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
//! | 4  | Single failure → gen_tps held for validity window | Some(…) held       | same        | GREEN |
//! | 5  | Omitted counter → None (not zero, not panic)      | None, baseline kept| None, kept  | GREEN |
//! | 6  | Omitted → success path preserves baseline (≠ Fail)| immediate recovery | same        | GREEN |
//! | 7  | Malformed payload → None (not zero, not panic)    | None               | None        | GREEN |
//! | 8  | RunningIdle → gen_tps present + running_reqs = 0  | Some(+), req=0     | same        | GREEN |
//! | 9  | Expiry boundary: held inside window, gone after   | B1 held, B2 none   | same        | GREEN |
//!
//! Scenarios 1–3, 5–8 verify already-correct or distinguishably-observable
//! behaviour.  All scenarios now serve as regression guards.
//!
//! The `lemonade_*` tests below the matrix cover Lemonade discovery and
//! ownership at the same seam, and are not rows in it: identity and the shared
//! last-request rate across several loaded models; external discovery, its
//! warnings (a seen endpoint closing, one never seen, an HTTP error on first
//! contact, recovery) and its removal of rows; a managed record taking over, or
//! being removed from, an endpoint; managed rewrites that reset the attributed
//! rate; scrape-only ticks; and IPv4/IPv6 dual-stack binds. The test names say
//! which.
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
//!   validity clock NOT reset, tracker baseline preserved.
//! - Unchanged valid counter after full rate window: `gen_tps = Some(0.0)`
//!   (Fresh — idle is not unknown).
//! - Scrape failure (HTTP non-200): gen_tps holds (Held) until validity expires,
//!   then `None`.  Tracker baseline cleared on failure so recovery re-baselines.
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

/// Read-only Lemonade fixture with a mutable health body. The fixture records
/// unexpected requests, including any attempt to manage or stop the server.
async fn lemonade_fixture(
    health: Arc<Mutex<String>>,
    unexpected: Arc<AtomicU64>,
) -> (u16, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let stats = Arc::new(Mutex::new(r#"{"tokens_per_second":42.0}"#.to_string()));
    lemonade_fixture_on(listener, health, stats, unexpected).await
}

/// Same fixture on a caller-chosen listener with a mutable stats body.
// Async so call-sites read `.await` like `lemonade_fixture`.
#[allow(clippy::unused_async)]
async fn lemonade_fixture_on(
    listener: TcpListener,
    health: Arc<Mutex<String>>,
    stats: Arc<Mutex<String>>,
    unexpected: Arc<AtomicU64>,
) -> (u16, tokio::task::JoinHandle<()>) {
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let health = health.clone();
            let stats = stats.clone();
            let unexpected = unexpected.clone();
            tokio::spawn(async move {
                let mut bytes = [0; 4096];
                let n = socket.read(&mut bytes).await.unwrap();
                let request = String::from_utf8_lossy(&bytes[..n]);
                let body = if request.starts_with("GET /api/v1/health ") {
                    health.lock().unwrap().clone()
                } else if request.starts_with("GET /api/v1/stats ") {
                    stats.lock().unwrap().clone()
                } else {
                    unexpected.fetch_add(1, Ordering::Relaxed);
                    "{}".to_string()
                };
                let response = if body == "HTTP_503" {
                    HTTP_503.to_string()
                } else {
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                };
                let _ = socket.write_all(response.as_bytes()).await;
            });
        }
    });
    (port, task)
}

#[tokio::test]
async fn lemonade_multiple_models_keep_identity_and_do_not_share_last_request_rate() {
    let health = Arc::new(Mutex::new(r#"{"status":"ok","all_models_loaded":[{"model_name":"Model-A"},{"model_name":"Model-B"},{"model_name":"Model-A"}]}"#.to_string()));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (port, server) = lemonade_fixture(health.clone(), unexpected.clone()).await;
    let (tx, mut rx) = broadcast::channel(64);
    let clock_dir = tempfile::tempdir().unwrap();
    let clock = clock_dir.path().join("clock");
    std::fs::write(&clock, "hold").unwrap();
    let opts = RunnerOptions {
        test_clock_offset_path: Some(clock),
        enable_lemonade: true,
        lemonade_port: port,
        disable_vllm_metrics: false,
        discovery_tick: TICK,
        instance_tick: TICK,
        ..RunnerOptions::default()
    };
    let task = spawn_runner(tx, opts).await;
    let result = wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 2).then(|| snap.instances.clone())
    })
    .await;
    if result.is_err() {
        task.abort();
        server.abort();
    }
    let rows = result.expect("two deduplicated loaded model rows");
    assert!(rows.iter().all(|row| row.gen_tps.is_none()));
    let model_a_id = rows
        .iter()
        .find(|row| row.model_name == "Model-A")
        .unwrap()
        .container_id
        .clone();
    let model_b_id = rows
        .iter()
        .find(|row| row.model_name == "Model-B")
        .unwrap()
        .container_id
        .clone();
    assert_ne!(model_a_id, model_b_id);
    *health.lock().unwrap() = r#"{"status":"ok","model_loaded":"Model-B","all_models_loaded":[{"model_name":"Model-B"},{"model_name":"Model-A"}]}"#.to_string();
    wait_for_snapshot(&mut rx, |snap| {
        let a = snap
            .instances
            .iter()
            .find(|row| row.container_id == model_a_id)?;
        let b = snap
            .instances
            .iter()
            .find(|row| row.container_id == model_b_id)?;
        (a.gen_tps.is_none() && b.gen_tps.is_none()).then_some(())
    })
    .await
    .expect("last access is not last request; multiple loaded models have no attributable rate");
    *health.lock().unwrap() =
        r#"{"status":"ok","all_models_loaded":[{"model_name":"Model-A"}]}"#.to_string();
    wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 1
            && snap.instances[0].container_id == model_a_id
            && snap.instances[0].gen_tps == Some(42.0))
        .then_some(())
    })
    .await
    .expect("unloaded model disappears; singleton receives rate");
    // Freeze observation time: expiry cannot make a missing invalidation pass,
    // even if the test is descheduled or waits through its entire deadline.
    drain_snapshots(&mut rx);
    *health.lock().unwrap() =
        r#"{"status":"ok","model_loaded":"Model-A","all_models_loaded":["Model-A","Model-B"]}"#
            .to_string();
    let ambiguous = wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 2).then(|| snap.instances.clone())
    })
    .await
    .expect("adding another model produces both rows on the next discovery");
    assert!(
        ambiguous
            .iter()
            .all(|row| row.gen_tps.is_none() && row.gen_tps_observation.is_none()),
        "the first multi-model snapshot must invalidate the singleton rate and observation"
    );
    *health.lock().unwrap() =
        r#"{"status":"error","all_models_loaded":["Model-A","Model-B"]}"#.to_string();
    wait_for_snapshot(&mut rx, |snap| snap.instances.is_empty().then_some(()))
        .await
        .expect("non-ok health never remains ready");
    *health.lock().unwrap() = r#"{"status":"ok","all_models_loaded":["Model-A"]}"#.to_string();
    wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 1 && snap.instances[0].container_id == model_a_id).then_some(())
    })
    .await
    .expect("healthy recovery restores the same model identity");
    *health.lock().unwrap() = r#"{"status":"ok","all_models_loaded":[]}"#.to_string();
    wait_for_snapshot(&mut rx, |snap| snap.instances.is_empty().then_some(()))
        .await
        .expect("healthy router without loaded models is not serving");
    task.abort();
    server.abort();
    assert_eq!(
        unexpected.load(Ordering::Relaxed),
        0,
        "only read-only health/stats requests are allowed"
    );
}

#[tokio::test]
async fn lemonade_managed_nondefault_port_scrapes_even_when_external_discovery_disabled() {
    let health = Arc::new(Mutex::new(
        r#"{"status":"ok","model_loaded":"Model-A"}"#.to_string(),
    ));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (port, server) = lemonade_fixture(health, unexpected.clone()).await;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("managed.json"), format!(r#"{{"service_id":"managed-lemonade","engine":"lemonade","model_ref":"model-alias","canonical_model_id":"Model-A","host":"127.0.0.1","port":{port},"status":"ready"}}"#)).unwrap();
    let (tx, mut rx) = broadcast::channel(64);
    let opts = RunnerOptions {
        services_dir: Some(dir.path().to_path_buf()),
        enable_lemonade: false,
        discovery_tick: TICK,
        instance_tick: TICK,
        ..RunnerOptions::default()
    };
    let task = spawn_runner(tx, opts).await;
    let result = wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 1
            && snap.instances[0].container_id == "managed-lemonade"
            && snap.instances[0].managed
            && snap.instances[0].gen_tps == Some(42.0)
            && snap.instances[0].gen_tps_last_request)
            .then_some(())
    })
    .await;
    task.abort();
    server.abort();
    result.expect("managed Lemonade is scraped at its registry port without external discovery");
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn lemonade_equivalent_endpoint_records_do_not_duplicate_singleton_rate() {
    let health = Arc::new(Mutex::new(
        r#"{"status":"ok","model_loaded":"Model-A"}"#.to_string(),
    ));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (port, server) = lemonade_fixture(health, unexpected.clone()).await;
    let dir = tempfile::tempdir().unwrap();
    for (id, host) in [("first", "127.0.0.1"), ("second", "LOCALHOST")] {
        std::fs::write(dir.path().join(format!("{id}.json")), format!(r#"{{"service_id":"{id}","engine":"lemonade","model_ref":"Model-A","host":"{host}","port":{port},"status":"ready"}}"#)).unwrap();
    }
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            services_dir: Some(dir.path().to_path_buf()),
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let result = wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 2).then(|| snap.instances.clone())
    })
    .await;
    task.abort();
    server.abort();
    let rows = result.expect("both registry identities remain present");
    assert!(
        rows.iter()
            .all(|row| row.managed && row.gen_tps.is_none() && row.gen_tps_observation.is_none()),
        "equivalent endpoint records cannot each receive its singleton rate: {rows:?}"
    );
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn lemonade_external_endpoint_deduplicates_managed_owner_then_reappears_after_record_removal()
{
    let health = Arc::new(Mutex::new(r#"{"status":"ok","all_models_loaded":[{"model_name":"Model-A"},{"model_name":"Model-B"}]}"#.to_string()));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (port, server) = lemonade_fixture(health, unexpected.clone()).await;
    let dir = tempfile::tempdir().unwrap();
    let record = dir.path().join("managed.json");
    std::fs::write(&record, format!(r#"{{"service_id":"managed-lemonade","engine":"lemonade","model_ref":"Model-A","host":"localhost","port":{port},"status":"ready"}}"#)).unwrap();
    let (tx, mut rx) = broadcast::channel(64);
    let opts = RunnerOptions {
        services_dir: Some(dir.path().to_path_buf()),
        enable_lemonade: true,
        lemonade_port: port,
        discovery_tick: TICK,
        instance_tick: TICK,
        ..RunnerOptions::default()
    };
    let task = spawn_runner(tx, opts).await;
    wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 1
            && snap.instances[0].container_id == "managed-lemonade"
            && snap.instances[0].gen_tps.is_none())
        .then_some(())
    })
    .await
    .expect("managed endpoint is not duplicated or assigned ambiguous stats");
    std::fs::remove_file(record).unwrap();
    wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 2
            && snap.instances.iter().all(|row| {
                !row.managed && row.container_id != "managed-lemonade" && row.gen_tps.is_none()
            }))
        .then_some(())
    })
    .await
    .expect("removing the registry file does not stop an externally reachable server");
    task.abort();
    server.abort();
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn lemonade_managed_record_taking_over_endpoint_drops_existing_external_rows() {
    let health = Arc::new(Mutex::new(r#"{"status":"ok","all_models_loaded":[{"model_name":"Model-A"},{"model_name":"Model-B"}]}"#.to_string()));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (port, server) = lemonade_fixture(health, unexpected.clone()).await;
    let dir = tempfile::tempdir().unwrap();
    let (tx, mut rx) = broadcast::channel(64);
    let opts = RunnerOptions {
        services_dir: Some(dir.path().to_path_buf()),
        enable_lemonade: true,
        lemonade_port: port,
        discovery_tick: TICK,
        instance_tick: TICK,
        ..RunnerOptions::default()
    };
    let task = spawn_runner(tx, opts).await;
    wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 2 && snap.instances.iter().all(|row| !row.managed)).then_some(())
    })
    .await
    .expect("external rows are discovered before any managed record exists");
    std::fs::write(dir.path().join("managed.json"), format!(r#"{{"service_id":"managed-lemonade","engine":"lemonade","model_ref":"Model-A","host":"localhost","port":{port},"status":"ready"}}"#)).unwrap();
    wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 1
            && snap.instances[0].container_id == "managed-lemonade"
            && snap.instances[0].managed)
            .then_some(())
    })
    .await
    .expect("managed takeover removes the external rows and keeps the managed row");
    task.abort();
    server.abort();
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn lemonade_external_health_failure_warns_and_removes_rows_without_grace() {
    let health = Arc::new(Mutex::new(
        r#"{"status":"ok","model_loaded":"Model-A"}"#.to_string(),
    ));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (port, server) = lemonade_fixture(health.clone(), unexpected.clone()).await;
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            enable_lemonade: true,
            lemonade_port: port,
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let row = wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 1 && snap.instances[0].gen_tps == Some(42.0))
            .then(|| snap.instances[0].clone())
    })
    .await
    .expect("external singleton established");
    assert!(
        row.gen_tps_last_request,
        "a Lemonade rate is the last request's and must be flagged as such"
    );
    drain_snapshots(&mut rx);
    *health.lock().unwrap() = "HTTP_503".to_string();
    // Discovery runs every tick here, so the rows go on the next one. A hold of
    // even a few ticks (the observation validity window is 6 s, some thirty
    // ticks) is a regression: count the snapshots that still carry the row.
    let still_listed = AtomicU64::new(0);
    let result = wait_for_snapshot(&mut rx, |snap| {
        if snap.instances.is_empty() {
            Some(snap.warnings.clone())
        } else {
            still_listed.fetch_add(1, Ordering::Relaxed);
            None
        }
    })
    .await;
    task.abort();
    server.abort();
    let warnings = result.expect("external failure removes rows on next discovery, without grace");
    assert!(
        still_listed.load(Ordering::Relaxed) <= 3,
        "rows were held for {} snapshots after the health failure",
        still_listed.load(Ordering::Relaxed)
    );
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("lemonade health")
                && warning.contains(&port.to_string())
                && warning.contains("503")),
        "failed disappearance carries the health error: {warnings:?}"
    );
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn lemonade_seen_external_endpoint_closing_warns_and_removes_rows() {
    let health = Arc::new(Mutex::new(
        r#"{"status":"ok","model_loaded":"Model-A"}"#.to_string(),
    ));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (port, server) = lemonade_fixture(health, unexpected.clone()).await;
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            enable_lemonade: true,
            lemonade_port: port,
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    wait_for_snapshot(&mut rx, |snap| (snap.instances.len() == 1).then_some(()))
        .await
        .unwrap();
    server.abort();
    let _ = server.await;
    drain_snapshots(&mut rx);
    let result = wait_for_snapshot(&mut rx, |snap| {
        snap.instances.is_empty().then(|| snap.warnings.clone())
    })
    .await;
    let warnings = result.expect("closing a seen endpoint removes its rows");
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("lemonade health")
                && warning.contains(&port.to_string())),
        "disappearance must name the endpoint: {warnings:?}"
    );

    // The endpoint comes back on the same port: its row returns and the warning
    // about its disappearance is cleared, not kept for as long as the daemon runs.
    let listener = TcpListener::bind(("127.0.0.1", port)).await.unwrap();
    let stats = Arc::new(Mutex::new(r#"{"tokens_per_second":42.0}"#.to_string()));
    let health = Arc::new(Mutex::new(
        r#"{"status":"ok","model_loaded":"Model-A"}"#.to_string(),
    ));
    let (_, revived) = lemonade_fixture_on(listener, health, stats, unexpected.clone()).await;
    drain_snapshots(&mut rx);
    let recovered = wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 1).then(|| snap.warnings.clone())
    })
    .await;
    task.abort();
    revived.abort();
    let warnings = recovered.expect("a recovered endpoint's row returns");
    assert!(
        !warnings.iter().any(|warning| warning.contains("lemonade")),
        "recovery must clear the disappearance warning: {warnings:?}"
    );
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn lemonade_never_seen_absent_external_endpoint_is_quiet() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            enable_lemonade: true,
            lemonade_port: port,
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let snap = wait_for_snapshot(&mut rx, |snap| Some(snap.clone())).await;
    task.abort();
    let snap = snap.expect("absent endpoint does not block snapshots");
    assert!(snap.instances.is_empty());
    assert!(
        !snap
            .warnings
            .iter()
            .any(|warning| warning.contains("lemonade")),
        "normal absent external endpoint is not a noisy warning"
    );
}

/// A never-seen external endpoint that answers with an HTTP error is present
/// but unusable (auth needed or broken), so it warns on first contact.
async fn never_seen_endpoint_answering_warns(status_line: &'static str, needle: &str) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            tokio::spawn(async move {
                let mut bytes = [0; 4096];
                let _ = socket.read(&mut bytes).await;
                let response = format!(
                    "HTTP/1.1 {status_line}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                let _ = socket.write_all(response.as_bytes()).await;
            });
        }
    });
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            enable_lemonade: true,
            lemonade_port: port,
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let warnings = wait_for_snapshot(&mut rx, |snap| {
        snap.warnings
            .iter()
            .any(|w| w.contains("lemonade"))
            .then(|| snap.warnings.clone())
    })
    .await;
    task.abort();
    server.abort();
    let warnings = warnings.expect("first-contact HTTP error warns");
    assert!(
        warnings.iter().any(|w| w.contains("lemonade health")
            && w.contains(&port.to_string())
            && w.contains(needle)),
        "warning names the endpoint and status: {warnings:?}"
    );
}

#[tokio::test]
async fn lemonade_never_seen_endpoint_answering_401_warns() {
    never_seen_endpoint_answering_warns("401 Unauthorized", "401").await;
}

#[tokio::test]
async fn lemonade_never_seen_endpoint_answering_503_warns() {
    never_seen_endpoint_answering_warns("503 Service Unavailable", "503").await;
}

#[tokio::test]
async fn lemonade_protected_managed_endpoint_keeps_registry_status_without_telemetry_warning() {
    for status in ["401 Unauthorized", "403 Forbidden"] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let unexpected = Arc::new(AtomicU64::new(0));
        let seen = unexpected.clone();
        let server = tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let seen = seen.clone();
                tokio::spawn(async move {
                    let mut bytes = [0; 4096];
                    let n = socket.read(&mut bytes).await.unwrap();
                    let request = String::from_utf8_lossy(&bytes[..n]);
                    if !request.starts_with("GET /api/v1/health ")
                        || request.to_ascii_lowercase().contains("authorization:")
                    {
                        seen.fetch_add(1, Ordering::Relaxed);
                    }
                    let _ = socket.write_all(format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await;
                });
            }
        });
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("managed.json"), format!(r#"{{"service_id":"protected","engine":"lemonade","model_ref":"Model-A","host":"127.0.0.1","port":{port},"status":"starting"}}"#)).unwrap();
        let (tx, mut rx) = broadcast::channel(64);
        let task = spawn_runner(
            tx,
            RunnerOptions {
                services_dir: Some(dir.path().to_path_buf()),
                enable_lemonade: true,
                lemonade_port: port,
                discovery_tick: TICK,
                instance_tick: TICK,
                ..RunnerOptions::default()
            },
        )
        .await;
        let snapshot = wait_for_snapshot(&mut rx, |snap| {
            (snap.instances.len() == 1).then(|| snap.clone())
        })
        .await;
        task.abort();
        server.abort();
        let snapshot = snapshot.unwrap();
        let row = &snapshot.instances[0];
        assert!(row.managed);
        assert_eq!(row.container_id, "protected");
        assert!(matches!(
            row.status,
            rocm_dash_core::metrics::InstanceStatus::Starting { .. }
        ));
        assert!(row.gen_tps.is_none() && row.gen_tps_observation.is_none());
        assert!(
            !snapshot
                .warnings
                .iter()
                .any(|warning| warning.contains("lemonade")),
            "protected telemetry is unavailable, not a recurring health failure: {:?}",
            snapshot.warnings
        );
        assert_eq!(
            unexpected.load(Ordering::Relaxed),
            0,
            "no stats, fallback, credentials or lifecycle requests"
        );
    }
}

/// A slow health response can finish only after the other endpoint's stats
/// request. Both serial orders fail: fast health waits for slow health to start.
#[tokio::test]
async fn lemonade_independent_endpoints_overlap_and_health_failures_warn() {
    let slow_started = Arc::new(tokio::sync::Notify::new());
    let fast_stats = Arc::new(tokio::sync::Notify::new());
    let fast_stats_count = Arc::new(AtomicU64::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let unexpected = Arc::new(AtomicU64::new(0));
    let mut servers = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    for (id, model, slow) in [("slow", "Model-S", true), ("fast", "Model-F", false)] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        std::fs::write(dir.path().join(format!("{id}.json")), format!(r#"{{"service_id":"{id}","engine":"lemonade","model_ref":"{model}","host":"127.0.0.1","port":{port},"status":"ready"}}"#)).unwrap();
        let slow_started = slow_started.clone();
        let fast_stats = fast_stats.clone();
        let fast_stats_count = fast_stats_count.clone();
        let failed = failed.clone();
        let unexpected = unexpected.clone();
        let health_succeeded = Arc::new(AtomicBool::new(false));
        servers.push(tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let slow_started = slow_started.clone();
                let fast_stats = fast_stats.clone();
                let fast_stats_count = fast_stats_count.clone();
                let failed = failed.clone();
                let unexpected = unexpected.clone();
                let health_succeeded = health_succeeded.clone();
                tokio::spawn(async move {
                    let mut bytes = [0; 4096];
                    let n = socket.read(&mut bytes).await.unwrap();
                    let request = String::from_utf8_lossy(&bytes[..n]);
                    let response = if request.starts_with("GET /api/v1/health ") {
                        health_succeeded.store(false, Ordering::SeqCst);
                        if slow && failed.load(Ordering::SeqCst) {
                            HTTP_503.to_string()
                        } else {
                            let overlap = if slow {
                                slow_started.notify_one();
                                timeout(Duration::from_secs(1), fast_stats.notified()).await
                            } else {
                                timeout(Duration::from_secs(1), slow_started.notified()).await
                            };
                            if overlap.is_ok() {
                                health_succeeded.store(true, Ordering::SeqCst);
                                http_ok(&format!(r#"{{"status":"ok","model_loaded":"{model}"}}"#))
                            } else {
                                HTTP_503.to_string()
                            }
                        }
                    } else if request.starts_with("GET /api/v1/stats ") {
                        if !health_succeeded.swap(false, Ordering::SeqCst) {
                            unexpected.fetch_add(1, Ordering::Relaxed);
                        }
                        if !slow {
                            fast_stats_count.fetch_add(1, Ordering::SeqCst);
                            fast_stats.notify_one();
                        }
                        http_ok(r#"{"tokens_per_second":42.0}"#)
                    } else {
                        unexpected.fetch_add(1, Ordering::Relaxed);
                        HTTP_503.to_string()
                    };
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        }));
    }
    let (tx, mut rx) = broadcast::channel(64);
    let clock = dir.path().join("clock");
    std::fs::write(&clock, "hold").unwrap();
    let task = spawn_runner(
        tx,
        RunnerOptions {
            services_dir: Some(dir.path().to_path_buf()),
            discovery_tick: TICK,
            instance_tick: TICK,
            test_clock_offset_path: Some(clock),
            ..RunnerOptions::default()
        },
    )
    .await;
    let overlap = wait_for_snapshot(&mut rx, |snap| {
        (snap.instances.len() == 2).then(|| snap.instances.clone())
    })
    .await;
    if overlap.is_err() {
        task.abort();
        for server in &servers {
            server.abort();
        }
    }
    let rows = overlap.expect("both registry rows appear in the first scrape snapshot");
    assert!(
        rows.iter().all(|row| row.gen_tps == Some(42.0)),
        "independent endpoint health and stats must overlap in the same cycle, not run serially: {rows:?}"
    );
    drain_snapshots(&mut rx);
    let previous_fast_stats = fast_stats_count.load(Ordering::SeqCst);
    failed.store(true, Ordering::SeqCst);
    // Allow fast health through while slow health is now failing immediately.
    slow_started.notify_one();
    let failure = wait_for_snapshot(&mut rx, |snap| {
        snap.warnings
            .iter()
            .any(|warning| {
                warning.contains("lemonade health")
                    && warning.contains("slow")
                    && warning.contains("503")
            })
            .then(|| snap.instances.clone())
    })
    .await;
    task.abort();
    for server in servers {
        server.abort();
    }
    let rows =
        failure.expect("health failures must produce endpoint/row-context snapshot warnings");
    assert!(
        fast_stats_count.load(Ordering::SeqCst) > previous_fast_stats,
        "independent fast endpoint completes another stats scrape during slow health failure"
    );
    assert!(
        rows.iter().all(|row| row.gen_tps == Some(42.0)),
        "failed health holds existing rates; independent fast endpoint still scrapes"
    );
    assert_eq!(
        unexpected.load(Ordering::Relaxed),
        0,
        "only health-before-stats reads are permitted"
    );
}

/// A managed record whose identity (model or host) is rewritten while a rate is
/// attributed must drop that rate and its observation. The test clock is held so
/// expiry cannot clear it, and the stats route stops reporting a rate after the
/// rewrite so only the identity reset can remove the held value.
async fn assert_record_rewrite_resets_attributed_rate(
    write_initial: fn(&std::path::Path, u16),
    rewrite: fn(&std::path::Path, u16),
) {
    let health = Arc::new(Mutex::new(
        r#"{"status":"ok","model_loaded":"Model-A"}"#.to_string(),
    ));
    let stats = Arc::new(Mutex::new(r#"{"tokens_per_second":42.0}"#.to_string()));
    let unexpected = Arc::new(AtomicU64::new(0));
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let (port, server) =
        lemonade_fixture_on(listener, health, stats.clone(), unexpected.clone()).await;
    let dir = tempfile::tempdir().unwrap();
    let record = dir.path().join("managed.json");
    write_initial(&record, port);
    let clock_dir = tempfile::tempdir().unwrap();
    let clock = clock_dir.path().join("clock");
    std::fs::write(&clock, "hold").unwrap();
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            services_dir: Some(dir.path().to_path_buf()),
            enable_lemonade: false,
            discovery_tick: TICK,
            instance_tick: TICK,
            test_clock_offset_path: Some(clock),
            ..RunnerOptions::default()
        },
    )
    .await;
    let outcome = async {
        wait_for_snapshot(&mut rx, |snap| {
            (snap.instances.len() == 1 && snap.instances[0].gen_tps == Some(42.0)).then_some(())
        })
        .await
        .map_err(|e| format!("rate attributed: {e}"))?;
        // Control: with the stats rate gone and identity unchanged, the held
        // value survives, so a later None can only come from the reset.
        *stats.lock().unwrap() = "{}".to_string();
        for _ in 0..4 {
            let held = wait_for_snapshot(&mut rx, |snap| Some(snap.instances.clone())).await?;
            if held.len() != 1 || held[0].gen_tps != Some(42.0) {
                return Err(format!(
                    "held rate lost without an identity change: {held:?}"
                ));
            }
        }
        rewrite(&record, port);
        wait_for_snapshot(&mut rx, |snap| {
            (snap.instances.len() == 1
                && snap.instances[0].gen_tps.is_none()
                && snap.instances[0].gen_tps_observation.is_none())
            .then_some(())
        })
        .await
        .map_err(|e| format!("identity rewrite must reset the attributed rate: {e}"))
    }
    .await;
    task.abort();
    server.abort();
    outcome.unwrap();
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

fn write_managed_record(path: &std::path::Path, model_ref: &str, host: &str, port: u16) {
    std::fs::write(path, format!(r#"{{"service_id":"managed-lemonade","engine":"lemonade","model_ref":"{model_ref}","canonical_model_id":"Model-A","host":"{host}","port":{port},"status":"ready"}}"#)).unwrap();
}

#[tokio::test]
async fn lemonade_managed_model_rewrite_resets_attributed_rate() {
    assert_record_rewrite_resets_attributed_rate(
        |path, port| write_managed_record(path, "alias-one", "127.0.0.1", port),
        |path, port| write_managed_record(path, "alias-two", "127.0.0.1", port),
    )
    .await;
}

/// Rewriting `canonical_model_id` to a name health does not report leaves the row
/// matching nothing, so the attributed rate is cleared because attribution is
/// lost. The same-endpoint identity clause is isolated by
/// `lemonade_managed_canonical_rewrite_that_still_matches_health_resets_rate`.
#[tokio::test]
async fn lemonade_managed_canonical_rewrite_that_unmatches_health_clears_rate() {
    assert_record_rewrite_resets_attributed_rate(
        |path, port| write_managed_record(path, "alias-one", "127.0.0.1", port),
        |path, port| {
            write_managed_record(path, "alias-one", "127.0.0.1", port);
            let mut record: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            record["canonical_model_id"] = serde_json::json!("Model-B");
            std::fs::write(path, serde_json::to_vec(&record).unwrap()).unwrap();
        },
    )
    .await;
}

/// Only `canonical_model_id` changes, and the row still matches health through
/// its `model_ref`. Nothing else then clears the rate (the row is not
/// unmatched), so removing the canonical-id clause from the same-endpoint check
/// leaves the old rate attributed to the rewritten service.
#[tokio::test]
async fn lemonade_managed_canonical_rewrite_that_still_matches_health_resets_rate() {
    fn write(path: &std::path::Path, canonical: &str, port: u16) {
        std::fs::write(path, format!(r#"{{"service_id":"managed-lemonade","engine":"lemonade","model_ref":"Model-A","canonical_model_id":"{canonical}","host":"127.0.0.1","port":{port},"status":"ready"}}"#)).unwrap();
    }
    assert_record_rewrite_resets_attributed_rate(
        |path, port| write(path, "Model-A", port),
        |path, port| write(path, "Model-B", port),
    )
    .await;
}

#[tokio::test]
async fn lemonade_managed_host_rewrite_resets_attributed_rate() {
    assert_record_rewrite_resets_attributed_rate(
        |path, port| write_managed_record(path, "alias-one", "127.0.0.1", port),
        |path, port| write_managed_record(path, "alias-one", "127.0.0.2", port),
    )
    .await;
}

/// With discovery three times slower than scraping, a health failure on a
/// scrape-only tick holds the external rows and warns; the next discovery tick
/// with the same failure removes them. The fixture counts health requests (one
/// per tick) and fails from the fourth, so the first failing ticks are
/// deterministically scrape-only (discovery runs on ticks 1, 3, 6, ...).
///
/// `failure` is the status the health route starts answering. A 401/403 must
/// behave like the 503: the external endpoint is checked as strictly on a
/// scrape-only tick as on a discovery tick, so the warning appears on the first
/// failing tick rather than waiting for the next discovery tick.
async fn scrape_only_health_failure_holds_external_rows(failure: &'static str) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits = Arc::new(AtomicU64::new(0));
    let server_hits = hits.clone();
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let hits = server_hits.clone();
            tokio::spawn(async move {
                let mut bytes = [0; 4096];
                let n = socket.read(&mut bytes).await.unwrap();
                let request = String::from_utf8_lossy(&bytes[..n]);
                let response = if request.starts_with("GET /api/v1/health ") {
                    if hits.fetch_add(1, Ordering::SeqCst) + 1 >= 4 {
                        format!(
                            "HTTP/1.1 {failure}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                    } else {
                        http_ok(r#"{"status":"ok","model_loaded":"Model-A"}"#)
                    }
                } else if request.starts_with("GET /api/v1/stats ") {
                    http_ok(r#"{"tokens_per_second":42.0}"#)
                } else {
                    HTTP_503.to_string()
                };
                let _ = socket.write_all(response.as_bytes()).await;
            });
        }
    });
    let (tx, mut rx) = broadcast::channel(256);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            enable_lemonade: true,
            lemonade_port: port,
            discovery_tick: 3 * TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let outcome = async {
        let mut saw_held = false;
        loop {
            let (rows, warned) = wait_for_snapshot(&mut rx, |snap| {
                Some((
                    snap.instances.len(),
                    snap.warnings.iter().any(|w| w.contains("lemonade health")),
                ))
            })
            .await?;
            match (rows, warned) {
                (1, true) => saw_held = true,
                (0, true) => {
                    return if saw_held {
                        Ok(())
                    } else {
                        Err("rows removed by a scrape-only tick".to_string())
                    };
                }
                (0, false) if hits.load(Ordering::SeqCst) >= 4 => {
                    return Err("rows removed without a warning".to_string());
                }
                _ => {}
            }
        }
    }
    .await;
    task.abort();
    server.abort();
    outcome.unwrap();
}

#[tokio::test]
async fn lemonade_scrape_only_tick_holds_external_rows_until_next_discovery() {
    scrape_only_health_failure_holds_external_rows("503 Service Unavailable").await;
}

/// An already-known external endpoint that starts requiring credentials is
/// reported on the scrape-only tick that finds out, not one discovery interval
/// later (the managed-target tolerance for 401/403 does not apply to it).
#[tokio::test]
async fn lemonade_scrape_only_tick_warns_when_a_known_external_endpoint_requires_credentials() {
    for status in ["401 Unauthorized", "403 Forbidden"] {
        scrape_only_health_failure_holds_external_rows(status).await;
    }
}

/// A known external endpoint whose stats route fails while health is fine and one
/// model is loaded (so the rate is attributable): the failure is reported, the row
/// stays, and nothing but reads reaches the server.
#[tokio::test]
async fn lemonade_stats_failure_on_an_attributed_endpoint_warns_and_keeps_the_row() {
    let health = Arc::new(Mutex::new(
        r#"{"status":"ok","model_loaded":"Model-A"}"#.to_string(),
    ));
    let stats = Arc::new(Mutex::new("HTTP_503".to_string()));
    let unexpected = Arc::new(AtomicU64::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (port, server) = lemonade_fixture_on(listener, health, stats, unexpected.clone()).await;
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            enable_lemonade: true,
            lemonade_port: port,
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let seen = wait_for_snapshot(&mut rx, |snap| {
        snap.warnings
            .iter()
            .any(|w| w.contains("lemonade stats"))
            .then(|| (snap.instances.clone(), snap.warnings.clone()))
    })
    .await;
    task.abort();
    server.abort();
    let (rows, warnings) = seen.expect("a stats failure is reported");
    assert_eq!(rows.len(), 1, "the row stays listed: {rows:?}");
    assert!(rows[0].gen_tps.is_none(), "{rows:?}");
    assert!(
        warnings.iter().any(|w| w.contains("lemonade stats")
            && w.contains(&port.to_string())
            && w.contains("503")),
        "the warning names the endpoint and the status: {warnings:?}"
    );
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

/// A managed target whose health route answers with a genuine 5xx (not the
/// 404/401/403 that mean "no router" or "credentials needed") is a failure to
/// report: the registry row stays with its registry status, and the warning
/// names the endpoint and the managed service.
#[tokio::test]
async fn lemonade_managed_endpoint_health_5xx_warns_and_keeps_the_registry_row() {
    let health = Arc::new(Mutex::new("HTTP_503".to_string()));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (port, server) = lemonade_fixture(health, unexpected.clone()).await;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("managed.json"), format!(r#"{{"service_id":"managed-5xx","engine":"lemonade","model_ref":"Model-A","host":"127.0.0.1","port":{port},"status":"ready"}}"#)).unwrap();
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            services_dir: Some(dir.path().to_path_buf()),
            enable_lemonade: false,
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let seen = wait_for_snapshot(&mut rx, |snap| {
        snap.warnings
            .iter()
            .any(|w| w.contains("lemonade health"))
            .then(|| (snap.instances.clone(), snap.warnings.clone()))
    })
    .await;
    task.abort();
    server.abort();
    let (rows, warnings) = seen.expect("a managed health 5xx is reported");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(
        rows[0].managed && rows[0].container_id == "managed-5xx",
        "{rows:?}"
    );
    assert!(rows[0].gen_tps.is_none(), "{rows:?}");
    assert!(
        warnings.iter().any(|w| w.contains("lemonade health")
            && w.contains(&port.to_string())
            && w.contains("managed-5xx")
            && w.contains("503")),
        "{warnings:?}"
    );
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

/// A dual-stack `::` bind at the external endpoint's port is the same service as
/// the IPv4 external probe; it must not also appear as an external row.
#[tokio::test]
async fn lemonade_dual_stack_managed_bind_does_not_duplicate_external_row() {
    let Ok(listener) = TcpListener::bind("[::]:0").await else {
        eprintln!("SKIPPED: IPv6 dual-stack bind unavailable; test did not run");
        return;
    };
    let health = Arc::new(Mutex::new(
        r#"{"status":"ok","model_loaded":"Model-A"}"#.to_string(),
    ));
    let stats = Arc::new(Mutex::new(r#"{"tokens_per_second":42.0}"#.to_string()));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (port, server) = lemonade_fixture_on(listener, health, stats, unexpected.clone()).await;
    let dir = tempfile::tempdir().unwrap();
    write_managed_record(&dir.path().join("managed.json"), "Model-A", "::", port);
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            services_dir: Some(dir.path().to_path_buf()),
            enable_lemonade: true,
            lemonade_port: port,
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let outcome = async {
        wait_for_snapshot(&mut rx, |snap| {
            snap.instances
                .iter()
                .any(|row| row.container_id == "managed-lemonade" && row.gen_tps == Some(42.0))
                .then_some(())
        })
        .await?;
        for _ in 0..5 {
            let rows = wait_for_snapshot(&mut rx, |snap| Some(snap.instances.clone())).await?;
            if rows.len() != 1 || rows[0].container_id != "managed-lemonade" {
                return Err(format!("duplicate external row: {rows:?}"));
            }
        }
        Ok(())
    }
    .await;
    task.abort();
    server.abort();
    outcome.unwrap();
    assert_eq!(unexpected.load(Ordering::Relaxed), 0);
}

/// A managed `::` record replaced by a `127.0.0.1` record on the same port
/// must not keep dialing the old IPv6 address: the IPv4-only listener has to
/// come up healthy with no standing warning.
#[tokio::test]
async fn lemonade_managed_ipv6_record_replaced_by_ipv4_record_on_same_port_recovers() {
    // Bind IPv4 loopback first, then require IPv6 on the same port; skip when
    // either family is unavailable.
    let Ok(v6) = TcpListener::bind("[::1]:0").await else {
        eprintln!("SKIPPED: IPv6 loopback bind unavailable; test did not run");
        return;
    };
    let port = v6.local_addr().unwrap().port();
    let Ok(v4) = TcpListener::bind(("127.0.0.1", port)).await else {
        eprintln!("SKIPPED: IPv4 bind on the IPv6 test port unavailable; test did not run");
        return;
    };
    let health = Arc::new(Mutex::new(
        r#"{"status":"ok","model_loaded":"Model-A"}"#.to_string(),
    ));
    let stats = Arc::new(Mutex::new(r#"{"tokens_per_second":42.0}"#.to_string()));
    let unexpected = Arc::new(AtomicU64::new(0));
    let (_, v6_server) =
        lemonade_fixture_on(v6, health.clone(), stats.clone(), unexpected.clone()).await;
    let dir = tempfile::tempdir().unwrap();
    let record = dir.path().join("managed.json");
    write_managed_record(&record, "Model-A", "::", port);
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            services_dir: Some(dir.path().to_path_buf()),
            enable_lemonade: false,
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let outcome = async {
        wait_for_snapshot(&mut rx, |snap| {
            (snap.instances.len() == 1 && snap.instances[0].gen_tps == Some(42.0)).then_some(())
        })
        .await
        .map_err(|e| format!("ipv6 record healthy: {e}"))?;
        // The old listener goes away and an IPv4-only service takes the port.
        v6_server.abort();
        let _ = v6_server.await;
        let (_, _v4_server) =
            lemonade_fixture_on(v4, health.clone(), stats.clone(), unexpected.clone()).await;
        // The replacement lands in one step, so no discovery tick sees an empty
        // registry that would drop the cached collector on its own.
        write_managed_record(&record, "Model-A", "127.0.0.1", port);
        wait_for_snapshot(&mut rx, |snap| {
            (snap.instances.len() == 1 && snap.instances[0].gen_tps == Some(42.0)).then_some(())
        })
        .await
        .map_err(|e| format!("ipv4 record healthy: {e}"))?;
        for _ in 0..4 {
            let warnings = wait_for_snapshot(&mut rx, |snap| Some(snap.warnings.clone())).await?;
            if warnings.iter().any(|w| w.contains("lemonade")) {
                return Err(format!("standing warning after takeover: {warnings:?}"));
            }
        }
        Ok::<(), String>(())
    }
    .await;
    task.abort();
    outcome.unwrap();
}

/// A managed direct llama-server row (the server answers only `/health` and
/// `/v1/models`, with no stats route) stays listed across repeated scrapes with
/// `gen_tps` left at `None` and no standing Lemonade warning.
#[tokio::test]
async fn lemonade_managed_direct_llama_server_row_has_no_standing_warning() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            tokio::spawn(async move {
                let mut bytes = [0; 4096];
                let n = socket.read(&mut bytes).await.unwrap();
                let request = String::from_utf8_lossy(&bytes[..n]);
                let response = if request.starts_with("GET /health ") {
                    http_ok(r#"{"status":"ok"}"#)
                } else if request.starts_with("GET /v1/models ") {
                    http_ok(r#"{"data":[{"id":"Model-A"}]}"#)
                } else {
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        .to_string()
                };
                let _ = socket.write_all(response.as_bytes()).await;
            });
        }
    });
    let dir = tempfile::tempdir().unwrap();
    write_managed_record(
        &dir.path().join("direct.json"),
        "Model-A",
        "127.0.0.1",
        port,
    );
    let (tx, mut rx) = broadcast::channel(64);
    let task = spawn_runner(
        tx,
        RunnerOptions {
            services_dir: Some(dir.path().to_path_buf()),
            enable_lemonade: false,
            discovery_tick: TICK,
            instance_tick: TICK,
            ..RunnerOptions::default()
        },
    )
    .await;
    let outcome = async {
        wait_for_snapshot(&mut rx, |snap| (snap.instances.len() == 1).then_some(())).await?;
        for _ in 0..6 {
            let (rows, warnings) = wait_for_snapshot(&mut rx, |snap| {
                Some((snap.instances.clone(), snap.warnings.clone()))
            })
            .await?;
            if rows.len() != 1 || rows[0].gen_tps.is_some() {
                return Err(format!("row must stay visible without a rate: {rows:?}"));
            }
            if warnings.iter().any(|w| w.contains("lemonade")) {
                return Err(format!("standing warning on a direct server: {warnings:?}"));
            }
        }
        Ok(())
    }
    .await;
    task.abort();
    server.abort();
    outcome.unwrap();
}

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

// `spawn_runner` is called with `.await` at every call-site (awaiting the
// returned JoinHandle would block forever); keep `async` so the call-sites
// read `spawn_runner(...).await` consistently and the JoinHandle is obtained
// without needing a separate let-binding.
#[allow(clippy::unused_async)]
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
            Ok(Ok(_) | Err(broadcast::error::RecvError::Lagged(_))) => {}
            Ok(Err(broadcast::error::RecvError::Closed)) => {
                return Err("broadcast channel closed".into());
            }
            Err(_) => return Err(format!("timed out after {SCENARIO_DEADLINE:?}")),
        }
    }
}

/// Extract the `contract-svc` instance's `gen_tps` from a snapshot.
/// `Some(None)` = instance present with `gen_tps` not yet computed;
/// `Some(Some(v))` = instance present with rate `v`; `None` = instance absent.
// The double Option is intentional: outer = found-instance, inner = gen_tps.
#[allow(clippy::option_option)]
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
/// GREEN: `GenerationObservationTracker` yields `Some(0.0)` when the counter is
/// unchanged after the full rate window.
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
/// GREEN: `GenerationObservationTracker` detects `cur < prev_val` and immediately
/// yields `None`, re-inserting the new lower value as the baseline.
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
/// GREEN: runner.rs:446-460 computes `known_services.difference(&disc.seen)`
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
            Ok(Ok(_) | Err(broadcast::error::RecvError::Lagged(_))) => {}
            Ok(Err(broadcast::error::RecvError::Closed)) => panic!("broadcast closed"),
            Err(e) => panic!("timed out waiting for InstanceGone: {e}"),
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

// ── Scenario 4: Single failure → gen_tps held for validity window ───────────

/// Contract: after a single failed `/metrics` scrape, `gen_tps` must remain
/// held at its last positive value for the validity window
/// `clamp(3 × instance_tick, 6 s, 30 s)` before clearing.
///
/// This is now a GREEN regression guard: the fix in `GenerationObservationTracker`
/// holds the last observed value for `clamp(3 × instance_tick, 6 s, 30 s)` before
/// clearing.  The assertion below will FAIL if the hold-window is regressed.
///
/// Complements cucumber Scenario 8 at the daemon broadcast seam (no PTY / TUI).
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

    // CONTRACT: the held value must still be present (regression guard).
    // If this fails, GenerationObservationTracker no longer holds gen_tps on failure.
    assert!(
        post_failure_gen_tps.is_some(),
        "EAI-7960 REGRESSION (daemon broadcast seam): gen_tps was cleared to None \
         immediately after the first failed /metrics scrape — regression detected.\n\
         Contract: hold for clamp(3 × {INSTANCE_TICK:?}, 6 s, 30 s).\n\
         Baseline was {baseline:.2} tok/s; post-failure was None.\n\
         This is a GREEN regression guard: if it fires, the EAI-7960 fix was reverted."
    );
}

// ── Scenario 5: Omitted counter → None (not zero, not panic) ───────────────

/// Contract: when the `/metrics` endpoint returns HTTP 200 but the body omits
/// `vllm:generation_tokens_total`, `gen_tps` is `None` — not `Some(0.0)`.
/// The tracker baseline must NOT be reset (success path, not failure path).
///
/// GREEN: `GenerationObservationTracker::observe()` is only called when
/// `gen_tokens_total` is `Some`; an absent counter is not forwarded and leaves
/// the tracker baseline untouched.
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
        if let Ok(Ok(Event::Snapshot(snap))) = timeout(remaining, rx.recv()).await
            && let Some(gen_tps_opt) = svc_gen_tps(&snap)
        {
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
/// very next scrape — because the tracker baseline was preserved on the success
/// path. This distinguishes `Omitted` from `Failure`, which resets the baseline.
///
/// GREEN: the success path only forwards `gen_tokens_total` to the tracker when
/// it is `Some`; an absent counter is not forwarded and leaves the tracker
/// baseline untouched.
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

    // Step 2: one omitted-counter scrape (success path, tracker baseline preserved).
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

    // Step 3: recover immediately — Growing again with preserved tracker baseline.
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
        if let Ok(Ok(Event::Snapshot(snap))) = timeout(remaining, rx.recv()).await
            && let Some(gen_tps_opt) = svc_gen_tps(&snap)
        {
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
                let has_gen_tps = i.gen_tps.is_some_and(|v| v > 0.0);
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
/// **BOUNDARY 1 (held assertion — regression guard):** immediately after the first
/// failed scrape, `gen_tps` must still be `Some(_)` (held at its last observed
/// value).  If this fails, `GenerationObservationTracker` no longer holds on
/// failure — the EAI-7960 fix was regressed.
///
/// **BOUNDARY 2 (expiry assertion):** after the full validity window
/// `clamp(3 × instance_tick, 6 s, 30 s)` = 6 s elapses, the held value must be
/// cleared and `gen_tps` must be `None`.
///
/// Both snapshots are captured before any assertion runs so both boundaries are
/// documented.  Both assertions are GREEN regression guards.
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
        "EAI-7960 BOUNDARY-1 REGRESSION (daemon broadcast seam): gen_tps cleared to None \
         immediately after first failure — regression detected.\n\
         Contract: hold for clamp(3 × {INSTANCE_TICK:?}, 6 s, 30 s).\n\
         Baseline was {baseline:.2} tok/s; post-failure was None.\n\
         This is a GREEN regression guard: if it fires, the EAI-7960 fix was reverted."
    );

    // Assert BOUNDARY 2 (reachable only after boundary-1 is fixed).
    assert!(
        boundary2_gen_tps.is_none(),
        "EAI-7960 BOUNDARY-2 FAILED: gen_tps is still Some({boundary2_gen_tps:?}) after the \
         validity window ({validity_window:?}) elapsed.\n\
         Contract: held value must expire to None after the window."
    );
}
