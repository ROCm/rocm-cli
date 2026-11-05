// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Characterisation tests for the runner's service-discovery reconcile.
//!
//! `run_loop` reconciles three discovery sources against its reducer state:
//! Docker, the probe-based Lemonade endpoint, and the managed-service registry.
//! These tests drive the real loop through the sources that can run without a
//! container daemon (the registry and Lemonade) and pin what a user of the
//! broadcast stream can observe. On the registry source: a service is announced
//! once, keeps its live scrape fields across rediscovery, and starts clean when
//! its endpoint changes. On Lemonade: the endpoint is announced once, scraped,
//! and announced gone when it stops answering.
//!
//! The registry's retirement of a vanished service is driven through the loop
//! here too, for an empty registry and for one whose directory cannot be read.
//! Lemonade's change of id is not driven through the loop here; it is covered only
//! by the unit tests in `runner.rs`. The Docker source has no seam that works without a container
//! daemon, so its reconcile is covered only by the unit tests next to the shared
//! helper in `runner.rs`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rocm_dash_core::metrics::{Instance, InstanceStatus, Snapshot};
use rocm_dash_core::protocol::Event;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio::time::{timeout, timeout_at};

use rocm_dash_daemon::bench_ring::BenchRing;
use rocm_dash_daemon::runner::{self, RunnerOptions};
use rocm_dash_daemon::snapshot_ring::SnapshotRing;

const TICK: Duration = Duration::from_millis(200);
const INSTANCE_TICK: Duration = Duration::from_millis(400);
const DEADLINE: Duration = Duration::from_secs(20);
const SVC_ID: &str = "reconcile-svc";

// ── Tiny scriptable HTTP server ─────────────────────────────────────────────

/// A local HTTP server whose reply is chosen per request path. Setting `down`
/// makes it answer 503 to everything, which is how a test takes a service away
/// without closing the port.
struct MockServer {
    port: u16,
    down: Arc<AtomicBool>,
}

async fn spawn_server(respond: impl Fn(&str) -> String + Send + Sync + 'static) -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let down = Arc::new(AtomicBool::new(false));
    let respond = Arc::new(respond);
    let down_for_task = Arc::clone(&down);
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let respond = Arc::clone(&respond);
            let down = Arc::clone(&down_for_task);
            tokio::spawn(async move {
                let mut buf = vec![0u8; 4096];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/")
                    .to_string();
                let reply = if down.load(Ordering::Relaxed) {
                    http(503, "text/plain", "")
                } else {
                    respond(&path)
                };
                let _ = sock.write_all(reply.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    MockServer { port, down }
}

fn http(status: u16, content_type: &str, body: &str) -> String {
    let reason = if status == 200 { "OK" } else { "Unavailable" };
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// A vLLM `/metrics` body with a fixed token counter, one running request and a
/// 25% KV cache.
fn vllm_metrics() -> String {
    http(
        200,
        "text/plain; version=0.0.4",
        "\
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model=\"mock\"} 1
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model=\"mock\"} 0
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model=\"mock\"} 0.25
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model=\"mock\"} 100
",
    )
}

// ── Runner plumbing ─────────────────────────────────────────────────────────

fn spawn_runner(opts: RunnerOptions) -> (broadcast::Receiver<Event>, tokio::task::JoinHandle<()>) {
    let (tx, rx) = broadcast::channel::<Event>(1024);
    let ring = Arc::new(Mutex::new(SnapshotRing::new(16)));
    let bench_ring = Arc::new(Mutex::new(BenchRing::new(4)));
    let handle = tokio::spawn(runner::run_loop(
        Some(TICK),
        tx,
        ring,
        bench_ring,
        None,
        opts,
    ));
    (rx, handle)
}

fn managed_opts(services_dir: std::path::PathBuf) -> RunnerOptions {
    RunnerOptions {
        services_dir: Some(services_dir),
        discovery_tick: TICK,
        instance_tick: INSTANCE_TICK,
        gpu_tick: TICK,
        ..RunnerOptions::default()
    }
}

fn lemonade_opts(port: u16) -> RunnerOptions {
    RunnerOptions {
        enable_lemonade: true,
        lemonade_host: "127.0.0.1".into(),
        lemonade_port: port,
        discovery_tick: TICK,
        instance_tick: INSTANCE_TICK,
        gpu_tick: TICK,
        ..RunnerOptions::default()
    }
}

/// Write the service's registry record. The record is written to a sibling temp
/// file and renamed into place, so the discovery tick can never read half a
/// record: a torn read would look like the service vanishing and coming back,
/// which is exactly what the tests below assert does not happen.
fn write_record(services_dir: &std::path::Path, port: u16) {
    std::fs::create_dir_all(services_dir).unwrap();
    let json = format!(
        r#"{{"service_id":"{SVC_ID}","engine":"vllm","model_ref":"Reconcile/1B","canonical_model_id":"Reconcile/1B","host":"127.0.0.1","port":{port},"status":"ready","created_at_unix_ms":0}}"#
    );
    // The registry only reads `*.json`, so the `.tmp` suffix keeps the
    // half-written file invisible to it.
    let tmp = services_dir.join(format!("{SVC_ID}.json.tmp"));
    std::fs::write(&tmp, json).unwrap();
    std::fs::rename(&tmp, services_dir.join(format!("{SVC_ID}.json"))).unwrap();
}

/// Read events until `pick` returns a value, or fail at the deadline.
async fn next_event<T>(
    rx: &mut broadcast::Receiver<Event>,
    pick: impl Fn(&Event) -> Option<T>,
) -> T {
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        assert!(
            !remaining.is_zero(),
            "no matching event within {DEADLINE:?}"
        );
        let Ok(received) = timeout(remaining, rx.recv()).await else {
            panic!("no matching event within {DEADLINE:?}");
        };
        match received {
            Ok(ev) => {
                if let Some(v) = pick(&ev) {
                    return v;
                }
            }
            Err(broadcast::error::RecvError::Lagged(_)) => {}
            Err(broadcast::error::RecvError::Closed) => panic!("broadcast channel closed"),
        }
    }
}

async fn next_snapshot(rx: &mut broadcast::Receiver<Event>) -> Snapshot {
    next_event(rx, |ev| match ev {
        Event::Snapshot(snap) => Some(snap.clone()),
        _ => None,
    })
    .await
}

fn instance<'a>(snap: &'a Snapshot, id: &str) -> Option<&'a Instance> {
    snap.instances.iter().find(|i| i.container_id == id)
}

fn approx(value: Option<f32>, expected: f32) -> bool {
    value.is_some_and(|v| (v - expected).abs() < 1e-3)
}

/// The mock's `0.25` KV-cache fraction, which the collector reports as a percent.
const KV_PCT: f32 = 25.0;

/// Snapshots to keep watching after an event of interest, so an announcement
/// that trails it by a tick or two is still seen.
const SETTLE_SNAPSHOTS: usize = 3;

/// Read snapshots until `pick` returns a value. The deadline is checked before
/// each snapshot, so a condition that never holds fails instead of looping on
/// fresh snapshots forever. Each wait for the next snapshot still has its own
/// [`DEADLINE`], so if the runner stops broadcasting, the failure can come up to
/// one further [`DEADLINE`] after the first.
async fn snapshot_where<T>(
    rx: &mut broadcast::Receiver<Event>,
    pick: impl Fn(&Snapshot) -> Option<T>,
) -> T {
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no snapshot satisfied the condition within {DEADLINE:?}"
        );
        let snap = next_snapshot(rx).await;
        if let Some(v) = pick(&snap) {
            return v;
        }
    }
}

/// Feed events to `step` until it returns true, under ONE deadline for the whole
/// observation. A loop that calls [`next_event`] repeatedly gets a fresh
/// deadline per call, so a runner that keeps broadcasting events the loop does
/// not count would spin forever; this fails at [`DEADLINE`] instead, naming what
/// was being waited for and, through `progress`, how far it got.
async fn observe<S>(
    rx: &mut broadcast::Receiver<Event>,
    waiting_for: &str,
    state: &mut S,
    mut step: impl FnMut(&mut S, &Event) -> bool,
    progress: impl Fn(&S) -> String,
) {
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        let Ok(received) = timeout_at(deadline, rx.recv()).await else {
            panic!(
                "timed out after {DEADLINE:?} waiting for {waiting_for}; seen so far: {}",
                progress(state)
            );
        };
        match received {
            Ok(ev) => {
                if step(state, &ev) {
                    return;
                }
            }
            Err(broadcast::error::RecvError::Lagged(_)) => {}
            Err(broadcast::error::RecvError::Closed) => {
                panic!("broadcast channel closed while waiting for {waiting_for}")
            }
        }
    }
}

// ── Managed-service registry ────────────────────────────────────────────────

/// Rediscovery runs every tick but the scrape only every second one, so on the
/// ticks without a scrape the instance's KV cache and request counts survive
/// only because the discovery merge carries them over from the previous state.
#[tokio::test]
async fn managed_rediscovery_keeps_live_scrape_fields_between_scrapes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");
    let server = spawn_server(|_| vllm_metrics()).await;
    write_record(&services_dir, server.port);
    let (mut rx, _runner) = spawn_runner(managed_opts(services_dir));

    snapshot_where(&mut rx, |snap| {
        instance(snap, SVC_ID)
            .filter(|i| approx(i.kv_cache_usage_pct, KV_PCT))
            .map(|_| ())
    })
    .await;

    for n in 0..10 {
        let snap = next_snapshot(&mut rx).await;
        let inst = instance(&snap, SVC_ID)
            .unwrap_or_else(|| panic!("snapshot {n} after the first scrape lost the instance"));
        assert!(
            approx(inst.kv_cache_usage_pct, KV_PCT),
            "snapshot {n}: kv cache was dropped on a tick without a scrape: {:?}",
            inst.kv_cache_usage_pct
        );
        assert_eq!(
            inst.running_reqs,
            Some(1),
            "snapshot {n}: running requests were dropped on a tick without a scrape"
        );
    }
}

/// A service is announced as discovered once, however many ticks rediscover it.
#[tokio::test]
async fn managed_service_is_announced_once_across_rediscovery() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");
    let server = spawn_server(|_| vllm_metrics()).await;
    write_record(&services_dir, server.port);
    let (mut rx, _runner) = spawn_runner(managed_opts(services_dir));

    // (announcements of the service, snapshots that contain it)
    let mut seen = (0usize, 0usize);
    observe(
        &mut rx,
        "8 snapshots containing the managed service",
        &mut seen,
        |seen, ev| {
            match ev {
                Event::InstanceDiscovered(inst) if inst.container_id == SVC_ID => seen.0 += 1,
                Event::Snapshot(snap) if instance(snap, SVC_ID).is_some() => seen.1 += 1,
                _ => {}
            }
            seen.1 >= 8
        },
        |seen| {
            format!(
                "{} snapshots containing it, {} announcements of it",
                seen.1, seen.0
            )
        },
    )
    .await;
    let (announced, _) = seen;
    assert_eq!(
        announced, 1,
        "the service must be announced exactly once, not once per discovery tick"
    );
}

/// What the stream showed from the moment a service's record changed port.
#[derive(Default)]
struct Change {
    /// The instance as first seen on its new port.
    moved: Option<Instance>,
    snapshots_after_move: usize,
    /// Every discovered/gone announcement for the service in the window.
    announcements: Vec<&'static str>,
}

/// The registry's port is the authority: when it changes under the same service
/// id, the live fields scraped from the old port must not leak onto the new
/// endpoint, and the change must not be announced as a new service.
#[tokio::test]
async fn managed_endpoint_change_starts_the_instance_clean() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");
    let healthy = spawn_server(|_| vllm_metrics()).await;
    // The new endpoint never answers a scrape, so any live field on the instance
    // when it first appears on the new port was carried over from the old one.
    let silent = spawn_server(|_| http(503, "text/plain", "")).await;
    write_record(&services_dir, healthy.port);
    let (mut rx, _runner) = spawn_runner(managed_opts(services_dir.clone()));

    snapshot_where(&mut rx, |snap| {
        instance(snap, SVC_ID)
            .filter(|i| approx(i.kv_cache_usage_pct, KV_PCT))
            .map(|_| ())
    })
    .await;

    write_record(&services_dir, silent.port);

    // Watch the stream from the moment the record changes until the instance has
    // been seen on its new port and a few more snapshots have followed, under one
    // deadline. Any announcement or retirement of the service in that window
    // means the change was treated as a different service rather than the same
    // one moving.
    let mut change = Change::default();
    observe(
        &mut rx,
        "the managed service to appear on its new port",
        &mut change,
        |change, ev| {
            match ev {
                Event::InstanceDiscovered(inst) if inst.container_id == SVC_ID => {
                    change.announcements.push("discovered");
                }
                Event::InstanceGone { container_id } if container_id == SVC_ID => {
                    change.announcements.push("gone");
                }
                Event::Snapshot(snap) if change.moved.is_none() => {
                    change.moved = instance(snap, SVC_ID)
                        .filter(|i| i.port == Some(silent.port))
                        .cloned();
                }
                Event::Snapshot(_) => change.snapshots_after_move += 1,
                _ => {}
            }
            change.snapshots_after_move >= SETTLE_SNAPSHOTS
        },
        |change| {
            format!(
                "moved={}, {} snapshots after the move, announcements {:?}",
                change.moved.is_some(),
                change.snapshots_after_move,
                change.announcements
            )
        },
    )
    .await;

    assert!(
        change.announcements.is_empty(),
        "an endpoint change must not announce the service as discovered or gone: {:?}",
        change.announcements
    );
    let moved = change.moved.expect("observe returned before the move");
    assert_eq!(
        moved.kv_cache_usage_pct, None,
        "the old endpoint's kv cache leaked onto the new one"
    );
    assert_eq!(
        moved.running_reqs, None,
        "the old endpoint's request count leaked onto the new one"
    );
}

// ── Managed-service registry: what the loop does when the registry changes ──

/// A running loop with one managed service that has been scraped at least once,
/// so the tests below start from an instance that carries live fields.
struct LiveManaged {
    rx: broadcast::Receiver<Event>,
    services_dir: std::path::PathBuf,
    // Held so the registry directory and the mock endpoint outlive the test body.
    _tmp: tempfile::TempDir,
    _server: MockServer,
    _runner: tokio::task::JoinHandle<()>,
}

async fn start_live_managed() -> LiveManaged {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");
    let server = spawn_server(|_| vllm_metrics()).await;
    write_record(&services_dir, server.port);
    let (mut rx, runner) = spawn_runner(managed_opts(services_dir.clone()));
    snapshot_where(&mut rx, |snap| {
        instance(snap, SVC_ID)
            .filter(|i| approx(i.kv_cache_usage_pct, KV_PCT))
            .map(|_| ())
    })
    .await;
    LiveManaged {
        rx,
        services_dir,
        _tmp: tmp,
        _server: server,
        _runner: runner,
    }
}

/// Wait until the service is announced gone and a snapshot no longer lists it:
/// the announcement and the snapshot are the two things a dashboard shows.
async fn assert_service_is_retired(rx: &mut broadcast::Receiver<Event>) {
    next_event(rx, |ev| match ev {
        Event::InstanceGone { container_id } if container_id == SVC_ID => Some(()),
        _ => None,
    })
    .await;
    snapshot_where(rx, |snap| instance(snap, SVC_ID).is_none().then_some(())).await;
}

/// Replace the registry directory with a regular file of the same name, so
/// listing it fails with something other than "not found". Portable: it needs no
/// permission change, so it behaves the same for a privileged user and on Windows.
fn replace_registry_dir_with_a_file(services_dir: &std::path::Path) {
    std::fs::remove_dir_all(services_dir).unwrap();
    std::fs::write(services_dir, b"not a directory").unwrap();
}

/// A registry that is confirmed empty (the directory is there and holds no
/// record) retires the service it knew about.
#[tokio::test]
async fn managed_confirmed_empty_registry_retires_the_known_instance() {
    let mut live = start_live_managed().await;

    std::fs::remove_file(live.services_dir.join(format!("{SVC_ID}.json"))).unwrap();

    assert_service_is_retired(&mut live.rx).await;
}

/// A registry directory that does not exist is also confirmed empty: it is
/// created the first time something is served, so its absence means no services.
#[tokio::test]
async fn managed_absent_registry_dir_retires_the_known_instance() {
    let mut live = start_live_managed().await;

    std::fs::remove_dir_all(&live.services_dir).unwrap();

    assert_service_is_retired(&mut live.rx).await;
}

/// A registry directory that cannot be listed retires the known service today,
/// exactly as an empty one does: the loop cannot tell a failed read from a
/// registry with nothing in it. This pins that defect; the next commit makes the
/// read failure leave the instance alone and flips this test.
#[tokio::test]
async fn managed_unlistable_registry_dir_retires_the_known_instance_today() {
    let mut live = start_live_managed().await;

    replace_registry_dir_with_a_file(&live.services_dir);

    assert_service_is_retired(&mut live.rx).await;
}

/// Restores a directory's permissions on drop, so the temp dir can be cleaned
/// up even when the test fails part-way.
#[cfg(unix)]
struct RestorePermissions(std::path::PathBuf);

#[cfg(unix)]
impl Drop for RestorePermissions {
    fn drop(&mut self) {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
    }
}

/// The same defect through a permission error rather than a path error. Unix
/// only, because Windows has no mode bits to drop; skipped when the process can
/// still list the directory with its read bit removed (running as root).
#[cfg(unix)]
#[tokio::test]
async fn managed_unreadable_registry_dir_retires_the_known_instance_today() {
    use std::os::unix::fs::PermissionsExt;
    let mut live = start_live_managed().await;
    let _restore = RestorePermissions(live.services_dir.clone());

    std::fs::set_permissions(&live.services_dir, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::read_dir(&live.services_dir).is_ok() {
        eprintln!("skipped: this process can list a mode-000 directory (running as root)");
        return;
    }

    assert_service_is_retired(&mut live.rx).await;
}

// ── Lemonade endpoint ───────────────────────────────────────────────────────

fn lemonade_responder(path: &str) -> String {
    match path {
        "/api/v1/health" => http(
            200,
            "application/json",
            r#"{"status":"ok","model_loaded":"Mock-Lemonade-1B"}"#,
        ),
        "/api/v1/stats" => http(200, "application/json", r#"{"tokens_per_second":33.5}"#),
        _ => http(404, "text/plain", ""),
    }
}

/// A reachable Lemonade endpoint is announced once with its health-reported
/// model, scraped for its rate, and announced gone when the endpoint stops
/// answering, after which it leaves the snapshots.
#[tokio::test]
async fn lemonade_endpoint_is_announced_scraped_and_announced_gone() {
    let server = spawn_server(lemonade_responder).await;
    let id = format!("lemonade-127.0.0.1-{}", server.port);
    let (mut rx, _runner) = spawn_runner(lemonade_opts(server.port));

    let announced = next_event(&mut rx, |ev| match ev {
        Event::InstanceDiscovered(inst) if inst.container_id == id => Some(inst.clone()),
        _ => None,
    })
    .await;
    assert_eq!(announced.model_name, "Mock-Lemonade-1B");
    assert_eq!(announced.status, InstanceStatus::Ready);
    assert_eq!(announced.port, Some(server.port));

    // Scraped: the stats endpoint's rate reaches the snapshot.
    snapshot_where(&mut rx, |snap| {
        instance(snap, &id)
            .and_then(|i| i.gen_tps)
            .filter(|tps| (tps - 33.5).abs() < 1e-6)
            .map(|_| ())
    })
    .await;

    // Rediscovery does not announce it again. One deadline covers the whole
    // watch, so a runner that stops broadcasting snapshots fails here instead of
    // each wait starting its own.
    let mut snapshots = 0usize;
    observe(
        &mut rx,
        "6 snapshots with no second announcement of the Lemonade endpoint",
        &mut snapshots,
        |snapshots, ev| {
            match ev {
                Event::InstanceDiscovered(inst) if inst.container_id == id => {
                    panic!("a rediscovered Lemonade endpoint was announced a second time")
                }
                Event::Snapshot(_) => *snapshots += 1,
                _ => {}
            }
            *snapshots >= 6
        },
        |snapshots| format!("{snapshots} snapshots"),
    )
    .await;

    server.down.store(true, Ordering::Relaxed);
    next_event(&mut rx, |ev| match ev {
        Event::InstanceGone { container_id } if *container_id == id => Some(()),
        _ => None,
    })
    .await;
    snapshot_where(&mut rx, |snap| instance(snap, &id).is_none().then_some(())).await;
}
