// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What the run loop logs about the managed-service registry's health.
//!
//! A read failure that lasts many discovery ticks is logged once, its recovery
//! once, and a second outage once more: one line per streak, not one per tick.
//!
//! This is its own test binary, with a single test, on purpose. `tracing` caches
//! whether a callsite is enabled the first time any thread reaches it; when a
//! test thread installs a per-thread subscriber after another thread has already
//! run the same `warn!` with none installed, the cached answer drops the event
//! and the capture comes up empty. In `discovery_reconcile.rs` the tests share a
//! process, so that is what happened there. Here nothing else logs first.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rocm_dash_core::protocol::Event;
use tokio::sync::broadcast;
use tokio::time::timeout;

use rocm_dash_daemon::bench_ring::BenchRing;
use rocm_dash_daemon::runner::{self, RunnerOptions};
use rocm_dash_daemon::snapshot_ring::SnapshotRing;

const TICK: Duration = Duration::from_millis(100);
const DEADLINE: Duration = Duration::from_secs(20);
const SVC_ID: &str = "logging-svc";
/// Snapshots to watch across a streak: several discovery ticks.
const STREAK_SNAPSHOTS: usize = 8;
const FAILED: &str = "managed-service registry could not be read";
const RECOVERED: &str = "managed-service registry is readable again";

/// A shared buffer a `tracing` subscriber writes to, so the test can count what
/// the loop logged.
#[derive(Clone, Default)]
struct LogCapture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for LogCapture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl LogCapture {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }

    fn lines_containing(&self, needle: &str) -> usize {
        self.text()
            .lines()
            .filter(|line| line.contains(needle))
            .count()
    }
}

fn write_record(services_dir: &std::path::Path) {
    std::fs::create_dir_all(services_dir).unwrap();
    // Nothing listens on the port, so the service is never scraped: this test is
    // about discovery only. A rename keeps the tick from reading half a record.
    let json = format!(
        r#"{{"service_id":"{SVC_ID}","engine":"vllm","model_ref":"Logging/1B","canonical_model_id":"Logging/1B","host":"127.0.0.1","port":9,"status":"ready","created_at_unix_ms":0}}"#
    );
    let tmp = services_dir.join(format!("{SVC_ID}.json.tmp"));
    std::fs::write(&tmp, json).unwrap();
    std::fs::rename(&tmp, services_dir.join(format!("{SVC_ID}.json"))).unwrap();
}

/// Replace the registry directory with a regular file, so listing it fails.
fn break_registry(services_dir: &std::path::Path) {
    std::fs::remove_dir_all(services_dir).unwrap();
    std::fs::write(services_dir, b"not a directory").unwrap();
}

/// Wait for `count` more snapshots, under one deadline.
async fn watch(rx: &mut broadcast::Receiver<Event>, count: usize) {
    let deadline = tokio::time::Instant::now() + DEADLINE;
    let mut seen = 0;
    while seen < count {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        match timeout(remaining, rx.recv()).await {
            Ok(Ok(Event::Snapshot(_))) => seen += 1,
            Ok(Ok(_) | Err(broadcast::error::RecvError::Lagged(_))) => {}
            Ok(Err(broadcast::error::RecvError::Closed)) => panic!("broadcast channel closed"),
            Err(elapsed) => {
                panic!("only {seen} of {count} snapshots within {DEADLINE:?} ({elapsed})")
            }
        }
    }
}

/// The default subscriber is per thread and `#[tokio::test]` runs the loop on
/// the test's own thread, so the capture sees the loop's output.
#[tokio::test]
async fn registry_read_failure_is_logged_once_per_streak() {
    let capture = LogCapture::default();
    let writer = capture.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");
    write_record(&services_dir);
    let (tx, mut rx) = broadcast::channel::<Event>(1024);
    let _runner = tokio::spawn(runner::run_loop(
        Some(TICK),
        tx,
        Arc::new(Mutex::new(SnapshotRing::new(16))),
        Arc::new(Mutex::new(BenchRing::new(4))),
        None,
        RunnerOptions {
            services_dir: Some(services_dir.clone()),
            discovery_tick: TICK,
            instance_tick: TICK * 2,
            gpu_tick: TICK,
            ..RunnerOptions::default()
        },
    ));
    watch(&mut rx, 3).await;
    assert_eq!(
        capture.lines_containing(FAILED),
        0,
        "a readable registry logs nothing: {}",
        capture.text()
    );

    break_registry(&services_dir);
    watch(&mut rx, STREAK_SNAPSHOTS).await;
    assert_eq!(
        capture.lines_containing(FAILED),
        1,
        "a streak of failed ticks must log one warning: {}",
        capture.text()
    );
    assert_eq!(capture.lines_containing(RECOVERED), 0);

    std::fs::remove_file(&services_dir).unwrap();
    write_record(&services_dir);
    watch(&mut rx, STREAK_SNAPSHOTS).await;
    assert_eq!(capture.lines_containing(RECOVERED), 1, "{}", capture.text());
    assert_eq!(capture.lines_containing(FAILED), 1);

    break_registry(&services_dir);
    watch(&mut rx, STREAK_SNAPSHOTS).await;
    assert_eq!(
        capture.lines_containing(FAILED),
        2,
        "a second streak is a new warning: {}",
        capture.text()
    );
    assert_eq!(capture.lines_containing(RECOVERED), 1);
}
