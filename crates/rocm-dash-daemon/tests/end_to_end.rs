// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! End-to-end integration test: spawn the runner with a CSV, drive a broadcast
//! subscriber from the same process, and assert Snapshot + BenchmarkRowsAppended
//! events arrive.
//!
//! Exercises the daemon's public library surface (`rocm_dash_daemon::*`).

use std::time::Duration;

use rocm_dash_core::protocol::Event;
use tokio::sync::broadcast;
use tokio::time::timeout;

use std::sync::{Arc, Mutex};

use rocm_dash_daemon::bench_ring::BenchRing;
use rocm_dash_daemon::runner;
use rocm_dash_daemon::snapshot_ring::SnapshotRing;

const HEADER: &str = "cell,run,wall_s,n_requests,main_prompt_n,prompt_tokens,prompt_tps,\
    completion_tokens,gen_tps,max_running_reqs,max_waiting_reqs,out_chars,rc,\
    assertion_pass,assertion_fail_count,assertion_summary,quality_score,\
    judge_pass_fail,judge_model,model,endpoint,tp,pp,dtype,max_num_seqs,\
    attention_backend,concurrency,extra_args,safety_pass,safety_violations\n";
const ROW: &str = "O-arch,1,42.3,8,512,4096,1240.5,2048,68.2,8,2,8192,0,true,0,all-pass,\
    4.5,pass,claude,deepseek-r1,http://vllm:8000,8,1,fp8,32,triton,1,,true,0\n";

#[tokio::test]
async fn runner_broadcasts_snapshots_and_bench_rows() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "rocm-dash-runner-test-{}-{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, HEADER).unwrap();

    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let path_for_runner = path.clone();
    let _handle = tokio::spawn(async move {
        // 250ms tick — faster than 1Hz so the test doesn't stall.
        let opts = runner::RunnerOptions {
            bench_csv: Some(path_for_runner),
            ..Default::default()
        };
        let ring = Arc::new(Mutex::new(SnapshotRing::new(8)));
        let bench_ring = Arc::new(Mutex::new(BenchRing::new(8)));
        runner::run_loop(
            Some(Duration::from_millis(250)),
            tx,
            ring,
            bench_ring,
            None,
            opts,
        )
        .await;
    });

    // First Snapshot should land within ~500ms.
    let first = timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("snapshot timeout")
        .expect("recv");
    assert!(matches!(first, Event::Snapshot(_)));

    // No bench rows in the file yet — drain returns empty, no event.
    // Now append a row and expect a BenchmarkRowsAppended within the next tick.
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(ROW.as_bytes()).unwrap();
    }

    let mut saw_rows = false;
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while std::time::Instant::now() < deadline && !saw_rows {
        let ev = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("event timeout")
            .expect("recv");
        if let Event::BenchmarkRowsAppended { rows } = ev {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].cell, "O-arch");
            assert_eq!(rows[0].run, 1);
            saw_rows = true;
        }
    }
    assert!(saw_rows, "never saw BenchmarkRowsAppended");

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn ring_accumulates_snapshots_for_replay() {
    use std::time::Instant;

    let (tx, _rx) = broadcast::channel::<Event>(64);
    let ring = Arc::new(Mutex::new(SnapshotRing::new(4)));
    let bench_ring = Arc::new(Mutex::new(BenchRing::new(4)));
    let runner_ring = ring.clone();
    let runner_bench_ring = bench_ring.clone();
    let _handle = tokio::spawn(async move {
        let opts = runner::RunnerOptions::default();
        runner::run_loop(
            Some(Duration::from_millis(100)),
            tx,
            runner_ring,
            runner_bench_ring,
            None,
            opts,
        )
        .await;
    });

    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let n = ring.lock().unwrap().len();
        if n >= 3 {
            break;
        }
        assert!(
            Instant::now() <= deadline,
            "ring never reached 3 snapshots; len={n}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    tokio::time::sleep(Duration::from_millis(800)).await;
    let snaps = ring.lock().unwrap().snapshot();
    assert!(snaps.len() <= 4, "ring exceeded cap: {}", snaps.len());
    assert!(snaps.len() >= 3, "ring underfull: {}", snaps.len());

    for w in snaps.windows(2) {
        assert!(w[0].timestamp <= w[1].timestamp);
    }
}

#[tokio::test]
async fn bench_ring_accumulates_rows_for_replay() {
    use std::time::Instant;

    let mut path = std::env::temp_dir();
    path.push(format!(
        "rocm-dash-ring-test-{}-{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, HEADER).unwrap();

    let (tx, _rx) = broadcast::channel::<Event>(64);
    let ring = Arc::new(Mutex::new(SnapshotRing::new(8)));
    let bench_ring = Arc::new(Mutex::new(BenchRing::new(8)));
    let runner_ring = ring.clone();
    let runner_bench_ring = bench_ring.clone();
    let path_for_runner = path.clone();
    let _handle = tokio::spawn(async move {
        let opts = runner::RunnerOptions {
            bench_csv: Some(path_for_runner),
            ..Default::default()
        };
        runner::run_loop(
            Some(Duration::from_millis(100)),
            tx,
            runner_ring,
            runner_bench_ring,
            None,
            opts,
        )
        .await;
    });

    // Let the runner take at least one tick so the tailer is initialized.
    tokio::time::sleep(Duration::from_millis(200)).await;

    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(ROW.as_bytes()).unwrap();
    }

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let n = bench_ring.lock().unwrap().len();
        if n >= 1 {
            break;
        }
        assert!(
            Instant::now() <= deadline,
            "bench ring never accumulated a row; len={n}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let rows = bench_ring.lock().unwrap().snapshot();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].cell, "O-arch");
    assert_eq!(rows[0].run, 1);

    let _ = std::fs::remove_file(&path);
}

/// Verify the unix socket is created with mode 0o600 (not world-accessible).
#[cfg(unix)]
#[tokio::test]
async fn socket_is_created_with_restricted_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().expect("tempdir");
    let sock_path = dir.path().join("test.sock");
    let listen = format!("unix:{}", sock_path.display());

    let handle = tokio::spawn({
        let listen = listen.clone();
        async move {
            // Run until aborted; ignore the resulting error on abort.
            let _ =
                rocm_dash_daemon::run(&listen, rocm_dash_daemon::runner::RunnerOptions::default())
                    .await;
        }
    });

    // Poll until the socket exists *and* its mode is 0o600. Polling for
    // existence alone is racy: set_permissions runs after UnixListener::bind,
    // so the file can appear before the mode is restricted.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mode = loop {
        if let Ok(meta) = std::fs::metadata(&sock_path) {
            let m = meta.permissions().mode() & 0o777;
            if m == 0o600 {
                break m;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "daemon did not create socket with mode 0o600 within 5 s"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };

    handle.abort();
    // Await the handle so the task is fully cancelled and has released the
    // socket before the TempDir destructor removes the directory.
    let _ = handle.await;
    assert_eq!(mode, 0o600, "socket must not be group- or world-accessible");
}

/// Regression: a scrape warning must persist on the ticks BETWEEN scrapes.
/// The vLLM scrape runs on the slower `instance_tick` cadence, so rebuilding
/// `warnings` per tick made the header ⚠ badge appear and vanish alternately.
#[tokio::test]
async fn scrape_warning_persists_between_scrape_ticks() {
    let dir = tempfile::tempdir().unwrap();
    // A "running" managed vLLM service on a port nothing listens on: every
    // scrape fails, so the warning must be present on EVERY snapshot.
    std::fs::write(
        dir.path().join("svc.json"),
        r#"{"service_id":"svc-dead","engine":"vllm","model_ref":"m","canonical_model_id":"m",
            "host":"127.0.0.1","port":1,"endpoint_url":"http://127.0.0.1:1/v1","mode":"managed",
            "status":"running","created_at_unix_ms":1}"#,
    )
    .unwrap();

    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let services_dir = dir.path().to_path_buf();
    let handle = tokio::spawn(async move {
        let opts = runner::RunnerOptions {
            services_dir: Some(services_dir),
            ..Default::default()
        };
        runner::run_loop(
            Some(Duration::from_millis(100)),
            tx,
            Arc::new(Mutex::new(SnapshotRing::new(32))),
            Arc::new(Mutex::new(BenchRing::new(4))),
            None,
            opts,
        )
        .await;
    });

    // Collect snapshots after the first failed scrape and require an unbroken
    // run of warnings — the pre-fix behavior alternated warned/clean.
    let mut seen_warned = false;
    let mut checked = 0;
    while checked < 6 {
        let ev = timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event timeout")
            .expect("recv");
        let Event::Snapshot(snap) = ev else { continue };
        let warned = snap.warnings.iter().any(|w| w.starts_with("vllm scrape:"));
        if !seen_warned {
            seen_warned = warned;
            continue;
        }
        assert!(
            warned,
            "vllm scrape warning dropped on a non-scrape tick: {:?}",
            snap.warnings
        );
        checked += 1;
    }

    handle.abort();
    let _ = handle.await;
}

/// Regression: a slow `amd-smi` startup must NOT delay managed-service discovery.
///
/// GPU detection (`amd-smi version` + the first `system_info()`) can take ~15s on
/// real hardware. It used to run inline before the loop started, so the very
/// first managed-service discovery + snapshot were blocked behind it and an
/// already-running model appeared in the dashboard ~15-20s late (while
/// `rocm services` already listed it). Detection now runs off the critical path,
/// so a `ready` service surfaces on the first discovery tick while GPU detection
/// is still in flight.
///
/// This asserts on ORDERING rather than wall-clock timing (delivery to an
/// in-process subscriber can be starved while the fake subprocess runs, which
/// would make a pure "arrived within Ns" check flaky): the instance MUST appear
/// in a snapshot whose `gpu_system_info` is still `None`. Post-fix that window
/// exists (many ~100ms ticks before the 4s fake detection lands). Pre-fix it
/// cannot: the loop only starts ticking after detection completes, so the very
/// first snapshot carrying the instance already has `gpu_system_info == Some`.
///
/// On hosts without `/dev/kfd` the fake is never invoked (detection
/// short-circuits to "no GPU", leaving `gpu_system_info == None` forever), which
/// is exactly the case where this bug cannot occur — the assertion still holds.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_gpu_detection_does_not_delay_service_discovery() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();

    // A fake amd-smi that is deliberately slow: whatever the subcommand, sleep
    // well past the discovery deadline, then emit trivially-valid JSON so both
    // `version` and the `system_info` `--json` calls "succeed".
    let fake = dir.path().join("amd-smi-slow");
    std::fs::write(&fake, "#!/bin/sh\nsleep 4\necho '{}'\n").unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();

    // A ready managed vLLM service on disk — the authority `rocm services` reads.
    std::fs::write(
        dir.path().join("svc.json"),
        r#"{"service_id":"vllm-ready-fast","engine":"vllm","model_ref":"m","canonical_model_id":"m",
            "host":"127.0.0.1","port":11435,"endpoint_url":"http://127.0.0.1:11435/v1",
            "mode":"managed","status":"ready","created_at_unix_ms":1}"#,
    )
    .unwrap();

    // Roomy channel so buffered early snapshots are never dropped (Lagged) even
    // if the subscriber is briefly starved while the fake subprocess runs.
    let (tx, mut rx) = broadcast::channel::<Event>(512);
    let services_dir = dir.path().to_path_buf();
    let fake_bin = fake.into_os_string();
    let handle = tokio::spawn(async move {
        let opts = runner::RunnerOptions {
            services_dir: Some(services_dir),
            amd_smi_binary: Some(fake_bin),
            disable_vllm_metrics: true,
            ..Default::default()
        };
        runner::run_loop(
            Some(Duration::from_millis(100)),
            tx,
            Arc::new(Mutex::new(SnapshotRing::new(512))),
            Arc::new(Mutex::new(BenchRing::new(4))),
            None,
            opts,
        )
        .await;
    });

    // Walk snapshots in order. The FIRST snapshot that carries the instance
    // decides: post-fix it has no GPU info yet (pass); pre-fix it already has
    // GPU info because detection ran to completion before the loop started.
    // The generous timeout covers detection + any subscriber starvation.
    let mut surfaced_before_gpu = None;
    let overall = Duration::from_secs(15);
    loop {
        let ev = match timeout(overall, rx.recv()).await {
            Ok(Ok(ev)) => ev,
            Ok(Err(broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(broadcast::error::RecvError::Closed)) => break,
            Err(_) => break,
        };
        if let Event::Snapshot(snap) = ev
            && snap
                .instances
                .iter()
                .any(|i| i.container_id == "vllm-ready-fast")
        {
            surfaced_before_gpu = Some(snap.gpu_system_info.is_none());
            break;
        }
    }

    handle.abort();
    let _ = handle.await;

    match surfaced_before_gpu {
        Some(true) => {} // instance surfaced while GPU detection was still in flight
        Some(false) => panic!(
            "managed service only surfaced after GPU detection completed \
             (the ~15-20s dashboard startup lag regressed)"
        ),
        None => panic!("ready managed service never surfaced in a snapshot"),
    }
}
