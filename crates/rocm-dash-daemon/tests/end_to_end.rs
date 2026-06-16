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
        if Instant::now() > deadline {
            panic!("ring never reached 3 snapshots; len={n}");
        }
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
        if Instant::now() > deadline {
            panic!("bench ring never accumulated a row; len={n}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let rows = bench_ring.lock().unwrap().snapshot();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].cell, "O-arch");
    assert_eq!(rows[0].run, 1);

    let _ = std::fs::remove_file(&path);
}
