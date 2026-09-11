// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm bench load`.
//!
//! These pin two things the command previously got wrong together: which chat
//! route the load generator posts to, and whether a run in which every request
//! failed is reported as a failure. Either alone is insufficient — the wrong
//! route was only invisible because the failures were swallowed.

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::MockServer;

use crate::E2eWorld;

/// Requests per cell. Small: these scenarios assert on reporting behaviour, not
/// on throughput accuracy, and the GPU lane pays real inference time for each.
const BENCH_REQUESTS: &str = "2";

/// Windowed `tpot_ms` the metrics mock is built to produce, in milliseconds.
///
/// Its TPOT histogram adds 0.4 s of `_sum` per 20 `_count` on every scrape, so
/// `Δsum/Δcount` is exactly 0.020 s however many scrapes the cell's window
/// spans. Asserting the value rather than just its sign is what makes the CSV
/// scenario pin the windowed arithmetic — a lifetime `sum/count` backfill or a
/// unit slip would also be "positive".
const EXPECTED_TPOT_MS: f64 = 20.0;

/// Deterministic path, inside the scenario's isolated root, that the CSV-content
/// scenario writes to via `--out` and reads back — so the `then` step can assert
/// on the emitted row without depending on the default `<data_dir>` layout.
fn bench_out_path(world: &E2eWorld) -> std::path::PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root for the benchmark output")
        .path()
        .join("bench-results.csv")
}

#[given("an endpoint that rejects every request")]
async fn setup_rejecting_endpoint(world: &mut E2eWorld) {
    let mock = MockServer::start_rejecting().await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some("TestModel/E2E-1B".to_string());
    world.mock = Some(mock);
}

#[given("a model is being served with a metrics endpoint")]
async fn setup_model_server_with_metrics(world: &mut E2eWorld) {
    // `start_with_metrics` exposes a vLLM-flavoured `/metrics` route whose
    // TTFT/TPOT histograms advance every scrape, so the bench cell's before/
    // after window measures a real per-output-token latency (not a flat one).
    let mock = MockServer::start_with_metrics("TestModel/E2E-1B").await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some("TestModel/E2E-1B".to_string());
    world.mock = Some(mock);
}

/// Benchmark the endpoint exactly as the CLI reports it — for a served model
/// that is the `/v1`-suffixed form printed by `rocm services list`.
#[when("the user benchmarks the served endpoint")]
async fn benchmark_served_endpoint(world: &mut E2eWorld) {
    let endpoint = world
        .endpoint
        .clone()
        .expect("no endpoint configured for the benchmark");
    run_bench(world, &endpoint, None);
}

/// Benchmark using the bare `scheme://host:port` form, dropping the API-root
/// suffix. Users type this because it is what the `--endpoint` help used to
/// show; it must reach the same place as the fuller form.
#[when("the user benchmarks the server using its plain host address")]
async fn benchmark_plain_host_address(world: &mut E2eWorld) {
    let endpoint = world
        .endpoint
        .clone()
        .expect("no endpoint configured for the benchmark");
    let plain = endpoint
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .trim_end_matches('/')
        .to_string();
    assert!(
        !plain.ends_with("/v1"),
        "the plain form must not keep the API-root suffix: {plain}"
    );
    run_bench(world, &plain, None);
}

/// Run one `rocm bench load` cell against `endpoint`, optionally recording the
/// row to `out`, and park the outcome on the world for the `then` steps.
///
/// Every scenario here runs the same single-cell shape; only the endpoint form
/// and whether the row is written to a known path differ, so they share one
/// argv builder rather than each keeping its own copy to drift.
fn run_bench(world: &mut E2eWorld, endpoint: &str, out: Option<&std::path::Path>) {
    let model = world
        .model_name
        .clone()
        .expect("no model configured for the benchmark");
    // Without `--out` the row lands inside the scenario's isolated data dir by
    // default; the explicit model avoids depending on the endpoint's
    // model-listing route, which the rejecting server deliberately fails.
    let mut args = vec![
        "bench",
        "load",
        "--endpoint",
        endpoint,
        "--model",
        &model,
        "--concurrency",
        "1",
        "--isl",
        "8",
        "--osl",
        "4",
        "--requests",
        BENCH_REQUESTS,
    ];
    let out = out.map(|path| {
        path.to_str()
            .expect("bench output path is not valid UTF-8")
            .to_string()
    });
    if let Some(out) = out.as_deref() {
        args.extend_from_slice(&["--out", out]);
    }

    let (stdout, stderr, rc) = crate::run_rocm(world, &args);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Benchmark the metrics-backed endpoint, writing the row to a known `--out`
/// file so the row's `engine`/`tpot_ms` columns can be asserted directly.
#[when("the user benchmarks the served endpoint recording results to a file")]
async fn benchmark_recording_results(world: &mut E2eWorld) {
    run_bench_recording(world);
}

fn run_bench_recording(world: &mut E2eWorld) {
    let endpoint = world
        .endpoint
        .clone()
        .expect("no endpoint configured for the benchmark");
    let out = bench_out_path(world);
    run_bench(world, &endpoint, Some(&out));
}

/// Leave behind the results file a build that never populated `engine` would
/// have written for this same cell.
///
/// Produced by running the benchmark once and then blanking the `engine` column
/// of the row it wrote, rather than by pasting a fixture: an append whose header
/// does not match byte-for-byte is refused outright, so a hand-written file
/// would decay into a header-mismatch failure that proves nothing about the
/// warning. Blanking is also exactly how a run against an endpoint with no
/// reachable `/metrics` leaves the column, which is the other way the split
/// happens.
#[given("earlier results for the same cell were recorded without an engine")]
async fn seed_results_recorded_without_an_engine(world: &mut E2eWorld) {
    run_bench_recording(world);
    let rc = world.cli_rc.expect("no command was run");
    assert!(
        rc == 0,
        "the seeding benchmark run failed (rc={rc}):\n{}{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );

    let path = bench_out_path(world);
    let csv = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read bench CSV {}: {e}", path.display()));
    let blanked = blank_engine_column(&csv);
    assert_ne!(
        blanked, csv,
        "the seeding run wrote no engine value to blank:\n{csv}"
    );
    std::fs::write(&path, blanked).expect("could not rewrite the seeded bench CSV");
}

/// Empty the `engine` field of every data row, locating it by header name so a
/// column reorder cannot silently blank the wrong one.
fn blank_engine_column(csv: &str) -> String {
    let mut lines = csv.lines();
    let header = lines.next().unwrap_or_else(|| panic!("bench CSV is empty"));
    let idx = header
        .split(',')
        .position(|c| c == "engine")
        .unwrap_or_else(|| panic!("no `engine` column in header: {header}"));

    let mut out = format!("{header}\n");
    for line in lines {
        let mut fields: Vec<&str> = line.split(',').collect();
        if let Some(field) = fields.get_mut(idx) {
            *field = "";
        }
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    out
}

#[then("the benchmark warns that the earlier rows group separately")]
async fn assert_engine_split_warning(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or("");
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let rc = world.cli_rc.expect("no command was run");
    // A split is worth saying out loud, but the run still measured what it was
    // asked to: warn, do not fail.
    assert!(
        rc == 0,
        "rocm bench load failed (rc={rc}):\n{stdout}{stderr}"
    );

    let warning = stderr
        .lines()
        .find(|line| line.contains("blank engine column"))
        .unwrap_or_else(|| {
            panic!("the run did not warn about the pre-existing blank-engine rows:\n{stderr}")
        });
    // Naming the cell and the file is what makes the warning actionable: the
    // Bench panel renders no `engine` column, so without them there is nothing
    // to connect a halved group to.
    assert!(
        warning.contains("bench-c1"),
        "the warning must name the affected cell:\n{warning}"
    );
    assert!(
        warning.contains("bench-results.csv"),
        "the warning must name the file holding both groups:\n{warning}"
    );
    assert!(
        stderr.contains("rotate"),
        "the warning must say what to do about it:\n{stderr}"
    );
}

#[then("the recorded benchmark row is labelled the vLLM engine with a per-output-token latency")]
async fn assert_row_engine_and_tpot(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or("");
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let rc = world.cli_rc.expect("no command was run");
    assert!(
        rc == 0,
        "rocm bench load failed (rc={rc}):\n{stdout}{stderr}"
    );

    let path = bench_out_path(world);
    let csv = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read bench CSV {}: {e}", path.display()));
    let mut lines = csv.lines();
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("bench CSV is empty:\n{csv}"));
    let data = lines
        .next()
        .unwrap_or_else(|| panic!("bench CSV has no data row:\n{csv}"));

    let cols: Vec<&str> = header.split(',').collect();
    let col = |name: &str| {
        let idx = cols
            .iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("no `{name}` column in header: {header}"));
        data.split(',').nth(idx).unwrap_or("")
    };

    assert_eq!(
        col("engine"),
        "vllm",
        "the emitted row must carry engine=vllm from the recognised /metrics scrape:\n{data}"
    );
    let tpot = col("tpot_ms");
    assert!(
        tpot.parse::<f64>()
            .is_ok_and(|v| (v - EXPECTED_TPOT_MS).abs() < 0.5),
        "the emitted row must carry the mock's windowed tpot_ms of {EXPECTED_TPOT_MS} ms, got {tpot:?}:\n{data}"
    );
}

#[then("the benchmark reports measured throughput")]
async fn assert_throughput_reported(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or("");
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let rc = world.cli_rc.expect("no command was run");
    assert!(
        rc == 0,
        "rocm bench load failed (rc={rc}):\n{stdout}{stderr}"
    );

    let cell = stdout
        .lines()
        .find(|line| line.starts_with("cell="))
        .unwrap_or_else(|| panic!("no benchmark cell line in output:\n{stdout}"));

    // `n=` counts requests that returned usable token counts. Zero is the exact
    // symptom this coverage exists to catch: the command used to print a cell
    // line with `n=0` and exit 0.
    let served = cell
        .split_whitespace()
        .find_map(|field| field.strip_prefix("n="))
        .unwrap_or("");
    assert!(
        served.parse::<u32>().is_ok_and(|n| n > 0),
        "no requests were served (n={served}):\n{cell}"
    );

    let gen_tps = cell
        .split_whitespace()
        .find_map(|field| field.strip_prefix("gen_tps="))
        .unwrap_or("");
    assert!(
        gen_tps != "-" && gen_tps.parse::<f64>().is_ok_and(|v| v > 0.0),
        "no throughput was measured (gen_tps={gen_tps}):\n{cell}"
    );
}

#[then("the benchmark requests reached the versioned chat route")]
async fn assert_versioned_route_used(world: &mut E2eWorld) {
    // The mock answers chat on both the versioned and unversioned routes, so a
    // successful benchmark alone does not prove the client used the route a
    // real engine serves. Assert the path it actually hit.
    let mock = world
        .mock
        .as_ref()
        .expect("this assertion needs the mock server");
    let paths = mock.chat_paths();
    assert!(
        !paths.is_empty(),
        "the benchmark sent no chat requests to the mock"
    );
    assert!(
        paths.iter().all(|path| path == "/v1/chat/completions"),
        "benchmark requests must use the versioned chat route, got: {paths:?}"
    );
}

#[then("the benchmark reports that the requests failed")]
async fn assert_failures_reported(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or("");
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("failed"),
        "the run reported no failure at all:\n{combined}"
    );
    // The reason must be actionable, not just a count — the original defect was
    // that the user had nothing to act on.
    assert!(
        combined.contains("503") || combined.contains("chat/completions"),
        "the failure was reported without naming a cause:\n{combined}"
    );
}

#[then("the benchmark does not report a successful run")]
async fn assert_run_not_successful(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command was run");
    let stdout = world.cli_output.as_deref().unwrap_or("");
    assert!(
        rc != 0,
        "a benchmark in which every request failed exited 0:\n{stdout}"
    );
}
