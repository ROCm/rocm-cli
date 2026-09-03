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
    run_bench(world, &endpoint);
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
    run_bench(world, &plain);
}

fn run_bench(world: &mut E2eWorld, endpoint: &str) {
    let model = world
        .model_name
        .clone()
        .expect("no model configured for the benchmark");
    // `--out` lands inside the scenario's isolated data dir by default; the
    // explicit model avoids depending on the endpoint's model-listing route,
    // which the rejecting server deliberately fails.
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
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
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Benchmark the metrics-backed endpoint, writing the row to a known `--out`
/// file so the row's `engine`/`tpot_ms` columns can be asserted directly.
#[when("the user benchmarks the served endpoint recording results to a file")]
async fn benchmark_recording_results(world: &mut E2eWorld) {
    let endpoint = world
        .endpoint
        .clone()
        .expect("no endpoint configured for the benchmark");
    let model = world
        .model_name
        .clone()
        .expect("no model configured for the benchmark");
    let out = bench_out_path(world);
    let out = out.to_str().expect("bench output path is not valid UTF-8");
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "bench",
            "load",
            "--endpoint",
            endpoint.as_str(),
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
            "--out",
            out,
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
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
        tpot.parse::<f64>().is_ok_and(|v| v > 0.0),
        "the emitted row must carry a positive tpot_ms from the advancing counter, got {tpot:?}:\n{data}"
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
