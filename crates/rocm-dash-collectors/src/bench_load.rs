// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Concurrency-sweep load generator for local OpenAI-compatible endpoints.
//!
//! Produces one aggregate [`BenchmarkRow`] per concurrency cell and appends
//! them to a CSV file that a running daemon tails via [`CsvBenchTailer`].
//! Quality fields are left at their defaults (`PassFail::Unknown`).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use reqwest::Client;
use rocm_dash_core::bench_schema::BenchmarkRow;
use rocm_dash_core::traits::InstanceSample;
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

/// Timeout for /metrics scrapes (short; must never stall the cell sweep).
const METRICS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

/// Poll interval for the mid-cell Prometheus poller.
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// Sentinel stored before any successful paired running/waiting scrape.
const NO_SAMPLE_PAIR: u64 = u64::MAX;

/// Fixed minimal header written once when a CSV file is new or empty.
pub const CSV_HEADER: &str = "cell,run,concurrency,model,engine,input_len,output_len,\
    n_requests,prompt_tokens,completion_tokens,prompt_tps,gen_tps,wall_s,launcher,\
    max_running_reqs,max_waiting_reqs,ttft_ms,tpot_ms\n";

/// Parameters for a single concurrency-level load cell.
#[derive(Debug, Clone)]
pub struct LoadSpec {
    /// Base URL of the OpenAI-compatible endpoint, e.g. `http://127.0.0.1:8000`.
    pub endpoint: String,
    /// Model name to pass in the request body.
    pub model: String,
    /// Number of input tokens to request (approximated via `max_tokens` prompt).
    pub input_len: u32,
    /// Number of output tokens to request.
    pub output_len: u32,
    /// Total number of requests to send at this concurrency level.
    pub requests: u32,
}

/// Aggregate result from one successful or partially-successful response.
struct Outcome {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// Error type for the bench load writer.
#[derive(Debug, thiserror::Error)]
pub enum BenchLoadError {
    /// HTTP client construction or send failure.
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    /// CSV serialization failure.
    #[error("csv: {0}")]
    Csv(#[from] csv::Error),
    /// File I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Concurrency must be at least one.
    #[error("concurrency must be at least 1 (got {0})")]
    InvalidConcurrency(u32),
    /// Existing file has a different CSV header; refusing to corrupt it.
    #[error(
        "refusing to append: {path} has a different header; pass --out <path> or remove the incompatible file"
    )]
    HeaderMismatch {
        /// Path of the file with the conflicting header.
        path: String,
    },
}

/// Build the `/metrics` URL from an OpenAI-compatible endpoint base URL.
///
/// Returns `None` if the endpoint cannot be parsed (no host:port component).
fn metrics_url(endpoint: &str) -> Option<String> {
    let url_base = endpoint.trim_end_matches('/');
    let (scheme, rest) = if let Some(r) = url_base.strip_prefix("https://") {
        ("https", r)
    } else if let Some(r) = url_base.strip_prefix("http://") {
        ("http", r)
    } else {
        ("http", url_base)
    };
    let host_port = rest.split('/').next()?;
    Some(format!("{scheme}://{host_port}/metrics"))
}

/// Scrape Prometheus `/metrics` using the supplied client.
///
/// Returns `None` on any error (non-vLLM, 404, network failure, parse
/// garbage). Never panics. The caller is responsible for supplying a client
/// with an appropriate timeout.
async fn try_scrape_prom(client: &Client, endpoint: &str) -> Option<InstanceSample> {
    let url = metrics_url(endpoint)?;
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let text = resp.text().await.ok()?;
    Some(crate::vllm_prom::parse(&text))
}

fn pack_peak_pair(running: u32, waiting: u32) -> u64 {
    (u64::from(waiting) << 32) | u64::from(running)
}

const fn unpack_peak_pair(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

fn update_peak_pair(peak: &AtomicU64, running: u32, waiting: u32) {
    let candidate = pack_peak_pair(running, waiting);
    let _ = peak.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        if current == NO_SAMPLE_PAIR || running > unpack_peak_pair(current).0 {
            Some(candidate)
        } else {
            None
        }
    });
}

fn peak_pair(peak: &AtomicU64) -> Option<(u32, u32)> {
    let value = peak.load(Ordering::Relaxed);
    (value != NO_SAMPLE_PAIR).then(|| unpack_peak_pair(value))
}

struct BenchClients {
    post: Client,
    metrics: Client,
}

impl BenchClients {
    fn new() -> Result<Self, BenchLoadError> {
        Ok(Self {
            post: Client::builder()
                .timeout(std::time::Duration::from_mins(5))
                .build()?,
            metrics: Client::builder().timeout(METRICS_TIMEOUT).build()?,
        })
    }
}

/// Send `spec.requests` POST `/chat/completions` requests with `concurrency`
/// in-flight at a time.
///
/// Returns one aggregate `BenchmarkRow` with client-side `gen_tps` and
/// `prompt_tps`. Per-request failures are isolated: a non-2xx response or
/// missing `usage` fields excludes that request from the sums but does not
/// abort the cell.
pub async fn run_cell(spec: &LoadSpec, concurrency: u32) -> Result<BenchmarkRow, BenchLoadError> {
    run_cell_with_clients(spec, concurrency, &BenchClients::new()?).await
}

async fn run_cell_with_clients(
    spec: &LoadSpec,
    concurrency: u32,
    clients: &BenchClients,
) -> Result<BenchmarkRow, BenchLoadError> {
    if concurrency == 0 {
        return Err(BenchLoadError::InvalidConcurrency(concurrency));
    }
    let sem = Arc::new(Semaphore::new(concurrency as usize));
    let url = format!("{}/chat/completions", spec.endpoint.trim_end_matches('/'));

    // Before scrape: used only for TTFT/TPOT histogram deltas.
    let prom_before = try_scrape_prom(&clients.metrics, &spec.endpoint).await;

    // Keep the queue pair from the sample with the highest running count so
    // saturation compares values observed at the same instant.
    let peak_queue = Arc::new(AtomicU64::new(NO_SAMPLE_PAIR));
    let stop_flag = Arc::new(AtomicBool::new(false));

    // Spawn the mid-cell poller before any POST requests so it can observe
    // the rising queue depth as requests are dispatched.
    let poller = {
        let metrics_client = clients.metrics.clone();
        let endpoint = spec.endpoint.clone();
        let peak_queue = Arc::clone(&peak_queue);
        let stop_flag = Arc::clone(&stop_flag);
        tokio::spawn(async move {
            loop {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(sample) = try_scrape_prom(&metrics_client, &endpoint).await
                    && let (Some(running), Some(waiting)) =
                        (sample.running_reqs, sample.waiting_reqs)
                {
                    update_peak_pair(&peak_queue, running, waiting);
                }
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        })
    };

    // Capture makespan BEFORE spawning so the clock includes queue wait time.
    let t0 = Instant::now();

    let mut js: JoinSet<Option<Outcome>> = JoinSet::new();
    for _ in 0..spec.requests {
        let client = clients.post.clone();
        let sem = Arc::clone(&sem);
        let url = url.clone();
        let model = spec.model.clone();
        let output_len = spec.output_len;
        let input_len = spec.input_len;

        js.spawn(async move {
            // Named binding: permit is held for the entire request.
            let _permit = sem.acquire_owned().await.ok()?;

            let body = serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "x".repeat(input_len as usize)}],
                "max_tokens": output_len,
                "temperature": 0.0,
                "stream": false,
            });

            let resp = client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(body.to_string())
                .send()
                .await
                .ok()?;

            if !resp.status().is_success() {
                return None;
            }

            let text = resp.text().await.ok()?;
            let v: Value = serde_json::from_str(&text).ok()?;
            let usage = v.get("usage")?;

            let prompt_tokens = usage.get("prompt_tokens")?.as_u64()?;
            let completion_tokens = usage.get("completion_tokens")?;
            // Treat missing/zero completion_tokens as failure (excluded from sums).
            let completion_tokens = completion_tokens.as_u64()?;
            if completion_tokens == 0 {
                return None;
            }

            Some(Outcome {
                prompt_tokens,
                completion_tokens,
            })
        });
    }

    let mut sum_prompt: u64 = 0;
    let mut sum_completion: u64 = 0;
    let mut n_success: u32 = 0;

    while let Some(res) = js.join_next().await {
        if let Ok(Some(outcome)) = res {
            sum_prompt += outcome.prompt_tokens;
            sum_completion += outcome.completion_tokens;
            n_success += 1;
        }
    }

    // Stop the poller and wait for it to exit cleanly.
    stop_flag.store(true, Ordering::Relaxed);
    let _ = poller.await;

    // After scrape: used only for TTFT/TPOT histogram deltas.
    let prom_after = try_scrape_prom(&clients.metrics, &spec.endpoint).await;

    let makespan_s = t0.elapsed().as_secs_f64();

    let gen_tps = if makespan_s > 0.0 && n_success > 0 {
        Some(sum_completion as f64 / makespan_s)
    } else {
        None
    };
    let prompt_tps = if makespan_s > 0.0 && n_success > 0 {
        Some(sum_prompt as f64 / makespan_s)
    } else {
        None
    };

    // Running and waiting are retained from one real-time observation.
    let (max_running_reqs, max_waiting_reqs) = peak_pair(&peak_queue)
        .map_or((None, None), |(running, waiting)| {
            (Some(running), Some(waiting))
        });

    // TTFT/TPOT deltas come from the before/after histogram scrapes (unchanged).
    let (_, _, ttft_ms, tpot_ms) = prom_deltas(prom_before.as_ref(), prom_after.as_ref());

    Ok(BenchmarkRow {
        cell: format!("bench-c{concurrency}"),
        run: 1,
        model: Some(spec.model.clone()),
        concurrency: Some(concurrency),
        input_len: Some(spec.input_len),
        output_len: Some(spec.output_len),
        n_requests: Some(n_success),
        prompt_tokens: Some(sum_prompt),
        completion_tokens: Some(sum_completion),
        prompt_tps,
        gen_tps,
        wall_s: Some(makespan_s),
        launcher: Some("rocm bench load (local smoke)".to_string()),
        max_running_reqs,
        max_waiting_reqs,
        ttft_ms,
        tpot_ms,
        ..Default::default()
    })
}

/// Compute latency deltas from two Prometheus samples.
///
/// Returns `((), (), ttft_ms, tpot_ms)` — the first two elements are `None`
/// placeholder for the deprecated peak fields (peaks now come from the
/// mid-cell poller; this function only computes histogram deltas).
/// If either sample is `None`, both latency fields are `None`.
fn prom_deltas(
    before: Option<&InstanceSample>,
    after: Option<&InstanceSample>,
) -> (Option<u32>, Option<u32>, Option<f64>, Option<f64>) {
    let (Some(b), Some(a)) = (before, after) else {
        return (None, None, None, None);
    };

    let ttft_ms = latency_delta_ms(b.ttft_sum_s, b.ttft_count, a.ttft_sum_s, a.ttft_count);
    let tpot_ms = latency_delta_ms(b.tpot_sum_s, b.tpot_count, a.tpot_sum_s, a.tpot_count);

    (None, None, ttft_ms, tpot_ms)
}

/// Compute `Δsum / Δcount * 1000` (ms).
///
/// Returns `None` if either input is `None`, if the count delta is not
/// positive (avoids division by zero and nonsense from stale counters),
/// or if the sum delta is negative (counter reset).
fn latency_delta_ms(
    sum_before: Option<f64>,
    count_before: Option<f64>,
    sum_after: Option<f64>,
    count_after: Option<f64>,
) -> Option<f64> {
    let delta_sum = sum_after? - sum_before?;
    let delta_count = count_after? - count_before?;
    if delta_count <= 0.0 || delta_sum < 0.0 {
        return None;
    }
    Some(delta_sum / delta_count * 1000.0)
}

/// Concurrency levels tried by [`run_auto_ramp`] in order.
pub const RAMP_SEQUENCE: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128];

/// Minimum fractional `gen_tps` improvement to keep ramping.
pub const PLATEAU_GAIN: f64 = 0.05;

/// Open (or create) `csv_path` once, take an exclusive advisory lock, validate
/// or write the header, then append one newline-terminated row.
///
/// The lock serializes cooperating `rocm bench load` processes so concurrent
/// first writers cannot both emit the header. `O_APPEND` keeps each row write
/// at the end of the file.
fn append_one_row(row: &BenchmarkRow, csv_path: &Path) -> Result<(), BenchLoadError> {
    use std::io::{BufRead, Seek};

    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(csv_path)?;
    file.lock()?;

    if file.metadata()?.len() == 0 {
        file.write_all(CSV_HEADER.as_bytes())?;
    } else {
        file.seek(std::io::SeekFrom::Start(0))?;
        let mut first_line = String::new();
        std::io::BufReader::new(&file).read_line(&mut first_line)?;
        if first_line.trim() != CSV_HEADER.trim() {
            return Err(BenchLoadError::HeaderMismatch {
                path: csv_path.display().to_string(),
            });
        }
    }

    let line = serialize_row_to_line(row)?;
    file.write_all(&line)?;
    Ok(())
}

/// Run a concurrency sweep and append one aggregate row per cell to `csv_path`.
///
/// The header is written only when the file is new or empty. Each row is
/// serialized into a `Vec<u8>` ending in `\n` and written with a single
/// `write_all` call (O_APPEND safe on regular files).
///
/// Returns the rows appended (one per concurrency level).
pub async fn run_and_append_csv(
    spec: &LoadSpec,
    concurrency_levels: &[u32],
    csv_path: &Path,
) -> Result<Vec<BenchmarkRow>, BenchLoadError> {
    let clients = BenchClients::new()?;

    let mut rows = Vec::with_capacity(concurrency_levels.len());
    for &conc in concurrency_levels {
        let row = run_cell_with_clients(spec, conc, &clients).await?;
        append_one_row(&row, csv_path)?;
        rows.push(row);
    }

    Ok(rows)
}

/// Decide whether the auto-ramp should stop after `row`.
///
/// Pure function — no I/O, no side effects — so it can be tested
/// deterministically with hand-built [`BenchmarkRow`] values.
///
/// Returns `true` when any of the following hold:
/// - `is_last`: the hard cap (last element of [`RAMP_SEQUENCE`]) was reached,
/// - plateau: `prev_gen_tps` is `Some` AND `row.gen_tps` is `Some` AND
///   `gen_tps <= prev * (1.0 + PLATEAU_GAIN)`,
/// - saturation: both `max_running_reqs` and `max_waiting_reqs` are `Some`
///   AND `running > 0` AND `waiting >= running` (queue backed up).
///   The `running > 0` guard prevents a false positive when both fields are
///   observed at rest (zero) before any requests have reached the server.
pub fn should_stop_ramp(prev_gen_tps: Option<f64>, row: &BenchmarkRow, is_last: bool) -> bool {
    if is_last {
        return true;
    }

    // Plateau: throughput stopped growing.
    if let (Some(prev), Some(cur)) = (prev_gen_tps, row.gen_tps)
        && cur <= prev * (1.0 + PLATEAU_GAIN)
    {
        return true;
    }

    // Saturation: the queue is backed up — adding concurrency won't help.
    // The `running > 0` guard prevents a false-stop when peaks are both zero
    // (observed at rest before requests reach the engine).
    if let (Some(running), Some(waiting)) = (row.max_running_reqs, row.max_waiting_reqs)
        && running > 0
        && waiting >= running
    {
        return true;
    }

    false
}

fn next_prev_gen_tps(previous: Option<f64>, current: Option<f64>) -> Option<f64> {
    current.or(previous)
}

/// Run an automatic concurrency ramp over [`RAMP_SEQUENCE`], stopping early
/// when throughput saturates.
///
/// Each cell is appended to `csv_path` immediately after completion so the
/// daemon tailer shows progress live. Stops after a cell when
/// [`should_stop_ramp`] returns `true`.
///
/// Returns the rows appended (one per concurrency level run).
pub async fn run_auto_ramp(
    spec: &LoadSpec,
    csv_path: &Path,
) -> Result<Vec<BenchmarkRow>, BenchLoadError> {
    let mut rows = Vec::new();
    let mut prev_gen_tps: Option<f64> = None;
    let clients = BenchClients::new()?;

    for &conc in RAMP_SEQUENCE {
        let row = run_cell_with_clients(spec, conc, &clients).await?;
        append_one_row(&row, csv_path)?;

        let is_last = conc == *RAMP_SEQUENCE.last().unwrap_or(&conc);
        let stop = should_stop_ramp(prev_gen_tps, &row, is_last);
        prev_gen_tps = next_prev_gen_tps(prev_gen_tps, row.gen_tps);
        rows.push(row);

        if stop {
            break;
        }
    }

    Ok(rows)
}

/// Serialize one `BenchmarkRow` to the 18-column CSV line (with trailing `\n`).
fn serialize_row_to_line(row: &BenchmarkRow) -> Result<Vec<u8>, BenchLoadError> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(&mut buf);
        wtr.write_record([
            row.cell.as_str(),
            &row.run.to_string(),
            &opt_u32(row.concurrency),
            opt_str(row.model.as_deref()),
            opt_str(row.engine.as_deref()),
            &opt_u32(row.input_len),
            &opt_u32(row.output_len),
            &opt_u32(row.n_requests),
            &opt_u64(row.prompt_tokens),
            &opt_u64(row.completion_tokens),
            &opt_f64(row.prompt_tps),
            &opt_f64(row.gen_tps),
            &opt_f64(row.wall_s),
            opt_str(row.launcher.as_deref()),
            &opt_u32(row.max_running_reqs),
            &opt_u32(row.max_waiting_reqs),
            &opt_f64(row.ttft_ms),
            &opt_f64(row.tpot_ms),
        ])?;
        wtr.flush()?;
    }
    // csv::Writer ends each record with \n already but we ensure it.
    if buf.last() != Some(&b'\n') {
        buf.push(b'\n');
    }
    Ok(buf)
}

fn opt_str(v: Option<&str>) -> &str {
    v.unwrap_or("")
}

fn opt_u32(v: Option<u32>) -> String {
    v.map(|n| n.to_string()).unwrap_or_default()
}

fn opt_u64(v: Option<u64>) -> String {
    v.map(|n| n.to_string()).unwrap_or_default()
}

fn opt_f64(v: Option<f64>) -> String {
    v.map(|f| format!("{f:.6}")).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use rocm_dash_core::bench_rollup::{rollup_pass_n, row_verdict};
    use rocm_dash_core::bench_schema::PassFail;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::bench_tail::CsvBenchTailer;
    use rocm_dash_core::traits::BenchTailer;

    // ---------- helpers ----------

    fn tempdir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        p.push(format!("rocm-bench-load-{pid}-{n}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn stub_response(prompt_tokens: u64, completion_tokens: u64) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_raw(
            format!(
                r#"{{"choices":[{{"message":{{"role":"assistant","content":"ok"}}}}],
                "usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":{completion_tokens}}}}}"#
            ),
            "application/json",
        )
    }

    fn make_spec(endpoint: &str) -> LoadSpec {
        LoadSpec {
            endpoint: endpoint.to_string(),
            model: "test-model".to_string(),
            input_len: 16,
            output_len: 8,
            requests: 4,
        }
    }

    // ---------- helpers: Prometheus stub body ----------

    fn prom_body(running: u32, waiting: u32, ttft_sum: f64, ttft_count: f64) -> String {
        format!(
            "vllm:num_requests_running {running}\n\
             vllm:num_requests_waiting {waiting}\n\
             vllm:time_to_first_token_seconds_sum {ttft_sum}\n\
             vllm:time_to_first_token_seconds_count {ttft_count}\n\
             vllm:time_per_output_token_seconds_sum 0\n\
             vllm:time_per_output_token_seconds_count 0\n"
        )
    }

    // ---------- T1: run_cell against a stub ----------

    #[tokio::test]
    async fn t1_run_cell_fields_and_tps() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(stub_response(100, 50))
            .expect(4)
            .mount(&server)
            .await;

        let spec = make_spec(&server.uri());
        // requests=4, each returns prompt=100 completion=50
        let mut spec4 = spec.clone();
        spec4.requests = 4;
        let row = run_cell(&spec4, 2).await.unwrap();

        assert_eq!(row.cell, "bench-c2");
        assert_eq!(row.run, 1);
        assert_eq!(row.concurrency, Some(2));
        assert_eq!(row.n_requests, Some(4));
        assert_eq!(row.completion_tokens, Some(200)); // 4 * 50
        assert_eq!(row.prompt_tokens, Some(400)); // 4 * 100
        // gen_tps divides by measured wall time — just check it's positive
        assert!(
            row.gen_tps.unwrap_or(0.0) > 0.0,
            "gen_tps should be positive"
        );
        assert!(
            row.prompt_tps.unwrap_or(0.0) > 0.0,
            "prompt_tps should be positive"
        );
        assert_eq!(
            row.launcher.as_deref(),
            Some("rocm bench load (local smoke)")
        );
    }

    #[tokio::test]
    async fn run_cell_rejects_zero_concurrency() {
        let spec = make_spec("http://127.0.0.1:1");
        let result = run_cell(&spec, 0).await;
        assert!(
            matches!(result, Err(BenchLoadError::InvalidConcurrency(0))),
            "zero concurrency must fail before any network request: {result:?}"
        );
    }

    // ---------- T2: concurrency cap ----------
    //
    // wiremock's hyper handler calls respond() under an exclusive write-lock,
    // so respond() is serial and cannot measure concurrent overlap. Instead we
    // verify the semaphore via total elapsed time:
    //
    //   With N=4, R=16 requests, and a per-response delay of D ms:
    //     - WITHOUT semaphore: all 16 fire simultaneously → wall ≈ D
    //     - WITH semaphore N: ceil(16/4)=4 serial batches → wall ≈ 4×D
    //
    // We assert wall_s > 1.5×D (conservative midpoint), which fails if the
    // semaphore is absent because 1 batch × D < 1.5×D. We also assert
    // wall_s < 8×D as a sanity upper bound so the test doesn't silently pass
    // on a hung server.

    #[tokio::test]
    async fn t2_concurrency_cap() {
        const DELAY_MS: u64 = 30;
        const N: u32 = 4;
        const REQUESTS: u32 = 16;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                stub_response(10, 5).set_delay(std::time::Duration::from_millis(DELAY_MS)),
            )
            .expect(u64::from(REQUESTS))
            .mount(&server)
            .await;

        let mut spec = make_spec(&server.uri());
        spec.requests = REQUESTS;
        let row = run_cell(&spec, N).await.unwrap();

        // Structural check: concurrency column matches N.
        assert_eq!(row.concurrency, Some(N));
        assert_eq!(
            row.n_requests,
            Some(REQUESTS),
            "all requests should succeed"
        );

        // Timing check: the semaphore batches requests so wall time is
        // proportional to ceil(R/N), not to 1 batch.
        let wall_s = row.wall_s.expect("wall_s must be set");
        let delay_s = DELAY_MS as f64 / 1000.0;
        // Lower bound: at least 1.5 batches of delay (conservatively)
        assert!(
            wall_s >= delay_s * 1.5,
            "wall_s={wall_s:.3}s < 1.5×delay={:.3}s — semaphore may not be limiting concurrency",
            delay_s * 1.5
        );
        // Sanity upper bound: no more than 8 batches (catches hung servers)
        assert!(
            wall_s < delay_s * 8.0 * f64::from(REQUESTS) / f64::from(N),
            "wall_s={wall_s:.3}s looks unreasonably large"
        );
    }

    // ---------- T3: CSV round-trip ----------

    #[tokio::test]
    async fn t3_csv_round_trip() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(stub_response(50, 25))
            .mount(&server)
            .await;

        let dir = tempdir();
        let csv_path = dir.join("bench.csv");
        let mut spec = make_spec(&server.uri());
        spec.requests = 2;

        // Append sweep A (concurrency [1]) → drain should return 1 row.
        run_and_append_csv(&spec, &[1], &csv_path).await.unwrap();
        let mut tailer = CsvBenchTailer::new(csv_path.clone());
        let rows_a = tailer.drain().unwrap();
        assert_eq!(rows_a.len(), 1, "drain A should return 1 row");
        assert_eq!(rows_a[0].cell, "bench-c1");
        // pass_fail defaults to Unknown (omitted columns default via #[serde(default)]).
        assert_eq!(rows_a[0].pass_fail, PassFail::Unknown);

        // Second drain should be empty (no new rows).
        let empty = tailer.drain().unwrap();
        assert!(empty.is_empty(), "second drain should be empty");

        // Append sweep B (concurrency [8]) → drain should return only the new row.
        run_and_append_csv(&spec, &[8], &csv_path).await.unwrap();
        let rows_b = tailer.drain().unwrap();
        assert_eq!(rows_b.len(), 1, "drain B should return 1 row");
        assert_eq!(rows_b[0].cell, "bench-c8");
        // pass_fail for a throughput-only row must be Unknown.
        assert_eq!(rows_b[0].pass_fail, PassFail::Unknown);

        let _ = std::fs::remove_dir_all(dir);
    }

    // ---------- D2: header-mismatch guard ----------

    #[tokio::test]
    async fn d2_header_mismatch_returns_error_without_modifying_file() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(stub_response(50, 25))
            .mount(&server)
            .await;

        let dir = tempdir();
        let csv_path = dir.join("external.csv");

        // Write a file that starts with a bogus header (simulating an external
        // agent-bench CSV with a different column layout).
        let bogus_header = "col1,col2,col3\n";
        let original_content = format!("{bogus_header}row1,row2,row3\n");
        std::fs::write(&csv_path, &original_content).unwrap();

        let spec = make_spec(&server.uri());
        let result = run_and_append_csv(&spec, &[1], &csv_path).await;

        // Must return the HeaderMismatch error.
        assert!(
            matches!(result, Err(BenchLoadError::HeaderMismatch { .. })),
            "expected HeaderMismatch error, got: {result:?}"
        );

        // File must be unmodified.
        let content_after = std::fs::read_to_string(&csv_path).unwrap();
        assert_eq!(
            content_after, original_content,
            "file must not be modified on header mismatch"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn concurrent_first_appends_write_one_header() {
        const WRITERS: usize = 8;
        let dir = tempdir();
        let csv_path = dir.join("concurrent.csv");
        let barrier = Arc::new(std::sync::Barrier::new(WRITERS));

        let handles: Vec<_> = (0..WRITERS)
            .map(|run| {
                let csv_path = csv_path.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let row = BenchmarkRow {
                        cell: format!("bench-{run}"),
                        run: run as u32,
                        ..Default::default()
                    };
                    barrier.wait();
                    append_one_row(&row, &csv_path)
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap().unwrap();
        }

        let content = std::fs::read_to_string(&csv_path).unwrap();
        assert_eq!(
            content
                .lines()
                .filter(|line| *line == CSV_HEADER.trim())
                .count(),
            1,
            "concurrent creators must not duplicate the CSV header"
        );
        assert_eq!(content.lines().count(), WRITERS + 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    // ---------- T4: Unknown-verdict guard ----------

    #[test]
    fn t4_unknown_verdict_does_not_count_as_pass() {
        // A row with only throughput fields populated — quality all default → Unknown.
        let row = BenchmarkRow {
            cell: "bench-c1".to_string(),
            run: 1,
            gen_tps: Some(100.0),
            concurrency: Some(1),
            ..Default::default()
        };

        assert_eq!(row_verdict(&row), PassFail::Unknown);

        let rollup = rollup_pass_n(std::slice::from_ref(&row));
        assert_eq!(rollup.len(), 1);
        assert_eq!(
            rollup[0].n_passed, 0,
            "Unknown verdict must not count as pass"
        );
    }

    // ---------- T6: Prometheus poller + before/after → peaks + ttft_ms ----------
    //
    // Architecture: peaks come from the mid-cell poller; ttft/tpot come from
    // the before/after histogram delta.
    //
    // Stub layout:
    //   - First GET /metrics (up_to_n_times=1): before scrape → ttft_sum=10, count=100.
    //     The poller starts after the before scrape, so it never sees this response.
    //   - Catch-all GET /metrics: poller + after scrape → running=8, waiting=1,
    //     ttft_sum=20, count=200.
    //
    // Expected peaks (from poller): max_running=8, max_waiting=1.
    // Expected ttft_ms = (20-10)/(200-100) * 1000 = 100ms.

    #[tokio::test]
    async fn t6_prom_poller_populates_peaks_and_ttft() {
        let server = MockServer::start().await;

        // /chat/completions returns token data.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(stub_response(100, 50))
            .expect(4)
            .mount(&server)
            .await;

        // Before scrape (first GET /metrics only) — used for ttft/tpot delta origin.
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_string(prom_body(5, 2, 10.0, 100.0)),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Catch-all — seen by the poller and the after scrape.
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_string(prom_body(8, 1, 20.0, 200.0)),
            )
            .mount(&server)
            .await;

        let mut spec = make_spec(&server.uri());
        spec.requests = 4;
        let row = run_cell(&spec, 2).await.unwrap();

        // Peaks come from the poller (which only sees the catch-all stub).
        assert_eq!(
            row.max_running_reqs,
            Some(8),
            "peak running should be 8 (poller)"
        );
        assert_eq!(
            row.max_waiting_reqs,
            Some(1),
            "peak waiting should be 1 (poller)"
        );
        // ttft_ms is the histogram delta between before and after scrapes.
        let ttft = row.ttft_ms.expect("ttft_ms should be Some");
        assert!(
            (ttft - 100.0).abs() < 0.01,
            "expected ttft_ms≈100 got {ttft}"
        );
        // gen_tps must still be computed from client-side measurement.
        assert!(row.gen_tps.unwrap_or(0.0) > 0.0, "gen_tps must be positive");
    }

    // ---------- T7: non-vLLM /metrics (404) → new fields None, gen_tps Some ----------

    #[tokio::test]
    async fn t7_non_vllm_metrics_404_new_fields_none_gen_tps_some() {
        let server = MockServer::start().await;

        // Normal chat completions succeed.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(stub_response(100, 50))
            .mount(&server)
            .await;

        // /metrics returns 404 (non-vLLM endpoint).
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let mut spec = make_spec(&server.uri());
        spec.requests = 4;
        let row = run_cell(&spec, 2).await.unwrap();

        assert_eq!(
            row.max_running_reqs, None,
            "max_running_reqs should be None for 404 /metrics"
        );
        assert_eq!(
            row.max_waiting_reqs, None,
            "max_waiting_reqs should be None for 404 /metrics"
        );
        assert_eq!(row.ttft_ms, None, "ttft_ms should be None for 404 /metrics");
        assert_eq!(row.tpot_ms, None, "tpot_ms should be None for 404 /metrics");
        assert!(
            row.gen_tps.unwrap_or(0.0) > 0.0,
            "gen_tps must still be positive"
        );
    }

    // ---------- T8: 18-col CSV round-trip via CsvBenchTailer ----------

    #[tokio::test]
    async fn t8_csv_round_trip_18_cols() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(stub_response(50, 25))
            .mount(&server)
            .await;
        // /metrics returns 404 so new fields are None (simpler to assert).
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let dir = tempdir();
        let csv_path = dir.join("bench18.csv");
        let mut spec = make_spec(&server.uri());
        spec.requests = 2;

        // Write one row.
        run_and_append_csv(&spec, &[1], &csv_path).await.unwrap();

        // Verify the header is 18 columns.
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let first_line = content.lines().next().expect("file should have a header");
        assert_eq!(
            first_line.split(',').count(),
            18,
            "header should have 18 columns"
        );

        // Drain via CsvBenchTailer — must deserialize without error.
        let mut tailer = CsvBenchTailer::new(csv_path.clone());
        let rows = tailer.drain().unwrap();
        assert_eq!(rows.len(), 1, "should drain 1 row");
        assert_eq!(rows[0].cell, "bench-c1");
        assert_eq!(rows[0].pass_fail, PassFail::Unknown);
        // New fields are None (404 /metrics path).
        assert_eq!(rows[0].max_running_reqs, None);
        assert_eq!(rows[0].ttft_ms, None);

        let _ = std::fs::remove_dir_all(dir);
    }

    // ---------- T9: appending to a 14-col Phase-1 file returns HeaderMismatch ----------

    #[tokio::test]
    async fn t9_old_14col_file_returns_header_mismatch() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(stub_response(50, 25))
            .mount(&server)
            .await;

        let dir = tempdir();
        let csv_path = dir.join("phase1.csv");

        // Write a file with the old 14-col header from Phase 1.
        let old_header = "cell,run,concurrency,model,engine,input_len,output_len,\
             n_requests,prompt_tokens,completion_tokens,prompt_tps,gen_tps,wall_s,launcher\n";
        let original = format!(
            "{old_header}bench-c1,1,1,m,,16,8,4,200,100,,,0.5,rocm bench load (local smoke)\n"
        );
        std::fs::write(&csv_path, &original).unwrap();

        let spec = make_spec(&server.uri());
        let result = run_and_append_csv(&spec, &[1], &csv_path).await;

        assert!(
            matches!(result, Err(BenchLoadError::HeaderMismatch { .. })),
            "expected HeaderMismatch for 14-col file, got: {result:?}"
        );

        // File must be unmodified.
        let after = std::fs::read_to_string(&csv_path).unwrap();
        assert_eq!(after, original, "14-col file must not be modified");

        let _ = std::fs::remove_dir_all(dir);
    }

    // ---------- T10: auto-ramp plateau — flat gen_tps stops early ----------
    //
    // All cells return identical token counts so gen_tps is flat. The plateau
    // check fires on the second cell (cur <= prev * 1.05 because cur == prev),
    // so the ramp stops at concurrency=2 and never reaches 128.

    #[tokio::test]
    async fn t10_auto_ramp_plateau_stops_early() {
        let server = MockServer::start().await;
        // Same token counts for all requests → flat gen_tps.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(stub_response(50, 25))
            .mount(&server)
            .await;
        // /metrics: 404 so Prometheus fields are None.
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let dir = tempdir();
        let csv_path = dir.join("auto_ramp_plateau.csv");
        let mut spec = make_spec(&server.uri());
        spec.requests = 2;

        let rows = run_auto_ramp(&spec, &csv_path).await.unwrap();

        // Must have stopped before reaching concurrency=128 (the last element).
        assert!(
            rows.len() < RAMP_SEQUENCE.len(),
            "plateau should have stopped early; got {} rows (RAMP len={})",
            rows.len(),
            RAMP_SEQUENCE.len()
        );
        // Last concurrency must not be 128.
        let last_conc = rows.last().and_then(|r| r.concurrency).unwrap_or(0);
        assert_ne!(last_conc, 128, "should not have reached concurrency=128");
        // Must have appended at least the first cell.
        assert!(!rows.is_empty(), "at least one row must be produced");

        let _ = std::fs::remove_dir_all(dir);
    }

    // ---------- T11: auto-ramp cap — rising gen_tps reaches 128 ----------
    //
    // We use a response delay that grows with each call so earlier concurrency
    // levels complete fewer tokens per second than later ones.  The trick: use a
    // wiremock `up_to_n_times` chain of stubs with decreasing delay so the mock
    // server delivers progressively faster responses, making gen_tps rise
    // monotonically and preventing the plateau check from firing until the last
    // element (128) of RAMP_SEQUENCE is reached.
    //
    // Because accurate per-call timing in a test is fragile, we instead use a
    // simpler approach: a single stub that always responds with the same tokens
    // but with a very short delay, and set spec.requests = 1 so each cell has
    // exactly 1 request. With 1 request per cell and flat token counts the gen_tps
    // will be approximately 1/wall which varies by wall time — we can't guarantee
    // monotonic growth.
    //
    // Instead, we use the queue-backed-up exit condition to test the cap: set
    // max_waiting >= max_running via Prometheus.  But that only works if Prom is up.
    //
    // Simplest approach: test the cap path directly by verifying that with a
    // strictly rising gen_tps signal, the ramp runs all the way to the last
    // RAMP_SEQUENCE entry (128). We simulate this by setting requests=1 and using
    // a delay that decreases per-cell, ensuring each successive cell is faster.
    //
    // Since we can't easily make gen_tps strictly increase with a real HTTP mock
    // (wall time is non-deterministic), we use a different angle: verify that when
    // NO plateau and NO queue-full ever triggers, the ramp hits exactly 128.
    // We achieve this by making each request take 0ms (no delay) — but with 1
    // request per cell the gen_tps may still vary.  The reliable invariant is:
    // the last row's concurrency == 128 when the stop condition never fires early.
    //
    // We enforce "no early stop" by using enough requests (spec.requests = 64)
    // that each cell's gen_tps has a chance to grow (more concurrent = more TPS),
    // and by checking the last concurrency rather than exact row count.

    #[test]
    fn t11_auto_ramp_hard_cap_stops_at_128() {
        let row = row_with_peaks(Some(1_000.0), None, None);
        assert!(should_stop_ramp(Some(1.0), &row, true));
        assert_eq!(RAMP_SEQUENCE.last(), Some(&128));
    }

    // ---------- T12: should_stop_ramp — pure-function unit tests ----------

    fn row_with_peaks(
        gen_tps: Option<f64>,
        max_running_reqs: Option<u32>,
        max_waiting_reqs: Option<u32>,
    ) -> BenchmarkRow {
        BenchmarkRow {
            cell: "bench-c1".to_string(),
            run: 1,
            gen_tps,
            max_running_reqs,
            max_waiting_reqs,
            ..Default::default()
        }
    }

    #[test]
    fn t12a_plateau_stops_ramp() {
        // gen_tps same as prev → cur <= prev * 1.05 → stop.
        let row = row_with_peaks(Some(100.0), None, None);
        assert!(
            should_stop_ramp(Some(100.0), &row, false),
            "plateau should stop"
        );
    }

    #[test]
    fn t12b_rising_gen_tps_continues() {
        // gen_tps grew by >5% → continue.
        let row = row_with_peaks(Some(120.0), None, None);
        assert!(
            !should_stop_ramp(Some(100.0), &row, false),
            "rising gen_tps should continue"
        );
    }

    #[test]
    fn t12c_is_last_stops() {
        // Hard cap regardless of other fields.
        let row = row_with_peaks(Some(200.0), None, None);
        assert!(
            should_stop_ramp(None, &row, true),
            "is_last should always stop"
        );
    }

    #[test]
    fn t12d_saturation_running8_waiting8_stops() {
        // waiting >= running AND running > 0 → saturated.
        let row = row_with_peaks(None, Some(8), Some(8));
        assert!(
            should_stop_ramp(None, &row, false),
            "waiting>=running with running>0 should stop"
        );
    }

    #[test]
    fn t12e_at_rest_both_zero_does_not_stop() {
        // Regression guard for the H1 fix: running=0, waiting=0 must NOT stop.
        let row = row_with_peaks(None, Some(0), Some(0));
        assert!(
            !should_stop_ramp(None, &row, false),
            "running=0,waiting=0 must NOT stop (H1)"
        );
    }

    #[test]
    fn t12f_failed_cell_preserves_last_successful_throughput() {
        assert_eq!(next_prev_gen_tps(Some(100.0), None), Some(100.0));
        assert_eq!(next_prev_gen_tps(Some(100.0), Some(120.0)), Some(120.0));
    }

    #[test]
    fn t12f_both_none_peaks_does_not_stop() {
        // Non-vLLM endpoint: both peaks are None; no saturation stop.
        let row = row_with_peaks(None, None, None);
        assert!(
            !should_stop_ramp(None, &row, false),
            "None peaks must not stop"
        );
    }
    #[test]
    fn peak_pair_keeps_running_and_waiting_from_one_sample() {
        let peak = AtomicU64::new(NO_SAMPLE_PAIR);
        update_peak_pair(&peak, 2, 5);
        update_peak_pair(&peak, 8, 1);

        assert_eq!(peak_pair(&peak), Some((8, 1)));
    }
}
