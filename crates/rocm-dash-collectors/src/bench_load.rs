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

use crate::engine_registry::EngineKind;

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
    /// OpenAI-compatible endpoint, e.g. `http://127.0.0.1:8000/v1` — the same
    /// form `rocm services list` prints. A plain host address without the `/v1`
    /// suffix is also accepted: request URLs are built through [`v1_base`],
    /// which supplies the suffix when it is missing.
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

/// One cell's [`BenchmarkRow`] plus the delivery facts the row cannot carry.
///
/// [`BenchmarkRow`] is the CSV schema and is guarded by a header check, so
/// per-request failure detail cannot live there. It travels alongside instead,
/// so the CLI can warn about a partially-failed cell and refuse to report a
/// run in which nothing succeeded as a success.
#[derive(Debug, Clone)]
pub struct CellReport {
    /// The aggregate row for this cell, as appended to the CSV.
    pub row: BenchmarkRow,
    /// Requests dispatched for this cell.
    pub attempted: u32,
    /// Requests that returned usable token counts.
    pub succeeded: u32,
    /// Requests that failed for any reason.
    pub failed: u32,
    /// Reason from the first failure observed, for the operator-facing warning.
    pub first_error: Option<String>,
}

/// Why a single benchmark request produced no usable measurement.
///
/// Kept as a short human-readable string rather than a typed enum: it is only
/// ever rendered into a warning line, and the underlying causes (transport,
/// status, malformed body) have no distinct programmatic handling.
type RequestFailure = String;

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

/// Normalise an endpoint to the OpenAI-compatible `/v1` API root that request
/// paths are built relative to.
///
/// Everything else in the product treats an "endpoint" as already carrying the
/// `/v1` suffix — `rocm services list` prints one (see `endpoint_url`, built as
/// `{base}/v1`), and the chat client appends `chat/completions` to it. Bench
/// used to be split-brained: it POSTed to `{endpoint}/chat/completions` (which
/// assumes the suffix is present) while probing `{endpoint}/v1/models` (which
/// assumes it is absent), so whichever form the user supplied, one of the two
/// 404'd. Routing both through here removes the ambiguity and lets a user paste
/// either form.
///
/// Idempotent: an endpoint already ending in `/v1` is returned unchanged apart
/// from trailing-slash trimming.
#[must_use]
pub fn v1_base(endpoint: &str) -> String {
    let trimmed = endpoint.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

/// Build the `/metrics` URL from an OpenAI-compatible endpoint base URL.
///
/// Returns `None` if the endpoint cannot be parsed (no host:port component).
/// Unaffected by the `/v1` suffix: the path is discarded and `/metrics` is
/// resolved against `host:port`, which is where the engine serves it.
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

/// Send `spec.requests` POST `/v1/chat/completions` requests with `concurrency`
/// in-flight at a time.
///
/// Returns one aggregate [`CellReport`] with client-side `gen_tps` and
/// `prompt_tps`. Per-request failures are isolated: a non-2xx response or
/// missing `usage` fields excludes that request from the sums but does not
/// abort the cell. Unlike earlier revisions, such failures are *counted and
/// reported* rather than silently dropped — a cell where every request failed
/// used to be indistinguishable from a cell that was never asked to do work.
pub async fn run_cell(spec: &LoadSpec, concurrency: u32) -> Result<CellReport, BenchLoadError> {
    run_cell_with_clients(spec, concurrency, &BenchClients::new()?).await
}

async fn run_cell_with_clients(
    spec: &LoadSpec,
    concurrency: u32,
    clients: &BenchClients,
) -> Result<CellReport, BenchLoadError> {
    if concurrency == 0 {
        return Err(BenchLoadError::InvalidConcurrency(concurrency));
    }
    let sem = Arc::new(Semaphore::new(concurrency as usize));
    let url = format!("{}/chat/completions", v1_base(&spec.endpoint));

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

    // Each request reports *why* it produced no measurement rather than
    // collapsing to `None`. A cell whose every request 404'd used to be
    // indistinguishable from a healthy cell that measured nothing, which is how
    // a wrong request path reached users as a silent empty result.
    let mut js: JoinSet<Result<Outcome, RequestFailure>> = JoinSet::new();
    for _ in 0..spec.requests {
        let client = clients.post.clone();
        let sem = Arc::clone(&sem);
        let url = url.clone();
        let model = spec.model.clone();
        let output_len = spec.output_len;
        let input_len = spec.input_len;

        js.spawn(async move {
            // Named binding: permit is held for the entire request.
            let _permit = sem
                .acquire_owned()
                .await
                .map_err(|_| "load generator stopped before the request started".to_owned())?;

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
                .map_err(|error| format!("POST {url} failed: {error}"))?;

            let status = resp.status();
            if !status.is_success() {
                return Err(format!("POST {url} returned HTTP {status}"));
            }

            let text = resp
                .text()
                .await
                .map_err(|error| format!("reading the response body failed: {error}"))?;
            let value: Value = serde_json::from_str(&text)
                .map_err(|error| format!("the response was not JSON: {error}"))?;
            let usage = value
                .get("usage")
                .ok_or_else(|| "the response carried no `usage` object".to_owned())?;

            let prompt_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .ok_or_else(|| "`usage` carried no numeric `prompt_tokens`".to_owned())?;
            let completion_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .ok_or_else(|| "`usage` carried no numeric `completion_tokens`".to_owned())?;
            // A response that generated nothing measures nothing: exclude it from
            // the sums rather than crediting a makespan it did not earn.
            if completion_tokens == 0 {
                return Err("the response reported zero completion_tokens".to_owned());
            }

            Ok(Outcome {
                prompt_tokens,
                completion_tokens,
            })
        });
    }

    let mut sum_prompt: u64 = 0;
    let mut sum_completion: u64 = 0;
    let mut n_success: u32 = 0;
    let mut n_failed: u32 = 0;
    let mut first_error: Option<String> = None;

    while let Some(res) = js.join_next().await {
        let failure = match res {
            Ok(Ok(outcome)) => {
                sum_prompt += outcome.prompt_tokens;
                sum_completion += outcome.completion_tokens;
                n_success += 1;
                continue;
            }
            Ok(Err(reason)) => reason,
            // A panicked request task is still a request that did not measure
            // anything, so it counts as a failure rather than vanishing.
            Err(join_error) => format!("the request task did not complete: {join_error}"),
        };
        n_failed += 1;
        if first_error.is_none() {
            first_error = Some(failure);
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

    // TTFT/TPOT latency from the before/after histogram scrapes: the windowed
    // mean over just the requests this cell issued. A field stays blank when the
    // window is not measurable (a scrape missing, the counter flat, or a reset)
    // rather than borrowing the endpoint's lifetime average, which would fold in
    // traffic from earlier ramp cells or other clients.
    let (ttft_ms, tpot_ms) = prom_latency(prom_before.as_ref(), prom_after.as_ref());

    // The `/metrics` scraper only understands vLLM's `vllm:` series, so a
    // recognisable sample is proof the endpoint is vLLM; a non-vLLM endpoint
    // leaves the column blank rather than guessing. `engine` is a new column:
    // rows appended before this change have it blank, and since `engine` is a
    // rollup key, a shared `results.csv` spanning the upgrade splits a cell's
    // trials into blank and `vllm` groups — rotate the file to regroup them.
    let engine = detect_engine(prom_before.as_ref(), prom_after.as_ref());

    let row = BenchmarkRow {
        cell: format!("bench-c{concurrency}"),
        run: 1,
        engine,
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
    };

    Ok(CellReport {
        row,
        attempted: spec.requests,
        succeeded: n_success,
        failed: n_failed,
        first_error,
    })
}

/// Best-effort engine label for the CSV `engine` column from a scrape pair.
///
/// The load generator's only view of the backend is its Prometheus `/metrics`
/// endpoint, and [`crate::vllm_prom::parse`] only recognises vLLM's `vllm:`
/// series. A sample carrying any recognised field is therefore proof the
/// endpoint is vLLM. Returns `None` when neither scrape produced a recognisable
/// sample (a non-vLLM endpoint, a 404, or a malformed body), leaving the column
/// blank rather than guessing.
fn detect_engine(
    before: Option<&InstanceSample>,
    after: Option<&InstanceSample>,
) -> Option<String> {
    let recognised = [after, before].into_iter().flatten().any(sample_is_vllm);
    // Take the label from the registry (`engine_registry.rs`) rather than
    // respelling it here: `engine` is a rollup key, so a spelling that drifted
    // from the registry would silently split rollup groups.
    recognised.then(|| EngineKind::Vllm.label().to_string())
}

/// Whether a parsed sample carries any vLLM-specific field.
///
/// `gen_tps` is deliberately excluded: it is the rate-reporting seam other
/// engines (e.g. Lemonade) populate, not a vLLM signal, and the vLLM parser
/// never sets it.
const fn sample_is_vllm(s: &InstanceSample) -> bool {
    s.kv_cache_usage_pct.is_some()
        || s.running_reqs.is_some()
        || s.waiting_reqs.is_some()
        || s.gen_tokens_total.is_some()
        || s.ttft_sum_s.is_some()
        || s.ttft_count.is_some()
        || s.tpot_sum_s.is_some()
        || s.tpot_count.is_some()
}

/// Compute TTFT/TPOT latency (ms) from two Prometheus samples.
///
/// Returns `(ttft_ms, tpot_ms)`, each the windowed mean over just the requests
/// this cell issued (see [`latency_ms`]). Either field is `None` when its
/// window is not measurable — including a flat TPOT counter, which for vLLM
/// means no inter-token gap was recorded in the window, i.e. the value is
/// genuinely unmeasured for this cell rather than zero.
fn prom_latency(
    before: Option<&InstanceSample>,
    after: Option<&InstanceSample>,
) -> (Option<f64>, Option<f64>) {
    let ttft_ms = latency_ms(
        before.and_then(|s| s.ttft_sum_s),
        before.and_then(|s| s.ttft_count),
        after.and_then(|s| s.ttft_sum_s),
        after.and_then(|s| s.ttft_count),
    );
    let tpot_ms = latency_ms(
        before.and_then(|s| s.tpot_sum_s),
        before.and_then(|s| s.tpot_count),
        after.and_then(|s| s.tpot_sum_s),
        after.and_then(|s| s.tpot_count),
    );
    (ttft_ms, tpot_ms)
}

/// Windowed latency (ms) across the cell's two histogram scrapes.
///
/// Computes `Δsum/Δcount × 1000` — the mean latency of just the requests this
/// cell issued between the before- and after-scrape. Returns `None` when the
/// window cannot be measured: either scrape missing, the observation count did
/// not advance (`Δcount ≤ 0`, no request recorded this metric in the window),
/// or the counter reset (`Δsum < 0`, a server restart).
///
/// The value is deliberately *not* backfilled from the after-scrape's lifetime
/// `sum/count` average. That average covers every request the server process
/// ever handled — earlier ramp cells and other clients included — so writing it
/// into this cell's immutable CSV row would describe a different population than
/// the row names, with nothing to flag it. A blank column therefore honestly
/// means "not measured for this cell". (This is narrower than the telemetry
/// daemon's `avg_ms_from_histogram`, which *does* fall back: it is a repeated
/// live poll with a rolling baseline that self-corrects on the next tick and
/// persists nothing, so the trade-offs differ.)
fn latency_ms(
    sum_before: Option<f64>,
    count_before: Option<f64>,
    sum_after: Option<f64>,
    count_after: Option<f64>,
) -> Option<f64> {
    let delta_sum = sum_after? - sum_before?;
    let delta_count = count_after? - count_before?;
    // Guard division by zero (flat counter) and counter resets.
    (delta_count > 0.0 && delta_sum >= 0.0).then_some(delta_sum / delta_count * 1000.0)
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
/// Returns the reports for the rows appended (one per concurrency level).
pub async fn run_and_append_csv(
    spec: &LoadSpec,
    concurrency_levels: &[u32],
    csv_path: &Path,
) -> Result<Vec<CellReport>, BenchLoadError> {
    let clients = BenchClients::new()?;

    let mut reports = Vec::with_capacity(concurrency_levels.len());
    for &conc in concurrency_levels {
        let report = run_cell_with_clients(spec, conc, &clients).await?;
        // A fully-failed cell is still appended: the dashboard tailer expects one
        // row per cell, and the caller reports the failure separately.
        append_one_row(&report.row, csv_path)?;
        reports.push(report);
    }

    Ok(reports)
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
/// Returns the reports for the rows appended (one per concurrency level run).
pub async fn run_auto_ramp(
    spec: &LoadSpec,
    csv_path: &Path,
) -> Result<Vec<CellReport>, BenchLoadError> {
    let mut reports = Vec::new();
    let mut prev_gen_tps: Option<f64> = None;
    let clients = BenchClients::new()?;

    for &conc in RAMP_SEQUENCE {
        let report = run_cell_with_clients(spec, conc, &clients).await?;
        append_one_row(&report.row, csv_path)?;

        let is_last = conc == *RAMP_SEQUENCE.last().unwrap_or(&conc);
        let stop = should_stop_ramp(prev_gen_tps, &report.row, is_last);
        prev_gen_tps = next_prev_gen_tps(prev_gen_tps, report.row.gen_tps);
        reports.push(report);

        if stop {
            break;
        }
    }

    Ok(reports)
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
            .and(path("/v1/chat/completions"))
            .respond_with(stub_response(100, 50))
            .expect(4)
            .mount(&server)
            .await;

        let spec = make_spec(&server.uri());
        // requests=4, each returns prompt=100 completion=50
        let mut spec4 = spec.clone();
        spec4.requests = 4;
        let row = run_cell(&spec4, 2).await.unwrap().row;

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
            .and(path("/v1/chat/completions"))
            .respond_with(
                stub_response(10, 5).set_delay(std::time::Duration::from_millis(DELAY_MS)),
            )
            .expect(u64::from(REQUESTS))
            .mount(&server)
            .await;

        let mut spec = make_spec(&server.uri());
        spec.requests = REQUESTS;
        let row = run_cell(&spec, N).await.unwrap().row;

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
            .and(path("/v1/chat/completions"))
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
            .and(path("/v1/chat/completions"))
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
            .and(path("/v1/chat/completions"))
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
        let row = run_cell(&spec, 2).await.unwrap().row;

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
            .and(path("/v1/chat/completions"))
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
        let row = run_cell(&spec, 2).await.unwrap().row;

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
        assert_eq!(
            row.engine, None,
            "engine should be blank for a non-vLLM (404 /metrics) endpoint"
        );
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
            .and(path("/v1/chat/completions"))
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
            .and(path("/v1/chat/completions"))
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
            .and(path("/v1/chat/completions"))
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
        let last_conc = rows.last().and_then(|r| r.row.concurrency).unwrap_or(0);
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

    // ---------- endpoint normalisation ----------

    #[test]
    fn v1_base_supplies_the_suffix_when_it_is_missing() {
        // The form `--endpoint`'s help used to advertise, and the form a user
        // types from memory. Both must reach the versioned API root.
        assert_eq!(v1_base("http://127.0.0.1:8000"), "http://127.0.0.1:8000/v1");
        assert_eq!(
            v1_base("http://127.0.0.1:8000/"),
            "http://127.0.0.1:8000/v1"
        );
        assert_eq!(
            v1_base("  http://127.0.0.1:8000  "),
            "http://127.0.0.1:8000/v1"
        );
    }

    #[test]
    fn v1_base_is_idempotent_for_an_already_versioned_endpoint() {
        // The canonical form — what `rocm services list` prints. Appending a
        // second `/v1` here is the mirror-image bug of omitting the first.
        assert_eq!(
            v1_base("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/v1"
        );
        assert_eq!(
            v1_base("http://127.0.0.1:8000/v1/"),
            "http://127.0.0.1:8000/v1"
        );
        assert_eq!(
            v1_base(&v1_base("http://127.0.0.1:8000")),
            "http://127.0.0.1:8000/v1"
        );
    }

    #[test]
    fn v1_base_preserves_a_gateway_path_prefix() {
        // A reverse-proxied endpoint keeps its prefix; only the API root is added.
        assert_eq!(
            v1_base("http://gw.example.com/openai"),
            "http://gw.example.com/openai/v1"
        );
        assert_eq!(
            v1_base("http://gw.example.com/openai/v1"),
            "http://gw.example.com/openai/v1"
        );
    }

    /// The regression guard for the reported bug: requests must land on the
    /// versioned path even when the caller supplies a bare host address. A mock
    /// that answers ONLY `/v1/chat/completions` fails this if the `/v1` is
    /// dropped, which is exactly what shipped.
    #[tokio::test]
    async fn plain_host_endpoint_still_reaches_the_versioned_chat_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(stub_response(100, 50))
            .mount(&server)
            .await;

        // Deliberately NOT `{uri}/v1` — the bare form the old help text taught.
        let spec = make_spec(&server.uri());
        let report = run_cell(&spec, 2).await.unwrap();

        assert_eq!(report.succeeded, 4, "every request should have been served");
        assert_eq!(report.failed, 0);
        assert_eq!(report.row.n_requests, Some(4));
    }

    // ---------- failures are reported, not swallowed ----------

    #[tokio::test]
    async fn a_cell_whose_every_request_is_rejected_reports_the_failures() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let spec = make_spec(&server.uri());
        let report = run_cell(&spec, 2).await.unwrap();

        assert_eq!(report.succeeded, 0);
        assert_eq!(report.failed, 4, "all four requests must be counted failed");
        assert_eq!(report.attempted, 4);
        let reason = report
            .first_error
            .expect("a failure reason must be recorded");
        assert!(
            reason.contains("500"),
            "the reason should name the status the server returned, got: {reason}"
        );
        // The row still reports nothing measured — the point is that the caller
        // can now tell that apart from a healthy idle cell.
        assert_eq!(report.row.gen_tps, None);
        assert_eq!(report.row.n_requests, Some(0));
    }

    #[tokio::test]
    async fn an_unreachable_endpoint_reports_a_transport_failure() {
        // Port 1 on loopback: reserved and never listening, so every send errors.
        let spec = make_spec("http://127.0.0.1:1");
        let report = run_cell(&spec, 1).await.unwrap();

        assert_eq!(report.succeeded, 0);
        assert_eq!(report.failed, 4);
        let reason = report
            .first_error
            .expect("a failure reason must be recorded");
        assert!(
            reason.contains("POST") && reason.contains("failed"),
            "the reason should describe the failed request, got: {reason}"
        );
    }

    #[tokio::test]
    async fn a_partially_failing_cell_reports_both_throughput_and_failures() {
        let server = MockServer::start().await;
        // First two requests succeed, the rest are rejected.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(stub_response(100, 50))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let spec = make_spec(&server.uri());
        let report = run_cell(&spec, 1).await.unwrap();

        assert_eq!(report.succeeded, 2);
        assert_eq!(report.failed, 2);
        assert!(
            report.row.gen_tps.unwrap_or(0.0) > 0.0,
            "throughput from the successful requests must still be reported"
        );
        assert_eq!(report.row.completion_tokens, Some(100), "2 * 50");
        assert!(report.first_error.is_some());
    }

    #[tokio::test]
    async fn a_response_without_usage_counts_as_a_named_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
                "application/json",
            ))
            .mount(&server)
            .await;

        let spec = make_spec(&server.uri());
        let report = run_cell(&spec, 2).await.unwrap();

        assert_eq!(report.succeeded, 0);
        assert_eq!(report.failed, 4);
        let reason = report
            .first_error
            .expect("a failure reason must be recorded");
        assert!(
            reason.contains("usage"),
            "the reason should name the missing field, got: {reason}"
        );
    }

    // ---------- engine + tpot column population ----------

    fn sample_with_tpot(
        ttft_sum: Option<f64>,
        ttft_count: Option<f64>,
        tpot_sum: Option<f64>,
        tpot_count: Option<f64>,
    ) -> InstanceSample {
        InstanceSample {
            running_reqs: Some(1),
            ttft_sum_s: ttft_sum,
            ttft_count,
            tpot_sum_s: tpot_sum,
            tpot_count,
            ..Default::default()
        }
    }

    #[test]
    fn latency_ms_prefers_the_window_when_the_counter_advanced() {
        // Δsum=1.0s over Δcount=2 → 500 ms.
        assert_eq!(
            latency_ms(Some(1.0), Some(10.0), Some(2.0), Some(12.0)),
            Some(500.0)
        );
    }

    #[test]
    fn latency_ms_is_none_when_the_counter_did_not_advance() {
        // A flat window (Δcount = 0) is unmeasured for this cell, not the
        // endpoint's lifetime average. For vLLM's TPOT this is the load-bearing
        // case: a flat counter means zero inter-token gaps were recorded here,
        // so a blank column is the honest answer.
        assert_eq!(
            latency_ms(Some(2.0), Some(100.0), Some(2.0), Some(100.0)),
            None
        );
    }

    #[test]
    fn latency_ms_is_none_when_the_before_scrape_is_missing() {
        // With no baseline there was never a window over this cell; the row must
        // not borrow the after-scrape's lifetime average.
        assert_eq!(latency_ms(None, None, Some(3.0), Some(30.0)), None);
    }

    #[test]
    fn latency_ms_is_none_on_a_counter_reset() {
        // sum dropped (server restart) → the negative window is discarded and
        // the row stays blank rather than reporting an average from a different
        // process lifetime.
        assert_eq!(
            latency_ms(Some(10.0), Some(100.0), Some(0.2), Some(2.0)),
            None
        );
    }

    #[test]
    fn latency_ms_is_none_without_any_data() {
        assert_eq!(latency_ms(None, None, None, None), None);
        // count == 0 is not divisible → None, never a divide-by-zero number.
        assert_eq!(latency_ms(None, None, Some(0.0), Some(0.0)), None);
    }

    #[test]
    fn prom_latency_computes_ttft_and_tpot_independently() {
        // ttft advanced (windowed → 500 ms); tpot flat across the window, so it
        // is genuinely unmeasured for this cell and stays blank rather than
        // reporting the endpoint's lifetime average.
        let before = sample_with_tpot(Some(1.0), Some(10.0), Some(2.0), Some(100.0));
        let after = sample_with_tpot(Some(2.0), Some(12.0), Some(2.0), Some(100.0));
        let (ttft_ms, tpot_ms) = prom_latency(Some(&before), Some(&after));
        assert_eq!(ttft_ms, Some(500.0));
        assert_eq!(tpot_ms, None);
    }

    #[test]
    fn prom_latency_reports_tpot_when_the_counter_advances() {
        // The window advances for both histograms → both populated. ttft:
        // (2.0-1.0)/(12-10)*1000 = 500 ms; tpot: (2.4-2.0)/(120-100)*1000 = 20 ms.
        let before = sample_with_tpot(Some(1.0), Some(10.0), Some(2.0), Some(100.0));
        let after = sample_with_tpot(Some(2.0), Some(12.0), Some(2.4), Some(120.0));
        let (ttft_ms, tpot_ms) = prom_latency(Some(&before), Some(&after));
        assert_eq!(ttft_ms, Some(500.0));
        let tpot = tpot_ms.expect("tpot_ms should be Some when the counter advances");
        assert!((tpot - 20.0).abs() < 1e-9, "expected tpot≈20 got {tpot}");
    }

    #[test]
    fn detect_engine_labels_a_recognised_sample_vllm() {
        let sample = sample_with_tpot(Some(1.0), Some(10.0), Some(2.0), Some(100.0));
        assert_eq!(detect_engine(Some(&sample), None).as_deref(), Some("vllm"));
        assert_eq!(detect_engine(None, Some(&sample)).as_deref(), Some("vllm"));
    }

    #[test]
    fn detect_engine_leaves_an_unrecognised_endpoint_blank() {
        // No scrape at all (404 / non-vLLM), and an all-`None` (malformed body)
        // sample must both leave the column blank rather than guess "vllm".
        assert_eq!(detect_engine(None, None), None);
        let empty = InstanceSample::default();
        assert_eq!(detect_engine(Some(&empty), Some(&empty)), None);
    }

    fn prom_body_full(
        running: u32,
        waiting: u32,
        ttft_sum: f64,
        ttft_count: f64,
        tpot_sum: f64,
        tpot_count: f64,
    ) -> String {
        format!(
            "vllm:num_requests_running {running}\n\
             vllm:num_requests_waiting {waiting}\n\
             vllm:time_to_first_token_seconds_sum {ttft_sum}\n\
             vllm:time_to_first_token_seconds_count {ttft_count}\n\
             vllm:time_per_output_token_seconds_sum {tpot_sum}\n\
             vllm:time_per_output_token_seconds_count {tpot_count}\n"
        )
    }

    // A vLLM endpoint whose TPOT counter did not advance between the before and
    // after scrapes (while TTFT did): the `engine` column is populated from the
    // recognised scrape, `ttft_ms` carries its windowed value, and `tpot_ms`
    // stays blank — a flat TPOT window is unmeasured for this cell, not a value
    // borrowed from the endpoint's lifetime average.
    #[tokio::test]
    async fn bench_row_labels_engine_and_leaves_flat_tpot_blank() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(stub_response(100, 50))
            .mount(&server)
            .await;

        // Before scrape: ttft cumulative 1.0s/10, tpot cumulative 2.0s/100.
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(prom_body_full(5, 2, 1.0, 10.0, 2.0, 100.0)),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // After/poller: ttft advanced (sum 2.0/count 12) but tpot counter is
        // unchanged (still 2.0/100) — the exact real-vLLM symptom.
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(prom_body_full(8, 1, 2.0, 12.0, 2.0, 100.0)),
            )
            .mount(&server)
            .await;

        let mut spec = make_spec(&server.uri());
        spec.requests = 2;
        let row = run_cell(&spec, 1).await.unwrap().row;

        assert_eq!(
            row.engine.as_deref(),
            Some("vllm"),
            "engine must be labelled from the recognised vLLM scrape"
        );
        // ttft: windowed (2.0-1.0)/(12-10)*1000 = 500 ms.
        let ttft = row.ttft_ms.expect("ttft_ms should be Some");
        assert!((ttft - 500.0).abs() < 0.01, "expected ttft≈500 got {ttft}");
        // tpot: flat window → genuinely unmeasured for this cell → blank.
        assert_eq!(
            row.tpot_ms, None,
            "a flat TPOT window must stay blank, not borrow the lifetime average"
        );
    }

    // When the TPOT counter advances across the cell's scrapes, the windowed
    // value is populated — the ordinary served case.
    #[tokio::test]
    async fn bench_row_populates_tpot_when_the_counter_advances() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(stub_response(100, 50))
            .mount(&server)
            .await;

        // Before: tpot cumulative 2.0s/100. After: 2.4s/120 → Δ 0.4s/20 = 20 ms.
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(prom_body_full(5, 2, 1.0, 10.0, 2.0, 100.0)),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(prom_body_full(8, 1, 2.0, 12.0, 2.4, 120.0)),
            )
            .mount(&server)
            .await;

        let mut spec = make_spec(&server.uri());
        spec.requests = 2;
        let row = run_cell(&spec, 1).await.unwrap().row;

        assert_eq!(row.engine.as_deref(), Some("vllm"));
        let tpot = row
            .tpot_ms
            .expect("tpot_ms must be populated when the counter advanced");
        assert!((tpot - 20.0).abs() < 0.01, "expected tpot≈20 got {tpot}");
    }
}
