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
use std::time::Instant;

use reqwest::Client;
use rocm_dash_core::bench_schema::BenchmarkRow;
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

/// Fixed minimal header written once when a CSV file is new or empty.
pub const CSV_HEADER: &str = "cell,run,concurrency,model,engine,input_len,output_len,\
    n_requests,prompt_tokens,completion_tokens,prompt_tps,gen_tps,wall_s,launcher\n";

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
    /// Existing file has a different CSV header; refusing to corrupt it.
    #[error("refusing to append: {path} has a different header")]
    HeaderMismatch {
        /// Path of the file with the conflicting header.
        path: String,
    },
}

/// Send `spec.requests` POST `/chat/completions` requests with `concurrency`
/// in-flight at a time.
///
/// Returns one aggregate `BenchmarkRow` with client-side `gen_tps` and
/// `prompt_tps`. Per-request failures are isolated: a non-2xx response or
/// missing `usage` fields excludes that request from the sums but does not
/// abort the cell.
pub async fn run_cell(spec: &LoadSpec, concurrency: u32) -> Result<BenchmarkRow, BenchLoadError> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_mins(5))
        .build()?;
    let sem = Arc::new(Semaphore::new(concurrency as usize));
    let url = format!("{}/chat/completions", spec.endpoint.trim_end_matches('/'));

    // Capture makespan BEFORE spawning so the clock includes queue wait time.
    let t0 = Instant::now();

    let mut js: JoinSet<Option<Outcome>> = JoinSet::new();
    for _ in 0..spec.requests {
        let client = client.clone();
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
        ..Default::default()
    })
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
    let is_empty = csv_path.metadata().map_or(true, |m| m.len() == 0);

    // Guard: if the file already has content, verify the header matches before
    // appending. A mismatch (e.g. an external 30-col agent-bench CSV) would
    // silently corrupt the file, so we refuse with a clear error instead.
    if !is_empty {
        use std::io::BufRead;
        let f = std::fs::File::open(csv_path)?;
        let mut reader = std::io::BufReader::new(f);
        let mut first_line = String::new();
        reader.read_line(&mut first_line)?;
        if first_line.trim() != CSV_HEADER.trim() {
            return Err(BenchLoadError::HeaderMismatch {
                path: csv_path.display().to_string(),
            });
        }
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)?;

    if is_empty {
        file.write_all(CSV_HEADER.as_bytes())?;
    }

    let mut rows = Vec::with_capacity(concurrency_levels.len());
    for &conc in concurrency_levels {
        let row = run_cell(spec, conc).await?;
        let line = serialize_row_to_line(&row)?;
        file.write_all(&line)?;
        rows.push(row);
    }

    Ok(rows)
}

/// Serialize one `BenchmarkRow` to the 14-column CSV line (with trailing `\n`).
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
}
