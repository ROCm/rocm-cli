// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Tail a normalized benchmark CSV from instinct-agent-bench.
//!
//! Strategy: every `drain()` re-reads the entire file and yields rows we haven't
//! returned yet. Simpler and correct under truncate-rewrite rotation (no need
//! to chase mtime / inode / offset heuristics). Benchmark CSVs are small —
//! tens to thousands of rows — so the cost is fine for our cadence.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use rocm_dash_core::bench_schema::BenchmarkRow;
use rocm_dash_core::traits::{BenchTailer, CollectorError, Result};

pub struct CsvBenchTailer {
    path: PathBuf,
    rows_seen: usize,
}

impl CsvBenchTailer {
    pub const fn new(path: PathBuf) -> Self {
        Self { path, rows_seen: 0 }
    }
}

impl BenchTailer for CsvBenchTailer {
    fn name(&self) -> &'static str {
        "csv-bench-tailer"
    }

    fn drain(&mut self) -> Result<Vec<BenchmarkRow>> {
        let file = File::open(&self.path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(BufReader::new(file));
        let header = rdr.headers().map_err(map_csv_err)?.clone();

        let mut all: Vec<BenchmarkRow> = Vec::new();
        for rec in rdr.records() {
            let rec = rec.map_err(map_csv_err)?;
            let row: BenchmarkRow = rec.deserialize(Some(&header)).map_err(map_csv_err)?;
            all.push(row);
        }

        // Rotation (or row deletion) — file got shorter than what we last saw.
        // Reset and re-emit everything.
        if all.len() < self.rows_seen {
            self.rows_seen = 0;
        }
        let new_rows: Vec<_> = all.into_iter().skip(self.rows_seen).collect();
        self.rows_seen += new_rows.len();
        Ok(new_rows)
    }
}

fn map_csv_err(e: csv::Error) -> CollectorError {
    CollectorError::Parse(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(path: &std::path::Path, s: &str) {
        std::fs::write(path, s).unwrap();
    }

    fn append(path: &std::path::Path, s: &str) {
        let mut f = std::fs::OpenOptions::new().append(true).open(path).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    const HEADER: &str = "cell,run,wall_s,n_requests,main_prompt_n,prompt_tokens,prompt_tps,\
        completion_tokens,gen_tps,max_running_reqs,max_waiting_reqs,out_chars,rc,\
        assertion_pass,assertion_fail_count,assertion_summary,quality_score,\
        judge_pass_fail,judge_model,model,endpoint,tp,pp,dtype,max_num_seqs,\
        attention_backend,concurrency,extra_args,safety_pass,safety_violations\n";
    const ROW1: &str = "O-arch,1,42.3,8,512,4096,1240.5,2048,68.2,8,2,8192,0,true,0,all-pass,\
        4.5,pass,claude-sonnet-4-6,deepseek-r1,http://vllm:8000,8,1,fp8,32,triton,1,,true,0\n";
    const ROW2: &str = "B-code,1,55.8,4,1024,3200,980.2,1600,52.4,4,0,6400,0,false,2,assert.miss,\
        2.1,fail,claude-sonnet-4-6,llama-3.1-70b,http://vllm:8000,4,1,fp16,16,flash,1,,true,0\n";

    #[test]
    fn drain_returns_only_appended_rows() {
        let dir = tempdir();
        let path = dir.join("results.csv");
        write(&path, &format!("{HEADER}{ROW1}"));
        let mut tailer = CsvBenchTailer::new(path.clone());

        let first = tailer.drain().unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].cell, "O-arch");
        assert_eq!(first[0].prompt_tps, Some(1240.5));
        assert_eq!(first[0].gen_tps, Some(68.2));
        assert_eq!(first[0].assertion_pass, Some(true));

        // Idempotent: no new rows → empty.
        assert!(tailer.drain().unwrap().is_empty());

        // Append one row → one row returned.
        append(&path, ROW2);
        let next = tailer.drain().unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].cell, "B-code");
        assert_eq!(next[0].assertion_pass, Some(false));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rotation_truncate_rewrite_re_emits_rows() {
        let dir = tempdir();
        let path = dir.join("results.csv");
        write(&path, &format!("{HEADER}{ROW1}{ROW2}"));
        let mut tailer = CsvBenchTailer::new(path.clone());
        assert_eq!(tailer.drain().unwrap().len(), 2);

        // Rotation: rewrite with just ROW2 → rows_seen (2) > new total (1) → reset + re-emit.
        write(&path, &format!("{HEADER}{ROW2}"));
        let after = tailer.drain().unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].cell, "B-code");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_30col_fields_default_safely() {
        // A minimal header missing many optional columns shouldn't error.
        let dir = tempdir();
        let path = dir.join("results.csv");
        write(&path, "cell,run\nO-arch,7\n");
        let mut t = CsvBenchTailer::new(path);
        let rows = t.drain().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].cell, "O-arch");
        assert_eq!(rows[0].run, 7);
        assert_eq!(rows[0].prompt_tps, None);

        let _ = std::fs::remove_dir_all(dir);
    }

    fn tempdir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        p.push(format!("rocm-dash-bench-tail-{pid}-{n}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
