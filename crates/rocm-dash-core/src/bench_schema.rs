// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Benchmark row schema, vendored from instinct-agent-bench.
//! See `../wiki/concepts/benchmark-result-schema.md` and `../wiki/entities/csv-emitter.md`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PassFail {
    Pass,
    Fail,
    #[default]
    Unknown,
}

/// The canonical row from a benchmark run.
///
/// Fields are a superset of the 30-col CSV from `csv_emitter.py` plus the expanded fields
/// from `normalize-results.py`. Optional where upstream sets defaults from `UNKNOWN_DEFAULTS`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct BenchmarkRow {
    // identity
    pub cell: String,
    pub run: u32,
    pub task: Option<String>,
    pub prompt_class: Option<String>,
    pub trial_index: Option<u32>,

    // platform
    pub kernel: Option<String>,
    pub rocm_version: Option<String>,
    pub gpu_arch: Option<String>,
    pub hbm_per_gpu_gb: Option<u32>,
    pub num_gpus: Option<u32>,

    // backend config
    pub engine: Option<String>,
    pub model: Option<String>,
    pub endpoint: Option<String>,
    pub tp: Option<u32>,
    pub pp: Option<u32>,
    pub dtype: Option<String>,
    pub max_num_seqs: Option<u32>,
    pub attention_backend: Option<String>,
    pub kv_dtype: Option<String>,
    pub spec: Option<String>,

    // benchmark shape
    pub input_len: Option<u32>,
    pub output_len: Option<u32>,
    pub cache_mode: Option<String>,
    pub concurrency: Option<u32>,

    // request stats
    pub n_requests: Option<u32>,
    pub main_prompt_n: Option<u32>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub out_chars: Option<u64>,
    pub rc: Option<i32>,

    // throughput
    pub prompt_tps: Option<f64>,
    pub gen_tps: Option<f64>,

    // concurrency peaks
    pub max_running_reqs: Option<u32>,
    pub max_waiting_reqs: Option<u32>,

    // latency
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    pub wall_s: Option<f64>,

    // quality (C1 assertions)
    pub assertion_pass: Option<bool>,
    pub assertion_fail_count: Option<u32>,
    pub assertion_summary: Option<String>,
    pub safety_pass: Option<bool>,
    pub safety_violations: Option<u32>,
    pub safety_class: Option<String>,

    // quality (C4 judge)
    pub quality_score: Option<f32>,
    pub judge_pass_fail: PassFail,
    pub judge_model: Option<String>,

    // rollup
    pub resolved: Option<bool>,
    pub pass_fail: PassFail,
    pub failure_class: Option<String>,
    pub pass_n_of_n: Option<u32>,
    pub pass_at_n: Option<u32>,

    // provenance
    pub cmd: Option<String>,
    pub launcher: Option<String>,
    pub server_log: Option<String>,
    pub client_log: Option<String>,
    pub extra_args: Option<String>,
}
