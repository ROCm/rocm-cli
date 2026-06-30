// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Synthetic-session generator (single source of truth for demo data).
//!
//! Simulates a single 8× AMD Instinct MI355X node serving inference traffic
//! through three vLLM containers. Output is the same `PersistedEntry` NDJSON
//! the daemon writes when `--persist-dir` is set, so `rocm replay` (and
//! `rocm demo`) play it back unmodified.
//!
//! Deterministic: the same [`DemoOptions`] always produce byte-identical
//! output (seeded xorshift jitter + a fixed wallclock anchor), which the
//! tests below pin as a hard invariant. `rocm gen-demo`, `rocm demo`, and the
//! marketing screenshot/cast examples all consume this one generator.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use chrono::{TimeZone, Utc};
use rocm_dash_core::bench_schema::{BenchmarkRow, PassFail};
use rocm_dash_core::metrics::{
    GpuMetrics, GpuSystemInfo, Instance, InstanceStatus, Snapshot, SystemMetrics,
};
use rocm_dash_core::partition::{ComputePartitionMode, MemoryPartitionMode};
use rocm_dash_core::persist::PersistedEntry;
use rocm_dash_core::protocol::{Event, PROTOCOL_VERSION};

/// Parameters for a synthetic session. Same options → byte-identical output.
#[derive(Debug, Clone, Copy)]
pub struct DemoOptions {
    /// Number of seconds to simulate (one snapshot per second).
    pub seconds: u64,
    /// Seed for the deterministic synthetic data. Same seed → same bytes.
    pub seed: u64,
}

impl Default for DemoOptions {
    fn default() -> Self {
        // Historical defaults from the former `gen_demo` example.
        Self {
            seconds: 120,
            seed: 42,
        }
    }
}

/// Write a synthetic session as `PersistedEntry` NDJSON to `w`.
///
/// The byte stream is identical for identical [`DemoOptions`].
pub fn generate_to_writer<W: Write>(opts: &DemoOptions, w: &mut W) -> anyhow::Result<()> {
    // Guard absurd durations: keep the i64 second-of-epoch math (and the run
    // time / output size) bounded. Far beyond any real demo; rejects values
    // that would otherwise overflow the snapshot timestamp. Does not affect the
    // byte stream for any realistic `seconds`.
    anyhow::ensure!(
        opts.seconds <= MAX_DEMO_SECONDS,
        "seconds={} exceeds the maximum of {} (~115 days at 1Hz)",
        opts.seconds,
        MAX_DEMO_SECONDS
    );

    // Anchor wallclock to a fixed point so the file is byte-stable for any
    // given seed — useful for golden test fixtures and reproducible demos.
    // Wallclock used for replay pacing; the recording's `Snapshot.timestamp`
    // uses the same base.
    let wall_base_us: u64 = 1_750_000_000_000_000; // 2025-06-15T22:13:20Z
    let base_secs: i64 = 1_750_000_000;
    let mut seed = opts.seed;

    // Synthetic Welcome — the replayer emits its own ClientMsg::Connected,
    // but having a Welcome in-file makes the recording self-describing.
    let welcome = PersistedEntry {
        ts_us: wall_base_us,
        event: Event::Welcome {
            protocol_version: PROTOCOL_VERSION,
            daemon_version: "demo/gen_demo".into(),
            host: "demo-mi355x-node".into(),
        },
    };
    write_line(w, &welcome)?;

    // Schedule bench rows by their offset_s.
    let mut bench = demo_bench_rows();
    bench.sort_by_key(|(t, _)| *t);
    let mut bench_iter = bench.into_iter().peekable();

    for t in 0..opts.seconds {
        let t_s = t as f64;
        // Wallclock advances 1s per snapshot.
        let ts_us = wall_base_us + (t * 1_000_000);
        let snap = build_snapshot(t_s, base_secs, &mut seed);

        // Snapshot.
        let snap_entry = PersistedEntry {
            ts_us,
            event: Event::Snapshot(snap),
        };
        write_line(w, &snap_entry)?;

        // Instance discovery events on the very first tick.
        if t == 0 {
            for inst in build_instances(0.0, &mut seed) {
                let ev = PersistedEntry {
                    ts_us: ts_us + 1_000,
                    event: Event::InstanceDiscovered(inst),
                };
                write_line(w, &ev)?;
            }
        }

        // Any bench rows scheduled at this second.
        while bench_iter.peek().is_some_and(|(t_off, _)| *t_off == t) {
            let (_t_off, row) = bench_iter.next().unwrap();
            let ev = PersistedEntry {
                ts_us: ts_us + 500_000, // half a tick after the snapshot
                event: Event::BenchmarkRowsAppended { rows: vec![row] },
            };
            write_line(w, &ev)?;
        }
    }

    // Final Bye marker — replay's EOF path doesn't need this but it makes
    // the file complete-looking when curled.
    let bye = PersistedEntry {
        ts_us: wall_base_us + opts.seconds * 1_000_000 + 100_000,
        event: Event::Bye,
    };
    write_line(w, &bye)?;

    w.flush()?;
    Ok(())
}

/// Write a synthetic session to `path` (via a buffered file writer).
///
/// On a mid-write failure the partial file is removed so a failed run never
/// leaves a corrupt session behind.
pub fn generate_file(opts: &DemoOptions, path: &Path) -> anyhow::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    if let Err(e) = generate_to_writer(opts, &mut w) {
        drop(w);
        let _ = std::fs::remove_file(path);
        return Err(e);
    }
    Ok(())
}

/// Upper bound on `DemoOptions::seconds` — ~115 days at 1Hz. Keeps generation
/// O(seconds) bounded and the snapshot-timestamp math safely within `i64`.
const MAX_DEMO_SECONDS: u64 = 10_000_000;

const HOST_CORES: usize = 192;
const NUM_GPUS: usize = 8;
const VRAM_TOTAL_MB: u64 = 288 * 1024; // MI355X: 288 GiB HBM3E per GPU
const MEMORY_TOTAL_MB: u64 = 1024 * 1024; // 1 TiB host RAM
const SWAP_TOTAL_MB: u64 = 0; // typical for an inference box

/// Inference container template — a "tenant" of the node.
struct ContainerSpec {
    container_id: &'static str,
    container_name: &'static str,
    model_name: &'static str,
    gpu_ids: &'static [&'static str],
    tp: u32,
    port: u16,
    /// Phase of the activity sinusoid in seconds — staggers the three
    /// instances so the heatmap and per-GPU sparklines aren't synchronised.
    phase_s: f64,
    /// Mean kv-cache utilization (0..100) around which it oscillates.
    kv_mean: f32,
    /// Amplitude of the kv-cache oscillation.
    kv_amp: f32,
    /// Mean GPU utilization 0..100 used by the GPUs assigned to this container.
    util_mean: f32,
    util_amp: f32,
    /// Typical request concurrency (running_reqs mean).
    req_mean: u32,
    /// Quantization label shown in the detail modal.
    quantization: Option<&'static str>,
    /// Launch args excerpt rendered in the instance card / detail modal.
    launch_args: &'static [&'static str],
}

const CONTAINERS: &[ContainerSpec] = &[
    ContainerSpec {
        container_id: "c-llama3-70b-fp8-tp4",
        container_name: "vllm-llama3-70b",
        model_name: "meta-llama/Llama-3-70B-Instruct",
        gpu_ids: &["0", "1", "2", "3"],
        tp: 4,
        port: 8000,
        phase_s: 0.0,
        kv_mean: 55.0,
        kv_amp: 30.0,
        util_mean: 78.0,
        util_amp: 15.0,
        req_mean: 18,
        quantization: Some("fp8"),
        launch_args: &[
            "--model",
            "meta-llama/Llama-3-70B-Instruct",
            "--tensor-parallel-size",
            "4",
            "--dtype",
            "fp8",
            "--max-num-seqs",
            "256",
        ],
    },
    ContainerSpec {
        container_id: "c-mixtral-8x7b-bf16-tp2",
        container_name: "vllm-mixtral-8x7b",
        model_name: "mistralai/Mixtral-8x7B-Instruct-v0.1",
        gpu_ids: &["4", "5"],
        tp: 2,
        port: 8001,
        phase_s: 17.0,
        kv_mean: 35.0,
        kv_amp: 25.0,
        util_mean: 62.0,
        util_amp: 22.0,
        req_mean: 8,
        quantization: None,
        launch_args: &[
            "--model",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "--tensor-parallel-size",
            "2",
            "--dtype",
            "bfloat16",
            "--max-num-seqs",
            "128",
        ],
    },
    ContainerSpec {
        container_id: "c-deepseek-r1-fp8-tp2",
        container_name: "vllm-deepseek-r1",
        model_name: "deepseek-ai/DeepSeek-R1",
        gpu_ids: &["6", "7"],
        tp: 2,
        port: 8002,
        phase_s: 33.0,
        kv_mean: 70.0,
        kv_amp: 22.0,
        util_mean: 85.0,
        util_amp: 12.0,
        req_mean: 12,
        quantization: Some("fp8"),
        launch_args: &[
            "--model",
            "deepseek-ai/DeepSeek-R1",
            "--tensor-parallel-size",
            "2",
            "--dtype",
            "fp8",
            "--max-num-seqs",
            "192",
            "--enable-prefix-caching",
        ],
    },
];

/// Bench rows we sprinkle through the timeline. Each tuple is
/// (offset_s, BenchmarkRow). Times are picked so the Bench tab fills
/// gradually rather than all at once.
fn demo_bench_rows() -> Vec<(u64, BenchmarkRow)> {
    let mk = |cell: &str,
              run: u32,
              model: &str,
              tp: u32,
              dtype: &str,
              wall: f64,
              ptps: f64,
              gtps: f64,
              pass: PassFail|
     -> BenchmarkRow {
        BenchmarkRow {
            cell: cell.into(),
            run,
            model: Some(model.into()),
            engine: Some("vllm".into()),
            endpoint: Some("http://127.0.0.1:8000".to_string()),
            tp: Some(tp),
            dtype: Some(dtype.into()),
            attention_backend: Some("flash-attn-3".into()),
            concurrency: Some(64),
            max_num_seqs: Some(256),
            wall_s: Some(wall),
            n_requests: Some(64),
            prompt_tokens: Some(64 * 1024),
            completion_tokens: Some(64 * 512),
            prompt_tps: Some(ptps),
            gen_tps: Some(gtps),
            max_running_reqs: Some(48),
            max_waiting_reqs: Some(4),
            out_chars: Some(64 * 1500),
            pass_fail: pass,
            judge_pass_fail: pass,
            assertion_pass: Some(matches!(pass, PassFail::Pass)),
            ..Default::default()
        }
    };

    vec![
        (
            8,
            mk(
                "O-arch-llama3",
                1,
                "meta-llama/Llama-3-70B-Instruct",
                4,
                "fp8",
                42.3,
                1240.5,
                68.2,
                PassFail::Pass,
            ),
        ),
        (
            18,
            mk(
                "O-arch-mixtral",
                1,
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                2,
                "bf16",
                55.7,
                980.1,
                54.6,
                PassFail::Pass,
            ),
        ),
        (
            32,
            mk(
                "S-bench-deepseek",
                1,
                "deepseek-ai/DeepSeek-R1",
                2,
                "fp8",
                38.9,
                1610.4,
                82.1,
                PassFail::Pass,
            ),
        ),
        (
            47,
            mk(
                "O-arch-llama3",
                2,
                "meta-llama/Llama-3-70B-Instruct",
                4,
                "fp8",
                41.8,
                1268.2,
                69.4,
                PassFail::Pass,
            ),
        ),
        (
            62,
            mk(
                "S-bench-deepseek",
                2,
                "deepseek-ai/DeepSeek-R1",
                2,
                "fp8",
                40.1,
                1572.0,
                80.7,
                PassFail::Pass,
            ),
        ),
        (
            74,
            mk(
                "S-stress-mixtral",
                1,
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                2,
                "bf16",
                92.6,
                612.0,
                31.8,
                PassFail::Fail,
            ),
        ),
        (
            98,
            mk(
                "S-stress-mixtral",
                2,
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                2,
                "bf16",
                58.0,
                890.0,
                49.0,
                PassFail::Pass,
            ),
        ),
        (
            89,
            mk(
                "O-arch-mixtral",
                2,
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                2,
                "bf16",
                54.2,
                1006.3,
                56.0,
                PassFail::Pass,
            ),
        ),
        (
            104,
            mk(
                "R-reproduce",
                1,
                "deepseek-ai/DeepSeek-R1",
                2,
                "fp8",
                37.4,
                1644.8,
                84.0,
                PassFail::Pass,
            ),
        ),
    ]
}

/// Deterministic pseudo-random oscillator. Same seed + t → same value.
fn osc(t_s: f64, period_s: f64, phase_s: f64, mean: f32, amp: f32) -> f32 {
    let theta = 2.0 * PI * (t_s + phase_s) / period_s;
    amp.mul_add(theta.sin() as f32, mean)
}

/// Tiny xorshift32 used to give each value a small jitter so the trace
/// doesn't look like a perfect sine wave. Returns 0.0..1.0.
fn jitter(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed % 1000) as f32) / 1000.0
}

fn build_host(t_s: f64, seed: &mut u64) -> SystemMetrics {
    // Inference is GPU-bound, so CPU stays moderate with some noise.
    let agg = (jitter(seed) - 0.5).mul_add(4.0, osc(t_s, 23.0, 0.0, 28.0, 8.0));
    let agg = agg.clamp(0.0, 100.0);

    let mut per_core: Vec<f32> = Vec::with_capacity(HOST_CORES);
    for i in 0..HOST_CORES {
        // Each core's load is the aggregate ± a per-core offset so the
        // CoreBars widget shows visible variation across cores.
        let off = osc(t_s, 7.0 + (i as f64 % 11.0), i as f64 * 0.7, 0.0, 18.0);
        let v = (jitter(seed) - 0.5)
            .mul_add(6.0, agg + off)
            .clamp(0.0, 100.0);
        per_core.push(v);
    }

    let mem_used =
        (MEMORY_TOTAL_MB as f64 * 0.42 * 0.02f64.mul_add((t_s / 31.0).sin(), 1.0)) as u64;

    SystemMetrics {
        cpu_overall_pct: agg,
        cpu_per_core_pct: per_core,
        memory_used_mb: mem_used,
        memory_total_mb: MEMORY_TOTAL_MB,
        swap_used_mb: 0,
        swap_total_mb: SWAP_TOTAL_MB,
        disk_read_bps: (jitter(seed) * 50_000_000.0) as u64,
        disk_write_bps: (jitter(seed) * 20_000_000.0) as u64,
        net_rx_bps: (jitter(seed) * 80_000_000.0) as u64,
        net_tx_bps: (jitter(seed) * 80_000_000.0) as u64,
    }
}

fn build_gpu(idx: usize, t_s: f64, seed: &mut u64) -> GpuMetrics {
    // Find which container owns this GPU and inherit its activity profile.
    let (util_mean, util_amp, phase) = CONTAINERS
        .iter()
        .find(|c| {
            c.gpu_ids
                .iter()
                .any(|g| g.parse::<usize>().ok() == Some(idx))
        })
        .map_or((20.0, 5.0, 0.0), |c| (c.util_mean, c.util_amp, c.phase_s));

    let util = (jitter(seed) - 0.5).mul_add(
        5.0,
        osc(
            t_s,
            11.0,
            (idx as f64).mul_add(1.3, phase),
            util_mean,
            util_amp,
        ),
    );
    let util = util.clamp(0.0, 100.0);

    // Temp tracks util with thermal lag — small offset, narrower swing.
    let temp = (jitter(seed) - 0.5).mul_add(1.5, 0.22f32.mul_add(util, 55.0));
    // Power tracks util more linearly. MI355X TDP ~ 750W.
    let power = (jitter(seed) - 0.5).mul_add(18.0, 5.2f32.mul_add(util, 220.0));
    // VRAM is mostly the model weights once loaded.
    let vram_used = (f64::from(jitter(seed)) * VRAM_TOTAL_MB as f64)
        .mul_add(0.02, VRAM_TOTAL_MB as f64 * 0.74) as u64;

    GpuMetrics {
        device_id: format!("gpu-{idx}"),
        vram_used_mb: vram_used,
        vram_total_mb: VRAM_TOTAL_MB,
        gpu_utilization_pct: util,
        temperature_c: temp,
        power_w: power,
        clock_mhz: Some((jitter(seed) - 0.5).mul_add(80.0, 1850.0)),
    }
}

fn build_sysinfo() -> GpuSystemInfo {
    GpuSystemInfo {
        rocm_version: Some("7.13.0".into()),
        driver_version: Some("6.10.5".into()),
        gpu_model: "AMD Instinct MI355X".into(),
        physical_gpu_count: NUM_GPUS as u32,
        logical_gpu_count: NUM_GPUS as u32,
        partition_mode: ComputePartitionMode::Spx,
        memory_partition_mode: MemoryPartitionMode::Nps1,
        compute_partition_mode: ComputePartitionMode::Spx,
        vram_per_logical_gpu_mb: VRAM_TOTAL_MB,
        lemond_version: None,
        llama_server_build: None,
        ccr_version: None,
        llamacpp_backend: None,
    }
}

fn build_instances(t_s: f64, seed: &mut u64) -> Vec<Instance> {
    CONTAINERS
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            let kv =
                (jitter(seed) - 0.5).mul_add(4.0, osc(t_s, 13.0, c.phase_s, c.kv_mean, c.kv_amp));
            let kv = kv.clamp(0.0, 100.0);
            let running = ((c.req_mean as f32
                + osc(t_s, 17.0, c.phase_s, 0.0, c.req_mean as f32 * 0.6))
            .max(0.0)) as u32;
            // Waiting reqs occasionally burst.
            let waiting_base = osc(t_s, 29.0, c.phase_s + 5.0, 1.5, 1.5).max(0.0);
            let waiting = if jitter(seed) > 0.92 {
                (waiting_base + 4.0) as u32
            } else {
                waiting_base as u32
            };

            // VRAM per instance = sum of its assigned GPUs' weight footprint.
            let vram_used = c.gpu_ids.len() as u64 * (VRAM_TOTAL_MB * 74 / 100);
            let vram_total = c.gpu_ids.len() as u64 * VRAM_TOTAL_MB;

            let mut env_vars = BTreeMap::new();
            env_vars.insert("HIP_VISIBLE_DEVICES".into(), c.gpu_ids.join(","));
            env_vars.insert("VLLM_USE_TRITON_FLASH_ATTN".into(), "1".into());
            env_vars.insert("HF_HOME".into(), "/data/hf".into());

            // Generation throughput scales with the instance's GPU count and
            // oscillates with load; tokens_per_watt is filled in build_snapshot
            // once GPU power is known (mirrors the runner's assembly step).
            let gen_tps = (c.gpu_ids.len() as f64
                * f64::from(osc(t_s, 19.0, c.phase_s, 180.0, 90.0)))
            .max(0.0);

            // Synthesize plausible live latency so `rocm dash --demo` shows real
            // TTFT/TPOT values: TTFT ~80–360 ms, TPOT ~10–40 ms, oscillating
            // with load. The first instance leaves them `None` to exercise the
            // honest `—` path (an engine that doesn't expose the histogram).
            let (ttft_ms, tpot_ms) = if idx == 0 {
                (None, None)
            } else {
                (
                    Some(f64::from(osc(t_s, 23.0, c.phase_s, 220.0, 140.0)).max(20.0)),
                    Some(f64::from(osc(t_s, 11.0, c.phase_s + 3.0, 25.0, 15.0)).max(5.0)),
                )
            };

            Instance {
                container_id: c.container_id.into(),
                container_name: c.container_name.into(),
                status: InstanceStatus::Running,
                model_name: c.model_name.into(),
                gpu_ids: c.gpu_ids.iter().map(|s| (*s).to_string()).collect(),
                partition_info: Some("SPX/NPS1".into()),
                quantization: c.quantization.map(std::string::ToString::to_string),
                tensor_parallel_size: c.tp,
                port: Some(c.port),
                vram_used_mb: vram_used,
                vram_total_mb: vram_total,
                kv_cache_usage_pct: Some(kv),
                running_reqs: Some(running),
                waiting_reqs: Some(waiting),
                gen_tps: Some(gen_tps),
                tokens_per_watt: None,
                ttft_ms,
                tpot_ms,
                launch_args: c.launch_args.iter().map(|s| (*s).to_string()).collect(),
                env_vars,
                log_file: Some(format!("/var/log/vllm/{}.log", c.container_name)),
            }
        })
        .collect()
}

fn build_snapshot(t_s: f64, base_secs: i64, seed: &mut u64) -> Snapshot {
    let timestamp = Utc.timestamp_opt(base_secs + t_s as i64, 0).unwrap();
    let gpus: Vec<_> = (0..NUM_GPUS).map(|i| build_gpu(i, t_s, seed)).collect();
    let mut instances = build_instances(t_s, seed);
    // Mirror the runner: derive tokens_per_watt once both throughput and GPU
    // power are in hand, exercising the same id-normalizing join.
    for inst in &mut instances {
        inst.tokens_per_watt =
            rocm_dash_core::efficiency::tokens_per_watt(inst.gen_tps, &inst.gpu_ids, &gpus);
    }
    Snapshot {
        timestamp,
        host: build_host(t_s, seed),
        gpus,
        gpu_system_info: Some(build_sysinfo()),
        instances,
        warnings: Vec::new(),
    }
}

fn write_line<W: Write + ?Sized>(w: &mut W, entry: &PersistedEntry) -> anyhow::Result<()> {
    let line = serde_json::to_string(entry)?;
    w.write_all(line.as_bytes())?;
    w.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_synthesizes_finite_ttft_tpot_with_one_honest_none() {
        // `rocm dash --demo` must show real TTFT/TPOT (Phase 4): the synthesized
        // instances carry finite latency, with the first left None to exercise
        // the honest `—` path (an engine without the histogram).
        let mut seed = 42;
        let insts = build_instances(5.0, &mut seed);
        assert!(insts.len() >= 2, "need >1 demo instance to test both paths");
        assert_eq!(insts[0].ttft_ms, None, "first instance is the honest None");
        assert_eq!(insts[0].tpot_ms, None);
        for inst in &insts[1..] {
            let ttft = inst.ttft_ms.expect("synthesized TTFT");
            let tpot = inst.tpot_ms.expect("synthesized TPOT");
            assert!(
                ttft.is_finite() && ttft > 0.0,
                "TTFT finite/positive: {ttft}"
            );
            assert!(
                tpot.is_finite() && tpot > 0.0,
                "TPOT finite/positive: {tpot}"
            );
        }
    }

    /// Same options must yield byte-identical output (hard invariant).
    #[test]
    fn deterministic_same_seed_byte_identical() {
        let opts = DemoOptions {
            seconds: 30,
            seed: 7,
        };
        let mut a = Vec::new();
        let mut b = Vec::new();
        generate_to_writer(&opts, &mut a).unwrap();
        generate_to_writer(&opts, &mut b).unwrap();
        assert_eq!(a, b, "identical DemoOptions produced differing bytes");
        assert!(!a.is_empty(), "generator produced no output");
    }

    /// Different seeds should diverge (sanity: the seed actually matters).
    #[test]
    fn different_seed_differs() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        generate_to_writer(
            &DemoOptions {
                seconds: 20,
                seed: 1,
            },
            &mut a,
        )
        .unwrap();
        generate_to_writer(
            &DemoOptions {
                seconds: 20,
                seed: 2,
            },
            &mut b,
        )
        .unwrap();
        assert_ne!(a, b, "distinct seeds produced identical output");
    }

    /// Every emitted line must deserialize as a `PersistedEntry`, the first
    /// event must be a `Welcome`, and the stream must end with `Event::Bye`.
    #[test]
    fn replay_parity_lines_deserialize_and_end_with_bye() {
        let opts = DemoOptions {
            seconds: 15,
            seed: 42,
        };
        let mut buf = Vec::new();
        generate_to_writer(&opts, &mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();

        let entries: Vec<PersistedEntry> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                serde_json::from_str::<PersistedEntry>(l)
                    .unwrap_or_else(|e| panic!("line failed to deserialize: {e}\n{l}"))
            })
            .collect();

        assert!(!entries.is_empty(), "no entries produced");
        assert!(
            matches!(entries.first().unwrap().event, Event::Welcome { .. }),
            "stream did not start with Welcome"
        );
        assert!(
            matches!(entries.last().unwrap().event, Event::Bye),
            "stream did not end with Bye"
        );
        // At least one Snapshot-bearing entry exists.
        assert!(
            entries
                .iter()
                .any(|e| matches!(e.event, Event::Snapshot(_))),
            "no Snapshot events in stream"
        );
    }

    /// seconds = 0 still yields a valid minimal stream (Welcome + Bye).
    #[test]
    fn seconds_zero_is_valid_minimal_stream() {
        let opts = DemoOptions {
            seconds: 0,
            seed: 42,
        };
        let mut buf = Vec::new();
        generate_to_writer(&opts, &mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let entries: Vec<PersistedEntry> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<PersistedEntry>(l).unwrap())
            .collect();
        assert_eq!(entries.len(), 2, "expected exactly Welcome + Bye");
        assert!(matches!(entries[0].event, Event::Welcome { .. }));
        assert!(matches!(entries[1].event, Event::Bye));
    }

    /// `generate_file` writes the same bytes as `generate_to_writer`.
    #[test]
    fn generate_file_matches_writer() {
        let opts = DemoOptions {
            seconds: 10,
            seed: 3,
        };
        let mut expected = Vec::new();
        generate_to_writer(&opts, &mut expected).unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "rocm-dash-demo-test-{}-{}.ndjson",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        generate_file(&opts, &path).unwrap();
        let on_disk = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(on_disk, expected, "generate_file diverged from writer");
    }

    /// Absurd durations are rejected with a clean error, not a panic.
    #[test]
    fn seconds_over_cap_errors_cleanly() {
        let opts = DemoOptions {
            seconds: MAX_DEMO_SECONDS + 1,
            seed: 42,
        };
        let mut buf = Vec::new();
        let err = generate_to_writer(&opts, &mut buf).unwrap_err();
        assert!(
            err.to_string().contains("exceeds the maximum"),
            "unexpected error: {err}"
        );
    }
}
