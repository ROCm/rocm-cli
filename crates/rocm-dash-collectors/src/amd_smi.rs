// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! amd-smi subprocess + JSON parse.
//!
//! Field paths and the KFD pre-flight check are vendored from the TypeScript
//! `AmdSmiProvider` in instinct-dash. See `../wiki/entities/amd-smi.md`.

use std::ffi::OsString;
use std::io;
use std::time::Duration;

use rocm_dash_core::metrics::{GpuMetrics, GpuSystemInfo};
use rocm_dash_core::partition::{ComputePartitionMode, MemoryPartitionMode};
use rocm_dash_core::traits::GpuProcess;
use serde_json::Value;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::warn;

const KFD_DEVICE: &str = "/dev/kfd";
const DETECT_TIMEOUT: Duration = Duration::from_secs(5);
const RUN_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug)]
pub struct AmdSmiCollector {
    binary: OsString,
}

impl AmdSmiCollector {
    /// Returns `Some` only if `/dev/kfd` is readable AND `amd-smi version` succeeds.
    ///
    /// The KFD pre-flight is mandatory: without it, `amd-smi` blocks in
    /// uninterruptible kernel sleep (D-state) that no signal can escape.
    pub async fn detect() -> Option<Self> {
        Self::detect_with_binary("amd-smi").await
    }

    /// Like [`detect`](Self::detect) but uses an explicit `amd-smi` binary path.
    ///
    /// The managed ROCm SDK ships `amd-smi` inside the runtime wheel's bin
    /// directory rather than on `PATH`, so callers resolve the path or command
    /// name (via `rocm_core::resolve_amd_smi_binary`) and pass it here.
    pub async fn detect_with_binary(binary: impl Into<OsString>) -> Option<Self> {
        if !kfd_accessible() {
            return None;
        }
        let me = Self {
            binary: binary.into(),
        };
        match timeout(DETECT_TIMEOUT, me.run(&["version"])).await {
            Ok(Ok(_)) => Some(me),
            Ok(Err(e)) => {
                warn!(error = %e, binary = ?me.binary, "amd-smi present but `version` failed");
                None
            }
            Err(_) => {
                warn!(binary = ?me.binary, "amd-smi `version` timed out");
                None
            }
        }
    }

    pub async fn metrics(&self) -> io::Result<Vec<GpuMetrics>> {
        let v = self.run_json(&["metric", "--json"]).await?;
        Ok(parse_metrics(&v))
    }

    /// Per-process GPU VRAM usage from `amd-smi process --json`.
    ///
    /// Inherent async (mirrors [`metrics`]); the runner uses this directly
    /// rather than the sync `GpuCollector::processes` trait method. Parsing is
    /// delegated to the defensive pure [`parse_processes`]; see its docs for
    /// the assumed schema and version-variance handling.
    pub async fn processes(&self) -> io::Result<Vec<GpuProcess>> {
        let v = self.run_json(&["process", "--json"]).await?;
        Ok(parse_processes(&v))
    }

    /// Best-effort system info — each sub-call is tolerated independently
    /// (matches the `Promise.allSettled` pattern in instinct-dash).
    pub async fn system_info(&self) -> GpuSystemInfo {
        let (ver, stat, topo) = tokio::join!(
            self.run_json(&["version", "--json"]),
            self.run_json(&["static", "--json"]),
            self.run_json(&["topology", "--json"]),
        );
        parse_system_info(ver.ok(), stat.ok(), topo.ok())
    }

    async fn run(&self, args: &[&str]) -> io::Result<String> {
        let out = Command::new(&self.binary).args(args).output().await?;
        if !out.status.success() {
            return Err(io::Error::other(format!(
                "amd-smi {args:?} exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    async fn run_json(&self, args: &[&str]) -> io::Result<Value> {
        let s = timeout(RUN_TIMEOUT, self.run(args))
            .await
            .map_err(|_| io::Error::other(format!("amd-smi {args:?} timed out")))??;
        serde_json::from_str(&s).map_err(|e| io::Error::other(format!("parse: {e}")))
    }
}

fn kfd_accessible() -> bool {
    std::fs::OpenOptions::new()
        .read(true)
        .open(KFD_DEVICE)
        .is_ok()
}

fn val_u64(v: Option<&Value>) -> Option<u64> {
    v?.as_u64()
        .or_else(|| v?.as_f64().map(|x| x.round() as u64))
}

fn val_f32(v: Option<&Value>) -> Option<f32> {
    v?.as_f64().map(|x| x as f32)
}

fn nested<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(v, |cur, k| cur.get(*k))
}

pub fn parse_metrics(v: &Value) -> Vec<GpuMetrics> {
    let Some(gpus) = v.get("gpu_data").and_then(|x| x.as_array()) else {
        return Vec::new();
    };
    gpus.iter()
        .map(|g| {
            let id = g
                .get("gpu")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            // Prefer hotspot — edge is N/A on MI300X SR-IOV.
            let temperature_c = val_f32(nested(g, &["temperature", "hotspot", "value"]))
                .or_else(|| val_f32(nested(g, &["temperature", "edge", "value"])))
                .unwrap_or(0.0);
            GpuMetrics {
                device_id: format!("gpu-{id}"),
                vram_used_mb: val_u64(nested(g, &["mem_usage", "used_vram", "value"])).unwrap_or(0),
                vram_total_mb: val_u64(nested(g, &["mem_usage", "total_vram", "value"]))
                    .unwrap_or(0),
                gpu_utilization_pct: val_f32(nested(g, &["usage", "gfx_activity", "value"]))
                    .unwrap_or(0.0),
                temperature_c,
                power_w: val_f32(nested(g, &["power", "socket_power", "value"])).unwrap_or(0.0),
                clock_mhz: val_f32(nested(g, &["clock", "gfx_0", "clk", "value"])),
            }
        })
        .collect()
}

/// Candidate `(parent, child)` JSON paths for a process's VRAM figure, in
/// priority order. amd-smi naming varies by version, so we probe several.
const VRAM_PATHS: &[(&str, &str)] = &[
    ("memory_usage", "vram_mem"),
    ("mem_usage", "vram_mem"),
    ("memory_usage", "vram_usage"),
    ("mem_usage", "vram_usage"),
];

/// Convert a memory figure to MB given its unit string. Unrecognized or absent
/// units are treated as MB (the unit amd-smi reports for the structured
/// `{value,unit}` form). Case-insensitive; binary multiples (KiB/GiB) alias the
/// decimal-looking unit names amd-smi emits.
fn mem_unit_to_mb(value: u64, unit: &str) -> u64 {
    match unit.to_ascii_uppercase().as_str() {
        "B" | "BYTES" => value / (1024 * 1024),
        "KB" | "KIB" => value / 1024,
        "GB" | "GIB" => value * 1024,
        // "MB" / "MIB" / unknown → already MB.
        _ => value,
    }
}

/// Extract a u64 from a value that is either a raw number or a `{value}` wrapper.
fn u64_or_wrapped(v: &Value) -> Option<u64> {
    val_u64(Some(v)).or_else(|| val_u64(v.get("value")))
}

/// Resolve a process entry's VRAM in MB across schema variants. A raw numeric
/// VRAM (no `{value,unit}` wrapper) is assumed to be **bytes** — the form
/// amd-smi's `vram_mem` field takes when emitted without a unit annotation.
/// Absent → 0.
fn process_vram_mb(p: &Value) -> u64 {
    let raw = VRAM_PATHS
        .iter()
        .find_map(|(parent, child)| p.get(parent).and_then(|m| m.get(child)));
    let Some(v) = raw else { return 0 };
    if v.is_object() {
        let value = val_u64(v.get("value")).unwrap_or(0);
        let unit = v.get("unit").and_then(|x| x.as_str()).unwrap_or("MB");
        mem_unit_to_mb(value, unit)
    } else {
        // Raw number with no unit → bytes (documented assumption).
        mem_unit_to_mb(val_u64(Some(v)).unwrap_or(0), "B")
    }
}

/// Parse `amd-smi process --json` into per-process VRAM records.
///
/// **Defensive across amd-smi versions.** Assumed schema (GPU-indexed):
///  - Top level: `{ "gpu_data": [ <gpu>, ... ] }` OR a bare `[ <gpu>, ... ]`.
///  - Each `<gpu>`: `{ "gpu": <index>, "process_list" | "processes": [ <item>, ... ] }`.
///  - Each `<item>`: flat, or wrapped in `{ "process_info": { ... } }`.
///  - PID: `pid` as a raw number or a `{ "value": N }` wrapper.
///  - VRAM: `memory_usage.vram_mem` (or `mem_usage.vram_mem` /
///    `*.vram_usage`); a `{value,unit}` wrapper (unit normalized to MB) or a
///    raw number (assumed bytes). See [`process_vram_mb`].
///
/// Entries with no resolvable PID are skipped; VRAM defaults to 0 when absent.
/// Never panics on shape mismatch — unknown shapes yield an empty `Vec`.
pub fn parse_processes(v: &Value) -> Vec<GpuProcess> {
    let entries: Vec<&Value> = if let Some(arr) = v.get("gpu_data").and_then(|x| x.as_array()) {
        arr.iter().collect()
    } else if let Some(arr) = v.as_array() {
        arr.iter().collect()
    } else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for g in entries {
        let id = g
            .get("gpu")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let device_id = format!("gpu-{id}");
        let plist = g
            .get("process_list")
            .and_then(|x| x.as_array())
            .or_else(|| g.get("processes").and_then(|x| x.as_array()));
        let Some(plist) = plist else { continue };
        for item in plist {
            let p = item.get("process_info").unwrap_or(item);
            let Some(pid) = p.get("pid").and_then(u64_or_wrapped) else {
                continue;
            };
            out.push(GpuProcess {
                pid: pid as u32,
                device_id: device_id.clone(),
                vram_used_mb: process_vram_mb(p),
            });
        }
    }
    out
}

pub fn parse_system_info(
    ver: Option<Value>,
    stat: Option<Value>,
    topo: Option<Value>,
) -> GpuSystemInfo {
    let mut info = GpuSystemInfo::default();

    if let Some(v) = ver {
        let item = match v {
            Value::Array(mut a) if !a.is_empty() => a.swap_remove(0),
            other => other,
        };
        info.rocm_version = item
            .get("rocm_version")
            .or_else(|| item.get("ROCm_version"))
            .and_then(|x| x.as_str())
            .map(str::to_owned);
        info.driver_version = item
            .get("amdgpu_version")
            .or_else(|| item.get("driver"))
            .and_then(|x| x.as_str())
            .map(str::to_owned);
    }

    if let Some(s) = stat
        && let Some(gpus) = s.get("gpu_data").and_then(|x| x.as_array())
    {
        info.physical_gpu_count = gpus.len() as u32;
        info.logical_gpu_count = info.physical_gpu_count;
        if let Some(first) = gpus.first() {
            info.gpu_model = nested(first, &["asic", "market_name"])
                .and_then(|x| x.as_str())
                .filter(|s| *s != "N/A")
                .or_else(|| nested(first, &["board", "product_name"]).and_then(|x| x.as_str()))
                .map_or_else(|| "Unknown".into(), str::to_owned);
        }
    }

    if let Some(t) = topo {
        info.partition_mode = parse_compute(
            t.get("partition_mode")
                .or_else(|| t.get("compute_partition_mode")),
        );
        info.memory_partition_mode = parse_memory(t.get("memory_partition_mode"));
        info.compute_partition_mode = parse_compute(
            t.get("compute_partition_mode")
                .or_else(|| t.get("partition_mode")),
        );
    }

    info
}

fn parse_compute(v: Option<&Value>) -> ComputePartitionMode {
    match v
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_ascii_uppercase()
        .as_str()
    {
        "SPX" => ComputePartitionMode::Spx,
        "DPX" => ComputePartitionMode::Dpx,
        "QPX" => ComputePartitionMode::Qpx,
        "CPX" => ComputePartitionMode::Cpx,
        _ => ComputePartitionMode::Unknown,
    }
}

fn parse_memory(v: Option<&Value>) -> MemoryPartitionMode {
    match v
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_ascii_uppercase()
        .as_str()
    {
        "NPS1" => MemoryPartitionMode::Nps1,
        "NPS2" => MemoryPartitionMode::Nps2,
        "NPS4" => MemoryPartitionMode::Nps4,
        _ => MemoryPartitionMode::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_METRIC: &str = r#"{
      "gpu_data": [
        {
          "gpu": 0,
          "mem_usage":   { "used_vram":  { "value": 1234, "unit": "MB" },
                           "total_vram": { "value": 196608, "unit": "MB" } },
          "usage":       { "gfx_activity":  { "value": 42.5, "unit": "%" } },
          "temperature": { "hotspot":  { "value": 68.0, "unit": "C" },
                           "edge":     { "value": null } },
          "power":       { "socket_power": { "value": 410.5, "unit": "W" } }
        }
      ]
    }"#;

    #[test]
    fn parses_a_single_gpu_metric_block() {
        let v: Value = serde_json::from_str(SAMPLE_METRIC).unwrap();
        let m = parse_metrics(&v);
        assert_eq!(m.len(), 1);
        let g = &m[0];
        assert_eq!(g.device_id, "gpu-0");
        assert_eq!(g.vram_used_mb, 1234);
        assert_eq!(g.vram_total_mb, 196_608);
        assert!((g.gpu_utilization_pct - 42.5).abs() < 0.01);
        assert!((g.temperature_c - 68.0).abs() < 0.01);
        assert!((g.power_w - 410.5).abs() < 0.01);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn missing_fields_default_safely() {
        let v: Value = serde_json::from_str(r#"{ "gpu_data": [ { "gpu": 7 } ] }"#).unwrap();
        let m = parse_metrics(&v);
        assert_eq!(m[0].device_id, "gpu-7");
        assert_eq!(m[0].vram_used_mb, 0);
        assert_eq!(m[0].temperature_c, 0.0);
    }

    #[test]
    fn temperature_falls_back_to_edge_when_no_hotspot() {
        let v: Value = serde_json::from_str(
            r#"{ "gpu_data": [ { "gpu": 0, "temperature": { "edge": { "value": 55.0 } } } ] }"#,
        )
        .unwrap();
        let m = parse_metrics(&v);
        assert!((m[0].temperature_c - 55.0).abs() < 0.01);
    }

    #[test]
    fn partition_modes_parse_case_insensitive() {
        let topo = serde_json::json!({
            "partition_mode": "spx",
            "memory_partition_mode": "NPS4",
            "compute_partition_mode": "QPX"
        });
        let info = parse_system_info(None, None, Some(topo));
        assert_eq!(info.partition_mode, ComputePartitionMode::Spx);
        assert_eq!(info.memory_partition_mode, MemoryPartitionMode::Nps4);
        assert_eq!(info.compute_partition_mode, ComputePartitionMode::Qpx);
    }

    #[test]
    fn system_info_uses_market_name_when_available() {
        let stat = serde_json::json!({
            "gpu_data": [
                { "asic": { "market_name": "Instinct MI300X" } },
                { "asic": { "market_name": "Instinct MI300X" } }
            ]
        });
        let info = parse_system_info(None, Some(stat), None);
        assert_eq!(info.physical_gpu_count, 2);
        assert_eq!(info.gpu_model, "Instinct MI300X");
    }

    // --- process parsing -------------------------------------------------

    // gpu_data wrapper + process_list + process_info wrapper +
    // memory_usage.vram_mem {value, unit:"MB"}; second entry has no pid.
    const SAMPLE_PROCESS_WRAPPED: &str = r#"{
      "gpu_data": [
        {
          "gpu": 0,
          "process_list": [
            { "process_info": { "pid": 12345, "name": "vllm",
                "memory_usage": { "vram_mem": { "value": 4096, "unit": "MB" } } } },
            { "process_info": { "name": "no-pid",
                "memory_usage": { "vram_mem": { "value": 100, "unit": "MB" } } } }
          ]
        }
      ]
    }"#;

    // bare top-level array + flat process (no process_info) + raw-bytes vram_mem.
    const SAMPLE_PROCESS_BARE: &str = r#"[
      { "gpu": 3, "processes": [ { "pid": 999, "mem_usage": { "vram_mem": 2147483648 } } ] }
    ]"#;

    #[test]
    fn parses_wrapped_process_list_value_unit_mb() {
        let v: Value = serde_json::from_str(SAMPLE_PROCESS_WRAPPED).unwrap();
        let procs = parse_processes(&v);
        assert_eq!(procs.len(), 1, "the no-pid entry must be skipped");
        assert_eq!(procs[0].pid, 12345);
        assert_eq!(procs[0].device_id, "gpu-0");
        assert_eq!(procs[0].vram_used_mb, 4096);
    }

    #[test]
    fn parses_bare_array_flat_process_raw_bytes() {
        let v: Value = serde_json::from_str(SAMPLE_PROCESS_BARE).unwrap();
        let procs = parse_processes(&v);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].pid, 999);
        assert_eq!(procs[0].device_id, "gpu-3");
        // 2 GiB raw bytes → 2048 MB.
        assert_eq!(procs[0].vram_used_mb, 2048);
    }

    #[test]
    fn vram_unit_normalization_bytes_and_mb() {
        // {value, unit:"B"}: 1 GiB → 1024 MB.
        let v: Value = serde_json::from_str(
            r#"{ "gpu_data": [ { "gpu": 0, "process_list": [
                { "pid": 1, "memory_usage": { "vram_mem": { "value": 1073741824, "unit": "B" } } }
            ] } ] }"#,
        )
        .unwrap();
        assert_eq!(parse_processes(&v)[0].vram_used_mb, 1024);

        // {value, unit:"MB"} stays as-is.
        let v: Value = serde_json::from_str(
            r#"{ "gpu_data": [ { "gpu": 0, "process_list": [
                { "pid": 2, "memory_usage": { "vram_mem": { "value": 512, "unit": "MB" } } }
            ] } ] }"#,
        )
        .unwrap();
        assert_eq!(parse_processes(&v)[0].vram_used_mb, 512);
    }

    #[test]
    fn empty_and_garbage_yield_no_processes() {
        assert!(parse_processes(&serde_json::json!({})).is_empty());
        assert!(parse_processes(&serde_json::json!([])).is_empty());
        assert!(parse_processes(&serde_json::json!("garbage")).is_empty());
        assert!(parse_processes(&serde_json::json!(42)).is_empty());
        // gpu entry present but no process list.
        assert!(parse_processes(&serde_json::json!({ "gpu_data": [ { "gpu": 0 } ] })).is_empty());
    }

    #[test]
    fn process_without_pid_is_skipped() {
        let v: Value = serde_json::from_str(
            r#"{ "gpu_data": [ { "gpu": 0, "process_list": [
                { "memory_usage": { "vram_mem": { "value": 100, "unit": "MB" } } }
            ] } ] }"#,
        )
        .unwrap();
        assert!(parse_processes(&v).is_empty());
    }

    #[tokio::test]
    #[ignore = "requires a real AMD GPU + amd-smi; run manually on hardware"]
    async fn live_processes_no_panic() {
        if let Some(c) = AmdSmiCollector::detect().await {
            // Either Ok or Err is acceptable; the contract is "does not panic".
            let _ = c.processes().await;
        }
    }
}
