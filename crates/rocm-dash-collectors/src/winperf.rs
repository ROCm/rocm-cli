// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Live Windows GPU metrics via OS performance counters, no AMD tool needed.
//!
//! Reads the `GPU Engine`/`GPU Adapter Memory` categories — the same
//! WDDM/DXGK-backed source Task Manager's GPU tab reads. Mirrors
//! `amd_smi.rs`'s subprocess + pure-parse shape, shelling out to
//! `powershell.exe` instead of `amd-smi`. Not `#[cfg(windows)]`-gated: the
//! shellout fails cleanly (`powershell.exe` not found) on other platforms,
//! same as `amd_smi.rs` when `amd-smi` isn't installed.

use std::collections::BTreeMap;
use std::io;
use std::time::Duration;

use rocm_dash_core::metrics::{GpuMetrics, GpuSystemInfo};
use tokio::process::Command;
use tokio::time::timeout;

// 5s was too tight: on real hardware `powershell.exe` child-process spawn
// (not the Get-Counter query itself, which returns near-instantly once
// running) can take well over 5s — observed consistently timing out every
// tick on a Strix Halo laptop while the identical script ran fine
// interactively. 15s gives real-world spawn latency room without masking an
// actually-hung process forever.
const RUN_TIMEOUT: Duration = Duration::from_secs(15);

/// `kind<TAB>key<TAB>value[<TAB>value2]` lines: per-engine utilization,
/// per-adapter dedicated VRAM usage, and a best-effort adapter inventory
/// (name/RAM/driver) for `GpuSystemInfo`. One process spawn for all three.
const WINPERF_SCRIPT: &str = r#"
$ErrorActionPreference = 'SilentlyContinue'
(Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples |
    ForEach-Object { "UTIL`t$($_.InstanceName)`t$($_.CookedValue)" }
(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples |
    ForEach-Object { "VRAM`t$($_.InstanceName)`t$($_.CookedValue)" }
Get-CimInstance -ClassName Win32_VideoController -Property Name,AdapterRAM,DriverVersion -ErrorAction SilentlyContinue |
    ForEach-Object { "GPU`t$($_.Name)`t$($_.AdapterRAM)`t$($_.DriverVersion)" }
"#;

#[derive(Debug, Default)]
pub struct WinPerfCollector;

impl WinPerfCollector {
    pub const fn new() -> Self {
        Self
    }

    /// One `powershell.exe` spawn covers both metrics and system info (the
    /// script already emits both sections) — calling `metrics()` and
    /// `system_info()` separately each tick would shell out twice for the
    /// same output, doubling exposure to spawn latency for nothing.
    pub async fn snapshot(&self) -> io::Result<(Vec<GpuMetrics>, GpuSystemInfo)> {
        let out = Self::run_script().await?;
        Ok((parse_metrics(&out), parse_system_info(&out)))
    }

    async fn run_script() -> io::Result<String> {
        let out = timeout(
            RUN_TIMEOUT,
            Command::new("powershell.exe")
                .args(["-NoProfile", "-NonInteractive", "-Command", WINPERF_SCRIPT])
                .output(),
        )
        .await
        .map_err(|_| io::Error::other("powershell Get-Counter query timed out"))??;
        if !out.status.success() {
            return Err(io::Error::other(format!(
                "powershell Get-Counter exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

/// `GPU Engine` instances are per-process-per-engine
/// (`pid_1234_luid_0x..._0x..._phys_0_eng_0_engtype_3d`); `GPU Adapter
/// Memory` instances are per-adapter only (`luid_0x..._0x..._phys_0`, no
/// pid). Strip any `pid_NNNN_` prefix before keying so both counter
/// families collapse onto the same per-adapter key — otherwise every
/// process using the GPU looks like a distinct phantom device.
fn device_key(instance: &str) -> &str {
    let instance = instance.find("luid_").map_or(instance, |i| &instance[i..]);
    instance.split("_phys_").next().unwrap_or(instance)
}

/// Builds one `GpuMetrics` per distinct adapter LUID seen in the counter
/// output. Utilization is the max across that adapter's engines (3D/copy/
/// video/etc.) — the same "overall GPU" reading Task Manager shows.
///
/// `vram_total_mb` is only filled in when there is exactly one adapter: with
/// more than one, correlating a `GPU Adapter Memory` LUID to a specific
/// `Win32_VideoController` entry isn't reliably possible without vendor-
/// specific APIs, so it's left at 0 (every TUI consumer already guards
/// `vram_total_mb > 0` before dividing).
/// ponytail: single-adapter case (the realistic target: an integrated AMD
/// APU) gets an exact total; multi-GPU hosts degrade to 0 instead of a wrong
/// number. Upgrade path: WMI `MSPower_DeviceEnable`/DXGI LUID correlation if
/// multi-GPU Windows hosts become a real target.
fn parse_metrics(raw: &str) -> Vec<GpuMetrics> {
    let mut util: BTreeMap<&str, f32> = BTreeMap::new();
    let mut vram_mb: BTreeMap<&str, u64> = BTreeMap::new();
    let mut order: Vec<&str> = Vec::new();

    for line in raw.lines() {
        let mut parts = line.splitn(3, '\t');
        let (Some(kind), Some(instance), Some(value)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if kind != "UTIL" && kind != "VRAM" {
            continue; // GPU inventory lines are handled separately by parse_adapters.
        }
        let key = device_key(instance);
        if !util.contains_key(key) && !vram_mb.contains_key(key) {
            order.push(key);
        }
        match kind {
            "UTIL" => {
                let pct: f32 = value.trim().parse().unwrap_or(0.0);
                let entry = util.entry(key).or_insert(0.0);
                if pct > *entry {
                    *entry = pct;
                }
            }
            "VRAM" => {
                let bytes: f64 = value.trim().parse().unwrap_or(0.0);
                *vram_mb.entry(key).or_insert(0) += (bytes / (1024.0 * 1024.0)) as u64;
            }
            _ => unreachable!(),
        }
    }

    let adapters = parse_adapters(raw);
    let single_adapter_ram_mb = match (order.len(), adapters.len()) {
        (1, 1) => adapters[0].ram_bytes / (1024 * 1024),
        _ => 0,
    };

    order
        .into_iter()
        .enumerate()
        .map(|(i, key)| GpuMetrics {
            device_id: format!("win-gpu-{i}"),
            vram_used_mb: vram_mb.get(key).copied().unwrap_or(0),
            vram_total_mb: single_adapter_ram_mb,
            gpu_utilization_pct: util.get(key).copied().unwrap_or(0.0).min(100.0),
            temperature_c: 0.0,
            power_w: 0.0,
            clock_mhz: None,
        })
        .collect()
}

struct Adapter {
    name: String,
    ram_bytes: u64,
    driver_version: String,
}

fn parse_adapters(raw: &str) -> Vec<Adapter> {
    raw.lines()
        .filter_map(|line| {
            let mut parts = line.splitn(4, '\t');
            match (parts.next(), parts.next(), parts.next(), parts.next()) {
                (Some("GPU"), Some(name), Some(ram), Some(driver)) => Some(Adapter {
                    name: name.to_string(),
                    ram_bytes: ram.trim().parse().unwrap_or(0),
                    driver_version: driver.trim().to_string(),
                }),
                _ => None,
            }
        })
        .collect()
}

fn parse_system_info(raw: &str) -> GpuSystemInfo {
    let adapters = parse_adapters(raw);
    GpuSystemInfo {
        gpu_model: adapters.first().map(|a| a.name.clone()).unwrap_or_default(),
        driver_version: adapters.first().map(|a| a.driver_version.clone()),
        physical_gpu_count: adapters.len() as u32,
        logical_gpu_count: adapters.len() as u32,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real `GPU Engine` instances are per-process (`pid_NNNN_luid_...`); mix
    // in a pid prefix here so the tests actually exercise that shape.
    const SAMPLE: &str = "UTIL\tpid_1234_luid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t12.5\nUTIL\tpid_1234_luid_0x00000000_0x0000b3a4_phys_0_eng_1_engtype_video\t4.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t8589934592\nGPU\tAMD Radeon(TM) Graphics\t17179869184\t31.0.21021.1000";

    #[test]
    fn parses_single_adapter_metrics_with_total_ram() {
        let metrics = parse_metrics(SAMPLE);
        assert_eq!(metrics.len(), 1);
        let gpu = &metrics[0];
        // Max across the two engine readings, not the sum.
        assert!((gpu.gpu_utilization_pct - 12.5).abs() < 0.01);
        assert_eq!(gpu.vram_used_mb, 8192);
        assert_eq!(gpu.vram_total_mb, 16384);
    }

    #[test]
    fn parses_system_info_from_adapter_inventory() {
        let info = parse_system_info(SAMPLE);
        assert_eq!(info.gpu_model, "AMD Radeon(TM) Graphics");
        assert_eq!(info.driver_version.as_deref(), Some("31.0.21021.1000"));
        assert_eq!(info.physical_gpu_count, 1);
    }

    #[test]
    fn multiple_adapters_leave_vram_total_zero() {
        let two_adapters = format!(
            "{SAMPLE}\nUTIL\tluid_0x00000000_0x0000cafe_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000cafe_phys_0\t1048576\nGPU\tOther GPU\t4294967296\t1.0"
        );
        let metrics = parse_metrics(&two_adapters);
        assert_eq!(metrics.len(), 2);
        assert!(metrics.iter().all(|g| g.vram_total_mb == 0));
    }

    #[test]
    fn multiple_processes_on_one_adapter_collapse_to_one_device() {
        // Two different pids hitting the same adapter's engines, plus a
        // single per-adapter VRAM sample (as Windows actually reports it).
        let sample = "UTIL\tpid_111_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t0.682\nUTIL\tpid_222_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t0.514\nVRAM\tluid_0x00000000_0x0002ff2a_phys_0\t902787072\nGPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004";
        let metrics = parse_metrics(sample);
        assert_eq!(metrics.len(), 1);
        let gpu = &metrics[0];
        assert!((gpu.gpu_utilization_pct - 0.682).abs() < 0.001);
        assert_eq!(gpu.vram_used_mb, 902_787_072 / (1024 * 1024));
        assert_eq!(gpu.vram_total_mb, 1_073_741_824 / (1024 * 1024));
    }

    #[test]
    fn empty_output_yields_no_metrics() {
        assert!(parse_metrics("").is_empty());
        assert_eq!(parse_system_info("").physical_gpu_count, 0);
    }
}
