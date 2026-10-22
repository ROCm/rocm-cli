// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Live Windows GPU metrics via OS performance counters, no AMD tool needed
//! for utilization/VRAM.
//!
//! Temperature, power, and clock speed additionally use ADLX (AMD Device
//! Library eXtra, via the `adlx` crate) when the AMD driver's bundled
//! `amdadlx64.dll` is present; absence (no AMD driver, non-AMD GPU) degrades
//! those three fields to 0.0/None rather than failing the whole snapshot.
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
        Ok((
            parse_metrics(&out, &adlx_readings()),
            parse_system_info(&out),
        ))
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
/// Windows can report more than one `GPU Adapter Memory` LUID for a single
/// physical GPU (stale instances left behind by a display-mode switch, or
/// separate WDDM engine nodes) — on an integrated APU this shows up as a
/// "3 GPUs" readout with two phantom zero-VRAM devices. When
/// `Win32_VideoController` says there's exactly one real adapter, collapse
/// every LUID into that one device instead of reporting one per LUID: sum
/// VRAM used (a phantom LUID contributes 0), take the max utilization.
///
/// With more than one real adapter, correlating a `GPU Adapter Memory` LUID
/// to a specific `Win32_VideoController` entry isn't reliably possible
/// without vendor-specific APIs, so `vram_total_mb` is left at 0 (every TUI
/// consumer already guards `vram_total_mb > 0` before dividing).
/// ponytail: single-adapter case (the realistic target: an integrated AMD
/// APU) gets an exact total; multi-GPU hosts degrade to 0 instead of a wrong
/// number. Upgrade path: WMI `MSPower_DeviceEnable`/DXGI LUID correlation if
/// multi-GPU Windows hosts become a real target.
///
/// `adlx` is a parameter (rather than this function calling `adlx_readings()`
/// itself) so tests can exercise the ADLX-matching/VRAM-fallback logic below
/// with fixed readings — `adlx_readings()`'s real FFI chain is Windows+driver
/// only and not unit-testable.
fn parse_metrics(raw: &str, adlx: &[AdlxReading]) -> Vec<GpuMetrics> {
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
    if let [adapter] = adapters.as_slice() {
        if order.is_empty() {
            return Vec::new();
        }
        let vram_used_mb: u64 = vram_mb.values().sum();
        let gpu_utilization_pct = util.values().copied().fold(0.0f32, f32::max).min(100.0);
        let (temperature_c, power_w, clock_mhz) = apply_adlx(adlx.first());
        // `Win32_VideoController.AdapterRAM` is a 32-bit WMI field: any card
        // with >=4GiB VRAM truncates/overflows there. ADLX's own VRAM query
        // isn't subject to that, so prefer it when exactly one ADLX GPU
        // reading exists (matching this single-real-adapter branch) and
        // fall back to the WMI value otherwise (no AMD driver, non-AMD GPU).
        let vram_total_mb = match adlx {
            [reading] => reading
                .total_vram_mb
                .map_or(adapter.ram_bytes / (1024 * 1024), u64::from),
            _ => adapter.ram_bytes / (1024 * 1024),
        };
        return vec![GpuMetrics {
            device_id: "win-gpu-0".to_string(),
            vram_used_mb,
            vram_total_mb,
            gpu_utilization_pct,
            temperature_c,
            power_w,
            clock_mhz,
        }];
    }

    order
        .into_iter()
        .enumerate()
        .map(|(i, key)| {
            let (temperature_c, power_w, clock_mhz) = apply_adlx(adlx.get(i));
            GpuMetrics {
                device_id: format!("win-gpu-{i}"),
                vram_used_mb: vram_mb.get(key).copied().unwrap_or(0),
                vram_total_mb: 0,
                gpu_utilization_pct: util.get(key).copied().unwrap_or(0.0).min(100.0),
                temperature_c,
                power_w,
                clock_mhz,
            }
        })
        .collect()
}

/// Per-GPU temperature/power/clock reading from ADLX. Each field is a plain
/// value (not `Option`): `adlx_readings` only emits an entry for a GPU once
/// `current_gpu_metrics` has already succeeded for it, so unsupported
/// individual metrics on that entry fall back to 0.0/None right there.
struct AdlxReading {
    temperature_c: f32,
    power_w: f32,
    clock_mhz: Option<f32>,
    /// ADLX's own VRAM-size query (`IADLXGPU::TotalVRAM`, already in MB).
    /// `Win32_VideoController.AdapterRAM` is a 32-bit WMI field that
    /// silently truncates/overflows above 4GiB — common on any discrete
    /// card, just not on the integrated APU this was tested against — so
    /// `parse_metrics` prefers this reading over WMI's when available.
    total_vram_mb: Option<u32>,
}

/// Matches an ADLX reading to a `parse_metrics` device slot by index (ADLX's
/// GPU list order isn't otherwise correlated to the perf-counter LUIDs here).
/// No reading for that slot, or a 0.0/None `AdlxReading`, degrades to the
/// same defaults `parse_metrics` used before ADLX existed.
fn apply_adlx(reading: Option<&AdlxReading>) -> (f32, f32, Option<f32>) {
    (
        reading.map_or(0.0, |r| r.temperature_c),
        reading.map_or(0.0, |r| r.power_w),
        reading.and_then(|r| r.clock_mhz),
    )
}

/// `AdlxHelper::new()` fails cleanly (`Err`, not a panic) when
/// `amdadlx64.dll` isn't found, so every step here can just bail to an empty
/// Vec via `.ok()?` — same graceful-degrade shape as the rest of this file.
#[cfg(windows)]
fn adlx_readings() -> Vec<AdlxReading> {
    (|| -> Option<Vec<AdlxReading>> {
        let helper = adlx::helper::AdlxHelper::new().ok()?;
        let system = helper.system();
        let gpus = system.gpus().ok()?;
        let services = system.performance_monitoring_services().ok()?;
        Some(
            gpus.iter()
                .filter_map(|gpu| {
                    let m = services.current_gpu_metrics(&gpu).ok()?;
                    Some((gpu, m))
                })
                .map(|(gpu, m)| AdlxReading {
                    temperature_c: m.temperature().unwrap_or(0.0) as f32,
                    power_w: m.power().unwrap_or(0.0) as f32,
                    clock_mhz: m.clock_speed().ok().map(|c| c as f32),
                    total_vram_mb: gpu.total_vram().ok(),
                })
                .collect(),
        )
    })()
    .unwrap_or_default()
}

#[cfg(not(windows))]
const fn adlx_readings() -> Vec<AdlxReading> {
    Vec::new()
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
        let metrics = parse_metrics(SAMPLE, &[]);
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
        let metrics = parse_metrics(&two_adapters, &[]);
        assert_eq!(metrics.len(), 2);
        assert!(metrics.iter().all(|g| g.vram_total_mb == 0));
    }

    #[test]
    fn multiple_processes_on_one_adapter_collapse_to_one_device() {
        // Two different pids hitting the same adapter's engines, plus a
        // single per-adapter VRAM sample (as Windows actually reports it).
        let sample = "UTIL\tpid_111_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t0.682\nUTIL\tpid_222_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t0.514\nVRAM\tluid_0x00000000_0x0002ff2a_phys_0\t902787072\nGPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004";
        let metrics = parse_metrics(sample, &[]);
        assert_eq!(metrics.len(), 1);
        let gpu = &metrics[0];
        assert!((gpu.gpu_utilization_pct - 0.682).abs() < 0.001);
        assert_eq!(gpu.vram_used_mb, 902_787_072 / (1024 * 1024));
        assert_eq!(gpu.vram_total_mb, 1_073_741_824 / (1024 * 1024));
    }

    #[test]
    fn phantom_adapter_memory_luids_collapse_to_one_device_when_only_one_real_adapter() {
        // Real-world capture from an AMD Radeon 780M laptop: Windows reports
        // 3 distinct `GPU Adapter Memory` LUIDs (2 of them permanently
        // zero-VRAM phantoms) for the single physical adapter
        // `Win32_VideoController` actually lists.
        let sample = "UTIL\tpid_1234_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t0.5\nVRAM\tluid_0x00000000_0x0002ff2a_phys_0\t897261568\nVRAM\tluid_0x00000000_0x000325a2_phys_0\t0\nVRAM\tluid_0x00000000_0x000325f9_phys_0\t0\nGPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004";
        let metrics = parse_metrics(sample, &[]);
        assert_eq!(metrics.len(), 1);
        let gpu = &metrics[0];
        assert!((gpu.gpu_utilization_pct - 0.5).abs() < 0.001);
        assert_eq!(gpu.vram_used_mb, 897_261_568 / (1024 * 1024));
        assert_eq!(gpu.vram_total_mb, 1_073_741_824 / (1024 * 1024));
    }

    #[test]
    fn single_adapter_prefers_adlx_vram_over_wmi_truncated_value() {
        // WMI's AdapterRAM here (4294967296 = exactly 4GiB) is the classic
        // 32-bit-field overflow case; ADLX's own query reports the real
        // 8GiB so parse_metrics should prefer it.
        let sample = "UTIL\tluid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1048576\nGPU\tDiscrete GPU\t4294967296\t1.0";
        let adlx = [AdlxReading {
            temperature_c: 50.0,
            power_w: 10.0,
            clock_mhz: None,
            total_vram_mb: Some(8192),
        }];
        let metrics = parse_metrics(sample, &adlx);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].vram_total_mb, 8192);
    }

    #[test]
    fn multiple_adapters_ignore_adlx_vram_and_leave_total_zero() {
        let two_adapters = format!(
            "{SAMPLE}\nUTIL\tluid_0x00000000_0x0000cafe_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000cafe_phys_0\t1048576\nGPU\tOther GPU\t4294967296\t1.0"
        );
        let adlx = [AdlxReading {
            temperature_c: 50.0,
            power_w: 10.0,
            clock_mhz: None,
            total_vram_mb: Some(8192),
        }];
        let metrics = parse_metrics(&two_adapters, &adlx);
        assert_eq!(metrics.len(), 2);
        assert!(metrics.iter().all(|g| g.vram_total_mb == 0));
    }

    #[test]
    fn single_adapter_falls_back_to_wmi_vram_when_adlx_reading_count_mismatches() {
        let sample = "UTIL\tluid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1048576\nGPU\tDiscrete GPU\t4294967296\t1.0";
        let two_adlx = [
            AdlxReading {
                temperature_c: 50.0,
                power_w: 10.0,
                clock_mhz: None,
                total_vram_mb: Some(8192),
            },
            AdlxReading {
                temperature_c: 40.0,
                power_w: 8.0,
                clock_mhz: None,
                total_vram_mb: Some(4096),
            },
        ];
        let metrics = parse_metrics(sample, &two_adlx);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].vram_total_mb, 4_294_967_296 / (1024 * 1024));
    }

    #[test]
    fn empty_output_yields_no_metrics() {
        assert!(parse_metrics("", &[]).is_empty());
        assert_eq!(parse_system_info("").physical_gpu_count, 0);
    }

    #[test]
    fn apply_adlx_falls_back_to_defaults_when_missing() {
        assert_eq!(apply_adlx(None), (0.0, 0.0, None));
    }

    #[test]
    fn apply_adlx_uses_reading_when_present() {
        let reading = AdlxReading {
            temperature_c: 65.0,
            power_w: 12.5,
            clock_mhz: Some(1900.0),
            total_vram_mb: Some(16384),
        };
        assert_eq!(apply_adlx(Some(&reading)), (65.0, 12.5, Some(1900.0)));
    }

    // adlx_readings() itself (the real ADLX FFI chain) is Windows-only and
    // hardware/driver-dependent — not unit-testable here; CI's
    // windows-build-and-test job covers compilation of the #[cfg(windows)]
    // path, and apply_adlx above covers the pure mapping logic.
}
