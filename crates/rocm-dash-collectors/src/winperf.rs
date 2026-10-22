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
use tokio::task;
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
        // adlx_readings() calls into amdadlx64.dll via FFI, which blocks on
        // driver I/O — running it directly here would stall a tokio worker
        // thread, same hazard run_script() avoids by using an async Command.
        let adlx = task::spawn_blocking(adlx_readings)
            .await
            .unwrap_or_default();
        Ok((parse_metrics(&out, &adlx), parse_system_info(&out)))
    }

    async fn run_script() -> io::Result<String> {
        let out = timeout(
            RUN_TIMEOUT,
            Command::new("powershell.exe")
                .args(["-NoProfile", "-NonInteractive", "-Command", WINPERF_SCRIPT])
                // Dropping the child on timeout must actually kill it, or a
                // hung powershell.exe (and its driver query) leaks forever
                // instead of exiting with the future that gave up on it.
                .kill_on_drop(true)
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
/// number. Upgrade path: DXGI/LUID-based adapter correlation if multi-GPU
/// Windows hosts become a real target.
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
        // ADLX's GPU list isn't otherwise correlated to the perf-counter
        // LUIDs here, so it's only trusted at all when there's exactly one
        // ADLX reading to match this one real adapter against — same
        // single-reading guard `vram_total_mb` below already uses.
        let (temperature_c, power_w, clock_mhz) = match adlx {
            [reading] => apply_adlx(Some(reading)),
            _ => apply_adlx(None),
        };
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

    // More than one real adapter: an ADLX GPU-list index has no reliable
    // correlation to a specific perf-counter LUID (same reason
    // `vram_total_mb` is left at 0 below), so every slot degrades to
    // ADLX-absent defaults rather than risk reporting one GPU's temp/power/
    // clock under a different GPU's device_id.
    let (temperature_c, power_w, clock_mhz) = apply_adlx(None);
    order
        .into_iter()
        .enumerate()
        .map(|(i, key)| GpuMetrics {
            device_id: format!("win-gpu-{i}"),
            vram_used_mb: vram_mb.get(key).copied().unwrap_or(0),
            vram_total_mb: 0,
            gpu_utilization_pct: util.get(key).copied().unwrap_or(0.0).min(100.0),
            temperature_c,
            power_w,
            clock_mhz,
        })
        .collect()
}

/// Per-GPU temperature/power/clock reading from ADLX. `adlx_readings` emits
/// exactly one entry per GPU in ADLX's list, whether or not
/// `current_gpu_metrics` succeeded for it, so a failure degrades the fields
/// to 0.0/None in place rather than shifting every later GPU's reading out
/// of position.
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

/// `None` (no device-identity correlation to trust, or ADLX unavailable)
/// degrades to the same 0.0/None defaults `parse_metrics` used before ADLX
/// existed. Callers only pass `Some` when there's exactly one real adapter
/// and exactly one ADLX reading — the only case this file can match an ADLX
/// reading to a device without index-based guessing.
fn apply_adlx(reading: Option<&AdlxReading>) -> (f32, f32, Option<f32>) {
    (
        reading.map_or(0.0, |r| r.temperature_c),
        reading.map_or(0.0, |r| r.power_w),
        reading.and_then(|r| r.clock_mhz),
    )
}

/// `AdlxHelper::new()` fails cleanly (`Err`, not a panic) when
/// `amdadlx64.dll` isn't found, so a total init failure can just bail to an
/// empty Vec via `.ok()?` — same graceful-degrade shape as the rest of this
/// file. A single GPU's metrics query failing, though, must NOT drop that
/// GPU's slot (a `filter_map` here would shift every later GPU's reading one
/// position to the left, silently misattributing it in `parse_metrics`'s
/// index-based correlation) — it degrades to a zeroed `AdlxReading` instead,
/// keeping one slot per GPU.
#[cfg(windows)]
fn adlx_readings() -> Vec<AdlxReading> {
    // `adlx::helper::AdlxHelper::new()` loads `amdadlx64.dll` by bare name
    // (`libloading::Library::new("amdadlx64.dll")`, no search-order
    // hardening, no way to pass flags into the vendored crate). Restricting
    // the process-wide default DLL search path to System32 closes the
    // planted-DLL-in-PATH/cwd hijack vector for that unflagged load. Must
    // run before the first `AdlxHelper::new()` call; `Once` makes repeated
    // per-tick calls to this function a no-op after the first.
    static HARDEN_DLL_SEARCH_PATH: std::sync::Once = std::sync::Once::new();
    #[allow(unsafe_code)] // Win32 FFI
    HARDEN_DLL_SEARCH_PATH.call_once(|| unsafe {
        windows_sys::Win32::System::LibraryLoader::SetDefaultDllDirectories(
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        );
    });

    (|| -> Option<Vec<AdlxReading>> {
        let helper = adlx::helper::AdlxHelper::new().ok()?;
        let system = helper.system();
        let gpus = system.gpus().ok()?;
        let services = system.performance_monitoring_services().ok()?;
        Some(
            gpus.iter()
                .map(|gpu| match services.current_gpu_metrics(&gpu) {
                    Ok(m) => AdlxReading {
                        temperature_c: m.temperature().unwrap_or(0.0) as f32,
                        power_w: m.power().unwrap_or(0.0) as f32,
                        clock_mhz: m.clock_speed().ok().map(|c| c as f32),
                        total_vram_mb: gpu.total_vram().ok(),
                    },
                    Err(_) => AdlxReading {
                        temperature_c: 0.0,
                        power_w: 0.0,
                        clock_mhz: None,
                        total_vram_mb: None,
                    },
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
    // 3221225472 (3GiB) and 1073741824 (1GiB) are both representable in the
    // real 32-bit `AdapterRAM` WMI field (unlike 16GiB, which would already
    // have overflowed it).
    const SAMPLE: &str = "UTIL\tpid_1234_luid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t12.5\nUTIL\tpid_1234_luid_0x00000000_0x0000b3a4_phys_0_eng_1_engtype_video\t4.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1073741824\nGPU\tAMD Radeon(TM) Graphics\t3221225472\t31.0.21021.1000";

    #[test]
    fn parses_single_adapter_metrics_with_total_ram() {
        let metrics = parse_metrics(SAMPLE, &[]);
        assert_eq!(metrics.len(), 1);
        let gpu = &metrics[0];
        // Max across the two engine readings, not the sum.
        assert!((gpu.gpu_utilization_pct - 12.5).abs() < 0.01);
        assert_eq!(gpu.vram_used_mb, 1024);
        assert_eq!(gpu.vram_total_mb, 3072);
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
        // *second*, unrelated real adapter. With only one real adapter (as
        // the original fixture had), the single-real-adapter collapse rescue
        // below would also force everything to one device, masking whether
        // this is actually testing `device_key`'s pid-prefix stripping; with
        // two real adapters that rescue never fires (its pattern match
        // requires exactly one), isolating the two code paths.
        let sample = "UTIL\tpid_111_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t0.682\nUTIL\tpid_222_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t0.514\nVRAM\tluid_0x00000000_0x0002ff2a_phys_0\t902787072\nUTIL\tluid_0x00000000_0x0000cafe_phys_0_eng_0_engtype_3d\t0.1\nVRAM\tluid_0x00000000_0x0000cafe_phys_0\t1048576\nGPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004\nGPU\tOther GPU\t4294967296\t1.0";
        let metrics = parse_metrics(sample, &[]);
        assert_eq!(metrics.len(), 2);
        let gpu = &metrics[0];
        assert!((gpu.gpu_utilization_pct - 0.682).abs() < 0.001);
        assert_eq!(gpu.vram_used_mb, 902_787_072 / (1024 * 1024));
        // Multi-real-adapter case: vram_total_mb always 0, no LUID correlation.
        assert_eq!(gpu.vram_total_mb, 0);
    }

    #[test]
    fn single_real_adapter_sums_vram_used_across_distinct_luids() {
        // Two *distinct* non-zero per-LUID VRAM readings on the one real
        // adapter — unlike the phantom-LUID fixture below (0 + 0 + real),
        // this is the only fixture where summing vram_used_mb actually
        // differs from taking its max, so a sum->max mutation would be
        // caught here.
        let sample = "UTIL\tluid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0002ff2a_phys_0\t536870912\nVRAM\tluid_0x00000000_0x000325a2_phys_0\t268435456\nGPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004";
        let metrics = parse_metrics(sample, &[]);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].vram_used_mb, 512 + 256);
    }

    #[test]
    fn gpu_inventory_without_counter_samples_yields_no_metrics() {
        // A real adapter shows up in Win32_VideoController but Get-Counter
        // returned nothing for it (e.g. a locale where Get-Counter's
        // -ErrorAction silently swallows a localized-path mismatch) — must
        // not fabricate a zeroed phantom device for it.
        let sample = "GPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004";
        assert!(parse_metrics(sample, &[]).is_empty());
        assert_eq!(parse_system_info(sample).physical_gpu_count, 1);
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
    #[allow(clippy::float_cmp)]
    fn single_adapter_prefers_adlx_vram_over_wmi_truncated_value() {
        // WMI's AdapterRAM here (4294967296 = exactly 4GiB) is the classic
        // 32-bit-field overflow case; ADLX's own query reports the real
        // 8GiB so parse_metrics should prefer it.
        let sample = "UTIL\tluid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1048576\nGPU\tDiscrete GPU\t4294967296\t1.0";
        let adlx = [AdlxReading {
            temperature_c: 50.0,
            power_w: 10.0,
            clock_mhz: Some(1800.0),
            total_vram_mb: Some(8192),
        }];
        let metrics = parse_metrics(sample, &adlx);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].vram_total_mb, 8192);
        assert_eq!(metrics[0].temperature_c, 50.0);
        assert_eq!(metrics[0].power_w, 10.0);
        assert_eq!(metrics[0].clock_mhz, Some(1800.0));
    }

    #[test]
    fn multiple_adapters_ignore_adlx_vram_and_leave_total_zero() {
        let two_adapters = format!(
            "{SAMPLE}\nUTIL\tluid_0x00000000_0x0000cafe_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000cafe_phys_0\t1048576\nGPU\tOther GPU\t4294967296\t1.0"
        );
        let adlx = [AdlxReading {
            temperature_c: 50.0,
            power_w: 10.0,
            clock_mhz: Some(1800.0),
            total_vram_mb: Some(8192),
        }];
        let metrics = parse_metrics(&two_adapters, &adlx);
        assert_eq!(metrics.len(), 2);
        assert!(metrics.iter().all(|g| g.vram_total_mb == 0));
        assert!(
            metrics
                .iter()
                .all(|g| g.temperature_c == 0.0 && g.power_w == 0.0 && g.clock_mhz.is_none())
        );
    }

    #[test]
    fn multiple_adapters_never_trust_adlx_even_when_reading_count_matches_adapter_count() {
        // Two ADLX readings for two real adapters: if `parse_metrics` ever
        // started correlating by list position again, this is exactly the
        // shape that would make it look like it "worked" by coincidence.
        let two_adapters = format!(
            "{SAMPLE}\nUTIL\tluid_0x00000000_0x0000cafe_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000cafe_phys_0\t1048576\nGPU\tOther GPU\t4294967296\t1.0"
        );
        let two_adlx = [
            AdlxReading {
                temperature_c: 50.0,
                power_w: 10.0,
                clock_mhz: Some(1800.0),
                total_vram_mb: Some(8192),
            },
            AdlxReading {
                temperature_c: 40.0,
                power_w: 8.0,
                clock_mhz: Some(1200.0),
                total_vram_mb: Some(4096),
            },
        ];
        let metrics = parse_metrics(&two_adapters, &two_adlx);
        assert_eq!(metrics.len(), 2);
        assert!(metrics.iter().all(|g| g.vram_total_mb == 0));
        assert!(
            metrics
                .iter()
                .all(|g| g.temperature_c == 0.0 && g.power_w == 0.0 && g.clock_mhz.is_none())
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn single_adapter_falls_back_to_wmi_vram_when_adlx_reading_count_mismatches() {
        let sample = "UTIL\tluid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1048576\nGPU\tDiscrete GPU\t4294967296\t1.0";
        let two_adlx = [
            AdlxReading {
                temperature_c: 50.0,
                power_w: 10.0,
                clock_mhz: Some(1800.0),
                total_vram_mb: Some(8192),
            },
            AdlxReading {
                temperature_c: 40.0,
                power_w: 8.0,
                clock_mhz: Some(1200.0),
                total_vram_mb: Some(4096),
            },
        ];
        let metrics = parse_metrics(sample, &two_adlx);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].vram_total_mb, 4_294_967_296 / (1024 * 1024));
        assert_eq!(metrics[0].temperature_c, 0.0);
        assert_eq!(metrics[0].power_w, 0.0);
        assert_eq!(metrics[0].clock_mhz, None);
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
