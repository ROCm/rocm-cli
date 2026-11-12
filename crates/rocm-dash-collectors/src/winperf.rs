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
//! `powershell.exe` instead of `amd-smi`. Not `#[cfg(windows)]`-gated at the
//! module level, but `snapshot()`'s only caller today is itself
//! `#[cfg(windows)]` (see `client.rs`'s `connect_and_run`). On a plain Linux
//! host `powershell.exe` isn't on `PATH` and this fails cleanly like
//! `amd_smi.rs` does when `amd-smi` is missing — but under WSL interop,
//! `powershell.exe` *is* on `PATH` and this would run against the Windows
//! host's GPUs, so don't add a Linux-reachable caller without reconsidering
//! that.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Bounds the whole request to the resident reader — handing the request
/// over, the driver read on that thread, and the answer coming back. None of
/// it should take anywhere near this, but a wedged driver call must not
/// freeze CPU/memory updates forever alongside it — same reasoning as
/// `RUN_TIMEOUT`, just a tighter bound since there's no process spawn latency
/// to absorb here.
const ADLX_TIMEOUT: Duration = Duration::from_secs(5);

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
    ///
    /// The returned `Vec<String>` surfaces non-fatal problems the caller
    /// should show the user (e.g. a locale mismatch, see
    /// `counter_samples_missing_warning`) without failing the whole tick.
    pub async fn snapshot(&self) -> io::Result<(Vec<GpuMetrics>, GpuSystemInfo, Vec<String>)> {
        let out = Self::run_script().await?;
        // adlx_readings() hands its request to the resident ADLX reader
        // thread and blocks until that thread answers — the FFI into
        // amdadlx64.dll runs there, not here. Blocking at all would stall a
        // tokio worker, so it still runs under spawn_blocking. A native call
        // cannot be cancelled, so the timeout below only stops *waiting*: a
        // wedged driver keeps both this blocking-pool thread and the one
        // reader. `run_single_flight` therefore refuses to start another call
        // while one is still running, which bounds that to a single
        // outstanding request instead of one per tick.
        //
        // A panic, timeout or still-wedged call surfaces as a warning, as
        // does the reader failing or dying (ReaderGone). A driver that is
        // absent or fails to initialise answers Ok with no readings and no
        // warning, since "no AMD driver" is an expected state.
        let (adlx, adlx_warning) =
            adlx_outcome(run_single_flight(&ADLX_IN_FLIGHT, ADLX_TIMEOUT, adlx_readings).await);
        let metrics = parse_metrics(&out, &adlx);
        let warnings = counter_samples_missing_warning(&out, &metrics)
            .into_iter()
            .chain(adlx_warning)
            .collect();
        Ok((metrics, parse_system_info(&out), warnings))
    }

    async fn run_script() -> io::Result<String> {
        let out = timeout(
            RUN_TIMEOUT,
            Command::new(powershell_path(
                std::env::var_os("SystemRoot").as_deref(),
                Path::exists,
            ))
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

/// Set while an `adlx_readings` call is running on a blocking-pool thread.
static ADLX_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

#[derive(Debug, PartialEq, Eq)]
enum SingleFlight<T> {
    Done(T),
    Panicked,
    /// The call did not finish within the deadline; it is still running and
    /// `in_flight` stays set until it does.
    TimedOut,
    /// An earlier call is still running, so none was started.
    StillRunning,
}

/// Clears the in-flight flag when the blocking closure ends, panic included.
struct ClearOnDrop(&'static AtomicBool);

impl Drop for ClearOnDrop {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Runs `f` on the blocking pool, waiting at most `deadline`, and never
/// starts a second `f` while a previous one (timed out, but uncancellable) is
/// still running on `in_flight`.
async fn run_single_flight<T, F>(
    in_flight: &'static AtomicBool,
    deadline: Duration,
    f: F,
) -> SingleFlight<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    if in_flight.swap(true, Ordering::AcqRel) {
        return SingleFlight::StillRunning;
    }
    let handle = task::spawn_blocking(move || {
        let _clear = ClearOnDrop(in_flight);
        f()
    });
    match timeout(deadline, handle).await {
        Ok(Ok(v)) => SingleFlight::Done(v),
        Ok(Err(_)) => SingleFlight::Panicked,
        Err(_) => SingleFlight::TimedOut,
    }
}

/// Absolute `System32` PowerShell under `%SystemRoot%` when it exists, else
/// the bare name. A bare name lets a `PATH`-order entry shadow the system
/// binary; `crates/rocm-core/src/host_gpu.rs` resolves the same way, but this
/// crate does not depend on `rocm-core`.
fn powershell_path(system_root: Option<&OsStr>, exists: impl Fn(&Path) -> bool) -> PathBuf {
    system_root
        .map(|root| {
            Path::new(root)
                .join("System32")
                .join("WindowsPowerShell")
                .join("v1.0")
                .join("powershell.exe")
        })
        .filter(|p| exists(p))
        .unwrap_or_else(|| PathBuf::from("powershell.exe"))
}

/// `Win32_VideoController.AdapterRAM` is a 32-bit WMI field: a card with real
/// VRAM at or beyond its ~4GiB ceiling gets silently truncated/capped rather
/// than reporting its true size. A value this close to that ceiling is far
/// more likely a truncated sentinel than an improbably-exact ~4GiB card, so
/// treat it as unknown (0) rather than report a number that's probably wrong.
/// Set ~295MB below the true 2^32 boundary so near-cap values are caught too.
const WMI_ADAPTER_RAM_CAP_THRESHOLD: u64 = 4_000_000_000;

const fn wmi_vram_total_mb(ram_bytes: u64) -> u64 {
    if ram_bytes >= WMI_ADAPTER_RAM_CAP_THRESHOLD {
        0
    } else {
        ram_bytes / (1024 * 1024)
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
    instance
        .split_once("_phys_")
        .map_or(instance, |(head, _)| head)
}

/// `GPU Engine` instance names end in `..._phys_P_eng_N_engtype_TYPE`; the
/// part after the device key (`phys_P_eng_N_engtype_TYPE`) identifies one
/// physical engine, shared across every process using it. Pairs with
/// `device_key` in `parse_metrics` to key utilization by (device, engine
/// instance). Two engines of the same type (e.g. two copy engines) are
/// distinct and must not be summed.
fn engine_id(instance: &str) -> &str {
    instance.split_once("_phys_").map_or(instance, |(_, e)| e)
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
/// consumer already guards `vram_total_mb > 0` before dividing). The
/// single-adapter case — the realistic target, an integrated AMD APU — gets
/// an exact total; multi-GPU hosts degrade to 0 instead of a wrong number.
/// Upgrade path if multi-GPU Windows hosts become a real target: DXGI/LUID-
/// based adapter correlation.
///
/// `adlx` is a parameter (rather than this function calling `adlx_readings()`
/// itself) so tests can exercise the ADLX-matching/VRAM-fallback logic below
/// with fixed readings — `adlx_readings()`'s real FFI chain is Windows+driver
/// only and not unit-testable.
fn parse_metrics(raw: &str, adlx: &[AdlxReading]) -> Vec<GpuMetrics> {
    // Keyed by (device, engine instance): `GPU Engine` instances are
    // per-process-per-engine, so two processes both drawing on the same
    // engine each contribute their own share of that engine's capacity and
    // must be summed, matching Task Manager. Distinct engines (3D vs. copy,
    // or two copy engines) run concurrently on separate hardware, so those
    // are maxed together below instead of summed.
    let mut engine_util: BTreeMap<(&str, &str), f32> = BTreeMap::new();
    let mut vram_mb: BTreeMap<&str, u64> = BTreeMap::new();
    // A `BTreeSet` rather than first-seen order: `powershell.exe`'s adapter
    // enumeration order isn't guaranteed stable between ticks, which would
    // otherwise make `win-gpu-{i}` swap which physical device it names from
    // one snapshot to the next (mixing per-device history in the TUI).
    // Sorting by the LUID-derived key instead keeps each device's index
    // stable for as long as its LUID is (i.e. until reboot/driver reload).
    let mut seen: BTreeSet<&str> = BTreeSet::new();

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
        seen.insert(key);
        match kind {
            "UTIL" => {
                let pct: f32 = value.trim().parse().unwrap_or(0.0);
                *engine_util.entry((key, engine_id(instance))).or_insert(0.0) += pct;
            }
            "VRAM" => {
                let bytes: f64 = value.trim().parse().unwrap_or(0.0);
                *vram_mb.entry(key).or_insert(0) += (bytes / (1024.0 * 1024.0)) as u64;
            }
            _ => unreachable!(),
        }
    }

    let mut util: BTreeMap<&str, f32> = BTreeMap::new();
    for (&(device, _engine), &pct) in &engine_util {
        let entry = util.entry(device).or_insert(0.0);
        if pct > *entry {
            *entry = pct;
        }
    }

    let adapters = parse_adapters(raw);
    if let [adapter] = adapters.as_slice() {
        if seen.is_empty() {
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
                .map_or_else(|| wmi_vram_total_mb(adapter.ram_bytes), u64::from),
            _ => wmi_vram_total_mb(adapter.ram_bytes),
        };
        return vec![GpuMetrics {
            device_id: "win-gpu-0".to_string(),
            vram_used_mb,
            vram_total_mb,
            gpu_utilization_pct,
            temperature_c,
            power_w,
            power_limit_w: None,
            clock_mhz,
        }];
    }

    // More than one real adapter: an ADLX GPU-list index has no reliable
    // correlation to a specific perf-counter LUID (same reason
    // `vram_total_mb` is left at 0 below), so every slot degrades to
    // ADLX-absent defaults rather than risk reporting one GPU's temp/power/
    // clock under a different GPU's device_id.
    let (temperature_c, power_w, clock_mhz) = apply_adlx(None);
    seen.into_iter()
        .enumerate()
        .map(|(i, key)| GpuMetrics {
            device_id: format!("win-gpu-{i}"),
            vram_used_mb: vram_mb.get(key).copied().unwrap_or(0),
            vram_total_mb: 0,
            gpu_utilization_pct: util.get(key).copied().unwrap_or(0.0).min(100.0),
            temperature_c,
            power_w,
            // No limit source on Windows: unknown, so power is not colored.
            power_limit_w: None,
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

/// Owns a value that is built once, on its own thread, and serves every read
/// from that thread for as long as the process lives.
///
/// ADLX is meant to be initialised once per process. The `adlx` crate's
/// `AdlxHelper` loads `amdadlx64.dll` and calls `ADLXInitialize` in `new()`,
/// and `ADLXTerminate` plus the DLL unload in `Drop`. Doing that every tick,
/// each time on a different blocking-pool thread, loads and unloads
/// `amdadlx64.dll` (and the `d3d9`/`OPENGL32`/`dbghelp` it pulls in) over and
/// over while the driver's `atiadlxx.dll` stays resident. After a while
/// `atiadlxx.dll` fail-fasts (`0xC0000409`, `FAST_FAIL_FATAL_APP_EXIT`)
/// and takes `rocm dash` down with no panic message or stack trace.
///
/// `init` and `read` are parameters so the "built once, same thread for every
/// read" contract is unit-testable without a driver (see the tests).
#[cfg(any(windows, test))]
struct ResidentReader {
    requests: std::sync::mpsc::Sender<std::sync::mpsc::Sender<ReaderAnswer>>,
}

/// Why the reader could not answer, as distinct from answering with no
/// readings. The difference matters: no readings is what an absent driver
/// looks like and is expected, while these mean the reader itself is in
/// trouble and the user is told. Since the reader is built once per process
/// and never replaced, a failure that reported nothing would blank
/// temperature, power and clock for the rest of the session and look exactly
/// like a machine with no AMD driver.
///
/// Not gated: `snapshot` matches on it and the non-Windows `adlx_readings`
/// stub returns it, on every platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReaderGone {
    /// The read step panicked. Caught, so the reader survives for the next
    /// tick; reported so a driver that panics every read is not silent.
    #[cfg_attr(not(any(windows, test)), allow(dead_code))]
    ReadPanicked,
    /// The init step panicked. Caught for the same reason as a read panic, and
    /// kept apart from it because they are not the same fault: a read panic is
    /// per-tick and the next read may succeed, while this one is terminal —
    /// the state was never built and this design never rebuilds it.
    ///
    /// Before it was caught, an init panic unwound the reader thread outright,
    /// so every later tick met a closed channel and reported
    /// [`Self::ThreadGone`] — indistinguishable from the thread never having
    /// started. Catching it keeps the thread alive to answer and names the real
    /// cause. It does **not** restore the pre-resident-reader behaviour of
    /// retrying init on the next tick; that is the deliberate trade this
    /// design makes, and the reason the distinct variant is worth having.
    #[cfg_attr(not(any(windows, test)), allow(dead_code))]
    InitPanicked,
    /// The thread never started, or exited before answering.
    #[cfg_attr(not(any(windows, test)), allow(dead_code))]
    ThreadGone,
}

type ReaderAnswer = Result<Vec<AdlxReading>, ReaderGone>;

/// What one ADLX attempt contributes to a snapshot: the readings to use, and
/// the warning to show the user, if any.
///
/// Split out of [`WinPerfCollector::snapshot`] purely so it can be tested.
/// `snapshot` itself calls `Self::run_script()` unconditionally, which shells
/// out to PowerShell, so no test can drive it — the same impracticality
/// `adlx_readings`'s own FFI chain is documented with. That left the mapping
/// below, including the `Done(Err(_))` arm this change introduced, as the one
/// piece of new wiring nothing exercised. It is pure and total, so testing it
/// here costs nothing and covers every arm.
///
/// The asymmetry worth keeping: a driver that is absent or failed to
/// initialise answers `Ok` with no readings and **no** warning, because "no AMD
/// driver" is an expected state rather than a fault. Everything else is a
/// fault, and says so.
fn adlx_outcome(outcome: SingleFlight<ReaderAnswer>) -> (Vec<AdlxReading>, Option<String>) {
    match outcome {
        SingleFlight::Done(Ok(readings)) => (readings, None),
        SingleFlight::Done(Err(gone)) => (Vec::new(), Some(gone.warning())),
        SingleFlight::Panicked => (Vec::new(), Some("ADLX reading task panicked".to_string())),
        SingleFlight::TimedOut => (
            Vec::new(),
            Some(format!("ADLX query timed out after {ADLX_TIMEOUT:?}")),
        ),
        SingleFlight::StillRunning => (
            Vec::new(),
            Some("ADLX query skipped: a previous call is still running".to_string()),
        ),
    }
}

impl ReaderGone {
    fn warning(self) -> String {
        match self {
            Self::ReadPanicked => "ADLX read panicked".to_string(),
            Self::InitPanicked => "ADLX initialisation panicked".to_string(),
            Self::ThreadGone => "the ADLX reader thread is gone".to_string(),
        }
    }
}

#[cfg(any(windows, test))]
impl ResidentReader {
    fn spawn<S>(
        init: impl FnOnce() -> Option<S> + Send + 'static,
        read: impl Fn(&S) -> Vec<AdlxReading> + Send + 'static,
    ) -> Self {
        let (requests, inbox) = std::sync::mpsc::channel::<std::sync::mpsc::Sender<ReaderAnswer>>();
        let spawned = std::thread::Builder::new()
            .name("adlx-reader".into())
            .spawn(move || {
                // `state` is deliberately held until the request channel closes
                // (never, for the process-wide reader) so it is not torn down
                // and re-initialised between ticks.
                //
                // Init is caught for the same reason the read below is, and the
                // asymmetry mattered more here: a panic escaping this line
                // unwound the thread before the request loop started, so
                // `inbox` dropped and every later tick reported `ThreadGone` —
                // the one state that cannot be told apart from "the thread
                // never started". Catching keeps the reader answering, with a
                // variant that names what actually happened.
                let state = std::panic::catch_unwind(std::panic::AssertUnwindSafe(init))
                    .map_err(|_| ReaderGone::InitPanicked);
                for reply in inbox {
                    // A panicking read would otherwise unwind this thread and
                    // leave every later tick answering from a closed channel,
                    // with no way to tell that from an absent driver. Catching
                    // it keeps the one reader alive and names the failure.
                    let answer = match state.as_ref() {
                        Ok(Some(state)) => {
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read(state)))
                                .map_err(|_| ReaderGone::ReadPanicked)
                        }
                        Ok(None) => Ok(Vec::new()),
                        Err(gone) => Err(*gone),
                    };
                    let _ = reply.send(answer);
                }
            });
        if let Err(e) = spawned {
            // Without the thread `inbox` is dropped, so every `read` below
            // sees a closed channel and degrades to no readings.
            tracing::warn!("failed to start the ADLX reader thread: {e}");
        }
        Self { requests }
    }

    /// One reading per GPU, or an empty `Ok` when ADLX is unavailable or
    /// failed to initialise — an absent driver is an expected state, not a
    /// fault. [`ReaderGone`] is reserved for the reader itself failing.
    /// Blocks until the reader answers, so callers run it under
    /// `spawn_blocking`.
    fn read(&self) -> ReaderAnswer {
        let (reply, answer) = std::sync::mpsc::channel();
        if self.requests.send(reply).is_err() {
            return Err(ReaderGone::ThreadGone);
        }
        answer.recv().unwrap_or(Err(ReaderGone::ThreadGone))
    }
}

/// `AdlxHelper::new()` fails cleanly (`Err`, not a panic) when
/// `amdadlx64.dll` isn't found, so a total init failure leaves the reader
/// answering with an empty Vec — same graceful-degrade shape as the rest of
/// this file. A single GPU's metrics query failing, though, must NOT drop that
/// GPU's slot (a `filter_map` here would silently change the reading count,
/// which `parse_metrics`'s single-reading guard relies on to decide whether
/// an ADLX reading can be trusted at all) — it degrades to a zeroed
/// `AdlxReading` instead, keeping one slot per GPU.
///
/// Initialisation is attempted once per process. A missing driver is a static
/// condition, and retrying a failed `ADLXInitialize` would repeat the DLL
/// load/unload cycle that [`ResidentReader`] exists to avoid.
#[cfg(windows)]
fn adlx_readings() -> ReaderAnswer {
    static READER: std::sync::OnceLock<ResidentReader> = std::sync::OnceLock::new();
    READER
        .get_or_init(|| ResidentReader::spawn(init_adlx, read_adlx))
        .read()
}

#[cfg(windows)]
fn init_adlx() -> Option<adlx::helper::AdlxHelper> {
    // `adlx::helper::AdlxHelper::new()` loads `amdadlx64.dll` by bare name
    // (`libloading::Library::new("amdadlx64.dll")`, no search-order
    // hardening, no way to pass flags into the `adlx` crate). Restricting
    // the process-wide default DLL search path to System32 — this call
    // affects every subsequent bare-name `LoadLibrary` in the whole process,
    // not just `adlx`'s — closes the planted-DLL-in-PATH/cwd hijack vector
    // for that unflagged load. Must run before the `AdlxHelper::new()` call,
    // which this function makes exactly once per process.
    #[allow(unsafe_code)] // Win32 FFI
    let hardened = unsafe {
        windows_sys::Win32::System::LibraryLoader::SetDefaultDllDirectories(
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    };
    // Failure here (`FALSE`) means the process keeps its default search
    // order (which can include the cwd) rather than System32-only; still
    // attempt the `AdlxHelper` load afterwards; a stricter search path is
    // defense in depth, not a precondition for ADLX to work.
    if hardened == 0 {
        tracing::warn!(
            "SetDefaultDllDirectories failed; amdadlx64.dll load falls back to the default search order"
        );
    }
    adlx::helper::AdlxHelper::new().ok()
}

#[cfg(windows)]
fn read_adlx(helper: &adlx::helper::AdlxHelper) -> Vec<AdlxReading> {
    (|| -> Option<Vec<AdlxReading>> {
        let system = helper.system();
        let gpus = system.gpus().ok()?;
        let services = system.performance_monitoring_services().ok()?;
        Some(
            gpus.iter()
                .map(|gpu| {
                    // Independent of current_gpu_metrics: a transient
                    // metrics-query failure shouldn't also discard an
                    // otherwise-good VRAM-size reading.
                    let total_vram_mb = gpu.total_vram().ok();
                    match services.current_gpu_metrics(&gpu) {
                        Ok(m) => AdlxReading {
                            temperature_c: m.temperature().unwrap_or(0.0) as f32,
                            power_w: m.power().unwrap_or(0.0) as f32,
                            clock_mhz: m.clock_speed().ok().map(|c| c as f32),
                            total_vram_mb,
                        },
                        Err(_) => AdlxReading {
                            temperature_c: 0.0,
                            power_w: 0.0,
                            clock_mhz: None,
                            total_vram_mb,
                        },
                    }
                })
                .collect(),
        )
    })()
    .unwrap_or_default()
}

#[cfg(not(windows))]
const fn adlx_readings() -> ReaderAnswer {
    Ok(Vec::new())
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

/// `Win32_VideoController` (WMI) and `Get-Counter` (the perf-counter
/// subsystem) are independent queries in the same script: WMI finding an
/// adapter but every counter query coming back empty isn't "no GPU", it's
/// the counters failing silently. The likeliest cause is `Get-Counter`'s
/// English-only counter paths (`\GPU Engine(*)\...`) not matching on a
/// localized Windows install — the script's `-ErrorAction
/// SilentlyContinue` swallows that mismatch rather than erroring, so this
/// is the only signal left to surface it to the user instead of silently
/// showing an empty GPU panel.
fn counter_samples_missing_warning(raw: &str, metrics: &[GpuMetrics]) -> Option<String> {
    if metrics.is_empty() && !parse_adapters(raw).is_empty() {
        Some(
            "Windows GPU utilization/VRAM counters returned no samples (possible non-English \
             Windows locale - Get-Counter counter paths are localized)"
                .to_string(),
        )
    } else {
        None
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
    fn multiple_processes_on_one_adapter_collapse_to_one_device() {
        // Two different pids hitting the same adapter's *same* engine, plus a
        // *second*, unrelated real adapter. With only one real adapter (as
        // the original fixture had), the single-real-adapter collapse rescue
        // below would also force everything to one device, masking whether
        // this is actually testing `device_key`'s pid-prefix stripping; with
        // two real adapters that rescue never fires (its pattern match
        // requires exactly one), isolating the two code paths. Both pids
        // share the `3d` engine, so their utilization sums (Task Manager
        // semantics) rather than taking the max of the two.
        let sample = "UTIL\tpid_111_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t0.682\nUTIL\tpid_222_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t0.514\nVRAM\tluid_0x00000000_0x0002ff2a_phys_0\t902787072\nUTIL\tluid_0x00000000_0x0000cafe_phys_0_eng_0_engtype_3d\t0.1\nVRAM\tluid_0x00000000_0x0000cafe_phys_0\t1048576\nGPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004\nGPU\tOther GPU\t4293918720\t1.0";
        let metrics = parse_metrics(sample, &[]);
        assert_eq!(metrics.len(), 2);
        // Devices are keyed by sorted LUID string, not first-seen order (see
        // `parse_metrics`'s `BTreeSet` comment), so find the device under
        // test by its VRAM reading rather than assuming an index.
        let gpu = metrics
            .iter()
            .find(|g| g.vram_used_mb == 902_787_072 / (1024 * 1024))
            .expect("the pid-prefixed adapter's metrics");
        assert!((gpu.gpu_utilization_pct - 1.196).abs() < 0.001);
        // Multi-real-adapter case: vram_total_mb always 0, no LUID correlation.
        assert_eq!(gpu.vram_total_mb, 0);
    }

    #[test]
    fn utilization_across_different_engine_types_takes_max_not_sum() {
        // Same pid, same device, two *different* engine types running
        // concurrently on separate hardware — these must be maxed, not
        // summed, unlike same-engine readings from different processes.
        let sample = "UTIL\tpid_1_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t90.0\nUTIL\tpid_1_luid_0x00000000_0x0002ff2a_phys_0_eng_1_engtype_copy\t20.0\nVRAM\tluid_0x00000000_0x0002ff2a_phys_0\t1048576\nGPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004";
        let metrics = parse_metrics(sample, &[]);
        assert_eq!(metrics.len(), 1);
        assert!((metrics[0].gpu_utilization_pct - 90.0).abs() < 0.001);
    }

    #[test]
    fn summed_engine_utilization_above_100_percent_clamps() {
        // Several processes hammering the same engine can sum past 100% of
        // a single engine's nominal capacity; the reported utilization must
        // still clamp at 100, matching Task Manager's own ceiling.
        let sample = "UTIL\tpid_1_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t60.0\nUTIL\tpid_2_luid_0x00000000_0x0002ff2a_phys_0_eng_0_engtype_3d\t70.0\nVRAM\tluid_0x00000000_0x0002ff2a_phys_0\t1048576\nGPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004";
        let metrics = parse_metrics(sample, &[]);
        assert_eq!(metrics.len(), 1);
        assert!((metrics[0].gpu_utilization_pct - 100.0).abs() < 0.001);
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
        let metrics = parse_metrics(sample, &[]);
        assert!(metrics.is_empty());
        assert_eq!(parse_system_info(sample).physical_gpu_count, 1);
        // The empty result must not be silently indistinguishable from "no
        // GPU" — the caller needs a warning to show instead.
        assert!(counter_samples_missing_warning(sample, &metrics).is_some());
    }

    #[test]
    fn counter_samples_missing_warning_is_none_when_metrics_present() {
        let sample = "UTIL\tluid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1048576\nGPU\tDiscrete GPU\t3221225472\t1.0";
        let metrics = parse_metrics(sample, &[]);
        assert!(!metrics.is_empty());
        assert!(counter_samples_missing_warning(sample, &metrics).is_none());
    }

    #[test]
    fn counter_samples_missing_warning_is_none_when_no_adapters_at_all() {
        // No `Win32_VideoController` line either — genuinely no GPU, not a
        // counter-query failure, so no warning is warranted.
        assert!(counter_samples_missing_warning("", &[]).is_none());
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
        // WMI's AdapterRAM here (4293918720, just under 2^32 and
        // above the coded cap threshold) is the classic 32-bit-field truncation case; ADLX's own
        // query reports the real 8GiB so parse_metrics should prefer it.
        let sample = "UTIL\tluid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1048576\nGPU\tDiscrete GPU\t4293918720\t1.0";
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
            "{SAMPLE}\nUTIL\tluid_0x00000000_0x0000cafe_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000cafe_phys_0\t1048576\nGPU\tOther GPU\t4293918720\t1.0"
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
            "{SAMPLE}\nUTIL\tluid_0x00000000_0x0000cafe_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000cafe_phys_0\t1048576\nGPU\tOther GPU\t4293918720\t1.0"
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
    fn multi_adapter_utilization_clamps_to_100_when_summed_processes_exceed_it() {
        // Two processes both drawing on the same engine of the same adapter
        // (summed per the doc comment on `engine_util`) push the raw total
        // past 100 on a *second real adapter* alongside the one in `SAMPLE` —
        // the multi-adapter branch's own clamp must still catch it, same as
        // the single-adapter branch above it.
        let two_adapters = format!(
            "{SAMPLE}\nUTIL\tpid_9998_luid_0x00000000_0x0000cafe_phys_0_eng_2_engtype_3d\t95.0\nUTIL\tpid_9999_luid_0x00000000_0x0000cafe_phys_0_eng_2_engtype_3d\t95.0\nGPU\tOther GPU\t4293918720\t1.0"
        );
        let metrics = parse_metrics(&two_adapters, &[]);
        assert_eq!(metrics.len(), 2);
        let gpu = metrics
            .iter()
            .find(|g| g.vram_used_mb == 0)
            .expect("the cafe device's metrics");
        assert_eq!(gpu.gpu_utilization_pct, 100.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn utilization_across_two_engines_of_the_same_type_takes_max_not_sum() {
        let sample = "UTIL\tpid_1_luid_0x00000000_0x0002ff2a_phys_0_eng_2_engtype_copy\t60.0\nUTIL\tpid_1_luid_0x00000000_0x0002ff2a_phys_0_eng_3_engtype_copy\t60.0\nVRAM\tluid_0x00000000_0x0002ff2a_phys_0\t1048576\nGPU\tAMD Radeon 780M Graphics\t1073741824\t32.0.31041.1004";
        let metrics = parse_metrics(sample, &[]);
        assert_eq!(metrics[0].gpu_utilization_pct, 60.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn single_adapter_falls_back_to_wmi_vram_when_adlx_reading_count_mismatches() {
        // WMI's AdapterRAM here is also at the truncation-cap threshold, so
        // the fallback value itself degrades to 0 per `wmi_vram_total_mb` —
        // see `single_adapter_falls_back_to_wmi_vram_below_cap_threshold`
        // for the case where the WMI fallback value is actually trusted.
        let sample = "UTIL\tluid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1048576\nGPU\tDiscrete GPU\t4293918720\t1.0";
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
        assert_eq!(metrics[0].vram_total_mb, 0);
        assert_eq!(metrics[0].temperature_c, 0.0);
        assert_eq!(metrics[0].power_w, 0.0);
        assert_eq!(metrics[0].clock_mhz, None);
    }

    #[test]
    fn single_adapter_falls_back_to_wmi_vram_below_cap_threshold() {
        // No ADLX reading at all, and the WMI value is comfortably below the
        // truncation-cap threshold, so it's trusted as-is.
        let sample = "UTIL\tluid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1048576\nGPU\tDiscrete GPU\t3221225472\t1.0";
        let metrics = parse_metrics(sample, &[]);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].vram_total_mb, 3_221_225_472 / (1024 * 1024));
    }

    #[test]
    fn single_adapter_adlx_reading_with_no_total_vram_falls_back_to_wmi() {
        // The ADLX reading matches (exactly one, for exactly one real
        // adapter) but its own TotalVRAM query failed (`None`) — still
        // falls back to the WMI value rather than reporting 0.
        let sample = "UTIL\tluid_0x00000000_0x0000b3a4_phys_0_eng_0_engtype_3d\t1.0\nVRAM\tluid_0x00000000_0x0000b3a4_phys_0\t1048576\nGPU\tDiscrete GPU\t3221225472\t1.0";
        let adlx = [AdlxReading {
            temperature_c: 50.0,
            power_w: 10.0,
            clock_mhz: Some(1800.0),
            total_vram_mb: None,
        }];
        let metrics = parse_metrics(sample, &adlx);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].vram_total_mb, 3_221_225_472 / (1024 * 1024));
    }

    #[test]
    fn device_key_without_luid_substring_falls_back_to_whole_instance() {
        // Defensive fallback for an unexpected instance-name shape (no
        // `luid_` segment at all) — must not panic, and should still key
        // consistently rather than silently merging unrelated instances.
        assert_eq!(
            device_key("some_unusual_instance_name"),
            "some_unusual_instance_name"
        );
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

    fn reading(temperature_c: f32) -> AdlxReading {
        AdlxReading {
            temperature_c,
            power_w: 0.0,
            clock_mhz: None,
            total_vram_mb: None,
        }
    }

    /// The crash this guards against came from initialising and tearing down
    /// ADLX on every tick. Initialisation must run exactly once however many
    /// reads follow, and every read must be served by that one thread.
    #[test]
    fn resident_reader_initialises_once_and_reads_on_one_thread() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;

        let inits = Arc::new(AtomicUsize::new(0));
        let inits_in_worker = inits.clone();
        let reader = ResidentReader::spawn(
            move || {
                inits_in_worker.fetch_add(1, Ordering::SeqCst);
                Some(std::thread::current().id())
            },
            |init_thread| {
                // Temperature encodes whether this read ran on the thread
                // that initialised the state.
                let same = *init_thread == std::thread::current().id();
                vec![reading(if same { 1.0 } else { 0.0 })]
            },
        );

        for _ in 0..25 {
            let out = reader.read().expect("a live reader answers");
            assert_eq!(out.len(), 1);
            assert!((out[0].temperature_c - 1.0).abs() < f32::EPSILON);
        }
        assert_eq!(inits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn resident_reader_degrades_to_empty_when_init_fails_and_does_not_retry() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;

        let inits = Arc::new(AtomicUsize::new(0));
        let inits_in_worker = inits.clone();
        let reader = ResidentReader::spawn(
            move || {
                inits_in_worker.fetch_add(1, Ordering::SeqCst);
                None::<()>
            },
            |()| vec![reading(1.0)],
        );

        for _ in 0..5 {
            // An absent driver answers Ok with nothing, never ReaderGone:
            // the dashboard must not warn about a machine that simply has no
            // AMD GPU.
            assert!(
                matches!(reader.read(), Ok(readings) if readings.is_empty()),
                "a failed init must read as an absent driver, not a reader fault"
            );
        }
        assert_eq!(inits.load(Ordering::SeqCst), 1);
    }

    /// The asymmetry raised in review on #1020: `read` was wrapped in
    /// `catch_unwind` and `init` was not.
    ///
    /// A panic escaping `init` unwound the reader thread before the request
    /// loop began, so `inbox` dropped and every later tick answered
    /// `ThreadGone` — the one verdict indistinguishable from "the thread never
    /// started", for a cause that is neither. Worse, the reader is built once
    /// per process, so that verdict was permanent.
    ///
    /// Both halves are asserted, and the second is what makes this
    /// discriminating: the answer must name the init panic, **and** the reader
    /// must still be answering on later calls rather than having died. Before
    /// the fix the first assertion fails (`ThreadGone`, not `InitPanicked`).
    #[test]
    fn a_panicking_init_is_named_and_leaves_the_reader_answering() {
        let reader = ResidentReader::spawn(
            || -> Option<()> { panic!("ADLX init blew up") },
            |()| vec![reading(1.0)],
        );

        for _ in 0..5 {
            // `matches!`, not `assert_eq!`: `AdlxReading` carries no `Debug`
            // or `PartialEq`, and adding them to a production struct to satisfy
            // a test would be the tail wagging the dog.
            assert!(
                matches!(reader.read(), Err(ReaderGone::InitPanicked)),
                "an init panic must be named, not reported as a thread that never started"
            );
        }
        assert_eq!(
            ReaderGone::InitPanicked.warning(),
            "ADLX initialisation panicked"
        );
    }

    /// The `snapshot` mapping, which nothing reached: `snapshot` shells out to
    /// PowerShell unconditionally, so the `Done(Err(_))` arm this change added
    /// was wired up and never executed by a test. See `adlx_outcome`.
    ///
    /// Every arm is covered, including the one that must stay silent: an absent
    /// driver answers `Ok` with no readings and raises no warning, because a
    /// machine with no AMD GPU is not a fault to report.
    #[test]
    fn the_adlx_outcome_mapping_warns_for_faults_and_stays_silent_for_an_absent_driver() {
        let (readings, warning) = adlx_outcome(SingleFlight::Done(Ok(vec![reading(42.0)])));
        assert_eq!(readings.len(), 1);
        assert!(warning.is_none(), "a good read raises nothing");

        let (readings, warning) = adlx_outcome(SingleFlight::Done(Ok(Vec::new())));
        assert!(readings.is_empty());
        assert!(
            warning.is_none(),
            "an absent driver is an expected state, not a warning"
        );

        for gone in [
            ReaderGone::ReadPanicked,
            ReaderGone::InitPanicked,
            ReaderGone::ThreadGone,
        ] {
            let (readings, warning) = adlx_outcome(SingleFlight::Done(Err(gone)));
            assert!(readings.is_empty());
            assert_eq!(
                warning.as_deref(),
                Some(gone.warning().as_str()),
                "a reader fault must surface its own wording: {gone:?}"
            );
        }

        for (outcome, expected) in [
            (SingleFlight::Panicked, "ADLX reading task panicked"),
            (
                SingleFlight::StillRunning,
                "a previous call is still running",
            ),
        ] {
            let (readings, warning) = adlx_outcome(outcome);
            assert!(readings.is_empty());
            assert!(
                warning.as_deref().is_some_and(|w| w.contains(expected)),
                "expected {expected:?}, got {warning:?}"
            );
        }

        let (readings, warning) = adlx_outcome(SingleFlight::TimedOut);
        assert!(readings.is_empty());
        assert!(
            warning.as_deref().is_some_and(|w| w.contains("timed out")),
            "{warning:?}"
        );
    }

    #[test]
    fn a_panicking_read_is_reported_and_leaves_the_reader_usable() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;

        let reads = Arc::new(AtomicUsize::new(0));
        let reads_in_worker = reads.clone();
        // Panics on the first read only, so the second shows the reader is
        // still answering rather than having unwound.
        let reader = ResidentReader::spawn(
            || Some(()),
            move |()| {
                assert!(
                    reads_in_worker.fetch_add(1, Ordering::SeqCst) != 0,
                    "driver blew up"
                );
                vec![reading(1.0)]
            },
        );

        // The panic is caught and named, never silently empty: the reader is
        // built once per process, so a silent failure would blank
        // temperature, power and clock for the whole session and read as a
        // machine with no AMD driver.
        assert_eq!(reader.read().err(), Some(ReaderGone::ReadPanicked));
        let after = reader.read().expect("the reader survives a caught panic");
        assert_eq!(after.len(), 1);
        assert!((after[0].temperature_c - 1.0).abs() < f32::EPSILON);
        assert_eq!(reads.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn every_reader_fault_warns_in_words_that_name_adlx_and_tell_the_faults_apart() {
        // `snapshot` pairs each of these with no readings, so the state the
        // user lands in is "the ADLX columns are blank, and here is why".
        // An absent driver is the other case and deliberately has no warning.
        for gone in [ReaderGone::ReadPanicked, ReaderGone::ThreadGone] {
            let warning = gone.warning();
            assert!(
                warning.to_lowercase().contains("adlx"),
                "a warning shown next to blank columns must name the subsystem: {warning}"
            );
        }
        assert_ne!(
            ReaderGone::ReadPanicked.warning(),
            ReaderGone::ThreadGone.warning(),
            "a panicking driver and a dead reader need different text to be told apart"
        );
    }

    #[test]
    fn a_reader_whose_thread_is_gone_reports_it_rather_than_answering_empty() {
        // A sender whose receiver has been dropped is what a dead or
        // never-started reader thread looks like from the caller's side.
        let (requests, inbox) = std::sync::mpsc::channel();
        drop(inbox);
        let dead = ResidentReader { requests };
        assert_eq!(dead.read().err(), Some(ReaderGone::ThreadGone));
    }

    #[tokio::test]
    async fn single_flight_returns_value_and_clears_flag() {
        static FLAG: AtomicBool = AtomicBool::new(false);
        let out = run_single_flight(&FLAG, Duration::from_secs(5), || 7).await;
        assert_eq!(out, SingleFlight::Done(7));
        assert!(!FLAG.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn single_flight_does_not_start_second_call_while_first_is_wedged() {
        static FLAG: AtomicBool = AtomicBool::new(false);
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let out = run_single_flight(&FLAG, Duration::from_millis(50), move || {
            let _ = release_rx.recv();
        })
        .await;
        assert_eq!(out, SingleFlight::TimedOut);

        // The wedged call still holds its thread: a second call must not run.
        let ran = std::sync::Arc::new(AtomicBool::new(false));
        let ran2 = ran.clone();
        let out = run_single_flight(&FLAG, Duration::from_millis(50), move || {
            ran2.store(true, Ordering::Release);
        })
        .await;
        assert_eq!(out, SingleFlight::StillRunning);
        assert!(!ran.load(Ordering::Acquire));

        // Once the wedged call returns, the flag clears and calls resume.
        release_tx.send(()).unwrap();
        for _ in 0..100 {
            if !FLAG.load(Ordering::Acquire) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let out = run_single_flight(&FLAG, Duration::from_secs(5), || 1).await;
        assert_eq!(out, SingleFlight::Done(1));
    }

    #[tokio::test]
    async fn single_flight_panic_clears_flag() {
        static FLAG: AtomicBool = AtomicBool::new(false);
        let out: SingleFlight<()> =
            run_single_flight(&FLAG, Duration::from_secs(5), || panic!("boom")).await;
        assert_eq!(out, SingleFlight::Panicked);
        assert!(!FLAG.load(Ordering::Acquire));
    }

    #[test]
    fn powershell_path_prefers_existing_system32_binary() {
        let p = powershell_path(Some(OsStr::new("C:\\Windows")), |_| true);
        assert!(p.starts_with("C:\\Windows"), "{p:?}");
        assert!(p.ends_with("powershell.exe"));
        assert!(p.components().count() > 1);
    }

    #[test]
    fn powershell_path_falls_back_to_bare_name() {
        let missing = PathBuf::from("powershell.exe");
        assert_eq!(
            powershell_path(Some(OsStr::new("C:\\Windows")), |_| false),
            missing
        );
        assert_eq!(powershell_path(None, |_| true), missing);
    }

    // adlx_readings() itself (the real ADLX FFI chain) is Windows-only and
    // hardware/driver-dependent — not unit-testable here; CI's
    // windows-build-and-test job covers compilation of the #[cfg(windows)]
    // path, and apply_adlx above covers the pure mapping logic.
}
