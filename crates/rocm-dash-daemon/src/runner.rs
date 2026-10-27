// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Owner task: drives collectors on tick cadences and broadcasts Snapshots.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod discovery;

use discovery::{
    DiscoverySource, DiscoveryState, Reconcile, TelemetryState, registry_read_warning,
    vllm_scrape_targets,
};

use crate::bench_ring::BenchRing;
use crate::persist::SessionWriter;
use crate::snapshot_ring::SnapshotRing;

use chrono::{DateTime, Utc};
use rocm_dash_collectors::amd_smi::AmdSmiCollector;
/// The real `/dev/kfd`, re-exported so callers without a direct
/// `rocm-dash-collectors` dependency can resolve their host-root view of it.
pub use rocm_dash_collectors::amd_smi::KFD_DEVICE;
use rocm_dash_collectors::bench_tail::CsvBenchTailer;
use rocm_dash_collectors::docker::DockerDiscovery;
use rocm_dash_collectors::host::HostCollector;
use rocm_dash_collectors::lemonade::LemonadeCollector;
use rocm_dash_collectors::parallel::parallel_scrape;
use rocm_dash_collectors::vllm_prom::VllmPrometheusCollector;
use rocm_dash_core::metrics::{
    GpuMetrics, GpuSystemInfo, Instance, InstanceStatus, ObservationMetadata, Snapshot,
};
use rocm_dash_core::observation::GenerationObservationTracker;
use rocm_dash_core::protocol::Event;
use rocm_dash_core::state::{State, StateEvent};
use rocm_dash_core::traits::{BenchTailer, DiscoveredService, InstanceSample, merge_instance};
use tokio::sync::broadcast;
use tokio::time::{MissedTickBehavior, interval};
use tracing::{info, trace, warn};

/// Refresh GPU system info (versions, partition modes) every N seconds.
const SYSINFO_REFRESH_SECS: u64 = 30;

/// Row identity and accepted model names for singleton Lemonade attribution.
#[derive(Clone)]
struct LemonadeRow {
    id: String,
    model_names: Vec<String>,
}

#[derive(Clone)]
struct LemonadeEndpoint {
    host: String,
    port: u16,
    collector: LemonadeCollector,
    rows: Vec<LemonadeRow>,
    /// The endpoint is the external probe (not a registry-managed target). This
    /// decides how strictly health is read: an external server that starts to
    /// require credentials is a failure to report, a managed target that does not
    /// serve the Lemonade protocol is not. It is a property of the endpoint, not
    /// of the tick.
    external: bool,
    /// This tick is a discovery tick: rows are rebuilt from health. External
    /// endpoints also appear on scrape-only ticks, with this `false`.
    discover_external: bool,
}

impl LemonadeEndpoint {
    fn warning(&self, phase: &str, error: &impl std::fmt::Display) -> String {
        let mut ids: Vec<&str> = self.rows.iter().map(|row| row.id.as_str()).collect();
        ids.sort_unstable();
        format!(
            "lemonade {phase} {}:{} [{}]: {error}",
            self.host,
            self.port,
            ids.join(", ")
        )
    }
}

/// Options for `run_loop`. Mirrors daemon CLI flags + config.
#[derive(Debug, Clone)]
pub struct RunnerOptions {
    pub bench_csv: Option<PathBuf>,
    pub enable_docker: bool,
    pub image_patterns: Option<String>,
    pub gpu_tick: Duration,
    pub discovery_tick: Duration,
    pub instance_tick: Duration,
    /// Internal gate for the per-instance vLLM Prometheus scrape. Currently
    /// always `false` (the scrape runs) — it is NOT yet wired to a CLI flag or
    /// config field, so callers cannot turn it off today; it exists as the
    /// single seam a future opt-out would flip. Independent of `enable_docker`:
    /// the scraper is a plain HTTP GET against a vLLM instance's `/metrics`
    /// endpoint and runs for BOTH Docker-discovered and managed (native,
    /// `services_dir`-discovered) instances.
    pub disable_vllm_metrics: bool,
    /// Hostname to scrape; `127.0.0.1` for the typical co-located daemon.
    pub vllm_metrics_host: String,
    /// Probe-based Lemonade discovery: when enabled, probe a local Lemonade
    /// endpoint each discovery tick and surface its healthy loaded models.
    /// `rocm dash` passes the dashboard config value, which defaults to on;
    /// only `RunnerOptions::default()` is off, for standalone use and tests.
    pub enable_lemonade: bool,
    /// Lemonade endpoint host + port (defaults `127.0.0.1:13305`).
    pub lemonade_host: String,
    pub lemonade_port: u16,
    /// When set, every broadcast Event is appended to
    /// `{persist_dir}/session-{ts}.ndjson` for offline replay.
    pub persist_dir: Option<PathBuf>,
    /// rocm-cli managed-service registry directory (`AppPaths::services_dir()`).
    /// When set, the daemon reads `ManagedServiceRecord`s each discovery tick and
    /// surfaces live ones as scrape targets (port from the registry), so a model
    /// served via `rocm serve` appears in the dashboard with live `gen_tps`
    /// without Docker discovery. `None` by default; `rocm dash` sets it.
    pub services_dir: Option<PathBuf>,
    /// Path or command name for the `amd-smi` binary. The managed ROCm SDK
    /// ships it inside the runtime wheel's bin directory rather than on `PATH`,
    /// so the caller resolves it (via `rocm_core::resolve_amd_smi_binary`) and
    /// passes it here. `None` falls back to looking up `amd-smi` on `PATH`.
    pub amd_smi_binary: Option<OsString>,
    /// **Test-only.** Skip the mandatory GPU-device pre-flight in amd-smi
    /// detection so a *fake* `amd_smi_binary` is actually invoked on a GPU-less
    /// CI host instead of short-circuiting to "no GPU". Never set in
    /// production: the device guard prevents a *real* `amd-smi` from hanging in
    /// uninterruptible D-state. Only the daemon integration test that points
    /// `amd_smi_binary` at a deliberately-slow fake script flips this, so the
    /// off-critical-path detection behaviour is genuinely exercised.
    pub amd_smi_skip_device_preflight: bool,
    /// Precomputed GPU-reachability verdict (from `rocm_core::has_usable_amd_gpu`)
    /// that lets the amd-smi device pre-flight pass without a readable
    /// `/dev/kfd` — the WSL case, where that verdict is the same one `serve`
    /// and `examine` already act on. This crate has no `rocm-core` dependency
    /// to compute it with; `rocm-dash-collectors`, which does have one since
    /// EAI-8457, still takes it threaded through rather than recomputing it, so
    /// the dashboard cannot reach a different verdict about the same host than
    /// `serve` and `examine` did. The caller (`apps/rocm`) computes it once and
    /// passes the answer through both. `false` (the default) preserves the
    /// bare-metal-only `/dev/kfd` check.
    pub amd_smi_gpu_reachable: bool,
    /// Where the caller's hardware probes put `/dev/kfd`.
    ///
    /// This is `rocm_core::host_path`'s answer, threaded in like
    /// `amd_smi_gpu_reachable`. The amd-smi pre-flight needs both this and
    /// the real `/dev/kfd` to be readable, so a simulated host root can hide
    /// the runner's GPU but never start `amd-smi` on a host without one.
    /// Defaults to the real [`KFD_DEVICE`], and equals it in every build
    /// without a simulated root, which leaves the pre-flight unchanged.
    pub amd_smi_host_kfd_device: PathBuf,
    /// **Test-only.** When set, cycle timestamps come from the logical clock
    /// this file controls instead of `Utc::now()` — see [`TestClockDirective`]
    /// for the file's grammar. Production callers leave this unset; E2E
    /// scenarios use it to sit still on, or step across, observation-validity
    /// boundaries without racing wall-clock scheduling.
    pub test_clock_offset_path: Option<PathBuf>,
}

impl Default for RunnerOptions {
    fn default() -> Self {
        Self {
            bench_csv: None,
            enable_docker: false,
            image_patterns: None,
            gpu_tick: Duration::from_secs(1),
            discovery_tick: Duration::from_secs(5),
            instance_tick: Duration::from_secs(2),
            disable_vllm_metrics: false,
            vllm_metrics_host: "127.0.0.1".into(),
            enable_lemonade: false,
            lemonade_host: "127.0.0.1".into(),
            lemonade_port: rocm_dash_collectors::lemonade::LEMONADE_PORT,
            persist_dir: None,
            services_dir: None,
            amd_smi_binary: None,
            amd_smi_skip_device_preflight: false,
            amd_smi_gpu_reachable: false,
            amd_smi_host_kfd_device: PathBuf::from(KFD_DEVICE),
            test_clock_offset_path: None,
        }
    }
}

#[derive(Default)]
pub struct Runner {
    pub state: State,
}

/// Whether `run_loop` should construct the [`VllmPrometheusCollector`].
///
/// EAI-7359: the scraper is a plain HTTP GET against a vLLM instance's
/// `/metrics` endpoint — it has no Docker dependency. Managed (native)
/// instances are already discovered via `services_dir` independently of
/// Docker, so this is gated on `disable_vllm_metrics` ALONE; `enable_docker`
/// must never factor in here (it continues to gate only `DockerDiscovery`).
const fn vllm_metrics_enabled(opts: &RunnerOptions) -> bool {
    !opts.disable_vllm_metrics
}

/// **Test-only.** A directive read from the file `test_clock_offset_path`
/// points at, controlling the daemon's logical observation clock.
///
/// Grammar — the file holds one trimmed token sequence:
///
/// | Content     | Meaning                                                      |
/// |-------------|--------------------------------------------------------------|
/// | `<int>`     | free-running: one `gpu_tick` of logical time per daemon cycle, shifted by `<int>` seconds |
/// | `hold`      | held: logical time stops at the cycle that first reads this   |
/// | `hold <int>`| held at that same cycle, then shifted by `<int>` seconds      |
///
/// `hold` exists because free-running logical time is NOT decoupled from the
/// host: it advances one tick per cycle and the cycles are paced by a
/// wall-clock interval, so a scenario descheduled between two assertions still
/// loses observation-validity budget it never meant to spend (EAI-7960). A held
/// clock cannot be moved by anything except the scenario rewriting the file, so
/// a "still held" assertion can be made at any later moment and an expiry is
/// crossed only by an explicit `hold <int>` step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestClockDirective {
    /// Logical time advances one tick per cycle, shifted by the given seconds.
    FreeRunning(i64),
    /// Logical time stands still at the hold point, shifted by the given seconds.
    Held(i64),
}

impl Default for TestClockDirective {
    /// What an absent or unparsable file means: the pre-`hold` behaviour.
    fn default() -> Self {
        Self::FreeRunning(0)
    }
}

/// **Test-only.** The daemon's logical observation clock, driven by the file at
/// `RunnerOptions::test_clock_offset_path`. Carried across cycles by `run_loop`
/// because a held clock has to remember *where* it was held.
#[derive(Debug, Default)]
struct TestClock {
    /// Last directive parsed. Retained when a read fails or yields content that
    /// does not parse, so a transient unreadable file (a scenario rewriting it
    /// non-atomically) can never silently resume a held clock.
    directive: TestClockDirective,
    /// Cycle count the clock was held at; `None` while free-running.
    held_at_tick: Option<u64>,
}

impl TestClock {
    /// Timestamp for this cycle: `Utc::now()` in production (no clock file), or
    /// the logical time the file's directive describes.
    fn cycle_timestamp(
        &mut self,
        epoch: DateTime<Utc>,
        tick: Duration,
        tick_count: u64,
        offset_path: Option<&std::path::Path>,
    ) -> DateTime<Utc> {
        let Some(path) = offset_path else {
            return Utc::now();
        };
        if let Some(directive) = read_test_clock_directive(path) {
            self.directive = directive;
        }
        let (logical_ticks, offset_secs) = match self.directive {
            TestClockDirective::FreeRunning(offset_secs) => {
                self.held_at_tick = None;
                (tick_count, offset_secs)
            }
            // Hold at the first cycle that saw a `hold`; later `hold <int>`
            // directives shift from that same point rather than re-holding,
            // so the expiry step's advance is exactly the offset it writes.
            TestClockDirective::Held(offset_secs) => {
                (*self.held_at_tick.get_or_insert(tick_count), offset_secs)
            }
        };
        let tick_millis = tick.as_millis().saturating_mul(u128::from(logical_ticks));
        let logical_millis = i64::try_from(tick_millis).unwrap_or(i64::MAX);
        epoch
            .checked_add_signed(chrono::TimeDelta::milliseconds(logical_millis))
            .and_then(|at| at.checked_add_signed(chrono::TimeDelta::seconds(offset_secs)))
            .unwrap_or(DateTime::<Utc>::MAX_UTC)
    }
}

/// Read and parse the test clock file. `None` when it cannot be read or does
/// not parse — the caller then keeps the directive already in force.
fn read_test_clock_directive(path: &std::path::Path) -> Option<TestClockDirective> {
    parse_test_clock_directive(std::fs::read_to_string(path).ok()?.trim())
}

/// Parse one [`TestClockDirective`]; see its grammar table.
fn parse_test_clock_directive(value: &str) -> Option<TestClockDirective> {
    if let Some(offset) = value.strip_prefix("hold") {
        let offset = offset.trim();
        return if offset.is_empty() {
            Some(TestClockDirective::Held(0))
        } else {
            offset.parse().ok().map(TestClockDirective::Held)
        };
    }
    value.parse().ok().map(TestClockDirective::FreeRunning)
}

/// Which detection call `run_loop`'s background gpu-init task makes, decided
/// up front from [`RunnerOptions`] so the decision itself is a plain value a
/// test can assert on without spawning the task or touching a real binary.
enum AmdSmiDetectPlan {
    SkipDevicePreflight(OsString),
    /// Binary, `gpu_reachable` verdict, host-root view of `/dev/kfd`.
    Detect(OsString, bool, PathBuf),
}

/// `amd_smi_binary: None` means "no override", not "no detection" — it still
/// runs the real pre-flight against the literal `amd-smi` command name, with
/// the same threaded `gpu_reachable` verdict and `host_kfd_device` as an
/// explicit binary.
fn amd_smi_detect_plan(
    binary: Option<OsString>,
    skip_device_preflight: bool,
    gpu_reachable: bool,
    host_kfd_device: PathBuf,
) -> AmdSmiDetectPlan {
    match binary {
        Some(binary) if skip_device_preflight => AmdSmiDetectPlan::SkipDevicePreflight(binary),
        Some(binary) => AmdSmiDetectPlan::Detect(binary, gpu_reachable, host_kfd_device),
        None => AmdSmiDetectPlan::Detect("amd-smi".into(), gpu_reachable, host_kfd_device),
    }
}

/// Loop forever: tick host + gpu metrics + bench rows, apply through reducer, broadcast.
///
/// `tick_override` lets tests run faster than `opts.gpu_tick`; production passes
/// `None` so the configured cadence drives the loop.
pub async fn run_loop(
    tick_override: Option<Duration>,
    tx: broadcast::Sender<Event>,
    ring: Arc<Mutex<SnapshotRing>>,
    bench_ring: Arc<Mutex<BenchRing>>,
    persist: Option<Arc<Mutex<SessionWriter>>>,
    opts: RunnerOptions,
) {
    let mut runner = Runner::default();
    let mut host = HostCollector::new();
    let tick = tick_override.unwrap_or(opts.gpu_tick);
    let mut ticker = interval(tick);
    let test_clock_epoch = Utc::now();
    let mut test_clock = TestClock::default();
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    // Compute multipliers vs the gpu tick.
    let discovery_ticks = ticks_per(opts.discovery_tick, tick);
    let instance_ticks = ticks_per(opts.instance_tick, tick);
    let sysinfo_refresh_ticks = ticks_per(Duration::from_secs(SYSINFO_REFRESH_SECS), tick);

    let vllm = if vllm_metrics_enabled(&opts) {
        Some(Arc::new(VllmPrometheusCollector::new(
            opts.vllm_metrics_host.clone(),
            Duration::from_millis(1500),
        )))
    } else {
        None
    };

    // Probe-based Lemonade discovery, gated on `opts.enable_lemonade` (the
    // dashboard config default is on; `RunnerOptions::default()` is off).
    // Distinct from Docker/vLLM discovery — a local server, not a
    // container — so it is tracked separately from the Docker known set.
    let lemonade = if opts.enable_lemonade {
        info!(
            host = %opts.lemonade_host,
            port = opts.lemonade_port,
            "lemonade discovery enabled"
        );
        Some(LemonadeCollector::new(
            opts.lemonade_host.clone(),
            opts.lemonade_port,
            Duration::from_millis(1500),
        ))
    } else {
        None
    };
    // Persistent endpoint collectors share their reqwest connection pools across cycles.
    let mut managed_lemonade_collectors: HashMap<(String, u16), LemonadeCollector> = HashMap::new();
    let mut bench = opts.bench_csv.as_ref().map(|p| {
        info!(path = %p.display(), "tailing benchmark CSV");
        CsvBenchTailer::new(p.clone())
    });

    let docker = if opts.enable_docker {
        if let Some(d) = DockerDiscovery::detect(opts.image_patterns.clone()).await {
            info!("docker discovery enabled");
            Some(d)
        } else {
            warn!("docker discovery requested but daemon unreachable; disabled");
            None
        }
    } else {
        None
    };
    let mut discovery = DiscoveryState::default();
    let mut telemetry = TelemetryState::default();
    // Per-container VRAM (MB) from the last amd-smi `process` scrape. Refreshed
    // on the instance cadence and reused every tick so the attributed value is
    // stable between scrapes (mirrors how GPU power drives tokens_per_watt).
    let mut per_container_used: HashMap<String, u64> = HashMap::new();

    // amd-smi GPU detection plus the first `system_info()` can take up to ~15s
    // on some hosts (subprocess spawn plus the per-call detect/run timeouts).
    // Doing it inline here blocked the run loop's very first tick — and thus
    // managed-service discovery and the first snapshot broadcast — behind it, so
    // an already-running model did not surface in the dashboard until GPU
    // detection finished (a visible ~15-20s "0 models running" lag while
    // `rocm services` already reported it). Run detection off the critical path:
    // the loop starts ticking immediately (surfacing serving instances within
    // one discovery tick) and GPU metrics fill in the moment detection lands.
    let detect_plan = amd_smi_detect_plan(
        opts.amd_smi_binary.clone(),
        opts.amd_smi_skip_device_preflight,
        opts.amd_smi_gpu_reachable,
        opts.amd_smi_host_kfd_device.clone(),
    );
    let (gpu_init_tx, mut gpu_init_rx) =
        tokio::sync::oneshot::channel::<(Option<AmdSmiCollector>, Option<GpuSystemInfo>)>();
    tokio::spawn(async move {
        let gpu = match detect_plan {
            AmdSmiDetectPlan::SkipDevicePreflight(binary) => {
                AmdSmiCollector::detect_with_binary_skipping_device_preflight(binary).await
            }
            AmdSmiDetectPlan::Detect(binary, gpu_reachable, host_kfd_device) => {
                AmdSmiCollector::detect_with_binary(binary, gpu_reachable, &host_kfd_device).await
            }
        };
        let info = match &gpu {
            Some(g) => Some(g.system_info().await),
            None => None,
        };
        let _ = gpu_init_tx.send((gpu, info));
    });
    let mut gpu: Option<AmdSmiCollector> = None;
    let mut gpu_system_info: Option<GpuSystemInfo> = None;
    let mut gpu_init_done = false;

    let mut tick_count: u64 = 0;
    let mut last_sysinfo_refresh: u64 = 0;
    // Static per-GPU power caps (`amd-smi static`), refreshed on the sysinfo
    // cadence and stamped onto each tick's metrics. `None` until first fetched.
    let mut power_limits: HashMap<String, f32> = HashMap::new();
    let mut last_power_limit_refresh: Option<u64> = None;
    let mut warned_no_power_limits = false;
    // Discovery/scrape warnings are produced on their own (slower) cadences, so
    // holding them only for the tick that produced them makes the header ⚠ badge
    // blink on and off. Keep the last result until that cadence runs again.
    let mut docker_warn: Option<String> = None;
    let mut registry_warn: Option<String> = None;
    let mut scrape_warns: Vec<String> = Vec::new();
    let mut lemonade_discovery_warn: Option<String> = None;

    loop {
        ticker.tick().await;
        tick_count += 1;
        // Single wall-clock anchor for this loop iteration. All counter/direct
        // observations and the assembled Snapshot timestamp share this instant so
        // every instance refreshed in this cycle serialises Fresh deterministically.
        let cycle_at = test_clock.cycle_timestamp(
            test_clock_epoch,
            tick,
            tick_count,
            opts.test_clock_offset_path.as_deref(),
        );

        // Adopt the background amd-smi detection result the moment it lands,
        // without ever blocking the loop while it is still in flight. Until then
        // `gpu` stays `None` (GPU panels read "unavailable") but discovery and
        // snapshots run every tick, so serving instances appear promptly.
        if !gpu_init_done {
            match gpu_init_rx.try_recv() {
                Ok((detected, info)) => {
                    gpu_init_done = true;
                    if let Some(i) = &info {
                        info!(
                            gpus = i.physical_gpu_count,
                            model = %i.gpu_model,
                            rocm = i.rocm_version.as_deref().unwrap_or("?"),
                            "amd-smi detected"
                        );
                    } else {
                        warn!(
                            "amd-smi not available (no accessible GPU device or `amd-smi version` failed); GPU disabled"
                        );
                    }
                    gpu = detected;
                    gpu_system_info = info;
                    last_sysinfo_refresh = tick_count;
                }
                // The detection task dropped its sender without ever sending.
                // It has no fallible await today, so this is latent rather than
                // a live path — but if it ever regressed, polling `Empty`
                // forever would hang `gpu_init_done` at false and suppress the
                // honest "unavailable" warning below. Settle as unavailable and
                // say so once, so the badge resolves instead of silently never
                // clearing.
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    gpu_init_done = true;
                    warn!("amd-smi detection task ended without a result; GPU disabled");
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }
        }

        let mut warnings = Vec::new();
        // Power caps are static, so they refresh on the sysinfo cadence. Fetch
        // them alongside this tick's metrics instead of ahead of it, so the
        // refresh tick pays one amd-smi round trip, not two in a row.
        let refresh_limits = gpu.is_some()
            && last_power_limit_refresh
                .is_none_or(|at| tick_count.saturating_sub(at) >= sysinfo_refresh_ticks);
        let mut gpus = if let Some(g) = &gpu {
            let (limits, metrics) = tokio::join!(
                async {
                    if refresh_limits {
                        Some(g.power_limits().await)
                    } else {
                        None
                    }
                },
                g.metrics()
            );
            if let Some(res) = limits {
                match res {
                    // An empty map (a schema the parser no longer matches) must
                    // not wipe caps we already hold, and is worth saying once.
                    Ok(l) if l.is_empty() => {
                        if power_limits.is_empty() && !warned_no_power_limits {
                            warn!(
                                "amd-smi static reported no readable power cap; power stays uncolored"
                            );
                            warned_no_power_limits = true;
                        }
                    }
                    Ok(l) => power_limits = l,
                    // A failed fetch keeps the previous caps; retry next interval.
                    Err(_) => {}
                }
                last_power_limit_refresh = Some(tick_count);
            }
            match metrics {
                Ok(v) => v,
                Err(e) => {
                    warnings.push(format!("amd-smi metric: {e}"));
                    Vec::new()
                }
            }
        } else if gpu_init_done {
            // If the caller passed `amd_smi_gpu_reachable: true` (in practice,
            // only the WSL wiring in `apps/rocm/src/dash.rs` does — it gates the
            // rocm-core verdict on `is_wsl_host()`) and amd-smi *still* found
            // nothing, that's a real contradiction worth calling out — plumbing
            // readiness (e.g. `wsl_rocdxg_ready`) is not the same as amd-smi
            // enumerating a supported GPU, and a generic "device inaccessible"
            // message would flatly contradict what `examine` just told the same
            // user. Otherwise (bare metal, or nothing told us the GPU is
            // reachable) the device pre-flight itself may have failed, or it
            // passed but the binary was missing/unresolvable/failed to run —
            // both causes are named since either is possible here.
            warnings.push(if opts.amd_smi_gpu_reachable {
                "amd-smi is missing, unresolvable, or failed to run, even though a GPU was \
                 detected by other means (WSL ROCDXG bridge) - if it is installed, this GPU \
                 model may not be supported by the installed amd-smi/ROCm, or not supported on \
                 WSL yet"
                    .into()
            } else {
                "amd-smi unavailable (not installed, unresolvable, or the GPU device is \
                 inaccessible)"
                    .into()
            });
            Vec::new()
        } else {
            // Detection is still in flight (spawned off the critical path), so
            // `gpu == None` is transient here. Stay silent instead of flashing a
            // false "unavailable" alarm on healthy hardware for the ~15s window
            // until the first detection result lands.
            Vec::new()
        };

        for g in &mut gpus {
            g.power_limit_w = power_limits.get(&g.device_id).copied();
        }

        if gpu.is_some()
            && tick_count.saturating_sub(last_sysinfo_refresh) >= sysinfo_refresh_ticks
            && let Some(g) = &gpu
        {
            gpu_system_info = Some(g.system_info().await);
            last_sysinfo_refresh = tick_count;
        }

        // Service discovery — every `discovery_ticks` ticks, reconcile each
        // source against what it had live before, emit Discovered/Gone events,
        // and update reducer state.
        let discovery_due = tick_count == 1 || tick_count.is_multiple_of(discovery_ticks);
        let mut reconcile = Reconcile {
            state: &mut runner.state,
            telemetry: &mut telemetry,
            tx: &tx,
            persist: persist.as_ref(),
        };
        if let Some(d) = docker.as_ref()
            && discovery_due
        {
            match d.discover_async().await {
                Ok(svcs) => {
                    let seen: HashSet<String> =
                        svcs.iter().map(|s| s.container_id.clone()).collect();
                    reconcile.reconcile_set(
                        DiscoverySource::Docker,
                        &svcs,
                        &discovery.known_instances,
                        &seen,
                    );
                    discovery.known_instances = seen;
                    docker_warn = None;
                }
                Err(e) => docker_warn = Some(format!("docker discover: {e}")),
            }
        }

        // Managed-service registry discovery — read the rocm-cli
        // `serve` records and surface live ones as scrape targets. The port is
        // the registry's authority; non-vLLM engines are tracked so the vLLM
        // scrape below skips them. A model served via `rocm serve` thus appears
        // in the dashboard and is scraped for `gen_tps` without Docker.
        //
        // A pass that could not read the registry in full retires nothing, as
        // with the Docker source above: what is known stays known (no `Gone`),
        // and the instances keep the data they hold. Unlike Docker, whose listing
        // is all or nothing, a registry can fail on one record while the rest
        // read, so the readable records still add and refresh their services
        // (announced once, as ever). Only a registry that was read in full, even
        // an empty one, retires what it lacks.
        if let Some(services_dir) = opts.services_dir.as_ref()
            && discovery_due
        {
            let read = crate::registry::read_service_records(services_dir);
            let change = discovery.registry_health.observe(&read);
            change.log();
            match read {
                Ok(read) => {
                    let disc = crate::registry::discover_managed_services(&read.records);
                    discovery.managed_non_vllm = disc.non_vllm;
                    discovery.managed_roots = disc.roots;
                    let previous =
                        std::mem::replace(&mut discovery.managed_lemonade, disc.lemonade);
                    reconcile.reset_moved_lemonade_targets(
                        &disc.svcs,
                        &previous,
                        &discovery.managed_lemonade,
                    );
                    reconcile.reconcile_set(
                        DiscoverySource::Managed,
                        &disc.svcs,
                        &discovery.known_services,
                        &disc.seen,
                    );
                    discovery.known_services = disc.seen;
                    registry_warn = None;
                }
                Err(e) => {
                    // The records that did read still add and refresh their
                    // services; nothing is retired, because the one that did
                    // not read could be any of the known ones.
                    let mut disc = crate::registry::discover_managed_services(&e.partial.records);
                    // Lemonade targets follow the same rule: the ones this pass
                    // read are refreshed, the others are kept.
                    let mut next: Vec<_> = discovery
                        .managed_lemonade
                        .iter()
                        .filter(|target| !disc.seen.contains(&target.service_id))
                        .cloned()
                        .collect();
                    next.append(&mut disc.lemonade);
                    let previous = std::mem::replace(&mut discovery.managed_lemonade, next);
                    reconcile.reset_moved_lemonade_targets(
                        &disc.svcs,
                        &previous,
                        &discovery.managed_lemonade,
                    );
                    reconcile.upsert_set(
                        DiscoverySource::Managed,
                        &disc.svcs,
                        &discovery.known_services,
                    );
                    discovery.keep_through_partial_registry(disc);
                    registry_warn = Some(registry_read_warning(&e));
                }
            }
        }
        if tick_count.is_multiple_of(instance_ticks) {
            scrape_warns.clear();
        }

        // Independent endpoints run concurrently, but health always precedes
        // stats within each endpoint. Last-request stats have no model identity:
        // only a singleton loaded set matching exactly one row can own the rate.
        let scrape_now = tick_count.is_multiple_of(instance_ticks);
        if discovery_due || scrape_now {
            let managed_endpoint = discovery.managed_lemonade.iter().any(|target| {
                crate::registry::endpoints_match(
                    &target.host,
                    target.port,
                    &opts.lemonade_host,
                    opts.lemonade_port,
                )
            });
            let mut endpoints = Vec::new();
            if let Some(collector) = lemonade.as_ref() {
                if managed_endpoint && discovery_due {
                    // Registry binding wins even if its configured model differs
                    // from health. Never attach external controls to that binding.
                    Reconcile {
                        state: &mut runner.state,
                        telemetry: &mut telemetry,
                        tx: &tx,
                        persist: persist.as_ref(),
                    }
                    .reconcile_lemonade(&[], &mut discovery.lemonade_ids);
                    lemonade_discovery_warn = None;
                } else if !managed_endpoint && (discovery_due || !discovery.lemonade_ids.is_empty())
                {
                    endpoints.push(LemonadeEndpoint {
                        host: opts.lemonade_host.clone(),
                        port: opts.lemonade_port,
                        collector: collector.clone(),
                        rows: discovery
                            .lemonade_ids
                            .iter()
                            .filter_map(|id| {
                                runner.state.instances.get(id).map(|inst| LemonadeRow {
                                    id: id.clone(),
                                    model_names: vec![inst.model_name.clone()],
                                })
                            })
                            .collect(),
                        external: true,
                        discover_external: discovery_due,
                    });
                }
            }
            let mut groups: HashMap<(String, u16), Vec<LemonadeRow>> = HashMap::new();
            for target in &discovery.managed_lemonade {
                // Use the registry's endpoint equivalence without rewriting the
                // selected request host or conflating IPv4 and IPv6 listeners.
                // Prefer the cached representative so reorder keeps its client.
                // A cached collector dials a fixed address, so it is reused
                // only when it would dial the same host as this target.
                let endpoint = groups
                    .keys()
                    .find(|(host, port)| {
                        crate::registry::endpoints_match(host, *port, &target.host, target.port)
                            && crate::registry::cached_collector_reusable(host, &target.host)
                    })
                    .or_else(|| {
                        managed_lemonade_collectors.keys().find(|(host, port)| {
                            crate::registry::endpoints_match(host, *port, &target.host, target.port)
                                && crate::registry::cached_collector_reusable(host, &target.host)
                        })
                    })
                    .cloned()
                    .unwrap_or_else(|| (target.host.clone(), target.port));
                groups.entry(endpoint).or_default().push(LemonadeRow {
                    id: target.service_id.clone(),
                    model_names: vec![target.model_ref.clone(), target.canonical_model_id.clone()],
                });
            }
            managed_lemonade_collectors.retain(|endpoint, _| groups.contains_key(endpoint));
            if scrape_now {
                for ((host, port), rows) in groups {
                    let collector = managed_lemonade_collectors
                        .entry((host.clone(), port))
                        .or_insert_with(|| {
                            LemonadeCollector::new(
                                crate::registry::scrape_host(&host),
                                port,
                                Duration::from_millis(1500),
                            )
                        })
                        .clone();
                    endpoints.push(LemonadeEndpoint {
                        host,
                        port,
                        collector,
                        rows,
                        external: false,
                        discover_external: false,
                    });
                }
            }
            let results = parallel_scrape(endpoints, move |endpoint| async move {
                // A managed row can be a server without the Lemonade protocol
                // (a direct llama-server). It stays listed with throughput
                // unavailable and no standing warning; real failures still warn.
                let health = if endpoint.external {
                    endpoint.collector.fetch_health().await?
                } else if let Some(health) = endpoint.collector.fetch_health_if_router().await? {
                    health
                } else {
                    return Ok((None, endpoint.rows.clone(), None, None));
                };
                let discovered = endpoint.discover_external.then(|| {
                    rocm_dash_collectors::lemonade::services_from_health(
                        &endpoint.host,
                        endpoint.port,
                        &health,
                    )
                });
                let rows = discovered.as_ref().map_or_else(
                    || endpoint.rows.clone(),
                    |services| {
                        services
                            .iter()
                            .map(|service| LemonadeRow {
                                id: service.container_id.clone(),
                                model_names: vec![service.model_name.clone()],
                            })
                            .collect()
                    },
                );
                let attribution = health.stats_model();
                let mut matching = rows.iter().filter(|row| {
                    attribution
                        .as_ref()
                        .is_some_and(|model| row.model_names.iter().any(|name| name == model))
                });
                let attributed_id = matching
                    .next()
                    .filter(|_| matching.next().is_none())
                    .map(|row| row.id.clone());
                let stats = if scrape_now && attributed_id.is_some() {
                    Some(endpoint.collector.fetch_stats().await)
                } else {
                    None
                };
                Ok::<_, rocm_dash_core::traits::CollectorError>((
                    discovered,
                    rows,
                    attributed_id,
                    stats,
                ))
            })
            .await;
            for (endpoint, result) in results {
                let (discovered, rows, attributed_id, stats) = match result {
                    Ok(result) => {
                        if endpoint.discover_external {
                            lemonade_discovery_warn = None;
                        }
                        result
                    }
                    Err(error) => {
                        // A failed external discovery still removes rows immediately,
                        // but reports why they disappeared. A never-seen endpoint
                        // with nothing listening stays quiet; one that answers with
                        // an HTTP or body error warns even on first contact.
                        // Managed failures retain only the existing hold.
                        if endpoint.discover_external {
                            let absent = endpoint.rows.is_empty()
                                && rocm_dash_collectors::lemonade::is_connect_error(&error);
                            lemonade_discovery_warn =
                                (!absent).then(|| endpoint.warning("health", &error));
                            (Some(Vec::new()), Vec::new(), None, None)
                        } else {
                            scrape_warns.push(endpoint.warning("health", &error));
                            continue;
                        }
                    }
                };
                if let Some(services) = discovered {
                    Reconcile {
                        state: &mut runner.state,
                        telemetry: &mut telemetry,
                        tx: &tx,
                        persist: persist.as_ref(),
                    }
                    .reconcile_lemonade(&services, &mut discovery.lemonade_ids);
                }
                // Healthy contradictory/ambiguous evidence invalidates prior
                // attribution immediately, even off the stats cadence.
                for row in &rows {
                    if Some(&row.id) != attributed_id.as_ref() {
                        telemetry.gen_trackers.remove(&row.id);
                    }
                }
                if let Some(stats) = stats {
                    match stats {
                        Ok(sample) => {
                            if let Some(rate) = sample.gen_tps {
                                telemetry
                                    .gen_trackers
                                    .entry(attributed_id.expect("stats require attribution"))
                                    .or_insert_with(|| {
                                        GenerationObservationTracker::new(opts.instance_tick)
                                    })
                                    .observe_direct(rate, cycle_at);
                            }
                        }
                        Err(error) => scrape_warns.push(endpoint.warning("stats", &error)),
                    }
                }
            }
        }

        // Per-instance vLLM metric scrape, parallel, on its own cadence.
        // Lemonade rows are scraped via their endpoint collectors above, so
        // exclude them from the vLLM Prometheus targets. Managed non-vLLM services
        // are excluded too (their engine uses a different parser).
        if let Some(prom) = vllm.as_ref()
            && !runner.state.instances.is_empty()
            && tick_count.is_multiple_of(instance_ticks)
        {
            let targets = vllm_scrape_targets(
                &runner.state.instances,
                &discovery.lemonade_ids,
                &discovery.managed_non_vllm,
            );
            let prom = prom.clone();
            let results = parallel_scrape(targets, move |(id, port)| {
                let prom = prom.clone();
                async move {
                    let svc = DiscoveredService {
                        container_id: id.clone(),
                        port: Some(port),
                        ..Default::default()
                    };
                    prom.fetch_async(&svc).await
                }
            })
            .await;
            let mut fail_count: usize = 0;
            let mut last_err: Option<String> = None;
            for ((id, _port), fetch) in results {
                match fetch {
                    Ok(sample) => {
                        // EAI-7960: drive the per-instance tracker with the
                        // cumulative counter when present; omitted counter leaves
                        // the tracker (and its held value) untouched.
                        if let Some(cur) = sample.gen_tokens_total {
                            telemetry
                                .gen_trackers
                                .entry(id.clone())
                                .or_insert_with(|| {
                                    GenerationObservationTracker::new(opts.instance_tick)
                                })
                                .observe_counter(cur, cycle_at);
                        }
                        // gen_tps is written to the instance in the snapshot
                        // assembly below (tracker.snapshot per cycle_at).
                        //
                        // Windowed avg latency (ms) from the cumulative TTFT/TPOT
                        // histograms. Only a window with new requests produces a
                        // value; the tracker holds it, then expires it, in the
                        // snapshot assembly below.
                        observe_latency(
                            &mut telemetry.prev_ttft,
                            &id,
                            opts.instance_tick,
                            sample.ttft_sum_s,
                            sample.ttft_count,
                            cycle_at,
                        );
                        observe_latency(
                            &mut telemetry.prev_tpot,
                            &id,
                            opts.instance_tick,
                            sample.tpot_sum_s,
                            sample.tpot_count,
                            cycle_at,
                        );
                        if let Some(mut inst) = runner.state.instances.get(&id).cloned() {
                            inst.kv_cache_usage_pct = sample.kv_cache_usage_pct;
                            inst.running_reqs = sample.running_reqs;
                            inst.waiting_reqs = sample.waiting_reqs;
                            // Docker-discovered instances have no
                            // `ManagedServiceRecord` to promote their status, so
                            // they start life as `Starting`. A successful vLLM
                            // Prometheus scrape is the first hard evidence the
                            // instance is actually serving requests, so promote
                            // it to `Ready` here.
                            if matches!(inst.status, InstanceStatus::Starting { .. }) {
                                inst.status = InstanceStatus::Ready;
                            }
                            runner.state.apply(StateEvent::InstanceUpserted(inst));
                        }
                    }
                    Err(e) => {
                        fail_count += 1;
                        let msg = format!("{e}");
                        trace!(id = %id, error = %msg, "vllm scrape failed");
                        last_err = Some(msg);
                        // EAI-7960: do NOT clear or remove gen_trackers on failure.
                        // The tracker holds the last-observed rate for the validity
                        // window (clamp(3×instance_tick, 6s, 30s)) and only clears
                        // it after expiry — satisfying the "held" contract.
                        //
                        // Drop the TTFT/TPOT baselines so recovery re-baselines
                        // from the next successful scrape; the trackers keep the
                        // last windowed value Held until its validity lapses.
                        rebaseline_latency(&mut telemetry.prev_ttft, &id);
                        rebaseline_latency(&mut telemetry.prev_tpot, &id);
                    }
                }
            }
            if fail_count > 0 {
                scrape_warns.push(match last_err {
                    Some(e) => format!("vllm scrape: {fail_count} failed (last: {e})"),
                    None => format!("vllm scrape: {fail_count} failed"),
                });
            }
        }

        // Per-process VRAM attribution: refresh the per-container map on the
        // instance cadence via one amd-smi `process` scrape, joining GPU-process
        // host PIDs to container ids through `/proc/<pid>/cgroup`. On Err/empty
        // the device-summed fallback still applies. `procs_nonempty` gates the
        // fallback warning so we only warn on a real (non-empty) scrape.
        let mut procs_nonempty = false;
        if let Some(g) = &gpu
            && !runner.state.instances.is_empty()
            && tick_count.is_multiple_of(instance_ticks)
        {
            match g.processes().await {
                Ok(procs) => {
                    procs_nonempty = !procs.is_empty();
                    // A container id when the process is in a docker/containerd
                    // cgroup, otherwise the managed service whose engine it
                    // descends from. Both key the same map the instances read.
                    per_container_used =
                        rocm_dash_core::vram::aggregate_process_vram(&procs, |pid| {
                            rocm_dash_collectors::cgroup::container_id_for_pid(pid).or_else(|| {
                                rocm_dash_collectors::ancestry::owner_of_pid(
                                    pid,
                                    &discovery.managed_roots,
                                )
                            })
                        });
                }
                Err(e) => {
                    // Keep the last-known map; device fallback covers the gap.
                    trace!(error = %e, "amd-smi process scrape failed");
                }
            }
        }

        let mut instances: Vec<Instance> = runner.state.instances.values().cloned().collect();
        // EAI-7960: apply per-instance tracker snapshots (gen_tps + observation
        // metadata) before computing tokens_per_watt so the efficiency field
        // always derives from the current cycle's valid/held/expired rate.
        for inst in &mut instances {
            let (gen_tps, meta) = telemetry
                .gen_trackers
                .get_mut(&inst.container_id)
                .map_or((None, None), |t| t.snapshot(cycle_at));
            inst.gen_tps = gen_tps;
            inst.gen_tps_observation = meta;
            (inst.ttft_ms, inst.ttft_observation) =
                snapshot_latency(&mut telemetry.prev_ttft, &inst.container_id, cycle_at);
            (inst.tpot_ms, inst.tpot_observation) =
                snapshot_latency(&mut telemetry.prev_tpot, &inst.container_id, cycle_at);
            inst.gen_tps_last_request = telemetry
                .gen_trackers
                .get(&inst.container_id)
                .is_some_and(GenerationObservationTracker::is_direct);
        }
        // Derive per-instance efficiency now that this tick's GPU power is known
        // and gen_tps reflects the tracker state for this cycle.
        // Count instances that have throughput + live GPUs but whose gpu_ids
        // don't line up with any amd-smi device_id — the silent-None failure
        // mode on real hardware, surfaced via the header ⚠ badge.
        let mut id_join_misses = 0usize;
        // Split each GPU's power among the instances sharing it, so two
        // instances on one board don't both claim all of it.
        let attributed =
            rocm_dash_core::efficiency::attribute_power_w(&instances, &gpus, &per_container_used);
        for (inst, power_w) in instances.iter_mut().zip(attributed) {
            inst.power_w = power_w;
            inst.tokens_per_watt =
                rocm_dash_core::efficiency::tokens_per_watt(inst.gen_tps, power_w);
            if inst.gen_tps.is_some()
                && !inst.gpu_ids.is_empty()
                && !gpus.is_empty()
                && !rocm_dash_core::efficiency::gpu_ids_overlap(&inst.gpu_ids, &gpus)
            {
                id_join_misses += 1;
            }
        }
        if id_join_misses > 0 {
            warnings.push(format!(
                "tokens_per_watt: gpu_ids matched no GPU for {id_join_misses} instance(s) \
                 (check HIP_VISIBLE_DEVICES vs amd-smi device_id)"
            ));
        }
        // Write tracker-derived gen_tps, observation metadata, and TPW back into
        // runner.state.instances so that the next discovery merge reads current
        // values (not stale pre-snapshot values from the last upsert).
        for inst in &instances {
            if let Some(entry) = runner.state.instances.get_mut(&inst.container_id) {
                entry.gen_tps = inst.gen_tps;
                entry.gen_tps_observation = inst.gen_tps_observation.clone();
                entry.ttft_ms = inst.ttft_ms;
                entry.ttft_observation = inst.ttft_observation.clone();
                entry.tpot_ms = inst.tpot_ms;
                entry.tpot_observation = inst.tpot_observation.clone();
                entry.gen_tps_last_request = inst.gen_tps_last_request;
                entry.tokens_per_watt = inst.tokens_per_watt;
                entry.power_w = inst.power_w;
            }
        }
        // Attribute per-instance VRAM (per-process where the cgroup join hit,
        // device-summed otherwise). Pure; uses this tick's `gpus` for totals.
        enrich_instance_vram(&mut instances, &gpus, &per_container_used);
        // Warn only when a real process scrape happened but an instance with
        // live GPUs still fell back to device-summed (cgroup join missed).
        if procs_nonempty {
            let vram_fallbacks = count_vram_join_misses(
                &instances,
                &gpus,
                &per_container_used,
                &discovery.managed_roots,
                &discovery.managed_non_vllm,
            );
            if vram_fallbacks > 0 {
                scrape_warns.push(format!(
                    "vram attribution: {vram_fallbacks} instance(s) fell back to device-summed \
                     VRAM (no per-process cgroup or ancestry match; check /proc access)"
                ));
            }
        }
        warnings.extend(docker_warn.iter().cloned());
        warnings.extend(registry_warn.iter().cloned());
        warnings.extend(lemonade_discovery_warn.iter().cloned());
        warnings.extend(scrape_warns.iter().cloned());
        let snap = Snapshot {
            timestamp: cycle_at,
            host: host.tick(),
            gpus,
            gpu_system_info: gpu_system_info.clone(),
            instances,
            warnings,
        };
        runner.state.apply(StateEvent::Tick(snap.clone()));
        if let Ok(mut r) = ring.lock() {
            r.push(snap.clone());
        }
        broadcast_and_persist(&tx, persist.as_ref(), Event::Snapshot(snap));
        trace!(tick = tick_count, "snapshot broadcast");

        // Drain any new benchmark rows that landed since the last tick.
        if let Some(b) = bench.as_mut() {
            match b.drain() {
                Ok(rows) if !rows.is_empty() => {
                    info!(count = rows.len(), "bench rows broadcast");
                    runner.state.apply(StateEvent::BenchmarkRows(rows.clone()));
                    if let Ok(mut br) = bench_ring.lock() {
                        for row in &rows {
                            br.push(row.clone());
                        }
                    }
                    broadcast_and_persist(
                        &tx,
                        persist.as_ref(),
                        Event::BenchmarkRowsAppended { rows },
                    );
                }
                Ok(_) => {}
                Err(e) => warn!(error = %e, "bench tailer drain failed"),
            }
        }
    }
}

/// Instances that *could* have joined the per-process VRAM map but did not.
///
/// Two kinds of instance can hit the map: one whose id is a docker/containerd
/// id read from `/proc/<pid>/cgroup`, and a managed `rocm serve` service that
/// has recorded `roots` (its engine/supervisor PIDs) to match by ancestry. A
/// managed service with no roots cannot join and falls back to device-summed
/// VRAM by design, so it is not counted. One that is not yet `Ready`/`Running`
/// may still be loading its model with no GPU process to attribute, so it is
/// not counted either; a serving one always holds VRAM, so a missing entry
/// means its ancestry walk failed (unreadable `/proc`, a recycled root PID).
/// A non-vLLM managed service (e.g. Lemonade on a backend `amd-smi` does not
/// list) is not counted: it may hold GPU memory without ever appearing in the
/// process list, which is not an access problem.
fn count_vram_join_misses<S: std::hash::BuildHasher>(
    instances: &[Instance],
    gpus: &[GpuMetrics],
    per_container_used: &HashMap<String, u64, S>,
    roots: &[rocm_dash_collectors::ancestry::Root],
    non_vllm: &HashSet<String>,
) -> usize {
    instances
        .iter()
        .filter(|i| {
            let can_join = rocm_dash_collectors::cgroup::is_container_id(&i.container_id)
                || (matches!(i.status, InstanceStatus::Ready | InstanceStatus::Running)
                    && !non_vllm.contains(&i.container_id)
                    && roots.iter().any(|r| r.owner == i.container_id));
            can_join
                && !i.gpu_ids.is_empty()
                && rocm_dash_core::efficiency::gpu_ids_overlap(&i.gpu_ids, gpus)
                && !per_container_used.contains_key(&i.container_id)
        })
        .count()
}

/// Build an `Instance` from a `DiscoveredService` with no live KV/req sample yet.
///
/// Status is taken from `svc.status` — the caller sets it from whatever real
/// signal its discovery source has (a managed-service record's status, a
/// Lemonade health probe, or `Starting` for a Docker container with no status
/// source until its first successful vLLM Prometheus scrape). vLLM Prometheus
/// scraping fills the KV/req sample fields in a later collector tick.
pub(crate) fn instance_from_discovered(svc: &DiscoveredService) -> Instance {
    merge_instance(svc, &InstanceSample::default(), 0, 0)
}

/// Merge a rediscovered service's identity/metadata into a fresh Instance,
/// preserving live collector-owned fields from the existing state when the
/// identity is unchanged.
///
/// **Discovery metadata wins** (model name, GPU ids, port, status, launch args).
/// **Collector-owned fields survive** (KV/running/waiting, gen_tps, observation
/// metadata, tokens_per_watt, TTFT/TPOT). Passing `existing = None` — on a new
/// identity or after an endpoint change — starts the instance with all-None live
/// fields (correct baseline).
///
/// This is the sole point for merging discovery and collector state; call it
/// from every discovery source (Docker, managed registry, Lemonade) to keep
/// per-service telemetry visible across rediscovery without stale telemetry
/// leaking across identity boundaries.
pub(crate) fn merge_discovery_refresh(
    svc: &DiscoveredService,
    existing: Option<&Instance>,
) -> Instance {
    let mut inst = instance_from_discovered(svc);
    if let Some(ex) = existing {
        inst.kv_cache_usage_pct = ex.kv_cache_usage_pct;
        inst.running_reqs = ex.running_reqs;
        inst.waiting_reqs = ex.waiting_reqs;
        inst.gen_tps = ex.gen_tps;
        inst.gen_tps_observation = ex.gen_tps_observation.clone();
        inst.gen_tps_last_request = ex.gen_tps_last_request;
        inst.tokens_per_watt = ex.tokens_per_watt;
        inst.power_w = ex.power_w;
        inst.ttft_ms = ex.ttft_ms;
        inst.ttft_observation = ex.ttft_observation.clone();
        inst.tpot_ms = ex.tpot_ms;
        inst.tpot_observation = ex.tpot_observation.clone();
    }
    inst
}

/// Set `vram_used_mb`/`vram_total_mb` on each instance from the per-process
/// attribution map, falling back to device-summed VRAM over the instance's
/// GPUs when its container has no per-process entry. Pure — the runner does the
/// amd-smi + cgroup I/O and passes `per_container_used` in. `total` is always
/// device-summed over `gpu_ids`; an instance with empty `gpu_ids` and no map
/// entry (e.g. Lemonade) stays at `(0, 0)`.
fn enrich_instance_vram(
    instances: &mut [Instance],
    gpus: &[GpuMetrics],
    per_container_used: &HashMap<String, u64>,
) {
    for inst in instances {
        let (used, total) = rocm_dash_core::vram::resolve_instance_vram(
            &inst.container_id,
            &inst.gpu_ids,
            gpus,
            per_container_used,
        );
        inst.vram_used_mb = used;
        inst.vram_total_mb = total;
    }
}

/// Broadcast `ev` to subscribers and, if a session writer is wired, append
/// the same event to disk for `--replay`. Persistence is best-effort — a
/// write failure logs at warn level but does not interrupt the loop.
fn broadcast_and_persist(
    tx: &broadcast::Sender<Event>,
    persist: Option<&Arc<Mutex<SessionWriter>>>,
    ev: Event,
) {
    if let Some(w) = persist
        && let Ok(mut writer) = w.lock()
        && let Err(e) = writer.append(&ev)
    {
        warn!(error = %e, "session persist failed");
    }
    let _ = tx.send(ev);
}

/// How many `tick`s fit into `period`, rounded to the nearest, minimum 1.
fn ticks_per(period: Duration, tick: Duration) -> u64 {
    let n = (period.as_secs_f64() / tick.as_secs_f64()).round() as i64;
    n.max(1) as u64
}

/// Upper bound on the scrape interval accepted by [`observe_latency`].
/// Applies to TTFT/TPOT latency histograms only; generation-counter staleness
/// is managed by `GenerationObservationTracker` (see `rocm-dash-core`).
/// A gap longer than this (outage, forward wall-clock jump / NTP / VM resume)
/// makes the windowed delta meaningless, so that sample yields no value and
/// only re-baselines.
const MAX_RATE_INTERVAL_S: f64 = 60.0;

/// Per-instance latency state for one histogram: the previous cumulative
/// reading and the tracker that holds and then expires the derived average.
pub(crate) struct LatencyWindow {
    /// Previous `(sum_s, count, at)` reading; `None` until baselined.
    prev: Option<(f64, f64, DateTime<Utc>)>,
    /// Holds the last windowed average for the observation validity window.
    tracker: GenerationObservationTracker,
}

pub(crate) type LatencyWindows = HashMap<String, LatencyWindow>;

/// Feed one scrape of a cumulative latency histogram (`_sum` in seconds,
/// `_count`) into `id`'s window. A value is recorded only when the window saw
/// new requests: `(Δsum / Δcount) × 1000` ms. The first reading, a counter
/// reset, a stale interval, or a window with no new requests only moves the
/// baseline; the lifetime average is never reported, so an idle server's old
/// latency ages out instead of persisting. An absent histogram (`None`) leaves
/// the window untouched.
fn observe_latency(
    windows: &mut LatencyWindows,
    id: &str,
    instance_tick: Duration,
    sum_s: Option<f64>,
    count: Option<f64>,
    now: DateTime<Utc>,
) {
    let (Some(sum_s), Some(count)) = (sum_s, count) else {
        return;
    };
    let w = windows
        .entry(id.to_string())
        .or_insert_with(|| LatencyWindow {
            prev: None,
            tracker: GenerationObservationTracker::new(instance_tick),
        });
    if let Some((psum, pcount, pat)) = w.prev.replace((sum_s, count, now)) {
        let dcount = count - pcount;
        let dsum = sum_s - psum;
        let dt = (now - pat).num_milliseconds() as f64 / 1000.0;
        if dcount > 0.0 && dsum >= 0.0 && dt > 0.0 && dt <= MAX_RATE_INTERVAL_S {
            w.tracker.observe_direct(dsum / dcount * 1000.0, now);
        }
    }
}

/// Forget the previous reading after a failed scrape so recovery re-baselines,
/// while the tracker keeps the last value held until it expires.
fn rebaseline_latency(windows: &mut LatencyWindows, id: &str) {
    if let Some(w) = windows.get_mut(id) {
        w.prev = None;
    }
}

/// Current held/fresh/expired latency for `id` at `now`.
fn snapshot_latency(
    windows: &mut LatencyWindows,
    id: &str,
    now: DateTime<Utc>,
) -> (Option<f64>, Option<ObservationMetadata>) {
    windows
        .get_mut(id)
        .map_or((None, None), |w| w.tracker.snapshot(now))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// Pins the `None` binary arm specifically: a revert back to hardcoding
    /// `false` here (rather than threading `gpu_reachable` through, as the
    /// `Some` arm already does) would not be caught by any integration test,
    /// since exercising it end to end needs a real or PATH-resolved `amd-smi`
    /// binary named literally `amd-smi`.
    #[test]
    fn detect_plan_threads_gpu_reachable_through_the_none_binary_arm() {
        match amd_smi_detect_plan(None, false, true, PathBuf::from("/fake-root/dev/kfd")) {
            AmdSmiDetectPlan::Detect(binary, gpu_reachable, host_kfd_device) => {
                assert_eq!(binary, OsString::from("amd-smi"));
                assert!(gpu_reachable);
                assert_eq!(host_kfd_device, PathBuf::from("/fake-root/dev/kfd"));
            }
            AmdSmiDetectPlan::SkipDevicePreflight(_) => panic!("expected Detect"),
        }

        match amd_smi_detect_plan(None, false, false, PathBuf::from(KFD_DEVICE)) {
            AmdSmiDetectPlan::Detect(_, gpu_reachable, _) => assert!(!gpu_reachable),
            AmdSmiDetectPlan::SkipDevicePreflight(_) => panic!("expected Detect"),
        }
    }

    #[test]
    fn detect_plan_skip_preflight_takes_priority_over_an_explicit_binary() {
        match amd_smi_detect_plan(
            Some("fake-amd-smi".into()),
            true,
            false,
            PathBuf::from(KFD_DEVICE),
        ) {
            AmdSmiDetectPlan::SkipDevicePreflight(binary) => {
                assert_eq!(binary, OsString::from("fake-amd-smi"));
            }
            AmdSmiDetectPlan::Detect(..) => panic!("expected SkipDevicePreflight"),
        }
    }

    #[test]
    fn test_clock_advances_by_ticks_and_external_offset() {
        let dir = tempfile::tempdir().unwrap();
        let offset = dir.path().join("offset-secs");
        std::fs::write(&offset, "0").unwrap();
        let epoch = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
        let mut clock = TestClock::default();

        assert_eq!(
            clock.cycle_timestamp(epoch, Duration::from_secs(1), 3, Some(&offset)),
            epoch + chrono::TimeDelta::seconds(3)
        );

        std::fs::write(&offset, "7").unwrap();
        assert_eq!(
            clock.cycle_timestamp(epoch, Duration::from_secs(1), 4, Some(&offset)),
            epoch + chrono::TimeDelta::seconds(11)
        );
    }

    /// EAI-7960 regression (the `hold` directive's whole reason to exist): once
    /// held, cycles keep happening but logical time does not move, so an
    /// observation's age — and therefore its validity budget — is frozen no
    /// matter how long the scenario takes to reach its next assertion.
    #[test]
    fn held_test_clock_stops_advancing_with_ticks() {
        let dir = tempfile::tempdir().unwrap();
        let offset = dir.path().join("offset-secs");
        std::fs::write(&offset, "0").unwrap();
        let epoch = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
        let mut clock = TestClock::default();

        assert_eq!(
            clock.cycle_timestamp(epoch, Duration::from_secs(1), 5, Some(&offset)),
            epoch + chrono::TimeDelta::seconds(5)
        );

        std::fs::write(&offset, "hold").unwrap();
        let held = clock.cycle_timestamp(epoch, Duration::from_secs(1), 6, Some(&offset));
        assert_eq!(held, epoch + chrono::TimeDelta::seconds(6));
        // Hundreds of cycles later, still the same instant.
        for tick_count in 7..300 {
            assert_eq!(
                clock.cycle_timestamp(epoch, Duration::from_secs(1), tick_count, Some(&offset)),
                held,
                "a held clock must not advance with cycles"
            );
        }

        // An explicit advance steps from the hold point — not from "now" — so
        // the step size a scenario writes is exactly the age it adds.
        std::fs::write(&offset, "hold 7").unwrap();
        assert_eq!(
            clock.cycle_timestamp(epoch, Duration::from_secs(1), 400, Some(&offset)),
            held + chrono::TimeDelta::seconds(7)
        );

        // Releasing the hold resumes tick-driven time from the current cycle.
        std::fs::write(&offset, "0").unwrap();
        assert_eq!(
            clock.cycle_timestamp(epoch, Duration::from_secs(1), 401, Some(&offset)),
            epoch + chrono::TimeDelta::seconds(401)
        );
    }

    /// A scenario rewriting the file non-atomically can expose an empty read;
    /// that must never resume a held clock (it would silently re-arm the very
    /// race `hold` removes).
    #[test]
    fn unreadable_test_clock_file_keeps_the_directive_in_force() {
        let dir = tempfile::tempdir().unwrap();
        let offset = dir.path().join("offset-secs");
        std::fs::write(&offset, "hold").unwrap();
        let epoch = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
        let mut clock = TestClock::default();

        let held = clock.cycle_timestamp(epoch, Duration::from_secs(1), 9, Some(&offset));

        std::fs::write(&offset, "").unwrap();
        assert_eq!(
            clock.cycle_timestamp(epoch, Duration::from_secs(1), 10, Some(&offset)),
            held
        );
        std::fs::remove_file(&offset).unwrap();
        assert_eq!(
            clock.cycle_timestamp(epoch, Duration::from_secs(1), 11, Some(&offset)),
            held
        );
    }

    #[test]
    fn test_clock_directive_grammar() {
        assert_eq!(
            parse_test_clock_directive("0"),
            Some(TestClockDirective::FreeRunning(0))
        );
        assert_eq!(
            parse_test_clock_directive("-3"),
            Some(TestClockDirective::FreeRunning(-3))
        );
        assert_eq!(
            parse_test_clock_directive("hold"),
            Some(TestClockDirective::Held(0))
        );
        assert_eq!(
            parse_test_clock_directive("hold 7"),
            Some(TestClockDirective::Held(7))
        );
        assert_eq!(parse_test_clock_directive(""), None);
        assert_eq!(parse_test_clock_directive("later"), None);
        assert_eq!(parse_test_clock_directive("hold soon"), None);
        assert_eq!(
            TestClockDirective::default(),
            TestClockDirective::FreeRunning(0)
        );
    }

    pub(super) fn at(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).unwrap()
    }

    use rocm_dash_core::metrics::ObservationFreshness;

    const TICK: Duration = Duration::from_secs(2);

    pub(super) fn baseline(sum_s: f64, count: f64) -> LatencyWindow {
        LatencyWindow {
            prev: Some((sum_s, count, at(10))),
            tracker: GenerationObservationTracker::new(TICK),
        }
    }

    fn snap_at(
        w: &mut LatencyWindows,
        id: &str,
        secs: i64,
    ) -> (Option<f64>, Option<ObservationFreshness>) {
        let (v, m) = snapshot_latency(w, id, at(secs));
        (v, m.map(|m| m.freshness))
    }

    fn inst_on(id: &str, gpu: &str) -> Instance {
        Instance {
            container_id: id.into(),
            gpu_ids: vec![gpu.into()],
            ..Instance::default()
        }
    }

    #[test]
    fn only_instances_that_can_join_count_as_vram_join_misses() {
        use rocm_dash_collectors::ancestry::Root;
        let docker_id = "ab".repeat(32);
        let rooted = "vllm-qwen-qwen2-5-0-5b-instru-1793361705247";
        let unrooted = "vllm-other-1793361705248";
        let gpus = [gpu("gpu-0", 1000, 8000)];
        let serving = |id: &str| Instance {
            status: InstanceStatus::Ready,
            ..inst_on(id, "0")
        };
        let insts = [serving(rooted), serving(unrooted), inst_on(&docker_id, "0")];
        let roots = [Root {
            pid: 10,
            start_ticks: None,
            owner: rooted.into(),
        }];
        // The docker instance and the rooted managed service could both have
        // joined and did not; the managed service with no roots never could.
        let empty: HashMap<String, u64> = HashMap::new();
        assert_eq!(
            count_vram_join_misses(&insts, &gpus, &empty, &roots, &HashSet::new()),
            2
        );
        // With no roots recorded, only the docker instance is a miss.
        assert_eq!(
            count_vram_join_misses(&insts, &gpus, &empty, &[], &HashSet::new()),
            1
        );
        // Once both joined, nothing is missed.
        let joined: HashMap<String, u64> = [(docker_id, 512), (rooted.to_string(), 256)].into();
        assert_eq!(
            count_vram_join_misses(&insts, &gpus, &joined, &roots, &HashSet::new()),
            0
        );
    }

    #[test]
    fn a_rooted_non_vllm_service_is_not_a_vram_join_miss() {
        use rocm_dash_collectors::ancestry::Root;
        let id = "lemonade-gguf-1793361705247";
        let gpus = [gpu("gpu-0", 1000, 8000)];
        let serving = Instance {
            status: InstanceStatus::Ready,
            ..inst_on(id, "0")
        };
        let roots = [Root {
            pid: 10,
            start_ticks: None,
            owner: id.into(),
        }];
        let empty: HashMap<String, u64> = HashMap::new();
        // As a vLLM service it would be a miss; marked non-vLLM it is not, since
        // its backend may hold GPU memory without `amd-smi` listing the process.
        assert_eq!(
            count_vram_join_misses(
                std::slice::from_ref(&serving),
                &gpus,
                &empty,
                &roots,
                &HashSet::new()
            ),
            1
        );
        let non_vllm: HashSet<String> = [id.to_string()].into();
        assert_eq!(
            count_vram_join_misses(&[serving], &gpus, &empty, &roots, &non_vllm),
            0
        );
    }

    #[test]
    fn a_rooted_service_still_loading_is_not_a_vram_join_miss() {
        use rocm_dash_collectors::ancestry::Root;
        let id = "vllm-qwen-1793361705247";
        let gpus = [gpu("gpu-0", 1000, 8000)];
        let loading = Instance {
            status: InstanceStatus::Starting { phase: None },
            ..inst_on(id, "0")
        };
        let roots = [Root {
            pid: 10,
            start_ticks: None,
            owner: id.into(),
        }];
        let empty: HashMap<String, u64> = HashMap::new();
        assert_eq!(
            count_vram_join_misses(&[loading], &gpus, &empty, &roots, &HashSet::new()),
            0
        );
    }

    #[test]
    fn latency_first_scrape_reports_nothing_then_windows() {
        let mut w = LatencyWindows::new();
        // First scrape only baselines: the lifetime average is not reported.
        observe_latency(&mut w, "a", TICK, Some(2.0), Some(4.0), at(10));
        assert_eq!(snap_at(&mut w, "a", 10), (None, None));
        // Second scrape: Δsum=1.0s over Δcount=10 → 100 ms, fresh.
        observe_latency(&mut w, "a", TICK, Some(3.0), Some(14.0), at(12));
        assert_eq!(
            snap_at(&mut w, "a", 12),
            (Some(100.0), Some(ObservationFreshness::Fresh))
        );
    }

    #[test]
    fn latency_is_held_without_new_requests_then_expires() {
        let mut w = LatencyWindows::new();
        observe_latency(&mut w, "a", TICK, Some(2.0), Some(4.0), at(10));
        observe_latency(&mut w, "a", TICK, Some(3.0), Some(14.0), at(12));
        // Idle scrapes: counters unchanged → no new value, previous is held.
        observe_latency(&mut w, "a", TICK, Some(3.0), Some(14.0), at(14));
        assert_eq!(
            snap_at(&mut w, "a", 14),
            (Some(100.0), Some(ObservationFreshness::Held))
        );
        // Validity for a 2 s tick is the 6 s floor: still held at +6 s...
        assert_eq!(
            snap_at(&mut w, "a", 18),
            (Some(100.0), Some(ObservationFreshness::Held))
        );
        // ...and gone just past it. The stale lifetime average never returns.
        observe_latency(&mut w, "a", TICK, Some(3.0), Some(14.0), at(19));
        assert_eq!(snap_at(&mut w, "a", 19), (None, None));
    }

    #[test]
    fn latency_absent_empty_reset_and_stale_interval_report_nothing() {
        let mut w = LatencyWindows::new();
        // Absent histogram / zero count never fabricate a number.
        observe_latency(&mut w, "b", TICK, None, None, at(12));
        assert_eq!(snap_at(&mut w, "b", 12), (None, None));
        observe_latency(&mut w, "c", TICK, Some(0.0), Some(0.0), at(12));
        observe_latency(&mut w, "c", TICK, Some(0.0), Some(0.0), at(14));
        assert_eq!(snap_at(&mut w, "c", 14), (None, None));
        // A counter reset (sum drops) only re-baselines.
        observe_latency(&mut w, "d", TICK, Some(10.0), Some(100.0), at(10));
        observe_latency(&mut w, "d", TICK, Some(0.2), Some(2.0), at(12));
        assert_eq!(snap_at(&mut w, "d", 12), (None, None));
        // A gap beyond MAX_RATE_INTERVAL_S is not a usable window.
        observe_latency(&mut w, "e", TICK, Some(1.0), Some(10.0), at(0));
        observe_latency(&mut w, "e", TICK, Some(2.0), Some(20.0), at(100));
        assert_eq!(snap_at(&mut w, "e", 100), (None, None));
    }

    #[test]
    fn failed_scrape_rebaselines_but_keeps_value_held() {
        let mut w = LatencyWindows::new();
        observe_latency(&mut w, "a", TICK, Some(2.0), Some(4.0), at(10));
        observe_latency(&mut w, "a", TICK, Some(3.0), Some(14.0), at(12));
        rebaseline_latency(&mut w, "a");
        assert_eq!(
            snap_at(&mut w, "a", 14),
            (Some(100.0), Some(ObservationFreshness::Held))
        );
        // Recovery re-baselines: the first post-failure scrape yields nothing new.
        observe_latency(&mut w, "a", TICK, Some(9.0), Some(50.0), at(15));
        assert_eq!(
            snap_at(&mut w, "a", 15),
            (Some(100.0), Some(ObservationFreshness::Held))
        );
    }

    fn gpu(device_id: &str, used: u64, total: u64) -> GpuMetrics {
        GpuMetrics {
            device_id: device_id.into(),
            vram_used_mb: used,
            vram_total_mb: total,
            ..GpuMetrics::default()
        }
    }

    fn inst(container_id: &str, gpu_ids: &[&str]) -> Instance {
        Instance {
            container_id: container_id.into(),
            gpu_ids: gpu_ids
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            ..Instance::default()
        }
    }

    #[test]
    fn enrich_uses_per_process_used_with_device_total() {
        let gpus = [gpu("gpu-0", 1000, 8000), gpu("gpu-1", 2000, 8000)];
        let mut per = HashMap::new();
        per.insert("abc".to_string(), 4242);
        let mut instances = [inst("abc", &["0", "1"])];
        enrich_instance_vram(&mut instances, &gpus, &per);
        // used from the per-process map, total device-summed over gpu 0+1.
        assert_eq!(instances[0].vram_used_mb, 4242);
        assert_eq!(instances[0].vram_total_mb, 16000);
    }

    #[test]
    fn enrich_falls_back_to_device_summed_when_unmatched() {
        let gpus = [gpu("gpu-0", 1000, 8000), gpu("gpu-1", 2000, 8000)];
        let per = HashMap::new(); // container not present
        let mut instances = [inst("missing", &["0", "1"])];
        enrich_instance_vram(&mut instances, &gpus, &per);
        assert_eq!(instances[0].vram_used_mb, 3000); // device-summed used
        assert_eq!(instances[0].vram_total_mb, 16000);
    }

    #[test]
    fn enrich_leaves_lemonade_style_instance_at_zero() {
        // Empty gpu_ids + synthetic id not in map → (0, 0), no panic.
        let gpus = [gpu("gpu-0", 1000, 8000)];
        let per = HashMap::new();
        let mut instances = [inst("lemonade-synthetic", &[])];
        enrich_instance_vram(&mut instances, &gpus, &per);
        assert_eq!(instances[0].vram_used_mb, 0);
        assert_eq!(instances[0].vram_total_mb, 0);
    }

    /// EAI-7359 regression: vLLM Prometheus scraping must be constructed
    /// regardless of `enable_docker`. Before the fix this was
    /// `enable_docker && !disable_vllm_metrics`, which permanently killed the
    /// scraper for the embedded daemon (always launched with
    /// `enable_docker = false`).
    ///
    /// The `disable_vllm_metrics = true` cases below exercise the internal
    /// gate directly; it is not currently user-reachable (no CLI flag / config
    /// field sets it), so this asserts the gate's logic, not a shipped
    /// off-switch.
    #[test]
    fn vllm_metrics_enabled_is_independent_of_docker() {
        let mut opts = RunnerOptions {
            enable_docker: false,
            disable_vllm_metrics: false,
            ..RunnerOptions::default()
        };
        assert!(
            vllm_metrics_enabled(&opts),
            "no-Docker case must still scrape"
        );

        opts.enable_docker = true;
        assert!(
            vllm_metrics_enabled(&opts),
            "Docker enabled must still scrape"
        );

        // Internal gate (not user-reachable today): flipping it off must win
        // over any Docker state.
        opts.disable_vllm_metrics = true;
        assert!(
            !vllm_metrics_enabled(&opts),
            "internal disable gate must always win"
        );

        opts.enable_docker = false;
        assert!(
            !vllm_metrics_enabled(&opts),
            "internal disable gate must always win regardless of docker"
        );
    }

    #[test]
    fn ticks_per_rounds_and_floors_to_one() {
        assert_eq!(ticks_per(Duration::from_secs(5), Duration::from_secs(1)), 5);
        assert_eq!(
            ticks_per(Duration::from_millis(900), Duration::from_secs(1)),
            1
        );
        assert_eq!(
            ticks_per(Duration::from_millis(0), Duration::from_secs(1)),
            1
        );
        assert_eq!(
            ticks_per(Duration::from_secs(30), Duration::from_millis(250)),
            120
        );
    }
}
