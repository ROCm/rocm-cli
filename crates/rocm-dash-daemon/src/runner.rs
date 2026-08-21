// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Owner task: drives collectors on tick cadences and broadcasts Snapshots.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::bench_ring::BenchRing;
use crate::persist::SessionWriter;
use crate::snapshot_ring::SnapshotRing;

use chrono::{DateTime, Utc};
use rocm_dash_collectors::amd_smi::AmdSmiCollector;
use rocm_dash_collectors::bench_tail::CsvBenchTailer;
use rocm_dash_collectors::docker::DockerDiscovery;
use rocm_dash_collectors::host::HostCollector;
use rocm_dash_collectors::lemonade::LemonadeCollector;
use rocm_dash_collectors::parallel::parallel_scrape;
use rocm_dash_collectors::vllm_prom::VllmPrometheusCollector;
use rocm_dash_core::metrics::{GpuMetrics, GpuSystemInfo, Instance, InstanceStatus, Snapshot};
use rocm_dash_core::observation::GenerationObservationTracker;
use rocm_dash_core::protocol::Event;
use rocm_dash_core::state::{State, StateEvent};
use rocm_dash_core::traits::{BenchTailer, DiscoveredService, InstanceSample, merge_instance};
use tokio::sync::broadcast;
use tokio::time::{MissedTickBehavior, interval};
use tracing::{info, trace, warn};

/// Refresh GPU system info (versions, partition modes) every N seconds.
const SYSINFO_REFRESH_SECS: u64 = 30;

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
    /// Opt-in probe-based Lemonade discovery: when enabled, probe a local
    /// Lemonade endpoint each discovery tick and surface it as an Instance.
    /// Off by default so hosts with no Lemonade server never poll a dead port.
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
    /// without Docker discovery. Off by default.
    pub services_dir: Option<PathBuf>,
    /// Path or command name for the `amd-smi` binary. The managed ROCm SDK
    /// ships it inside the runtime wheel's bin directory rather than on `PATH`,
    /// so the caller resolves it (via `rocm_core::resolve_amd_smi_binary`) and
    /// passes it here. `None` falls back to looking up `amd-smi` on `PATH`.
    pub amd_smi_binary: Option<OsString>,
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

    // Opt-in Lemonade discovery (probe-based; off unless a Lemonade endpoint is
    // configured). Distinct from Docker/vLLM discovery — a local server, not a
    // container — so it is tracked separately from `known_instances`.
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
    // The id of the currently-live Lemonade instance, if any.
    let mut lemonade_id: Option<String> = None;

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
    let mut known_instances: HashSet<String> = HashSet::new();
    // Managed-service registry: ids surfaced from the rocm-cli
    // `serve` registry, and the subset whose engine is NOT vLLM (excluded from
    // the vLLM Prometheus scrape so they aren't mis-parsed).
    let mut known_services: HashSet<String> = HashSet::new();
    let mut managed_non_vllm: HashSet<String> = HashSet::new();
    // Per-instance generation-throughput tracker. Keyed by container_id.
    // Constructed lazily from opts.instance_tick; survives scrape failures so
    // the last-observed rate is held for the validity window before clearing.
    let mut gen_trackers: HashMap<String, GenerationObservationTracker> = HashMap::new();
    // Previous TTFT / TPOT histogram readings (sum_s, count, at) per instance,
    // for the windowed avg-ms derivation (mirrors `prev_gen_tokens`).
    let mut prev_ttft: HashMap<String, (f64, f64, DateTime<Utc>)> = HashMap::new();
    let mut prev_tpot: HashMap<String, (f64, f64, DateTime<Utc>)> = HashMap::new();
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
    let amd_smi_binary = opts.amd_smi_binary.clone();
    let (gpu_init_tx, mut gpu_init_rx) =
        tokio::sync::oneshot::channel::<(Option<AmdSmiCollector>, Option<GpuSystemInfo>)>();
    tokio::spawn(async move {
        let gpu = match amd_smi_binary {
            Some(binary) => AmdSmiCollector::detect_with_binary(binary).await,
            None => AmdSmiCollector::detect().await,
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
    // Discovery/scrape warnings are produced on their own (slower) cadences, so
    // holding them only for the tick that produced them makes the header ⚠ badge
    // blink on and off. Keep the last result until that cadence runs again.
    let mut docker_warn: Option<String> = None;
    let mut scrape_warns: Vec<String> = Vec::new();

    loop {
        ticker.tick().await;
        tick_count += 1;
        // Single wall-clock anchor for this loop iteration. All counter/direct
        // observations and the assembled Snapshot timestamp share this instant so
        // every instance refreshed in this cycle serialises Fresh deterministically.
        let cycle_at = Utc::now();

        // Adopt the background amd-smi detection result the moment it lands,
        // without ever blocking the loop while it is still in flight. Until then
        // `gpu` stays `None` (GPU panels read "unavailable") but discovery and
        // snapshots run every tick, so serving instances appear promptly.
        if !gpu_init_done && let Ok((detected, info)) = gpu_init_rx.try_recv() {
            gpu_init_done = true;
            if detected.is_some() {
                if let Some(i) = &info {
                    info!(
                        gpus = i.physical_gpu_count,
                        model = %i.gpu_model,
                        rocm = i.rocm_version.as_deref().unwrap_or("?"),
                        "amd-smi detected"
                    );
                }
            } else {
                warn!(
                    "amd-smi not available (no /dev/kfd or `amd-smi version` failed); GPU disabled"
                );
            }
            gpu = detected;
            gpu_system_info = info;
            last_sysinfo_refresh = tick_count;
        }

        let mut warnings = Vec::new();
        let gpus = if let Some(g) = &gpu {
            match g.metrics().await {
                Ok(v) => v,
                Err(e) => {
                    warnings.push(format!("amd-smi metric: {e}"));
                    Vec::new()
                }
            }
        } else {
            warnings.push("amd-smi unavailable (no /dev/kfd or binary missing)".into());
            Vec::new()
        };

        if gpu.is_some()
            && tick_count.saturating_sub(last_sysinfo_refresh) >= sysinfo_refresh_ticks
            && let Some(g) = &gpu
        {
            gpu_system_info = Some(g.system_info().await);
            last_sysinfo_refresh = tick_count;
        }

        // Service discovery — every DISCOVERY_TICKS ticks, diff vs known set,
        // emit Discovered/Gone events, and update reducer state.
        if let Some(d) = docker.as_ref()
            && (tick_count == 1 || tick_count.is_multiple_of(discovery_ticks))
        {
            match d.discover_async().await {
                Ok(svcs) => {
                    let seen: HashSet<String> =
                        svcs.iter().map(|s| s.container_id.clone()).collect();
                    for svc in &svcs {
                        let existing = runner.state.instances.get(&svc.container_id).cloned();
                        // Endpoint change: same container-id, new port → new logical service.
                        // Invalidate the rate baseline and latency windows so the new
                        // scrape address starts fresh.
                        let same_endpoint = existing.as_ref().is_none_or(|ex| ex.port == svc.port);
                        if !same_endpoint {
                            invalidate_on_endpoint_change(
                                &svc.container_id,
                                &mut gen_trackers,
                                &mut prev_ttft,
                                &mut prev_tpot,
                            );
                        }
                        let inst = merge_discovery_refresh(
                            svc,
                            if same_endpoint {
                                existing.as_ref()
                            } else {
                                None
                            },
                        );
                        runner
                            .state
                            .apply(StateEvent::InstanceUpserted(inst.clone()));
                        if !known_instances.contains(&svc.container_id) {
                            info!(
                                id = %svc.container_id,
                                name = %svc.container_name,
                                model = %svc.model_name,
                                "instance discovered"
                            );
                            // InstanceDiscovered is a discovery-metadata event; broadcast
                            // a clean instance (no prior collector-owned telemetry).
                            broadcast_and_persist(
                                &tx,
                                persist.as_ref(),
                                Event::InstanceDiscovered(instance_from_discovered(svc)),
                            );
                        }
                    }
                    for gone in known_instances.difference(&seen) {
                        info!(id = %gone, "instance gone");
                        purge_telemetry_state(
                            gone,
                            &mut gen_trackers,
                            &mut prev_ttft,
                            &mut prev_tpot,
                        );
                        runner
                            .state
                            .apply(StateEvent::InstanceRemoved(gone.clone()));
                        broadcast_and_persist(
                            &tx,
                            persist.as_ref(),
                            Event::InstanceGone {
                                container_id: gone.clone(),
                            },
                        );
                    }
                    known_instances = seen;
                    docker_warn = None;
                }
                Err(e) => docker_warn = Some(format!("docker discover: {e}")),
            }
        }

        // Lemonade discovery — probe the endpoint on the discovery cadence; add a
        // Lemonade Instance when reachable, emit Gone when it disappears, and stay
        // a clean no-op (no warning/panic) when no endpoint is configured.
        //
        // The Lemonade `container_id` encodes the host and port
        // (`<host>:<port>`), so a port or host change produces a new id string
        // and is treated as a full identity change (old gone, new discovered).
        if let Some(l) = lemonade.as_ref()
            && (tick_count == 1 || tick_count.is_multiple_of(discovery_ticks))
        {
            match l.discover().await {
                Some(svc) => {
                    let is_new_id = lemonade_id.as_deref() != Some(svc.container_id.as_str());
                    if is_new_id {
                        // New identity: purge all telemetry state for the old id so
                        // stale rate baseline, TTFT, and TPOT do not bleed across.
                        if let Some(old_id) = lemonade_id.as_deref() {
                            purge_telemetry_state(
                                old_id,
                                &mut gen_trackers,
                                &mut prev_ttft,
                                &mut prev_tpot,
                            );
                        }
                    }
                    let existing = runner.state.instances.get(&svc.container_id).cloned();
                    let inst = merge_discovery_refresh(
                        &svc,
                        if is_new_id { None } else { existing.as_ref() },
                    );
                    runner
                        .state
                        .apply(StateEvent::InstanceUpserted(inst.clone()));
                    if is_new_id {
                        info!(
                            id = %svc.container_id,
                            model = %svc.model_name,
                            "lemonade instance discovered"
                        );
                        // InstanceDiscovered is a discovery-metadata event; broadcast
                        // a clean instance (no prior collector-owned telemetry).
                        broadcast_and_persist(
                            &tx,
                            persist.as_ref(),
                            Event::InstanceDiscovered(instance_from_discovered(&svc)),
                        );
                        lemonade_id = Some(svc.container_id.clone());
                    }
                }
                None => {
                    if let Some(id) = lemonade_id.take() {
                        info!(id = %id, "lemonade instance gone");
                        purge_telemetry_state(
                            &id,
                            &mut gen_trackers,
                            &mut prev_ttft,
                            &mut prev_tpot,
                        );
                        runner.state.apply(StateEvent::InstanceRemoved(id.clone()));
                        broadcast_and_persist(
                            &tx,
                            persist.as_ref(),
                            Event::InstanceGone { container_id: id },
                        );
                    }
                }
            }
        }

        // Managed-service registry discovery — read the rocm-cli
        // `serve` records and surface live ones as scrape targets. The port is
        // the registry's authority; non-vLLM engines are tracked so the vLLM
        // scrape below skips them. A model served via `rocm serve` thus appears
        // in the dashboard and is scraped for `gen_tps` without Docker.
        if let Some(services_dir) = opts.services_dir.as_ref()
            && (tick_count == 1 || tick_count.is_multiple_of(discovery_ticks))
        {
            let records = crate::registry::load_service_records(services_dir);
            let disc = crate::registry::discover_managed_services(&records);
            managed_non_vllm = disc.non_vllm;
            for svc in disc.svcs {
                let id = svc.container_id.clone();
                let is_new = !known_services.contains(&id);
                let existing = runner.state.instances.get(&id).cloned();
                // Endpoint change → new logical service; reset rate baseline and
                // latency windows so the new port starts fresh.
                let same_endpoint = existing.as_ref().is_none_or(|ex| ex.port == svc.port);
                if !same_endpoint {
                    invalidate_on_endpoint_change(
                        &id,
                        &mut gen_trackers,
                        &mut prev_ttft,
                        &mut prev_tpot,
                    );
                }
                let inst = merge_discovery_refresh(
                    &svc,
                    if same_endpoint {
                        existing.as_ref()
                    } else {
                        None
                    },
                );
                runner
                    .state
                    .apply(StateEvent::InstanceUpserted(inst.clone()));
                if is_new {
                    info!(id = %id, "managed service discovered");
                    // InstanceDiscovered is a discovery-metadata event; broadcast
                    // a clean instance (no prior collector-owned telemetry).
                    broadcast_and_persist(
                        &tx,
                        persist.as_ref(),
                        Event::InstanceDiscovered(instance_from_discovered(&svc)),
                    );
                }
            }
            for gone in known_services.difference(&disc.seen) {
                info!(id = %gone, "managed service gone");
                purge_telemetry_state(gone, &mut gen_trackers, &mut prev_ttft, &mut prev_tpot);
                runner
                    .state
                    .apply(StateEvent::InstanceRemoved(gone.clone()));
                broadcast_and_persist(
                    &tx,
                    persist.as_ref(),
                    Event::InstanceGone {
                        container_id: gone.clone(),
                    },
                );
            }
            known_services = disc.seen;
        }
        if tick_count.is_multiple_of(instance_ticks) {
            scrape_warns.clear();
        }

        // Per-instance vLLM metric scrape, parallel, on its own cadence. The
        // Lemonade instance (if any) is scraped via its own collector below, so
        // exclude it from the vLLM Prometheus targets. Managed non-vLLM services
        // are excluded too (their engine uses a different parser).
        if let Some(prom) = vllm.as_ref()
            && !runner.state.instances.is_empty()
            && tick_count.is_multiple_of(instance_ticks)
        {
            let targets: Vec<(String, u16)> = runner
                .state
                .instances
                .values()
                .filter(|i| Some(i.container_id.as_str()) != lemonade_id.as_deref())
                .filter(|i| !managed_non_vllm.contains(&i.container_id))
                .filter_map(|i| i.port.map(|p| (i.container_id.clone(), p)))
                .collect();
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
                            gen_trackers
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
                        // histograms; cumulative-average fallback on first scrape.
                        let ttft_ms = avg_ms_from_histogram(
                            &mut prev_ttft,
                            &id,
                            sample.ttft_sum_s,
                            sample.ttft_count,
                            cycle_at,
                        );
                        let tpot_ms = avg_ms_from_histogram(
                            &mut prev_tpot,
                            &id,
                            sample.tpot_sum_s,
                            sample.tpot_count,
                            cycle_at,
                        );
                        if let Some(mut inst) = runner.state.instances.get(&id).cloned() {
                            inst.kv_cache_usage_pct = sample.kv_cache_usage_pct;
                            inst.running_reqs = sample.running_reqs;
                            inst.waiting_reqs = sample.waiting_reqs;
                            inst.ttft_ms = ttft_ms;
                            inst.tpot_ms = tpot_ms;
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
                        // Clear TTFT/TPOT baselines so recovery re-baselines from
                        // the next successful scrape (latency clearing semantics
                        // are unchanged; freshness generalization is out of scope).
                        prev_ttft.remove(&id);
                        prev_tpot.remove(&id);
                        if let Some(mut inst) = runner.state.instances.get(&id).cloned()
                            && (inst.ttft_ms.is_some() || inst.tpot_ms.is_some())
                        {
                            inst.ttft_ms = None;
                            inst.tpot_ms = None;
                            runner.state.apply(StateEvent::InstanceUpserted(inst));
                        }
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

        // Lemonade per-instance scrape — reports an instantaneous rate directly
        // (`gen_tps`), so no counter-differencing. A scrape failure leaves the
        // last-known fields and warns; it never panics.
        if let (Some(l), Some(id)) = (lemonade.as_ref(), lemonade_id.clone())
            && tick_count.is_multiple_of(instance_ticks)
        {
            match l.fetch_stats().await {
                Ok(sample) => {
                    // EAI-7960: drive the tracker with a direct rate reading when
                    // the sample carries one. A missing direct rate (None) does not
                    // erase or refresh the tracker; the held value stays valid until
                    // the validity window expires.
                    if let Some(rate) = sample.gen_tps {
                        gen_trackers
                            .entry(id.clone())
                            .or_insert_with(|| {
                                GenerationObservationTracker::new(opts.instance_tick)
                            })
                            .observe_direct(rate, cycle_at);
                    }
                    if let Some(mut inst) = runner.state.instances.get(&id).cloned() {
                        // gen_tps is written in the snapshot assembly below;
                        // set the KV/request fields immediately so they are
                        // present in state for subsequent snapshots.
                        inst.kv_cache_usage_pct = sample.kv_cache_usage_pct;
                        inst.running_reqs = sample.running_reqs;
                        inst.waiting_reqs = sample.waiting_reqs;
                        // See the vLLM scrape-success handler: a successful stats
                        // fetch proves the instance is serving; promote out of Starting.
                        if matches!(inst.status, InstanceStatus::Starting { .. }) {
                            inst.status = InstanceStatus::Ready;
                        }
                        runner.state.apply(StateEvent::InstanceUpserted(inst));
                    }
                }
                Err(e) => {
                    trace!(id = %id, error = %e, "lemonade scrape failed");
                    scrape_warns.push(format!("lemonade scrape: {e}"));
                    // EAI-7960: retain tracker during validity window on failure.
                    // gen_tps is assembled from the tracker snapshot below;
                    // no explicit clearing needed here.
                }
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
                    per_container_used = rocm_dash_core::vram::aggregate_process_vram(
                        &procs,
                        rocm_dash_collectors::cgroup::container_id_for_pid,
                    );
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
            let (gen_tps, meta) = gen_trackers
                .get_mut(&inst.container_id)
                .map_or((None, None), |t| t.snapshot(cycle_at));
            inst.gen_tps = gen_tps;
            inst.gen_tps_observation = meta;
        }
        // Derive per-instance efficiency now that this tick's GPU power is known
        // and gen_tps reflects the tracker state for this cycle.
        // Count instances that have throughput + live GPUs but whose gpu_ids
        // don't line up with any amd-smi device_id — the silent-None failure
        // mode on real hardware, surfaced via the header ⚠ badge.
        let mut id_join_misses = 0usize;
        for inst in &mut instances {
            inst.tokens_per_watt =
                rocm_dash_core::efficiency::tokens_per_watt(inst.gen_tps, &inst.gpu_ids, &gpus);
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
                entry.tokens_per_watt = inst.tokens_per_watt;
            }
        }
        // Attribute per-instance VRAM (per-process where the cgroup join hit,
        // device-summed otherwise). Pure; uses this tick's `gpus` for totals.
        enrich_instance_vram(&mut instances, &gpus, &per_container_used);
        // Warn only when a real process scrape happened but an instance with
        // live GPUs still fell back to device-summed (cgroup join missed).
        if procs_nonempty {
            let vram_fallbacks = instances
                .iter()
                .filter(|i| {
                    !i.gpu_ids.is_empty()
                        && rocm_dash_core::efficiency::gpu_ids_overlap(&i.gpu_ids, &gpus)
                        && !per_container_used.contains_key(&i.container_id)
                })
                .count();
            if vram_fallbacks > 0 {
                scrape_warns.push(format!(
                    "vram attribution: {vram_fallbacks} instance(s) fell back to device-summed \
                     VRAM (no per-process cgroup match; check /proc access)"
                ));
            }
        }
        warnings.extend(docker_warn.iter().cloned());
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

/// Remove all per-service telemetry state for `id`: tracker, TTFT, and TPOT
/// baselines.  Call on explicit removal (Docker Gone, Lemonade Gone, managed
/// Gone) and whenever an identity change requires a clean slate.
pub(crate) fn purge_telemetry_state(
    id: &str,
    gen_trackers: &mut HashMap<String, GenerationObservationTracker>,
    prev_ttft: &mut HashMap<String, (f64, f64, DateTime<Utc>)>,
    prev_tpot: &mut HashMap<String, (f64, f64, DateTime<Utc>)>,
) {
    gen_trackers.remove(id);
    prev_ttft.remove(id);
    prev_tpot.remove(id);
}

/// Invalidate the generation tracker for `id` (reset baseline without
/// removing the map entry) and clear the TTFT/TPOT latency windows.
/// Called when an endpoint changes within the same container/service id so
/// the new scrape address starts with a fresh rate baseline.
pub(crate) fn invalidate_on_endpoint_change(
    id: &str,
    gen_trackers: &mut HashMap<String, GenerationObservationTracker>,
    prev_ttft: &mut HashMap<String, (f64, f64, DateTime<Utc>)>,
    prev_tpot: &mut HashMap<String, (f64, f64, DateTime<Utc>)>,
) {
    if let Some(t) = gen_trackers.get_mut(id) {
        t.invalidate();
    }
    prev_ttft.remove(id);
    prev_tpot.remove(id);
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
        inst.tokens_per_watt = ex.tokens_per_watt;
        inst.ttft_ms = ex.ttft_ms;
        inst.tpot_ms = ex.tpot_ms;
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

/// How many `tick`s fit into `period`, rounded to the nearest, minimum 1.
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

fn ticks_per(period: Duration, tick: Duration) -> u64 {
    let n = (period.as_secs_f64() / tick.as_secs_f64()).round() as i64;
    n.max(1) as u64
}

/// Upper bound on the scrape interval accepted by `avg_ms_from_histogram`.
/// Applies to TTFT/TPOT latency histograms only; generation-counter staleness
/// is managed by `GenerationObservationTracker` (see `rocm-dash-core`).
/// A gap longer than this makes the windowed delta stale (outage or forward
/// wall-clock jump / NTP / VM resume), so the histogram falls back to the
/// cumulative average for that sample.
const MAX_RATE_INTERVAL_S: f64 = 60.0;

/// Average latency (ms) from a cumulative histogram's `_sum` (seconds) and
/// `_count` series, windowed over successive scrapes: `(Δsum / Δcount) × 1000`.
/// Updates the per-instance baseline in `prev`. On the FIRST reading (or after a
/// counter reset / stale interval / no new requests) it falls back to the
/// cumulative average `sum/count × 1000`, so a single scrape already yields a
/// value. Returns `None` when the histogram is absent (`sum`/`count` `None`) or
/// `count == 0` — Observe then shows `—` (never a fabricated number).
fn avg_ms_from_histogram(
    prev: &mut HashMap<String, (f64, f64, DateTime<Utc>)>,
    id: &str,
    sum_s: Option<f64>,
    count: Option<f64>,
    now: DateTime<Utc>,
) -> Option<f64> {
    let (sum_s, count) = (sum_s?, count?);
    let last = prev.insert(id.to_string(), (sum_s, count, now));
    if let Some((psum, pcount, pat)) = last {
        let dcount = count - pcount;
        let dsum = sum_s - psum;
        let dt = (now - pat).num_milliseconds() as f64 / 1000.0;
        if dcount > 0.0 && dsum >= 0.0 && dt > 0.0 && dt <= MAX_RATE_INTERVAL_S {
            return Some(dsum / dcount * 1000.0);
        }
    }
    // Cumulative-average fallback: first scrape, no new requests, or a reset.
    if count > 0.0 {
        Some(sum_s / count * 1000.0)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).unwrap()
    }

    #[test]
    fn avg_ms_first_scrape_uses_cumulative_then_windows() {
        let mut prev = HashMap::new();
        // First scrape: cumulative average = 2.0s / 4 × 1000 = 500 ms.
        assert_eq!(
            avg_ms_from_histogram(&mut prev, "a", Some(2.0), Some(4.0), at(10)),
            Some(500.0)
        );
        // Second scrape: Δsum=1.0s over Δcount=10 → 100 ms (windowed).
        assert_eq!(
            avg_ms_from_histogram(&mut prev, "a", Some(3.0), Some(14.0), at(12)),
            Some(100.0)
        );
        // Absent histogram → None (Observe shows `—`).
        assert_eq!(
            avg_ms_from_histogram(&mut prev, "b", None, None, at(12)),
            None
        );
        // count == 0 (no requests yet) → None, never a divide-by-zero number.
        assert_eq!(
            avg_ms_from_histogram(&mut prev, "c", Some(0.0), Some(0.0), at(12)),
            None
        );
        // A counter reset (sum drops) re-bases to the cumulative average.
        let mut p2 = HashMap::new();
        avg_ms_from_histogram(&mut p2, "d", Some(10.0), Some(100.0), at(10));
        assert_eq!(
            avg_ms_from_histogram(&mut p2, "d", Some(0.2), Some(2.0), at(12)),
            Some(100.0),
            "reset falls back to cumulative 0.2/2×1000"
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

    // -- Identity/endpoint cleanup helper tests -----------------------------------

    fn make_tracker(tick_secs: u64) -> GenerationObservationTracker {
        GenerationObservationTracker::new(Duration::from_secs(tick_secs))
    }

    /// `purge_telemetry_state` removes the tracker entry, TTFT baseline, and
    /// TPOT baseline for the target id and leaves other ids untouched.
    #[test]
    fn purge_telemetry_state_removes_all_three_maps_for_id() {
        let mut trackers: HashMap<String, GenerationObservationTracker> = HashMap::new();
        let mut ttft: HashMap<String, (f64, f64, DateTime<Utc>)> = HashMap::new();
        let mut tpot: HashMap<String, (f64, f64, DateTime<Utc>)> = HashMap::new();

        // Populate two ids.
        trackers.insert("a".into(), make_tracker(2));
        trackers.insert("b".into(), make_tracker(2));
        ttft.insert("a".into(), (1.0, 1.0, at(10)));
        ttft.insert("b".into(), (2.0, 2.0, at(10)));
        tpot.insert("a".into(), (3.0, 3.0, at(10)));
        tpot.insert("b".into(), (4.0, 4.0, at(10)));

        purge_telemetry_state("a", &mut trackers, &mut ttft, &mut tpot);

        // "a" must be gone from all maps.
        assert!(!trackers.contains_key("a"), "tracker for a must be removed");
        assert!(!ttft.contains_key("a"), "ttft for a must be removed");
        assert!(!tpot.contains_key("a"), "tpot for a must be removed");
        // "b" must be unaffected.
        assert!(trackers.contains_key("b"), "tracker for b must survive");
        assert!(ttft.contains_key("b"), "ttft for b must survive");
        assert!(tpot.contains_key("b"), "tpot for b must survive");
    }

    /// `invalidate_on_endpoint_change` invalidates the tracker (resets its
    /// window so subsequent snapshots return None) and removes the latency
    /// baselines, without removing the tracker entry itself.
    #[test]
    fn invalidate_on_endpoint_change_resets_tracker_and_clears_latency() {
        let mut trackers: HashMap<String, GenerationObservationTracker> = HashMap::new();
        let mut ttft: HashMap<String, (f64, f64, DateTime<Utc>)> = HashMap::new();
        let mut tpot: HashMap<String, (f64, f64, DateTime<Utc>)> = HashMap::new();

        let mut tracker = make_tracker(2);
        // Give the tracker a reading so it has a non-None baseline.
        tracker.observe_counter(100.0, at(10));
        tracker.observe_counter(200.0, at(12));
        trackers.insert("svc".into(), tracker);
        ttft.insert("svc".into(), (1.0, 10.0, at(10)));
        tpot.insert("svc".into(), (0.5, 8.0, at(10)));

        invalidate_on_endpoint_change("svc", &mut trackers, &mut ttft, &mut tpot);

        // Tracker still exists but its window is reset: snapshot returns None.
        let (rate, _meta) = trackers
            .get_mut("svc")
            .expect("tracker entry must remain")
            .snapshot(at(12));
        assert!(rate.is_none(), "invalidated tracker must yield None rate");
        // Latency baselines must be cleared.
        assert!(
            !ttft.contains_key("svc"),
            "ttft baseline must be cleared on endpoint change"
        );
        assert!(
            !tpot.contains_key("svc"),
            "tpot baseline must be cleared on endpoint change"
        );
    }
}
