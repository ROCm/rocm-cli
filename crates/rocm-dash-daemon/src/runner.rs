//! Owner task: drives collectors on tick cadences and broadcasts Snapshots.

use std::collections::{HashMap, HashSet};
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
    /// Disable the per-instance Prometheus scrape (otherwise runs whenever
    /// docker discovery is enabled).
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
        }
    }
}

#[derive(Default)]
pub struct Runner {
    pub state: State,
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

    let vllm = if opts.enable_docker && !opts.disable_vllm_metrics {
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
        match DockerDiscovery::detect(opts.image_patterns.clone()).await {
            Some(d) => {
                info!("docker discovery enabled");
                Some(d)
            }
            None => {
                warn!("docker discovery requested but daemon unreachable; disabled");
                None
            }
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
    // Previous `generation_tokens_total` reading per instance, for rate calc.
    let mut prev_gen_tokens: HashMap<String, (f64, DateTime<Utc>)> = HashMap::new();
    // Per-container VRAM (MB) from the last amd-smi `process` scrape. Refreshed
    // on the instance cadence and reused every tick so the attributed value is
    // stable between scrapes (mirrors how GPU power drives tokens_per_watt).
    let mut per_container_used: HashMap<String, u64> = HashMap::new();

    let gpu = AmdSmiCollector::detect().await;
    let mut gpu_system_info: Option<GpuSystemInfo> = if let Some(g) = &gpu {
        let info = g.system_info().await;
        info!(
            gpus = info.physical_gpu_count,
            model = %info.gpu_model,
            rocm = info.rocm_version.as_deref().unwrap_or("?"),
            "amd-smi detected"
        );
        Some(info)
    } else {
        warn!("amd-smi not available (no /dev/kfd or `amd-smi version` failed); GPU disabled");
        None
    };

    let mut tick_count: u64 = 0;
    let mut last_sysinfo_refresh: u64 = 0;

    loop {
        ticker.tick().await;
        tick_count += 1;

        let mut warnings = Vec::new();
        let gpus = match &gpu {
            Some(g) => match g.metrics().await {
                Ok(v) => v,
                Err(e) => {
                    warnings.push(format!("amd-smi metric: {e}"));
                    Vec::new()
                }
            },
            None => {
                warnings.push("amd-smi unavailable (no /dev/kfd or binary missing)".into());
                Vec::new()
            }
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
                        let inst = instance_from_discovered(svc);
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
                            broadcast_and_persist(
                                &tx,
                                persist.as_ref(),
                                Event::InstanceDiscovered(inst),
                            );
                        }
                    }
                    for gone in known_instances.difference(&seen) {
                        info!(id = %gone, "instance gone");
                        prev_gen_tokens.remove(gone);
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
                }
                Err(e) => warnings.push(format!("docker discover: {e}")),
            }
        }

        // Lemonade discovery — probe the endpoint on the discovery cadence; add a
        // Lemonade Instance when reachable, emit Gone when it disappears, and stay
        // a clean no-op (no warning/panic) when no endpoint is configured.
        if let Some(l) = lemonade.as_ref()
            && (tick_count == 1 || tick_count.is_multiple_of(discovery_ticks))
        {
            match l.discover().await {
                Some(svc) => {
                    let inst = instance_from_discovered(&svc);
                    runner
                        .state
                        .apply(StateEvent::InstanceUpserted(inst.clone()));
                    if lemonade_id.as_deref() != Some(svc.container_id.as_str()) {
                        info!(
                            id = %svc.container_id,
                            model = %svc.model_name,
                            "lemonade instance discovered"
                        );
                        broadcast_and_persist(
                            &tx,
                            persist.as_ref(),
                            Event::InstanceDiscovered(inst),
                        );
                        lemonade_id = Some(svc.container_id.clone());
                    }
                }
                None => {
                    if let Some(id) = lemonade_id.take() {
                        info!(id = %id, "lemonade instance gone");
                        prev_gen_tokens.remove(&id);
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
            for inst in disc.instances {
                let is_new = !known_services.contains(&inst.container_id);
                let id = inst.container_id.clone();
                runner
                    .state
                    .apply(StateEvent::InstanceUpserted(inst.clone()));
                if is_new {
                    info!(id = %id, "managed service discovered");
                    broadcast_and_persist(&tx, persist.as_ref(), Event::InstanceDiscovered(inst));
                }
            }
            for gone in known_services.difference(&disc.seen) {
                info!(id = %gone, "managed service gone");
                prev_gen_tokens.remove(gone);
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
                        // Difference the cumulative token counter into a live
                        // rate; first reading (or a restart) yields None.
                        let gen_tps = sample.gen_tokens_total.and_then(|cur| {
                            let now = Utc::now();
                            let prev = prev_gen_tokens.insert(id.clone(), (cur, now));
                            gen_tps_from_delta(prev, cur, now)
                        });
                        if let Some(mut inst) = runner.state.instances.get(&id).cloned() {
                            inst.kv_cache_usage_pct = sample.kv_cache_usage_pct;
                            inst.running_reqs = sample.running_reqs;
                            inst.waiting_reqs = sample.waiting_reqs;
                            inst.gen_tps = gen_tps;
                            runner.state.apply(StateEvent::InstanceUpserted(inst));
                        }
                    }
                    Err(e) => {
                        fail_count += 1;
                        let msg = format!("{e}");
                        trace!(id = %id, error = %msg, "vllm scrape failed");
                        last_err = Some(msg);
                        // Don't let a dead instance show a frozen rate: clear
                        // throughput and drop the baseline so recovery re-bases.
                        prev_gen_tokens.remove(&id);
                        if let Some(mut inst) = runner.state.instances.get(&id).cloned()
                            && inst.gen_tps.is_some()
                        {
                            inst.gen_tps = None;
                            runner.state.apply(StateEvent::InstanceUpserted(inst));
                        }
                    }
                }
            }
            if fail_count > 0 {
                warnings.push(match last_err {
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
                    if let Some(mut inst) = runner.state.instances.get(&id).cloned() {
                        inst.gen_tps = sample.gen_tps;
                        inst.kv_cache_usage_pct = sample.kv_cache_usage_pct;
                        inst.running_reqs = sample.running_reqs;
                        inst.waiting_reqs = sample.waiting_reqs;
                        runner.state.apply(StateEvent::InstanceUpserted(inst));
                    }
                }
                Err(e) => {
                    trace!(id = %id, error = %e, "lemonade scrape failed");
                    warnings.push(format!("lemonade scrape: {e}"));
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
        // Derive per-instance efficiency now that this tick's GPU power is known.
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
                warnings.push(format!(
                    "vram attribution: {vram_fallbacks} instance(s) fell back to device-summed \
                     VRAM (no per-process cgroup match; check /proc access)"
                ));
            }
        }
        let snap = Snapshot {
            timestamp: Utc::now(),
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

/// Build an `Instance` from a `DiscoveredService` with no live KV/req sample yet.
/// vLLM Prometheus scraping will fill these fields in a later collector.
pub(crate) fn instance_from_discovered(svc: &DiscoveredService) -> Instance {
    let mut inst = merge_instance(svc, &InstanceSample::default(), 0, 0);
    inst.status = InstanceStatus::Running;
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

/// A gap longer than this between counter readings makes the baseline stale —
/// the rate would be a misleadingly low average across an outage or a forward
/// wall-clock jump (NTP, VM resume), so we re-baseline instead.
const MAX_RATE_INTERVAL_S: f64 = 60.0;

/// Instantaneous generation tok/s from two cumulative counter readings.
///
/// Returns `None` on the first reading (`prev` is `None`), a non-positive or
/// stale interval (backwards/forward clock jump, scrape outage), or a counter
/// reset (`cur < prev`, e.g. the vLLM process restarted) — so the rate is
/// never negative, stale, or otherwise bogus.
pub(crate) fn gen_tps_from_delta(
    prev: Option<(f64, DateTime<Utc>)>,
    cur: f64,
    now: DateTime<Utc>,
) -> Option<f64> {
    let (prev_val, prev_at) = prev?;
    let dt = (now - prev_at).num_milliseconds() as f64 / 1000.0;
    if dt <= 0.0 || dt > MAX_RATE_INTERVAL_S || cur < prev_val {
        return None;
    }
    Some((cur - prev_val) / dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).unwrap()
    }

    #[test]
    fn gen_tps_none_on_first_reading() {
        assert_eq!(gen_tps_from_delta(None, 100.0, at(10)), None);
    }

    #[test]
    fn gen_tps_computes_rate() {
        // 200 tokens accumulated over 2 s → 100 tok/s.
        assert_eq!(
            gen_tps_from_delta(Some((100.0, at(10))), 300.0, at(12)),
            Some(100.0)
        );
    }

    #[test]
    fn gen_tps_none_on_stale_interval() {
        // Gap beyond MAX_RATE_INTERVAL_S → re-baseline rather than a bogus avg.
        assert_eq!(
            gen_tps_from_delta(Some((100.0, at(10))), 9000.0, at(10 + 120)),
            None
        );
    }

    #[test]
    fn gen_tps_none_on_counter_reset() {
        // Process restarted: cur < prev → no negative rate.
        assert_eq!(
            gen_tps_from_delta(Some((500.0, at(10))), 20.0, at(12)),
            None
        );
    }

    #[test]
    fn gen_tps_none_on_nonpositive_interval() {
        assert_eq!(
            gen_tps_from_delta(Some((100.0, at(12))), 300.0, at(12)),
            None
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
            gpu_ids: gpu_ids.iter().map(|s| s.to_string()).collect(),
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
