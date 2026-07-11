// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Managed-service registry → scrape-target seam.
//!
//! The rocm-cli `serve` lifecycle records every managed engine as a
//! `ManagedServiceRecord` JSON file under `AppPaths::services_dir()`. This
//! module reads that registry and converts live records into the collector
//! pipeline's existing [`DiscoveredService`] shape, so a model served via
//! `rocm serve` shows up in the dashboard and gets scraped for `gen_tps` —
//! without going through Docker discovery.
//!
//! The on-disk record is read through [`ServiceRecord`], a minimal mirror of the
//! stable fields of `rocm_core::ManagedServiceRecord` (every field
//! `#[serde(default)]`, unknown fields ignored). This keeps the async daemon
//! decoupled from `rocm-core`'s sync/ureq surface and its churn; the shape is
//! drift-tolerant because rocm-cli serializes the record as a JSON object and we
//! read only the fields the scrape seam needs.
//!
//! **Port authority:** the bound port comes from the registry record
//! (`ServiceRecord.port`), never a hardcoded default. The engine defaults
//! (`EngineKind::default_port`, `docker.rs`, `lemonade.rs`) remain only as a
//! fallback for *unmanaged/external* discovery.
//!
//! **Co-located scope (known limitation):** scrape targets carry only a port;
//! the runner's vLLM collector scrapes `opts.vllm_metrics_host` (default
//! `127.0.0.1`), so `record.host` is honored only for the co-located daemon
//! (the same assumption the Docker-discovery path already makes). A managed
//! service on a non-loopback host is surfaced but its metrics are scraped at the
//! local address. Per-host scrape targeting is tracked as a D7 follow-up.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::runner::instance_from_discovered;
use rocm_dash_collectors::engine_registry::EngineKind;
use rocm_dash_core::metrics::{Instance, InstanceStatus, StartupPhase};
use rocm_dash_core::traits::DiscoveredService;
use serde::Deserialize;

/// Minimal read-only view of a rocm-cli `ManagedServiceRecord` on disk.
///
/// Mirrors the stable subset the scrape seam needs; unknown fields (manifest paths,
/// pids, recipe json, …) are ignored. Every field defaults so partial/older
/// records still parse.
#[derive(Debug, Clone, Deserialize)]
pub struct ServiceRecord {
    #[serde(default)]
    pub service_id: String,
    #[serde(default)]
    pub engine: String,
    #[serde(default)]
    pub model_ref: String,
    #[serde(default)]
    pub canonical_model_id: String,
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub port: u16,
    #[serde(default)]
    pub status: String,
    /// Coarse startup stage (`downloading`/`loading`/`warmup`) the rocm-cli
    /// supervisor parsed from the serve logs while the service was coming up.
    /// Absent on older records and once the service reaches `ready`.
    #[serde(default)]
    pub startup_phase: Option<String>,
    #[serde(default)]
    pub created_at_unix_ms: u128,
}

/// Service statuses worth scraping.
///
/// Matches the live set the rocm-cli supervisor overlays onto a record
/// (`ready`/`running`/`starting`/`recovering` — see e.g. `apps/rocm/src/main.rs`'s
/// own "is this service live" checks); a `failed`/`stopped` record is skipped
/// so we never poll a dead port.
pub fn is_scrapeable_status(status: &str) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "ready" | "running" | "starting" | "recovering"
    )
}

/// Map a rocm-cli managed-service record's `status` string (plus any parsed
/// `startup_phase`) onto the dashboard's `InstanceStatus`. Only called after
/// `is_scrapeable_status` has already filtered out `failed`/`stopped`/unrecognized
/// strings, so the `_` fallback arm is unreachable in practice but kept total
/// for safety.
///
/// **`running` + `startup_phase` overlap:** the rocm-cli supervisor
/// (`apps/rocmd/src/lib.rs::supervise_service`) flips the on-disk record to
/// `running` immediately after spawning the engine child — *before* it starts
/// polling and parsing `startup_phase` from the serve log — and only clears
/// `startup_phase` once the service reaches `ready`. So a record can be
/// `running` with a `startup_phase` still set while the model is downloading
/// or loading; that combination must still surface as `Starting{phase}`, not
/// `Running`, or the coarse phase never renders during a real cold start. A
/// `running` record without a phase (the steady-state case) still maps to
/// plain `Running`.
fn instance_status_for_record(status: &str, startup_phase: Option<&str>) -> InstanceStatus {
    let phase = || startup_phase.and_then(StartupPhase::from_token);
    match status.trim().to_ascii_lowercase().as_str() {
        "ready" => InstanceStatus::Ready,
        "running" => startup_phase.map_or(InstanceStatus::Running, |_| InstanceStatus::Starting {
            phase: phase(),
        }),
        "starting" | "recovering" => InstanceStatus::Starting { phase: phase() },
        _ => InstanceStatus::Unknown,
    }
}

/// Load every managed-service record under `services_dir`, newest first.
///
/// Best-effort and side-effect-free: a missing directory yields an empty list,
/// and an individual unreadable/malformed `*.json` is skipped rather than
/// failing the whole load (mirrors the canonical rocm-cli `load_managed_services`
/// semantics, minus the engine-state status overlay which the supervisor owns).
pub fn load_service_records(services_dir: &Path) -> Vec<ServiceRecord> {
    let Ok(entries) = fs::read_dir(services_dir) else {
        return Vec::new();
    };
    let mut records = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|v| v.to_str()) != Some("json") {
            continue;
        }
        if let Ok(bytes) = fs::read(&path)
            && let Ok(record) = serde_json::from_slice::<ServiceRecord>(&bytes)
        {
            records.push(record);
        }
    }
    records.sort_by_key(|record| std::cmp::Reverse(record.created_at_unix_ms));
    records
}

/// Convert a managed-service record into a [`DiscoveredService`] for the
/// collector pipeline, or `None` when the record is not in a scrapeable state.
///
/// The port is taken verbatim from the registry (`record.port`) — the registry
/// is the authority, never a hardcoded default. `gpu_ids` is left empty because
/// the record carries no concrete device ids; VRAM and tokens/W attribution
/// degrade gracefully (device-summed / `None`) for managed services until a
/// richer device mapping is recorded.
pub fn discovered_from_record(record: &ServiceRecord) -> Option<DiscoveredService> {
    if !is_scrapeable_status(&record.status) {
        return None;
    }
    // A `#[serde(default)]` u16 missing from the JSON deserializes to 0; an
    // unbound/malformed port is not a real scrape target (would poll :0 forever).
    if record.port == 0 {
        return None;
    }
    let model_name = if record.model_ref.is_empty() {
        record.canonical_model_id.clone()
    } else {
        record.model_ref.clone()
    };
    Some(DiscoveredService {
        container_id: record.service_id.clone(),
        container_name: record.service_id.clone(),
        model_name,
        port: Some(record.port),
        status: instance_status_for_record(&record.status, record.startup_phase.as_deref()),
        ..Default::default()
    })
}

/// The engine kind for a record's `engine` label, if recognized. Lets the
/// scrape pipeline pick the right per-engine parser (vLLM Prometheus vs
/// Lemonade JSON) for a managed service.
pub fn engine_kind_for(record: &ServiceRecord) -> Option<EngineKind> {
    EngineKind::from_label(&record.engine)
}

/// The result of turning a batch of registry records into dashboard instances.
///
/// Contains the live instances to upsert, their ids (`seen`), and the subset whose engine
/// is NOT vLLM (`non_vllm`, excluded from the vLLM Prometheus scrape so they are
/// not mis-parsed). Pure — the runner applies it (upsert/broadcast/Gone-diff).
#[derive(Debug, Default)]
pub struct ManagedDiscovery {
    pub instances: Vec<Instance>,
    pub seen: HashSet<String>,
    pub non_vllm: HashSet<String>,
}

/// Convert the loaded registry records into a [`ManagedDiscovery`] — the pure core of the daemon's managed-service discovery tick.
///
/// Non-scrapeable records (bad status / port 0) are dropped; vLLM-engine services flow to the
/// Prometheus scrape, others are flagged in `non_vllm`.
///
/// NOTE: gen_tps for managed **non-vLLM** engines (e.g. Lemonade) is not yet
/// wired — they appear in the dashboard but are excluded from the vLLM scrape
/// and not routed to a per-engine collector here. vLLM (the Phase-2 acceptance
/// target) is fully wired. Non-vLLM managed scraping is a D7 follow-up.
pub fn discover_managed_services(records: &[ServiceRecord]) -> ManagedDiscovery {
    let mut out = ManagedDiscovery::default();
    for record in records {
        let Some(svc) = discovered_from_record(record) else {
            continue;
        };
        out.seen.insert(svc.container_id.clone());
        if engine_kind_for(record) != Some(EngineKind::Vllm) {
            out.non_vllm.insert(svc.container_id.clone());
        }
        out.instances.push(instance_from_discovered(&svc));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::gen_tps_from_delta;
    use chrono::{DateTime, TimeZone, Utc};
    use std::path::PathBuf;

    fn at(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).unwrap()
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("daemon-registry")
            .join(format!("{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A `ManagedServiceRecord`-shaped JSON object (extra rocm-cli fields the
    /// seam ignores are included to prove unknown-field tolerance).
    fn record_json(
        service_id: &str,
        engine: &str,
        port: u16,
        status: &str,
        created: u128,
    ) -> String {
        format!(
            r#"{{
              "service_id": "{service_id}",
              "engine": "{engine}",
              "model_ref": "meta-llama/Llama-3.1-8B",
              "canonical_model_id": "llama-3.1-8b",
              "host": "127.0.0.1",
              "port": {port},
              "endpoint_url": "http://127.0.0.1:{port}/v1",
              "mode": "managed",
              "status": "{status}",
              "supervisor_pid": 4242,
              "manifest_path": "/tmp/{service_id}.json",
              "log_path": "/tmp/{service_id}.log",
              "engine_state_path": "/tmp/{service_id}-state.json",
              "created_at_unix_ms": {created}
            }}"#
        )
    }

    #[test]
    fn load_missing_services_dir_is_empty() {
        let dir = test_dir("absent").join("nope");
        assert!(load_service_records(&dir).is_empty());
    }

    #[test]
    fn load_reads_json_records_newest_first() {
        let dir = test_dir("load-sort");
        fs::write(
            dir.join("a.json"),
            record_json("svc-a", "vllm", 8000, "running", 100),
        )
        .unwrap();
        fs::write(
            dir.join("b.json"),
            record_json("svc-b", "vllm", 8001, "running", 200),
        )
        .unwrap();
        // A non-json file and a malformed json are skipped, not fatal.
        fs::write(dir.join("notes.txt"), "ignore me").unwrap();
        fs::write(dir.join("bad.json"), "{ not valid").unwrap();

        let records = load_service_records(&dir);
        assert_eq!(records.len(), 2);
        // Newest (created=200) first.
        assert_eq!(records[0].service_id, "svc-b");
        assert_eq!(records[1].service_id, "svc-a");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discovered_uses_registry_port_as_authority() {
        let dir = test_dir("port-authority");
        // A non-default port (8123, not the vLLM hardcoded 8000) must survive.
        fs::write(
            dir.join("svc.json"),
            record_json("svc-hot", "vllm", 8123, "running", 1),
        )
        .unwrap();
        let records = load_service_records(&dir);
        let svc = discovered_from_record(&records[0]).expect("live record converts");
        assert_eq!(svc.port, Some(8123));
        assert_eq!(svc.container_id, "svc-hot");
        assert_eq!(svc.model_name, "meta-llama/Llama-3.1-8B");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discovered_skips_non_scrapeable_status() {
        for dead in ["stopped", "failed", "exited", ""] {
            let rec: ServiceRecord =
                serde_json::from_str(&record_json("svc", "vllm", 8000, dead, 1)).unwrap();
            assert!(
                discovered_from_record(&rec).is_none(),
                "status {dead:?} must not be scraped"
            );
        }
        for live in ["running", "ready", "starting", "RUNNING"] {
            let rec: ServiceRecord =
                serde_json::from_str(&record_json("svc", "vllm", 8000, live, 1)).unwrap();
            assert!(
                discovered_from_record(&rec).is_some(),
                "status {live:?} must be scraped"
            );
        }
    }

    #[test]
    fn discovered_from_record_maps_status_string_to_instance_status() {
        // The registry seam must surface the record's real lifecycle state,
        // not silently downgrade everything to `Running`/`Unknown`.
        let cases = [
            ("ready", InstanceStatus::Ready),
            ("running", InstanceStatus::Running),
            ("RUNNING", InstanceStatus::Running),
            ("starting", InstanceStatus::Starting { phase: None }),
            ("recovering", InstanceStatus::Starting { phase: None }),
        ];
        for (status, expected) in cases {
            let rec: ServiceRecord =
                serde_json::from_str(&record_json("svc", "vllm", 8000, status, 1)).unwrap();
            let svc = discovered_from_record(&rec)
                .unwrap_or_else(|| panic!("status {status:?} must be scrapeable"));
            assert_eq!(svc.status, expected, "status {status:?}");
        }
    }

    #[test]
    fn discovered_from_record_surfaces_startup_phase_while_starting() {
        // A `starting`/`recovering` record with a parsed phase must carry that
        // phase into the instance status so the dashboard can show
        // DOWNLOADING/LOADING/WARMUP.
        for status in ["starting", "recovering"] {
            let rec: ServiceRecord = serde_json::from_str(&format!(
                r#"{{
                  "service_id": "svc-load", "engine": "vllm", "model_ref": "m",
                  "canonical_model_id": "m", "host": "127.0.0.1", "port": 8000,
                  "status": "{status}", "startup_phase": "loading",
                  "created_at_unix_ms": 1
                }}"#
            ))
            .unwrap();
            let svc = discovered_from_record(&rec).expect("record is scrapeable");
            assert_eq!(
                svc.status,
                InstanceStatus::Starting {
                    phase: Some(StartupPhase::Loading)
                },
                "status {status:?}"
            );
        }

        // An unrecognized phase token degrades to a phase-less Starting.
        let rec: ServiceRecord = serde_json::from_str(
            r#"{
              "service_id": "svc-x", "engine": "vllm", "model_ref": "m",
              "canonical_model_id": "m", "host": "127.0.0.1", "port": 8000,
              "status": "starting", "startup_phase": "bogus",
              "created_at_unix_ms": 1
            }"#,
        )
        .unwrap();
        let svc = discovered_from_record(&rec).expect("starting record is scrapeable");
        assert_eq!(svc.status, InstanceStatus::Starting { phase: None });
    }

    #[test]
    fn discovered_from_record_surfaces_startup_phase_while_running() {
        // The real producer (`apps/rocmd/src/lib.rs::supervise_service`) flips
        // the on-disk record to `running` *before* it starts polling and
        // parsing `startup_phase` from the serve log, and only clears
        // `startup_phase` once the service reaches `ready`. Exercise that
        // exact `running` + `startup_phase` shape end to end through
        // `load_service_records` + `discovered_from_record` (not a
        // hand-picked shape the producer never writes) to prove the coarse
        // phase actually renders during a real cold start.
        let dir = test_dir("running-phase");
        fs::write(
            dir.join("svc.json"),
            r#"{
              "service_id": "svc-cold", "engine": "vllm", "model_ref": "m",
              "canonical_model_id": "m", "host": "127.0.0.1", "port": 8000,
              "status": "running", "startup_phase": "downloading",
              "created_at_unix_ms": 1
            }"#,
        )
        .unwrap();

        let records = load_service_records(&dir);
        let svc = discovered_from_record(&records[0]).expect("running record is scrapeable");
        assert_eq!(
            svc.status,
            InstanceStatus::Starting {
                phase: Some(StartupPhase::Downloading)
            },
            "a `running` record with a live startup_phase must still render as Starting"
        );
        let _ = fs::remove_dir_all(&dir);

        // Once the supervisor clears `startup_phase` (service reached
        // `ready`... though status flips to `ready` too — this proves the
        // `running`-with-no-phase steady state still maps to plain Running).
        let dir = test_dir("running-no-phase");
        fs::write(
            dir.join("svc.json"),
            r#"{
              "service_id": "svc-steady", "engine": "vllm", "model_ref": "m",
              "canonical_model_id": "m", "host": "127.0.0.1", "port": 8000,
              "status": "running", "created_at_unix_ms": 1
            }"#,
        )
        .unwrap();
        let records = load_service_records(&dir);
        let svc = discovered_from_record(&records[0]).expect("running record is scrapeable");
        assert_eq!(svc.status, InstanceStatus::Running);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discovered_skips_zero_port() {
        // A record whose port field is absent (serde default 0) is not a real
        // scrape target even when its status is live.
        let rec: ServiceRecord =
            serde_json::from_str(&record_json("svc", "vllm", 0, "running", 1)).unwrap();
        assert!(discovered_from_record(&rec).is_none());
        // A missing `port` key entirely → default 0 → also skipped.
        let rec: ServiceRecord =
            serde_json::from_str(r#"{"service_id":"x","engine":"vllm","status":"running"}"#)
                .unwrap();
        assert_eq!(rec.port, 0);
        assert!(discovered_from_record(&rec).is_none());
    }

    #[test]
    fn discover_managed_services_classifies_and_filters() {
        let records: Vec<ServiceRecord> = [
            record_json("svc-vllm", "vllm", 8000, "running", 3),
            record_json("svc-lemon", "lemonade", 13305, "ready", 2),
            record_json("svc-dead", "vllm", 9000, "stopped", 1),
            record_json("svc-noport", "vllm", 0, "running", 0),
        ]
        .iter()
        .map(|j| serde_json::from_str(j).unwrap())
        .collect();

        let disc = discover_managed_services(&records);
        // Two live, scrapeable instances (dead + zero-port dropped).
        assert_eq!(disc.instances.len(), 2);
        assert!(disc.seen.contains("svc-vllm"));
        assert!(disc.seen.contains("svc-lemon"));
        assert!(!disc.seen.contains("svc-dead"));
        assert!(!disc.seen.contains("svc-noport"));
        // Only the Lemonade service is excluded from the vLLM scrape.
        assert!(disc.non_vllm.contains("svc-lemon"));
        assert!(!disc.non_vllm.contains("svc-vllm"));
        // Instances carry the registry port + Running status.
        let vllm = disc
            .instances
            .iter()
            .find(|i| i.container_id == "svc-vllm")
            .unwrap();
        assert_eq!(vllm.port, Some(8000));
        assert_eq!(
            vllm.status,
            rocm_dash_core::metrics::InstanceStatus::Running
        );
    }

    /// Deterministic end-to-end of the registry→scrape→dashboard data path (no
    /// GPU required): a `rocm serve`-style managed record on disk → loaded →
    /// converted to a scrape target on the **registry port** → the engine-kind
    /// parser turns two successive vLLM Prometheus bodies into a cumulative
    /// counter → the runner's delta yields a live `gen_tps`. This is the
    /// test-level proof for Phase-2 acceptance criterion 3 (no ROCm GPU here).
    #[test]
    fn serve_record_to_live_gen_tps_end_to_end() {
        let dir = test_dir("e2e-gen-tps");
        fs::write(
            dir.join("svc.json"),
            record_json("svc-llama", "vllm", 8123, "running", 1),
        )
        .unwrap();

        // 1. Registry → scrape target (port authority is the registry's 8123).
        let records = load_service_records(&dir);
        assert_eq!(records.len(), 1);
        let svc = discovered_from_record(&records[0]).expect("live record");
        let scrape_port = svc.port.expect("registry port");
        assert_eq!(scrape_port, 8123);

        // 2. Engine-kind seam picks the vLLM Prometheus parser for this record.
        let kind = engine_kind_for(&records[0]).expect("known engine");
        assert_eq!(kind, EngineKind::Vllm);

        // 3. Two successive scrapes of that port's /metrics → cumulative counter.
        let body_t0 = "vllm:generation_tokens_total 1000\n";
        let body_t1 = "vllm:generation_tokens_total 1400\n";
        let s0 = kind.parse_sample(body_t0);
        let s1 = kind.parse_sample(body_t1);
        let c0 = s0.gen_tokens_total.expect("counter at t0");
        let c1 = s1.gen_tokens_total.expect("counter at t1");

        // 4. Runner delta → live gen_tps: 400 tokens over 2 s = 200 tok/s.
        let gen_tps = gen_tps_from_delta(Some((c0, at(10))), c1, at(12));
        assert_eq!(gen_tps, Some(200.0));
        let _ = fs::remove_dir_all(&dir);
    }
}
