// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Managed-service liveness classification and the dashboard-sidebar
//! status/counts helpers derived from it.
//!
//! Mechanically relocated from `main.rs` with no behavior change — Phase 6f
//! (ROCMAI-211) of the `main.rs` modularization (see
//! `docs/architecture.md`). `dispatch()`'s call sites are unaffected: none of
//! this module's functions are called directly from `dispatch()`.
//!
//! This module owns: reading and liveness-refreshing the on-disk managed-service
//! registry (`load_managed_services`, `refresh_managed_service_runtime_liveness`,
//! `recorded_service_pids`, `settle_pending_stop_key_cleanup`), the idempotency
//! guard that finds an already-live service for a given engine+model
//! (`existing_live_managed_service`, and its records-in-hand form
//! `find_live_managed_service`), the serve reuse pre-gate that decides whether
//! asking the engine about reuse is worth a round-trip
//! (`live_managed_services_for_model`), the "is this record live" classifier
//! (`managed_service_is_live`), and the dashboard sidebar's ready/starting/past-
//! attempts counts and one-line status text
//! (`ManagedServiceSidebarCounts`/`managed_service_sidebar_counts`/
//! `local_server_sidebar_status`) plus the engine display-name helper
//! (`friendly_engine_label`), which only `chat.rs` reads. `main.rs` calls into
//! it for the in-use check behind service-record removal and pruning
//! (`managed_service_is_live`, `recorded_service_pids`) and for
//! `load_managed_services` in prune planning and the codex-bridge snapshot;
//! `services_text.rs` reads the sidebar helpers for the local-servers view;
//! `chat.rs` calls into it for the deterministic services-tool summary;
//! `runtime_services.rs` calls into it when reconciling live servers against a
//! runtime activation; `service_launch.rs` is the only caller of the
//! `existing_live_managed_service` duplicate-launch guard it runs before
//! spawning a new managed service. Shared types/config (`ManagedServiceRecord`,
//! `AppPaths`) and the `endpoint_keys` module stay reachable through
//! `crate::`/`rocm_core::`, as for other mechanically relocated modules.

use std::fs;
use std::time::Duration;

use anyhow::{Context, Result};
use rocm_core::{
    AppPaths, EndpointReadiness, EndpointReadinessOutcome, ManagedServiceRecord,
    managed_service_endpoint_readiness, process_is_running,
};

use crate::endpoint_keys;
use crate::engine_runtime_select::service_model_names_match;

pub(crate) fn friendly_engine_label(engine: &str) -> &str {
    match engine {
        "lemonade" => "Lemonade",
        "vllm" => "vLLM",
        "aim" => "AIM",
        other => other,
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ManagedServiceSidebarCounts {
    pub(crate) ready: usize,
    pub(crate) starting: usize,
    pub(crate) past_attempts: usize,
}

pub(crate) fn managed_service_sidebar_counts(
    records: &[ManagedServiceRecord],
) -> ManagedServiceSidebarCounts {
    let mut counts = ManagedServiceSidebarCounts::default();
    for record in records {
        match record.status.as_str() {
            "ready" | "running" => counts.ready += 1,
            "starting" | "recovering" => counts.starting += 1,
            _ => counts.past_attempts += 1,
        }
    }
    counts
}

pub(crate) fn managed_service_is_live(record: &ManagedServiceRecord) -> bool {
    record.is_live()
}

/// Idempotency guard for managed launches: returns an already-live managed
/// service for this engine+model, if any. The caller compares its effective
/// launch recipe before deciding whether reuse is safe.
///
/// Keyed on `(engine, canonical_model_id)` — NOT `service_id`. `generate_service_id`
/// embeds `unix_time_millis()`, so every launch mints a unique id; matching on it
/// would never catch a duplicate. `load_managed_services` refreshes liveness, so
/// stale manifests (dead PIDs) demote to "stopped" and are skipped, letting a
/// genuine relaunch proceed. Records are sorted newest-first, so `find` returns
/// the newest live match. Prevents a second `serve --managed` for the same
/// engine+model from spawning a duplicate process once the TUI job-bridge guard
/// has cleared.
pub(crate) fn existing_live_managed_service(
    paths: &AppPaths,
    engine: &str,
    canonical_model_id: &str,
) -> Option<ManagedServiceRecord> {
    find_live_managed_service(
        &load_managed_services(paths).ok()?,
        engine,
        canonical_model_id,
    )
    .cloned()
}

/// [`existing_live_managed_service`]'s match rule, over records already in hand.
///
/// Split out so a caller that has just loaded and liveness-refreshed the
/// registry can ask the same question again without a second disk scan, and so
/// the two spellings of "which record would be reused" cannot drift apart.
pub(crate) fn find_live_managed_service<'a>(
    records: &'a [ManagedServiceRecord],
    engine: &str,
    canonical_model_id: &str,
) -> Option<&'a ManagedServiceRecord> {
    records.iter().find(|record| {
        record.engine == engine
            && record.canonical_model_id == canonical_model_id
            && managed_service_is_live(record)
    })
}

/// The liveness-refreshed registry, returned only when one of its records is a
/// live service for `engine` that plausibly already serves `model_ref` —
/// decided from the records alone, with no canonical model id and so no engine
/// round-trip.
///
/// Pre-gate for the reuse detection in `serve`. Everything that check does is
/// real engine work: a `ResolveModel` round-trip and, for a self-managing
/// engine, an `ensure_self_managed_engine_ready` that may print
/// "Preparing <engine> for GPU serving..." and install. That work runs ahead of
/// the no-usable-GPU pre-flight, so it must be reserved for invocations that can
/// actually reuse something: keying on the engine alone let a live service for an
/// unrelated model pull an install in front of the bail on a GPU-less host.
///
/// Cheap only *relative* to what it guards — it is not free. `load_managed_services`
/// refreshes liveness for every record before this function's model filter is
/// applied, so each live record can cost an endpoint listing probe, an inference
/// probe, and a `record.write()`. It buys no engine round-trip and no install;
/// it does not buy "no I/O".
///
/// Hands the whole refreshed `Vec` back rather than a `bool` precisely because
/// that scan is not free: the caller's follow-up canonical-id lookup runs over
/// these records via [`find_live_managed_service`] instead of paying for the
/// same scan a second time. The authoritative check inside
/// `spawn_managed_engine_child` still re-reads from disk, and must: it runs
/// under the launch lock, after GPU selection, and detecting that this set went
/// stale in the meantime is the whole point of it.
///
/// Matching uses [`service_model_names_match`] — the same lenient relation the
/// service-listing surfaces already use to tie a user-typed name to a record — so
/// a short-vs-canonical spelling still reaches the probe. The probe then decides
/// reuse authoritatively on the canonical id; this only decides whether asking is
/// worth the engine round-trip.
pub(crate) fn live_managed_services_for_model(
    paths: &AppPaths,
    engine: &str,
    model_ref: &str,
) -> Option<Vec<ManagedServiceRecord>> {
    let records = load_managed_services(paths).ok()?;
    records
        .iter()
        .any(|record| {
            record.engine == engine
                && (service_model_names_match(&record.model_ref, model_ref)
                    || service_model_names_match(&record.canonical_model_id, model_ref))
                && managed_service_is_live(record)
        })
        .then_some(records)
}

const SERVICE_LIVENESS_CHECK_TIMEOUT: Duration = Duration::from_millis(750);

/// The real process ids a record tracks.
///
/// The launcher pid is always recorded; `engine_pid` is adopted once the engine
/// state file reports one. A `0` pid is a placeholder, never a real process.
pub(crate) fn recorded_service_pids(record: &ManagedServiceRecord) -> Vec<u32> {
    [record.engine_pid, Some(record.supervisor_pid)]
        .into_iter()
        .flatten()
        .filter(|pid| *pid != 0)
        .collect()
}

/// Complete a stop that was requested but could not confirm termination, once
/// the processes are actually observed gone: drop the endpoint key and clear the
/// marker. Returns whether the record changed.
///
/// This is deliberately keyed on `stop_requested_unix_ms` rather than on "the
/// tracked pids are dead". A *crashed* service also has dead pids, but its
/// operator never asked for it to go away and will want it back: dropping the
/// key there would make `rocm services restart` and daemon recovery refuse it
/// permanently — fail-closed guards have no way to re-mint a key. The cost of
/// this choice is that a crashed public service's 0600 key file outlives the
/// process until the service is stopped or successfully restarted.
///
/// Called before the liveness gate in
/// [`refresh_managed_service_runtime_liveness`], so a record that reached
/// "stopped" by another route still gets its deferred cleanup rather than
/// stranding the key.
fn settle_pending_stop_key_cleanup(paths: &AppPaths, record: &mut ManagedServiceRecord) -> bool {
    if record.stop_requested_unix_ms.is_none() {
        return false;
    }
    if recorded_service_pids(record)
        .iter()
        .any(|pid| process_is_running(*pid))
    {
        return false;
    }
    endpoint_keys::clear_endpoint_api_key(paths, &record.service_id);
    record.stop_requested_unix_ms = None;
    true
}

pub(crate) fn refresh_managed_service_runtime_liveness(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
) -> bool {
    let settled_pending_stop = settle_pending_stop_key_cleanup(paths, record);
    if !managed_service_is_live(record) {
        return settled_pending_stop;
    }

    // Probe with the service's key so a protected public service is not mistaken
    // for dead (an anonymous /v1/models would 401).
    let endpoint_api_key = endpoint_keys::endpoint_api_key(paths, &record.service_id);
    let outcome = if matches!(record.status.as_str(), "ready" | "running") {
        managed_service_endpoint_readiness(
            record,
            endpoint_api_key.as_deref(),
            SERVICE_LIVENESS_CHECK_TIMEOUT,
            rocm_core::INFERENCE_PROBE_TIMEOUT,
        )
    } else {
        EndpointReadinessOutcome {
            readiness: EndpointReadiness::Unreachable,
            record_changed: false,
        }
    };
    let readiness = outcome.readiness;
    if readiness == EndpointReadiness::Serving {
        // Mirror `providers.rs::ready_local_services()`: a live, probe-passing
        // service should be persisted as "ready", not left stuck at "running".
        if record.status == "running" {
            record.status = "ready".to_owned();
            return true;
        }
        // A newly latched verification is itself worth persisting, so the next
        // check reads it instead of sending another inference request.
        return settled_pending_stop || outcome.record_changed;
    }
    if readiness == EndpointReadiness::Listing {
        // The server is up and advertising the model; the weights are simply not
        // usable yet. Leave the status alone rather than demoting to "starting" —
        // `rocmd` restarts a service that sits in "starting" past its stale
        // window, which for a slow-loading model would restart it mid-load and
        // begin the wait all over again. Still report a change when the probe
        // bookkeeping moved, so the retry throttle is persisted and the next
        // process does not spend another full probe timeout.
        return settled_pending_stop || outcome.record_changed;
    }

    let tracked_pids = recorded_service_pids(record);
    let has_tracked_pid = !tracked_pids.is_empty();
    let has_live_pid = tracked_pids.iter().any(|pid| process_is_running(*pid));
    if has_tracked_pid && !has_live_pid {
        // Reconcile the status only. The endpoint key is *not* dropped here:
        // dead pids alone do not distinguish a stopped service from a crashed
        // one, and a crashed public service needs its key to be restartable.
        // `settle_pending_stop_key_cleanup` above owns that cleanup, gated on a
        // stop having actually been requested.
        if record.status != "stopped" {
            record.status = "stopped".to_owned();
            return true;
        }
        return settled_pending_stop;
    }

    if matches!(record.status.as_str(), "ready" | "running") {
        let new_status = if has_live_pid { "starting" } else { "stopped" };
        if record.status != new_status {
            record.status = new_status.to_owned();
            return true;
        }
    }

    settled_pending_stop
}

/// One-line header for the local-servers views.
///
/// Records that are neither ready nor starting used to be invisible here, so a
/// host with three failed servers and nothing live read "none ready" - true of
/// the live set, misleading about what is on disk. They now get their own
/// clause. The ready/starting wording is unchanged, so a header with no past
/// attempts reads exactly as before; one with past attempts gains a third
/// clause ("1 ready, 1 starting, 1 not running").
pub(crate) fn local_server_sidebar_status(counts: &ManagedServiceSidebarCounts) -> String {
    let mut parts = Vec::new();
    if counts.ready > 0 {
        parts.push(format!("{} ready", counts.ready));
    }
    if counts.starting > 0 {
        parts.push(format!("{} starting", counts.starting));
    }
    if counts.past_attempts > 0 {
        parts.push(format!("{} not running", counts.past_attempts));
    }
    if parts.is_empty() {
        "none ready".to_owned()
    } else {
        parts.join(", ")
    }
}

pub(crate) fn load_managed_services(paths: &AppPaths) -> Result<Vec<ManagedServiceRecord>> {
    let services_dir = paths.services_dir();
    if !services_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    for entry in fs::read_dir(&services_dir)
        .with_context(|| format!("failed to read {}", services_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if let Ok(mut record) = serde_json::from_slice::<ManagedServiceRecord>(&bytes) {
            record.normalize_paths_for_host();
            let refreshed_from_engine = record.refresh_from_engine_state().unwrap_or(false);
            let refreshed_liveness = refresh_managed_service_runtime_liveness(paths, &mut record);
            if refreshed_from_engine || refreshed_liveness {
                let _ = record.write();
            }
            records.push(record);
        }
    }

    records.sort_by_key(|record| std::cmp::Reverse(record.created_at_unix_ms));
    Ok(records)
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::services_text::load_managed_service;
    use crate::tests::test_paths;

    #[test]
    fn a_loading_service_keeps_its_status_instead_of_being_demoted() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // A model that is listed but cannot serve yet is coming up normally. It
        // must not be demoted to "starting": `rocmd` restarts a service that sits
        // in "starting" past its stale window, which would kill a slow-loading
        // model mid-load and start the wait over.
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        // Keeps listing the model for as long as the test asks, so a second
        // readiness check reaches the probe throttle rather than an unanswered
        // socket. Counts the inference requests that actually got sent.
        let probes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let server_probes = std::sync::Arc::clone(&probes);
        let server = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 1024];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                if request.starts_with("POST /v1/chat/completions ") {
                    server_probes.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let body = r#"{"error":"loading model"}"#;
                    let _ = write!(
                        stream,
                        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                } else {
                    let body = r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#;
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                }
            }
        });

        let (root, paths) = test_paths("liveness-loading-service");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-loading",
            "vllm",
            "Qwen3-0.6B-GGUF",
            "Qwen3-0.6B-GGUF",
            "127.0.0.1",
            port,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "running".to_owned();
        record.write()?;

        let changed = refresh_managed_service_runtime_liveness(&paths, &mut record);

        assert_eq!(
            record.status, "running",
            "a loading service keeps its status rather than being demoted"
        );
        assert!(
            record.inference_verified_at_unix_ms.is_none(),
            "nothing is latched until inference actually answers"
        );
        assert!(
            changed && record.inference_probe_attempted_at_unix_ms.is_some(),
            "the probe attempt is recorded and reported so the caller persists \
             the retry throttle"
        );

        // Second pass, inside the retry interval: the throttle holds, so the
        // model is not asked to generate again just because someone re-listed.
        let changed_again = refresh_managed_service_runtime_liveness(&paths, &mut record);
        assert!(!changed_again, "a throttled check changes nothing");
        assert_eq!(record.status, "running");
        assert_eq!(
            probes.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a warming service must not be re-probed by every poll"
        );

        drop(server);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn load_managed_services_promotes_running_to_ready_once_probe_passes() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        // Three requests hit this mock. `load_managed_services` lists the model
        // and then confirms inference, which promotes the record and latches the
        // verification; the `load_managed_service` re-read below (which proves the
        // promotion was persisted, not just returned in-memory) reads that latch
        // and so only re-lists.
        let server = thread::spawn(move || -> Result<Vec<String>> {
            let mut requests = Vec::new();
            for _ in 0..3 {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut request_bytes = Vec::new();
                let mut buffer = [0_u8; 1024];
                loop {
                    let read = stream.read(&mut buffer)?;
                    if read == 0 {
                        break;
                    }
                    request_bytes.extend_from_slice(&buffer[..read]);
                    if String::from_utf8_lossy(&request_bytes).contains("\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&request_bytes).into_owned();
                let body = if request.starts_with("POST /v1/chat/completions ") {
                    r#"{"choices":[{"message":{"content":"ok"}}]}"#
                } else {
                    r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
                requests.push(request);
            }
            Ok(requests)
        });

        let (root, paths) = test_paths("load-managed-services-promote-ready");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-qwen-promote",
            "vllm",
            "Qwen3-0.6B-GGUF",
            "Qwen3-0.6B-GGUF",
            "127.0.0.1",
            port,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        // A supervisor that has already observed the engine come up reports
        // "running"; only the HTTP model-ready probe should promote it further.
        record.status = "running".to_owned();
        record.write()?;

        let records = load_managed_services(&paths)?;
        let promoted = records
            .iter()
            .find(|found| found.service_id == "svc-qwen-promote")
            .expect("service should be present");
        assert_eq!(promoted.status, "ready");

        // Re-read the manifest file directly (bypassing any code path that
        // could itself re-run the promotion) to prove the transition was
        // actually written to disk, not just returned in the in-memory
        // `Vec<ManagedServiceRecord>` above. `load_managed_service` below
        // also calls `refresh_managed_service_runtime_liveness` on every
        // read, so asserting only on its return value would pass even if
        // `load_managed_services` never persisted anything.
        let on_disk_bytes = fs::read(&record.manifest_path)?;
        let on_disk = serde_json::from_slice::<ManagedServiceRecord>(&on_disk_bytes)?;
        assert_eq!(on_disk.status, "ready");

        // The promotion must have been persisted to disk, not just returned
        // in-memory, since chat's `pick_managed_chat_endpoint` re-reads it.
        let reloaded = load_managed_service(&paths, "svc-qwen-promote")?;
        assert_eq!(reloaded.status, "ready");
        assert!(
            reloaded.inference_verified_at_unix_ms.is_some(),
            "the inference verification is persisted with the promotion"
        );

        let requests = server.join().expect("server thread should not panic")?;
        let lines: Vec<&str> = requests
            .iter()
            .filter_map(|request| request.lines().next())
            .collect();
        assert_eq!(
            lines,
            vec![
                "GET /v1/models HTTP/1.1",
                "POST /v1/chat/completions HTTP/1.1",
                "GET /v1/models HTTP/1.1",
            ],
            "promotion confirms inference once; the re-read reads the latch"
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn duplicate_managed_launch_detected_across_distinct_service_ids() -> Result<()> {
        // `generate_service_id` embeds a timestamp, so a second launch for the
        // same engine+model has a DIFFERENT service_id. The guard must still
        // detect the live service by (engine, canonical_model_id), and return
        // the newest live match. `starting` skips the endpoint probe; the
        // current process id is a guaranteed-live PID.
        let (root, paths) = test_paths("dup-managed-distinct-ids");
        paths.ensure()?;

        // Older, dead manifest for the same engine+model (distinct service_id).
        let mut dead = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-1000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11500,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        dead.status = "ready".to_owned();
        dead.engine_pid = Some(999_999_999);
        dead.created_at_unix_ms = 1000;
        dead.write()?;

        // Newer, live manifest for the same engine+model (distinct service_id).
        let mut live = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-2000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11501,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        live.status = "starting".to_owned();
        live.engine_pid = Some(std::process::id());
        live.created_at_unix_ms = 2000;
        live.write()?;

        let found = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        let found = found.expect("a live managed service should be detected by engine+model");
        assert_eq!(
            found.service_id, "lemonade-qwen-2000",
            "should return the newest live match"
        );
        assert!(managed_service_is_live(&found));
        Ok(())
    }

    #[test]
    fn dead_managed_service_allows_relaunch() -> Result<()> {
        // A stale manifest with dead PIDs must NOT block a relaunch: liveness
        // refresh demotes it to "stopped", so the guard returns None.
        let (root, paths) = test_paths("dup-managed-dead");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-3000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11502,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.engine_pid = Some(999_999_999);
        record.write()?;

        let found = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        assert!(
            found.is_none(),
            "a dead managed service must not block relaunch"
        );
        Ok(())
    }

    #[test]
    fn live_service_for_other_model_does_not_block() -> Result<()> {
        // A live service for a DIFFERENT model must not match — the guard keys
        // on the model, not just the engine.
        let (root, paths) = test_paths("dup-managed-other-model");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-other-1",
            "lemonade",
            "other",
            "other-canonical",
            "127.0.0.1",
            11503,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "starting".to_owned();
        record.engine_pid = Some(std::process::id());
        record.write()?;

        let found = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        assert!(
            found.is_none(),
            "a live service for a different model must not match"
        );
        Ok(())
    }

    #[test]
    fn missing_manifest_allows_launch() {
        // No services dir / manifests → nothing to detect, launch proceeds.
        let (root, paths) = test_paths("dup-managed-missing");
        let found = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);
        assert!(found.is_none());
    }

    /// A public record with a dead pid and a stored endpoint key, for the
    /// liveness-refresh cases below. The port is one nothing listens on, so the
    /// refresh's endpoint probe fails and it falls through to the pid check.
    fn dead_public_service_with_key(
        paths: &AppPaths,
        service_id: &str,
        port: u16,
    ) -> ManagedServiceRecord {
        paths.ensure().unwrap();
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id,
            "vllm",
            "model-ref",
            "canonical/model",
            "0.0.0.0",
            port,
            "managed",
            // A pid far above any plausible live process, as in
            // `dead_managed_service_allows_relaunch`.
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "running".to_owned();
        record.write().unwrap();
        endpoint_keys::store_endpoint_api_key(paths, service_id, "secret-key").unwrap();
        record
    }

    #[test]
    fn crashed_public_service_keeps_its_endpoint_key() {
        // A crash (OOM kill, host reboot, panic) leaves the record at "running"
        // with dead pids and no stop marker. Dropping the key here would make
        // the fail-closed respawn guards refuse every later `rocm services
        // restart` and every daemon recovery attempt — permanently, because
        // nothing can re-mint the key. The refresh happens on every read
        // (`rocm services list`), so this must survive it.
        let (root, paths) = test_paths("liveness-crash-keeps-key");
        let service_id = "svc-crashed-public";
        let mut record = dead_public_service_with_key(&paths, service_id, 11982);

        let changed = refresh_managed_service_runtime_liveness(&paths, &mut record);

        assert!(changed, "a dead service must be demoted to stopped");
        assert_eq!(record.status, "stopped");
        assert_eq!(
            endpoint_keys::endpoint_api_key(&paths, service_id).as_deref(),
            Some("secret-key"),
            "a crashed public service must stay restartable"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unconfirmed_stop_clears_the_endpoint_key_once_the_processes_are_gone() {
        // The other half: a stop that could not confirm termination leaves the
        // key in place (the engine may still be alive and enforcing it) and
        // records the intent. Once the processes are observed gone, the deferred
        // cleanup runs, so no plaintext secret is stranded for a service the
        // operator did ask to stop.
        let (root, paths) = test_paths("liveness-pending-stop-clears-key");
        let service_id = "svc-pending-stop";
        let mut record = dead_public_service_with_key(&paths, service_id, 11983);
        record.stop_requested_unix_ms = Some(1);

        let changed = refresh_managed_service_runtime_liveness(&paths, &mut record);

        assert!(changed);
        assert_eq!(record.status, "stopped");
        assert_eq!(
            endpoint_keys::endpoint_api_key(&paths, service_id),
            None,
            "a requested stop must still drop the key"
        );
        assert_eq!(
            record.stop_requested_unix_ms, None,
            "the marker is consumed, so the cleanup does not run again"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pending_stop_cleanup_runs_even_once_the_record_reads_stopped() {
        // Proves the cleanup sits *before* the `managed_service_is_live` gate:
        // a record that reached "stopped" by another route (an engine state
        // refresh, a concurrent writer) would otherwise early-return and strand
        // the key of a service the operator stopped.
        let (root, paths) = test_paths("liveness-pending-stop-when-stopped");
        let service_id = "svc-pending-stop-stopped";
        let mut record = dead_public_service_with_key(&paths, service_id, 11984);
        record.status = "stopped".to_owned();
        record.stop_requested_unix_ms = Some(1);

        let changed = refresh_managed_service_runtime_liveness(&paths, &mut record);

        assert!(
            changed,
            "consuming the marker is a record change worth writing"
        );
        assert_eq!(endpoint_keys::endpoint_api_key(&paths, service_id), None);
        assert_eq!(record.stop_requested_unix_ms, None);
        let _ = fs::remove_dir_all(root);
    }

    /// Write one live managed record and report what the reuse pre-gate makes of
    /// a serve for `queried_model_ref`.
    fn reuse_pregate_for(
        label: &str,
        record_model_ref: &str,
        record_canonical_model_id: &str,
        queried_engine: &str,
        queried_model_ref: &str,
    ) -> Result<bool> {
        reuse_pregate_for_record(
            label,
            record_model_ref,
            record_canonical_model_id,
            queried_engine,
            queried_model_ref,
            "starting",
        )
    }

    /// As [`reuse_pregate_for`], but with the record's `status` chosen by the
    /// caller so the liveness arm of the pre-gate can be discriminated. Every
    /// other test plants a live record, which left `managed_service_is_live`
    /// deletable with the suite still green.
    fn reuse_pregate_for_record(
        label: &str,
        record_model_ref: &str,
        record_canonical_model_id: &str,
        queried_engine: &str,
        queried_model_ref: &str,
        record_status: &str,
    ) -> Result<bool> {
        let (root, paths) = test_paths(label);
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            format!("lemonade-{label}"),
            "lemonade",
            record_model_ref,
            record_canonical_model_id,
            "127.0.0.1",
            11_520,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = record_status.to_owned();
        record.engine_pid = Some(std::process::id());
        record.write()?;

        let gated =
            live_managed_services_for_model(&paths, queried_engine, queried_model_ref).is_some();
        let _ = fs::remove_dir_all(root);
        Ok(gated)
    }

    #[test]
    fn reuse_pregate_skips_engine_work_for_an_unrelated_live_model() -> Result<()> {
        // EAI-8059 review: the reuse probe runs a `ResolveModel` round-trip and,
        // for a self-managing engine, an install that prints "Preparing ... for
        // GPU serving" — all of it ahead of the no-usable-GPU bail. Keying the
        // pre-gate on the engine alone let a live service for a model this
        // invocation can never reuse pull that work in front of the bail. Gating
        // on the model keeps the fail-fast contract true.
        assert!(
            !reuse_pregate_for(
                "reuse-pregate-other-model",
                "other",
                "other-canonical",
                "lemonade",
                "qwen",
            )?,
            "a live service for an unrelated model must not unlock the reuse probe"
        );
        Ok(())
    }

    #[test]
    fn reuse_pregate_admits_the_same_model_including_a_short_spelling() -> Result<()> {
        // The probe must still be reached for anything that could genuinely be
        // reused, so the gate must not narrow reuse detection: an exact ref, a
        // canonical-id-only match, and a short-vs-canonical spelling all pass.
        // The probe then decides reuse authoritatively on the canonical id.
        assert!(
            reuse_pregate_for(
                "reuse-pregate-exact",
                "qwen",
                "qwen-canonical",
                "lemonade",
                "qwen",
            )?,
            "the same raw model ref must unlock the reuse probe"
        );
        assert!(
            reuse_pregate_for(
                "reuse-pregate-canonical",
                "some-alias",
                "Qwen/Qwen3-8B",
                "lemonade",
                "Qwen/Qwen3-8B",
            )?,
            "a canonical-id match must unlock the reuse probe"
        );
        assert!(
            reuse_pregate_for(
                "reuse-pregate-short",
                "Qwen/Qwen3-8B",
                "Qwen/Qwen3-8B",
                "lemonade",
                "qwen3-8b",
            )?,
            "a short spelling of the same model must unlock the reuse probe"
        );
        Ok(())
    }

    #[test]
    fn reuse_pregate_admits_the_model_id_the_reuse_scenario_plants() -> Result<()> {
        // `@id:serve-reuse-exempt-from-no-gpu-refusal` plants a live record for
        // this exact id and then serves it, relying on the reuse short-circuit
        // to reach the summary on a GPU-less host. That scenario only runs on
        // the `@requires-no-gpu` mock lane, so pin the pre-gate's verdict for
        // its literal model id here too: a matcher change that silently turned
        // the scenario into a launch attempt would otherwise only show up there.
        assert!(
            reuse_pregate_for(
                "reuse-pregate-e2e-reuse",
                "e2e/reuse-model",
                "e2e/reuse-model",
                "lemonade",
                "e2e/reuse-model",
            )?,
            "the reuse scenario's planted service must unlock the reuse probe"
        );
        Ok(())
    }

    #[test]
    fn reuse_pregate_admits_a_match_on_the_raw_model_ref_alone() -> Result<()> {
        // Discriminates the `record.model_ref` arm. Every other positive case
        // also matches on `canonical_model_id` (`"qwen-canonical"` contains
        // `"qwen"`, and the rest use identical fields), so deleting that arm
        // left the whole suite green. Here the canonical id shares nothing with
        // the query, so only the raw ref can open the gate — which is the case
        // that matters in practice: a user types the name they launched with
        // before the engine has reported any canonical id for it.
        assert!(
            reuse_pregate_for(
                "reuse-pregate-model-ref-only",
                "qwen",
                "unrelated-canonical-id",
                "lemonade",
                "qwen",
            )?,
            "a match on the record's raw model ref alone must unlock the reuse probe"
        );
        Ok(())
    }

    #[test]
    fn reuse_pregate_rejects_a_record_that_is_no_longer_live() -> Result<()> {
        // Discriminates the `managed_service_is_live(record)` arm. The shared
        // helper otherwise always plants a live record, so deleting that arm
        // also left the suite green. A stopped or failed service cannot be
        // reused, so letting its record open the gate would drag the
        // `ResolveModel` round-trip and any self-managed engine install in front
        // of the no-usable-GPU bail for nothing — exactly the fail-fast contract
        // the pre-gate exists to keep.
        for status in ["stopped", "failed"] {
            assert!(
                !reuse_pregate_for_record(
                    &format!("reuse-pregate-{status}"),
                    "qwen",
                    "qwen-canonical",
                    "lemonade",
                    "qwen",
                    status,
                )?,
                "a {status} service must not unlock the reuse probe"
            );
        }
        Ok(())
    }

    #[test]
    fn reuse_pregate_still_keys_on_the_engine() -> Result<()> {
        assert!(
            !reuse_pregate_for(
                "reuse-pregate-other-engine",
                "qwen",
                "qwen-canonical",
                "vllm",
                "qwen",
            )?,
            "a live service for a different engine must not unlock the reuse probe"
        );
        Ok(())
    }

    /// The canonical-id lookup the serve path runs over the pre-gate's own
    /// records must agree with the disk-reading guard the spawn uses, or the
    /// single-scan optimisation would have changed behaviour rather than cost.
    #[test]
    fn find_live_managed_service_agrees_with_the_disk_reading_guard() -> Result<()> {
        let (root, paths) = test_paths("find-live-matches-existing");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-find-live",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11_521,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "starting".to_owned();
        record.engine_pid = Some(std::process::id());
        record.write()?;

        let records = live_managed_services_for_model(&paths, "lemonade", "qwen")
            .expect("the planted live record opens the pre-gate");
        assert_eq!(
            find_live_managed_service(&records, "lemonade", "qwen-canonical")
                .map(|r| r.service_id.clone()),
            existing_live_managed_service(&paths, "lemonade", "qwen-canonical")
                .map(|r| r.service_id),
        );
        assert!(
            find_live_managed_service(&records, "lemonade", "not-this-model").is_none(),
            "the canonical-id term must still discriminate"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
