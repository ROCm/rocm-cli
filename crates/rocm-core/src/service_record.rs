// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `ManagedServiceRecord`: the on-disk manifest for a managed inference service.
//!
//! Identity, endpoint, process-identity (PID + start-ticks for safe stop /
//! PID-reuse protection), and inference-readiness bookkeeping.
//!
//! Used by `apps/rocm` and `apps/rocmd`; `rocm-dash-daemon` also reads this
//! on-disk format at runtime, through its own `ServiceRecord` mirror struct
//! in `registry.rs` rather than this type, and the serde test guards that
//! contract. `rocm-engine-protocol` mentions `ManagedServiceRecord` solely in
//! a doc comment — `rocm-dash-core` and `rocm-dash-collectors` do not depend
//! on `rocm-core` at all.

use anyhow::{Context, Result};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

use crate::AppPaths;
use crate::INFERENCE_VERIFIED_STATE_KEY;
use crate::format_http_base_url;
use crate::unix_time_millis;
use crate::{normalize_runtime_path_for_host, normalize_runtime_path_for_storage};

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ManagedServiceRecord {
    pub service_id: String,
    pub engine: String,
    pub model_ref: String,
    pub canonical_model_id: String,
    pub host: String,
    pub port: u16,
    pub endpoint_url: String,
    pub mode: String,
    pub status: String,
    pub supervisor_pid: u32,
    pub engine_pid: Option<u32>,
    /// Kernel start-time of the supervisor (launcher) process, captured at
    /// spawn. Paired with `supervisor_pid` it forms an identity that survives
    /// PID recycling, so a later stop never signals a reused PID. `None` for
    /// records written before this field existed.
    #[serde(default)]
    pub supervisor_start_ticks: Option<u64>,
    /// Kernel start-time of the engine server process, adopted from the engine
    /// state file whenever `engine_pid` is refreshed from it. The launcher and
    /// the engine server are distinct processes, so each PID carries its own
    /// identity token. `None` until the engine state records one.
    #[serde(default)]
    pub engine_start_ticks: Option<u64>,
    /// Whether this service must never come back up without an endpoint key.
    ///
    /// The bind address alone cannot answer that. A service bound to loopback is
    /// unreachable from elsewhere *until something republishes the port* — a
    /// tailnet publish, a proxy, a container port map — and the publish outlives
    /// the process. So the requirement has to be recorded next to the service and
    /// survive a restart, exactly as the key itself does.
    #[serde(default)]
    pub requires_api_key: bool,
    #[serde(default)]
    pub runtime_id: Option<String>,
    #[serde(default)]
    pub env_id: Option<String>,
    #[serde(default)]
    pub device_policy: Option<String>,
    #[serde(default)]
    pub gpu_indices: Vec<u32>,
    #[serde(default)]
    pub engine_recipe_json: Option<String>,
    #[serde(default)]
    pub restart_count: u32,
    #[serde(default)]
    pub last_restart_unix_ms: Option<u128>,
    /// When a stop was requested but could not confirm that every recorded
    /// process died. It records *intent*: the operator asked for this service to
    /// go away, so once the processes are observed gone the endpoint key may be
    /// dropped. A service that merely crashed carries no such intent and keeps
    /// its key, so it stays restartable/recoverable. Cleared on a confirmed stop
    /// and on a successful respawn. `None` on records written before this field
    /// existed, which is the safe default (keep the key).
    #[serde(default)]
    pub stop_requested_unix_ms: Option<u128>,
    /// When the last inference probe was attempted. Throttles re-probing of a
    /// service that is listed but still loading — see
    /// [`crate::INFERENCE_PROBE_RETRY_INTERVAL`]. Absent on records written before
    /// readiness was gated on inference.
    #[serde(default)]
    pub inference_probe_attempted_at_unix_ms: Option<u64>,
    /// Coarse startup stage (`downloading`/`loading`/`warmup`) parsed from the
    /// serve process's own log output while it is coming up. Set to `None` once
    /// the service reaches `ready`, and absent on older on-disk records.
    #[serde(default)]
    pub startup_phase: Option<String>,
    /// When a real inference request first succeeded against this service. Once
    /// set, readiness checks stop re-probing and fall back to the cheap endpoint
    /// query — see [`crate::managed_service_endpoint_readiness`]. Adopted from the
    /// engine state file when present, and absent on records written before
    /// readiness was gated on inference.
    #[serde(default)]
    pub inference_verified_at_unix_ms: Option<u64>,
    pub manifest_path: PathBuf,
    pub log_path: PathBuf,
    pub engine_state_path: PathBuf,
    pub created_at_unix_ms: u128,
}

impl ManagedServiceRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        paths: &AppPaths,
        service_id: impl Into<String>,
        engine: impl Into<String>,
        model_ref: impl Into<String>,
        canonical_model_id: impl Into<String>,
        host: impl Into<String>,
        port: u16,
        mode: impl Into<String>,
        supervisor_pid: u32,
        runtime_id: Option<String>,
        env_id: Option<String>,
        device_policy: Option<String>,
    ) -> Self {
        let service_id = service_id.into();
        let engine = engine.into();
        let host = host.into();
        let manifest_path = paths.service_manifest_path(&service_id);
        let log_path = paths.service_log_path(&service_id);
        let engine_state_path = paths.service_engine_state_path(&engine, &service_id);
        Self {
            endpoint_url: format!("{}/v1", format_http_base_url(&host, port)),
            service_id,
            engine,
            model_ref: model_ref.into(),
            canonical_model_id: canonical_model_id.into(),
            host,
            port,
            mode: mode.into(),
            status: "starting".to_owned(),
            supervisor_pid,
            engine_pid: None,
            supervisor_start_ticks: None,
            engine_start_ticks: None,
            // Off unless a caller says otherwise: local loopback serving stays
            // credential-free, which is the unchanged default.
            requires_api_key: false,
            runtime_id,
            env_id,
            device_policy,
            gpu_indices: Vec::new(),
            engine_recipe_json: None,
            restart_count: 0,
            last_restart_unix_ms: None,
            stop_requested_unix_ms: None,
            startup_phase: None,
            inference_verified_at_unix_ms: None,
            inference_probe_attempted_at_unix_ms: None,
            manifest_path,
            log_path,
            engine_state_path,
            created_at_unix_ms: unix_time_millis(),
        }
    }

    /// Drop the per-run state that a restart invalidates, and count the restart.
    ///
    /// A restart reuses this manifest but spawns a different server with an
    /// unloaded model, so anything describing the previous run has to go. Chiefly
    /// the inference verification: left set, it short-circuits readiness straight
    /// back to "ready" as soon as the new server lists the model, reinstating the
    /// false positive the probe exists to prevent. The engine's own state file is
    /// rewritten from scratch on restart, so only this copy needs clearing — and
    /// [`Self::refresh_from_engine_state`] only ever adopts a verification, never
    /// clears one, so a stale value here would survive indefinitely.
    pub fn reset_for_restart(&mut self) {
        self.inference_verified_at_unix_ms = None;
        self.inference_probe_attempted_at_unix_ms = None;
        self.restart_count = self.restart_count.saturating_add(1);
        self.last_restart_unix_ms = Some(unix_time_millis());
    }

    /// Record `pid` as the supervisor, together with its start-time token.
    ///
    /// Call it while the process is known to be alive — just after spawning
    /// it. A stop verifies a recorded PID against its token before signalling
    /// it, and [`crate::identity_state`] treats a PID recorded *without* one as
    /// a match, so a PID written on its own is signalled on trust alone once it
    /// has been recycled. Writing the two together is what keeps that from
    /// happening at whichever site records a PID next.
    pub fn record_supervisor_identity(&mut self, pid: u32) {
        self.supervisor_pid = pid;
        self.supervisor_start_ticks = crate::process_start_ticks(pid);
    }

    /// Whether `self` and `other` — two snapshots of one service's record —
    /// name the same processes, token for token.
    ///
    /// A stop reads the record, signals what it names, and only then writes its
    /// verdict. If the record names different processes by then — a restart
    /// recorded its new supervisor meanwhile — the verdict does not cover them,
    /// and both `rocm services stop` and `rocmd`'s stop gate their "stopped"
    /// claim on this.
    pub fn names_same_processes(&self, other: &Self) -> bool {
        self.supervisor_pid == other.supervisor_pid
            && self.supervisor_start_ticks == other.supervisor_start_ticks
            && self.engine_pid == other.engine_pid
            && self.engine_start_ticks == other.engine_start_ticks
    }

    /// Record `pid` as the engine, together with its start-time token. See
    /// [`Self::record_supervisor_identity`].
    pub fn record_engine_identity(&mut self, pid: u32) {
        self.engine_pid = Some(pid);
        self.engine_start_ticks = crate::process_start_ticks(pid);
    }

    pub fn normalize_paths_for_host(&mut self) {
        self.manifest_path = normalize_runtime_path_for_host(&self.manifest_path);
        self.log_path = normalize_runtime_path_for_host(&self.log_path);
        self.engine_state_path = normalize_runtime_path_for_host(&self.engine_state_path);
    }

    pub fn refresh_from_engine_state(&mut self) -> Result<bool> {
        if !matches!(
            self.status.as_str(),
            "starting" | "running" | "recovering" | "ready"
        ) {
            return Ok(false);
        }
        self.normalize_paths_for_host();
        if !self.engine_state_path.is_file() {
            return Ok(false);
        }
        let bytes = fs::read(&self.engine_state_path)
            .with_context(|| format!("failed to read {}", self.engine_state_path.display()))?;
        let state = serde_json::from_slice::<serde_json::Value>(&bytes)
            .with_context(|| format!("failed to parse {}", self.engine_state_path.display()))?;
        let Some(status) = state
            .get("status")
            .and_then(serde_json::Value::as_str)
            .filter(|value| matches!(*value, "ready" | "running" | "starting" | "failed"))
        else {
            return Ok(false);
        };

        let previous = self.status.clone();
        status.clone_into(&mut self.status);
        if let Some(endpoint_url) = state
            .get("endpoint_url")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            endpoint_url.clone_into(&mut self.endpoint_url);
        }
        // `requested_runtime_id` is the selector the launch asked for: an exact
        // runtime key or a family id, whichever was requested. `runtime_id` is
        // only what the engine resolved that to, which for a TheRock runtime is
        // the manifest's family id (`therock-release:gfx120X-all`), shared by
        // every installed version of that family. When the launch pinned an
        // exact key, adopting the resolved form would throw away the version
        // the service is really on, and the record is what a restart re-pins
        // and what runtime activation compares against. Prefer the requested
        // value, falling back to the resolved one for engines that record only
        // that. A blank value in either field is ignored.
        if let Some(runtime_id) = ["requested_runtime_id", "runtime_id"]
            .into_iter()
            .filter_map(|key| state.get(key).and_then(serde_json::Value::as_str))
            .find(|value| !value.trim().is_empty())
        {
            self.runtime_id = Some(runtime_id.to_owned());
        }
        if let Some(env_id) = state
            .get("env_id")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            self.env_id = Some(env_id.to_owned());
        }
        // Adopt the engine server PID together with ITS OWN start-time token so
        // a later stop verifies the server process, not the launcher. The ticks
        // key must match the PID key: `server_pid`↔`server_start_ticks`,
        // `pid`↔`start_ticks`.
        let engine_pid = state
            .get("server_pid")
            .and_then(serde_json::Value::as_u64)
            .map(|pid| (pid, "server_start_ticks"))
            .or_else(|| {
                state
                    .get("pid")
                    .and_then(serde_json::Value::as_u64)
                    .map(|pid| (pid, "start_ticks"))
            });
        if let Some((pid, ticks_key)) = engine_pid
            && let Ok(engine_pid) = u32::try_from(pid)
        {
            self.engine_pid = Some(engine_pid);
            // When `server_pid` names the same process as `pid`, `start_ticks`
            // is that process's token too. vLLM wrote its state that way —
            // `server_pid` equal to `pid`, and only `start_ticks` — before it
            // wrote `server_start_ticks`, so without this the PID of every such
            // service was adopted with no token, and a stop could only signal
            // it blind once the PID had been recycled.
            self.engine_start_ticks = state
                .get(ticks_key)
                .and_then(serde_json::Value::as_u64)
                .or_else(|| {
                    (state.get("pid").and_then(serde_json::Value::as_u64) == Some(pid))
                        .then(|| state.get("start_ticks").and_then(serde_json::Value::as_u64))
                        .flatten()
                });
        }
        // Adopt the engine's inference verification so the CLI side does not
        // re-probe a service the engine healthcheck already confirmed.
        if self.inference_verified_at_unix_ms.is_none()
            && let Some(verified_at) = state
                .get(INFERENCE_VERIFIED_STATE_KEY)
                .and_then(serde_json::Value::as_u64)
        {
            self.inference_verified_at_unix_ms = Some(verified_at);
        }
        Ok(self.status != previous)
    }

    fn with_storage_paths(&self) -> Self {
        let mut record = self.clone();
        record.manifest_path = normalize_runtime_path_for_storage(&record.manifest_path);
        record.log_path = normalize_runtime_path_for_storage(&record.log_path);
        record.engine_state_path = normalize_runtime_path_for_storage(&record.engine_state_path);
        record
    }

    pub fn write(&self) -> Result<()> {
        let mut host_record = self.clone();
        host_record.normalize_paths_for_host();
        let parent = host_record
            .manifest_path
            .parent()
            .context("service manifest path must have a parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let storage_record = host_record.with_storage_paths();
        let bytes = serde_json::to_vec_pretty(&storage_record)
            .context("failed to serialize service record")?;
        crate::write_atomic(&host_record.manifest_path, &bytes, "json").with_context(|| {
            format!(
                "failed to write service manifest {}",
                host_record.manifest_path.display()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{probe_test_record, temp_app_paths};

    #[test]
    fn restarting_drops_the_previous_runs_inference_verification() {
        // The restarted child is a different server with an unloaded model. If the
        // verification carried over, readiness would short-circuit to "ready" the
        // moment the new server listed the model — the original false positive,
        // reinstated. `refresh_from_engine_state` only ever adopts a verification,
        // so nothing downstream would clear it.
        let mut record = probe_test_record(11435);
        record.inference_verified_at_unix_ms = Some(1);
        record.inference_probe_attempted_at_unix_ms = Some(1);
        record.restart_count = 2;

        record.reset_for_restart();

        assert_eq!(record.inference_verified_at_unix_ms, None);
        assert_eq!(
            record.inference_probe_attempted_at_unix_ms, None,
            "the retry throttle is per-run too; the new child deserves an              immediate first probe"
        );
        assert_eq!(record.restart_count, 3);
        assert!(record.last_restart_unix_ms.is_some());
    }

    /// Writes `state` as the record's engine state file and runs the refresh.
    fn refresh_from_state(name: &str, state: &serde_json::Value) -> ManagedServiceRecord {
        let dir = std::env::temp_dir().join(format!(
            "rocm-core-engine-state-{name}-{}-{}",
            std::process::id(),
            crate::unix_time_millis()
        ));
        fs::create_dir_all(&dir).expect("create state dir");
        let mut record = probe_test_record(11437);
        record.status = "running".to_owned();
        record.engine_state_path = dir.join("state.json");
        fs::write(
            &record.engine_state_path,
            serde_json::to_vec(state).expect("serialize state"),
        )
        .expect("write state");
        record.refresh_from_engine_state().expect("refresh");
        let _ = fs::remove_dir_all(dir);
        record
    }

    /// vLLM records one process as both `pid` and `server_pid` with only
    /// `start_ticks`. The refresh prefers `server_pid`, so it has to find that
    /// process's token under `start_ticks`, or the PID is adopted with none.
    #[test]
    fn engine_refresh_takes_start_ticks_when_server_pid_is_the_same_process() {
        let record = refresh_from_state(
            "same-pid",
            &serde_json::json!({
                "status": "running",
                "pid": 4321,
                "server_pid": 4321,
                "start_ticks": 987_u64,
            }),
        );
        assert_eq!(record.engine_pid, Some(4321));
        assert_eq!(record.engine_start_ticks, Some(987));
    }

    /// The fallback is only for the same process: a distinct server PID never
    /// borrows the launcher's token, which would make the stop's identity check
    /// refute the server and leave it running.
    #[test]
    fn engine_refresh_never_gives_a_server_pid_the_launchers_token() {
        let record = refresh_from_state(
            "distinct-pid",
            &serde_json::json!({
                "status": "running",
                "pid": 100,
                "server_pid": 200,
                "start_ticks": 111_u64,
            }),
        );
        assert_eq!(record.engine_pid, Some(200));
        assert_eq!(record.engine_start_ticks, None);

        let record = refresh_from_state(
            "distinct-pid-own-token",
            &serde_json::json!({
                "status": "running",
                "pid": 100,
                "server_pid": 200,
                "start_ticks": 111_u64,
                "server_start_ticks": 222_u64,
            }),
        );
        assert_eq!(record.engine_start_ticks, Some(222));
    }

    /// Each role is written with its own PID's token, so a stop can verify it.
    /// A role recorded without one is signalled on trust alone once its PID has
    /// been recycled. One live process stands in for both roles, as it does for
    /// a `rocm serve --background` launch.
    ///
    /// Linux-only: `process_start_ticks` reads `/proc`, and is `None` elsewhere.
    #[cfg(target_os = "linux")]
    #[test]
    fn recording_a_process_identity_writes_the_pid_with_its_own_token() {
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn child");
        let pid = child.id();
        let expected = crate::process_start_ticks(pid);
        let mut record = probe_test_record(11436);

        record.record_supervisor_identity(pid);
        record.record_engine_identity(pid);
        let _ = child.kill();
        let _ = child.wait();

        assert!(expected.is_some(), "precondition: a live child has a token");
        assert_eq!(record.supervisor_pid, pid);
        assert_eq!(record.supervisor_start_ticks, expected);
        assert_eq!(record.engine_pid, Some(pid));
        assert_eq!(record.engine_start_ticks, expected);
    }

    #[test]
    fn engine_state_refresh_keeps_the_runtime_selector_the_launch_requested() -> Result<()> {
        let (root, paths) = temp_app_paths("record-runtime-pin");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-runtime-pin",
            "vllm",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-0.6B",
            "127.0.0.1",
            9,
            "managed",
            std::process::id(),
            Some("release-pip-gfx120x-all-7-13-0".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        fs::create_dir_all(
            record
                .engine_state_path
                .parent()
                .expect("engine state path has a parent"),
        )?;
        // What a vLLM server writes: `runtime_id` is the manifest's family id,
        // shared by every installed version of that family, while
        // `requested_runtime_id` is the selector the launch asked for, here an
        // exact runtime key.
        fs::write(
            &record.engine_state_path,
            serde_json::to_vec(&serde_json::json!({
                "status": "running",
                "runtime_id": "therock-release:gfx120X-all",
                "requested_runtime_id": "release-pip-gfx120x-all-7-13-0",
            }))?,
        )?;

        let refreshed = record.refresh_from_engine_state()?;
        let _ = fs::remove_dir_all(&root);

        assert!(refreshed);
        assert_eq!(
            record.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0"),
            "adopting the resolved family id would throw away which installed \
             version the service is actually on — the record is what a restart \
             re-pins and what runtime activation compares against"
        );
        Ok(())
    }

    #[test]
    fn engine_state_refresh_falls_back_to_the_resolved_runtime_id() -> Result<()> {
        let (root, paths) = temp_app_paths("record-runtime-fallback");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-runtime-fallback",
            "lemonade",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-0.6B",
            "127.0.0.1",
            9,
            "managed",
            std::process::id(),
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        fs::create_dir_all(
            record
                .engine_state_path
                .parent()
                .expect("engine state path has a parent"),
        )?;
        // An engine that records only the resolved runtime must still be read:
        // preferring `requested_runtime_id` must not mean ignoring `runtime_id`.
        fs::write(
            &record.engine_state_path,
            serde_json::to_vec(&serde_json::json!({
                "status": "running",
                "runtime_id": "lemonade-embeddable-8.1.11",
                "requested_runtime_id": serde_json::Value::Null,
            }))?,
        )?;

        record.refresh_from_engine_state()?;
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            record.runtime_id.as_deref(),
            Some("lemonade-embeddable-8.1.11")
        );
        Ok(())
    }

    #[test]
    fn engine_state_refresh_ignores_a_blank_requested_runtime_id() -> Result<()> {
        // A blank selector names no runtime, so adopting it would leave the
        // record with an empty runtime that matches nothing and that a restart
        // would re-pin. Both an empty string and whitespace must fall through
        // to the resolved `runtime_id`.
        for (blank, name) in [("", "empty"), ("   ", "whitespace")] {
            let (root, paths) = temp_app_paths(&format!("record-runtime-blank-{name}"));
            let mut record = ManagedServiceRecord::new(
                &paths,
                "svc-runtime-blank",
                "lemonade",
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3-0.6B",
                "127.0.0.1",
                9,
                "managed",
                std::process::id(),
                None,
                None,
                Some("gpu_required".to_owned()),
            );
            fs::create_dir_all(
                record
                    .engine_state_path
                    .parent()
                    .expect("engine state path has a parent"),
            )?;
            fs::write(
                &record.engine_state_path,
                serde_json::to_vec(&serde_json::json!({
                    "status": "running",
                    "runtime_id": "lemonade-embeddable-8.1.11",
                    "requested_runtime_id": blank,
                }))?,
            )?;

            let refreshed = record.refresh_from_engine_state()?;
            let _ = fs::remove_dir_all(&root);

            assert!(refreshed, "{name}: the engine state must still be adopted");
            assert_eq!(
                record.runtime_id.as_deref(),
                Some("lemonade-embeddable-8.1.11"),
                "{name}: a blank requested_runtime_id must fall through to runtime_id"
            );
        }
        Ok(())
    }
}
