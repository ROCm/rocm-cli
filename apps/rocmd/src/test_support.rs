// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared test-only fixtures for `rocmd`'s module test suites.
//!
//! Every `#[cfg(test)] mod tests` in this crate that exercises `AppPaths`
//! needs the same workspace-local scratch directory under
//! `.rocm-work/tests/rocmd`. Keeping one copy here (rather than one per
//! module) avoids re-diverging these helpers as `lib.rs` continues to be
//! split into focused modules (see `docs/architecture.md`).

use anyhow::Result;
use rocm_core::{AppPaths, ManagedServiceRecord, unix_time_millis};
use std::fs;
use std::path::PathBuf;

pub(crate) fn temp_app_paths(name: &str) -> (PathBuf, AppPaths) {
    let root = unique_test_root(&format!(
        "rocmd-{name}-{}-{}",
        std::process::id(),
        unix_time_millis()
    ));
    let paths = AppPaths {
        config_dir: root.join("config"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
    };
    (root, paths)
}

pub(crate) fn unique_test_root(label: &str) -> PathBuf {
    let root = workspace_test_artifact_dir().join(label);
    fs::create_dir_all(&root).expect("create workspace-local test root");
    root
}

pub(crate) fn workspace_test_artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".rocm-work")
        .join("tests")
        .join("rocmd")
}

/// Build the record a supervisor would construct for `service_id`, to learn
/// the paths it derives before the real call does.
#[cfg(target_os = "linux")]
pub(crate) fn identity_probe_record(
    paths: &AppPaths,
    service_id: &str,
    port: u16,
) -> rocm_core::ManagedServiceRecord {
    rocm_core::ManagedServiceRecord::new(
        paths,
        service_id,
        "llamacpp",
        "a-model",
        "a-model",
        "127.0.0.1",
        port,
        "managed",
        std::process::id(),
        None,
        None,
        Some("gpu_required".to_owned()),
    )
}

/// Stop `service_id` with every recorded-PID termination reporting
/// `outcome`, without signalling anything.
///
/// An unconfirmed stop needs a process that survives `SIGKILL`, or a live
/// one whose start-time cannot be read; a test can create neither on demand.
/// This stands in for the identity check *and* the signalling — the whole of
/// `terminate_recorded_pid` — so it proves nothing about either. What it
/// does exercise for real is everything around them: the PID list, the
/// verdict, the manifest writes, the key cleanup and the report.
pub(crate) fn stop_with_outcome(
    paths: &AppPaths,
    service_id: &str,
    outcome: rocm_core::TerminationOutcome,
) -> Result<serde_json::Value> {
    crate::service::stop_managed_service_with(paths, service_id, |_| outcome)
}

/// The recorded PID for the unconfirmed-stop tests. Never signalled: those
/// tests force the termination outcome, so the value only has to be neither
/// zero nor the test process itself.
pub(crate) const UNCONFIRMED_STOP_PID: u32 = 999_999_999;

/// Seed a ready, publicly bound service that has an endpoint key on disk.
pub(crate) fn seed_keyed_service(paths: &AppPaths, service_id: &str, port: u16) -> Result<PathBuf> {
    let mut record = ManagedServiceRecord::new(
        paths,
        service_id,
        "vllm",
        "qwen",
        "Qwen/Qwen3.5",
        "0.0.0.0",
        port,
        "managed",
        UNCONFIRMED_STOP_PID,
        None,
        None,
        None,
    );
    record.status = "ready".to_owned();
    record.write()?;
    let key_path = rocm_engine_protocol::endpoint_key_file_path(paths, service_id);
    fs::create_dir_all(paths.services_dir())?;
    fs::write(&key_path, "secret-key")?;
    Ok(key_path)
}

/// The state an unconfirmed stop must leave behind: not marked stopped, the
/// PID kept so a later stop can still reach it, the deferred-cleanup marker
/// set, and the endpoint key still on disk.
pub(crate) fn assert_unconfirmed_stop_kept_the_service(
    reloaded: &ManagedServiceRecord,
    key_kept: bool,
) {
    assert_eq!(
        reloaded.status, "ready",
        "an unconfirmed stop must not mark the service stopped"
    );
    assert_eq!(
        reloaded.supervisor_pid, UNCONFIRMED_STOP_PID,
        "an unconfirmed stop must keep the PID a later stop needs"
    );
    assert!(
        reloaded.stop_requested_unix_ms.is_some(),
        "an unconfirmed stop must hand key cleanup to the liveness refresh"
    );
    assert!(key_kept, "an unconfirmed stop must keep the endpoint key");
}
