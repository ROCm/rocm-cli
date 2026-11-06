// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The host snapshot behind the internal bridge surface:
//! `build_codex_bridge_snapshot` assembles a `rocm_core::CodexBridgeSnapshot`
//! from the examination, config, automation state and managed services,
//! `build_codex_bridge_gpu_snapshot` builds its GPU part, and
//! `builtin_codex_bridge_engine_inventory` its engine list, which it derives
//! from `main.rs`'s `engine_inventory`, the single table of built-in engine
//! ids and summaries that `rocm examine`, `rocm update` and the `main.rs`
//! test checking `--engine`'s help reads too.
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. `dispatch()`'s `InternalBridgeSnapshot` arm calls
//! `build_codex_bridge_snapshot` through a re-import in `main.rs`, and
//! `chat.rs`'s internal MCP tools (`bridge_snapshot`, `gpu_snapshot` and
//! `engines`) reach `build_codex_bridge_snapshot`,
//! `build_codex_bridge_gpu_snapshot` and `builtin_codex_bridge_engine_inventory`
//! as `crate::<name>` through the same re-import, so those three are
//! `pub(crate)`.

use anyhow::Result;
use rocm_core::{
    AppPaths, AutomationRuntimeState, CodexBridgeEngine, CodexBridgeGpuSnapshot,
    CodexBridgeSnapshot, ExamineSummary, RocmCliConfig, daemon_binary_path,
    default_engine_for_platform, load_recent_automation_events,
};

use crate::engine_inventory;
use crate::managed_service_status::load_managed_services;

pub(crate) fn build_codex_bridge_snapshot(paths: &AppPaths) -> Result<CodexBridgeSnapshot> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    Ok(CodexBridgeSnapshot {
        protocol: "rocmd-codex-bridge-v0".to_owned(),
        generated_at_unix_ms: rocm_core::unix_time_millis(),
        examine: ExamineSummary::gather()?,
        gpu: build_codex_bridge_gpu_snapshot(&config),
        config,
        automation_runtime: AutomationRuntimeState::load(paths)?,
        recent_automation_events: load_recent_automation_events(paths, 32)?,
        engines: builtin_codex_bridge_engine_inventory(),
        services: load_managed_services(paths)?,
    })
}

pub(crate) fn build_codex_bridge_gpu_snapshot(config: &RocmCliConfig) -> CodexBridgeGpuSnapshot {
    if !config.telemetry.local_inspection_enabled() {
        return CodexBridgeGpuSnapshot {
            amd_smi_available: false,
            static_snapshot: None,
            monitor_snapshot: None,
            note: Some("GPU telemetry is disabled by rocm-cli config.".to_owned()),
        };
    }

    CodexBridgeGpuSnapshot {
        amd_smi_available: false,
        static_snapshot: None,
        monitor_snapshot: None,
        note: Some("Use `rocm examine` for the current local AMD GPU summary.".to_owned()),
    }
}

pub(crate) fn builtin_codex_bridge_engine_inventory() -> Vec<CodexBridgeEngine> {
    let default_engine = default_engine_for_platform();
    let binary_path = daemon_binary_path()
        .ok()
        .map(|path| path.display().to_string());
    engine_inventory()
        .iter()
        .map(|(id, summary)| CodexBridgeEngine {
            id: (*id).to_owned(),
            summary: (*summary).to_owned(),
            default_for_platform: *id == default_engine,
            installed_binary: true,
            binary_path: binary_path.clone(),
        })
        .collect()
}
