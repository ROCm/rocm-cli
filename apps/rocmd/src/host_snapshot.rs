// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use rocm_core::{
    AppPaths, AutomationRuntimeState, CodexBridgeEngine, CodexBridgeGpuSnapshot,
    CodexBridgeSnapshot, ExamineSummary, RocmCliConfig, default_engine_for_platform,
    load_recent_automation_events, resolve_amd_smi_binary, unix_time_millis,
};
use serde_json::Value;
use std::process::Command as ProcessCommand;
use std::time::Duration;

const AMD_SMI_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

pub(crate) fn build_bridge_snapshot(paths: &AppPaths) -> Result<CodexBridgeSnapshot> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    Ok(CodexBridgeSnapshot {
        protocol: "rocmd-codex-bridge-v0".to_owned(),
        generated_at_unix_ms: unix_time_millis(),
        examine: ExamineSummary::gather()?,
        gpu: gather_gpu_snapshot_for_config(&config),
        config,
        automation_runtime: AutomationRuntimeState::load(paths)?,
        recent_automation_events: load_recent_automation_events(paths, 32)?,
        engines: bridge_engine_inventory(),
        services: crate::persistence::load_managed_services(paths)?,
    })
}

fn gather_gpu_snapshot() -> CodexBridgeGpuSnapshot {
    let static_snapshot = match capture_amd_smi_json(&["static", "-a", "-g", "all", "--json"]) {
        Ok(value) => Some(value),
        Err(error) => {
            return CodexBridgeGpuSnapshot {
                // `{error:#}` rather than `to_string()`: the outermost context
                // alone says "failed to launch amd-smi ...", which names the
                // wrong thing when what actually happened was a timeout. Pinned
                // by `examine-25`, which asserts that the rendered note names
                // the timeout and not just the launch.
                //
                // Every failure `run_with_timeout` raises carries its cause the
                // same way, since all of them arrive under that one context:
                // the timeout, the failure to spawn, and the poll and reap
                // failures of its wait loop. Spawn is the case on a host where
                // none of the paths `resolve_amd_smi_binary` searches holds an
                // `amd-smi`, down to the bare name on `PATH`; the cause is then
                // the OS's own "not found" text, which is not the same string
                // on Linux and Windows, and `rocmd` runs this probe on both --
                // so none of it is quoted here, and no test pins it. A host
                // with no GPU but an amd-smi installed is a different case: it
                // spawns and fails at a non-zero exit. (The non-zero-exit and
                // parse failures below are their own outermost context and
                // already read correctly.)
                //
                // The note is the only account of any of this that reaches the
                // bridge snapshot and the automation watcher, so it carries the
                // causes too -- bounded, because it is persisted; see
                // `run_with_timeout`'s `MAX_QUOTED_OUTPUT_CHARS`.
                amd_smi_available: false,
                static_snapshot: None,
                monitor_snapshot: None,
                note: Some(format!("{error:#}")),
            };
        }
    };

    let monitor_snapshot = match capture_amd_smi_json(&[
        "monitor", "-p", "-t", "-u", "-m", "-v", "-g", "all", "--json",
    ]) {
        Ok(value) => Some(value),
        Err(error) => {
            return CodexBridgeGpuSnapshot {
                // Same reasoning as the `static` probe above, but a different
                // test: examine-25's stalled `static` probe returns before this
                // branch is ever reached, so reverting this `{error:#}` is
                // caught by `examine-26` alone. Pinned by `examine-26`.
                amd_smi_available: true,
                static_snapshot,
                monitor_snapshot: None,
                note: Some(format!("{error:#}")),
            };
        }
    };

    CodexBridgeGpuSnapshot {
        amd_smi_available: true,
        static_snapshot,
        monitor_snapshot,
        note: None,
    }
}

pub(crate) fn gather_gpu_snapshot_for_config(config: &RocmCliConfig) -> CodexBridgeGpuSnapshot {
    if config.telemetry.local_inspection_enabled() {
        gather_gpu_snapshot()
    } else {
        CodexBridgeGpuSnapshot {
            amd_smi_available: false,
            static_snapshot: None,
            monitor_snapshot: None,
            note: Some(
                "gpu telemetry is disabled by rocm-cli config; no external reporting is implemented"
                    .to_owned(),
            ),
        }
    }
}

fn capture_amd_smi_json(args: &[&str]) -> Result<Value> {
    let amd_smi_binary = resolve_amd_smi_binary();
    let mut command = ProcessCommand::new(&amd_smi_binary);
    // No stdio here: `run_with_timeout` sets null stdin and piped stdout/stderr
    // itself, since capturing both is its contract.
    command.args(args);
    let output = rocm_core::process::run_with_timeout(command, AMD_SMI_PROBE_TIMEOUT, "process")
        .with_context(|| format!("failed to launch amd-smi {}", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        anyhow::bail!(
            "amd-smi {} failed: {}",
            args.join(" "),
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            }
        );
    }

    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to parse amd-smi {} json", args.join(" ")))
}

pub(crate) fn bridge_engine_inventory() -> Vec<CodexBridgeEngine> {
    let default_engine = default_engine_for_platform();
    let current_exe = std::env::current_exe().ok();
    rocmd_engine_inventory()
        .iter()
        .map(|(id, summary)| CodexBridgeEngine {
            id: (*id).to_owned(),
            summary: (*summary).to_owned(),
            default_for_platform: *id == default_engine,
            installed_binary: true,
            binary_path: current_exe.as_ref().map(|path| path.display().to_string()),
        })
        .collect()
}

const fn rocmd_engine_inventory() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "lemonade",
            "embedded Lemonade server with ROCm llama.cpp backend",
        ),
        (
            "vllm",
            "Linux/WSL ROCm GPU serving engine through external vLLM",
        ),
        (
            "aim",
            "AMD Enterprise AI pre-built Docker serving containers",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_snapshot_respects_disabled_telemetry_policy() {
        let mut config = RocmCliConfig::default();
        config.telemetry.mode = rocm_core::TELEMETRY_MODE_OFF.to_owned();

        let snapshot = gather_gpu_snapshot_for_config(&config);

        assert!(!snapshot.amd_smi_available);
        assert!(snapshot.static_snapshot.is_none());
        assert!(snapshot.monitor_snapshot.is_none());
        assert!(
            snapshot
                .note
                .as_deref()
                .is_some_and(|note| note.contains("disabled by rocm-cli config"))
        );
    }
}
