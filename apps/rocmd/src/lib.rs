// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#![allow(clippy::items_after_test_module)]

mod cli;
mod common;
mod mcp;
mod persistence;
mod sandbox;
mod service;
#[cfg(test)]
mod test_support;
mod watchers;
mod webhook;

pub use cli::{run_bin_cli, run_from_args};

use std::time::Duration;

const WATCHER_TICK_INTERVAL: Duration = Duration::from_secs(5);
const ARTIFACT_PREFETCH_TIMEOUT: Duration = Duration::from_mins(10);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::sandbox_stop_server_value;
    use crate::test_support::{
        assert_unconfirmed_stop_kept_the_service, seed_keyed_service, stop_with_outcome,
        temp_app_paths,
    };
    use crate::watchers::load_service_record;
    use anyhow::Result;
    use rocm_core::ManagedServiceRecord;
    use serde_json::Value;
    use std::fs;

    #[test]
    fn sandbox_tool_stop_server_updates_manifest_and_skips_current_pid() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-stop-current-pid");
        paths.ensure()?;
        let current_pid = std::process::id();
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-current",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            current_pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(current_pid);
        record.status = "ready".to_owned();
        record.write()?;

        let value = sandbox::run_sandbox_tool(
            &paths,
            cli::SandboxToolArg::StopServer,
            Some("svc-current".to_owned()),
            None,
            None,
            cli::SandboxToolPolicy::default(),
        )?;
        let reloaded = watchers::load_service_record(&paths, "svc-current")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("stopped"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(true));
        assert_eq!(reloaded.status, "stopped");
        assert!(
            value
                .get("result")
                .and_then(|result| result.get("skipped_pids"))
                .and_then(Value::as_array)
                .is_some_and(|pids| pids
                    .iter()
                    .any(|pid| pid.as_u64() == Some(u64::from(current_pid))))
        );
        Ok(())
    }

    /// The sandbox tool used to report `status: stopped` unconditionally.
    #[test]
    fn sandbox_tool_stop_server_reports_an_unconfirmed_stop() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-stop-unconfirmed");
        paths.ensure()?;
        let service_id = "svc-sandbox-stop-unconfirmed";
        let key_path = seed_keyed_service(&paths, service_id, 11448)?;

        // The tool is `stop_managed_service` feeding `sandbox_stop_server_value`;
        // `sandbox_tool_stop_server_updates_manifest_and_skips_current_pid`
        // drives that whole tool for real.
        let result = stop_with_outcome(
            &paths,
            service_id,
            rocm_core::TerminationOutcome::Unverified,
        )
        .map(sandbox_stop_server_value);
        let key_kept = key_path.exists();
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("stop_unconfirmed"),
            "{value}"
        );
        assert_eq!(
            value.pointer("/result/stopped").and_then(Value::as_bool),
            Some(false),
            "{value}"
        );
        assert_unconfirmed_stop_kept_the_service(&reloaded?, key_kept);
        Ok(())
    }
}
