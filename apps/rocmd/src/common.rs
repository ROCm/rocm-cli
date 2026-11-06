// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::Result;
use rocm_core::{AppPaths, CommandCapture};
use serde_json::Value;
use std::time::Duration;

pub(crate) fn update_check_message(status: &str) -> &'static str {
    match status {
        "repair_available" => {
            "ran read-only `rocm update`; a ROCm runtime repair is available because its package composition changed; no updates were applied"
        }
        "update_available" => {
            "ran read-only `rocm update`; a ROCm runtime update is available; no updates were applied"
        }
        "error" => "read-only `rocm update` failed; no updates were applied",
        _ => "ran read-only `rocm update`; no updates were applied",
    }
}

/// Run `rocm` on `paths`' config/data/cache roots, via the shared
/// [`rocm_core::run_rocm_capture_with_env`].
pub(crate) fn run_rocm_capture_for_paths(
    paths: &AppPaths,
    args: &[&str],
    timeout: Duration,
) -> Result<CommandCapture> {
    rocm_core::run_rocm_capture_with_env(
        args,
        &[
            ("ROCM_CLI_CONFIG_DIR", paths.config_dir.as_path()),
            ("ROCM_CLI_DATA_DIR", paths.data_dir.as_path()),
            ("ROCM_CLI_CACHE_DIR", paths.cache_dir.as_path()),
        ],
        timeout,
    )
}

pub(crate) fn optional_arg(flag: &str, value: Option<&str>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value.to_owned()],
        None => Vec::new(),
    }
}

pub(crate) fn payload_string(payload: &Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}
