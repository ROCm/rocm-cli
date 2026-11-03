// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#[cfg(test)]
use anyhow::Context;
use anyhow::Result;
use rocm_core::{AppPaths, CommandCapture};
#[cfg(test)]
use rocm_engine_protocol::EnginePluginDescriptor;
use serde_json::Value;
#[cfg(test)]
use std::path::PathBuf;
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

#[cfg(test)]
pub(crate) fn find_engine_plugin_binary<I, P>(
    engine: &str,
    plugin_dirs: I,
) -> Result<Option<PathBuf>>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    Ok(rocm_engine_protocol::discover_engine_plugins(plugin_dirs)
        .context("failed to discover engine plugin binaries")?
        .into_iter()
        .find(|plugin: &EnginePluginDescriptor| plugin.id == engine)
        .map(|plugin| plugin.executable_path))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use rocm_core::engine_plugin_dirs;
    use std::fs;

    #[test]
    fn engine_plugin_discovery_finds_runtime_binary() -> Result<()> {
        let (root, paths) = temp_app_paths("engine-plugin");
        let plugin_dir = paths.data_dir.join("engines").join("plugins");
        fs::create_dir_all(&plugin_dir)?;
        let plugin_path = plugin_dir.join(
            rocm_engine_protocol::platform_engine_plugin_binary_name("vllm"),
        );
        fs::write(&plugin_path, "plugin")?;

        let discovered = find_engine_plugin_binary("vllm", engine_plugin_dirs(&paths))?;
        fs::remove_dir_all(root).ok();

        assert_eq!(discovered, Some(plugin_path));
        Ok(())
    }
}
