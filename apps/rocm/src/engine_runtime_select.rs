// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Engine/runtime selection and HTTP readiness polling: deciding which
//! runtime (or TheRock env) a `serve`/`engines install` invocation resolves
//! to (`EngineSelection`, `resolve_engine_selection`,
//! `validate_engine_selection_runtime`, `single_ready_runtime_key`,
//! `resolve_runtime_selector_to_exact_key`), small argv-building helpers used
//! when relaunching a managed engine (`optional_arg`,
//! `engine_recipe_json_arg`, `parse_engine_recipe_json_arg`,
//! `daemon_run_argv`, `managed_service_launcher_path`), the `ROCM_CLI_*`
//! environment variables every managed child inherits
//! (`app_path_env_vars`/`app_path_env_var_values`/`app_path_env_var_refs`/
//! `apply_app_path_env`), and polling a freshly launched service until it can
//! actually answer inference requests (`wait_for_service_http_ready`,
//! `wait_for_service_http_ready_with_progress`,
//! `service_http_readiness_paths`, `http_get_local_service`,
//! `http_post_local_service_json`, `service_http_readiness_response_ready`,
//! `lemonade_health_ready_for_model`, `model_list_ready_for_model`,
//! `service_model_gpu_ready`, `service_model_reports_rocm_backend`,
//! `service_model_names_match`).
//!
//! Mechanically relocated from `main.rs` with no behavior change — Phase 6f
//! (ROCMAI-215) of the `main.rs` modularization (see
//! `docs/architecture.md`). This is the last sub-phase of the `main.rs`
//! Phase 6f split that lands before the file's single shared
//! `#[cfg(test)] mod tests` block starts: the function immediately after this
//! cluster in the original file, `treat_as_natural_language`, is
//! dispatch-adjacent natural-language-invocation routing left over from an
//! earlier phase (ROCMAI-87) and stays in `main.rs` on purpose — it is not
//! part of this module and is not touched by this change.
//!
//! One item physically interleaved in this span in `main.rs` stayed there
//! instead of moving: `const fn status_for_readiness` sits between
//! `wait_for_service_http_ready_with_progress` and
//! `service_http_readiness_paths`, but it already had callers outside this
//! cluster — `main.rs`'s own `restart_internal_managed_service_with` and
//! `service_launch.rs`, which reaches it unqualified via `use
//! crate::status_for_readiness;` — so it stays a shared crate-root
//! function, reached from here (and from the retained tests below) the same
//! way, via `use crate::status_for_readiness;`.
//!
//! `EngineSelection` is not named in isolation anywhere in `main.rs`'s
//! dispatch/call-site re-import contract, but it is this cluster's own type —
//! returned by `resolve_engine_selection`/`validate_engine_selection_runtime`
//! and read nowhere else — so it moves with the functions that produce it,
//! the same way `serve_cmd.rs`'s `ServeEngineSelection` stayed with
//! `select_serve_engine`.
//!
//! `resolve_engine_selection` and `validate_engine_selection_runtime` are
//! `pub(crate)`, read from `engines_cmd.rs` and `serve_cmd.rs` (neither is
//! called from `main.rs` itself outside its own now-relocated tests);
//! `EngineSelection`, and its fields, are `pub(crate)` too, since those two
//! callers read `.runtime_id`/`.env_id`/`.source` directly off the value
//! `resolve_engine_selection`/`validate_engine_selection_runtime` return.
//! `resolve_runtime_selector_to_exact_key` is also `pub(crate)`: `engines_cmd.rs`
//! calls it in production, and `runtimes_cmd.rs`'s own test module calls it
//! directly too. `single_ready_runtime_key` is called only from
//! `validate_engine_selection_runtime` in this same module and stays private.
//! `optional_arg` and
//! `engine_recipe_json_arg` are `pub(crate)`, read from `engine_ipc.rs`'s
//! `engine_request`; `parse_engine_recipe_json_arg` is `pub(crate)`, read
//! from `main.rs`'s `dispatch()` and `restart_child_argv`.
//! `app_path_env_vars`/`apply_app_path_env` are `pub(crate)`, read from
//! `engine_ipc.rs`, and `apply_app_path_env` is also read from `chat.rs`,
//! `engines_cmd.rs`, `service_launch.rs`, and `main.rs`'s
//! `restart_internal_managed_service_with`; `app_path_env_var_values`/
//! `app_path_env_var_refs` are `pub(crate)`, read from `service_launch.rs`
//! and `main.rs`'s `restart_internal_managed_service_with`.
//! `daemon_run_argv` is `pub(crate)`, read from `main.rs`'s `dispatch()`;
//! `managed_service_launcher_path` is `pub(crate)`, read from
//! `service_launch.rs` and `main.rs`'s `restart_internal_managed_service_with`.
//! `wait_for_service_http_ready` is `pub(crate)`, read from `main.rs`'s
//! `restart_internal_managed_service_with`; `wait_for_service_http_ready_with_progress`
//! is `pub(crate)`, read from `service_launch.rs`. `http_post_local_service_json`
//! is `pub(crate)`, read from `main.rs`'s `unload_lemonade_service_model`.
//! `service_model_names_match` is `pub(crate)`, read from `chat.rs`.
//! `service_http_readiness_paths`, `http_get_local_service`,
//! `service_http_readiness_response_ready`, `lemonade_health_ready_for_model`,
//! `model_list_ready_for_model`, `service_model_gpu_ready`, and
//! `service_model_reports_rocm_backend` have no callers outside this module
//! and stay private.
//!
//! The cluster's regression tests live in this module's own `mod tests`,
//! reaching the shared `test_app_paths` fixture (needed by the
//! `app_path_env_*` tests) through `crate::tests::test_app_paths` the same
//! way other mechanically relocated modules reach shared fixtures, and
//! `status_for_readiness` (needed by the readiness-wait tests) through
//! `crate::status_for_readiness`, since it stays at the crate root per above.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, EndpointReadiness, RocmCliConfig, connect_tcp_stream, daemon_binary_path,
    format_host_port, format_http_base_url, read_http_response_bounded, write_all_tcp_stream,
};
use rocm_engine_protocol::EngineRecipeHint;

use crate::runtimes_cmd::{
    recover_setup_runtime_registration, select_runtime_manifest,
    validate_runtime_manifest_for_activation,
};
use crate::therock;

#[derive(Debug, Clone)]
pub(crate) struct EngineSelection {
    pub(crate) runtime_id: Option<String>,
    pub(crate) env_id: Option<String>,
    pub(crate) source: Option<String>,
}

pub(crate) fn resolve_engine_selection(
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> EngineSelection {
    if let Some(env_id) = env_id {
        return EngineSelection {
            runtime_id: None,
            env_id: Some(env_id.to_owned()),
            source: Some("cli_env_id".to_owned()),
        };
    }
    if let Some(runtime_id) = runtime_id {
        return EngineSelection {
            runtime_id: Some(runtime_id.to_owned()),
            env_id: None,
            source: Some("cli_runtime_id".to_owned()),
        };
    }

    if let Some(runtime_key) = config.active_runtime_key.as_ref() {
        return EngineSelection {
            runtime_id: Some(runtime_key.clone()),
            env_id: None,
            source: Some("config_active_runtime_key".to_owned()),
        };
    }

    if let Some(entry) = config.engine_config(engine) {
        if let Some(env_id) = entry.preferred_env_id.as_ref() {
            return EngineSelection {
                runtime_id: None,
                env_id: Some(env_id.clone()),
                source: Some("config_preferred_env_id".to_owned()),
            };
        }
        if let Some(runtime_id) = entry.preferred_runtime_id.as_ref() {
            return EngineSelection {
                runtime_id: Some(runtime_id.clone()),
                env_id: None,
                source: Some("config_preferred_runtime_id".to_owned()),
            };
        }
        if let Some(env_id) = entry.last_installed_env_id.as_ref() {
            return EngineSelection {
                runtime_id: None,
                env_id: Some(env_id.clone()),
                source: Some("config_last_installed_env_id".to_owned()),
            };
        }
        if let Some(runtime_id) = entry.last_installed_runtime_id.as_ref() {
            return EngineSelection {
                runtime_id: Some(runtime_id.clone()),
                env_id: None,
                source: Some("config_last_installed_runtime_id".to_owned()),
            };
        }
    }

    if let Some(runtime_id) = config.default_runtime_id.as_ref() {
        return EngineSelection {
            runtime_id: Some(runtime_id.clone()),
            env_id: None,
            source: Some("config_default_runtime_id".to_owned()),
        };
    }

    EngineSelection {
        runtime_id: None,
        env_id: None,
        source: None,
    }
}

pub(crate) fn validate_engine_selection_runtime(
    paths: &AppPaths,
    mut selection: EngineSelection,
) -> Result<EngineSelection> {
    if let Some(runtime_id) = selection.runtime_id.as_deref() {
        let source = selection.source.as_deref().unwrap_or("runtime selection");
        selection.runtime_id = Some(resolve_runtime_selector_to_exact_key(
            paths, runtime_id, source,
        )?);
    } else if selection.env_id.is_none()
        && let Some(runtime_key) = single_ready_runtime_key(paths)?
    {
        selection.runtime_id = Some(runtime_key);
        selection.source = Some("single_ready_runtime".to_owned());
    }
    Ok(selection)
}

fn single_ready_runtime_key(paths: &AppPaths) -> Result<Option<String>> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    recover_setup_runtime_registration(paths, &config)?;
    let manifests = therock::load_runtime_manifests(paths)?;
    let ready = manifests
        .iter()
        .filter(|manifest| validate_runtime_manifest_for_activation(manifest).is_ok())
        .collect::<Vec<_>>();
    Ok(match ready.as_slice() {
        [manifest] => Some(manifest.runtime_key.clone()),
        _ => None,
    })
}

pub(crate) fn resolve_runtime_selector_to_exact_key(
    paths: &AppPaths,
    selector: &str,
    source: &str,
) -> Result<String> {
    let manifests = therock::load_runtime_manifests(paths)?;
    match select_runtime_manifest(&manifests, selector) {
        Ok(manifest) => Ok(manifest.runtime_key.clone()),
        Err(error) => {
            let config = RocmCliConfig::load(paths).unwrap_or_default();
            if recover_setup_runtime_registration(paths, &config)?.is_some() {
                let manifests = therock::load_runtime_manifests(paths)?;
                if let Ok(manifest) = select_runtime_manifest(&manifests, selector) {
                    return Ok(manifest.runtime_key.clone());
                }
            }
            bail!(
                "runtime selector `{selector}` from {source} is not an exact usable runtime: {error}; run `rocm runtimes list` and `rocm runtimes activate <runtime_key>`, or pass --runtime-id <runtime_key>"
            )
        }
    }
}

pub(crate) fn optional_arg(flag: &str, value: Option<&str>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value.to_owned()],
        None => Vec::new(),
    }
}

pub(crate) fn engine_recipe_json_arg(
    engine_recipe: Option<&EngineRecipeHint>,
) -> Result<Vec<String>> {
    match engine_recipe {
        Some(engine_recipe) => Ok(vec![
            "--engine-recipe-json".to_owned(),
            serde_json::to_string(engine_recipe).context("failed to encode engine recipe hint")?,
        ]),
        None => Ok(Vec::new()),
    }
}

pub(crate) fn parse_engine_recipe_json_arg(
    value: Option<String>,
) -> Result<Option<EngineRecipeHint>> {
    value
        .map(|value| {
            serde_json::from_str(&value).context("failed to parse --engine-recipe-json payload")
        })
        .transpose()
}

pub(crate) fn app_path_env_vars(paths: &AppPaths) -> [(&'static str, &Path); 3] {
    [
        ("ROCM_CLI_CONFIG_DIR", paths.config_dir.as_path()),
        ("ROCM_CLI_DATA_DIR", paths.data_dir.as_path()),
        ("ROCM_CLI_CACHE_DIR", paths.cache_dir.as_path()),
    ]
}

#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn app_path_env_var_values(
    paths: &AppPaths,
    env_root: Option<&Path>,
) -> Vec<(&'static str, PathBuf)> {
    let mut vars = app_path_env_vars(paths)
        .into_iter()
        .map(|(key, value)| (key, value.to_path_buf()))
        .collect::<Vec<_>>();
    if let Some(env_root) = env_root {
        vars.push(("ROCM_CLI_ENGINE_ENVS_ROOT", env_root.to_path_buf()));
    }
    vars
}

#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn app_path_env_var_refs<'a>(
    vars: &'a [(&'static str, PathBuf)],
) -> Vec<(&'static str, &'a Path)> {
    vars.iter()
        .map(|(key, value)| (*key, value.as_path()))
        .collect()
}

/// Argv passed to the embedded `rocmd` library to run the real foreground
/// automation loop (the same path as `rocmd run --automations-enabled`).
pub(crate) fn daemon_run_argv() -> Vec<OsString> {
    vec![
        OsString::from("rocmd"),
        OsString::from("run"),
        OsString::from("--automations-enabled"),
    ]
}

pub(crate) fn managed_service_launcher_path() -> Result<PathBuf> {
    let current_exe = daemon_binary_path()?;
    if rocm_core::runtime_is_windows() {
        return Ok(rocm_core::normalize_runtime_path_for_storage(&current_exe));
    }
    Ok(current_exe)
}

pub(crate) fn apply_app_path_env(command: &mut ProcessCommand, paths: &AppPaths) {
    for (key, value) in app_path_env_vars(paths) {
        command.env(key, value);
    }
}

pub(crate) fn wait_for_service_http_ready(
    engine: &str,
    host: &str,
    port: u16,
    canonical_model_id: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> EndpointReadiness {
    wait_for_service_http_ready_with_progress(
        engine,
        host,
        port,
        &|| canonical_model_id.to_owned(),
        endpoint_api_key,
        timeout,
        &mut |_elapsed| {},
    )
}

/// Wait until the service can actually serve, reporting how far it got.
///
/// A health or model-listing endpoint answering is not enough to call a service
/// ready: engines advertise a model within seconds of accepting its name, while
/// the weights can take minutes to become usable, so a caller that sends its
/// first request on that signal gets a hang. Only [`EndpointReadiness::Serving`]
/// — an inference request that came back — means ready; a service that lists the
/// model without answering one is [`EndpointReadiness::Listing`], and the best
/// result seen before `timeout` is what gets returned.
///
/// Polls until `timeout` elapses, invoking `on_tick(elapsed)` once per iteration
/// so a caller can animate a spinner. Engine-neutral: `service_http_readiness_paths`
/// maps each engine to the right listing path. `endpoint_api_key` is sent as a
/// bearer token so both the listing check and the inference probe still succeed
/// against a public endpoint that requires authentication.
pub(crate) fn wait_for_service_http_ready_with_progress(
    engine: &str,
    host: &str,
    port: u16,
    model_name: &dyn Fn() -> String,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
    on_tick: &mut dyn FnMut(Duration),
) -> EndpointReadiness {
    let start = std::time::Instant::now();
    let endpoint = format_http_base_url(host, port);
    let mut best = EndpointReadiness::Unreachable;
    while start.elapsed() < timeout {
        // Re-evaluated each pass: a container engine reports the name its server
        // answers to only after it starts, so a name fixed up front can be wrong.
        let canonical_model_id = model_name();
        let canonical_model_id = canonical_model_id.as_str();
        let listed = service_http_readiness_paths(engine).iter().any(|path| {
            http_get_local_service(
                host,
                port,
                path,
                endpoint_api_key,
                Duration::from_millis(750),
            )
            .is_ok_and(|(status, body)| {
                service_http_readiness_response_ready(
                    engine,
                    path,
                    status,
                    &body,
                    canonical_model_id,
                )
            })
        });
        if listed {
            best = EndpointReadiness::Listing;
            if rocm_core::openai_chat_completion_probe(
                &endpoint,
                canonical_model_id,
                endpoint_api_key,
                rocm_core::INFERENCE_PROBE_TIMEOUT,
            )
            .unwrap_or(false)
            {
                return EndpointReadiness::Serving;
            }
        }
        on_tick(start.elapsed());
        thread::sleep(Duration::from_millis(250));
    }
    best
}

fn service_http_readiness_paths(engine: &str) -> &'static [&'static str] {
    match engine {
        "lemonade" => &["/v1/health", "/v1/models"],
        _ => &["/v1/models", "/v1/health", "/health", "/healthz"],
    }
}

fn http_get_local_service(
    host: &str,
    port: u16,
    path: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<(u16, String)> {
    let deadline = std::time::Instant::now() + timeout;
    let mut stream = connect_tcp_stream(host, port, timeout)?;
    let host_header = format_host_port(host, port);
    // Authenticate the probe when the endpoint is protected; loopback endpoints
    // pass `None` and the header is omitted.
    let auth_header = match endpoint_api_key {
        Some(key) => format!("Authorization: Bearer {key}\r\n"),
        None => String::new(),
    };
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host_header}\r\n{auth_header}Connection: close\r\n\r\n"
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .context("failed to write service readiness request")?;
    let response = read_http_response_bounded(&mut stream, deadline)
        .context("failed to read service readiness response")?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or((response.as_str(), ""));
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(0);
    Ok((status, body.to_owned()))
}

pub(crate) fn http_post_local_service_json(
    host: &str,
    port: u16,
    path: &str,
    body: &serde_json::Value,
    timeout: Duration,
) -> Result<(u16, String)> {
    let deadline = std::time::Instant::now() + timeout;
    let mut stream = connect_tcp_stream(host, port, timeout)?;
    let host_header = format_host_port(host, port);
    let body = serde_json::to_string(body).context("failed to serialize service request")?;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .context("failed to write service request")?;
    let response = read_http_response_bounded(&mut stream, deadline)
        .context("failed to read service response")?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or((response.as_str(), ""));
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(0);
    Ok((status, body.to_owned()))
}

fn service_http_readiness_response_ready(
    engine: &str,
    path: &str,
    status: u16,
    body: &str,
    canonical_model_id: &str,
) -> bool {
    if status != 200 {
        return false;
    }
    match (engine, path) {
        ("lemonade", "/v1/health") => lemonade_health_ready_for_model(body, canonical_model_id),
        ("lemonade", "/v1/models") => model_list_ready_for_model(body, canonical_model_id, true),
        (_, "/v1/models") => model_list_ready_for_model(body, canonical_model_id, false),
        _ => false,
    }
}

fn lemonade_health_ready_for_model(body: &str, canonical_model_id: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body.trim()) else {
        return false;
    };
    value
        .get("all_models_loaded")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                let name_matches = ["model_name", "id", "name"]
                    .into_iter()
                    .filter_map(|field| model.get(field).and_then(serde_json::Value::as_str))
                    .any(|loaded| service_model_names_match(loaded, canonical_model_id));
                name_matches && service_model_reports_rocm_backend(model)
            })
        })
}

fn model_list_ready_for_model(
    body: &str,
    canonical_model_id: &str,
    require_rocm_backend: bool,
) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body.trim()) else {
        return false;
    };
    value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                let name_matches = ["id", "model", "name"]
                    .into_iter()
                    .filter_map(|field| model.get(field).and_then(serde_json::Value::as_str))
                    .any(|loaded| service_model_names_match(loaded, canonical_model_id));
                name_matches && (!require_rocm_backend || service_model_gpu_ready(model))
            })
        })
}

/// Whether a lemonade `/v1/models` entry is served on GPU. A stock `llama-server`
/// (direct-serve) entry has no `recipe_options` and is accepted (that path only runs
/// GPU backends); a Lemonade-router entry carries `recipe_options`, so it must report a
/// ROCm backend — which keeps a registered-but-unloaded model (empty `recipe_options`)
/// from reading as ready.
fn service_model_gpu_ready(model: &serde_json::Value) -> bool {
    match model.get("recipe_options") {
        None => true,
        Some(_) => service_model_reports_rocm_backend(model),
    }
}

fn service_model_reports_rocm_backend(model: &serde_json::Value) -> bool {
    model
        .get("recipe_options")
        .and_then(|options| options.get("llamacpp_backend"))
        .or_else(|| model.get("llamacpp_backend"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|backend| backend.trim().to_ascii_lowercase().starts_with("rocm"))
}

pub(crate) fn service_model_names_match(left: &str, right: &str) -> bool {
    let left = left.trim();
    let right = right.trim();
    if left.is_empty() || right.is_empty() {
        return false;
    }
    if left.eq_ignore_ascii_case(right) {
        return true;
    }
    let left = left
        .trim_end_matches(".gguf")
        .trim_end_matches(".safetensors")
        .to_ascii_lowercase();
    let right = right
        .trim_end_matches(".gguf")
        .trim_end_matches(".safetensors")
        .to_ascii_lowercase();
    left.contains(&right) || right.contains(&left)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::status_for_readiness;
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::{test_app_paths, test_paths};

    #[test]
    fn service_http_readiness_requires_loaded_lemonade_model() {
        let loading = json!({ "all_models_loaded": [] }).to_string();
        assert!(!service_http_readiness_response_ready(
            "lemonade",
            "/v1/health",
            200,
            &loading,
            "Qwen3-0.6B-GGUF"
        ));

        let loaded = json!({
            "all_models_loaded": [{
                "model_name": "Qwen3-0.6B-GGUF",
                "recipe_options": { "llamacpp_backend": "rocm" }
            }]
        })
        .to_string();
        assert!(service_http_readiness_response_ready(
            "lemonade",
            "/v1/health",
            200,
            &loaded,
            "Qwen3-0.6B-GGUF"
        ));

        let loaded_cpu = json!({
            "all_models_loaded": [{
                "model_name": "Qwen3-0.6B-GGUF",
                "recipe_options": { "llamacpp_backend": "cpu" }
            }]
        })
        .to_string();
        assert!(!service_http_readiness_response_ready(
            "lemonade",
            "/v1/health",
            200,
            &loaded_cpu,
            "Qwen3-0.6B-GGUF"
        ));
    }

    #[test]
    fn service_http_readiness_requires_model_list_entry() {
        let empty = json!({ "data": [] }).to_string();
        assert!(!service_http_readiness_response_ready(
            "vllm",
            "/v1/models",
            200,
            &empty,
            "tiny.gguf"
        ));

        let models = json!({ "data": [{ "id": "tiny.gguf" }] }).to_string();
        assert!(service_http_readiness_response_ready(
            "vllm",
            "/v1/models",
            200,
            &models,
            "tiny.gguf"
        ));

        let lemonade_cpu_models = json!({
            "data": [{
                "id": "Qwen3-0.6B-GGUF",
                "recipe_options": { "llamacpp_backend": "cpu" }
            }]
        })
        .to_string();
        assert!(!service_http_readiness_response_ready(
            "lemonade",
            "/v1/models",
            200,
            &lemonade_cpu_models,
            "Qwen3-0.6B-GGUF"
        ));

        let lemonade_rocm_models = json!({
            "data": [{
                "id": "Qwen3-0.6B-GGUF",
                "recipe_options": { "llamacpp_backend": "rocm" }
            }]
        })
        .to_string();
        assert!(service_http_readiness_response_ready(
            "lemonade",
            "/v1/models",
            200,
            &lemonade_rocm_models,
            "Qwen3-0.6B-GGUF"
        ));

        assert!(!service_http_readiness_response_ready(
            "vllm",
            "/health",
            200,
            "OK",
            "tiny.gguf"
        ));
        assert!(!service_http_readiness_response_ready(
            "vllm",
            "/healthz",
            200,
            "OK",
            "Qwen3-0.6B-GGUF"
        ));
    }

    #[test]
    fn lemonade_direct_serve_model_reads_ready_without_recipe_options() {
        // The HF direct-serve path runs a stock llama-server whose `/v1/models` entry
        // has no `recipe_options`. It must read as ready by name (that path is GPU-only),
        // while a registered-but-unloaded lemonade entry (empty `recipe_options`) must not.
        let direct = json!({
            "data": [{ "id": "LiquidAI/LFM2.5-230M-GGUF:Q4_0", "object": "model" }]
        })
        .to_string();
        assert!(service_http_readiness_response_ready(
            "lemonade",
            "/v1/models",
            200,
            &direct,
            "LiquidAI/LFM2.5-230M-GGUF:Q4_0"
        ));

        let registered = json!({
            "data": [{ "id": "LiquidAI/LFM2.5-230M-GGUF:Q4_0", "recipe_options": {} }]
        })
        .to_string();
        assert!(!service_http_readiness_response_ready(
            "lemonade",
            "/v1/models",
            200,
            &registered,
            "LiquidAI/LFM2.5-230M-GGUF:Q4_0"
        ));
    }

    #[test]
    fn serve_readiness_wait_withholds_ready_while_the_model_only_lists() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // This is the wait behind `rocm serve`, and its verdict is what
        // `services list` later prints. A model listing on `/v1/models` while
        // inference still fails must come back as `Listing` — reporting it ready
        // is the false positive users and automation trip over.
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<()> {
            // Serve until the wait below gives up and the test drops the socket.
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 1024];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                let (status_line, body) = if request.starts_with("POST /v1/chat/completions ") {
                    ("HTTP/1.1 503 Service Unavailable", r#"{"error":"loading"}"#)
                } else {
                    ("HTTP/1.1 200 OK", r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#)
                };
                let _ = write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
            }
            Ok(())
        });

        let readiness = wait_for_service_http_ready(
            "vllm",
            "127.0.0.1",
            port,
            "Qwen3-0.6B-GGUF",
            None,
            Duration::from_millis(600),
        );

        assert_eq!(
            readiness,
            EndpointReadiness::Listing,
            "a listed-but-unservable model is not ready"
        );
        assert_eq!(
            status_for_readiness(readiness),
            "running",
            "a loading service must not be recorded as `starting`, which rocmd \
             restarts once stale"
        );
        drop(server);
        Ok(())
    }

    #[test]
    fn serve_readiness_wait_reports_ready_once_inference_answers() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<()> {
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 1024];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                let body = if request.starts_with("POST /v1/chat/completions ") {
                    r#"{"choices":[{"message":{"content":"ok"}}]}"#
                } else {
                    r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
            }
            Ok(())
        });

        let readiness = wait_for_service_http_ready(
            "vllm",
            "127.0.0.1",
            port,
            "Qwen3-0.6B-GGUF",
            None,
            Duration::from_secs(5),
        );

        assert_eq!(readiness, EndpointReadiness::Serving);
        assert_eq!(status_for_readiness(readiness), "ready");
        drop(server);
        Ok(())
    }

    #[test]
    fn serve_readiness_ready_verdict_does_not_wait_for_the_peer_to_close() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::time::Instant;

        // Regression test for the other half of this PR's fix: a response is
        // read to completion by its own framing, not by waiting for the peer
        // to close. Before this fix, `read_tcp_stream_to_string` blocked
        // until EOF, so a keep-alive engine that answers correctly but never
        // closes the socket looked identical to a hung one — the readiness
        // probe ran out its timeout and reported not-ready even though the
        // answer had already arrived.
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                thread::spawn(move || {
                    stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                    let mut buffer = [0_u8; 1024];
                    let Ok(read) = stream.read(&mut buffer) else {
                        return;
                    };
                    let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                    let body = if request.starts_with("POST /v1/chat/completions ") {
                        r#"{"choices":[{"message":{"content":"ok"}}]}"#
                    } else {
                        r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#
                    };
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    // Hold the connection open well past the readiness wait's
                    // timeout below, and deliberately omit `Connection:
                    // close`. The client must not need EOF to recognize the
                    // response as complete.
                    thread::sleep(Duration::from_secs(10));
                });
            }
        });

        let started = Instant::now();
        let readiness = wait_for_service_http_ready(
            "vllm",
            "127.0.0.1",
            port,
            "Qwen3-0.6B-GGUF",
            None,
            Duration::from_secs(5),
        );

        assert_eq!(readiness, EndpointReadiness::Serving);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "a complete response must be recognized without waiting on the peer to close"
        );
        // `server`'s accept loop runs forever; dropping the JoinHandle detaches
        // it rather than joining, and the thread dies with the test process.
        drop(server);
        Ok(())
    }

    #[test]
    fn app_path_env_vars_include_config_data_and_cache() {
        let paths = test_app_paths();
        let vars = app_path_env_vars(&paths);

        assert_eq!(vars[0], ("ROCM_CLI_CONFIG_DIR", paths.config_dir.as_path()));
        assert_eq!(vars[1], ("ROCM_CLI_DATA_DIR", paths.data_dir.as_path()));
        assert_eq!(vars[2], ("ROCM_CLI_CACHE_DIR", paths.cache_dir.as_path()));
    }

    #[test]
    fn app_path_env_var_values_include_engine_env_root_when_needed() {
        let paths = test_app_paths();
        let engine_root = PathBuf::from("D:/rocm-data/runtime/engines");
        let vars = app_path_env_var_values(&paths, Some(&engine_root));

        assert_eq!(
            vars.last().map(|(key, value)| (*key, value.as_path())),
            Some(("ROCM_CLI_ENGINE_ENVS_ROOT", engine_root.as_path()))
        );
    }

    #[test]
    fn resolve_engine_selection_uses_default_runtime_after_engine_prefs() {
        let mut config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let selection = resolve_engine_selection(&config, "vllm", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("therock-release:gfx120X-all")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_default_runtime_id")
        );

        config.active_runtime_key = Some("release-pip-gfx120x-all-7-13-0".to_owned());
        let selection = resolve_engine_selection(&config, "vllm", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_active_runtime_key")
        );

        config.engine_config_mut("vllm").preferred_runtime_id =
            Some("therock-nightly:gfx120X-all".to_owned());
        let selection = resolve_engine_selection(&config, "vllm", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_active_runtime_key")
        );

        config.active_runtime_key = None;
        let selection = resolve_engine_selection(&config, "vllm", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("therock-nightly:gfx120X-all")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_preferred_runtime_id")
        );
    }

    #[test]
    fn engine_selection_uses_single_ready_runtime_without_active_marker() -> Result<()> {
        let (root, paths) = test_paths("single-ready-runtime-selection");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
            "therock-release:gfx120X-all",
            "7.14.0",
            20,
        )?;
        let selection = validate_engine_selection_runtime(
            &paths,
            resolve_engine_selection(&RocmCliConfig::default(), "vllm", None, None),
        )?;

        assert_eq!(
            selection.runtime_id.as_deref(),
            Some(manifest.runtime_key.as_str())
        );
        assert_eq!(selection.source.as_deref(), Some("single_ready_runtime"));
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn engine_selection_keeps_multiple_ready_runtimes_explicit() -> Result<()> {
        let (root, paths) = test_paths("multiple-ready-runtime-selection");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
            "therock-release:gfx120X-all",
            "7.14.0",
            20,
        )?;
        let selection = validate_engine_selection_runtime(
            &paths,
            resolve_engine_selection(&RocmCliConfig::default(), "vllm", None, None),
        )?;

        assert!(selection.runtime_id.is_none());
        assert!(selection.env_id.is_none());
        assert!(selection.source.is_none());
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }
}
