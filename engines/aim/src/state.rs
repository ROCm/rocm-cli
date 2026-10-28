// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use rocm_core::{AppPaths, openai_models_endpoint_has_model, require_nonempty};
use rocm_engine_protocol::{
    DEFAULT_LOG_TAIL_LINES, DevicePolicy, EndpointRequest, EndpointResponse, EngineRecipeHint,
    HealthcheckRequest, HealthcheckResponse, LogsRequest, LogsResponse,
};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

const HEALTHCHECK_TIMEOUT_MS: u64 = 700;

pub(crate) fn healthcheck_service(request: HealthcheckRequest) -> Result<HealthcheckResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let state_path = state_path(&request.service_id);
    let state = read_service_state(&state_path).ok();

    let endpoint_url = state
        .as_ref()
        .and_then(endpoint_url_from_state)
        .unwrap_or_default();
    // The name the container lists in `/v1/models`, which is not the image ref
    // `model_ref` holds (see `process::resolve_model_response`).
    let model_ref = state
        .as_ref()
        .and_then(|s| value_string(s, "served_model_id").or_else(|| value_string(s, "model_ref")));
    let state_status = state
        .as_ref()
        .and_then(|s| value_string(s, "status"))
        .unwrap_or_else(|| "unknown".to_owned());
    let device = if state.is_some() {
        "rocm_gpu"
    } else {
        "unknown"
    };

    let listed = !endpoint_url.is_empty()
        && query_loaded_model_endpoint(&endpoint_url, model_ref.as_deref()).unwrap_or(false);
    let ready = listed
        && model_ref.as_deref().is_some_and(|model_ref| {
            inference_verified(&state_path, state.as_ref(), &endpoint_url, model_ref)
        });

    Ok(HealthcheckResponse::for_readiness(
        listed,
        ready,
        &state_status,
        device,
    ))
}

pub(crate) fn endpoint_response(request: EndpointRequest) -> Result<EndpointResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let state = read_service_state(&state_path(&request.service_id))
        .with_context(|| format!("service state not found for `{}`", request.service_id))?;
    let endpoint_url = endpoint_url_from_state(&state)
        .with_context(|| format!("service `{}` has no endpoint URL", request.service_id))?;
    Ok(EndpointResponse {
        endpoint_url,
        api_style: "openai".to_owned(),
        supported_routes: vec![
            "/health".to_owned(),
            "/v1/models".to_owned(),
            "/v1/chat/completions".to_owned(),
            "/v1/completions".to_owned(),
        ],
    })
}

pub(crate) fn logs_response(request: LogsRequest) -> Result<LogsResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let limit = request.tail_lines.unwrap_or(DEFAULT_LOG_TAIL_LINES);
    let name = crate::process::container_name(&request.service_id);
    let recent_lines = crate::docker::logs(&name, limit).unwrap_or_default();
    Ok(LogsResponse {
        log_path: format!("docker://{name}"),
        recent_lines,
    })
}

pub(crate) fn state_path(service_id: &str) -> PathBuf {
    AppPaths::discover().map_or_else(
        |_| PathBuf::from(format!("{service_id}.json")),
        |paths| {
            paths
                .engine_state_dir(crate::ENGINE_NAME)
                .join(format!("{service_id}.json"))
        },
    )
}

/// Replaces every `extra_env` value with a redaction marker, keeping the keys.
fn redact_engine_recipe_extra_env(hint: &EngineRecipeHint) -> EngineRecipeHint {
    let mut hint = hint.clone();
    for value in hint.extra_env.values_mut() {
        *value = "[redacted]".to_owned();
    }
    hint
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn write_running_state(
    state_path: &Path,
    service_id: &str,
    model_ref: &str,
    served_model_id: Option<&str>,
    host: &str,
    port: u16,
    device_policy: DevicePolicy,
    engine_recipe: Option<&EngineRecipeHint>,
) -> Result<()> {
    let endpoint = crate::process::endpoint_url(host, port);
    // `extra_env` can carry secrets (e.g. an HF token passed via `--engine-env`);
    // this state file is status/introspection only (never read back to relaunch
    // the container, see `crate::process::launch_service`), so redact values
    // before writing it to disk while keeping the keys for visibility.
    let engine_recipe = engine_recipe.map(redact_engine_recipe_extra_env);
    write_state(
        state_path,
        &json!({
            "service_id": service_id,
            "engine": crate::ENGINE_NAME,
            "status": "running",
            "container_name": crate::process::container_name(service_id),
            "model_ref": model_ref,
            "served_model_id": served_model_id,
            "host": host,
            "port": port,
            "endpoint_url": endpoint,
            "device_policy": device_policy,
            "engine_recipe": engine_recipe,
            "started_at_unix_ms": current_unix_millis(),
        }),
    )
}

/// Records the model name the running container actually serves under.
pub(crate) fn record_served_model_id(state_path: &Path, served_model_id: &str) -> Result<()> {
    merge_state_field(state_path, "served_model_id", json!(served_model_id))
}

/// Sets a non-terminal status (`stopping`) without stamping `stopped_at`.
pub(crate) fn write_status(state_path: &Path, status: &str) -> Result<()> {
    merge_state_field(state_path, "status", json!(status))
}

pub(crate) fn read_status(state_path: &Path) -> Option<String> {
    value_string(&read_service_state(state_path).ok()?, "status")
}

fn merge_state_field(state_path: &Path, key: &str, value: Value) -> Result<()> {
    let mut state = read_service_state(state_path).unwrap_or_else(|_| json!({}));
    if let Some(object) = state.as_object_mut() {
        object.insert(key.to_owned(), value);
    }
    write_state(state_path, &state)
}

pub(crate) fn write_terminal_state(state_path: &Path, status: &str) -> Result<()> {
    let mut state = read_service_state(state_path).unwrap_or_else(|_| json!({}));
    if let Some(object) = state.as_object_mut() {
        object.insert("status".to_owned(), json!(status));
        object.insert(
            "stopped_at_unix_ms".to_owned(),
            json!(current_unix_millis()),
        );
    }
    write_state(state_path, &state)
}

pub(crate) fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_state(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn endpoint_url_from_state(state: &Value) -> Option<String> {
    if let Some(endpoint) = value_string(state, "endpoint_url") {
        return Some(endpoint);
    }
    let host = value_string(state, "host")?;
    let port = state.get("port")?.as_u64()? as u16;
    Some(crate::process::endpoint_url(&host, port))
}

pub(crate) fn query_loaded_model_endpoint(
    endpoint_url: &str,
    model_ref: Option<&str>,
) -> Result<bool> {
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    openai_models_endpoint_has_model(
        endpoint_url,
        model_ref,
        endpoint_api_key.as_deref(),
        Duration::from_millis(HEALTHCHECK_TIMEOUT_MS),
    )
}

/// The first model the endpoint lists, for images that don't label the name
/// they serve under.
pub(crate) fn discover_served_model(endpoint_url: &str) -> Option<String> {
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    let body = rocm_core::http_get_text_with_auth(
        endpoint_url,
        "/v1/models",
        endpoint_api_key.as_deref(),
        Duration::from_millis(HEALTHCHECK_TIMEOUT_MS),
    )
    .ok()?;
    first_listed_model_id(&body)
}

fn first_listed_model_id(body: &str) -> Option<String> {
    serde_json::from_str::<Value>(body.trim())
        .ok()?
        .get("data")?
        .as_array()?
        .iter()
        .find_map(|item| value_string(item, "id").filter(|id| !id.trim().is_empty()))
}

pub(crate) fn query_inference_probe_endpoint(endpoint_url: &str, model_ref: &str) -> Result<bool> {
    if model_ref.trim().is_empty() {
        return Ok(false);
    }
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    rocm_core::openai_chat_completion_probe(
        endpoint_url,
        model_ref,
        endpoint_api_key.as_deref(),
        rocm_core::INFERENCE_PROBE_TIMEOUT,
    )
}

fn inference_verified(
    state_path: &Path,
    state: Option<&Value>,
    endpoint_url: &str,
    model_ref: &str,
) -> bool {
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    rocm_core::engine_state_inference_verified(
        state_path,
        state,
        endpoint_url,
        model_ref,
        endpoint_api_key.as_deref(),
    )
}

fn value_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn current_unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_url_falls_back_to_host_and_port() {
        let state = json!({"host": "127.0.0.1", "port": 8000});
        assert_eq!(
            endpoint_url_from_state(&state).as_deref(),
            Some("http://127.0.0.1:8000/v1")
        );
    }

    #[test]
    fn endpoint_url_prefers_explicit_value() {
        let state = json!({"endpoint_url": "http://example.test/v1", "host": "ignored", "port": 1});
        assert_eq!(
            endpoint_url_from_state(&state).as_deref(),
            Some("http://example.test/v1")
        );
    }

    #[test]
    fn healthcheck_reports_unknown_for_a_service_with_no_state_file() {
        let response = healthcheck_service(HealthcheckRequest {
            service_id: format!("svc-never-tracked-{}", current_unix_millis()),
        })
        .unwrap();
        assert!(!response.model_loaded);
        assert_eq!(response.status, "unknown");
        assert_eq!(response.device, "unknown");
    }

    #[test]
    fn redact_engine_recipe_extra_env_masks_values_but_keeps_keys() {
        let hint = EngineRecipeHint {
            contract_version: "0.1.0".to_owned(),
            engine: "aim".to_owned(),
            extra_env: std::collections::BTreeMap::from([(
                "HF_TOKEN".to_owned(),
                "hf_super_secret".to_owned(),
            )]),
            ..EngineRecipeHint::default()
        };
        let redacted = redact_engine_recipe_extra_env(&hint);
        assert_eq!(
            redacted.extra_env.get("HF_TOKEN").map(String::as_str),
            Some("[redacted]")
        );
    }

    #[test]
    fn write_running_state_never_persists_extra_env_secrets_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc-secret.json");
        let hint = EngineRecipeHint {
            contract_version: "0.1.0".to_owned(),
            engine: "aim".to_owned(),
            extra_env: std::collections::BTreeMap::from([(
                "HF_TOKEN".to_owned(),
                "hf_super_secret".to_owned(),
            )]),
            ..EngineRecipeHint::default()
        };
        write_running_state(
            &path,
            "svc-secret",
            "some/model",
            Some("Org/Served"),
            "127.0.0.1",
            8000,
            DevicePolicy::GpuPreferred,
            Some(&hint),
        )
        .unwrap();
        let on_disk = fs::read_to_string(&path).unwrap();
        assert!(!on_disk.contains("hf_super_secret"));
        assert!(on_disk.contains("HF_TOKEN"));
    }

    #[test]
    fn status_helpers_round_trip_and_keep_other_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc.json");
        assert_eq!(read_status(&path), None);
        write_status(&path, "stopping").unwrap();
        assert_eq!(read_status(&path).as_deref(), Some("stopping"));
        record_served_model_id(&path, "Org/Served").unwrap();
        let state = read_service_state(&path).unwrap();
        assert_eq!(value_string(&state, "status").as_deref(), Some("stopping"));
        assert_eq!(
            value_string(&state, "served_model_id").as_deref(),
            Some("Org/Served")
        );
    }

    #[test]
    fn first_listed_model_id_reads_the_first_nonempty_id() {
        assert_eq!(
            first_listed_model_id(r#"{"data":[{"id":""},{"id":"a/b"},{"id":"c"}]}"#).as_deref(),
            Some("a/b")
        );
        assert_eq!(first_listed_model_id(r#"{"data":[]}"#), None);
        assert_eq!(first_listed_model_id("not json"), None);
    }

    #[test]
    fn write_and_read_terminal_state_round_trips() {
        let dir = std::env::temp_dir().join(format!(
            "rocm-aim-state-test-{}-{}",
            std::process::id(),
            current_unix_millis()
        ));
        let path = dir.join("svc.json");
        write_state(&path, &json!({"status": "running"})).unwrap();
        write_terminal_state(&path, "stopped").unwrap();
        let state = read_service_state(&path).unwrap();
        assert_eq!(value_string(&state, "status").as_deref(), Some("stopped"));
        assert!(state.get("stopped_at_unix_ms").is_some());
        let _ = fs::remove_dir_all(&dir);
    }
}
