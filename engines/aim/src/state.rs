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
    healthcheck_service_at(&state_path(&request.service_id))
}

/// [`healthcheck_service`] against an explicit state path.
///
/// Split out for the same reason `engines/lemonade`'s `healthcheck_service_at`
/// is: [`state_path`] resolves through the caller's real `AppPaths`, so the
/// RPC handler itself could not be driven from a test, and the `device` field
/// it answers with was only ever covered by calling [`reported_device`]
/// directly — a test that stays green if this handler stops consulting it at
/// all (see `healthcheck_rpc_answers_with_the_device_the_state_file_describes`).
pub(crate) fn healthcheck_service_at(state_path: &Path) -> Result<HealthcheckResponse> {
    let state = read_service_state(state_path).ok();

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
    let device = match state.as_ref() {
        Some(state) => reported_device(state),
        None => "unknown".to_owned(),
    };

    let listed = !endpoint_url.is_empty()
        && query_loaded_model_endpoint(&endpoint_url, model_ref.as_deref()).unwrap_or(false);
    let ready = listed
        && model_ref.as_deref().is_some_and(|model_ref| {
            inference_verified(state_path, state.as_ref(), &endpoint_url, model_ref)
        });

    Ok(HealthcheckResponse::for_readiness(
        listed,
        ready,
        &state_status,
        &device,
    ))
}

/// The device string reported once state exists for the service.
///
/// The shape is whatever [`rocm_engine_protocol::format_reported_device`]
/// renders — the same one `engines/lemonade` and `engines/vllm` use, because
/// `rocm serve`'s `device:` row and `rocm services list` print whichever engine
/// answered verbatim and must not change format per engine. The ordinal comes
/// from [`rocm_engine_protocol::first_gpu_index_from_state`], the same read
/// those two make of the same `gpu_indices` key.
///
/// The backend is **not** fixed. AIM's images run on ROCm, but
/// [`rocm_engine_protocol::DevicePolicy::CpuOnly`] is a real variant of this
/// crate's own launch path (`process::resolve_use_gpu`) and
/// `docker::RunSpec` then hands the container no `/dev/kfd`, no `/dev/dri` and
/// no `HIP_VISIBLE_DEVICES` at all. Claiming `rocm`, let alone naming a card,
/// for a container that was given no GPU is exactly the invented device row
/// ROCMAI-550 exists to remove, so the policy the launch recorded decides:
/// `cpu_only` reports `cpu` and names nothing. A state file from before this
/// key was written has no policy, and ROCm is the right assumption there —
/// `cpu_only` could not be reached through `rocm serve` then either.
fn reported_device(state: &Value) -> String {
    if recorded_device_policy(state) == Some(DevicePolicy::CpuOnly) {
        return rocm_engine_protocol::format_reported_device("cpu", None, None);
    }
    rocm_engine_protocol::format_reported_device(
        "rocm",
        rocm_engine_protocol::first_gpu_index_from_state(state),
        value_string(state, "gpu_name").as_deref(),
    )
}

/// The `device_policy` the launch recorded, or `None` when the state file
/// predates the key or holds something that is not one of the three policies.
fn recorded_device_policy(state: &Value) -> Option<DevicePolicy> {
    serde_json::from_value(state.get("device_policy")?.clone()).ok()
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
    let name = crate::docker::container_name(&request.service_id);
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
    gpu_indices: Option<&[u32]>,
    engine_recipe: Option<&EngineRecipeHint>,
) -> Result<()> {
    let endpoint = endpoint_url(host, port);
    // No GPU was attached, so there is no device to probe for or name. See
    // [`reported_device`] for why this is gated rather than probed anyway.
    let uses_gpu = device_policy != DevicePolicy::CpuOnly;
    // `None` means this launch reattached to a container that was already
    // running, so it never chose any ordinals. This is a whole-document write
    // (`write_json_state_file`), so writing an empty list here would not record
    // "unknown" — it would delete the ordinals and the GPU name the launch that
    // actually started the container recorded, leaving `device:` unable to name
    // a device for a server that still has one. Carry both forward instead.
    let (gpu_indices, gpu_name) = match gpu_indices {
        Some(indices) => (
            json!(indices),
            json!(
                uses_gpu
                    .then(|| rocm_core::pinned_gpu_name(indices))
                    .flatten()
            ),
        ),
        None => recorded_gpu_identity(state_path),
    };
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
            "container_name": crate::docker::container_name(service_id),
            "model_ref": model_ref,
            "served_model_id": served_model_id,
            "host": host,
            "port": port,
            "endpoint_url": endpoint,
            "device_policy": device_policy,
            // The ordinals `process::plan_gpus` resolved and `docker::RunSpec`
            // narrowed the container to with `HIP_VISIBLE_DEVICES` — the same
            // key, read the same way, as `engines/lemonade` and
            // `engines/vllm`, so one multi-GPU host does not report
            // `rocm gpu 1 (…)` for those two and a bare `rocm` for this one.
            // Empty for a CPU-only launch; carried forward from the previous
            // state file for a container this call found already running (that
            // launch pinned it, and this one cannot know what it chose).
            "gpu_indices": gpu_indices,
            // The host's GPU name, so the `device:` row names a device and not
            // only a backend (ROCMAI-550). Captured once here rather than
            // probed per healthcheck: `rocm services list` healthchecks every
            // ready service on its render path. `rocm_core::pinned_gpu_name`
            // declines on any host where more than one AMD GPU could be the
            // one that answered, and is not asked at all when no GPU was
            // attached. Carried forward alongside the ordinals on a reattach,
            // since re-probing could answer for a different GPU than the one
            // that launch pinned.
            "gpu_name": gpu_name,
            "engine_recipe": engine_recipe,
            "started_at_unix_ms": current_unix_millis(),
        }),
    )
}

/// The `gpu_indices` and `gpu_name` already on disk for this service, for a
/// launch that reattached and so cannot supply its own.
///
/// Returns JSON `null` for either key that is absent or unreadable, which is
/// what every other writer of this file already means by "not known" — a
/// first-ever write has no prior document, and a truncated one is no more
/// authoritative than none.
fn recorded_gpu_identity(state_path: &Path) -> (Value, Value) {
    let Ok(previous) = read_service_state(state_path) else {
        return (Value::Null, Value::Null);
    };
    (
        previous.get("gpu_indices").cloned().unwrap_or(Value::Null),
        previous.get("gpu_name").cloned().unwrap_or(Value::Null),
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

/// Patch one key into the service's state file under the state directory's
/// lock ([`rocm_core::merge_json_state_file`]) — the same lock the
/// inference-verification latch and `rocmd`'s Anthropic-endpoint publish take,
/// so no writer drops a key another writer recorded.
///
/// It used to read, patch and `fs::write` with no lock at all, which could
/// drop either of those. A state file that is empty, torn or not a JSON object
/// is now an error and left as it was, rather than replaced wholesale.
fn merge_state_field(state_path: &Path, key: &str, value: Value) -> Result<()> {
    rocm_core::merge_json_state_file(state_path, &json!({ key: value }))
}

pub(crate) fn write_terminal_state(state_path: &Path, status: &str) -> Result<()> {
    rocm_core::merge_json_state_file(
        state_path,
        &json!({
            "status": status,
            "stopped_at_unix_ms": current_unix_millis(),
        }),
    )
}

pub(crate) fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

/// Replace the service's state file whole, under the state directory's lock
/// and published atomically ([`rocm_core::write_json_state_file`]).
///
/// [`write_running_state`] is the only caller; the status writers patch
/// ([`merge_state_field`]) instead. It used to be a lock-free `fs::write`,
/// which is a problem this PR's own change made reachable: the document being
/// written now includes a host GPU probe (`rocm_core::pinned_gpu_name`, up to
/// seconds on Windows), and `rocmd`'s Anthropic-proxy thread — a thread of
/// *this* process — publishes its endpoint into the same file in that window.
/// Going through `write_json_state_file` both orders this against every merge
/// and carries that publication through the rewrite, the way
/// `engines/lemonade`'s `write_state` does.
fn write_state(path: &Path, value: &Value) -> Result<()> {
    rocm_core::write_json_state_file(path, value)
}

pub(crate) fn endpoint_url(host: &str, port: u16) -> String {
    format!("{}/v1", rocm_core::format_http_base_url(host, port))
}

fn endpoint_url_from_state(state: &Value) -> Option<String> {
    if let Some(endpoint) = value_string(state, "endpoint_url") {
        return Some(endpoint);
    }
    let host = value_string(state, "host")?;
    let port = state.get("port")?.as_u64()? as u16;
    Some(endpoint_url(&host, port))
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
            Some(&[0]),
            Some(&hint),
        )
        .unwrap();
        let on_disk = fs::read_to_string(&path).unwrap();
        assert!(!on_disk.contains("hf_super_secret"));
        assert!(on_disk.contains("HF_TOKEN"));
    }

    /// `rocm serve` against a container that is already up reattaches instead
    /// of launching, so it never chooses GPU ordinals — but it still rewrites
    /// this state file, and [`write_state`] replaces the document whole. The
    /// launch that *did* start the container is the only one that knows what it
    /// pinned, so a reattach must carry those forward rather than write an
    /// empty list over them: otherwise `rocm serve` on a running service
    /// silently drops the `device:` row's GPU for a server that still has one.
    ///
    /// Both keys are asserted. `gpu_name` is what the row actually prints, and
    /// `first_gpu_index_from_state` reads `gpu_indices` to decide whether to
    /// print an ordinal at all, so losing either one degrades the row.
    #[test]
    fn a_reattach_keeps_the_gpu_the_launch_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc-reattach.json");
        let write = |gpu_indices: Option<&[u32]>| {
            write_running_state(
                &path,
                "svc-reattach",
                "some/model",
                Some("Org/Served"),
                "127.0.0.1",
                8000,
                DevicePolicy::GpuPreferred,
                gpu_indices,
                None,
            )
            .unwrap();
        };

        // The launch. Plant a name the way a real one would, since
        // `pinned_gpu_name` declines on a host like CI's.
        write(Some(&[1]));
        let mut planted = read_service_state(&path).unwrap();
        planted["gpu_name"] = json!("AMD Instinct MI300X");
        write_state(&path, &planted).unwrap();

        // The reattach: same service, no ordinals of its own.
        write(None);

        let after = read_service_state(&path).unwrap();
        assert_eq!(
            after.get("gpu_indices"),
            Some(&json!([1])),
            "a reattach must not erase the ordinals the launch pinned: {after}"
        );
        assert_eq!(
            after.get("gpu_name"),
            Some(&json!("AMD Instinct MI300X")),
            "nor the GPU name they resolved to: {after}"
        );
        assert_eq!(
            rocm_engine_protocol::first_gpu_index_from_state(&after),
            Some(1),
            "and the device row must still find an ordinal to print: {after}"
        );
    }

    /// ROCMAI-550 changed the device this engine reports for an existing
    /// service from the literal `"rocm_gpu"` to the shared
    /// `<backend> [gpu <n>] [(<name>)]` shape, and nothing pinned it.
    ///
    /// This drives the real persist-then-report path:
    /// [`write_running_state`] writes the state file, [`read_service_state`]
    /// reads it back and [`reported_device`] renders it — exactly what
    /// [`healthcheck_service`] does for a tracked service, minus the
    /// `state_path` lookup, which resolves through the caller's real
    /// `AppPaths` and so cannot be driven from a unit test (the no-state half
    /// is covered by
    /// `healthcheck_reports_unknown_for_a_service_with_no_state_file`).
    ///
    /// Reverting the string to `"rocm_gpu"`, dropping the `gpu_name` field or
    /// dropping the `gpu_indices` field from the state write fails this.
    #[test]
    fn reported_device_for_existing_state_matches_the_shared_engine_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc-device.json");
        write_running_state(
            &path,
            "svc-device",
            "some/model",
            Some("Org/Served"),
            "127.0.0.1",
            8000,
            DevicePolicy::GpuRequired,
            Some(&[1]),
            None,
        )
        .unwrap();

        // The probed name is whatever the test host has, so assert the shape
        // rather than a value: `rocm gpu 1`, optionally followed by
        // `(<name>)`, and never the old `rocm_gpu`.
        let state = read_service_state(&path).unwrap();
        let device = reported_device(&state);
        assert!(
            device == "rocm gpu 1" || (device.starts_with("rocm gpu 1 (") && device.ends_with(')')),
            "AIM must report the shared `<backend> gpu <n> [(<name>)]` shape, with the \
             ordinal the container was narrowed to: {device}"
        );
        assert!(
            !device.contains("rocm_gpu"),
            "the pre-ROCMAI-550 literal must not come back: {device}"
        );
        // The state file must actually carry both keys the row is built from,
        // even when the host probe found no name.
        assert!(
            state.get("gpu_name").is_some(),
            "write_running_state must record gpu_name: {state}"
        );
        assert_eq!(
            state.get("gpu_indices"),
            Some(&json!([1])),
            "write_running_state must record the ordinals the container was \
             given, the way lemonade and vLLM do: {state}"
        );

        // And the name, when the host has one, reaches the row.
        assert_eq!(
            reported_device(&json!({ "gpu_indices": [0], "gpu_name": "AMD Instinct MI300X" })),
            "rocm gpu 0 (AMD Instinct MI300X)"
        );
    }

    /// A `cpu_only` launch attaches no GPU at all (`process::resolve_use_gpu`
    /// returns `false`, and `docker::RunSpec` then passes no `/dev/kfd`, no
    /// `/dev/dri` and no `HIP_VISIBLE_DEVICES`), so neither the `rocm` backend
    /// nor a GPU name is a fact about it. Before ROCMAI-550's fix this path
    /// still probed the host and rendered e.g.
    /// `rocm (AMD Instinct MI300X)` for a container that used no GPU — the
    /// invented claim the whole ticket exists to remove.
    ///
    /// Drops to red if `write_running_state` probes for a name regardless of
    /// the policy, or if `reported_device` goes back to hardcoding `rocm`.
    #[test]
    fn a_cpu_only_launch_claims_neither_rocm_nor_a_gpu_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc-cpu.json");
        write_running_state(
            &path,
            "svc-cpu",
            "some/model",
            Some("Org/Served"),
            "127.0.0.1",
            8000,
            DevicePolicy::CpuOnly,
            Some(&[]),
            None,
        )
        .unwrap();

        let state = read_service_state(&path).unwrap();
        assert_eq!(
            state.get("gpu_name"),
            Some(&Value::Null),
            "a cpu-only launch must not have the host GPU probed for it: {state}"
        );
        assert_eq!(reported_device(&state), "cpu");

        // And the rendering is driven by the recorded policy, not by the
        // absence of a name: a state file that somehow carries both still
        // reports `cpu`.
        assert_eq!(
            reported_device(&json!({
                "device_policy": "cpu_only",
                "gpu_indices": [0],
                "gpu_name": "AMD Instinct MI300X",
            })),
            "cpu"
        );
        // A state file written before the key existed keeps the old reading.
        assert_eq!(
            reported_device(&json!({ "gpu_name": "AMD Instinct MI300X" })),
            "rocm (AMD Instinct MI300X)"
        );
    }

    /// The `device` the Healthcheck RPC *itself* answers with, not the private
    /// formatter it is supposed to call.
    ///
    /// Every other device test in the three engines calls `reported_device`
    /// directly, so reverting the handler's `Some(state) => reported_device`
    /// wiring to a hardcoded literal kept the whole suite green. This drives
    /// the handler.
    ///
    /// The planted state carries no endpoint, so no HTTP probe is made and
    /// `listed`/`ready` are false — which is the point: this engine reports
    /// the device as soon as it has state, independently of readiness (see
    /// `apps/rocm/src/engine_ipc.rs`'s `device_worth_reporting`).
    #[test]
    fn healthcheck_rpc_answers_with_the_device_the_state_file_describes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc-rpc.json");
        write_state(
            &path,
            &json!({
                "service_id": "svc-rpc",
                "status": "running",
                "device_policy": "gpu_required",
                "gpu_indices": [2],
                "gpu_name": "AMD Instinct MI300X",
            }),
        )
        .unwrap();

        let response = healthcheck_service_at(&path).unwrap();
        assert_eq!(response.device, "rocm gpu 2 (AMD Instinct MI300X)");
        assert_eq!(response.status, "running");
        assert!(!response.model_loaded);

        // And a service with no state on disk is the `"unknown"` the CLI
        // filters out rather than prints.
        let response = healthcheck_service_at(&dir.path().join("absent.json")).unwrap();
        assert_eq!(response.device, "unknown");
    }

    /// The post-spawn whole-file rewrite must not drop the Anthropic endpoint
    /// a sibling thread of this same process published moments earlier.
    ///
    /// `rocm`'s `run_builtin_engine_serve_http` runs `rocmd`'s Anthropic proxy
    /// as a thread of the process running this adapter, and ROCMAI-550 put a
    /// host GPU probe (seconds, on Windows) inside the rewrite — wide enough
    /// for the publish to land inside it. Routing [`write_state`] through
    /// `rocm_core::write_json_state_file` is what carries it through; putting
    /// the raw `fs::write` back drops the key and fails this.
    #[test]
    fn a_running_state_write_keeps_the_anthropic_endpoint_this_process_published() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc-anthropic.json");
        rocm_core::publish_anthropic_endpoint(
            &path,
            "http://127.0.0.1:9931",
            rocm_core::AnthropicEndpointMode::Proxy,
        )
        .unwrap();

        write_running_state(
            &path,
            "svc-anthropic",
            "some/model",
            None,
            "127.0.0.1",
            8000,
            DevicePolicy::GpuRequired,
            Some(&[0]),
            None,
        )
        .unwrap();

        let state = read_service_state(&path).unwrap();
        assert_eq!(
            value_string(&state, rocm_core::ANTHROPIC_ENDPOINT_URL_STATE_KEY).as_deref(),
            Some("http://127.0.0.1:9931"),
            "the whole-file rewrite dropped the endpoint the proxy thread \
             published: {state}"
        );
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
