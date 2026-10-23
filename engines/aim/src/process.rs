// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, require_nonempty};
use rocm_engine_protocol::{
    DevicePolicy, ENGINE_RECIPE_CONTRACT_VERSION, EngineRecipeHint, GpuSelection, LaunchRequest,
    LaunchResponse, ResolveModelRequest, ResolveModelResponse, StopRequest, StopResponse,
};
use serde_json::json;
use std::time::{Duration, Instant};

use crate::ServeHttpRequest;

const STOP_GRACE: Duration = Duration::from_secs(10);
const READY_POLL_INTERVAL: Duration = Duration::from_millis(500);
const DEFAULT_READY_TIMEOUT: Duration = Duration::from_mins(5);

/// Time to wait for the AIM container to become ready before reporting failure.
///
/// Defaults to [`DEFAULT_READY_TIMEOUT`]. A cold start (image's first HF weight
/// download, first-decode compile) can exceed the default, so the timeout is
/// configurable via `ROCM_CLI_AIM_READY_TIMEOUT_SECS` (a positive integer number
/// of seconds) — mirrors `engines/vllm`'s `ROCM_CLI_VLLM_READY_TIMEOUT_SECS`.
fn aim_ready_timeout() -> Duration {
    resolve_aim_ready_timeout(std::env::var("ROCM_CLI_AIM_READY_TIMEOUT_SECS").ok())
}

fn resolve_aim_ready_timeout(override_value: Option<String>) -> Duration {
    override_value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map_or(DEFAULT_READY_TIMEOUT, Duration::from_secs)
}

/// Deterministic, Docker-name-sanitized container identity for a service.
/// This *is* the service's identity for the AIM engine — no PID/start-ticks
/// tracking, since `docker run -d` leaves no long-lived client process to
/// supervise (see plan: SIGKILL to a foreground `docker run` client orphans
/// the container, so this engine never signals a client at all).
pub(crate) fn container_name(service_id: &str) -> String {
    let sanitized: String = service_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    format!("rocm-aim-{sanitized}")
}

pub(crate) fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    require_nonempty(&request.model_ref, "model_ref")?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;

    let metadata = crate::registry::fetch_image_metadata(&request.model_ref)?;
    let mut warnings = Vec::new();

    let device_policy = match metadata.accelerator_type.as_deref() {
        Some("gpu") => DevicePolicy::GpuRequired,
        Some("cpu") => DevicePolicy::CpuOnly,
        _ => {
            warnings.push(
                "AIM image did not report AIM_ACCELERATOR_TYPE; assuming GPU-preferred".to_owned(),
            );
            DevicePolicy::GpuPreferred
        }
    };

    if device_policy != DevicePolicy::CpuOnly
        && let Some(host_gfx) = rocm_core::detect_host_gfx_target()
        && !metadata.pytorch_rocm_arch.is_empty()
        && !metadata
            .pytorch_rocm_arch
            .iter()
            .any(|arch| arch == &host_gfx)
    {
        warnings.push(format!(
            "AIM image targets {:?} but this host reports {host_gfx}; it may not run",
            metadata.pytorch_rocm_arch
        ));
    }

    Ok(ResolveModelResponse {
        // Must stay the Docker image reference, not `metadata.canonical_name`'s
        // human-readable HF-style id: `apps/rocm` re-threads `canonical_model_id`
        // back into Launch/ServeHttp as `model_ref` (service id generation,
        // `__engine-serve-http` argv, `docker pull`/`docker run`) for every
        // built-in engine uniformly. For vLLM/Lemonade that coincides with the
        // resolved HF repo id; for AIM it would swap in a non-pullable display
        // string instead of the image (found via live Strix Halo validation:
        // `docker pull meta-llama/Llama-3.2-1B-Instruct` failed outright).
        canonical_model_id: request.model_ref,
        task: "text-generation".to_owned(),
        source: "docker_registry".to_owned(),
        revision: "main".to_owned(),
        loader: "aim".to_owned(),
        trust_remote_code: false,
        chat_template_mode: "engine_default".to_owned(),
        dtype: "auto".to_owned(),
        device_policy,
        estimated_memory: "engine-reported".to_owned(),
        launch_defaults: json!({
            "endpoint_mode": "openai",
            "host": crate::DEFAULT_HOST,
            "port": crate::DEFAULT_AIM_PORT,
        }),
        engine_recipe,
        warnings,
    })
}

fn accepted_engine_recipe(
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<Option<EngineRecipeHint>> {
    let Some(hint) = engine_recipe else {
        return Ok(None);
    };
    if hint.engine != crate::ENGINE_NAME || hint.contract_version != ENGINE_RECIPE_CONTRACT_VERSION
    {
        bail!(
            "engine recipe hint is not valid for `{}` (contract {ENGINE_RECIPE_CONTRACT_VERSION})",
            crate::ENGINE_NAME
        );
    }
    Ok(Some(hint))
}

pub(crate) fn parse_engine_recipe_json(value: Option<String>) -> Result<Option<EngineRecipeHint>> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let hint: EngineRecipeHint =
        serde_json::from_str(&raw).context("failed to parse engine recipe JSON")?;
    accepted_engine_recipe(Some(hint))
}

pub(crate) fn launch_service(request: LaunchRequest) -> Result<LaunchResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    require_nonempty(&request.model_ref, "model_ref")?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let gpu_indices = rocm_engine_protocol::launch_gpu_indices(request.gpu_selection.as_ref());
    let device_policy = request.device_policy.unwrap_or(DevicePolicy::GpuPreferred);

    let paths = AppPaths::discover()?;
    let state_path = paths
        .engine_state_dir(crate::ENGINE_NAME)
        .join(format!("{}.json", request.service_id));

    run_container(
        &request.service_id,
        &request.model_ref,
        &request.host,
        request.port,
        &gpu_indices,
        engine_recipe.as_ref().map(|hint| &hint.extra_env),
    )?;
    let endpoint_url = endpoint_url(&request.host, request.port);
    crate::state::write_running_state(
        &request.service_id,
        &request.model_ref,
        &request.host,
        request.port,
        device_policy,
        engine_recipe.as_ref(),
    )?;

    let log_path = format!("docker://{}", container_name(&request.service_id));
    Ok(LaunchResponse {
        service_id: request.service_id,
        // ponytail: `docker run -d` backgrounds the container itself — there is
        // no long-lived client process to report a real PID for.
        pid: 0,
        endpoint_url,
        log_path,
        state_path: state_path.display().to_string(),
    })
}

/// Starts the named service's container if it isn't already running.
/// Container names are globally unique in Docker (unlike host PIDs), so
/// unconditionally calling `docker run --name X` twice for the same service
/// fails with a name conflict — `launch_service` and `serve_http` must agree
/// on "already running? skip `docker run`" instead of vLLM's host-process
/// model, where two independent spawns are safe because each gets its own PID.
fn run_container(
    service_id: &str,
    model_ref: &str,
    host: &str,
    port: u16,
    gpu_indices: &[u32],
    extra_env: Option<&std::collections::BTreeMap<String, String>>,
) -> Result<Option<String>> {
    let name = container_name(service_id);
    let metadata = crate::registry::fetch_image_metadata(model_ref).unwrap_or_default();
    if crate::docker::inspect_running(&name)?.unwrap_or(false) {
        return Ok(metadata.canonical_name);
    }

    if !crate::docker::image_present(model_ref)? {
        crate::docker::pull(model_ref)?;
    }

    let use_gpu = metadata.accelerator_type.as_deref() != Some("cpu") && !gpu_indices.is_empty();
    let render_gid = if use_gpu {
        crate::docker::render_group_gid()
    } else {
        None
    };

    let empty_env = std::collections::BTreeMap::default();
    crate::docker::run(
        &name,
        model_ref,
        host,
        port,
        crate::DEFAULT_AIM_PORT,
        if use_gpu { gpu_indices } else { &[] },
        render_gid,
        extra_env.unwrap_or(&empty_env),
    )?;
    Ok(metadata.canonical_name)
}

pub(crate) fn serve_http(request: ServeHttpRequest) -> Result<()> {
    let canonical_name = run_container(
        &request.service_id,
        &request.model_ref,
        &request.host,
        request.port,
        &request.gpu_indices,
        request.engine_recipe.as_ref().map(|hint| &hint.extra_env),
    )?;

    crate::state::write_running_state(
        &request.service_id,
        &request.model_ref,
        &request.host,
        request.port,
        request.device_policy,
        request.engine_recipe.as_ref(),
    )?;

    let name = container_name(&request.service_id);
    let timeout = aim_ready_timeout();
    // The container self-reports under the registry's HF-style canonical name
    // (e.g. `meta-llama/Llama-3.2-1B-Instruct`), not the Docker image reference
    // used for `docker pull`/`run` — matching `/v1/models` against the image ref
    // would never find it. Fall back to the image ref only when the registry
    // didn't advertise a canonical name.
    let expected_model_id = canonical_name.as_deref().unwrap_or(&request.model_ref);
    match wait_for_container_ready(
        &name,
        &request.host,
        request.port,
        expected_model_id,
        timeout,
    ) {
        Ok(()) => Ok(()),
        Err(error) => {
            crate::state::write_terminal_state(&request.state_path, "failed")?;
            Err(error)
        }
    }
}

/// Mirrors `engines/vllm/src/process.rs`'s `wait_for_vllm_ready` architecture:
/// early-exit check, monotonic timeout, listed-then-inference-verified gate.
/// `docker::inspect_running` replaces `child.try_wait()` since there is no
/// `std::process::Child` for a detached container.
fn wait_for_container_ready(
    name: &str,
    host: &str,
    port: u16,
    model_ref: &str,
    timeout: Duration,
) -> Result<()> {
    let start = Instant::now();
    let endpoint = endpoint_url(host, port);
    loop {
        if crate::docker::inspect_running(name)?.is_some_and(|running| !running) {
            let exit_code = crate::docker::inspect_exit_code(name)?.unwrap_or(-1);
            let tail = crate::docker::logs(name, 80).unwrap_or_default().join("\n");
            bail!("AIM container `{name}` exited early (exit code {exit_code}):\n{tail}");
        }
        if start.elapsed() > timeout {
            let tail = crate::docker::logs(name, 80).unwrap_or_default().join("\n");
            bail!(
                "AIM container `{name}` did not become ready within {}s:\n{tail}",
                timeout.as_secs()
            );
        }

        let listed =
            crate::state::query_loaded_model_endpoint(&endpoint, Some(model_ref)).unwrap_or(false);
        if listed
            && crate::state::query_inference_probe_endpoint(&endpoint, model_ref).unwrap_or(false)
        {
            return Ok(());
        }

        std::thread::sleep(READY_POLL_INTERVAL);
    }
}

pub(crate) fn stop_service(request: StopRequest) -> Result<StopResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let name = container_name(&request.service_id);
    let stopped = crate::docker::stop(&name, request.force, STOP_GRACE)?;
    if stopped {
        let state_path = AppPaths::discover()?
            .engine_state_dir(crate::ENGINE_NAME)
            .join(format!("{}.json", request.service_id));
        crate::state::write_terminal_state(&state_path, "stopped")?;
    }
    Ok(StopResponse {
        stopped,
        graceful: stopped && !request.force,
    })
}

pub(crate) fn endpoint_url(host: &str, port: u16) -> String {
    format!("{}/v1", rocm_core::format_http_base_url(host, port))
}

pub(crate) fn parse_device_policy_arg(policy: &str) -> Result<DevicePolicy> {
    match policy {
        "gpu" | "gpu_required" => Ok(DevicePolicy::GpuRequired),
        "gpu_preferred" => Ok(DevicePolicy::GpuPreferred),
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        other => bail!("unsupported device policy: {other}"),
    }
}

pub(crate) fn parse_gpu_selection_arg(value: Option<&str>) -> Result<Option<GpuSelection>> {
    value
        .map(|raw| GpuSelection::parse_cli_value(raw).map_err(anyhow::Error::msg))
        .transpose()
}

pub(crate) fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    Ok(rocm_engine_protocol::launch_gpu_indices(
        parse_gpu_selection_arg(value)?.as_ref(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aim_ready_timeout_uses_default_without_override() {
        assert_eq!(resolve_aim_ready_timeout(None), DEFAULT_READY_TIMEOUT);
    }

    #[test]
    fn aim_ready_timeout_honors_positive_override() {
        assert_eq!(
            resolve_aim_ready_timeout(Some(" 900 ".to_owned())),
            Duration::from_mins(15)
        );
    }

    #[test]
    fn aim_ready_timeout_ignores_invalid_or_zero_override() {
        assert_eq!(
            resolve_aim_ready_timeout(Some("0".to_owned())),
            DEFAULT_READY_TIMEOUT
        );
        assert_eq!(
            resolve_aim_ready_timeout(Some("not-a-number".to_owned())),
            DEFAULT_READY_TIMEOUT
        );
    }

    #[test]
    fn container_name_sanitizes_unsafe_characters() {
        assert_eq!(container_name("svc 1/a"), "rocm-aim-svc-1-a");
    }

    #[test]
    fn container_name_preserves_safe_characters() {
        assert_eq!(container_name("svc_1.a-b"), "rocm-aim-svc_1.a-b");
    }

    #[test]
    fn parse_device_policy_arg_accepts_known_aliases() {
        assert_eq!(
            parse_device_policy_arg("gpu").unwrap(),
            DevicePolicy::GpuRequired
        );
        assert_eq!(
            parse_device_policy_arg("gpu_preferred").unwrap(),
            DevicePolicy::GpuPreferred
        );
        assert_eq!(
            parse_device_policy_arg("cpu_only").unwrap(),
            DevicePolicy::CpuOnly
        );
        assert!(parse_device_policy_arg("bogus").is_err());
    }
}
