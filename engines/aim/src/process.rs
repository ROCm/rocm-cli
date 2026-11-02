// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::require_nonempty;
use rocm_engine_protocol::{
    DevicePolicy, ENGINE_RECIPE_CONTRACT_VERSION, EngineRecipeHint, GpuSelection, LaunchRequest,
    LaunchResponse, ResolveModelRequest, ResolveModelResponse, StopRequest, StopResponse,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::ServeHttpRequest;
use crate::docker::ContainerState;
use crate::registry::AimImageMetadata;

const STOP_GRACE: Duration = Duration::from_secs(10);
const READY_POLL_INTERVAL: Duration = Duration::from_millis(500);
const DEFAULT_READY_TIMEOUT: Duration = Duration::from_mins(5);
/// Consecutive failed `docker inspect` calls (daemon down or erroring, as
/// opposed to a container that is definitively gone) tolerated before the
/// supervisor gives up: 60 polls is about 30 seconds.
const MAX_CONSECUTIVE_INSPECT_ERRORS: u32 = 60;

/// Env vars this engine sets itself; a caller-supplied `extra_env` may not
/// override them.
const RESERVED_ENV_KEYS: [&str; 2] = ["HIP_VISIBLE_DEVICES", "VLLM_API_KEY"];

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
/// The container is stopped and removed by name (`docker kill`/`rm -f`), never
/// by signalling a client process, so it cannot be orphaned by a killed
/// `docker run` client. `serve_http` is the long-lived process rocmd
/// supervises: it blocks until the container exits and records how it ended.
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

/// Image metadata for the pre-launch `ResolveModel` step: the local daemon's
/// copy when the image is already present, otherwise a best-effort Docker Hub
/// lookup. Both are hints; callers must tolerate failure.
fn lookup_image_metadata(image_ref: &str) -> Result<AimImageMetadata> {
    let image_ref = crate::registry::normalize_image_ref(image_ref);
    if let Some(local) = crate::registry::local_image_metadata(&image_ref) {
        return Ok(local);
    }
    crate::registry::fetch_image_metadata(&image_ref)
}

pub(crate) fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    resolve_model_response_with(request, lookup_image_metadata)
}

fn resolve_model_response_with(
    request: ResolveModelRequest,
    lookup: impl FnOnce(&str) -> Result<AimImageMetadata>,
) -> Result<ResolveModelResponse> {
    require_nonempty(&request.model_ref, "model_ref")?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;

    let mut warnings = Vec::new();
    let metadata = lookup(&request.model_ref).unwrap_or_else(|error| {
        warnings.push(format!(
            "could not read AIM image metadata for `{}` ({error:#}); it will be read from the \
             image after the pull",
            request.model_ref
        ));
        AimImageMetadata::default()
    });

    // The caller's `--device` wins; image metadata only fills in when none was given.
    let device_policy =
        request
            .device_policy
            .unwrap_or(match metadata.accelerator_type.as_deref() {
                Some("gpu") => DevicePolicy::GpuRequired,
                Some("cpu") => DevicePolicy::CpuOnly,
                _ => DevicePolicy::GpuPreferred,
            });

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
        // Must stay the Docker image reference: `apps/rocm` re-threads
        // `canonical_model_id` back into Launch/ServeHttp as `model_ref`
        // (service id generation, `__engine-serve-http` argv, `docker pull`/`run`)
        // for every built-in engine uniformly. The name the container serves
        // under (the image's HF-style canonical name) is reported separately in
        // `served_model_id`, which readiness, the smoke test and the summary use.
        canonical_model_id: request.model_ref,
        served_model_id: metadata.canonical_name,
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

    let state_path = crate::state::state_path(&request.service_id);

    let served_model_id = run_container(
        &request.service_id,
        &request.model_ref,
        &request.host,
        request.port,
        &device_policy,
        &gpu_indices,
        engine_recipe.as_ref().map(|hint| &hint.extra_env),
    )?;
    let endpoint_url = endpoint_url(&request.host, request.port);
    crate::state::write_running_state(
        &state_path,
        &request.service_id,
        &request.model_ref,
        served_model_id.as_deref(),
        &request.host,
        request.port,
        device_policy,
        engine_recipe.as_ref(),
    )?;

    let log_path = format!("docker://{}", container_name(&request.service_id));
    Ok(LaunchResponse {
        service_id: request.service_id,
        // ponytail: `docker run -d` backgrounds the container itself, so there is
        // no client process to report a real PID for.
        pid: 0,
        endpoint_url,
        log_path,
        state_path: state_path.display().to_string(),
    })
}

/// Which GPUs, if any, the container may use.
#[derive(Debug, PartialEq, Eq)]
struct GpuPlan {
    available: bool,
    indices: Vec<u32>,
}

/// Resolves the caller's GPU request against what the host can actually use,
/// the way `engines/vllm` does: an authoritative empty probe means no usable
/// GPU, an empty request picks the first usable device, and an unknown probe
/// (WSL and other unprobeable surfaces) stays permissive. Native Windows has no
/// GPU passthrough shape (`docker::gpu_device_args`), so it never has a GPU.
fn plan_gpus(requested: &[u32], usable: Option<Vec<u32>>, windows: bool) -> Result<GpuPlan> {
    let none = GpuPlan {
        available: false,
        indices: Vec::new(),
    };
    if windows {
        return Ok(none);
    }
    let Some(usable) = usable else {
        return Ok(GpuPlan {
            available: true,
            indices: requested.to_vec(),
        });
    };
    if usable.is_empty() {
        return Ok(none);
    }
    if requested.is_empty() {
        return Ok(GpuPlan {
            available: true,
            indices: vec![usable[0]],
        });
    }
    if let Some(missing) = requested.iter().find(|index| !usable.contains(index)) {
        bail!(
            "requested GPU {missing} is not available on this host; usable GPU indices: {usable:?}"
        );
    }
    Ok(GpuPlan {
        available: true,
        indices: requested.to_vec(),
    })
}

/// Starts the named service's container unless it is already running, and
/// returns the model name the image says it serves under, if known.
///
/// Container names are unique in Docker, so `docker run --name X` against a
/// name still held by an exited container (a crash, a reboot, a failed
/// readiness check) would fail with "name already in use" forever; a stale
/// container is therefore removed first.
fn run_container(
    service_id: &str,
    model_ref: &str,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    requested_gpus: &[u32],
    extra_env: Option<&BTreeMap<String, String>>,
) -> Result<Option<String>> {
    let name = container_name(service_id);
    // Normalize once so the pull, the metadata and the run all use the same image.
    let image_ref = crate::registry::normalize_image_ref(model_ref);

    match crate::docker::container_state(&name)? {
        ContainerState::Running => {
            return Ok(crate::registry::local_image_metadata(&image_ref)
                .and_then(|metadata| metadata.canonical_name));
        }
        ContainerState::Exited(_) => crate::docker::remove(&name)?,
        ContainerState::Missing => {}
    }

    if !crate::docker::image_present(&image_ref)? {
        crate::docker::pull(&image_ref)?;
    }
    // Read from the pulled image itself, so this works for any registry and
    // offline, unlike the Docker Hub lookup.
    let metadata = crate::registry::local_image_metadata(&image_ref).unwrap_or_default();

    let gpus = plan_gpus(
        requested_gpus,
        rocm_core::usable_amd_gpu_indices(),
        cfg!(windows),
    )?;
    let image_wants_cpu = metadata.accelerator_type.as_deref() == Some("cpu");
    let use_gpu = resolve_use_gpu(
        service_id,
        &image_ref,
        device_policy,
        gpus.available,
        image_wants_cpu,
    )?;
    let render_gid = if use_gpu {
        crate::docker::render_group_gid()
    } else {
        None
    };

    let mut env = extra_env.cloned().unwrap_or_default();
    if let Some(key) = RESERVED_ENV_KEYS.iter().find(|key| env.contains_key(**key)) {
        bail!("engine env `{key}` is managed by rocm-cli and cannot be overridden");
    }
    // AIM images bundle vLLM's OpenAI server, so it is `VLLM_API_KEY` that this
    // engine's actual server enforces. The container is the only process this
    // engine runs, so without this the require-api-key setting a public
    // `rocm serve --engine aim` was given is silently never enforced. The value
    // reaches docker through a private env file, never argv.
    if let Some(api_key) = rocm_engine_protocol::resolve_endpoint_api_key() {
        env.insert("VLLM_API_KEY".to_owned(), api_key);
    }
    crate::docker::run(&crate::docker::RunSpec {
        name: &name,
        image: &image_ref,
        host,
        port,
        container_port: crate::DEFAULT_AIM_PORT,
        use_gpu,
        gpu_indices: if use_gpu { &gpus.indices } else { &[] },
        render_gid,
        env: &env,
    })?;
    Ok(metadata.canonical_name)
}

/// Enforces `device_policy` instead of treating GPU use as a soft heuristic:
/// `CpuOnly` never requests a GPU, and `GpuRequired` fails fast instead of
/// silently falling back to a CPU launch when no GPU turns out to be usable.
fn resolve_use_gpu(
    service_id: &str,
    model_ref: &str,
    device_policy: &DevicePolicy,
    gpu_usable: bool,
    image_wants_cpu: bool,
) -> Result<bool> {
    Ok(match device_policy {
        DevicePolicy::CpuOnly => false,
        // `GpuPreferred` is normalised to `GpuRequired`, as the sibling engines
        // do: no silent CPU fallback (AGENTS.md section 6). CPU is reachable
        // only through an explicit `CpuOnly`.
        DevicePolicy::GpuRequired | DevicePolicy::GpuPreferred => {
            if !gpu_usable || image_wants_cpu {
                bail!(
                    "AIM service `{service_id}` has device_policy=gpu_required but no GPU is \
                     usable for `{model_ref}` (native Windows has no GPU passthrough, the host \
                     reports no usable GPU, or the image reports AIM_ACCELERATOR_TYPE=cpu)"
                );
            }
            true
        }
    })
}

pub(crate) fn serve_http(request: ServeHttpRequest) -> Result<()> {
    let name = container_name(&request.service_id);
    let label_name = run_container(
        &request.service_id,
        &request.model_ref,
        &request.host,
        request.port,
        &request.device_policy,
        &request.gpu_indices,
        request.engine_recipe.as_ref().map(|hint| &hint.extra_env),
    )
    .inspect_err(|_| {
        // `docker run -d` can create the container and then fail to start it
        // (port already allocated); service ids are unique, so nothing would
        // ever reclaim it.
        let _ = crate::docker::remove(&name);
    })?;

    crate::state::write_running_state(
        &request.state_path,
        &request.service_id,
        &request.model_ref,
        label_name.as_deref(),
        &request.host,
        request.port,
        request.device_policy,
        request.engine_recipe.as_ref(),
    )?;

    // The container serves under the image's HF-style canonical name, not the
    // Docker image reference used for `docker pull`/`run`. When the image didn't
    // label one, wait for the container to list whatever it serves instead.
    match wait_for_container_ready(
        &name,
        &request.host,
        request.port,
        label_name.as_deref(),
        aim_ready_timeout(),
    ) {
        Ok(served_model_id) => {
            crate::state::record_served_model_id(&request.state_path, &served_model_id)?;
            let exit = wait_for_exit(
                || crate::docker::container_state(&name),
                || std::thread::sleep(READY_POLL_INTERVAL),
            );
            finish_after_exit(&name, &request.state_path, exit)
        }
        Err(error) => {
            // The logs are already in `error`; remove the container so a failed
            // launch leaks neither a running container nothing will signal
            // again nor the stopped one that would hold its name.
            let _ = crate::docker::remove(&name);
            crate::state::write_terminal_state(&request.state_path, "failed")?;
            Err(error)
        }
    }
}

/// How a supervised container ended.
#[derive(Debug, PartialEq, Eq)]
enum ContainerExit {
    Code(i32),
    /// Removed out from under the supervisor (`docker rm -f`, `docker system prune`).
    Gone,
    /// The daemon stayed unreachable past the tolerance.
    DaemonLost,
}

/// Blocks until the container stops, mirroring `engines/vllm` and
/// `engines/lemonade`'s blocking `child.wait()`: `apps/rocmd`'s generic
/// supervisor treats this process's exit as the service's exit, so returning
/// early would make a live container read `stopped`. A definitively missing
/// container ends the wait; transient `docker inspect` errors are tolerated
/// only [`MAX_CONSECUTIVE_INSPECT_ERRORS`] times in a row.
fn wait_for_exit(
    mut poll: impl FnMut() -> Result<ContainerState>,
    mut pause: impl FnMut(),
) -> ContainerExit {
    let mut consecutive_errors = 0;
    loop {
        match poll() {
            Ok(ContainerState::Running) => consecutive_errors = 0,
            Ok(ContainerState::Exited(code)) => return ContainerExit::Code(code),
            Ok(ContainerState::Missing) => return ContainerExit::Gone,
            Err(_) => {
                consecutive_errors += 1;
                if consecutive_errors >= MAX_CONSECUTIVE_INSPECT_ERRORS {
                    return ContainerExit::DaemonLost;
                }
            }
        }
        pause();
    }
}

/// Records how the container ended, as `engines/vllm` does for its child: a
/// deliberate stop or clean exit is `stopped`; anything else is `failed` and
/// surfaces as an error (non-zero exit) so rocmd's recovery sees a crash.
fn finish_after_exit(name: &str, state_path: &Path, exit: ContainerExit) -> Result<()> {
    // `stop_service` records its intent before killing, so a kill (exit 137)
    // it caused is not mistaken for a crash.
    let deliberate = matches!(
        crate::state::read_status(state_path).as_deref(),
        Some("stopping" | "stopped")
    );
    if deliberate || exit == ContainerExit::Code(0) {
        let _ = crate::docker::remove(name);
        return crate::state::write_terminal_state(state_path, "stopped");
    }
    crate::state::write_terminal_state(state_path, "failed")?;
    match exit {
        ContainerExit::Code(code) => bail!("AIM container `{name}` exited with code {code}"),
        ContainerExit::Gone => bail!(
            "AIM container `{name}` was removed while it was serving (`docker rm` or a prune?)"
        ),
        ContainerExit::DaemonLost => {
            bail!("lost contact with the Docker daemon while supervising AIM container `{name}`")
        }
    }
}

/// Mirrors `engines/vllm/src/process.rs`'s `wait_for_vllm_ready` architecture:
/// early-exit check, monotonic timeout, listed-then-inference-verified gate.
/// Returns the model name the server serves under (`expected_model` when the
/// image labelled one, otherwise whatever the server lists).
fn wait_for_container_ready(
    name: &str,
    host: &str,
    port: u16,
    expected_model: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    let start = Instant::now();
    let endpoint = endpoint_url(host, port);
    let mut consecutive_errors = 0;
    loop {
        match crate::docker::container_state(name) {
            Ok(ContainerState::Running) => consecutive_errors = 0,
            Ok(ContainerState::Exited(exit_code)) => {
                let tail = crate::docker::logs(name, 80).unwrap_or_default().join("\n");
                bail!("AIM container `{name}` exited early (exit code {exit_code}):\n{tail}");
            }
            Ok(ContainerState::Missing) => {
                bail!("AIM container `{name}` no longer exists; it was removed during startup")
            }
            Err(error) => {
                consecutive_errors += 1;
                if consecutive_errors >= MAX_CONSECUTIVE_INSPECT_ERRORS {
                    return Err(error.context(format!(
                        "lost contact with the Docker daemon while starting AIM container `{name}`"
                    )));
                }
            }
        }
        if start.elapsed() > timeout {
            let tail = crate::docker::logs(name, 80).unwrap_or_default().join("\n");
            bail!(
                "AIM container `{name}` did not become ready within {}s:\n{tail}",
                timeout.as_secs()
            );
        }

        let served = match expected_model {
            Some(model) => crate::state::query_loaded_model_endpoint(&endpoint, Some(model))
                .unwrap_or(false)
                .then(|| model.to_owned()),
            None => crate::state::discover_served_model(&endpoint),
        };
        if let Some(model) = served
            && crate::state::query_inference_probe_endpoint(&endpoint, &model).unwrap_or(false)
        {
            return Ok(model);
        }

        std::thread::sleep(READY_POLL_INTERVAL);
    }
}

pub(crate) fn stop_service(request: StopRequest) -> Result<StopResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let name = container_name(&request.service_id);
    let state_path = crate::state::state_path(&request.service_id);
    let prior_status = crate::state::read_status(&state_path);
    // Record the intent first: the container's supervisor (`serve_http`) sees the
    // exit within a poll interval and must read a deliberate stop, not a crash.
    crate::state::write_status(&state_path, "stopping")?;
    let stopped = match crate::docker::stop(&name, request.force, STOP_GRACE) {
        Ok(stopped) => stopped,
        Err(error) => {
            if let Some(status) = prior_status {
                let _ = crate::state::write_status(&state_path, &status);
            }
            return Err(error);
        }
    };
    // Release the name and the writable layer (downloaded weights) so a later
    // restart can `docker run` the same name again.
    crate::docker::remove(&name)?;
    crate::state::write_terminal_state(&state_path, "stopped")?;
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
    use std::cell::Cell;

    fn resolve_request(device_policy: Option<DevicePolicy>) -> ResolveModelRequest {
        ResolveModelRequest {
            model_ref: "org/aim-foo:1".to_owned(),
            runtime_id: None,
            device_policy,
            recipe_override: None,
            engine_recipe: None,
        }
    }

    fn metadata(accelerator_type: Option<&str>, canonical_name: Option<&str>) -> AimImageMetadata {
        AimImageMetadata {
            accelerator_type: accelerator_type.map(ToOwned::to_owned),
            canonical_name: canonical_name.map(ToOwned::to_owned),
            ..AimImageMetadata::default()
        }
    }

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

    #[test]
    fn resolve_use_gpu_cpu_only_never_uses_gpu() {
        assert!(!resolve_use_gpu("svc", "model", &DevicePolicy::CpuOnly, true, false).unwrap());
    }

    #[test]
    fn resolve_use_gpu_gpu_required_uses_gpu_when_usable() {
        assert!(resolve_use_gpu("svc", "model", &DevicePolicy::GpuRequired, true, false).unwrap());
    }

    #[test]
    fn resolve_use_gpu_gpu_required_fails_fast_when_no_gpu_usable() {
        assert!(resolve_use_gpu("svc", "model", &DevicePolicy::GpuRequired, false, false).is_err());
    }

    #[test]
    fn resolve_use_gpu_gpu_required_fails_fast_when_image_wants_cpu() {
        assert!(resolve_use_gpu("svc", "model", &DevicePolicy::GpuRequired, true, true).is_err());
    }

    #[test]
    fn resolve_use_gpu_gpu_preferred_never_falls_back_to_cpu() {
        assert!(
            resolve_use_gpu("svc", "model", &DevicePolicy::GpuPreferred, false, false).is_err()
        );
        assert!(resolve_use_gpu("svc", "model", &DevicePolicy::GpuPreferred, true, true).is_err());
        assert!(resolve_use_gpu("svc", "model", &DevicePolicy::GpuPreferred, true, false).unwrap());
    }

    #[test]
    fn plan_gpus_resolves_an_empty_request_to_the_first_usable_gpu() {
        assert_eq!(
            plan_gpus(&[], Some(vec![1, 2]), false).unwrap(),
            GpuPlan {
                available: true,
                indices: vec![1]
            }
        );
    }

    #[test]
    fn plan_gpus_keeps_a_valid_request_and_rejects_an_unusable_one() {
        assert_eq!(
            plan_gpus(&[2], Some(vec![1, 2]), false).unwrap().indices,
            vec![2]
        );
        assert!(plan_gpus(&[5], Some(vec![1, 2]), false).is_err());
    }

    #[test]
    fn plan_gpus_reports_no_gpu_for_an_empty_probe_and_on_windows() {
        assert!(!plan_gpus(&[], Some(vec![]), false).unwrap().available);
        assert!(!plan_gpus(&[0], Some(vec![0]), true).unwrap().available);
    }

    #[test]
    fn plan_gpus_stays_permissive_when_the_probe_is_unavailable() {
        assert_eq!(
            plan_gpus(&[], None, false).unwrap(),
            GpuPlan {
                available: true,
                indices: vec![]
            }
        );
    }

    #[test]
    fn resolve_reports_the_served_name_and_keeps_the_image_ref_as_canonical() {
        let response = resolve_model_response_with(resolve_request(None), |_| {
            Ok(metadata(Some("gpu"), Some("Org/Served-Model")))
        })
        .unwrap();
        assert_eq!(response.canonical_model_id, "org/aim-foo:1");
        assert_eq!(
            response.served_model_id.as_deref(),
            Some("Org/Served-Model")
        );
        assert_eq!(response.device_policy, DevicePolicy::GpuRequired);
    }

    #[test]
    fn resolve_honors_the_callers_device_policy_over_image_metadata() {
        let response =
            resolve_model_response_with(resolve_request(Some(DevicePolicy::CpuOnly)), |_| {
                Ok(metadata(Some("gpu"), None))
            })
            .unwrap();
        assert_eq!(response.device_policy, DevicePolicy::CpuOnly);
    }

    #[test]
    fn resolve_tolerates_a_failed_metadata_lookup_with_a_warning() {
        let response =
            resolve_model_response_with(resolve_request(None), |_| anyhow::bail!("offline"))
                .unwrap();
        assert_eq!(response.served_model_id, None);
        assert_eq!(response.device_policy, DevicePolicy::GpuPreferred);
        assert!(
            response
                .warnings
                .iter()
                .any(|warning| warning.contains("offline")),
            "{:?}",
            response.warnings
        );
    }

    #[test]
    fn wait_for_exit_returns_the_exit_code() {
        let mut states = vec![
            Ok(ContainerState::Running),
            Ok(ContainerState::Running),
            Ok(ContainerState::Exited(137)),
        ]
        .into_iter();
        let exit = wait_for_exit(|| states.next().unwrap(), || {});
        assert_eq!(exit, ContainerExit::Code(137));
    }

    #[test]
    fn wait_for_exit_ends_when_the_container_is_removed_instead_of_spinning() {
        let mut states = vec![Ok(ContainerState::Running), Ok(ContainerState::Missing)].into_iter();
        assert_eq!(
            wait_for_exit(|| states.next().unwrap(), || {}),
            ContainerExit::Gone
        );
    }

    #[test]
    fn wait_for_exit_tolerates_a_few_transient_errors_but_not_a_dead_daemon() {
        let calls = Cell::new(0);
        let transient = wait_for_exit(
            || {
                calls.set(calls.get() + 1);
                match calls.get() {
                    1..=3 => anyhow::bail!("daemon hiccup"),
                    4 => Ok(ContainerState::Running),
                    _ => Ok(ContainerState::Exited(0)),
                }
            },
            || {},
        );
        assert_eq!(transient, ContainerExit::Code(0));

        let polls = Cell::new(0);
        let dead = wait_for_exit(
            || {
                polls.set(polls.get() + 1);
                anyhow::bail!("daemon down")
            },
            || {},
        );
        assert_eq!(dead, ContainerExit::DaemonLost);
        assert_eq!(polls.get(), MAX_CONSECUTIVE_INSPECT_ERRORS);
    }

    #[test]
    fn finish_after_exit_records_failed_for_a_crash_and_stopped_for_a_clean_exit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc.json");
        crate::state::write_status(&path, "running").unwrap();

        // `docker rm` is best-effort here and may not exist on the test host.
        let error = finish_after_exit("aim-x", &path, ContainerExit::Code(137))
            .expect_err("a crash must surface as an error so recovery runs");
        assert!(error.to_string().contains("137"), "{error}");
        assert_eq!(crate::state::read_status(&path).as_deref(), Some("failed"));

        finish_after_exit("aim-x", &path, ContainerExit::Code(0)).unwrap();
        assert_eq!(crate::state::read_status(&path).as_deref(), Some("stopped"));
    }

    #[test]
    fn finish_after_exit_treats_a_deliberate_stop_as_stopped_even_after_a_kill() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc.json");
        crate::state::write_status(&path, "stopping").unwrap();
        finish_after_exit("aim-x", &path, ContainerExit::Code(137)).unwrap();
        assert_eq!(crate::state::read_status(&path).as_deref(), Some("stopped"));

        crate::state::write_status(&path, "stopping").unwrap();
        finish_after_exit("aim-x", &path, ContainerExit::Gone).unwrap();
        assert_eq!(crate::state::read_status(&path).as_deref(), Some("stopped"));
    }

    #[test]
    fn finish_after_exit_fails_when_the_container_vanished_unexpectedly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("svc.json");
        crate::state::write_status(&path, "running").unwrap();
        assert!(finish_after_exit("aim-x", &path, ContainerExit::Gone).is_err());
        assert_eq!(crate::state::read_status(&path).as_deref(), Some("failed"));
        assert!(finish_after_exit("aim-x", &path, ContainerExit::DaemonLost).is_err());
    }
}

/// `run_container`/`wait_for_container_ready`/`serve_http`/`stop_service` all
/// shell out to `docker`, so each scenario re-spawns this test binary with the
/// shared fake `docker` on `PATH` (see `docker::subprocess_tests`), a data dir
/// under the scratch dir, and, where readiness is reached, a minimal in-process
/// OpenAI-compatible server on loopback. `docker image inspect` is answered by
/// the fake, so nothing here touches the network or the real data dir.
#[cfg(all(test, unix))]
mod subprocess_tests {
    use super::*;
    use crate::docker::subprocess_tests::{in_subprocess, run_with_fake_docker};
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    const IMAGE: &str = "org/aim-fixture:1";
    const SERVED: &str = "Org/Served-Model";

    /// Serves `GET /v1/models` (listing `model`) and `POST /v1/chat/completions`
    /// (always 200) on an ephemeral loopback port; returns the port.
    fn spawn_fake_openai(model: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut request_line = String::new();
                    reader.read_line(&mut request_line).ok();
                    let mut content_length = 0usize;
                    loop {
                        let mut header = String::new();
                        if reader.read_line(&mut header).unwrap_or(0) == 0 || header == "\r\n" {
                            break;
                        }
                        if let Some(value) =
                            header.to_ascii_lowercase().strip_prefix("content-length:")
                        {
                            content_length = value.trim().parse().unwrap_or(0);
                        }
                    }
                    let mut body = vec![0; content_length];
                    reader.read_exact(&mut body).ok();
                    let payload = if request_line.starts_with("GET /v1/models") {
                        format!(r#"{{"data":[{{"id":"{model}"}}]}}"#)
                    } else {
                        r#"{"choices":[]}"#.to_owned()
                    };
                    let mut stream = stream;
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                        payload.len()
                    );
                });
            }
        });
        port
    }

    fn data_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var_os("ROCM_CLI_DATA_DIR").expect("data dir"))
    }

    fn serve_request(
        service_id: &str,
        port: u16,
        state_path: std::path::PathBuf,
    ) -> ServeHttpRequest {
        ServeHttpRequest {
            service_id: service_id.to_owned(),
            model_ref: IMAGE.to_owned(),
            host: "127.0.0.1".to_owned(),
            port,
            device_policy: DevicePolicy::CpuOnly,
            gpu_indices: vec![],
            state_path,
            engine_recipe: None,
        }
    }

    fn status_of(path: &Path) -> Option<String> {
        crate::state::read_status(path)
    }

    const LABELLED_IMAGE: (&str, &str) = (
        "FAKE_DOCKER_IMAGE_CONFIG",
        r#"{"Env":["AIM_ACCELERATOR_TYPE=cpu"],"Labels":{"com.amd.aim.model.canonicalName":"Org/Served-Model"}}"#,
    );

    #[test]
    fn serve_http_reaches_ready_then_marks_a_clean_container_exit_stopped() {
        if in_subprocess() {
            let port = spawn_fake_openai(SERVED);
            let state_path = data_dir().join("svc.json");
            serve_http(serve_request("svc-clean", port, state_path.clone())).unwrap();
            assert_eq!(status_of(&state_path).as_deref(), Some("stopped"));
            let state = crate::state::read_service_state(&state_path).unwrap();
            assert_eq!(state["served_model_id"], SERVED);
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::serve_http_reaches_ready_then_marks_a_clean_container_exit_stopped",
            &[
                LABELLED_IMAGE,
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                (
                    "FAKE_DOCKER_INSPECT_SEQUENCE",
                    "missing,true 0,true 0,false 0",
                ),
            ],
        );
        let argv = fixture.argv();
        assert!(
            argv.iter()
                .any(|line| line.starts_with("run -d --name rocm-aim-svc-clean")),
            "{argv:?}"
        );
        assert!(
            argv.contains(&"rm -f rocm-aim-svc-clean".to_owned()),
            "{argv:?}"
        );
    }

    #[test]
    fn serve_http_discovers_the_served_name_when_the_image_has_no_label() {
        if in_subprocess() {
            let port = spawn_fake_openai(SERVED);
            let state_path = data_dir().join("svc.json");
            serve_http(serve_request("svc-discover", port, state_path.clone())).unwrap();
            let state = crate::state::read_service_state(&state_path).unwrap();
            assert_eq!(state["served_model_id"], SERVED);
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::serve_http_discovers_the_served_name_when_the_image_has_no_label",
            &[
                ("FAKE_DOCKER_IMAGE_CONFIG", r#"{"Env":[]}"#),
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                (
                    "FAKE_DOCKER_INSPECT_SEQUENCE",
                    "missing,true 0,true 0,false 0",
                ),
            ],
        );
    }

    #[test]
    fn serve_http_records_failed_and_errors_when_the_container_crashes_after_ready() {
        if in_subprocess() {
            let port = spawn_fake_openai(SERVED);
            let state_path = data_dir().join("svc.json");
            let error = serve_http(serve_request("svc-crash", port, state_path.clone()))
                .expect_err("a crash after readiness must be an error so rocmd recovers it");
            assert!(error.to_string().contains("137"), "{error}");
            assert_eq!(status_of(&state_path).as_deref(), Some("failed"));
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::serve_http_records_failed_and_errors_when_the_container_crashes_after_ready",
            &[
                LABELLED_IMAGE,
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                (
                    "FAKE_DOCKER_INSPECT_SEQUENCE",
                    "missing,true 0,true 0,false 137",
                ),
            ],
        );
    }

    #[test]
    fn serve_http_records_failed_when_the_container_is_removed_while_serving() {
        if in_subprocess() {
            let port = spawn_fake_openai(SERVED);
            let state_path = data_dir().join("svc.json");
            serve_http(serve_request("svc-gone", port, state_path.clone()))
                .expect_err("a vanished container must end the supervisor, not spin");
            assert_eq!(status_of(&state_path).as_deref(), Some("failed"));
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::serve_http_records_failed_when_the_container_is_removed_while_serving",
            &[
                LABELLED_IMAGE,
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                (
                    "FAKE_DOCKER_INSPECT_SEQUENCE",
                    "missing,true 0,true 0,missing",
                ),
            ],
        );
    }

    #[test]
    fn serve_http_removes_the_container_and_marks_failed_on_readiness_failure() {
        if in_subprocess() {
            let state_path = data_dir().join("svc.json");
            serve_http(serve_request("svc-not-ready", 18084, state_path.clone()))
                .expect_err("readiness failure should surface as an error");
            assert_eq!(status_of(&state_path).as_deref(), Some("failed"));
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::serve_http_removes_the_container_and_marks_failed_on_readiness_failure",
            &[
                LABELLED_IMAGE,
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "missing,false 1"),
                ("FAKE_DOCKER_LOGS", "widget crashed"),
            ],
        );
        assert!(
            fixture
                .argv()
                .contains(&"rm -f rocm-aim-svc-not-ready".to_owned()),
            "{:?}",
            fixture.argv()
        );
    }

    #[test]
    fn run_container_removes_a_stale_exited_container_before_running_again() {
        if in_subprocess() {
            run_container(
                "svc-stale",
                IMAGE,
                "127.0.0.1",
                18085,
                &DevicePolicy::CpuOnly,
                &[],
                None,
            )
            .unwrap();
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::run_container_removes_a_stale_exited_container_before_running_again",
            &[
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "false 137"),
            ],
        );
        let argv = fixture.argv();
        let rm = argv
            .iter()
            .position(|line| line == "rm -f rocm-aim-svc-stale");
        let run = argv
            .iter()
            .position(|line| line.starts_with("run -d --name rocm-aim-svc-stale"));
        assert!(
            matches!((rm, run), (Some(rm), Some(run)) if rm < run),
            "the stale container must be removed before `docker run`: {argv:?}"
        );
    }

    #[test]
    fn run_container_skips_docker_run_when_already_running() {
        if in_subprocess() {
            run_container(
                "svc-already-running",
                IMAGE,
                "127.0.0.1",
                18080,
                &DevicePolicy::GpuPreferred,
                &[],
                None,
            )
            .unwrap();
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::run_container_skips_docker_run_when_already_running",
            &[
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "true 0"),
                ("FAKE_DOCKER_PULL_EXIT", "1"),
                ("FAKE_DOCKER_RUN_EXIT", "1"),
            ],
        );
        assert!(
            !fixture.argv().iter().any(|line| line.starts_with("run ")),
            "{:?}",
            fixture.argv()
        );
    }

    #[test]
    fn run_container_pulls_when_image_absent_then_runs() {
        if in_subprocess() {
            run_container(
                "svc-pull-then-run",
                IMAGE,
                "127.0.0.1",
                18081,
                &DevicePolicy::CpuOnly,
                &[],
                None,
            )
            .unwrap();
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::run_container_pulls_when_image_absent_then_runs",
            &[
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "missing"),
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "1"),
            ],
        );
        assert!(
            fixture
                .argv()
                .iter()
                .any(|line| line == "pull org/aim-fixture:1")
        );
    }

    #[test]
    fn run_container_bails_when_pull_fails() {
        if in_subprocess() {
            let error = run_container(
                "svc-pull-fails",
                IMAGE,
                "127.0.0.1",
                18082,
                &DevicePolicy::GpuPreferred,
                &[],
                None,
            )
            .expect_err("pull failure should bail");
            assert!(error.to_string().contains("pull"), "{error}");
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::run_container_bails_when_pull_fails",
            &[
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "missing"),
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "1"),
                ("FAKE_DOCKER_PULL_EXIT", "1"),
            ],
        );
    }

    #[test]
    fn run_container_bails_when_docker_run_fails() {
        if in_subprocess() {
            let error = run_container(
                "svc-run-fails",
                IMAGE,
                "127.0.0.1",
                18083,
                &DevicePolicy::CpuOnly,
                &[],
                None,
            )
            .expect_err("docker run failure should bail");
            let message = error.to_string();
            assert!(message.contains("rocm-aim-svc-run-fails"), "{message}");
            assert!(message.contains("boom"), "{message}");
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::run_container_bails_when_docker_run_fails",
            &[
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "missing"),
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                ("FAKE_DOCKER_RUN_EXIT", "1"),
                ("FAKE_DOCKER_RUN_STDERR", "boom"),
            ],
        );
    }

    #[test]
    fn run_container_rejects_reserved_env_overrides_before_running_docker() {
        if in_subprocess() {
            for key in ["VLLM_API_KEY", "HIP_VISIBLE_DEVICES"] {
                let env = BTreeMap::from([(key.to_owned(), "x".to_owned())]);
                let error = run_container(
                    "svc-reserved",
                    IMAGE,
                    "127.0.0.1",
                    18086,
                    &DevicePolicy::CpuOnly,
                    &[],
                    Some(&env),
                )
                .expect_err("reserved key must be rejected");
                assert!(error.to_string().contains(key), "{error}");
            }
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::run_container_rejects_reserved_env_overrides_before_running_docker",
            &[
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "missing"),
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
            ],
        );
        assert!(!fixture.argv().iter().any(|line| line.starts_with("run ")));
    }

    #[test]
    fn run_container_passes_the_catalog_env_through_the_env_file_not_argv() {
        if in_subprocess() {
            let env = BTreeMap::from([("AIM_PROFILE_ID".to_owned(), "vllm-profile".to_owned())]);
            run_container(
                "svc-env",
                IMAGE,
                "127.0.0.1",
                18087,
                &DevicePolicy::CpuOnly,
                &[],
                Some(&env),
            )
            .unwrap();
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::run_container_passes_the_catalog_env_through_the_env_file_not_argv",
            &[
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "missing"),
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
            ],
        );
        assert!(!fixture.argv().join("\n").contains("vllm-profile"));
        assert_eq!(
            fixture.env_copy().as_deref(),
            Some("AIM_PROFILE_ID=vllm-profile\n")
        );
    }

    #[test]
    fn run_container_forwards_the_endpoint_api_key_as_vllm_api_key() {
        if in_subprocess() {
            run_container(
                "svc-key",
                IMAGE,
                "127.0.0.1",
                18088,
                &DevicePolicy::CpuOnly,
                &[],
                None,
            )
            .unwrap();
            return;
        }
        let key_file = std::env::temp_dir().join(format!("aim-key-{}", std::process::id()));
        std::fs::write(&key_file, "s3cret\n").unwrap();
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::run_container_forwards_the_endpoint_api_key_as_vllm_api_key",
            &[
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "missing"),
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                (
                    rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV,
                    key_file.to_str().unwrap(),
                ),
            ],
        );
        std::fs::remove_file(&key_file).ok();
        assert!(!fixture.argv().join("\n").contains("s3cret"));
        assert_eq!(fixture.env_copy().as_deref(), Some("VLLM_API_KEY=s3cret\n"));
    }

    #[test]
    fn serve_http_removes_the_container_when_docker_run_fails() {
        if in_subprocess() {
            let state_path = data_dir().join("svc.json");
            serve_http(serve_request("svc-run-leak", 18089, state_path))
                .expect_err("docker run failure should surface");
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::serve_http_removes_the_container_when_docker_run_fails",
            &[
                ("FAKE_DOCKER_INSPECT_SEQUENCE", "missing"),
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                ("FAKE_DOCKER_RUN_EXIT", "1"),
                ("FAKE_DOCKER_RUN_STDERR", "port is already allocated"),
            ],
        );
        assert!(
            fixture
                .argv()
                .contains(&"rm -f rocm-aim-svc-run-leak".to_owned()),
            "{:?}",
            fixture.argv()
        );
    }

    #[test]
    fn wait_for_container_ready_bails_when_container_exits_early() {
        if in_subprocess() {
            let error = wait_for_container_ready(
                "fake-container",
                "127.0.0.1",
                19999,
                Some("model-x"),
                Duration::from_secs(5),
            )
            .expect_err("early container exit should bail");
            let message = error.to_string();
            assert!(message.contains("exited early"), "{message}");
            assert!(message.contains('7'), "{message}");
            assert!(message.contains("widget crashed"), "{message}");
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::wait_for_container_ready_bails_when_container_exits_early",
            &[
                ("FAKE_DOCKER_RUNNING", "false"),
                ("FAKE_DOCKER_EXIT_CODE", "7"),
                ("FAKE_DOCKER_LOGS", "widget crashed"),
            ],
        );
    }

    #[test]
    fn wait_for_container_ready_bails_when_the_container_is_removed() {
        if in_subprocess() {
            let error = wait_for_container_ready(
                "fake-container",
                "127.0.0.1",
                19998,
                Some("model-x"),
                Duration::from_secs(5),
            )
            .expect_err("a missing container should bail");
            assert!(error.to_string().contains("no longer exists"), "{error}");
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::wait_for_container_ready_bails_when_the_container_is_removed",
            &[("FAKE_DOCKER_INSPECT_MISSING", "1")],
        );
    }

    #[test]
    fn wait_for_container_ready_bails_on_timeout() {
        if in_subprocess() {
            // Unbound port: the readiness HTTP probe fails fast (connection refused).
            let error = wait_for_container_ready(
                "fake-container",
                "127.0.0.1",
                59123,
                Some("model-x"),
                Duration::from_millis(10),
            )
            .expect_err("timeout should bail");
            assert!(
                error.to_string().contains("did not become ready"),
                "{error}"
            );
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::wait_for_container_ready_bails_on_timeout",
            &[("FAKE_DOCKER_RUNNING", "true")],
        );
    }

    #[test]
    fn stop_service_records_intent_kills_removes_and_writes_stopped_state() {
        if in_subprocess() {
            let response = stop_service(StopRequest {
                service_id: "svc-stop-succeeds".to_owned(),
                force: true,
            })
            .unwrap();
            assert!(response.stopped);
            assert!(!response.graceful);
            let state_path = crate::state::state_path("svc-stop-succeeds");
            assert_eq!(status_of(&state_path).as_deref(), Some("stopped"));
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::stop_service_records_intent_kills_removes_and_writes_stopped_state",
            &[],
        );
        assert_eq!(
            fixture.argv(),
            vec![
                "kill rocm-aim-svc-stop-succeeds",
                "rm -f rocm-aim-svc-stop-succeeds"
            ]
        );
        // The deliberate-stop intent must be on disk before the kill, or the
        // supervisor reads the container's exit 137 as a crash.
        let at_stop = fixture.state_at_stop().unwrap_or_default();
        assert!(at_stop.contains("stopping"), "{at_stop}");
    }

    #[test]
    fn stop_service_of_a_missing_container_reports_not_stopped_but_still_cleans_up() {
        if in_subprocess() {
            let response = stop_service(StopRequest {
                service_id: "svc-stop-missing".to_owned(),
                force: true,
            })
            .unwrap();
            assert!(!response.stopped);
            assert!(!response.graceful);
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::stop_service_of_a_missing_container_reports_not_stopped_but_still_cleans_up",
            &[
                ("FAKE_DOCKER_STOP_NO_SUCH_CONTAINER", "1"),
                ("FAKE_DOCKER_RM_NO_SUCH_CONTAINER", "1"),
            ],
        );
    }

    #[test]
    fn stop_service_failure_restores_the_prior_status_and_keeps_the_container() {
        if in_subprocess() {
            let state_path = crate::state::state_path("svc-stop-fails");
            crate::state::write_status(&state_path, "running").unwrap();
            stop_service(StopRequest {
                service_id: "svc-stop-fails".to_owned(),
                force: true,
            })
            .expect_err("a failed kill must be reported, not swallowed");
            assert_eq!(status_of(&state_path).as_deref(), Some("running"));
            return;
        }
        let fixture = run_with_fake_docker(
            "process::subprocess_tests::stop_service_failure_restores_the_prior_status_and_keeps_the_container",
            &[
                ("FAKE_DOCKER_STOP_EXIT", "1"),
                ("FAKE_DOCKER_STOP_STDERR", "permission denied"),
            ],
        );
        assert!(!fixture.argv().iter().any(|line| line.starts_with("rm ")));
    }
}
