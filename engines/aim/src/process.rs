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
/// supervise. Unlike vLLM/Lemonade, this engine never signals a client
/// process at all: the container is stopped by name (`docker stop`/`kill`),
/// which sidesteps the SIGKILL-orphan hazard those engines have to guard
/// against (a killed `docker run` client would otherwise leave the container
/// running, with nothing left to track it by).
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
        &device_policy,
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
    device_policy: &DevicePolicy,
    gpu_indices: &[u32],
    extra_env: Option<&std::collections::BTreeMap<String, String>>,
) -> Result<Option<String>> {
    let name = container_name(service_id);
    // Normalize once and reuse everywhere below: metadata and the actual
    // `docker pull`/`run` must resolve the exact same image, or the fetched
    // accelerator/GPU-arch metadata silently describes a different image
    // than the one that runs (ROCMAI-554 review finding).
    let image_ref = crate::registry::normalize_image_ref(model_ref);
    let metadata = crate::registry::fetch_image_metadata(&image_ref).unwrap_or_else(|error| {
        eprintln!(
            "Warning: could not fetch AIM image metadata for `{image_ref}`: {error:#}; \
             assuming GPU-preferred with no accelerator-arch check."
        );
        crate::registry::AimImageMetadata::default()
    });
    if crate::docker::inspect_running(&name)?.unwrap_or(false) {
        return Ok(metadata.canonical_name);
    }

    if !crate::docker::image_present(&image_ref)? {
        crate::docker::pull(&image_ref)?;
    }

    // `crate::docker::gpu_device_args` has no native-Windows device shape, and
    // an image reporting AIM_ACCELERATOR_TYPE=cpu has nothing to pass GPU args
    // to, so neither counts as "a GPU is usable" regardless of what the caller
    // selected.
    let gpu_usable = !cfg!(windows) && !gpu_indices.is_empty();
    let image_wants_cpu = metadata.accelerator_type.as_deref() == Some("cpu");
    let use_gpu = resolve_use_gpu(
        service_id,
        &image_ref,
        device_policy,
        gpu_usable,
        image_wants_cpu,
    )?;
    let render_gid = if use_gpu {
        crate::docker::render_group_gid()
    } else {
        None
    };

    let mut env = extra_env.cloned().unwrap_or_default();
    // AIM images bundle vLLM's OpenAI server, so it is `VLLM_API_KEY` — the
    // same variable `engines/vllm` sets on its own child process — that this
    // engine's actual server enforces. The container is the only process this
    // engine ever runs, so without this the require-api-key setting a public
    // `rocm serve --engine aim` was given is silently never enforced.
    if let Some(api_key) = rocm_engine_protocol::resolve_endpoint_api_key() {
        env.insert("VLLM_API_KEY".to_owned(), api_key);
    }
    crate::docker::run(
        &name,
        &image_ref,
        host,
        port,
        crate::DEFAULT_AIM_PORT,
        if use_gpu { gpu_indices } else { &[] },
        render_gid,
        &env,
    )?;
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
        DevicePolicy::GpuRequired => {
            if !gpu_usable || image_wants_cpu {
                bail!(
                    "AIM service `{service_id}` has device_policy=gpu_required but no GPU is \
                     usable for `{model_ref}` (native Windows has no GPU passthrough, no GPU \
                     indices were selected, or the image reports AIM_ACCELERATOR_TYPE=cpu)"
                );
            }
            true
        }
        DevicePolicy::GpuPreferred => gpu_usable && !image_wants_cpu,
    })
}

pub(crate) fn serve_http(request: ServeHttpRequest) -> Result<()> {
    let canonical_name = run_container(
        &request.service_id,
        &request.model_ref,
        &request.host,
        request.port,
        &request.device_policy,
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
        Ok(()) => block_until_container_stops(&name, &request.state_path),
        Err(error) => {
            // A readiness timeout (unlike an early container exit) leaves the
            // container running; stop it so a failed launch doesn't leak an
            // orphaned container nothing will ever signal again (the service
            // record is about to be marked `failed`, not `running`, so
            // `apps/rocmd`'s own stop path will never find it).
            let _ = crate::docker::stop(&name, true, STOP_GRACE);
            crate::state::write_terminal_state(&request.state_path, "failed")?;
            Err(error)
        }
    }
}

/// Blocks for the container's remaining lifetime, mirroring `engines/vllm`
/// and `engines/lemonade`'s blocking `child.wait()`: `apps/rocmd`'s generic
/// supervisor (`supervise_service`) treats this process's exit as the
/// service's exit, so returning right after readiness (as this used to)
/// made a still-running, actively-serving container read `stopped` the
/// instant it became ready. A transient `docker inspect` failure is treated
/// as "still running" rather than ending the loop, matching this file's
/// other `unwrap_or`-style tolerance of CLI hiccups elsewhere.
fn block_until_container_stops(name: &str, state_path: &std::path::Path) -> Result<()> {
    while crate::docker::inspect_running(name).unwrap_or(Some(true)) != Some(false) {
        std::thread::sleep(READY_POLL_INTERVAL);
    }
    crate::state::write_terminal_state(state_path, "stopped")
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
    fn resolve_use_gpu_gpu_preferred_falls_back_to_cpu_without_failing() {
        assert!(
            !resolve_use_gpu("svc", "model", &DevicePolicy::GpuPreferred, false, false).unwrap()
        );
        assert!(!resolve_use_gpu("svc", "model", &DevicePolicy::GpuPreferred, true, true).unwrap());
        assert!(resolve_use_gpu("svc", "model", &DevicePolicy::GpuPreferred, true, false).unwrap());
    }
}

/// `run_container`/`wait_for_container_ready`/`serve_http`/`stop_service` all
/// shell out to `docker`, so (same reasoning as `docker.rs`'s
/// `subprocess_tests`) each scenario below re-spawns this test binary with a
/// fake `docker` on `PATH`. `fetch_image_metadata` (called unconditionally by
/// `run_container`) hits the real Docker Hub registry for a deliberately
/// bogus image and is expected to 404; that failure is already handled by a
/// production fallback (`unwrap_or_else` to `AimImageMetadata::default()`),
/// so no mock/seam is needed, just the real ~1s network round trip.
#[cfg(all(test, unix))]
mod subprocess_tests {
    use super::*;
    use rocm_core::fake_executable::write_fake_executable;
    use std::process::{Command, Stdio};

    const SENTINEL: &str = "ROCM_AIM_PROCESS_SUBPROCESS_TEST";

    const FAKE_DOCKER_SCRIPT: &str = r#"#!/bin/sh
case "$1" in
  run)
    [ -n "$FAKE_DOCKER_RUN_STDERR" ] && echo "$FAKE_DOCKER_RUN_STDERR" >&2
    exit "${FAKE_DOCKER_RUN_EXIT:-0}"
    ;;
  kill|stop)
    if [ -n "$FAKE_DOCKER_STOP_NO_SUCH_CONTAINER" ]; then
      echo "Error response from daemon: No such container: fake" >&2
      exit 1
    fi
    exit "${FAKE_DOCKER_STOP_EXIT:-0}"
    ;;
  inspect)
    case "$3" in
      *Running*) echo "${FAKE_DOCKER_RUNNING:-false}" ;;
      *ExitCode*) echo "${FAKE_DOCKER_EXIT_CODE:-0}" ;;
    esac
    exit 0
    ;;
  image)
    exit "${FAKE_DOCKER_IMAGE_PRESENT_EXIT:-1}"
    ;;
  pull)
    exit "${FAKE_DOCKER_PULL_EXIT:-0}"
    ;;
  logs)
    [ -n "$FAKE_DOCKER_LOGS" ] && echo "$FAKE_DOCKER_LOGS"
    exit 0
    ;;
esac
exit 0
"#;

    fn in_subprocess() -> bool {
        std::env::var_os(SENTINEL).is_some()
    }

    fn run_with_fake_docker(test_path: &str, extra_env: &[(&str, &str)]) {
        let dir = std::env::temp_dir().join(format!(
            "aim-process-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fake_bin = dir.join("bin");
        std::fs::create_dir_all(&fake_bin).unwrap();
        write_fake_executable(&fake_bin.join("docker"), FAKE_DOCKER_SCRIPT).unwrap();

        let mut entries = vec![fake_bin];
        entries.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));

        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", test_path, "--nocapture"])
            .env(SENTINEL, "1")
            .env("PATH", std::env::join_paths(entries).unwrap())
            .stdin(Stdio::null());
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let status = command.status().unwrap();
        assert!(status.success(), "helper subprocess failed for {test_path}");

        std::fs::remove_dir_all(&dir).ok();
    }

    const BOGUS_IMAGE: &str = "rocm-cli-test/does-not-exist-on-docker-hub-aim-fixture";

    #[test]
    fn run_container_skips_docker_run_when_already_running() {
        if in_subprocess() {
            let result = run_container(
                "svc-already-running",
                BOGUS_IMAGE,
                "127.0.0.1",
                18080,
                &DevicePolicy::GpuPreferred,
                &[],
                None,
            );
            assert!(result.is_ok(), "{result:?}");
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::run_container_skips_docker_run_when_already_running",
            &[
                ("FAKE_DOCKER_RUNNING", "true"),
                ("FAKE_DOCKER_PULL_EXIT", "1"),
                ("FAKE_DOCKER_RUN_EXIT", "1"),
            ],
        );
    }

    #[test]
    fn run_container_pulls_when_image_absent_then_runs() {
        if in_subprocess() {
            let result = run_container(
                "svc-pull-then-run",
                BOGUS_IMAGE,
                "127.0.0.1",
                18081,
                &DevicePolicy::GpuPreferred,
                &[],
                None,
            );
            assert!(result.is_ok(), "{result:?}");
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::run_container_pulls_when_image_absent_then_runs",
            &[
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "1"),
                ("FAKE_DOCKER_PULL_EXIT", "0"),
                ("FAKE_DOCKER_RUN_EXIT", "0"),
            ],
        );
    }

    #[test]
    fn run_container_bails_when_pull_fails() {
        if in_subprocess() {
            let error = run_container(
                "svc-pull-fails",
                BOGUS_IMAGE,
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
                BOGUS_IMAGE,
                "127.0.0.1",
                18083,
                &DevicePolicy::GpuPreferred,
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
                ("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0"),
                ("FAKE_DOCKER_RUN_EXIT", "1"),
                ("FAKE_DOCKER_RUN_STDERR", "boom"),
            ],
        );
    }

    #[test]
    fn wait_for_container_ready_bails_when_container_exits_early() {
        if in_subprocess() {
            let error = wait_for_container_ready(
                "fake-container",
                "127.0.0.1",
                19999,
                "model-x",
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
    fn wait_for_container_ready_bails_on_timeout() {
        if in_subprocess() {
            // Unbound port: the readiness HTTP probe fails fast (connection
            // refused), so this only exercises the timeout bookkeeping, not a
            // real OpenAI-compatible server (left as a judgment call — see the
            // PR summary).
            let error = wait_for_container_ready(
                "fake-container",
                "127.0.0.1",
                59123,
                "model-x",
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
    fn serve_http_stops_container_and_marks_state_failed_on_readiness_failure() {
        if in_subprocess() {
            let dir = std::env::temp_dir().join(format!(
                "aim-serve-http-test-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let state_path = dir.join("svc.json");

            let request = ServeHttpRequest {
                service_id: "svc-serve-http-fails".to_owned(),
                model_ref: BOGUS_IMAGE.to_owned(),
                host: "127.0.0.1".to_owned(),
                port: 18084,
                device_policy: DevicePolicy::GpuPreferred,
                gpu_indices: vec![],
                runtime_id: None,
                env_id: None,
                state_path: state_path.clone(),
                log_path: None,
                engine_recipe: None,
            };
            serve_http(request).expect_err("readiness failure should surface as an error");

            let state = crate::state::read_service_state(&state_path).unwrap();
            assert_eq!(state.get("status").and_then(|v| v.as_str()), Some("failed"));

            std::fs::remove_dir_all(&dir).ok();
            return;
        }
        run_with_fake_docker(
            "process::subprocess_tests::serve_http_stops_container_and_marks_state_failed_on_readiness_failure",
            &[
                ("FAKE_DOCKER_RUNNING", "false"),
                ("FAKE_DOCKER_EXIT_CODE", "1"),
            ],
        );
    }

    #[test]
    fn stop_service_force_stop_succeeds_and_writes_stopped_state() {
        if in_subprocess() {
            let response = stop_service(StopRequest {
                service_id: "svc-stop-succeeds".to_owned(),
                force: true,
            })
            .unwrap();
            assert!(response.stopped);
            assert!(!response.graceful);

            let state_path = crate::state::state_path("svc-stop-succeeds");
            let state = crate::state::read_service_state(&state_path).unwrap();
            assert_eq!(
                state.get("status").and_then(|v| v.as_str()),
                Some("stopped")
            );
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "aim-stop-service-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        run_with_fake_docker(
            "process::subprocess_tests::stop_service_force_stop_succeeds_and_writes_stopped_state",
            &[("ROCM_CLI_DATA_DIR", dir.to_str().unwrap())],
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stop_service_missing_container_reports_not_stopped_without_writing_state() {
        if in_subprocess() {
            let response = stop_service(StopRequest {
                service_id: "svc-stop-missing".to_owned(),
                force: true,
            })
            .unwrap();
            assert!(!response.stopped);
            assert!(!response.graceful);

            let state_path = crate::state::state_path("svc-stop-missing");
            assert!(!state_path.exists());
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "aim-stop-service-missing-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        run_with_fake_docker(
            "process::subprocess_tests::stop_service_missing_container_reports_not_stopped_without_writing_state",
            &[
                ("ROCM_CLI_DATA_DIR", dir.to_str().unwrap()),
                ("FAKE_DOCKER_STOP_NO_SUCH_CONTAINER", "1"),
            ],
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
