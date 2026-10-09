// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use rocm_core::{AppPaths, DEFAULT_LOCAL_PORT, normalize_runtime_path_for_host, runtime_is_linux};
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, DevicePolicy, EndpointRequest, EndpointResponse,
    EngineCapabilities, EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope,
    HealthcheckRequest, HealthcheckResponse, InstallRequest, InstallResponse, LaunchRequest,
    LaunchResponse, LogsRequest, LogsResponse, ResolveModelRequest, ResolveModelResponse,
    StopRequest, StopResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::Read;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

mod backend_alignment;
mod direct_llama;
mod install;
mod process;
mod runtime_dir;
mod state;
#[cfg(test)]
mod test_support;

// Re-exported so this stays reachable at its pre-split crate-root path —
// `backend_alignment` is a private module, but this constant was `pub` at
// the crate root before the split, and `apps/rocm` depends on this crate's
// lib target.
pub use backend_alignment::LEMONADE_BACKEND_ALIGNMENT_DISABLED_ENV;

pub(crate) const ENGINE_NAME: &str = "lemonade";
pub(crate) const DEFAULT_HOST: &str = "127.0.0.1";
pub(crate) const DEFAULT_MODEL: &str = "Qwen3-4B-Instruct-2507-GGUF";
pub(crate) const DEFAULT_MODEL_REPO_DIR: &str = "models--unsloth--Qwen3-4B-Instruct-2507-GGUF";
pub(crate) const DEFAULT_MODEL_GGUF: &str = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
pub(crate) const LLAMACPP_RECIPE: &str = "llamacpp";
pub(crate) const ROCM_BACKEND_NAME: &str = "rocm";
#[cfg(feature = "e2e-test-hooks")]
const BACKEND_INSTALL_FAILURE_TEST_ENV: &str = "ROCM_E2E_LEMONADE_BACKEND_INSTALL_FAILURE";
const DEFAULT_LOG_TAIL_LINES: usize = 200;
/// How long a stop waits for the server to actually exit after each signal
/// before reporting a timeout (or, under `force`, escalating to `SIGKILL`).
const STOP_GRACE: Duration = Duration::from_secs(10);
/// Lemonade state identifies the actual llama-server process directly.
const STOP_SCOPE: rocm_core::KillScope = rocm_core::KillScope::Single;

#[derive(Parser)]
#[command(name = "rocm-engine-lemonade")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand)]
enum CommandKind {
    Detect,
    Capabilities,
    Install {
        #[arg(long)]
        runtime_id: String,
        #[arg(long)]
        reinstall: bool,
    },
    ResolveModel {
        model_ref: String,
    },
    Launch {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long)]
        device_policy: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
        #[arg(long)]
        gpu: Option<String>,
    },
    Stdio,
    ServeHttp {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long)]
        device_policy: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
        #[arg(long)]
        state_path: PathBuf,
        #[arg(long)]
        log_path: Option<PathBuf>,
        #[arg(long)]
        engine_recipe_json: Option<String>,
        #[arg(long)]
        gpu: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct ServeHttpRequest {
    pub(crate) service_id: String,
    pub(crate) model_ref: String,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) device_policy: DevicePolicy,
    pub(crate) gpu_indices: Vec<u32>,
    pub(crate) runtime_id: Option<String>,
    pub(crate) env_id: Option<String>,
    pub(crate) state_path: PathBuf,
    pub(crate) log_path: Option<PathBuf>,
    pub(crate) engine_recipe: Option<rocm_engine_protocol::EngineRecipeHint>,
}

pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Detect => print_json(&detect_response())?,
        CommandKind::Capabilities => print_json(&capabilities())?,
        CommandKind::Install {
            runtime_id,
            reinstall,
        } => print_json(&install_response(InstallRequest {
            runtime_id,
            python_version: None,
            env_root: None,
            reinstall,
        })?)?,
        CommandKind::ResolveModel { model_ref } => {
            print_json(&resolve_model_response(ResolveModelRequest {
                model_ref,
                runtime_id: None,
                device_policy: None,
                recipe_override: None,
                engine_recipe: None,
            })?)?;
        }
        CommandKind::Launch {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            runtime_id,
            env_id,
            gpu,
        } => print_json(&launch_service(LaunchRequest {
            service_id,
            env_id,
            runtime_id,
            model_ref,
            host,
            port,
            device_policy: Some(crate::state::parse_device_policy_arg(
                device_policy.as_deref(),
            )?),
            endpoint_mode: Some("openai".to_owned()),
            engine_recipe: None,
            gpu_selection: crate::state::parse_gpu_selection_arg(gpu.as_deref())?,
        })?)?,
        CommandKind::Stdio => {
            let envelope = read_request()?;
            print_json(&handle_envelope(envelope))?;
        }
        CommandKind::ServeHttp {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe_json,
            gpu,
        } => serve_http(ServeHttpRequest {
            service_id,
            model_ref,
            host,
            port,
            device_policy: crate::state::parse_device_policy_arg(device_policy.as_deref())?,
            gpu_indices: crate::state::parse_gpu_indices_arg(gpu.as_deref())?,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe: crate::state::parse_engine_recipe_json(engine_recipe_json)?,
        })?,
    }
    Ok(())
}

pub fn builtin_handle_envelope(envelope: EngineRequestEnvelope) -> EngineResponseEnvelope {
    handle_envelope(envelope)
}

#[allow(clippy::too_many_arguments)]
pub fn builtin_serve_http(
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: DevicePolicy,
    gpu_indices: Vec<u32>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<rocm_engine_protocol::EngineRecipeHint>,
) -> Result<()> {
    serve_http(ServeHttpRequest {
        service_id,
        model_ref,
        host,
        port,
        device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        state_path,
        log_path,
        engine_recipe,
    })
}

fn handle_envelope(envelope: EngineRequestEnvelope) -> EngineResponseEnvelope {
    match envelope.method {
        EngineMethod::Detect => {
            deserialize_and_respond::<DetectRequest, _, _>(envelope.payload, |_| {
                Ok(detect_response())
            })
        }
        EngineMethod::Capabilities => EngineResponseEnvelope::success(capabilities()),
        EngineMethod::Install => {
            deserialize_and_respond::<InstallRequest, _, _>(envelope.payload, install_response)
        }
        EngineMethod::ResolveModel => deserialize_and_respond::<ResolveModelRequest, _, _>(
            envelope.payload,
            resolve_model_response,
        ),
        EngineMethod::Launch => {
            deserialize_and_respond::<LaunchRequest, _, _>(envelope.payload, launch_service)
        }
        EngineMethod::Healthcheck => deserialize_and_respond::<HealthcheckRequest, _, _>(
            envelope.payload,
            healthcheck_service,
        ),
        EngineMethod::Endpoint => {
            deserialize_and_respond::<EndpointRequest, _, _>(envelope.payload, endpoint_response)
        }
        EngineMethod::Stop => {
            deserialize_and_respond::<StopRequest, _, _>(envelope.payload, stop_service)
        }
        EngineMethod::Logs => {
            deserialize_and_respond::<LogsRequest, _, _>(envelope.payload, logs_response)
        }
    }
}

fn deserialize_and_respond<T, F, U>(payload: Value, handler: F) -> EngineResponseEnvelope
where
    T: for<'de> Deserialize<'de>,
    F: FnOnce(T) -> Result<U>,
    U: Serialize,
{
    match serde_json::from_value::<T>(payload) {
        Ok(request) => match handler(request) {
            Ok(response) => EngineResponseEnvelope::success(response),
            Err(error) => EngineResponseEnvelope::failure("request_failed", format_error(&error)),
        },
        Err(error) => EngineResponseEnvelope::failure("invalid_payload", error.to_string()),
    }
}

fn format_error(error: &anyhow::Error) -> String {
    let mut lines = Vec::new();
    for cause in error.chain() {
        let text = cause.to_string();
        if !lines.iter().any(|line| line == &text) {
            lines.push(text);
        }
    }
    lines.join(": ")
}

fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        cpu: false,
        rocm_gpu: true,
        openai_compatible: true,
        tool_calling: true,
        quantized_models:
            "GGUF through Lemonade llama.cpp (ROCm or Vulkan GPU backend auto-selected)".to_owned(),
        reasoning_parser: false,
    }
}

fn detect_response() -> DetectResponse {
    let runtime = crate::backend_alignment::resolve_runtime().ok();
    let mut notes = Vec::new();
    if let Some(runtime) = runtime.as_ref() {
        notes.push(format!(
            "Lemonade embeddable {} is installed at {}",
            runtime.manifest.version,
            runtime.manifest.runtime_dir.display()
        ));
        notes.push(format!(
            "Lemonade llama.cpp backend selected for this GPU: {}:{}",
            runtime.manifest.backend_recipe, runtime.manifest.backend_name
        ));
    } else {
        notes.push(
            "Lemonade embeddable is not installed yet; run `rocm engines install lemonade`"
                .to_owned(),
        );
    }
    DetectResponse {
        installed: runtime.is_some(),
        env_id: runtime
            .as_ref()
            .map(|runtime| runtime.manifest.env_id.clone()),
        runtime_kind: Some("lemonade_embeddable".to_owned()),
        runtime_executable: runtime
            .as_ref()
            .map(|runtime| runtime.manifest.lemond.display().to_string()),
        managed_env: Some(true),
        python_version: None,
        torch_version: None,
        transformers_version: None,
        available_devices: vec![crate::state::gpu_availability_device(runtime.is_some())],
        capabilities: capabilities(),
        notes,
    }
}

fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    // Debug builds expose a deterministic failure seam for the black-box CLI
    // scenario that pins retry count and terminal recovery guidance. Keep it at
    // the backend phase: the real defect happens after the embeddable is ready,
    // and exercising it must not download or alter a runtime on the test host.
    #[cfg(feature = "e2e-test-hooks")]
    if std::env::var_os(BACKEND_INSTALL_FAILURE_TEST_ENV).is_some() {
        crate::backend_alignment::install_llamacpp_backend_with_retry(|| {
            bail!("scripted Lemonade backend install failure")
        })?;
    }
    eprintln!(
        "Preparing Lemonade embeddable {}...",
        rocm_deps::LEMONADE_VERSION
    );
    let env_root = request
        .env_root
        .as_deref()
        .map(normalize_runtime_path_for_host);
    let mut manifest =
        crate::install::prepare_embeddable(&paths, env_root.as_deref(), request.reinstall)?;
    eprintln!("Detecting best supported Lemonade llama.cpp backend...");
    let aligned_version =
        crate::backend_alignment::prepare_llamacpp_backend_for_active_rocm(&paths, &mut manifest)?;
    crate::install::write_manifest(&paths, &manifest)?;
    let mut warnings = vec![
        "Lemonade is installed as a rocm-cli managed embeddable runtime".to_owned(),
        format!(
            "Selected the best supported llama.cpp backend for this GPU: {}:{}",
            manifest.backend_recipe, manifest.backend_name
        ),
    ];
    if let Some(version) = aligned_version {
        warnings.push(format!(
            "Aligned Lemonade's ROCm llama.cpp backend to match the installed ROCm SDK ({version})"
        ));
    }
    Ok(InstallResponse {
        env_id: manifest.env_id.clone(),
        env_path: manifest.runtime_dir.display().to_string(),
        python_executable: manifest.lemonade.display().to_string(),
        runtime_kind: Some("lemonade_embeddable".to_owned()),
        runtime_executable: Some(manifest.lemond.display().to_string()),
        managed_env: Some(true),
        installed_packages: vec![
            format!("lemonade-embeddable=={}", manifest.version),
            format!(
                "lemonade-backend={}:{}",
                manifest.backend_recipe, manifest.backend_name
            ),
        ],
        capabilities: capabilities(),
        lock_hash: crate::state::manifest_lock_hash(&manifest),
        warnings,
    })
}

fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    let device_policy = crate::state::normalize_device_policy(request.device_policy)?;
    let engine_recipe = crate::state::accepted_engine_recipe(request.engine_recipe)?;
    let canonical_model_id = crate::state::resolve_lemonade_model_ref(&request.model_ref);
    Ok(ResolveModelResponse {
        canonical_model_id,
        task: "chat-completions".to_owned(),
        source: "lemonade".to_owned(),
        revision: "main".to_owned(),
        loader: "llamacpp".to_owned(),
        trust_remote_code: false,
        chat_template_mode: "lemonade".to_owned(),
        dtype: "gguf".to_owned(),
        device_policy,
        estimated_memory: "about 4 GiB plus context for Qwen3-4B-Instruct-2507-GGUF".to_owned(),
        launch_defaults: json!({
            "host": DEFAULT_HOST,
            "port": DEFAULT_LOCAL_PORT,
            "endpoint_mode": "openai"
        }),
        engine_recipe,
        warnings: vec![
            "Lemonade auto-selects the best supported GPU llama.cpp backend for this host (ROCm, then Vulkan); CPU is never used under the GPU-required policy".to_owned(),
        ],
    })
}

fn launch_service(mut request: LaunchRequest) -> Result<LaunchResponse> {
    rocm_core::require_nonempty(&request.service_id, "service_id")?;
    rocm_core::require_nonempty(&request.model_ref, "model_ref")?;
    request.device_policy = Some(crate::state::normalize_device_policy(
        request.device_policy.clone(),
    )?);
    request.engine_recipe = crate::state::accepted_engine_recipe(request.engine_recipe)?;
    let runtime = crate::backend_alignment::resolve_runtime()?;
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    std::fs::create_dir_all(paths.engine_logs_dir(ENGINE_NAME))?;
    std::fs::create_dir_all(paths.engine_state_dir(ENGINE_NAME))?;
    let log_path = paths
        .engine_logs_dir(ENGINE_NAME)
        .join(format!("{}.log", request.service_id));
    let state_path = paths
        .engine_state_dir(ENGINE_NAME)
        .join(format!("{}.json", request.service_id));
    let endpoint_url = crate::state::endpoint_url(&request.host, request.port);
    let serve_request = ServeHttpRequest {
        service_id: request.service_id.clone(),
        model_ref: crate::state::resolve_lemonade_model_ref(&request.model_ref),
        host: request.host.clone(),
        port: request.port,
        device_policy: request
            .device_policy
            .clone()
            .unwrap_or(DevicePolicy::GpuRequired),
        gpu_indices: rocm_engine_protocol::launch_gpu_indices(request.gpu_selection.as_ref()),
        runtime_id: request.runtime_id.clone(),
        env_id: request.env_id.clone(),
        state_path: state_path.clone(),
        log_path: Some(log_path.clone()),
        engine_recipe: request.engine_recipe.clone(),
    };
    let current_exe =
        std::env::current_exe().context("failed to discover current Lemonade engine binary")?;
    let args = crate::state::serve_http_command_args(&serve_request);
    crate::state::write_running_state(
        &serve_request,
        &runtime,
        std::process::id(),
        None,
        "starting",
    )?;
    let wrapper_pid = crate::state::spawn_serve_http_background(&current_exe, &args)?;
    crate::state::merge_json_state(
        &state_path,
        &json!({
            "pid": wrapper_pid,
            "wrapper_pid": wrapper_pid,
            // Refresh the identity token in lockstep with the PID so a stop that
            // lands before the wrapper rewrites its own state still verifies it.
            "start_ticks": rocm_core::process_start_ticks(wrapper_pid),
        }),
    )?;
    Ok(LaunchResponse {
        service_id: request.service_id,
        pid: wrapper_pid,
        endpoint_url,
        log_path: log_path.display().to_string(),
        state_path: state_path.display().to_string(),
    })
}

fn serve_http(mut request: ServeHttpRequest) -> Result<()> {
    crate::state::require_gpu_required(&request.device_policy)?;
    // Verify a real GPU is available before launching anything: zero usable
    // devices fails fast, `auto` resolves to the first present device (never an
    // assumed device 0), and an explicit `--gpu` index is rejected when it is not
    // actually present. This runs before any server process is spawned.
    request.gpu_indices = crate::state::resolve_serve_gpu_indices(&request.gpu_indices)?;
    let mut runtime = crate::backend_alignment::resolve_runtime()?;
    let mut process_env = crate::process::lemonade_process_environment()?;
    process_env.gpu_indices = request.gpu_indices.clone();
    let log_path = request.log_path.as_deref();
    crate::state::write_running_state(&request, &runtime, std::process::id(), None, "starting")?;
    // A canonical Hugging Face checkpoint (owner/repo:variant) cannot be served under its
    // exact name through Lemonade's model router (Lemonade renames registered models and
    // its registry has several naming quirks). Download the GGUF and run a packaged
    // llama-server on it directly with `--alias`, which serves it under exactly that name.
    if let Some(checkpoint) = crate::direct_llama::parse_hf_checkpoint(&request.model_ref) {
        return crate::direct_llama::serve_hf_checkpoint(
            &request,
            &runtime,
            &process_env,
            log_path,
            &checkpoint,
        );
    }
    if runtime_is_linux()
        && let Some(server) = crate::direct_llama::find_llama_server_binary(&runtime.manifest)
    {
        crate::direct_llama::ensure_direct_llama_model_available(
            &request,
            &runtime,
            &process_env,
            log_path,
        )?;
        let backend = crate::direct_llama::llama_server_backend_label(&server);
        return crate::process::serve_direct_llama_server(
            &request,
            &runtime,
            &process_env,
            &server,
            log_path,
            &anyhow!("using Lemonade packaged {backend} llama-server directly on Linux"),
        );
    }
    let mut child = crate::process::spawn_lemond(
        &runtime.manifest,
        &request.host,
        request.port,
        log_path,
        &process_env,
    )?;
    crate::state::write_running_state(
        &request,
        &runtime,
        std::process::id(),
        Some(child.id()),
        "running",
    )?;
    crate::process::wait_for_lemonade_cli_status(
        &runtime.manifest,
        &request.host,
        request.port,
        Duration::from_secs(30),
        log_path,
        &process_env,
    )
    .context("Lemonade server did not become ready")?;
    let backend = crate::backend_alignment::ensure_best_llamacpp_backend(
        &mut runtime.manifest,
        &request.host,
        request.port,
        &process_env,
        false,
    )
    .context("failed to select a supported Lemonade llama.cpp backend")?;
    let load_result = crate::process::run_lemonade_model_load(
        &runtime.manifest,
        &request.host,
        request.port,
        &request.model_ref,
        &backend,
        request.engine_recipe.as_ref(),
        log_path,
        &process_env,
    );
    let router_ready = load_result.is_ok()
        && crate::process::query_loaded_model_endpoint(
            &crate::state::endpoint_url(&request.host, request.port),
            &request.model_ref,
            &backend,
        )
        .unwrap_or(false)
        && crate::process::query_chat_smoke_endpoint(
            &request.host,
            request.port,
            &request.model_ref,
        )
        .unwrap_or(false);
    if let Err(error) = load_result {
        if runtime_is_linux()
            && let Some(direct_server) =
                crate::direct_llama::find_llama_server_binary(&runtime.manifest)
        {
            let _ = crate::state::terminate_pid(child.id(), true);
            let _ = child.wait();
            return crate::process::serve_direct_llama_server(
                &request,
                &runtime,
                &process_env,
                &direct_server,
                log_path,
                &error,
            );
        }
        return Err(error).with_context(|| {
            format!(
                "failed to load {} with Lemonade {LLAMACPP_RECIPE}:{backend}",
                request.model_ref
            )
        });
    }
    if !router_ready {
        let error = anyhow!(
            "Lemonade load completed but the endpoint did not report a {LLAMACPP_RECIPE}:{backend}-loaded model"
        );
        if runtime_is_linux()
            && let Some(direct_server) =
                crate::direct_llama::find_llama_server_binary(&runtime.manifest)
        {
            let _ = crate::state::terminate_pid(child.id(), true);
            let _ = child.wait();
            return crate::process::serve_direct_llama_server(
                &request,
                &runtime,
                &process_env,
                &direct_server,
                log_path,
                &error,
            );
        }
        return Err(error).with_context(|| {
            format!(
                "failed to verify {} with Lemonade {LLAMACPP_RECIPE}:{backend}",
                request.model_ref
            )
        });
    }
    crate::state::merge_json_state(
        &request.state_path,
        &json!({
            "status": "ready",
            "server_pid": child.id(),
            // Identity token for the server PID, captured while the child is alive.
            "server_start_ticks": rocm_core::process_start_ticks(child.id()),
            "backend_state": "ready",
            "backend_requested": backend,
            "load_response": {
                "status": "loaded",
                "method": "lemonade-cli",
                "model_name": request.model_ref,
                "llamacpp_backend": backend
            },
        }),
    )?;
    let status = child.wait().context("failed waiting for Lemonade server")?;
    crate::state::mark_json_status(
        &request.state_path,
        if status.success() {
            "stopped"
        } else {
            "failed"
        },
    )?;
    if status.success() {
        Ok(())
    } else {
        bail!("Lemonade server exited with status {status}")
    }
}

fn healthcheck_service(request: HealthcheckRequest) -> Result<HealthcheckResponse> {
    rocm_core::require_nonempty(&request.service_id, "service_id")?;
    let files = crate::state::service_files(&request.service_id)?;
    let state = crate::state::read_service_state(&files.state_path).ok();
    let endpoint_url = state
        .as_ref()
        .and_then(crate::state::endpoint_url_from_state);
    let state_status = state
        .as_ref()
        .and_then(|value| crate::state::value_string(value, "status"))
        .unwrap_or_else(|| "unknown".to_owned());
    let model_ref = state
        .as_ref()
        .and_then(|value| {
            crate::state::value_string(value, "canonical_model_id")
                .or_else(|| crate::state::value_string(value, "model_ref"))
        })
        .unwrap_or_default();
    let backend = state
        .as_ref()
        .and_then(|value| crate::state::value_string(value, "backend_requested"))
        .unwrap_or_else(|| ROCM_BACKEND_NAME.to_owned());
    let listed = state_status == "ready"
        && !model_ref.is_empty()
        && endpoint_url
            .as_deref()
            .map(|endpoint| {
                crate::process::query_loaded_model_endpoint(endpoint, &model_ref, &backend)
            })
            .transpose()
            .unwrap_or(None)
            .unwrap_or(false);
    // Listing a model is not the same as being able to serve it: the endpoint can
    // answer within seconds while the weights load for minutes. Confirm inference
    // once before reporting ready.
    let ready = listed
        && endpoint_url.as_deref().is_some_and(|endpoint| {
            crate::state::inference_verified(
                &files.state_path,
                state.as_ref(),
                endpoint,
                &model_ref,
            )
        });
    let device = if ready {
        crate::state::reported_device(state.as_ref(), &backend)
    } else {
        "unknown".to_owned()
    };
    Ok(HealthcheckResponse::for_readiness(
        listed,
        ready,
        &state_status,
        &device,
    ))
}

fn endpoint_response(request: EndpointRequest) -> Result<EndpointResponse> {
    rocm_core::require_nonempty(&request.service_id, "service_id")?;
    let files = crate::state::service_files(&request.service_id)?;
    let state = crate::state::read_service_state(&files.state_path)
        .with_context(|| format!("service state not found for `{}`", request.service_id))?;
    let endpoint_url = crate::state::endpoint_url_from_state(&state)
        .with_context(|| format!("service `{}` has no endpoint URL", request.service_id))?;
    Ok(EndpointResponse {
        endpoint_url,
        api_style: "openai".to_owned(),
        supported_routes: vec![
            "/v1/health".to_owned(),
            "/v1/models".to_owned(),
            "/v1/chat/completions".to_owned(),
            "/v1/completions".to_owned(),
        ],
    })
}

fn logs_response(request: LogsRequest) -> Result<LogsResponse> {
    rocm_core::require_nonempty(&request.service_id, "service_id")?;
    let files = crate::state::service_files(&request.service_id)?;
    let limit = request.tail_lines.unwrap_or(DEFAULT_LOG_TAIL_LINES);
    Ok(LogsResponse {
        log_path: files.log_path.display().to_string(),
        recent_lines: if files.log_path.is_file() {
            crate::state::tail_lines(&files.log_path, limit)?
        } else {
            Vec::new()
        },
    })
}

fn stop_service(request: StopRequest) -> Result<StopResponse> {
    rocm_core::require_nonempty(&request.service_id, "service_id")?;
    let files = crate::state::service_files(&request.service_id)?;
    let state = crate::state::read_service_state(&files.state_path).ok();
    // Verify the recorded PID still belongs to our server before signalling it,
    // then wait for it to actually exit so the reported result is truthful.
    let outcome = state
        .as_ref()
        .and_then(crate::state::identity_from_state)
        .map(|identity| {
            rocm_core::terminate_verified(&identity, STOP_SCOPE, STOP_GRACE, request.force)
        });
    let (stopped, graceful) = match outcome {
        Some(outcome) => (outcome.stopped(), outcome.graceful()),
        None => (false, false),
    };
    if stopped {
        crate::state::mark_json_status(&files.state_path, "stopped")?;
    }
    Ok(StopResponse { stopped, graceful })
}

pub(crate) fn current_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn read_request() -> Result<EngineRequestEnvelope> {
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .context("failed to read stdin for engine request")?;
    serde_json::from_str(&buffer).context("failed to parse engine request envelope")
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    use std::io::Write;

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer_pretty(&mut handle, value)?;
    writeln!(&mut handle)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_scope_targets_only_the_recorded_lemonade_server() {
        assert_eq!(STOP_SCOPE, rocm_core::KillScope::Single);
    }
}
