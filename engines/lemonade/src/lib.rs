// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
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
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

mod backend_alignment;
mod direct_llama;
mod install;
mod process;
mod router_serve;
mod runtime_dir;
mod state;
#[cfg(test)]
mod test_http;
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
            anyhow::bail!("scripted Lemonade backend install failure")
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
    let serve_request = launch_serve_request(&request, &state_path, &log_path);
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

/// The serve request the wrapper process is started with. The model reference is
/// normalized once here, so a launch serves (and records) the same identity that
/// `resolve` reports for it: `qwen-smoke` launches as `Qwen3-0.6B-GGUF`. A
/// canonical Hugging Face checkpoint passes through unchanged. `serve-http` itself
/// never normalizes, so a direct serve-http run keeps exactly the name it was given.
fn launch_serve_request(
    request: &LaunchRequest,
    state_path: &Path,
    log_path: &Path,
) -> ServeHttpRequest {
    ServeHttpRequest {
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
        state_path: state_path.to_path_buf(),
        log_path: Some(log_path.to_path_buf()),
        engine_recipe: request.engine_recipe.clone(),
    }
}

/// Everything a serve needs from its environment, resolved once up front so tests
/// can supply their own (a confined cache and parent environment, a fake runtime).
struct ServeContext {
    runtime: crate::backend_alignment::LemonadeRuntime,
    process_env: crate::process::LemonadeProcessEnvironment,
    paths: AppPaths,
    parent: crate::runtime_dir::ParentRuntimeEnvironment,
}

fn serve_http(request: ServeHttpRequest) -> Result<()> {
    serve_http_resolving(request, |request| {
        crate::state::require_gpu_required(&request.device_policy)?;
        // Verify a real GPU is available before launching anything: zero usable
        // devices fails fast, `auto` resolves to the first present device (never an
        // assumed device 0), and an explicit `--gpu` index is rejected when it is not
        // actually present. This runs before any server process is spawned.
        request.gpu_indices = crate::state::resolve_serve_gpu_indices(&request.gpu_indices)?;
        let runtime = crate::backend_alignment::resolve_runtime()?;
        let mut process_env = crate::process::lemonade_process_environment()?;
        process_env.gpu_indices = request.gpu_indices.clone();
        Ok(ServeContext {
            runtime,
            process_env,
            paths: AppPaths::discover()?,
            parent: crate::runtime_dir::ParentRuntimeEnvironment::current(),
        })
    })
}

/// Run a serve and make its outcome visible in the persisted state: any error,
/// whichever step raised it, leaves a state file that exists at `failed` rather
/// than at the `starting`/`running` it last recorded, and the error is returned
/// unchanged. Servers spawned inside [`serve_with_context`] are stopped by the
/// code that spawned them, which owns the child.
fn serve_http_resolving(
    mut request: ServeHttpRequest,
    resolve: impl FnOnce(&mut ServeHttpRequest) -> Result<ServeContext>,
) -> Result<()> {
    let state_path = request.state_path.clone();
    let result = resolve(&mut request).and_then(|context| serve_with_context(&request, context));
    if result.is_err() && state_path.is_file() {
        let _ = crate::state::mark_json_status(&state_path, "failed");
    }
    result
}

fn serve_with_context(request: &ServeHttpRequest, context: ServeContext) -> Result<()> {
    let ServeContext {
        runtime,
        process_env,
        paths,
        parent,
    } = context;
    let log_path = request.log_path.as_deref();
    crate::state::write_running_state(request, &runtime, std::process::id(), None, "starting")?;
    let route = crate::direct_llama::serving_route(&runtime.manifest, request, runtime_is_linux())?;
    let router_model_ref = match route {
        crate::direct_llama::ServeRoute::Direct { model, server } => {
            let model_path = crate::direct_llama::ensure_model_in(
                &paths,
                request,
                &runtime,
                &process_env,
                log_path,
                &model,
                &parent,
            )?;
            let backend = crate::direct_llama::llama_server_backend_label(&server);
            return crate::process::serve_direct_llama_server(
                request,
                &runtime,
                &process_env,
                &server,
                &model_path,
                log_path,
                &format!("using Lemonade packaged {backend} llama-server directly"),
                &parent,
            );
        }
        crate::direct_llama::ServeRoute::Router { model_ref } => model_ref,
    };
    let mut steps = crate::router_serve::LemondRouterSteps {
        runtime,
        request,
        process_env: &process_env,
        log_path,
        parent,
        child: None,
    };
    let result = crate::router_serve::serve_router(request, &mut steps, &router_model_ref);
    with_direct_fallback(
        result,
        &paths,
        request,
        &steps.runtime,
        &process_env,
        log_path,
        &steps.parent,
        runtime_is_linux(),
    )
}

/// Route a router serve's outcome. The router already stopped its server and
/// marked the state failed. Only a model load or verification failure may be
/// retried, and only through the GPU llama-server of the backend Lemonade
/// selected; every other outcome is returned as it is.
#[allow(clippy::too_many_arguments)]
fn with_direct_fallback(
    result: Result<()>,
    paths: &AppPaths,
    request: &ServeHttpRequest,
    runtime: &crate::backend_alignment::LemonadeRuntime,
    process_env: &crate::process::LemonadeProcessEnvironment,
    log_path: Option<&Path>,
    parent: &crate::runtime_dir::ParentRuntimeEnvironment,
    is_linux: bool,
) -> Result<()> {
    match result {
        Err(error) if crate::router_serve::is_router_load_failure(&error) => {
            crate::direct_llama::serve_direct_after_router_failure(
                paths,
                request,
                runtime,
                process_env,
                log_path,
                parent,
                is_linux,
                error,
            )
        }
        result => result,
    }
}

fn healthcheck_service(request: HealthcheckRequest) -> Result<HealthcheckResponse> {
    rocm_core::require_nonempty(&request.service_id, "service_id")?;
    let files = crate::state::service_files(&request.service_id)?;
    healthcheck_service_at(&files.state_path)
}

fn healthcheck_service_at(state_path: &Path) -> Result<HealthcheckResponse> {
    let state = crate::state::read_service_state(state_path).ok();
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
            crate::state::inference_verified(state_path, state.as_ref(), endpoint, &model_ref)
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
            rocm_core::tail_lines(&files.log_path, limit, crate::state::MAX_TAIL_READ)?
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
    fn serving_regression_persisted_identity_drives_health_chat_model() {
        use crate::test_http::{Reply, spawn_fixture};
        for (tag, served_name) in [("router", "Qwen3-0.6B-GGUF"), ("direct", "qwen-smoke")] {
            let expected = served_name.to_owned();
            let (port, server) = spawn_fixture(Duration::from_secs(5), move |recorded| {
                let chat = recorded.method == "POST";
                Reply {
                    body: if chat {
                        json!({"choices": [{"message": {"content": "ok"}}]})
                    } else {
                        json!({"all_models_loaded": [{"model_name": expected, "recipe_options": {"llamacpp_backend": "vulkan"}}]})
                    }
                    .to_string(),
                    last: chat,
                }
            });
            let temporary = tempfile::tempdir().unwrap();
            let dir = temporary.path();
            let request = ServeHttpRequest {
                service_id: tag.to_owned(),
                model_ref: "qwen-smoke".to_owned(),
                host: "127.0.0.1".to_owned(),
                port,
                device_policy: DevicePolicy::GpuRequired,
                gpu_indices: vec![1],
                runtime_id: None,
                env_id: None,
                state_path: dir.join("state.json"),
                log_path: None,
                engine_recipe: None,
            };
            let runtime = crate::backend_alignment::LemonadeRuntime {
                manifest: crate::install::LemonadeInstallManifest {
                    env_id: "test".to_owned(),
                    version: "test".to_owned(),
                    runtime_dir: dir.join("runtime"),
                    lemond: dir.join("lemond"),
                    lemonade: dir.join("lemonade"),
                    backend_recipe: LLAMACPP_RECIPE.to_owned(),
                    backend_name: "vulkan".to_owned(),
                    installed_at_unix_ms: 0,
                },
            };
            if tag == "direct" {
                crate::state::write_running_state(
                    &request,
                    &runtime,
                    std::process::id(),
                    None,
                    "ready",
                )
                .unwrap();
            } else {
                crate::state::write_running_state_for_model(
                    &request,
                    &runtime,
                    std::process::id(),
                    None,
                    "ready",
                    served_name,
                )
                .unwrap();
            }
            crate::state::merge_json_state(
                &request.state_path,
                &json!({"backend_requested": "vulkan"}),
            )
            .unwrap();
            let response = healthcheck_service_at(&request.state_path).unwrap();
            let seen = server.join().unwrap();
            let chat = seen
                .iter()
                .find(|recorded| recorded.method == "POST")
                .expect("health inference request was not received");
            assert_eq!(chat.path, "/v1/chat/completions");
            let chat_model = serde_json::from_str::<Value>(&chat.body).unwrap()["model"]
                .as_str()
                .unwrap()
                .to_owned();
            let state = crate::state::read_service_state(&request.state_path).unwrap();
            assert_eq!(response.status, "ready");
            assert_eq!(state["model_ref"], "qwen-smoke");
            assert_eq!(chat_model, served_name);
            assert_eq!(state["canonical_model_id"], served_name);
            assert!(state.get(rocm_core::INFERENCE_VERIFIED_STATE_KEY).is_some());
        }
    }

    fn launch_request(model_ref: &str) -> LaunchRequest {
        LaunchRequest {
            service_id: "svc".to_owned(),
            env_id: None,
            runtime_id: None,
            model_ref: model_ref.to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: None,
            endpoint_mode: None,
            engine_recipe: None,
            gpu_selection: None,
        }
    }

    #[test]
    fn launch_serves_the_identity_resolve_reports() {
        // The engine's Launch method normalizes the requested reference once, so what
        // it serves and records is what `resolve` reports for it. `serve-http`
        // itself never normalizes (direct serving keeps the exact name it is given).
        for (requested, reported) in [
            ("qwen-smoke", "Qwen3-0.6B-GGUF"),
            ("assistant", DEFAULT_MODEL),
            ("owner/repo:Q4_0", "owner/repo:Q4_0"),
            ("Some-Registered-Model", "Some-Registered-Model"),
        ] {
            let resolved = resolve_model_response(ResolveModelRequest {
                model_ref: requested.to_owned(),
                runtime_id: None,
                device_policy: None,
                recipe_override: None,
                engine_recipe: None,
            })
            .unwrap();
            assert_eq!(resolved.canonical_model_id, reported);
            let serve = launch_serve_request(
                &launch_request(requested),
                Path::new("state.json"),
                Path::new("service.log"),
            );
            assert_eq!(serve.model_ref, resolved.canonical_model_id, "{requested}");
            // The wrapper process is started with that same identity.
            let args = crate::state::serve_http_command_args(&serve);
            assert_eq!(args[2], reported, "{args:?}");
        }
    }

    /// A confined serve environment: a fake runtime under `dir`, a model cache under
    /// `dir/cache`, and a parent environment rooted at `dir`.
    #[cfg(target_os = "linux")]
    fn fake_context(dir: &Path, manifest: install::LemonadeInstallManifest) -> ServeContext {
        ServeContext {
            runtime: crate::backend_alignment::LemonadeRuntime { manifest },
            process_env: crate::process::LemonadeProcessEnvironment::default(),
            paths: AppPaths {
                config_dir: dir.join("config"),
                data_dir: dir.join("data"),
                cache_dir: dir.join("cache"),
            },
            parent: crate::runtime_dir::ParentRuntimeEnvironment::rooted_at(dir),
        }
    }

    #[cfg(target_os = "linux")]
    fn fake_request(dir: &Path, model_ref: &str) -> ServeHttpRequest {
        ServeHttpRequest {
            service_id: "svc".to_owned(),
            model_ref: model_ref.to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: DevicePolicy::GpuRequired,
            gpu_indices: vec![0],
            runtime_id: None,
            env_id: None,
            state_path: dir.join("state.json"),
            log_path: None,
            engine_recipe: None,
        }
    }

    #[cfg(target_os = "linux")]
    fn install_fake_llama_server(manifest: &install::LemonadeInstallManifest, script: &str) {
        crate::test_http::fake_runtime::write_executable(
            &manifest
                .runtime_dir
                .join("bin/llamacpp/vulkan/llama-server"),
            script,
        );
    }

    #[cfg(target_os = "linux")]
    fn write_registry(manifest: &install::LemonadeInstallManifest) {
        let resources = manifest.runtime_dir.join("resources");
        std::fs::create_dir_all(&resources).unwrap();
        std::fs::write(
            resources.join("server_models.json"),
            json!({
                "Qwen3-0.6B-GGUF": {
                    "recipe": "llamacpp",
                    "checkpoint": "owner/serve-test-repo:Q4_0",
                    "suggested": true,
                    "size": 0.38,
                    "labels": ["reasoning"]
                }
            })
            .to_string(),
        )
        .unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn serve_errors_before_the_server_starts_leave_state_failed() {
        use crate::test_http::fake_runtime;
        // Each case fails somewhere between the `starting` state being written and a
        // server being spawned. None may leave the state at `starting`, and the
        // error that reaches the caller is the original one.
        for (case, model_ref, with_binary, pull_exit, expected) in [
            // serving_route: the selected backend's binary is missing.
            (
                "missing binary",
                "owner/serve-test-repo:Q4_0",
                false,
                0,
                "no selected GPU llama-server",
            ),
            // serving_route: a checkpoint without a variant.
            (
                "invalid checkpoint",
                "owner/serve-test-repo",
                true,
                0,
                "needs an explicit quantization variant",
            ),
            // ensure_model_in: the download fails.
            (
                "failed pull",
                "owner/serve-test-repo:Q4_0",
                true,
                7,
                "Lemonade pull of `owner/serve-test-repo:Q4_0` failed",
            ),
            // ensure_model_in: the download succeeds but yields no usable GGUF.
            (
                "pull without file",
                "owner/serve-test-repo:Q4_0",
                true,
                0,
                "did not download",
            ),
        ] {
            let temporary = tempfile::tempdir().unwrap();
            let dir = temporary.path();
            let manifest = fake_runtime::manifest(dir, pull_exit, None);
            if with_binary {
                install_fake_llama_server(&manifest, "#!/bin/sh\nexit 0\n");
            }
            let request = fake_request(dir, model_ref);
            let context = fake_context(dir, manifest);
            let error = serve_http_resolving(request.clone(), |_| Ok(context))
                .expect_err(case)
                .to_string();
            assert!(error.contains(expected), "{case}: {error}");
            let state = crate::state::read_service_state(&request.state_path).unwrap();
            assert_eq!(state["status"], "failed", "{case}");
            assert_eq!(state["model_ref"], model_ref, "{case}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn serve_error_while_resolving_the_runtime_marks_launch_state_failed() {
        // `launch` wrote `starting` before the wrapper ran; a wrapper that fails to
        // resolve its runtime must not leave that behind.
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let request = fake_request(dir, "qwen-smoke");
        std::fs::write(
            &request.state_path,
            json!({"status": "starting"}).to_string(),
        )
        .unwrap();
        let error = serve_http_resolving(request.clone(), |_| anyhow::bail!("no runtime"))
            .unwrap_err()
            .to_string();
        assert_eq!(error, "no runtime");
        let state = crate::state::read_service_state(&request.state_path).unwrap();
        assert_eq!(state["status"], "failed");
        // With no state file there is nothing to mark, and none is invented.
        let missing = fake_request(&dir.join("elsewhere"), "qwen-smoke");
        let _ = serve_http_resolving(missing.clone(), |_| anyhow::bail!("no runtime"));
        assert!(!missing.state_path.exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn direct_serve_through_serve_http_uses_the_requested_alias() {
        use crate::test_http::{Reply, fake_runtime, spawn_fixture};
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let manifest = fake_runtime::manifest(dir, 0, None);
        write_registry(&manifest);
        let args_file = dir.join("server-args.txt");
        install_fake_llama_server(
            &manifest,
            &format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nexit 0\n",
                args_file.display()
            ),
        );
        // The registered checkpoint is already cached, so nothing is downloaded.
        let gguf = dir.join(
            "cache/huggingface/hub/models--owner--serve-test-repo/snapshots/rev/model-Q4_0.gguf",
        );
        std::fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        std::fs::write(&gguf, b"gguf").unwrap();
        let (port, server) = spawn_fixture(Duration::from_secs(10), |recorded| {
            let chat = recorded.method == "POST";
            Reply {
                body: if chat {
                    json!({"choices": [{"message": {"content": "ok"}}]})
                } else {
                    json!({"data": [{"id": "qwen-smoke"}]})
                }
                .to_string(),
                last: chat,
            }
        });
        let mut request = fake_request(dir, "qwen-smoke");
        request.port = port;
        let context = fake_context(dir, manifest);
        serve_http_resolving(request.clone(), |_| Ok(context)).unwrap();

        let args = std::fs::read_to_string(&args_file)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let after = |flag: &str| {
            let at = args.iter().position(|arg| arg == flag).expect(flag);
            args[at + 1].clone()
        };
        // Served under exactly the name that was requested, not the registered name.
        assert_eq!(after("--alias"), "qwen-smoke", "{args:?}");
        assert_eq!(after("-m"), gguf.display().to_string(), "{args:?}");
        assert_eq!(after("--port"), port.to_string(), "{args:?}");
        let chat = server
            .join()
            .unwrap()
            .into_iter()
            .find(|recorded| recorded.method == "POST")
            .expect("smoke request");
        assert_eq!(
            serde_json::from_str::<Value>(&chat.body).unwrap()["model"],
            "qwen-smoke"
        );
        let state = crate::state::read_service_state(&request.state_path).unwrap();
        assert_eq!(state["model_ref"], "qwen-smoke");
        assert_eq!(state["canonical_model_id"], "qwen-smoke");
        assert_eq!(state["status"], "stopped");
    }

    #[cfg(target_os = "linux")]
    fn fallback_fixture(
        dir: &Path,
        script: &str,
    ) -> (
        ServeContext,
        ServeHttpRequest,
        std::path::PathBuf,
        u16,
        std::thread::JoinHandle<Vec<crate::test_http::Recorded>>,
    ) {
        use crate::test_http::{Reply, fake_runtime, spawn_fixture};
        let manifest = fake_runtime::manifest(dir, 0, None);
        write_registry(&manifest);
        install_fake_llama_server(&manifest, script);
        let gguf = dir.join(
            "cache/huggingface/hub/models--owner--serve-test-repo/snapshots/rev/model-Q4_0.gguf",
        );
        std::fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        std::fs::write(&gguf, b"gguf").unwrap();
        let (port, server) = spawn_fixture(Duration::from_secs(10), |recorded| {
            let chat = recorded.method == "POST";
            Reply {
                body: if chat {
                    json!({"choices": [{"message": {"content": "ok"}}]})
                } else {
                    json!({"data": [{"id": "qwen-smoke"}]})
                }
                .to_string(),
                last: chat,
            }
        });
        let mut request = fake_request(dir, "qwen-smoke");
        request.port = port;
        request.log_path = Some(dir.join("serve.log"));
        (fake_context(dir, manifest), request, gguf, port, server)
    }

    #[cfg(target_os = "linux")]
    fn args_script(args_file: &Path) -> String {
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nexit 0\n",
            args_file.display()
        )
    }

    #[cfg(target_os = "linux")]
    fn marked_load_failure() -> anyhow::Error {
        anyhow::Error::msg("load exploded").context(crate::router_serve::RouterLoadFailure(
            "failed to load qwen-smoke".to_owned(),
        ))
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn router_load_failure_falls_back_to_the_selected_gpu_llama_server() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let args_file = dir.join("server-args.txt");
        let (context, request, gguf, port, server) =
            fallback_fixture(dir, &args_script(&args_file));
        with_direct_fallback(
            Err(marked_load_failure()),
            &context.paths,
            &request,
            &context.runtime,
            &context.process_env,
            request.log_path.as_deref(),
            &context.parent,
            true,
        )
        .unwrap();
        server.join().unwrap();

        // The state shows a direct serve that ran, under the requested name.
        let args = std::fs::read_to_string(&args_file).unwrap();
        assert!(
            args.contains(&format!("-m\n{}\n", gguf.display())),
            "{args}"
        );
        assert!(args.contains(&format!("--port\n{port}\n")), "{args}");
        assert!(args.contains("--alias\nqwen-smoke\n"), "{args}");
        let state = crate::state::read_service_state(&request.state_path).unwrap();
        assert_eq!(state["status"], "stopped");
        // The message names the backend of the server it launched, derived from
        // the same path, and says it is not ROCm.
        let log = std::fs::read_to_string(request.log_path.as_ref().unwrap()).unwrap();
        assert!(
            log.contains("serving it directly with the selected vulkan GPU llama-server"),
            "{log}"
        );
        assert!(log.contains("load exploded"), "{log}");
        assert!(
            log.contains("Lemonade is serving on the vulkan GPU backend, not ROCm"),
            "{log}"
        );
        assert!(!log.contains("cpu"), "{log}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn only_a_router_load_failure_may_fall_back_and_only_on_linux() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let args_file = dir.join("server-args.txt");
        let (context, request, _gguf, _port, server) =
            fallback_fixture(dir, &args_script(&args_file));
        let run = |result: Result<()>, is_linux: bool| {
            with_direct_fallback(
                result,
                &context.paths,
                &request,
                &context.runtime,
                &context.process_env,
                request.log_path.as_deref(),
                &context.parent,
                is_linux,
            )
        };
        // A server that never became ready, a failed backend selection or a later
        // exit is returned as it is, though a direct route is open.
        let other = run(
            Err(anyhow::anyhow!("Lemonade server did not become ready")),
            true,
        )
        .unwrap_err()
        .to_string();
        assert_eq!(other, "Lemonade server did not become ready");
        // Off Linux the router's own error is returned unchanged.
        let windows = run(Err(marked_load_failure()), false)
            .unwrap_err()
            .to_string();
        assert_eq!(windows, "failed to load qwen-smoke");
        assert!(!args_file.exists(), "no llama-server may have started");
        assert!(!request.log_path.as_ref().unwrap().exists());
        assert!(!request.state_path.exists());
        drop(server);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fallback_never_serves_through_a_missing_or_cpu_backend() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let manifest = crate::test_http::fake_runtime::manifest(dir, 0, None);
        write_registry(&manifest);
        let request = fake_request(dir, "qwen-smoke");
        // The selected GPU backend has no llama-server: nothing to fall back to.
        assert!(crate::direct_llama::direct_fallback_route(&manifest, &request, true).is_none());
        // Only a CPU llama-server exists and the backend is CPU: still refused.
        let mut cpu = manifest.clone();
        cpu.backend_name = "cpu".to_owned();
        crate::test_http::fake_runtime::write_executable(
            &cpu.runtime_dir.join("bin/llamacpp/cpu/llama-server"),
            "#!/bin/sh\nexit 0\n",
        );
        assert!(crate::direct_llama::direct_fallback_route(&cpu, &request, true).is_none());
        // With the selected GPU backend installed the route opens on Linux only.
        install_fake_llama_server(&manifest, "#!/bin/sh\nexit 0\n");
        assert!(crate::direct_llama::direct_fallback_route(&manifest, &request, true).is_some());
        assert!(crate::direct_llama::direct_fallback_route(&manifest, &request, false).is_none());
    }

    #[test]
    fn stop_scope_targets_only_the_recorded_lemonade_server() {
        assert_eq!(STOP_SCOPE, rocm_core::KillScope::Single);
    }
}
