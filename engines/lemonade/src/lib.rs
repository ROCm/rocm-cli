// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_PORT, RocmCliConfig, active_managed_therock_environment,
    download_file_to_path, format_host_port, format_http_base_url, http_get_text,
    normalize_runtime_path_for_host, prepend_runtime_paths, require_nonempty, runtime_is_linux,
    runtime_is_windows,
};
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, DevicePolicy, ENGINE_RECIPE_CONTRACT_VERSION, EndpointRequest,
    EndpointResponse, EngineCapabilities, EngineDeviceAvailability, EngineMethod, EngineRecipeHint,
    EngineRequestEnvelope, EngineResponseEnvelope, GpuSelection, HealthcheckRequest,
    HealthcheckResponse, InstallRequest, InstallResponse, LaunchRequest, LaunchResponse,
    LogsRequest, LogsResponse, ResolveModelRequest, ResolveModelResponse, StopRequest,
    StopResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{VecDeque, hash_map::DefaultHasher};
use std::ffi::OsString;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Read, Seek, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ENGINE_NAME: &str = "lemonade";
const LEMONADE_VERSION: &str = "10.6.0";
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_MODEL: &str = "Qwen3-4B-Instruct-2507-GGUF";
const DEFAULT_MODEL_REPO_DIR: &str = "models--unsloth--Qwen3-4B-Instruct-2507-GGUF";
const DEFAULT_MODEL_GGUF: &str = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
const LLAMACPP_RECIPE: &str = "llamacpp";
const ROCM_BACKEND_NAME: &str = "rocm";
/// Preferred llama.cpp backends, best first. Lemonade reports per-GPU support;
/// we pick the highest-priority backend it considers supported on this host.
const LLAMACPP_BACKEND_PRIORITY: [&str; 3] = ["rocm", "vulkan", "cpu"];
const DEFAULT_LOG_TAIL_LINES: usize = 200;
const STARTUP_FAILURE_LOG_TAIL_LINES: usize = 80;
/// Maximum bytes to read from the end of a log file when extracting tail lines.
/// Prevents reading entire gigabyte-sized logs on startup timeout.
const MAX_TAIL_READ: u64 = 4 * 1024 * 1024; // 4MB

const EMBEDDABLE_WINDOWS_ARCHIVE_NAME: &str = "lemonade-embeddable-10.6.0-windows-x64.zip";
const EMBEDDABLE_LINUX_ARCHIVE_NAME: &str = "lemonade-embeddable-10.6.0-ubuntu-x64.tar.gz";
const EMBEDDABLE_WINDOWS_URL: &str = "https://github.com/lemonade-sdk/lemonade/releases/download/v10.6.0/lemonade-embeddable-10.6.0-windows-x64.zip";
const EMBEDDABLE_LINUX_URL: &str = "https://github.com/lemonade-sdk/lemonade/releases/download/v10.6.0/lemonade-embeddable-10.6.0-ubuntu-x64.tar.gz";

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LemonadeInstallManifest {
    env_id: String,
    version: String,
    runtime_dir: PathBuf,
    lemond: PathBuf,
    lemonade: PathBuf,
    backend_recipe: String,
    backend_name: String,
    installed_at_unix_ms: u128,
}

#[derive(Debug, Clone)]
struct LemonadeRuntime {
    manifest: LemonadeInstallManifest,
}

#[derive(Debug, Clone, Default)]
struct LemonadeProcessEnvironment {
    rocm_root: Option<PathBuf>,
    path_entries: Vec<PathBuf>,
    library_entries: Vec<PathBuf>,
    gpu_indices: Vec<u32>,
}

#[derive(Debug, Clone)]
struct ServeHttpRequest {
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
    engine_recipe: Option<EngineRecipeHint>,
}

#[derive(Debug, Clone)]
struct ServiceFiles {
    state_path: PathBuf,
    log_path: PathBuf,
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
                gpu_selection: None,
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
            device_policy: Some(parse_device_policy_arg(device_policy.as_deref())?),
            endpoint_mode: Some("openai".to_owned()),
            engine_recipe: None,
            gpu_selection: parse_gpu_selection_arg(gpu.as_deref())?,
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
            device_policy: parse_device_policy_arg(device_policy.as_deref())?,
            gpu_indices: parse_gpu_indices_arg(gpu.as_deref())?,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe: parse_engine_recipe_json(engine_recipe_json)?,
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
    engine_recipe: Option<EngineRecipeHint>,
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
        quantized_models: "GGUF through Lemonade llama.cpp (ROCm/Vulkan/CPU auto-selected)"
            .to_owned(),
        reasoning_parser: false,
    }
}

fn detect_response() -> DetectResponse {
    let runtime = resolve_runtime().ok();
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
        available_devices: vec![EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: runtime.is_some(),
            reason: if runtime.is_some() {
                None
            } else {
                Some("Lemonade embeddable runtime is not installed".to_owned())
            },
        }],
        capabilities: capabilities(),
        notes,
    }
}

fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    eprintln!("Preparing Lemonade embeddable {LEMONADE_VERSION}...");
    let env_root = request
        .env_root
        .as_deref()
        .map(normalize_runtime_path_for_host);
    let mut manifest = prepare_embeddable(&paths, env_root.as_deref(), request.reinstall)?;
    eprintln!("Detecting best supported Lemonade llama.cpp backend...");
    install_best_llamacpp_backend(&mut manifest)?;
    write_manifest(&paths, &manifest)?;
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
        lock_hash: manifest_lock_hash(&manifest),
        warnings: vec![
            "Lemonade is installed as a rocm-cli managed embeddable runtime".to_owned(),
            format!(
                "Selected the best supported llama.cpp backend for this GPU: {}:{}",
                manifest.backend_recipe, manifest.backend_name
            ),
        ],
    })
}

fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    let device_policy = normalize_device_policy(request.device_policy)?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let canonical_model_id = resolve_lemonade_model_ref(&request.model_ref);
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
            "Lemonade auto-selects the best supported llama.cpp backend for this GPU (ROCm, then Vulkan, then CPU)".to_owned(),
        ],
    })
}

fn launch_service(mut request: LaunchRequest) -> Result<LaunchResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    require_nonempty(&request.model_ref, "model_ref")?;
    request.device_policy = Some(normalize_device_policy(request.device_policy.clone())?);
    request.engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let runtime = resolve_runtime()?;
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    fs::create_dir_all(paths.engine_logs_dir(ENGINE_NAME))?;
    fs::create_dir_all(paths.engine_state_dir(ENGINE_NAME))?;
    let log_path = paths
        .engine_logs_dir(ENGINE_NAME)
        .join(format!("{}.log", request.service_id));
    let state_path = paths
        .engine_state_dir(ENGINE_NAME)
        .join(format!("{}.json", request.service_id));
    let endpoint_url = endpoint_url(&request.host, request.port);
    let serve_request = ServeHttpRequest {
        service_id: request.service_id.clone(),
        model_ref: resolve_lemonade_model_ref(&request.model_ref),
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
    let args = serve_http_command_args(&serve_request);
    write_running_state(
        &serve_request,
        &runtime,
        std::process::id(),
        None,
        "starting",
    )?;
    let wrapper_pid = spawn_serve_http_background(&current_exe, &args)?;
    merge_json_state(
        &state_path,
        &json!({
            "pid": wrapper_pid,
            "wrapper_pid": wrapper_pid,
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

fn serve_http(request: ServeHttpRequest) -> Result<()> {
    require_gpu_required(&request.device_policy)?;
    let runtime = resolve_runtime()?;
    let mut process_env = lemonade_process_environment()?;
    process_env.gpu_indices = request.gpu_indices.clone();
    let log_path = request.log_path.as_deref();
    write_running_state(&request, &runtime, std::process::id(), None, "starting")?;
    if runtime_is_linux() && direct_llama_server_path(&runtime.manifest).is_file() {
        ensure_direct_llama_model_available(&request, &runtime, &process_env, log_path)?;
        return serve_direct_rocm_llama_server(
            &request,
            &runtime,
            &process_env,
            log_path,
            &anyhow!("using Lemonade packaged ROCm llama-server directly on Linux"),
        );
    }
    let mut child = spawn_lemond(
        &runtime.manifest,
        &request.host,
        request.port,
        log_path,
        &process_env,
    )?;
    write_running_state(
        &request,
        &runtime,
        std::process::id(),
        Some(child.id()),
        "running",
    )?;
    wait_for_lemonade_cli_status(
        &runtime.manifest,
        &request.host,
        request.port,
        Duration::from_secs(30),
        log_path,
        &process_env,
    )
    .context("Lemonade server did not become ready")?;
    let backend =
        ensure_best_llamacpp_backend(&runtime.manifest, &request.host, request.port, &process_env)
            .context("failed to select a supported Lemonade llama.cpp backend")?;
    let load_result = run_lemonade_model_load(
        &runtime.manifest,
        &request.host,
        request.port,
        &request.model_ref,
        &backend,
        log_path,
        &process_env,
    );
    let router_ready = load_result.is_ok()
        && query_loaded_model_endpoint(
            &endpoint_url(&request.host, request.port),
            &request.model_ref,
            &backend,
        )
        .unwrap_or(false)
        && query_chat_smoke_endpoint(&request.host, request.port, &request.model_ref)
            .unwrap_or(false);
    if let Err(error) = load_result {
        if runtime_is_linux() && direct_llama_server_path(&runtime.manifest).is_file() {
            let _ = terminate_pid(child.id(), true);
            let _ = child.wait();
            return serve_direct_rocm_llama_server(
                &request,
                &runtime,
                &process_env,
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
        if runtime_is_linux() && direct_llama_server_path(&runtime.manifest).is_file() {
            let _ = terminate_pid(child.id(), true);
            let _ = child.wait();
            return serve_direct_rocm_llama_server(
                &request,
                &runtime,
                &process_env,
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
    merge_json_state(
        &request.state_path,
        &json!({
            "status": "ready",
            "server_pid": child.id(),
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
    mark_json_status(
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

fn ensure_direct_llama_model_available(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    if direct_llama_model_path(&paths, &request.model_ref).is_some() {
        return Ok(());
    }

    let download_port = free_local_port()?;
    let mut child = spawn_lemond(
        &runtime.manifest,
        DEFAULT_HOST,
        download_port,
        log_path,
        process_env,
    )?;
    let result = (|| -> Result<()> {
        wait_for_lemonade_cli_status(
            &runtime.manifest,
            DEFAULT_HOST,
            download_port,
            Duration::from_secs(30),
            log_path,
            process_env,
        )?;
        let _ = run_lemonade_model_load(
            &runtime.manifest,
            DEFAULT_HOST,
            download_port,
            &request.model_ref,
            ROCM_BACKEND_NAME,
            log_path,
            process_env,
        );
        if direct_llama_model_path(&paths, &request.model_ref).is_some() {
            Ok(())
        } else {
            bail!(
                "Lemonade did not download `{}` for direct ROCm serving",
                request.model_ref
            )
        }
    })();
    let _ = terminate_pid(child.id(), true);
    let _ = child.wait();
    result
}

fn healthcheck_service(request: HealthcheckRequest) -> Result<HealthcheckResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path).ok();
    let endpoint_url = state.as_ref().and_then(endpoint_url_from_state);
    let state_status = state
        .as_ref()
        .and_then(|value| value_string(value, "status"))
        .unwrap_or_else(|| "unknown".to_owned());
    let model_ref = state
        .as_ref()
        .and_then(|value| {
            value_string(value, "canonical_model_id").or_else(|| value_string(value, "model_ref"))
        })
        .unwrap_or_default();
    let backend = state
        .as_ref()
        .and_then(|value| value_string(value, "backend_requested"))
        .unwrap_or_else(|| ROCM_BACKEND_NAME.to_owned());
    let ready = state_status == "ready"
        && !model_ref.is_empty()
        && endpoint_url
            .as_deref()
            .map(|endpoint| query_loaded_model_endpoint(endpoint, &model_ref, &backend))
            .transpose()
            .unwrap_or(None)
            .unwrap_or(false);
    let status = if ready {
        "ready".to_owned()
    } else {
        state_status
    };
    Ok(HealthcheckResponse {
        status,
        model_loaded: ready,
        device: if ready {
            "rocm_gpu".to_owned()
        } else {
            "unknown".to_owned()
        },
        uptime_sec: 0,
        queue_depth: 0,
        last_error: None,
        tokens_per_sec: None,
    })
}

fn endpoint_response(request: EndpointRequest) -> Result<EndpointResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path)
        .with_context(|| format!("service state not found for `{}`", request.service_id))?;
    let endpoint_url = endpoint_url_from_state(&state)
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
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let limit = request.tail_lines.unwrap_or(DEFAULT_LOG_TAIL_LINES);
    Ok(LogsResponse {
        log_path: files.log_path.display().to_string(),
        recent_lines: if files.log_path.is_file() {
            tail_lines(&files.log_path, limit)?
        } else {
            Vec::new()
        },
    })
}

fn stop_service(request: StopRequest) -> Result<StopResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path).ok();
    let stopped = match state.as_ref().and_then(pid_to_terminate_from_state) {
        Some(pid) => terminate_pid(pid, request.force),
        None => false,
    };
    if stopped {
        mark_json_status(&files.state_path, "stopped")?;
    }
    Ok(StopResponse {
        stopped,
        graceful: stopped && !request.force,
    })
}

fn prepare_embeddable(
    paths: &AppPaths,
    env_root: Option<&Path>,
    reinstall: bool,
) -> Result<LemonadeInstallManifest> {
    let root = lemonade_root(paths, env_root);
    let archive_name = embeddable_archive_name();
    let archive_url = embeddable_url();
    let downloads = root.join("downloads");
    let archive = downloads.join(archive_name);
    fs::create_dir_all(&downloads)?;
    if archive.is_file() {
        eprintln!("Using cached {archive_name}.");
    } else {
        eprintln!("Downloading {archive_name}...");
        download_file(archive_url, &archive)?;
    }
    let runtime_dir = runtime_dir_in(&root);
    if reinstall || !lemond_path_in(&runtime_dir).is_file() {
        if runtime_dir.exists() {
            fs::remove_dir_all(&runtime_dir)
                .with_context(|| format!("failed to clear {}", runtime_dir.display()))?;
        }
        let extract_root = root
            .join("extract")
            .join(format!("{}", current_unix_millis()));
        fs::create_dir_all(&extract_root)?;
        extract_archive(&archive, &extract_root)?;
        let embeddable_root = find_embeddable_root(&extract_root)?;
        copy_tree(&embeddable_root, &runtime_dir)?;
        fs::remove_dir_all(&extract_root).ok();
    }
    let lemond = lemond_path_in(&runtime_dir);
    let lemonade = lemonade_path_in(&runtime_dir);
    if !lemond.is_file() || !lemonade.is_file() {
        bail!(
            "Lemonade embeddable extraction did not produce lemonade/lemond binaries in {}",
            runtime_dir.display()
        );
    }
    Ok(LemonadeInstallManifest {
        env_id: format!("lemonade-embeddable-{LEMONADE_VERSION}"),
        version: LEMONADE_VERSION.to_owned(),
        runtime_dir,
        lemond,
        lemonade,
        backend_recipe: LLAMACPP_RECIPE.to_owned(),
        // Resolved during install once Lemonade reports per-GPU backend support.
        backend_name: ROCM_BACKEND_NAME.to_owned(),
        installed_at_unix_ms: current_unix_millis(),
    })
}

fn install_best_llamacpp_backend(manifest: &mut LemonadeInstallManifest) -> Result<()> {
    let port = free_local_port()?;
    let log_path_buf = manifest.runtime_dir.join("install-lemond.log");
    let log_path = Some(log_path_buf.as_path());
    let process_env = lemonade_process_environment()?;
    let mut child = spawn_lemond(manifest, DEFAULT_HOST, port, log_path, &process_env)?;
    let result = (|| -> Result<String> {
        wait_for_lemonade_cli_status(
            manifest,
            DEFAULT_HOST,
            port,
            Duration::from_secs(30),
            log_path,
            &process_env,
        )?;
        ensure_best_llamacpp_backend(manifest, DEFAULT_HOST, port, &process_env)
    })();
    let _ = terminate_pid(child.id(), true);
    let _ = child.wait();
    manifest.backend_name = result?;
    Ok(())
}

/// Ask Lemonade which llama.cpp backends it supports on this GPU, choose the best
/// one (`LLAMACPP_BACKEND_PRIORITY`), install it if necessary, and return its name.
fn ensure_best_llamacpp_backend(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    process_env: &LemonadeProcessEnvironment,
) -> Result<String> {
    let listing = run_lemonade_backends_list(manifest, process_env)?;
    let backends = parse_llamacpp_backend_statuses(&listing);
    let Some((backend, already_installed)) = select_best_llamacpp_backend(&backends) else {
        bail!(
            "Lemonade reports no supported llama.cpp backend for this GPU (status: {})",
            describe_llamacpp_backends(&backends)
        );
    };
    if already_installed {
        eprintln!("Using installed Lemonade {LLAMACPP_RECIPE}:{backend} backend.");
    } else {
        eprintln!("Installing Lemonade {LLAMACPP_RECIPE}:{backend} backend...");
        run_lemonade_backend_install(manifest, host, port, &backend, process_env)?;
    }
    Ok(backend)
}

/// Parse `lemonade backends` table output into `(backend, status)` pairs for the
/// `llamacpp` recipe only. Status is one of `installed`/`installable`/`unsupported`.
fn parse_llamacpp_backend_statuses(output: &str) -> Vec<(String, String)> {
    const RECIPES: [&str; 8] = [
        "flm",
        "kokoro",
        "llamacpp",
        "ryzenai-llm",
        "sd-cpp",
        "vllm",
        "whispercpp",
        "embeddings",
    ];
    const STATUSES: [&str; 3] = ["installed", "installable", "unsupported"];
    let mut current_recipe = "";
    let mut result = Vec::new();
    for line in output.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&first) = tokens.first() else {
            continue;
        };
        if first == "Recipe" || first.starts_with("---") {
            continue;
        }
        // A row either starts with a recipe name (recipe + its first backend) or,
        // for grouped recipes, with a backend name continuing the previous recipe.
        let (backend, rest) = if RECIPES.contains(&first) {
            current_recipe = match RECIPES.iter().find(|r| **r == first) {
                Some(r) => r,
                None => current_recipe,
            };
            match tokens.get(1) {
                Some(backend) => (*backend, &tokens[2..]),
                None => continue,
            }
        } else {
            (first, &tokens[1..])
        };
        if current_recipe != LLAMACPP_RECIPE {
            continue;
        }
        if let Some(status) = rest.iter().find(|t| STATUSES.contains(t)) {
            result.push((backend.to_owned(), (*status).to_owned()));
        }
    }
    result
}

/// Choose the highest-priority backend Lemonade considers usable (installed or
/// installable). Returns `(backend, already_installed)`.
fn select_best_llamacpp_backend(backends: &[(String, String)]) -> Option<(String, bool)> {
    for candidate in LLAMACPP_BACKEND_PRIORITY {
        if let Some((name, status)) = backends
            .iter()
            .find(|(b, s)| b == candidate && (s == "installed" || s == "installable"))
        {
            return Some((name.clone(), status == "installed"));
        }
    }
    None
}

fn describe_llamacpp_backends(backends: &[(String, String)]) -> String {
    if backends.is_empty() {
        return "none reported".to_owned();
    }
    backends
        .iter()
        .map(|(b, s)| format!("{b}={s}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn run_lemonade_backends_list(
    manifest: &LemonadeInstallManifest,
    process_env: &LemonadeProcessEnvironment,
) -> Result<String> {
    let mut command = ProcessCommand::new(&manifest.lemonade);
    command
        .arg("backends")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_lemonade_process_environment(&mut command, process_env)?;
    hide_child_console_window(&mut command);
    let output = command
        .output()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !output.status.success() {
        bail!(
            "Lemonade backends query failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn resolve_runtime() -> Result<LemonadeRuntime> {
    let paths = AppPaths::discover()?;
    let manifest = read_manifest(&paths)?;
    if !manifest.lemond.is_file() {
        bail!(
            "Lemonade runtime is missing {}; run `rocm engines install lemonade`",
            manifest.lemond.display()
        );
    }
    Ok(LemonadeRuntime { manifest })
}

fn manifest_path(paths: &AppPaths) -> PathBuf {
    paths.engine_manifests_dir(ENGINE_NAME).join("runtime.json")
}

fn lemonade_root(paths: &AppPaths, env_root: Option<&Path>) -> PathBuf {
    env_root.map_or_else(
        || paths.engine_dir(ENGINE_NAME),
        |root| normalize_runtime_path_for_host(root).join(ENGINE_NAME),
    )
}

fn runtime_dir_in(root: &Path) -> PathBuf {
    root.join("runtime")
}

fn read_manifest(paths: &AppPaths) -> Result<LemonadeInstallManifest> {
    let path = manifest_path(paths);
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_manifest(paths: &AppPaths, manifest: &LemonadeInstallManifest) -> Result<()> {
    let path = manifest_path(paths);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_vec_pretty(manifest)?)?;
    Ok(())
}

fn lemond_path_in(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(platform_binary_name("lemond"))
}

fn lemonade_path_in(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(platform_binary_name("lemonade"))
}

fn platform_binary_name(name: &str) -> String {
    if runtime_is_windows() {
        format!("{name}.exe")
    } else {
        name.to_owned()
    }
}

const fn embeddable_archive_name() -> &'static str {
    if runtime_is_windows() {
        EMBEDDABLE_WINDOWS_ARCHIVE_NAME
    } else {
        EMBEDDABLE_LINUX_ARCHIVE_NAME
    }
}

const fn embeddable_url() -> &'static str {
    if runtime_is_windows() {
        EMBEDDABLE_WINDOWS_URL
    } else {
        EMBEDDABLE_LINUX_URL
    }
}

fn download_file(url: &str, destination: &Path) -> Result<()> {
    download_file_to_path(url, destination, Duration::from_mins(15))
}

fn extract_archive(archive: &Path, destination: &Path) -> Result<()> {
    let stderr_path = destination.join("extract-stderr.txt");
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let mut command = ProcessCommand::new("tar");
    command.arg(if runtime_is_windows() { "-xf" } else { "-xzf" });
    command.arg(archive).arg("-C").arg(destination);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file));
    let status = command.status().context("failed to run tar")?;
    if status.success() {
        let _ = fs::remove_file(stderr_path);
        Ok(())
    } else {
        let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
        let _ = fs::remove_file(stderr_path);
        bail!(
            "tar extraction failed with status {}; stderr: {}",
            status,
            stderr.trim()
        )
    }
}

fn windows_child_path(path: &Path) -> String {
    let raw = path.display().to_string();
    let normalized = raw.replace('\\', "/");
    let bytes = normalized.as_bytes();
    if bytes.len() >= 3 && bytes[0] == b'/' && bytes[1].is_ascii_alphabetic() && bytes[2] == b'/' {
        let drive = (bytes[1] as char).to_ascii_uppercase();
        let rest = normalized[3..].replace('/', "\\");
        return format!("{drive}:\\{rest}");
    }
    if bytes.len() == 2 && bytes[0] == b'/' && bytes[1].is_ascii_alphabetic() {
        let drive = (bytes[1] as char).to_ascii_uppercase();
        return format!("{drive}:\\");
    }
    raw
}

fn find_embeddable_root(extract_root: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    collect_embeddable_roots(extract_root, &mut candidates, 0)?;
    candidates.into_iter().next().with_context(|| {
        format!(
            "no Lemonade embeddable root found in {}",
            extract_root.display()
        )
    })
}

fn collect_embeddable_roots(
    path: &Path,
    candidates: &mut Vec<PathBuf>,
    depth: usize,
) -> Result<()> {
    if depth > 4 {
        return Ok(());
    }
    if lemond_path_in(path).is_file() && lemonade_path_in(path).is_file() {
        candidates.push(path.to_path_buf());
        return Ok(());
    }
    if !path.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.path().is_dir() {
            collect_embeddable_roots(&entry.path(), candidates, depth + 1)?;
        }
    }
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_tree(&source_path, &destination_path)?;
        } else if source_path.is_file() {
            fs::copy(&source_path, &destination_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
            if !cfg!(windows) {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let metadata = fs::metadata(&source_path)?;
                    fs::set_permissions(
                        &destination_path,
                        fs::Permissions::from_mode(metadata.permissions().mode()),
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn spawn_lemond(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
) -> Result<LemondChild> {
    #[cfg(windows)]
    if let Some(log_path) = log_path {
        let args = vec![
            child_process_path(&manifest.runtime_dir),
            "--host".to_owned(),
            host.to_owned(),
            "--port".to_owned(),
            port.to_string(),
        ];
        let pid = rocm_core::spawn_hidden_console_with_log(&manifest.lemond, &args, &[], log_path)
            .with_context(|| format!("failed to start {}", manifest.lemond.display()))?;
        return Ok(LemondChild::Pid(pid));
    }

    let mut command = ProcessCommand::new(&manifest.lemond);
    command
        .arg(child_process_path(&manifest.runtime_dir))
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null());
    apply_lemonade_process_environment(&mut command, process_env)?;
    if let Some(log_path) = log_path {
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let log = fs::File::create(log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        command.stdout(Stdio::from(log.try_clone()?));
        command.stderr(Stdio::from(log));
    } else {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }
    hide_child_console_window(&mut command);
    command
        .spawn()
        .with_context(|| format!("failed to start {}", manifest.lemond.display()))
        .map(LemondChild::Child)
}

enum LemondChild {
    Child(std::process::Child),
    #[cfg(windows)]
    Pid(u32),
}

impl LemondChild {
    fn id(&self) -> u32 {
        match self {
            Self::Child(child) => child.id(),
            #[cfg(windows)]
            Self::Pid(pid) => *pid,
        }
    }

    fn wait(&mut self) -> Result<LemondExitStatus> {
        match self {
            Self::Child(child) => {
                let status = child.wait().context("failed waiting for Lemonade server")?;
                Ok(LemondExitStatus {
                    success: status.success(),
                    description: status.to_string(),
                })
            }
            #[cfg(windows)]
            Self::Pid(pid) => {
                let code = rocm_core::wait_for_process_exit(*pid)?;
                Ok(LemondExitStatus {
                    success: code == 0,
                    description: format!("exit code {code}"),
                })
            }
        }
    }
}

struct LemondExitStatus {
    success: bool,
    description: String,
}

impl LemondExitStatus {
    const fn success(&self) -> bool {
        self.success
    }
}

impl std::fmt::Display for LemondExitStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.description)
    }
}

#[cfg(windows)]
fn hide_child_console_window(command: &mut ProcessCommand) {
    use std::os::windows::process::CommandExt;

    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
const fn hide_child_console_window(_command: &mut ProcessCommand) {}

fn child_process_path(path: &Path) -> String {
    if runtime_is_windows() {
        windows_child_path(path)
    } else {
        path.display().to_string()
    }
}

fn lemonade_process_environment() -> Result<LemonadeProcessEnvironment> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    let Some(env) = active_managed_therock_environment(&paths, &config)? else {
        return Ok(LemonadeProcessEnvironment::default());
    };
    Ok(LemonadeProcessEnvironment {
        rocm_root: env.rocm_root,
        path_entries: env.path_entries,
        library_entries: env.library_entries,
        gpu_indices: Vec::new(),
    })
}

fn apply_lemonade_process_environment(
    command: &mut ProcessCommand,
    env: &LemonadeProcessEnvironment,
) -> Result<()> {
    for (key, value) in lemonade_process_environment_vars(env)? {
        command.env(key, value);
    }
    Ok(())
}

fn lemonade_process_environment_vars(
    env: &LemonadeProcessEnvironment,
) -> Result<Vec<(&'static str, OsString)>> {
    let mut vars = Vec::new();
    if let Some(rocm_root) = env.rocm_root.as_ref() {
        vars.push(("ROCM_PATH", rocm_root.as_os_str().to_owned()));
    }
    let mut path_entries = env.path_entries.clone();
    if runtime_is_windows() {
        path_entries.extend(env.library_entries.iter().cloned());
    }
    if let Some(path) = prepend_runtime_paths(&path_entries, std::env::var_os("PATH"))? {
        vars.push(("PATH", path));
    }
    if runtime_is_linux()
        && let Some(ld_library_path) =
            prepend_runtime_paths(&env.library_entries, std::env::var_os("LD_LIBRARY_PATH"))?
    {
        vars.push(("LD_LIBRARY_PATH", ld_library_path));
    }
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(&env.gpu_indices) {
        vars.push(("HIP_VISIBLE_DEVICES", OsString::from(csv)));
    }
    Ok(vars)
}

fn push_existing_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !path.exists() || paths.iter().any(|existing| existing == &path) {
        return;
    }
    paths.push(path);
}

fn wait_for_lemonade_cli_status(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    timeout: Duration,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
) -> Result<()> {
    let start = std::time::Instant::now();
    let mut last_status = None;
    while start.elapsed() < timeout {
        let mut command = ProcessCommand::new(&manifest.lemonade);
        command
            .arg("--host")
            .arg(host)
            .arg("--port")
            .arg(port.to_string())
            .arg("status")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        apply_lemonade_process_environment(&mut command, process_env)?;
        hide_child_console_window(&mut command);
        match command.status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => last_status = Some(status.to_string()),
            Err(error) => last_status = Some(error.to_string()),
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let startup_log_summary = summarize_startup_log_tail(log_path, STARTUP_FAILURE_LOG_TAIL_LINES);
    bail!(
        "Lemonade server did not become ready: {}; {}",
        last_status.unwrap_or_else(|| "not checked".to_owned()),
        startup_log_summary
    )
}

fn summarize_startup_log_tail(log_path: Option<&Path>, limit: usize) -> String {
    let Some(log_path) = log_path else {
        return "no Lemonade startup log path was configured".to_owned();
    };
    if !log_path.is_file() {
        return format!("Lemonade startup log not found at {}", log_path.display());
    }
    match tail_lines(log_path, limit) {
        Ok(lines) if lines.is_empty() => {
            format!("Lemonade startup log {} is empty", log_path.display())
        }
        Ok(lines) => format!(
            "Lemonade startup log tail ({}):\n{}",
            log_path.display(),
            lines.join("\n")
        ),
        Err(error) => format!(
            "failed to read Lemonade startup log {}: {error}",
            log_path.display()
        ),
    }
}

fn run_lemonade_backend_install(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    backend: &str,
    process_env: &LemonadeProcessEnvironment,
) -> Result<()> {
    let mut command = ProcessCommand::new(&manifest.lemonade);
    command
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("backends")
        .arg("install")
        .arg(format!("{LLAMACPP_RECIPE}:{backend}"))
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_lemonade_process_environment(&mut command, process_env)?;
    hide_child_console_window(&mut command);
    let status = command
        .status()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !status.success() {
        bail!("Lemonade backend install failed with status {status}");
    }
    Ok(())
}

fn run_lemonade_model_load(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    model_ref: &str,
    backend: &str,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
) -> Result<()> {
    let mut command = ProcessCommand::new(&manifest.lemonade);
    command
        .args(lemonade_model_load_args(host, port, model_ref, backend))
        .stdin(Stdio::null());
    if let Some(log_path) = log_path {
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("failed to open {}", log_path.display()))?;
        command.stdout(Stdio::from(log.try_clone()?));
        command.stderr(Stdio::from(log));
    } else {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }
    apply_lemonade_process_environment(&mut command, process_env)?;
    hide_child_console_window(&mut command);
    let status = command
        .status()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !status.success() {
        bail!("Lemonade model load failed with status {status}");
    }
    Ok(())
}

fn serve_direct_rocm_llama_server(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
    router_error: &anyhow::Error,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    let model_path = direct_llama_model_path(&paths, &request.model_ref).with_context(|| {
        format!(
            "Lemonade downloaded model `{}` was not found after its ROCm router refused to load it",
            request.model_ref
        )
    })?;
    let server = direct_llama_server_path(&runtime.manifest);
    if !server.is_file() {
        bail!(
            "Lemonade ROCm llama-server is missing at {}",
            server.display()
        );
    }

    if let Some(log_path) = log_path
        && let Some(parent) = log_path.parent()
    {
        fs::create_dir_all(parent)?;
    }
    if let Some(log_path) = log_path {
        let mut log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("failed to open {}", log_path.display()))?;
        writeln!(
            log,
            "\nLemonade router refused ROCm, launching Lemonade packaged llama-server directly: {router_error:#}"
        )
        .ok();
    }

    let mut direct_env = process_env.clone();
    if let Some(server_dir) = server.parent() {
        push_existing_path(&mut direct_env.path_entries, server_dir.to_path_buf());
        push_existing_path(&mut direct_env.library_entries, server_dir.to_path_buf());
    }

    let mut command = ProcessCommand::new(&server);
    command
        .arg("-m")
        .arg(&model_path)
        .arg("--host")
        .arg(&request.host)
        .arg("--port")
        .arg(request.port.to_string())
        .arg("--n-gpu-layers")
        .arg("999")
        .arg("--alias")
        .arg(&request.model_ref)
        .stdin(Stdio::null());
    if let Some(log_path) = log_path {
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("failed to open {}", log_path.display()))?;
        command.stdout(Stdio::from(log.try_clone()?));
        command.stderr(Stdio::from(log));
    } else {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }
    apply_lemonade_process_environment(&mut command, &direct_env)?;
    hide_child_console_window(&mut command);
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to start {}", server.display()))?;
    write_running_state(
        request,
        runtime,
        std::process::id(),
        Some(child.id()),
        "running",
    )?;
    wait_for_openai_models_ready(
        &request.host,
        request.port,
        &request.model_ref,
        Duration::from_mins(2),
    )?;
    if !query_chat_smoke_endpoint(&request.host, request.port, &request.model_ref)? {
        bail!("Lemonade packaged llama-server did not pass a chat-completion smoke test");
    }
    merge_json_state(
        &request.state_path,
        &json!({
            "status": "ready",
            "server_pid": child.id(),
            "backend_state": "ready",
            "backend_requested": ROCM_BACKEND_NAME,
            "backend_mode": "lemonade-packaged-llama-server-rocm",
            "load_response": {
                "status": "loaded",
                "method": "lemonade-packaged-llama-server",
                "model_name": request.model_ref,
                "model_path": model_path,
                "llamacpp_backend": ROCM_BACKEND_NAME
            },
        }),
    )?;
    let status = child
        .wait()
        .context("failed waiting for Lemonade packaged llama-server")?;
    mark_json_status(
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
        bail!("Lemonade packaged llama-server exited with status {status}")
    }
}

fn direct_llama_server_path(manifest: &LemonadeInstallManifest) -> PathBuf {
    manifest
        .runtime_dir
        .join("bin")
        .join("llamacpp")
        .join("rocm-stable")
        .join(platform_binary_name("llama-server"))
}

fn direct_llama_model_path(paths: &AppPaths, model_ref: &str) -> Option<PathBuf> {
    let as_path = PathBuf::from(model_ref);
    if as_path.is_file() {
        return Some(as_path);
    }
    if resolve_lemonade_model_ref(model_ref) != DEFAULT_MODEL {
        return None;
    }
    default_qwen_cache_roots(paths)
        .into_iter()
        .find_map(find_default_qwen_gguf)
}

fn default_qwen_cache_roots(paths: &AppPaths) -> Vec<PathBuf> {
    default_qwen_cache_roots_from(paths, |name| std::env::var_os(name).map(PathBuf::from))
}

fn default_qwen_cache_roots_from<F>(paths: &AppPaths, mut env_path: F) -> Vec<PathBuf>
where
    F: FnMut(&str) -> Option<PathBuf>,
{
    let mut roots = Vec::new();
    if let Some(hub_cache) = env_path("HUGGINGFACE_HUB_CACHE") {
        push_qwen_cache_root(&mut roots, hub_cache);
    }
    if let Some(hf_home) = env_path("HF_HOME") {
        push_qwen_cache_root(&mut roots, hf_home.join("hub"));
    }
    push_qwen_cache_root(&mut roots, paths.cache_dir.join("huggingface").join("hub"));
    if let Some(home) = env_path("HOME") {
        push_qwen_cache_root(
            &mut roots,
            home.join(".cache").join("huggingface").join("hub"),
        );
    }
    roots
}

fn push_qwen_cache_root(roots: &mut Vec<PathBuf>, path: PathBuf) {
    if path.as_os_str().is_empty() || roots.iter().any(|existing| existing == &path) {
        return;
    }
    roots.push(path);
}

fn find_default_qwen_gguf(cache_root: PathBuf) -> Option<PathBuf> {
    let snapshots = cache_root.join(DEFAULT_MODEL_REPO_DIR).join("snapshots");
    let entries = fs::read_dir(snapshots).ok()?;
    entries
        .flatten()
        .map(|entry| entry.path().join(DEFAULT_MODEL_GGUF))
        .find(|path| path.is_file())
}

fn wait_for_openai_models_ready(
    host: &str,
    port: u16,
    model_ref: &str,
    timeout: Duration,
) -> Result<()> {
    let endpoint = format_http_base_url(host, port);
    let start = std::time::Instant::now();
    let mut last_error = None;
    while start.elapsed() < timeout {
        match http_get_text(&endpoint, "/v1/models", Duration::from_secs(3))
            .and_then(|body| parse_models_ready(&body, model_ref))
        {
            Ok(true) => return Ok(()),
            Ok(false) => last_error = Some("model was not reported by /v1/models".to_owned()),
            Err(error) => last_error = Some(error.to_string()),
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    bail!(
        "Lemonade packaged llama-server did not become ready: {}",
        last_error.unwrap_or_else(|| "not checked".to_owned())
    )
}

fn parse_models_ready(body: &str, model_ref: &str) -> Result<bool> {
    let value = serde_json::from_str::<Value>(body.trim())
        .context("failed to parse /v1/models response")?;
    Ok(models_payload_has_loaded_model(&value, model_ref))
}

fn models_payload_has_loaded_model(value: &Value, model_ref: &str) -> bool {
    value
        .get("data")
        .or_else(|| value.get("models"))
        .and_then(Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                ["id", "model", "name"]
                    .into_iter()
                    .filter_map(|field| model.get(field).and_then(Value::as_str))
                    .any(|loaded| model_names_match(loaded, model_ref))
            })
        })
}

fn lemonade_model_load_args(host: &str, port: u16, model_ref: &str, backend: &str) -> Vec<String> {
    vec![
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "load".to_owned(),
        model_ref.to_owned(),
        "--llamacpp".to_owned(),
        backend.to_owned(),
        "--save-options".to_owned(),
    ]
}

fn query_health_json(host: &str, port: u16) -> Result<Value> {
    let endpoint = format_http_base_url(host, port);
    let body = http_get_text(&endpoint, "/v1/health", Duration::from_secs(3))
        .with_context(|| format!("failed to query Lemonade health at {endpoint}/v1/health"))?;
    serde_json::from_str(&body).context("failed to parse Lemonade health JSON")
}

fn query_loaded_model_endpoint(endpoint_url: &str, model_ref: &str, backend: &str) -> Result<bool> {
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    if let Ok(health) = query_health_json(&host, port) {
        Ok(health_has_loaded_model(&health, model_ref, backend))
    } else {
        let endpoint = format_http_base_url(&host, port);
        let body = http_get_text(&endpoint, "/v1/models", Duration::from_secs(3))
            .with_context(|| format!("failed to query Lemonade models at {endpoint}/v1/models"))?;
        let models = serde_json::from_str::<Value>(&body)
            .context("failed to parse Lemonade /v1/models JSON")?;
        Ok(models_payload_has_loaded_model(&models, model_ref))
    }
}

fn query_chat_smoke_endpoint(host: &str, port: u16, model_ref: &str) -> Result<bool> {
    let addr = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve {host}:{port}"))?
        .next()
        .with_context(|| format!("no socket addresses resolved for {host}:{port}"))?;
    let timeout = Duration::from_secs(8);
    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .with_context(|| format!("failed to connect to {host}:{port}"))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    let body = json!({
        "model": model_ref,
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": 2,
        "stream": false
    });
    let body = serde_json::to_string(&body).context("failed to serialize chat smoke request")?;
    let host_header = format_host_port(host, port);
    write!(
        stream,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
    .context("failed to write chat smoke request")?;
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .context("failed to read chat smoke response")?;
    let status_line = response.lines().next().unwrap_or_default();
    Ok(status_line.contains(" 200 "))
}

fn health_has_loaded_model(health: &Value, model_ref: &str, backend: &str) -> bool {
    let model_ref = model_ref.trim();
    if model_ref.is_empty() {
        return false;
    }
    health
        .get("all_models_loaded")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|model| {
            let name_matches = ["model_name", "id", "name"]
                .into_iter()
                .filter_map(|field| model.get(field).and_then(Value::as_str))
                .any(|loaded| model_names_match(loaded, model_ref));
            let backend_matches = model
                .get("recipe_options")
                .and_then(|options| options.get("llamacpp_backend"))
                .and_then(Value::as_str)
                .is_some_and(|loaded| lemonade_backend_matches(loaded, backend));
            name_matches && backend_matches
        })
}

fn model_names_match(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
        || resolve_lemonade_model_ref(left).eq_ignore_ascii_case(&resolve_lemonade_model_ref(right))
}

fn lemonade_backend_matches(value: &str, backend: &str) -> bool {
    value
        .trim()
        .to_ascii_lowercase()
        .starts_with(&backend.to_ascii_lowercase())
}

fn serve_http_command_args(request: &ServeHttpRequest) -> Vec<String> {
    let mut args = vec![
        "serve-http".to_owned(),
        request.service_id.clone(),
        request.model_ref.clone(),
        "--host".to_owned(),
        request.host.clone(),
        "--port".to_owned(),
        request.port.to_string(),
        "--device-policy".to_owned(),
        device_policy_name(&request.device_policy).to_owned(),
        "--state-path".to_owned(),
        request.state_path.display().to_string(),
    ];
    if let Some(runtime_id) = request.runtime_id.as_deref() {
        args.extend(["--runtime-id".to_owned(), runtime_id.to_owned()]);
    }
    if let Some(env_id) = request.env_id.as_deref() {
        args.extend(["--env-id".to_owned(), env_id.to_owned()]);
    }
    if let Some(log_path) = request.log_path.as_ref() {
        args.extend(["--log-path".to_owned(), log_path.display().to_string()]);
    }
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(&request.gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    if let Some(engine_recipe) = request.engine_recipe.as_ref() {
        args.extend([
            "--engine-recipe-json".to_owned(),
            serde_json::to_string(engine_recipe).expect("engine recipe serializes"),
        ]);
    }
    args
}

#[cfg(windows)]
fn spawn_serve_http_background(current_exe: &Path, serve_args: &[String]) -> Result<u32> {
    rocm_core::spawn_detached_no_inherit(current_exe, serve_args, &[])
        .context("failed to launch Lemonade serve-http background process")
}

#[cfg(not(windows))]
fn spawn_serve_http_background(current_exe: &Path, serve_args: &[String]) -> Result<u32> {
    let child = ProcessCommand::new(current_exe)
        .args(serve_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to launch Lemonade serve-http background process")?;
    Ok(child.id())
}

fn write_running_state(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    pid: u32,
    server_pid: Option<u32>,
    status: &str,
) -> Result<()> {
    write_state(
        &request.state_path,
        &json!({
            "service_id": request.service_id,
            "engine": ENGINE_NAME,
            "status": status,
            "pid": pid,
            "server_pid": server_pid,
            "model_ref": request.model_ref,
            "canonical_model_id": request.model_ref,
            "host": request.host,
            "port": request.port,
            "endpoint_url": endpoint_url(&request.host, request.port),
            "device_policy": device_policy_name(&request.device_policy),
            "runtime_id": request.runtime_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "env_id": request.env_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "runtime_kind": "lemonade_embeddable",
            "runtime_executable": runtime.manifest.lemond,
            "log_path": request.log_path.as_ref().map(|path| path.display().to_string()),
            "engine_recipe": request.engine_recipe,
            "started_at_unix_ms": current_unix_millis()
        }),
    )
}

fn service_files(service_id: &str) -> Result<ServiceFiles> {
    let paths = AppPaths::discover()?;
    Ok(ServiceFiles {
        state_path: paths
            .engine_state_dir(ENGINE_NAME)
            .join(format!("{service_id}.json")),
        log_path: paths
            .engine_logs_dir(ENGINE_NAME)
            .join(format!("{service_id}.log")),
    })
}

fn endpoint_url(host: &str, port: u16) -> String {
    format!("{}/v1", format_http_base_url(host, port))
}

fn endpoint_url_from_state(state: &Value) -> Option<String> {
    value_string(state, "endpoint_url").or_else(|| {
        let host = value_string(state, "host")?;
        let port = value_u32(state, "port")?;
        let port = u16::try_from(port).ok()?;
        Some(endpoint_url(&host, port))
    })
}

fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_state(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn merge_json_state(path: &Path, patch: &Value) -> Result<()> {
    let mut value = read_service_state(path).unwrap_or_else(|_| json!({}));
    if !value.is_object() {
        value = json!({});
    }
    let object = value.as_object_mut().expect("object checked above");
    if let Some(patch) = patch.as_object() {
        for (key, value) in patch {
            object.insert(key.clone(), value.clone());
        }
    }
    write_state(path, &value)
}

fn mark_json_status(path: &Path, status: &str) -> Result<()> {
    merge_json_state(
        path,
        &json!({
            "engine": ENGINE_NAME,
            "status": status,
            "stopped_at_unix_ms": current_unix_millis(),
        }),
    )
}

fn value_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
}

fn value_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn pid_to_terminate_from_state(state: &Value) -> Option<u32> {
    value_u32(state, "server_pid").or_else(|| value_u32(state, "pid"))
}

fn tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?;
    let file_size = metadata.len();

    // For small files, read normally to preserve exact line count.
    if file_size <= MAX_TAIL_READ {
        let reader = std::io::BufReader::new(file);
        let mut lines = VecDeque::with_capacity(limit);
        for line in reader.lines() {
            let line = line.with_context(|| format!("failed to read {}", path.display()))?;
            if lines.len() == limit {
                lines.pop_front();
            }
            lines.push_back(line);
        }
        return Ok(lines.into_iter().collect());
    }

    // For large files, seek near the end and read only the final chunk.
    // This prevents reading multi-gigabyte logs during timeout errors.
    let mut file = file;
    let seek_pos = file_size.saturating_sub(MAX_TAIL_READ);
    file.seek(std::io::SeekFrom::Start(seek_pos))
        .with_context(|| format!("failed to seek in {}", path.display()))?;

    let reader = std::io::BufReader::new(file);
    let mut lines = VecDeque::with_capacity(limit);
    let mut skipped_first = seek_pos == 0;
    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read {}", path.display()))?;
        // Skip the first line after seeking, as it may be partial.
        if !skipped_first {
            skipped_first = true;
            continue;
        }
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(line);
    }
    Ok(lines.into_iter().collect())
}

fn terminate_pid(pid: u32, _force: bool) -> bool {
    rocm_core::terminate_process(pid).is_ok()
}

fn free_local_port() -> Result<u16> {
    let listener = TcpListener::bind((DEFAULT_HOST, 0)).context("failed to reserve local port")?;
    Ok(listener.local_addr()?.port())
}

fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
    let without_scheme = endpoint_url.trim().strip_prefix("http://")?;
    let authority = without_scheme.split('/').next()?.trim();
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']')?;
        let host = rest[..end].to_owned();
        let port = rest[end + 1..].strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    Some((host.to_owned(), port.parse().ok()?))
}

fn normalize_device_policy(policy: Option<DevicePolicy>) -> Result<DevicePolicy> {
    match policy.unwrap_or(DevicePolicy::GpuRequired) {
        DevicePolicy::GpuRequired | DevicePolicy::GpuPreferred => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::CpuOnly => {
            bail!("Lemonade adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

fn require_gpu_required(policy: &DevicePolicy) -> Result<()> {
    match policy {
        DevicePolicy::GpuRequired | DevicePolicy::GpuPreferred => Ok(()),
        DevicePolicy::CpuOnly => {
            bail!("Lemonade adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

fn parse_device_policy_arg(value: Option<&str>) -> Result<DevicePolicy> {
    match value.unwrap_or("gpu_required") {
        "gpu" | "gpu_required" | "gpu_preferred" => Ok(DevicePolicy::GpuRequired),
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        other => bail!("unknown device policy `{other}`"),
    }
}

/// Parse a `--gpu` CLI value into an optional `GpuSelection` for `LaunchRequest`.
fn parse_gpu_selection_arg(value: Option<&str>) -> Result<Option<GpuSelection>> {
    value
        .map(|raw| GpuSelection::parse_cli_value(raw).map_err(anyhow::Error::msg))
        .transpose()
}

/// Parse a `--gpu` CLI value into explicit device ordinals (empty for `auto`).
fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    Ok(rocm_engine_protocol::launch_gpu_indices(
        parse_gpu_selection_arg(value)?.as_ref(),
    ))
}

const fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
}

fn accepted_engine_recipe(
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<Option<EngineRecipeHint>> {
    if let Some(hint) = &engine_recipe {
        if hint.engine != ENGINE_NAME {
            bail!(
                "engine_recipe target `{}` does not match adapter `{}`",
                hint.engine,
                ENGINE_NAME
            );
        }
        if hint.contract_version != ENGINE_RECIPE_CONTRACT_VERSION {
            bail!(
                "engine_recipe contract `{}` is unsupported; expected `{}`",
                hint.contract_version,
                ENGINE_RECIPE_CONTRACT_VERSION
            );
        }
    }
    Ok(engine_recipe)
}

fn parse_engine_recipe_json(value: Option<String>) -> Result<Option<EngineRecipeHint>> {
    value
        .map(|text| {
            serde_json::from_str::<EngineRecipeHint>(&text)
                .context("failed to parse engine recipe JSON")
        })
        .transpose()
        .and_then(accepted_engine_recipe)
}

fn resolve_lemonade_model_ref(model_ref: &str) -> String {
    let trimmed = model_ref.trim();
    let lower = trimmed.to_ascii_lowercase();
    if trimmed.is_empty()
        || matches!(
            lower.as_str(),
            "qwen"
                | "assistant"
                | "default"
                | "small"
                | "lemonade-qwen"
                | "qwen-gguf"
                | "qwen3-4b"
                | "qwen3-4b-instruct"
                | "qwen3-4b-instruct-2507-gguf"
        )
        || lower.contains("qwen2.5-1.5b")
        || lower.contains("qwen3.5-0.8b")
    {
        DEFAULT_MODEL.to_owned()
    } else if matches!(
        lower.as_str(),
        "tiny" | "qwen-smoke" | "lemonade-tiny" | "qwen3-0.6b-gguf"
    ) {
        "Qwen3-0.6B-GGUF".to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn manifest_lock_hash(manifest: &LemonadeInstallManifest) -> String {
    let mut hasher = DefaultHasher::new();
    manifest.env_id.hash(&mut hasher);
    manifest.version.hash(&mut hasher);
    manifest.runtime_dir.hash(&mut hasher);
    manifest.lemond.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn current_unix_millis() -> u128 {
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
    fn qwen_alias_resolves_to_validated_assistant_gguf_model() {
        assert_eq!(resolve_lemonade_model_ref("qwen"), DEFAULT_MODEL);
        assert_eq!(
            resolve_lemonade_model_ref("Qwen/Qwen2.5-1.5B-Instruct"),
            DEFAULT_MODEL
        );
        assert_eq!(resolve_lemonade_model_ref("qwen-smoke"), "Qwen3-0.6B-GGUF");
    }

    #[test]
    fn embeddable_package_matches_runtime_os() {
        if runtime_is_windows() {
            assert!(embeddable_archive_name().ends_with("windows-x64.zip"));
            assert!(embeddable_url().ends_with("windows-x64.zip"));
            assert_eq!(platform_binary_name("lemond"), "lemond.exe");
        } else {
            assert!(embeddable_archive_name().ends_with("ubuntu-x64.tar.gz"));
            assert!(embeddable_url().ends_with("ubuntu-x64.tar.gz"));
            assert_eq!(platform_binary_name("lemond"), "lemond");
        }
    }

    #[test]
    fn lemonade_root_uses_requested_engine_root() {
        let paths = AppPaths {
            config_dir: PathBuf::from("C:/Users/test/.rocm"),
            data_dir: PathBuf::from("C:/Users/test/.rocm"),
            cache_dir: PathBuf::from("C:/Users/test/.rocm/cache"),
        };
        let engine_root = PathBuf::from("D:/path/to/therock_venvs/engines");

        assert_eq!(
            lemonade_root(&paths, Some(&engine_root)),
            normalize_runtime_path_for_host(&engine_root).join(ENGINE_NAME)
        );
    }

    #[test]
    fn windows_child_path_maps_ape_drive_paths() {
        assert_eq!(
            windows_child_path(Path::new("/D/path/to/rocm-cli/file.zip")),
            r"D:\path\to\rocm-cli\file.zip"
        );
        assert_eq!(windows_child_path(Path::new("/c")), r"C:\");
    }

    #[test]
    fn lemonade_model_load_uses_selected_backend() {
        let args = lemonade_model_load_args("127.0.0.1", 11435, DEFAULT_MODEL, "vulkan");
        assert_eq!(
            args,
            vec![
                "--host",
                "127.0.0.1",
                "--port",
                "11435",
                "load",
                DEFAULT_MODEL,
                "--llamacpp",
                "vulkan",
                "--save-options",
            ]
        );
    }

    #[test]
    fn parses_llamacpp_backends_from_table() {
        let output = "\
Recipe              Backend     Status          Message/Version                               Action
----------------------------------------------------------------------------------------------------
kokoro              cpu         installable     Backend is supported but not installed.       lemonade backends install kokoro:cpu
                    metal       unsupported     Requires macOS                                -
llamacpp            cpu         installable     Backend is supported but not installed.       lemonade backends install llamacpp:cpu
                    metal       unsupported     Requires macOS                                -
                    rocm        unsupported     Unsupported GPU                               -
                    system      unsupported     Requires Linux                               -
                    vulkan      installable     Backend is supported but not installed.       lemonade backends install llamacpp:vulkan
vllm                rocm        unsupported     Requires Linux                               -
";
        let backends = parse_llamacpp_backend_statuses(output);
        assert_eq!(
            backends,
            vec![
                ("cpu".to_owned(), "installable".to_owned()),
                ("metal".to_owned(), "unsupported".to_owned()),
                ("rocm".to_owned(), "unsupported".to_owned()),
                ("system".to_owned(), "unsupported".to_owned()),
                ("vulkan".to_owned(), "installable".to_owned()),
            ]
        );
    }

    #[test]
    fn selects_vulkan_when_rocm_unsupported() {
        let backends = vec![
            ("cpu".to_owned(), "installable".to_owned()),
            ("rocm".to_owned(), "unsupported".to_owned()),
            ("vulkan".to_owned(), "installable".to_owned()),
        ];
        assert_eq!(
            select_best_llamacpp_backend(&backends),
            Some(("vulkan".to_owned(), false))
        );
    }

    #[test]
    fn prefers_installed_rocm_when_supported() {
        let backends = vec![
            ("rocm".to_owned(), "installed".to_owned()),
            ("vulkan".to_owned(), "installable".to_owned()),
        ];
        assert_eq!(
            select_best_llamacpp_backend(&backends),
            Some(("rocm".to_owned(), true))
        );
    }

    #[test]
    fn selects_cpu_when_no_gpu_backend() {
        let backends = vec![
            ("rocm".to_owned(), "unsupported".to_owned()),
            ("cpu".to_owned(), "installable".to_owned()),
        ];
        assert_eq!(
            select_best_llamacpp_backend(&backends),
            Some(("cpu".to_owned(), false))
        );
    }

    #[test]
    fn selects_nothing_when_all_unsupported() {
        let backends = vec![("rocm".to_owned(), "unsupported".to_owned())];
        assert_eq!(select_best_llamacpp_backend(&backends), None);
    }

    #[test]
    fn direct_qwen_lookup_checks_huggingface_cache_env() {
        let paths = AppPaths {
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("rocm-cache"),
        };
        let roots = default_qwen_cache_roots_from(&paths, |name| match name {
            "HUGGINGFACE_HUB_CACHE" => Some(PathBuf::from("hf-hub")),
            "HF_HOME" => Some(PathBuf::from("hf-home")),
            "HOME" => Some(PathBuf::from("home")),
            _ => None,
        });

        assert_eq!(
            roots,
            vec![
                PathBuf::from("hf-hub"),
                PathBuf::from("hf-home").join("hub"),
                PathBuf::from("rocm-cache").join("huggingface").join("hub"),
                PathBuf::from("home")
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            ]
        );
    }

    #[test]
    fn device_policy_rejects_cpu_without_fallback() {
        let error = normalize_device_policy(Some(DevicePolicy::CpuOnly))
            .expect_err("cpu should be rejected")
            .to_string();
        assert!(error.contains("no CPU fallback"));
    }

    #[test]
    fn endpoint_parser_supports_ipv6_loopback() {
        assert_eq!(
            parse_http_endpoint("http://[::1]:11435/v1"),
            Some(("::1".to_owned(), 11435))
        );
    }

    #[test]
    fn serve_http_args_preserve_runtime_selection() {
        let request = ServeHttpRequest {
            service_id: "svc".to_owned(),
            model_ref: DEFAULT_MODEL.to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: DevicePolicy::GpuRequired,
            gpu_indices: Vec::new(),
            runtime_id: Some("runtime".to_owned()),
            env_id: Some("env".to_owned()),
            state_path: PathBuf::from("state.json"),
            log_path: Some(PathBuf::from("service.log")),
            engine_recipe: None,
        };
        let args = serve_http_command_args(&request);
        assert!(args.contains(&"--runtime-id".to_owned()));
        assert!(args.contains(&"runtime".to_owned()));
        assert!(args.contains(&"--env-id".to_owned()));
        assert!(args.contains(&"env".to_owned()));
        assert!(!args.iter().any(|arg| arg == "cpu"));
    }

    #[test]
    fn health_parser_requires_loaded_requested_model() {
        let unloaded = json!({
            "status": "ok",
            "model_loaded": null,
            "all_models_loaded": []
        });
        assert!(!health_has_loaded_model(&unloaded, DEFAULT_MODEL, "vulkan"));

        let loaded = json!({
            "status": "ok",
            "model_loaded": DEFAULT_MODEL,
            "all_models_loaded": [{
                "model_name": DEFAULT_MODEL,
                "recipe": "llamacpp",
                "recipe_options": {
                    "llamacpp_backend": "vulkan"
                }
            }]
        });
        assert!(health_has_loaded_model(&loaded, "lemonade-qwen", "vulkan"));
        assert!(!health_has_loaded_model(&loaded, "lemonade-qwen", "rocm"));
    }
}
