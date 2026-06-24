// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_PORT, format_http_base_url, openai_models_endpoint_has_model,
    require_nonempty,
};
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, DevicePolicy, ENGINE_RECIPE_CONTRACT_VERSION, EndpointRequest,
    EndpointResponse, EngineCapabilities, EngineDeviceAvailability, EngineMethod, EngineRecipeHint,
    EngineRequestEnvelope, EngineResponseEnvelope, HealthcheckRequest, HealthcheckResponse,
    InstallRequest, InstallResponse, LaunchRequest, LaunchResponse, LogsRequest, LogsResponse,
    ResolveModelRequest, ResolveModelResponse, StopRequest, StopResponse,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{VecDeque, hash_map::DefaultHasher};
use std::ffi::OsString;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ENGINE_NAME: &str = "llama.cpp";
const DEFAULT_HOST: &str = "127.0.0.1";
const HEALTHCHECK_TIMEOUT_MS: u64 = 700;
const DEFAULT_LOG_TAIL_LINES: usize = 200;
const LLAMA_GPU_LAYERS_VALUE: &str = "-1";
const REQUIRED_WINDOWS_THEROCK_EXACT_DLLS: &[&str] = &[
    "amdhip64_7.dll",
    "hipblas.dll",
    "libhipblaslt.dll",
    "rocblas.dll",
    "rocm_kpack.dll",
    "rocsolver.dll",
];
const REQUIRED_WINDOWS_THEROCK_VERSIONED_DLLS: &[(&str, &str)] = &[
    ("amd_comgr", ".dll"),
    ("hiprtc-builtins", ".dll"),
    ("hiprtc", ".dll"),
];
const REQUIRED_WINDOWS_THEROCK_DATA_DIRS: &[&[&str]] =
    &[&["rocblas", "library"], &["hipblaslt", "library"]];

#[derive(Parser)]
#[command(name = "rocm-engine-llama.cpp")]
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
    },
}

#[derive(Debug, Clone)]
struct LlamaServer {
    program: String,
    display: String,
}

#[derive(Debug, Clone)]
struct PreparedLlamaServer {
    program: PathBuf,
    display: String,
    staged_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct ServiceFiles {
    state_path: PathBuf,
    log_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ServeHttpRequest {
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: Option<String>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
}

#[derive(Debug, Clone)]
struct TheRockHipRuntimeEnv {
    runtime_id: String,
    runtime_key: Option<String>,
    root_path: PathBuf,
    bin_path: PathBuf,
    bin_paths: Vec<PathBuf>,
    library_paths: Vec<PathBuf>,
    source: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TheRockRuntimeManifest {
    #[serde(default)]
    runtime_key: Option<String>,
    #[serde(default)]
    runtime_id: Option<String>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    rocm_sdk: Option<RocmSdkRuntimeProbe>,
    #[serde(default)]
    installed_at_unix_ms: Option<u128>,
}

#[derive(Debug, Clone, Deserialize)]
struct RocmSdkRuntimeProbe {
    #[serde(default)]
    import_ok: bool,
    #[serde(default)]
    root_path: Option<PathBuf>,
    #[serde(default)]
    bin_path: Option<PathBuf>,
    #[serde(default)]
    bin_paths: Vec<PathBuf>,
    #[serde(default)]
    library_paths: Vec<PathBuf>,
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
        } => serve_http(ServeHttpRequest {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
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
    device_policy: Option<String>,
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
    T: DeserializeOwned,
    F: FnOnce(T) -> Result<U>,
    U: Serialize,
{
    match serde_json::from_value::<T>(payload) {
        Ok(request) => match handler(request) {
            Ok(response) => EngineResponseEnvelope::success(response),
            Err(error) => EngineResponseEnvelope::failure("request_failed", error.to_string()),
        },
        Err(error) => EngineResponseEnvelope::failure("invalid_payload", error.to_string()),
    }
}

fn detect_response() -> DetectResponse {
    let server = resolve_llama_server().ok();
    let runtime_executable = server.as_ref().map(|server| server.program.clone());
    let runtime_env = resolve_therock_hip_runtime_env(None);
    let rocm_gpu_available = server.as_ref().is_some_and(llama_server_has_hip_backend)
        && matches!(runtime_env.as_ref(), Ok(Some(_)));
    let mut notes = Vec::new();
    match server.as_ref() {
        Some(server) => notes.push(format!("llama-server detected: {}", server.display)),
        None => notes.push(
            "llama-server not found; set ROCM_CLI_LLAMA_CPP_SERVER or install llama.cpp".to_owned(),
        ),
    }
    match runtime_env.as_ref() {
        Ok(Some(runtime_env)) => notes.push(format!(
            "TheRock HIP runtime env available: root={} bin={} source={}",
            runtime_env.root_path.display(),
            runtime_env.bin_path.display(),
            runtime_env.source
        )),
        Ok(None) => notes.push(
            "no managed TheRock HIP runtime manifest with rocm_sdk root/bin was found; install a rocm-cli managed SDK before GPU llama.cpp serving".to_owned(),
        ),
        Err(error) => notes.push(format!("TheRock HIP runtime env probe failed: {error}")),
    }
    DetectResponse {
        installed: server.is_some(),
        env_id: server.as_ref().map(|_| "external-llama.cpp".to_owned()),
        runtime_kind: Some("external_llama_server".to_owned()),
        runtime_executable,
        managed_env: Some(false),
        python_version: None,
        torch_version: None,
        transformers_version: None,
        available_devices: vec![
            EngineDeviceAvailability {
                kind: "cpu".to_owned(),
                available: false,
                reason: Some(
                    "rocm-cli does not offer llama.cpp CPU serving; a managed TheRock HIP runtime is required"
                        .to_owned(),
                ),
            },
            EngineDeviceAvailability {
                kind: "rocm_gpu".to_owned(),
                available: rocm_gpu_available,
                reason: Some(rocm_gpu_reason(server.as_ref(), runtime_env.as_ref().ok())),
            },
        ],
        capabilities: engine_capabilities(rocm_gpu_available),
        notes,
    }
}

fn capabilities() -> EngineCapabilities {
    engine_capabilities(false)
}

fn engine_capabilities(rocm_gpu: bool) -> EngineCapabilities {
    EngineCapabilities {
        cpu: false,
        rocm_gpu,
        multi_gpu: rocm_gpu,
        openai_compatible: true,
        tool_calling: false,
        quantized_models: "gguf".to_owned(),
        distributed_serving: false,
        reasoning_parser: false,
    }
}

fn rocm_gpu_reason(
    server: Option<&LlamaServer>,
    runtime_env: Option<&Option<TheRockHipRuntimeEnv>>,
) -> String {
    let Some(server) = server else {
        return "llama-server was not found".to_owned();
    };
    if !llama_server_has_hip_backend(server) {
        return "llama-server was found, but no sibling ggml-hip backend library was detected"
            .to_owned();
    }
    match runtime_env {
        Some(Some(runtime_env)) => format!(
            "llama-server has a HIP backend and managed TheRock SDK paths are available from {}",
            runtime_env.source
        ),
        _ => "llama-server has a HIP backend, but no managed TheRock runtime manifest with rocm_sdk root/bin was found".to_owned(),
    }
}

fn llama_server_has_hip_backend(server: &LlamaServer) -> bool {
    let Ok(program) = resolve_llama_server_program_path(server) else {
        return false;
    };
    let Some(dir) = program.parent() else {
        return false;
    };
    if cfg!(windows) {
        dir.join("ggml-hip.dll").is_file()
    } else if cfg!(target_os = "macos") {
        dir.join("libggml-hip.dylib").is_file()
    } else {
        dir.join("libggml-hip.so").is_file() || dir.join("ggml-hip.so").is_file()
    }
}

fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    let server = resolve_llama_server().ok();
    let runtime_env = resolve_therock_hip_runtime_env(Some(&request.runtime_id))?;
    let rocm_gpu_available =
        server.as_ref().is_some_and(llama_server_has_hip_backend) && runtime_env.is_some();
    let paths = AppPaths::discover()?;
    let runtime_executable = server.as_ref().map(|server| server.program.clone());
    let mut installed_packages = server
        .as_ref()
        .map(|server| vec![format!("llama-server={}", server.display)])
        .unwrap_or_default();
    if let Some(runtime_env) = runtime_env.as_ref() {
        installed_packages.push(format!(
            "therock-hip-runtime-root={}",
            runtime_env.root_path.display()
        ));
        installed_packages.push(format!(
            "therock-hip-runtime-bin={}",
            runtime_env.bin_path.display()
        ));
    }
    let mut warnings = Vec::new();
    if server.is_some() {
        warnings.push(
            "llama.cpp adapter uses an external llama-server binary; rocm-cli does not build llama.cpp yet".to_owned(),
        );
    } else {
        warnings.push(
            "llama-server was not found; set ROCM_CLI_LLAMA_CPP_SERVER to a llama-server executable".to_owned(),
        );
    }
    match runtime_env.as_ref() {
        Some(runtime_env) => warnings.push(format!(
            "external HIP binaries will inherit TheRock SDK paths from managed runtime manifest {}",
            runtime_env.root_path.display()
        )),
        None => warnings.push(
            "no managed TheRock SDK root/bin was resolved; GPU llama.cpp serving will fail without a rocm-cli managed runtime manifest".to_owned(),
        ),
    }
    Ok(InstallResponse {
        env_id: "external-llama.cpp".to_owned(),
        env_path: paths.engine_dir(ENGINE_NAME).display().to_string(),
        python_executable: server.as_ref().map_or_else(
            || "<external llama-server not found>".to_owned(),
            |server| server.display.clone(),
        ),
        runtime_kind: Some("external_llama_server".to_owned()),
        runtime_executable,
        managed_env: Some(false),
        installed_packages,
        capabilities: engine_capabilities(rocm_gpu_available),
        lock_hash: "external".to_owned(),
        warnings,
    })
}

fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    require_nonempty(&request.model_ref, "model_ref")?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let mut warnings = Vec::new();
    let device_policy = normalize_llama_device_policy(request.device_policy)?;
    let canonical_model_id = resolve_llama_model_ref(&request.model_ref, &mut warnings);
    Ok(ResolveModelResponse {
        canonical_model_id,
        task: "chat-completions".to_owned(),
        source: "llama.cpp".to_owned(),
        revision: "local".to_owned(),
        loader: "llama.cpp".to_owned(),
        trust_remote_code: false,
        chat_template_mode: "llama.cpp".to_owned(),
        dtype: "gguf".to_owned(),
        device_policy,
        estimated_memory: "depends on GGUF quantization and context size".to_owned(),
        launch_defaults: json!({
            "host": DEFAULT_HOST,
            "port": DEFAULT_LOCAL_PORT,
            "endpoint_mode": "openai"
        }),
        engine_recipe,
        warnings,
    })
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

fn normalize_llama_device_policy(policy: Option<DevicePolicy>) -> Result<DevicePolicy> {
    match policy.unwrap_or(DevicePolicy::GpuRequired) {
        DevicePolicy::GpuRequired => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::GpuPreferred => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::CpuOnly => {
            bail!("llama.cpp adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

fn resolve_llama_model_ref(model_ref: &str, warnings: &mut Vec<String>) -> String {
    let trimmed = model_ref.trim();
    if !trimmed.to_ascii_lowercase().ends_with(".gguf") {
        warnings.push(
            "llama.cpp adapter expects a local GGUF model path or a llama-server-compatible model reference".to_owned(),
        );
        return trimmed.to_owned();
    }

    let expanded = expand_home_path(trimmed);
    let mut candidates = Vec::new();
    if expanded.is_absolute() {
        candidates.push(expanded);
    } else {
        if let Ok(current_dir) = std::env::current_dir() {
            candidates.push(current_dir.join(&expanded));
        }
        if let Ok(paths) = AppPaths::discover() {
            candidates.push(paths.data_dir.join("models").join(&expanded));
        }
    }

    for candidate in candidates {
        if candidate.is_file() {
            return candidate
                .canonicalize()
                .unwrap_or(candidate)
                .display()
                .to_string();
        }
    }

    warnings.push(format!(
        "GGUF model file `{trimmed}` was not found locally; llama-server may still resolve it if it supports this reference"
    ));
    trimmed.to_owned()
}

fn expand_home_path(value: &str) -> PathBuf {
    let Some(rest) = value
        .strip_prefix("~/")
        .or_else(|| value.strip_prefix("~\\"))
    else {
        return PathBuf::from(value);
    };
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map_or_else(
            || PathBuf::from(value),
            |home| PathBuf::from(home).join(rest),
        )
}

fn launch_service(mut request: LaunchRequest) -> Result<LaunchResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    require_nonempty(&request.model_ref, "model_ref")?;
    let warnings = Vec::<String>::new();
    let device_policy = normalize_llama_device_policy(request.device_policy.clone())?;
    require_managed_therock_hip_runtime_env(
        resolve_therock_hip_runtime_env(request.runtime_id.as_deref())?,
        request.runtime_id.as_deref(),
        device_policy_name(&device_policy),
    )?;
    request.device_policy = Some(device_policy);
    request.engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
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
    let current_exe =
        std::env::current_exe().context("failed to discover current engine binary")?;
    let serve_args = serve_http_command_args(&request, &state_path, Some(&log_path));
    let endpoint_url = format!("{}/v1", format_http_base_url(&request.host, request.port));
    write_state(
        &state_path,
        &json!({
            "engine": ENGINE_NAME,
            "service_id": request.service_id,
            "model_ref": request.model_ref,
            "host": request.host,
            "port": request.port,
            "status": "starting",
            "endpoint_url": endpoint_url,
            "log_path": log_path.display().to_string(),
            "device_policy": request.device_policy.as_ref().map(device_policy_name),
            "runtime_id": request.runtime_id,
            "env_id": request.env_id,
            "warnings": warnings,
        }),
    )?;
    let wrapper_pid = spawn_serve_http_background(&current_exe, &serve_args, &state_path)
        .context("failed to spawn llama.cpp serve-http process")?;

    merge_json_state(
        &state_path,
        &json!({
            "pid": wrapper_pid,
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

fn serve_http_command_args(
    request: &LaunchRequest,
    state_path: &Path,
    log_path: Option<&Path>,
) -> Vec<String> {
    let mut args = vec![
        "serve-http".to_owned(),
        request.service_id.clone(),
        request.model_ref.clone(),
        "--host".to_owned(),
        request.host.clone(),
        "--port".to_owned(),
        request.port.to_string(),
    ];
    if let Some(device_policy) = request.device_policy.as_ref() {
        args.push("--device-policy".to_owned());
        args.push(device_policy_name(device_policy).to_owned());
    }
    if let Some(runtime_id) = request.runtime_id.as_deref() {
        args.push("--runtime-id".to_owned());
        args.push(runtime_id.to_owned());
    }
    if let Some(env_id) = request.env_id.as_deref() {
        args.push("--env-id".to_owned());
        args.push(env_id.to_owned());
    }
    args.push("--state-path".to_owned());
    args.push(state_path.display().to_string());
    if let Some(log_path) = log_path {
        args.push("--log-path".to_owned());
        args.push(log_path.display().to_string());
    }
    if let Some(engine_recipe) = request.engine_recipe.as_ref() {
        args.push("--engine-recipe-json".to_owned());
        args.push(serde_json::to_string(engine_recipe).expect("engine recipe serializes"));
    }
    args
}

fn normalize_serve_http_request(request: ServeHttpRequest) -> Result<ServeHttpRequest> {
    match request.device_policy.as_deref() {
        Some("gpu_required" | "gpu_preferred") | None => {}
        Some("cpu" | "cpu_only") => {
            bail!("llama.cpp adapter requires ROCm GPU execution; no CPU fallback is used")
        }
        Some(other) => bail!("unknown llama.cpp device policy `{other}`"),
    }
    Ok(request)
}

fn serve_http(request: ServeHttpRequest) -> Result<()> {
    let request = normalize_serve_http_request(request)?;
    let server = resolve_llama_server()?;
    let paths = AppPaths::discover()?;
    let runtime_env = resolve_therock_hip_runtime_env(request.runtime_id.as_deref())?;
    let gpu_requested = serve_http_gpu_requested(request.device_policy.as_deref());
    let runtime_env = if gpu_requested {
        Some(require_managed_therock_hip_runtime_env(
            runtime_env,
            request.runtime_id.as_deref(),
            request.device_policy.as_deref().unwrap_or("gpu_required"),
        )?)
    } else {
        runtime_env
    };
    let prepared_server = prepare_llama_server_for_launch(
        &paths,
        &server,
        runtime_env.as_ref(),
        &request.service_id,
    )?;
    let endpoint_url = format!("{}/v1", format_http_base_url(&request.host, request.port));
    write_state(
        &request.state_path,
        &json!({
            "engine": ENGINE_NAME,
            "service_id": request.service_id,
            "model_ref": request.model_ref,
            "host": request.host,
            "port": request.port,
            "pid": std::process::id(),
            "wrapper_pid": std::process::id(),
            "status": "starting",
            "endpoint_url": endpoint_url,
            "log_path": request
                .log_path
                .as_ref()
                .map(|path| path.display().to_string()),
            "server": prepared_server.display,
            "server_source": server.display,
            "staged_runtime_dir": prepared_server
                .staged_dir
                .as_ref()
                .map(|path| path.display().to_string()),
            "device_policy": request.device_policy,
            "runtime_id": request.runtime_id,
            "env_id": request.env_id,
            "engine_recipe": request.engine_recipe,
            "engine_recipe_required_flags": engine_recipe_launch_args(request.engine_recipe.as_ref()),
            "therock_runtime_env": runtime_env.as_ref().map(|runtime_env| json!({
                "runtime_id": runtime_env.runtime_id.clone(),
                "runtime_key": runtime_env.runtime_key.clone(),
                "root": runtime_env.root_path.display().to_string(),
                "bin": runtime_env.bin_path.display().to_string(),
                "bin_paths": runtime_bin_paths(runtime_env)
                    .into_iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>(),
                "library_paths": runtime_env.library_paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>(),
                "source": runtime_env.source.clone(),
            })),
        }),
    )?;
    let mut command = ProcessCommand::new(&prepared_server.program);
    let server_args = llama_server_args(&request, gpu_requested);
    command.args(&server_args).stdin(Stdio::null());
    if let Some(log_path) = request.log_path.as_ref() {
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let log_file = fs::File::create(log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        let log_file_err = log_file
            .try_clone()
            .context("failed to clone log file handle")?;
        command
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_file_err));
    } else {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }
    if let Some(runtime_env) = runtime_env.as_ref() {
        apply_therock_hip_runtime_env(&mut command, runtime_env)?;
    }
    let mut child = command.spawn().context("failed to start llama-server")?;
    merge_json_state(
        &request.state_path,
        &json!({
            "status": "running",
            "server_pid": child.id(),
        }),
    )?;
    let status = child.wait().context("failed waiting for llama-server")?;
    let final_status = if status.success() {
        "stopped"
    } else {
        "failed"
    };
    mark_json_status(&request.state_path, final_status)?;
    if status.success() {
        Ok(())
    } else {
        bail!("llama-server exited with status {status}")
    }
}

fn serve_http_gpu_requested(policy: Option<&str>) -> bool {
    !matches!(policy, Some("cpu" | "cpu_only"))
}

#[cfg(windows)]
fn spawn_serve_http_background(
    current_exe: &Path,
    serve_args: &[String],
    state_path: &Path,
) -> Result<u32> {
    let launcher_path = background_launcher_script_path(state_path);
    fs::write(&launcher_path, windows_background_launcher_script())
        .with_context(|| format!("failed to write {}", launcher_path.display()))?;
    let output = ProcessCommand::new("powershell.exe")
        .arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(&launcher_path)
        .arg(current_exe)
        .args(serve_args)
        .stdin(Stdio::null())
        .output();
    let _ = fs::remove_file(&launcher_path);
    let output = output.context("failed to invoke PowerShell background launcher")?;
    if !output.status.success() {
        bail!(
            "PowerShell background launcher failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let pid_text = String::from_utf8_lossy(&output.stdout);
    let pid_text = pid_text.trim();
    pid_text.parse::<u32>().with_context(|| {
        format!("PowerShell background launcher returned invalid pid `{pid_text}`")
    })
}

#[cfg(windows)]
fn background_launcher_script_path(state_path: &Path) -> PathBuf {
    state_path.with_extension("launch.ps1")
}

#[cfg(windows)]
fn windows_background_launcher_script() -> &'static str {
    r#"$ErrorActionPreference = 'Stop'
if ($args.Count -lt 1) {
  throw 'missing executable path'
}
$exe = $args[0]
$childArgs = @()
if ($args.Count -gt 1) {
  $childArgs = $args[1..($args.Count - 1)]
}
$p = Start-Process -FilePath $exe -ArgumentList $childArgs -WindowStyle Hidden -PassThru
[Console]::Out.Write($p.Id)
"#
}

#[cfg(not(windows))]
fn spawn_serve_http_background(
    current_exe: &Path,
    serve_args: &[String],
    _state_path: &Path,
) -> Result<u32> {
    let child = ProcessCommand::new(current_exe)
        .args(serve_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    Ok(child.id())
}

fn require_managed_therock_hip_runtime_env(
    runtime_env: Option<TheRockHipRuntimeEnv>,
    runtime_id: Option<&str>,
    policy: &str,
) -> Result<TheRockHipRuntimeEnv> {
    let Some(runtime_env) = runtime_env else {
        let runtime_hint = runtime_id
            .map(|runtime_id| format!(" for runtime_id `{runtime_id}`"))
            .unwrap_or_default();
        bail!(
            "llama.cpp {policy} requires a rocm-cli managed TheRock runtime manifest with rocm_sdk.root_path and rocm_sdk.bin_path{runtime_hint}; no CPU fallback is applied"
        );
    };
    if !runtime_env.source.starts_with("managed_runtime_manifest") {
        bail!(
            "llama.cpp {policy} requires TheRock SDK paths from a rocm-cli managed runtime manifest; source `{}` is not allowed and no CPU fallback is applied",
            runtime_env.source
        );
    }
    Ok(runtime_env)
}

fn llama_server_args(request: &ServeHttpRequest, gpu_requested: bool) -> Vec<String> {
    let mut args = vec![
        "-m".to_owned(),
        request.model_ref.clone(),
        "--host".to_owned(),
        request.host.clone(),
        "--port".to_owned(),
        request.port.to_string(),
    ];
    if gpu_requested {
        args.push("--n-gpu-layers".to_owned());
        args.push(LLAMA_GPU_LAYERS_VALUE.to_owned());
    }
    args.extend(engine_recipe_launch_args(request.engine_recipe.as_ref()));
    args
}

fn engine_recipe_launch_args(engine_recipe: Option<&EngineRecipeHint>) -> Vec<String> {
    engine_recipe
        .map(|hint| hint.required_flags.clone())
        .unwrap_or_default()
}

fn healthcheck_service(request: HealthcheckRequest) -> Result<HealthcheckResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path).ok();
    let endpoint_url = state.as_ref().and_then(endpoint_url_from_state);
    let model_ref = state
        .as_ref()
        .and_then(|value| value_string(value, "model_ref"));
    let ready = endpoint_url
        .as_deref()
        .map(|endpoint| query_loaded_model_endpoint(endpoint, model_ref.as_deref()))
        .transpose()
        .unwrap_or(None)
        .unwrap_or(false);
    let status = if ready {
        "ready".to_owned()
    } else {
        state
            .as_ref()
            .and_then(|value| value_string(value, "status"))
            .unwrap_or_else(|| "unknown".to_owned())
    };
    Ok(HealthcheckResponse {
        status,
        model_loaded: ready,
        device: healthcheck_device_from_state(state.as_ref()),
        uptime_sec: 0,
        queue_depth: 0,
        last_error: None,
        tokens_per_sec: None,
    })
}

fn healthcheck_device_from_state(state: Option<&Value>) -> String {
    let Some(state) = state else {
        return "unknown".to_owned();
    };
    match value_string(state, "device_policy").as_deref() {
        Some("gpu_required" | "gpu_preferred") => "rocm_gpu".to_owned(),
        _ if state.get("therock_runtime_env").is_some() => "rocm_gpu".to_owned(),
        _ => "unknown".to_owned(),
    }
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
            "/health".to_owned(),
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

fn pid_to_terminate_from_state(state: &Value) -> Option<u32> {
    value_u32(state, "server_pid").or_else(|| value_u32(state, "pid"))
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

fn resolve_llama_server() -> Result<LlamaServer> {
    for env in ["ROCM_CLI_LLAMA_CPP_SERVER", "LLAMA_CPP_SERVER"] {
        if let Some(value) = std::env::var_os(env) {
            let path = PathBuf::from(value);
            if path.is_file() {
                let path = path.canonicalize().unwrap_or(path);
                return Ok(LlamaServer {
                    program: path.display().to_string(),
                    display: format!("{env}={}", path.display()),
                });
            }
        }
    }
    if let Some(path) = find_program_on_path("llama-server") {
        return Ok(LlamaServer {
            program: path.display().to_string(),
            display: format!("llama-server on PATH ({})", path.display()),
        });
    }
    bail!("unable to locate llama-server; set ROCM_CLI_LLAMA_CPP_SERVER")
}

fn prepare_llama_server_for_launch(
    paths: &AppPaths,
    server: &LlamaServer,
    runtime_env: Option<&TheRockHipRuntimeEnv>,
    service_id: &str,
) -> Result<PreparedLlamaServer> {
    if cfg!(windows)
        && let Some(runtime_env) = runtime_env
    {
        return stage_windows_therock_llama_server(paths, server, runtime_env, service_id);
    }

    Ok(PreparedLlamaServer {
        program: PathBuf::from(&server.program),
        display: server.display.clone(),
        staged_dir: None,
    })
}

fn stage_windows_therock_llama_server(
    paths: &AppPaths,
    server: &LlamaServer,
    runtime_env: &TheRockHipRuntimeEnv,
    service_id: &str,
) -> Result<PreparedLlamaServer> {
    let server_program = resolve_llama_server_program_path(server)?;
    let server_file_name = server_program
        .file_name()
        .context("llama-server path has no file name")?;
    let stage_dir = paths
        .engine_dir(ENGINE_NAME)
        .join("runtime-bin")
        .join(stage_dir_name(service_id, server, runtime_env));
    if stage_dir.exists() {
        fs::remove_dir_all(&stage_dir).with_context(|| {
            format!(
                "failed to clear stale staged runtime {}",
                stage_dir.display()
            )
        })?;
    }
    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;

    stage_llama_server_files(&server_program, &stage_dir)?;
    stage_therock_runtime_files(runtime_env, &stage_dir)?;

    let staged_program = stage_dir.join(server_file_name);
    Ok(PreparedLlamaServer {
        program: staged_program.clone(),
        display: format!(
            "staged llama-server with managed TheRock runtime: {}",
            staged_program.display()
        ),
        staged_dir: Some(stage_dir),
    })
}

fn stage_dir_name(
    service_id: &str,
    server: &LlamaServer,
    runtime_env: &TheRockHipRuntimeEnv,
) -> String {
    let mut hasher = DefaultHasher::new();
    server.program.hash(&mut hasher);
    runtime_env.runtime_id.hash(&mut hasher);
    runtime_env.bin_path.hash(&mut hasher);
    for path in &runtime_env.bin_paths {
        path.hash(&mut hasher);
    }
    format!(
        "{}-{:016x}",
        safe_path_component(service_id),
        hasher.finish()
    )
}

fn stage_llama_server_files(server_program: &Path, stage_dir: &Path) -> Result<()> {
    let server_dir = server_program
        .parent()
        .context("llama-server path has no parent directory")?;
    let server_file_name = server_program
        .file_name()
        .context("llama-server path has no file name")?;
    stage_file(server_program, &stage_dir.join(server_file_name), false)?;

    for entry in fs::read_dir(server_dir)
        .with_context(|| format!("failed to read {}", server_dir.display()))?
    {
        let entry = entry?;
        let source = entry.path();
        if source == server_program || !is_windows_dll(&source) {
            continue;
        }
        let destination = stage_dir.join(entry.file_name());
        stage_file(&source, &destination, false)?;
    }
    Ok(())
}

fn stage_therock_runtime_files(runtime_env: &TheRockHipRuntimeEnv, stage_dir: &Path) -> Result<()> {
    validate_therock_windows_runtime_files(runtime_env)?;
    for bin_path in runtime_bin_paths(runtime_env) {
        for entry in fs::read_dir(&bin_path)
            .with_context(|| format!("failed to read {}", bin_path.display()))?
        {
            let entry = entry?;
            let source = entry.path();
            if !source.is_file() || !is_windows_dll(&source) {
                continue;
            }
            let destination = stage_dir.join(entry.file_name());
            stage_file(&source, &destination, true)?;
        }

        for dirname in ["rocblas", "hipblaslt"] {
            let source = bin_path.join(dirname);
            if source.is_dir() {
                stage_runtime_tree(&source, &stage_dir.join(dirname))?;
            }
        }
    }
    Ok(())
}

fn validate_therock_windows_runtime_files(runtime_env: &TheRockHipRuntimeEnv) -> Result<()> {
    for filename in REQUIRED_WINDOWS_THEROCK_EXACT_DLLS {
        if find_runtime_bin_file(runtime_env, filename).is_none() {
            bail!(
                "managed TheRock runtime is missing required llama.cpp HIP DLL {filename}; no CPU fallback is applied"
            );
        }
    }
    for (prefix, suffix) in REQUIRED_WINDOWS_THEROCK_VERSIONED_DLLS {
        if find_runtime_bin_file_by_prefix_suffix(runtime_env, prefix, suffix).is_none() {
            bail!(
                "managed TheRock runtime is missing required llama.cpp HIP DLL matching {prefix}*{suffix}; no CPU fallback is applied"
            );
        }
    }
    for components in REQUIRED_WINDOWS_THEROCK_DATA_DIRS {
        if find_runtime_bin_dir(runtime_env, components).is_none() {
            bail!(
                "managed TheRock runtime is missing required llama.cpp HIP data directory {}; no CPU fallback is applied",
                components.join("/")
            );
        }
    }
    Ok(())
}

fn runtime_bin_paths(runtime_env: &TheRockHipRuntimeEnv) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for path in std::iter::once(&runtime_env.bin_path).chain(runtime_env.bin_paths.iter()) {
        if path.is_dir() && !paths.iter().any(|existing| existing == path) {
            paths.push(path.clone());
        }
    }
    paths
}

fn find_runtime_bin_file(runtime_env: &TheRockHipRuntimeEnv, filename: &str) -> Option<PathBuf> {
    runtime_bin_paths(runtime_env)
        .into_iter()
        .map(|path| path.join(filename))
        .find(|path| path.is_file())
}

fn find_runtime_bin_file_by_prefix_suffix(
    runtime_env: &TheRockHipRuntimeEnv,
    prefix: &str,
    suffix: &str,
) -> Option<PathBuf> {
    for bin_path in runtime_bin_paths(runtime_env) {
        let Ok(entries) = fs::read_dir(&bin_path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if starts_with_ascii_case_insensitive(file_name, prefix)
                && ends_with_ascii_case_insensitive(file_name, suffix)
            {
                return Some(path);
            }
        }
    }
    None
}

fn starts_with_ascii_case_insensitive(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

fn ends_with_ascii_case_insensitive(value: &str, suffix: &str) -> bool {
    value
        .get(value.len().saturating_sub(suffix.len())..)
        .is_some_and(|tail| tail.eq_ignore_ascii_case(suffix))
}

fn find_runtime_bin_dir(
    runtime_env: &TheRockHipRuntimeEnv,
    components: &[&str],
) -> Option<PathBuf> {
    runtime_bin_paths(runtime_env)
        .into_iter()
        .map(|path| {
            components
                .iter()
                .fold(path, |path, component| path.join(component))
        })
        .find(|path| path.is_dir())
}

fn stage_runtime_tree(source_dir: &Path, destination_dir: &Path) -> Result<()> {
    fs::create_dir_all(destination_dir)
        .with_context(|| format!("failed to create {}", destination_dir.display()))?;
    for entry in fs::read_dir(source_dir)
        .with_context(|| format!("failed to read {}", source_dir.display()))?
    {
        let entry = entry?;
        let source = entry.path();
        let destination = destination_dir.join(entry.file_name());
        if source.is_dir() {
            stage_runtime_tree(&source, &destination)?;
        } else if source.is_file() {
            stage_file(&source, &destination, true)?;
        }
    }
    Ok(())
}

fn stage_file(source: &Path, destination: &Path, prefer_hardlink: bool) -> Result<()> {
    if staged_file_is_current(source, destination) {
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    if destination.exists() {
        fs::remove_file(destination)
            .with_context(|| format!("failed to replace {}", destination.display()))?;
    }
    if prefer_hardlink && fs::hard_link(source, destination).is_ok() {
        return Ok(());
    }
    fs::copy(source, destination).with_context(|| {
        format!(
            "failed to stage {} as {}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn staged_file_is_current(source: &Path, destination: &Path) -> bool {
    let (Ok(source_meta), Ok(destination_meta)) = (source.metadata(), destination.metadata())
    else {
        return false;
    };
    if source_meta.len() != destination_meta.len() {
        return false;
    }
    let (Ok(source_modified), Ok(destination_modified)) =
        (source_meta.modified(), destination_meta.modified())
    else {
        return true;
    };
    destination_modified >= source_modified
}

fn is_windows_dll(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"))
}

fn safe_path_component(value: &str) -> String {
    let mut safe = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
            safe.push(ch);
        } else {
            safe.push('_');
        }
    }
    let safe = safe.trim_matches('.');
    if safe.is_empty() {
        "service".to_owned()
    } else {
        safe.to_owned()
    }
}

fn resolve_llama_server_program_path(server: &LlamaServer) -> Result<PathBuf> {
    let program = PathBuf::from(&server.program);
    if program.is_file() {
        return Ok(program.canonicalize().unwrap_or(program));
    }
    find_program_on_path(&server.program).with_context(|| {
        format!(
            "unable to resolve llama-server executable `{}`",
            server.program
        )
    })
}

fn find_program_on_path(program: &str) -> Option<PathBuf> {
    let program_path = Path::new(program);
    if program_path.components().count() > 1 && program_path.is_file() {
        return Some(program_path.to_path_buf());
    }

    let path = std::env::var_os("PATH")?;
    let extensions = executable_extensions(program);
    for dir in std::env::split_paths(&path) {
        for extension in &extensions {
            let candidate = dir.join(format!("{program}{extension}"));
            if candidate.is_file() {
                return Some(candidate.canonicalize().unwrap_or(candidate));
            }
        }
    }
    None
}

fn executable_extensions(program: &str) -> Vec<String> {
    if Path::new(program).extension().is_some() {
        return vec![String::new()];
    }
    if cfg!(windows) {
        let pathext = std::env::var_os("PATHEXT")
            .and_then(|value| value.into_string().ok())
            .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".to_owned());
        let mut extensions = vec![String::new()];
        for extension in pathext.split(';') {
            if extension.is_empty() {
                continue;
            }
            let extension = if extension.starts_with('.') {
                extension.to_owned()
            } else {
                format!(".{extension}")
            };
            if !extensions
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&extension))
            {
                extensions.push(extension);
            }
        }
        extensions
    } else {
        vec![String::new()]
    }
}

fn resolve_therock_hip_runtime_env(
    runtime_id: Option<&str>,
) -> Result<Option<TheRockHipRuntimeEnv>> {
    let paths = AppPaths::discover()?;
    resolve_managed_therock_hip_runtime_env(&paths, runtime_id)
}

fn resolve_managed_therock_hip_runtime_env(
    paths: &AppPaths,
    runtime_id: Option<&str>,
) -> Result<Option<TheRockHipRuntimeEnv>> {
    let registry_dir = paths.data_dir.join("runtimes").join("registry");
    if !registry_dir.is_dir() {
        return Ok(None);
    }

    let mut manifests = Vec::new();
    for entry in fs::read_dir(&registry_dir)
        .with_context(|| format!("failed to read {}", registry_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let Ok(manifest) = serde_json::from_slice::<TheRockRuntimeManifest>(&bytes) else {
            continue;
        };
        if !therock_runtime_matches(&manifest, runtime_id) {
            continue;
        }
        manifests.push((manifest.installed_at_unix_ms.unwrap_or(0), manifest));
    }
    manifests.sort_by_key(|(installed_at, _)| std::cmp::Reverse(*installed_at));

    for (_, manifest) in manifests {
        if let Some(env) = therock_env_from_manifest(&manifest)? {
            return Ok(Some(env));
        }
    }
    Ok(None)
}

fn therock_runtime_matches(manifest: &TheRockRuntimeManifest, requested: Option<&str>) -> bool {
    let Some(requested) = requested.map(str::trim).filter(|value| !value.is_empty()) else {
        return true;
    };
    let requested = requested.to_ascii_lowercase();
    if requested == "external" || requested == "external-llama.cpp" {
        return false;
    }

    for candidate in [
        manifest.runtime_id.as_deref(),
        manifest.runtime_key.as_deref(),
        manifest.format.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        let candidate = candidate.to_ascii_lowercase();
        if candidate == requested || candidate.starts_with(&requested) {
            return true;
        }
    }
    false
}

fn therock_env_from_manifest(
    manifest: &TheRockRuntimeManifest,
) -> Result<Option<TheRockHipRuntimeEnv>> {
    let runtime_id = manifest
        .runtime_id
        .clone()
        .unwrap_or_else(|| "therock-runtime".to_owned());
    let runtime_key = manifest.runtime_key.clone();
    let source = runtime_key.as_deref().map_or_else(
        || "managed_runtime_manifest".to_owned(),
        |key| format!("managed_runtime_manifest:{key}"),
    );

    if let Some(probe) = manifest.rocm_sdk.as_ref()
        && probe.import_ok
        && let (Some(root_path), Some(bin_path)) =
            (probe.root_path.as_ref(), probe.bin_path.as_ref())
        && root_path.is_dir()
        && bin_path.is_dir()
    {
        let bin_paths = probe
            .bin_paths
            .iter()
            .filter(|path| path.is_dir())
            .cloned()
            .collect::<Vec<_>>();
        let library_paths = probe
            .library_paths
            .iter()
            .filter(|path| path.is_dir())
            .cloned()
            .collect::<Vec<_>>();
        return Ok(Some(TheRockHipRuntimeEnv {
            runtime_id,
            runtime_key,
            root_path: root_path.clone(),
            bin_path: bin_path.clone(),
            bin_paths,
            library_paths,
            source,
        }));
    }

    Ok(None)
}

fn apply_therock_hip_runtime_env(
    command: &mut ProcessCommand,
    runtime_env: &TheRockHipRuntimeEnv,
) -> Result<()> {
    command
        .env("ROCM_SDK_ROOT", &runtime_env.root_path)
        .env("ROCM_PATH", &runtime_env.root_path)
        .env("ROCM_HOME", &runtime_env.root_path)
        .env("HIP_PATH", &runtime_env.root_path)
        .env("ROCM_CLI_THEROCK_RUNTIME_ID", &runtime_env.runtime_id)
        .env("ROCM_CLI_THEROCK_SDK_BIN", &runtime_env.bin_path)
        .env(
            "PATH",
            prepend_path_entries(&runtime_bin_paths(runtime_env), std::env::var_os("PATH"))?,
        );

    let cmake_path = runtime_env.root_path.join("lib").join("cmake");
    command.env(
        "CMAKE_PREFIX_PATH",
        prepend_path_entries(
            &[runtime_env.root_path.clone(), cmake_path],
            std::env::var_os("CMAKE_PREFIX_PATH"),
        )?,
    );

    if !cfg!(windows) {
        command.env(
            "LD_LIBRARY_PATH",
            prepend_path_entries(
                &therock_library_path_entries(runtime_env),
                std::env::var_os("LD_LIBRARY_PATH"),
            )?,
        );
    }
    Ok(())
}

fn therock_library_path_entries(runtime_env: &TheRockHipRuntimeEnv) -> Vec<PathBuf> {
    let mut entries = runtime_env.library_paths.clone();
    entries.extend([
        runtime_env.root_path.join("lib"),
        runtime_env.root_path.join("lib64"),
        runtime_env
            .root_path
            .join("lib")
            .join("rocm_sysdeps")
            .join("lib"),
    ]);
    if cfg!(target_os = "linux") {
        let wsl_dxcore_lib = PathBuf::from("/usr/lib/wsl/lib");
        if wsl_dxcore_lib.is_dir() {
            entries.push(wsl_dxcore_lib);
        }
    }
    entries
}

fn prepend_path_entries(entries: &[PathBuf], current: Option<OsString>) -> Result<OsString> {
    let mut parts = Vec::new();
    for entry in entries {
        if !entry.as_os_str().is_empty() && !parts.iter().any(|part: &PathBuf| part == entry) {
            parts.push(entry.clone());
        }
    }
    if let Some(current) = current {
        for entry in std::env::split_paths(&current) {
            if !entry.as_os_str().is_empty() && !parts.iter().any(|part| part == &entry) {
                parts.push(entry);
            }
        }
    }
    std::env::join_paths(parts).context("failed to compose TheRock runtime path")
}

fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_state(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(value).context("failed to serialize llama.cpp state")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
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

fn endpoint_url_from_state(state: &Value) -> Option<String> {
    value_string(state, "endpoint_url").or_else(|| {
        let host = value_string(state, "host")?;
        let port = value_u32(state, "port")?;
        let port = u16::try_from(port).ok()?;
        Some(format!("{}/v1", format_http_base_url(&host, port)))
    })
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

fn query_loaded_model_endpoint(endpoint_url: &str, model_ref: Option<&str>) -> Result<bool> {
    openai_models_endpoint_has_model(
        endpoint_url,
        model_ref,
        Duration::from_millis(HEALTHCHECK_TIMEOUT_MS),
    )
}

#[cfg(test)]
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

fn tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut lines = VecDeque::with_capacity(limit);
    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read {}", path.display()))?;
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

const fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
}

fn parse_device_policy_arg(value: Option<&str>) -> Result<DevicePolicy> {
    match value.unwrap_or("gpu_required") {
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        "gpu" | "gpu_preferred" => Ok(DevicePolicy::GpuPreferred),
        "gpu_required" => Ok(DevicePolicy::GpuRequired),
        other => bail!("unknown device policy `{other}`"),
    }
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

    fn test_engine_recipe(engine: &str, contract_version: &str) -> EngineRecipeHint {
        EngineRecipeHint {
            contract_version: contract_version.to_owned(),
            engine: engine.to_owned(),
            required_flags: vec!["--jinja".to_owned()],
            parser_settings: std::collections::BTreeMap::default(),
            preferred_endpoint: None,
            unsupported_combinations: Vec::new(),
            notes: vec!["test recipe".to_owned()],
        }
    }

    #[test]
    fn resolve_model_rejects_cpu_policy_without_fallback() {
        let error = resolve_model_response(ResolveModelRequest {
            model_ref: "Qwen/Qwen3.5".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::CpuOnly),
            recipe_override: None,
            engine_recipe: None,
        })
        .expect_err("cpu policy should not resolve");
        assert!(error.to_string().contains("no CPU fallback is used"));
    }

    #[test]
    fn resolve_model_accepts_gpu_preferred_without_cpu_fallback() -> Result<()> {
        let response = resolve_model_response(ResolveModelRequest {
            model_ref: "tiny.gguf".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuPreferred),
            recipe_override: None,
            engine_recipe: None,
        })?;

        assert_eq!(response.device_policy, DevicePolicy::GpuRequired);
        Ok(())
    }

    #[test]
    fn resolve_model_accepts_gpu_required() -> Result<()> {
        let response = resolve_model_response(ResolveModelRequest {
            model_ref: "tiny.gguf".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: None,
        })?;

        assert_eq!(response.device_policy, DevicePolicy::GpuRequired);
        Ok(())
    }

    #[test]
    fn resolve_model_canonicalizes_existing_gguf_path() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "rocm-llama-model-{}-{}.gguf",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, "fake")?;
        let canonical = path.canonicalize()?.display().to_string();
        let response = resolve_model_response(ResolveModelRequest {
            model_ref: path.display().to_string(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: None,
        })?;
        fs::remove_file(&path).ok();

        assert_eq!(response.canonical_model_id, canonical);
        assert!(response.warnings.is_empty());
        Ok(())
    }

    #[test]
    fn resolve_model_warns_for_missing_gguf_path() -> Result<()> {
        let response = resolve_model_response(ResolveModelRequest {
            model_ref: "missing-model.gguf".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: None,
        })?;

        assert_eq!(response.canonical_model_id, "missing-model.gguf");
        assert!(
            response
                .warnings
                .iter()
                .any(|warning| warning.contains("not found locally"))
        );
        Ok(())
    }

    #[test]
    fn resolve_model_echoes_matching_engine_recipe() -> Result<()> {
        let hint = test_engine_recipe(ENGINE_NAME, ENGINE_RECIPE_CONTRACT_VERSION);
        let response = resolve_model_response(ResolveModelRequest {
            model_ref: "tiny.gguf".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(hint.clone()),
        })?;

        assert_eq!(response.engine_recipe, Some(hint));
        Ok(())
    }

    #[test]
    fn resolve_model_rejects_mismatched_engine_recipe() {
        let error = resolve_model_response(ResolveModelRequest {
            model_ref: "tiny.gguf".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(test_engine_recipe(
                "pytorch",
                ENGINE_RECIPE_CONTRACT_VERSION,
            )),
        })
        .expect_err("mismatched engine recipe should fail");

        assert!(error.to_string().contains("does not match adapter"));
    }

    #[test]
    fn resolve_model_rejects_unsupported_engine_recipe_contract() {
        let error = resolve_model_response(ResolveModelRequest {
            model_ref: "tiny.gguf".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(test_engine_recipe(ENGINE_NAME, "999.0.0")),
        })
        .expect_err("unsupported recipe contract should fail");

        assert!(error.to_string().contains("unsupported"));
    }

    #[test]
    fn install_response_marks_llama_server_as_external_runtime() -> Result<()> {
        let response = install_response(InstallRequest {
            runtime_id: "external".to_owned(),
            python_version: None,
            env_root: None,
            reinstall: false,
        })?;

        assert_eq!(
            response.runtime_kind.as_deref(),
            Some("external_llama_server")
        );
        assert_eq!(response.managed_env, Some(false));
        Ok(())
    }

    #[test]
    fn managed_therock_env_uses_rocm_sdk_root_for_hip_apps() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("rocm-llama-therock-env-{}", current_unix_millis()));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        let sdk_root = root.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&sdk_bin)?;
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("release-pip-gfx120x-all.json"),
            serde_json::to_vec_pretty(&json!({
                "runtime_key": "release-pip-gfx120x-all",
                "runtime_id": "therock-release:gfx120X-all",
                "format": "pip",
                "installed_at_unix_ms": 2,
                "rocm_sdk": {
                    "import_ok": true,
                    "root_path": sdk_root,
                    "bin_path": sdk_bin
                }
            }))?,
        )?;

        let env = resolve_managed_therock_hip_runtime_env(&paths, Some("therock-release"))?
            .expect("managed TheRock env");
        fs::remove_dir_all(root).ok();

        assert_eq!(env.runtime_id, "therock-release:gfx120X-all");
        assert_eq!(env.runtime_key.as_deref(), Some("release-pip-gfx120x-all"));
        assert!(env.root_path.ends_with("_rocm_sdk_devel"));
        assert!(env.bin_path.ends_with("bin"));
        Ok(())
    }

    #[test]
    fn external_runtime_id_does_not_pick_managed_therock_env() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("rocm-llama-external-env-{}", current_unix_millis()));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("release-pip-gfx120x-all.json"),
            serde_json::to_vec_pretty(&json!({
                "runtime_key": "release-pip-gfx120x-all",
                "runtime_id": "therock-release:gfx120X-all",
                "format": "pip",
                "installed_at_unix_ms": 2,
                "rocm_sdk": {
                    "import_ok": true,
                    "root_path": root.join("_rocm_sdk_devel"),
                    "bin_path": root.join("_rocm_sdk_devel").join("bin")
                }
            }))?,
        )?;

        let env = resolve_managed_therock_hip_runtime_env(&paths, Some("external"))?;
        fs::remove_dir_all(root).ok();

        assert!(env.is_none());
        Ok(())
    }

    #[test]
    fn prepend_path_entries_puts_therock_bin_first() -> Result<()> {
        let root = if cfg!(windows) {
            "C:\\rocm"
        } else {
            "/tmp/rocm"
        };
        let current = std::env::join_paths([PathBuf::from("existing")])?;
        let composed = prepend_path_entries(&[PathBuf::from(root).join("bin")], Some(current))?;
        let parts = std::env::split_paths(&composed).collect::<Vec<_>>();

        assert_eq!(parts.first(), Some(&PathBuf::from(root).join("bin")));
        assert!(parts.iter().any(|part| part == &PathBuf::from("existing")));
        Ok(())
    }

    #[test]
    fn therock_library_path_entries_include_sysdeps_for_hip_apps() {
        let env = TheRockHipRuntimeEnv {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: Some("release-pip-gfx120x-all".to_owned()),
            root_path: PathBuf::from(if cfg!(windows) {
                "C:\\rocm"
            } else {
                "/tmp/rocm"
            }),
            bin_path: PathBuf::from(if cfg!(windows) {
                "C:\\rocm\\bin"
            } else {
                "/tmp/rocm/bin"
            }),
            bin_paths: Vec::new(),
            library_paths: Vec::new(),
            source: "managed_runtime_manifest:release-pip-gfx120x-all".to_owned(),
        };
        let entries = therock_library_path_entries(&env);

        assert!(entries.iter().any(|entry| entry.ends_with("lib")));
        assert!(entries.iter().any(|entry| entry.ends_with("lib64")));
        assert!(
            entries
                .iter()
                .any(|entry| entry.ends_with(Path::new("lib").join("rocm_sysdeps").join("lib")))
        );
    }

    #[test]
    fn llama_server_has_hip_backend_detects_sibling_backend() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("rocm-llama-hip-backend-{}", current_unix_millis()));
        fs::create_dir_all(&root)?;
        let server_path = root.join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        });
        fs::write(&server_path, "exe")?;
        let backend_name = if cfg!(windows) {
            "ggml-hip.dll"
        } else if cfg!(target_os = "macos") {
            "libggml-hip.dylib"
        } else {
            "libggml-hip.so"
        };
        fs::write(root.join(backend_name), "hip")?;
        let server = LlamaServer {
            program: server_path.display().to_string(),
            display: "test".to_owned(),
        };

        let has_backend = llama_server_has_hip_backend(&server);
        fs::remove_dir_all(root).ok();

        assert!(has_backend);
        Ok(())
    }

    #[test]
    fn windows_therock_staging_places_runtime_next_to_llama_server() -> Result<()> {
        let root = std::env::temp_dir().join(format!("rocm-llama-stage-{}", current_unix_millis()));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        let server_dir = root.join("server");
        fs::create_dir_all(&server_dir)?;
        let server_path = server_dir.join("llama-server.exe");
        fs::write(&server_path, "exe")?;
        fs::write(server_dir.join("ggml-hip.dll"), "backend")?;

        let sdk_root = root.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(sdk_bin.join("rocblas").join("library"))?;
        fs::create_dir_all(sdk_bin.join("hipblaslt").join("library"))?;
        for filename in REQUIRED_WINDOWS_THEROCK_EXACT_DLLS {
            fs::write(sdk_bin.join(filename), format!("managed {filename}"))?;
        }
        for filename in [
            "amd_comgr0713.dll",
            "hiprtc-builtins07013.dll",
            "hiprtc07013.dll",
        ] {
            fs::write(sdk_bin.join(filename), format!("managed {filename}"))?;
        }
        fs::write(
            sdk_bin
                .join("rocblas")
                .join("library")
                .join("TensileManifest.txt"),
            "rocblas data",
        )?;
        fs::write(
            sdk_bin
                .join("hipblaslt")
                .join("library")
                .join("TensileLibrary.dat"),
            "hipblaslt data",
        )?;
        let server = LlamaServer {
            program: server_path.display().to_string(),
            display: "test".to_owned(),
        };
        let runtime_env = TheRockHipRuntimeEnv {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: Some("release-pip-gfx120x-all".to_owned()),
            root_path: sdk_root,
            bin_path: sdk_bin.clone(),
            bin_paths: vec![sdk_bin],
            library_paths: Vec::new(),
            source: "managed_runtime_manifest:release-pip-gfx120x-all".to_owned(),
        };

        let prepared =
            stage_windows_therock_llama_server(&paths, &server, &runtime_env, "svc:one")?;
        let staged_dir = prepared.staged_dir.expect("staged dir");

        assert!(prepared.program.ends_with("llama-server.exe"));
        assert!(staged_dir.join("llama-server.exe").is_file());
        assert!(staged_dir.join("ggml-hip.dll").is_file());
        assert!(staged_dir.join("amd_comgr0713.dll").is_file());
        assert!(staged_dir.join("amdhip64_7.dll").is_file());
        assert!(staged_dir.join("hiprtc-builtins07013.dll").is_file());
        assert!(staged_dir.join("hiprtc07013.dll").is_file());
        assert!(staged_dir.join("hipblas.dll").is_file());
        assert!(
            staged_dir
                .join("rocblas")
                .join("library")
                .join("TensileManifest.txt")
                .is_file()
        );
        assert!(
            staged_dir
                .join("hipblaslt")
                .join("library")
                .join("TensileLibrary.dat")
                .is_file()
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn windows_therock_staging_supports_split_runtime_wheel_bins() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("rocm-llama-stage-split-{}", current_unix_millis()));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        let server_dir = root.join("server");
        fs::create_dir_all(&server_dir)?;
        let server_path = server_dir.join("llama-server.exe");
        fs::write(&server_path, "exe")?;
        fs::write(server_dir.join("ggml-hip.dll"), "backend")?;

        let core_root = root.join("_rocm_sdk_core");
        let core_bin = core_root.join("bin");
        let libraries_root = root.join("_rocm_sdk_libraries_gfx120X_all");
        let libraries_bin = libraries_root.join("bin");
        fs::create_dir_all(&core_bin)?;
        fs::create_dir_all(libraries_bin.join("rocblas").join("library"))?;
        fs::create_dir_all(libraries_bin.join("hipblaslt").join("library"))?;
        for filename in [
            "amd_comgr0713.dll",
            "amdhip64_7.dll",
            "hiprtc-builtins07013.dll",
            "hiprtc07013.dll",
            "rocm_kpack.dll",
        ] {
            fs::write(core_bin.join(filename), format!("core {filename}"))?;
        }
        for filename in [
            "hipblas.dll",
            "libhipblaslt.dll",
            "rocblas.dll",
            "rocsolver.dll",
        ] {
            fs::write(
                libraries_bin.join(filename),
                format!("libraries {filename}"),
            )?;
        }
        fs::write(
            libraries_bin
                .join("rocblas")
                .join("library")
                .join("TensileManifest.txt"),
            "rocblas data",
        )?;
        fs::write(
            libraries_bin
                .join("hipblaslt")
                .join("library")
                .join("TensileLibrary.dat"),
            "hipblaslt data",
        )?;
        let server = LlamaServer {
            program: server_path.display().to_string(),
            display: "test".to_owned(),
        };
        let runtime_env = TheRockHipRuntimeEnv {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: Some("release-pip-gfx120x-all".to_owned()),
            root_path: core_root,
            bin_path: core_bin.clone(),
            bin_paths: vec![core_bin, libraries_bin],
            library_paths: Vec::new(),
            source: "managed_runtime_manifest:release-pip-gfx120x-all".to_owned(),
        };

        let prepared =
            stage_windows_therock_llama_server(&paths, &server, &runtime_env, "svc:split")?;
        let staged_dir = prepared.staged_dir.expect("staged dir");

        assert!(staged_dir.join("amd_comgr0713.dll").is_file());
        assert!(staged_dir.join("amdhip64_7.dll").is_file());
        assert!(staged_dir.join("hiprtc-builtins07013.dll").is_file());
        assert!(staged_dir.join("hiprtc07013.dll").is_file());
        assert!(staged_dir.join("hipblas.dll").is_file());
        assert!(staged_dir.join("rocblas.dll").is_file());
        assert!(
            staged_dir
                .join("rocblas")
                .join("library")
                .join("TensileManifest.txt")
                .is_file()
        );
        assert!(
            staged_dir
                .join("hipblaslt")
                .join("library")
                .join("TensileLibrary.dat")
                .is_file()
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn windows_therock_staging_fails_when_required_runtime_file_is_missing() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-llama-stage-missing-{}",
            current_unix_millis()
        ));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        let server_dir = root.join("server");
        fs::create_dir_all(&server_dir)?;
        let server_path = server_dir.join("llama-server.exe");
        fs::write(&server_path, "exe")?;
        fs::write(server_dir.join("ggml-hip.dll"), "backend")?;

        let sdk_root = root.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(sdk_bin.join("rocblas").join("library"))?;
        fs::create_dir_all(sdk_bin.join("hipblaslt").join("library"))?;
        for filename in REQUIRED_WINDOWS_THEROCK_EXACT_DLLS {
            if *filename != "amdhip64_7.dll" {
                fs::write(sdk_bin.join(filename), format!("managed {filename}"))?;
            }
        }
        for filename in [
            "amd_comgr0713.dll",
            "hiprtc-builtins07013.dll",
            "hiprtc07013.dll",
        ] {
            fs::write(sdk_bin.join(filename), format!("managed {filename}"))?;
        }
        fs::write(
            sdk_bin.join("rocblas").join("library").join("manifest.txt"),
            "rocblas data",
        )?;
        fs::write(
            sdk_bin
                .join("hipblaslt")
                .join("library")
                .join("manifest.txt"),
            "hipblaslt data",
        )?;
        let server = LlamaServer {
            program: server_path.display().to_string(),
            display: "test".to_owned(),
        };
        let runtime_env = TheRockHipRuntimeEnv {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: Some("release-pip-gfx120x-all".to_owned()),
            root_path: sdk_root,
            bin_path: sdk_bin.clone(),
            bin_paths: vec![sdk_bin],
            library_paths: Vec::new(),
            source: "managed_runtime_manifest:release-pip-gfx120x-all".to_owned(),
        };

        let error = stage_windows_therock_llama_server(&paths, &server, &runtime_env, "svc:one")
            .expect_err("missing amdhip64_7.dll must fail before launch");
        fs::remove_dir_all(root).ok();

        let message = error.to_string();
        assert!(message.contains("amdhip64_7.dll"));
        assert!(message.contains("no CPU fallback"));
        Ok(())
    }

    #[test]
    fn serve_http_cli_accepts_protocol_runtime_args() {
        let cli = Cli::try_parse_from([
            "rocm-engine-llama-cpp",
            "serve-http",
            "svc",
            "tiny.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "11435",
            "--device-policy",
            "gpu_required",
            "--runtime-id",
            "therock-release:gfx120X-all",
            "--env-id",
            "external-llama.cpp",
            "--state-path",
            "state.json",
        ])
        .expect("serve-http should accept protocol runtime args");

        match cli.command {
            CommandKind::ServeHttp {
                device_policy,
                runtime_id,
                env_id,
                ..
            } => {
                assert_eq!(device_policy.as_deref(), Some("gpu_required"));
                assert_eq!(runtime_id.as_deref(), Some("therock-release:gfx120X-all"));
                assert_eq!(env_id.as_deref(), Some("external-llama.cpp"));
            }
            _ => panic!("expected serve-http command"),
        }
    }

    #[test]
    fn launch_cli_accepts_runtime_selection_args() {
        let cli = Cli::try_parse_from([
            "rocm-engine-llama-cpp",
            "launch",
            "svc",
            "tiny.gguf",
            "--device-policy",
            "gpu_preferred",
            "--runtime-id",
            "runtime-1",
            "--env-id",
            "external-llama.cpp",
        ])
        .expect("launch should accept protocol runtime args");

        match cli.command {
            CommandKind::Launch {
                device_policy,
                runtime_id,
                env_id,
                ..
            } => {
                assert_eq!(device_policy.as_deref(), Some("gpu_preferred"));
                assert_eq!(runtime_id.as_deref(), Some("runtime-1"));
                assert_eq!(env_id.as_deref(), Some("external-llama.cpp"));
            }
            _ => panic!("expected launch command"),
        }
    }

    #[test]
    fn serve_http_request_accepts_gpu_required_for_runtime_validation() -> Result<()> {
        let request = normalize_serve_http_request(ServeHttpRequest {
            service_id: "svc".to_owned(),
            model_ref: "tiny.gguf".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: Some("gpu_required".to_owned()),
            runtime_id: None,
            env_id: None,
            state_path: PathBuf::from("state.json"),
            log_path: None,
            engine_recipe: None,
        })?;

        assert_eq!(request.device_policy.as_deref(), Some("gpu_required"));
        Ok(())
    }

    #[test]
    fn gpu_required_fails_loudly_without_managed_therock_sdk() {
        let error =
            require_managed_therock_hip_runtime_env(None, Some("therock-release"), "gpu_required")
                .expect_err("gpu_required should require managed TheRock SDK paths");

        let message = error.to_string();
        assert!(message.contains("managed TheRock runtime manifest"));
        assert!(message.contains("rocm_sdk.root_path"));
        assert!(message.contains("rocm_sdk.bin_path"));
        assert!(message.contains("no CPU fallback"));
    }

    #[test]
    fn gpu_required_rejects_non_manifest_therock_source_without_cpu_fallback() {
        let error = require_managed_therock_hip_runtime_env(
            Some(TheRockHipRuntimeEnv {
                runtime_id: "therock-env".to_owned(),
                runtime_key: None,
                root_path: PathBuf::from("sdk"),
                bin_path: PathBuf::from("sdk").join("bin"),
                bin_paths: Vec::new(),
                library_paths: Vec::new(),
                source: "ROCM_CLI_THEROCK_SDK_ROOT".to_owned(),
            }),
            None,
            "gpu_required",
        )
        .expect_err("external TheRock SDK roots should not satisfy gpu_required");

        let message = error.to_string();
        assert!(message.contains("managed runtime manifest"));
        assert!(message.contains("no CPU fallback"));
    }

    #[test]
    fn parse_device_policy_arg_accepts_cli_aliases() -> Result<()> {
        assert_eq!(parse_device_policy_arg(None)?, DevicePolicy::GpuRequired);
        assert_eq!(
            parse_device_policy_arg(Some("gpu"))?,
            DevicePolicy::GpuPreferred
        );
        assert_eq!(parse_device_policy_arg(Some("cpu"))?, DevicePolicy::CpuOnly);
        assert!(
            normalize_llama_device_policy(Some(DevicePolicy::CpuOnly))
                .unwrap_err()
                .to_string()
                .contains("no CPU fallback is used")
        );
        Ok(())
    }

    #[test]
    fn launch_serve_http_args_preserve_runtime_selection() {
        let request = LaunchRequest {
            service_id: "svc".to_owned(),
            env_id: Some("external-llama.cpp".to_owned()),
            runtime_id: Some("runtime-1".to_owned()),
            model_ref: "tiny.gguf".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: Some(DevicePolicy::GpuRequired),
            endpoint_mode: Some("openai".to_owned()),
            engine_recipe: None,
        };

        let args = serve_http_command_args(
            &request,
            Path::new("state.json"),
            Some(Path::new("log.txt")),
        );

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--device-policy" && pair[1] == "gpu_required")
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--runtime-id" && pair[1] == "runtime-1")
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--env-id" && pair[1] == "external-llama.cpp")
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--state-path" && pair[1] == "state.json")
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--log-path" && pair[1] == "log.txt")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_background_launcher_uses_hidden_start_process() {
        let script = windows_background_launcher_script();

        assert!(script.contains("Start-Process"));
        assert!(script.contains("-WindowStyle Hidden"));
        assert!(script.contains("-PassThru"));
        assert!(script.contains("$args"));
    }

    #[test]
    fn llama_server_args_add_gpu_layers_for_gpu_required() {
        let request = ServeHttpRequest {
            service_id: "svc".to_owned(),
            model_ref: "tiny.gguf".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: Some("gpu_required".to_owned()),
            runtime_id: Some("therock-release".to_owned()),
            env_id: None,
            state_path: PathBuf::from("state.json"),
            log_path: None,
            engine_recipe: None,
        };

        let args = llama_server_args(&request, true);

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "-m" && pair[1] == "tiny.gguf")
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--n-gpu-layers" && pair[1] == LLAMA_GPU_LAYERS_VALUE)
        );
    }

    #[test]
    fn llama_server_args_forward_engine_recipe_flags() {
        let request = ServeHttpRequest {
            service_id: "svc".to_owned(),
            model_ref: "tiny.gguf".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: Some("gpu_required".to_owned()),
            runtime_id: Some("therock-release".to_owned()),
            env_id: None,
            state_path: PathBuf::from("state.json"),
            log_path: None,
            engine_recipe: Some(test_engine_recipe(
                ENGINE_NAME,
                ENGINE_RECIPE_CONTRACT_VERSION,
            )),
        };

        let args = llama_server_args(&request, true);

        assert!(args.iter().any(|arg| arg == "--jinja"));
    }

    #[test]
    fn serve_http_request_rejects_cpu_policy_without_fallback() {
        let error = normalize_serve_http_request(ServeHttpRequest {
            service_id: "svc".to_owned(),
            model_ref: "tiny.gguf".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: Some("cpu_only".to_owned()),
            runtime_id: None,
            env_id: None,
            state_path: PathBuf::from("state.json"),
            log_path: None,
            engine_recipe: None,
        })
        .expect_err("cpu policy should be rejected before launch");

        assert!(error.to_string().contains("no CPU fallback is used"));
    }

    #[test]
    fn healthcheck_device_reflects_gpu_required_state() {
        let state = json!({
            "device_policy": "gpu_required",
            "therock_runtime_env": {
                "runtime_id": "therock-release:gfx120X-all"
            }
        });

        assert_eq!(healthcheck_device_from_state(Some(&state)), "rocm_gpu");
        assert_eq!(
            healthcheck_device_from_state(Some(&json!({ "device_policy": "cpu_only" }))),
            "unknown"
        );
    }

    #[test]
    fn endpoint_url_falls_back_to_host_and_port() {
        let state = json!({
            "host": "127.0.0.1",
            "port": 12345
        });
        assert_eq!(
            endpoint_url_from_state(&state),
            Some("http://127.0.0.1:12345/v1".to_owned())
        );
    }

    #[test]
    fn endpoint_url_and_parser_support_ipv6_loopback() {
        let state = json!({
            "host": "::1",
            "port": 12345
        });
        assert_eq!(
            endpoint_url_from_state(&state),
            Some("http://[::1]:12345/v1".to_owned())
        );
        assert_eq!(
            parse_http_endpoint("http://[::1]:12345/v1"),
            Some(("::1".to_owned(), 12345))
        );
    }

    #[test]
    fn endpoint_response_errors_without_service_state() {
        let error = endpoint_response(EndpointRequest {
            service_id: format!("missing-{}", current_unix_millis()),
        })
        .expect_err("missing service state should not produce a default endpoint");

        assert!(error.to_string().contains("service state not found"));
    }

    #[test]
    fn stdio_protocol_routes_all_methods_without_side_effects() {
        let service_id = format!(
            "missing-protocol-{}-{}",
            std::process::id(),
            current_unix_millis()
        );
        let success_cases = [
            (EngineMethod::Detect, json!({})),
            (EngineMethod::Capabilities, json!({})),
            (
                EngineMethod::ResolveModel,
                json!({
                    "model_ref": "missing.gguf",
                    "device_policy": "gpu_required"
                }),
            ),
            (
                EngineMethod::Healthcheck,
                json!({
                    "service_id": service_id.as_str()
                }),
            ),
            (
                EngineMethod::Logs,
                json!({
                    "service_id": service_id.as_str(),
                    "tail_lines": 4
                }),
            ),
            (
                EngineMethod::Stop,
                json!({
                    "service_id": service_id.as_str(),
                    "force": false
                }),
            ),
        ];

        for (method, payload) in success_cases {
            let response = handle_envelope(EngineRequestEnvelope { method, payload });
            assert!(
                response.ok,
                "expected protocol method to return a typed success envelope: {:?}",
                response.error
            );
        }

        let endpoint = handle_envelope(EngineRequestEnvelope {
            method: EngineMethod::Endpoint,
            payload: json!({
                "service_id": service_id.as_str()
            }),
        });
        assert!(!endpoint.ok);
        assert_eq!(
            endpoint.error.as_ref().map(|error| error.code.as_str()),
            Some("request_failed")
        );

        for method in [EngineMethod::Install, EngineMethod::Launch] {
            let response = handle_envelope(EngineRequestEnvelope {
                method,
                payload: json!({}),
            });
            assert!(!response.ok);
            assert_eq!(
                response.error.as_ref().map(|error| error.code.as_str()),
                Some("invalid_payload")
            );
        }
    }

    #[test]
    fn stop_prefers_recorded_llama_server_pid_over_wrapper_pid() {
        let state = json!({
            "pid": 100,
            "wrapper_pid": 100,
            "server_pid": 200,
        });
        assert_eq!(pid_to_terminate_from_state(&state), Some(200));

        let wrapper_only = json!({ "pid": 100 });
        assert_eq!(pid_to_terminate_from_state(&wrapper_only), Some(100));
    }

    #[test]
    fn tail_lines_returns_suffix() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "rocm-llama-tail-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, "a\nb\nc\n")?;
        let lines = tail_lines(&path, 2)?;
        fs::remove_file(path).ok();
        assert_eq!(lines, vec!["b".to_owned(), "c".to_owned()]);
        Ok(())
    }
}
