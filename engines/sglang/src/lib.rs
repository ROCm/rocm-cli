// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_PORT, format_http_base_url, openai_models_endpoint_has_model,
    require_nonempty,
};
use rocm_engine_protocol::{
    DEFAULT_LOG_TAIL_LINES, DetectRequest, DetectResponse, DevicePolicy,
    ENGINE_RECIPE_CONTRACT_VERSION, EndpointRequest, EndpointResponse, EngineCapabilities,
    EngineDeviceAvailability, EngineMethod, EngineRecipeHint, EngineRequestEnvelope,
    EngineResponseEnvelope, GpuSelection, HealthcheckRequest, HealthcheckResponse, InstallRequest,
    InstallResponse, LaunchRequest, LaunchResponse, LogsRequest, LogsResponse, ResolveModelRequest,
    ResolveModelResponse, StopRequest, StopResponse,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::hash_map::DefaultHasher;
use std::ffi::OsString;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ENGINE_NAME: &str = "sglang";
const DEFAULT_HOST: &str = "127.0.0.1";
const HEALTHCHECK_TIMEOUT_MS: u64 = 700;

#[derive(Parser, Debug)]
#[command(name = "rocm-engine-sglang", about = "rocm-cli SGLang engine adapter")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand, Debug)]
enum CommandKind {
    Detect,
    Capabilities,
    Install {
        #[arg(long, default_value = "external-sglang")]
        runtime_id: String,
        #[arg(long)]
        reinstall: bool,
    },
    ResolveModel {
        model_ref: String,
        #[arg(long)]
        device_policy: Option<String>,
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
        #[arg(long, default_value = "gpu_required")]
        device_policy: String,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
        #[arg(long)]
        state_path: PathBuf,
        #[arg(long)]
        engine_recipe_json: Option<String>,
        #[arg(long)]
        gpu: Option<String>,
    },
}

#[derive(Debug, Clone)]
struct SglangRuntime {
    runtime_id: String,
    env_id: String,
    command: PathBuf,
    launcher: SglangLauncher,
    python_executable: Option<PathBuf>,
    version: Option<String>,
    source: String,
    sdk_root: Option<PathBuf>,
    sdk_bin: Option<PathBuf>,
    sdk_bin_paths: Vec<PathBuf>,
    sdk_library_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SglangLauncher {
    Command,
    PythonModule,
}

#[derive(Debug, Clone, Deserialize)]
struct TheRockRuntimeManifest {
    #[serde(default)]
    runtime_key: Option<String>,
    #[serde(default)]
    runtime_id: Option<String>,
    #[serde(default)]
    python_executable: Option<PathBuf>,
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
        CommandKind::ResolveModel {
            model_ref,
            device_policy,
        } => print_json(&resolve_model_response(ResolveModelRequest {
            model_ref,
            runtime_id: None,
            device_policy: device_policy
                .as_deref()
                .map(|value| parse_device_policy_arg(Some(value)))
                .transpose()?,
            recipe_override: None,
            engine_recipe: None,
        })?)?,
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
            engine_recipe_json,
            gpu,
        } => serve_http(ServeHttpRequest {
            service_id,
            model_ref,
            host,
            port,
            device_policy: parse_device_policy_arg(Some(&device_policy))?,
            gpu_indices: parse_gpu_indices_arg(gpu.as_deref())?,
            runtime_id,
            env_id,
            state_path,
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
    let runtime = resolve_sglang_runtime(None);
    let installed = runtime.is_ok();
    let mut notes = Vec::new();
    if cfg!(windows) {
        notes.push(windows_unsupported_message().to_owned());
    } else if let Err(error) = runtime.as_ref() {
        notes.push(error.to_string());
    }
    let runtime = runtime.ok();
    if let Some(runtime) = runtime.as_ref() {
        notes.push(format!(
            "SGLang launcher resolved from {}; no CPU fallback is used",
            runtime.source
        ));
    }

    DetectResponse {
        installed,
        env_id: runtime.as_ref().map(|runtime| runtime.env_id.clone()),
        runtime_kind: Some("external_sglang".to_owned()),
        runtime_executable: runtime
            .as_ref()
            .map(|runtime| runtime.command.display().to_string()),
        managed_env: Some(runtime.as_ref().is_some_and(runtime_is_managed)),
        python_version: runtime
            .as_ref()
            .and_then(|runtime| runtime.version.as_ref())
            .map(|version| format!("SGLang {version}")),
        torch_version: None,
        transformers_version: None,
        available_devices: vec![EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: installed && !cfg!(windows),
            reason: if cfg!(windows) {
                Some(windows_unsupported_message().to_owned())
            } else if installed {
                None
            } else {
                Some("SGLang is not installed in a Linux/WSL ROCm Python environment".to_owned())
            },
        }],
        capabilities: capabilities(),
        notes,
    }
}

fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        cpu: false,
        rocm_gpu: !cfg!(windows),
        openai_compatible: true,
        tool_calling: false,
        quantized_models: "sglang-supported".to_owned(),
        reasoning_parser: false,
    }
}

fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    let runtime = resolve_sglang_runtime(Some(&request.runtime_id))?;
    let env_path = runtime
        .command
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    Ok(InstallResponse {
        env_id: runtime.env_id.clone(),
        env_path: env_path.display().to_string(),
        python_executable: runtime
            .python_executable
            .as_ref()
            .unwrap_or(&runtime.command)
            .display()
            .to_string(),
        runtime_kind: Some("external_sglang".to_owned()),
        runtime_executable: Some(runtime.command.display().to_string()),
        managed_env: Some(runtime_is_managed(&runtime)),
        installed_packages: vec![format!(
            "sglang{}",
            runtime
                .version
                .as_deref()
                .map(|version| format!("=={version}"))
                .unwrap_or_default()
        )],
        capabilities: capabilities(),
        lock_hash: runtime_lock_hash(&runtime),
        warnings: sglang_runtime_warnings(&runtime),
    })
}

fn runtime_is_managed(runtime: &SglangRuntime) -> bool {
    runtime.source.starts_with("managed_runtime_manifest")
}

fn sglang_runtime_warnings(runtime: &SglangRuntime) -> Vec<String> {
    let runtime_scope = if runtime_is_managed(runtime) {
        "rocm-cli records this SGLang launcher from a managed TheRock runtime; it does not pip install SGLang automatically"
    } else {
        "rocm-cli records this as an external SGLang runtime; it does not pip install SGLang automatically"
    };
    vec![
        runtime_scope.to_owned(),
        "SGLang serving remains ROCm GPU required; no CPU fallback is used".to_owned(),
    ]
}

fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    let device_policy = normalize_sglang_device_policy(request.device_policy)?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    Ok(ResolveModelResponse {
        canonical_model_id: request.model_ref,
        task: "text-generation".to_owned(),
        source: "huggingface_or_local".to_owned(),
        revision: "main".to_owned(),
        loader: "sglang".to_owned(),
        trust_remote_code: false,
        chat_template_mode: "engine_default".to_owned(),
        dtype: "auto".to_owned(),
        device_policy,
        estimated_memory: "engine-reported".to_owned(),
        launch_defaults: json!({
            "endpoint_mode": "openai",
            "host": DEFAULT_HOST,
            "port": DEFAULT_LOCAL_PORT
        }),
        engine_recipe,
        warnings: vec![
            "SGLang is treated as a ROCm GPU engine in rocm-cli; select another engine explicitly for CPU serving".to_owned(),
        ],
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

fn launch_service(request: LaunchRequest) -> Result<LaunchResponse> {
    let device_policy = normalize_sglang_device_policy(request.device_policy)?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let runtime = resolve_sglang_runtime(request.runtime_id.as_deref())?;
    let state_path = AppPaths::discover()?
        .engine_state_dir(ENGINE_NAME)
        .join(format!("{}.json", request.service_id));
    let serve_request = ServeHttpRequest {
        service_id: request.service_id.clone(),
        model_ref: request.model_ref.clone(),
        host: request.host.clone(),
        port: request.port,
        device_policy,
        gpu_indices: rocm_engine_protocol::launch_gpu_indices(request.gpu_selection.as_ref()),
        runtime_id: request.runtime_id.clone(),
        env_id: request.env_id.clone(),
        state_path: state_path.clone(),
        engine_recipe,
    };
    let log_path = AppPaths::discover()?
        .engine_logs_dir(ENGINE_NAME)
        .join(format!("{}.log", request.service_id));
    let child = spawn_sglang_server(&serve_request, &runtime, Some(&log_path))?;
    let pid = child.id();
    write_running_state(&serve_request, &runtime, pid)?;
    Ok(LaunchResponse {
        service_id: request.service_id,
        pid,
        endpoint_url: endpoint_url(&request.host, request.port),
        log_path: log_path.display().to_string(),
        state_path: state_path.display().to_string(),
    })
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
    engine_recipe: Option<EngineRecipeHint>,
}

fn serve_http(request: ServeHttpRequest) -> Result<()> {
    let runtime = resolve_sglang_runtime(request.runtime_id.as_deref())?;
    let mut child = spawn_sglang_server(&request, &runtime, None)?;
    write_running_state(&request, &runtime, child.id())?;
    let status = child.wait().context("failed waiting for SGLang server")?;
    write_terminal_state(
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
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn spawn_sglang_server(
    request: &ServeHttpRequest,
    runtime: &SglangRuntime,
    log_path: Option<&Path>,
) -> Result<std::process::Child> {
    require_nonempty(&request.service_id, "service_id")?;
    require_nonempty(&request.model_ref, "model_ref")?;
    if !matches!(request.device_policy, DevicePolicy::GpuRequired) {
        bail!("SGLang launch requires ROCm GPU execution; no CPU fallback is used");
    }

    if let Some(parent) = request.state_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    if let Some(log_path) = log_path
        && let Some(parent) = log_path.parent()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut command = ProcessCommand::new(&runtime.command);
    command
        .args(sglang_server_args(
            runtime.launcher,
            &request.model_ref,
            &request.host,
            request.port,
        ))
        .args(engine_recipe_launch_args(request.engine_recipe.as_ref()))
        .stdin(Stdio::null());
    apply_therock_env(&mut command, runtime)?;
    rocm_engine_protocol::apply_gpu_visibility(&mut command, &request.gpu_indices);
    if let Some(log_path) = log_path {
        let log = fs::File::create(log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        command.stdout(Stdio::from(
            log.try_clone().context("failed to clone log handle")?,
        ));
        command.stderr(Stdio::from(log));
    }

    command.spawn().with_context(|| {
        format!(
            "failed to spawn SGLang command {}",
            runtime.command.display()
        )
    })
}

fn sglang_server_args(
    launcher: SglangLauncher,
    model_ref: &str,
    host: &str,
    port: u16,
) -> Vec<String> {
    let mut args = Vec::new();
    match launcher {
        SglangLauncher::Command => {
            args.push("serve".to_owned());
        }
        SglangLauncher::PythonModule => {
            args.push("-m".to_owned());
            args.push("sglang.launch_server".to_owned());
        }
    }
    args.extend([
        "--model-path".to_owned(),
        model_ref.to_owned(),
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "--attention-backend".to_owned(),
        "triton".to_owned(),
    ]);
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
        device: if state.is_some() {
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
    let stopped = match state.as_ref().and_then(pid_from_state) {
        Some(pid) => terminate_pid(pid, request.force),
        None => false,
    };
    if stopped {
        write_terminal_state(&files.state_path, "stopped")?;
    }
    Ok(StopResponse {
        stopped,
        graceful: stopped && !request.force,
    })
}

fn resolve_sglang_runtime(runtime_id: Option<&str>) -> Result<SglangRuntime> {
    if cfg!(windows) {
        bail!("{}", windows_unsupported_message());
    }

    if let Some(command) = std::env::var_os("ROCM_CLI_SGLANG_COMMAND")
        .or_else(|| std::env::var_os("SGLANG_COMMAND"))
        .map(PathBuf::from)
    {
        let command = resolve_command_path(&command)?;
        return Ok(SglangRuntime {
            runtime_id: runtime_id.unwrap_or("external-sglang").to_owned(),
            env_id: "external-sglang-command".to_owned(),
            command,
            launcher: SglangLauncher::Command,
            python_executable: None,
            version: None,
            source: "environment command".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
        });
    }

    if let Some(python) = std::env::var_os("ROCM_CLI_SGLANG_PYTHON")
        .or_else(|| std::env::var_os("SGLANG_PYTHON"))
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return runtime_from_python(
            python,
            runtime_id.unwrap_or("external-sglang-python"),
            "environment python",
            None,
            None,
            Vec::new(),
            Vec::new(),
        );
    }

    if let Some(runtime) = resolve_managed_runtime(runtime_id)? {
        return Ok(runtime);
    }

    if let Some(command) = find_command_on_path("sglang") {
        return Ok(SglangRuntime {
            runtime_id: runtime_id.unwrap_or("external-sglang-path").to_owned(),
            env_id: "external-sglang-path".to_owned(),
            command,
            launcher: SglangLauncher::Command,
            python_executable: None,
            version: None,
            source: "PATH".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
        });
    }

    bail!(
        "SGLang is not installed in a Linux/WSL ROCm Python environment. Install/build SGLang against a ROCm-capable Python environment, then set ROCM_CLI_SGLANG_COMMAND, set ROCM_CLI_SGLANG_PYTHON, or install it into the active rocm-cli TheRock runtime. Native Windows is skipped; no CPU fallback is used."
    )
}

fn runtime_from_python(
    python: PathBuf,
    runtime_id: &str,
    source: &str,
    sdk_root: Option<PathBuf>,
    sdk_bin: Option<PathBuf>,
    sdk_bin_paths: Vec<PathBuf>,
    sdk_library_paths: Vec<PathBuf>,
) -> Result<SglangRuntime> {
    let (command, launcher, version) = if let Some(command) = sglang_command_from_python(&python) {
        (
            command,
            SglangLauncher::Command,
            probe_sglang_version(&python).ok().flatten(),
        )
    } else {
        let version = probe_sglang_version(&python)
            .with_context(|| format!("SGLang package not found in {}", python.display()))?
            .unwrap_or_else(|| "unknown".to_owned());
        (python.clone(), SglangLauncher::PythonModule, Some(version))
    };
    Ok(SglangRuntime {
        runtime_id: runtime_id.to_owned(),
        env_id: format!("external-sglang-{}", stable_id_component(runtime_id)),
        command,
        launcher,
        python_executable: Some(python),
        version,
        source: source.to_owned(),
        sdk_root,
        sdk_bin,
        sdk_bin_paths,
        sdk_library_paths,
    })
}

fn resolve_managed_runtime(runtime_id: Option<&str>) -> Result<Option<SglangRuntime>> {
    let paths = AppPaths::discover()?;
    let registry = paths.data_dir.join("runtimes").join("registry");
    if !registry.is_dir() {
        return Ok(None);
    }
    let mut manifests = Vec::new();
    for entry in
        fs::read_dir(&registry).with_context(|| format!("failed to read {}", registry.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let Ok(manifest) = serde_json::from_slice::<TheRockRuntimeManifest>(&bytes) else {
            continue;
        };
        if !runtime_matches(&manifest, runtime_id) {
            continue;
        }
        manifests.push((manifest.installed_at_unix_ms.unwrap_or(0), manifest));
    }
    manifests.sort_by_key(|(installed_at, _)| std::cmp::Reverse(*installed_at));

    for (_, manifest) in manifests {
        let Some(python) = manifest
            .python_executable
            .clone()
            .filter(|path| path.is_file())
        else {
            continue;
        };
        let runtime_id = manifest
            .runtime_id
            .as_deref()
            .unwrap_or("therock-sglang-runtime")
            .to_owned();
        let source = manifest.runtime_key.as_deref().map_or_else(
            || "managed_runtime_manifest".to_owned(),
            |key| format!("managed_runtime_manifest:{key}"),
        );
        let (sdk_root, sdk_bin, sdk_bin_paths, sdk_library_paths) = manifest
            .rocm_sdk
            .as_ref()
            .filter(|probe| probe.import_ok)
            .map_or((None, None, Vec::new(), Vec::new()), |probe| {
                (
                    probe.root_path.clone(),
                    probe.bin_path.clone(),
                    probe.bin_paths.clone(),
                    probe.library_paths.clone(),
                )
            });
        if let Ok(runtime) = runtime_from_python(
            python,
            &runtime_id,
            &source,
            sdk_root,
            sdk_bin,
            sdk_bin_paths,
            sdk_library_paths,
        ) {
            return Ok(Some(runtime));
        }
    }
    Ok(None)
}

fn runtime_matches(manifest: &TheRockRuntimeManifest, requested: Option<&str>) -> bool {
    let Some(requested) = requested.map(str::trim).filter(|value| !value.is_empty()) else {
        return true;
    };
    let requested = requested.to_ascii_lowercase();
    if requested == "external" || requested == "external-sglang" {
        return false;
    }
    for candidate in [
        manifest.runtime_id.as_deref(),
        manifest.runtime_key.as_deref(),
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

fn normalize_sglang_device_policy(policy: Option<DevicePolicy>) -> Result<DevicePolicy> {
    match policy.unwrap_or(DevicePolicy::GpuRequired) {
        DevicePolicy::GpuRequired => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::GpuPreferred => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::CpuOnly => {
            bail!("SGLang adapter is ROCm GPU-only in rocm-cli; no CPU fallback is used")
        }
    }
}

fn parse_device_policy_arg(policy: Option<&str>) -> Result<DevicePolicy> {
    match policy.unwrap_or("gpu_required") {
        "gpu" | "gpu_required" => Ok(DevicePolicy::GpuRequired),
        "gpu_preferred" => Ok(DevicePolicy::GpuPreferred),
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        other => bail!("unsupported device policy: {other}"),
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

fn sglang_command_from_python(python: &Path) -> Option<PathBuf> {
    let dir = python.parent()?;
    candidate_command_names("sglang")
        .into_iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

fn find_command_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for candidate in candidate_command_names(name) {
            let path = dir.join(candidate);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn resolve_command_path(command: &Path) -> Result<PathBuf> {
    if command.components().count() > 1 || command.is_absolute() {
        if command.is_file() {
            return Ok(command.to_path_buf());
        }
        bail!(
            "configured SGLang command is not a file: {}",
            command.display()
        );
    }
    find_command_on_path(&command.display().to_string()).with_context(|| {
        format!(
            "configured SGLang command `{}` was not found on PATH",
            command.display()
        )
    })
}

fn candidate_command_names(name: &str) -> Vec<String> {
    if cfg!(windows) {
        vec![
            format!("{name}.exe"),
            format!("{name}.cmd"),
            name.to_owned(),
        ]
    } else {
        vec![name.to_owned()]
    }
}

fn probe_sglang_version(python: &Path) -> Result<Option<String>> {
    let script = r#"import importlib.metadata, importlib.util, json
spec = importlib.util.find_spec("sglang")
version = None
if spec is not None:
    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
print(json.dumps({"present": spec is not None, "version": version}))
"#;
    let output = ProcessCommand::new(python)
        .arg("-c")
        .arg(script)
        .output()
        .with_context(|| format!("failed to probe SGLang with {}", python.display()))?;
    if !output.status.success() {
        bail!(
            "SGLang probe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let value: Value =
        serde_json::from_slice(&output.stdout).context("invalid SGLang probe JSON")?;
    if value
        .get("present")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        Ok(value
            .get("version")
            .and_then(Value::as_str)
            .map(str::to_owned))
    } else {
        bail!("Python environment does not contain the SGLang package")
    }
}

fn apply_therock_env(command: &mut ProcessCommand, runtime: &SglangRuntime) -> Result<()> {
    let Some(root) = runtime.sdk_root.as_ref() else {
        return Ok(());
    };
    let bin = runtime.sdk_bin.as_ref();
    command
        .env("ROCM_SDK_ROOT", root)
        .env("ROCM_PATH", root)
        .env("ROCM_HOME", root)
        .env("HIP_PATH", root)
        .env("ROCM_CLI_THEROCK_RUNTIME_ID", &runtime.runtime_id);
    if std::env::var_os("GPU_ARCHS").is_none()
        && let Some(arch) = sglang_rocm_gpu_arch(runtime)
    {
        command
            .env("GPU_ARCHS", &arch)
            .env("AMDGPU_TARGET", &arch)
            .env("PYTORCH_ROCM_ARCH", &arch);
    }
    if let Some(bin) = bin {
        command.env("ROCM_CLI_THEROCK_SDK_BIN", bin).env(
            "PATH",
            prepend_path_entries(&runtime_bin_paths(runtime), std::env::var_os("PATH"))?,
        );
    } else if !runtime.sdk_bin_paths.is_empty() {
        command.env(
            "PATH",
            prepend_path_entries(&runtime_bin_paths(runtime), std::env::var_os("PATH"))?,
        );
    }
    if !cfg!(windows) {
        command.env(
            "LD_LIBRARY_PATH",
            prepend_path_entries(
                &therock_library_path_entries(runtime),
                std::env::var_os("LD_LIBRARY_PATH"),
            )?,
        );
    }
    Ok(())
}

fn sglang_rocm_gpu_arch(runtime: &SglangRuntime) -> Option<String> {
    detect_sglang_rocm_gpu_arch_from_sdk(runtime)
        .or_else(|| sglang_rocm_gpu_arch_from_text(&runtime.runtime_id))
        .or_else(|| sglang_rocm_gpu_arch_from_text(&runtime.source))
}

fn detect_sglang_rocm_gpu_arch_from_sdk(runtime: &SglangRuntime) -> Option<String> {
    for bin_dir in runtime_bin_paths(runtime) {
        let tool = bin_dir.join(if cfg!(windows) {
            "rocm_agent_enumerator.exe"
        } else {
            "rocm_agent_enumerator"
        });
        if !tool.is_file() {
            continue;
        }
        let Ok(output) = ProcessCommand::new(&tool).output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(arch) = sglang_rocm_gpu_arch_from_text(&stdout) {
            return Some(arch);
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if let Some(arch) = sglang_rocm_gpu_arch_from_text(&stderr) {
            return Some(arch);
        }
    }
    None
}

fn sglang_rocm_gpu_arch_from_text(text: &str) -> Option<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .find_map(normalize_sglang_rocm_gpu_arch)
}

fn normalize_sglang_rocm_gpu_arch(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();
    match value.as_str() {
        "gfx90a" | "gfx940" | "gfx941" | "gfx942" | "gfx950" => Some(value),
        value if value.starts_with("gfx942") || value.starts_with("gfx94") => {
            Some("gfx942".to_owned())
        }
        value if value.starts_with("gfx950") => Some("gfx950".to_owned()),
        _ => None,
    }
}

fn runtime_bin_paths(runtime: &SglangRuntime) -> Vec<PathBuf> {
    let mut entries = Vec::new();
    if let Some(bin) = runtime.sdk_bin.as_ref() {
        entries.push(bin.clone());
    }
    entries.extend(runtime.sdk_bin_paths.iter().cloned());
    dedupe_paths(entries)
}

fn therock_library_path_entries(runtime: &SglangRuntime) -> Vec<PathBuf> {
    let Some(root) = runtime.sdk_root.as_ref() else {
        return dedupe_paths(runtime.sdk_library_paths.clone());
    };
    let mut entries = runtime.sdk_library_paths.clone();
    entries.extend([
        root.join("lib"),
        root.join("lib64"),
        root.join("lib").join("rocm_sysdeps").join("lib"),
    ]);
    if cfg!(target_os = "linux") {
        let wsl_dxcore_lib = PathBuf::from("/usr/lib/wsl/lib");
        if wsl_dxcore_lib.is_dir() {
            entries.push(wsl_dxcore_lib);
        }
    }
    dedupe_paths(entries)
}

fn dedupe_paths(entries: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for entry in entries {
        if !entry.as_os_str().is_empty() && !deduped.iter().any(|seen| seen == &entry) {
            deduped.push(entry);
        }
    }
    deduped
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
    std::env::join_paths(parts).context("failed to compose runtime path")
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

fn write_running_state(
    request: &ServeHttpRequest,
    runtime: &SglangRuntime,
    pid: u32,
) -> Result<()> {
    write_state(
        &request.state_path,
        &json!({
            "service_id": request.service_id,
            "engine": ENGINE_NAME,
            "status": "running",
            "pid": pid,
            "model_ref": request.model_ref,
            "host": request.host,
            "port": request.port,
            "endpoint_url": endpoint_url(&request.host, request.port),
            "device_policy": "gpu_required",
            "runtime_id": runtime.runtime_id,
            "requested_runtime_id": request.runtime_id,
            "env_id": request.env_id.as_deref().unwrap_or(runtime.env_id.as_str()),
            "runtime_executable": runtime.command,
            "server_pid": pid,
            "engine_recipe": request.engine_recipe,
            "engine_recipe_required_flags": engine_recipe_launch_args(request.engine_recipe.as_ref()),
            "therock_runtime_env": therock_runtime_env_state(runtime),
            "started_at_unix_ms": current_unix_millis()
        }),
    )
}

fn therock_runtime_env_state(runtime: &SglangRuntime) -> Option<Value> {
    let root = runtime.sdk_root.as_ref()?;
    Some(json!({
        "runtime_id": runtime.runtime_id,
        "env_id": runtime.env_id,
        "root": root.display().to_string(),
        "bin": runtime.sdk_bin.as_ref().map(|path| path.display().to_string()),
        "bin_paths": runtime_bin_paths(runtime)
            .into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>(),
        "library_paths": therock_library_path_entries(runtime)
            .into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>(),
        "source": runtime.source,
    }))
}

fn write_terminal_state(state_path: &Path, status: &str) -> Result<()> {
    let mut state = read_service_state(state_path).unwrap_or_else(|_| json!({}));
    if let Some(object) = state.as_object_mut() {
        object.insert("status".to_owned(), Value::String(status.to_owned()));
        object.insert(
            "stopped_at_unix_ms".to_owned(),
            Value::from(current_unix_millis() as u64),
        );
    }
    write_state(state_path, &state)
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
        serde_json::to_vec_pretty(value).context("failed to serialize SGLang state")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn endpoint_url(host: &str, port: u16) -> String {
    format!("{}/v1", format_http_base_url(host, port))
}

fn endpoint_url_from_state(state: &Value) -> Option<String> {
    value_string(state, "endpoint_url").or_else(|| {
        let host = value_string(state, "host")?;
        let port = state.get("port")?.as_u64()?;
        let port = u16::try_from(port).ok()?;
        Some(endpoint_url(&host, port))
    })
}

fn query_loaded_model_endpoint(endpoint_url: &str, model_ref: Option<&str>) -> Result<bool> {
    openai_models_endpoint_has_model(
        endpoint_url,
        model_ref,
        Duration::from_millis(HEALTHCHECK_TIMEOUT_MS),
    )
}

fn pid_from_state(state: &Value) -> Option<u32> {
    state
        .get("pid")?
        .as_u64()
        .and_then(|pid| pid.try_into().ok())
}

fn terminate_pid(pid: u32, _force: bool) -> bool {
    rocm_core::terminate_process(pid).is_ok()
}

fn tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(tail_lines_from_text(&text, limit))
}

fn tail_lines_from_text(text: &str, limit: usize) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }
    let mut lines = text
        .lines()
        .rev()
        .take(limit)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    lines.reverse();
    lines
}

fn value_string(value: &Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(str::to_owned)
}

fn stable_id_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn runtime_lock_hash(runtime: &SglangRuntime) -> String {
    let mut hasher = DefaultHasher::new();
    runtime.runtime_id.hash(&mut hasher);
    runtime.command.hash(&mut hasher);
    runtime.version.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn current_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

const fn windows_unsupported_message() -> &'static str {
    "SGLang ROCm serving is supported by rocm-cli only on Linux/WSL; native Windows SGLang is skipped. No CPU fallback is used."
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
            required_flags: vec!["--reasoning-parser".to_owned(), "qwen3".to_owned()],
            parser_settings: std::collections::BTreeMap::default(),
            preferred_endpoint: None,
            unsupported_combinations: Vec::new(),
            notes: vec!["test recipe".to_owned()],
        }
    }

    #[test]
    fn parse_gpu_args_map_to_indices() {
        assert_eq!(parse_gpu_indices_arg(None).unwrap(), Vec::<u32>::new());
        assert_eq!(
            parse_gpu_indices_arg(Some("auto")).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(parse_gpu_indices_arg(Some("3")).unwrap(), vec![3]);
        assert!(parse_gpu_selection_arg(Some("x")).is_err());
        assert!(parse_gpu_selection_arg(Some("1,3")).is_err());
    }

    #[test]
    fn cpu_policy_is_rejected_without_fallback() {
        let error = normalize_sglang_device_policy(Some(DevicePolicy::CpuOnly))
            .expect_err("SGLang CPU policy must fail");
        assert!(error.to_string().contains("no CPU fallback is used"));
    }

    #[test]
    fn gpu_preferred_resolves_to_gpu_required() -> Result<()> {
        assert_eq!(
            normalize_sglang_device_policy(Some(DevicePolicy::GpuPreferred))?,
            DevicePolicy::GpuRequired
        );
        Ok(())
    }

    #[test]
    fn engine_recipe_launch_args_forward_required_flags() {
        let hint = test_engine_recipe(ENGINE_NAME, ENGINE_RECIPE_CONTRACT_VERSION);

        assert_eq!(
            engine_recipe_launch_args(Some(&hint)),
            vec!["--reasoning-parser".to_owned(), "qwen3".to_owned()]
        );
    }

    #[test]
    fn server_args_support_command_and_python_module_launchers() {
        assert_eq!(
            sglang_server_args(SglangLauncher::Command, "qwen", "127.0.0.1", 30000),
            vec![
                "serve",
                "--model-path",
                "qwen",
                "--host",
                "127.0.0.1",
                "--port",
                "30000",
                "--attention-backend",
                "triton"
            ]
        );
        assert_eq!(
            sglang_server_args(SglangLauncher::PythonModule, "qwen", "0.0.0.0", 30001),
            vec![
                "-m",
                "sglang.launch_server",
                "--model-path",
                "qwen",
                "--host",
                "0.0.0.0",
                "--port",
                "30001",
                "--attention-backend",
                "triton"
            ]
        );
    }

    #[test]
    fn managed_runtime_allows_sibling_command_without_shared_python_package() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-sglang-runtime-test-{}",
            current_unix_millis()
        ));
        fs::create_dir_all(&root)?;
        let python = root.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let command = root.join(
            candidate_command_names("sglang")
                .into_iter()
                .next()
                .expect("candidate command name"),
        );
        fs::write(&python, "")?;
        fs::write(&command, "")?;

        let runtime = runtime_from_python(
            python.clone(),
            "therock-release:gfx120X-all",
            "managed_runtime_manifest:test",
            None,
            None,
            Vec::new(),
            Vec::new(),
        )?;

        assert_eq!(runtime.command, command);
        assert_eq!(runtime.launcher, SglangLauncher::Command);
        assert_eq!(runtime.python_executable.as_deref(), Some(python.as_path()));
        assert_eq!(runtime.version, None);

        fs::remove_dir_all(root)?;
        Ok(())
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
    fn endpoint_url_falls_back_to_host_and_port() {
        let state = json!({
            "host": "127.0.0.1",
            "port": 12345
        });
        assert_eq!(
            endpoint_url_from_state(&state),
            Some("http://127.0.0.1:12345/v1".to_owned())
        );
        let ipv6_state = json!({
            "host": "::1",
            "port": 12345
        });
        assert_eq!(
            endpoint_url_from_state(&ipv6_state),
            Some("http://[::1]:12345/v1".to_owned())
        );
    }

    #[test]
    fn resolve_model_echoes_matching_engine_recipe() -> Result<()> {
        let hint = test_engine_recipe(ENGINE_NAME, ENGINE_RECIPE_CONTRACT_VERSION);
        let response = resolve_model_response(ResolveModelRequest {
            model_ref: "Qwen/Qwen3.5-4B".to_owned(),
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
            model_ref: "Qwen/Qwen3.5-4B".to_owned(),
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
            model_ref: "Qwen/Qwen3.5-4B".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(test_engine_recipe(ENGINE_NAME, "999.0.0")),
        })
        .expect_err("unsupported recipe contract should fail");

        assert!(error.to_string().contains("unsupported"));
    }

    #[test]
    fn tail_lines_returns_suffix() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "rocm-sglang-tail-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, "a\nb\nc\n")?;
        let lines = tail_lines(&path, 2)?;
        fs::remove_file(path).ok();
        assert_eq!(lines, vec!["b".to_owned(), "c".to_owned()]);
        Ok(())
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
                    "model_ref": "Qwen/Qwen3.5-4B",
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
    fn therock_library_path_entries_include_sysdeps_for_hip_apps() {
        let root = PathBuf::from(if cfg!(windows) {
            r"C:\rocm-sdk"
        } else {
            "/tmp/rocm-sdk"
        });
        let runtime = SglangRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-sglang-therock".to_owned(),
            command: PathBuf::from("sglang"),
            launcher: SglangLauncher::Command,
            python_executable: None,
            version: None,
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(root.clone()),
            sdk_bin: Some(root.join("bin")),
            sdk_bin_paths: vec![root.join("runtime").join("bin")],
            sdk_library_paths: vec![root.join("runtime").join("lib")],
        };
        let entries = therock_library_path_entries(&runtime);
        assert!(entries.contains(&root.join("runtime").join("lib")));
        assert!(entries.contains(&root.join("lib")));
        assert!(
            entries
                .iter()
                .any(|entry| entry.ends_with(Path::new("lib").join("rocm_sysdeps").join("lib")))
        );
    }

    #[test]
    fn sglang_rocm_gpu_arch_maps_runtime_families_for_aiter() {
        assert_eq!(
            sglang_rocm_gpu_arch_from_text("therock-release:gfx94X-dcgpu"),
            Some("gfx942".to_owned())
        );
        assert_eq!(
            sglang_rocm_gpu_arch_from_text("gfx942:sramecc+:xnack-"),
            Some("gfx942".to_owned())
        );
        assert_eq!(
            sglang_rocm_gpu_arch_from_text("release-pip-gfx950-dcgpu"),
            Some("gfx950".to_owned())
        );
        assert_eq!(sglang_rocm_gpu_arch_from_text("gfx120X-all"), None);
    }

    #[test]
    fn managed_env_reflects_managed_runtime_manifest_source() {
        let runtime = SglangRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-sglang-therock".to_owned(),
            command: PathBuf::from(if cfg!(windows) {
                r"C:\venv\Scripts\sglang.exe"
            } else {
                "/home/user/.venv/bin/sglang"
            }),
            launcher: SglangLauncher::Command,
            python_executable: None,
            version: Some("0.5.12".to_owned()),
            source: "managed_runtime_manifest:sglang-source-pip-gfx120x-all".to_owned(),
            sdk_root: Some(PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk"
            } else {
                "/home/user/.venv/lib/python/site-packages/_rocm_sdk_devel"
            })),
            sdk_bin: Some(PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk\bin"
            } else {
                "/home/user/.venv/lib/python/site-packages/_rocm_sdk_devel/bin"
            })),
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
        };

        assert!(runtime_is_managed(&runtime));
        assert!(
            sglang_runtime_warnings(&runtime)
                .join("\n")
                .contains("managed TheRock runtime")
        );

        let external = SglangRuntime {
            source: "environment command".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            ..runtime
        };
        assert!(!runtime_is_managed(&external));
        assert!(
            sglang_runtime_warnings(&external)
                .join("\n")
                .contains("external SGLang runtime")
        );
    }

    #[test]
    fn running_state_records_managed_therock_env_for_gpu_verification() -> Result<()> {
        let state_path = std::env::temp_dir().join(format!(
            "rocm-sglang-state-{}-{}.json",
            std::process::id(),
            current_unix_millis()
        ));
        let request = ServeHttpRequest {
            service_id: "sglang-test".to_owned(),
            model_ref: "Qwen/Qwen2.5-0.5B-Instruct".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: DevicePolicy::GpuRequired,
            gpu_indices: Vec::new(),
            runtime_id: Some("runtime-key-gfx120x".to_owned()),
            env_id: None,
            state_path: state_path.clone(),
            engine_recipe: None,
        };
        let runtime = SglangRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-sglang-therock".to_owned(),
            command: PathBuf::from(if cfg!(windows) {
                r"C:\venv\Scripts\sglang.exe"
            } else {
                "/home/user/.venv/bin/sglang"
            }),
            launcher: SglangLauncher::Command,
            python_executable: None,
            version: Some("0.5.12".to_owned()),
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk"
            } else {
                "/home/user/.venv/lib/python/site-packages/_rocm_sdk_devel"
            })),
            sdk_bin: Some(PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk\bin"
            } else {
                "/home/user/.venv/lib/python/site-packages/_rocm_sdk_devel/bin"
            })),
            sdk_bin_paths: vec![PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk\extra-bin"
            } else {
                "/home/user/.venv/lib/python/site-packages/_rocm_sdk_libraries/bin"
            })],
            sdk_library_paths: vec![PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk\extra-lib"
            } else {
                "/home/user/.venv/lib/python/site-packages/_rocm_sdk_libraries/lib"
            })],
        };

        write_running_state(&request, &runtime, 12345)?;
        let state = read_service_state(&state_path)?;
        fs::remove_file(&state_path).ok();

        assert_eq!(state.get("server_pid").and_then(Value::as_u64), Some(12345));
        assert_eq!(
            state.get("runtime_id").and_then(Value::as_str),
            Some("therock-release:gfx120X-all")
        );
        assert_eq!(
            state.get("requested_runtime_id").and_then(Value::as_str),
            Some("runtime-key-gfx120x")
        );
        let runtime_env = state
            .get("therock_runtime_env")
            .expect("runtime env should be recorded");
        assert_eq!(
            runtime_env.get("runtime_id").and_then(Value::as_str),
            Some("therock-release:gfx120X-all")
        );
        assert!(
            runtime_env
                .get("root")
                .and_then(Value::as_str)
                .is_some_and(|root| root.contains("rocm"))
        );
        assert!(
            runtime_env
                .get("bin_paths")
                .and_then(Value::as_array)
                .is_some_and(|paths| paths.len() >= 2)
        );
        assert!(
            runtime_env
                .get("library_paths")
                .and_then(Value::as_array)
                .is_some_and(|paths| !paths.is_empty())
        );
        Ok(())
    }
}
