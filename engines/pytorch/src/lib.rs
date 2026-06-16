use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_PORT, ModelRecipeRecord, detect_host_therock_family, ensure_uv_binary,
    extract_first_gfx_token, format_host_port, format_http_base_url, interactive_terminal,
    normalize_runtime_path_for_host, normalize_runtime_path_text_for_host,
    normalize_therock_family, require_nonempty,
    resolve_model_recipe as resolve_shared_model_recipe, runtime_is_windows, unix_time_millis,
    uv_command_env, uv_pip_freeze_args, uv_pip_install_base, uv_venv_args,
};
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, DevicePolicy, ENGINE_RECIPE_CONTRACT_VERSION, EndpointRequest,
    EndpointResponse, EngineCapabilities, EngineDeviceAvailability, EngineMethod, EngineRecipeHint,
    EngineRequestEnvelope, EngineResponseEnvelope, HealthcheckRequest, HealthcheckResponse,
    InstallRequest, InstallResponse, LaunchRequest, LaunchResponse, LogsRequest, LogsResponse,
    ResolveModelRequest, ResolveModelResponse, StopRequest, StopResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ENGINE_NAME: &str = "pytorch";
const DEFAULT_RUNTIME_ID: &str = "therock-release";
const THEROCK_SIMPLE_INDEX_BASE: &str = "https://rocm.nightlies.amd.com/v2";
const PYTHON_WORKER_SOURCE: &str = include_str!("python_worker.py");
const ENGINE_DEPENDENCIES: &[&str] = &[
    "fastapi",
    "uvicorn",
    "pydantic",
    "transformers<5",
    "safetensors",
    "tokenizers",
    "huggingface_hub<1",
    "jinja2",
];
const THEROCK_TORCH_PACKAGES: &[&str] = &["torch", "torchvision", "torchaudio"];
const TORCH_STACK_DEPENDENCIES: &[&str] = &["accelerate"];
const DEFAULT_LOG_TAIL_LINES: usize = 200;
const MAX_LOG_TAIL_LINES: usize = 1000;
const HEALTHCHECK_TIMEOUT_MS: u64 = 500;
const REMOVE_ENV_RETRIES: usize = 8;
const REMOVE_ENV_RETRY_DELAY: Duration = Duration::from_millis(250);
const KNOWN_THEROCK_FAMILIES: &[&str] = &[
    "gfx94X-dcgpu",
    "gfx950-dcgpu",
    "gfx110X-all",
    "gfx1151",
    "gfx120X-all",
];

#[derive(Parser, Debug)]
#[command(
    name = "rocm-engine-pytorch",
    about = "rocm-cli PyTorch engine",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand, Debug)]
enum CommandKind {
    Detect,
    Capabilities,
    Install {
        #[arg(long, default_value = DEFAULT_RUNTIME_ID)]
        runtime_id: String,
        #[arg(long)]
        python_version: Option<String>,
        #[arg(long)]
        reinstall: bool,
    },
    ResolveModel {
        model_ref: String,
        #[arg(long)]
        device_policy: Option<DevicePolicyArg>,
    },
    Launch {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long)]
        device_policy: Option<DevicePolicyArg>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
    },
    Stdio,
    #[command(hide = true)]
    ServeHttp {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long, default_value = "gpu_required")]
        device_policy: String,
        #[arg(long)]
        env_id: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        state_path: PathBuf,
        #[arg(long)]
        engine_recipe_json: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DevicePolicyArg {
    GpuRequired,
    GpuPreferred,
    CpuOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EngineEnvManifest {
    env_id: String,
    runtime_id: String,
    requested_python_version: Option<String>,
    python_launcher: String,
    python_executable: String,
    env_path: PathBuf,
    manifest_path: PathBuf,
    lock_path: PathBuf,
    installed_packages: Vec<String>,
    lock_hash: String,
    #[serde(default)]
    pip_cache_dir: Option<PathBuf>,
    #[serde(default)]
    therock_channel: Option<String>,
    #[serde(default)]
    therock_family: Option<String>,
    #[serde(default)]
    therock_index_url: Option<String>,
    #[serde(default)]
    therock_packages: Vec<String>,
    #[serde(default)]
    torch_runtime_probe: Option<TorchRuntimeProbe>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct PythonLauncher {
    program: String,
    args: Vec<String>,
    display: String,
}

#[derive(Debug, Clone, Copy)]
enum TheRockChannel {
    Release,
    Nightly,
}

impl TheRockChannel {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Nightly => "nightly",
        }
    }
}

#[derive(Debug, Clone)]
struct TheRockRuntimeRequest {
    channel: TheRockChannel,
    family_override: Option<String>,
}

#[derive(Debug, Clone)]
struct TheRockTorchResolution {
    channel: TheRockChannel,
    family: String,
    index_url: String,
    packages: Vec<String>,
    source: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeRegistryManifest {
    runtime_key: String,
    runtime_id: String,
    channel: String,
    format: String,
    family: String,
    index_url: Option<String>,
    selected_artifact_url: Option<String>,
    python_executable: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum ModelFamily {
    Generic,
    Qwen,
    Glm,
    Llama,
    Gpt2,
}

#[derive(Debug, Clone)]
struct ModelRecipe {
    canonical_model_id: String,
    task: &'static str,
    source: String,
    loader: &'static str,
    trust_remote_code: bool,
    chat_template_mode: &'static str,
    preferred_dtype: String,
    device_policy: DevicePolicy,
    estimated_memory: String,
    min_gpu_mem_gb: Option<u32>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TorchRuntimeProbe {
    #[serde(default)]
    import_ok: bool,
    #[serde(default)]
    torch_version: Option<String>,
    #[serde(default)]
    cuda_available: bool,
    #[serde(default)]
    device_count: u32,
    #[serde(default)]
    devices: Vec<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    rocm_sdk: Option<RocmSdkProbe>,
    #[serde(default)]
    torch_rocm_init: Option<TorchRocmInitProbe>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RocmSdkProbe {
    #[serde(default)]
    import_ok: bool,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    site_packages: Option<String>,
    #[serde(default)]
    default_target_family: Option<String>,
    #[serde(default)]
    available_target_families: Vec<String>,
    #[serde(default)]
    resolved_target_family: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TorchRocmInitProbe {
    #[serde(default)]
    present: bool,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    check_version: Option<String>,
    #[serde(default)]
    preload_shortnames: Vec<String>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ServiceRecordSnapshot {
    #[serde(default)]
    engine: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    supervisor_pid: Option<u32>,
    #[serde(default)]
    engine_pid: Option<u32>,
    #[serde(default)]
    log_path: Option<PathBuf>,
    #[serde(default)]
    engine_state_path: Option<PathBuf>,
    #[serde(default)]
    endpoint_url: Option<String>,
}

#[derive(Debug)]
struct ServiceFiles {
    record_path: PathBuf,
    record: Option<ServiceRecordSnapshot>,
    record_matches_engine: bool,
    state_path: PathBuf,
    log_path: PathBuf,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum PidTermination {
    Terminated,
    Failed,
}

#[tokio::main]
pub async fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Detect => {
            print_json(&detect_response())?;
        }
        CommandKind::Capabilities => {
            print_json(&capabilities())?;
        }
        CommandKind::Install {
            runtime_id,
            python_version,
            reinstall,
        } => {
            let response = install_response(InstallRequest {
                runtime_id,
                python_version,
                env_root: None,
                reinstall,
            })?;
            print_json(&response)?;
        }
        CommandKind::ResolveModel {
            model_ref,
            device_policy,
        } => {
            let response = resolve_model_response(ResolveModelRequest {
                model_ref,
                runtime_id: None,
                device_policy: device_policy.map(Into::into),
                recipe_override: None,
                engine_recipe: None,
            })?;
            print_json(&response)?;
        }
        CommandKind::Launch {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            runtime_id,
            env_id,
        } => {
            let response = launch_service(LaunchRequest {
                service_id,
                env_id,
                runtime_id,
                model_ref,
                host,
                port,
                device_policy: device_policy.map(Into::into),
                endpoint_mode: Some("openai".to_owned()),
                engine_recipe: None,
            })?;
            print_json(&response)?;
        }
        CommandKind::Stdio => {
            let envelope = read_request()?;
            let response = handle_envelope(envelope);
            print_json(&response)?;
        }
        CommandKind::ServeHttp {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            env_id,
            runtime_id,
            state_path,
            engine_recipe_json,
        } => {
            serve_http(
                service_id,
                model_ref,
                host,
                port,
                parse_device_policy(&device_policy)?,
                env_id,
                runtime_id,
                state_path,
                parse_engine_recipe_json(engine_recipe_json)?,
            )?;
        }
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
    env_id: Option<String>,
    runtime_id: Option<String>,
    state_path: PathBuf,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<()> {
    serve_http(
        service_id,
        model_ref,
        host,
        port,
        device_policy,
        env_id,
        runtime_id,
        state_path,
        engine_recipe,
    )
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

fn deserialize_and_respond<T, F, U>(
    payload: serde_json::Value,
    handler: F,
) -> EngineResponseEnvelope
where
    T: for<'de> Deserialize<'de>,
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
    use std::fmt::Write as _;
    let manifest = latest_env_manifest().ok().flatten();
    let installed = manifest.is_some();
    let detected_family = detect_host_therock_family();
    let env_id = manifest.as_ref().map(|value| value.env_id.clone());
    let python_version = manifest
        .as_ref()
        .and_then(|value| value.requested_python_version.clone())
        .or_else(|| Some(default_python_version().to_owned()));
    let transformers_version = manifest
        .as_ref()
        .and_then(|value| find_installed_package(&value.installed_packages, "transformers"));

    let mut notes = Vec::new();
    let torch_probe = manifest.as_ref().and_then(|manifest| {
        match probe_torch_runtime(&manifest.python_executable) {
            Ok(probe) => {
                if probe.import_ok {
                    notes.push(format!(
                        "torch probe: cuda_available={} device_count={}",
                        probe.cuda_available, probe.device_count
                    ));
                    if !probe.devices.is_empty() {
                        notes.push(format!("torch devices: {}", probe.devices.join(", ")));
                    }
                    if let Some(init) = probe.torch_rocm_init.as_ref()
                        && init.present
                    {
                        let version = init.check_version.as_deref().unwrap_or("<unknown>");
                        notes.push(format!(
                            "torch._rocm_init: check_version={} preload_count={}",
                            version,
                            init.preload_shortnames.len()
                        ));
                    }
                    if let Some(sdk) = probe.rocm_sdk.as_ref()
                        && sdk.import_ok
                    {
                        let version = sdk.version.as_deref().unwrap_or("<unknown>");
                        let family = sdk.resolved_target_family.as_deref().unwrap_or("<unknown>");
                        notes.push(format!(
                            "rocm_sdk: version={version} target_family={family}"
                        ));
                    }
                } else if let Some(error) = probe.error.as_deref() {
                    notes.push(format!("torch probe import failed: {error}"));
                }
                Some(probe)
            }
            Err(error) => {
                notes.push(format!("torch probe failed: {error}"));
                None
            }
        }
    });
    let torch_version = torch_probe
        .as_ref()
        .and_then(|probe| probe.torch_version.clone())
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|value| find_installed_package(&value.installed_packages, "torch"))
        });
    if let Some(manifest) = &manifest {
        notes.push(format!(
            "managed env detected at {}",
            manifest.env_path.display()
        ));
        if let Some(family) = manifest.therock_family.as_deref() {
            notes.push(format!("TheRock family: {family}"));
        }
        if let Some(channel) = manifest.therock_channel.as_deref() {
            notes.push(format!("TheRock channel: {channel}"));
        }
        if let Some(index_url) = manifest.therock_index_url.as_deref() {
            notes.push(format!("TheRock index: {index_url}"));
        }
        notes.extend(manifest.warnings.iter().cloned());
    } else {
        notes.push("no managed PyTorch envs found; run `rocm engines install pytorch`".to_owned());
    }

    let rocm_gpu_available = torch_probe
        .as_ref()
        .map_or_else(|| detected_family.is_some(), |probe| probe.cuda_available);
    let rocm_gpu_reason = match torch_probe.as_ref() {
        Some(probe) if probe.cuda_available => {
            let mut reason = format!("torch.cuda reports {} device(s)", probe.device_count);
            if !probe.devices.is_empty() {
                let _ = write!(reason, ": {}", probe.devices.join(", "));
            }
            Some(reason)
        }
        Some(probe) => probe
            .error
            .clone()
            .or_else(|| Some("torch.cuda is not available in the managed env".to_owned())),
        None => detected_family
            .as_ref()
            .map(|family| format!("detected host family {family}"))
            .or_else(|| Some("no supported TheRock GPU family detected on this host".to_owned())),
    };

    DetectResponse {
        installed,
        env_id,
        runtime_kind: Some("managed_python".to_owned()),
        runtime_executable: manifest
            .as_ref()
            .map(|manifest| manifest.python_executable.clone()),
        managed_env: Some(true),
        python_version,
        torch_version,
        transformers_version,
        available_devices: vec![
            EngineDeviceAvailability {
                kind: "cpu".to_owned(),
                available: false,
                reason: Some(
                    "rocm-cli does not offer PyTorch CPU serving; use ROCm GPU execution"
                        .to_owned(),
                ),
            },
            EngineDeviceAvailability {
                kind: "rocm_gpu".to_owned(),
                available: rocm_gpu_available,
                reason: rocm_gpu_reason,
            },
        ],
        capabilities: capabilities(),
        notes,
    }
}

fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        cpu: false,
        rocm_gpu: true,
        multi_gpu: true,
        openai_compatible: true,
        tool_calling: true,
        quantized_models: "limited".to_owned(),
        distributed_serving: false,
        reasoning_parser: false,
    }
}

fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    require_nonempty(&request.runtime_id, "runtime_id")?;
    let manifest = create_or_update_env_manifest(&request)?;
    Ok(InstallResponse {
        env_id: manifest.env_id,
        env_path: manifest.env_path.display().to_string(),
        python_executable: manifest.python_executable.clone(),
        runtime_kind: Some("managed_python".to_owned()),
        runtime_executable: Some(manifest.python_executable),
        managed_env: Some(true),
        installed_packages: manifest.installed_packages,
        capabilities: capabilities(),
        lock_hash: manifest.lock_hash,
        warnings: manifest.warnings,
    })
}

fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    require_nonempty(&request.model_ref, "model_ref")?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let recipe = resolve_model_recipe(&request.model_ref)?;
    let device_policy = normalize_pytorch_device_policy(
        request
            .device_policy
            .unwrap_or_else(|| default_device_policy_for_recipe(&recipe)),
    )?;
    Ok(ResolveModelResponse {
        canonical_model_id: recipe.canonical_model_id,
        task: recipe.task.to_owned(),
        source: recipe.source,
        revision: "main".to_owned(),
        loader: recipe.loader.to_owned(),
        trust_remote_code: recipe.trust_remote_code,
        chat_template_mode: recipe.chat_template_mode.to_owned(),
        dtype: recipe.preferred_dtype.clone(),
        device_policy,
        estimated_memory: recipe.estimated_memory,
        launch_defaults: json!({
            "host": "127.0.0.1",
            "port": DEFAULT_LOCAL_PORT,
            "endpoint_mode": "openai"
        }),
        engine_recipe,
        warnings: recipe.warnings,
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
    require_nonempty(&request.service_id, "service_id")?;
    require_nonempty(&request.model_ref, "model_ref")?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe.clone())?;
    let device_policy = normalize_pytorch_device_policy(
        request.device_policy.unwrap_or(DevicePolicy::GpuRequired),
    )?;

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
    let log_file = fs::File::create(&log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone log file handle")?;

    let current_exe =
        std::env::current_exe().context("failed to discover current engine binary")?;
    let child = Command::new(command_path(&current_exe))
        .arg("serve-http")
        .arg(&request.service_id)
        .arg(&request.model_ref)
        .arg("--host")
        .arg(&request.host)
        .arg("--port")
        .arg(request.port.to_string())
        .arg("--device-policy")
        .arg(device_policy_name(&device_policy))
        .args(optional_arg("--env-id", request.env_id.as_deref()))
        .args(optional_arg("--runtime-id", request.runtime_id.as_deref()))
        .args(engine_recipe_json_arg(engine_recipe.as_ref())?)
        .arg("--state-path")
        .arg(&state_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to spawn pytorch serve-http process")?;

    fs::write(
        &state_path,
        serde_json::to_vec_pretty(&json!({
            "engine": ENGINE_NAME,
            "service_id": request.service_id,
            "model_ref": request.model_ref,
            "host": request.host,
            "port": request.port,
            "pid": child.id(),
            "status": "starting",
            "engine_recipe": engine_recipe,
            "engine_recipe_required_flags": engine_recipe_launch_args(engine_recipe.as_ref())
        }))?,
    )
    .with_context(|| format!("failed to write {}", state_path.display()))?;

    let endpoint_url = format!("{}/v1", format_http_base_url(&request.host, request.port));
    Ok(LaunchResponse {
        service_id: request.service_id,
        pid: child.id(),
        endpoint_url,
        log_path: log_path.display().to_string(),
        state_path: state_path.display().to_string(),
    })
}

fn healthcheck_service(request: HealthcheckRequest) -> Result<HealthcheckResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let (state, state_modified, state_error) = read_service_state(&files.state_path);
    let state_status = state
        .as_ref()
        .and_then(|value| value_string(value, "status"))
        .unwrap_or_else(|| "unknown".to_owned());
    let endpoint_url = state.as_ref().and_then(endpoint_url_from_state);
    let (health_payload, probe_error) = if state_status == "ready" || state_status == "running" {
        match endpoint_url.as_deref() {
            Some(endpoint_url) => match query_health_endpoint(endpoint_url) {
                Ok(payload) => (Some(payload), None),
                Err(error) => (None, Some(error.to_string())),
            },
            None => (None, None),
        }
    } else {
        (None, None)
    };

    Ok(build_healthcheck_response(
        state.as_ref(),
        state_modified,
        state_error,
        health_payload.as_ref(),
        probe_error,
        SystemTime::now(),
    ))
}

fn logs_response(request: LogsRequest) -> Result<LogsResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let limit = normalize_tail_limit(request.tail_lines);
    let recent_lines = if files.log_path.is_file() {
        tail_lines(&files.log_path, limit)?
    } else {
        Vec::new()
    };

    Ok(LogsResponse {
        log_path: files.log_path.display().to_string(),
        recent_lines,
    })
}

fn endpoint_response(request: EndpointRequest) -> Result<EndpointResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    endpoint_response_from_files(&files)
}

fn endpoint_response_from_files(files: &ServiceFiles) -> Result<EndpointResponse> {
    let (state, _, _) = read_service_state(&files.state_path);
    let endpoint_url = state
        .as_ref()
        .and_then(endpoint_url_from_state)
        .or_else(|| {
            files
                .record
                .as_ref()
                .filter(|_| files.record_matches_engine)
                .and_then(|record| record.endpoint_url.clone())
        })
        .unwrap_or_else(|| format!("http://127.0.0.1:{DEFAULT_LOCAL_PORT}/v1"));
    Ok(openai_endpoint_response(endpoint_url))
}

fn openai_endpoint_response(endpoint_url: String) -> EndpointResponse {
    EndpointResponse {
        endpoint_url,
        api_style: "openai".to_owned(),
        supported_routes: vec![
            "/healthz".to_owned(),
            "/v1/models".to_owned(),
            "/v1/chat/completions".to_owned(),
            "/v1/completions".to_owned(),
        ],
    }
}

fn stop_service(request: StopRequest) -> Result<StopResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let (state, _, _) = read_service_state(&files.state_path);
    let mut pids = Vec::new();
    if let Some(pid) = state.as_ref().and_then(|value| value_u32(value, "pid")) {
        pids.push(pid);
    }
    if files.record_matches_engine
        && let Some(record) = files.record.as_ref()
    {
        if let Some(pid) = record.engine_pid {
            pids.push(pid);
        }
        if request.force
            && let Some(pid) = record.supervisor_pid
        {
            pids.push(pid);
        }
    }
    dedupe_pids(&mut pids);

    let had_pid = !pids.is_empty();
    let mut stopped_any = false;
    let mut failed_any = false;
    for pid in pids {
        match terminate_pid(pid, request.force) {
            PidTermination::Terminated => stopped_any = true,
            PidTermination::Failed => failed_any = true,
        }
    }

    let state_status = state
        .as_ref()
        .and_then(|value| value_string(value, "status"))
        .or_else(|| {
            files
                .record
                .as_ref()
                .and_then(|record| record.status.clone())
        });
    let already_terminal = matches!(
        state_status.as_deref(),
        Some("stopped" | "failed" | "exited")
    );
    let stopped = stopped_any || (had_pid && !failed_any) || (!had_pid && already_terminal);
    if stopped {
        mark_json_status(&files.state_path, "stopped")?;
        if files.record_matches_engine && files.record_path.is_file() {
            mark_json_status(&files.record_path, "stopped")?;
        }
    }

    Ok(StopResponse {
        stopped,
        graceful: stopped && !request.force && !failed_any,
    })
}

fn service_files(service_id: &str) -> Result<ServiceFiles> {
    let paths = AppPaths::discover()?;
    let record_path = paths.service_manifest_path(service_id);
    let record = load_service_record(&record_path)?;
    let record_matches_engine = record
        .as_ref()
        .and_then(|record| record.engine.as_deref())
        .map_or_else(|| record.is_some(), |engine| engine == ENGINE_NAME);
    let state_path = record
        .as_ref()
        .filter(|_| record_matches_engine)
        .and_then(|record| record.engine_state_path.clone())
        .unwrap_or_else(|| {
            paths
                .engine_state_dir(ENGINE_NAME)
                .join(format!("{service_id}.json"))
        });
    let engine_log_path = paths
        .engine_logs_dir(ENGINE_NAME)
        .join(format!("{service_id}.log"));
    let log_path = record
        .as_ref()
        .filter(|_| record_matches_engine)
        .and_then(|record| record.log_path.clone())
        .or_else(|| {
            if record.is_some() {
                return None;
            }
            let service_log_path = paths.service_log_path(service_id);
            service_log_path.is_file().then_some(service_log_path)
        })
        .unwrap_or(engine_log_path);

    Ok(ServiceFiles {
        record_path,
        record,
        record_matches_engine,
        state_path,
        log_path,
    })
}

fn load_service_record(path: &Path) -> Result<Option<ServiceRecordSnapshot>> {
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))
        .map(Some)
}

fn read_service_state(path: &Path) -> (Option<Value>, Option<SystemTime>, Option<String>) {
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok();
    match fs::read_to_string(path) {
        Ok(content) => match serde_json::from_str::<Value>(&content) {
            Ok(value) => (Some(value), modified, None),
            Err(error) => (
                None,
                modified,
                Some(format!("failed to parse {}: {error}", path.display())),
            ),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (
            None,
            None,
            Some(format!("state file not found: {}", path.display())),
        ),
        Err(error) => (
            None,
            modified,
            Some(format!("failed to read {}: {error}", path.display())),
        ),
    }
}

fn build_healthcheck_response(
    state: Option<&Value>,
    state_modified: Option<SystemTime>,
    state_error: Option<String>,
    health_payload: Option<&Value>,
    probe_error: Option<String>,
    now: SystemTime,
) -> HealthcheckResponse {
    let state_status = state
        .and_then(|value| value_string(value, "status"))
        .unwrap_or_else(|| "unknown".to_owned());
    let health_status = health_payload.and_then(|value| value_string(value, "status"));
    let status = if matches!(state_status.as_str(), "ready" | "running")
        && health_payload.is_none()
        && probe_error.is_some()
    {
        "unreachable".to_owned()
    } else if health_status.as_deref() == Some("ok") {
        "ready".to_owned()
    } else {
        state_status
    };

    let device = health_payload
        .and_then(|value| value_string(value, "device"))
        .or_else(|| state.and_then(|value| value_string(value, "device")))
        .or_else(|| state.and_then(|value| value_string(value, "input_device")))
        .unwrap_or_else(|| "unknown".to_owned());
    let uptime_sec = health_payload
        .and_then(|value| value_f64(value, "loaded_at"))
        .and_then(|loaded_at| uptime_from_epoch_secs(loaded_at, now))
        .or_else(|| state_modified.and_then(|modified| uptime_from_modified(modified, now)))
        .unwrap_or(0);
    let queue_depth = health_payload
        .and_then(|value| value_u32(value, "queue_depth"))
        .or_else(|| state.and_then(|value| value_u32(value, "queue_depth")))
        .unwrap_or(0);
    let last_error = state_error
        .or_else(|| state.and_then(|value| value_string(value, "last_error")))
        .or_else(|| state.and_then(|value| value_string(value, "error")))
        .or_else(|| {
            if status == "unreachable" {
                probe_error
            } else {
                None
            }
        });
    let tokens_per_sec = health_payload
        .and_then(|value| value_f64(value, "tokens_per_sec"))
        .or_else(|| state.and_then(|value| value_f64(value, "tokens_per_sec")))
        .map(|value| value as f32);
    let model_loaded = status == "ready";

    HealthcheckResponse {
        status,
        model_loaded,
        device,
        uptime_sec,
        queue_depth,
        last_error,
        tokens_per_sec,
    }
}

fn endpoint_url_from_state(state: &Value) -> Option<String> {
    value_string(state, "endpoint_url").or_else(|| {
        let host = value_string(state, "host")?;
        let port = value_u32(state, "port")?;
        let port = u16::try_from(port).ok()?;
        Some(format!("{}/v1", format_http_base_url(&host, port)))
    })
}

fn query_health_endpoint(endpoint_url: &str) -> Result<Value> {
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    let addr = (host.as_str(), port)
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve {host}:{port}"))?
        .next()
        .with_context(|| format!("no socket addresses resolved for {host}:{port}"))?;
    let timeout = Duration::from_millis(HEALTHCHECK_TIMEOUT_MS);
    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .with_context(|| format!("failed to connect to {host}:{port}"))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    let host_header = format_host_port(&host, port);
    write!(
        stream,
        "GET /healthz HTTP/1.1\r\nHost: {host_header}\r\nConnection: close\r\n\r\n"
    )
    .context("failed to send health request")?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .context("failed to read health response")?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .context("health response was missing HTTP body")?;
    let status_line = headers.lines().next().unwrap_or_default();
    if !status_line.contains(" 200 ") {
        bail!("health endpoint returned {status_line}");
    }
    serde_json::from_str(body.trim()).context("failed to parse health response body")
}

fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
    let without_scheme = endpoint_url.trim().strip_prefix("http://")?;
    let authority = without_scheme.split('/').next()?.trim();
    if authority.is_empty() {
        return None;
    }
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']')?;
        let host = rest[..end].to_owned();
        let port = rest[end + 1..].strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    Some((host.to_owned(), port.parse().ok()?))
}

fn normalize_tail_limit(tail_lines: Option<usize>) -> usize {
    tail_lines
        .unwrap_or(DEFAULT_LOG_TAIL_LINES)
        .min(MAX_LOG_TAIL_LINES)
}

fn tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(tail_lines_from_text(&content, limit))
}

fn tail_lines_from_text(content: &str, limit: usize) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }
    let mut lines = VecDeque::with_capacity(limit);
    for line in content.lines() {
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(line.to_owned());
    }
    lines.into_iter().collect()
}

fn dedupe_pids(pids: &mut Vec<u32>) {
    let current_pid = std::process::id();
    let mut unique = Vec::new();
    for pid in pids.drain(..) {
        if pid == 0 || pid == current_pid || unique.contains(&pid) {
            continue;
        }
        unique.push(pid);
    }
    *pids = unique;
}

fn terminate_pid(pid: u32, _force: bool) -> PidTermination {
    match rocm_core::terminate_process(pid) {
        Ok(()) => PidTermination::Terminated,
        Err(_) => PidTermination::Failed,
    }
}

fn mark_json_status(path: &Path, status: &str) -> Result<()> {
    let mut value = if path.is_file() {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str::<Value>(&content).unwrap_or_else(|_| json!({}))
    } else {
        json!({})
    };
    if !value.is_object() {
        value = json!({});
    }
    let object = value.as_object_mut().expect("object checked above");
    object.insert("engine".to_owned(), json!(ENGINE_NAME));
    object.insert("status".to_owned(), json!(status));
    object.insert(
        "stopped_at_unix_ms".to_owned(),
        json!(current_unix_millis()),
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(&value).context("failed to serialize status update")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
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

fn value_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn uptime_from_epoch_secs(loaded_at: f64, now: SystemTime) -> Option<u64> {
    if !loaded_at.is_finite() || loaded_at < 0.0 {
        return None;
    }
    let now_secs = now.duration_since(UNIX_EPOCH).ok()?.as_secs_f64();
    Some((now_secs - loaded_at).max(0.0) as u64)
}

fn uptime_from_modified(modified: SystemTime, now: SystemTime) -> Option<u64> {
    now.duration_since(modified)
        .ok()
        .map(|value| value.as_secs())
}

fn current_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn remove_managed_env_dir_with_retry(env_path: &Path) -> Result<()> {
    if !env_path.exists() {
        return Ok(());
    }

    let mut delete_target = env_path.to_owned();
    let mut last_error = None;
    for attempt in 0..=REMOVE_ENV_RETRIES {
        clear_readonly_recursive(&delete_target).ok();
        match fs::remove_dir_all(&delete_target) {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                if attempt == 1
                    && delete_target == env_path
                    && let Some(file_name) = env_path.file_name().and_then(|value| value.to_str())
                    && let Some(parent) = env_path.parent()
                {
                    let moved_as =
                        parent.join(format!("{file_name}.deleting-{}", current_unix_millis()));
                    if fs::rename(env_path, &moved_as).is_ok() {
                        eprintln!(
                            "Managed PyTorch env was locked, moved it aside so a fresh env can be created: {}",
                            moved_as.display()
                        );
                        delete_target = moved_as;
                        if !env_path.exists() {
                            return Ok(());
                        }
                    }
                }
                if attempt < REMOVE_ENV_RETRIES {
                    thread::sleep(REMOVE_ENV_RETRY_DELAY);
                }
            }
        }
    }

    let error = last_error.map_or_else(|| "unknown error".to_owned(), |error| error.to_string());
    bail!(
        "failed to remove managed PyTorch env {} after {} attempts. Close any ROCm, Python, or model server process using this folder and try again. Last error: {error}",
        env_path.display(),
        REMOVE_ENV_RETRIES + 1
    )
}

#[cfg(windows)]
#[allow(clippy::permissions_set_readonly_false)]
fn clear_readonly_recursive(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    let mut permissions = metadata.permissions();
    if permissions.readonly() {
        permissions.set_readonly(false);
        fs::set_permissions(path, permissions)
            .with_context(|| format!("failed to clear readonly attribute on {}", path.display()))?;
    }
    if metadata.is_dir() {
        for entry in
            fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))?
        {
            clear_readonly_recursive(&entry?.path())?;
        }
    }
    Ok(())
}

#[cfg(not(windows))]
const fn clear_readonly_recursive(_path: &Path) -> Result<()> {
    Ok(())
}

fn create_or_update_env_manifest(request: &InstallRequest) -> Result<EngineEnvManifest> {
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    let engine_envs_dir = request.env_root.as_ref().map_or_else(
        || paths.engine_envs_dir(ENGINE_NAME),
        |root| {
            normalize_runtime_path_for_host(root)
                .join(ENGINE_NAME)
                .join("envs")
        },
    );
    fs::create_dir_all(&engine_envs_dir)?;
    fs::create_dir_all(paths.engine_locks_dir(ENGINE_NAME))?;
    fs::create_dir_all(paths.engine_manifests_dir(ENGINE_NAME))?;

    let runtime_python_executable = if request.python_version.is_none() {
        runtime_python_executable_for_selector(&paths, &request.runtime_id)?
    } else {
        None
    };
    let effective_python_version = request.python_version.clone().or(runtime_python_executable
        .as_deref()
        .map(runtime_python_major_minor)
        .transpose()?);

    let env_id = managed_env_id(&request.runtime_id, effective_python_version.as_deref());
    let env_path = engine_envs_dir.join(&env_id);
    let lock_path = paths
        .engine_locks_dir(ENGINE_NAME)
        .join(format!("{env_id}.txt"));
    let manifest_path = paths
        .engine_manifests_dir(ENGINE_NAME)
        .join(format!("{env_id}.json"));
    let existing_manifest = if manifest_path.is_file() {
        Some(load_manifest(&manifest_path)?)
    } else {
        None
    };
    let therock_resolution = if std::env::var("ROCM_CLI_PYTORCH_PACKAGE_SPEC").is_ok() {
        None
    } else {
        resolve_therock_torch_resolution(&paths, &request.runtime_id)?
    };

    if !request.reinstall
        && let Some(manifest) = existing_manifest
        && manifest.env_path.is_dir()
        && (manifest_has_torch(&manifest) || therock_resolution.is_none())
    {
        return Ok(manifest);
    }

    remove_managed_env_dir_with_retry(&env_path)?;

    let launcher = discover_python_launcher(
        effective_python_version.as_deref(),
        runtime_python_executable.as_deref(),
    )?;
    let uv = ensure_uv_binary(&paths)?;
    let launcher_path = PathBuf::from(&launcher.program);
    let venv_args = uv_venv_args(&launcher_path, &env_path);
    run_uv_command(
        &uv,
        venv_args.iter().map(String::as_str),
        "create managed pytorch venv",
    )?;

    let python_executable = venv_python_path(&env_path);

    let mut engine_dep_args = uv_pip_install_base(&python_executable);
    engine_dep_args.extend(["--only-binary".to_owned(), ":all:".to_owned()]);
    engine_dep_args.extend(
        ENGINE_DEPENDENCIES
            .iter()
            .map(std::string::ToString::to_string),
    );
    run_uv_progress_command(
        &uv,
        engine_dep_args.iter().map(String::as_str),
        "install managed pytorch engine dependencies",
    )?;

    let python_executable_string = python_executable.to_string_lossy().to_string();
    let mut warnings = Vec::new();
    let mut therock_channel = None;
    let mut therock_family = None;
    let mut therock_index_url = None;
    let mut therock_packages = Vec::new();
    let maybe_torch_spec = std::env::var("ROCM_CLI_PYTORCH_PACKAGE_SPEC").ok();
    let maybe_extra_index = std::env::var("ROCM_CLI_PYTORCH_EXTRA_INDEX_URL").ok();
    if let Some(torch_spec) = maybe_torch_spec.as_deref() {
        let mut args = uv_pip_install_base(&python_executable);
        args.push("--only-binary".to_owned());
        args.push(":all:".to_owned());
        if let Some(extra_index) = maybe_extra_index.as_deref() {
            args.push("--extra-index-url".to_owned());
            args.push(extra_index.to_owned());
        }
        args.push(torch_spec.to_owned());
        run_uv_progress_command(
            &uv,
            args.iter().map(String::as_str),
            "install torch package into managed pytorch env",
        )?;
        let mut stack_args = uv_pip_install_base(&python_executable);
        stack_args.extend(["--only-binary".to_owned(), ":all:".to_owned()]);
        stack_args.extend(
            TORCH_STACK_DEPENDENCIES
                .iter()
                .map(std::string::ToString::to_string),
        );
        run_uv_progress_command(
            &uv,
            stack_args.iter().map(String::as_str),
            "install pytorch engine runtime dependencies",
        )?;
        warnings.push(
            "using manual torch package override from ROCM_CLI_PYTORCH_PACKAGE_SPEC".to_owned(),
        );
    } else {
        match therock_resolution {
            Some(resolution) => {
                install_therock_torch_packages(&uv, &python_executable, &resolution)?;
                let mut stack_args = uv_pip_install_base(&python_executable);
                stack_args.extend(["--only-binary".to_owned(), ":all:".to_owned()]);
                stack_args.extend(
                    TORCH_STACK_DEPENDENCIES
                        .iter()
                        .map(std::string::ToString::to_string),
                );
                run_uv_progress_command(
                    &uv,
                    stack_args.iter().map(String::as_str),
                    "install pytorch engine runtime dependencies",
                )?;
                therock_channel = Some(resolution.channel.as_str().to_owned());
                therock_family = Some(resolution.family.clone());
                therock_index_url = Some(resolution.index_url.clone());
                therock_packages = resolution.packages.clone();
                warnings.push(format!(
                    "installed TheRock PyTorch packages from {} ({}, source={})",
                    resolution.index_url, resolution.family, resolution.source
                ));
            }
            None => {
                warnings.push(
                    "torch installation deferred because no supported TheRock GPU family could be resolved; set ROCM_CLI_THEROCK_FAMILY or use a runtime_id like therock-release:gfx950-dcgpu".to_owned(),
                );
            }
        }
    }

    let installed_packages = if cosmo_windows_host() {
        installed_package_specs_from_runtime_metadata(&python_executable_string)
            .context("capture managed pytorch env lockfile from package metadata")?
    } else {
        let freeze_args = uv_pip_freeze_args(&python_executable);
        let freeze = capture_uv_command(
            &uv,
            freeze_args.iter().map(String::as_str),
            "capture managed pytorch env lockfile",
        )?;
        freeze
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    };
    let freeze = installed_packages.join("\n") + "\n";
    fs::write(&lock_path, &freeze)
        .with_context(|| format!("failed to write {}", lock_path.display()))?;
    let lock_hash = simple_hash(&freeze);
    let torch_runtime_probe = if cosmo_windows_host() {
        warnings.push(
            "managed PyTorch runtime probe deferred for the universal Windows launcher".to_owned(),
        );
        None
    } else {
        match probe_torch_runtime(&python_executable_string) {
            Ok(probe) => {
                if therock_family.is_some() {
                    match probe.rocm_sdk.as_ref() {
                        Some(sdk) if sdk.import_ok => {
                            if let (Some(expected), Some(actual)) = (
                                probe
                                    .torch_rocm_init
                                    .as_ref()
                                    .and_then(|init| init.check_version.as_deref()),
                                sdk.version.as_deref(),
                            ) && expected != actual
                            {
                                warnings.push(format!(
                                "torch._rocm_init expects ROCm SDK {expected}, but rocm_sdk reports {actual}"
                            ));
                            }
                        }
                        Some(sdk) => warnings.push(format!(
                            "rocm_sdk import probe failed in managed PyTorch env: {}",
                            sdk.error.as_deref().unwrap_or("unknown error")
                        )),
                        None => warnings.push(
                            "rocm_sdk probe was not reported by the managed PyTorch env".to_owned(),
                        ),
                    }
                    if !probe
                        .torch_rocm_init
                        .as_ref()
                        .is_some_and(|init| init.present)
                    {
                        warnings.push(
                        "torch._rocm_init was not found; TheRock library preloading may be unavailable"
                            .to_owned(),
                    );
                    }
                }
                Some(probe)
            }
            Err(error) => {
                warnings.push(format!("managed PyTorch runtime probe failed: {error}"));
                None
            }
        }
    };

    let manifest = EngineEnvManifest {
        env_id,
        runtime_id: request.runtime_id.clone(),
        requested_python_version: effective_python_version,
        python_launcher: launcher.display,
        python_executable: python_executable.display().to_string(),
        env_path,
        manifest_path,
        lock_path,
        installed_packages,
        lock_hash,
        pip_cache_dir: None,
        therock_channel,
        therock_family,
        therock_index_url,
        therock_packages,
        torch_runtime_probe,
        warnings,
    };
    write_manifest(&manifest)?;
    Ok(manifest)
}

fn latest_env_manifest() -> Result<Option<EngineEnvManifest>> {
    let paths = AppPaths::discover()?;
    let manifests_dir = paths.engine_manifests_dir(ENGINE_NAME);
    if !manifests_dir.is_dir() {
        return Ok(None);
    }

    let mut manifests = Vec::new();
    for entry in fs::read_dir(&manifests_dir)
        .with_context(|| format!("failed to read {}", manifests_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        manifests.push(load_manifest(&path)?);
    }
    manifests.sort_by(|left, right| left.env_id.cmp(&right.env_id));
    Ok(manifests.pop())
}

fn load_manifest(path: &Path) -> Result<EngineEnvManifest> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read engine manifest {}", path.display()))?;
    let mut manifest: EngineEnvManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse engine manifest {}", path.display()))?;
    normalize_manifest_paths_for_host(&mut manifest);
    manifest.python_executable = resolve_manifest_python_executable(&manifest)
        .display()
        .to_string();
    Ok(manifest)
}

fn normalize_manifest_paths_for_host(manifest: &mut EngineEnvManifest) {
    manifest.python_executable = normalize_runtime_path_text_for_host(&manifest.python_executable);
    manifest.env_path = normalize_runtime_path_for_host(&manifest.env_path);
    manifest.manifest_path = normalize_runtime_path_for_host(&manifest.manifest_path);
    manifest.lock_path = normalize_runtime_path_for_host(&manifest.lock_path);
    manifest.pip_cache_dir = manifest
        .pip_cache_dir
        .as_ref()
        .map(|path| normalize_runtime_path_for_host(path));
}

fn write_manifest(manifest: &EngineEnvManifest) -> Result<()> {
    fs::write(
        &manifest.manifest_path,
        serde_json::to_vec_pretty(manifest).context("failed to serialize engine manifest")?,
    )
    .with_context(|| format!("failed to write {}", manifest.manifest_path.display()))?;
    Ok(())
}

fn discover_python_launcher(
    requested_version: Option<&str>,
    preferred_python: Option<&str>,
) -> Result<PythonLauncher> {
    let mut candidates = Vec::new();
    if let Some(python) = preferred_python
        && !python.trim().is_empty()
    {
        let normalized = normalize_runtime_path_text_for_host(python);
        if runtime_is_windows() {
            return Ok(PythonLauncher {
                program: normalized.clone(),
                args: Vec::new(),
                display: normalized,
            });
        }
        candidates.push(PythonLauncher {
            program: normalized.clone(),
            args: Vec::new(),
            display: normalized,
        });
    }
    if runtime_is_windows() {
        if let Some(version) = requested_version {
            candidates.push(PythonLauncher {
                program: "py".to_owned(),
                args: vec![format!("-{version}")],
                display: format!("py -{version}"),
            });
        }
        candidates.push(PythonLauncher {
            program: "py".to_owned(),
            args: Vec::new(),
            display: "py".to_owned(),
        });
        candidates.push(PythonLauncher {
            program: "python".to_owned(),
            args: Vec::new(),
            display: "python".to_owned(),
        });
    } else {
        if let Some(version) = requested_version {
            candidates.push(PythonLauncher {
                program: format!("python{version}"),
                args: Vec::new(),
                display: format!("python{version}"),
            });
        }
        candidates.push(PythonLauncher {
            program: "python3".to_owned(),
            args: Vec::new(),
            display: "python3".to_owned(),
        });
        candidates.push(PythonLauncher {
            program: "python".to_owned(),
            args: Vec::new(),
            display: "python".to_owned(),
        });
    }

    let mut attempts = Vec::new();
    for launcher in candidates {
        let status = Command::new(command_program(&launcher.program))
            .args(&launcher.args)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        attempts.push(format!("{} -> {:?}", launcher.display, status));
        if matches!(status, Ok(value) if value.success()) {
            return Ok(launcher);
        }
    }

    bail!(
        "unable to locate a usable Python launcher for the pytorch engine (runtime_os={}, requested_version={}, preferred_python={}, attempts={})",
        rocm_core::runtime_os_name(),
        requested_version.unwrap_or("<none>"),
        preferred_python.unwrap_or("<none>"),
        attempts.join("; ")
    )
}

fn runtime_python_executable_for_selector(
    paths: &AppPaths,
    selector: &str,
) -> Result<Option<String>> {
    let Some(manifest) = load_runtime_registry_manifest(paths, selector)? else {
        return Ok(None);
    };
    if manifest.format != "wheel" {
        return Ok(None);
    }
    Ok(manifest.python_executable)
}

fn runtime_python_major_minor(python_executable: &str) -> Result<String> {
    if cosmo_windows_host() {
        return Ok("3.12".to_owned());
    }
    let output = capture_command(
        python_executable,
        [
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        "inspect managed runtime Python version",
    )?;
    let version = output.trim();
    if version.is_empty() {
        bail!("managed runtime Python did not report a version");
    }
    Ok(version.to_owned())
}

fn venv_python_path(env_path: &Path) -> PathBuf {
    if runtime_is_windows() {
        env_path.join("Scripts").join("python.exe")
    } else {
        env_path.join("bin").join("python")
    }
}

fn resolve_manifest_python_executable(manifest: &EngineEnvManifest) -> PathBuf {
    let recorded = PathBuf::from(&manifest.python_executable);
    if recorded.is_file() {
        return recorded;
    }

    for candidate in venv_python_candidates(&manifest.env_path) {
        if candidate.is_file() {
            return candidate;
        }
    }

    recorded
}

fn venv_python_candidates(env_path: &Path) -> Vec<PathBuf> {
    if runtime_is_windows() {
        return vec![
            env_path.join("Scripts").join("python.exe"),
            env_path.join("Scripts").join("python3.exe"),
        ];
    }

    let bin = env_path.join("bin");
    vec![
        bin.join("python"),
        bin.join("python3"),
        bin.join("python3.12"),
        bin.join("python3.11"),
        bin.join("python3.10"),
        bin.join("python3.9"),
    ]
}

#[allow(clippy::too_many_arguments)]
fn serve_http(
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: DevicePolicy,
    env_id: Option<String>,
    runtime_id: Option<String>,
    state_path: PathBuf,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<()> {
    let manifest = ensure_service_env(runtime_id.as_deref(), env_id.as_deref())?;
    let device_policy = normalize_pytorch_device_policy(device_policy)?;
    let recipe = apply_pytorch_engine_recipe_overrides(
        resolve_model_recipe(&model_ref)?,
        engine_recipe.as_ref(),
    )?;
    let worker_script = materialize_python_worker()?;
    let endpoint_url = format!("{}/v1", format_http_base_url(&host, port));
    fs::write(
        &state_path,
        serde_json::to_vec_pretty(&json!({
            "engine": ENGINE_NAME,
            "service_id": service_id,
            "model_ref": model_ref,
            "status": "starting",
            "pid": std::process::id(),
            "host": host,
            "port": port,
            "device_policy": device_policy_name(&device_policy),
            "env_id": manifest.env_id,
            "runtime_id": manifest.runtime_id,
            "python_executable": manifest.python_executable,
            "preferred_dtype": &recipe.preferred_dtype,
            "estimated_memory": &recipe.estimated_memory,
            "trust_remote_code": recipe.trust_remote_code,
            "engine_recipe": engine_recipe,
            "engine_recipe_required_flags": engine_recipe_launch_args(engine_recipe.as_ref()),
            "endpoint_url": endpoint_url
        }))?,
    )?;
    let mut worker_command = Command::new(command_program(&manifest.python_executable));
    worker_command
        .arg(&worker_script)
        .arg("--service-id")
        .arg(&service_id)
        .arg("--model-ref")
        .arg(&model_ref)
        .arg("--host")
        .arg(&host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--device-policy")
        .arg(device_policy_name(&device_policy))
        .arg("--state-path")
        .arg(&state_path)
        .arg("--env-id")
        .arg(&manifest.env_id)
        .arg("--runtime-id")
        .arg(&manifest.runtime_id)
        .arg("--preferred-dtype")
        .arg(&recipe.preferred_dtype)
        .args(optional_arg_owned(
            "--min-gpu-mem-gb",
            recipe.min_gpu_mem_gb.map(|value| value.to_string()),
        ))
        .args(flag_arg("--trust-remote-code", recipe.trust_remote_code))
        .env("PYTHONUNBUFFERED", "1")
        .env("TOKENIZERS_PARALLELISM", "false")
        .stdin(Stdio::null());
    if cosmo_windows_host() {
        worker_command.stdout(Stdio::null()).stderr(Stdio::null());
    } else {
        worker_command
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
    }
    let mut child = worker_command
        .spawn()
        .context("failed to start python worker for pytorch engine")?;

    let status = child
        .wait()
        .context("failed waiting for pytorch python worker")?;
    if status.success() {
        mark_json_status(&state_path, "stopped")?;
        Ok(())
    } else {
        let current_status = read_service_state(&state_path)
            .0
            .and_then(|value| value_string(&value, "status"));
        if matches!(current_status.as_deref(), Some("stopped" | "stopping")) {
            mark_json_status(&state_path, "stopped")?;
            return Ok(());
        }
        mark_json_status(&state_path, "failed")?;
        bail!("pytorch worker exited with status {status} for service {service_id}")
    }
}

fn canonical_model_id(model_ref: &str) -> String {
    if let Ok(Some(recipe)) = resolve_shared_model_recipe(model_ref)
        && shared_recipe_supports_engine(&recipe, ENGINE_NAME)
    {
        return recipe.canonical_model_id;
    }

    match model_ref.to_ascii_lowercase().as_str() {
        "qwen" | "qwen2.5" | "qwen2.5-1.5b" => "Qwen/Qwen2.5-1.5B-Instruct".to_owned(),
        "qwen-tiny" | "tiny-qwen" | "qwen2.5-0.5b" => "Qwen/Qwen2.5-0.5B-Instruct".to_owned(),
        "qwen3.5" => "Qwen/Qwen3.5-4B".to_owned(),
        "qwen32b" | "qwen3.5-32b" | "qwen3-32b" => "Qwen/Qwen3-32B-FP8".to_owned(),
        "glm5" | "glm-5" => "zai-org/GLM-5-FP8".to_owned(),
        "llama3.2" | "llama" => "meta-llama/Llama-3.2-3B-Instruct".to_owned(),
        "tiny-gpt2" | "gpt2tiny" => "sshleifer/tiny-gpt2".to_owned(),
        other if other.contains('/') => model_ref.to_owned(),
        _ => model_ref.to_owned(),
    }
}

fn resolve_model_recipe(model_ref: &str) -> Result<ModelRecipe> {
    if let Some(recipe) = known_recipe_for_model(model_ref)? {
        return Ok(recipe);
    }

    let canonical_model_id = canonical_model_id(model_ref);
    ensure_pytorch_model_supported(&canonical_model_id)?;
    let source = if model_ref.contains('/') {
        "huggingface".to_owned()
    } else {
        "alias".to_owned()
    };
    let family = infer_model_family(&canonical_model_id);
    let trust_remote_code = matches!(family, ModelFamily::Glm);
    let preferred_dtype = preferred_dtype_for_model(&canonical_model_id, family);
    let min_gpu_mem_gb = estimate_gpu_memory_gib(&canonical_model_id);
    let estimated_memory = format_estimated_memory(&canonical_model_id, min_gpu_mem_gb);

    let mut warnings = Vec::new();
    if trust_remote_code {
        warnings.push(
            "this model family is configured with trust_remote_code enabled by recipe".to_owned(),
        );
    }
    if let Some(min_gpu_mem_gb) = min_gpu_mem_gb {
        if min_gpu_mem_gb >= 48 {
            warnings.push(format!(
                "this model looks large (~{min_gpu_mem_gb} GiB GPU memory recommended); implicit CPU serving is disabled for safety"
            ));
        } else if min_gpu_mem_gb >= 16 {
            warnings.push(format!(
                "this model may need about {min_gpu_mem_gb} GiB of GPU memory for comfortable serving"
            ));
        }
    }

    Ok(ModelRecipe {
        canonical_model_id,
        task: "chat",
        source,
        loader: "transformers",
        trust_remote_code,
        chat_template_mode: "auto",
        preferred_dtype: preferred_dtype.to_owned(),
        device_policy: default_device_policy_for_memory(min_gpu_mem_gb),
        estimated_memory,
        min_gpu_mem_gb,
        warnings,
    })
}

fn known_recipe_for_model(model_ref: &str) -> Result<Option<ModelRecipe>> {
    let Some(recipe) = resolve_shared_model_recipe(model_ref)? else {
        return Ok(None);
    };
    if !shared_recipe_supports_engine(&recipe, ENGINE_NAME) {
        return Ok(None);
    }
    let device_policy = parse_device_policy(&recipe.device_policy)
        .unwrap_or_else(|_| default_device_policy_for_memory(recipe.min_gpu_mem_gb));
    let canonical_model_id = recipe.canonical_model_id;
    ensure_pytorch_model_supported(&canonical_model_id)?;
    let source = recipe.source;
    let trust_remote_code = recipe.trust_remote_code;
    let dtype = recipe.dtype;
    let min_gpu_mem_gb = recipe.min_gpu_mem_gb;
    let warnings = recipe.warnings;
    Ok(Some(build_known_recipe(
        &canonical_model_id,
        source,
        trust_remote_code,
        &dtype,
        device_policy,
        min_gpu_mem_gb,
        warnings,
    )))
}

fn shared_recipe_supports_engine(recipe: &ModelRecipeRecord, engine: &str) -> bool {
    recipe
        .preferred_engines
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(engine))
        || recipe
            .engine_recipes
            .iter()
            .any(|candidate| candidate.engine.eq_ignore_ascii_case(engine))
}

fn ensure_pytorch_model_supported(canonical_model_id: &str) -> Result<()> {
    if canonical_model_id.eq_ignore_ascii_case("Qwen/Qwen3.5-4B") {
        bail!(
            "Qwen/Qwen3.5-4B is not supported by the managed PyTorch engine yet: the current Transformers line reports unknown architecture `qwen3_5`. Use `qwen` for the recommended Qwen2.5 1.5B local assistant recipe, or use an engine/runtime that explicitly supports Qwen3.5."
        );
    }
    Ok(())
}

fn apply_pytorch_engine_recipe_overrides(
    mut recipe: ModelRecipe,
    engine_recipe: Option<&EngineRecipeHint>,
) -> Result<ModelRecipe> {
    let Some(engine_recipe) = engine_recipe else {
        return Ok(recipe);
    };

    let flags = &engine_recipe.required_flags;
    let mut index = 0;
    while index < flags.len() {
        let flag = flags[index].trim();
        match flag {
            "--trust-remote-code" | "--trust-remote-code=true" => {
                recipe.trust_remote_code = true;
            }
            "--no-trust-remote-code" | "--trust-remote-code=false" => {
                recipe.trust_remote_code = false;
            }
            "--preferred-dtype" => {
                index += 1;
                let value = engine_recipe_flag_value(flags, index, "--preferred-dtype")?;
                recipe.preferred_dtype = value.to_owned();
            }
            _ if flag.starts_with("--preferred-dtype=") => {
                let value = flag
                    .split_once('=')
                    .map(|(_, value)| value.trim())
                    .unwrap_or_default();
                require_nonempty(value, "--preferred-dtype value")?;
                recipe.preferred_dtype = value.to_owned();
            }
            "--min-gpu-mem-gb" => {
                index += 1;
                let value = engine_recipe_flag_value(flags, index, "--min-gpu-mem-gb")?;
                recipe.min_gpu_mem_gb = Some(parse_engine_recipe_memory_gb(value)?);
                recipe.estimated_memory =
                    format_estimated_memory(&recipe.canonical_model_id, recipe.min_gpu_mem_gb);
            }
            _ if flag.starts_with("--min-gpu-mem-gb=") => {
                let value = flag
                    .split_once('=')
                    .map(|(_, value)| value.trim())
                    .unwrap_or_default();
                require_nonempty(value, "--min-gpu-mem-gb value")?;
                recipe.min_gpu_mem_gb = Some(parse_engine_recipe_memory_gb(value)?);
                recipe.estimated_memory =
                    format_estimated_memory(&recipe.canonical_model_id, recipe.min_gpu_mem_gb);
            }
            unsupported => {
                bail!("unsupported PyTorch launch recipe flag `{unsupported}`");
            }
        }
        index += 1;
    }

    Ok(recipe)
}

fn engine_recipe_flag_value<'a>(flags: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    let Some(value) = flags.get(index).map(|value| value.trim()) else {
        bail!("{flag} requires a value");
    };
    require_nonempty(value, &format!("{flag} value"))?;
    if value.starts_with("--") {
        bail!("{flag} requires a value before `{value}`");
    }
    Ok(value)
}

fn parse_engine_recipe_memory_gb(value: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .with_context(|| format!("invalid --min-gpu-mem-gb value `{value}`"))
}

fn build_known_recipe(
    canonical_model_id: &str,
    source: String,
    trust_remote_code: bool,
    preferred_dtype: &str,
    device_policy: DevicePolicy,
    min_gpu_mem_gb: Option<u32>,
    mut warnings: Vec<String>,
) -> ModelRecipe {
    if let Some(min_gpu_mem_gb) = min_gpu_mem_gb {
        if min_gpu_mem_gb >= 48
            && !warnings
                .iter()
                .any(|value| value.contains("implicit CPU serving"))
        {
            warnings.push(format!(
                "this model looks large (~{min_gpu_mem_gb} GiB GPU memory recommended); implicit CPU serving is disabled for safety"
            ));
            if !warnings.iter().any(|value| value.contains("visible GPUs")) {
                warnings.push(
                    "startup will attempt auto device_map placement across visible GPUs when aggregate memory is sufficient"
                        .to_owned(),
                );
            }
        } else if min_gpu_mem_gb >= 16 && !warnings.iter().any(|value| value.contains("GPU memory"))
        {
            warnings.push(format!(
                "this model may need about {min_gpu_mem_gb} GiB of GPU memory for comfortable serving"
            ));
        }
    }

    ModelRecipe {
        canonical_model_id: canonical_model_id.to_owned(),
        task: "chat",
        source,
        loader: "transformers",
        trust_remote_code,
        chat_template_mode: "auto",
        preferred_dtype: preferred_dtype.to_owned(),
        device_policy,
        estimated_memory: format_estimated_memory(canonical_model_id, min_gpu_mem_gb),
        min_gpu_mem_gb,
        warnings,
    }
}

fn infer_model_family(model_ref: &str) -> ModelFamily {
    let lower = model_ref.to_ascii_lowercase();
    if lower.contains("qwen") {
        ModelFamily::Qwen
    } else if lower.contains("glm") {
        ModelFamily::Glm
    } else if lower.contains("llama") {
        ModelFamily::Llama
    } else if lower.contains("gpt2") {
        ModelFamily::Gpt2
    } else {
        ModelFamily::Generic
    }
}

fn preferred_dtype_for_model(model_ref: &str, family: ModelFamily) -> &'static str {
    let lower = model_ref.to_ascii_lowercase();
    if lower.contains("fp8") || lower.contains("gptq") || lower.contains("awq") {
        "auto"
    } else if matches!(
        family,
        ModelFamily::Qwen | ModelFamily::Glm | ModelFamily::Llama
    ) {
        "bfloat16"
    } else {
        "auto"
    }
}

fn infer_parameter_billions(model_ref: &str) -> Option<f32> {
    let lower = model_ref.to_ascii_lowercase();
    if lower == "zai-org/glm-5-fp8" {
        return Some(754.0);
    }

    lower
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '.'))
        .find_map(|token| {
            token
                .strip_suffix('b')
                .and_then(|value| value.parse::<f32>().ok())
                .or_else(|| {
                    token
                        .strip_suffix('m')
                        .and_then(|value| value.parse::<f32>().ok())
                        .map(|value| value / 1000.0)
                })
        })
}

fn infer_weight_bytes_per_param(model_ref: &str) -> f32 {
    let lower = model_ref.to_ascii_lowercase();
    if lower.contains("int4") || lower.contains("awq") || lower.contains("gptq") {
        0.5
    } else if lower.contains("fp8") || lower.contains("int8") {
        1.0
    } else {
        2.0
    }
}

fn estimate_gpu_memory_gib(model_ref: &str) -> Option<u32> {
    let params = infer_parameter_billions(model_ref)?;
    let bytes_per_param = infer_weight_bytes_per_param(model_ref);
    let overhead = if bytes_per_param <= 1.0 { 1.20 } else { 1.35 };
    Some((params * bytes_per_param * overhead).ceil().max(2.0) as u32)
}

fn format_estimated_memory(model_ref: &str, min_gpu_mem_gb: Option<u32>) -> String {
    match (infer_parameter_billions(model_ref), min_gpu_mem_gb) {
        (Some(params), Some(min_gpu_mem_gb)) => {
            format!(
                "~{min_gpu_mem_gb} GiB GPU memory recommended for ~{}B parameters",
                trim_float(params)
            )
        }
        (Some(params), None) => format!(
            "~{}B parameters; memory estimate unavailable",
            trim_float(params)
        ),
        (None, Some(min_gpu_mem_gb)) => format!("~{min_gpu_mem_gb} GiB GPU memory recommended"),
        (None, None) => "memory estimate unavailable".to_owned(),
    }
}

fn default_device_policy_for_recipe(recipe: &ModelRecipe) -> DevicePolicy {
    recipe.device_policy.clone()
}

fn default_device_policy_for_memory(min_gpu_mem_gb: Option<u32>) -> DevicePolicy {
    if min_gpu_mem_gb.unwrap_or_default() >= 48 {
        DevicePolicy::GpuRequired
    } else {
        DevicePolicy::GpuPreferred
    }
}

fn trim_float(value: f32) -> String {
    if (value.fract() - 0.0).abs() < f32::EPSILON {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

fn parse_device_policy(value: &str) -> Result<DevicePolicy> {
    match value {
        "gpu_required" => Ok(DevicePolicy::GpuRequired),
        "gpu_preferred" => Ok(DevicePolicy::GpuPreferred),
        "cpu_only" => Ok(DevicePolicy::CpuOnly),
        _ => bail!("unknown device policy: {value}"),
    }
}

fn normalize_pytorch_device_policy(policy: DevicePolicy) -> Result<DevicePolicy> {
    match policy {
        DevicePolicy::GpuRequired => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::GpuPreferred => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::CpuOnly => {
            bail!("PyTorch adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

const fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
}

fn slugify(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '-',
        })
        .collect()
}

fn managed_env_id(runtime_id: &str, python_version: Option<&str>) -> String {
    let python = python_version.unwrap_or(default_python_version());
    format!(
        "{}-{}-{}",
        rocm_core::runtime_os_name(),
        slugify(runtime_id),
        slugify(python)
    )
}

const fn default_python_version() -> &'static str {
    "3.12"
}

fn parse_therock_runtime_request(runtime_id: &str) -> TheRockRuntimeRequest {
    let normalized = runtime_id.trim().to_ascii_lowercase();
    let channel = therock_channel_from_str(&normalized);

    let family_override = KNOWN_THEROCK_FAMILIES
        .iter()
        .find(|family| normalized.contains(&family.to_ascii_lowercase()))
        .map(|family| (*family).to_owned())
        .or_else(|| {
            extract_first_gfx_token(&normalized)
                .and_then(|target| normalize_therock_family(&target))
        });

    TheRockRuntimeRequest {
        channel,
        family_override,
    }
}

fn therock_channel_from_str(value: &str) -> TheRockChannel {
    if value.trim().to_ascii_lowercase().contains("nightly") {
        TheRockChannel::Nightly
    } else {
        TheRockChannel::Release
    }
}

fn resolve_therock_torch_resolution(
    paths: &AppPaths,
    runtime_id: &str,
) -> Result<Option<TheRockTorchResolution>> {
    let runtime_request = parse_therock_runtime_request(runtime_id);

    if let Some(manifest) = load_runtime_registry_manifest(paths, runtime_id)? {
        if manifest.format != "wheel" {
            return Ok(None);
        }
        let family = normalize_therock_family(&manifest.family)
            .or_else(|| parse_therock_runtime_request(&manifest.runtime_id).family_override)
            .with_context(|| {
                format!(
                    "managed runtime `{}` did not report a supported TheRock family",
                    manifest.runtime_key
                )
            })?;
        let python = manifest.python_executable.as_deref().with_context(|| {
            format!(
                "managed runtime `{}` did not record a Python executable",
                manifest.runtime_key
            )
        })?;
        let packages = pinned_torch_package_specs_from_runtime(python).with_context(|| {
            format!(
                "managed runtime `{}` did not expose exact torch package versions",
                manifest.runtime_key
            )
        })?;
        let channel = therock_channel_from_str(&manifest.channel);
        return Ok(Some(TheRockTorchResolution {
            channel,
            family: family.clone(),
            index_url: manifest
                .index_url
                .or(manifest.selected_artifact_url)
                .unwrap_or_else(|| therock_index_url(&family)),
            packages,
            source: format!("managed_runtime_manifest:{}", manifest.runtime_key),
        }));
    }

    if let Some(family) = runtime_request.family_override {
        return Ok(Some(TheRockTorchResolution {
            channel: runtime_request.channel,
            family: family.clone(),
            index_url: therock_index_url(&family),
            packages: THEROCK_TORCH_PACKAGES
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
            source: "runtime_id".to_owned(),
        }));
    }

    if let Ok(value) = std::env::var("ROCM_CLI_THEROCK_FAMILY")
        && let Some(family) = normalize_therock_family(&value)
    {
        return Ok(Some(TheRockTorchResolution {
            channel: runtime_request.channel,
            family: family.clone(),
            index_url: therock_index_url(&family),
            packages: THEROCK_TORCH_PACKAGES
                .iter()
                .map(|item| (*item).to_owned())
                .collect(),
            source: "env".to_owned(),
        }));
    }

    if let Some(family) = detect_host_therock_family() {
        return Ok(Some(TheRockTorchResolution {
            channel: runtime_request.channel,
            family: family.clone(),
            index_url: therock_index_url(&family),
            packages: THEROCK_TORCH_PACKAGES
                .iter()
                .map(|item| (*item).to_owned())
                .collect(),
            source: "host".to_owned(),
        }));
    }

    Ok(None)
}

fn runtime_registry_dir(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("runtimes").join("registry")
}

fn load_runtime_registry_manifest(
    paths: &AppPaths,
    selector: &str,
) -> Result<Option<RuntimeRegistryManifest>> {
    let registry_dir = runtime_registry_dir(paths);
    if !registry_dir.is_dir() {
        return Ok(None);
    }

    let exact_path = registry_dir.join(format!("{selector}.json"));
    if exact_path.is_file() {
        return load_runtime_registry_manifest_path(&exact_path).map(Some);
    }

    let mut matches = Vec::new();
    for entry in fs::read_dir(&registry_dir)
        .with_context(|| format!("failed to read {}", registry_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let manifest = load_runtime_registry_manifest_path(&path)?;
        if manifest.runtime_id.eq_ignore_ascii_case(selector) {
            matches.push(manifest);
        }
    }
    if matches.len() == 1 {
        return Ok(matches.pop());
    }
    Ok(None)
}

fn load_runtime_registry_manifest_path(path: &Path) -> Result<RuntimeRegistryManifest> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse runtime manifest {}", path.display()))
}

fn pinned_torch_package_specs_from_runtime(python_executable: &str) -> Result<Vec<String>> {
    match pinned_torch_package_specs_from_runtime_metadata(python_executable) {
        Ok(specs) => return Ok(specs),
        Err(metadata_error) if cosmo_windows_host() => {
            bail!("failed to inspect runtime package metadata: {metadata_error}");
        }
        Err(_) => {}
    }

    let code = format!(
        "import importlib.metadata as md, json; names = {THEROCK_TORCH_PACKAGES:?}; print(json.dumps({{name: md.version(name) for name in names}}, sort_keys=True))"
    );
    let output = capture_command(
        python_executable,
        ["-c", code.as_str()],
        "inspect managed runtime torch package versions",
    )?;
    parse_torch_package_version_specs(&output)
}

fn pinned_torch_package_specs_from_runtime_metadata(
    python_executable: &str,
) -> Result<Vec<String>> {
    let packages = installed_package_specs_from_runtime_metadata(python_executable)?;
    let versions = packages
        .iter()
        .filter_map(|spec| spec.split_once("=="))
        .map(|(name, version)| (name.to_ascii_lowercase(), version.to_owned()))
        .collect::<BTreeMap<_, _>>();
    let mut specs = Vec::new();
    for name in THEROCK_TORCH_PACKAGES {
        let version = versions
            .get(*name)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("runtime metadata is missing package `{name}`"))?;
        specs.push(format!("{name}=={version}"));
    }
    Ok(specs)
}

fn installed_package_specs_from_runtime_metadata(python_executable: &str) -> Result<Vec<String>> {
    let mut errors = Vec::new();
    for site_packages in site_packages_candidates_for_python(python_executable) {
        match installed_package_specs_from_dist_info_dir(&site_packages) {
            Ok(specs) => return Ok(specs),
            Err(error) => errors.push(format!("{}: {error}", site_packages.display())),
        }
    }
    bail!(
        "could not find installed package metadata ({})",
        errors.join("; ")
    )
}

fn site_packages_candidates_for_python(python_executable: &str) -> Vec<PathBuf> {
    let python_path = PathBuf::from(normalize_runtime_path_text_for_host(python_executable));
    let Some(parent) = python_path.parent() else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    if runtime_is_windows() {
        if parent
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("scripts"))
            && let Some(env_root) = parent.parent()
        {
            candidates.push(env_root.join("Lib").join("site-packages"));
        }
        candidates.push(parent.join("Lib").join("site-packages"));
    } else if parent
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value == "bin")
        && let Some(env_root) = parent.parent()
    {
        let lib_dir = env_root.join("lib");
        if let Ok(entries) = fs::read_dir(&lib_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("python") {
                    candidates.push(path.join("site-packages"));
                }
            }
        }
        candidates.push(
            lib_dir
                .join(format!("python{}", default_python_version()))
                .join("site-packages"),
        );
    }
    candidates.sort();
    candidates.dedup();
    candidates
}

fn installed_package_specs_from_dist_info_dir(site_packages: &Path) -> Result<Vec<String>> {
    let mut versions = BTreeMap::new();
    let entries = fs::read_dir(site_packages)
        .with_context(|| format!("failed to read {}", site_packages.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let is_dist_info = path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.to_ascii_lowercase().ends_with(".dist-info"));
        if !path.is_dir() || !is_dist_info {
            continue;
        }
        let metadata_path = path.join("METADATA");
        if !metadata_path.is_file() {
            continue;
        }
        if let Some((name, version)) = dist_info_name_version(&metadata_path)?
            && !name.trim().is_empty()
            && !version.trim().is_empty()
        {
            versions.insert(name, version);
        }
    }
    if versions.is_empty() {
        bail!("no installed package metadata found");
    }
    Ok(versions
        .into_iter()
        .map(|(name, version)| format!("{name}=={version}"))
        .collect())
}

#[cfg(test)]
fn pinned_torch_package_specs_from_dist_info_dir(site_packages: &Path) -> Result<Vec<String>> {
    let packages = installed_package_specs_from_dist_info_dir(site_packages)?;
    let versions = packages
        .iter()
        .filter_map(|spec| spec.split_once("=="))
        .map(|(name, version)| (name.to_ascii_lowercase(), version.to_owned()))
        .collect::<BTreeMap<_, _>>();
    let mut specs = Vec::new();
    for name in THEROCK_TORCH_PACKAGES {
        let version = versions
            .get(*name)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("runtime metadata is missing package `{name}`"))?;
        specs.push(format!("{name}=={version}"));
    }
    Ok(specs)
}

fn dist_info_name_version(metadata_path: &Path) -> Result<Option<(String, String)>> {
    let metadata = fs::read_to_string(metadata_path)
        .with_context(|| format!("failed to read {}", metadata_path.display()))?;
    let mut name = None;
    let mut version = None;
    for line in metadata.lines() {
        if let Some(value) = line.strip_prefix("Name:") {
            name = Some(value.trim().to_ascii_lowercase());
        } else if let Some(value) = line.strip_prefix("Version:") {
            version = Some(value.trim().to_owned());
        }
        if name.is_some() && version.is_some() {
            break;
        }
    }
    Ok(name.zip(version))
}

fn parse_torch_package_version_specs(output: &str) -> Result<Vec<String>> {
    let versions: BTreeMap<String, String> =
        serde_json::from_str(output.trim()).context("failed to parse torch package versions")?;
    let mut specs = Vec::new();
    for name in THEROCK_TORCH_PACKAGES {
        let version = versions
            .get(*name)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("runtime Python is missing package `{name}`"))?;
        specs.push(format!("{name}=={version}"));
    }
    Ok(specs)
}

fn install_therock_torch_packages(
    uv: &Path,
    python_executable: &Path,
    resolution: &TheRockTorchResolution,
) -> Result<()> {
    let mut args = uv_pip_install_base(python_executable);
    args.push("--index-url".to_owned());
    args.push(resolution.index_url.clone());
    if matches!(resolution.channel, TheRockChannel::Nightly) {
        args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
    }
    args.extend(resolution.packages.iter().cloned());
    run_uv_progress_command(
        uv,
        args.iter().map(String::as_str),
        "install TheRock torch packages into managed pytorch env",
    )
}

fn therock_index_url(family: &str) -> String {
    format!("{THEROCK_SIMPLE_INDEX_BASE}/{family}/")
}

fn simple_hash(value: &str) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn find_installed_package(packages: &[String], name: &str) -> Option<String> {
    packages.iter().find_map(|entry| {
        entry
            .strip_prefix(&format!("{name}=="))
            .map(ToOwned::to_owned)
    })
}

fn manifest_has_torch(manifest: &EngineEnvManifest) -> bool {
    find_installed_package(&manifest.installed_packages, "torch").is_some()
}

fn probe_torch_runtime(python_executable: &str) -> Result<TorchRuntimeProbe> {
    let output = capture_command(
        python_executable,
        ["-c", PYTORCH_PROBE_SCRIPT],
        "probe managed pytorch runtime",
    )?;
    parse_torch_runtime_probe(&output)
}

fn parse_torch_runtime_probe(output: &str) -> Result<TorchRuntimeProbe> {
    serde_json::from_str(output.trim()).context("failed to parse managed pytorch runtime probe")
}

const PYTORCH_PROBE_SCRIPT: &str = r#"
import ast
import importlib.util
import json

def inspect_rocm_sdk():
    out = {
        "import_ok": False,
        "version": None,
        "site_packages": None,
        "default_target_family": None,
        "available_target_families": [],
        "resolved_target_family": None,
        "error": None,
    }
    try:
        import sysconfig
        import rocm_sdk
        from rocm_sdk import _dist_info as di

        out["import_ok"] = True
        out["version"] = getattr(rocm_sdk, "__version__", None)
        out["site_packages"] = sysconfig.get_paths().get("purelib")
        out["default_target_family"] = getattr(di, "DEFAULT_TARGET_FAMILY", None)
        out["available_target_families"] = list(getattr(di, "AVAILABLE_TARGET_FAMILIES", []))
        try:
            out["resolved_target_family"] = di.determine_target_family()
        except Exception as exc:
            out["error"] = type(exc).__name__ + ": " + str(exc)
    except Exception as exc:
        out["error"] = type(exc).__name__ + ": " + str(exc)
    return out

def inspect_torch_rocm_init():
    out = {
        "present": False,
        "path": None,
        "check_version": None,
        "preload_shortnames": [],
        "error": None,
    }
    try:
        spec = importlib.util.find_spec("torch")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if not locations:
            return out
        from pathlib import Path
        path = Path(locations[0]) / "_rocm_init.py"
        if not path.exists():
            return out
        out["present"] = True
        out["path"] = str(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "initialize_process"
            ):
                continue
            for kw in node.keywords:
                if kw.arg == "check_version":
                    out["check_version"] = ast.literal_eval(kw.value)
                elif kw.arg == "preload_shortnames":
                    out["preload_shortnames"] = list(ast.literal_eval(kw.value))
            break
    except Exception as exc:
        out["error"] = type(exc).__name__ + ": " + str(exc)
    return out

rocm_sdk_probe = inspect_rocm_sdk()
rocm_init_probe = inspect_torch_rocm_init()

try:
    import torch
    devices = []
    try:
        count = torch.cuda.device_count()
        devices = [torch.cuda.get_device_name(i) for i in range(count)]
    except Exception:
        count = 0
    print(json.dumps({
        "import_ok": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(count),
        "devices": devices,
        "rocm_sdk": rocm_sdk_probe,
        "torch_rocm_init": rocm_init_probe,
    }))
except Exception as exc:
    print(json.dumps({
        "import_ok": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "error": str(exc),
        "rocm_sdk": rocm_sdk_probe,
        "torch_rocm_init": rocm_init_probe,
    }))
"#;

fn run_uv_command<'a, I>(uv: &Path, args: I, context_text: &str) -> Result<()>
where
    I: IntoIterator<Item = &'a str>,
{
    let args: Vec<_> = args.into_iter().collect();
    let output = Command::new(uv)
        .args(&args)
        .envs(uv_command_env())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to launch uv for {context_text}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    bail!("{context_text}: uv exited with {}: {stderr}", output.status)
}

fn run_uv_progress_command<'a, I>(uv: &Path, args: I, context_text: &str) -> Result<()>
where
    I: IntoIterator<Item = &'a str>,
{
    let args: Vec<_> = args.into_iter().collect();
    let status = Command::new(uv)
        .args(&args)
        .envs(uv_command_env())
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("failed to launch uv for {context_text}"))?;
    if status.success() {
        return Ok(());
    }
    bail!("{context_text}: uv exited with {status}")
}

fn capture_uv_command<'a, I>(uv: &Path, args: I, context_text: &str) -> Result<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let args: Vec<_> = args.into_iter().collect();
    let output = Command::new(uv)
        .args(&args)
        .envs(uv_command_env())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .with_context(|| format!("failed to launch uv for {context_text}"))?;
    if output.status.success() {
        return String::from_utf8(output.stdout).context("uv output was not valid UTF-8");
    }
    bail!("{context_text}: uv exited with {}", output.status)
}

#[allow(dead_code)]
fn command_succeeds<'a, I>(program: &str, args: I) -> Result<bool>
where
    I: IntoIterator<Item = &'a str>,
{
    let args = args.into_iter().map(ToOwned::to_owned).collect::<Vec<_>>();
    let status = Command::new(command_program(program))
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("failed to start {}", args.join(" ")))?;
    Ok(status.success())
}

fn ensure_service_env(runtime_id: Option<&str>, env_id: Option<&str>) -> Result<EngineEnvManifest> {
    if let Some(env_id) = env_id {
        let manifest = load_manifest_by_env_id(env_id)?;
        if manifest_has_torch(&manifest) {
            return Ok(manifest);
        }
        return create_or_update_env_manifest(&InstallRequest {
            runtime_id: manifest.runtime_id.clone(),
            python_version: manifest.requested_python_version,
            env_root: None,
            reinstall: true,
        });
    }

    if let Some(runtime_id) = runtime_id {
        return create_or_update_env_manifest(&InstallRequest {
            runtime_id: runtime_id.to_owned(),
            python_version: None,
            env_root: None,
            reinstall: false,
        });
    }

    if let Some(manifest) = latest_runnable_env_manifest()? {
        return Ok(manifest);
    }

    create_or_update_env_manifest(&InstallRequest {
        runtime_id: DEFAULT_RUNTIME_ID.to_owned(),
        python_version: None,
        env_root: None,
        reinstall: false,
    })
}

fn latest_runnable_env_manifest() -> Result<Option<EngineEnvManifest>> {
    let paths = AppPaths::discover()?;
    let manifests_dir = paths.engine_manifests_dir(ENGINE_NAME);
    if !manifests_dir.is_dir() {
        return Ok(None);
    }

    let mut manifests = Vec::new();
    for entry in fs::read_dir(&manifests_dir)
        .with_context(|| format!("failed to read {}", manifests_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let manifest = load_manifest(&path)?;
        if manifest.env_path.is_dir() && manifest_has_torch(&manifest) {
            manifests.push(manifest);
        }
    }
    manifests.sort_by(|left, right| left.env_id.cmp(&right.env_id));
    Ok(manifests.pop())
}

fn load_manifest_by_env_id(env_id: &str) -> Result<EngineEnvManifest> {
    let paths = AppPaths::discover()?;
    let path = paths
        .engine_manifests_dir(ENGINE_NAME)
        .join(format!("{env_id}.json"));
    load_manifest(&path)
}

fn materialize_python_worker() -> Result<PathBuf> {
    let paths = AppPaths::discover()?;
    let worker_dir = paths.engine_dir(ENGINE_NAME).join("worker");
    fs::create_dir_all(&worker_dir)
        .with_context(|| format!("failed to create {}", worker_dir.display()))?;

    let worker_path = worker_dir.join("python_worker.py");
    let needs_write = match fs::read_to_string(&worker_path) {
        Ok(current) => current != PYTHON_WORKER_SOURCE,
        Err(_) => true,
    };
    if needs_write {
        fs::write(&worker_path, PYTHON_WORKER_SOURCE)
            .with_context(|| format!("failed to write {}", worker_path.display()))?;
    }
    Ok(worker_path)
}

fn optional_arg(flag: &str, value: Option<&str>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value.to_owned()],
        None => Vec::new(),
    }
}

fn engine_recipe_json_arg(engine_recipe: Option<&EngineRecipeHint>) -> Result<Vec<String>> {
    match engine_recipe {
        Some(engine_recipe) => Ok(vec![
            "--engine-recipe-json".to_owned(),
            serde_json::to_string(engine_recipe).context("failed to encode engine recipe hint")?,
        ]),
        None => Ok(Vec::new()),
    }
}

fn engine_recipe_launch_args(engine_recipe: Option<&EngineRecipeHint>) -> Vec<String> {
    engine_recipe
        .map(|hint| hint.required_flags.clone())
        .unwrap_or_default()
}

fn optional_arg_owned(flag: &str, value: Option<String>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value],
        None => Vec::new(),
    }
}

fn flag_arg(flag: &str, enabled: bool) -> Vec<String> {
    if enabled {
        vec![flag.to_owned()]
    } else {
        Vec::new()
    }
}

#[allow(dead_code)]
fn run_command<'a, I>(program: &str, args: I, context_label: &str) -> Result<()>
where
    I: IntoIterator<Item = &'a str>,
{
    let args = args.into_iter().map(ToOwned::to_owned).collect::<Vec<_>>();
    let output = capture_command_files(program, &args, context_label)?;
    if output.status.success() {
        Ok(())
    } else {
        bail!(
            "{} failed (status {}): {}",
            context_label,
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[allow(dead_code)]
fn run_progress_command<'a, I>(program: &str, args: I, context_label: &str) -> Result<()>
where
    I: IntoIterator<Item = &'a str>,
{
    let args = args.into_iter().map(ToOwned::to_owned).collect::<Vec<_>>();
    if interactive_terminal() || cosmo_windows_host() {
        let status = Command::new(command_program(program))
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .with_context(|| format!("failed to start {context_label}"))?;
        if status.success() {
            return Ok(());
        }
        bail!("{context_label} failed (status {status})");
    }
    if engine_progress_stderr_enabled() {
        return run_progress_command_forwarded(program, &args, context_label);
    }
    run_command(program, args.iter().map(String::as_str), context_label)
}

#[allow(dead_code)]
fn engine_progress_stderr_enabled() -> bool {
    env_value_truthy(std::env::var("ROCM_ENGINE_PROGRESS_STDERR").ok().as_deref())
}

#[allow(dead_code)]
fn env_value_truthy(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

#[allow(dead_code)]
fn run_progress_command_forwarded(
    program: &str,
    args: &[String],
    context_label: &str,
) -> Result<()> {
    let mut child = Command::new(command_program(program))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to start {context_label}"))?;
    let stdout = child
        .stdout
        .take()
        .map(forward_child_output_to_stderr)
        .context("progress child did not expose stdout")?;
    let stderr = child
        .stderr
        .take()
        .map(forward_child_output_to_stderr)
        .context("progress child did not expose stderr")?;
    let status = child
        .wait()
        .with_context(|| format!("failed waiting for {context_label}"))?;
    let stdout = stdout.join().unwrap_or_default();
    let stderr = stderr.join().unwrap_or_default();
    if status.success() {
        return Ok(());
    }
    let mut combined = Vec::with_capacity(stdout.len() + stderr.len());
    combined.extend_from_slice(&stdout);
    combined.extend_from_slice(&stderr);
    bail!(
        "{} failed (status {}): {}",
        context_label,
        status,
        String::from_utf8_lossy(&combined)
    );
}

#[allow(dead_code)]
fn forward_child_output_to_stderr<R>(mut reader: R) -> thread::JoinHandle<Vec<u8>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut collected = Vec::new();
        let mut buffer = [0_u8; 8192];
        let mut stderr = io::stderr();
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => {
                    let chunk = &buffer[..read];
                    let _ = stderr.write_all(chunk);
                    let _ = stderr.flush();
                    collected.extend_from_slice(chunk);
                }
                Err(error) => {
                    let message = format!("\nfailed to read progress output: {error}\n");
                    let _ = stderr.write_all(message.as_bytes());
                    collected.extend_from_slice(message.as_bytes());
                    break;
                }
            }
        }
        collected
    })
}

fn capture_command<'a, I>(program: &str, args: I, context_label: &str) -> Result<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let args = args.into_iter().map(ToOwned::to_owned).collect::<Vec<_>>();
    let output = capture_command_files(program, &args, context_label)?;
    if !output.status.success() {
        bail!(
            "{} failed (status {}): {}",
            context_label,
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).context("command output was not valid utf-8")
}

fn command_program(program: &str) -> String {
    normalize_runtime_path_text_for_host(program)
}

fn command_path(path: &Path) -> String {
    normalize_runtime_path_text_for_host(&path.display().to_string())
}

const fn cosmo_windows_host() -> bool {
    runtime_is_windows() && !cfg!(windows)
}

struct CapturedCommand {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn capture_command_files(
    program: &str,
    args: &[String],
    context_label: &str,
) -> Result<CapturedCommand> {
    let temp_dir = std::env::temp_dir();
    let stem = format!(
        "rocm-pytorch-command-{}-{}",
        std::process::id(),
        unix_time_millis()
    );
    let stdout_path = temp_dir.join(format!("{stem}.out"));
    let stderr_path = temp_dir.join(format!("{stem}.err"));
    let stdout_file = fs::File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let status_result = Command::new(command_program(program))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .status()
        .with_context(|| format!("failed to start {context_label}"));
    let stdout = fs::read(&stdout_path).unwrap_or_default();
    let stderr = fs::read(&stderr_path).unwrap_or_default();
    let _ = fs::remove_file(&stdout_path);
    let _ = fs::remove_file(&stderr_path);
    Ok(CapturedCommand {
        status: status_result?,
        stdout,
        stderr,
    })
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

impl From<DevicePolicyArg> for DevicePolicy {
    fn from(value: DevicePolicyArg) -> Self {
        match value {
            DevicePolicyArg::GpuRequired => Self::GpuRequired,
            DevicePolicyArg::GpuPreferred => Self::GpuPreferred,
            DevicePolicyArg::CpuOnly => Self::CpuOnly,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine_recipe(engine: &str, contract_version: &str) -> EngineRecipeHint {
        EngineRecipeHint {
            contract_version: contract_version.to_owned(),
            engine: engine.to_owned(),
            required_flags: vec!["--trust-remote-code=false".to_owned()],
            parser_settings: BTreeMap::default(),
            preferred_endpoint: None,
            unsupported_combinations: Vec::new(),
            notes: vec!["test recipe".to_owned()],
        }
    }

    #[test]
    fn normalize_therock_family_maps_gfx1103_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1103"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1101_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1101"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn tail_lines_from_text_returns_requested_suffix() {
        assert_eq!(
            tail_lines_from_text("one\ntwo\nthree\nfour\n", 2),
            vec!["three".to_owned(), "four".to_owned()]
        );
        assert!(tail_lines_from_text("one\ntwo\n", 0).is_empty());
    }

    #[test]
    fn healthcheck_response_maps_ready_state() {
        let state = json!({
            "status": "ready",
            "device": "cuda",
            "queue_depth": 3,
            "tokens_per_sec": 42.5
        });
        let response = build_healthcheck_response(
            Some(&state),
            Some(UNIX_EPOCH + Duration::from_secs(5)),
            None,
            None,
            None,
            UNIX_EPOCH + Duration::from_secs(15),
        );

        assert_eq!(response.status, "ready");
        assert!(response.model_loaded);
        assert_eq!(response.device, "cuda");
        assert_eq!(response.uptime_sec, 10);
        assert_eq!(response.queue_depth, 3);
        assert_eq!(response.tokens_per_sec, Some(42.5));
    }

    #[test]
    fn healthcheck_response_marks_ready_state_unreachable_when_probe_fails() {
        let state = json!({
            "status": "ready",
            "device": "cpu"
        });
        let response = build_healthcheck_response(
            Some(&state),
            None,
            None,
            None,
            Some("connection refused".to_owned()),
            UNIX_EPOCH,
        );

        assert_eq!(response.status, "unreachable");
        assert!(!response.model_loaded);
        assert_eq!(response.device, "cpu");
        assert_eq!(response.last_error.as_deref(), Some("connection refused"));
    }

    #[test]
    fn parse_http_endpoint_supports_v1_urls() {
        assert_eq!(
            parse_http_endpoint("http://127.0.0.1:11435/v1"),
            Some(("127.0.0.1".to_owned(), 11435))
        );
        assert_eq!(
            parse_http_endpoint("http://[::1]:11435/v1"),
            Some(("::1".to_owned(), 11435))
        );
    }

    #[test]
    fn endpoint_url_fallback_brackets_ipv6_loopback() {
        let state = json!({
            "host": "::1",
            "port": 11435
        });

        assert_eq!(
            endpoint_url_from_state(&state),
            Some("http://[::1]:11435/v1".to_owned())
        );
    }

    #[test]
    fn serve_http_cli_defaults_to_gpu_required_without_cpu_fallback() {
        let cli = Cli::try_parse_from([
            "rocm-engine-pytorch",
            "serve-http",
            "svc",
            "qwen",
            "--env-id",
            "env-1",
            "--runtime-id",
            "therock-release:gfx120X-all",
            "--state-path",
            "state.json",
        ])
        .expect("serve-http args should parse");

        match cli.command {
            CommandKind::ServeHttp { device_policy, .. } => {
                assert_eq!(device_policy, "gpu_required");
            }
            _ => panic!("expected serve-http command"),
        }
    }

    #[test]
    fn launch_cli_accepts_runtime_selection_args() {
        let cli = Cli::try_parse_from([
            "rocm-engine-pytorch",
            "launch",
            "svc",
            "qwen",
            "--runtime-id",
            "therock-release:gfx120X-all",
            "--env-id",
            "pytorch-env-1",
        ])
        .expect("launch should accept protocol runtime args");

        match cli.command {
            CommandKind::Launch {
                runtime_id, env_id, ..
            } => {
                assert_eq!(runtime_id.as_deref(), Some("therock-release:gfx120X-all"));
                assert_eq!(env_id.as_deref(), Some("pytorch-env-1"));
            }
            _ => panic!("expected launch command"),
        }
    }

    #[test]
    fn python_worker_disallows_implicit_cpu_fallback_for_gpu_policies() {
        let worker = include_str!("python_worker.py");

        assert!(!worker.contains("cpu_fallback"));
        assert!(worker.contains("policy in {\"gpu_required\", \"gpu_preferred\"}"));
        assert!(worker.contains("PyTorch CPU serving is not offered by rocm-cli"));
        assert!(worker.contains("no CPU fallback is used"));
    }

    #[test]
    fn python_worker_bridges_qwen_xml_tool_calls_to_openai_tool_calls() {
        let worker = include_str!("python_worker.py");

        assert!(worker.contains("kwargs[\"tools\"] = tools"));
        assert!(worker.contains("TOOL_CALL_PATTERN"));
        assert!(worker.contains("\"tool_calls\""));
        assert!(worker.contains("\"finish_reason\": finish_reason"));
    }

    #[test]
    fn pytorch_capabilities_advertise_tool_calling() {
        assert!(capabilities().tool_calling);
    }

    #[test]
    fn engine_progress_forwarding_env_uses_explicit_truthy_values() {
        for value in [Some("1"), Some("true"), Some("YES"), Some(" on ")] {
            assert!(env_value_truthy(value));
        }
        for value in [None, Some(""), Some("0"), Some("false"), Some("please")] {
            assert!(!env_value_truthy(value));
        }
    }

    #[test]
    fn cpu_policy_is_rejected_without_fallback() {
        let error = normalize_pytorch_device_policy(DevicePolicy::CpuOnly)
            .expect_err("cpu policy should not be accepted by rocm-cli");
        assert!(error.to_string().contains("no CPU fallback is used"));
    }

    #[test]
    fn stdio_protocol_routes_all_methods_without_side_effects() {
        let service_id = format!(
            "missing-protocol-{}-{}",
            std::process::id(),
            rocm_core::unix_time_millis()
        );
        let ok_cases = [
            (EngineMethod::Detect, json!({})),
            (EngineMethod::Capabilities, json!({})),
            (
                EngineMethod::ResolveModel,
                json!({
                    "model_ref": "qwen",
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
                EngineMethod::Endpoint,
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

        for (method, payload) in ok_cases {
            let response = handle_envelope(EngineRequestEnvelope { method, payload });
            assert!(
                response.ok,
                "expected protocol method to return a typed success envelope: {:?}",
                response.error
            );
        }

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
    fn endpoint_response_uses_service_state_endpoint() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-pytorch-endpoint-{}",
            rocm_core::unix_time_millis()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root)?;
        let state_path = root.join("state.json");
        fs::write(
            &state_path,
            serde_json::to_vec(&json!({
                "status": "ready",
                "endpoint_url": "http://127.0.0.1:32123/v1"
            }))?,
        )?;
        let files = ServiceFiles {
            record_path: root.join("service.json"),
            record: None,
            record_matches_engine: false,
            state_path,
            log_path: root.join("service.log"),
        };

        let response = endpoint_response_from_files(&files)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(response.endpoint_url, "http://127.0.0.1:32123/v1");
        assert_eq!(response.api_style, "openai");
        assert!(response.supported_routes.contains(&"/healthz".to_owned()));
        assert!(
            response
                .supported_routes
                .contains(&"/v1/chat/completions".to_owned())
        );
        Ok(())
    }

    #[test]
    fn manifest_python_path_falls_back_to_existing_venv_interpreter() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-pytorch-python-fallback-{}",
            rocm_core::unix_time_millis()
        ));
        let _ = fs::remove_dir_all(&root);
        let env_path = root.join("env");
        let bin_dir = if cfg!(windows) {
            env_path.join("Scripts")
        } else {
            env_path.join("bin")
        };
        fs::create_dir_all(&bin_dir)?;
        let missing_recorded = if cfg!(windows) {
            bin_dir.join("python.exe")
        } else {
            bin_dir.join("python")
        };
        let existing = if cfg!(windows) {
            bin_dir.join("python3.exe")
        } else {
            bin_dir.join("python3")
        };
        fs::write(&existing, "python")?;

        let manifest = EngineEnvManifest {
            env_id: "env".to_owned(),
            runtime_id: "runtime".to_owned(),
            requested_python_version: Some("3.12".to_owned()),
            python_launcher: "python3.12".to_owned(),
            python_executable: missing_recorded.display().to_string(),
            env_path,
            manifest_path: root.join("manifest.json"),
            lock_path: root.join("lock.txt"),
            installed_packages: vec!["torch==2.11.0".to_owned()],
            lock_hash: "hash".to_owned(),
            pip_cache_dir: None,
            therock_channel: None,
            therock_family: None,
            therock_index_url: None,
            therock_packages: Vec::new(),
            torch_runtime_probe: None,
            warnings: Vec::new(),
        };

        let resolved = resolve_manifest_python_executable(&manifest);
        fs::remove_dir_all(root).ok();

        assert_eq!(resolved, existing);
        Ok(())
    }

    #[test]
    fn parses_torch_runtime_probe() -> Result<()> {
        let probe = parse_torch_runtime_probe(
            r#"{"import_ok":true,"torch_version":"2.9.0+rocm","cuda_available":true,"device_count":1,"devices":["AMD Radeon"]}"#,
        )?;
        assert!(probe.import_ok);
        assert_eq!(probe.torch_version.as_deref(), Some("2.9.0+rocm"));
        assert!(probe.cuda_available);
        assert_eq!(probe.device_count, 1);
        assert_eq!(probe.devices, vec!["AMD Radeon".to_owned()]);
        Ok(())
    }

    #[test]
    fn parses_torch_runtime_probe_with_rocm_init_contract() -> Result<()> {
        let probe = parse_torch_runtime_probe(
            r#"{"import_ok":true,"torch_version":"2.10.0+rocm7.13.0a20260423","cuda_available":true,"device_count":1,"devices":["AMD Radeon RX 9070 XT"],"rocm_sdk":{"import_ok":true,"version":"7.13.0a20260423","site_packages":"C:\\venv\\Lib\\site-packages","default_target_family":"gfx1151","available_target_families":["gfx1151"],"resolved_target_family":"gfx1151","error":null},"torch_rocm_init":{"present":true,"path":"C:\\venv\\Lib\\site-packages\\torch\\_rocm_init.py","check_version":"7.13.0a20260423","preload_shortnames":["amd_comgr","amdhip64","hipblas"],"error":null}}"#,
        )?;

        let sdk = probe.rocm_sdk.as_ref().expect("rocm_sdk probe");
        assert!(sdk.import_ok);
        assert_eq!(sdk.version.as_deref(), Some("7.13.0a20260423"));
        assert_eq!(sdk.resolved_target_family.as_deref(), Some("gfx1151"));

        let init = probe
            .torch_rocm_init
            .as_ref()
            .expect("torch._rocm_init probe");
        assert!(init.present);
        assert_eq!(init.check_version.as_deref(), Some("7.13.0a20260423"));
        assert_eq!(
            init.preload_shortnames,
            vec![
                "amd_comgr".to_owned(),
                "amdhip64".to_owned(),
                "hipblas".to_owned()
            ]
        );
        Ok(())
    }

    #[test]
    fn uv_install_base_targets_venv_python() {
        let python = PathBuf::from("/envs/pytorch/bin/python");
        let args = uv_pip_install_base(&python);
        assert_eq!(args[0], "pip");
        assert_eq!(args[1], "install");
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--python", "/envs/pytorch/bin/python"])
        );
    }

    #[test]
    fn therock_torch_package_specs_include_full_torch_stack() {
        assert_eq!(
            THEROCK_TORCH_PACKAGES,
            &["torch", "torchvision", "torchaudio"]
        );
    }

    #[test]
    fn engine_dependencies_keep_transformers_on_windows_rocm_compatible_line() {
        assert!(ENGINE_DEPENDENCIES.contains(&"transformers<5"));
        assert!(ENGINE_DEPENDENCIES.contains(&"huggingface_hub<1"));
        assert!(!ENGINE_DEPENDENCIES.contains(&"transformers"));
    }

    #[test]
    fn parses_runtime_torch_versions_as_exact_pip_specs() -> Result<()> {
        let specs = parse_torch_package_version_specs(
            r#"{"torch":"2.11.0+rocm7.13.0a20260416","torchvision":"0.26.0+rocm7.13.0a20260416","torchaudio":"2.11.0+rocm7.13.0a20260416"}"#,
        )?;
        assert_eq!(
            specs,
            vec![
                "torch==2.11.0+rocm7.13.0a20260416",
                "torchvision==0.26.0+rocm7.13.0a20260416",
                "torchaudio==2.11.0+rocm7.13.0a20260416",
            ]
        );
        Ok(())
    }

    #[test]
    fn reads_runtime_torch_versions_from_dist_info_metadata() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-pytorch-dist-info-{}",
            rocm_core::unix_time_millis()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root)?;
        for (name, version) in [
            ("torch", "2.10.0+rocm7.13.0a20260511"),
            ("torchvision", "0.25.0+rocm7.13.0a20260511"),
            ("torchaudio", "2.10.0+rocm7.13.0a20260511"),
        ] {
            let dist_info = root.join(format!("{name}-{version}.dist-info"));
            fs::create_dir_all(&dist_info)?;
            fs::write(
                dist_info.join("METADATA"),
                format!("Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"),
            )?;
        }

        let specs = pinned_torch_package_specs_from_dist_info_dir(&root)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            specs,
            vec![
                "torch==2.10.0+rocm7.13.0a20260511",
                "torchvision==0.25.0+rocm7.13.0a20260511",
                "torchaudio==2.10.0+rocm7.13.0a20260511",
            ]
        );
        Ok(())
    }

    #[test]
    fn runtime_torch_version_parse_requires_full_stack() {
        let error = parse_torch_package_version_specs(
            r#"{"torch":"2.11.0+rocm7.13.0a20260416","torchvision":"0.26.0+rocm7.13.0a20260416"}"#,
        )
        .expect_err("missing torchaudio must fail");
        assert!(error.to_string().contains("torchaudio"));
    }

    #[test]
    fn known_model_recipe_comes_from_shared_registry() {
        let recipe = resolve_model_recipe("qwen").expect("recipe should resolve");
        assert_eq!(recipe.canonical_model_id, "Qwen/Qwen2.5-1.5B-Instruct");
        assert_eq!(recipe.source, "alias");
        assert_eq!(recipe.preferred_dtype, "bfloat16");
        assert_eq!(recipe.device_policy, DevicePolicy::GpuPreferred);
    }

    #[test]
    fn qwen35_is_rejected_before_pytorch_launch() {
        let error =
            resolve_model_recipe("qwen3.5").expect_err("qwen3.5 should be gated before launch");

        assert!(error.to_string().contains("unknown architecture `qwen3_5`"));
        assert!(error.to_string().contains("Use `qwen`"));
    }

    #[test]
    fn engine_recipe_overrides_supported_pytorch_worker_settings() -> Result<()> {
        let mut hint = test_engine_recipe(ENGINE_NAME, ENGINE_RECIPE_CONTRACT_VERSION);
        hint.required_flags = vec![
            "--trust-remote-code".to_owned(),
            "--preferred-dtype".to_owned(),
            "float16".to_owned(),
            "--min-gpu-mem-gb=16".to_owned(),
        ];

        let recipe =
            apply_pytorch_engine_recipe_overrides(resolve_model_recipe("tiny-gpt2")?, Some(&hint))?;

        assert!(recipe.trust_remote_code);
        assert_eq!(recipe.preferred_dtype, "float16");
        assert_eq!(recipe.min_gpu_mem_gb, Some(16));
        assert!(recipe.estimated_memory.contains("16 GiB"));
        Ok(())
    }

    #[test]
    fn engine_recipe_rejects_unknown_pytorch_launch_flags() {
        let mut hint = test_engine_recipe(ENGINE_NAME, ENGINE_RECIPE_CONTRACT_VERSION);
        hint.required_flags = vec!["--enable-auto-tool-choice".to_owned()];

        let error = apply_pytorch_engine_recipe_overrides(
            resolve_model_recipe("tiny-gpt2").unwrap(),
            Some(&hint),
        )
        .expect_err("unknown PyTorch launch flag should fail");

        assert!(
            error
                .to_string()
                .contains("unsupported PyTorch launch recipe flag")
        );
    }

    #[test]
    fn resolve_model_echoes_matching_engine_recipe() -> Result<()> {
        let hint = test_engine_recipe(ENGINE_NAME, ENGINE_RECIPE_CONTRACT_VERSION);
        let response = resolve_model_response(ResolveModelRequest {
            model_ref: "qwen".to_owned(),
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
            model_ref: "qwen".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(test_engine_recipe("vllm", ENGINE_RECIPE_CONTRACT_VERSION)),
        })
        .expect_err("mismatched engine recipe should fail");

        assert!(error.to_string().contains("does not match adapter"));
    }

    #[test]
    fn resolve_model_rejects_unsupported_engine_recipe_contract() {
        let error = resolve_model_response(ResolveModelRequest {
            model_ref: "qwen".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(test_engine_recipe(ENGINE_NAME, "999.0.0")),
        })
        .expect_err("unsupported recipe contract should fail");

        assert!(error.to_string().contains("unsupported"));
    }

    #[test]
    fn tiny_model_recipe_uses_gpu_policy_from_registry() {
        let recipe = resolve_model_recipe("tiny-gpt2").expect("recipe should resolve");
        assert_eq!(recipe.canonical_model_id, "sshleifer/tiny-gpt2");
        assert_eq!(recipe.device_policy, DevicePolicy::GpuRequired);
        assert_eq!(recipe.min_gpu_mem_gb, Some(2));
        assert!(!recipe.trust_remote_code);
    }
}
