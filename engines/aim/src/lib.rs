// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rocm_core::AppPaths;
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, DevicePolicy, EngineCapabilities, EngineDeviceAvailability,
    EngineMethod, EngineRecipeHint, EngineRequestEnvelope, EngineResponseEnvelope, InstallRequest,
    LaunchRequest, ResolveModelRequest,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::io::{Read, Write};
use std::path::PathBuf;

mod docker;
mod install;
mod process;
mod registry;
mod state;

pub(crate) const ENGINE_NAME: &str = "aim";
pub(crate) const DEFAULT_HOST: &str = "127.0.0.1";
/// AIM images bundle vLLM, which defaults to port 8000 inside the container.
pub(crate) const DEFAULT_AIM_PORT: u16 = 8000;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand)]
enum CommandKind {
    Detect,
    Capabilities,
    Install {
        #[arg(long, default_value = "aim")]
        runtime_id: String,
        #[arg(long)]
        reinstall: bool,
    },
    ResolveModel {
        #[arg(long)]
        model_ref: String,
        #[arg(long)]
        device_policy: Option<String>,
    },
    Launch {
        #[arg(long)]
        service_id: String,
        #[arg(long)]
        model_ref: String,
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        #[arg(long, default_value_t = DEFAULT_AIM_PORT)]
        port: u16,
        #[arg(long)]
        device_policy: Option<String>,
        #[arg(long)]
        gpu: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
    },
    Stdio,
    ServeHttp {
        #[arg(long)]
        service_id: String,
        #[arg(long)]
        model_ref: String,
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        #[arg(long, default_value_t = DEFAULT_AIM_PORT)]
        port: u16,
        #[arg(long, default_value = "gpu_required")]
        device_policy: String,
        #[arg(long)]
        gpu: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
        #[arg(long)]
        state_path: PathBuf,
        #[arg(long)]
        engine_recipe_json: Option<String>,
    },
}

pub(crate) struct ServeHttpRequest {
    pub service_id: String,
    pub model_ref: String,
    pub host: String,
    pub port: u16,
    pub device_policy: DevicePolicy,
    pub gpu_indices: Vec<u32>,
    #[allow(dead_code)] // ponytail: accepted for protocol-shape parity; AIM has no runtime concept
    pub runtime_id: Option<String>,
    #[allow(dead_code)]
    // ponytail: accepted for protocol-shape parity; AIM has no managed-env concept
    pub env_id: Option<String>,
    pub state_path: PathBuf,
    #[allow(dead_code)]
    // ponytail: no host-side log file in `docker run -d` mode; logs come from `docker logs`
    pub log_path: Option<PathBuf>,
    pub engine_recipe: Option<EngineRecipeHint>,
}

pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Detect => print_json(&detect_response()),
        CommandKind::Capabilities => print_json(&capabilities()),
        CommandKind::Install {
            runtime_id,
            reinstall,
        } => print_json(&crate::install::install_response(InstallRequest {
            runtime_id,
            python_version: None,
            reinstall,
            env_root: None,
        })?),
        CommandKind::ResolveModel {
            model_ref,
            device_policy,
        } => print_json(&crate::process::resolve_model_response(
            ResolveModelRequest {
                model_ref,
                runtime_id: None,
                device_policy: device_policy
                    .as_deref()
                    .map(crate::process::parse_device_policy_arg)
                    .transpose()?,
                recipe_override: None,
                engine_recipe: None,
            },
        )?),
        CommandKind::Launch {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            gpu,
            runtime_id,
            env_id,
        } => print_json(&crate::process::launch_service(LaunchRequest {
            service_id,
            env_id,
            runtime_id,
            model_ref,
            host,
            port,
            device_policy: device_policy
                .as_deref()
                .map(crate::process::parse_device_policy_arg)
                .transpose()?,
            endpoint_mode: None,
            engine_recipe: None,
            gpu_selection: crate::process::parse_gpu_selection_arg(gpu.as_deref())?,
        })?),
        CommandKind::Stdio => {
            let envelope = read_request()?;
            let response = handle_envelope(envelope);
            print_json(&response)
        }
        CommandKind::ServeHttp {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            gpu,
            runtime_id,
            env_id,
            state_path,
            engine_recipe_json,
        } => {
            let gpu_indices = crate::process::parse_gpu_indices_arg(gpu.as_deref())?;
            let log_path = Some(
                AppPaths::discover()?
                    .engine_logs_dir(ENGINE_NAME)
                    .join(format!("{service_id}.log")),
            );
            crate::process::serve_http(ServeHttpRequest {
                service_id,
                model_ref,
                host,
                port,
                device_policy: crate::process::parse_device_policy_arg(&device_policy)?,
                gpu_indices,
                runtime_id,
                env_id,
                state_path,
                log_path,
                engine_recipe: crate::process::parse_engine_recipe_json(engine_recipe_json)?,
            })
        }
    }
}

pub fn builtin_handle_envelope(envelope: EngineRequestEnvelope) -> EngineResponseEnvelope {
    handle_envelope(envelope)
}

/// Whether the Docker daemon is reachable on this host.
///
/// Exposed so `apps/rocm`'s host-fit gate can rule the `aim` engine out
/// before a model is served, not just report it after the fact via `Detect`.
pub fn builtin_docker_daemon_reachable() -> bool {
    crate::docker::daemon_reachable()
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
    crate::process::serve_http(ServeHttpRequest {
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
            deserialize_and_respond::<DetectRequest, _, _>(envelope.payload, |_request| {
                Ok(detect_response())
            })
        }
        EngineMethod::Capabilities => EngineResponseEnvelope::success(capabilities()),
        EngineMethod::Install => {
            deserialize_and_respond(envelope.payload, crate::install::install_response)
        }
        EngineMethod::ResolveModel => {
            deserialize_and_respond(envelope.payload, crate::process::resolve_model_response)
        }
        EngineMethod::Launch => {
            deserialize_and_respond(envelope.payload, crate::process::launch_service)
        }
        EngineMethod::Healthcheck => {
            deserialize_and_respond(envelope.payload, crate::state::healthcheck_service)
        }
        EngineMethod::Endpoint => {
            deserialize_and_respond(envelope.payload, crate::state::endpoint_response)
        }
        EngineMethod::Stop => {
            deserialize_and_respond(envelope.payload, crate::process::stop_service)
        }
        EngineMethod::Logs => {
            deserialize_and_respond(envelope.payload, crate::state::logs_response)
        }
    }
}

fn deserialize_and_respond<T, F, U>(payload: Value, handler: F) -> EngineResponseEnvelope
where
    T: DeserializeOwned,
    F: FnOnce(T) -> Result<U>,
    U: Serialize,
{
    let request: T = match serde_json::from_value(payload) {
        Ok(request) => request,
        Err(error) => {
            return EngineResponseEnvelope::failure("invalid_payload", error.to_string());
        }
    };
    match handler(request) {
        Ok(response) => EngineResponseEnvelope::success(response),
        Err(error) => EngineResponseEnvelope::failure("request_failed", error.to_string()),
    }
}

fn detect_response() -> DetectResponse {
    let reachable = crate::docker::daemon_reachable();
    let mut notes = Vec::new();
    if !reachable {
        notes.push(
            "docker daemon is not reachable; install Docker and ensure it is running to use the AIM engine"
                .to_owned(),
        );
    }
    DetectResponse {
        installed: reachable,
        env_id: None,
        runtime_kind: Some("docker".to_owned()),
        runtime_executable: Some("docker".to_owned()),
        managed_env: Some(false),
        python_version: None,
        torch_version: None,
        transformers_version: None,
        available_devices: vec![EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: reachable,
            reason: if reachable {
                None
            } else {
                Some("docker daemon unreachable".to_owned())
            },
        }],
        capabilities: capabilities(),
        notes,
    }
}

pub(crate) fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        cpu: true,
        rocm_gpu: true,
        openai_compatible: true,
        tool_calling: false,
        quantized_models: "image-reported".to_owned(),
        reasoning_parser: false,
    }
}

fn read_request() -> Result<EngineRequestEnvelope> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .context("failed to read engine stdio request")?;
    serde_json::from_str(&input).context("failed to parse engine stdio request")
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer_pretty(&mut handle, value)?;
    writeln!(handle)?;
    Ok(())
}
