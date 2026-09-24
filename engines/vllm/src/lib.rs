// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rocm_core::{AppPaths, DEFAULT_LOCAL_PORT};
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, DevicePolicy, EndpointRequest, EngineCapabilities,
    EngineDeviceAvailability, EngineMethod, EngineRecipeHint, EngineRequestEnvelope,
    EngineResponseEnvelope, HealthcheckRequest, InstallRequest, LaunchRequest, LogsRequest,
    ResolveModelRequest, StopRequest,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::io::{Read, Write};
use std::path::PathBuf;

mod install;
mod process;
mod runtime;
mod state;

pub(crate) const ENGINE_NAME: &str = "vllm";

pub(crate) const DEFAULT_HOST: &str = "127.0.0.1";

#[derive(Parser, Debug)]
#[command(name = "rocm-engine-vllm", about = "rocm-cli vLLM engine adapter")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand, Debug)]
enum CommandKind {
    Detect,
    Capabilities,
    Install {
        #[arg(long, default_value = "external-vllm")]
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
        gpu: Option<String>,
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

pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Detect => print_json(&detect_response())?,
        CommandKind::Capabilities => print_json(&capabilities())?,
        CommandKind::Install {
            runtime_id,
            reinstall,
        } => print_json(&crate::install::install_response(InstallRequest {
            runtime_id,
            python_version: None,
            env_root: None,
            reinstall,
        })?)?,
        CommandKind::ResolveModel {
            model_ref,
            device_policy,
        } => print_json(&crate::process::resolve_model_response(
            ResolveModelRequest {
                model_ref,
                runtime_id: None,
                device_policy: device_policy
                    .as_deref()
                    .map(|value| crate::process::parse_device_policy_arg(Some(value)))
                    .transpose()?,
                recipe_override: None,
                engine_recipe: None,
            },
        )?)?,
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
            device_policy: Some(crate::process::parse_device_policy_arg(
                device_policy.as_deref(),
            )?),
            endpoint_mode: Some("openai".to_owned()),
            engine_recipe: None,
            gpu_selection: crate::process::parse_gpu_selection_arg(gpu.as_deref())?,
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
            gpu,
            runtime_id,
            env_id,
            state_path,
            engine_recipe_json,
        } => {
            let log_path = AppPaths::discover()?
                .engine_logs_dir(ENGINE_NAME)
                .join(format!("{service_id}.log"));
            crate::process::serve_http(crate::process::ServeHttpRequest {
                service_id,
                model_ref,
                host,
                port,
                device_policy: crate::process::parse_device_policy_arg(Some(&device_policy))?,
                gpu_indices: crate::process::parse_gpu_indices_arg(gpu.as_deref())?,
                runtime_id,
                env_id,
                state_path,
                log_path: Some(log_path),
                engine_recipe: crate::process::parse_engine_recipe_json(engine_recipe_json)?,
            })?;
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
    gpu_indices: Vec<u32>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<()> {
    crate::process::serve_http(crate::process::ServeHttpRequest {
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
        EngineMethod::Install => deserialize_and_respond::<InstallRequest, _, _>(
            envelope.payload,
            crate::install::install_response,
        ),
        EngineMethod::ResolveModel => deserialize_and_respond::<ResolveModelRequest, _, _>(
            envelope.payload,
            crate::process::resolve_model_response,
        ),
        EngineMethod::Launch => deserialize_and_respond::<LaunchRequest, _, _>(
            envelope.payload,
            crate::process::launch_service,
        ),
        EngineMethod::Healthcheck => deserialize_and_respond::<HealthcheckRequest, _, _>(
            envelope.payload,
            crate::state::healthcheck_service,
        ),
        EngineMethod::Endpoint => deserialize_and_respond::<EndpointRequest, _, _>(
            envelope.payload,
            crate::state::endpoint_response,
        ),
        EngineMethod::Stop => deserialize_and_respond::<StopRequest, _, _>(
            envelope.payload,
            crate::process::stop_service,
        ),
        EngineMethod::Logs => deserialize_and_respond::<LogsRequest, _, _>(
            envelope.payload,
            crate::state::logs_response,
        ),
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
    let runtime = crate::runtime::resolve_vllm_runtime(None);
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
            "vLLM command resolved from {}; no CPU fallback is used",
            runtime.source
        ));
    }

    DetectResponse {
        installed,
        env_id: runtime.as_ref().map(|runtime| runtime.env_id.clone()),
        runtime_kind: Some("external_vllm".to_owned()),
        runtime_executable: runtime
            .as_ref()
            .map(|runtime| runtime.command.display().to_string()),
        managed_env: Some(
            runtime
                .as_ref()
                .is_some_and(crate::runtime::runtime_is_managed),
        ),
        python_version: runtime
            .as_ref()
            .and_then(|runtime| runtime.version.as_ref())
            .map(|version| format!("vllm {version}")),
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
                Some("vLLM is not installed in a Linux/WSL ROCm Python environment".to_owned())
            },
        }],
        capabilities: capabilities(),
        notes,
    }
}

pub(crate) fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        cpu: false,
        rocm_gpu: !cfg!(windows),
        openai_compatible: true,
        tool_calling: false,
        quantized_models: "vllm-supported".to_owned(),
        reasoning_parser: false,
    }
}

pub(crate) const fn windows_unsupported_message() -> &'static str {
    "vLLM ROCm serving is supported by rocm-cli only on Linux/WSL; native Windows vLLM is skipped. No CPU fallback is used."
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
    use serde_json::json;

    #[test]
    fn stdio_protocol_routes_all_methods_without_side_effects() {
        let service_id = format!(
            "missing-protocol-{}-{}",
            std::process::id(),
            crate::state::current_unix_millis()
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
}
