// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
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
    pub state_path: PathBuf,
    pub engine_recipe: Option<EngineRecipeHint>,
}

pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Detect => print_json(&detect_response(None)),
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
            // AIM has no managed runtime or env; accepted for protocol-shape parity.
            runtime_id: _,
            env_id: _,
            state_path,
            engine_recipe_json,
        } => {
            let gpu_indices = crate::process::parse_gpu_indices_arg(gpu.as_deref())?;
            crate::process::serve_http(ServeHttpRequest {
                service_id,
                model_ref,
                host,
                port,
                device_policy: crate::process::parse_device_policy_arg(&device_policy)?,
                gpu_indices,
                state_path,
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
    // AIM has no managed runtime, env or host-side log file (container logs come
    // from `docker logs`); accepted for parity with the other built-in engines.
    _runtime_id: Option<String>,
    _env_id: Option<String>,
    state_path: PathBuf,
    _log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<()> {
    crate::process::serve_http(ServeHttpRequest {
        service_id,
        model_ref,
        host,
        port,
        device_policy,
        gpu_indices,
        state_path,
        engine_recipe,
    })
}

fn handle_envelope(envelope: EngineRequestEnvelope) -> EngineResponseEnvelope {
    match envelope.method {
        EngineMethod::Detect => {
            deserialize_and_respond::<DetectRequest, _, _>(envelope.payload, |request| {
                Ok(detect_response(request.runtime_id))
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

/// `env_id` echoes back `runtime_id` when Docker is reachable, matching what
/// `install_response` reports for the same `runtime_id` -- aim has no actual
/// install step beyond "Docker is present", so Detect and Install have to
/// agree on readiness or every serve re-runs Install for nothing.
fn detect_response(runtime_id: Option<String>) -> DetectResponse {
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
        env_id: reachable.then_some(runtime_id).flatten(),
        runtime_kind: Some("docker".to_owned()),
        runtime_executable: Some("docker".to_owned()),
        managed_env: Some(false),
        python_version: None,
        torch_version: None,
        transformers_version: None,
        available_devices: vec![EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: reachable && !cfg!(windows),
            reason: if cfg!(windows) {
                Some(windows_gpu_unsupported_message().to_owned())
            } else if reachable {
                None
            } else {
                Some("docker daemon unreachable".to_owned())
            },
        }],
        capabilities: capabilities(),
        notes,
    }
}

#[cfg(all(test, unix))]
mod detect_tests {
    use super::detect_response;
    use rocm_core::fake_executable::write_fake_executable;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    const DOCKER_REACHABLE_TEST_DIR: &str = "ROCM_AIM_DOCKER_REACHABLE_TEST_DIR";

    /// Child half of [`detect_response_echoes_runtime_id_as_env_id_when_docker_is_reachable`].
    ///
    /// Runs in its own process so the `PATH` override (a fake `docker` that
    /// always succeeds) can be set with `Command::env` -- mutating `PATH`
    /// in-process would need `unsafe`, which the workspace denies.
    #[test]
    fn docker_reachable_subprocess_helper() {
        if std::env::var_os(DOCKER_REACHABLE_TEST_DIR).is_none() {
            return;
        }
        let response = detect_response(Some("aim".to_owned()));
        assert_eq!(
            response.env_id,
            Some("aim".to_owned()),
            "env_id must echo the requested runtime_id once Detect reports installed, or \
             `ensure_self_managed_engine_ready` will treat aim as never installed"
        );
    }

    /// Fix for the review finding that `ensure_self_managed_engine_ready`
    /// re-runs Install on every aim serve: Detect always reported `env_id:
    /// None`, which could never equal the `runtime_id` the caller compares it
    /// against.
    #[test]
    fn detect_response_echoes_runtime_id_as_env_id_when_docker_is_reachable() {
        let dir = std::env::temp_dir().join(format!(
            "aim-detect-docker-reachable-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fake_bin = dir.join("bin");
        std::fs::create_dir_all(&fake_bin).unwrap();
        write_fake_executable(&fake_bin.join("docker"), "#!/bin/sh\nexit 0\n").unwrap();

        let mut entries = vec![fake_bin];
        entries.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));

        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "detect_tests::docker_reachable_subprocess_helper",
                "--nocapture",
            ])
            .env(DOCKER_REACHABLE_TEST_DIR, PathBuf::from("1"))
            .env("PATH", std::env::join_paths(entries).unwrap())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "helper subprocess failed");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_response_has_no_env_id_when_runtime_id_is_absent() {
        // Standalone CLI mode (`rocm-engine-aim detect`) passes no runtime_id;
        // there is nothing to echo back regardless of Docker reachability.
        assert_eq!(detect_response(None).env_id, None);
    }
}

pub(crate) fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        cpu: true,
        // `crate::docker::gpu_device_args` only knows WSL2 (`/dev/dxg`) and
        // assumed-native-Linux (`/dev/kfd`/`/dev/dri`) device shapes; native
        // Windows has neither, so GPU passthrough is unimplemented there
        // (mirrors `engines/vllm`'s `rocm_gpu: !cfg!(windows)`).
        rocm_gpu: !cfg!(windows),
        openai_compatible: true,
        tool_calling: false,
        quantized_models: "image-reported".to_owned(),
        reasoning_parser: false,
    }
}

pub(crate) const fn windows_gpu_unsupported_message() -> &'static str {
    "AIM GPU passthrough is supported by rocm-cli only on Linux/WSL; native Windows AIM GPU \
serving is skipped."
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
