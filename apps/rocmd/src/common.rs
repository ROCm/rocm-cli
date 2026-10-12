// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AutomationRuntimeState, CodexBridgeEngine, CodexBridgeGpuSnapshot,
    CodexBridgeSnapshot, ExamineSummary, RocmCliConfig, command_failure_detail, daemon_binary_path,
    default_engine_for_platform, format_host_port, load_recent_automation_events,
    resolve_amd_smi_binary, unix_time_millis,
};
#[cfg(test)]
use rocm_engine_protocol::EnginePluginDescriptor;
use rocm_engine_protocol::{
    EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope, HealthcheckRequest,
    HealthcheckResponse,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
#[cfg(test)]
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;

const AMD_SMI_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

pub(crate) fn build_bridge_snapshot(paths: &AppPaths) -> Result<CodexBridgeSnapshot> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    Ok(CodexBridgeSnapshot {
        protocol: "rocmd-codex-bridge-v0".to_owned(),
        generated_at_unix_ms: unix_time_millis(),
        examine: ExamineSummary::gather()?,
        gpu: gather_gpu_snapshot_for_config(&config),
        config,
        automation_runtime: AutomationRuntimeState::load(paths)?,
        recent_automation_events: load_recent_automation_events(paths, 32)?,
        engines: bridge_engine_inventory(),
        services: crate::persistence::load_managed_services(paths)?,
    })
}

pub(crate) fn update_check_message(status: &str) -> &'static str {
    match status {
        "repair_available" => {
            "ran read-only `rocm update`; a ROCm runtime repair is available because its package composition changed; no updates were applied"
        }
        "update_available" => {
            "ran read-only `rocm update`; a ROCm runtime update is available; no updates were applied"
        }
        "error" => "read-only `rocm update` failed; no updates were applied",
        _ => "ran read-only `rocm update`; no updates were applied",
    }
}

pub(crate) fn gather_gpu_snapshot() -> CodexBridgeGpuSnapshot {
    let static_snapshot = match capture_amd_smi_json(&["static", "-a", "-g", "all", "--json"]) {
        Ok(value) => Some(value),
        Err(error) => {
            return CodexBridgeGpuSnapshot {
                amd_smi_available: false,
                static_snapshot: None,
                monitor_snapshot: None,
                note: Some(error.to_string()),
            };
        }
    };

    let monitor_snapshot = match capture_amd_smi_json(&[
        "monitor", "-p", "-t", "-u", "-m", "-v", "-g", "all", "--json",
    ]) {
        Ok(value) => Some(value),
        Err(error) => {
            return CodexBridgeGpuSnapshot {
                amd_smi_available: true,
                static_snapshot,
                monitor_snapshot: None,
                note: Some(error.to_string()),
            };
        }
    };

    CodexBridgeGpuSnapshot {
        amd_smi_available: true,
        static_snapshot,
        monitor_snapshot,
        note: None,
    }
}

pub(crate) fn gather_gpu_snapshot_for_config(config: &RocmCliConfig) -> CodexBridgeGpuSnapshot {
    if config.telemetry.local_inspection_enabled() {
        gather_gpu_snapshot()
    } else {
        CodexBridgeGpuSnapshot {
            amd_smi_available: false,
            static_snapshot: None,
            monitor_snapshot: None,
            note: Some(
                "gpu telemetry is disabled by rocm-cli config; no external reporting is implemented"
                    .to_owned(),
            ),
        }
    }
}

fn capture_amd_smi_json(args: &[&str]) -> Result<Value> {
    let amd_smi_binary = resolve_amd_smi_binary();
    let mut command = ProcessCommand::new(&amd_smi_binary);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = run_command_with_timeout(command, AMD_SMI_PROBE_TIMEOUT)
        .with_context(|| format!("failed to launch amd-smi {}", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        anyhow::bail!(
            "amd-smi {} failed: {}",
            args.join(" "),
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            }
        );
    }

    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to parse amd-smi {} json", args.join(" ")))
}

pub(crate) fn bridge_engine_inventory() -> Vec<CodexBridgeEngine> {
    let default_engine = default_engine_for_platform();
    let current_exe = std::env::current_exe().ok();
    rocmd_engine_inventory()
        .iter()
        .map(|(id, summary)| CodexBridgeEngine {
            id: (*id).to_owned(),
            summary: (*summary).to_owned(),
            default_for_platform: *id == default_engine,
            installed_binary: true,
            binary_path: current_exe.as_ref().map(|path| path.display().to_string()),
        })
        .collect()
}

const fn rocmd_engine_inventory() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "lemonade",
            "embedded Lemonade server with ROCm llama.cpp backend",
        ),
        (
            "vllm",
            "Linux/WSL ROCm GPU serving engine through external vLLM",
        ),
    ]
}

#[cfg(test)]
pub(crate) fn find_engine_plugin_binary<I, P>(
    engine: &str,
    plugin_dirs: I,
) -> Result<Option<PathBuf>>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    Ok(rocm_engine_protocol::discover_engine_plugins(plugin_dirs)
        .context("failed to discover engine plugin binaries")?
        .into_iter()
        .find(|plugin: &EnginePluginDescriptor| plugin.id == engine)
        .map(|plugin| plugin.executable_path))
}

#[derive(Debug)]
pub(crate) struct CommandCapture {
    pub(crate) argv: Vec<String>,
    pub(crate) exit_status: i32,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

pub(crate) fn run_command_with_timeout(
    mut command: ProcessCommand,
    timeout: Duration,
) -> Result<std::process::Output> {
    let mut child = command.spawn().context("failed to spawn child process")?;
    let started = std::time::Instant::now();
    loop {
        if child
            .try_wait()
            .context("failed to poll child process")?
            .is_some()
        {
            return child
                .wait_with_output()
                .context("failed to collect child process output");
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let output = child
                .wait_with_output()
                .context("failed to collect timed-out child process output")?;
            bail!(
                "process exceeded {}s timeout: {}",
                timeout.as_secs(),
                command_failure_detail(&output)
            );
        }
        thread::sleep(Duration::from_millis(50));
    }
}

pub(crate) fn run_rocm_capture(args: &[&str]) -> Result<CommandCapture> {
    let paths = AppPaths::discover()?;
    run_rocm_capture_for_paths(&paths, args, Duration::from_mins(2))
}

pub(crate) fn run_rocm_capture_for_paths(
    paths: &AppPaths,
    args: &[&str],
    timeout: Duration,
) -> Result<CommandCapture> {
    let rocm_binary = daemon_binary_path()?;
    let mut command = ProcessCommand::new(&rocm_binary);
    command
        .args(args)
        .env("ROCM_CLI_CONFIG_DIR", &paths.config_dir)
        .env("ROCM_CLI_DATA_DIR", &paths.data_dir)
        .env("ROCM_CLI_CACHE_DIR", &paths.cache_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = run_command_with_timeout(command, timeout)
        .with_context(|| format!("failed to run {}", rocm_binary.display()))?;
    Ok(CommandCapture {
        argv: std::iter::once(rocm_binary.display().to_string())
            .chain(args.iter().map(|value| (*value).to_owned()))
            .collect(),
        exit_status: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

pub(crate) fn optional_arg(flag: &str, value: Option<&str>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value.to_owned()],
        None => Vec::new(),
    }
}

pub(crate) fn engine_healthcheck_response(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
) -> Result<HealthcheckResponse> {
    engine_request::<_, HealthcheckResponse>(
        paths,
        engine,
        service_id,
        EngineMethod::Healthcheck,
        &HealthcheckRequest {
            service_id: service_id.to_owned(),
        },
    )
}

pub(crate) fn healthcheck_response_ready(response: &HealthcheckResponse) -> bool {
    response.status == "ready" && response.model_loaded
}

pub(crate) fn healthcheck_response_recoverable(response: &HealthcheckResponse) -> bool {
    matches!(
        response.status.as_str(),
        "failed" | "unreachable" | "exited"
    )
}

/// Re-thread the endpoint key file (public bind only) onto an engine child's
/// environment, mirroring the initial `rocm serve` spawn. A loopback service
/// with no stored key leaves the command's environment untouched, matching the
/// unauthenticated default.
///
/// Returns whether a key was applied. Callers that spawn a *listener* must
/// check it against the service's host — a public bind with no key would come
/// up anonymous (see [`ensure_public_service_has_endpoint_key`]). Callers that
/// only make a stdio plugin call can ignore it.
///
/// The file must hold a *usable* key, not merely exist: the engine adapters
/// resolve it with `endpoint_api_key_from_file` and enforce nothing when that
/// yields `None`, so treating an empty or malformed file as "protected" would
/// let the guard pass while the endpoint served anonymously.
#[must_use]
pub(crate) fn apply_endpoint_key_env(
    command: &mut ProcessCommand,
    paths: &AppPaths,
    service_id: &str,
) -> bool {
    if let Some(key_file) = rocm_engine_protocol::endpoint_key_file_if_present(paths, service_id)
        .and_then(|path| rocm_engine_protocol::endpoint_api_key_file_if_valid(&path))
    {
        command.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file);
        return true;
    }
    false
}

/// Refuse to respawn a recorded service on a public host without its endpoint
/// API key, so daemon recovery cannot reopen a protected endpoint anonymously.
///
/// Mirrors the guard of the same name in `rocm`; the shared
/// [`rocm_engine_protocol::is_public_bind_host`] keeps the two classifications
/// identical for a given `ManagedServiceRecord::host`.
pub(crate) fn ensure_public_service_has_endpoint_key(
    host: &str,
    key_present: bool,
    requires_api_key: bool,
) -> Result<()> {
    // The bind address is not the whole story. A service bound to loopback is
    // only private until something republishes the port, and a tailnet publish
    // outlives both the process and this daemon. The requirement is recorded on
    // the service precisely so recovery can honour it without re-deriving it
    // from an address that no longer answers the question.
    if requires_api_key && !key_present {
        bail!(
            "refusing to recover a service that was launched with an endpoint API key but no \
             longer has one: it would come back up without authentication, and something \
             outside this machine may still be publishing its port. Relaunch it with \
             `rocm serve --require-api-key` to issue a new key."
        );
    }
    if rocm_engine_protocol::is_public_bind_host(host) && !key_present {
        bail!(
            "refusing to respawn a service bound to the public host `{host}` without an endpoint \
             API key: it would come back up without authentication. Relaunch it with \
             `rocm serve --host {host} --allow-public-bind` to issue a new key."
        );
    }
    Ok(())
}

pub(crate) fn engine_request<T, R>(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    method: EngineMethod,
    request: &T,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    let envelope = EngineRequestEnvelope {
        method,
        payload: serde_json::to_value(request).context("failed to serialize engine request")?,
    };
    let engine_binary =
        std::env::current_exe().context("failed to resolve current rocm executable path")?;
    let mut command = ProcessCommand::new(&engine_binary);
    command
        .arg("__engine-stdio")
        .arg(engine)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // A protected public endpoint enforces its key on every request, including
    // this stdio-transported healthcheck — without the carrier the engine
    // adapter's HTTP probe is unauthenticated and the endpoint looks
    // unreachable, which would misclassify a healthy protected service as
    // recoverable.
    //
    // No host to check here: this is a stdio plugin call, not a listener, so a
    // missing key only weakens this probe rather than opening a port.
    let _ = apply_endpoint_key_env(&mut command, paths, service_id);
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to spawn engine stdio process {}",
            engine_binary.display()
        )
    })?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .context("engine stdio child did not expose stdin")?;
        serde_json::to_writer(&mut *stdin, &envelope).context("failed to write engine request")?;
        stdin.write_all(b"\n")?;
    }

    let output = child
        .wait_with_output()
        .context("failed waiting for engine stdio response")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        if stderr.is_empty() {
            bail!("engine stdio process exited with status {}", output.status);
        }
        bail!(
            "engine stdio process exited with status {}: {}",
            output.status,
            stderr
        );
    }
    let envelope: EngineResponseEnvelope =
        serde_json::from_slice(&output.stdout).context("failed to parse engine response")?;
    if !envelope.ok {
        let detail = envelope.error.map_or_else(
            || "unknown engine error".to_owned(),
            |error| format!("{}: {}", error.code, error.message),
        );
        bail!(detail);
    }
    let data = envelope
        .data
        .context("engine response did not include data")?;
    serde_json::from_value(data).context("failed to deserialize engine response data")
}

pub(crate) fn wait_for_port(host: &str, port: u16, timeout: Duration) -> bool {
    let address: SocketAddr = match format_host_port(host, port).parse() {
        Ok(value) => value,
        Err(_) => return false,
    };

    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use rocm_core::engine_plugin_dirs;
    use std::fs;

    #[test]
    fn recovery_refuses_a_service_that_lost_a_key_it_was_launched_with() {
        // The daemon keeps its own copy of this guard, and it only knew about
        // public binds. A loopback service that something republishes — a
        // tailnet publish outlives this daemon, let alone the process — would
        // be recovered without authentication, and the rebuilt record would
        // then disarm `rocm services restart` too.
        for host in ["127.0.0.1", "localhost", "::1"] {
            let error = ensure_public_service_has_endpoint_key(host, false, true)
                .expect_err("a service launched with a key must not be recovered without one");
            assert!(
                format!("{error:#}").contains("without authentication"),
                "{error:#}"
            );
        }
        // With the key still present, recovery proceeds.
        ensure_public_service_has_endpoint_key("127.0.0.1", true, true).unwrap();
        // And a service that never had one is untouched.
        ensure_public_service_has_endpoint_key("127.0.0.1", false, false).unwrap();
    }

    #[test]
    fn recovery_respawn_fails_closed_for_a_public_service_without_a_key() {
        // Daemon recovery re-execs `rocmd supervise` for the recorded host. With
        // the key gone the child would listen on that public host anonymously,
        // so the spawn must be refused instead.
        let error = ensure_public_service_has_endpoint_key("0.0.0.0", false, false).unwrap_err();
        assert!(
            error.to_string().contains("without authentication"),
            "{error:#}"
        );

        ensure_public_service_has_endpoint_key("0.0.0.0", true, false).unwrap();
        for host in ["127.0.0.1", "localhost", "::1"] {
            ensure_public_service_has_endpoint_key(host, false, false)
                .unwrap_or_else(|error| panic!("{host} must not require a key: {error:#}"));
        }
    }

    #[test]
    fn apply_endpoint_key_env_sets_var_only_when_key_file_present() {
        let (_root, paths) = temp_app_paths("apply-endpoint-key-env");
        let service_id = "svc-endpoint-key-env";

        // Loopback service: no key file has ever been written, so the child's
        // environment must be left untouched.
        let mut command = ProcessCommand::new("true");
        assert!(!apply_endpoint_key_env(&mut command, &paths, service_id));
        assert!(
            command
                .get_envs()
                .all(|(key, _)| key != rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV),
            "loopback service must not receive the endpoint key env var"
        );

        // Public service: once a key file exists at the deterministic path,
        // it must be threaded onto the child so the engine's HTTP probe
        // authenticates.
        let key_path = rocm_engine_protocol::endpoint_key_file_path(&paths, service_id);
        fs::create_dir_all(paths.services_dir()).unwrap();
        fs::write(&key_path, "secret-key").unwrap();
        let mut command = ProcessCommand::new("true");
        assert!(apply_endpoint_key_env(&mut command, &paths, service_id));
        let env_value = command
            .get_envs()
            .find_map(|(key, value)| {
                (key == rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV)
                    .then_some(value)
                    .flatten()
            })
            .expect("endpoint key env var must be set once the key file exists");
        assert_eq!(env_value, key_path.as_os_str());

        // An empty (or otherwise unusable) key file is not a key: the engine
        // adapters would enforce nothing, so reporting `true` here would let the
        // respawn guard pass and still open an anonymous public listener.
        fs::write(&key_path, "   \n").unwrap();
        let mut command = ProcessCommand::new("true");
        assert!(!apply_endpoint_key_env(&mut command, &paths, service_id));
        assert!(
            command
                .get_envs()
                .all(|(key, _)| key != rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV),
            "an unusable key file must not be threaded onto the child"
        );
    }

    #[test]
    fn healthcheck_readiness_requires_ready_loaded_model() {
        let ready = HealthcheckResponse {
            status: "ready".to_owned(),
            model_loaded: true,
            device: "cuda".to_owned(),
            uptime_sec: 1,
            queue_depth: 0,
            last_error: None,
            tokens_per_sec: None,
        };
        assert!(healthcheck_response_ready(&ready));

        let mut loading = ready.clone();
        loading.status = "loading_model".to_owned();
        assert!(!healthcheck_response_ready(&loading));

        let mut unloaded = ready;
        unloaded.model_loaded = false;
        assert!(!healthcheck_response_ready(&unloaded));
    }

    #[test]
    fn healthcheck_recoverability_tracks_failed_endpoint_state() {
        let mut response = HealthcheckResponse {
            status: "ready".to_owned(),
            model_loaded: true,
            device: "cuda".to_owned(),
            uptime_sec: 1,
            queue_depth: 0,
            last_error: None,
            tokens_per_sec: None,
        };
        assert!(!healthcheck_response_recoverable(&response));

        response.status = "unreachable".to_owned();
        assert!(healthcheck_response_recoverable(&response));

        response.status = "failed".to_owned();
        assert!(healthcheck_response_recoverable(&response));

        response.status = "loading_model".to_owned();
        assert!(!healthcheck_response_recoverable(&response));

        // The status the engines report for a model that is listed but has not
        // yet served an inference request. Restarting it would kill a model
        // mid-load and start the wait over.
        response.status = "loading".to_owned();
        assert!(!healthcheck_response_recoverable(&response));
    }

    #[test]
    fn healthcheck_readiness_withheld_while_the_model_only_lists() {
        // What an engine reports once `/v1/models` answers but inference has not:
        // not ready, so `rocm serve` keeps waiting instead of handing the caller
        // an endpoint that will hang on its first request.
        let listing_only = HealthcheckResponse {
            status: "loading".to_owned(),
            model_loaded: false,
            device: "unknown".to_owned(),
            uptime_sec: 1,
            queue_depth: 0,
            last_error: None,
            tokens_per_sec: None,
        };
        assert!(!healthcheck_response_ready(&listing_only));
    }

    #[test]
    fn gpu_snapshot_respects_disabled_telemetry_policy() {
        let mut config = RocmCliConfig::default();
        config.telemetry.mode = rocm_core::TELEMETRY_MODE_OFF.to_owned();

        let snapshot = gather_gpu_snapshot_for_config(&config);

        assert!(!snapshot.amd_smi_available);
        assert!(snapshot.static_snapshot.is_none());
        assert!(snapshot.monitor_snapshot.is_none());
        assert!(
            snapshot
                .note
                .as_deref()
                .is_some_and(|note| note.contains("disabled by rocm-cli config"))
        );
    }

    #[test]
    fn engine_plugin_discovery_finds_runtime_binary() -> Result<()> {
        let (root, paths) = temp_app_paths("engine-plugin");
        let plugin_dir = paths.data_dir.join("engines").join("plugins");
        fs::create_dir_all(&plugin_dir)?;
        let plugin_path = plugin_dir.join(
            rocm_engine_protocol::platform_engine_plugin_binary_name("vllm"),
        );
        fs::write(&plugin_path, "plugin")?;

        let discovered = find_engine_plugin_binary("vllm", engine_plugin_dirs(&paths))?;
        fs::remove_dir_all(root).ok();

        assert_eq!(discovered, Some(plugin_path));
        Ok(())
    }
}
