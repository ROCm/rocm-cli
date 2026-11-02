// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Engine plugin binary resolution and the engine IPC request/response
//! envelope: locating an engine's executable (packaged plugin vs. the
//! platform-packaged fallback), encoding/decoding the JSON-over-stdio
//! envelope used to talk to it, and the built-in (in-process) fast path for
//! `lemonade`/`vllm` that skips spawning a child process entirely.
//!
//! Mechanically relocated from `main.rs` with no behavior change — Phase 6f
//! (ROCMAI-213) of the `main.rs` modularization (see `docs/architecture.md`).
//! `dispatch()` calls directly into this module at two points —
//! `Command::EngineStdio { engine } => run_builtin_engine_stdio(&engine)` and
//! the `__engine-serve-http` arm's `run_builtin_engine_serve_http(...)` call —
//! both re-imported via `use crate::engine_ipc::{run_builtin_engine_serve_http,
//! run_builtin_engine_stdio};`; `dispatch()`'s own body is otherwise
//! unaffected.
//!
//! This module owns: resolving an engine's binary path
//! (`resolve_engine_binary_path`, `resolve_engine_binary_path_with_paths`,
//! `find_engine_plugin_binary`), the stdio request/response envelope used to
//! talk to an external engine binary or the in-process built-in handler
//! (`engine_request`, `engine_request_with_env_root`,
//! `decode_engine_response`, the `ScopedEnvVar`/
//! `with_scoped_builtin_engine_env` env-scoping helpers and the
//! `BUILTIN_ENGINE_ENV_LOCK` mutex they serialize on, `builtin_engine_request`,
//! `run_builtin_engine_stdio`), and the built-in HTTP-serving path
//! (`builtin_engine_available`, `run_builtin_engine_serve_http`,
//! `builtin_engine_serve_http_args`). Two items sit physically inside this
//! span in `main.rs` but stay there rather than moving: `const fn
//! missing_packaged_engine_reason` has callers outside this cluster (and its
//! own test) in addition to the one inside it, and `impl From<WatcherModeArg>
//! for WatcherMode` is unrelated clap-arg-to-domain-type conversion that only
//! happens to sit between `find_engine_plugin_binary` and `engine_request` in
//! the original file.
//!
//! `main.rs` calls into it for `resolve_engine_binary_path`,
//! `resolve_engine_binary_path_with_paths`, `engine_request`,
//! `builtin_engine_available`, and `builtin_engine_serve_http_args`, plus the
//! two `dispatch()` call sites above; `engines_cmd.rs` and `serve_cmd.rs` call
//! into it for `engine_request`; `engines_cmd.rs` and
//! `runtime_install_finalize.rs` call into it for
//! `engine_request_with_env_root`; `service_launch.rs` calls into it for
//! `builtin_engine_serve_http_args`. `find_engine_plugin_binary`,
//! `decode_engine_response`, `with_scoped_builtin_engine_env`,
//! `builtin_engine_request`, and `ScopedEnvVar` have no callers outside this
//! module and stay private. Shared helpers this module calls back into at the
//! crate root — `with_sigpipe_ignored`, `device_policy_name`,
//! `missing_packaged_engine_reason` — stay in `main.rs` and are reached
//! unqualified via `use crate::{...}`, same as for other mechanically
//! relocated modules. `parse_device_policy` moved out of `main.rs` into
//! `gpu_policy.rs` in ROCMAI-214 (Phase 6f); it is reached via `use
//! crate::gpu_policy::parse_device_policy;` instead. `apply_app_path_env`,
//! `app_path_env_vars`, `optional_arg`, and `engine_recipe_json_arg` moved out
//! of `main.rs` into `engine_runtime_select.rs` in ROCMAI-215 (Phase 6f); this
//! file's `use` list below points there now instead of at the crate root.

use std::ffi::OsString;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, engine_binary_path, engine_plugin_dirs};
use rocm_engine_protocol::{
    DevicePolicy, EngineMethod, EnginePluginDescriptor, EngineRecipeHint, EngineRequestEnvelope,
    EngineResponseEnvelope, HealthcheckRequest, HealthcheckResponse,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::engine_runtime_select::{
    app_path_env_vars, apply_app_path_env, engine_recipe_json_arg, optional_arg,
};
use crate::gpu_policy::parse_device_policy;
use crate::{device_policy_name, missing_packaged_engine_reason, with_sigpipe_ignored};

static BUILTIN_ENGINE_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn resolve_engine_binary_path(engine: &str) -> Result<PathBuf> {
    let paths = AppPaths::discover()?;
    resolve_engine_binary_path_with_paths(engine, &paths)
}

pub(crate) fn resolve_engine_binary_path_with_paths(
    engine: &str,
    paths: &AppPaths,
) -> Result<PathBuf> {
    if let Some(path) = find_engine_plugin_binary(engine, engine_plugin_dirs(paths))? {
        return Ok(path);
    }
    if let Some(reason) = missing_packaged_engine_reason(engine) {
        bail!("{reason}");
    }
    engine_binary_path(engine)
}

fn find_engine_plugin_binary<I, P>(engine: &str, plugin_dirs: I) -> Result<Option<PathBuf>>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    Ok(rocm_engine_protocol::discover_engine_plugins(plugin_dirs)
        .context("failed to discover engine plugin binaries")?
        .into_iter()
        .find(|plugin: &EnginePluginDescriptor| plugin.id == engine)
        .map(|plugin| plugin.executable_path))
}

pub(crate) fn engine_request<T, R>(
    paths: Option<&AppPaths>,
    engine: &str,
    method: EngineMethod,
    request: &T,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    engine_request_with_env_root(paths, engine, method, request, None)
}

pub(crate) fn engine_request_with_env_root<T, R>(
    paths: Option<&AppPaths>,
    engine: &str,
    method: EngineMethod,
    request: &T,
    env_root: Option<&Path>,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    engine_request_with_scoped_env(paths, engine, method, request, env_root, None)
}

/// [`engine_request_with_env_root`], additionally handing the engine the 0600
/// endpoint key file for one service via `ROCM_SERVE_API_KEY_FILE`.
///
/// An engine that probes its own server's HTTP endpoint while answering a
/// request reads that variable to authenticate
/// (`rocm_engine_protocol::resolve_endpoint_api_key`, reached from
/// `engines/lemonade`'s and `engines/vllm`'s `/v1/models` probes). Without it,
/// a public or `--require-api-key` service answers 401 and the engine reports
/// the service as not listed - indistinguishable, from the CLI's side, from a
/// server that is simply down. It is threaded per call rather than per process
/// because the key is per service id.
fn engine_request_with_scoped_env<T, R>(
    paths: Option<&AppPaths>,
    engine: &str,
    method: EngineMethod,
    request: &T,
    env_root: Option<&Path>,
    endpoint_key_file: Option<&Path>,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    let stream_progress = matches!(&method, EngineMethod::Install);
    let envelope = EngineRequestEnvelope {
        method,
        payload: serde_json::to_value(request)
            .context("failed to encode engine request payload")?,
    };
    if let Some(envelope) =
        with_scoped_builtin_engine_env(paths, env_root, endpoint_key_file, || {
            builtin_engine_request(engine, &envelope)
        })
    {
        return decode_engine_response(envelope);
    }

    let engine_binary = resolve_engine_binary_path(engine).with_context(|| {
        format!(
            "unable to locate engine binary for {engine}; build the workspace or install the engine package"
        )
    })?;
    let mut command = ProcessCommand::new(engine_binary);
    command.arg("stdio");
    if let Some(paths) = paths {
        apply_app_path_env(&mut command, paths);
    }
    if let Some(env_root) = env_root {
        command.env("ROCM_CLI_ENGINE_ENVS_ROOT", env_root);
    }
    if let Some(endpoint_key_file) = endpoint_key_file {
        command.env(
            rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV,
            endpoint_key_file,
        );
    }
    if stream_progress {
        command.env("ROCM_ENGINE_PROGRESS_STDERR", "1");
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn engine stdio process")?;

    {
        let mut stdin = child
            .stdin
            .take()
            .context("engine stdio child did not expose stdin")?;
        // Ignore SIGPIPE for the duration of the write: with SIG_DFL in effect
        // process-wide, an engine that exits before reading its stdin would
        // otherwise kill `rocm` with SIGPIPE before the exit-status diagnostics
        // below can run. Under SIG_IGN the write returns EPIPE instead, which we
        // surface as the normal "failed to write engine request" error and then
        // let `wait()` report why the engine exited.
        with_sigpipe_ignored(|| {
            serde_json::to_writer(&mut stdin, &envelope)
                .context("failed to write engine request")?;
            stdin.write_all(b"\n")?;
            Ok::<(), anyhow::Error>(())
        })?;
    }

    let stderr_handle = child.stderr.take().map(|stderr| {
        thread::spawn(move || {
            let mut reader = io::BufReader::new(stderr);
            let mut collected = Vec::new();
            let mut line = Vec::new();
            loop {
                line.clear();
                match reader.read_until(b'\n', &mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Ok(text) = std::str::from_utf8(&line) {
                            eprint!("{text}");
                        } else {
                            let _ = io::stderr().write_all(&line);
                        }
                        collected.extend_from_slice(&line);
                    }
                    Err(error) => {
                        let message = format!("failed to read engine progress: {error}\n");
                        eprint!("{message}");
                        collected.extend_from_slice(message.as_bytes());
                        break;
                    }
                }
            }
            collected
        })
    });
    let mut stdout = Vec::new();
    child
        .stdout
        .take()
        .context("engine stdio child did not expose stdout")?
        .read_to_end(&mut stdout)
        .context("failed to read engine stdio response")?;
    let status = child
        .wait()
        .context("failed waiting for engine stdio response")?;
    let stderr = stderr_handle
        .map(|handle| handle.join().unwrap_or_else(|_| Vec::new()))
        .unwrap_or_default();

    if !status.success() && stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
        if stderr.is_empty() {
            bail!("engine stdio process exited with status {status}");
        }
        bail!("engine stdio process exited with status {status}: {stderr}");
    }
    let envelope: EngineResponseEnvelope = serde_json::from_slice(&stdout).with_context(|| {
        let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
        if stderr.is_empty() {
            "failed to parse engine response envelope".to_owned()
        } else {
            format!("failed to parse engine response envelope; stderr: {stderr}")
        }
    })?;
    decode_engine_response(envelope)
}

/// Best-effort query of the backend/device a running managed service actually
/// selected (e.g. `"vulkan gpu 0"` / `"rocm gpu 0"`), via the same
/// `Healthcheck` RPC every engine plugin already answers
/// (`engines/lemonade` reads it back from its own `backend_requested` state;
/// `engines/vllm` and `engines/aim` report the ROCm device they pinned). Read
/// from the engine, never guessed from the launch-time GPU selection, so it
/// reflects what the server is actually running on. `None` on any failure -
/// this is informational text, not worth failing `serve`/`services list` over.
///
/// How long one call can take, since `rocm services list` makes one per ready
/// service: [`engine_request`] itself imposes no deadline, so the bound comes
/// from the engine. The three built-in engines answer in-process and every
/// endpoint probe they make is timeout-bounded (`engines/vllm`'s
/// `HEALTHCHECK_TIMEOUT_MS`, `engines/lemonade`'s
/// `rocm_core::INFERENCE_PROBE_TIMEOUT`), so a stalled server costs seconds,
/// not forever. An *external* plugin is a spawned child that
/// `engine_request` waits on unbounded, exactly as every other engine call
/// site in the CLI does. [`fetch_service_devices`] is what keeps a list of
/// services from multiplying that cost by the number of services.
pub(crate) fn fetch_service_device(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
) -> Option<String> {
    // Hand the engine this service's 0600 key file. Both built-in engines
    // authenticate their own `/v1/models` probe from
    // `ROCM_SERVE_API_KEY_FILE`, so without it a public or `--require-api-key`
    // service answers 401, the engine reports it as not listed, and the device
    // comes back `"unknown"` - the row would vanish for exactly the services
    // that are protected. `None` for a loopback service, which has no key file
    // and needs none.
    let endpoint_key_file = crate::endpoint_keys::endpoint_key_file_if_present(paths, service_id);
    let response: HealthcheckResponse = engine_request_with_scoped_env(
        Some(paths),
        engine,
        EngineMethod::Healthcheck,
        &HealthcheckRequest {
            service_id: service_id.to_owned(),
        },
        None,
        endpoint_key_file.as_deref(),
    )
    .ok()?;
    // `"unknown"` is the engines' own device value when no state is on disk for
    // this service id yet (see `engines/vllm/src/state.rs` and
    // `engines/lemonade/src/lib.rs`), which is not more informative than
    // showing nothing.
    (!response.device.is_empty() && response.device != "unknown").then_some(response.device)
}

/// How long [`fetch_service_devices`] may spend on live `Healthcheck` RPCs in
/// total, across however many services were listed.
///
/// `rocm services list` was pure local I/O before the `device:` row; the row is
/// worth a short wait but not an unbounded one, and the cost has to stay flat
/// in the number of services rather than growing with it. Sized to let a
/// couple of healthy engines answer (`engines/lemonade`'s endpoint probe is
/// the slowest at `rocm_core::INFERENCE_PROBE_TIMEOUT`, 8s worst case, and
/// only on a miss) while still bounding the whole render.
pub(crate) const SERVICE_DEVICE_PROBE_BUDGET: Duration = Duration::from_secs(10);

/// Devices for several services at once, keyed by service id, under one shared
/// time budget.
///
/// The probes run sequentially on purpose. They cannot usefully be run in
/// parallel: [`engine_request`] routes the built-in engines through
/// [`with_scoped_builtin_engine_env`], which takes a process-wide mutex *and*
/// mutates process-wide `ROCM_CLI_*` env vars for the duration, so concurrent
/// callers would serialize on that lock anyway and racing them would be
/// unsound. What this does instead is cap the aggregate: once
/// [`SERVICE_DEVICE_PROBE_BUDGET`] is spent, the remaining services are left
/// without a `device:` row rather than each adding its own wait. The row is
/// informational, so dropping it is strictly better than making
/// `rocm services list --all` hang behind one unresponsive server.
///
/// The budget is checked between probes, not inside one: a single external
/// plugin that never exits still blocks, because `engine_request` waits on its
/// child unbounded. That is the pre-existing behavior of every engine call
/// site in the CLI and is not made worse here.
pub(crate) fn fetch_service_devices<'a, I>(
    paths: &AppPaths,
    services: I,
) -> std::collections::HashMap<String, String>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let started = Instant::now();
    let mut devices = std::collections::HashMap::new();
    for (service_id, engine) in services {
        if started.elapsed() >= SERVICE_DEVICE_PROBE_BUDGET {
            break;
        }
        if let Some(device) = fetch_service_device(paths, engine, service_id) {
            devices.insert(service_id.to_owned(), device);
        }
    }
    devices
}

fn decode_engine_response<R>(envelope: EngineResponseEnvelope) -> Result<R>
where
    R: DeserializeOwned,
{
    if !envelope.ok {
        let error = envelope.error.map_or_else(
            || "unknown engine error".to_owned(),
            |value| format!("{}: {}", value.code, value.message),
        );
        bail!("{error}");
    }
    let data = envelope
        .data
        .context("engine response envelope did not contain data")?;
    serde_json::from_value(data).context("failed to decode engine response payload")
}

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn set_path(key: &'static str, value: &Path) -> Self {
        let previous = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    #[allow(unsafe_code)] // std::env::set_var/remove_var are unsafe in edition 2024
    fn drop(&mut self) {
        unsafe {
            match self.previous.as_ref() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

fn with_scoped_builtin_engine_env<R>(
    paths: Option<&AppPaths>,
    env_root: Option<&Path>,
    endpoint_key_file: Option<&Path>,
    f: impl FnOnce() -> R,
) -> R {
    if paths.is_none() && env_root.is_none() && endpoint_key_file.is_none() {
        return f();
    }
    let lock = BUILTIN_ENGINE_ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("builtin engine env lock poisoned");
    let mut vars = Vec::new();
    if let Some(paths) = paths {
        for (key, value) in app_path_env_vars(paths) {
            vars.push(ScopedEnvVar::set_path(key, value));
        }
    }
    if let Some(env_root) = env_root {
        vars.push(ScopedEnvVar::set_path(
            "ROCM_CLI_ENGINE_ENVS_ROOT",
            env_root,
        ));
    }
    if let Some(endpoint_key_file) = endpoint_key_file {
        vars.push(ScopedEnvVar::set_path(
            rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV,
            endpoint_key_file,
        ));
    }
    let result = f();
    drop(vars);
    result
}

fn builtin_engine_request(
    engine: &str,
    envelope: &EngineRequestEnvelope,
) -> Option<EngineResponseEnvelope> {
    match engine {
        "lemonade" => Some(rocm_engine_lemonade::builtin_handle_envelope(
            envelope.clone(),
        )),
        "vllm" => Some(rocm_engine_vllm::builtin_handle_envelope(envelope.clone())),
        "aim" => Some(rocm_engine_aim::builtin_handle_envelope(envelope.clone())),
        _ => None,
    }
}

pub(crate) fn run_builtin_engine_stdio(engine: &str) -> Result<()> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .context("failed to read engine stdio request")?;
    let envelope: EngineRequestEnvelope =
        serde_json::from_str(&input).context("failed to parse engine stdio request")?;
    let response = builtin_engine_request(engine, &envelope)
        .with_context(|| format!("engine `{engine}` is not built into this rocm binary"))?;
    print!("{}", serde_json::to_string(&response)?);
    Ok(())
}

pub(crate) fn builtin_engine_available(engine: &str) -> bool {
    matches!(engine, "lemonade" | "vllm" | "aim")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_builtin_engine_serve_http(
    engine: &str,
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: &str,
    gpu_indices: Vec<u32>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<()> {
    let parsed_policy = parse_device_policy(Some(device_policy))?;
    match engine {
        "lemonade" => rocm_engine_lemonade::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe,
        ),
        "vllm" => rocm_engine_vllm::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe,
        ),
        "aim" => rocm_engine_aim::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe,
        ),
        other => bail!("engine `{other}` is not built into this rocm binary"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn builtin_engine_serve_http_args(
    engine: &str,
    service_id: &str,
    canonical_model_id: &str,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
    state_path: &Path,
    log_path: Option<&Path>,
) -> Result<Vec<String>> {
    let mut args = vec![
        "__engine-serve-http".to_owned(),
        engine.to_owned(),
        service_id.to_owned(),
        canonical_model_id.to_owned(),
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "--device-policy".to_owned(),
        device_policy_name(device_policy).to_owned(),
    ];
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    if env_id.is_none() {
        args.extend(optional_arg("--runtime-id", runtime_id));
    }
    args.extend(optional_arg("--env-id", env_id));
    args.extend(engine_recipe_json_arg(engine_recipe)?);
    args.extend(["--state-path".to_owned(), state_path.display().to_string()]);
    if let Some(log_path) = log_path {
        args.extend(["--log-path".to_owned(), log_path.display().to_string()]);
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::tests::test_paths;

    #[test]
    fn engine_plugin_discovery_finds_runtime_binary() -> Result<()> {
        let (root, paths) = test_paths("engine-plugin");
        let plugin_dir = paths.primary_engine_plugin_dir();
        fs::create_dir_all(&plugin_dir)?;
        let plugin_path = plugin_dir.join(
            rocm_engine_protocol::platform_engine_plugin_binary_name("vllm"),
        );
        fs::write(&plugin_path, "plugin")?;

        let discovered = find_engine_plugin_binary("vllm", engine_plugin_dirs(&paths))?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(discovered, Some(plugin_path));
        Ok(())
    }

    #[test]
    fn engine_plugin_discovery_prefers_primary_plugin_dir() -> Result<()> {
        let (root, paths) = test_paths("engine-plugin-precedence");
        let primary_dir = paths.primary_engine_plugin_dir();
        let compatibility_dir = paths.data_dir.join("engines");
        fs::create_dir_all(&primary_dir)?;
        fs::create_dir_all(&compatibility_dir)?;
        let name = rocm_engine_protocol::platform_engine_plugin_binary_name("vllm");
        let primary_path = primary_dir.join(&name);
        fs::write(&primary_path, "primary")?;
        fs::write(compatibility_dir.join(&name), "compatibility")?;

        let discovered = find_engine_plugin_binary("vllm", engine_plugin_dirs(&paths))?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(discovered, Some(primary_path));
        Ok(())
    }
}
