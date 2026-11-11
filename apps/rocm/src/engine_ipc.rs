// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Engine plugin binary resolution and the engine IPC request/response
//! envelope: locating an engine's executable (packaged plugin vs. the
//! platform-packaged fallback), talking to it over JSON-over-stdio, and the
//! built-in (in-process) fast path for `lemonade`/`vllm`/`aim` that skips
//! spawning a child process entirely.
//!
//! The stdio conversation itself is not implemented here. The request framing,
//! the child's environment, stderr handling, the SIGPIPE guard around the write
//! and the reading of the answer are the client in
//! [`rocm_engine_protocol`]'s `stdio_client` module, which `rocmd` uses too.
//! This module chooses what to run (the built-in handler, or the resolved
//! engine binary with `stdio`) and what `rocm` asks of the client for it
//! ([`engine_stdio_child_request`]).
//!
//! `dispatch()` calls directly into this module at two points —
//! `Command::EngineStdio { engine } => run_builtin_engine_stdio(&engine)` and
//! the `__engine-serve-http` arm's `run_builtin_engine_serve_http(...)` call —
//! both re-imported via `use crate::engine_ipc::{run_builtin_engine_serve_http,
//! run_builtin_engine_stdio};`.
//!
//! This module owns: resolving an engine's binary path
//! (`resolve_engine_binary_path`, `resolve_engine_binary_path_with_paths`,
//! `find_engine_plugin_binary`), the request entry points
//! (`engine_request`, `engine_request_with_env_root`,
//! `engine_stdio_child_request`), the `ScopedEnvVar`/
//! `with_scoped_builtin_engine_env` env-scoping helpers and the
//! `BUILTIN_ENGINE_ENV_LOCK` mutex they serialize on, `builtin_engine_request`,
//! `run_builtin_engine_stdio`, and the built-in HTTP-serving path
//! (`builtin_engine_available`, `run_builtin_engine_serve_http`,
//! `builtin_engine_serve_http_args`).
//!
//! `main.rs` calls into it for `resolve_engine_binary_path`,
//! `resolve_engine_binary_path_with_paths`, `engine_request`,
//! `builtin_engine_available`, and `builtin_engine_serve_http_args`, plus the
//! two `dispatch()` call sites above; `engines_cmd.rs` and `serve_cmd.rs` call
//! into it for `engine_request`; `engines_cmd.rs` and
//! `runtime_install_finalize.rs` call into it for
//! `engine_request_with_env_root`; `service_launch.rs` calls into it for
//! `builtin_engine_serve_http_args`. `find_engine_plugin_binary`,
//! `with_scoped_builtin_engine_env`, `builtin_engine_request`, and
//! `ScopedEnvVar` have no callers outside this module and stay private. The
//! one helper this module calls back into at the crate root, `device_policy_name`,
//! stays in `main.rs` and is reached via `use crate::device_policy_name;`.
//! `parse_device_policy` is in `gpu_policy.rs`, `engine_recipe_json_arg` is in
//! `engine_runtime_select.rs`, and `optional_arg` is in `rocm_engine_protocol`.

use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
// Both are reached only from `run_scripted_e2e_engine_serve_http`, which is
// itself behind this feature; ungated they are unused in a default build.
#[cfg(feature = "e2e-test-hooks")]
use std::thread;
#[cfg(feature = "e2e-test-hooks")]
use std::time::Duration;

#[cfg(feature = "e2e-test-hooks")]
use anyhow::bail;
use anyhow::{Context, Result};
use rocm_core::{
    AppPaths, ENGINE_ENVS_ROOT_ENV, ManagedServiceRecord, engine_binary_path, engine_plugin_dirs,
};
use rocm_engine_protocol::{
    DevicePolicy, EngineMethod, EnginePluginDescriptor, EngineRecipeHint, EngineRequestEnvelope,
    EngineResponseEnvelope, EngineStdioOptions, decode_engine_response, engine_request_envelope,
    optional_arg, run_engine_stdio,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::device_policy_name;
use crate::engine_runtime_select::engine_recipe_json_arg;
use crate::gpu_policy::parse_device_policy;

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
    let stream_progress = matches!(&method, EngineMethod::Install);
    let envelope = engine_request_envelope(method, request)?;
    if let Some(envelope) = with_scoped_builtin_engine_env(paths, env_root, || {
        builtin_engine_request(engine, &envelope)
    }) {
        return decode_engine_response(envelope);
    }

    let engine_binary = resolve_engine_binary_path(engine).with_context(|| {
        format!(
            "unable to locate engine binary for {engine}; build the workspace or install the engine package"
        )
    })?;
    engine_stdio_child_request(&engine_binary, paths, &envelope, env_root, stream_progress)
}

/// The out-of-process half of [`engine_request_with_env_root`]: run
/// `engine_binary stdio` through the client `rocm` shares with `rocmd`, send
/// `envelope` on its stdin and decode its answer.
///
/// What `rocm` asks of that client: the child runs on this process's app paths
/// when it has them and on the engine environments root when the request names
/// one, an install asks for progress on stderr, and the child's stderr is
/// written to this process's as it arrives, which is how a terminal user
/// watches an install. No endpoint key is handed over, because none of the
/// requests `rocm` makes is about a running service's endpoint. Split out so a
/// test can point it at a stand-in binary.
fn engine_stdio_child_request<R>(
    engine_binary: &Path,
    paths: Option<&AppPaths>,
    envelope: &EngineRequestEnvelope,
    env_root: Option<&Path>,
    stream_progress: bool,
) -> Result<R>
where
    R: DeserializeOwned,
{
    let options = stdio_options(paths, env_root, stream_progress);
    decode_engine_response(run_engine_stdio(
        engine_binary,
        ["stdio"],
        &options,
        envelope,
    )?)
}

/// What `rocm` asks of the shared stdio client; see
/// [`engine_stdio_child_request`].
const fn stdio_options<'a>(
    paths: Option<&'a AppPaths>,
    env_root: Option<&'a Path>,
    stream_progress: bool,
) -> EngineStdioOptions<'a> {
    EngineStdioOptions {
        app_paths: paths,
        env_root,
        endpoint_key_file: None,
        request_progress: stream_progress,
        forward_stderr: true,
        note_failing_exit_after_answer: false,
    }
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
    f: impl FnOnce() -> R,
) -> R {
    if paths.is_none() && env_root.is_none() {
        return f();
    }
    let lock = BUILTIN_ENGINE_ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("builtin engine env lock poisoned");
    let mut vars = Vec::new();
    if let Some(paths) = paths {
        for (key, value) in paths.env_vars() {
            vars.push(ScopedEnvVar::set_path(key, value));
        }
    }
    if let Some(env_root) = env_root {
        vars.push(ScopedEnvVar::set_path(ENGINE_ENVS_ROOT_ENV, env_root));
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

/// Entry point for the hidden `__engine-serve-http` subcommand: the process
/// `spawn_managed_engine_child`/`restart_internal_managed_service_with` launch
/// detached to actually run the engine.
///
/// ROCMAI-541: on Windows this process's stdout/stderr reach the service log
/// only because the parent attached them at spawn time
/// (`rocm_core::spawn_detached_with_log`); a gap there (or in a future
/// caller that spawns this subcommand some other way) would otherwise lose an
/// engine's failure reason completely, since nothing else reads a detached
/// child's output. So this wrapper does not rely on that attachment alone: on
/// any error from the engine dispatch below, it appends a one-line summary of
/// the error chain directly to the service log file named by `--log-path` (the
/// same file `rocm services logs <service_id>` tails) — one line, because this process's
/// stderr reaches that same file, so the full chain is already there whenever
/// the attachment worked — and the full chain to this process's own
/// structured log, and marks the engine state file `failed` — regardless of whether the
/// engine already did either on this particular error path. A later success
/// (this function only runs once per process, so there is no later state to
/// race against) is unaffected; only the error path writes anything here.
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
    // ROCMAI-541: short-circuit every real engine entirely when the
    // scripted-engine E2E hook is armed, so a scenario can exercise a managed
    // child that survives its early-startup liveness checks and then dies (or
    // runs until stopped) without any real engine logic — GPU, Python venv, or
    // network download — in the way. Ahead of the Anthropic endpoint below,
    // which fronts a real engine this path never starts. See
    // `run_scripted_e2e_engine_serve_http`.
    #[cfg(feature = "e2e-test-hooks")]
    if std::env::var_os("ROCM_E2E_SCRIPTED_ENGINE").is_some() {
        let result =
            run_scripted_e2e_engine_serve_http(&service_id, &state_path, log_path.as_deref());
        if let Err(error) = &result {
            record_engine_serve_http_failure(&service_id, &state_path, log_path.as_deref(), error);
        }
        return result;
    }
    // Not `?`: an unparseable policy is as much an engine failure as any other,
    // and the state file has to say so or the service reads `starting` forever
    // (ROCMAI-541).
    let parsed_policy = match parse_device_policy(Some(device_policy)) {
        Ok(policy) => policy,
        Err(error) => {
            record_engine_serve_http_failure(&service_id, &state_path, log_path.as_deref(), &error);
            return Err(error);
        }
    };
    // Every engine is fronted the same way, so the Anthropic endpoint is started
    // here once rather than inside each adapter. It runs on a thread of this
    // process, which is the service's supervisor: it ends when the service does.
    rocmd::anthropic_proxy::spawn_alongside_engine(rocmd::anthropic_proxy::AnthropicProxyConfig {
        host: host.clone(),
        engine_port: port,
        served_model: model_ref.clone(),
        state_path: state_path.clone(),
        probe_api_key: rocm_engine_protocol::resolve_endpoint_api_key(),
    });
    let result = match engine {
        "lemonade" => rocm_engine_lemonade::builtin_serve_http(
            service_id.clone(),
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path.clone(),
            log_path.clone(),
            engine_recipe,
        ),
        "vllm" => rocm_engine_vllm::builtin_serve_http(
            service_id.clone(),
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path.clone(),
            log_path.clone(),
            engine_recipe,
        ),
        "aim" => rocm_engine_aim::builtin_serve_http(
            service_id.clone(),
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path.clone(),
            log_path.clone(),
            engine_recipe,
        ),
        other => Err(anyhow::anyhow!(
            "engine `{other}` is not built into this rocm binary"
        )),
    };
    if let Err(error) = &result {
        record_engine_serve_http_failure(&service_id, &state_path, log_path.as_deref(), error);
    }
    result
}

/// Best-effort, platform-generic failure capture for
/// [`run_builtin_engine_serve_http`] — see its doc comment for why this exists
/// alongside the Windows stdio-attach fix rather than instead of it.
///
/// Never fails itself: a service already failing must not fail a second time,
/// less legibly, because its own failure-reporting hit an error.
fn record_engine_serve_http_failure(
    service_id: &str,
    state_path: &Path,
    log_path: Option<&Path>,
    error: &anyhow::Error,
) {
    let detail = format!("{error:?}");
    tracing::error!(service_id, %detail, "managed engine serve-http failed");
    // Decided before anything is written: `lemonade` and `vllm` return `Err`
    // for a SIGTERM from `rocm services stop` too, and a line reading
    // "failed:" in `rocm services logs` is wrong for a stop the operator
    // asked for — the state file was already being spared, but the log was
    // not.
    let deliberate = engine_stop_was_deliberate(service_id, state_path);
    if let Some(log_path) = log_path
        && let Ok(mut file) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
    {
        // One compact line (`{:#}`), not the full `{:?}` chain `detail` holds:
        // this process's stderr is attached to this same log, so whatever the
        // binary prints on its way out already lands here. The backstop still
        // has to exist — that attach is exactly what can be missing — but
        // writing the identical multi-line chain made `rocm services logs`
        // show it twice. A single `service_id`-tagged summary stays legible
        // beside the real stderr copy, and is still the only record when
        // there is none.
        let outcome = if deliberate { "stopped" } else { "failed" };
        let _ = writeln!(
            file,
            "__engine-serve-http service_id={service_id} {outcome}: {error:#}"
        );
    }
    if deliberate {
        return;
    }
    let _ =
        rocm_core::merge_json_state_file(state_path, &serde_json::json!({ "status": "failed" }));
}

/// Whether this exit is an operator-requested stop rather than a crash, so the
/// blanket `failed` stamp above must not fire.
///
/// Needed because the adapters disagree. `aim` makes the distinction itself
/// (`finish_after_exit` reads `stopping`/`stopped` off its own state file,
/// writes `stopped` and returns `Ok`), so it never reaches the stamp. `lemonade`
/// and `vllm` have no such carve-out: a SIGTERM from `rocm services stop` makes
/// `status.success()` false, so they write `failed` themselves and return `Err`.
/// They were already mislabelling a deliberate stop before this change — the
/// stamp would merely agree with them — but agreeing with a wrong answer is
/// still wrong, so check both places an intent is recorded:
///
/// - the engine's own state file, which is what `aim`'s `stop_service` stamps
///   `stopping` before it kills anything; and
/// - the service manifest's `stop_requested_unix_ms`, which
///   `stop_internal_managed_service_confirmed_with` writes before it signals
///   anything, for every engine.
///
/// Best-effort on purpose: when neither can be read this returns `false` and the
/// stamp proceeds, because failing to prove a stop must never swallow a crash
/// report.
fn engine_stop_was_deliberate(service_id: &str, state_path: &Path) -> bool {
    engine_stop_was_deliberate_in(AppPaths::discover().ok().as_ref(), service_id, state_path)
}

/// [`engine_stop_was_deliberate`] with the paths injected.
///
/// Split so a test can plant the manifest marker under an isolated root: the
/// production entry point resolves `AppPaths::discover()`, which reads the real
/// environment, so a test calling it would answer from the developer's own
/// services directory rather than its fixture.
fn engine_stop_was_deliberate_in(
    paths: Option<&AppPaths>,
    service_id: &str,
    state_path: &Path,
) -> bool {
    let engine_status = fs::read(state_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|state| {
            state
                .get("status")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        });
    if matches!(engine_status.as_deref(), Some("stopping" | "stopped")) {
        return true;
    }
    paths.is_some_and(|paths| {
        ManagedServiceRecord::read_manifest(paths, service_id)
            .is_ok_and(|record| record.stop_requested_unix_ms.is_some())
    })
}

/// ROCMAI-541 E2E hook, armed by `ROCM_E2E_SCRIPTED_ENGINE`: stands in for
/// every real engine so a scenario can exercise a managed child dying (or being
/// stopped) *after* its early-startup window, on a host with no GPU — the gap
/// the original ROCMAI-541 fix left uncovered (see
/// `serve-scripted-death-after-startup` and
/// `serve-scripted-external-stop-while-attached` in `model_serving.feature`).
///
/// Marks the engine state file `running` (mirroring a real engine reaching
/// that point) and appends a progress line to the service log — the same two
/// signals `finalize_managed_launch_readiness`/`managed_engine_exit_failure_detail`
/// read — then either:
///
/// - `ROCM_E2E_SCRIPTED_ENGINE_DEATH_MS` set to a parseable delay: sleeps that
///   long (long enough to clear both platforms' early-startup liveness checks,
///   short enough to land well inside the 45s readiness window) and returns an
///   error, simulating a crash after startup;
/// - unset: loops until something kills this process (e.g. `rocm services
///   stop`), capped at a generous safety ceiling so a scenario that forgets to
///   stop it cannot hang a CI job forever.
///
/// Compiled only into the `e2e-test-hooks` binary; production builds never see
/// this function or the env vars it reads.
#[cfg(feature = "e2e-test-hooks")]
fn run_scripted_e2e_engine_serve_http(
    service_id: &str,
    state_path: &Path,
    log_path: Option<&Path>,
) -> Result<()> {
    let append_log = |line: &str| {
        if let Some(log_path) = log_path
            && let Ok(mut file) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
        {
            let _ = writeln!(file, "{line}");
        }
    };
    append_log(&format!(
        "scripted e2e engine: started service_id={service_id} pid={}",
        std::process::id()
    ));
    let _ =
        rocm_core::merge_json_state_file(state_path, &serde_json::json!({ "status": "running" }));
    let death_after_ms = std::env::var("ROCM_E2E_SCRIPTED_ENGINE_DEATH_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok());
    if let Some(delay_ms) = death_after_ms {
        thread::sleep(Duration::from_millis(delay_ms));
        append_log("scripted e2e engine: dying on schedule");
        bail!("scripted e2e engine death after startup (service `{service_id}`)");
    }
    // No scheduled death: run until something stops this process. The ceiling
    // only guards against a scenario that forgot to stop it — a real scenario
    // kills this well before 2 minutes.
    let deadline = std::time::Instant::now() + Duration::from_mins(2);
    while std::time::Instant::now() < deadline {
        thread::sleep(Duration::from_millis(200));
    }
    append_log("scripted e2e engine: safety ceiling reached, exiting cleanly");
    Ok(())
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

    /// The backstop itself, which nothing covered directly: on Linux the same
    /// text also arrives through the stderr attach, so deleting the file write
    /// would have stayed green on every lane that runs.
    ///
    /// Pins all three of its outputs together — the log line, the `failed`
    /// stamp, and the fact that a deliberate stop gets neither verdict.
    #[test]
    fn the_serve_http_backstop_writes_the_log_line_and_the_stamp() {
        let (root, _paths) = test_paths("engine-ipc-backstop");
        let state_path = root.join("engine-state.json");
        let log_path = root.join("service.log");
        fs::create_dir_all(&root).unwrap();
        fs::write(&state_path, br#"{"status":"running"}"#).unwrap();
        let error = anyhow::anyhow!("engine blew up").context("while serving");

        record_engine_serve_http_failure(
            "svc-backstop",
            &state_path,
            Some(log_path.as_path()),
            &error,
        );

        let log = fs::read_to_string(&log_path).unwrap();
        let state: serde_json::Value =
            serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();
        assert!(
            log.contains("svc-backstop") && log.contains("failed:"),
            "the crash must reach the service log even with no stderr attach: {log}"
        );
        assert!(
            log.contains("engine blew up"),
            "and it must carry the engine's own reason: {log}"
        );
        assert_eq!(state["status"], "failed");

        // A deliberate stop gets neither verdict: not stamped, and not called
        // "failed" in a log the user reads.
        fs::write(&state_path, br#"{"status":"stopping"}"#).unwrap();
        fs::write(&log_path, b"").unwrap();
        record_engine_serve_http_failure(
            "svc-backstop",
            &state_path,
            Some(log_path.as_path()),
            &error,
        );
        let log = fs::read_to_string(&log_path).unwrap();
        let state: serde_json::Value =
            serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();
        let _ = fs::remove_dir_all(root);
        assert!(
            !log.contains("failed:"),
            "a stop the operator asked for must not read as a failure: {log}"
        );
        assert_ne!(state["status"], "failed");
    }

    /// The blanket `failed` stamp must not relabel an operator-requested stop.
    ///
    /// `aim` already makes this distinction itself and returns `Ok`, so it never
    /// reaches the stamp; `lemonade` and `vllm` do not — a SIGTERM from
    /// `rocm services stop` makes `status.success()` false and they return
    /// `Err`. This is the guard that keeps the stamp from agreeing with them.
    #[test]
    fn a_deliberate_stop_is_not_stamped_failed_by_the_backstop() {
        let (root, paths) = test_paths("engine-ipc-deliberate-stop");
        let state_path = root.join("engine-state.json");
        fs::create_dir_all(&root).unwrap();

        // What `aim`'s `stop_service` writes before it kills the container.
        fs::write(&state_path, br#"{"status":"stopping"}"#).unwrap();
        assert!(
            engine_stop_was_deliberate_in(Some(&paths), "svc-stopping", &state_path),
            "a `stopping` engine state is an operator stop, not a crash"
        );

        fs::write(&state_path, br#"{"status":"running"}"#).unwrap();
        assert!(
            !engine_stop_was_deliberate_in(Some(&paths), "svc-no-such-service", &state_path),
            "a running engine with no stop recorded anywhere must still be \
             reported as a crash — failing to prove a stop must never swallow one"
        );

        // The manifest-marker branch, which is the one `lemonade` and `vllm`
        // actually rely on: their engine state never says `stopping`, so the
        // only trace of an operator stop is `stop_requested_unix_ms` on the
        // record. Planted under the isolated root, not the real environment.
        paths.ensure().unwrap();
        let mut record = rocm_core::ManagedServiceRecord::new(
            &paths,
            "svc-marked-stopping",
            "lemonade",
            "model-ref",
            "canonical/model",
            "127.0.0.1",
            11_500,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.stop_requested_unix_ms = Some(1_234);
        record.write().unwrap();
        assert!(
            engine_stop_was_deliberate_in(Some(&paths), "svc-marked-stopping", &state_path),
            "a stop marker on the manifest is an operator stop even when the \
             engine state file says nothing about it"
        );

        let _ = fs::remove_dir_all(root);
    }

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

    // ---- What `rocm` asks of the shared stdio client. ----
    //
    // The conversation itself (framing, SIGPIPE guard, exit status and answer
    // handling) is the client's, tested in `rocm-engine-protocol`. These tests
    // run `rocm`'s entry point against a stand-in engine child and pin what
    // `rocm` chooses: the verb, the environment and the stderr handling. The
    // other caller's choices are pinned in `apps/rocmd/src/engine_ipc.rs`.

    #[test]
    fn rocm_asks_for_forwarded_stderr_and_no_endpoint_key() {
        let (root, paths) = crate::tests::test_paths("engine-ipc-options");
        let env_root = root.join("envs");

        let options = stdio_options(Some(&paths), Some(&env_root), true);
        assert_eq!(
            options.app_paths.map(|p| &p.data_dir),
            Some(&paths.data_dir)
        );
        assert_eq!(options.env_root, Some(env_root.as_path()));
        assert!(options.request_progress);
        // The terminal user watches an install through the child's stderr.
        assert!(options.forward_stderr);
        assert!(!options.note_failing_exit_after_answer);
        // No request `rocm` makes is about a running service's endpoint.
        assert_eq!(options.endpoint_key_file, None);

        let bare = stdio_options(None, None, false);
        assert!(bare.app_paths.is_none() && bare.env_root.is_none());
        assert!(!bare.request_progress);
        assert!(bare.forward_stderr);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    mod stdio_child {
        use super::*;
        use rocm_engine_protocol::fake_engine::FakeEngine;
        use serde_json::{Value, json};
        use std::os::unix::process::ExitStatusExt as _;

        const STOPPED: &str = r#"{"ok":true,"data":{"stopped":true}}"#;

        fn envelope(method: EngineMethod, payload: serde_json::Value) -> EngineRequestEnvelope {
            EngineRequestEnvelope { method, payload }
        }

        fn stop_request() -> EngineRequestEnvelope {
            envelope(EngineMethod::Stop, json!({"service_id": "svc"}))
        }

        fn request(
            engine: &FakeEngine,
            paths: Option<&AppPaths>,
            env_root: Option<&Path>,
            stream_progress: bool,
        ) -> Result<Value> {
            engine_stdio_child_request::<Value>(
                engine.path(),
                paths,
                &stop_request(),
                env_root,
                stream_progress,
            )
        }

        /// Run `f` holding the lock that serialises every test that sets
        /// `ROCM_CLI_*` process-wide (`with_scoped_builtin_engine_env`), so that
        /// what a child inherits when `f` starts it and what [`ambient`] reads
        /// inside `f` come from the same instant.
        fn under_env_lock<R>(f: impl FnOnce() -> R) -> R {
            let lock = BUILTIN_ENGINE_ENV_LOCK.get_or_init(|| Mutex::new(()));
            let _guard = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            f()
        }

        fn ambient(key: &str) -> Option<String> {
            std::env::var(key).ok()
        }

        #[test]
        fn the_child_gets_the_stdio_verb_and_one_line_of_request() {
            let engine = FakeEngine::answering("rocm-framing", STOPPED, 0);
            let answer = request(&engine, None, None, false).unwrap();

            assert_eq!(answer, json!({"stopped": true}));
            assert_eq!(engine.argv(), ["stdio"]);
            assert_eq!(
                engine.stdin(),
                "{\"method\":\"stop\",\"payload\":{\"service_id\":\"svc\"}}\n"
            );
        }

        #[test]
        fn the_child_runs_on_the_callers_app_paths_when_given_them() {
            let (root, paths) = crate::tests::test_paths("engine-ipc-app-paths");
            let engine = FakeEngine::answering("rocm-app-paths", STOPPED, 0);
            request(&engine, Some(&paths), None, false).unwrap();
            let _ = fs::remove_dir_all(root);

            assert_eq!(
                engine.env("ROCM_CLI_CONFIG_DIR").as_deref(),
                paths.config_dir.to_str()
            );
            assert_eq!(
                engine.env("ROCM_CLI_DATA_DIR").as_deref(),
                paths.data_dir.to_str()
            );
            assert_eq!(
                engine.env("ROCM_CLI_CACHE_DIR").as_deref(),
                paths.cache_dir.to_str()
            );
        }

        #[test]
        fn the_child_inherits_the_callers_environment_when_no_app_paths_are_given() {
            const KEYS: [&str; 3] = [
                "ROCM_CLI_CONFIG_DIR",
                "ROCM_CLI_DATA_DIR",
                "ROCM_CLI_CACHE_DIR",
            ];
            let engine = FakeEngine::answering("rocm-no-app-paths", STOPPED, 0);
            let expected = under_env_lock(|| {
                request(&engine, None, None, false).unwrap();
                KEYS.map(ambient)
            });

            for (key, expected) in KEYS.into_iter().zip(expected) {
                assert_eq!(engine.env(key), expected, "{key}");
            }
        }

        /// The comparison of a child's inherited environment with the ambient one
        /// holds while another thread scopes `ROCM_CLI_*` process-wide, which
        /// other tests of this module do: the spawn and the read happen under the
        /// same lock. Without the lock the child can start while the variables
        /// are scoped and the read see them restored (or the other way round).
        #[test]
        fn the_inherited_environment_is_compared_at_the_instant_the_child_starts() {
            use std::sync::Arc;
            use std::sync::atomic::{AtomicBool, Ordering};

            let (root, paths) = crate::tests::test_paths("engine-ipc-env-race");
            let done = Arc::new(AtomicBool::new(false));
            let flipper = {
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    while !done.load(Ordering::Relaxed) {
                        with_scoped_builtin_engine_env(Some(&paths), None, || {
                            std::thread::sleep(std::time::Duration::from_millis(2));
                        });
                        // Outside the lock, so the test's own turn is not starved.
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                })
            };

            for round in 0..25 {
                let engine = FakeEngine::answering("rocm-env-race", STOPPED, 0);
                let expected = under_env_lock(|| {
                    request(&engine, None, None, false).unwrap();
                    ambient("ROCM_CLI_DATA_DIR")
                });
                assert_eq!(engine.env("ROCM_CLI_DATA_DIR"), expected, "round {round}");
            }

            done.store(true, Ordering::Relaxed);
            flipper.join().unwrap();
            let _ = fs::remove_dir_all(root);
        }

        #[test]
        fn an_env_root_reaches_the_child_as_the_engine_envs_root() {
            let engine = FakeEngine::answering("rocm-env-root", STOPPED, 0);
            let env_root = Path::new("/an/engine/envs/root");
            request(&engine, None, Some(env_root), false).unwrap();
            assert_eq!(
                engine.env("ROCM_CLI_ENGINE_ENVS_ROOT").as_deref(),
                Some("/an/engine/envs/root")
            );

            let without = FakeEngine::answering("rocm-no-env-root", STOPPED, 0);
            let expected = under_env_lock(|| {
                request(&without, None, None, false).unwrap();
                ambient("ROCM_CLI_ENGINE_ENVS_ROOT")
            });
            assert_eq!(without.env("ROCM_CLI_ENGINE_ENVS_ROOT"), expected);
        }

        #[test]
        fn progress_streaming_is_asked_for_only_when_the_caller_wants_it() {
            let streaming = FakeEngine::answering("rocm-progress-on", STOPPED, 0);
            request(&streaming, None, None, true).unwrap();
            assert_eq!(
                streaming.env("ROCM_ENGINE_PROGRESS_STDERR").as_deref(),
                Some("1")
            );

            let quiet = FakeEngine::answering("rocm-progress-off", STOPPED, 0);
            let expected = under_env_lock(|| {
                request(&quiet, None, None, false).unwrap();
                ambient("ROCM_ENGINE_PROGRESS_STDERR")
            });
            assert_eq!(quiet.env("ROCM_ENGINE_PROGRESS_STDERR"), expected);
        }

        #[test]
        fn the_endpoint_key_is_not_passed_to_the_child() {
            // `rocm`'s requests are not about a running service's endpoint (detect,
            // install, resolve-model, stop), so this client has no service to find
            // a key for. `rocmd`'s healthcheck is the call that threads one.
            let engine = FakeEngine::answering("rocm-no-key", STOPPED, 0);
            let key = rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV;
            let expected = under_env_lock(|| {
                request(&engine, None, None, false).unwrap();
                ambient(key)
            });
            assert_eq!(engine.env(key), expected);
        }

        #[test]
        fn a_nonzero_exit_that_still_answered_is_decoded_not_discarded() {
            // The engine's own error envelope is the useful diagnostic, so an
            // answer on stdout wins over the exit status.
            let engine = FakeEngine::answering(
                "rocm-answers-then-fails",
                r#"{"ok":false,"error":{"code":"stop_failed","message":"it broke"}}"#,
                1,
            );
            let error = request(&engine, None, None, false).unwrap_err();
            assert_eq!(error.to_string(), "stop_failed: it broke");

            let engine = FakeEngine::answering("rocm-answers-then-exits-2", STOPPED, 2);
            assert_eq!(
                request(&engine, None, None, false).unwrap(),
                json!({"stopped": true})
            );
        }

        /// The write of a request to an engine that has already exited must come
        /// back as an error even where SIGPIPE is at its default action, which is
        /// how `main` runs `rocm`. Re-executes this test in a fresh process that
        /// installs `SIG_DFL` first (the test harness leaves SIGPIPE ignored, and
        /// changing it process-wide would race sibling tests).
        #[test]
        fn a_request_to_an_engine_that_already_exited_is_an_error_under_sig_dfl() {
            if std::env::var_os("ROCM_TEST_ENGINE_SIGPIPE_ARM").is_some() {
                crate::reset_sigpipe();
                let engine = FakeEngine::new("rocm-sigpipe", false, "exit 3");
                // Larger than a pipe buffer, so the write reaches the closed read end.
                let big = "x".repeat(1 << 20);
                let result = engine_stdio_child_request::<Value>(
                    engine.path(),
                    None,
                    &envelope(EngineMethod::Stop, json!({ "blob": big })),
                    None,
                    false,
                );
                let error = result.expect_err("the engine never answered");
                assert!(
                    format!("{error:#}").contains("failed to write engine request"),
                    "{error:#}"
                );
                return;
            }

            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "engine_ipc::tests::stdio_child::a_request_to_an_engine_that_already_exited_is_an_error_under_sig_dfl",
                    "--exact",
                    "--nocapture",
                ])
                .env("ROCM_TEST_ENGINE_SIGPIPE_ARM", "1")
                .status()
                .unwrap();
            assert_eq!(
                status.signal(),
                None,
                "SIGPIPE killed the process instead of surfacing as an error: {status:?}"
            );
            assert!(status.success(), "{status:?}");
        }
    }
}
