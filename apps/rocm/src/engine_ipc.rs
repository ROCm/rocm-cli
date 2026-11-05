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
//! `builtin_engine_serve_http_args`). One item sits physically inside this
//! span in `main.rs` but stays there rather than moving: `impl
//! From<WatcherModeArg> for WatcherMode` is unrelated clap-arg-to-domain-type
//! conversion that only happens to sit between `find_engine_plugin_binary` and
//! `engine_request` in the original file (it has since moved to `cli.rs`,
//! next to the enum it converts).
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
//! crate root — `with_sigpipe_ignored` and `device_policy_name` — stay in
//! `main.rs` and are reached unqualified via `use crate::{...}`, same as for
//! other mechanically relocated modules. `parse_device_policy` moved out of
//! `main.rs` into `gpu_policy.rs` in ROCMAI-214 (Phase 6f); it is reached via
//! `use crate::gpu_policy::parse_device_policy;` instead. `apply_app_path_env`,
//! `app_path_env_vars`, `optional_arg`, and `engine_recipe_json_arg` moved out
//! of `main.rs` into `engine_runtime_select.rs` in ROCMAI-215 (Phase 6f); this
//! file's `use` list below points there now instead of at the crate root.

use std::ffi::OsString;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, engine_binary_path, engine_plugin_dirs};
use rocm_engine_protocol::{
    DevicePolicy, EngineMethod, EnginePluginDescriptor, EngineRecipeHint, EngineRequestEnvelope,
    EngineResponseEnvelope,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::engine_runtime_select::{
    app_path_env_vars, apply_app_path_env, engine_recipe_json_arg, optional_arg,
};
use crate::gpu_policy::parse_device_policy;
use crate::{device_policy_name, with_sigpipe_ignored};

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
    let envelope = EngineRequestEnvelope {
        method,
        payload: serde_json::to_value(request)
            .context("failed to encode engine request payload")?,
    };
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
/// `engine_binary stdio`, send `envelope` on its stdin and decode its answer.
/// Split out so a test can point it at a stand-in binary.
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
    let mut command = ProcessCommand::new(engine_binary);
    command.arg("stdio");
    if let Some(paths) = paths {
        apply_app_path_env(&mut command, paths);
    }
    if let Some(env_root) = env_root {
        command.env("ROCM_CLI_ENGINE_ENVS_ROOT", env_root);
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
            serde_json::to_writer(&mut stdin, envelope)
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
    f: impl FnOnce() -> R,
) -> R {
    if paths.is_none() && env_root.is_none() {
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

    // ---- The stdio client, driven against a stand-in engine child. ----
    //
    // These pin what `rocm` puts on an engine child's command line, environment
    // and stdin and how it reads the answer back. `rocmd` keeps its own client
    // with a different contract; its tests are in `apps/rocmd/src/engine_ipc.rs`.

    fn envelope(method: EngineMethod, payload: serde_json::Value) -> EngineRequestEnvelope {
        EngineRequestEnvelope { method, payload }
    }

    #[test]
    fn decode_engine_response_reports_each_way_an_answer_can_be_unusable() {
        let failed = EngineResponseEnvelope {
            ok: false,
            data: None,
            error: Some(rocm_engine_protocol::EngineErrorDetail {
                code: "stop_failed".to_owned(),
                message: "it broke".to_owned(),
            }),
        };
        let error = decode_engine_response::<serde_json::Value>(failed).unwrap_err();
        assert_eq!(error.to_string(), "stop_failed: it broke");

        let unexplained = EngineResponseEnvelope {
            ok: false,
            data: None,
            error: None,
        };
        let error = decode_engine_response::<serde_json::Value>(unexplained).unwrap_err();
        assert_eq!(error.to_string(), "unknown engine error");

        let empty = EngineResponseEnvelope {
            ok: true,
            data: None,
            error: None,
        };
        let error = decode_engine_response::<serde_json::Value>(empty).unwrap_err();
        assert_eq!(
            error.to_string(),
            "engine response envelope did not contain data"
        );

        let wrong_shape = EngineResponseEnvelope::success(serde_json::json!({"stopped": "yes"}));
        let error =
            decode_engine_response::<rocm_engine_protocol::StopResponse>(wrong_shape).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to decode engine response payload"
        );
    }

    #[cfg(unix)]
    mod stdio_child {
        use super::*;
        use rocm_engine_protocol::fake_engine::FakeEngine;
        use serde_json::{Value, json};
        use std::os::unix::process::ExitStatusExt as _;

        const STOPPED: &str = r#"{"ok":true,"data":{"stopped":true}}"#;

        fn stop_request() -> EngineRequestEnvelope {
            envelope(EngineMethod::Stop, json!({"service_id": "svc"}))
        }

        fn request(
            engine: &FakeEngine,
            paths: Option<&AppPaths>,
            env_root: Option<&Path>,
            stream_progress: bool,
        ) -> Result<Value> {
            engine.run(|| {
                engine_stdio_child_request::<Value>(
                    engine.path(),
                    paths,
                    &stop_request(),
                    env_root,
                    stream_progress,
                )
            })
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
            let engine = FakeEngine::answering("rocm-no-app-paths", STOPPED, 0);
            request(&engine, None, None, false).unwrap();

            for key in [
                "ROCM_CLI_CONFIG_DIR",
                "ROCM_CLI_DATA_DIR",
                "ROCM_CLI_CACHE_DIR",
            ] {
                assert_eq!(engine.env(key), ambient(key), "{key}");
            }
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
            request(&without, None, None, false).unwrap();
            assert_eq!(
                without.env("ROCM_CLI_ENGINE_ENVS_ROOT"),
                ambient("ROCM_CLI_ENGINE_ENVS_ROOT")
            );
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
            request(&quiet, None, None, false).unwrap();
            assert_eq!(
                quiet.env("ROCM_ENGINE_PROGRESS_STDERR"),
                ambient("ROCM_ENGINE_PROGRESS_STDERR")
            );
        }

        #[test]
        fn the_endpoint_key_is_not_passed_to_the_child() {
            // `rocm`'s requests are not about a running service's endpoint (detect,
            // install, resolve-model, stop), so this client has no service to find
            // a key for. `rocmd`'s healthcheck is the call that threads one.
            let engine = FakeEngine::answering("rocm-no-key", STOPPED, 0);
            request(&engine, None, None, false).unwrap();
            assert_eq!(
                engine.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV),
                ambient(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV)
            );
        }

        #[test]
        fn a_nonzero_exit_with_no_answer_reports_the_status_and_stderr() {
            let engine = FakeEngine::new("rocm-crash", true, "echo boom >&2\nexit 3");
            let error = request(&engine, None, None, false).unwrap_err();
            assert_eq!(
                error.to_string(),
                "engine stdio process exited with status exit status: 3: boom"
            );

            let silent = FakeEngine::new("rocm-silent-crash", true, "exit 4");
            let error = request(&silent, None, None, false).unwrap_err();
            assert_eq!(
                error.to_string(),
                "engine stdio process exited with status exit status: 4"
            );
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

        #[test]
        fn an_unparseable_answer_carries_the_childs_stderr() {
            let engine = FakeEngine::new(
                "rocm-garbage",
                true,
                "printf 'not json'\necho oops >&2\nexit 0",
            );
            let error = request(&engine, None, None, false).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope; stderr: oops"
            );

            let silent = FakeEngine::new("rocm-garbage-quiet", true, "printf 'not json'");
            let error = request(&silent, None, None, false).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope"
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
                let result = engine.run(|| {
                    engine_stdio_child_request::<Value>(
                        engine.path(),
                        None,
                        &envelope(EngineMethod::Stop, json!({ "blob": big })),
                        None,
                        false,
                    )
                });
                let error = result.expect_err("the engine never answered");
                assert!(
                    format!("{error:#}").contains("failed to write engine request"),
                    "{error:#}"
                );
                return;
            }

            let status = ProcessCommand::new(std::env::current_exe().unwrap())
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
