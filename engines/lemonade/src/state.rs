// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, format_http_base_url};
use rocm_engine_protocol::{ENGINE_RECIPE_CONTRACT_VERSION, EngineRecipeHint};
use serde_json::{Value, json};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Child;

use crate::device_policy::device_policy_name;
use crate::install::{LemonadeInstallManifest, LemonadeRuntime};
use crate::{DEFAULT_HOST, DEFAULT_MODEL, ENGINE_NAME, ServeHttpRequest, current_unix_millis};

/// Maximum bytes to read from the end of a log file when extracting tail lines.
/// Prevents reading entire gigabyte-sized logs on startup timeout.
pub(crate) const MAX_TAIL_READ: u64 = 4 * 1024 * 1024; // 4MB

#[derive(Debug, Clone)]
pub(crate) struct ServiceFiles {
    pub(crate) state_path: PathBuf,
    pub(crate) log_path: PathBuf,
}

/// Whether a real inference request has succeeded against this service.
///
/// Latch and backoff bookkeeping lives in `rocm-core` so both engines share one
/// implementation — what counts as *listed* differs per engine, what counts as
/// *serving* does not.
pub(crate) fn inference_verified(
    state_path: &Path,
    state: Option<&Value>,
    endpoint_url: &str,
    model_ref: &str,
) -> bool {
    let Some((host, port)) = parse_http_endpoint(endpoint_url) else {
        return false;
    };
    rocm_core::engine_state_inference_verified(
        state_path,
        state,
        &format_http_base_url(&host, port),
        model_ref,
        rocm_engine_protocol::resolve_endpoint_api_key().as_deref(),
    )
}

pub(crate) fn write_running_state(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    pid: u32,
    server_pid: Option<u32>,
    status: &str,
) -> Result<()> {
    write_state(
        &request.state_path,
        &json!({
            "service_id": request.service_id,
            "engine": ENGINE_NAME,
            "status": status,
            "pid": pid,
            "server_pid": server_pid,
            "model_ref": request.model_ref,
            "canonical_model_id": request.model_ref,
            "host": request.host,
            "port": request.port,
            "endpoint_url": endpoint_url(&request.host, request.port),
            "device_policy": device_policy_name(&request.device_policy),
            "gpu_indices": request.gpu_indices,
            "runtime_id": request.runtime_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "env_id": request.env_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "runtime_kind": "lemonade_embeddable",
            "runtime_executable": runtime.manifest.lemond,
            "log_path": request.log_path.as_ref().map(|path| path.display().to_string()),
            "engine_recipe": request.engine_recipe,
            "started_at_unix_ms": current_unix_millis(),
            // Kernel start-times captured while each PID is alive. Paired with the
            // PIDs, they identify these exact processes across PID recycling so a
            // later stop never signals a reused PID.
            "start_ticks": rocm_core::process_start_ticks(pid),
            "server_start_ticks": server_pid.and_then(rocm_core::process_start_ticks)
        }),
    )
}

pub(crate) fn service_files(service_id: &str) -> Result<ServiceFiles> {
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

pub(crate) fn endpoint_url(host: &str, port: u16) -> String {
    format!("{}/v1", format_http_base_url(host, port))
}

pub(crate) fn endpoint_url_from_state(state: &Value) -> Option<String> {
    value_string(state, "endpoint_url").or_else(|| {
        let host = value_string(state, "host")?;
        let port = value_u32(state, "port")?;
        let port = u16::try_from(port).ok()?;
        Some(endpoint_url(&host, port))
    })
}

pub(crate) fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

/// Replace the service's state file whole. Shares the state directory's lock
/// with every merge (`rocm_core::write_json_state_file`), so a full rewrite
/// never lands inside another writer's merge, and publishes atomically, so a
/// lock-free reader such as `rocm services list` never sees half a document.
fn write_state(path: &Path, value: &Value) -> Result<()> {
    rocm_core::write_json_state_file(path, value)
}

/// Merge `patch` into the service's state file under the lock the healthcheck's
/// inference latch and `rocmd`'s endpoint publish also take
/// (`rocm_core::merge_json_state_file`), so none of them drops a key another
/// just wrote — `server_pid` above all, which a later stop needs. A state file
/// that is empty, torn or not a JSON object is an error and left as it was.
pub(crate) fn merge_json_state(path: &Path, patch: &Value) -> Result<()> {
    rocm_core::merge_json_state_file(path, patch)
}

pub(crate) fn mark_json_status(path: &Path, status: &str) -> Result<()> {
    merge_json_state(
        path,
        &json!({
            "engine": ENGINE_NAME,
            "status": status,
            "stopped_at_unix_ms": current_unix_millis(),
        }),
    )
}

pub(crate) fn value_string(value: &Value, key: &str) -> Option<String> {
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

/// Reconstruct the identity (PID + kernel start-time) of the process a stop
/// should terminate. The server PID is preferred, falling back to the launcher
/// PID, each paired with its recorded start-time. `start_ticks` is absent in
/// pre-existing state files, in which case verification degrades to best-effort.
pub(crate) fn identity_from_state(state: &Value) -> Option<rocm_core::ProcessIdentity> {
    if let Some(server_pid) = value_u32(state, "server_pid") {
        let start_ticks = state.get("server_start_ticks").and_then(Value::as_u64);
        return Some(rocm_core::ProcessIdentity::new(server_pid, start_ticks));
    }
    let pid = value_u32(state, "pid")?;
    let start_ticks = state.get("start_ticks").and_then(Value::as_u64);
    Some(rocm_core::ProcessIdentity::new(pid, start_ticks))
}

pub(crate) fn terminate_pid(pid: u32, _force: bool) -> bool {
    rocm_core::terminate_process(pid).is_ok()
}

/// A server process the wrapper started and has to reap before it returns.
pub(crate) trait StartedServer {
    fn pid(&self) -> u32;
    /// Kill the server outright; called only when terminating it by pid failed.
    fn force_kill(&mut self);
    /// Wait for the server to exit.
    fn wait_for_exit(&mut self) -> Result<()>;
}

impl StartedServer for Child {
    fn pid(&self) -> u32 {
        self.id()
    }

    fn force_kill(&mut self) {
        let _ = self.kill();
    }

    fn wait_for_exit(&mut self) -> Result<()> {
        self.wait().map(|_| ()).map_err(Into::into)
    }
}

/// Run `record`, the state write that names a server the wrapper has just
/// started; when it fails, stop and reap `server` before returning the error.
///
/// The wrapper waits on its server only after recording it, so returning the
/// error alone would leave a server running that no state file names, which
/// `rocm services stop` could then not find.
pub(crate) fn record_server_or_stop_it(
    server: &mut impl StartedServer,
    record: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let Err(error) = record() else {
        return Ok(());
    };
    let pid = server.pid();
    if !terminate_pid(pid, true) {
        server.force_kill();
    }
    let outcome = match server.wait_for_exit() {
        Ok(()) => "stopped it".to_owned(),
        Err(wait_error) => {
            format!("tried to stop it, but waiting for it failed: {wait_error:#}")
        }
    };
    Err(error.context(format!(
        "could not record the server (pid {pid}) in its state file, so {outcome}"
    )))
}

pub(crate) fn free_local_port() -> Result<u16> {
    let listener = TcpListener::bind((DEFAULT_HOST, 0)).context("failed to reserve local port")?;
    Ok(listener.local_addr()?.port())
}

pub(crate) fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
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

pub(crate) fn accepted_engine_recipe(
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

pub(crate) fn parse_engine_recipe_json(value: Option<String>) -> Result<Option<EngineRecipeHint>> {
    value
        .map(|text| {
            serde_json::from_str::<EngineRecipeHint>(&text)
                .context("failed to parse engine recipe JSON")
        })
        .transpose()
        .and_then(accepted_engine_recipe)
}

pub(crate) fn resolve_lemonade_model_ref(model_ref: &str) -> String {
    let trimmed = model_ref.trim();
    let lower = trimmed.to_ascii_lowercase();
    // Only exact, recognized shorthand aliases map to the bundled assistant GGUF.
    // A syntactically valid `owner/repo` Hugging Face checkpoint must never be
    // silently rewritten here — e.g. `Qwen/Qwen2.5-1.5B-Instruct` is a real
    // checkpoint id, not a shorthand, and is passed through unchanged so the
    // serve path can honour (or explicitly reject) it rather than substituting
    // an unrelated model behind the user's back.
    if trimmed.is_empty()
        || matches!(
            lower.as_str(),
            "qwen"
                | "assistant"
                | "default"
                | "small"
                | "lemonade-qwen"
                | "qwen-gguf"
                | "qwen3-4b"
                | "qwen3-4b-instruct"
                | "qwen3-4b-instruct-2507-gguf"
        )
    {
        DEFAULT_MODEL.to_owned()
    } else if matches!(
        lower.as_str(),
        "tiny" | "qwen-smoke" | "lemonade-tiny" | "qwen3-0.6b-gguf"
    ) {
        "Qwen3-0.6B-GGUF".to_owned()
    } else {
        trimmed.to_owned()
    }
}

pub(crate) fn manifest_lock_hash(manifest: &LemonadeInstallManifest) -> String {
    let mut hasher = DefaultHasher::new();
    manifest.env_id.hash(&mut hasher);
    manifest.version.hash(&mut hasher);
    manifest.runtime_dir.hash(&mut hasher);
    manifest.lemond.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scratch_dir;
    use std::io::{Read, Write};
    use std::time::Duration;

    #[test]
    fn identity_from_state_prefers_server_pid_and_its_start_ticks() {
        let state = json!({
            "pid": 100,
            "start_ticks": 111_u64,
            "server_pid": 200,
            "server_start_ticks": 222_u64,
        });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 200);
        assert_eq!(identity.start_ticks, Some(222));
    }

    #[test]
    fn identity_from_state_falls_back_to_launcher_pid() {
        let state = json!({ "pid": 100, "start_ticks": 111_u64 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 100);
        assert_eq!(identity.start_ticks, Some(111));
    }

    #[test]
    fn identity_from_legacy_state_has_no_start_ticks() {
        // Pre-existing state files carry only PIDs; verification must degrade.
        let state = json!({ "server_pid": 200 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 200);
        assert_eq!(identity.start_ticks, None);
    }

    #[test]
    fn identity_from_state_without_any_pid_is_none() {
        assert!(identity_from_state(&json!({ "status": "running" })).is_none());
    }

    #[test]
    fn qwen_alias_resolves_to_validated_assistant_gguf_model() {
        assert_eq!(resolve_lemonade_model_ref("qwen"), DEFAULT_MODEL);
        assert_eq!(resolve_lemonade_model_ref("qwen-smoke"), "Qwen3-0.6B-GGUF");
    }

    #[test]
    fn canonical_hugging_face_name_passes_through_unchanged() {
        assert_eq!(
            resolve_lemonade_model_ref("LiquidAI/LFM2.5-230M-GGUF:Q4_0"),
            "LiquidAI/LFM2.5-230M-GGUF:Q4_0"
        );
        // Regression (EAI-7370): a fully-qualified checkpoint id that merely
        // contains a known-alias substring must not be swapped for the bundled
        // default. `Qwen/Qwen2.5-1.5B-Instruct` was silently served as
        // `Qwen3-4B-Instruct-2507-GGUF`; it must now pass through verbatim.
        assert_eq!(
            resolve_lemonade_model_ref("Qwen/Qwen2.5-1.5B-Instruct"),
            "Qwen/Qwen2.5-1.5B-Instruct"
        );
    }

    /// Answer `count` chat requests on a loopback port with the given status,
    /// recording how many arrived.
    fn spawn_chat_endpoint(
        status_line: &'static str,
        count: usize,
    ) -> (u16, std::thread::JoinHandle<usize>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let handle = std::thread::spawn(move || {
            let mut served = 0;
            for _ in 0..count {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 1024];
                let _ = stream.read(&mut buffer);
                let body = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
                let _ = write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                served += 1;
            }
            served
        });
        (port, handle)
    }

    #[test]
    fn inference_verification_latches_into_the_state_file() {
        // First check probes and records the verdict; the second reads the latch
        // and leaves the model alone.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 200 OK", 1);
        let dir = scratch_dir("inference-latch");
        let state_path = dir.join("state.json");
        fs::write(&state_path, json!({"status": "ready"}).to_string()).expect("seed state");
        let endpoint = format_http_base_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            DEFAULT_MODEL
        ));

        let state = read_service_state(&state_path).expect("state readable");
        assert!(
            state
                .get(rocm_core::INFERENCE_VERIFIED_STATE_KEY)
                .and_then(Value::as_u64)
                .is_some(),
            "a passing probe is latched so later healthchecks skip it"
        );
        assert!(inference_verified(
            &state_path,
            Some(&state),
            &endpoint,
            DEFAULT_MODEL
        ));

        assert_eq!(
            server.join().expect("server thread"),
            1,
            "the latched check must not send a second inference request"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inference_verification_withheld_while_the_endpoint_cannot_serve() {
        // The reported failure: the model is listed but inference still 5xxs.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 503 Service Unavailable", 1);
        let dir = scratch_dir("inference-unverified");
        let state_path = dir.join("state.json");
        fs::write(&state_path, json!({"status": "ready"}).to_string()).expect("seed state");
        let endpoint = format_http_base_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(!inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            DEFAULT_MODEL
        ));

        let state = read_service_state(&state_path).expect("state readable");
        assert!(
            state.get(rocm_core::INFERENCE_VERIFIED_STATE_KEY).is_none(),
            "nothing is latched until inference actually answers"
        );

        server.join().expect("server thread");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn endpoint_parser_supports_ipv6_loopback() {
        assert_eq!(
            parse_http_endpoint("http://[::1]:11435/v1"),
            Some(("::1".to_owned(), 11435))
        );
    }

    /// A wrapper merge into a torn state file — the half-written document the
    /// old non-atomic writers could leave — fails and keeps the file as it
    /// was, rather than replacing it with the patch and losing `server_pid`.
    #[test]
    fn a_merge_into_a_torn_state_file_fails_and_keeps_it() {
        let dir = scratch_dir("torn-state");
        let path = dir.join("svc.json");
        let torn = br#"{"status": "running", "server_pid": 4242, "server_st"#;
        fs::write(&path, torn).unwrap();

        let merge = merge_json_state(&path, &json!({ "status": "ready" }));
        let mark = mark_json_status(&path, "stopped");

        let after = fs::read(&path).unwrap();
        fs::remove_dir_all(&dir).ok();
        for result in [merge, mark] {
            let error = format!(
                "{:#}",
                result.expect_err("a torn state file must be refused")
            );
            assert!(error.contains("is not valid JSON"), "{error}");
            assert!(
                error.contains("left it untouched and recorded nothing"),
                "{error}"
            );
        }
        assert_eq!(after, torn, "the torn file must be left as it was");
    }

    /// The race behind the lost `server_pid`: the wrapper merging into its own
    /// state file while a healthcheck latches its inference probe into the same
    /// file. Both must keep the server's identity, so a later stop still
    /// targets the server rather than the wrapper.
    #[test]
    fn wrapper_and_healthcheck_merges_keep_the_server_identity() {
        const ROUNDS: u64 = 150;
        let dir = scratch_dir("merge-race");
        let path = dir.join("svc.json");
        write_state(
            &path,
            &json!({
                "status": "running",
                "pid": 100,
                "start_ticks": 111_u64,
                "server_pid": 4242,
                "server_start_ticks": 99_u64,
            }),
        )
        .unwrap();

        let (wrapper, healthcheck) = std::thread::scope(|scope| {
            let wrapper = scope.spawn(|| -> Result<()> {
                for round in 0..ROUNDS {
                    merge_json_state(&path, &json!({ "backend_state": "ready", "round": round }))?;
                    mark_json_status(&path, "ready")?;
                }
                Ok(())
            });
            let healthcheck = scope.spawn(|| -> Result<()> {
                for round in 0..ROUNDS {
                    rocm_core::merge_json_state_file(
                        &path,
                        &json!({ rocm_core::INFERENCE_PROBE_ATTEMPTED_STATE_KEY: round }),
                    )?;
                }
                Ok(())
            });
            (
                wrapper.join().expect("wrapper thread"),
                healthcheck.join().expect("healthcheck thread"),
            )
        });

        let state = read_service_state(&path);
        fs::remove_dir_all(&dir).ok();
        wrapper.expect("every wrapper merge succeeds");
        healthcheck.expect("every healthcheck merge succeeds");
        let state = state.expect("the state file is still a whole document");
        let identity = identity_from_state(&state).expect("an identity to stop");
        assert_eq!(identity.pid, 4242, "a stop must target the server: {state}");
        assert_eq!(identity.start_ticks, Some(99), "{state}");
        assert_eq!(state["round"], ROUNDS - 1, "{state}");
        assert_eq!(
            state[rocm_core::INFERENCE_PROBE_ATTEMPTED_STATE_KEY],
            ROUNDS - 1,
            "{state}"
        );
    }

    const SLEEPER_CHILD_ENV: &str = "ROCM_LEMONADE_TEST_SLEEPER_CHILD";

    /// This test binary, run again as a stand-in server that stays up for a
    /// minute unless something stops it.
    fn spawn_stand_in_server() -> Child {
        std::process::Command::new(std::env::current_exe().expect("current test executable"))
            .args([
                "state::tests::a_server_whose_state_cannot_be_recorded_is_stopped",
                "--exact",
                "--quiet",
            ])
            .env(SLEEPER_CHILD_ENV, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn the stand-in server")
    }

    #[test]
    fn a_server_whose_state_cannot_be_recorded_is_stopped() {
        if std::env::var_os(SLEEPER_CHILD_ENV).is_some() {
            std::thread::sleep(Duration::from_mins(1));
            return;
        }
        let mut server = spawn_stand_in_server();
        let pid = server.id();

        let result =
            record_server_or_stop_it(&mut server, || Err(anyhow::anyhow!("state file is torn")));

        // Already reaped by the helper, so this returns the recorded status at
        // once; `None` would mean the server is still running.
        let exited = server.try_wait().expect("query the stand-in server");
        if exited.is_none() {
            let _ = server.kill();
            let _ = server.wait();
        }
        let error = format!("{:#}", result.expect_err("the record failed"));
        assert_eq!(
            error,
            format!(
                "could not record the server (pid {pid}) in its state file, so stopped it: \
                 state file is torn"
            )
        );
        let status = exited.expect("the server must have been stopped and reaped");
        assert!(
            !status.success(),
            "the server was stopped, not left to finish its minute: {status}"
        );
    }

    #[test]
    fn a_server_whose_state_is_recorded_is_left_running() {
        let mut server = spawn_stand_in_server();

        let result = record_server_or_stop_it(&mut server, || Ok(()));

        let running = server
            .try_wait()
            .expect("query the stand-in server")
            .is_none();
        let _ = server.kill();
        let _ = server.wait();
        result.expect("the record succeeded");
        assert!(running, "a recorded server is the wrapper's to wait on");
    }
}
