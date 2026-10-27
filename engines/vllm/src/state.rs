// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use rocm_core::{
    AppPaths, format_http_base_url, openai_models_endpoint_has_model, require_nonempty,
};
use rocm_engine_protocol::{
    DEFAULT_LOG_TAIL_LINES, EndpointRequest, EndpointResponse, HealthcheckRequest,
    HealthcheckResponse, LogsRequest, LogsResponse,
};
use serde_json::{Value, json};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::ServeHttpRequest;
use crate::runtime::VllmRuntime;

const HEALTHCHECK_TIMEOUT_MS: u64 = 700;
/// Maximum bytes to read from the end of a log file when extracting tail lines.
/// Prevents reading entire gigabyte-sized logs on startup timeout.
pub(crate) const MAX_TAIL_READ: u64 = 4 * 1024 * 1024; // 4MB

#[derive(Debug, Clone)]
pub(crate) struct ServiceFiles {
    pub state_path: PathBuf,
    pub log_path: PathBuf,
}

pub(crate) fn healthcheck_service(request: HealthcheckRequest) -> Result<HealthcheckResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path).ok();
    let endpoint_url = state.as_ref().and_then(endpoint_url_from_state);
    let model_ref = state
        .as_ref()
        .and_then(|value| value_string(value, "model_ref"));
    let listed = endpoint_url
        .as_deref()
        .map(|endpoint| query_loaded_model_endpoint(endpoint, model_ref.as_deref()))
        .transpose()
        .unwrap_or(None)
        .unwrap_or(false);
    // `/v1/models` lists a model as soon as the server accepts its name, which can
    // be minutes before the weights are resident. Confirm inference once before
    // reporting ready.
    let ready = listed
        && endpoint_url.as_deref().is_some_and(|endpoint| {
            inference_verified(
                &files.state_path,
                state.as_ref(),
                endpoint,
                model_ref.as_deref().unwrap_or_default(),
            )
        });
    let state_status = state
        .as_ref()
        .and_then(|value| value_string(value, "status"))
        .unwrap_or_else(|| "unknown".to_owned());
    let device = if state.is_some() {
        "rocm_gpu"
    } else {
        "unknown"
    };
    Ok(HealthcheckResponse::for_readiness(
        listed,
        ready,
        &state_status,
        device,
    ))
}

pub(crate) fn endpoint_response(request: EndpointRequest) -> Result<EndpointResponse> {
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

pub(crate) fn logs_response(request: LogsRequest) -> Result<LogsResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let limit = request.tail_lines.unwrap_or(DEFAULT_LOG_TAIL_LINES);
    Ok(LogsResponse {
        log_path: files.log_path.display().to_string(),
        recent_lines: if files.log_path.is_file() {
            rocm_core::tail_lines(&files.log_path, limit, MAX_TAIL_READ)?
        } else {
            Vec::new()
        },
    })
}

pub(crate) fn service_files(service_id: &str) -> Result<ServiceFiles> {
    let paths = AppPaths::discover()?;
    Ok(ServiceFiles {
        state_path: paths
            .engine_state_dir(crate::ENGINE_NAME)
            .join(format!("{service_id}.json")),
        log_path: paths
            .engine_logs_dir(crate::ENGINE_NAME)
            .join(format!("{service_id}.log")),
    })
}

/// Assembles the JSON payload persisted as a running service's state and
/// writes it. Kept here, paired with the readers that reconstruct the PID,
/// process identity, and endpoint from it ([`pid_from_state`],
/// [`identity_from_state`], [`endpoint_url_from_state`]) and with
/// [`write_terminal_state`] — `engines/lemonade/src/state.rs`'s
/// `write_running_state` pairs with matching readers, though lemonade has no
/// `pid_from_state`, and its `write_terminal_state` counterpart is
/// `mark_json_status`. The `VllmRuntime`-derived fields come from
/// `crate::runtime`, never from `process.rs`, so this module still never
/// reaches back into it.
pub(crate) fn write_running_state(
    request: &ServeHttpRequest,
    runtime: &VllmRuntime,
    pid: u32,
) -> Result<()> {
    // Kernel start-time of the launcher PID, captured while it is alive. Paired
    // with `pid`, it identifies this exact process across PID recycling so a
    // later stop never signals a reused PID.
    let start_ticks = rocm_core::process_start_ticks(pid);
    write_state(
        &request.state_path,
        &json!({
            "service_id": request.service_id,
            "engine": crate::ENGINE_NAME,
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
            "engine_recipe_required_flags": crate::engine_recipe_launch_args(request.engine_recipe.as_ref()),
            "therock_runtime_env": crate::runtime::therock_runtime_env_payload(runtime),
            "started_at_unix_ms": current_unix_millis(),
            "start_ticks": start_ticks,
            // `server_pid` is the same process, so the same token. `rocm-core`
            // pairs `server_pid` with this key, so it must be written too —
            // without it the server PID is adopted with no token at all.
            "server_start_ticks": start_ticks
        }),
    )
}

pub(crate) fn write_terminal_state(state_path: &Path, status: &str) -> Result<()> {
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

pub(crate) fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

/// Serializes `value` to `path` as pretty JSON, creating the parent directory
/// if needed. The low-level primitive [`write_running_state`] and
/// [`write_terminal_state`] both go through — it is not the only way this
/// module's state files are written: inference-verification latches go
/// through `rocm_core::merge_json_state_file` instead (see
/// [`inference_verified`]), which patches the file directly rather than
/// routing through here.
fn write_state(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(value).context("failed to serialize vLLM state")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

pub(crate) fn endpoint_url(host: &str, port: u16) -> String {
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

pub(crate) fn query_loaded_model_endpoint(
    endpoint_url: &str,
    model_ref: Option<&str>,
) -> Result<bool> {
    // Send the endpoint key when the server is protected so the healthcheck does
    // not read a 401 as "not ready" and kill a correctly-authenticated server.
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    openai_models_endpoint_has_model(
        endpoint_url,
        model_ref,
        endpoint_api_key.as_deref(),
        Duration::from_millis(HEALTHCHECK_TIMEOUT_MS),
    )
}
/// Whether the endpoint can actually complete a chat request, as opposed to
/// merely listing the model.
pub(crate) fn query_inference_probe_endpoint(endpoint_url: &str, model_ref: &str) -> Result<bool> {
    if model_ref.trim().is_empty() {
        return Ok(false);
    }
    // Send the endpoint key for the same reason the models query does: a 401 from
    // a correctly-protected server must not read as "cannot serve".
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    rocm_core::openai_chat_completion_probe(
        endpoint_url,
        model_ref,
        endpoint_api_key.as_deref(),
        rocm_core::INFERENCE_PROBE_TIMEOUT,
    )
}
/// Whether a real inference request has succeeded against this service.
///
/// Latch and backoff bookkeeping lives in `rocm-core` so both engines share one
/// implementation — what counts as *listed* differs per engine, what counts as
/// *serving* does not.
fn inference_verified(
    state_path: &Path,
    state: Option<&Value>,
    endpoint_url: &str,
    model_ref: &str,
) -> bool {
    rocm_core::engine_state_inference_verified(
        state_path,
        state,
        endpoint_url,
        model_ref,
        rocm_engine_protocol::resolve_endpoint_api_key().as_deref(),
    )
}

fn pid_from_state(state: &Value) -> Option<u32> {
    state
        .get("pid")?
        .as_u64()
        .and_then(|pid| pid.try_into().ok())
}
/// Reconstruct the recorded process identity (PID + kernel start-time) from a
/// service state file. `start_ticks` is absent in state files written before
/// this field existed, in which case verification degrades to best-effort.
pub(crate) fn identity_from_state(state: &Value) -> Option<rocm_core::ProcessIdentity> {
    let pid = pid_from_state(state)?;
    let start_ticks = state.get("start_ticks").and_then(Value::as_u64);
    Some(rocm_core::ProcessIdentity::new(pid, start_ticks))
}

fn value_string(value: &Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(str::to_owned)
}

pub(crate) fn runtime_lock_hash(runtime: &VllmRuntime) -> String {
    let mut hasher = DefaultHasher::new();
    runtime.runtime_id.hash(&mut hasher);
    runtime.command.hash(&mut hasher);
    runtime.version.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub(crate) fn current_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_engine_protocol::{DevicePolicy, EngineRecipeHint};
    use std::io::Read;
    use std::io::Write;

    /// Answer `count` chat requests on a loopback port with the given status,
    /// reporting how many arrived.
    fn spawn_chat_endpoint(
        status_line: &'static str,
        count: usize,
    ) -> (u16, std::thread::JoinHandle<usize>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
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
    fn probe_state_path(tag: &str) -> PathBuf {
        // pid + time alone repeat when two threads of one test process ask for
        // the same label in the same tick; the counter makes every call unique.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "rocm-vllm-probe-{tag}-{}-{}-{}.json",
            std::process::id(),
            current_unix_millis(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }
    /// Call `make_path` from many threads at once and assert no two calls
    /// returned the same path.
    ///
    /// Concurrent rather than sequential, because that is how tests call these
    /// helpers: calls made one after another let the clock tick between them
    /// and can pass by luck, while threads released together all read the same
    /// millisecond, so a name keyed on pid and time alone repeats.
    fn assert_each_call_gets_its_own_path(make_path: impl Fn() -> PathBuf + Sync) {
        const CALLERS: usize = 16;
        let start = std::sync::Barrier::new(CALLERS);
        let paths: Vec<PathBuf> = std::thread::scope(|scope| {
            let callers: Vec<_> = (0..CALLERS)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        make_path()
                    })
                })
                .collect();
            callers
                .into_iter()
                .map(|caller| caller.join().expect("a caller panicked"))
                .collect()
        });
        let distinct: std::collections::HashSet<_> = paths.iter().collect();
        assert_eq!(
            distinct.len(),
            CALLERS,
            "two calls with one tag must not share a path"
        );
    }
    #[test]
    fn probe_state_path_gives_each_call_its_own_file_even_for_one_tag() {
        assert_each_call_gets_its_own_path(|| probe_state_path("same-tag"));
    }
    #[test]
    fn inference_verification_latches_into_the_state_file() -> Result<()> {
        // First check probes and records the verdict; the second reads the latch
        // and leaves the model alone.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 200 OK", 1);
        let state_path = probe_state_path("latch");
        write_state(&state_path, &json!({"status": "running"}))?;
        let endpoint = endpoint_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            "facebook/opt-125m"
        ));

        let state = read_service_state(&state_path)?;
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
            "facebook/opt-125m"
        ));

        assert_eq!(
            server.join().expect("server thread"),
            1,
            "the latched check must not send a second inference request"
        );
        fs::remove_file(&state_path).ok();
        Ok(())
    }
    #[test]
    fn inference_verification_withheld_while_the_model_is_still_loading() -> Result<()> {
        // The reported failure: `/v1/models` answers but inference does not.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 503 Service Unavailable", 1);
        let state_path = probe_state_path("loading");
        write_state(&state_path, &json!({"status": "running"}))?;
        let endpoint = endpoint_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(!inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            "facebook/opt-125m"
        ));
        assert!(
            read_service_state(&state_path)?
                .get(rocm_core::INFERENCE_VERIFIED_STATE_KEY)
                .is_none(),
            "nothing is latched until inference actually answers"
        );

        server.join().expect("server thread");
        fs::remove_file(&state_path).ok();
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
    fn write_running_state_records_managed_therock_env_for_gpu_verification() -> Result<()> {
        // Uses the test's own live PID rather than a fixed placeholder so that,
        // on Linux, the `start_ticks` assertion below exercises the real
        // `rocm_core::process_start_ticks` lookup rather than only checking
        // the field is present. On other platforms `process_start_ticks` is a
        // stub that always returns `None`, so there the assertion only checks
        // `None == None`.
        let own_pid = std::process::id();
        let state_path = probe_state_path("therock-env");
        let request = ServeHttpRequest {
            service_id: "svc-vllm".to_owned(),
            model_ref: "facebook/opt-125m".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11439,
            device_policy: DevicePolicy::GpuRequired,
            gpu_indices: Vec::new(),
            runtime_id: Some("runtime-key-gfx120x".to_owned()),
            env_id: None,
            state_path: state_path.clone(),
            log_path: None,
            engine_recipe: None,
        };
        let runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from(if cfg!(windows) {
                r"C:\venv\Scripts\vllm.exe"
            } else {
                "/home/user/.venv/bin/vllm"
            }),
            python_executable: None,
            version: Some("test".to_owned()),
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk"
            } else {
                "/home/user/.venv/lib/python/site-packages/rocm_sdk"
            })),
            sdk_bin: Some(PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk\bin"
            } else {
                "/home/user/.venv/lib/python/site-packages/rocm_sdk/bin"
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
            rocm_sdk_version: None,
        };

        write_running_state(&request, &runtime, own_pid)?;
        let state = read_service_state(&state_path)?;
        fs::remove_file(&state_path).ok();

        assert_eq!(
            state.get("server_pid").and_then(Value::as_u64),
            Some(u64::from(own_pid))
        );
        assert_eq!(
            state.get("service_id").and_then(Value::as_str),
            Some("svc-vllm")
        );
        assert_eq!(
            state.get("runtime_id").and_then(Value::as_str),
            Some("therock-release:gfx120X-all")
        );
        assert_eq!(
            state.get("requested_runtime_id").and_then(Value::as_str),
            Some("runtime-key-gfx120x")
        );
        assert_eq!(
            state.get("endpoint_url").and_then(Value::as_str),
            Some("http://127.0.0.1:11439/v1")
        );
        assert_eq!(
            state.get("env_id").and_then(Value::as_str),
            Some("external-vllm-therock")
        );
        assert!(
            state
                .get("engine_recipe_required_flags")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
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
        // The identity token must be recorded so a later stop can verify it
        // against the live process, not merely be present in the JSON — under
        // both PID keys, since `server_pid` is the same process and is the
        // one `rocm-core` adopts, together with `server_start_ticks`. The
        // equality checks alone are a `None == None` trap if
        // `process_start_ticks` ever regressed to always returning `None` on
        // Linux, so also pin that a live PID actually has a token there.
        #[cfg(target_os = "linux")]
        assert!(
            state
                .get("server_start_ticks")
                .and_then(Value::as_u64)
                .is_some(),
            "a live PID has a token on Linux"
        );
        assert_eq!(
            state.get("start_ticks").and_then(Value::as_u64),
            rocm_core::process_start_ticks(own_pid)
        );
        assert_eq!(
            state.get("server_start_ticks").and_then(Value::as_u64),
            rocm_core::process_start_ticks(own_pid)
        );

        // Round-trips through disk via `write_running_state` and
        // `read_service_state`, then checks that the reader side
        // (`identity_from_state`) reconstructs the PID this writer recorded —
        // the contract the split between the two sides of this module depends
        // on.
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, own_pid);
        #[cfg(target_os = "linux")]
        assert!(
            identity.start_ticks.is_some(),
            "a live PID has a token on Linux"
        );
        assert_eq!(
            identity.start_ticks,
            rocm_core::process_start_ticks(own_pid)
        );
        Ok(())
    }
    #[test]
    fn write_running_state_forwards_engine_recipe_required_flags() -> Result<()> {
        // The sibling therock-env test above always passes `engine_recipe: None`,
        // so its `engine_recipe_required_flags` assertion (`is_some_and(Vec::is_empty)`)
        // can't tell "the forwarding call was deleted" apart from "the forwarding
        // call is present but handed an empty list" — both leave the key
        // `Some([])`. This test uses a recipe with a non-empty `required_flags`
        // so the assertion can only pass if the flags genuinely reach the state
        // file.
        let state_path = probe_state_path("recipe-flags");
        let recipe = EngineRecipeHint {
            contract_version: "1".to_owned(),
            engine: crate::ENGINE_NAME.to_owned(),
            required_flags: vec!["--enable-auto-tool-choice".to_owned()],
            parser_settings: std::collections::BTreeMap::default(),
            preferred_endpoint: None,
            unsupported_combinations: Vec::new(),
            notes: Vec::new(),
        };
        let request = ServeHttpRequest {
            service_id: "svc-vllm-recipe".to_owned(),
            model_ref: "facebook/opt-125m".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11440,
            device_policy: DevicePolicy::GpuRequired,
            gpu_indices: Vec::new(),
            runtime_id: Some("runtime-key-gfx120x".to_owned()),
            env_id: None,
            state_path: state_path.clone(),
            log_path: None,
            engine_recipe: Some(recipe),
        };
        let runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable: None,
            version: Some("test".to_owned()),
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        };

        write_running_state(&request, &runtime, std::process::id())?;
        let state = read_service_state(&state_path)?;
        fs::remove_file(&state_path).ok();

        assert_eq!(
            state
                .get("engine_recipe_required_flags")
                .and_then(Value::as_array),
            Some(&vec![Value::String("--enable-auto-tool-choice".to_owned())])
        );
        Ok(())
    }
    #[test]
    fn identity_from_state_carries_pid_and_start_ticks() {
        let state = json!({ "pid": 4321, "start_ticks": 987_654_u64 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 4321);
        assert_eq!(identity.start_ticks, Some(987_654));
    }
    #[test]
    fn identity_from_legacy_state_has_no_start_ticks() {
        // State files written before this change carry only `pid`; verification
        // must degrade gracefully rather than fail to parse.
        let state = json!({ "pid": 4321 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 4321);
        assert_eq!(identity.start_ticks, None);
    }
    #[test]
    fn identity_from_state_without_pid_is_none() {
        assert!(identity_from_state(&json!({ "status": "running" })).is_none());
    }
}
