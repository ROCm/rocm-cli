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
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::runtime::VllmRuntime;

const HEALTHCHECK_TIMEOUT_MS: u64 = 700;
/// For files larger than this, [`tail_lines`] seeks to this many bytes from
/// the end instead of reading the whole file.
const MAX_TAIL_READ: u64 = 4 * 1024 * 1024;

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
            tail_lines(&files.log_path, limit)?
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
/// if needed. The one write path for every state-writing entry point in this
/// module, and for `process.rs`, which assembles the running-state payload
/// itself (it already has the launch request, resolved runtime, and PID) and
/// hands the finished [`Value`] here — this module never reaches back into
/// `process.rs` to compute any of it.
pub(crate) fn write_state(path: &Path, value: &Value) -> Result<()> {
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

/// Reads the last N lines from a file efficiently by seeking.
/// For files larger than [`MAX_TAIL_READ`], seeks to `MAX_TAIL_READ` from the end.
pub(crate) fn tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open log file {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    let file_size = metadata.len();

    // For large files, seek to MAX_TAIL_READ from the end. When the seek lands in
    // the middle of a line, the first line read back is a partial fragment that
    // must be dropped. When it lands exactly on a line boundary (the byte before
    // `seek_pos` is a newline) the first line is complete and must be kept.
    let mut first_line_is_partial = false;
    if file_size > MAX_TAIL_READ {
        let seek_pos = file_size - MAX_TAIL_READ;
        // Probe the byte preceding `seek_pos` to classify the first line, then
        // leave the cursor at `seek_pos` for the buffered read below.
        file.seek(SeekFrom::Start(seek_pos - 1))
            .with_context(|| format!("failed to seek in log file {}", path.display()))?;
        let mut probe = [0u8; 1];
        file.read_exact(&mut probe)
            .with_context(|| format!("failed to read from log file {}", path.display()))?;
        first_line_is_partial = probe[0] != b'\n';
    }

    let buffered = BufReader::new(file);
    let mut lines: Vec<String> = buffered
        .lines()
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to read lines from {}", path.display()))?;

    // Drop the leading partial line produced by seeking into the middle of a line.
    if first_line_is_partial && !lines.is_empty() {
        lines.remove(0);
    }

    // Return only the last `limit` lines
    let start_idx = if lines.len() > limit {
        lines.len() - limit
    } else {
        0
    };
    Ok(lines.into_iter().skip(start_idx).collect())
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
        std::env::temp_dir().join(format!(
            "rocm-vllm-probe-{tag}-{}-{}.json",
            std::process::id(),
            current_unix_millis()
        ))
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
    fn tail_lines_returns_suffix() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-tail-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, "a\nb\nc\n")?;
        let lines = tail_lines(&path, 2)?;
        fs::remove_file(path).ok();
        assert_eq!(lines, vec!["b".to_owned(), "c".to_owned()]);
        Ok(())
    }
    #[test]
    fn tail_lines_keeps_first_line_when_seek_lands_on_boundary() -> Result<()> {
        // Build a file where the MAX_TAIL_READ window starts exactly on a line
        // boundary: a prefix ending in '\n', followed by exactly MAX_TAIL_READ
        // bytes of complete lines. The first windowed line must NOT be dropped.
        let prefix = format!("{}\n", "p".repeat(63));
        let mut tail = String::from("FIRSTLINE\n");
        while tail.len() + 2 <= MAX_TAIL_READ as usize {
            tail.push_str("y\n");
        }
        while tail.len() < MAX_TAIL_READ as usize {
            tail.push('z');
        }
        assert_eq!(tail.len(), MAX_TAIL_READ as usize);

        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-tail-boundary-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, format!("{prefix}{tail}"))?;
        let lines = tail_lines(&path, usize::MAX)?;
        fs::remove_file(&path).ok();

        assert_eq!(
            lines.first().map(String::as_str),
            Some("FIRSTLINE"),
            "complete first line must be preserved when the seek lands on a newline boundary"
        );
        assert!(
            !lines.iter().any(|line| line.contains('p')),
            "bytes before the tail window must not appear"
        );
        Ok(())
    }
    #[test]
    fn tail_lines_drops_partial_first_line_when_seek_lands_midline() -> Result<()> {
        // The window starts in the middle of a line, so the leading fragment is
        // partial and must be dropped.
        let prefix = "p".repeat(64);
        let mut tail = String::from("PARTIALFRAGMENT");
        tail.push('\n');
        tail.push_str("SECONDLINE\n");
        while tail.len() < MAX_TAIL_READ as usize {
            tail.push_str("y\n");
        }
        // Trim back to exactly MAX_TAIL_READ bytes so the window starts mid-line.
        tail.truncate(MAX_TAIL_READ as usize);

        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-tail-midline-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, format!("{prefix}{tail}"))?;
        let lines = tail_lines(&path, usize::MAX)?;
        fs::remove_file(&path).ok();

        assert_eq!(
            lines.first().map(String::as_str),
            Some("SECONDLINE"),
            "partial leading fragment must be dropped when the seek lands mid-line"
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
