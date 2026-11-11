// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Managed-service endpoint readiness.
//!
//! How far along a managed service's OpenAI-compatible endpoint is: whether it
//! lists the served model, and whether a real single-token chat request has
//! come back (see [`EndpointReadiness`]). Also the shared latch-and-throttle
//! bookkeeping the engine adapters use to record that inference was verified
//! (`engine_state_inference_verified`), and the model-reference matching that
//! decides whether a listed model id is the one a service was started with.
//!
//! The engine state file that latch writes into has several writers, so the
//! locked, atomic writers for it live here too ([`merge_json_state_file`],
//! [`write_json_state_file`]).
//!
//! The loopback HTTP and TCP primitives these probes are built on live in
//! `net.rs`; the state-file keys and the probe retry interval that tie the
//! probes to a [`ManagedServiceRecord`] live in `service_record.rs`.

use anyhow::{Context, Result, anyhow};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::atomic_write::write_file_atomically;
use crate::model_catalog::builtin_model_recipes;
use crate::net::{http_get_text_with_auth, http_post_json_with_auth};
use crate::process::FileLock;
use crate::service_record::{
    INFERENCE_PROBE_ATTEMPTED_STATE_KEY, INFERENCE_PROBE_RETRY_INTERVAL,
    INFERENCE_VERIFIED_STATE_KEY, ManagedServiceRecord,
};
use crate::unix_time_millis;

/// Budget for the single-token chat request that proves a service can serve.
///
/// Generously longer than a model-listing timeout: the probe is a real inference
/// request, and a first request against a freshly loaded model pays for prompt
/// processing before it answers.
pub const INFERENCE_PROBE_TIMEOUT: Duration = Duration::from_secs(8);

/// Name of the lock file that serializes every write of a JSON state file in
/// one directory, taken by [`merge_json_state_file`] and
/// [`write_json_state_file`].
///
/// One lock per directory rather than one per state file, so it can be left in
/// place for good: deleting a lock file while another writer holds or waits on
/// it would let a third writer lock a fresh file of the same name and run
/// alongside them. It is never removed, holds no data, and its `.lock`
/// extension keeps it out of every scan that looks for `*.json` state files.
pub const JSON_STATE_LOCK_FILE_NAME: &str = ".json-state.lock";

/// The lock [`merge_json_state_file`] and [`write_json_state_file`] take for
/// `path`: [`JSON_STATE_LOCK_FILE_NAME`] in `path`'s directory.
#[must_use]
pub fn json_state_lock_path(path: &Path) -> PathBuf {
    path.parent()
        .unwrap_or_else(|| Path::new(""))
        .join(JSON_STATE_LOCK_FILE_NAME)
}

/// Merge `patch`'s top-level keys into the JSON object stored at `path`.
///
/// Every writer of the state file — the engine wrapper that owns the service,
/// a healthcheck latching that inference was verified, `rocmd` publishing the
/// Anthropic endpoint — patches the same document, so the read, the merge and
/// the write run under one cross-process lock ([`json_state_lock_path`]) and
/// no writer can drop a key another just recorded. The result is published
/// atomically ([`crate::atomic_write::write_file_atomically`]), so a reader
/// that takes no lock still sees a whole document, never a truncated one.
///
/// Creates the file (and its parent) when absent. An existing file that is
/// empty, is not valid JSON or does not hold a JSON object is an error, and the
/// file is left exactly as it was: it may hold the only record of a running
/// process (`server_pid`), and replacing it with `patch` alone would lose it.
/// A `patch` that is not an object has no keys to merge; the file is not
/// touched and the call succeeds.
pub fn merge_json_state_file(path: &Path, patch: &serde_json::Value) -> Result<()> {
    let Some(patch) = patch.as_object() else {
        return Ok(());
    };
    let _lock = FileLock::acquire(json_state_lock_path(path))?;
    let mut object = match fs::read(path) {
        Ok(bytes) => existing_state_object(path, &bytes)?,
        Err(error) if error.kind() == ErrorKind::NotFound => serde_json::Map::new(),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to read {}; left it untouched and recorded nothing",
                    path.display()
                )
            });
        }
    };
    for (key, patch_value) in patch {
        object.insert(key.clone(), patch_value.clone());
    }
    publish_json_state(path, &serde_json::Value::Object(object))
}

/// Replace the JSON state file at `path` with `value`.
///
/// Takes the same lock as [`merge_json_state_file`] and publishes the same
/// atomic way, so a full rewrite cannot land in the middle of another
/// writer's merge.
pub fn write_json_state_file(path: &Path, value: &serde_json::Value) -> Result<()> {
    let _lock = FileLock::acquire(json_state_lock_path(path))?;
    publish_json_state(path, value)
}

/// The object an existing state file holds, or an error that says the file
/// was left untouched.
fn existing_state_object(
    path: &Path,
    bytes: &[u8],
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let refuse = |what: String| {
        anyhow!(
            "{} {what}; left it untouched and recorded nothing, rather than replace what it holds",
            path.display()
        )
    };
    if bytes.is_empty() {
        return Err(refuse("is empty".to_owned()));
    }
    match serde_json::from_slice::<serde_json::Value>(bytes) {
        Ok(serde_json::Value::Object(object)) => Ok(object),
        Ok(other) => Err(refuse(format!(
            "holds a JSON {}, not an object",
            json_kind(&other)
        ))),
        Err(error) => Err(refuse(format!("is not valid JSON ({error})"))),
    }
}

const fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Atomic publish of `value`; the caller holds [`json_state_lock_path`].
fn publish_json_state(path: &Path, value: &serde_json::Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("failed to serialize service state")?;
    write_file_atomically(path, &bytes)
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Whether inference has been confirmed for a service, from its engine state
/// file — probing at most once, and at most once per
/// [`INFERENCE_PROBE_RETRY_INTERVAL`] while it is still loading.
///
/// Shared by the engine adapters so the latch and backoff bookkeeping has one
/// implementation: the engines differ in how they decide a model is *listed*,
/// but not in what confirming inference means.
///
/// The attempt is recorded before the probe runs, so a caller killed mid-probe
/// still leaves the throttle in place instead of freeing the next poll to spend
/// another full timeout.
pub fn engine_state_inference_verified(
    state_path: &Path,
    state: Option<&serde_json::Value>,
    endpoint_url: &str,
    model_ref: &str,
    endpoint_api_key: Option<&str>,
) -> bool {
    let state_u64 = |key: &str| {
        state
            .and_then(|value| value.get(key))
            .and_then(serde_json::Value::as_u64)
    };
    if state_u64(INFERENCE_VERIFIED_STATE_KEY).is_some() {
        return true;
    }
    if model_ref.trim().is_empty() {
        return false;
    }
    let now = unix_time_millis() as u64;
    if let Some(attempted_at) = state_u64(INFERENCE_PROBE_ATTEMPTED_STATE_KEY)
        && now.saturating_sub(attempted_at) < INFERENCE_PROBE_RETRY_INTERVAL.as_millis() as u64
    {
        return false;
    }
    let _ = merge_json_state_file(
        state_path,
        &serde_json::json!({ INFERENCE_PROBE_ATTEMPTED_STATE_KEY: now }),
    );
    if !openai_chat_completion_probe(
        endpoint_url,
        model_ref,
        endpoint_api_key,
        INFERENCE_PROBE_TIMEOUT,
    )
    .unwrap_or(false)
    {
        return false;
    }
    let _ = merge_json_state_file(
        state_path,
        &serde_json::json!({ INFERENCE_VERIFIED_STATE_KEY: unix_time_millis() as u64 }),
    );
    true
}

/// Ask the endpoint for a single token and report whether it answered.
///
/// This is the readiness signal that `/v1/models` cannot give: an engine lists a
/// model as soon as it accepts the name, which can be minutes before the weights
/// are resident and the first chat request stops hanging.
///
/// "Answered" means a complete HTTP response with a status below 500, not a
/// successful generation. A `4xx` still proves the inference path is up and the
/// model is loaded — the request was understood and refused on its merits — while
/// the failure this guards against is a hang, a dropped connection, or the `5xx`
/// an engine returns while it is still warming up. Insisting on `200` with
/// non-empty content would also wrongly fail a reasoning model, which can spend
/// its whole (tiny) token budget before emitting any content.
///
/// The rule does mean a `404` reads as serving. That is harmless for the engines
/// shipped today — both implement `/v1/chat/completions`, and a wrong key fails
/// the model listing that gates this call — but an engine that does not expose an
/// OpenAI-shaped chat route would need a different signal rather than this one.
pub fn openai_chat_completion_probe(
    endpoint_url: &str,
    model_ref: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<bool> {
    let status = openai_chat_completion_status(endpoint_url, model_ref, endpoint_api_key, timeout)?;
    Ok(status < 500)
}

/// Send the smallest possible chat request and return the HTTP status.
///
/// Callers pick their own bar. Readiness ([`openai_chat_completion_probe`]) only
/// needs to know the inference path answers at all, while a post-load smoke test
/// wants a real `200` — there, a `4xx` means the model that came up is not the
/// one that was asked for, which is a failure worth surfacing rather than
/// tolerating.
pub fn openai_chat_completion_status(
    endpoint_url: &str,
    model_ref: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<u16> {
    let body = serde_json::json!({
        "model": model_ref,
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": 2,
        "stream": false,
    });
    let response = http_post_json_with_auth(
        endpoint_url,
        "/v1/chat/completions",
        &body,
        endpoint_api_key,
        timeout,
    )?;
    Ok(response.status)
}

pub fn openai_models_endpoint_has_model(
    endpoint_url: &str,
    expected_model: Option<&str>,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<bool> {
    let body = http_get_text_with_auth(endpoint_url, "/v1/models", endpoint_api_key, timeout)?;
    let value = serde_json::from_str::<serde_json::Value>(body.trim())
        .context("failed to parse /v1/models JSON")?;
    let loaded_models = openai_loaded_model_ids(&value);
    if loaded_models.is_empty() {
        return Ok(false);
    }
    let Some(expected_model) = expected_model.filter(|value| !value.trim().is_empty()) else {
        return Ok(true);
    };
    Ok(loaded_models
        .iter()
        .any(|loaded| model_refs_match(loaded, expected_model)))
}

pub fn managed_service_endpoint_model_ready(
    record: &ManagedServiceRecord,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<bool> {
    if record.endpoint_url.trim().is_empty() {
        return Ok(false);
    }
    let expected = Some(record.served_model_name()).filter(|name| !name.trim().is_empty());
    openai_models_endpoint_has_model(&record.endpoint_url, expected, endpoint_api_key, timeout)
}

/// How far along a managed service's endpoint is.
///
/// The middle state is the one that matters: an engine lists a model within
/// seconds of accepting its name, while the weights can take minutes to become
/// usable. Callers must not treat `Listing` as ready — nor as dead, since the
/// service is coming up normally and restarting it would start the wait over.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EndpointReadiness {
    /// Not answering at all: wrong port, process gone, or nothing bound yet.
    Unreachable,
    /// Answering and advertising the model, but inference has not come back.
    Listing,
    /// A real inference request has succeeded.
    Serving,
}

/// The result of a readiness check, plus whether it left the record dirty.
#[derive(Debug, Clone, Copy)]
pub struct EndpointReadinessOutcome {
    pub readiness: EndpointReadiness,
    /// The check updated the record's probe bookkeeping. Persist it with
    /// [`ManagedServiceRecord::write`] — the throttle in
    /// [`managed_service_endpoint_readiness`] only works if the attempt survives
    /// the process, since each CLI invocation starts fresh.
    pub record_changed: bool,
}

/// How far along the service's endpoint is, probing inference at most once.
///
/// Stronger than [`managed_service_endpoint_model_ready`], which only asks
/// whether the endpoint lists the model. A service reaches [`Serving`] once a
/// real inference request has come back, and that verdict is **latched** into
/// `record.inference_verified_at_unix_ms`: readiness is polled repeatedly (by
/// `services list`, the dash, and the supervisor), and re-probing on every poll
/// would queue a generation request behind the user's own traffic. The trade-off
/// is that a service which degrades after start still reports ready — the same
/// as before this check existed.
///
/// A *failed* probe cannot latch, so those are throttled instead: a still-loading
/// service is re-probed at most once per [`INFERENCE_PROBE_RETRY_INTERVAL`],
/// which keeps a warming model from costing every caller a full
/// `probe_timeout`.
///
/// Mutates `record` when it probes; persist it when `record_changed` is set.
///
/// [`Serving`]: EndpointReadiness::Serving
pub fn managed_service_endpoint_readiness(
    record: &mut ManagedServiceRecord,
    endpoint_api_key: Option<&str>,
    listing_timeout: Duration,
    probe_timeout: Duration,
) -> EndpointReadinessOutcome {
    let outcome = |readiness, record_changed| EndpointReadinessOutcome {
        readiness,
        record_changed,
    };
    let listed = managed_service_endpoint_model_ready(record, endpoint_api_key, listing_timeout)
        .unwrap_or(false);
    if !listed {
        return outcome(EndpointReadiness::Unreachable, false);
    }
    if record.inference_verified_at_unix_ms.is_some() {
        return outcome(EndpointReadiness::Serving, false);
    }
    let now = unix_time_millis() as u64;
    if let Some(attempted_at) = record.inference_probe_attempted_at_unix_ms
        && now.saturating_sub(attempted_at) < INFERENCE_PROBE_RETRY_INTERVAL.as_millis() as u64
    {
        return outcome(EndpointReadiness::Listing, false);
    }
    let model_ref = record.served_model_name().to_owned();
    record.inference_probe_attempted_at_unix_ms = Some(now);
    if !openai_chat_completion_probe(
        &record.endpoint_url,
        &model_ref,
        endpoint_api_key,
        probe_timeout,
    )
    .unwrap_or(false)
    {
        return outcome(EndpointReadiness::Listing, true);
    }
    record.inference_verified_at_unix_ms = Some(unix_time_millis() as u64);
    outcome(EndpointReadiness::Serving, true)
}

fn openai_loaded_model_ids(value: &serde_json::Value) -> Vec<String> {
    value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            ["id", "model", "name"]
                .into_iter()
                .filter_map(|field| item.get(field).and_then(serde_json::Value::as_str))
                .find(|value| !value.trim().is_empty())
                .map(str::to_owned)
        })
        .collect()
}

fn model_refs_match(left: &str, right: &str) -> bool {
    let left = left.trim();
    let right = right.trim();
    if left.eq_ignore_ascii_case(right) || model_ref_basename(left).eq_ignore_ascii_case(right) {
        return true;
    }
    if model_ref_basename(right).eq_ignore_ascii_case(left)
        || model_ref_basename(left).eq_ignore_ascii_case(model_ref_basename(right))
    {
        return true;
    }
    builtin_model_recipes().into_iter().any(|recipe| {
        (recipe.matches_ref(left) || recipe.matches_ref(right))
            && (recipe.matches_ref(left) && recipe.matches_ref(right))
    }) || model_ref_family_matches(left, right)
        || model_ref_family_matches(right, left)
}

fn model_ref_basename(value: &str) -> &str {
    value
        .trim()
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_else(|| value.trim())
}

fn model_ref_family_matches(reported: &str, expected_family: &str) -> bool {
    let expected = normalize_model_ref_family(expected_family);
    if expected.len() < 3 || expected.chars().any(|ch| ch.is_ascii_digit()) {
        return false;
    }
    model_ref_tokens(reported)
        .into_iter()
        .any(|token| token == expected || token.starts_with(&expected))
}

fn normalize_model_ref_family(value: &str) -> String {
    value
        .trim()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect()
}

fn model_ref_tokens(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|token| {
            let token = normalize_model_ref_family(token);
            (!token.is_empty()).then_some(token)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::probe_test_record;
    use std::io::{Read, Write};
    use std::time::Instant;

    #[test]
    fn openai_models_endpoint_has_model_checks_loaded_model_ids() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<String> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut request_bytes = Vec::new();
            let mut buffer = [0_u8; 512];
            loop {
                let read = stream.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                request_bytes.extend_from_slice(&buffer[..read]);
                if String::from_utf8_lossy(&request_bytes).contains("\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            let body = r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )?;
            Ok(request)
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        assert!(openai_models_endpoint_has_model(
            &endpoint,
            Some("qwen"),
            None,
            Duration::from_secs(2)
        )?);

        let request = server.join().expect("server thread should not panic")?;
        assert!(request.starts_with("GET /v1/models HTTP/1.1"));
        Ok(())
    }
    #[test]
    fn openai_models_endpoint_sends_bearer_when_key_present() -> Result<()> {
        // A protected endpoint: 200 with the model list only when the request
        // carries `Authorization: Bearer test-key`, otherwise 401. Serves two
        // connections — the no-key probe, then the with-key probe.
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut request_bytes = Vec::new();
                let mut buffer = [0_u8; 512];
                loop {
                    let read = stream.read(&mut buffer)?;
                    if read == 0 {
                        break;
                    }
                    request_bytes.extend_from_slice(&buffer[..read]);
                    if String::from_utf8_lossy(&request_bytes).contains("\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&request_bytes).into_owned();
                if request.contains("Authorization: Bearer test-key") {
                    let body = r#"{"data":[{"id":"qwen"}]}"#;
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )?;
                } else {
                    write!(
                        stream,
                        "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )?;
                }
            }
            Ok(())
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        // Without the key the protected endpoint 401s → treated as not ready.
        assert!(
            openai_models_endpoint_has_model(&endpoint, Some("qwen"), None, Duration::from_secs(2))
                .is_err()
        );
        // With the key the bearer header is sent → the model reads as ready.
        assert!(openai_models_endpoint_has_model(
            &endpoint,
            Some("qwen"),
            Some("test-key"),
            Duration::from_secs(2)
        )?);

        server.join().expect("server thread should not panic")?;
        Ok(())
    }
    // Why readiness is gated on an inference probe (EAI-7333): a server that
    // lists the model on `/v1/models` but cannot yet serve
    // `/v1/chat/completions` still reports the model as present. This test pins
    // that `openai_models_endpoint_has_model` alone is a false positive for
    // inference-readiness, which is why callers must additionally probe
    // inference — see `openai_chat_completion_probe` and
    // `managed_service_endpoint_readiness`.
    #[test]
    fn models_endpoint_readiness_does_not_imply_inference_ready() -> Result<()> {
        // A server that answers `/v1/models` with the model listed, but would
        // hang/refuse an actual chat request (it only ever serves this one
        // response, then closes) — mirroring an engine whose model is still
        // loading while `/v1/models` already responds.
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buffer = [0_u8; 512];
            let _ = stream.read(&mut buffer)?;
            let body = r#"{"data":[{"id":"Qwen/Qwen2.5-1.5B-Instruct"}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )?;
            Ok(())
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        // The `/v1/models` probe reports the model present — this is the exact
        // signal the healthcheck uses to declare "ready".
        let models_ready = openai_models_endpoint_has_model(
            &endpoint,
            Some("Qwen/Qwen2.5-1.5B-Instruct"),
            None,
            Duration::from_secs(2),
        )?;
        assert!(
            models_ready,
            "/v1/models lists the model, so the current healthcheck would report ready"
        );

        // But that says nothing about inference: the server served only the
        // models response and closed, so a chat request would not succeed.
        // Readiness based on this signal alone is a false positive (EAI-7333).
        server.join().expect("server thread should not panic")?;
        Ok(())
    }
    /// Serve `count` canned HTTP responses on a loopback port, returning the port
    /// and a handle yielding the requests that were received. Each response is
    /// `(status_line, body)`; a `None` response accepts the connection and never
    /// answers, standing in for an engine that hangs.
    fn spawn_canned_http_server(
        responses: Vec<Option<(&'static str, &'static str)>>,
    ) -> Result<(u16, std::thread::JoinHandle<Result<Vec<String>>>)> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let handle = std::thread::spawn(move || -> Result<Vec<String>> {
            let mut requests = Vec::new();
            for response in responses {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut request_bytes = Vec::new();
                let mut buffer = [0_u8; 512];
                while let Ok(read) = stream.read(&mut buffer) {
                    if read == 0 {
                        break;
                    }
                    request_bytes.extend_from_slice(&buffer[..read]);
                    let text = String::from_utf8_lossy(&request_bytes);
                    // Requests with a body (POST) are complete once the declared
                    // content length has arrived after the header terminator.
                    if let Some((headers, body)) = text.split_once("\r\n\r\n") {
                        let declared = headers
                            .lines()
                            .find_map(|line| {
                                line.strip_prefix("Content-Length: ")?.trim().parse().ok()
                            })
                            .unwrap_or(0_usize);
                        if body.len() >= declared {
                            break;
                        }
                    }
                }
                requests.push(String::from_utf8_lossy(&request_bytes).into_owned());
                let Some((status_line, body)) = response else {
                    // Hang: hold the connection open without answering until the
                    // client's read timeout fires, then drop it.
                    std::thread::sleep(Duration::from_millis(1500));
                    continue;
                };
                write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
            }
            Ok(requests)
        });
        Ok((port, handle))
    }
    const CHAT_OK_BODY: &str = r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#;
    const MODELS_OK_BODY: &str = r#"{"data":[{"id":"Qwen/Qwen3-0.6B"}]}"#;
    #[test]
    fn chat_completion_probe_passes_when_the_endpoint_answers() -> Result<()> {
        let (port, server) =
            spawn_canned_http_server(vec![Some(("HTTP/1.1 200 OK", CHAT_OK_BODY))])?;
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        assert!(openai_chat_completion_probe(
            &endpoint,
            "Qwen/Qwen3-0.6B",
            None,
            Duration::from_secs(2)
        )?);

        let requests = server.join().expect("server thread should not panic")?;
        let request = requests.first().expect("the probe sends one request");
        assert!(
            request.starts_with("POST /v1/chat/completions HTTP/1.1"),
            "probe must exercise the inference path, got: {request}"
        );
        assert!(
            request.contains("\"model\":\"Qwen/Qwen3-0.6B\"") && request.contains("\"max_tokens\""),
            "probe asks the served model for a token-capped completion, got: {request}"
        );
        Ok(())
    }
    #[test]
    fn chat_completion_probe_separates_a_refusal_from_a_warmup_failure() -> Result<()> {
        // A refusal proves the inference path is up and the model is resident:
        // the request was understood and rejected on its merits. A 5xx is what an
        // engine returns while it is still warming up, which is not ready.
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 400 Bad Request", r#"{"error":"unsupported"}"#)),
            Some((
                "HTTP/1.1 503 Service Unavailable",
                r#"{"error":"loading model"}"#,
            )),
        ])?;
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        assert!(openai_chat_completion_probe(
            &endpoint,
            "qwen",
            None,
            Duration::from_secs(2)
        )?);
        assert!(!openai_chat_completion_probe(
            &endpoint,
            "qwen",
            None,
            Duration::from_secs(2)
        )?);

        server.join().expect("server thread should not panic")?;
        Ok(())
    }
    #[test]
    fn chat_completion_probe_accepts_a_response_from_a_server_that_holds_the_socket() -> Result<()>
    {
        // `Connection: close` is a request, not a guarantee — a server or an
        // intervening proxy may answer in full and keep the socket open. Reading
        // to EOF would stall until the timeout and throw the answer away, leaving
        // a perfectly healthy service stuck reporting "not ready".
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            let body = CHAT_OK_BODY;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )?;
            stream.flush()?;
            // Hold the connection open past the probe's timeout.
            std::thread::sleep(Duration::from_secs(3));
            Ok(())
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        let started = Instant::now();
        assert!(
            openai_chat_completion_probe(&endpoint, "qwen", None, Duration::from_secs(2))?,
            "a complete response counts even when the peer keeps the socket open"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the framed response is complete, so the probe must not wait for EOF"
        );

        server.join().expect("server thread should not panic")?;
        Ok(())
    }
    #[test]
    fn http_read_is_bounded_across_reads_not_just_per_read() -> Result<()> {
        // A socket read timeout bounds each `read`, not the sequence of them. A
        // server that dribbles bytes forever, each within the per-read timeout,
        // must still hit the caller's overall budget.
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            // Never declares a length and never finishes: one byte at a time,
            // comfortably inside any per-read timeout.
            for _ in 0..200 {
                if stream.write_all(b"x").is_err() || stream.flush().is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Ok(())
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        let started = Instant::now();
        assert!(
            openai_chat_completion_probe(&endpoint, "qwen", None, Duration::from_millis(500))
                .is_err(),
            "a response that never completes is not a passing probe"
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "the call must honor its own budget, not a multiple of it: took {:?}",
            started.elapsed()
        );

        let _ = server.join();
        Ok(())
    }
    /// A signal arriving mid-response must not fail the request.
    ///
    /// A signal delivered while the client is parked in `read` aborts it with
    /// `EINTR`. `SA_RESTART` does not save us: Linux never restarts a socket read
    /// that has a receive timeout set, and this client sets one on every pass.
    /// Any handler in the process is enough to trigger it — `crossterm`'s
    /// `SIGWINCH` hook is linked into the CLI — so treating `EINTR` as a
    /// transport error turned an unrelated signal into a spurious "endpoint
    /// unreachable".
    ///
    /// Linux-only on purpose. The guarantee being exercised — a receive timeout
    /// defeats `SA_RESTART` — is documented for Linux; BSD-derived kernels may
    /// restart the read instead, which would leave this passing without ever
    /// reaching the retry. CI has no macOS runner to tell the difference.
    #[cfg(target_os = "linux")]
    #[test]
    fn http_read_survives_a_signal_arriving_mid_response() -> Result<()> {
        extern "C" fn noop_handler(_signal: libc::c_int) {}

        let body = r#"{"data":[{"id":"qwen"}]}"#;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            // Answer late, so the client is blocked in `read` while the signals
            // land rather than racing them.
            std::thread::sleep(Duration::from_millis(400));
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )?;
            stream.flush()?;
            Ok(())
        });

        // Install the handler with SA_RESTART set, to show that the flag is not
        // what protects this read.
        //
        // The handler is left installed rather than restored: the default
        // disposition for SIGUSR1 is to kill the process, so putting it back
        // would let a signal still in flight take the whole test binary down.
        // Leaving a no-op handler is inert — nothing else here raises SIGUSR1,
        // and the signals below are aimed at this thread alone, so no other
        // test sharing this process can observe either one.
        #[allow(unsafe_code)] // libc FFI
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = noop_handler as *const () as libc::sighandler_t;
            action.sa_flags = libc::SA_RESTART;
            libc::sigemptyset(&raw mut action.sa_mask);
            assert_eq!(
                libc::sigaction(libc::SIGUSR1, &raw const action, std::ptr::null_mut()),
                0,
                "failed to install the SIGUSR1 handler"
            );
        }

        // Target this thread specifically: a process-directed signal could land
        // on any thread and disturb an unrelated test sharing this process.
        #[allow(unsafe_code)] // libc FFI
        let reader = unsafe { libc::pthread_self() };
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let signaller = {
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                // Let the connect and the request write finish first; only the
                // response read is under test.
                std::thread::sleep(Duration::from_millis(50));
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    #[allow(unsafe_code)] // libc FFI
                    unsafe {
                        libc::pthread_kill(reader, libc::SIGUSR1);
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            })
        };

        let endpoint = format!("http://127.0.0.1:{port}/v1");
        let response =
            http_get_text_with_auth(&endpoint, "/v1/models", None, Duration::from_secs(5));

        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        signaller.join().expect("signaller thread should not panic");
        server.join().expect("server thread should not panic")?;

        assert_eq!(
            response?, body,
            "an interrupted read is retryable, not a failed request"
        );
        Ok(())
    }
    #[test]
    fn chat_completion_probe_fails_on_a_hung_endpoint() -> Result<()> {
        // The reported symptom: the endpoint accepts the connection and never
        // answers. The probe must give up within its timeout, not wait forever.
        let (port, server) = spawn_canned_http_server(vec![None])?;
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        let started = Instant::now();
        assert!(
            openai_chat_completion_probe(&endpoint, "qwen", None, Duration::from_millis(300))
                .is_err(),
            "a hung endpoint is not inference-ready"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the probe must be bounded by its timeout"
        );

        server.join().expect("server thread should not panic")?;
        Ok(())
    }
    #[test]
    fn inference_readiness_latches_after_the_first_successful_probe() -> Result<()> {
        // First check: list the model, then probe inference. Second check: the
        // verdict is latched, so only the cheap listing is re-issued — repeated
        // `services list` polls must not queue generation work behind real
        // traffic.
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
            Some(("HTTP/1.1 200 OK", CHAT_OK_BODY)),
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
        ])?;
        let mut record = probe_test_record(port);

        assert_eq!(
            managed_service_endpoint_readiness(
                &mut record,
                None,
                Duration::from_secs(2),
                Duration::from_secs(2)
            )
            .readiness,
            EndpointReadiness::Serving
        );
        assert!(
            record.inference_verified_at_unix_ms.is_some(),
            "a passing probe is recorded so later checks can skip it"
        );

        assert_eq!(
            managed_service_endpoint_readiness(
                &mut record,
                None,
                Duration::from_secs(2),
                Duration::from_secs(2)
            )
            .readiness,
            EndpointReadiness::Serving
        );

        let requests = server.join().expect("server thread should not panic")?;
        let paths: Vec<&str> = requests
            .iter()
            .filter_map(|request| request.lines().next())
            .collect();
        assert_eq!(
            paths,
            vec![
                "GET /v1/models HTTP/1.1",
                "POST /v1/chat/completions HTTP/1.1",
                "GET /v1/models HTTP/1.1",
            ],
            "the second readiness check must not re-probe inference"
        );
        Ok(())
    }
    #[test]
    fn a_warming_service_is_not_re_probed_on_every_poll() -> Result<()> {
        // Only a successful probe latches, so a model that is listed but still
        // loading would otherwise be re-probed by every poll — and each attempt
        // costs the full probe timeout, paid by `services list` and the dash in
        // front of a user. The second check must cost a listing and nothing more.
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
            Some((
                "HTTP/1.1 503 Service Unavailable",
                r#"{"error":"loading model"}"#,
            )),
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
        ])?;
        let mut record = probe_test_record(port);

        let first = managed_service_endpoint_readiness(
            &mut record,
            None,
            Duration::from_secs(2),
            Duration::from_secs(2),
        );
        assert_eq!(first.readiness, EndpointReadiness::Listing);
        assert!(
            first.record_changed && record.inference_probe_attempted_at_unix_ms.is_some(),
            "the attempt must be recorded, and persisted by the caller — each CLI \
             run is a fresh process, so an unwritten attempt throttles nothing"
        );

        let second = managed_service_endpoint_readiness(
            &mut record,
            None,
            Duration::from_secs(2),
            Duration::from_secs(2),
        );
        assert_eq!(second.readiness, EndpointReadiness::Listing);
        assert!(!second.record_changed);

        let requests = server.join().expect("server thread should not panic")?;
        let paths: Vec<&str> = requests
            .iter()
            .filter_map(|request| request.lines().next())
            .collect();
        assert_eq!(
            paths,
            vec![
                "GET /v1/models HTTP/1.1",
                "POST /v1/chat/completions HTTP/1.1",
                "GET /v1/models HTTP/1.1",
            ],
            "the second check must not re-probe inside the retry interval"
        );
        Ok(())
    }
    #[test]
    fn inference_readiness_is_withheld_while_the_model_only_lists() -> Result<()> {
        // The bug: `/v1/models` answers within seconds while the model loads for
        // minutes and inference returns nothing. That service is not ready.
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
            Some((
                "HTTP/1.1 503 Service Unavailable",
                r#"{"error":"loading model"}"#,
            )),
        ])?;
        let mut record = probe_test_record(port);

        assert_eq!(
            managed_service_endpoint_readiness(
                &mut record,
                None,
                Duration::from_secs(2),
                Duration::from_secs(2)
            )
            .readiness,
            EndpointReadiness::Listing,
            "a listed-but-unservable model is coming up, not dead"
        );
        assert!(
            record.inference_verified_at_unix_ms.is_none(),
            "nothing is latched until inference actually succeeds"
        );

        server.join().expect("server thread should not panic")?;
        Ok(())
    }
    #[test]
    fn inference_probe_sends_the_service_key_to_a_protected_endpoint() -> Result<()> {
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
            Some(("HTTP/1.1 200 OK", CHAT_OK_BODY)),
        ])?;
        let mut record = probe_test_record(port);

        assert_eq!(
            managed_service_endpoint_readiness(
                &mut record,
                Some("test-key"),
                Duration::from_secs(2),
                Duration::from_secs(2)
            )
            .readiness,
            EndpointReadiness::Serving
        );

        let requests = server.join().expect("server thread should not panic")?;
        assert!(
            requests
                .iter()
                .all(|request| request.contains("Authorization: Bearer test-key")),
            "a protected service must not read as unready for want of its own key"
        );
        Ok(())
    }

    // The locked, atomic state-file writers. One test per state the file can
    // be found in when a merge starts: absent, empty, torn, valid but not an
    // object, unreadable, a valid object; plus a patch with nothing to merge.

    fn json_state_dir(tag: &str) -> std::path::PathBuf {
        let dir = crate::test_support::workspace_test_artifact_dir().join(format!(
            "json-state-{tag}-{}",
            crate::test_support::unique_suffix()
        ));
        fs::create_dir_all(&dir).expect("create the test directory");
        dir
    }

    /// The merge refused `path`, said so naming the file and saying it was left
    /// untouched, and the file does still hold exactly `before` — the message
    /// and the state it claims, checked together.
    fn assert_refused_and_untouched(result: Result<()>, path: &Path, before: &[u8], why: &str) {
        let error = format!(
            "{:#}",
            result.expect_err("an unusable state file must be refused")
        );
        assert!(
            error.contains(&path.display().to_string()),
            "the error names the file: {error}"
        );
        assert!(
            error.contains("left it untouched and recorded nothing"),
            "the error says the file was left as it was: {error}"
        );
        assert!(error.contains(why), "the error says why ({why}): {error}");
        assert_eq!(
            fs::read(path).expect("the refused file is still there"),
            before,
            "a refused merge must leave the file byte for byte as it was"
        );
    }

    #[test]
    fn merging_into_an_absent_state_file_creates_it_with_the_patch() -> Result<()> {
        let dir = json_state_dir("absent");
        let path = dir.join("not-yet").join("svc.json");

        merge_json_state_file(
            &path,
            &serde_json::json!({ "status": "starting", "pid": 7 }),
        )?;

        let written: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(
            written,
            serde_json::json!({ "status": "starting", "pid": 7 })
        );
        Ok(())
    }

    #[test]
    fn merging_into_an_empty_state_file_is_refused_and_leaves_it_untouched() {
        let dir = json_state_dir("empty");
        let path = dir.join("svc.json");
        fs::write(&path, b"").unwrap();

        let result = merge_json_state_file(&path, &serde_json::json!({ "status": "ready" }));

        assert_refused_and_untouched(result, &path, b"", "is empty");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The case behind the lost `server_pid`: a writer caught another
    /// mid-write, read half a document, and replaced it with its own patch.
    #[test]
    fn merging_into_a_torn_state_file_is_refused_and_leaves_it_untouched() {
        let dir = json_state_dir("torn");
        let path = dir.join("svc.json");
        let torn = br#"{"status": "running", "server_pid": 42"#;
        fs::write(&path, torn).unwrap();

        let result = merge_json_state_file(
            &path,
            &serde_json::json!({ INFERENCE_PROBE_ATTEMPTED_STATE_KEY: 1 }),
        );

        assert_refused_and_untouched(result, &path, torn, "is not valid JSON");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn merging_into_a_state_file_that_is_not_an_object_is_refused_and_leaves_it_untouched() {
        let dir = json_state_dir("non-object");
        let path = dir.join("svc.json");
        for (document, kind) in [
            (&b"[1, 2]"[..], "holds a JSON array, not an object"),
            (&b"null"[..], "holds a JSON null, not an object"),
            (&b"\"x\""[..], "holds a JSON string, not an object"),
        ] {
            fs::write(&path, document).unwrap();
            let result = merge_json_state_file(&path, &serde_json::json!({ "status": "ready" }));
            assert_refused_and_untouched(result, &path, document, kind);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_state_file_that_cannot_be_read_is_refused_and_left_in_place() {
        let dir = json_state_dir("unreadable");
        // A directory where the file should be: reading it fails with an
        // error other than "not found" on every platform.
        let path = dir.join("svc.json");
        fs::create_dir_all(&path).unwrap();

        let error = format!(
            "{:#}",
            merge_json_state_file(&path, &serde_json::json!({ "status": "ready" }))
                .expect_err("an unreadable state file must be refused")
        );

        let still_a_directory = path.is_dir();
        let _ = fs::remove_dir_all(&dir);
        assert!(
            error.contains(&format!("failed to read {}", path.display())),
            "{error}"
        );
        assert!(
            error.contains("left it untouched and recorded nothing"),
            "{error}"
        );
        assert!(still_a_directory, "the refused path must be left in place");
    }

    #[test]
    fn merging_into_a_state_object_keeps_its_other_keys() -> Result<()> {
        let dir = json_state_dir("merge");
        let path = dir.join("svc.json");
        fs::write(
            &path,
            br#"{"status": "running", "server_pid": 42, "server_start_ticks": 9}"#,
        )?;

        merge_json_state_file(
            &path,
            &serde_json::json!({ "status": "ready", INFERENCE_VERIFIED_STATE_KEY: 5 }),
        )?;

        let written: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(
            written,
            serde_json::json!({
                "status": "ready",
                "server_pid": 42,
                "server_start_ticks": 9,
                INFERENCE_VERIFIED_STATE_KEY: 5,
            })
        );
        Ok(())
    }

    #[test]
    fn a_patch_that_is_not_an_object_changes_nothing() -> Result<()> {
        let dir = json_state_dir("non-object-patch");
        let path = dir.join("svc.json");
        let before = br#"{"server_pid": 42}"#;
        fs::write(&path, before)?;
        let absent = dir.join("absent.json");

        merge_json_state_file(&path, &serde_json::json!(["status"]))?;
        merge_json_state_file(&absent, &serde_json::Value::Null)?;

        let after = fs::read(&path)?;
        let created = absent.exists();
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(after, before, "nothing to merge, so nothing is written");
        assert!(!created, "nothing to merge, so no file is created");
        Ok(())
    }

    #[test]
    fn writing_a_state_file_replaces_it_whole() -> Result<()> {
        let dir = json_state_dir("write");
        let path = dir.join("nested").join("svc.json");

        write_json_state_file(
            &path,
            &serde_json::json!({ "status": "starting", "pid": 1 }),
        )?;
        let first: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        write_json_state_file(&path, &serde_json::json!({ "status": "running" }))?;
        let second: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;

        let _ = fs::remove_dir_all(&dir);
        assert_eq!(first, serde_json::json!({ "status": "starting", "pid": 1 }));
        assert_eq!(second, serde_json::json!({ "status": "running" }));
        Ok(())
    }

    /// The writers leave behind the state file and the directory's one lock
    /// file, and nothing else: no temp file, no per-file lock.
    #[test]
    fn state_writes_leave_only_the_state_file_and_the_directory_lock() -> Result<()> {
        let dir = json_state_dir("leftovers");
        let path = dir.join("svc.json");

        write_json_state_file(&path, &serde_json::json!({ "status": "starting" }))?;
        merge_json_state_file(&path, &serde_json::json!({ "status": "ready" }))?;
        let _ = merge_json_state_file(&dir.join("other.json"), &serde_json::json!({ "a": 1 }));

        let mut names: Vec<String> = fs::read_dir(&dir)?
            .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
            .collect::<std::io::Result<_>>()?;
        names.sort();
        let lock = json_state_lock_path(&path);
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(names, [JSON_STATE_LOCK_FILE_NAME, "other.json", "svc.json"]);
        assert_eq!(lock, dir.join(JSON_STATE_LOCK_FILE_NAME));
        assert_eq!(
            lock.extension().and_then(|value| value.to_str()),
            Some("lock"),
            "scans for `*.json` state files must not pick the lock up"
        );
        Ok(())
    }

    /// A merge publishes through `atomic_write`, which reserves its own temp
    /// file: another writer's file sitting on the name the merge would try
    /// first is neither written through nor moved.
    #[test]
    fn a_merge_spares_another_writers_staged_file() -> Result<()> {
        let dir = json_state_dir("foreign-temp");
        let path = dir.join("svc.json");
        crate::foreign_temp_probe::assert_write_spares_foreign_temp_files(
            |millis| dir.join(format!("svc.json.tmp-{}-{millis}-0", std::process::id())),
            || {
                fs::write(&path, br#"{"server_pid": 42}"#)?;
                merge_json_state_file(&path, &serde_json::json!({ "status": "ready" }))
            },
        )?;
        let written: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(
            written,
            serde_json::json!({ "server_pid": 42, "status": "ready" })
        );
        Ok(())
    }

    const MERGE_CHILD_PATH_ENV: &str = "ROCM_CORE_TEST_JSON_STATE_MERGE_CHILD_PATH";
    const MERGE_CHILD_WRITER_ENV: &str = "ROCM_CORE_TEST_JSON_STATE_MERGE_CHILD_WRITER";
    const MERGES_PER_WRITER: usize = 40;

    fn merge_distinct_keys(path: &Path, writer: &str) -> Result<()> {
        for n in 0..MERGES_PER_WRITER {
            merge_json_state_file(path, &serde_json::json!({ format!("{writer}-{n}"): n }))?;
        }
        Ok(())
    }

    /// The regression for the lost `server_pid`: writers in this process and
    /// in two child processes (this test binary run again) each merge their
    /// own keys into one state file while a reader that takes no lock reads it
    /// over and over. Every key and the original `server_pid` must survive,
    /// and the reader must only ever see a whole object.
    #[test]
    fn concurrent_merges_across_threads_and_processes_keep_every_key() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};

        if let (Some(path), Some(writer)) = (
            std::env::var_os(MERGE_CHILD_PATH_ENV),
            std::env::var(MERGE_CHILD_WRITER_ENV).ok(),
        ) {
            return merge_distinct_keys(Path::new(&path), &writer);
        }

        let dir = json_state_dir("concurrent");
        let path = dir.join("svc.json");
        fs::write(&path, br#"{"server_pid": 4242, "server_start_ticks": 7}"#)?;

        let exe = std::env::current_exe().context("current test executable")?;
        let child_writers = ["process-a", "process-b"];
        let mut children = Vec::new();
        for writer in child_writers {
            children.push(
                std::process::Command::new(&exe)
                    .args([
                        "endpoint_readiness::tests::concurrent_merges_across_threads_and_processes_keep_every_key",
                        "--exact",
                        "--quiet",
                    ])
                    .env(MERGE_CHILD_PATH_ENV, &path)
                    .env(MERGE_CHILD_WRITER_ENV, writer)
                    .stdout(std::process::Stdio::null())
                    .spawn()
                    .context("spawn a merging child process")?,
            );
        }

        let done = AtomicBool::new(false);
        let thread_writers = ["thread-a", "thread-b", "thread-c", "thread-d"];
        let (thread_results, torn_reads) = std::thread::scope(|scope| {
            let reader = scope.spawn(|| {
                let mut torn = Vec::new();
                while !done.load(Ordering::Relaxed) {
                    // Windows may refuse a read while a rename is in flight; a
                    // refusal is not a torn document, so only what was read
                    // is judged.
                    let Ok(bytes) = fs::read(&path) else { continue };
                    let whole = serde_json::from_slice::<serde_json::Value>(&bytes)
                        .ok()
                        .is_some_and(|value| value.get("server_pid").is_some());
                    if !whole {
                        torn.push(String::from_utf8_lossy(&bytes).into_owned());
                    }
                }
                torn
            });
            // Every writer is started before any is joined, so they overlap.
            let shared_path = path.as_path();
            let mut writers = Vec::new();
            for writer in thread_writers {
                writers.push(scope.spawn(move || merge_distinct_keys(shared_path, writer)));
            }
            let results: Vec<Result<()>> = writers
                .into_iter()
                .map(|writer| writer.join().expect("writer thread"))
                .collect();
            for child in &mut children {
                let _ = child.wait();
            }
            done.store(true, Ordering::Relaxed);
            (results, reader.join().expect("reader thread"))
        });

        let statuses: Vec<_> = children
            .iter_mut()
            .map(|child| child.wait().map(|status| status.success()))
            .collect::<std::io::Result<_>>()?;
        let written: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        let _ = fs::remove_dir_all(&dir);

        for result in thread_results {
            result?;
        }
        assert_eq!(statuses, [true, true], "both merging children must succeed");
        assert!(
            torn_reads.is_empty(),
            "a lock-free reader saw {} partial documents, e.g. {:?}",
            torn_reads.len(),
            torn_reads.first()
        );
        assert_eq!(written["server_pid"], 4242, "{written}");
        assert_eq!(written["server_start_ticks"], 7, "{written}");
        let missing: Vec<String> = thread_writers
            .iter()
            .chain(child_writers.iter())
            .flat_map(|writer| (0..MERGES_PER_WRITER).map(move |n| format!("{writer}-{n}")))
            .filter(|key| written.get(key).is_none())
            .collect();
        assert!(
            missing.is_empty(),
            "{} merged keys were lost, e.g. {:?}",
            missing.len(),
            missing.first()
        );
        Ok(())
    }
}
