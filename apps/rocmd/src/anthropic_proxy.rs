// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Anthropic Messages front for a served model.
//!
//! The engines speak the OpenAI API only. This module gives a managed service an
//! Anthropic-compatible `POST /v1/messages` by translating each request to the
//! engine's `/v1/chat/completions` and the reply back. It runs inside the process
//! that already owns the engine (`rocm`'s hidden `__engine-serve-http` entry),
//! as a background thread, so it starts, stops and restarts with the service and
//! adds no process of its own to track.
//!
//! [`spawn_alongside_engine`] first waits for the engine to list its model, then
//! binds a port of its own and publishes that address in the engine state file
//! under the keys in `rocm_core::anthropic_endpoint`, which the service record
//! adopts. The proxy fronts the engine even when the engine has a `/v1/messages`
//! route of its own (vLLM, Lemonade): that route only accepts the served model's
//! name, and an Anthropic client sends a Claude one.
//!
//! The translation covers system prompts and text turns, sampling parameters,
//! stop sequences, usage and stop reasons, and tool use in both directions:
//! `tools` and `tool_choice` on the request, `tool_use` and `tool_result` blocks
//! in the conversation, and `tool_use` blocks with the `tool_use` stop reason in
//! the reply. A `stream: true` request is answered with Anthropic server-sent
//! events (`message_start` through `message_stop`) built from the engine's
//! streamed chat-completion chunks as they arrive, including streamed tool
//! calls (one `tool_use` block per call, its arguments as `input_json_delta`
//! fragments). Extended thinking, server tools and other non-text content
//! blocks (images) are refused with an Anthropic `invalid_request_error` that
//! says so, rather than being dropped silently. The engine reports token usage only at the end of a
//! stream, so `message_start` carries `input_tokens: 0` and the real figures
//! arrive in `message_delta`. Fields that only annotate a request (`metadata`)
//! are ignored.

use std::convert::Infallible;
use std::io::{BufRead, BufReader, Read};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rocm_core::{
    ANTHROPIC_ENDPOINT_MODE_STATE_KEY, ANTHROPIC_ENDPOINT_URL_STATE_KEY, AnthropicEndpointMode,
    JsonStateFileRefused, format_http_base_url, http_get_with_auth, http_post_json_with_auth,
    merge_json_state_file, openai_models_endpoint_has_model,
};
use serde_json::{Value, json};

/// How long to wait for the engine to list its model before giving up on
/// publishing an Anthropic endpoint. Model loads can take many minutes.
const ENGINE_LISTING_TIMEOUT: Duration = Duration::from_mins(30);

/// Budget for one proxied generation.
const UPSTREAM_REQUEST_TIMEOUT: Duration = Duration::from_mins(10);

/// What the proxy needs to know about the engine it fronts.
#[derive(Debug, Clone)]
pub struct AnthropicProxyConfig {
    /// Host the engine listens on, as given to `rocm serve`. The proxy binds the
    /// same host, so it is never reachable from further away than the engine.
    pub host: String,
    pub engine_port: u16,
    /// The model the engine was asked to serve. Every request is sent under the
    /// name the engine actually serves it as — `served_model_id` in the state
    /// file when the engine reported one, else this — whatever model the client
    /// asked for: an Anthropic client names a Claude model, no local engine's.
    pub served_model: String,
    /// The engine state file the endpoint is published to.
    pub state_path: PathBuf,
    /// The service's endpoint key, used only for the proxy's own probes. Client
    /// requests are never answered with it: a client brings its own.
    pub probe_api_key: Option<String>,
}

impl AnthropicProxyConfig {
    /// Where to reach the engine from this machine, even when it listens on a
    /// wildcard address.
    fn upstream_url(&self) -> String {
        let host = match self.host.as_str() {
            "0.0.0.0" => "127.0.0.1",
            "::" => "::1",
            other => other,
        };
        format_http_base_url(host, self.engine_port)
    }
}

/// Start publishing an Anthropic-compatible endpoint for the engine described by
/// `config`, on a background thread that lives as long as the process.
pub fn spawn_alongside_engine(config: AnthropicProxyConfig) {
    let _ = std::thread::Builder::new()
        .name("anthropic-endpoint".to_owned())
        .spawn(move || {
            if let Err(error) = run(&config) {
                eprintln!("anthropic endpoint unavailable: {error:#}");
            }
        });
}

fn run(config: &AnthropicProxyConfig) -> anyhow::Result<()> {
    let upstream = config.upstream_url();
    wait_for_engine_listing(config, &upstream)?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind((config.host.as_str(), 0)).await?;
        let address: SocketAddr = listener.local_addr()?;
        // Published only once the socket is bound, so a reader never sees a URL
        // nothing is listening on.
        publish(
            config,
            &format_http_base_url(&config.host, address.port()),
            AnthropicEndpointMode::Proxy,
        )?;
        axum::serve(listener, router(ProxyState::from_config(config))).await?;
        Ok(())
    })
}

fn wait_for_engine_listing(config: &AnthropicProxyConfig, upstream: &str) -> anyhow::Result<()> {
    let started = std::time::Instant::now();
    while started.elapsed() < ENGINE_LISTING_TIMEOUT {
        if openai_models_endpoint_has_model(
            upstream,
            // Any listed model counts: a container engine lists its own name,
            // not the reference it was asked to serve.
            None,
            config.probe_api_key.as_deref(),
            Duration::from_secs(3),
        )
        .unwrap_or(false)
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    anyhow::bail!("the engine did not list a model in time")
}

/// How many times [`publish`] tries to record the endpoint before it gives up.
const PUBLISH_ATTEMPTS: usize = 5;
/// The pause between two of [`publish`]'s attempts.
const PUBLISH_RETRY_PAUSE: Duration = Duration::from_millis(200);

/// Record the endpoint in the engine's state file.
///
/// `merge_json_state_file` refuses a state file it cannot merge into rather
/// than replace it. The vLLM and AIM engines still rewrite their state files
/// in place, so a file refused as empty or not valid JSON can be one caught
/// mid-write and whole again a moment later: that refusal is tried again, up
/// to [`PUBLISH_ATTEMPTS`] attempts [`PUBLISH_RETRY_PAUSE`] apart. Any other
/// failure (a whole document that is not an object, or failing to lock, read
/// or write the file) is returned at once.
fn publish(
    config: &AnthropicProxyConfig,
    url: &str,
    mode: AnthropicEndpointMode,
) -> anyhow::Result<()> {
    publish_with_pause(config, url, mode, std::thread::sleep)
}

/// [`publish`] with the pause between attempts supplied by the caller, so a
/// test can see each pause and change the file between attempts.
fn publish_with_pause(
    config: &AnthropicProxyConfig,
    url: &str,
    mode: AnthropicEndpointMode,
    mut pause: impl FnMut(Duration),
) -> anyhow::Result<()> {
    // The listener URL carries no `/v1`: Anthropic SDKs append `/v1/messages`.
    let base = url.trim_end_matches('/').trim_end_matches("/v1");
    let patch = json!({
        ANTHROPIC_ENDPOINT_URL_STATE_KEY: base,
        ANTHROPIC_ENDPOINT_MODE_STATE_KEY: mode.as_str(),
    });
    let mut attempt = 1;
    loop {
        let error = match merge_json_state_file(&config.state_path, &patch) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        let may_be_mid_write = error
            .downcast_ref::<JsonStateFileRefused>()
            .is_some_and(|refused| refused.refusal.may_be_mid_write());
        if may_be_mid_write && attempt < PUBLISH_ATTEMPTS {
            attempt += 1;
            pause(PUBLISH_RETRY_PAUSE);
            continue;
        }
        let attempts = if attempt == 1 {
            "1 attempt".to_owned()
        } else {
            format!("{attempt} attempts")
        };
        return Err(error.context(format!(
            "could not publish the Anthropic endpoint to {} after {attempts}",
            config.state_path.display()
        )));
    }
}

#[derive(Clone)]
struct ProxyState {
    upstream: String,
    served_model: String,
}

impl ProxyState {
    fn from_config(config: &AnthropicProxyConfig) -> Self {
        let reported = std::fs::read(&config.state_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .and_then(|state| {
                state
                    .get("served_model_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .filter(|name| !name.trim().is_empty());
        Self {
            upstream: config.upstream_url(),
            served_model: reported.unwrap_or_else(|| config.served_model.clone()),
        }
    }
}

fn router(state: ProxyState) -> Router {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/models", get(models))
        .with_state(state)
}

/// What went wrong, in the terms of the Anthropic error envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProxyError {
    status: u16,
    kind: &'static str,
    message: String,
}

impl ProxyError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    fn unsupported(what: &str) -> Self {
        Self::invalid(format!(
            "{what} is not supported by the rocm Anthropic-compatible endpoint yet"
        ))
    }

    /// An error the engine returned, relabelled with Anthropic's error types.
    fn from_upstream(status: u16, body: &str) -> Self {
        let message = serde_json::from_str::<Value>(body)
            .ok()
            .and_then(|value| {
                // OpenAI-style `{"error": {"message": ...}}` or `{"error": "..."}`,
                // and FastAPI-style `{"detail": "..."}`.
                let error = value.get("error").or_else(|| value.get("detail"))?;
                error
                    .get("message")
                    .or(Some(error))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| format!("the engine answered HTTP {status}"));
        let kind = match status {
            400 | 422 => "invalid_request_error",
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            _ => "api_error",
        };
        Self {
            status,
            kind,
            message,
        }
    }

    fn task_failed(error: &tokio::task::JoinError) -> Self {
        Self {
            status: 500,
            kind: "api_error",
            message: format!("the proxy task failed: {error}"),
        }
    }

    fn unreachable(error: &anyhow::Error) -> Self {
        let timed_out = error.chain().any(|cause| {
            cause.downcast_ref::<std::io::Error>().is_some_and(|io| {
                matches!(
                    io.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                )
            })
        });
        if timed_out {
            return Self {
                status: 504,
                kind: "api_error",
                message: format!("the engine did not answer in time: {error:#}"),
            };
        }
        Self {
            status: 502,
            kind: "api_error",
            message: format!("the engine could not be reached: {error:#}"),
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let body = json!({
            "type": "error",
            "error": { "type": self.kind, "message": self.message },
        });
        (
            StatusCode::from_u16(self.status).unwrap_or(StatusCode::BAD_GATEWAY),
            Json(body),
        )
            .into_response()
    }
}

/// The key a client presented, in either of the two places Anthropic clients put
/// it, as the bearer token the engine expects.
fn client_api_key(headers: &HeaderMap) -> Option<String> {
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    header("x-api-key")
        .or_else(|| {
            header("authorization").and_then(|value| match value.split_once(' ') {
                Some((scheme, token)) if scheme.eq_ignore_ascii_case("bearer") => Some(token),
                // Another scheme (`Basic ...`) carries no key the engine accepts.
                Some(_) => None,
                None => Some(value),
            })
        })
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_owned)
}

async fn messages(State(state): State<ProxyState>, headers: HeaderMap, body: Bytes) -> Response {
    let key = client_api_key(&headers);
    let upstream_request = match parse_request(&state, &body) {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    if upstream_request.get("stream").and_then(Value::as_bool) == Some(true) {
        return stream_messages(state, key, upstream_request).await;
    }
    let result = tokio::task::spawn_blocking(move || {
        forward_messages(&state, key.as_deref(), &upstream_request)
    })
    .await
    .unwrap_or_else(|error| Err(ProxyError::task_failed(&error)));
    match result {
        Ok(message) => Json(message).into_response(),
        Err(error) => error.into_response(),
    }
}

/// The chat-completions request an Anthropic request body translates to.
fn parse_request(state: &ProxyState, body: &[u8]) -> Result<Value, ProxyError> {
    let request: Value = serde_json::from_slice(body)
        .map_err(|error| ProxyError::invalid(format!("the request body is not JSON: {error}")))?;
    anthropic_to_openai(&request, &state.served_model)
}

fn forward_messages(
    state: &ProxyState,
    key: Option<&str>,
    upstream_request: &Value,
) -> Result<Value, ProxyError> {
    let response = http_post_json_with_auth(
        &state.upstream,
        "/v1/chat/completions",
        upstream_request,
        key,
        UPSTREAM_REQUEST_TIMEOUT,
    )
    .map_err(|error| ProxyError::unreachable(&error))?;
    if response.status != 200 {
        return Err(ProxyError::from_upstream(response.status, &response.body));
    }
    let reply: Value = serde_json::from_str(response.body.trim()).map_err(|error| ProxyError {
        status: 502,
        kind: "api_error",
        message: format!("the engine's reply was not JSON: {error}"),
    })?;
    openai_to_anthropic(&reply, &state.served_model)
}

/// Answer a streaming request.
///
/// The engine is contacted before any response byte is sent, so an engine that
/// refuses the request (a bad key, an unknown model) still gets a plain Anthropic
/// error with its real status. Once the engine has accepted it the answer is a
/// `text/event-stream`, fed by a blocking task that translates the engine's
/// chunks as they arrive; a failure after that point can only be reported as an
/// in-stream `error` event.
async fn stream_messages(state: ProxyState, key: Option<String>, request: Value) -> Response {
    let served_model = state.served_model.clone();
    let opened = tokio::task::spawn_blocking(move || {
        open_upstream_stream(&state.upstream, key.as_deref(), &request)
    })
    .await
    .unwrap_or_else(|error| Err(ProxyError::task_failed(&error)));
    let upstream = match opened {
        Ok(upstream) => upstream,
        Err(error) => return error.into_response(),
    };
    let (sender, mut receiver) = tokio::sync::mpsc::channel::<Bytes>(32);
    tokio::task::spawn_blocking(move || pump_stream(upstream, &served_model, &sender));
    let events = async_stream::stream! {
        while let Some(chunk) = receiver.recv().await {
            yield Ok::<Bytes, Infallible>(chunk);
        }
    };
    let mut response = Response::new(Body::from_stream(events));
    let headers = response.headers_mut();
    headers.insert("content-type", "text/event-stream".parse().expect("static"));
    headers.insert("cache-control", "no-cache".parse().expect("static"));
    response
}

/// POST `request` to the engine and hand back its (streaming) response body.
fn open_upstream_stream(
    upstream: &str,
    key: Option<&str>,
    request: &Value,
) -> Result<Box<dyn Read + Send>, ProxyError> {
    // The read timeout applies to each read, not to the whole generation: a
    // stream may run for as long as the engine keeps producing chunks.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(UPSTREAM_REQUEST_TIMEOUT)
        .build();
    let mut upstream_request = agent
        .post(&format!("{upstream}/v1/chat/completions"))
        .set("Content-Type", "application/json")
        .set("Accept", "text/event-stream");
    if let Some(key) = key {
        upstream_request = upstream_request.set("Authorization", &format!("Bearer {key}"));
    }
    match upstream_request.send_string(&request.to_string()) {
        Ok(response) => Ok(Box::new(response.into_reader())),
        Err(ureq::Error::Status(status, response)) => Err(ProxyError::from_upstream(
            status,
            &response.into_string().unwrap_or_default(),
        )),
        Err(error) => Err(ProxyError::unreachable(&anyhow::Error::new(error))),
    }
}

/// Read the engine's server-sent events and send their Anthropic translation
/// through `sender`, until the engine finishes or the client goes away (which
/// drops `upstream` and with it the engine connection).
fn pump_stream(
    upstream: Box<dyn Read + Send>,
    served_model: &str,
    sender: &tokio::sync::mpsc::Sender<Bytes>,
) {
    let mut translator = StreamTranslator::new(served_model);
    let send = |events: Vec<String>| {
        events
            .into_iter()
            .all(|event| sender.blocking_send(Bytes::from(event)).is_ok())
    };
    for line in BufReader::new(upstream).lines() {
        // Events are only sent when a chunk carries text or a tool call, so a
        // client that left during a long reasoning phase would otherwise go
        // unnoticed.
        if sender.is_closed() {
            return;
        }
        let events = match line {
            Ok(line) => match line.strip_prefix("data:") {
                Some(data) => translator.feed(data.trim()),
                // A 200 reply whose body is a plain JSON error rather than
                // `data:` lines: surface the engine's message.
                None if !translator.has_started() && line.trim_start().starts_with('{') => {
                    translator.feed(line.trim())
                }
                None => continue,
            },
            Err(error) => translator.fail(&format!("reading the engine's stream failed: {error}")),
        };
        if !send(events) || translator.is_finished() {
            return;
        }
    }
    send(translator.end_of_stream());
}

async fn models(State(state): State<ProxyState>, headers: HeaderMap) -> Response {
    let key = client_api_key(&headers);
    let result = tokio::task::spawn_blocking(move || {
        http_get_with_auth(
            &state.upstream,
            "/v1/models",
            key.as_deref(),
            Duration::from_secs(10),
        )
    })
    .await;
    match result {
        Ok(Ok(response)) if response.status == 200 => (
            StatusCode::OK,
            [("content-type", "application/json")],
            response.body,
        )
            .into_response(),
        Ok(Ok(response)) => {
            ProxyError::from_upstream(response.status, &response.body).into_response()
        }
        Ok(Err(error)) => ProxyError::unreachable(&error).into_response(),
        Err(error) => ProxyError::task_failed(&error).into_response(),
    }
}

/// The text of an Anthropic `system` or message `content` value.
///
/// A string is itself; an array must hold only `text` blocks.
fn text_of(content: &Value, field: &str) -> Result<String, ProxyError> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => {
            let mut parts = Vec::with_capacity(blocks.len());
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => parts.push(
                        block
                            .get("text")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                ProxyError::invalid(format!(
                                    "a `text` content block in `{field}` needs a string `text`"
                                ))
                            })?
                            .to_owned(),
                    ),
                    Some(other) => {
                        return Err(ProxyError::unsupported(&format!(
                            "a `{other}` content block in `{field}`"
                        )));
                    }
                    None => {
                        return Err(ProxyError::invalid(format!(
                            "a content block in `{field}` has no `type`"
                        )));
                    }
                }
            }
            Ok(parts.join("\n\n"))
        }
        _ => Err(ProxyError::invalid(format!(
            "`{field}` must be a string or an array of content blocks"
        ))),
    }
}

/// The text of a `tool_result` block's `content`: a string, or `text` blocks.
fn tool_result_text(block: &Value) -> Result<String, ProxyError> {
    let text = match block.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(content) => text_of(content, "tool_result content")?,
    };
    Ok(
        if block.get("is_error").and_then(Value::as_bool) == Some(true) {
            // OpenAI's tool message has no error flag; say it in the text instead.
            format!("Error: {text}")
        } else {
            text
        },
    )
}

/// Append one Anthropic turn to `messages` as OpenAI chat messages.
///
/// A string, or an array of `text` blocks, is one message. An assistant
/// `tool_use` block becomes a `tool_calls` entry on that message. A user
/// `tool_result` block becomes a `tool` message of its own, placed before the
/// turn's text: OpenAI wants tool results to follow the assistant's calls
/// directly, and Anthropic requires them first in the user turn.
fn push_turn(messages: &mut Vec<Value>, role: &str, content: &Value) -> Result<(), ProxyError> {
    let Value::Array(blocks) = content else {
        messages.push(json!({ "role": role, "content": text_of(content, "messages")? }));
        return Ok(());
    };
    let mut text = Vec::new();
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for block in blocks {
        let kind = block.get("type").and_then(Value::as_str);
        match (kind, role) {
            (Some("text"), _) => text.push(block.clone()),
            (Some("tool_use"), "assistant") => {
                let field = |name: &str| {
                    block
                        .get(name)
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .ok_or_else(|| {
                            ProxyError::invalid(format!(
                                "a `tool_use` content block needs a string `{name}`"
                            ))
                        })
                };
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                calls.push(json!({
                    "id": field("id")?,
                    "type": "function",
                    "function": { "name": field("name")?, "arguments": input.to_string() },
                }));
            }
            (Some("tool_result"), "user") => {
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        ProxyError::invalid("a `tool_result` content block needs a `tool_use_id`")
                    })?;
                results.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": tool_result_text(block)?,
                }));
            }
            (Some(kind @ ("tool_use" | "tool_result")), _) => {
                let allowed = if kind == "tool_use" {
                    "assistant"
                } else {
                    "user"
                };
                return Err(ProxyError::invalid(format!(
                    "messages: a `{kind}` content block belongs in a {allowed} message, not {role}"
                )));
            }
            (Some(other), _) => {
                return Err(ProxyError::unsupported(&format!(
                    "a `{other}` content block in `messages`"
                )));
            }
            (None, _) => {
                return Err(ProxyError::invalid(
                    "a content block in `messages` has no `type`",
                ));
            }
        }
    }
    messages.extend(results);
    let text = text_of(&Value::Array(text), "messages")?;
    if calls.is_empty() {
        if !text.is_empty() || blocks.is_empty() {
            messages.push(json!({ "role": role, "content": text }));
        }
    } else {
        let content = if text.is_empty() {
            Value::Null
        } else {
            json!(text)
        };
        messages.push(json!({ "role": role, "content": content, "tool_calls": calls }));
    }
    Ok(())
}

/// Anthropic `tools` as OpenAI function tools; `None` when none are given.
///
/// Server tools (web search, code execution, ...) run on Anthropic's side and
/// have no engine counterpart, so a tool with a `type` other than `custom` is
/// refused.
fn translate_tools(tools: Option<&Value>) -> Result<Option<Value>, ProxyError> {
    let Some(tools) = tools.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let tools = tools
        .as_array()
        .ok_or_else(|| ProxyError::invalid("tools: must be an array"))?;
    if tools.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        if let Some(kind) = tool
            .get("type")
            .and_then(Value::as_str)
            .filter(|kind| *kind != "custom")
        {
            return Err(ProxyError::unsupported(&format!(
                "a tool of type `{kind}` (only `custom` function tools are accepted)"
            )));
        }
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| ProxyError::invalid("tools: each tool needs a `name`"))?;
        let mut function = json!({
            "name": name,
            "parameters": tool
                .get("input_schema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
        });
        if let Some(description) = tool.get("description").filter(|value| value.is_string()) {
            function["description"] = description.clone();
        }
        out.push(json!({ "type": "function", "function": function }));
    }
    Ok(Some(Value::Array(out)))
}

/// Anthropic `tool_choice` as OpenAI's, plus whether parallel calls stay allowed.
fn translate_tool_choice(choice: &Value) -> Result<(Value, bool), ProxyError> {
    let parallel = choice
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        != Some(true);
    let translated = match choice.get("type").and_then(Value::as_str) {
        Some("auto") => json!("auto"),
        Some("any") => json!("required"),
        Some("none") => json!("none"),
        Some("tool") => {
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| ProxyError::invalid("tool_choice: `tool` needs a `name`"))?;
            json!({ "type": "function", "function": { "name": name } })
        }
        _ => {
            return Err(ProxyError::invalid(
                "tool_choice: `type` must be auto, any, tool or none",
            ));
        }
    };
    Ok((translated, parallel))
}

/// The engine's `tool_calls` as Anthropic `tool_use` blocks.
fn tool_use_blocks(calls: &[Value]) -> Result<Vec<Value>, ProxyError> {
    let bad_gateway = |message: String| ProxyError {
        status: 502,
        kind: "api_error",
        message,
    };
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let name = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| bad_gateway("the engine's tool call had no name".to_owned()))?;
            let input = match call.pointer("/function/arguments") {
                None | Some(Value::Null) => json!({}),
                Some(Value::String(raw)) if raw.trim().is_empty() => json!({}),
                Some(Value::String(raw)) => serde_json::from_str::<Value>(raw)
                    .ok()
                    .filter(Value::is_object)
                    .ok_or_else(|| {
                        bad_gateway(format!(
                            "the model's arguments for tool `{name}` were not a JSON object: {raw}"
                        ))
                    })?,
                Some(object @ Value::Object(_)) => object.clone(),
                Some(_) => {
                    return Err(bad_gateway(format!(
                        "the engine's arguments for tool `{name}` were not a JSON object"
                    )));
                }
            };
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map_or_else(|| format!("toolu_rocm_{index}"), str::to_owned);
            Ok(json!({ "type": "tool_use", "id": id, "name": name, "input": input }))
        })
        .collect()
}

/// Translate an Anthropic Messages request into an OpenAI chat-completions one.
fn anthropic_to_openai(request: &Value, served_model: &str) -> Result<Value, ProxyError> {
    let object = request
        .as_object()
        .ok_or_else(|| ProxyError::invalid("the request body must be a JSON object"))?;
    let stream = object.get("stream").and_then(Value::as_bool) == Some(true);
    if object
        .get("thinking")
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| kind != "disabled")
    {
        return Err(ProxyError::unsupported("extended thinking (`thinking`)"));
    }
    let max_tokens = object
        .get("max_tokens")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| ProxyError::invalid("max_tokens: a positive integer is required"))?;
    let turns = object
        .get("messages")
        .and_then(Value::as_array)
        .filter(|turns| !turns.is_empty())
        .ok_or_else(|| ProxyError::invalid("messages: at least one message is required"))?;

    let mut messages = Vec::with_capacity(turns.len() + 1);
    // PROTOTYPE (ROCMAI-489): Claude Code also sends `system`-role turns inside
    // `messages`, after the user turn. Many local chat templates refuse a
    // system message that is not first, so their text joins the leading system
    // prompt instead of keeping its place. A request still needs at least one
    // user or assistant turn.
    let is_system = |turn: &&Value| turn.get("role").and_then(Value::as_str) == Some("system");
    let mut system_text = match object.get("system").filter(|value| !value.is_null()) {
        Some(system) => text_of(system, "system")?,
        None => String::new(),
    };
    for turn in turns.iter().filter(is_system) {
        let content = turn
            .get("content")
            .ok_or_else(|| ProxyError::invalid("messages: each message needs `content`"))?;
        let text = text_of(content, "system message")?;
        if !text.is_empty() {
            if !system_text.is_empty() {
                system_text.push_str("\n\n");
            }
            system_text.push_str(&text);
        }
    }
    if !turns.iter().any(|turn| !is_system(&turn)) {
        return Err(ProxyError::invalid(
            "messages: needs at least one user or assistant turn",
        ));
    }
    if !system_text.is_empty() {
        messages.push(json!({ "role": "system", "content": system_text }));
    }
    for turn in turns.iter().filter(|turn| !is_system(turn)) {
        let role = turn
            .get("role")
            .and_then(Value::as_str)
            .filter(|role| matches!(*role, "user" | "assistant"))
            .ok_or_else(|| {
                ProxyError::invalid("messages: each message needs a `role` of user or assistant")
            })?;
        let content = turn
            .get("content")
            .ok_or_else(|| ProxyError::invalid("messages: each message needs `content`"))?;
        push_turn(&mut messages, role, content)?;
    }

    let tools = translate_tools(object.get("tools"))?;
    let mut out = json!({
        "model": served_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    });
    if stream {
        // Without this the engine omits token counts from a stream, and the
        // closing `message_delta` would report zero output tokens.
        out["stream_options"] = json!({ "include_usage": true });
    }
    for key in ["temperature", "top_p", "top_k"] {
        if let Some(value) = object.get(key).filter(|value| value.is_number()) {
            out[key] = value.clone();
        }
    }
    if let Some(tools) = tools {
        out["tools"] = tools;
        if let Some(choice) = object.get("tool_choice").filter(|value| !value.is_null()) {
            let (choice, parallel) = translate_tool_choice(choice)?;
            out["tool_choice"] = choice;
            if !parallel {
                out["parallel_tool_calls"] = json!(false);
            }
        }
    } else if object
        .get("tool_choice")
        .is_some_and(|choice| !choice.is_null())
    {
        return Err(ProxyError::invalid(
            "tool_choice: `tools` must list at least one tool to choose from",
        ));
    }
    if let Some(stops) = object.get("stop_sequences").and_then(Value::as_array)
        && !stops.is_empty()
    {
        out["stop"] = Value::Array(stops.clone());
    }
    Ok(out)
}

/// Translate an OpenAI chat-completions reply into an Anthropic Message.
fn openai_to_anthropic(reply: &Value, served_model: &str) -> Result<Value, ProxyError> {
    let choice = reply
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| ProxyError {
            status: 502,
            kind: "api_error",
            message: "the engine's reply had no choices".to_owned(),
        })?;
    let text = choice
        .pointer("/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let calls = choice
        .pointer("/message/tool_calls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let truncated = choice.get("finish_reason").and_then(Value::as_str) == Some("length");
    let mut tool_uses = tool_use_blocks(calls);
    if truncated && tool_uses.is_err() {
        // The token limit cut the arguments off mid-object; keep the text and
        // report `max_tokens` rather than failing the whole reply.
        tool_uses = Ok(Vec::new());
    }
    let tool_uses = tool_uses?;
    // Some engines report `stop` beside tool calls, so the calls decide.
    let stop_reason = if !tool_uses.is_empty() {
        "tool_use"
    } else if truncated {
        "max_tokens"
    } else {
        "end_turn"
    };
    let mut content = Vec::new();
    if !text.is_empty() || tool_uses.is_empty() {
        content.push(json!({ "type": "text", "text": text }));
    }
    content.extend(tool_uses);
    let id = reply
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map_or_else(
            || "msg_rocm".to_owned(),
            |id| format!("msg_{}", id.trim_start_matches("chatcmpl-")),
        );
    let token_count = |key: &str| {
        reply
            .pointer(&format!("/usage/{key}"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    Ok(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": reply
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
            .unwrap_or(served_model),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": token_count("prompt_tokens"),
            "output_tokens": token_count("completion_tokens"),
        },
    }))
}

/// A fragment of a `tool_use` block's JSON input.
fn input_json_delta(index: u32, fragment: &str) -> String {
    sse_event(
        "content_block_delta",
        &json!({
            "type": "content_block_delta",
            "index": index,
            "delta": { "type": "input_json_delta", "partial_json": fragment },
        }),
    )
}

/// One Anthropic server-sent event.
fn sse_event(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// Turns an engine's streamed chat-completion chunks into the Anthropic
/// streaming event sequence: `message_start`, then content blocks in order (a
/// text block per run of text, a `tool_use` block per tool call, each with a
/// `content_block_start`, deltas and a `content_block_stop`), then
/// `message_delta` and `message_stop` once the engine is done.
///
/// Only one block is open at a time. An engine streams a tool call as a first
/// delta carrying its `id` and function name followed by fragments of its JSON
/// arguments, and starts the next call (or more text) only after the previous
/// one is complete, so a block is closed when something else begins. An engine
/// that returns to a call it already left (interleaving two calls) cannot be
/// relayed that way, so the stream ends with an `error` event instead.
///
/// A call's `content_block_start` waits for its name. String argument
/// fragments that arrive before it are held and sent right after it; later
/// ones are relayed as they arrive, and the whole is checked when the block
/// closes. A call that never gets a name, or whose fragments do not add up to
/// a JSON object, then ends the stream with an `error` event in place of its
/// `content_block_stop`. Arguments of any other JSON type than a string, an
/// object or `null` end the stream with an `error` event at once. Either way
/// the client never gets a finished `tool_use` it cannot run (the
/// non-streaming reply answers 502 for the same calls).
///
/// A `length` finish is reported as `max_tokens` whatever came before it. The
/// blocks already relayed stay: a complete call, a call cut off mid-arguments
/// (closed with the partial JSON, which is not checked) and a call cut off
/// before any argument (closed with `{}`). Only a call cut off before its name
/// is dropped, since none of it reached the client. The non-streaming reply
/// differs here: it keeps complete calls with the `tool_use` stop reason, and
/// drops every call when one was cut off.
///
/// The closing events wait for the engine's `[DONE]` (or end of stream) rather
/// than its `finish_reason` chunk, because the usage totals arrive in a chunk
/// after it.
struct StreamTranslator {
    served_model: String,
    started: bool,
    /// The block currently open, if any.
    open: Option<OpenBlock>,
    /// Anthropic index the next block gets.
    next_index: u32,
    /// The reply's id, as `message_start` gave it.
    message_id: String,
    /// How many tool calls the reply has had. A call the engine gave no id is
    /// numbered by it under the reply's id, so when the engine's chunks carry
    /// an id, such calls in different replies do not share one. Known limit:
    /// chunks without an id give every reply `msg_rocm`, so its id-less calls
    /// are `toolu_rocm_0`, `toolu_rocm_1`... again, the ids the non-streaming
    /// reply gives such calls in every reply.
    tool_calls: usize,
    /// `(engine index, engine id)` of every tool call already closed, so a
    /// fragment for one of them is recognised.
    closed_calls: Vec<(u64, Option<String>)>,
    finished: bool,
    stop_reason: Option<&'static str>,
    input_tokens: u64,
    output_tokens: u64,
}

/// A content block that has been started and not yet stopped.
enum OpenBlock {
    Text { index: u32 },
    Tool(ToolCall),
}

/// A streamed tool call that has not been closed yet.
struct ToolCall {
    /// Its Anthropic index, once its `content_block_start` was sent. That waits
    /// for the call's name, which a client needs to run it.
    index: Option<u32>,
    /// The engine's position for the call in the reply.
    call_index: u64,
    /// The engine's id for the call, if it sent one.
    engine_id: Option<String>,
    /// The id the client sees (synthesised when the engine sent none).
    id: String,
    name: String,
    /// Every argument fragment received so far.
    arguments: String,
    /// How many bytes of `arguments` the client already has.
    relayed: usize,
}

impl StreamTranslator {
    fn new(served_model: &str) -> Self {
        Self {
            served_model: served_model.to_owned(),
            started: false,
            open: None,
            next_index: 0,
            message_id: String::new(),
            tool_calls: 0,
            closed_calls: Vec::new(),
            finished: false,
            stop_reason: None,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    const fn has_started(&self) -> bool {
        self.started
    }

    /// Whether the closing event (`message_stop` or `error`) has been produced.
    const fn is_finished(&self) -> bool {
        self.finished
    }

    /// Translate the payload of one engine `data:` line.
    fn feed(&mut self, data: &str) -> Vec<String> {
        if self.finished || data.is_empty() {
            return Vec::new();
        }
        if data == "[DONE]" {
            // `[DONE]` with no stop reason seen is an early end, like a cut stream.
            return self.end_of_stream();
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            return self.fail("the engine sent a stream chunk that was not JSON");
        };
        if let Some(error) = chunk.get("error") {
            let message = error
                .get("message")
                .or(Some(error))
                .and_then(Value::as_str)
                .unwrap_or("the engine reported an error");
            return self.fail(message);
        }
        let mut events = Vec::new();
        if !self.started {
            let id = chunk
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map_or_else(
                    || "msg_rocm".to_owned(),
                    |id| format!("msg_{}", id.trim_start_matches("chatcmpl-")),
                );
            let model = chunk
                .get("model")
                .and_then(Value::as_str)
                .filter(|model| !model.is_empty())
                .unwrap_or(&self.served_model)
                .to_owned();
            events.push(self.start(&id, &model));
        }
        let token_count = |key: &str| {
            chunk
                .pointer(&format!("/usage/{key}"))
                .and_then(Value::as_u64)
        };
        self.input_tokens = token_count("prompt_tokens").unwrap_or(self.input_tokens);
        self.output_tokens = token_count("completion_tokens").unwrap_or(self.output_tokens);
        if let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            && let Err(message) = self.choice_delta(choice, &mut events)
        {
            events.extend(self.fail(&message));
        }
        events
    }

    /// Translate one chunk's choice into `events`: its text, its tool-call
    /// fragments and its finish reason. An `Err` is the message of the `error`
    /// event the stream must end with.
    fn choice_delta(&mut self, choice: &Value, events: &mut Vec<String>) -> Result<(), String> {
        if let Some(text) = choice
            .pointer("/delta/content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            let index = self.open_text(events)?;
            events.push(sse_event(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "text_delta", "text": text },
                }),
            ));
        }
        if let Some(calls) = choice
            .pointer("/delta/tool_calls")
            .and_then(Value::as_array)
        {
            for (position, call) in calls.iter().enumerate() {
                self.tool_call_delta(position as u64, call, events)?;
            }
        }
        match choice.get("finish_reason").and_then(Value::as_str) {
            Some("length") => self.stop_reason = Some("max_tokens"),
            Some(_) => self.stop_reason = Some(self.natural_stop_reason()),
            None => {}
        }
        Ok(())
    }

    /// The events for an engine stream that is over: complete if a stop reason
    /// arrived, an `error` otherwise.
    fn end_of_stream(&mut self) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }
        if let Some(stop_reason) = self.stop_reason {
            return self.complete(stop_reason);
        }
        self.fail("the engine's stream ended before the reply was complete")
    }

    /// End the stream with an Anthropic `error` event.
    fn fail(&mut self, message: &str) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        vec![sse_event(
            "error",
            &json!({
                "type": "error",
                "error": { "type": "api_error", "message": message },
            }),
        )]
    }

    fn start(&mut self, id: &str, model: &str) -> String {
        self.started = true;
        id.clone_into(&mut self.message_id);
        sse_event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 },
                },
            }),
        )
    }

    /// `tool_use` when the reply contained a tool call, `end_turn` otherwise.
    /// Some engines report `stop` rather than `tool_calls` after a call.
    const fn natural_stop_reason(&self) -> &'static str {
        if self.tool_calls > 0 {
            "tool_use"
        } else {
            "end_turn"
        }
    }

    /// A new block's index.
    const fn take_index(&mut self) -> u32 {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    /// Stop the open block, if any, into `events`. A tool call is checked
    /// first: one with no name or with arguments that are not a JSON object is
    /// an `Err`, unless the token limit cut it off (then a nameless call is
    /// dropped and partial arguments are closed as they are). A call that
    /// streamed no arguments gets an empty object, so the client has JSON to
    /// parse.
    fn close_open(&mut self, events: &mut Vec<String>) -> Result<(), String> {
        let index = match self.open.take() {
            None => return Ok(()),
            Some(OpenBlock::Text { index }) => index,
            Some(OpenBlock::Tool(call)) => {
                let truncated = self.stop_reason == Some("max_tokens");
                self.closed_calls.push((call.call_index, call.engine_id));
                let Some(index) = call.index else {
                    // Nothing of the call reached the client, so a call the token
                    // limit cut off before its name is dropped.
                    return if truncated {
                        Ok(())
                    } else {
                        Err("the engine's tool call had no name".to_owned())
                    };
                };
                let raw = call.arguments.trim();
                if raw.is_empty() {
                    events.push(input_json_delta(index, "{}"));
                } else if !truncated
                    && !serde_json::from_str::<Value>(raw).is_ok_and(|input| input.is_object())
                {
                    return Err(format!(
                        "the model's arguments for tool `{}` were not a JSON object: {raw}",
                        call.name
                    ));
                }
                index
            }
        };
        events.push(sse_event(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": index }),
        ));
        Ok(())
    }

    /// Make a text block the open one, adding the events that takes to
    /// `events`, and return its index.
    fn open_text(&mut self, events: &mut Vec<String>) -> Result<u32, String> {
        if let Some(OpenBlock::Text { index }) = self.open {
            return Ok(index);
        }
        self.close_open(events)?;
        let index = self.take_index();
        self.open = Some(OpenBlock::Text { index });
        events.push(sse_event(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "text", "text": "" },
            }),
        ));
        Ok(index)
    }

    /// Translate one entry of a chunk's `delta.tool_calls` into `events`.
    ///
    /// An entry continues the open call when it has the same engine index and
    /// either no `id`, the open call's `id`, or an `id` where the open call has
    /// none yet (adopted as the call's id). Otherwise it starts a new call,
    /// unless it names a call already closed, which is an `Err`. A call's block
    /// starts once its name is known; string fragments are held until then
    /// and relayed as they arrive after it.
    fn tool_call_delta(
        &mut self,
        position: u64,
        call: &Value,
        events: &mut Vec<String>,
    ) -> Result<(), String> {
        let call_index = call
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(position);
        let engine_id = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        let continues = matches!(
            &self.open,
            Some(OpenBlock::Tool(open)) if open.call_index == call_index
                && engine_id.is_none_or(|id| open.engine_id.as_deref().is_none_or(|open_id| open_id == id))
        );
        if !continues {
            let returns_to_a_closed_call =
                self.closed_calls
                    .iter()
                    .any(|(closed_index, closed_id)| match engine_id {
                        Some(id) => closed_id.as_deref() == Some(id),
                        None => *closed_index == call_index,
                    });
            if returns_to_a_closed_call {
                return Err(
                    "the engine interleaved its tool calls, which cannot be streamed".to_owned(),
                );
            }
            self.close_open(events)?;
            let id = engine_id.map_or_else(
                || {
                    let reply = self.message_id.trim_start_matches("msg_");
                    format!("toolu_{reply}_{}", self.tool_calls)
                },
                str::to_owned,
            );
            self.tool_calls += 1;
            self.open = Some(OpenBlock::Tool(ToolCall {
                index: None,
                call_index,
                engine_id: engine_id.map(str::to_owned),
                id,
                name: String::new(),
                arguments: String::new(),
                relayed: 0,
            }));
        }
        let mut next_index = self.next_index;
        let Some(OpenBlock::Tool(open)) = &mut self.open else {
            unreachable!("a tool call was just made the open block");
        };
        if open.engine_id.is_none()
            && let Some(id) = engine_id
        {
            id.clone_into(&mut open.id);
            open.engine_id = Some(id.to_owned());
        }
        if let Some(name) = call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .filter(|name| open.name.is_empty() && !name.is_empty())
        {
            name.clone_into(&mut open.name);
        }
        match call.pointer("/function/arguments") {
            // A delta with nothing new for the arguments.
            None | Some(Value::Null) => {}
            Some(Value::String(fragment)) => open.arguments.push_str(fragment),
            // Not the OpenAI shape, but some engines send the object itself.
            Some(object @ Value::Object(_)) => open.arguments.push_str(&object.to_string()),
            Some(other) => {
                return Err(format!(
                    "the engine's arguments for tool `{}` were not a JSON object: {other}",
                    open.name
                ));
            }
        }
        if open.index.is_none() && !open.name.is_empty() {
            open.index = Some(next_index);
            next_index += 1;
            events.push(sse_event(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": next_index - 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": open.id,
                        "name": open.name,
                        "input": {},
                    },
                }),
            ));
        }
        if let Some(index) = open.index
            && open.relayed < open.arguments.len()
        {
            events.push(input_json_delta(index, &open.arguments[open.relayed..]));
            open.relayed = open.arguments.len();
        }
        self.next_index = next_index;
        Ok(())
    }

    /// The closing events, for a reply that finished with `stop_reason`. A
    /// stop reason only comes from a chunk, after `message_start`. An engine
    /// that produced no content still gets an empty text block, matching the
    /// non-streaming reply.
    fn complete(&mut self, stop_reason: &'static str) -> Vec<String> {
        let mut events = Vec::new();
        let closed = self.close_open(&mut events).and_then(|()| {
            if self.next_index == 0 {
                self.open_text(&mut events)?;
                self.close_open(&mut events)?;
            }
            Ok(())
        });
        if let Err(message) = closed {
            events.extend(self.fail(&message));
            return events;
        }
        events.push(sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": null,
                },
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                },
            }),
        ));
        events.push(sse_event(
            "message_stop",
            &json!({ "type": "message_stop" }),
        ));
        self.finished = true;
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    const MODEL: &str = "Org/Served-Model";

    /// A directory no other test in this process shares. Process id plus a
    /// millisecond clock is not enough: tests run in parallel threads and two can
    /// start in the same millisecond, then overwrite each other's state file.
    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "rocmd-anthropic-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    fn translate(request: &Value) -> Result<Value, ProxyError> {
        anthropic_to_openai(request, MODEL)
    }

    #[test]
    fn request_maps_system_turns_and_sampling_to_chat_completions() {
        let out = translate(&json!({
            "model": "claude-sonnet-4",
            "max_tokens": 64,
            "system": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "Be kind."}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                {"role": "user", "content": "again"},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"],
        }))
        .unwrap();
        assert_eq!(
            out,
            json!({
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Be brief.\n\nBe kind."},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "again"},
                ],
                "max_tokens": 64,
                "stream": false,
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "stop": ["END"],
            })
        );
    }

    #[test]
    fn a_streaming_request_asks_the_engine_to_stream_with_usage() {
        let out = translate(&json!({
            "max_tokens": 8,
            "stream": true,
            "messages": [{"role": "user", "content": "x"}],
        }))
        .unwrap();
        assert_eq!(out["stream"], true);
        assert_eq!(out["stream_options"], json!({"include_usage": true}));
        let out = translate(&json!({
            "max_tokens": 8,
            "stream": false,
            "messages": [{"role": "user", "content": "x"}],
        }))
        .unwrap();
        assert_eq!(out["stream"], false);
        assert!(out.get("stream_options").is_none());
    }

    #[test]
    fn request_always_names_the_served_model_not_the_clients() {
        let out = translate(&json!({
            "model": "claude-opus-4",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "x"}],
        }))
        .unwrap();
        assert_eq!(out["model"], MODEL);
    }

    #[test]
    fn request_refuses_what_it_cannot_translate_instead_of_dropping_it() {
        let base = || json!({"max_tokens": 8, "messages": [{"role": "user", "content": "x"}]});
        let mut server_tool = base();
        server_tool["tools"] = json!([{"type": "web_search_20250305", "name": "web_search"}]);
        let image = json!({"max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}
        ]}]});
        for (request, needle) in [
            (server_tool, "a tool of type `web_search_20250305`"),
            (image, "`image` content block"),
        ] {
            let error = translate(&request).unwrap_err();
            assert_eq!(error.status, 400);
            assert_eq!(error.kind, "invalid_request_error");
            assert!(error.message.contains(needle), "{}", error.message);
        }
    }

    #[test]
    fn request_rejects_malformed_input_with_invalid_request_errors() {
        for request in [
            json!([]),
            json!({"messages": [{"role": "user", "content": "x"}]}),
            json!({"max_tokens": 0, "messages": [{"role": "user", "content": "x"}]}),
            json!({"max_tokens": 8, "messages": []}),
            json!({"max_tokens": 8}),
            json!({"max_tokens": 8, "messages": [{"role": "system", "content": "x"}]}),
            json!({"max_tokens": 8, "messages": [{"role": "user"}]}),
        ] {
            let error = translate(&request).unwrap_err();
            assert_eq!(error.kind, "invalid_request_error", "{request}");
        }
    }

    #[test]
    fn response_becomes_a_valid_anthropic_message() {
        let message = openai_to_anthropic(
            &json!({
                "id": "chatcmpl-abc123",
                "model": MODEL,
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2},
            }),
            "fallback",
        )
        .unwrap();
        assert_eq!(
            message,
            json!({
                "id": "msg_abc123",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 11, "output_tokens": 2},
            })
        );
    }

    #[test]
    fn response_keeps_the_text_when_a_length_stop_truncates_tool_arguments() {
        let reply = |finish: &str| {
            json!({"choices": [{"message": {"content": "Checking.", "tool_calls": [
                {"id": "call_1", "function": {"name": "weather", "arguments": "{\"city\":\"Os"}}
            ]}, "finish_reason": finish}]})
        };
        let message = openai_to_anthropic(&reply("length"), MODEL).unwrap();
        assert_eq!(message["stop_reason"], "max_tokens");
        assert_eq!(
            message["content"],
            json!([{"type": "text", "text": "Checking."}])
        );
        // Without the token limit the same arguments are still a bad gateway.
        assert_eq!(
            openai_to_anthropic(&reply("tool_calls"), MODEL)
                .unwrap_err()
                .status,
            502
        );
    }

    #[test]
    fn system_role_turns_join_the_leading_system_prompt() {
        let out = translate(&json!({
            "max_tokens": 8,
            "system": "Be brief.",
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "system", "content": [{"type": "text", "text": "Reminder."}]},
            ],
        }))
        .unwrap();
        assert_eq!(
            out["messages"],
            json!([
                {"role": "system", "content": "Be brief.\n\nReminder."},
                {"role": "user", "content": "x"},
            ])
        );
    }

    #[test]
    fn response_keeps_complete_calls_as_tool_use_under_a_length_stop() {
        // Pins the non-streaming behaviour as it is: complete calls win over
        // the token limit. A stream has relayed them before the stop arrives,
        // so it reports `max_tokens` instead (see the stream tests).
        let message = openai_to_anthropic(
            &json!({"choices": [{"message": {"content": null, "tool_calls": [
                {"id": "call_1", "function": {"name": "weather", "arguments": "{\"city\":\"Oslo\"}"}}
            ]}, "finish_reason": "length"}]}),
            MODEL,
        )
        .unwrap();
        assert_eq!(message["stop_reason"], "tool_use");
        assert_eq!(message["content"][0]["input"], json!({"city": "Oslo"}));
    }

    #[test]
    fn response_maps_a_length_stop_and_falls_back_to_the_served_model() {
        let message = openai_to_anthropic(
            &json!({"choices": [{"message": {"content": null}, "finish_reason": "length"}]}),
            MODEL,
        )
        .unwrap();
        assert_eq!(message["stop_reason"], "max_tokens");
        assert_eq!(message["model"], MODEL);
        assert_eq!(message["content"][0]["text"], "");
        assert_eq!(message["usage"]["input_tokens"], 0);
    }

    #[test]
    fn response_without_choices_is_a_bad_gateway() {
        let error = openai_to_anthropic(&json!({"choices": []}), MODEL).unwrap_err();
        assert_eq!(error.status, 502);
    }

    #[test]
    fn upstream_errors_keep_their_status_and_message_with_anthropic_types() {
        let error =
            ProxyError::from_upstream(404, r#"{"error": {"message": "model `x` does not exist"}}"#);
        assert_eq!(
            (error.status, error.kind, error.message.as_str()),
            (404, "not_found_error", "model `x` does not exist")
        );
        let error = ProxyError::from_upstream(401, "not json");
        assert_eq!(error.kind, "authentication_error");
        assert_eq!(error.message, "the engine answered HTTP 401");
        assert_eq!(ProxyError::from_upstream(500, "").kind, "api_error");
    }

    #[test]
    fn client_key_is_read_from_either_header() {
        let mut headers = HeaderMap::new();
        assert_eq!(client_api_key(&headers), None);
        headers.insert("authorization", "Bearer from-bearer".parse().unwrap());
        assert_eq!(client_api_key(&headers).as_deref(), Some("from-bearer"));
        headers.insert("x-api-key", "from-x-api-key".parse().unwrap());
        assert_eq!(client_api_key(&headers).as_deref(), Some("from-x-api-key"));
    }

    #[test]
    fn an_empty_x_api_key_falls_back_to_the_authorization_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "  ".parse().unwrap());
        headers.insert("authorization", "bearer from-bearer".parse().unwrap());
        assert_eq!(client_api_key(&headers).as_deref(), Some("from-bearer"));
        headers.insert("authorization", "Basic dXNlcjpwdw==".parse().unwrap());
        assert_eq!(client_api_key(&headers), None);
    }

    #[test]
    fn engine_errors_shaped_as_detail_keep_their_message() {
        let error = ProxyError::from_upstream(404, r#"{"detail": "no such route"}"#);
        assert_eq!(error.message, "no such route");
    }

    #[test]
    fn a_timed_out_engine_is_a_gateway_timeout_not_unreachable() {
        let timeout = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::TimedOut))
            .context("failed to read HTTP POST /v1/chat/completions");
        assert_eq!(ProxyError::unreachable(&timeout).status, 504);
        let refused =
            anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::ConnectionRefused));
        assert_eq!(ProxyError::unreachable(&refused).status, 502);
    }

    #[test]
    fn requests_that_would_silently_lose_meaning_are_refused() {
        let base = json!({"max_tokens": 4, "messages": [{"role": "user", "content": "x"}]});
        let mut request = base.clone();
        request["thinking"] = json!({"type": "enabled", "budget_tokens": 1024});
        assert_eq!(translate(&request).unwrap_err().status, 400);
        let mut disabled = base;
        disabled["thinking"] = json!({"type": "disabled"});
        assert!(translate(&disabled).is_ok());
        let missing_text = json!({
            "max_tokens": 4,
            "messages": [{"role": "user", "content": [{"type": "text"}]}],
        });
        let error = translate(&missing_text).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("string `text`"), "{}", error.message);
    }

    fn weather_tool() -> Value {
        json!({
            "name": "get_weather",
            "description": "Current weather for a city.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        })
    }

    #[test]
    fn tools_and_tool_choice_become_function_tools() {
        let mut request = json!({
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [weather_tool()],
        });
        let out = translate(&request).unwrap();
        assert_eq!(
            out["tools"],
            json!([{"type": "function", "function": {
                "name": "get_weather",
                "description": "Current weather for a city.",
                "parameters": weather_tool()["input_schema"],
            }}])
        );
        assert!(out.get("tool_choice").is_none());
        for (choice, expected) in [
            (json!({"type": "auto"}), json!("auto")),
            (json!({"type": "any"}), json!("required")),
            (json!({"type": "none"}), json!("none")),
            (
                json!({"type": "tool", "name": "get_weather"}),
                json!({"type": "function", "function": {"name": "get_weather"}}),
            ),
        ] {
            request["tool_choice"] = choice;
            let out = translate(&request).unwrap();
            assert_eq!(out["tool_choice"], expected);
            assert!(out.get("parallel_tool_calls").is_none());
        }
        request["tool_choice"] = json!({"type": "auto", "disable_parallel_tool_use": true});
        assert_eq!(translate(&request).unwrap()["parallel_tool_calls"], false);
    }

    #[test]
    fn malformed_tool_definitions_and_choices_are_invalid_requests() {
        let base = json!({"max_tokens": 8, "messages": [{"role": "user", "content": "x"}]});
        for (tools, choice) in [
            (json!([{"input_schema": {}}]), json!(null)),
            (json!("get_weather"), json!(null)),
            (json!([weather_tool()]), json!({"type": "sometimes"})),
            (json!([weather_tool()]), json!({"type": "tool"})),
            // A choice with nothing to choose from.
            (json!([]), json!({"type": "any"})),
        ] {
            let mut request = base.clone();
            request["tools"] = tools;
            request["tool_choice"] = choice;
            assert_eq!(
                translate(&request).unwrap_err().kind,
                "invalid_request_error"
            );
        }
    }

    #[test]
    fn tool_use_and_tool_result_blocks_become_tool_calls_and_tool_messages() {
        let out = translate(&json!({
            "max_tokens": 8,
            "tools": [weather_tool()],
            "messages": [
                {"role": "user", "content": "weather in Oslo?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}},
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": "thanks"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "3C"}]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "no data", "is_error": true},
                ]},
            ],
        }))
        .unwrap();
        assert_eq!(
            out["messages"],
            json!([
                {"role": "user", "content": "weather in Oslo?"},
                {"role": "assistant", "content": "Checking.", "tool_calls": [{
                    "id": "toolu_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"},
                }]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "3C"},
                {"role": "user", "content": "thanks"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "toolu_2", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "toolu_2", "content": "Error: no data"},
            ])
        );
    }

    #[test]
    fn tool_blocks_in_the_wrong_role_or_missing_fields_are_invalid_requests() {
        for (role, block) in [
            (
                "user",
                json!({"type": "tool_use", "id": "a", "name": "n", "input": {}}),
            ),
            (
                "assistant",
                json!({"type": "tool_result", "tool_use_id": "a", "content": "x"}),
            ),
            (
                "assistant",
                json!({"type": "tool_use", "name": "n", "input": {}}),
            ),
            (
                "assistant",
                json!({"type": "tool_use", "id": "a", "input": {}}),
            ),
            ("user", json!({"type": "tool_result", "content": "x"})),
        ] {
            let error = translate(&json!({
                "max_tokens": 8,
                "messages": [{"role": role, "content": [block.clone()]}],
            }))
            .unwrap_err();
            assert_eq!(error.kind, "invalid_request_error", "{block}");
        }
    }

    #[test]
    fn tool_calls_in_the_reply_become_tool_use_blocks_with_the_tool_use_stop() {
        let message = openai_to_anthropic(
            &json!({
                "id": "chatcmpl-t1",
                "choices": [{
                    "message": {"content": "Let me check.", "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}},
                        {"type": "function", "function": {"name": "get_time", "arguments": ""}},
                    ]},
                    // Engines that report `stop` beside tool calls are still tool use.
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            }),
            MODEL,
        )
        .unwrap();
        assert_eq!(message["stop_reason"], "tool_use");
        assert_eq!(
            message["content"],
            json!([
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Oslo"}},
                {"type": "tool_use", "id": "toolu_rocm_1", "name": "get_time", "input": {}},
            ])
        );
    }

    #[test]
    fn a_reply_of_only_tool_calls_has_no_empty_text_block() {
        let message = openai_to_anthropic(
            &json!({"choices": [{
                "message": {"content": null, "tool_calls": [
                    {"id": "c", "function": {"name": "n", "arguments": "{}"}},
                ]},
                "finish_reason": "tool_calls",
            }]}),
            MODEL,
        )
        .unwrap();
        assert_eq!(message["content"].as_array().unwrap().len(), 1);
        assert_eq!(message["content"][0]["type"], "tool_use");
    }

    #[test]
    fn unusable_tool_calls_from_the_engine_are_a_bad_gateway() {
        for call in [
            json!({"id": "c", "function": {"name": "n", "arguments": "{not json"}}),
            json!({"id": "c", "function": {"name": "n", "arguments": "[1]"}}),
            json!({"id": "c", "function": {"arguments": "{}"}}),
        ] {
            let error = openai_to_anthropic(
                &json!({"choices": [{"message": {"tool_calls": [call]}, "finish_reason": "tool_calls"}]}),
                MODEL,
            )
            .unwrap_err();
            assert_eq!(error.status, 502);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn tool_use_round_trips_through_the_engine() {
        let engine = fake_engine();
        let base = proxy_over(&engine).await;
        let (status, message) = tokio::task::spawn_blocking(move || {
            post(
                &base,
                "/v1/messages",
                None,
                &json!({
                    "model": "claude-sonnet-4",
                    "max_tokens": 64,
                    "tools": [weather_tool()],
                    "tool_choice": {"type": "tool", "name": "get_weather"},
                    "messages": [{"role": "user", "content": "weather in Oslo?"}],
                }),
            )
        })
        .await
        .unwrap();
        assert_eq!(status, 200, "{message}");
        assert_eq!(message["stop_reason"], "tool_use");
        assert_eq!(
            message["content"],
            json!([{"type": "tool_use", "id": "call_9", "name": "get_weather", "input": {"city": "Oslo"}}])
        );
        let seen = engine.seen.lock().unwrap();
        let forwarded: Value = serde_json::from_str(&seen[0].1).unwrap();
        assert_eq!(forwarded["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(forwarded["tool_choice"]["function"]["name"], "get_weather");
    }

    #[test]
    fn wildcard_listen_hosts_are_reached_over_loopback() {
        let config = |host: &str| AnthropicProxyConfig {
            host: host.to_owned(),
            engine_port: 8000,
            served_model: MODEL.to_owned(),
            state_path: PathBuf::new(),
            probe_api_key: None,
        };
        assert_eq!(config("0.0.0.0").upstream_url(), "http://127.0.0.1:8000");
        assert_eq!(config("::").upstream_url(), "http://[::1]:8000");
        assert_eq!(config("127.0.0.1").upstream_url(), "http://127.0.0.1:8000");
    }

    fn chunk(delta: &Value, finish: Option<&str>) -> String {
        json!({
            "id": "chatcmpl-t1",
            "model": MODEL,
            "choices": [{"delta": delta, "finish_reason": finish}],
        })
        .to_string()
    }

    fn pump(body: &str) -> Vec<String> {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(64);
        pump_stream(
            Box::new(std::io::Cursor::new(body.to_owned())),
            MODEL,
            &sender,
        );
        drop(sender);
        let mut out = Vec::new();
        while let Ok(bytes) = receiver.try_recv() {
            out.push(String::from_utf8(bytes.to_vec()).unwrap());
        }
        out
    }

    #[test]
    fn a_plain_json_error_body_on_a_200_reply_reaches_the_client() {
        let events = pump(r#"{"error": {"message": "model is loading"}}"#);
        assert_eq!(names(&events), ["error"]);
        assert!(events[0].contains("model is loading"), "{}", events[0]);
    }

    fn names(events: &[String]) -> Vec<&str> {
        events
            .iter()
            .map(|event| {
                event
                    .strip_prefix("event: ")
                    .and_then(|rest| rest.split('\n').next())
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn the_stream_opens_once_and_each_text_fragment_is_a_delta() {
        let mut translator = StreamTranslator::new("fallback");
        let first = translator.feed(&chunk(&json!({"role": "assistant", "content": ""}), None));
        assert_eq!(names(&first), ["message_start"]);
        assert!(first[0].contains(r#""id":"msg_t1""#), "{}", first[0]);
        assert!(first[0].contains(MODEL), "{}", first[0]);
        let second = translator.feed(&chunk(&json!({"content": "Hel"}), None));
        assert_eq!(
            names(&second),
            ["content_block_start", "content_block_delta"]
        );
        let third = translator.feed(&chunk(&json!({"content": "lo"}), None));
        assert_eq!(names(&third), ["content_block_delta"]);
        assert!(third[0].contains(r#""text":"lo""#), "{}", third[0]);
        assert!(!translator.is_finished());
    }

    #[test]
    fn a_streaming_request_with_tools_is_forwarded_with_its_tools() {
        let out = translate(&json!({
            "max_tokens": 8,
            "stream": true,
            "tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "x"}],
        }))
        .unwrap();
        assert_eq!(out["stream"], true);
        assert_eq!(out["tools"][0]["function"]["name"], "get_weather");
    }

    /// The `data:` body an engine streams for the given chunks, `[DONE]` last.
    fn engine_stream(chunks: &[String]) -> String {
        let mut body = String::new();
        for chunk in chunks {
            body.push_str("data: ");
            body.push_str(chunk);
            body.push_str("\n\n");
        }
        body.push_str("data: [DONE]\n\n");
        body
    }

    fn call(index: u64, id: Option<&str>, name: Option<&str>, arguments: &str) -> Value {
        let mut call =
            json!({"index": index, "type": "function", "function": {"arguments": arguments}});
        if let Some(id) = id {
            call["id"] = json!(id);
        }
        if let Some(name) = name {
            call["function"]["name"] = json!(name);
        }
        call
    }

    /// The `(event, index, detail)` of every content-block event, where detail
    /// is the block type, the text, or the JSON fragment.
    fn blocks(events: &[String]) -> Vec<(String, u64, String)> {
        events
            .iter()
            .filter_map(|event| {
                let (name, data) = event.strip_prefix("event: ")?.split_once("\ndata: ")?;
                let data: Value = serde_json::from_str(data.trim()).ok()?;
                let detail = match name {
                    "content_block_start" => data["content_block"]["type"].as_str()?.to_owned(),
                    "content_block_delta" => data["delta"]["text"]
                        .as_str()
                        .or_else(|| data["delta"]["partial_json"].as_str())?
                        .to_owned(),
                    "content_block_stop" => String::new(),
                    _ => return None,
                };
                Some((name.to_owned(), data["index"].as_u64()?, detail))
            })
            .collect()
    }

    fn stop_reason(events: &[String]) -> String {
        let delta = events
            .iter()
            .find(|event| event.starts_with("event: message_delta"))
            .expect("a message_delta");
        let data: Value =
            serde_json::from_str(delta.split_once("data: ").unwrap().1.trim()).unwrap();
        data["delta"]["stop_reason"].as_str().unwrap().to_owned()
    }

    #[test]
    fn a_streamed_tool_call_becomes_a_tool_use_block_with_json_fragments() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"role": "assistant", "tool_calls": [call(0, Some("call_1"), Some("get_weather"), "")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(0, None, None, "{\"city\":")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(0, None, None, "\"Oslo\"}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(
            names(&events),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert!(events[1].contains(r#""id":"call_1""#), "{}", events[1]);
        assert!(
            events[1].contains(r#""name":"get_weather""#),
            "{}",
            events[1]
        );
        let fragments: String = blocks(&events)
            .into_iter()
            .filter(|(name, ..)| name == "content_block_delta")
            .map(|(.., detail)| detail)
            .collect();
        assert_eq!(fragments, r#"{"city":"Oslo"}"#);
        assert_eq!(stop_reason(&events), "tool_use");
    }

    #[test]
    fn text_then_a_tool_call_are_two_blocks_in_order() {
        let events = pump(&engine_stream(&[
            chunk(&json!({"content": "Checking."}), None),
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("run"), "{}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "text".into()),
                ("content_block_delta".into(), 0, "Checking.".into()),
                ("content_block_stop".into(), 0, String::new()),
                ("content_block_start".into(), 1, "tool_use".into()),
                ("content_block_delta".into(), 1, "{}".into()),
                ("content_block_stop".into(), 1, String::new()),
            ]
        );
    }

    #[test]
    fn two_tool_calls_get_their_own_blocks_whether_or_not_the_engine_numbers_them() {
        let numbered = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("a"), Some("one"), "{\"x\":1}")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(1, Some("b"), Some("two"), "{\"y\":2}")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(2, Some("c"), Some("three"), "{}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        // An engine that gives every call index 0 is told apart by `id`, and
        // a fresh id after two closed calls is a third call, not a return.
        let all_zero = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("a"), Some("one"), "{\"x\":1}")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(0, Some("b"), Some("two"), "{\"y\":2}")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("three"), "{}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        for events in [numbered, all_zero] {
            let starts: Vec<_> = blocks(&events)
                .into_iter()
                .filter(|(name, ..)| name == "content_block_start")
                .map(|(_, index, kind)| (index, kind))
                .collect();
            assert_eq!(
                starts,
                [
                    (0, "tool_use".to_owned()),
                    (1, "tool_use".to_owned()),
                    (2, "tool_use".to_owned()),
                ]
            );
            assert!(events.iter().any(|e| e.contains(r#""name":"two""#)));
            assert!(events.iter().any(|e| e.contains(r#""name":"three""#)));
            assert_eq!(stop_reason(&events), "tool_use");
        }
    }

    #[test]
    fn a_tool_call_without_arguments_still_gets_an_empty_json_object() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("list_files"), "")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "tool_use".into()),
                ("content_block_delta".into(), 0, "{}".into()),
                ("content_block_stop".into(), 0, String::new()),
            ]
        );
    }

    #[test]
    fn a_missing_call_id_is_synthesised_and_stop_after_a_call_means_tool_use() {
        // Some engines name no id and finish a tool call with `stop`. Each call
        // gets its own id, tied to the reply so a later turn's calls differ.
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, None, Some("run"), "{}")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(1, None, Some("run"), "{}")]}),
                None,
            ),
            chunk(&json!({}), Some("stop")),
        ]));
        assert_eq!(tool_use_ids(&events), ["toolu_t1_0", "toolu_t1_1"]);
        assert_eq!(stop_reason(&events), "tool_use");
    }

    #[test]
    fn an_id_arriving_after_the_call_opened_replaces_the_synthesised_one() {
        // The call opens with neither id nor name; its id comes with the name,
        // before the block starts, so the client sees the engine's id.
        let first = call(0, None, None, "");
        let events = pump(&engine_stream(&[
            chunk(&json!({"tool_calls": [first]}), None),
            chunk(
                &json!({"tool_calls": [call(0, Some("call_7"), Some("run"), "{}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert!(events[1].contains(r#""id":"call_7""#), "{}", events[1]);
    }

    #[test]
    fn a_named_call_cut_off_before_any_argument_is_closed_with_an_empty_object() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("run"), "")]}),
                None,
            ),
            chunk(&json!({}), Some("length")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "tool_use".into()),
                ("content_block_delta".into(), 0, "{}".into()),
                ("content_block_stop".into(), 0, String::new()),
            ]
        );
        assert_eq!(stop_reason(&events), "max_tokens");
    }

    #[test]
    fn a_delta_without_an_arguments_key_is_no_fragment() {
        let mut opening = call(0, Some("c"), Some("run"), "");
        opening["function"]
            .as_object_mut()
            .unwrap()
            .remove("arguments");
        let events = pump(&engine_stream(&[
            chunk(&json!({"tool_calls": [opening]}), None),
            chunk(
                &json!({"tool_calls": [call(0, None, None, "{\"a\":1}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "tool_use".into()),
                ("content_block_delta".into(), 0, "{\"a\":1}".into()),
                ("content_block_stop".into(), 0, String::new()),
            ]
        );
        assert_eq!(stop_reason(&events), "tool_use");
    }

    /// The `id` of every `tool_use` block a reply started.
    fn tool_use_ids(events: &[String]) -> Vec<String> {
        events
            .iter()
            .filter(|event| event.starts_with("event: content_block_start"))
            .map(|event| {
                let data: Value =
                    serde_json::from_str(event.split_once("data: ").unwrap().1.trim()).unwrap();
                data["content_block"]["id"].as_str().unwrap().to_owned()
            })
            .collect()
    }

    /// A reply of one id-less call to `run`, from chunks with id `chunk_id`.
    fn reply_with_an_id_less_call(chunk_id: Option<&str>) -> Vec<String> {
        let chunk = |delta: Value, finish: Option<&str>| {
            let mut chunk =
                json!({"model": MODEL, "choices": [{"delta": delta, "finish_reason": finish}]});
            if let Some(id) = chunk_id {
                chunk["id"] = json!(id);
            }
            chunk.to_string()
        };
        pump(&engine_stream(&[
            chunk(
                json!({"tool_calls": [call(0, None, Some("run"), "{}")]}),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
        ]))
    }

    #[test]
    fn id_less_calls_in_different_replies_get_different_ids_when_chunks_carry_one() {
        assert_eq!(
            tool_use_ids(&reply_with_an_id_less_call(Some("chatcmpl-one"))),
            ["toolu_one_0"]
        );
        assert_eq!(
            tool_use_ids(&reply_with_an_id_less_call(Some("chatcmpl-two"))),
            ["toolu_two_0"]
        );
        // Known limit: with no chunk id every reply falls back to the ids the
        // non-streaming reply uses, so two such replies share them.
        assert_eq!(
            tool_use_ids(&reply_with_an_id_less_call(None)),
            ["toolu_rocm_0"]
        );
    }

    #[test]
    fn running_out_of_tokens_mid_call_is_max_tokens_not_tool_use() {
        // The block and its fragments already reached the client, so they stay
        // and are closed; `max_tokens` tells the client not to run the call.
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("run"), "{\"a\":")]}),
                None,
            ),
            chunk(&json!({}), Some("length")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "tool_use".into()),
                ("content_block_delta".into(), 0, "{\"a\":".into()),
                ("content_block_stop".into(), 0, String::new()),
            ]
        );
        assert_eq!(stop_reason(&events), "max_tokens");
    }

    #[test]
    fn a_complete_call_before_a_length_stop_is_relayed_and_stops_for_max_tokens() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("run"), "{\"a\":1}")]}),
                None,
            ),
            chunk(&json!({}), Some("length")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "tool_use".into()),
                ("content_block_delta".into(), 0, "{\"a\":1}".into()),
                ("content_block_stop".into(), 0, String::new()),
            ]
        );
        assert_eq!(stop_reason(&events), "max_tokens");
    }

    #[test]
    fn arguments_that_are_neither_a_string_nor_an_object_are_an_error() {
        for arguments in [json!([1]), json!(5), json!(true)] {
            let mut entry = call(0, Some("c"), Some("run"), "");
            entry["function"]["arguments"] = arguments.clone();
            let events = pump(&engine_stream(&[
                chunk(&json!({"tool_calls": [entry]}), None),
                chunk(&json!({}), Some("tool_calls")),
            ]));
            let (name, message) = ending(&events);
            assert_eq!(name, "error", "{arguments}: {events:?}");
            // Refused on arrival, not relayed and refused at the block's close.
            assert!(
                message
                    .starts_with("the engine's arguments for tool `run` were not a JSON object: "),
                "{message}"
            );
            assert!(
                !events.iter().any(|e| e.contains("input_json_delta")),
                "{arguments}: {events:?}"
            );
        }
        // `null` is what a delta with nothing new carries, and is no fragment.
        let mut nothing_new = call(0, None, None, "");
        nothing_new["function"]["arguments"] = Value::Null;
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("run"), "{\"a\":1}")]}),
                None,
            ),
            chunk(&json!({"tool_calls": [nothing_new]}), None),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(stop_reason(&events), "tool_use");
        assert_eq!(blocks(&events)[1].2, "{\"a\":1}");
    }

    #[test]
    fn a_plain_text_stream_keeps_its_end_turn_stop_and_single_block() {
        let events = pump(&engine_stream(&[
            chunk(&json!({"content": "hi"}), None),
            chunk(&json!({}), Some("stop")),
        ]));
        assert_eq!(stop_reason(&events), "end_turn");
        assert_eq!(blocks(&events).len(), 3);
    }

    /// The final event's name and, for an `error`, its message.
    fn ending(events: &[String]) -> (String, String) {
        let last = events.last().expect("an event");
        let (name, data) = last
            .strip_prefix("event: ")
            .and_then(|rest| rest.split_once("\ndata: "))
            .unwrap();
        let data: Value = serde_json::from_str(data.trim()).unwrap();
        (
            name.to_owned(),
            data["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        )
    }

    #[test]
    fn an_engine_that_resends_the_id_and_name_on_every_chunk_streams_one_block() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("run"), "{\"a\":")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("run"), "1}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "tool_use".into()),
                ("content_block_delta".into(), 0, "{\"a\":".into()),
                ("content_block_delta".into(), 0, "1}".into()),
                ("content_block_stop".into(), 0, String::new()),
            ]
        );
    }

    #[test]
    fn calls_with_index_gaps_and_text_after_a_call_keep_dense_block_indexes() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(2, Some("a"), Some("one"), "{}")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(5, Some("b"), Some("two"), "{}")]}),
                None,
            ),
            chunk(&json!({"content": "done"}), None),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "tool_use".into()),
                ("content_block_delta".into(), 0, "{}".into()),
                ("content_block_stop".into(), 0, String::new()),
                ("content_block_start".into(), 1, "tool_use".into()),
                ("content_block_delta".into(), 1, "{}".into()),
                ("content_block_stop".into(), 1, String::new()),
                ("content_block_start".into(), 2, "text".into()),
                ("content_block_delta".into(), 2, "done".into()),
                ("content_block_stop".into(), 2, String::new()),
            ]
        );
        assert_eq!(stop_reason(&events), "tool_use");
    }

    #[test]
    fn text_after_a_call_with_bad_arguments_ends_the_stream_with_an_error() {
        // Text closes the open call, and closing it checks its arguments.
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("a"), Some("one"), "[1]")]}),
                None,
            ),
            chunk(&json!({"content": "done"}), None),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        let (name, message) = ending(&events);
        assert_eq!(name, "error", "{events:?}");
        assert!(
            message.contains("tool `one` were not a JSON object"),
            "{message}"
        );
        assert!(
            !events.iter().any(|e| e.contains("text_delta")),
            "{events:?}"
        );
    }

    #[test]
    fn a_call_named_after_its_first_delta_starts_once_the_name_arrives() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), None, "{\"a\"")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(0, None, Some("run"), ":1}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "tool_use".into()),
                ("content_block_delta".into(), 0, "{\"a\":1}".into()),
                ("content_block_stop".into(), 0, String::new()),
            ]
        );
        assert!(events[1].contains(r#""name":"run""#), "{}", events[1]);
    }

    #[test]
    fn arguments_sent_as_an_object_are_relayed_as_its_json() {
        let mut entry = call(0, Some("c"), Some("run"), "");
        entry["function"]["arguments"] = json!({"a": 1});
        let events = pump(&engine_stream(&[
            chunk(&json!({"tool_calls": [entry]}), None),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(blocks(&events)[1].2, r#"{"a":1}"#);
    }

    #[test]
    fn a_usage_only_chunk_after_a_call_reaches_message_delta() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("run"), "{}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
            json!({"id": "chatcmpl-t1", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
                .to_string(),
        ]));
        let delta = events
            .iter()
            .find(|event| event.starts_with("event: message_delta"))
            .unwrap();
        assert!(delta.contains(r#""output_tokens":3"#), "{delta}");
        assert!(delta.contains(r#""input_tokens":7"#), "{delta}");
        assert_eq!(stop_reason(&events), "tool_use");
    }

    #[test]
    fn an_engine_returning_to_a_closed_call_ends_the_stream_with_an_error() {
        // The first call is valid JSON when the second begins, so only the
        // return to it gives the interleaving away: by index when the fragment
        // has no id, by id (whatever index it comes with) when it has one.
        for returning in [call(0, None, None, " "), call(7, Some("a"), None, " ")] {
            let events = pump(&engine_stream(&[
                chunk(
                    &json!({"tool_calls": [call(0, Some("a"), Some("one"), "{\"x\":1}")]}),
                    None,
                ),
                chunk(
                    &json!({"tool_calls": [call(1, Some("b"), Some("two"), "{}")]}),
                    None,
                ),
                chunk(&json!({"tool_calls": [returning]}), None),
                chunk(&json!({}), Some("tool_calls")),
            ]));
            let (name, message) = ending(&events);
            assert_eq!(name, "error", "{events:?}");
            assert!(message.contains("interleaved"), "{message}");
            assert!(!events.iter().any(|e| e.contains("message_stop")));
        }
        // Left mid-object, the first call already fails its JSON check when
        // the second begins.
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("a"), Some("one"), "{\"x\":")]}),
                None,
            ),
            chunk(
                &json!({"tool_calls": [call(1, Some("b"), Some("two"), "{}")]}),
                None,
            ),
            chunk(&json!({"tool_calls": [call(0, None, None, "1}")]}), None),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        let (name, message) = ending(&events);
        assert_eq!(name, "error", "{events:?}");
        assert!(
            message.contains("tool `one` were not a JSON object"),
            "{message}"
        );
        assert!(!events.iter().any(|e| e.contains(r#""name":"two""#)));
    }

    #[test]
    fn a_call_that_never_gets_a_name_is_an_error_not_a_nameless_tool_use() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), None, "{}")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        assert_eq!(
            ending(&events),
            ("error".into(), "the engine's tool call had no name".into())
        );
        assert!(!events.iter().any(|e| e.contains("tool_use")), "{events:?}");
    }

    #[test]
    fn arguments_that_are_not_a_json_object_are_an_error() {
        let events = pump(&engine_stream(&[
            chunk(
                &json!({"tool_calls": [call(0, Some("c"), Some("run"), "[1]")]}),
                None,
            ),
            chunk(&json!({}), Some("tool_calls")),
        ]));
        let (name, message) = ending(&events);
        assert_eq!(name, "error");
        assert!(
            message.contains("arguments for tool `run` were not a JSON object: [1]"),
            "{message}"
        );
    }

    #[test]
    fn a_call_cut_off_before_its_name_is_dropped_and_the_reply_still_has_a_block() {
        let events = pump(&engine_stream(&[
            chunk(&json!({"tool_calls": [call(0, Some("c"), None, "")]}), None),
            chunk(&json!({}), Some("length")),
        ]));
        assert_eq!(
            blocks(&events),
            [
                ("content_block_start".into(), 0, "text".into()),
                ("content_block_stop".into(), 0, String::new()),
            ]
        );
        assert_eq!(stop_reason(&events), "max_tokens");
    }

    #[test]
    fn usage_after_the_finish_chunk_still_reaches_message_delta() {
        let mut translator = StreamTranslator::new(MODEL);
        translator.feed(&chunk(&json!({"content": "x"}), None));
        assert!(
            translator
                .feed(&chunk(&json!({}), Some("length")))
                .is_empty()
        );
        translator
            .feed(r#"{"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 9}}"#);
        let closing = translator.feed("[DONE]");
        assert_eq!(
            names(&closing),
            ["content_block_stop", "message_delta", "message_stop"]
        );
        assert!(
            closing[1].contains(r#""stop_reason":"max_tokens""#),
            "{}",
            closing[1]
        );
        assert!(closing[1].contains(r#""input_tokens":7"#), "{}", closing[1]);
        assert!(
            closing[1].contains(r#""output_tokens":9"#),
            "{}",
            closing[1]
        );
        assert!(translator.is_finished());
        assert!(translator.feed("[DONE]").is_empty());
    }

    #[test]
    fn a_reply_without_text_is_an_empty_text_block() {
        let mut translator = StreamTranslator::new(MODEL);
        let mut events = translator.feed(&chunk(&json!({}), Some("stop")));
        events.extend(translator.feed("[DONE]"));
        assert_eq!(
            names(&events),
            [
                "message_start",
                "content_block_start",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert!(events[0].contains(MODEL), "{}", events[0]);
    }

    #[test]
    fn done_without_a_stop_reason_is_an_error() {
        let mut translator = StreamTranslator::new(MODEL);
        let events = translator.feed("[DONE]");
        assert_eq!(names(&events), ["error"]);
        assert!(events[0].contains("ended before"), "{}", events[0]);
    }

    #[test]
    fn a_stream_that_ends_early_is_an_error_not_a_complete_reply() {
        let mut translator = StreamTranslator::new(MODEL);
        translator.feed(&chunk(&json!({"content": "x"}), None));
        let events = translator.end_of_stream();
        assert_eq!(names(&events), ["error"]);
        assert!(events[0].contains("ended before"), "{}", events[0]);
        assert!(translator.is_finished());
        // Ending after the finish chunk, without `[DONE]`, is still complete.
        let mut translator = StreamTranslator::new(MODEL);
        translator.feed(&chunk(&json!({"content": "x"}), Some("stop")));
        assert_eq!(
            names(&translator.end_of_stream()),
            ["content_block_stop", "message_delta", "message_stop"]
        );
    }

    #[test]
    fn an_engine_error_mid_stream_becomes_an_error_event_and_ends_the_stream() {
        let mut translator = StreamTranslator::new(MODEL);
        translator.feed(&chunk(&json!({"content": "x"}), None));
        let events = translator.feed(r#"{"error": {"message": "out of memory"}}"#);
        assert_eq!(names(&events), ["error"]);
        assert!(events[0].contains("out of memory"), "{}", events[0]);
        assert!(
            translator
                .feed(&chunk(&json!({"content": "y"}), None))
                .is_empty()
        );
        let events = StreamTranslator::new(MODEL).feed("not json");
        assert_eq!(names(&events), ["error"]);
    }

    /// A stand-in OpenAI engine: lists its model, answers one chat completion,
    /// and records the `Authorization` header and body of what it was sent.
    struct FakeEngine {
        port: u16,
        seen: Arc<Mutex<Vec<(String, String)>>>,
    }

    fn fake_engine() -> FakeEngine {
        fake_engine_with(false)
    }

    /// `native` makes it answer `/v1/messages` the way an engine that serves one
    /// does (refusing an empty body); otherwise that route is a 404.
    fn fake_engine_with(native: bool) -> FakeEngine {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
        let log = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut raw = Vec::new();
                let mut chunk = [0_u8; 4096];
                loop {
                    let read = stream.read(&mut chunk).unwrap_or(0);
                    raw.extend_from_slice(&chunk[..read]);
                    let text = String::from_utf8_lossy(&raw).into_owned();
                    let Some((head, body)) = text.split_once("\r\n\r\n") else {
                        if read == 0 {
                            break;
                        }
                        continue;
                    };
                    let length = head
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                        })
                        .unwrap_or(0);
                    if body.len() >= length || read == 0 {
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&raw).into_owned();
                let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
                let request_line = head.lines().next().unwrap_or_default().to_owned();
                let auth = head
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .starts_with("authorization:")
                            .then(|| line.split_once(':').map(|(_, v)| v.trim().to_owned()))
                            .flatten()
                    })
                    .unwrap_or_default();
                let streaming = body.contains("\"stream\":true");
                let (status, reply) = if request_line.contains("/v1/messages") {
                    if native {
                        ("400 Bad Request", json!({"error": "invalid request"}))
                    } else {
                        ("404 Not Found", json!({"detail": "Not Found"}))
                    }
                } else if auth == "Bearer wrong" {
                    ("401 Unauthorized", json!({"error": {"message": "bad key"}}))
                } else if request_line.contains("/v1/models") {
                    ("200 OK", json!({"data": [{"id": MODEL}]}))
                } else if streaming {
                    log.lock().unwrap().push((auth, body.to_owned()));
                    let chunk = |delta: Value, finish: Value| {
                        format!(
                            "data: {}\n\n",
                            json!({
                                "id": "chatcmpl-s1",
                                "model": MODEL,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                            })
                        )
                    };
                    let usage = json!({
                        "id": "chatcmpl-s1",
                        "model": MODEL,
                        "choices": [],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    });
                    // Offered tools and not yet answered by a tool result, the
                    // engine streams a call the way vLLM does: id and name
                    // first, then the arguments in fragments.
                    let events = if body.contains("\"tools\"")
                        && !body.contains("\"role\":\"tool\"")
                    {
                        let call = |head: Value, arguments: &str| {
                            let mut call =
                                json!({"index": 0, "function": {"arguments": arguments}});
                            if let Value::Object(head) = head {
                                call.as_object_mut().unwrap().extend(head);
                            }
                            json!({"tool_calls": [call]})
                        };
                        format!(
                            "{}{}{}{}data: {usage}\n\ndata: [DONE]\n\n",
                            chunk(
                                call(
                                    json!({"id": "call_9", "type": "function", "function": {"name": "get_weather", "arguments": ""}}),
                                    "",
                                ),
                                Value::Null,
                            ),
                            chunk(call(Value::Null, "{\"city\":"), Value::Null),
                            chunk(call(Value::Null, "\"Oslo\"}"), Value::Null),
                            chunk(json!({}), json!("tool_calls")),
                        )
                    } else {
                        format!(
                            "{}{}{}{}data: {usage}\n\ndata: [DONE]\n\n",
                            chunk(json!({"role": "assistant", "content": ""}), Value::Null),
                            chunk(json!({"content": "po"}), Value::Null),
                            chunk(json!({"content": "ng"}), Value::Null),
                            chunk(json!({}), json!("stop")),
                        )
                    };
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{events}"
                    );
                    continue;
                } else if body.contains("\"tools\"") {
                    log.lock().unwrap().push((auth, body.to_owned()));
                    (
                        "200 OK",
                        json!({
                            "id": "chatcmpl-2",
                            "model": MODEL,
                            "choices": [{
                                "message": {"content": null, "tool_calls": [{
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"},
                                }]},
                                "finish_reason": "tool_calls",
                            }],
                            "usage": {"prompt_tokens": 20, "completion_tokens": 9},
                        }),
                    )
                } else {
                    log.lock().unwrap().push((auth, body.to_owned()));
                    (
                        "200 OK",
                        json!({
                            "id": "chatcmpl-1",
                            "model": MODEL,
                            "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                        }),
                    )
                };
                let reply = reply.to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            }
        });
        FakeEngine { port, seen }
    }

    /// Start the proxy router over a fake engine and return its base URL.
    async fn proxy_over(engine: &FakeEngine) -> String {
        let state = ProxyState {
            upstream: format!("http://127.0.0.1:{}", engine.port),
            served_model: MODEL.to_owned(),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, router(state)).await });
        format!("http://127.0.0.1:{port}")
    }

    /// POST `body` to the proxy and return `(status, json body)`.
    fn post(base: &str, path: &str, key: Option<&str>, body: &Value) -> (u16, Value) {
        let mut request = ureq::post(&format!("{base}{path}"));
        if let Some(key) = key {
            request = request.set("x-api-key", key);
        }
        let parse = |response: ureq::Response| {
            let status = response.status();
            (
                status,
                serde_json::from_str(&response.into_string().unwrap()).unwrap(),
            )
        };
        match request
            .set("content-type", "application/json")
            .send_string(&body.to_string())
        {
            Ok(response) => parse(response),
            Err(ureq::Error::Status(_, response)) => parse(response),
            Err(error) => panic!("{error}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn messages_round_trips_through_the_engine_with_the_clients_key() {
        let engine = fake_engine();
        let base = proxy_over(&engine).await;
        let (status, message) = tokio::task::spawn_blocking(move || {
            post(
                &base,
                "/v1/messages",
                Some("client-key"),
                &json!({
                    "model": "claude-sonnet-4",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "ping"}],
                }),
            )
        })
        .await
        .unwrap();
        assert_eq!(status, 200);
        assert_eq!(message["type"], "message");
        assert_eq!(message["model"], MODEL);
        assert_eq!(message["content"][0]["text"], "pong");
        assert_eq!(message["stop_reason"], "end_turn");
        let seen = engine.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "Bearer client-key");
        let forwarded: Value = serde_json::from_str(&seen[0].1).unwrap();
        assert_eq!(forwarded["model"], MODEL);
        assert_eq!(forwarded["messages"][0]["content"], "ping");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn engine_refusals_reach_the_client_as_anthropic_errors() {
        let engine = fake_engine();
        let base = proxy_over(&engine).await;
        let (status, error) = tokio::task::spawn_blocking(move || {
            post(
                &base,
                "/v1/messages",
                Some("wrong"),
                &json!({"max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}),
            )
        })
        .await
        .unwrap();
        assert_eq!(status, 401);
        assert_eq!(
            error,
            json!({"type": "error", "error": {"type": "authentication_error", "message": "bad key"}})
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn untranslatable_requests_are_refused_before_reaching_the_engine() {
        let engine = fake_engine();
        let base = proxy_over(&engine).await;
        let (status, error) = tokio::task::spawn_blocking(move || {
            post(
                &base,
                "/v1/messages",
                None,
                &json!({"max_tokens": 4, "tools": [{"type": "web_search_20250305", "name": "web_search"}], "messages": [{"role": "user", "content": "x"}]}),
            )
        })
        .await
        .unwrap();
        assert_eq!(status, 400);
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(engine.seen.lock().unwrap().is_empty());
    }

    /// POST a streaming request and return `(status, content-type, raw body)`.
    fn post_streaming(base: &str, key: Option<&str>) -> (u16, String, String) {
        post_streaming_body(
            base,
            key,
            &json!({
                "model": "claude-sonnet-4",
                "max_tokens": 16,
                "stream": true,
                "messages": [{"role": "user", "content": "ping"}],
            }),
        )
    }

    /// POST `body` and return `(status, content-type, raw body)`.
    fn post_streaming_body(base: &str, key: Option<&str>, body: &Value) -> (u16, String, String) {
        let mut request = ureq::post(&format!("{base}/v1/messages"));
        if let Some(key) = key {
            request = request.set("x-api-key", key);
        }
        let read = |response: ureq::Response| {
            (
                response.status(),
                response.content_type().to_owned(),
                response.into_string().unwrap(),
            )
        };
        match request.send_string(&body.to_string()) {
            Ok(response) => read(response),
            Err(ureq::Error::Status(_, response)) => read(response),
            Err(error) => panic!("{error}"),
        }
    }

    /// The `(event name, data)` pairs of a server-sent-event body.
    fn sse_events(body: &str) -> Vec<(String, Value)> {
        body.split("\n\n")
            .filter(|frame| !frame.trim().is_empty())
            .map(|frame| {
                let mut lines = frame.lines();
                let event = lines.next().unwrap().strip_prefix("event: ").unwrap();
                let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
                assert!(lines.next().is_none(), "{frame}");
                (event.to_owned(), serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_streaming_request_is_answered_with_anthropic_server_sent_events() {
        let engine = fake_engine();
        let base = proxy_over(&engine).await;
        let (status, content_type, body) =
            tokio::task::spawn_blocking(move || post_streaming(&base, Some("client-key")))
                .await
                .unwrap();
        assert_eq!((status, content_type.as_str()), (200, "text/event-stream"));
        let events = sse_events(&body);
        let names: Vec<_> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(events[0].1["message"]["id"], "msg_s1");
        assert_eq!(events[0].1["message"]["model"], MODEL);
        let text: String = events
            .iter()
            .filter(|(name, _)| name == "content_block_delta")
            .map(|(_, data)| data["delta"]["text"].as_str().unwrap())
            .collect();
        assert_eq!(text, "pong");
        assert_eq!(events[5].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(events[5].1["usage"]["output_tokens"], 2);
        let seen = engine.seen.lock().unwrap();
        assert_eq!(seen[0].0, "Bearer client-key");
        let forwarded: Value = serde_json::from_str(&seen[0].1).unwrap();
        assert_eq!(forwarded["stream"], true);
        assert_eq!(forwarded["model"], MODEL);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_streamed_tool_call_round_trips_through_the_engine() {
        let engine = fake_engine();
        let base = proxy_over(&engine).await;
        let ask = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 64,
            "stream": true,
            "tools": [weather_tool()],
            "messages": [{"role": "user", "content": "weather in Oslo?"}],
        });
        let first_base = base.clone();
        let (status, content_type, body) =
            tokio::task::spawn_blocking(move || post_streaming_body(&first_base, None, &ask))
                .await
                .unwrap();
        assert_eq!(
            (status, content_type.as_str()),
            (200, "text/event-stream"),
            "{body}"
        );
        let events = sse_events(&body);
        let names: Vec<_> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(
            events[1].1["content_block"],
            json!({"type": "tool_use", "id": "call_9", "name": "get_weather", "input": {}})
        );
        let input: String = events
            .iter()
            .filter_map(|(_, data)| data["delta"]["partial_json"].as_str())
            .collect();
        assert_eq!(input, r#"{"city":"Oslo"}"#);
        assert_eq!(events[5].1["delta"]["stop_reason"], "tool_use");

        // The client sends the call back with its result, still streaming.
        let answer = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 64,
            "stream": true,
            "tools": [weather_tool()],
            "messages": [
                {"role": "user", "content": "weather in Oslo?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_9", "name": "get_weather", "input": {"city": "Oslo"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_9", "content": "sunny"}
                ]},
            ],
        });
        let (status, _, body) =
            tokio::task::spawn_blocking(move || post_streaming_body(&base, None, &answer))
                .await
                .unwrap();
        assert_eq!(status, 200, "{body}");
        let events = sse_events(&body);
        let text: String = events
            .iter()
            .filter_map(|(_, data)| data["delta"]["text"].as_str())
            .collect();
        assert_eq!(text, "pong");
        assert_eq!(events.last().unwrap().0, "message_stop");

        let seen = engine.seen.lock().unwrap();
        let first: Value = serde_json::from_str(&seen[0].1).unwrap();
        assert_eq!(first["stream"], true);
        assert_eq!(first["tools"][0]["function"]["name"], "get_weather");
        let second: Value = serde_json::from_str(&seen[1].1).unwrap();
        let messages = second["messages"].as_array().unwrap();
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_9");
        assert_eq!(
            messages[2],
            json!({"role": "tool", "tool_call_id": "call_9", "content": "sunny"})
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_streaming_request_the_engine_refuses_gets_a_plain_anthropic_error() {
        let engine = fake_engine();
        let base = proxy_over(&engine).await;
        let (status, content_type, body) =
            tokio::task::spawn_blocking(move || post_streaming(&base, Some("wrong")))
                .await
                .unwrap();
        assert_eq!((status, content_type.as_str()), (401, "application/json"));
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["type"], "authentication_error");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_models_auth_failure_is_relayed_not_reported_as_unreachable() {
        let engine = fake_engine();
        let base = proxy_over(&engine).await;
        let (status, error) = tokio::task::spawn_blocking(move || {
            match ureq::get(&format!("{base}/v1/models"))
                .set("x-api-key", "wrong")
                .call()
            {
                Err(ureq::Error::Status(status, response)) => (
                    status,
                    serde_json::from_str::<Value>(&response.into_string().unwrap()).unwrap(),
                ),
                other => panic!("expected an HTTP error, got {other:?}"),
            }
        })
        .await
        .unwrap();
        assert_eq!(status, 401);
        assert_eq!(error["error"]["type"], "authentication_error");
        assert_eq!(error["error"]["message"], "bad key");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn models_are_forwarded_so_readiness_covers_the_proxy() {
        let engine = fake_engine();
        let base = proxy_over(&engine).await;
        let ready = tokio::task::spawn_blocking(move || {
            rocm_core::anthropic_endpoint_ready(&base, Some(MODEL), None, Duration::from_secs(5))
        })
        .await
        .unwrap();
        assert!(ready);
    }

    #[test]
    fn proxy_state_prefers_the_served_name_the_engine_reported() {
        let dir = unique_temp_dir("served");
        std::fs::create_dir_all(&dir).unwrap();
        let mut config = AnthropicProxyConfig {
            host: "127.0.0.1".to_owned(),
            engine_port: 1,
            served_model: "org/image:1".to_owned(),
            state_path: dir.join("state.json"),
            probe_api_key: None,
        };
        assert_eq!(ProxyState::from_config(&config).served_model, "org/image:1");
        std::fs::write(&config.state_path, r#"{"served_model_id": "Org/Served"}"#).unwrap();
        assert_eq!(ProxyState::from_config(&config).served_model, "Org/Served");
        config.state_path = dir.join("missing.json");
        assert_eq!(ProxyState::from_config(&config).served_model, "org/image:1");
        let _ = std::fs::remove_dir_all(dir);
    }

    fn published_endpoint_for(engine: &FakeEngine) -> (String, String) {
        let dir = unique_temp_dir("run");
        std::fs::create_dir_all(&dir).unwrap();
        let state_path = dir.join("state.json");
        spawn_alongside_engine(AnthropicProxyConfig {
            host: "127.0.0.1".to_owned(),
            engine_port: engine.port,
            served_model: MODEL.to_owned(),
            state_path: state_path.clone(),
            probe_api_key: None,
        });
        let url =
            rocm_core::wait_for_published_anthropic_endpoint(&state_path, Duration::from_secs(30))
                .expect("no Anthropic endpoint was published");
        let state: Value = serde_json::from_slice(&std::fs::read(&state_path).unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(dir);
        (
            url,
            state["anthropic_endpoint_mode"]
                .as_str()
                .unwrap()
                .to_owned(),
        )
    }

    #[test]
    fn an_engine_without_the_route_is_fronted_by_a_proxy_on_its_own_port() {
        let engine = fake_engine_with(false);
        let (url, mode) = published_endpoint_for(&engine);
        assert_eq!(mode, "proxy");
        assert_ne!(url, format!("http://127.0.0.1:{}", engine.port));
        let (status, message) = post(
            &url,
            "/v1/messages",
            None,
            &json!({"max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]}),
        );
        assert_eq!(
            (status, message["content"][0]["text"].as_str()),
            (200, Some("pong"))
        );
    }

    #[test]
    fn an_engine_that_serves_the_route_is_still_fronted_by_the_proxy() {
        // vLLM and Lemonade answer `/v1/messages` themselves but reject any model
        // name other than the one they serve; the proxy answers under any name.
        let engine = fake_engine_with(true);
        let (url, mode) = published_endpoint_for(&engine);
        assert_eq!(mode, "proxy");
        assert_ne!(url, format!("http://127.0.0.1:{}", engine.port));
        let (status, message) = post(
            &url,
            "/v1/messages",
            None,
            &json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "ping"}],
            }),
        );
        assert_eq!(
            (status, message["content"][0]["text"].as_str()),
            (200, Some("pong"))
        );
    }

    fn publish_config(dir: &std::path::Path) -> AnthropicProxyConfig {
        std::fs::create_dir_all(dir).unwrap();
        AnthropicProxyConfig {
            host: "127.0.0.1".to_owned(),
            engine_port: 1,
            served_model: MODEL.to_owned(),
            state_path: dir.join("state.json"),
            probe_api_key: None,
        }
    }

    /// A state file caught mid-write by an engine that rewrites it in place is
    /// refused, and the publish tries again, after the production pause, once
    /// it is whole.
    #[test]
    fn publish_retries_a_state_file_caught_mid_write() {
        let dir = unique_temp_dir("publish-retry");
        let config = publish_config(&dir);
        std::fs::write(&config.state_path, b"").unwrap();
        let mut pauses = Vec::new();

        let result = publish_with_pause(
            &config,
            "http://127.0.0.1:8000/v1",
            AnthropicEndpointMode::Proxy,
            |pause| {
                pauses.push(pause);
                std::fs::write(&config.state_path, br#"{"status": "ready"}"#).unwrap();
            },
        );

        let state: Value =
            serde_json::from_slice(&std::fs::read(&config.state_path).unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(dir);
        result.expect("the second attempt publishes");
        assert_eq!(pauses, [Duration::from_millis(200)]);
        assert_eq!(state["status"], "ready", "{state}");
        assert_eq!(state["anthropic_endpoint_url"], "http://127.0.0.1:8000");
        assert_eq!(state["anthropic_endpoint_mode"], "proxy");
    }

    /// The production `publish`, against a state file that stays torn: five
    /// attempts with a real 200 ms pause between each, so it cannot give up
    /// in less than 800 ms; then an error naming the file, which is left as
    /// it was.
    #[test]
    fn publish_gives_up_on_a_state_file_that_stays_torn() {
        let dir = unique_temp_dir("publish-gives-up");
        let config = publish_config(&dir);
        let torn = br#"{"status": "ready", "server_pid": 4"#;
        std::fs::write(&config.state_path, torn).unwrap();
        let started = std::time::Instant::now();

        let result = publish(
            &config,
            "http://127.0.0.1:8000/v1",
            AnthropicEndpointMode::Proxy,
        );

        let elapsed = started.elapsed();
        let after = std::fs::read(&config.state_path).unwrap();
        let error = format!("{:#}", result.expect_err("a torn file is refused"));
        assert!(
            error.starts_with(&format!(
                "could not publish the Anthropic endpoint to {} after 5 attempts: ",
                config.state_path.display()
            )),
            "{error}"
        );
        assert!(error.contains("is not valid JSON"), "{error}");
        let _ = std::fs::remove_dir_all(dir);
        assert!(
            elapsed >= Duration::from_millis(800),
            "four 200 ms pauses between five attempts, took {elapsed:?}"
        );
        assert_eq!(after, torn, "the refused file is left as it was");
    }

    /// A failure a mid-write cannot explain is returned at once, without a
    /// pause: a whole document that is not an object, and a state file whose
    /// lock cannot be taken.
    #[test]
    fn publish_does_not_retry_a_failure_a_mid_write_cannot_explain() {
        let dir = unique_temp_dir("publish-no-retry");
        let config = publish_config(&dir);
        std::fs::write(&config.state_path, b"[1, 2]").unwrap();
        // A regular file where the state directory should be, so the lock file
        // beside the state file cannot be created.
        let blocked = AnthropicProxyConfig {
            state_path: dir.join("not-a-dir").join("state.json"),
            ..publish_config(&dir)
        };
        std::fs::write(dir.join("not-a-dir"), b"").unwrap();

        let mut pauses = Vec::new();
        let not_an_object = publish_with_pause(
            &config,
            "http://127.0.0.1:8000/v1",
            AnthropicEndpointMode::Proxy,
            |pause| pauses.push(pause),
        );
        let unlockable = publish_with_pause(
            &blocked,
            "http://127.0.0.1:8000/v1",
            AnthropicEndpointMode::Proxy,
            |pause| pauses.push(pause),
        );

        let after = std::fs::read(&config.state_path).unwrap();
        let _ = std::fs::remove_dir_all(dir);
        assert!(pauses.is_empty(), "no retry: {pauses:?}");
        let error = format!("{:#}", not_an_object.expect_err("an array is refused"));
        assert!(
            error.starts_with(&format!(
                "could not publish the Anthropic endpoint to {} after 1 attempt: ",
                config.state_path.display()
            )),
            "{error}"
        );
        assert!(
            error.contains("holds a JSON array, not an object"),
            "{error}"
        );
        assert_eq!(after, b"[1, 2]", "the refused file is left as it was");
        let error = format!("{:#}", unlockable.expect_err("the lock cannot be taken"));
        assert!(
            error.starts_with(&format!(
                "could not publish the Anthropic endpoint to {} after 1 attempt: ",
                blocked.state_path.display()
            )),
            "{error}"
        );
    }

    #[test]
    fn publish_records_the_endpoint_without_a_v1_suffix() {
        let dir = unique_temp_dir("publish");
        let config = AnthropicProxyConfig {
            host: "127.0.0.1".to_owned(),
            engine_port: 1,
            served_model: MODEL.to_owned(),
            state_path: dir.join("state.json"),
            probe_api_key: None,
        };
        publish(
            &config,
            "http://127.0.0.1:8000/v1",
            AnthropicEndpointMode::Proxy,
        )
        .unwrap();
        let state: Value =
            serde_json::from_slice(&std::fs::read(&config.state_path).unwrap()).unwrap();
        assert_eq!(state["anthropic_endpoint_url"], "http://127.0.0.1:8000");
        assert_eq!(state["anthropic_endpoint_mode"], "proxy");
        let _ = std::fs::remove_dir_all(dir);
    }
}
