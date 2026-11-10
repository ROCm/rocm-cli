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
//! The translation is the text subset of the Messages API: system prompts and
//! text turns, sampling parameters, stop sequences, usage and stop reasons.
//! A `stream: true` request is answered with Anthropic server-sent events
//! (`message_start` through `message_stop`) built from the engine's streamed
//! chat-completion chunks as they arrive. Tool use (`tools`, `tool_choice`),
//! extended thinking and non-text content blocks are refused with an Anthropic
//! `invalid_request_error` that says so, rather than being dropped silently.
//! The engine reports token usage only at the end of a stream, so `message_start`
//! carries `input_tokens: 0` and the real figures arrive in `message_delta`.
//! Fields that only annotate a request (`metadata`) are ignored.

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
    format_http_base_url, http_get_with_auth, http_post_json_with_auth, merge_json_state_file,
    openai_models_endpoint_has_model,
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

fn publish(
    config: &AnthropicProxyConfig,
    url: &str,
    mode: AnthropicEndpointMode,
) -> anyhow::Result<()> {
    // The listener URL carries no `/v1`: Anthropic SDKs append `/v1/messages`.
    let base = url.trim_end_matches('/').trim_end_matches("/v1");
    merge_json_state_file(
        &config.state_path,
        &json!({
            ANTHROPIC_ENDPOINT_URL_STATE_KEY: base,
            ANTHROPIC_ENDPOINT_MODE_STATE_KEY: mode.as_str(),
        }),
    )
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
        // Events are only sent when a chunk carries text, so a client that left
        // during a long reasoning phase would otherwise go unnoticed.
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

/// Translate an Anthropic Messages request into an OpenAI chat-completions one.
fn anthropic_to_openai(request: &Value, served_model: &str) -> Result<Value, ProxyError> {
    let object = request
        .as_object()
        .ok_or_else(|| ProxyError::invalid("the request body must be a JSON object"))?;
    let stream = object.get("stream").and_then(Value::as_bool) == Some(true);
    if object
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(ProxyError::unsupported("tool use (`tools`)"));
    }
    if object
        .get("tool_choice")
        .is_some_and(|value| !value.is_null())
    {
        return Err(ProxyError::unsupported("`tool_choice`"));
    }
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
    if let Some(system) = object.get("system").filter(|value| !value.is_null()) {
        let text = text_of(system, "system")?;
        if !text.is_empty() {
            messages.push(json!({ "role": "system", "content": text }));
        }
    }
    for turn in turns {
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
        messages.push(json!({ "role": role, "content": text_of(content, "messages")? }));
    }

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
    let stop_reason = match choice.get("finish_reason").and_then(Value::as_str) {
        Some("length") => "max_tokens",
        _ => "end_turn",
    };
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
        "content": [{ "type": "text", "text": text }],
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": token_count("prompt_tokens"),
            "output_tokens": token_count("completion_tokens"),
        },
    }))
}

/// One Anthropic server-sent event.
fn sse_event(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// Turns an engine's streamed chat-completion chunks into the Anthropic
/// streaming event sequence: `message_start`, one text `content_block_start`,
/// a `content_block_delta` per text fragment, then `content_block_stop`,
/// `message_delta` and `message_stop` once the engine is done.
///
/// The closing events wait for the engine's `[DONE]` (or end of stream) rather
/// than its `finish_reason` chunk, because the usage totals arrive in a chunk
/// after it.
struct StreamTranslator {
    served_model: String,
    started: bool,
    block_open: bool,
    finished: bool,
    stop_reason: Option<&'static str>,
    input_tokens: u64,
    output_tokens: u64,
}

impl StreamTranslator {
    fn new(served_model: &str) -> Self {
        Self {
            served_model: served_model.to_owned(),
            started: false,
            block_open: false,
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
        {
            if let Some(text) = choice
                .pointer("/delta/content")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            {
                events.extend(self.open_block());
                events.push(sse_event(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": { "type": "text_delta", "text": text },
                    }),
                ));
            }
            match choice.get("finish_reason").and_then(Value::as_str) {
                Some("length") => self.stop_reason = Some("max_tokens"),
                Some(_) => self.stop_reason = Some("end_turn"),
                None => {}
            }
        }
        events
    }

    /// The events for an engine stream that is over: complete if a stop reason
    /// arrived, an `error` otherwise.
    fn end_of_stream(&mut self) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }
        if self.stop_reason.is_some() {
            return self.complete();
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

    fn open_block(&mut self) -> Option<String> {
        if self.block_open {
            return None;
        }
        self.block_open = true;
        Some(sse_event(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" },
            }),
        ))
    }

    /// The closing events. An engine that produced no text still gets an empty
    /// text block, matching the non-streaming reply.
    fn complete(&mut self) -> Vec<String> {
        let mut events = Vec::new();
        if !self.started {
            let model = self.served_model.clone();
            events.push(self.start("msg_rocm", &model));
        }
        events.extend(self.open_block());
        events.push(sse_event(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 0 }),
        ));
        events.push(sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": self.stop_reason.unwrap_or("end_turn"),
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
        let mut tools = base();
        tools["tools"] = json!([{"name": "get_weather", "input_schema": {"type": "object"}}]);
        let image = json!({"max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}
        ]}]});
        for (request, needle) in [(tools, "tool use"), (image, "`image` content block")] {
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
        for (field, value) in [
            ("tool_choice", json!({"type": "auto"})),
            (
                "thinking",
                json!({"type": "enabled", "budget_tokens": 1024}),
            ),
        ] {
            let mut request = base.clone();
            request[field] = value;
            let error = translate(&request).unwrap_err();
            assert_eq!(error.status, 400, "{field}");
        }
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
                    let events = format!(
                        "{}{}{}{}data: {usage}\n\ndata: [DONE]\n\n",
                        chunk(json!({"role": "assistant", "content": ""}), Value::Null),
                        chunk(json!({"content": "po"}), Value::Null),
                        chunk(json!({"content": "ng"}), Value::Null),
                        chunk(json!({}), json!("stop")),
                    );
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{events}"
                    );
                    continue;
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
                &json!({"max_tokens": 4, "tools": [{"name": "t"}], "messages": [{"role": "user", "content": "x"}]}),
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
        let mut request = ureq::post(&format!("{base}/v1/messages"));
        if let Some(key) = key {
            request = request.set("x-api-key", key);
        }
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 16,
            "stream": true,
            "messages": [{"role": "user", "content": "ping"}],
        });
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
