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
//! the reply. Streaming, extended thinking, server tools and other non-text
//! content blocks (images) are refused with an Anthropic `invalid_request_error`
//! that says so, rather than being dropped silently. Fields that only annotate a
//! request (`metadata`) are ignored.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use axum::body::Bytes;
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
    let result =
        tokio::task::spawn_blocking(move || forward_messages(&state, key.as_deref(), &body))
            .await
            .unwrap_or_else(|error| {
                Err(ProxyError {
                    status: 500,
                    kind: "api_error",
                    message: format!("the proxy task failed: {error}"),
                })
            });
    match result {
        Ok(message) => Json(message).into_response(),
        Err(error) => error.into_response(),
    }
}

fn forward_messages(
    state: &ProxyState,
    key: Option<&str>,
    body: &[u8],
) -> Result<Value, ProxyError> {
    let request: Value = serde_json::from_slice(body)
        .map_err(|error| ProxyError::invalid(format!("the request body is not JSON: {error}")))?;
    let upstream_request = anthropic_to_openai(&request, &state.served_model)?;
    let response = http_post_json_with_auth(
        &state.upstream,
        "/v1/chat/completions",
        &upstream_request,
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
        Err(error) => ProxyError {
            status: 500,
            kind: "api_error",
            message: format!("the proxy task failed: {error}"),
        }
        .into_response(),
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
    if object.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(ProxyError::unsupported("streaming (`stream: true`)"));
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
        push_turn(&mut messages, role, content)?;
    }

    let tools = translate_tools(object.get("tools"))?;
    let mut out = json!({
        "model": served_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": false,
    });
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
        let mut streaming = base();
        streaming["stream"] = json!(true);
        let mut server_tool = base();
        server_tool["tools"] = json!([{"type": "web_search_20250305", "name": "web_search"}]);
        let image = json!({"max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}
        ]}]});
        for (request, needle) in [
            (streaming, "streaming"),
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
                &json!({"max_tokens": 4, "stream": true, "messages": [{"role": "user", "content": "x"}]}),
            )
        })
        .await
        .unwrap();
        assert_eq!(status, 400);
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(engine.seen.lock().unwrap().is_empty());
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
