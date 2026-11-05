// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The provider wire formats: the OpenAI-compatible and Anthropic chat request
//! bodies (including the local-model request defaults), the parsers for their
//! non-streaming responses, and the SSE emitters that turn a streamed response
//! into `ProviderStreamEvent`s.
//!
//! Depends on the shared request/response types in `providers` and on the ROCm
//! tool schema in `tool_schema`; the local HTTP transport in `local_http`
//! reads [`OpenAiSseEmitter`] from here, so the dependency is one-way.

use super::tool_schema::{rocm_anthropic_tool_definitions, rocm_openai_tool_definitions};
use super::{ChatMessage, ChatRequest, ChatToolCall, ProviderStreamEvent, SseBytesEmitter};
use anyhow::{Context, Result, bail};

pub(super) fn openai_chat_request_body(model: &str, request: &ChatRequest) -> serde_json::Value {
    openai_chat_request_body_with_stream(model, request, false)
}

pub(super) fn openai_stream_chat_request_body(
    model: &str,
    request: &ChatRequest,
) -> serde_json::Value {
    openai_chat_request_body_with_stream(model, request, true)
}

pub(super) fn openai_local_chat_request_body(
    model: &str,
    request: &ChatRequest,
) -> serde_json::Value {
    let request = local_openai_compatible_request(request);
    let mut body = openai_chat_request_body(model, &request);
    apply_local_model_request_defaults(model, &mut body);
    body
}

pub(super) fn openai_local_stream_chat_request_body(
    model: &str,
    request: &ChatRequest,
) -> serde_json::Value {
    let request = local_openai_compatible_request(request);
    let mut body = openai_stream_chat_request_body(model, &request);
    apply_local_model_request_defaults(model, &mut body);
    body
}

/// The chat-template rule itself lives in
/// [`rocm_dash_core::chat_template`], not here: the dashboard's chat client
/// sends the same request to the same locally served model and has to decide
/// this identically, and a rule kept in two places drifts.
fn apply_local_model_request_defaults(model: &str, body: &mut serde_json::Value) {
    if rocm_dash_core::chat_template::local_model_should_disable_thinking(model) {
        body["chat_template_kwargs"] = serde_json::json!({
            "enable_thinking": false
        });
    }
}

fn local_openai_compatible_request(request: &ChatRequest) -> ChatRequest {
    let system = request
        .messages
        .iter()
        .filter(|message| message.role == "system")
        .map(|message| message.content.trim())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut messages = request
        .messages
        .iter()
        .filter(|message| message.role != "system")
        .cloned()
        .collect::<Vec<_>>();
    if !system.is_empty() {
        if let Some(first_user) = messages.iter_mut().find(|message| message.role == "user") {
            first_user.content = format!(
                "ROCm CLI guidance:\n{system}\n\nUser message:\n{}",
                first_user.content
            );
        } else {
            messages.insert(
                0,
                ChatMessage {
                    role: "user".to_owned(),
                    content: format!("ROCm CLI guidance:\n{system}"),
                },
            );
        }
    }
    ChatRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        rocm_tools: request.rocm_tools,
    }
}

fn openai_chat_request_body_with_stream(
    model: &str,
    request: &ChatRequest,
    stream: bool,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": request.messages.iter().map(|message| {
            serde_json::json!({
                "role": message.role,
                "content": message.content,
            })
        }).collect::<Vec<_>>(),
        "stream": stream,
        "max_tokens": request.max_tokens.unwrap_or(512),
    });
    if let Some(temperature) = request.temperature {
        body["temperature"] = serde_json::json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if request.rocm_tools && !stream {
        body["tools"] = serde_json::Value::Array(rocm_openai_tool_definitions());
        body["tool_choice"] = serde_json::json!("auto");
    }
    body
}

pub(super) fn anthropic_chat_request_body(model: &str, request: &ChatRequest) -> serde_json::Value {
    anthropic_chat_request_body_with_stream(model, request, false)
}

pub(super) fn anthropic_stream_chat_request_body(
    model: &str,
    request: &ChatRequest,
) -> serde_json::Value {
    anthropic_chat_request_body_with_stream(model, request, true)
}

fn anthropic_chat_request_body_with_stream(
    model: &str,
    request: &ChatRequest,
    stream: bool,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": request.messages.iter().filter(|message| message.role != "system").map(|message| {
            serde_json::json!({
                "role": if message.role == "assistant" { "assistant" } else { "user" },
                "content": message.content,
            })
        }).collect::<Vec<_>>(),
        "max_tokens": request.max_tokens.unwrap_or(512),
        "stream": stream,
    });
    let system = request
        .messages
        .iter()
        .filter(|message| message.role == "system")
        .map(|message| message.content.trim())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if !system.is_empty() {
        body["system"] = serde_json::Value::String(system);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = serde_json::json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if request.rocm_tools && !stream {
        body["tools"] = serde_json::Value::Array(rocm_anthropic_tool_definitions());
    }
    body
}

#[cfg(test)]
fn parse_openai_chat_content(response: &serde_json::Value) -> Result<String> {
    let (content, tool_calls) = parse_openai_chat_message(response)?;
    if content.is_empty() && !tool_calls.is_empty() {
        bail!("OpenAI-compatible response contained tool calls but no text content");
    }
    Ok(content)
}

pub(super) fn parse_openai_chat_message(
    response: &serde_json::Value,
) -> Result<(String, Vec<ChatToolCall>)> {
    let message = response
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .context("OpenAI-compatible response did not include choices[0].message")?;
    let content = message
        .get("content")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_default();
    let tool_calls = parse_openai_tool_calls(message)?;
    if content.is_empty() && tool_calls.is_empty() {
        bail!("OpenAI-compatible response did not include message content or tool calls");
    }
    Ok((content, tool_calls))
}

fn parse_openai_tool_calls(message: &serde_json::Value) -> Result<Vec<ChatToolCall>> {
    let Some(items) = message
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(Vec::new());
    };
    let mut calls = Vec::new();
    for item in items {
        let function = item
            .get("function")
            .context("OpenAI-compatible tool call was missing function")?;
        let name = function
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .context("OpenAI-compatible tool call was missing function.name")?;
        let arguments = match function.get("arguments") {
            Some(serde_json::Value::String(text)) if text.trim().is_empty() => {
                serde_json::json!({})
            }
            Some(serde_json::Value::String(text)) => serde_json::from_str(text)
                .with_context(|| format!("tool call `{name}` arguments were not valid JSON"))?,
            Some(value) => value.clone(),
            None => serde_json::json!({}),
        };
        calls.push(ChatToolCall {
            id: item
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            name,
            arguments,
        });
    }
    Ok(calls)
}

#[cfg(test)]
fn parse_anthropic_chat_content(response: &serde_json::Value) -> Result<String> {
    let (content, tool_calls) = parse_anthropic_chat_message(response)?;
    if content.is_empty() && !tool_calls.is_empty() {
        bail!("anthropic provider response contained tool calls but no text content");
    }
    Ok(content)
}

pub(super) fn parse_anthropic_chat_message(
    response: &serde_json::Value,
) -> Result<(String, Vec<ChatToolCall>)> {
    let items = response
        .get("content")
        .and_then(serde_json::Value::as_array)
        .context("anthropic provider response did not include content blocks")?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for item in items {
        match item.get("type").and_then(serde_json::Value::as_str) {
            Some("text") => {
                if let Some(chunk) = item.get("text").and_then(serde_json::Value::as_str) {
                    text.push_str(chunk);
                }
            }
            Some("tool_use") => {
                let name = item
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .context("anthropic provider tool_use block was missing name")?;
                let arguments = item
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                tool_calls.push(ChatToolCall {
                    id: item
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned),
                    name,
                    arguments,
                });
            }
            _ => {}
        }
    }
    if text.is_empty() && tool_calls.is_empty() {
        bail!("anthropic provider response did not include text content or tool calls");
    }
    Ok((text, tool_calls))
}

pub(super) struct OpenAiSseEmitter<'a> {
    pending: Vec<u8>,
    done: bool,
    on_event: &'a mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
}

impl<'a> OpenAiSseEmitter<'a> {
    pub(super) fn new(on_event: &'a mut dyn FnMut(ProviderStreamEvent) -> Result<()>) -> Self {
        Self {
            pending: Vec::new(),
            done: false,
            on_event,
        }
    }

    pub(super) fn finish(mut self) -> Result<()> {
        if !self.pending.is_empty() {
            self.emit_pending_line()?;
        }
        if !self.done {
            (self.on_event)(ProviderStreamEvent {
                content: String::new(),
                done: true,
            })?;
            self.done = true;
        }
        Ok(())
    }

    fn emit_pending_line(&mut self) -> Result<()> {
        let line = String::from_utf8_lossy(&self.pending)
            .trim_end_matches('\r')
            .to_owned();
        self.pending.clear();
        self.emit_line(&line)
    }

    fn emit_line(&mut self, line: &str) -> Result<()> {
        if self.done {
            return Ok(());
        }
        let Some(event) = parse_openai_sse_line(line)? else {
            return Ok(());
        };
        if event.done {
            self.done = true;
        }
        (self.on_event)(event)
    }
}

impl SseBytesEmitter for OpenAiSseEmitter<'_> {
    fn push_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        for byte in bytes {
            if *byte == b'\n' {
                self.emit_pending_line()?;
            } else {
                self.pending.push(*byte);
            }
        }
        Ok(())
    }
}

pub(super) struct AnthropicSseEmitter<'a> {
    pending: Vec<u8>,
    done: bool,
    on_event: &'a mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
}

impl<'a> AnthropicSseEmitter<'a> {
    pub(super) fn new(on_event: &'a mut dyn FnMut(ProviderStreamEvent) -> Result<()>) -> Self {
        Self {
            pending: Vec::new(),
            done: false,
            on_event,
        }
    }

    pub(super) fn finish(mut self) -> Result<()> {
        if !self.pending.is_empty() {
            self.emit_pending_line()?;
        }
        if !self.done {
            (self.on_event)(ProviderStreamEvent {
                content: String::new(),
                done: true,
            })?;
            self.done = true;
        }
        Ok(())
    }

    fn emit_pending_line(&mut self) -> Result<()> {
        let line = String::from_utf8_lossy(&self.pending)
            .trim_end_matches('\r')
            .to_owned();
        self.pending.clear();
        self.emit_line(&line)
    }

    fn emit_line(&mut self, line: &str) -> Result<()> {
        if self.done {
            return Ok(());
        }
        let Some(event) = parse_anthropic_sse_line(line)? else {
            return Ok(());
        };
        if event.done {
            self.done = true;
        }
        (self.on_event)(event)
    }
}

impl SseBytesEmitter for AnthropicSseEmitter<'_> {
    fn push_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        for byte in bytes {
            if *byte == b'\n' {
                self.emit_pending_line()?;
            } else {
                self.pending.push(*byte);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
fn parse_openai_sse_chat_events(body: &str) -> Result<Vec<ProviderStreamEvent>> {
    let mut events = Vec::new();
    for line in body.lines() {
        let Some(event) = parse_openai_sse_line(line)? else {
            continue;
        };
        events.push(event);
    }
    if !events.iter().any(|event| event.done) {
        events.push(ProviderStreamEvent {
            content: String::new(),
            done: true,
        });
    }
    Ok(events)
}

fn parse_openai_sse_line(line: &str) -> Result<Option<ProviderStreamEvent>> {
    let line = line.trim();
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(Some(ProviderStreamEvent {
            content: String::new(),
            done: true,
        }));
    }
    let payload = serde_json::from_str::<serde_json::Value>(data)
        .context("failed to parse local provider SSE data")?;
    let content = payload
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("content"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            payload
                .get("choices")
                .and_then(serde_json::Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(serde_json::Value::as_str)
        });
    Ok(content.map(|content| ProviderStreamEvent {
        content: content.to_owned(),
        done: false,
    }))
}

fn parse_anthropic_sse_line(line: &str) -> Result<Option<ProviderStreamEvent>> {
    let line = line.trim();
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(Some(ProviderStreamEvent {
            content: String::new(),
            done: true,
        }));
    }
    let payload = serde_json::from_str::<serde_json::Value>(data)
        .context("failed to parse Anthropic provider SSE data")?;
    match payload.get("type").and_then(serde_json::Value::as_str) {
        Some("content_block_delta") => Ok(payload
            .get("delta")
            .and_then(|delta| delta.get("text"))
            .and_then(serde_json::Value::as_str)
            .map(|content| ProviderStreamEvent {
                content: content.to_owned(),
                done: false,
            })),
        Some("content_block_start") => Ok(payload
            .get("content_block")
            .and_then(|block| block.get("text"))
            .and_then(serde_json::Value::as_str)
            .filter(|content| !content.is_empty())
            .map(|content| ProviderStreamEvent {
                content: content.to_owned(),
                done: false,
            })),
        Some("message_stop") => Ok(Some(ProviderStreamEvent {
            content: String::new(),
            done: true,
        })),
        Some("error") => {
            let message = payload
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown Anthropic stream error");
            bail!("Anthropic provider stream failed: {message}");
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod sse_emitter_proptests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_chat_content() -> Result<()> {
        let payload = serde_json::json!({
            "choices": [
                {
                    "message": {
                        "content": "hello from local"
                    }
                }
            ]
        });
        assert_eq!(parse_openai_chat_content(&payload)?, "hello from local");
        Ok(())
    }

    #[test]
    fn parses_openai_chat_tool_calls_without_text() -> Result<()> {
        let payload = serde_json::json!({
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "install_sdk_dry_run",
                                    "arguments": "{\"channel\":\"release\",\"format\":\"pip\"}"
                                }
                            }
                        ]
                    }
                }
            ]
        });
        let (content, calls) = parse_openai_chat_message(&payload)?;
        assert!(content.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call-1"));
        assert_eq!(calls[0].name, "install_sdk_dry_run");
        assert_eq!(calls[0].arguments["channel"], "release");
        let error = parse_openai_chat_content(&payload).unwrap_err().to_string();
        assert!(error.contains("tool calls but no text content"));
        Ok(())
    }

    #[test]
    fn parses_anthropic_chat_content() -> Result<()> {
        let payload = serde_json::json!({
            "content": [
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "from anthropic"}
            ]
        });
        assert_eq!(
            parse_anthropic_chat_content(&payload)?,
            "hello from anthropic"
        );
        Ok(())
    }

    #[test]
    fn parses_anthropic_tool_use_without_text() -> Result<()> {
        let payload = serde_json::json!({
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "install_sdk_dry_run",
                    "input": {
                        "channel": "release",
                        "format": "wheel"
                    }
                }
            ]
        });

        let (content, calls) = parse_anthropic_chat_message(&payload)?;

        assert!(content.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("toolu_1"));
        assert_eq!(calls[0].name, "install_sdk_dry_run");
        assert_eq!(calls[0].arguments["channel"], "release");
        let error = parse_anthropic_chat_content(&payload)
            .unwrap_err()
            .to_string();
        assert!(error.contains("tool calls but no text content"));
        Ok(())
    }

    #[test]
    fn openai_and_anthropic_request_bodies_include_model_and_messages() {
        let request = ChatRequest {
            model: Some("model-a".to_owned()),
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: "hello".to_owned(),
            }],
            max_tokens: Some(42),
            rocm_tools: false,
            ..Default::default()
        };
        let openai = openai_chat_request_body("model-a", &request);
        assert_eq!(openai["model"], "model-a");
        assert_eq!(openai["messages"][0]["content"], "hello");
        assert_eq!(openai["stream"], false);
        assert!(openai.get("tools").is_none());
        let anthropic = anthropic_chat_request_body("claude-test", &request);
        assert_eq!(anthropic["model"], "claude-test");
        assert_eq!(anthropic["messages"][0]["role"], "user");
        assert_eq!(anthropic["max_tokens"], 42);
    }

    #[test]
    fn request_bodies_include_temperature_and_top_p_only_when_set() {
        let base = ChatRequest {
            model: Some("model-a".to_owned()),
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: "hello".to_owned(),
            }],
            max_tokens: Some(42),
            rocm_tools: false,
            ..Default::default()
        };

        // Omitted by default so provider/server defaults are preserved.
        let openai_default = openai_chat_request_body("model-a", &base);
        assert!(openai_default.get("temperature").is_none());
        assert!(openai_default.get("top_p").is_none());
        let anthropic_default = anthropic_chat_request_body("claude-test", &base);
        assert!(anthropic_default.get("temperature").is_none());
        assert!(anthropic_default.get("top_p").is_none());

        let tuned = ChatRequest {
            temperature: Some(0.25),
            top_p: Some(0.5),
            ..base
        };
        let openai = openai_chat_request_body("model-a", &tuned);
        assert_eq!(openai["temperature"], 0.25);
        assert_eq!(openai["top_p"], 0.5);
        let anthropic = anthropic_chat_request_body("claude-test", &tuned);
        assert_eq!(anthropic["temperature"], 0.25);
        assert_eq!(anthropic["top_p"], 0.5);
    }

    #[test]
    fn openai_local_qwen3_request_disables_thinking_for_visible_answers() {
        let request = ChatRequest {
            model: Some("Qwen3-0.6B-GGUF".to_owned()),
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: "hello".to_owned(),
            }],
            max_tokens: Some(16),
            rocm_tools: false,
            ..Default::default()
        };

        let body = openai_local_chat_request_body("Qwen3-0.6B-GGUF", &request);

        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn openai_request_body_includes_rocm_tools_only_for_non_streaming_tool_mode() {
        let request = ChatRequest {
            model: Some("model-a".to_owned()),
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: "check this host".to_owned(),
            }],
            max_tokens: Some(42),
            rocm_tools: true,
            ..Default::default()
        };
        let non_streaming = openai_chat_request_body("model-a", &request);
        assert_eq!(non_streaming["tool_choice"], "auto");
        let tools = non_streaming["tools"]
            .as_array()
            .expect("tools should be set");
        let tool_names = tools
            .iter()
            .filter_map(|tool| {
                tool.get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(serde_json::Value::as_str)
            })
            .collect::<Vec<_>>();
        for name in [
            "bridge_snapshot",
            "service_logs",
            "natural_language_plan",
            "path_exists",
            "port_status",
            "rocm_command",
            "update_check",
            "install_sdk_dry_run",
        ] {
            assert!(
                tool_names.contains(&name),
                "missing ROCm chat tool `{name}`"
            );
        }
        for name in [
            "install_sdk",
            "install_engine",
            "launch_server",
            "stop_server",
            "watcher_enable",
            "watcher_disable",
        ] {
            assert!(
                !tool_names.contains(&name),
                "mutating ROCm chat tool `{name}` should be requested through rocm_command"
            );
        }

        let streaming = openai_stream_chat_request_body("model-a", &request);
        assert!(streaming.get("tools").is_none());
        assert!(streaming.get("tool_choice").is_none());
    }

    #[test]
    fn anthropic_request_body_includes_rocm_tools_and_system_only_for_non_streaming_tool_mode() {
        let request = ChatRequest {
            model: Some("model-a".to_owned()),
            messages: vec![
                ChatMessage {
                    role: "system".to_owned(),
                    content: "Use ROCm tools only when useful.".to_owned(),
                },
                ChatMessage {
                    role: "user".to_owned(),
                    content: "check this host".to_owned(),
                },
            ],
            max_tokens: Some(42),
            rocm_tools: true,
            ..Default::default()
        };

        let non_streaming = anthropic_chat_request_body("claude-test", &request);

        assert_eq!(non_streaming["system"], "Use ROCm tools only when useful.");
        assert_eq!(non_streaming["messages"][0]["role"], "user");
        assert_eq!(non_streaming["messages"][0]["content"], "check this host");
        assert_eq!(non_streaming["messages"].as_array().unwrap().len(), 1);
        let tools = non_streaming["tools"]
            .as_array()
            .expect("tools should be set");
        let tool_names = tools
            .iter()
            .filter_map(|tool| tool.get("name").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>();
        for name in [
            "bridge_snapshot",
            "service_logs",
            "natural_language_plan",
            "path_exists",
            "port_status",
            "rocm_command",
            "update_check",
            "install_sdk_dry_run",
        ] {
            assert!(
                tool_names.contains(&name),
                "missing ROCm chat tool `{name}`"
            );
        }
        for name in [
            "install_sdk",
            "install_engine",
            "launch_server",
            "stop_server",
            "watcher_enable",
            "watcher_disable",
        ] {
            assert!(
                !tool_names.contains(&name),
                "mutating ROCm chat tool `{name}` should be requested through rocm_command"
            );
        }
        assert!(tools.iter().any(|tool| {
            tool.get("name").and_then(serde_json::Value::as_str) == Some("install_sdk_dry_run")
                && tool
                    .get("input_schema")
                    .and_then(|schema| schema.get("type"))
                    .and_then(serde_json::Value::as_str)
                    == Some("object")
                && tool
                    .get("input_schema")
                    .and_then(|schema| schema.get("properties"))
                    .and_then(|properties| properties.get("build_date"))
                    .is_some()
        }));

        let streaming = anthropic_stream_chat_request_body("claude-test", &request);
        assert!(streaming.get("tools").is_none());
        assert_eq!(streaming["system"], "Use ROCm tools only when useful.");
    }

    #[test]
    fn parses_openai_sse_chat_events() -> Result<()> {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n";
        let events = parse_openai_sse_chat_events(body)?;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].content, "hel");
        assert!(!events[0].done);
        assert_eq!(events[1].content, "lo");
        assert!(events[2].done);
        Ok(())
    }
}
