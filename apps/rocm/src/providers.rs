// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AuditEventRecord, ManagedServiceRecord, RocmCliConfig, append_audit_event,
    connect_tcp_stream, format_host_port, managed_service_endpoint_model_ready,
    read_tcp_stream_to_string, unix_time_millis, write_all_tcp_stream,
};
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::time::Duration;

pub(crate) const ROCM_TOOL_SCHEMA_ID: &str = "rocm-tools-v0";
pub(crate) const BUILTIN_ASSISTANT_MODEL_ALIAS: &str = "qwen";
pub(crate) const LEMONADE_ASSISTANT_MODEL_ID: &str = "Qwen3-4B-Instruct-2507-GGUF";
pub(crate) const BUILTIN_ASSISTANT_MODEL_ID: &str = LEMONADE_ASSISTANT_MODEL_ID;
const LOCAL_PROVIDER_HTTP_TIMEOUT: Duration = Duration::from_secs(30);
const LOCAL_SERVICE_READY_TIMEOUT: Duration = Duration::from_secs(2);
const REMOTE_PROVIDER_HTTP_TIMEOUT: Duration = Duration::from_mins(1);

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ProviderStatus {
    pub provider: String,
    pub auth_status: String,
    pub models: Vec<String>,
    pub tool_call_schema: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub rocm_tools: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChatToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChatResponse {
    pub provider: String,
    pub model: String,
    pub content: String,
    pub tool_calls: Vec<ChatToolCall>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ChatStreamSummary {
    pub provider: String,
    pub model: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ProviderStreamEvent {
    pub content: String,
    pub done: bool,
}

pub(crate) trait ProviderAdapter {
    fn status(&self) -> Result<ProviderStatus>;
    fn chat(&self, request: &ChatRequest) -> Result<ChatResponse>;
    fn stream_chat_with_callback(
        &self,
        request: &ChatRequest,
        on_event: &mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
    ) -> Result<ChatStreamSummary>;

    #[allow(dead_code)]
    fn stream_chat(&self, request: &ChatRequest) -> Result<Vec<ProviderStreamEvent>> {
        let mut events = Vec::new();
        self.stream_chat_with_callback(request, &mut |event| {
            events.push(event);
            Ok(())
        })?;
        Ok(events)
    }

    #[allow(dead_code)]
    fn models(&self) -> Result<Vec<String>> {
        Ok(self.status()?.models)
    }

    #[allow(dead_code)]
    fn tool_call_schema(&self) -> Result<String> {
        Ok(self.status()?.tool_call_schema)
    }

    #[allow(dead_code)]
    fn auth_status(&self) -> Result<String> {
        Ok(self.status()?.auth_status)
    }
}

struct LocalProvider<'a> {
    paths: &'a AppPaths,
}

struct RemoteProvider {
    provider: &'static str,
    api_key_env: &'static str,
    model_env: &'static str,
    endpoint_env: &'static str,
    default_endpoint: &'static str,
}

pub(crate) fn provider_status(paths: &AppPaths, provider: &str) -> Result<ProviderStatus> {
    if remote_provider(provider) && !provider_prompt_enabled(paths, provider)? {
        return Ok(ProviderStatus {
            provider: provider.to_owned(),
            auth_status: format!(
                "disabled: run `rocm config enable-provider {provider}` before sending prompts"
            ),
            models: remote_provider_models(provider),
            tool_call_schema: ROCM_TOOL_SCHEMA_ID.to_owned(),
        });
    }
    let status = provider_adapter(paths, provider)?.status()?;
    Ok(status)
}

pub(crate) fn provider_key_status_text(provider: &str) -> Result<String> {
    let (provider, env_name) = match provider {
        "local" => return Ok("not needed".to_owned()),
        "openai" => ("openai", "OPENAI_API_KEY"),
        "anthropic" => ("anthropic", "ANTHROPIC_API_KEY"),
        other => bail!("unsupported provider: {other}"),
    };
    let status = crate::provider_keys::provider_key_status(provider, env_name);
    Ok(crate::provider_keys::provider_key_status_label(&status))
}

pub(crate) fn provider_chat(
    paths: &AppPaths,
    provider: &str,
    request: &ChatRequest,
) -> Result<ChatResponse> {
    ensure_provider_prompt_enabled(paths, provider)?;
    let response = provider_adapter(paths, provider)?.chat(request)?;
    append_provider_audit_event(
        paths,
        provider,
        "chat",
        format!(
            "provider chat completed model={} messages={}",
            response.model,
            request.messages.len()
        ),
    )?;
    Ok(response)
}

#[allow(dead_code)]
pub(crate) fn provider_stream_chat(
    paths: &AppPaths,
    provider: &str,
    request: &ChatRequest,
) -> Result<Vec<ProviderStreamEvent>> {
    let mut events = Vec::new();
    provider_stream_chat_with_callback(paths, provider, request, &mut |event| {
        events.push(event);
        Ok(())
    })?;
    Ok(events)
}

pub(crate) fn provider_stream_chat_with_callback(
    paths: &AppPaths,
    provider: &str,
    request: &ChatRequest,
    on_event: &mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
) -> Result<()> {
    ensure_provider_prompt_enabled(paths, provider)?;
    let summary =
        provider_adapter(paths, provider)?.stream_chat_with_callback(request, on_event)?;
    append_provider_audit_event(
        paths,
        provider,
        "stream_chat",
        format!(
            "provider stream chat completed model={} messages={}",
            summary.model,
            request.messages.len()
        ),
    )
}

fn ensure_provider_prompt_enabled(paths: &AppPaths, provider: &str) -> Result<()> {
    if remote_provider(provider) && !provider_prompt_enabled(paths, provider)? {
        bail!(
            "cloud provider `{provider}` is disabled; run `rocm config enable-provider {provider}` before sending prompts"
        );
    }
    Ok(())
}

fn provider_prompt_enabled(paths: &AppPaths, provider: &str) -> Result<bool> {
    Ok(RocmCliConfig::load(paths)?.provider_enabled(provider))
}

fn remote_provider(provider: &str) -> bool {
    matches!(provider, "openai" | "anthropic")
}

fn remote_provider_models(provider: &str) -> Vec<String> {
    let model_env = match provider {
        "openai" => "ROCM_CLI_OPENAI_MODEL",
        "anthropic" => "ROCM_CLI_ANTHROPIC_MODEL",
        _ => return Vec::new(),
    };
    std::env::var(model_env)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .into_iter()
        .collect()
}

fn append_provider_audit_event(
    paths: &AppPaths,
    provider: &str,
    action: &str,
    message: String,
) -> Result<()> {
    append_audit_event(
        paths,
        &AuditEventRecord {
            at_unix_ms: unix_time_millis(),
            source: "rocm".to_owned(),
            category: "provider".to_owned(),
            actor: format!("provider:{provider}"),
            level: "info".to_owned(),
            action: action.to_owned(),
            message,
            watcher_id: None,
            service_id: None,
        },
    )
}

fn provider_adapter<'a>(
    paths: &'a AppPaths,
    provider: &str,
) -> Result<Box<dyn ProviderAdapter + 'a>> {
    Ok(match provider {
        "local" => Box::new(LocalProvider { paths }),
        "openai" => Box::new(RemoteProvider {
            provider: "openai",
            api_key_env: "OPENAI_API_KEY",
            model_env: "ROCM_CLI_OPENAI_MODEL",
            endpoint_env: "OPENAI_BASE_URL",
            default_endpoint: "https://api.openai.com/v1/chat/completions",
        }),
        "anthropic" => Box::new(RemoteProvider {
            provider: "anthropic",
            api_key_env: "ANTHROPIC_API_KEY",
            model_env: "ROCM_CLI_ANTHROPIC_MODEL",
            endpoint_env: "ANTHROPIC_BASE_URL",
            default_endpoint: "https://api.anthropic.com/v1/messages",
        }),
        other => bail!("unsupported provider: {other}"),
    })
}

impl ProviderAdapter for LocalProvider<'_> {
    fn status(&self) -> Result<ProviderStatus> {
        let services = ready_local_services(self.paths)?;
        let builtin_ready = services.iter().any(local_service_is_builtin_assistant);
        let mut models = services
            .into_iter()
            .map(|record| record.canonical_model_id)
            .collect::<Vec<_>>();
        models.sort();
        models.dedup();
        Ok(ProviderStatus {
            provider: "local".to_owned(),
            auth_status: if builtin_ready {
                "ready".to_owned()
            } else if models.is_empty() {
                "no_ready_local_service".to_owned()
            } else {
                "no_ready_builtin_assistant_service".to_owned()
            },
            models,
            tool_call_schema: ROCM_TOOL_SCHEMA_ID.to_owned(),
        })
    }

    fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let service = select_local_chat_service(self.paths, request.model.as_deref())?;
        let body = openai_local_chat_request_body(
            request
                .model
                .as_deref()
                .unwrap_or(&service.canonical_model_id),
            request,
        );
        let response = post_json_to_local_endpoint(
            &service.endpoint_url,
            "/v1/chat/completions",
            &body,
            LOCAL_PROVIDER_HTTP_TIMEOUT,
        )?;
        let (content, tool_calls) = parse_openai_chat_message(&response)?;
        Ok(ChatResponse {
            provider: "local".to_owned(),
            model: service.canonical_model_id,
            content,
            tool_calls,
        })
    }

    fn stream_chat_with_callback(
        &self,
        request: &ChatRequest,
        on_event: &mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
    ) -> Result<ChatStreamSummary> {
        let service = select_local_chat_service(self.paths, request.model.as_deref())?;
        let model = service.canonical_model_id.clone();
        let body = openai_local_stream_chat_request_body(
            request
                .model
                .as_deref()
                .unwrap_or(&service.canonical_model_id),
            request,
        );
        stream_json_from_local_endpoint(
            &service.endpoint_url,
            "/v1/chat/completions",
            &body,
            LOCAL_PROVIDER_HTTP_TIMEOUT,
            on_event,
        )?;
        Ok(ChatStreamSummary {
            provider: "local".to_owned(),
            model,
        })
    }
}

impl ProviderAdapter for RemoteProvider {
    fn status(&self) -> Result<ProviderStatus> {
        let models = std::env::var(self.model_env)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .into_iter()
            .collect::<Vec<_>>();
        let key_status = crate::provider_keys::provider_key_status(self.provider, self.api_key_env);
        Ok(ProviderStatus {
            provider: self.provider.to_owned(),
            auth_status: crate::provider_keys::provider_key_status_label(&key_status),
            models,
            tool_call_schema: ROCM_TOOL_SCHEMA_ID.to_owned(),
        })
    }

    fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let api_key =
            crate::provider_keys::resolve_provider_api_key(self.provider, self.api_key_env)?;
        let model = resolve_remote_model(self.provider, self.model_env, request.model.as_deref())?;
        let endpoint = remote_endpoint(self.endpoint_env, self.default_endpoint);
        let (content, tool_calls) = match self.provider {
            "openai" => {
                let body = openai_chat_request_body(&model, request);
                let json = post_json_with_headers(
                    &endpoint,
                    &[
                        ("Authorization", format!("Bearer {}", api_key.value)),
                        ("Content-Type", "application/json".to_owned()),
                    ],
                    &body,
                )?;
                parse_openai_chat_message(&json)?
            }
            "anthropic" => {
                let body = anthropic_chat_request_body(&model, request);
                let json = post_json_with_headers(
                    &endpoint,
                    &[
                        ("x-api-key", api_key.value),
                        ("anthropic-version", "2023-06-01".to_owned()),
                        ("Content-Type", "application/json".to_owned()),
                    ],
                    &body,
                )?;
                parse_anthropic_chat_message(&json)?
            }
            other => bail!("unsupported remote provider: {other}"),
        };
        Ok(ChatResponse {
            provider: self.provider.to_owned(),
            model,
            content,
            tool_calls,
        })
    }

    fn stream_chat_with_callback(
        &self,
        request: &ChatRequest,
        on_event: &mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
    ) -> Result<ChatStreamSummary> {
        let api_key =
            crate::provider_keys::resolve_provider_api_key(self.provider, self.api_key_env)?;
        let model = resolve_remote_model(self.provider, self.model_env, request.model.as_deref())?;
        let endpoint = remote_endpoint(self.endpoint_env, self.default_endpoint);
        match self.provider {
            "openai" => {
                let body = openai_stream_chat_request_body(&model, request);
                let mut emitter = OpenAiSseEmitter::new(on_event);
                stream_json_with_headers(
                    &endpoint,
                    &[
                        ("Authorization", format!("Bearer {}", api_key.value)),
                        ("Content-Type", "application/json".to_owned()),
                        ("Accept", "text/event-stream".to_owned()),
                    ],
                    &body,
                    &mut emitter,
                )?;
                emitter.finish()?;
            }
            "anthropic" => {
                let body = anthropic_stream_chat_request_body(&model, request);
                let mut emitter = AnthropicSseEmitter::new(on_event);
                stream_json_with_headers(
                    &endpoint,
                    &[
                        ("x-api-key", api_key.value),
                        ("anthropic-version", "2023-06-01".to_owned()),
                        ("Content-Type", "application/json".to_owned()),
                        ("Accept", "text/event-stream".to_owned()),
                    ],
                    &body,
                    &mut emitter,
                )?;
                emitter.finish()?;
            }
            other => bail!("unsupported remote provider: {other}"),
        }
        Ok(ChatStreamSummary {
            provider: self.provider.to_owned(),
            model,
        })
    }
}

fn select_local_chat_service(
    paths: &AppPaths,
    model: Option<&str>,
) -> Result<ManagedServiceRecord> {
    let mut services = ready_local_services(paths)?;
    if let Some(model) = model {
        services.retain(|service| {
            service.canonical_model_id.eq_ignore_ascii_case(model)
                || service.model_ref.eq_ignore_ascii_case(model)
        });
        return services
            .into_iter()
            .next()
            .context("local provider has no ready managed service for this request");
    }
    services.retain(local_service_is_builtin_assistant);
    services.sort_by_key(local_service_priority);
    services.into_iter().next()
        .with_context(|| {
            format!(
                "local provider has no ready managed service for the built-in assistant model {BUILTIN_ASSISTANT_MODEL_ID}; start `{BUILTIN_ASSISTANT_MODEL_ALIAS}` or pass --model for a custom/manual local service"
            )
        })
}

const fn local_service_is_builtin_assistant(service: &ManagedServiceRecord) -> bool {
    local_service_priority(service) != usize::MAX
}

const fn local_service_priority(service: &ManagedServiceRecord) -> usize {
    let model = service.canonical_model_id.as_str();
    if model.eq_ignore_ascii_case(LEMONADE_ASSISTANT_MODEL_ID) {
        0
    } else {
        usize::MAX
    }
}

fn ready_local_services(paths: &AppPaths) -> Result<Vec<ManagedServiceRecord>> {
    let services_dir = paths.services_dir();
    if !services_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut services = Vec::new();
    for entry in fs::read_dir(&services_dir)
        .with_context(|| format!("failed to read {}", services_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let Ok(mut record) = serde_json::from_slice::<ManagedServiceRecord>(&bytes) else {
            continue;
        };
        record.normalize_paths_for_host();
        if record.refresh_from_engine_state().unwrap_or(false) {
            let _ = record.write();
        }
        if matches!(record.status.as_str(), "ready" | "running")
            && managed_service_endpoint_model_ready(&record, LOCAL_SERVICE_READY_TIMEOUT)
                .unwrap_or(false)
        {
            if record.status != "ready" {
                record.status = "ready".to_owned();
                let _ = record.write();
            }
            services.push(record);
        }
    }
    services.sort_by_key(|service| std::cmp::Reverse(service.created_at_unix_ms));
    Ok(services)
}

fn post_json_to_local_endpoint(
    endpoint_url: &str,
    path: &str,
    body: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value> {
    let body = post_json_to_local_endpoint_body(endpoint_url, path, body, timeout)?;
    serde_json::from_str(body.trim()).context("failed to parse local provider chat JSON")
}

fn post_json_to_local_endpoint_body(
    endpoint_url: &str,
    path: &str,
    body: &serde_json::Value,
    timeout: Duration,
) -> Result<String> {
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported local endpoint URL `{endpoint_url}`"))?;
    let body = serde_json::to_string(body).context("failed to serialize chat request")?;
    let mut stream = connect_tcp_stream(&host, port, timeout)?;
    let host_header = format_host_port(&host, port);
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .context("failed to write local provider chat request")?;

    let response = read_tcp_stream_to_string(&mut stream)
        .context("failed to read local provider chat response")?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .context("local provider response was missing HTTP body")?;
    let status_line = headers.lines().next().unwrap_or_default();
    if !status_line.contains(" 200 ") {
        bail!("local provider returned {status_line}");
    }
    Ok(body.to_owned())
}

fn stream_json_from_local_endpoint(
    endpoint_url: &str,
    path: &str,
    body: &serde_json::Value,
    timeout: Duration,
    on_event: &mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
) -> Result<()> {
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported local endpoint URL `{endpoint_url}`"))?;
    let body = serde_json::to_string(body).context("failed to serialize chat request")?;
    let mut stream = connect_tcp_stream(&host, port, timeout)?;
    let host_header = format_host_port(&host, port);
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .context("failed to write local provider chat request")?;

    let mut reader = BufReader::new(stream);
    let mut headers = String::new();
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .context("failed to read local provider response headers")?;
        if read == 0 {
            bail!("local provider response ended before headers completed");
        }
        headers.push_str(&line);
        if line == "\r\n" || line == "\n" {
            break;
        }
    }
    let status_line = headers.lines().next().unwrap_or_default();
    if !status_line.contains(" 200 ") {
        bail!("local provider returned {status_line}");
    }

    let transfer_encoding = header_value(&headers, "Transfer-Encoding");
    let chunked = transfer_encoding
        .as_deref()
        .is_some_and(|value| value.to_ascii_lowercase().contains("chunked"));
    let content_length = header_value(&headers, "Content-Length")
        .and_then(|value| value.trim().parse::<usize>().ok());
    let mut emitter = OpenAiSseEmitter::new(on_event);
    if chunked {
        read_chunked_sse_body(&mut reader, &mut emitter)?;
    } else if let Some(length) = content_length {
        read_fixed_sse_body(&mut reader, length, &mut emitter)?;
    } else {
        read_close_delimited_sse_body(&mut reader, &mut emitter)?;
    }
    emitter.finish()
}

fn header_value(headers: &str, name: &str) -> Option<String> {
    headers.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name)
            .then(|| value.trim().to_owned())
    })
}

fn read_fixed_sse_body<R: Read>(
    reader: &mut R,
    mut remaining: usize,
    emitter: &mut OpenAiSseEmitter<'_>,
) -> Result<()> {
    let mut buffer = [0_u8; 1024];
    while remaining > 0 {
        let read_len = remaining.min(buffer.len());
        reader
            .read_exact(&mut buffer[..read_len])
            .context("failed to read local provider SSE body")?;
        emitter.push_bytes(&buffer[..read_len])?;
        remaining -= read_len;
    }
    Ok(())
}

fn read_close_delimited_sse_body<R: Read>(
    reader: &mut R,
    emitter: &mut OpenAiSseEmitter<'_>,
) -> Result<()> {
    let mut buffer = [0_u8; 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .context("failed to read local provider SSE body")?;
        if read == 0 {
            break;
        }
        emitter.push_bytes(&buffer[..read])?;
    }
    Ok(())
}

fn read_chunked_sse_body<R: BufRead>(
    reader: &mut R,
    emitter: &mut OpenAiSseEmitter<'_>,
) -> Result<()> {
    loop {
        let mut size_line = String::new();
        reader
            .read_line(&mut size_line)
            .context("failed to read local provider chunk size")?;
        if size_line.trim().is_empty() {
            continue;
        }
        let size_text = size_line
            .trim()
            .split(';')
            .next()
            .unwrap_or_default()
            .trim();
        let size = usize::from_str_radix(size_text, 16)
            .with_context(|| format!("invalid local provider chunk size `{size_text}`"))?;
        if size == 0 {
            let mut trailer = String::new();
            let _ = reader.read_line(&mut trailer);
            break;
        }
        read_fixed_sse_body(reader, size, emitter)?;
        let mut crlf = [0_u8; 2];
        reader
            .read_exact(&mut crlf)
            .context("failed to read local provider chunk terminator")?;
        if crlf != *b"\r\n" {
            bail!("invalid local provider chunk terminator");
        }
    }
    Ok(())
}

fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
    let without_scheme = endpoint_url.trim().strip_prefix("http://")?;
    let authority = without_scheme.split('/').next()?.trim();
    if authority.is_empty() {
        return None;
    }
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']')?;
        let host = rest[..end].to_owned();
        let port = rest[end + 1..].strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    Some((host.to_owned(), port.parse().ok()?))
}

#[cfg(test)]
fn parse_openai_chat_content(response: &serde_json::Value) -> Result<String> {
    let (content, tool_calls) = parse_openai_chat_message(response)?;
    if content.is_empty() && !tool_calls.is_empty() {
        bail!("OpenAI-compatible response contained tool calls but no text content");
    }
    Ok(content)
}

fn parse_openai_chat_message(response: &serde_json::Value) -> Result<(String, Vec<ChatToolCall>)> {
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

fn parse_anthropic_chat_message(
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

fn resolve_remote_model(
    provider: &str,
    model_env: &str,
    request_model: Option<&str>,
) -> Result<String> {
    request_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            std::env::var(model_env)
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
        .with_context(|| format!("{provider} provider requires a model via --model or {model_env}"))
}

fn remote_endpoint(endpoint_env: &str, default_endpoint: &str) -> String {
    std::env::var(endpoint_env)
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default_endpoint.to_owned())
}

fn openai_chat_request_body(model: &str, request: &ChatRequest) -> serde_json::Value {
    openai_chat_request_body_with_stream(model, request, false)
}

fn openai_stream_chat_request_body(model: &str, request: &ChatRequest) -> serde_json::Value {
    openai_chat_request_body_with_stream(model, request, true)
}

fn openai_local_chat_request_body(model: &str, request: &ChatRequest) -> serde_json::Value {
    let request = local_openai_compatible_request(request);
    let mut body = openai_chat_request_body(model, &request);
    apply_local_model_request_defaults(model, &mut body);
    body
}

fn openai_local_stream_chat_request_body(model: &str, request: &ChatRequest) -> serde_json::Value {
    let request = local_openai_compatible_request(request);
    let mut body = openai_stream_chat_request_body(model, &request);
    apply_local_model_request_defaults(model, &mut body);
    body
}

fn apply_local_model_request_defaults(model: &str, body: &mut serde_json::Value) {
    if local_model_should_disable_thinking(model) {
        body["chat_template_kwargs"] = serde_json::json!({
            "enable_thinking": false
        });
    }
}

fn local_model_should_disable_thinking(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    normalized.contains("qwen3") || normalized.contains("qwen/qwen3")
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
    if request.rocm_tools && !stream {
        body["tools"] = serde_json::Value::Array(rocm_openai_tool_definitions());
        body["tool_choice"] = serde_json::json!("auto");
    }
    body
}

fn rocm_openai_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        rocm_openai_tool(
            "examine",
            "Read the current ROCm host, GPU, runtime, driver, and engine status.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "bridge_snapshot",
            "Read a full ROCm snapshot including examine data, engines, services, automations, and GPU telemetry.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "gpu_snapshot",
            "Read the current local AMD GPU telemetry snapshot when available.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "engines",
            "List available ROCm serving engines and whether each is installed.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "services",
            "List managed local model servers and their current status.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "service_logs",
            "Read the recent log output for a managed local model server.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "service_id": { "type": "string" },
                    "lines": { "type": "integer", "minimum": 1, "maximum": 500 }
                },
                "required": ["service_id"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "automations",
            "List watcher and automation status plus recent automation events.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "event_limit": { "type": "integer", "minimum": 1, "maximum": 64 }
                },
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "natural_language_plan",
            "Ask ROCm CLI to translate a natural-language ROCm request into a visible plan without executing it.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "request": { "type": "string" }
                },
                "required": ["request"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "path_exists",
            "Read whether an exact local filesystem path exists and whether its parent folder exists. Use this only to check user-provided install folders before requesting a ROCm install.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "port_status",
            "Read whether a local loopback TCP port is listening and whether a ROCm-managed service owns it. Use this for questions like whether something is running on port 8188 or the local model server port. Do not use it for non-local hosts.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "host": { "type": "string" },
                    "port": { "type": "integer", "minimum": 1, "maximum": 65535 }
                },
                "required": ["port"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "rocm_command",
            "Run or request a supported ROCm CLI command with argv-style arguments. Known read-only inspection commands may run immediately. Commands that install, start, serve, stop, delete, or change ROCm state are paused for user review before they run. Use this tool for mutating TheRock setup, including --prefix PATH when the user gives an install folder, --build-date YYYY-MM-DD or --version VERSION, config changes, ComfyUI install/start, engine install, vLLM management, and local LLM serve actions; do not request shell commands, CPU execution, or public network binds.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "maxItems": 64
                    },
                    "reason": { "type": "string" }
                },
                "required": ["args"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "update_check",
            "Check whether ROCm CLI knows about newer TheRock runtime versions.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "install_sdk_dry_run",
            "Preview a TheRock SDK install without changing the machine.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "channel": { "type": "string", "enum": ["release", "nightly"] },
                    "format": { "type": "string", "enum": ["wheel", "tarball"] },
                    "prefix": { "type": "string" },
                    "version": { "type": "string" },
                    "build_date": { "type": "string" }
                },
                "additionalProperties": false
            }),
        ),
    ]
}

fn rocm_openai_tool(
    name: &str,
    description: &str,
    parameters: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    })
}

fn rocm_anthropic_tool_definitions() -> Vec<serde_json::Value> {
    rocm_openai_tool_definitions()
        .into_iter()
        .filter_map(|tool| {
            let function = tool.get("function")?;
            Some(serde_json::json!({
                "name": function.get("name")?.clone(),
                "description": function.get("description")?.clone(),
                "input_schema": function.get("parameters")?.clone(),
            }))
        })
        .collect()
}

fn anthropic_chat_request_body(model: &str, request: &ChatRequest) -> serde_json::Value {
    anthropic_chat_request_body_with_stream(model, request, false)
}

fn anthropic_stream_chat_request_body(model: &str, request: &ChatRequest) -> serde_json::Value {
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
    if request.rocm_tools && !stream {
        body["tools"] = serde_json::Value::Array(rocm_anthropic_tool_definitions());
    }
    body
}

fn post_json_with_headers(
    endpoint: &str,
    headers: &[(&str, String)],
    body: &serde_json::Value,
) -> Result<serde_json::Value> {
    let body = serde_json::to_string(body).context("failed to serialize provider chat request")?;
    let mut request = ureq::post(endpoint).timeout(REMOTE_PROVIDER_HTTP_TIMEOUT);
    for (name, value) in headers {
        request = request.set(name, value);
    }
    let response = match request.send_string(&body) {
        Ok(response) => response,
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            if detail.trim().is_empty() {
                bail!("provider chat request failed with HTTP {code}");
            }
            bail!(
                "provider chat request failed with HTTP {code}: {}",
                detail.trim()
            );
        }
        Err(ureq::Error::Transport(error)) => {
            bail!("provider chat request failed: {error}");
        }
    };
    let text = response
        .into_string()
        .context("failed to read provider chat response")?;
    serde_json::from_str(text.trim()).context("failed to parse provider chat JSON")
}

trait SseBytesEmitter {
    fn push_bytes(&mut self, bytes: &[u8]) -> Result<()>;
}

fn stream_json_with_headers<E: SseBytesEmitter>(
    endpoint: &str,
    headers: &[(&str, String)],
    body: &serde_json::Value,
    emitter: &mut E,
) -> Result<()> {
    let body =
        serde_json::to_string(body).context("failed to serialize provider stream request")?;
    let mut request = ureq::post(endpoint).timeout(REMOTE_PROVIDER_HTTP_TIMEOUT);
    for (name, value) in headers {
        request = request.set(name, value);
    }
    let response = match request.send_string(&body) {
        Ok(response) => response,
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            if detail.trim().is_empty() {
                bail!("provider stream request failed with HTTP {code}");
            }
            bail!(
                "provider stream request failed with HTTP {code}: {}",
                detail.trim()
            );
        }
        Err(ureq::Error::Transport(error)) => {
            bail!("provider stream request failed: {error}");
        }
    };
    let mut reader = response.into_reader();
    let mut buffer = [0_u8; 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .context("failed to read provider stream response")?;
        if read == 0 {
            break;
        }
        emitter.push_bytes(&buffer[..read])?;
    }
    Ok(())
}

struct OpenAiSseEmitter<'a> {
    pending: Vec<u8>,
    done: bool,
    on_event: &'a mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
}

impl<'a> OpenAiSseEmitter<'a> {
    fn new(on_event: &'a mut dyn FnMut(ProviderStreamEvent) -> Result<()>) -> Self {
        Self {
            pending: Vec::new(),
            done: false,
            on_event,
        }
    }

    fn finish(mut self) -> Result<()> {
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

struct AnthropicSseEmitter<'a> {
    pending: Vec<u8>,
    done: bool,
    on_event: &'a mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
}

impl<'a> AnthropicSseEmitter<'a> {
    fn new(on_event: &'a mut dyn FnMut(ProviderStreamEvent) -> Result<()>) -> Self {
        Self {
            pending: Vec::new(),
            done: false,
            on_event,
        }
    }

    fn finish(mut self) -> Result<()> {
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
mod tests {
    use super::*;
    use rocm_core::{AppPaths, ManagedServiceRecord, RocmCliConfig, unix_time_millis};
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn local_provider_endpoint_parser_supports_ipv6_loopback() {
        assert_eq!(
            parse_http_endpoint("http://[::1]:11435/v1"),
            Some(("::1".to_owned(), 11435))
        );
    }

    #[test]
    fn local_provider_reports_ready_managed_models() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-models");
        paths.ensure()?;
        let ready_server = spawn_models_only_server(BUILTIN_ASSISTANT_MODEL_ID)?;
        let mut ready = ManagedServiceRecord::new(
            &paths,
            "svc-ready",
            "pytorch",
            "qwen",
            BUILTIN_ASSISTANT_MODEL_ID,
            "127.0.0.1",
            ready_server.port(),
            "managed",
            123,
            None,
            None,
            None,
        );
        ready.status = "running".to_owned();
        ready.write()?;

        let mut stopped = ManagedServiceRecord::new(
            &paths,
            "svc-stopped",
            "pytorch",
            "llama",
            "meta-llama/Llama",
            "127.0.0.1",
            11436,
            "managed",
            124,
            None,
            None,
            None,
        );
        stopped.status = "stopped".to_owned();
        stopped.write()?;

        let status = provider_status(&paths, "local")?;
        let served = ready_server.stop()?;
        fs::remove_dir_all(root).ok();

        assert_eq!(status.provider, "local");
        assert_eq!(status.auth_status, "ready");
        assert_eq!(status.models, vec![BUILTIN_ASSISTANT_MODEL_ID.to_owned()]);
        assert_eq!(status.tool_call_schema, ROCM_TOOL_SCHEMA_ID);
        assert!(served > 0);
        Ok(())
    }

    #[test]
    fn local_provider_refreshes_starting_service_from_ready_engine_state() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-engine-state-ready");
        paths.ensure()?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<Vec<String>> {
            let mut requests = Vec::new();
            for _ in 0..2 {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut request_bytes = Vec::new();
                let mut buffer = [0_u8; 1024];
                loop {
                    let read = stream.read(&mut buffer)?;
                    if read == 0 {
                        break;
                    }
                    request_bytes.extend_from_slice(&buffer[..read]);
                    let request = String::from_utf8_lossy(&request_bytes);
                    if request.contains("\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&request_bytes).into_owned();
                let body = format!(r#"{{"data":[{{"id":"{LEMONADE_ASSISTANT_MODEL_ID}"}}]}}"#);
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
                requests.push(request);
            }
            Ok(requests)
        });
        let mut service = ManagedServiceRecord::new(
            &paths,
            "svc-lemonade-ready",
            "lemonade",
            "qwen",
            LEMONADE_ASSISTANT_MODEL_ID,
            "127.0.0.1",
            port,
            "managed",
            123,
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        service.status = "starting".to_owned();
        service.write()?;
        fs::create_dir_all(service.engine_state_path.parent().expect("state parent"))?;
        fs::write(
            &service.engine_state_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "status": "ready",
                "endpoint_url": format!("http://127.0.0.1:{port}/v1"),
                "server_pid": 456,
                "runtime_id": "release-pip-gfx120x-all-7-13-0a20260511",
                "env_id": "lemonade-embeddable-10.6.0"
            }))?,
        )?;

        let selected = select_local_chat_service(&paths, None)?;
        let status = provider_status(&paths, "local")?;
        let requests = server.join().expect("server thread should not panic")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(requests.len(), 2);
        assert!(
            requests
                .iter()
                .all(|request| request.starts_with("GET /v1/models HTTP/1.1"))
        );
        assert_eq!(selected.service_id, "svc-lemonade-ready");
        assert_eq!(selected.status, "ready");
        assert_eq!(selected.engine_pid, Some(456));
        assert_eq!(status.auth_status, "ready");
        assert_eq!(status.models, vec![LEMONADE_ASSISTANT_MODEL_ID.to_owned()]);
        Ok(())
    }

    #[test]
    fn local_provider_reports_no_ready_service_without_manifests() -> Result<()> {
        let (_root, paths) = temp_app_paths("local-provider-empty");
        let status = provider_status(&paths, "local")?;
        assert_eq!(status.auth_status, "no_ready_local_service");
        assert!(status.models.is_empty());
        Ok(())
    }

    #[test]
    fn remote_provider_status_reports_disabled_until_opt_in() -> Result<()> {
        let (_root, paths) = temp_app_paths("remote-provider-status-disabled");

        let status = provider_status(&paths, "openai")?;

        assert_eq!(status.provider, "openai");
        assert_eq!(
            status.auth_status,
            "disabled: run `rocm config enable-provider openai` before sending prompts"
        );
        Ok(())
    }

    #[test]
    fn remote_provider_chat_requires_config_opt_in_before_env_or_network() {
        let (_root, paths) = temp_app_paths("remote-provider-chat-disabled");

        let error = provider_chat(
            &paths,
            "openai",
            &ChatRequest {
                model: Some("test-model".to_owned()),
                messages: vec![ChatMessage {
                    role: "user".to_owned(),
                    content: "hello".to_owned(),
                }],
                max_tokens: Some(1),
                rocm_tools: false,
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("cloud provider `openai` is disabled"));
        assert!(error.contains("rocm config enable-provider openai"));
    }

    #[test]
    fn remote_provider_status_uses_env_auth_after_opt_in() -> Result<()> {
        let (root, paths) = temp_app_paths("remote-provider-status-enabled");
        let mut config = RocmCliConfig::default();
        config.provider_config_mut("openai").enabled = true;
        config.save(&paths)?;

        let status = provider_status(&paths, "openai")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(status.provider, "openai");
        assert!(!status.auth_status.starts_with("disabled:"));
        Ok(())
    }

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
    fn remote_model_prefers_request_model() -> Result<()> {
        assert_eq!(
            resolve_remote_model("openai", "ROCM_CLI_TEST_MODEL", Some("gpt-test"))?,
            "gpt-test"
        );
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
    fn openai_local_qwen3_request_disables_thinking_for_visible_answers() {
        let request = ChatRequest {
            model: Some("Qwen3-0.6B-GGUF".to_owned()),
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: "hello".to_owned(),
            }],
            max_tokens: Some(16),
            rocm_tools: false,
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

    #[test]
    fn local_provider_chat_posts_to_ready_service() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-chat");
        paths.ensure()?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = spawn_local_provider_test_server(
            listener,
            "Qwen/Qwen3.5",
            move |stream, _request| {
                let body = r#"{"choices":[{"message":{"content":"ready response"}}]}"#;
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
                Ok(())
            },
        );

        let mut ready = ManagedServiceRecord::new(
            &paths,
            "svc-ready",
            "pytorch",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            port,
            "managed",
            123,
            None,
            None,
            None,
        );
        ready.status = "ready".to_owned();
        ready.write()?;

        let response = provider_chat(
            &paths,
            "local",
            &ChatRequest {
                model: Some("Qwen/Qwen3.5".to_owned()),
                messages: vec![
                    ChatMessage {
                        role: "system".to_owned(),
                        content: "Keep answers focused on AMD ROCm.".to_owned(),
                    },
                    ChatMessage {
                        role: "user".to_owned(),
                        content: "hello".to_owned(),
                    },
                ],
                max_tokens: Some(16),
                rocm_tools: false,
            },
        )?;
        let request = server.join().expect("server thread should not panic")?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        let audit = serde_json::from_str::<AuditEventRecord>(audit_text.trim())?;
        fs::remove_dir_all(root).ok();

        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(request.contains("ROCm CLI guidance"));
        assert!(request.contains("Keep answers focused on AMD ROCm."));
        assert!(request.contains("User message:\\nhello"));
        assert!(!request.contains("\"role\":\"system\""));
        assert!(!request.contains("\"tools\""));
        assert_eq!(response.provider, "local");
        assert_eq!(response.model, "Qwen/Qwen3.5");
        assert_eq!(response.content, "ready response");
        assert!(response.tool_calls.is_empty());
        assert_eq!(audit.category, "provider");
        assert_eq!(audit.actor, "provider:local");
        assert_eq!(audit.action, "chat");
        Ok(())
    }

    #[test]
    fn local_provider_default_chat_requires_builtin_qwen_assistant() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-default-qwen");
        paths.ensure()?;
        let custom_server = spawn_models_only_server("sshleifer/tiny-gpt2")?;

        let mut custom = ManagedServiceRecord::new(
            &paths,
            "svc-custom",
            "pytorch",
            "tiny-gpt2",
            "sshleifer/tiny-gpt2",
            "127.0.0.1",
            custom_server.port(),
            "managed",
            123,
            None,
            None,
            None,
        );
        custom.status = "ready".to_owned();
        custom.write()?;

        let error = select_local_chat_service(&paths, None)
            .expect_err("default local assistant should not pick unvalidated services")
            .to_string();
        assert!(error.contains(BUILTIN_ASSISTANT_MODEL_ID));
        let status = provider_status(&paths, "local")?;
        assert_eq!(status.auth_status, "no_ready_builtin_assistant_service");
        assert_eq!(status.models, vec!["sshleifer/tiny-gpt2".to_owned()]);
        assert_eq!(
            select_local_chat_service(&paths, Some("tiny-gpt2"))?.service_id,
            "svc-custom"
        );

        let qwen_server = spawn_models_only_server(BUILTIN_ASSISTANT_MODEL_ID)?;
        let mut qwen = ManagedServiceRecord::new(
            &paths,
            "svc-qwen",
            "pytorch",
            BUILTIN_ASSISTANT_MODEL_ALIAS,
            BUILTIN_ASSISTANT_MODEL_ID,
            "127.0.0.1",
            qwen_server.port(),
            "managed",
            124,
            None,
            None,
            None,
        );
        qwen.status = "ready".to_owned();
        qwen.write()?;

        let selected = select_local_chat_service(&paths, None)?;
        let status = provider_status(&paths, "local")?;
        let custom_served = custom_server.stop()?;
        let qwen_served = qwen_server.stop()?;
        fs::remove_dir_all(root).ok();

        assert_eq!(selected.service_id, "svc-qwen");
        assert_eq!(selected.canonical_model_id, BUILTIN_ASSISTANT_MODEL_ID);
        assert_eq!(status.auth_status, "ready");
        assert!(custom_served > 0);
        assert!(qwen_served > 0);
        Ok(())
    }

    #[test]
    fn local_provider_accepts_lemonade_qwen_assistant() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-lemonade-qwen");
        paths.ensure()?;
        let lemonade_server = spawn_models_only_server(LEMONADE_ASSISTANT_MODEL_ID)?;

        let mut lemonade = ManagedServiceRecord::new(
            &paths,
            "svc-lemonade-qwen",
            "lemonade",
            "lemonade-qwen",
            LEMONADE_ASSISTANT_MODEL_ID,
            "127.0.0.1",
            lemonade_server.port(),
            "managed",
            124,
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        lemonade.status = "ready".to_owned();
        lemonade.write()?;

        let selected = select_local_chat_service(&paths, None)?;
        let status = provider_status(&paths, "local")?;
        let served = lemonade_server.stop()?;
        fs::remove_dir_all(root).ok();

        assert_eq!(selected.service_id, "svc-lemonade-qwen");
        assert_eq!(selected.canonical_model_id, LEMONADE_ASSISTANT_MODEL_ID);
        assert_eq!(status.auth_status, "ready");
        assert!(served > 0);
        Ok(())
    }

    #[test]
    fn local_provider_prefers_lemonade_over_stale_builtin_services() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-prefers-lemonade");
        paths.ensure()?;
        let stale_custom_server = spawn_models_only_server("not-the-built-in-model")?;
        let lemonade_server = spawn_models_only_server(LEMONADE_ASSISTANT_MODEL_ID)?;

        let mut custom = ManagedServiceRecord::new(
            &paths,
            "svc-custom-qwen",
            "pytorch",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            stale_custom_server.port(),
            "managed",
            124,
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        custom.status = "ready".to_owned();
        custom.write()?;

        let mut lemonade = ManagedServiceRecord::new(
            &paths,
            "svc-lemonade-qwen",
            "lemonade",
            "lemonade-qwen",
            LEMONADE_ASSISTANT_MODEL_ID,
            "127.0.0.1",
            lemonade_server.port(),
            "managed",
            125,
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        lemonade.status = "ready".to_owned();
        lemonade.write()?;

        let selected = select_local_chat_service(&paths, None)?;
        let stale_served = stale_custom_server.stop()?;
        let lemonade_served = lemonade_server.stop()?;
        fs::remove_dir_all(root).ok();

        assert_eq!(selected.service_id, "svc-lemonade-qwen");
        assert_eq!(selected.engine, "lemonade");
        assert_eq!(selected.canonical_model_id, LEMONADE_ASSISTANT_MODEL_ID);
        assert!(stale_served > 0);
        assert!(lemonade_served > 0);
        Ok(())
    }

    #[test]
    fn local_provider_chat_posts_rocm_tools_and_preserves_tool_call() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-chat-tools");
        paths.ensure()?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = spawn_local_provider_test_server(
            listener,
            "tiny.gguf",
            move |stream, _request| {
                let body = r#"{"choices":[{"message":{"content":"I will check first.","tool_calls":[{"id":"call-1","type":"function","function":{"name":"examine","arguments":"{}"}}]}}]}"#;
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
                Ok(())
            },
        );

        let mut ready = ManagedServiceRecord::new(
            &paths,
            "svc-ready",
            "llama.cpp",
            "tiny.gguf",
            "tiny.gguf",
            "127.0.0.1",
            port,
            "managed",
            123,
            None,
            None,
            None,
        );
        ready.status = "ready".to_owned();
        ready.write()?;

        let response = provider_chat(
            &paths,
            "local",
            &ChatRequest {
                model: Some("tiny.gguf".to_owned()),
                messages: vec![ChatMessage {
                    role: "user".to_owned(),
                    content: "check this host".to_owned(),
                }],
                max_tokens: Some(16),
                rocm_tools: true,
            },
        )?;
        let request = server.join().expect("server thread should not panic")?;
        fs::remove_dir_all(root).ok();

        assert!(request.contains("\"tools\""));
        assert!(request.contains("\"tool_choice\":\"auto\""));
        assert!(request.contains("\"name\":\"examine\""));
        assert_eq!(response.content, "I will check first.");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "examine");
        Ok(())
    }

    #[test]
    fn local_provider_stream_chat_posts_sse_request() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-stream");
        paths.ensure()?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = spawn_local_provider_test_server(
            listener,
            "Qwen/Qwen3.5",
            move |stream, _request| {
                let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n";
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
                Ok(())
            },
        );

        let mut ready = ManagedServiceRecord::new(
            &paths,
            "svc-ready",
            "pytorch",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            port,
            "managed",
            123,
            None,
            None,
            None,
        );
        ready.status = "ready".to_owned();
        ready.write()?;

        let events = provider_stream_chat(
            &paths,
            "local",
            &ChatRequest {
                model: Some("Qwen/Qwen3.5".to_owned()),
                messages: vec![
                    ChatMessage {
                        role: "system".to_owned(),
                        content: "Keep answers focused on AMD ROCm.".to_owned(),
                    },
                    ChatMessage {
                        role: "user".to_owned(),
                        content: "hello".to_owned(),
                    },
                ],
                max_tokens: Some(16),
                rocm_tools: false,
            },
        )?;
        let request = server.join().expect("server thread should not panic")?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        let audit = serde_json::from_str::<AuditEventRecord>(audit_text.trim())?;
        fs::remove_dir_all(root).ok();

        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(request.contains("\"stream\":true"));
        assert!(request.contains("ROCm CLI guidance"));
        assert!(request.contains("Keep answers focused on AMD ROCm."));
        assert!(!request.contains("\"role\":\"system\""));
        assert_eq!(audit.category, "provider");
        assert_eq!(audit.actor, "provider:local");
        assert_eq!(audit.action, "stream_chat");
        assert!(audit.message.contains("model=Qwen/Qwen3.5"));
        assert!(audit.message.contains("messages=2"));
        assert_eq!(
            events,
            vec![
                ProviderStreamEvent {
                    content: "hel".to_owned(),
                    done: false
                },
                ProviderStreamEvent {
                    content: "lo".to_owned(),
                    done: false
                },
                ProviderStreamEvent {
                    content: String::new(),
                    done: true
                }
            ]
        );
        Ok(())
    }

    #[test]
    fn local_provider_stream_callback_receives_event_before_connection_closes() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-live-stream");
        paths.ensure()?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let (first_seen_tx, first_seen_rx) = mpsc::channel();
        let server = spawn_local_provider_test_server(
            listener,
            "Qwen/Qwen3.5",
            move |stream, _request| {
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
                )?;
                write!(
                    stream,
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":\"hel\"}}}}]}}\n\n"
                )?;
                stream.flush()?;
                first_seen_rx
                    .recv_timeout(Duration::from_secs(2))
                    .context("first SSE event was not observed before connection close")?;
                write!(
                    stream,
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":\"lo\"}}}}]}}\n\ndata: [DONE]\n\n"
                )?;
                stream.flush()?;
                Ok(())
            },
        );

        let mut ready = ManagedServiceRecord::new(
            &paths,
            "svc-ready",
            "pytorch",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            port,
            "managed",
            123,
            None,
            None,
            None,
        );
        ready.status = "ready".to_owned();
        ready.write()?;
        let mut events = Vec::new();

        provider_stream_chat_with_callback(
            &paths,
            "local",
            &ChatRequest {
                model: Some("Qwen/Qwen3.5".to_owned()),
                messages: vec![ChatMessage {
                    role: "user".to_owned(),
                    content: "hello".to_owned(),
                }],
                max_tokens: Some(16),
                rocm_tools: false,
            },
            &mut |event| {
                if event.content == "hel" {
                    let _ = first_seen_tx.send(());
                }
                events.push(event);
                Ok(())
            },
        )?;
        let request = server.join().expect("server thread should not panic")?;
        fs::remove_dir_all(root).ok();

        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert_eq!(
            events,
            vec![
                ProviderStreamEvent {
                    content: "hel".to_owned(),
                    done: false
                },
                ProviderStreamEvent {
                    content: "lo".to_owned(),
                    done: false
                },
                ProviderStreamEvent {
                    content: String::new(),
                    done: true
                }
            ]
        );
        Ok(())
    }

    #[test]
    fn local_provider_stream_chat_decodes_chunked_sse() -> Result<()> {
        let (root, paths) = temp_app_paths("local-provider-chunked-stream");
        paths.ensure()?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = spawn_local_provider_test_server(
            listener,
            "Qwen/Qwen3.5",
            move |stream, _request| {
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                )?;
                let chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"chunk\"}}]}\n\n";
                write!(stream, "{:X}\r\n{}\r\n", chunk.len(), chunk)?;
                let done = "data: [DONE]\n\n";
                write!(stream, "{:X}\r\n{}\r\n0\r\n\r\n", done.len(), done)?;
                stream.flush()?;
                Ok(())
            },
        );

        let mut ready = ManagedServiceRecord::new(
            &paths,
            "svc-ready",
            "pytorch",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            port,
            "managed",
            123,
            None,
            None,
            None,
        );
        ready.status = "ready".to_owned();
        ready.write()?;

        let events = provider_stream_chat(
            &paths,
            "local",
            &ChatRequest {
                model: Some("Qwen/Qwen3.5".to_owned()),
                messages: vec![ChatMessage {
                    role: "user".to_owned(),
                    content: "hello".to_owned(),
                }],
                max_tokens: Some(16),
                rocm_tools: false,
            },
        )?;
        let request = server.join().expect("server thread should not panic")?;
        fs::remove_dir_all(root).ok();

        assert!(request.contains("\"stream\":true"));
        assert_eq!(
            events,
            vec![
                ProviderStreamEvent {
                    content: "chunk".to_owned(),
                    done: false
                },
                ProviderStreamEvent {
                    content: String::new(),
                    done: true
                }
            ]
        );
        Ok(())
    }

    #[test]
    fn remote_openai_stream_chat_uses_live_sse() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let (first_seen_tx, first_seen_rx) = mpsc::channel();
        let server = thread::spawn(move || -> Result<String> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut request_bytes = Vec::new();
            let mut buffer = [0_u8; 1024];
            loop {
                let read = stream.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                request_bytes.extend_from_slice(&buffer[..read]);
                let request = String::from_utf8_lossy(&request_bytes);
                if request_complete(&request) {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            )?;
            write!(
                stream,
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"hel\"}}}}]}}\n\n"
            )?;
            stream.flush()?;
            first_seen_rx
                .recv_timeout(Duration::from_secs(2))
                .context("first remote OpenAI SSE event was not observed before close")?;
            write!(
                stream,
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"lo\"}}}}]}}\n\ndata: [DONE]\n\n"
            )?;
            stream.flush()?;
            Ok(request)
        });

        let endpoint = format!("http://127.0.0.1:{port}/openai-stream");
        let chat_request = ChatRequest {
            model: None,
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: "hello".to_owned(),
            }],
            max_tokens: Some(16),
            rocm_tools: false,
        };
        let body = openai_stream_chat_request_body("gpt-test", &chat_request);
        let mut events = Vec::new();

        {
            let mut on_event = |event: ProviderStreamEvent| {
                if event.content == "hel" {
                    let _ = first_seen_tx.send(());
                }
                events.push(event);
                Ok(())
            };
            let mut emitter = OpenAiSseEmitter::new(&mut on_event);
            stream_json_with_headers(
                &endpoint,
                &[
                    ("Authorization", "Bearer test-openai-key".to_owned()),
                    ("Content-Type", "application/json".to_owned()),
                    ("Accept", "text/event-stream".to_owned()),
                ],
                &body,
                &mut emitter,
            )?;
            emitter.finish()?;
        }
        let request = server.join().expect("server thread should not panic")?;

        assert!(request.starts_with("POST /openai-stream HTTP/1.1"));
        assert!(request.contains("\"stream\":true"));
        assert!(request.contains("\"model\":\"gpt-test\""));
        assert_eq!(
            events,
            vec![
                ProviderStreamEvent {
                    content: "hel".to_owned(),
                    done: false
                },
                ProviderStreamEvent {
                    content: "lo".to_owned(),
                    done: false
                },
                ProviderStreamEvent {
                    content: String::new(),
                    done: true
                }
            ]
        );
        Ok(())
    }

    #[test]
    fn remote_anthropic_stream_chat_uses_live_sse() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let (first_seen_tx, first_seen_rx) = mpsc::channel();
        let server = thread::spawn(move || -> Result<String> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut request_bytes = Vec::new();
            let mut buffer = [0_u8; 1024];
            loop {
                let read = stream.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                request_bytes.extend_from_slice(&buffer[..read]);
                let request = String::from_utf8_lossy(&request_bytes);
                if request_complete(&request) {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            )?;
            write!(
                stream,
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"hel\"}}}}\n\n"
            )?;
            stream.flush()?;
            first_seen_rx
                .recv_timeout(Duration::from_secs(2))
                .context("first remote Anthropic SSE event was not observed before close")?;
            write!(
                stream,
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"lo\"}}}}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
            )?;
            stream.flush()?;
            Ok(request)
        });

        let endpoint = format!("http://127.0.0.1:{port}/anthropic-stream");
        let chat_request = ChatRequest {
            model: None,
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: "hello".to_owned(),
            }],
            max_tokens: Some(16),
            rocm_tools: false,
        };
        let body = anthropic_stream_chat_request_body("claude-test", &chat_request);
        let mut events = Vec::new();

        {
            let mut on_event = |event: ProviderStreamEvent| {
                if event.content == "hel" {
                    let _ = first_seen_tx.send(());
                }
                events.push(event);
                Ok(())
            };
            let mut emitter = AnthropicSseEmitter::new(&mut on_event);
            stream_json_with_headers(
                &endpoint,
                &[
                    ("x-api-key", "test-anthropic-key".to_owned()),
                    ("anthropic-version", "2023-06-01".to_owned()),
                    ("Content-Type", "application/json".to_owned()),
                    ("Accept", "text/event-stream".to_owned()),
                ],
                &body,
                &mut emitter,
            )?;
            emitter.finish()?;
        }
        let request = server.join().expect("server thread should not panic")?;

        assert!(request.starts_with("POST /anthropic-stream HTTP/1.1"));
        assert!(request.contains("\"stream\":true"));
        assert!(request.contains("\"model\":\"claude-test\""));
        assert!(request.contains("anthropic-version: 2023-06-01"));
        assert_eq!(
            events,
            vec![
                ProviderStreamEvent {
                    content: "hel".to_owned(),
                    done: false
                },
                ProviderStreamEvent {
                    content: "lo".to_owned(),
                    done: false
                },
                ProviderStreamEvent {
                    content: String::new(),
                    done: true
                }
            ]
        );
        Ok(())
    }

    struct ModelsOnlyServer {
        port: u16,
        stop_tx: mpsc::Sender<()>,
        handle: thread::JoinHandle<Result<usize>>,
    }

    impl ModelsOnlyServer {
        fn port(&self) -> u16 {
            self.port
        }

        fn stop(self) -> Result<usize> {
            let _ = self.stop_tx.send(());
            match self.handle.join() {
                Ok(result) => result,
                Err(_) => bail!("models server thread panicked"),
            }
        }
    }

    fn spawn_models_only_server(model_id: impl Into<String>) -> Result<ModelsOnlyServer> {
        let model_id = model_id.into();
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();
        let (stop_tx, stop_rx) = mpsc::channel();
        let handle = thread::spawn(move || -> Result<usize> {
            let mut served = 0;
            loop {
                if stop_rx.try_recv().is_ok() {
                    return Ok(served);
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                        let request = read_http_request(&mut stream)?;
                        if request.starts_with("GET /v1/models ") {
                            write_models_response(&mut stream, &model_id)?;
                        } else {
                            write_not_found_response(&mut stream)?;
                        }
                        served += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => return Err(error).context("models server accept failed"),
                }
            }
        });
        Ok(ModelsOnlyServer {
            port,
            stop_tx,
            handle,
        })
    }

    fn spawn_local_provider_test_server<F>(
        listener: TcpListener,
        model_id: impl Into<String>,
        mut responder: F,
    ) -> thread::JoinHandle<Result<String>>
    where
        F: FnMut(&mut TcpStream, &str) -> Result<()> + Send + 'static,
    {
        let model_id = model_id.into();
        thread::spawn(move || -> Result<String> {
            loop {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let request = read_http_request(&mut stream)?;
                if request.starts_with("GET /v1/models ") {
                    write_models_response(&mut stream, &model_id)?;
                    continue;
                }
                responder(&mut stream, &request)?;
                return Ok(request);
            }
        })
    }

    fn read_http_request(stream: &mut TcpStream) -> Result<String> {
        let mut request_bytes = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let read = stream.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            request_bytes.extend_from_slice(&buffer[..read]);
            let request = String::from_utf8_lossy(&request_bytes);
            if request_complete(&request) {
                break;
            }
        }
        Ok(String::from_utf8_lossy(&request_bytes).into_owned())
    }

    fn write_models_response(stream: &mut TcpStream, model_id: &str) -> Result<()> {
        let body = serde_json::json!({ "data": [{ "id": model_id }] }).to_string();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )?;
        Ok(())
    }

    fn write_not_found_response(stream: &mut TcpStream) -> Result<()> {
        let body = r#"{"error":"not found"}"#;
        write!(
            stream,
            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )?;
        Ok(())
    }

    fn request_complete(request: &str) -> bool {
        let Some((headers, body)) = request.split_once("\r\n\r\n") else {
            return false;
        };
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                key.eq_ignore_ascii_case("Content-Length")
                    .then(|| value.trim())
            })
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        body.len() >= content_length
    }

    fn temp_app_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("providers")
            .join(format!(
                "rocm-provider-{name}-{}-{}",
                std::process::id(),
                unix_time_millis()
            ));
        let _ = fs::remove_dir_all(&root);
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }
}
