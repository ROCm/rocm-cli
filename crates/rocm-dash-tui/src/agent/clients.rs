// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `AgentClient` backend implementations: Rig (OpenAI-compatible), ChatGPT
//! (OAuth device-code), Anthropic (Claude), and the in-memory `MockAgentClient`.
//!
//! Split out of `agent.rs` to keep the `AgentClient` seam focused. The shared
//! seam types (`StateSnapshot`, `AgentError`, `AgentClient`, `InferenceParams`,
//! `REQUEST_TIMEOUT`) stay in `agent/mod.rs`; the rig `Tool` registration these
//! backends call into lives in `agent::tools`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use tracing::{error, info, warn};

use crate::app::{ChatRole, ChatTurn};
use crate::client::ClientMsg;
use crate::llm::LlmConfig;
use crate::tool_exec::SharedRocmToolExecutor;

use tokio::sync::mpsc::UnboundedSender;

use super::tools::{
    FiredLog, register_rocm_mutating_tools, register_rocm_read_tools, register_telemetry_tools,
};
use super::{AgentClient, AgentError, InferenceParams, REQUEST_TIMEOUT, StateSnapshot};

/// Max tool-calling turns the model may take before producing a final answer.
const MAX_TOOL_TURNS: usize = 5;

/// Max tokens the model may emit in its final answer. Shared by all three
/// backends (RigAgentClient, ChatGptAgentClient, AnthropicAgentClient) so the
/// budget is defined once. `u64` to match rig's `AgentBuilder::max_tokens`.
/// Used as the fallback when no explicit `--max-tokens` is configured.
const MAX_AGENT_TOKENS: u64 = 1024;

/// Apply the optional sampling controls to a fresh Rig `AgentBuilder`.
///
/// `max_tokens` always resolves to a concrete value (the CLI override or the
/// shared [`MAX_AGENT_TOKENS`] default). `temperature` maps to the builder's
/// native `.temperature()`; `top_p` has no dedicated builder method in
/// rig-core, so it rides in `.additional_params({"top_p": ..})`, which the
/// provider merges into the request body. Both are set only when supplied, so
/// an unset knob leaves the request untouched.
fn apply_inference_params<M, P>(
    builder: rig::agent::AgentBuilder<M, P>,
    params: &InferenceParams,
) -> rig::agent::AgentBuilder<M, P>
where
    M: rig::completion::CompletionModel,
    P: rig::agent::PromptHook<M>,
{
    let mut builder = builder.max_tokens(params.max_tokens.map_or(MAX_AGENT_TOKENS, u64::from));
    if let Some(temperature) = params.temperature {
        builder = builder.temperature(f64::from(temperature));
    }
    if let Some(top_p) = params.top_p {
        builder = builder.additional_params(serde_json::json!({ "top_p": top_p }));
    }
    builder
}

fn validate_chatgpt_inference_params(params: &InferenceParams) -> Result<(), AgentError> {
    if params.temperature.is_some() && params.top_p.is_some() {
        return Err(AgentError::Build(
            "ChatGPT Responses accepts either temperature or top_p, not both".to_string(),
        ));
    }
    Ok(())
}

/// Default system preamble for the dashboard assistant.
///
/// The fallback only — a live dash replaces it via `with_preamble` with the
/// bin-composed prompt, which carries the ROCm tool-use rules AND this
/// machine's detected facts. This text stands alone for demo/replay/`--chat-mock`,
/// which have no bin seam to compose one.
const DEFAULT_PREAMBLE: &str = "You are the rocm-dash assistant, embedded in a terminal dashboard for AMD \
     Instinct GPU telemetry and benchmarks. Use the provided tools (gpu_status, \
     list_instances, bench_summary, tokens_per_watt) to answer questions about \
     live GPU, serving instance, and benchmark state. Prefer short, direct answers.";

/// Give one backend a `with_preamble` override for its system prompt.
///
/// All three backends carry the same `preamble: String`, and all three must
/// accept the bin's host-grounded prompt or the grounding would depend on which
/// provider the operator happened to pick. One macro keeps that parity the way
/// the tool-registration helpers already do.
///
/// An override is applied post-construction rather than as a `new` parameter so
/// the constructors keep their signatures; `None` or a blank string leaves
/// [`DEFAULT_PREAMBLE`] in place, which is what demo/replay/`--chat-mock` pass.
macro_rules! impl_with_preamble {
    ($ty:ident) => {
        impl $ty {
            #[must_use]
            pub fn with_preamble(mut self, preamble: Option<String>) -> Self {
                if let Some(prompt) = preamble.filter(|p| !p.trim().is_empty()) {
                    self.preamble = prompt;
                }
                self
            }

            /// The system prompt this client sends. Test-only: it exists so the
            /// override can be pinned without a network round-trip.
            #[cfg(test)]
            pub(crate) fn preamble(&self) -> &str {
                &self.preamble
            }
        }
    };
}

impl_with_preamble!(RigAgentClient);
impl_with_preamble!(ChatGptAgentClient);
impl_with_preamble!(AnthropicAgentClient);

/// Map our TUI-local turns to Rig messages, preserving role + order.
///
/// `Error` and `System` turns are UI-local annotations and are never sent to the
/// model (a `System` notice like "switched to local" is not something the
/// assistant said, so forwarding it would corrupt the model's context). Pure: no
/// I/O.
pub fn build_messages(turns: &[ChatTurn]) -> Vec<rig::completion::Message> {
    turns
        .iter()
        .filter_map(|t| match t.role {
            ChatRole::User => Some(rig::completion::Message::user(t.content.clone())),
            ChatRole::Agent => Some(rig::completion::Message::assistant(t.content.clone())),
            ChatRole::Error | ChatRole::System => None,
        })
        .collect()
}

/// Append a "via: tool, tool" annotation so the operator can see which Skills
/// fired. Deduplicates, preserving first-seen order. Pure.
pub fn annotate_reply(reply: String, skills: &[String]) -> String {
    if skills.is_empty() {
        return reply;
    }
    let mut seen: Vec<String> = Vec::new();
    for s in skills {
        if !seen.contains(s) {
            seen.push(s.clone());
        }
    }
    format!("{reply}\n⚙ via: {}", seen.join(", "))
}

/// Drive a built rig prompt request to completion under the shared
/// [`REQUEST_TIMEOUT`], then annotate the reply with the Skills that fired.
///
/// The shared `complete()` tail, called from [`run_agent_request`] — the one
/// call site all three backends (RigAgentClient, ChatGptAgentClient,
/// AnthropicAgentClient) now go through. Their `req` types differ (rig
/// typestate), so this is generic over `IntoFuture<Output = Result<String, E>>`
/// with a `Display` error. A timeout maps to [`AgentError::Timeout`]; a backend
/// error maps to [`AgentError::Request`]; success is annotated with the fired
/// Skills. ONE definition of the tail.
///
/// `backend` tags the lifecycle trace events (request-start / completion /
/// error / timeout) so a hung or failing chat is traceable to a specific
/// provider in the client log. The request is non-streaming (rig's
/// `prompt().await` resolves once with the full reply), so there is no
/// separate first-byte instant to record — "first-byte" and "completion" are
/// the same instant here; the elapsed time logged on success is the
/// request's effective TTFT.
async fn finish_agent_request<F, E>(
    backend: &'static str,
    req: F,
    fired: &FiredLog,
) -> Result<String, AgentError>
where
    F: std::future::IntoFuture<Output = Result<String, E>>,
    E: std::fmt::Display,
{
    let started = Instant::now();
    info!(backend, "chat request start");
    let reply = match tokio::time::timeout(REQUEST_TIMEOUT, req.into_future()).await {
        Err(_) => {
            warn!(
                backend,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "chat request timed out"
            );
            return Err(AgentError::Timeout);
        }
        Ok(Err(e)) => {
            error!(
                backend,
                elapsed_ms = started.elapsed().as_millis() as u64,
                error = %e,
                "chat request failed"
            );
            return Err(AgentError::Request(e.to_string()));
        }
        Ok(Ok(reply)) => reply,
    };
    let skills = fired.lock().map(|g| g.clone()).unwrap_or_default();
    info!(
        backend,
        elapsed_ms = started.elapsed().as_millis() as u64,
        reply_len = reply.len(),
        skills_fired = skills.len(),
        "chat request complete"
    );
    Ok(annotate_reply(reply, &skills))
}

/// Drive one `complete()` call from a preamble'd `AgentBuilder` through to a
/// reply string: apply the sampling knobs, register the SAME three tool sets
/// (telemetry/skill, ROCm read, ROCm mutating) in the SAME order, build the
/// prompt request with the shared history/turn-limit, and hand off to
/// [`finish_agent_request`].
///
/// The ONE definition of what used to be triplicated across
/// `RigAgentClient::complete`, `ChatGptAgentClient::complete`, and
/// `AnthropicAgentClient::complete` (ROCMAI-200). Each backend still builds its
/// own agent (client/model construction and any backend-specific setup, e.g.
/// ChatGPT's device-code auth, differ) and still owns its own backend tag; the
/// `history` split (and its `AgentError::Empty` case) lives here, the one
/// shared site. A backend with ordering constraints (ChatGPT's auth must not
/// run on an empty history) does its own cheap `history.is_empty()` guard
/// first — see `ChatGptAgentClient::complete` — without needing the split.
#[allow(clippy::too_many_arguments)]
async fn run_agent_request<M, P>(
    agent: rig::agent::AgentBuilder<M, P>,
    params: &InferenceParams,
    executor: Option<&SharedRocmToolExecutor>,
    approval_tx: Option<&UnboundedSender<ClientMsg>>,
    history: &[ChatTurn],
    snapshot: StateSnapshot,
    backend: &'static str,
) -> Result<String, AgentError>
where
    M: rig::completion::CompletionModel + 'static,
    P: rig::agent::PromptHook<M> + 'static,
{
    use rig::completion::Prompt;

    let Some((last, prior)) = history.split_last() else {
        return Err(AgentError::Empty);
    };

    let snap = Arc::new(snapshot);
    let fired: FiredLog = Arc::new(Mutex::new(Vec::new()));

    let agent = apply_inference_params(agent, params);
    // Telemetry + skill registry tools (shared registration site).
    let agent = register_telemetry_tools(agent, &snap, &fired);
    // Read-only ROCm machine-inspection tools (forward across the seam).
    let agent = register_rocm_read_tools(agent, executor, &fired);
    // Mutating ROCm tools (surface approval; never execute in the rig loop).
    let agent = register_rocm_mutating_tools(agent, executor, approval_tx, &fired).build();

    let req = agent
        .prompt(last.content.clone())
        .max_turns(MAX_TOOL_TURNS)
        .with_history(build_messages(prior));

    finish_agent_request(backend, req, &fired).await
}

/// Live Rig-backed client for an OpenAI-compatible endpoint. The Rig client is
/// constructed once; the agent + tools are rebuilt per request from the
/// captured snapshot.
pub struct RigAgentClient {
    client: rig::providers::openai::CompletionsClient,
    model: String,
    preamble: String,
    /// Optional sampling controls (temperature/top_p/max_tokens).
    params: InferenceParams,
    /// Bin-injected tool executor (None for tests / no live seam).
    executor: Option<SharedRocmToolExecutor>,
    /// Channel to surface mutating-tool approval intents to the app (None for
    /// tests / no live seam). Mutating tools post here instead of executing.
    approval_tx: Option<UnboundedSender<ClientMsg>>,
}

impl RigAgentClient {
    pub fn new(
        cfg: LlmConfig,
        params: InferenceParams,
        executor: Option<SharedRocmToolExecutor>,
        approval_tx: Option<UnboundedSender<ClientMsg>>,
    ) -> Result<Self, AgentError> {
        // Custom-auth gateway (e.g. Azure APIM `Ocp-Apim-Subscription-Key`):
        // the key goes in a custom header, NOT `Authorization: Bearer`. Rig
        // still requires an api_key, so pass a dummy Bearer the gateway ignores.
        let custom_headers = match (cfg.auth_header.as_deref(), cfg.api_key.as_deref()) {
            (Some(name), Some(key)) => Some(auth_header_map(name, key)?),
            _ => None,
        };
        // Bearer carries the real key ONLY in the standard (no custom header)
        // case; otherwise a dummy (local endpoints / custom-header gateways).
        let bearer = match (&custom_headers, cfg.api_key.as_deref()) {
            (None, Some(key)) => key.to_string(),
            _ => "sk-no-key".to_string(),
        };

        // `.api_key()` sets the builder typestate, so it must be in the chain;
        // `.base_url()` / `.http_headers()` return Self and can follow.
        let mut builder = rig::providers::openai::CompletionsClient::builder()
            .api_key(&bearer)
            .base_url(&cfg.base_url);
        if let Some(headers) = custom_headers {
            builder = builder.http_headers(headers);
        }

        let client = builder
            .build()
            .map_err(|e| AgentError::Build(e.to_string()))?;
        Ok(Self {
            client,
            model: cfg.model,
            preamble: DEFAULT_PREAMBLE.to_string(),
            params,
            executor,
            approval_tx,
        })
    }
}

/// Build a single-entry `HeaderMap` carrying the gateway's custom auth header.
/// The value is marked sensitive so the HTTP stack won't log it; errors never
/// embed the key value.
fn auth_header_map(name: &str, value: &str) -> Result<http::HeaderMap, AgentError> {
    let header_name = http::HeaderName::from_bytes(name.as_bytes())
        .map_err(|e| AgentError::Build(format!("invalid chat_auth_header name: {e}")))?;
    let mut header_value = http::HeaderValue::from_str(value)
        .map_err(|_| AgentError::Build("invalid auth header value".to_string()))?;
    header_value.set_sensitive(true);
    let mut map = http::HeaderMap::new();
    map.insert(header_name, header_value);
    Ok(map)
}

#[async_trait]
impl AgentClient for RigAgentClient {
    async fn complete(
        &self,
        history: &[ChatTurn],
        snapshot: StateSnapshot,
    ) -> Result<String, AgentError> {
        use rig::client::CompletionClient;

        let agent = self.client.agent(&self.model).preamble(&self.preamble);
        run_agent_request(
            agent,
            &self.params,
            self.executor.as_ref(),
            self.approval_tx.as_ref(),
            history,
            snapshot,
            "rig-openai",
        )
        .await
    }
}

/// No-key ChatGPT backend over Rig's native `chatgpt` OAuth provider.
///
/// This is the no-key default that restores the ChatGPT device-login the vendored Codex
/// path provided. It takes NO api_key (the env-only key invariant is untouched:
/// this path authenticates with an OAuth device-code flow, not a key). The
/// `on_device_code` callback surfaces the verification URL + user code so the
/// chat tab can show the operator how to sign in; the resulting token is
/// persisted by the provider so re-launches don't re-prompt.
pub struct ChatGptAgentClient {
    client: rig::providers::chatgpt::Client,
    model: String,
    preamble: String,
    /// Optional sampling controls (temperature/top_p/max_tokens).
    params: InferenceParams,
    /// Bin-injected tool executor (None for tests / no live seam).
    executor: Option<SharedRocmToolExecutor>,
    /// Channel to surface mutating-tool approval intents to the app (None for
    /// tests / no live seam).
    approval_tx: Option<UnboundedSender<ClientMsg>>,
}

impl ChatGptAgentClient {
    /// Build the OAuth client. `model` defaults to the provider's Codex model.
    /// `on_device_code(verification_uri, user_code)` is invoked during the first
    /// `authorize()` (device-code flow). No network I/O happens here — login is
    /// deferred to the first `complete()`.
    pub fn new<F>(
        model: Option<String>,
        params: InferenceParams,
        on_device_code: F,
        executor: Option<SharedRocmToolExecutor>,
        approval_tx: Option<UnboundedSender<ClientMsg>>,
    ) -> Result<Self, AgentError>
    where
        F: Fn(String, String) + Send + Sync + 'static,
    {
        use rig::providers::chatgpt;
        // Store OAuth tokens in a user-owned directory so the plaintext
        // access/refresh tokens are not readable by other local users.
        // Refuse to fall back to a predictable temp path: a pre-existing
        // directory there could be owned by another user.
        let token_dir = std::env::var_os("HOME")
            .filter(|v| !v.is_empty())
            .or_else(|| std::env::var_os("USERPROFILE").filter(|v| !v.is_empty())) // Windows
            .map(|h| {
                std::path::PathBuf::from(h)
                    .join(".rocm")
                    .join("data")
                    .join("chatgpt-tokens")
            })
            .ok_or_else(|| {
                AgentError::Build(
                    "home directory not found ($HOME/$USERPROFILE unset); \
                     cannot safely store OAuth tokens"
                        .to_string(),
                )
            })?;
        #[cfg(unix)]
        {
            // Create with mode 0o700 up-front so the directory is never
            // world-accessible even for the brief window before set_permissions.
            // The subsequent set_permissions call tightens pre-existing dirs.
            use std::os::unix::fs::DirBuilderExt;
            use std::os::unix::fs::PermissionsExt;
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(&token_dir)
                .map_err(|e| AgentError::Build(format!("cannot create token cache dir: {e}")))?;
            if let Err(e) =
                std::fs::set_permissions(&token_dir, std::fs::Permissions::from_mode(0o700))
            {
                tracing::warn!(dir = %token_dir.display(), error = %e, "could not restrict token cache dir permissions");
            }
        }
        #[cfg(not(unix))]
        std::fs::create_dir_all(&token_dir)
            .map_err(|e| AgentError::Build(format!("cannot create token cache dir: {e}")))?;
        // The closure param is the provider's `DeviceCodePrompt` (its `auth`
        // module is private, so we let inference name it); its `verification_uri`
        // and `user_code` fields are public.
        let client = chatgpt::Client::builder()
            .oauth()
            .token_dir(token_dir)
            .on_device_code(move |p| on_device_code(p.verification_uri, p.user_code))
            .build()
            .map_err(|e| AgentError::Build(e.to_string()))?;
        Ok(Self {
            client,
            model: model.unwrap_or_else(|| chatgpt::GPT_5_3_CODEX.to_string()),
            preamble: DEFAULT_PREAMBLE.to_string(),
            params,
            executor,
            approval_tx,
        })
    }
}

#[async_trait]
impl AgentClient for ChatGptAgentClient {
    async fn complete(
        &self,
        history: &[ChatTurn],
        snapshot: StateSnapshot,
    ) -> Result<String, AgentError> {
        use rig::agent::AgentBuilder;
        use rig::providers::chatgpt::ResponsesCompletionModel;

        // Cheap pre-check: reject an empty history before device login or
        // network I/O. `run_agent_request` re-derives this from `history`
        // itself (the one shared split site); this guard only exists for the
        // ordering, not the check.
        if history.is_empty() {
            return Err(AgentError::Empty);
        }

        // OpenAI's Responses API defines temperature and top_p as mutually
        // exclusive. Reject the combination before device login or network I/O.
        validate_chatgpt_inference_params(&self.params)?;

        // Device-code login on first use; the provider caches the token after.
        // A failed/declined sign-in is an Auth error (distinct from a build
        // failure), surfaced as a clear error turn — never leaks the token.
        self.client
            .authorize()
            .await
            .map_err(|e| AgentError::Auth(e.to_string()))?;

        let model = ResponsesCompletionModel::new(self.client.clone(), self.model.clone());
        let agent = AgentBuilder::new(model).preamble(&self.preamble);
        run_agent_request(
            agent,
            &self.params,
            self.executor.as_ref(),
            self.approval_tx.as_ref(),
            history,
            snapshot,
            "chatgpt-oauth",
        )
        .await
    }
}

/// Live Rig-backed client for Anthropic's Claude API.
///
/// Mirrors [`RigAgentClient`]: the Rig client is constructed once; the agent +
/// the SAME ROCm read/mutating tool set are rebuilt per request from the
/// captured snapshot, so tool + approval parity holds across every backend. The
/// key rides in `x-api-key` (handled inside the provider) — never in `base_url`
/// or the request path — so no key leaks into [`AgentError`] strings.
pub struct AnthropicAgentClient {
    client: rig::providers::anthropic::Client,
    model: String,
    preamble: String,
    /// Optional sampling controls (temperature/top_p/max_tokens).
    params: InferenceParams,
    /// Bin-injected tool executor (None for tests / no live seam).
    executor: Option<SharedRocmToolExecutor>,
    /// Channel to surface mutating-tool approval intents to the app (None for
    /// tests / no live seam). Mutating tools post here instead of executing.
    approval_tx: Option<UnboundedSender<ClientMsg>>,
}

impl AnthropicAgentClient {
    /// Build the Anthropic client from an [`LlmConfig`]. `api_key` is required
    /// (env / secure-store sourced by the bin and carried in-process via the
    /// seam — never argv). `base_url` is intentionally ignored: the provider's
    /// own default (`https://api.anthropic.com`) is used. An empty `model`
    /// falls back to [`CLAUDE_SONNET_4_6`](rig::providers::anthropic::completion::CLAUDE_SONNET_4_6).
    /// No network I/O happens here — the request is deferred to `complete()`.
    pub fn new(
        cfg: LlmConfig,
        params: InferenceParams,
        executor: Option<SharedRocmToolExecutor>,
        approval_tx: Option<UnboundedSender<ClientMsg>>,
    ) -> Result<Self, AgentError> {
        let key = cfg
            .api_key
            .as_deref()
            .ok_or_else(|| AgentError::Build("anthropic requires ANTHROPIC_API_KEY".to_string()))?;
        // Builder typestate: `.api_key()` then `.build()`. Leave base_url at the
        // provider default so we never point Claude at a non-Anthropic host.
        let client = rig::providers::anthropic::Client::builder()
            .api_key(key)
            .build()
            .map_err(|e| AgentError::Build(e.to_string()))?;
        let model = if cfg.model.is_empty() {
            rig::providers::anthropic::completion::CLAUDE_SONNET_4_6.to_string()
        } else {
            cfg.model
        };
        Ok(Self {
            client,
            model,
            preamble: DEFAULT_PREAMBLE.to_string(),
            params,
            executor,
            approval_tx,
        })
    }
}

#[async_trait]
impl AgentClient for AnthropicAgentClient {
    async fn complete(
        &self,
        history: &[ChatTurn],
        snapshot: StateSnapshot,
    ) -> Result<String, AgentError> {
        use rig::client::CompletionClient;

        // Tool registration is delegated to run_agent_request, the single
        // shared site all three backends (RigAgentClient, ChatGptAgentClient,
        // AnthropicAgentClient) now call: the SAME telemetry/skill tools +
        // every ROCm read + mutating tool, so capability and approval
        // behavior are uniform across backends.
        let agent = self.client.agent(&self.model).preamble(&self.preamble);
        run_agent_request(
            agent,
            &self.params,
            self.executor.as_ref(),
            self.approval_tx.as_ref(),
            history,
            snapshot,
            "anthropic",
        )
        .await
    }
}

/// When the last user turn contains `phrase` (case-insensitive), the mock
/// surfaces `intent` for approval over `tx` — mirroring what a real
/// `rocm_mutating_tool!`'s `call()` does from inside the rig tool loop — instead
/// of returning the normal canned reply. Lets `--chat-mock` drive e2e coverage
/// of the deny-by-default approval modal without a live model or a real
/// [`crate::tool_exec::RocmToolExecutor`].
struct MockApprovalTrigger {
    phrase: String,
    intent: crate::tool_exec::ApprovalIntent,
    tx: UnboundedSender<ClientMsg>,
}

/// Deterministic in-memory client for tests and the offline demo. Never touches
/// the network. Can emit a canned tool-calling-style answer (cites a Skill).
pub struct MockAgentClient {
    reply: String,
    fail: bool,
    cited: Vec<String>,
    approval: Option<MockApprovalTrigger>,
}

impl MockAgentClient {
    /// A mock that returns a fixed canned reply.
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
            fail: false,
            cited: Vec::new(),
            approval: None,
        }
    }

    /// A mock that returns a canned reply annotated as if `tool_name` fired —
    /// drives the offline tool-calling demo deterministically.
    pub fn with_tool_call(reply: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
            fail: false,
            cited: vec![tool_name.into()],
            approval: None,
        }
    }

    /// Like [`Self::with_tool_call`], but when the last user message contains
    /// `phrase` (case-insensitive) the mock instead sends `intent` over
    /// `approval_tx` as a `ClientMsg::ChatApprovalRequired` and replies with a
    /// "surfaced for approval" note — no tool actually executes.
    pub fn with_tool_call_and_approval_trigger(
        reply: impl Into<String>,
        tool_name: impl Into<String>,
        phrase: impl Into<String>,
        intent: crate::tool_exec::ApprovalIntent,
        approval_tx: UnboundedSender<ClientMsg>,
    ) -> Self {
        Self {
            approval: Some(MockApprovalTrigger {
                phrase: phrase.into().to_lowercase(),
                intent,
                tx: approval_tx,
            }),
            ..Self::with_tool_call(reply, tool_name)
        }
    }

    /// A mock whose `complete` always fails (to exercise the error path).
    pub const fn failing() -> Self {
        Self {
            reply: String::new(),
            fail: true,
            cited: Vec::new(),
            approval: None,
        }
    }
}

#[async_trait]
impl AgentClient for MockAgentClient {
    async fn complete(
        &self,
        history: &[ChatTurn],
        _snapshot: StateSnapshot,
    ) -> Result<String, AgentError> {
        if self.fail {
            return Err(AgentError::Request("mock failure".to_string()));
        }
        if history.is_empty() {
            return Err(AgentError::Empty);
        }
        if let Some(trigger) = &self.approval {
            let fires = matches!(
                history.last(),
                Some(t) if t.role == ChatRole::User
                    && t.content.to_lowercase().contains(&trigger.phrase)
            );
            if fires {
                if let Err(e) = trigger.tx.send(ClientMsg::ChatApprovalRequired {
                    intent: trigger.intent.clone(),
                }) {
                    warn!(error = %e, "mock approval trigger dropped: receiver gone");
                }
                return Ok(
                    "This action needs operator approval; it has been surfaced to \
                           the operator."
                        .to_string(),
                );
            }
        }
        Ok(annotate_reply(self.reply.clone(), &self.cited))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::agent::{
        ROCM_MUTATING_TOOL_NAMES, ROCM_READ_TOOL_NAMES, SKILL_NAMES, fixture_snapshot,
    };

    #[test]
    fn build_messages_preserves_role_and_order_and_drops_errors() {
        let turns = vec![
            ChatTurn::user("first user"),
            ChatTurn::agent("first agent"),
            ChatTurn::error("local error annotation"),
            ChatTurn::user("second user"),
        ];
        let msgs = build_messages(&turns);
        assert_eq!(msgs.len(), 3);
        let dbg = format!("{msgs:?}");
        let i_first_user = dbg.find("first user").expect("first user present");
        let i_first_agent = dbg.find("first agent").expect("first agent present");
        let i_second_user = dbg.find("second user").expect("second user present");
        assert!(i_first_user < i_first_agent);
        assert!(i_first_agent < i_second_user);
        assert!(!dbg.contains("local error annotation"));
    }

    #[test]
    fn annotate_reply_appends_deduped_skills() {
        let r = annotate_reply("hi".into(), &["gpu_status".into(), "gpu_status".into()]);
        assert!(r.contains("hi"));
        assert!(r.contains("via: gpu_status"));
        // No skills → unchanged.
        assert_eq!(annotate_reply("hi".into(), &[]), "hi");
    }

    #[tokio::test]
    async fn mock_tool_calling_answer_cites_skill() {
        // The offline "what's GPU-2 doing?" demo path — no live LLM.
        let agent =
            MockAgentClient::with_tool_call("GPU-2 is at 87% util, 71°C, 250 W.", "gpu_status");
        let history = vec![ChatTurn::user("what's GPU-2 doing?")];
        let reply = agent
            .complete(&history, fixture_snapshot())
            .await
            .expect("mock reply");
        assert!(reply.contains("87% util"));
        assert!(
            reply.contains("gpu_status"),
            "reply cites the Skill that fired"
        );
    }

    #[tokio::test]
    async fn mock_error_path_is_err_not_panic() {
        let agent = MockAgentClient::failing();
        let history = vec![ChatTurn::user("hi")];
        let err = agent
            .complete(&history, StateSnapshot::default())
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::Request(_)));
    }

    #[tokio::test]
    async fn mock_empty_history_is_empty_error() {
        let agent = MockAgentClient::new("x");
        let err = agent
            .complete(&[], StateSnapshot::default())
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::Empty));
    }

    #[tokio::test]
    async fn approval_trigger_does_not_refire_on_follow_up() {
        // Hand-builds the history shape that `app::on_approval_result`
        // produces in production (it appends the approved-action result as
        // an Agent turn, not a new User turn) and feeds it straight to
        // `MockAgentClient::complete` — this does not call
        // `on_approval_result` itself. It pins the mock's own approval
        // trigger to key off the *last* turn only, so a follow-up call
        // seeing `[User(trigger), Agent(result)]` does not re-surface
        // approval a second time.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ClientMsg>();
        let agent = MockAgentClient::with_tool_call_and_approval_trigger(
            "all good",
            "gpu_status",
            "install the sdk",
            crate::tool_exec::ApprovalIntent {
                title: "Install TheRock ROCm SDK?".to_string(),
                body: vec!["install_sdk".to_string()],
                name: "install_sdk".to_string(),
                arguments: json!({}),
            },
            tx,
        );

        let first = vec![ChatTurn::user("please install the sdk")];
        let reply = agent
            .complete(&first, fixture_snapshot())
            .await
            .expect("mock reply");
        assert!(reply.contains("surfaced to"));

        let follow_up = vec![
            ChatTurn::user("please install the sdk"),
            ChatTurn::agent("Installed successfully."),
        ];
        let reply = agent
            .complete(&follow_up, fixture_snapshot())
            .await
            .expect("mock reply");
        assert!(
            !reply.contains("surfaced to"),
            "approval trigger re-fired on the automatic follow-up: {reply}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn rig_client_stores_inference_params() {
        // Construction is offline (no network until complete()), so this pins
        // that the sampling knobs survive into the client that applies them to
        // every request builder. A default (all-None) client leaves them unset.
        let cfg = LlmConfig {
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            model: "local-model".to_string(),
            api_key: None,
            auth_header: None,
        };
        let params = InferenceParams {
            temperature: Some(0.25),
            top_p: Some(0.5),
            max_tokens: Some(512),
        };
        let client =
            RigAgentClient::new(cfg.clone(), params, None, None).expect("build rig client");
        assert_eq!(client.params, params);
        let client_default =
            RigAgentClient::new(cfg, InferenceParams::default(), None, None).expect("build");
        assert_eq!(client_default.params, InferenceParams::default());
    }

    #[test]
    fn a_bin_composed_prompt_replaces_the_default_preamble() {
        // Construction is offline, so this pins that the host-grounded prompt
        // the bin composes actually reaches the field every request's system
        // message is built from — and that demo/replay/mock, which pass None,
        // keep the standalone default.
        let cfg = LlmConfig {
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            model: "local-model".to_string(),
            api_key: None,
            auth_header: None,
        };
        let grounded = "You are ROCm CLI's local assistant. Operating system: Linux.";
        let client = RigAgentClient::new(cfg.clone(), InferenceParams::default(), None, None)
            .expect("build rig client")
            .with_preamble(Some(grounded.to_string()));
        assert_eq!(client.preamble(), grounded);

        for absent in [None, Some(String::new()), Some("   ".to_string())] {
            let fallback = RigAgentClient::new(cfg.clone(), InferenceParams::default(), None, None)
                .expect("build rig client")
                .with_preamble(absent.clone());
            assert_eq!(
                fallback.preamble(),
                DEFAULT_PREAMBLE,
                "no usable prompt ({absent:?}) must leave the built-in default"
            );
        }

        // Same for the other two backends: grounding must not depend on which
        // provider the operator picked, so all three accept the override.
        let chatgpt = ChatGptAgentClient::new(
            None,
            InferenceParams::default(),
            |_url, _code| {},
            None,
            None,
        )
        .expect("build chatgpt oauth client")
        .with_preamble(Some(grounded.to_string()));
        assert_eq!(chatgpt.preamble(), grounded);

        let anthropic = AnthropicAgentClient::new(
            LlmConfig {
                base_url: String::new(),
                model: String::new(),
                api_key: Some("dummy".to_string()),
                auth_header: None,
            },
            InferenceParams::default(),
            None,
            None,
        )
        .expect("build anthropic client")
        .with_preamble(Some(grounded.to_string()));
        assert_eq!(anthropic.preamble(), grounded);
    }

    #[test]
    fn chatgpt_rejects_temperature_and_top_p_together() {
        let params = InferenceParams {
            temperature: Some(0.25),
            top_p: Some(0.5),
            max_tokens: None,
        };
        let error = validate_chatgpt_inference_params(&params)
            .expect_err("Responses API must reject mutually exclusive controls");
        assert!(matches!(error, AgentError::Build(_)));
        assert!(error.to_string().contains("either temperature or top_p"));

        assert!(
            validate_chatgpt_inference_params(&InferenceParams {
                temperature: Some(0.25),
                ..InferenceParams::default()
            })
            .is_ok()
        );
        assert!(
            validate_chatgpt_inference_params(&InferenceParams {
                top_p: Some(0.5),
                ..InferenceParams::default()
            })
            .is_ok()
        );
    }

    /// Manual-demo verification of the live Rig path (tool-calling) against a
    /// local OpenAI-compatible endpoint. NOT run in CI (no live LLM). Run with:
    /// `cargo test -p rocm-dash-tui --lib rig_round_trip -- --ignored`
    /// after starting a local endpoint (e.g. vLLM/Ollama at 127.0.0.1:8000/v1).
    #[tokio::test]
    #[ignore = "requires a live local OpenAI-compatible endpoint"]
    async fn rig_round_trip_against_local_endpoint() {
        let cfg = LlmConfig {
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            model: "local-model".to_string(),
            api_key: None,
            auth_header: None,
        };
        let client = RigAgentClient::new(cfg, InferenceParams::default(), None, None)
            .expect("build rig client");
        let history = vec![ChatTurn::user("What's GPU-2 doing? Use the tools.")];
        let reply = client
            .complete(&history, fixture_snapshot())
            .await
            .expect("live reply");
        assert!(!reply.is_empty());
    }

    /// Live round-trip against an OpenAI-compatible endpoint with a custom auth
    /// header. NOT run in CI. Configure via environment variables:
    /// - `LLM_TEST_BASE_URL` (required): base URL of the endpoint
    /// - `LLM_TEST_API_KEY` (required): API key or subscription key
    /// - `LLM_TEST_AUTH_HEADER` (optional): auth header name; defaults to `Authorization` Bearer
    /// - `LLM_TEST_MODEL` (optional): model name; defaults to `gpt-4o-mini`
    ///
    /// Run with:
    /// `cargo test -p rocm-dash-tui --lib rig_round_trip_against_custom_endpoint -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requires LLM_TEST_BASE_URL and LLM_TEST_API_KEY environment variables"]
    async fn rig_round_trip_against_custom_endpoint() {
        let base_url = std::env::var("LLM_TEST_BASE_URL").expect("set LLM_TEST_BASE_URL");
        let key = std::env::var("LLM_TEST_API_KEY").expect("set LLM_TEST_API_KEY");
        let auth_header = std::env::var("LLM_TEST_AUTH_HEADER").ok();
        let model = std::env::var("LLM_TEST_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
        let cfg = LlmConfig {
            base_url,
            model,
            api_key: Some(key),
            auth_header,
        };
        let client = RigAgentClient::new(cfg, InferenceParams::default(), None, None)
            .expect("build rig client");
        let history = vec![ChatTurn::user("Reply with exactly: gateway ok")];
        let reply = client
            .complete(&history, fixture_snapshot())
            .await
            .expect("gateway reply");
        assert!(!reply.is_empty());
    }

    #[test]
    fn chatgpt_oauth_client_builds_offline_without_taking_a_key() {
        // Construction is offline (login is deferred to authorize() in
        // complete()); the device-code callback is wired but not yet invoked.
        // Crucially, the constructor signature takes NO api_key — the env-only
        // key invariant is structurally preserved on the no-key path.
        let fired = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = fired.clone();
        let client = ChatGptAgentClient::new(
            Some("gpt-5.3-codex".to_string()),
            InferenceParams::default(),
            move |url, code| {
                // Would surface in the chat tab during a real device-code login.
                sink.lock().unwrap().push(format!("{url}|{code}"));
            },
            None,
            None,
        )
        .expect("build chatgpt oauth client");
        assert_eq!(client.model, "gpt-5.3-codex");
        // No network happened, so the handler has not fired yet.
        assert!(fired.lock().unwrap().is_empty());
    }

    #[test]
    fn chatgpt_oauth_client_defaults_model_when_none() {
        let client = ChatGptAgentClient::new(
            None,
            InferenceParams::default(),
            |_url, _code| {},
            None,
            None,
        )
        .expect("build chatgpt oauth client");
        assert_eq!(
            client.model,
            rig::providers::chatgpt::GPT_5_3_CODEX,
            "the no-key default uses the provider's Codex model"
        );
    }

    #[test]
    fn anthropic_backend_constructs_and_is_agentclient() {
        // Offline construction with a dummy key (no network until complete()).
        // An empty model falls back to CLAUDE_SONNET_4_6.
        let client = AnthropicAgentClient::new(
            LlmConfig {
                base_url: String::new(),
                model: String::new(),
                api_key: Some("dummy".to_string()),
                auth_header: None,
            },
            InferenceParams::default(),
            None,
            None,
        )
        .expect("build anthropic client");
        assert_eq!(
            client.model,
            rig::providers::anthropic::completion::CLAUDE_SONNET_4_6,
            "empty model defaults to Claude Sonnet"
        );
        // It is usable behind the swappable `AgentClient` seam (object-safe).
        let _erased: Arc<dyn AgentClient> = Arc::new(client);
    }

    #[test]
    fn anthropic_requires_key() {
        // No api_key → a Build error naming the env var; nothing constructs.
        // (The Ok variant isn't Debug, so let-else rather than unwrap_err.)
        let Err(err) = AnthropicAgentClient::new(
            LlmConfig {
                base_url: String::new(),
                model: String::new(),
                api_key: None,
                auth_header: None,
            },
            InferenceParams::default(),
            None,
            None,
        ) else {
            panic!("expected a Build error without a key");
        };
        assert!(matches!(err, AgentError::Build(_)));
        assert!(
            err.to_string().contains("ANTHROPIC_API_KEY"),
            "error names the required env var: {err}"
        );
    }

    #[test]
    fn anthropic_honors_explicit_model() {
        let client = AnthropicAgentClient::new(
            LlmConfig {
                base_url: String::new(),
                model: "claude-opus-4-7".to_string(),
                api_key: Some("dummy".to_string()),
                auth_header: None,
            },
            InferenceParams::default(),
            None,
            None,
        )
        .expect("build anthropic client");
        assert_eq!(client.model, "claude-opus-4-7");
    }

    #[test]
    fn all_backends_register_same_rocm_tools() {
        // Contract: every backend's complete() registers the SAME three tool
        // sets — telemetry/skill (register_telemetry_tools), read ROCm
        // (register_rocm_read_tools), and mutating ROCm
        // (register_rocm_mutating_tools) — so capability + approval parity holds
        // across local/openai/anthropic. This enumerates the canonical sets that
        // those shared helpers register; the helper call sites now live in one
        // place, `run_agent_request`, which RigAgentClient, ChatGptAgentClient,
        // and AnthropicAgentClient all call into.
        // `run_agent_request_registers_tool_sets_and_enforces_turn_limit` below
        // asserts those tools are actually sent on the wire from that one site.
        //
        // Telemetry/skill set: register_telemetry_tools registers exactly the
        // SKILL_NAMES tools (GpuStatus, ListInstances, BenchSummary,
        // TokensPerWatt, ListSkills, SkillPlan). Pinning the size here means the
        // telemetry registration can't silently diverge from the canonical set.
        assert_eq!(SKILL_NAMES.len(), 6, "canonical telemetry/skill set size");
        for n in SKILL_NAMES {
            assert!(!n.is_empty(), "empty telemetry/skill tool name");
            // Telemetry tools are disjoint from both ROCm sets.
            assert!(
                !ROCM_READ_TOOL_NAMES.contains(&n),
                "telemetry tool {n} collides with read set"
            );
            assert!(
                !ROCM_MUTATING_TOOL_NAMES.contains(&n),
                "telemetry tool {n} collides with mutating set"
            );
        }
        assert_eq!(
            ROCM_READ_TOOL_NAMES.len(),
            14,
            "canonical read-tool set size"
        );
        assert_eq!(
            ROCM_MUTATING_TOOL_NAMES.len(),
            6,
            "canonical mutating-tool set size"
        );
        // The two sets are disjoint (no tool is both read and mutating).
        for n in ROCM_MUTATING_TOOL_NAMES {
            assert!(
                !ROCM_READ_TOOL_NAMES.contains(&n),
                "tool {n} is in both sets"
            );
        }
        // Every name is non-empty (a registered tool must have a NAME).
        for n in ROCM_READ_TOOL_NAMES
            .iter()
            .chain(ROCM_MUTATING_TOOL_NAMES.iter())
        {
            assert!(!n.is_empty(), "empty tool name in canonical set");
        }
    }

    /// Offline (no network) coverage for `run_agent_request` itself — the
    /// helper `MockAgentClient`-backed tests above never reach, since
    /// `MockAgentClient::complete` is independent mock logic that bypasses it
    /// entirely. Drives it through rig-core's `test_utils::MockCompletionModel`
    /// instead of a real provider.
    ///
    /// Scripts more tool-call turns than [`MAX_TOOL_TURNS`] so the agent loop
    /// can never produce a final answer and must exhaust the turn limit —
    /// that's what forces every registered tool set through `run_agent_request`
    /// onto the wire (the model only ever sees tool definitions on the request
    /// it's given) and pins `run_agent_request`'s `.max_turns(MAX_TOOL_TURNS)`
    /// call, both of which no ignored round-trip test and no `MockAgentClient`
    /// test exercises.
    #[tokio::test]
    async fn run_agent_request_registers_tool_sets_and_enforces_turn_limit() {
        use rig::test_utils::{MockCompletionModel, MockTurn};

        let turns: Vec<MockTurn> = (0..MAX_TOOL_TURNS + 2)
            .map(|i| MockTurn::tool_call(format!("call_{i}"), "gpu_status", json!({})))
            .collect();
        let model = MockCompletionModel::new(turns);
        let recorded = model.clone();
        let agent = rig::agent::AgentBuilder::new(model).preamble(DEFAULT_PREAMBLE);

        let params = InferenceParams::default();
        let history = [ChatTurn::user("loop forever")];

        let err = run_agent_request(
            agent,
            &params,
            None,
            None,
            &history,
            fixture_snapshot(),
            "mock-test",
        )
        .await
        .expect_err("a tool-call-only script must exhaust the turn limit");
        let AgentError::Request(msg) = err else {
            panic!("expected AgentError::Request wrapping rig's MaxTurnsError, got {err:?}");
        };
        assert!(
            msg.contains(&format!("limit: {MAX_TOOL_TURNS}")),
            "error should name the configured turn limit, got: {msg}"
        );

        // The model actually received the registered tool definitions on the
        // wire — proves run_agent_request's three register_* calls ran, not
        // just that they type-check.
        let requests = recorded.requests();
        let sent_tool_names: std::collections::BTreeSet<&str> = requests
            .first()
            .expect("at least one request was sent before the turn limit was hit")
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        for expected in SKILL_NAMES
            .iter()
            .chain(ROCM_READ_TOOL_NAMES.iter())
            .chain(ROCM_MUTATING_TOOL_NAMES.iter())
        {
            assert!(
                sent_tool_names.contains(expected),
                "registered tool {expected} missing from the wire request"
            );
        }

        // Sanity bound on the request count: enough to have tried every
        // scripted turn up to the limit, not an early bail-out or a runaway
        // loop. Not pinned to rig-core's exact internal +1/-1 boundary, which
        // this test does not mean to assert.
        assert!(
            (MAX_TOOL_TURNS..=MAX_TOOL_TURNS + 2).contains(&recorded.request_count()),
            "expected roughly MAX_TOOL_TURNS requests, got {}",
            recorded.request_count()
        );
    }

    /// Live round-trip against Anthropic's Claude API. NOT run in CI (network +
    /// a real key). Run with:
    /// `ANTHROPIC_API_KEY=… cargo test -p rocm-dash-tui --lib anthropic_round_trip -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY environment variable + network"]
    async fn anthropic_round_trip() {
        let key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");
        let client = AnthropicAgentClient::new(
            LlmConfig {
                base_url: String::new(),
                model: String::new(),
                api_key: Some(key),
                auth_header: None,
            },
            InferenceParams::default(),
            None,
            None,
        )
        .expect("build anthropic client");
        let history = vec![ChatTurn::user("Reply with exactly: anthropic ok")];
        let reply = client
            .complete(&history, fixture_snapshot())
            .await
            .expect("anthropic reply");
        assert!(!reply.is_empty());
    }

    /// Live no-key device-code round-trip against ChatGPT. NOT run in CI
    /// (interactive OAuth + network). Run with:
    /// `cargo test -p rocm-dash-tui --lib chatgpt_oauth_round_trip -- --ignored --nocapture`
    /// then complete the device login in a browser.
    #[tokio::test]
    #[ignore = "interactive ChatGPT OAuth device-code login + network"]
    async fn chatgpt_oauth_round_trip() {
        let client = ChatGptAgentClient::new(
            None,
            InferenceParams::default(),
            |url, code| {
                eprintln!("Sign in: open {url} and enter code {code}");
            },
            None,
            None,
        )
        .expect("build chatgpt oauth client");
        let history = vec![ChatTurn::user("Reply with exactly: oauth ok")];
        let reply = client
            .complete(&history, fixture_snapshot())
            .await
            .expect("oauth reply");
        eprintln!("ChatGPT OAuth reply: {reply}");
        assert!(!reply.is_empty());
    }
}
