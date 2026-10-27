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

/// The exact text of the warning `ChatGptAgentClient::complete` logs when
/// `temperature`/`top_p`/`max_tokens` is set. Shared with the test below so
/// the regression test asserts the SAME message the operator actually sees,
/// rather than a copy that can silently drift from it.
const CHATGPT_DROPPED_PARAMS_WARNING: &str =
    "temperature/top_p/max_tokens are ignored by the ChatGPT backend";

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
// All three clients carry an `H` (HTTP client) type parameter, defaulted to
// the live provider's `reqwest::Client` for production use and swapped for
// `rig::test_utils::RecordingHttpClient` in tests (see
// `rig_agent_client_complete_forwards_params_to_the_wire` /
// `anthropic_agent_client_complete_forwards_params_to_the_wire` /
// `chatgpt_agent_client_complete_forwards_model_and_instructions_to_the_wire`).
// Neither method below touches `self.client`, so no bound on `H` is needed;
// the optional `<$h>` just reproduces whichever of the two forms `$ty` takes.
macro_rules! impl_with_preamble {
    ($ty:ident $(<$h:ident>)?) => {
        impl $(<$h>)? $ty $(<$h>)? {
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

impl_with_preamble!(RigAgentClient<H>);
impl_with_preamble!(ChatGptAgentClient<H>);
impl_with_preamble!(AnthropicAgentClient<H>);

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
///
/// Generic over the HTTP client (`H`, defaulted to the real `reqwest::Client`)
/// so a test can swap in `rig::test_utils::RecordingHttpClient` and drive the
/// real `complete()` method over a scripted wire response instead of a live
/// endpoint — see `rig_agent_client_complete_forwards_params_to_the_wire`.
pub struct RigAgentClient<H = reqwest::Client> {
    client: rig::providers::openai::CompletionsClient<H>,
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

impl RigAgentClient<reqwest::Client> {
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
impl<H> AgentClient for RigAgentClient<H>
where
    H: rig::http_client::HttpClientExt + Clone + Default + std::fmt::Debug + Send + Sync + 'static,
{
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

/// Derives [`ChatGptAgentClient::new`]'s OAuth token-cache directory from
/// `$HOME` (falling back to `$USERPROFILE` on Windows), without reading the
/// process environment itself — a pure function over the two candidate
/// values, so the derivation (precedence, empty-value handling, and the
/// "home directory not found" error) is unit-testable without mutating or
/// depending on the real environment.
///
/// This precedence (prefer `$HOME`) differs from `rocm-core`'s
/// `runtime_home_dir`, which prefers `$USERPROFILE` on Windows — under Git
/// Bash/MSYS the ChatGPT token cache can therefore land under a different
/// home than the rest of `~/.rocm`. Pre-existing, and left as-is: fixing it
/// is a behavior change outside this function's own scope.
fn token_dir_from_home(
    home: Option<std::ffi::OsString>,
    home_fallback: Option<std::ffi::OsString>,
) -> Result<std::path::PathBuf, AgentError> {
    home.filter(|v| !v.is_empty())
        .or_else(|| home_fallback.filter(|v| !v.is_empty()))
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
        })
}

/// No-key ChatGPT backend over Rig's native `chatgpt` OAuth provider.
///
/// This is the no-key default that restores the ChatGPT device-login the vendored Codex
/// path provided. It takes NO api_key (the env-only key invariant is untouched:
/// this path authenticates with an OAuth device-code flow, not a key). The
/// `on_device_code` callback surfaces the verification URL + user code so the
/// chat tab can show the operator how to sign in; the resulting token is
/// persisted by the provider so re-launches don't re-prompt.
///
/// Generic over the HTTP client (`H`, defaulted to the real `reqwest::Client`)
/// so a test can swap in `rig::test_utils::RecordingHttpClient` and drive the
/// real `complete()` method over a scripted wire response instead of a live
/// endpoint — see `chatgpt_agent_client_complete_forwards_model_and_instructions_to_the_wire`.
/// `new`/`new_with_env`/`new_in` stay on the concrete `reqwest::Client` impl,
/// since they always build the live OAuth device-code client; only a test
/// constructs the struct literal directly with another `H`.
pub struct ChatGptAgentClient<H = reqwest::Client> {
    client: rig::providers::chatgpt::Client<H>,
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

impl ChatGptAgentClient<reqwest::Client> {
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
        Self::new_with_env(
            |k| std::env::var_os(k),
            model,
            params,
            on_device_code,
            executor,
            approval_tx,
        )
    }

    /// As [`Self::new`], but takes the `$HOME`/`$USERPROFILE` lookup as a
    /// function instead of reading the process environment directly.
    ///
    /// This is the seam a test needs to pin `new`'s own wiring into
    /// [`token_dir_from_home`]: `new_with_env`'s only non-test caller,
    /// `Self::new`, is one line, so without it a test can only exercise the
    /// real environment, which can't tell a correct call apart from a
    /// `HOME`/`USERPROFILE` swap on a host where only one of the two is ever
    /// set (every Linux CI runner). See
    /// `chatgpt_new_with_env_derives_token_dir_from_distinct_home_and_userprofile`.
    pub(crate) fn new_with_env<F, L>(
        lookup: L,
        model: Option<String>,
        params: InferenceParams,
        on_device_code: F,
        executor: Option<SharedRocmToolExecutor>,
        approval_tx: Option<UnboundedSender<ClientMsg>>,
    ) -> Result<Self, AgentError>
    where
        F: Fn(String, String) + Send + Sync + 'static,
        L: Fn(&str) -> Option<std::ffi::OsString>,
    {
        // Store OAuth tokens in a user-owned directory so the plaintext
        // access/refresh tokens are not readable by other local users.
        // Refuse to fall back to a predictable temp path: a pre-existing
        // directory there could be owned by another user.
        let token_dir = token_dir_from_home(
            lookup("HOME"),
            lookup("USERPROFILE"), // Windows
        )?;
        Self::new_in(
            token_dir,
            model,
            params,
            on_device_code,
            executor,
            approval_tx,
        )
    }

    /// As [`Self::new`], but takes the token-cache directory directly instead
    /// of deriving it from `$HOME`/`$USERPROFILE`.
    ///
    /// This is the test seam `xtask/src/env_mutation_contract.rs` recommends
    /// over mutating the process environment: a test can point the OAuth
    /// token cache at a throwaway directory without touching `$HOME` at all,
    /// so it never races the other tests that read it via [`Self::new`].
    pub(crate) fn new_in<F>(
        token_dir: std::path::PathBuf,
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
impl<H> AgentClient for ChatGptAgentClient<H>
where
    H: rig::http_client::HttpClientExt + Clone + Default + std::fmt::Debug + Send + Sync + 'static,
{
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

        // rig-core's ChatGPT provider unconditionally zeroes temperature,
        // max_output_tokens, and additional_parameters.top_p in its own
        // `create_request` before the request reaches the wire, regardless of
        // what `self.params` holds (tracked upstream in rig-core, not fixable
        // here — see https://github.com/ROCm/rocm-cli/issues/664). Warn so the
        // operator knows these flags are silently ignored for this backend.
        if self.params.temperature.is_some()
            || self.params.top_p.is_some()
            || self.params.max_tokens.is_some()
        {
            warn!(
                backend = "chatgpt-oauth",
                temperature = ?self.params.temperature,
                top_p = ?self.params.top_p,
                max_tokens = ?self.params.max_tokens,
                "{CHATGPT_DROPPED_PARAMS_WARNING}"
            );
        }

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
///
/// Generic over the HTTP client (`H`, defaulted to the real `reqwest::Client`)
/// so a test can swap in `rig::test_utils::RecordingHttpClient` and drive the
/// real `complete()` method over a scripted wire response instead of a live
/// endpoint — see `anthropic_agent_client_complete_forwards_params_to_the_wire`.
pub struct AnthropicAgentClient<H = reqwest::Client> {
    client: rig::providers::anthropic::Client<H>,
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

impl AnthropicAgentClient<reqwest::Client> {
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
impl<H> AgentClient for AnthropicAgentClient<H>
where
    H: rig::http_client::HttpClientExt + Clone + Default + std::fmt::Debug + Send + Sync + 'static,
{
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
    use serde_json::{Value, json};

    use crate::agent::{
        ROCM_MUTATING_TOOL_NAMES, ROCM_READ_TOOL_NAMES, SKILL_NAMES, fixture_snapshot,
    };

    /// A throwaway, never-pre-existing directory for
    /// `ChatGptAgentClient::new_in`'s OAuth token cache, so a test builds a
    /// ChatGPT client without ever touching the developer's real `$HOME` (and
    /// without the "Permission denied" a read-only `$HOME` would otherwise
    /// cause). `tag` only needs to be unique among the directories a single
    /// test creates. `new_in` creates the directory up front, and the client
    /// keeps reading and writing `auth.json` under it for its whole lifetime
    /// (inside `authorize()`) — callers must keep the guard alive for as long
    /// as the client itself is in use, not just until construction returns.
    ///
    /// "Never-pre-existing" is enforced, not assumed: a process-wide counter
    /// is folded into the name (pids are reused across processes and
    /// `ThreadId` numbering restarts in every process, so pid+tid alone can
    /// collide), and the call panics if the resulting path already exists —
    /// because `new_in`'s directory creation is `recursive(true)` precisely
    /// so it can reuse an already-there `.rocm/data` on a real `$HOME`, so it
    /// would silently reuse a leftover test directory too, which could let a
    /// stale `auth.json` slip a test onto the cached-token fast path it means
    /// to avoid.
    fn fresh_token_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rocm-dash-chatgpt-{tag}-{}-{:?}-{n}",
            std::process::id(),
            std::thread::current().id()
        ));
        assert!(
            !dir.exists(),
            "fresh_token_dir must be never-pre-existing, but {} already exists \
             (likely leaked by a crashed prior run); remove it before re-running \
             the suite",
            dir.display()
        );
        dir
    }

    /// Removes a [`fresh_token_dir`] directory when dropped — including on an
    /// early return or a test panic — so a failing `ChatGptAgentClient::new_in`
    /// call can't leak the directory into the OS temp dir.
    struct TokenDirGuard(std::path::PathBuf);

    impl Drop for TokenDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A [`crate::tool_exec::RocmToolExecutor`] double that records every
    /// non-`watcher_disable` call name it receives (standing in for a real
    /// read-tool invocation) and surfaces `watcher_disable` as
    /// `ApprovalRequired` (standing in for a real mutating-tool invocation,
    /// whose `call()` forwards that outcome over `approval_tx` — see
    /// `rocm_mutating_tool!` in `agent::tools`). Shared by
    /// `run_agent_request_forwards_executor_and_approval_channel` and the
    /// per-backend `*_complete_forwards_executor_and_approval_channel` wire
    /// tests below, which all need the same executor/approval-channel
    /// forwarding proof, just reached through a different call site.
    #[derive(Debug)]
    struct RecordingExec {
        read_calls: Arc<Mutex<Vec<String>>>,
    }
    impl crate::tool_exec::RocmToolExecutor for RecordingExec {
        fn execute(&self, name: &str, args: &Value) -> crate::tool_exec::RocmToolOutcome {
            if name == "watcher_disable" {
                crate::tool_exec::RocmToolOutcome::ApprovalRequired(
                    crate::tool_exec::ApprovalIntent {
                        title: "Disable watcher".to_string(),
                        body: vec!["watcher_disable".to_string()],
                        name: name.to_string(),
                        arguments: args.clone(),
                    },
                )
            } else {
                self.read_calls.lock().unwrap().push(name.to_string());
                crate::tool_exec::RocmToolOutcome::Result(json!({ "doctor": "ok" }))
            }
        }
        fn execute_approved(&self, name: &str, _args: &Value) -> crate::tool_exec::RocmToolOutcome {
            panic!("execute_approved must never run inside the rig tool loop (got {name})");
        }
    }

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
        let token_dir = fresh_token_dir("grounding");
        let _cleanup = TokenDirGuard(token_dir.clone());
        let chatgpt = ChatGptAgentClient::new_in(
            token_dir,
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
    fn token_dir_from_home_prefers_home_over_userprofile() {
        let dir = token_dir_from_home(Some("/home/alice".into()), Some("C:\\Users\\alice".into()))
            .expect("HOME is set");
        assert_eq!(
            dir,
            std::path::PathBuf::from("/home/alice")
                .join(".rocm")
                .join("data")
                .join("chatgpt-tokens")
        );
    }

    #[test]
    fn token_dir_from_home_falls_back_to_userprofile_when_home_unset() {
        let dir = token_dir_from_home(None, Some("C:\\Users\\alice".into()))
            .expect("USERPROFILE fallback is set");
        assert_eq!(
            dir,
            std::path::PathBuf::from("C:\\Users\\alice")
                .join(".rocm")
                .join("data")
                .join("chatgpt-tokens")
        );
    }

    #[test]
    fn token_dir_from_home_treats_empty_values_as_unset() {
        // An empty env var (e.g. `HOME=` in a stripped-down container) must
        // not be treated as a valid home directory.
        let err = token_dir_from_home(Some("".into()), Some("".into()))
            .expect_err("empty HOME and USERPROFILE must not count as set");
        assert!(matches!(err, AgentError::Build(_)));
        assert!(err.to_string().contains("home directory not found"));
    }

    #[test]
    fn token_dir_from_home_falls_back_to_userprofile_when_home_is_empty() {
        // `token_dir_from_home_treats_empty_values_as_unset` above only tries
        // an empty HOME alongside an empty USERPROFILE; a mutation that short-
        // circuits to the "unset" error as soon as `home` is empty — without
        // trying the fallback at all — would stay green there. This pins the
        // fallback firing when HOME is empty but USERPROFILE is actually set.
        let dir = token_dir_from_home(Some("".into()), Some("C:\\Users\\alice".into()))
            .expect("empty HOME must still fall back to a set USERPROFILE");
        assert_eq!(
            dir,
            std::path::PathBuf::from("C:\\Users\\alice")
                .join(".rocm")
                .join("data")
                .join("chatgpt-tokens")
        );
    }

    #[test]
    fn token_dir_from_home_errors_when_neither_is_set() {
        let err =
            token_dir_from_home(None, None).expect_err("must fail without $HOME or $USERPROFILE");
        assert!(matches!(err, AgentError::Build(_)));
        assert!(err.to_string().contains("$HOME/$USERPROFILE unset"));
    }

    /// `token_dir_from_home` itself is fully covered above, but every test
    /// that used to exercise [`ChatGptAgentClient::new`]'s own derivation —
    /// `std::env::var_os("HOME")`/`"USERPROFILE"` into that function — moved
    /// to [`ChatGptAgentClient::new_in`], which takes the directory directly
    /// and so proves nothing about `new`'s wiring to it. Production calls
    /// `new(...).ok()`, so a broken wiring would silently leave chat without
    /// an agent with nothing to catch it.
    ///
    /// Reads the real environment opportunistically (whatever `$HOME`/
    /// `$USERPROFILE` happen to be on the test runner) instead of asserting
    /// a specific directory, and checks `new`'s own outcome agrees with
    /// `token_dir_from_home` fed the same two values — so this needs no env
    /// mutation and cannot race the other tests that read `$HOME` via `new`.
    ///
    /// This *does* write under the real `$HOME`/`$USERPROFILE`, the same as
    /// any other call to `new`: `new_in` creates the token-cache directory up
    /// front (`new`'s own doc comment's "No network I/O happens here" is
    /// about the OAuth login, not the filesystem). On success this also
    /// checks that the directory `token_dir_from_home` derives is the one
    /// `new` actually created, which a bare `is_ok()`/`is_err()` comparison
    /// would miss. (A precedence bug inside `token_dir_from_home` itself is
    /// already covered by the dedicated `token_dir_from_home_*` tests above,
    /// since both `expected` and `new` feed it the same two values in the
    /// same order.) Cleanup removes every path segment this call created —
    /// not just the leaf — stopping at the first ancestor that already
    /// existed, so a `$HOME` with no prior `.rocm` directory at all doesn't
    /// leak one. Each created segment is removed individually with
    /// `remove_dir` (not `remove_dir_all`), deepest first: a plain
    /// `remove_dir_all` on the shallowest missing ancestor — which on a
    /// fresh runner can be `~/.rocm` itself — would delete anything another
    /// process wrote under it in this window (a parallel test binary, or a
    /// real `rocm` run on a developer machine); `remove_dir` only removes a
    /// directory this call made and that is still empty, so an unexpected
    /// concurrent write aborts the cleanup at that level instead of being
    /// silently deleted. Where `$HOME` exists but isn't writable, `new` is
    /// expected to fail with `Build("cannot create token cache dir…")`; that
    /// still counts as agreement; a failure for any other reason does not.
    ///
    /// This does NOT catch a `HOME`/`USERPROFILE` swap at `new`'s call site:
    /// on every host where only one of the two is ever set (every Linux CI
    /// runner, since `USERPROFILE` is Windows-only), `token_dir_from_home`'s
    /// own fallback makes the swapped call land on the same directory as the
    /// correct one. See
    /// `chatgpt_new_with_env_derives_token_dir_from_distinct_home_and_userprofile`
    /// below for that case, which sets both to distinct values deterministically.
    #[test]
    fn chatgpt_new_agrees_with_token_dir_from_home_on_the_real_environment() {
        let expected =
            token_dir_from_home(std::env::var_os("HOME"), std::env::var_os("USERPROFILE"));
        // Every ancestor of `expected` that doesn't exist yet, deepest
        // first — `new_in`'s `DirBuilder::recursive(true)` may create more
        // than just the leaf, and cleanup must remove exactly what it
        // created, one empty directory at a time (see doc comment above).
        let cleanup_dirs: Vec<std::path::PathBuf> = expected
            .as_ref()
            .ok()
            .map(|dir| {
                let mut created = Vec::new();
                let mut p: &std::path::Path = dir.as_path();
                while !p.exists() {
                    created.push(p.to_path_buf());
                    match p.parent() {
                        Some(parent) => p = parent,
                        None => break,
                    }
                }
                created
            })
            .unwrap_or_default();
        let result = ChatGptAgentClient::new(
            None,
            InferenceParams::default(),
            |_, _| panic!("device-code callback must not fire during construction"),
            None,
            None,
        );
        match &expected {
            Ok(dir) => match result {
                Ok(_) => {
                    assert!(
                        dir.is_dir(),
                        "new() succeeded but did not create the directory \
                         token_dir_from_home derived: {}",
                        dir.display()
                    );
                    for d in &cleanup_dirs {
                        // Ignore errors: a non-empty directory (something
                        // else wrote into it concurrently) just stops the
                        // cleanup at that level rather than deleting it.
                        let _ = std::fs::remove_dir(d);
                    }
                }
                Err(AgentError::Build(msg)) if msg.contains("cannot create token cache dir") => {
                    // `$HOME` exists but isn't writable on this runner — not
                    // a wiring bug, so this still counts as agreement.
                }
                Err(e) => {
                    panic!("token_dir_from_home succeeded but new() failed unexpectedly: {e}")
                }
            },
            Err(_) => assert!(
                matches!(result, Err(AgentError::Build(_))),
                "token_dir_from_home failed on this environment but new() did not"
            ),
        }
    }

    /// Pins `new`'s wiring into `token_dir_from_home` deterministically: two
    /// distinct, freshly-generated temp directories stand in for `$HOME` and
    /// `$USERPROFILE`, so a swap at `new`'s call site is caught on every
    /// host, unlike the real-environment test above, which can't tell a swap
    /// apart from a correct call wherever only one of the two is ever set.
    ///
    /// This always supplies `HOME`, so it cannot catch a typo'd
    /// `"USERPROFILE"` lookup key: `token_dir_from_home`'s own precedence
    /// means the fallback value is never read when `HOME` is set, so the
    /// wrong key and the right key produce the same directory here. See
    /// `chatgpt_new_with_env_falls_back_to_userprofile_when_home_is_unset`
    /// below for that case.
    #[test]
    fn chatgpt_new_with_env_derives_token_dir_from_distinct_home_and_userprofile() {
        let home_root = fresh_token_dir("new-with-env-home");
        let profile_root = fresh_token_dir("new-with-env-profile");
        let _cleanup_home = TokenDirGuard(home_root.clone());
        let _cleanup_profile = TokenDirGuard(profile_root.clone());

        let home_os = home_root.clone().into_os_string();
        let profile_os = profile_root.clone().into_os_string();
        let result = ChatGptAgentClient::new_with_env(
            move |k| match k {
                "HOME" => Some(home_os.clone()),
                "USERPROFILE" => Some(profile_os.clone()),
                _ => None,
            },
            None,
            InferenceParams::default(),
            |_, _| panic!("device-code callback must not fire during construction"),
            None,
            None,
        );
        if let Err(e) = &result {
            panic!("expected new_with_env to succeed: {e}");
        }

        let expected_dir = home_root.join(".rocm").join("data").join("chatgpt-tokens");
        assert!(
            expected_dir.is_dir(),
            "new_with_env must create the token dir under HOME, not USERPROFILE: {}",
            expected_dir.display()
        );
        assert!(
            !profile_root.join(".rocm").exists(),
            "new_with_env must not have written anything under USERPROFILE when HOME is set"
        );
    }

    /// Companion to the test above: that one always supplies `HOME`, so the
    /// `USERPROFILE` fallback through `new_with_env` is never taken, and a
    /// typo'd `"USERPROFILE"` lookup key at the call site goes undetected
    /// (the lookup falls through to `None`, same as the untouched key would
    /// when `HOME` already satisfies `token_dir_from_home`). This sets
    /// `HOME` to `None` and `USERPROFILE` to a distinct, freshly-generated
    /// temp directory, so the fallback path itself — and a typo in the
    /// `USERPROFILE` lookup key — are both exercised and caught: a typo
    /// makes the lookup return `None` for both variables, which
    /// `token_dir_from_home` rejects.
    #[test]
    fn chatgpt_new_with_env_falls_back_to_userprofile_when_home_is_unset() {
        let profile_root = fresh_token_dir("new-with-env-profile-only");
        let _cleanup_profile = TokenDirGuard(profile_root.clone());

        let profile_os = profile_root.clone().into_os_string();
        let result = ChatGptAgentClient::new_with_env(
            move |k| match k {
                "USERPROFILE" => Some(profile_os.clone()),
                _ => None,
            },
            None,
            InferenceParams::default(),
            |_, _| panic!("device-code callback must not fire during construction"),
            None,
            None,
        );
        if let Err(e) = &result {
            panic!("expected new_with_env to fall back to USERPROFILE when HOME is unset: {e}");
        }

        let expected_dir = profile_root
            .join(".rocm")
            .join("data")
            .join("chatgpt-tokens");
        assert!(
            expected_dir.is_dir(),
            "new_with_env must create the token dir under USERPROFILE when HOME is unset: {}",
            expected_dir.display()
        );
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
        let token_dir = fresh_token_dir("offline-no-key");
        let _cleanup = TokenDirGuard(token_dir.clone());
        let client = ChatGptAgentClient::new_in(
            token_dir,
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
        let token_dir = fresh_token_dir("default-model");
        let _cleanup = TokenDirGuard(token_dir.clone());
        let client = ChatGptAgentClient::new_in(
            token_dir,
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

    /// The token cache holds plaintext OAuth access/refresh tokens, so
    /// `new_in` creates its directory at mode `0o700` (owner-only) rather
    /// than leaving it world- or group-accessible. That mode-setting code
    /// pre-dates this PR and moved into `new_in` unchanged, but nothing
    /// asserted it — `new_in` now makes it a one-line check.
    ///
    /// The directory is pre-created at `0o755` before `new_in` runs, so this
    /// pins the `set_permissions` tightening branch specifically: against a
    /// brand-new (never-pre-existing) directory, the initial
    /// `DirBuilder::mode(0o700)` call alone would already leave the dir at
    /// `0o700`, and this test couldn't tell the tightening branch apart from
    /// removing it outright.
    #[cfg(unix)]
    #[test]
    fn chatgpt_new_in_creates_token_dir_with_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let token_dir = fresh_token_dir("owner-only-mode");
        let _cleanup = TokenDirGuard(token_dir.clone());
        std::fs::create_dir_all(&token_dir).expect("pre-create token dir");
        std::fs::set_permissions(&token_dir, std::fs::Permissions::from_mode(0o755))
            .expect("relax token dir to 0o755 before new_in tightens it");

        let _client = ChatGptAgentClient::new_in(
            token_dir.clone(),
            None,
            InferenceParams::default(),
            |_url, _code| {},
            None,
            None,
        )
        .expect("build chatgpt oauth client");

        let mode = std::fs::metadata(&token_dir)
            .expect("token dir should exist after new_in")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            mode, 0o700,
            "token cache dir must be owner-only (0o700), got {mode:o}"
        );
    }

    /// `ChatGptAgentClient::complete`'s empty-history guard (clients.rs, ahead
    /// of its param validation and device-code `authorize()`) is otherwise
    /// protected only by a comment. Pin it: an empty history must return
    /// `AgentError::Empty` without the device-code callback ever firing.
    ///
    /// Builds via [`ChatGptAgentClient::new_in`] with a fresh, token-free
    /// directory instead of `ChatGptAgentClient::new`'s real `$HOME`: a
    /// developer machine already signed in to ChatGPT has a cached token
    /// there, and against that real cache, deleting the guard entirely would
    /// still return `Empty` for the wrong reason — `authorize()` runs first
    /// regardless, returns early via the cached token without firing the
    /// device-code callback, and only then does `run_agent_request`'s own
    /// `history.split_last()` check reject the empty history. This test
    /// would pass without ever exercising the dedicated guard it exists to
    /// catch. `new_in` takes the directory directly, so no process-wide
    /// `$HOME` mutation (and no lock against the other tests that read it
    /// via `new`) is needed at all.
    ///
    /// This closes that gap but not a second one: rig-core's device-code
    /// endpoints (`auth.openai.com`) are compile-time constants with no
    /// override hook, so on a host with real internet access and the guard
    /// removed, this test would reach OpenAI's real auth service and poll for
    /// up to 15 minutes before failing. The `complete()` call below is
    /// wrapped in a 5s timeout so that regression still fails fast here.
    #[tokio::test]
    async fn chatgpt_empty_history_is_empty_error_without_auth() {
        let fired = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = fired.clone();

        let token_dir = fresh_token_dir("empty-history");
        let _cleanup = TokenDirGuard(token_dir.clone());
        let client = ChatGptAgentClient::new_in(
            token_dir,
            None,
            InferenceParams::default(),
            move |url, code| {
                sink.lock().unwrap().push(format!("{url}|{code}"));
            },
            None,
            None,
        )
        .expect("build chatgpt oauth client");

        let err = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.complete(&[], fixture_snapshot()),
        )
        .await
        .expect("complete() must not hang past 5s even if its empty-history guard regresses")
        .expect_err("empty history must be rejected");
        assert!(matches!(err, AgentError::Empty));
        assert!(
            fired.lock().unwrap().is_empty(),
            "device-code callback must not fire on the empty-history short-circuit"
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

        let turns: Vec<MockTurn> = (0..MAX_TOOL_TURNS + 5)
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
        // loop. The script above is deliberately longer than this bound
        // (MAX_TOOL_TURNS + 5 turns scripted, vs. MAX_TOOL_TURNS + 2 expected
        // requests) so the turn limit — not running out of scripted turns —
        // is what ends the loop; rig-core 0.38.1's own +1/-1 internal
        // bookkeeping is what fixes the request count at MAX_TOOL_TURNS + 2.
        assert_eq!(
            recorded.request_count(),
            MAX_TOOL_TURNS + 2,
            "rig-core 0.38.1 fixes the request count at MAX_TOOL_TURNS + 2"
        );
    }

    /// Offline coverage for the parts of `run_agent_request`'s wiring the test
    /// above does not reach: it drives a default-params, single-turn history
    /// through the turn-limit error path, so it would still pass with
    /// `apply_inference_params` dropped, the wrong end of `history` split off,
    /// a swapped prompt source, or the wrong `fired` log handed to
    /// `finish_agent_request` — none of those change a default `InferenceParams`
    /// or an empty `prior` slice.
    ///
    /// This test instead scripts a successful run: a history with two prior
    /// turns plus the live prompt, and non-default temperature/top_p/max_tokens.
    /// It asserts directly on the `CompletionRequest` `MockCompletionModel`
    /// recorded — chat history (order and content), the prompt, and every
    /// sampling field `apply_inference_params` sets — plus that a successful
    /// reply carries the `fired`-log annotation for the tool the script calls.
    #[tokio::test]
    async fn run_agent_request_forwards_history_and_params_and_annotates_reply() {
        use rig::completion::Message;
        use rig::completion::message::UserContent;
        use rig::test_utils::{MockCompletionModel, MockTurn};

        let model = MockCompletionModel::new([
            MockTurn::tool_call("call_0", "gpu_status", json!({})),
            MockTurn::text("GPU 0 is healthy."),
        ]);
        let recorded = model.clone();
        let agent = rig::agent::AgentBuilder::new(model).preamble(DEFAULT_PREAMBLE);

        let params = InferenceParams {
            temperature: Some(0.4),
            top_p: Some(0.2),
            max_tokens: Some(256),
        };
        let history = [
            ChatTurn::user("first, check gpu 0"),
            ChatTurn::agent("checked once already"),
            ChatTurn::user("check gpu 0 again"),
        ];

        let reply = run_agent_request(
            agent,
            &params,
            None,
            None,
            &history,
            fixture_snapshot(),
            "mock-test",
        )
        .await
        .expect("scripted tool call then text turn should succeed");

        assert!(
            reply.contains("⚙ via: gpu_status"),
            "reply should carry the fired-skill annotation: {reply}"
        );

        let requests = recorded.requests();
        let first = requests.first().expect("at least one request was sent");

        // Sampling knobs: all three of apply_inference_params's branches ran.
        assert_eq!(first.temperature, Some(f64::from(0.4_f32)));
        assert_eq!(first.max_tokens, Some(256));
        let top_p = first
            .additional_params
            .as_ref()
            .and_then(|v| v.get("top_p"))
            .and_then(serde_json::Value::as_f64);
        assert_eq!(top_p, Some(f64::from(0.2_f32)));

        // History: the preamble (rig's own leading system message), then
        // build_messages(prior) (the first two turns), then the live prompt
        // (the third turn, `last.content`) appended by rig itself — proves
        // `history.split_last()` split the right way and `last.content` (not
        // `prior`) became the prompt.
        let chat_history: Vec<Message> = first.chat_history.iter().cloned().collect();
        assert_eq!(chat_history.len(), 4, "got: {chat_history:?}");
        assert_eq!(chat_history[1], Message::user("first, check gpu 0"));
        assert_eq!(chat_history[2], Message::assistant("checked once already"));
        let Message::User { content } = &chat_history[3] else {
            panic!(
                "expected the live prompt as a trailing user message, got {:?}",
                chat_history[3]
            );
        };
        assert!(matches!(
            content.first(),
            UserContent::Text(t) if t.text == "check gpu 0 again"
        ));

        // The snapshot `run_agent_request` hands to `register_telemetry_tools`
        // reaches `gpu_status`'s own output, not just the fired-skill log: the
        // second request (after the tool call) must carry the tool result
        // derived from `fixture_snapshot()`'s GPUs, not an empty/default one.
        let second = requests
            .get(1)
            .expect("a follow-up request should carry the tool result");
        let second_history: Vec<Message> = second.chat_history.iter().cloned().collect();
        let tool_result_text = second_history
            .iter()
            .find_map(|m| match m {
                Message::User { content } => content.iter().find_map(|c| match c {
                    UserContent::ToolResult(r) => r.content.iter().find_map(|c| match c {
                        rig::completion::message::ToolResultContent::Text(t) => {
                            Some(t.text.clone())
                        }
                        rig::completion::message::ToolResultContent::Image(_) => None,
                    }),
                    _ => None,
                }),
                _ => None,
            })
            .expect("follow-up request should carry a tool-result message");
        assert!(
            tool_result_text.contains("gpu-1"),
            "tool result should carry fixture_snapshot()'s GPU data, got: {tool_result_text}"
        );
    }

    /// Pins the one remaining branch inside `run_agent_request` the two tests
    /// above never take: an empty history. Both Rig and Anthropic now reach
    /// `AgentError::Empty` only through `history.split_last()` here (ChatGPT
    /// still has its own earlier guard) — nothing calls `run_agent_request`
    /// with `&[]`, so a swap to a different error, or a change that lets an
    /// empty history reach the model, would stay green.
    #[tokio::test]
    async fn run_agent_request_returns_empty_for_an_empty_history() {
        use rig::test_utils::MockCompletionModel;

        let model = MockCompletionModel::new([]);
        let recorded = model.clone();
        let agent = rig::agent::AgentBuilder::new(model).preamble(DEFAULT_PREAMBLE);

        let err = run_agent_request(
            agent,
            &InferenceParams::default(),
            None,
            None,
            &[],
            fixture_snapshot(),
            "mock-test",
        )
        .await
        .expect_err("an empty history must be rejected before any request is built");

        assert!(matches!(err, AgentError::Empty), "got: {err:?}");
        assert_eq!(
            recorded.request_count(),
            0,
            "no request should reach the model for an empty history"
        );
    }

    /// Pins the other wiring `run_agent_request` moved to its one shared
    /// site: the optional ROCm tool executor and the mutating-tool approval
    /// channel. Both tests above pass `None` for each, so a dropped forward —
    /// e.g. `None` replacing `executor`/`approval_tx` at the call into
    /// `register_rocm_read_tools`/`register_rocm_mutating_tools` — would stay
    /// green. Scripts one read-tool call and one mutating-tool call through a
    /// real executor + channel and asserts both reached their registered
    /// tool: the executor saw the read call, `approval_tx` received the
    /// mutating call's approval intent, and the reply names both tools.
    #[tokio::test]
    async fn run_agent_request_forwards_executor_and_approval_channel() {
        use rig::test_utils::{MockCompletionModel, MockTurn};

        let read_calls = Arc::new(Mutex::new(Vec::new()));
        let exec: SharedRocmToolExecutor = Arc::new(RecordingExec {
            read_calls: read_calls.clone(),
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ClientMsg>();

        let model = MockCompletionModel::new([
            MockTurn::tool_call("call_0", "doctor", json!({})),
            MockTurn::tool_call("call_1", "watcher_disable", json!({ "watcher": "w1" })),
            MockTurn::text("done"),
        ]);
        let agent = rig::agent::AgentBuilder::new(model).preamble(DEFAULT_PREAMBLE);
        let history = [ChatTurn::user("do stuff")];

        let reply = run_agent_request(
            agent,
            &InferenceParams::default(),
            Some(&exec),
            Some(&tx),
            &history,
            fixture_snapshot(),
            "mock-test",
        )
        .await
        .expect("scripted read + mutating calls then a text turn should succeed");

        assert!(
            reply.contains("doctor") && reply.contains("watcher_disable"),
            "reply should annotate both tools that fired: {reply}"
        );
        assert_eq!(
            read_calls.lock().unwrap().as_slice(),
            ["doctor".to_string()],
            "the read tool must have reached the injected executor"
        );

        match rx
            .try_recv()
            .expect("approval intent must be posted for the mutating call")
        {
            ClientMsg::ChatApprovalRequired { intent } => {
                assert_eq!(intent.name, "watcher_disable");
                assert_eq!(intent.arguments["watcher"], "w1");
            }
            other => panic!("expected ChatApprovalRequired, got {other:?}"),
        }
    }

    /// Captures the `backend` field off every tracing event
    /// `finish_agent_request` emits (`info!(backend, "chat request
    /// start"/"chat request complete")`), for tests that need to assert
    /// which backend tag a `complete()` call site actually used. The tag
    /// never rides on the wire — `finish_agent_request`'s doc comment says
    /// it only "tags the lifecycle trace events" — so a captured HTTP
    /// request (`RecordingHttpClient`) can't see a swapped tag; this is the
    /// one seam that can.
    #[derive(Default)]
    struct CapturedBackendTags(Mutex<Vec<String>>);

    impl CapturedBackendTags {
        fn snapshot(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
    }

    impl tracing::Subscriber for CapturedBackendTags {
        fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            struct BackendFieldVisitor<'a>(&'a Mutex<Vec<String>>);
            impl tracing::field::Visit for BackendFieldVisitor<'_> {
                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    if field.name() == "backend" {
                        self.0.lock().unwrap().push(value.to_string());
                    }
                }
                fn record_debug(
                    &mut self,
                    _field: &tracing::field::Field,
                    _value: &dyn std::fmt::Debug,
                ) {
                }
            }
            event.record(&mut BackendFieldVisitor(&self.0));
        }

        fn enter(&self, _span: &tracing::span::Id) {}
        fn exit(&self, _span: &tracing::span::Id) {}
    }

    /// One tracing event's level, `backend` tag (if any), and message text —
    /// captured by [`CapturedWarnEvents`].
    struct CapturedEvent {
        level: tracing::Level,
        backend: Option<String>,
        message: Option<String>,
    }

    /// Unlike [`CapturedBackendTags`] (which records only the `backend`
    /// field's value and is blind to level/message), this captures each
    /// event's level and message text too, so a test can assert the SPECIFIC
    /// warn-level diagnostic fired — not merely that `complete()` produced
    /// one extra `chatgpt-oauth`-tagged event of any level and any text.
    /// Counting tags alone would stay green if `warn!` were silently changed
    /// to `info!`, or if the message text were altered or removed, since
    /// `CapturedBackendTags` ignores both.
    #[derive(Default)]
    struct CapturedWarnEvents(Mutex<Vec<CapturedEvent>>);

    impl CapturedWarnEvents {
        /// Count of events at exactly WARN level, tagged `chatgpt-oauth`,
        /// whose message is exactly [`CHATGPT_DROPPED_PARAMS_WARNING`].
        fn dropped_params_warning_count(&self) -> usize {
            self.0
                .lock()
                .unwrap()
                .iter()
                .filter(|e| {
                    e.level == tracing::Level::WARN
                        && e.backend.as_deref() == Some("chatgpt-oauth")
                        && e.message.as_deref() == Some(CHATGPT_DROPPED_PARAMS_WARNING)
                })
                .count()
        }
    }

    impl tracing::Subscriber for CapturedWarnEvents {
        fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            struct EventVisitor {
                backend: Option<String>,
                message: Option<String>,
            }
            impl tracing::field::Visit for EventVisitor {
                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    if field.name() == "backend" {
                        self.backend = Some(value.to_string());
                    }
                }
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    // tracing wraps a `Display` message in an internal type
                    // whose `Debug` impl forwards straight to `Display`, so
                    // `{value:?}` yields the exact message text here (this is
                    // also why `tracing-subscriber`'s own fmt layer shows
                    // plain, unquoted messages despite reading them through
                    // `record_debug`).
                    if field.name() == "message" {
                        self.message = Some(format!("{value:?}"));
                    }
                }
            }
            let mut visitor = EventVisitor {
                backend: None,
                message: None,
            };
            event.record(&mut visitor);
            self.0.lock().unwrap().push(CapturedEvent {
                level: *event.metadata().level(),
                backend: visitor.backend,
                message: visitor.message,
            });
        }

        fn enter(&self, _span: &tracing::span::Id) {}
        fn exit(&self, _span: &tracing::span::Id) {}
    }

    /// Ensures at least one always-interested [`tracing::Subscriber`] has
    /// been registered as the process-wide global default before any test
    /// installs its own thread-local one via [`tracing::subscriber::set_default`].
    ///
    /// Without this, the very first time one of `finish_agent_request`'s
    /// `info!(backend, ..)` callsites registers in the process, tracing-core
    /// caches that callsite's `Interest` by asking `dispatcher::get_default()`
    /// on whichever thread happens to trigger the registration — and with no
    /// global default and no thread-local override yet in scope on that
    /// thread, the answer is "never interested". A thread-local `set_default`
    /// elsewhere *does* rebuild interest for every already-registered
    /// callsite against every registered dispatcher, so this can self-heal —
    /// but only if nothing re-poisons the callsite afterward. Under threaded
    /// `cargo test`, a concurrent subscriber-less thread can still be the one
    /// that wins the race to trigger that first registration *after* the
    /// capturing test's own rebuild, caching "never interested" again with no
    /// further rebuild to follow. A permanent global default with
    /// `enabled() -> true` makes that condition — "no one is interested
    /// yet" — impossible from process start, regardless of which thread
    /// races which test. Its own capture buffer is never read; only the
    /// thread-local override installed below is.
    fn warm_tracing_interest_cache() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let _ = tracing::subscriber::set_global_default(CapturedBackendTags::default());
        });
    }

    /// Drives `client.complete(history, ..)` under a [`CapturedBackendTags`]
    /// subscriber and returns the reply plus every "backend" tag the
    /// lifecycle trace events recorded — the shared tail of
    /// `rig_agent_client_complete_forwards_params_to_the_wire`,
    /// `anthropic_agent_client_complete_forwards_params_to_the_wire`, and
    /// `chatgpt_agent_client_complete_forwards_model_and_instructions_to_the_wire`,
    /// which otherwise differ only in which concrete client type they drive.
    async fn complete_capturing_backend_tags(
        client: &impl AgentClient,
        history: &[ChatTurn],
    ) -> (String, Vec<String>) {
        warm_tracing_interest_cache();
        let backend_tags = Arc::new(CapturedBackendTags::default());
        let tracing_guard = tracing::subscriber::set_default(backend_tags.clone());
        let reply = client
            .complete(history, fixture_snapshot())
            .await
            .expect("scripted single-turn reply should succeed");
        drop(tracing_guard);
        (reply, backend_tags.snapshot())
    }

    /// A preamble distinct from [`DEFAULT_PREAMBLE`], so the
    /// forwards-params-to-the-wire tests below can tell "the client's own
    /// `preamble` field landed on the wire" apart from "the production
    /// default happened to land on the wire" — the two are indistinguishable
    /// if the test builds its client with `DEFAULT_PREAMBLE` itself.
    const TEST_PREAMBLE: &str = "test-only preamble, distinct from the production default";

    /// Asserts the `model`/`temperature`/`max_tokens` fields a captured wire
    /// request carries, shared by the two forwards-params-to-the-wire tests
    /// below. Each caller separately asserts where its provider places the
    /// preamble, since that location differs by wire format.
    fn assert_forwarded_params(
        body: &serde_json::Value,
        model: &str,
        temperature: f32,
        max_tokens: u64,
    ) {
        assert_eq!(
            body["model"].as_str(),
            Some(model),
            "complete() must forward &self.model, not a hardcoded one: {body}"
        );
        assert_eq!(
            body["temperature"].as_f64(),
            Some(f64::from(temperature)),
            "complete() must forward &self.params, not a default: {body}"
        );
        assert_eq!(body["max_tokens"].as_u64(), Some(max_tokens));
    }

    /// Pins `RigAgentClient::complete`'s own call site: that it forwards
    /// `&self.params` (not a default) and the `"rig-openai"` backend tag
    /// into `run_agent_request`. The test above drives `run_agent_request`
    /// directly with a `MockCompletionModel`, which proves the shared helper
    /// forwards correctly but says nothing about whether `complete()`
    /// actually passes it `&self.params` rather than, say,
    /// `&InferenceParams::default()`, or the right tag — a swap at that one
    /// call site stays green under every other test in this file; only a
    /// real `complete()` round-trip catches it.
    ///
    /// `RigAgentClient<H>`'s `H` (HTTP client) type parameter — defaulted to
    /// the live `reqwest::Client` for production — exists for exactly this:
    /// swap in `rig::test_utils::RecordingHttpClient` and drive the actual
    /// `complete()` method over a scripted response instead of a live
    /// endpoint.
    #[tokio::test]
    async fn rig_agent_client_complete_forwards_params_to_the_wire() {
        use rig::test_utils::RecordingHttpClient;

        let response_body = serde_json::to_vec(&json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "GPU 0 is healthy."},
                "logprobs": null,
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "total_tokens": 2},
        }))
        .expect("serialize stub OpenAI completion response");
        let http_client = RecordingHttpClient::new(response_body);
        let rig_client = rig::providers::openai::CompletionsClient::builder()
            .api_key("sk-test")
            .base_url("http://rocm-dash-tui-test.invalid")
            .http_client(http_client.clone())
            .build()
            .expect("build client over the recording HTTP double");

        let client = RigAgentClient {
            client: rig_client,
            model: "test-model".to_string(),
            preamble: TEST_PREAMBLE.to_string(),
            params: InferenceParams {
                temperature: Some(0.4),
                top_p: None,
                max_tokens: Some(256),
            },
            executor: None,
            approval_tx: None,
        };

        let history = [ChatTurn::user("check gpu 0")];
        let (reply, tags) = complete_capturing_backend_tags(&client, &history).await;
        assert_eq!(reply, "GPU 0 is healthy.");

        let requests = http_client.requests();
        assert_eq!(
            requests.len(),
            1,
            "exactly one request should have been sent"
        );
        let request = &requests[0];
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("request body is JSON");
        assert_forwarded_params(&body, "test-model", 0.4, 256);
        assert_eq!(
            body["messages"][0]["role"].as_str(),
            Some("system"),
            "preamble must land as the first message, as a system message: {body}"
        );
        assert!(
            body["messages"][0].to_string().contains(TEST_PREAMBLE),
            "complete() must forward &self.preamble, not a default: {body}"
        );

        assert!(
            !tags.is_empty() && tags.iter().all(|t| t == "rig-openai"),
            "complete() must tag its lifecycle trace events \"rig-openai\", got {tags:?}"
        );
    }

    /// As `rig_agent_client_complete_forwards_params_to_the_wire`, for
    /// `AnthropicAgentClient::complete`'s call site.
    #[tokio::test]
    async fn anthropic_agent_client_complete_forwards_params_to_the_wire() {
        use rig::test_utils::RecordingHttpClient;

        let response_body = serde_json::to_vec(&json!({
            "type": "message",
            "id": "msg-test",
            "model": "test-model",
            "role": "assistant",
            "content": [{"type": "text", "text": "GPU 0 is healthy."}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": null,
                "cache_creation_input_tokens": null,
            },
        }))
        .expect("serialize stub Anthropic completion response");
        let http_client = RecordingHttpClient::new(response_body);
        let anthropic_client = rig::providers::anthropic::Client::builder()
            .api_key("sk-ant-test")
            .http_client(http_client.clone())
            .build()
            .expect("build client over the recording HTTP double");

        let client = AnthropicAgentClient {
            client: anthropic_client,
            model: "test-model".to_string(),
            preamble: TEST_PREAMBLE.to_string(),
            params: InferenceParams {
                temperature: Some(0.4),
                top_p: None,
                max_tokens: Some(256),
            },
            executor: None,
            approval_tx: None,
        };

        let history = [ChatTurn::user("check gpu 0")];
        let (reply, tags) = complete_capturing_backend_tags(&client, &history).await;
        assert_eq!(reply, "GPU 0 is healthy.");

        let requests = http_client.requests();
        assert_eq!(
            requests.len(),
            1,
            "exactly one request should have been sent"
        );
        let request = &requests[0];
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("request body is JSON");
        assert_forwarded_params(&body, "test-model", 0.4, 256);
        assert!(
            body["system"].to_string().contains(TEST_PREAMBLE),
            "complete() must forward &self.preamble, not a default: {body}"
        );

        assert!(
            !tags.is_empty() && tags.iter().all(|t| t == "anthropic"),
            "complete() must tag its lifecycle trace events \"anthropic\", got {tags:?}"
        );
    }

    /// Pins `RigAgentClient::complete`'s forwarding of `self.executor` /
    /// `self.approval_tx` into `run_agent_request` — distinct from
    /// `run_agent_request_forwards_executor_and_approval_channel`, which
    /// proves the shared helper forwards correctly but drives it directly
    /// with a `MockCompletionModel`, so it says nothing about whether
    /// `complete()` actually passes its own `self.executor`/`self.approval_tx`
    /// into that call rather than, say, `None, None` — a swap at this call
    /// site stays green under every other test in this file, including the
    /// two `*_forwards_params_to_the_wire` tests above, which build their
    /// client with `executor: None, approval_tx: None`. Scripts one response
    /// with both a read-tool call (`doctor`) and a mutating-tool call
    /// (`watcher_disable`); the static `RecordingHttpClient` response repeats
    /// on every turn, so the agent loop keeps re-issuing both calls until it
    /// exhausts `MAX_TOOL_TURNS` and returns a turn-limit error — the test
    /// only needs each tool to have fired at least once before that happens.
    #[tokio::test]
    async fn rig_agent_client_complete_forwards_executor_and_approval_channel() {
        use rig::test_utils::RecordingHttpClient;

        let response_body = serde_json::to_vec(&json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_0",
                            "function": { "name": "doctor", "arguments": "{}" },
                        },
                        {
                            "type": "function",
                            "id": "call_1",
                            "function": {
                                "name": "watcher_disable",
                                "arguments": "{\"watcher\":\"w1\"}",
                            },
                        },
                    ],
                },
                "logprobs": null,
                "finish_reason": "tool_calls",
            }],
            "usage": { "prompt_tokens": 1, "total_tokens": 2 },
        }))
        .expect("serialize stub OpenAI tool-call response");
        let http_client = RecordingHttpClient::new(response_body);
        let rig_client = rig::providers::openai::CompletionsClient::builder()
            .api_key("sk-test")
            .base_url("http://rocm-dash-tui-test.invalid")
            .http_client(http_client.clone())
            .build()
            .expect("build client over the recording HTTP double");

        let read_calls = Arc::new(Mutex::new(Vec::new()));
        let exec: SharedRocmToolExecutor = Arc::new(RecordingExec {
            read_calls: read_calls.clone(),
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ClientMsg>();

        let client = RigAgentClient {
            client: rig_client,
            model: "test-model".to_string(),
            preamble: TEST_PREAMBLE.to_string(),
            params: InferenceParams::default(),
            executor: Some(exec),
            approval_tx: Some(tx),
        };

        let history = [ChatTurn::user("do stuff")];
        let err = client
            .complete(&history, fixture_snapshot())
            .await
            .expect_err("a tool-call-only script must exhaust the turn limit");
        assert!(
            matches!(err, AgentError::Request(_)),
            "expected run_agent_request's turn limit to be hit, got {err:?}"
        );

        assert!(
            read_calls.lock().unwrap().iter().any(|n| n == "doctor"),
            "complete() must forward &self.executor (not None): the read tool \
             never reached the recording executor"
        );
        match rx
            .try_recv()
            .expect("approval intent must be posted for the mutating call")
        {
            ClientMsg::ChatApprovalRequired { intent } => {
                assert_eq!(intent.name, "watcher_disable");
            }
            other => panic!("expected ChatApprovalRequired, got {other:?}"),
        }
    }

    /// As `rig_agent_client_complete_forwards_executor_and_approval_channel`,
    /// for `AnthropicAgentClient::complete`'s call site.
    #[tokio::test]
    async fn anthropic_agent_client_complete_forwards_executor_and_approval_channel() {
        use rig::test_utils::RecordingHttpClient;

        let response_body = serde_json::to_vec(&json!({
            "type": "message",
            "id": "msg-test",
            "model": "test-model",
            "role": "assistant",
            "content": [
                { "type": "tool_use", "id": "toolu_0", "name": "doctor", "input": {} },
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "watcher_disable",
                    "input": { "watcher": "w1" },
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": null,
                "cache_creation_input_tokens": null,
            },
        }))
        .expect("serialize stub Anthropic tool-call response");
        let http_client = RecordingHttpClient::new(response_body);
        let anthropic_client = rig::providers::anthropic::Client::builder()
            .api_key("sk-ant-test")
            .http_client(http_client.clone())
            .build()
            .expect("build client over the recording HTTP double");

        let read_calls = Arc::new(Mutex::new(Vec::new()));
        let exec: SharedRocmToolExecutor = Arc::new(RecordingExec {
            read_calls: read_calls.clone(),
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ClientMsg>();

        let client = AnthropicAgentClient {
            client: anthropic_client,
            model: "test-model".to_string(),
            preamble: TEST_PREAMBLE.to_string(),
            params: InferenceParams::default(),
            executor: Some(exec),
            approval_tx: Some(tx),
        };

        let history = [ChatTurn::user("do stuff")];
        let err = client
            .complete(&history, fixture_snapshot())
            .await
            .expect_err("a tool-call-only script must exhaust the turn limit");
        assert!(
            matches!(err, AgentError::Request(_)),
            "expected run_agent_request's turn limit to be hit, got {err:?}"
        );

        assert!(
            read_calls.lock().unwrap().iter().any(|n| n == "doctor"),
            "complete() must forward &self.executor (not None): the read tool \
             never reached the recording executor"
        );
        match rx
            .try_recv()
            .expect("approval intent must be posted for the mutating call")
        {
            ClientMsg::ChatApprovalRequired { intent } => {
                assert_eq!(intent.name, "watcher_disable");
            }
            other => panic!("expected ChatApprovalRequired, got {other:?}"),
        }
    }

    /// Builds the SSE bytes `rig-core`'s ChatGPT provider expects from a
    /// unary `send()` call: a single `response.completed` event wrapping a
    /// full `responses_api::CompletionResponse`. Every field the struct
    /// requires is given an explicit (usually null/empty) value rather than
    /// omitted, since `#[serde(flatten)]` on `additional_parameters` does not
    /// reliably tolerate missing keys the way a non-flattened `Option` field
    /// does. `output` is the one part callers vary.
    fn chatgpt_response_completed_sse(output: serde_json::Value) -> Vec<u8> {
        let response = json!({
            "id": "resp-test",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "test-model",
            "usage": null,
            "output": output,
            "tools": [],
            "background": null,
            "text": null,
            "include": null,
            "top_p": null,
            "truncation": null,
            "user": null,
            "metadata": {},
            "parallel_tool_calls": null,
            "previous_response_id": null,
            "reasoning": null,
            "service_tier": null,
            "store": null,
        });
        let event = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": response,
        });
        format!(
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&event).expect("serialize chatgpt response.completed event")
        )
        .into_bytes()
    }

    /// As `rig_agent_client_complete_forwards_params_to_the_wire` /
    /// `anthropic_agent_client_complete_forwards_params_to_the_wire`, for
    /// `ChatGptAgentClient::complete`'s call site. Uses rig-core 0.38.1's
    /// `ChatGPTAuth::AccessToken` variant, which skips device-code entirely
    /// (`auth_context()` returns immediately, no I/O), and `chatgpt::Client<H>`'s
    /// existing generic HTTP client — the completion path is a single unary
    /// `self.client.send(req)` whose body happens to be parsed as SSE text,
    /// not a `send_streaming` call, so `RecordingHttpClient` (a unary double)
    /// can serve it directly.
    ///
    /// Deliberately NOT asserted: `temperature`/`top_p`/`max_tokens`. Unlike
    /// Rig/Anthropic, rig-core's ChatGPT provider's own `create_request`
    /// unconditionally zeroes `temperature`, `max_output_tokens`, and
    /// `additional_parameters.top_p` before the request ever reaches the
    /// wire — regardless of what `self.params` holds. So a
    /// `&self.params` → `&InferenceParams::default()` swap at this call site
    /// is architecturally unobservable through the request body; the
    /// assertion below pins that absence instead, so a future rig-core
    /// upgrade that stops zeroing them doesn't silently leave params
    /// unpinned again — if it ever fires, replace it with a real
    /// params-forwarding assertion. (The dropped flags themselves are a
    /// pre-existing ChatGPT backend limitation, not something this test
    /// should paper over; tracked as <https://github.com/ROCm/rocm-cli/issues/664>.)
    ///
    /// Also NOT covered: this test builds `ChatGptAgentClient` from a struct
    /// literal with `AccessToken` auth, so `self.client.authorize()` returns
    /// immediately without ever reaching its `Err` arm. A broken mapping from
    /// a failed `authorize()` to `AgentError::Auth` would stay green here.
    #[tokio::test]
    async fn chatgpt_agent_client_complete_forwards_model_and_instructions_to_the_wire() {
        use rig::test_utils::RecordingHttpClient;

        let response_body = chatgpt_response_completed_sse(json!([{
            "type": "message",
            "id": "msg-test",
            "role": "assistant",
            "status": "completed",
            "content": [{ "type": "output_text", "text": "GPU 0 is healthy." }],
        }]));
        let http_client = RecordingHttpClient::new(response_body);
        let chatgpt_client = rig::providers::chatgpt::Client::builder()
            .api_key("test-access-token")
            .http_client(http_client.clone())
            .build()
            .expect("build client over the recording HTTP double");

        let client = ChatGptAgentClient {
            client: chatgpt_client,
            model: "test-model".to_string(),
            preamble: TEST_PREAMBLE.to_string(),
            params: InferenceParams {
                temperature: Some(0.4),
                top_p: None,
                max_tokens: Some(256),
            },
            executor: None,
            approval_tx: None,
        };

        let history = [ChatTurn::user("check gpu 0")];
        let (reply, tags) = complete_capturing_backend_tags(&client, &history).await;
        assert_eq!(reply, "GPU 0 is healthy.");

        let requests = http_client.requests();
        assert_eq!(
            requests.len(),
            1,
            "exactly one request should have been sent"
        );
        let request = &requests[0];
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("request body is JSON");
        assert_eq!(
            body["model"].as_str(),
            Some("test-model"),
            "complete() must forward &self.model, not a hardcoded one: {body}"
        );
        assert!(
            body["instructions"]
                .as_str()
                .is_some_and(|s| s.contains(TEST_PREAMBLE)),
            "complete() must forward &self.preamble, not a default: {body}"
        );
        let obj = body.as_object().expect("request body is a JSON object");
        assert!(
            !obj.contains_key("temperature") && !obj.contains_key("max_output_tokens"),
            "rig-core's ChatGPT provider is expected to zero temperature/\
             max_output_tokens before the wire (see doc comment above); if \
             this now fails, rig-core stopped doing that and params forwarding \
             needs a real assertion here: {body}"
        );

        assert!(
            !tags.is_empty() && tags.iter().all(|t| t == "chatgpt-oauth"),
            "complete() must tag its lifecycle trace events \"chatgpt-oauth\", got {tags:?}"
        );
    }

    /// Builds a `ChatGptAgentClient` wired to a `RecordingHttpClient` that
    /// replies with a single scripted turn, for the
    /// `chatgpt_agent_client_complete_warns_when_dropped_params_are_set` test
    /// below. Shared so both the non-default and the default-params run
    /// drive the exact same request/response shape; only `params` differs.
    /// Returns the `RecordingHttpClient` too, so the caller can inspect the
    /// actual wire request and check that the warned-about fields are
    /// omitted, not just that the diagnostic fired.
    fn chatgpt_client_for_dropped_params_warning_test(
        params: InferenceParams,
    ) -> (
        ChatGptAgentClient<rig::test_utils::RecordingHttpClient>,
        rig::test_utils::RecordingHttpClient,
    ) {
        use rig::test_utils::RecordingHttpClient;

        let response_body = chatgpt_response_completed_sse(json!([{
            "type": "message",
            "id": "msg-test",
            "role": "assistant",
            "status": "completed",
            "content": [{ "type": "output_text", "text": "GPU 0 is healthy." }],
        }]));
        let http_client = RecordingHttpClient::new(response_body);
        let chatgpt_client = rig::providers::chatgpt::Client::builder()
            .api_key("test-access-token")
            .http_client(http_client.clone())
            .build()
            .expect("build client over the recording HTTP double");

        (
            ChatGptAgentClient {
                client: chatgpt_client,
                model: "test-model".to_string(),
                preamble: TEST_PREAMBLE.to_string(),
                params,
                executor: None,
                approval_tx: None,
            },
            http_client,
        )
    }

    /// Regression test for <https://github.com/ROCm/rocm-cli/issues/664>:
    /// rig-core's ChatGPT provider silently zeroes `temperature`,
    /// `max_output_tokens`, and `top_p` before the wire (pinned by
    /// `chatgpt_agent_client_complete_forwards_model_and_instructions_to_the_wire`
    /// above), so `ChatGptAgentClient::complete` must instead warn the
    /// operator that these flags are ignored for this backend.
    ///
    /// Uses [`CapturedWarnEvents`], not [`CapturedBackendTags`]: a bare
    /// "chatgpt-oauth tag count went up by one" assertion stays green even if
    /// `warn!` were silently changed to `info!`, or the message text altered
    /// or removed — `CapturedBackendTags` ignores both level and message.
    /// [`CapturedWarnEvents::dropped_params_warning_count`] instead requires
    /// an EXACT match: WARN level, the `chatgpt-oauth` tag, AND the literal
    /// [`CHATGPT_DROPPED_PARAMS_WARNING`] text — so the test proves the
    /// SPECIFIC diagnostic fired, not merely that some extra event did.
    ///
    /// Each of `temperature`, `top_p`, and `max_tokens` is set ALONE in its
    /// own run (never paired with another) so a single dropped `.is_some()`
    /// disjunct in the production `if` — including the exact "forgot the
    /// `top_p` check" shape this issue is about — fails exactly one of these
    /// three cases instead of staying green under all of them. `top_p` can't
    /// be paired with `temperature` here regardless: ChatGPT Responses
    /// rejects that combination before the warn check ever runs (see
    /// `validate_chatgpt_inference_params`), which is also why the pre-existing
    /// `..._forwards_model_and_instructions_to_the_wire` test above never sets
    /// `top_p`.
    ///
    /// Each isolated case also inspects the actual wire request and asserts
    /// `temperature`/`max_output_tokens`/`top_p` are all absent from it — the
    /// warning's own claim, checked alongside the diagnostic rather than
    /// assumed.
    #[tokio::test]
    async fn chatgpt_agent_client_complete_warns_when_dropped_params_are_set() {
        warm_tracing_interest_cache();
        let history = [ChatTurn::user("check gpu 0")];

        let (default_client, default_http) =
            chatgpt_client_for_dropped_params_warning_test(InferenceParams::default());
        let default_warn_events = Arc::new(CapturedWarnEvents::default());
        let tracing_guard = tracing::subscriber::set_default(default_warn_events.clone());
        let reply = default_client
            .complete(&history, fixture_snapshot())
            .await
            .expect("scripted single-turn reply should succeed");
        drop(tracing_guard);
        assert_eq!(reply, "GPU 0 is healthy.");
        assert_eq!(
            default_warn_events.dropped_params_warning_count(),
            0,
            "InferenceParams::default() must not emit the dropped-params warning"
        );
        let default_requests = default_http.requests();
        assert_eq!(default_requests.len(), 1);
        let default_body: serde_json::Value =
            serde_json::from_slice(&default_requests[0].body).expect("request body is JSON");
        let default_obj = default_body
            .as_object()
            .expect("request body is a JSON object");
        for key in ["temperature", "max_output_tokens", "top_p"] {
            assert!(
                !default_obj.contains_key(key),
                "default params: {key} should not be on the wire either way: {default_body}"
            );
        }

        let isolated_cases = [
            (
                "temperature alone",
                InferenceParams {
                    temperature: Some(0.4),
                    top_p: None,
                    max_tokens: None,
                },
            ),
            (
                "top_p alone",
                InferenceParams {
                    temperature: None,
                    top_p: Some(0.9),
                    max_tokens: None,
                },
            ),
            (
                "max_tokens alone",
                InferenceParams {
                    temperature: None,
                    top_p: None,
                    max_tokens: Some(256),
                },
            ),
        ];

        for (label, params) in isolated_cases {
            let (client, http) = chatgpt_client_for_dropped_params_warning_test(params);
            let warn_events = Arc::new(CapturedWarnEvents::default());
            let tracing_guard = tracing::subscriber::set_default(warn_events.clone());
            let reply = client
                .complete(&history, fixture_snapshot())
                .await
                .expect("scripted single-turn reply should succeed");
            drop(tracing_guard);
            assert_eq!(reply, "GPU 0 is healthy.", "case: {label}");
            assert_eq!(
                warn_events.dropped_params_warning_count(),
                1,
                "case {label}: setting it alone must emit the dropped-params \
                 warning exactly once"
            );

            let requests = http.requests();
            assert_eq!(requests.len(), 1, "case {label}");
            let body: serde_json::Value =
                serde_json::from_slice(&requests[0].body).expect("request body is JSON");
            let obj = body.as_object().expect("request body is a JSON object");
            for key in ["temperature", "max_output_tokens", "top_p"] {
                assert!(
                    !obj.contains_key(key),
                    "case {label}: {key} must be absent from the wire request \
                     (rig-core zeroes it before sending) even though the \
                     warning fired for the dropped param: {body}"
                );
            }
        }
    }

    /// As `rig_agent_client_complete_forwards_executor_and_approval_channel`,
    /// for `ChatGptAgentClient::complete`'s call site.
    #[tokio::test]
    async fn chatgpt_agent_client_complete_forwards_executor_and_approval_channel() {
        use rig::test_utils::RecordingHttpClient;

        let response_body = chatgpt_response_completed_sse(json!([
            {
                "type": "function_call",
                "id": "fc-0",
                "call_id": "call_0",
                "name": "doctor",
                "arguments": "{}",
                "status": "completed",
            },
            {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call_1",
                "name": "watcher_disable",
                "arguments": "{\"watcher\":\"w1\"}",
                "status": "completed",
            },
        ]));
        let http_client = RecordingHttpClient::new(response_body);
        let chatgpt_client = rig::providers::chatgpt::Client::builder()
            .api_key("test-access-token")
            .http_client(http_client.clone())
            .build()
            .expect("build client over the recording HTTP double");

        let read_calls = Arc::new(Mutex::new(Vec::new()));
        let exec: SharedRocmToolExecutor = Arc::new(RecordingExec {
            read_calls: read_calls.clone(),
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ClientMsg>();

        let client = ChatGptAgentClient {
            client: chatgpt_client,
            model: "test-model".to_string(),
            preamble: TEST_PREAMBLE.to_string(),
            params: InferenceParams::default(),
            executor: Some(exec),
            approval_tx: Some(tx),
        };

        let history = [ChatTurn::user("do stuff")];
        let err = client
            .complete(&history, fixture_snapshot())
            .await
            .expect_err("a tool-call-only script must exhaust the turn limit");
        assert!(
            matches!(err, AgentError::Request(_)),
            "expected run_agent_request's turn limit to be hit, got {err:?}"
        );

        assert!(
            read_calls.lock().unwrap().iter().any(|n| n == "doctor"),
            "complete() must forward &self.executor (not None): the read tool \
             never reached the recording executor"
        );
        match rx
            .try_recv()
            .expect("approval intent must be posted for the mutating call")
        {
            ClientMsg::ChatApprovalRequired { intent } => {
                assert_eq!(intent.name, "watcher_disable");
            }
            other => panic!("expected ChatApprovalRequired, got {other:?}"),
        }
    }

    /// Pins that `ChatGptAgentClient::complete`'s own call site forwards
    /// `&self.params` into `validate_chatgpt_inference_params` and rejects
    /// before any wire request — `chatgpt_rejects_temperature_and_top_p_together`
    /// only drives the bare helper function and proves nothing about whether
    /// `complete()` actually calls it.
    #[tokio::test]
    async fn chatgpt_agent_client_complete_rejects_temperature_and_top_p_before_any_wire_request() {
        use rig::test_utils::RecordingHttpClient;

        let http_client = RecordingHttpClient::new(Vec::new());
        let chatgpt_client = rig::providers::chatgpt::Client::builder()
            .api_key("test-access-token")
            .http_client(http_client.clone())
            .build()
            .expect("build client over the recording HTTP double");

        let client = ChatGptAgentClient {
            client: chatgpt_client,
            model: "test-model".to_string(),
            preamble: TEST_PREAMBLE.to_string(),
            params: InferenceParams {
                temperature: Some(0.4),
                top_p: Some(0.5),
                max_tokens: None,
            },
            executor: None,
            approval_tx: None,
        };

        let history = [ChatTurn::user("check gpu 0")];
        let err = client
            .complete(&history, fixture_snapshot())
            .await
            .expect_err("mutually exclusive temperature/top_p must be rejected");
        assert!(matches!(err, AgentError::Build(_)));
        assert!(err.to_string().contains("either temperature or top_p"));
        assert!(
            http_client.requests().is_empty(),
            "complete() must reject before sending any request to the wire"
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
