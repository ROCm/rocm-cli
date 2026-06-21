//! Chat-backend construction, detection, and persistence.
//!
//! The provider→agent factory ([`build_chat_agent`]), the local-engine probe
//! ([`detect_local_chat`] + [`fetch_first_model`]), and the config persistence
//! for an accepted endpoint ([`persist_chat_endpoint`] + [`config_with_chat`]).
//! Split out of `app/mod.rs` to keep the core reducer + event loop focused. The
//! one reducer method here, [`AppState::set_chat_config`], lives with the rest
//! of the chat-backend resolution group it configures.

use tokio::sync::mpsc;

use super::{AppState, ChatConsent, ChatProvider, ResolvedArgs};
use crate::client::ClientMsg;

/// OpenAI default base URL when the `Openai` provider is selected.
const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// Default OpenAI model when none is configured via `chat_model`.
const OPENAI_DEFAULT_MODEL: &str = "gpt-5";

impl AppState {
    pub fn set_chat_config(&mut self, llm: Option<crate::llm::LlmConfig>, pre_consent: bool) {
        self.chat_consent = match (&llm, pre_consent) {
            (None, _) => ChatConsent::Unavailable,
            (Some(_), true) => ChatConsent::Accepted,
            (Some(_), false) => ChatConsent::Pending,
        };
        self.chat_llm = llm;
    }
}

/// Construct the chat backend for an explicitly-selected provider (Phase 8).
///
/// Construction only — NO network I/O happens here (rig clients defer the
/// request to `complete()`). Handles ONLY `Openai` and `Anthropic`; `Local`
/// returns `None` because its build (auto-detect probe → Rig/ChatGPT) is owned
/// by `event_loop`'s inline path and can't be reproduced from `ResolvedArgs`
/// alone. Keys come from `ResolvedArgs` (in-process seam), never argv. `None`
/// signals "couldn't build" (e.g. a missing key) so the caller surfaces an
/// actionable error turn instead of switching to a dead backend.
pub(super) fn build_chat_agent(
    provider: ChatProvider,
    args: &ResolvedArgs,
    executor: Option<crate::tool_exec::SharedRocmToolExecutor>,
    approval_tx: mpsc::UnboundedSender<ClientMsg>,
) -> Option<std::sync::Arc<dyn crate::agent::AgentClient>> {
    match provider {
        // Local is rebuilt by the caller's inline path (it needs the live probe).
        ChatProvider::Local => None,
        ChatProvider::Openai => {
            let cfg = crate::llm::LlmConfig {
                base_url: OPENAI_BASE_URL.to_string(),
                model: args
                    .chat_model
                    .clone()
                    .filter(|m| !m.is_empty())
                    .unwrap_or_else(|| OPENAI_DEFAULT_MODEL.to_string()),
                api_key: args.chat_api_key.clone(),
                auth_header: None,
            };
            crate::agent::RigAgentClient::new(cfg, executor, Some(approval_tx))
                .ok()
                .map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn crate::agent::AgentClient>)
        }
        ChatProvider::Anthropic => {
            // Leave base_url empty → the Anthropic backend uses rig's default
            // host. model "" → CLAUDE_SONNET_4_6 (resolved inside the backend).
            let cfg = crate::llm::LlmConfig {
                base_url: String::new(),
                model: args.chat_model.clone().unwrap_or_default(),
                api_key: args.anthropic_api_key.clone(),
                auth_header: None,
            };
            crate::agent::AnthropicAgentClient::new(cfg, executor, Some(approval_tx))
                .ok()
                .map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn crate::agent::AgentClient>)
        }
    }
}

/// Probe for a local chat engine, returning a ready [`LlmConfig`] or `None`.
pub(super) async fn detect_local_chat() -> Option<crate::llm::LlmConfig> {
    // TCP probe is blocking; keep it off the async reactor.
    let base = tokio::task::spawn_blocking(crate::llm::detect_local_endpoint)
        .await
        .ok()
        .flatten()?;

    // Best-effort model query; fall back to the neutral default on any failure.
    let model = fetch_first_model(base)
        .await
        .unwrap_or_else(|| crate::llm::DEFAULT_CHAT_MODEL.to_string());
    Some(crate::llm::detected_llm_config(base, &model))
}

/// Persist an accepted local endpoint to the user's `config.toml`: load the
/// existing config (or defaults), set `tui.chat_url`/`tui.chat_model`, and write
/// it back. Best-effort — returns a human error string on failure.
///
/// Uses [`default_config_path`] (a `--config` override is not honored by this
/// in-TUI save; that's a documented limitation). All I/O lives here.
pub(super) fn persist_chat_endpoint(
    base_url: &str,
    model: &str,
) -> Result<std::path::PathBuf, String> {
    use rocm_dash_core::config::{Config, default_config_path};
    let path = default_config_path().ok_or_else(|| "no config path available".to_string())?;
    let cfg = Config::load(&path).unwrap_or_default();
    let next = config_with_chat(cfg, base_url, model);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let toml = toml::to_string_pretty(&next).map_err(|e| e.to_string())?;
    std::fs::write(&path, toml).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Pure immutable transform: return a copy of `cfg` with the chat endpoint set
/// to a local engine (base_url + model), clearing any gateway auth header since
/// local engines need none.
pub(super) fn config_with_chat(
    mut cfg: rocm_dash_core::config::Config,
    base_url: &str,
    model: &str,
) -> rocm_dash_core::config::Config {
    cfg.tui.chat_url = Some(base_url.to_string());
    cfg.tui.chat_model = Some(model.to_string());
    cfg.tui.chat_auth_header = None;
    cfg
}

/// GET `{base}/models` and return the first served model id, or `None`.
async fn fetch_first_model(base_url: &str) -> Option<String> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok()?;
    let resp = client.get(&url).send().await.ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    crate::llm::pick_first_model(&json)
}
