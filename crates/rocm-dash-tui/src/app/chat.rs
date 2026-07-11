// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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
            // Require a real key. Without this, `RigAgentClient::new` falls back to
            // a dummy `sk-no-key` bearer and still builds, so the switch reports
            // success and then 401s at request time. Returning `None` here makes
            // the caller surface an actionable error and stay on the current
            // backend instead of switching to a dead one.
            let api_key = args.chat_api_key.clone().filter(|k| !k.trim().is_empty())?;
            let cfg = crate::llm::LlmConfig {
                base_url: OPENAI_BASE_URL.to_string(),
                model: args
                    .chat_model
                    .clone()
                    .filter(|m| !m.is_empty())
                    .unwrap_or_else(|| OPENAI_DEFAULT_MODEL.to_string()),
                api_key: Some(api_key),
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

/// Build the live agent for a `Local` (inline-detected) `LlmConfig`.
///
/// The single construction path for the local backend, shared by the event
/// loop's startup build and its endpoint-rebuild drain so any change to how a
/// local agent is constructed lives in exactly one place. Construction only —
/// no network I/O. Returns the [`AgentError`](crate::agent::AgentError) so the
/// caller can either discard it (`.ok()` at startup) or surface it as an error
/// turn (the rebuild drain).
pub(super) fn build_local_agent(
    cfg: crate::llm::LlmConfig,
    executor: Option<crate::tool_exec::SharedRocmToolExecutor>,
    approval_tx: mpsc::UnboundedSender<ClientMsg>,
) -> Result<std::sync::Arc<dyn crate::agent::AgentClient>, crate::agent::AgentError> {
    crate::agent::RigAgentClient::new(cfg, executor, Some(approval_tx))
        .map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn crate::agent::AgentClient>)
}

/// Local engines that expose an OpenAI-compatible `/v1` surface the dash chat
/// can talk to directly. A managed service running one of these is a valid
/// auto-detected chat endpoint regardless of which port it bound.
const OPENAI_COMPATIBLE_ENGINES: &[&str] = &["vllm", "lemonade"];

/// A managed-service endpoint the dash chat can route to, picked from the
/// read-only `services` tool payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManagedChatEndpoint {
    /// `endpoint_url` from the registry — already includes the `/v1` suffix.
    pub base_url: String,
    /// The service's model id (`canonical_model_id`, then `model_ref`), if any.
    pub model: Option<String>,
}

/// Pick the best **ready**, OpenAI-compatible managed-service endpoint from the
/// `services` tool's JSON envelope (`structuredContent.services`).
///
/// "Ready" mirrors the bin's own HTTP readiness check (what `rocm services`
/// reports), so a selected endpoint has been verified to actually serve. Among
/// ready candidates the most recently created wins. Pure — no I/O; the anchor
/// for the port-detection unit tests.
pub(crate) fn pick_managed_chat_endpoint(
    services_result: &serde_json::Value,
) -> Option<ManagedChatEndpoint> {
    let services = services_result
        .get("structuredContent")
        .and_then(|s| s.get("services"))
        .and_then(serde_json::Value::as_array)?;

    let mut best: Option<(&serde_json::Value, u64)> = None;
    for record in services {
        let engine = record
            .get("engine")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !OPENAI_COMPATIBLE_ENGINES
            .iter()
            .any(|known| engine.eq_ignore_ascii_case(known))
        {
            continue;
        }
        let ready = record
            .get("status")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("ready"));
        if !ready {
            continue;
        }
        let endpoint_present = record
            .get("endpoint_url")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|e| !e.is_empty());
        if !endpoint_present {
            continue;
        }
        let created = record
            .get("created_at_unix_ms")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if best
            .as_ref()
            .is_none_or(|(_, best_created)| created >= *best_created)
        {
            best = Some((record, created));
        }
    }

    let (record, _) = best?;
    let base_url = record
        .get("endpoint_url")
        .and_then(serde_json::Value::as_str)?
        .to_string();
    let model = record
        .get("canonical_model_id")
        .and_then(serde_json::Value::as_str)
        .filter(|m| !m.is_empty())
        .or_else(|| {
            record
                .get("model_ref")
                .and_then(serde_json::Value::as_str)
                .filter(|m| !m.is_empty())
        })
        .map(str::to_string);
    Some(ManagedChatEndpoint { base_url, model })
}

/// Query the bin's read-only `services` tool and, if a ready OpenAI-compatible
/// managed service exists, return its endpoint as an [`LlmConfig`].
///
/// This is how the dash learns the *actual* port it launched an engine on (e.g.
/// a tool-launched vLLM on a non-default port) instead of guessing the
/// well-known defaults. The seam call is blocking, so it runs off the reactor.
/// A best-effort `/v1/models` fetch both confirms the endpoint is live and
/// supplies the served model id; on fetch failure the registry-reported model
/// is used (the service was already readiness-verified by the bin).
pub(super) async fn detect_managed_chat(
    executor: Option<crate::tool_exec::SharedRocmToolExecutor>,
) -> Option<crate::llm::LlmConfig> {
    let executor = executor?;
    let outcome =
        tokio::task::spawn_blocking(move || executor.execute("services", &serde_json::json!({})))
            .await
            .ok()?;
    let crate::tool_exec::RocmToolOutcome::Result(value) = outcome else {
        return None;
    };
    let picked = pick_managed_chat_endpoint(&value)?;
    match fetch_first_model(&picked.base_url).await {
        Some(model) => Some(crate::llm::detected_llm_config(&picked.base_url, &model)),
        None => picked
            .model
            .map(|model| crate::llm::detected_llm_config(&picked.base_url, &model)),
    }
}

/// Probe for a local chat engine, returning a ready [`LlmConfig`] or `None`.
///
/// Registry-first: an engine we launched ourselves (known via the managed-
/// services registry, on whatever port it bound) takes priority over the
/// well-known default ports. Falls back to the TCP probe of the well-known
/// Lemonade/vLLM endpoints when no managed service is available.
pub(super) async fn detect_local_chat(
    executor: Option<crate::tool_exec::SharedRocmToolExecutor>,
) -> Option<crate::llm::LlmConfig> {
    if let Some(cfg) = detect_managed_chat(executor).await {
        return Some(cfg);
    }

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

/// Whether startup should run the local-engine auto-detection swap.
///
/// Detection produces a keyless [`crate::llm::detected_llm_config`], so it may
/// only fire when the user configured NO endpoint (URL/env URL) AND NO api key
/// — otherwise the swap would silently drop a configured credential and 401 at
/// request time. Pure; the anchor for the startup-gate unit test.
pub(super) const fn should_detect_local_chat(
    chat_url: Option<&str>,
    chat_env_url: Option<&str>,
    chat_api_key: Option<&str>,
) -> bool {
    chat_url.is_none() && chat_env_url.is_none() && chat_api_key.is_none()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrap service records in the `services` tool's success envelope shape
    /// (`structuredContent.services`), matching `internal_mcp_tool_success`.
    fn services_envelope(records: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "content": [{ "type": "text", "text": "services" }],
            "structuredContent": { "services": records },
            "isError": false,
        })
    }

    fn service(
        engine: &str,
        endpoint_url: &str,
        status: &str,
        model: &str,
        created: u64,
    ) -> serde_json::Value {
        serde_json::json!({
            "engine": engine,
            "endpoint_url": endpoint_url,
            "status": status,
            "canonical_model_id": model,
            "created_at_unix_ms": created,
        })
    }

    #[test]
    fn picks_ready_vllm_on_nondefault_port() {
        // The headline case: a tool-launched vLLM bound a non-default port; the
        // registry knows the real endpoint and it must be detected.
        let env = services_envelope(serde_json::json!([service(
            "vllm",
            "http://127.0.0.1:11435/v1",
            "ready",
            "Qwen3-8B",
            100
        )]));
        let picked = pick_managed_chat_endpoint(&env).expect("ready vLLM is picked");
        assert_eq!(picked.base_url, "http://127.0.0.1:11435/v1");
        assert_eq!(picked.model.as_deref(), Some("Qwen3-8B"));
    }

    #[test]
    fn skips_non_ready_services() {
        let env = services_envelope(serde_json::json!([service(
            "vllm",
            "http://127.0.0.1:11435/v1",
            "starting",
            "Qwen3-8B",
            100
        )]));
        assert_eq!(pick_managed_chat_endpoint(&env), None);
    }

    #[test]
    fn skips_non_openai_compatible_engines() {
        // A hypothetical non-OpenAI engine must not be offered as a chat endpoint.
        let env = services_envelope(serde_json::json!([service(
            "comfyui",
            "http://127.0.0.1:8188",
            "ready",
            "sd",
            100
        )]));
        assert_eq!(pick_managed_chat_endpoint(&env), None);
    }

    #[test]
    fn prefers_most_recently_created_among_ready() {
        let env = services_envelope(serde_json::json!([
            service("vllm", "http://127.0.0.1:8000/v1", "ready", "old", 100),
            service("lemonade", "http://127.0.0.1:13305/v1", "ready", "new", 200),
        ]));
        let picked = pick_managed_chat_endpoint(&env).expect("a ready endpoint");
        assert_eq!(picked.base_url, "http://127.0.0.1:13305/v1");
        assert_eq!(picked.model.as_deref(), Some("new"));
    }

    #[test]
    fn falls_back_to_model_ref_when_canonical_missing() {
        let env = services_envelope(serde_json::json!([{
            "engine": "vllm",
            "endpoint_url": "http://127.0.0.1:11435/v1",
            "status": "ready",
            "canonical_model_id": "",
            "model_ref": "org/Model-Ref",
            "created_at_unix_ms": 1u64,
        }]));
        let picked = pick_managed_chat_endpoint(&env).expect("ready endpoint");
        assert_eq!(picked.model.as_deref(), Some("org/Model-Ref"));
    }

    #[test]
    fn none_on_empty_or_malformed() {
        assert_eq!(
            pick_managed_chat_endpoint(&services_envelope(serde_json::json!([]))),
            None
        );
        assert_eq!(pick_managed_chat_endpoint(&serde_json::json!({})), None);
        assert_eq!(
            pick_managed_chat_endpoint(&serde_json::json!({ "structuredContent": {} })),
            None
        );
    }

    #[test]
    fn skips_ready_record_with_empty_endpoint() {
        let env = services_envelope(serde_json::json!([service("vllm", "", "ready", "m", 100)]));
        assert_eq!(pick_managed_chat_endpoint(&env), None);
    }

    /// EAI-7347: startup now reuses `detect_local_chat` (the same detection the
    /// manual 'd' path already used) instead of a single well-known-port probe,
    /// so a local server on a non-default well-known port (e.g. `rocm serve`'s
    /// :11435) is preferred over falling back to the ChatGPT cloud default.
    #[tokio::test]
    async fn detect_local_chat_prefers_local_server_over_no_endpoint() {
        // detect_local_chat probes candidates in priority order (Lemonade,
        // then vLLM, then rocm serve's default). For the rocm-serve listener
        // below to be the one detect_local_chat actually picks, the two
        // higher-priority ports must be free — an ambient listener on the CI
        // runner (e.g. another process already on :8000) would otherwise
        // make this test flake by returning a different endpoint than
        // expected. Note: unlike the guard below, these can only be checked
        // and released (not held) — the probe is a bare TCP connect, so a
        // held listener would itself register as "reachable" and defeat the
        // check.
        let Ok(lemonade_probe) =
            std::net::TcpListener::bind(("127.0.0.1", crate::skills::LEMONADE_PORT))
        else {
            return; // Port already bound in this environment; skip rather than flake.
        };
        drop(lemonade_probe);
        let Ok(vllm_probe) = std::net::TcpListener::bind(("127.0.0.1", crate::skills::VLLM_PORT))
        else {
            return; // Port already bound in this environment; skip rather than flake.
        };
        drop(vllm_probe);
        let Ok(listener) =
            std::net::TcpListener::bind(("127.0.0.1", crate::skills::ROCM_SERVE_PORT))
        else {
            return; // Port already bound in this environment; skip rather than flake.
        };
        // No executor (no managed-services registry available) — falls through
        // to the well-known-port probe, which must still find the listener.
        let cfg = detect_local_chat(None)
            .await
            .expect("local server detected");
        assert_eq!(cfg.base_url, crate::skills::ROCM_SERVE_ENDPOINT);
        drop(listener);
    }

    #[test]
    fn should_detect_only_when_no_url_env_or_key_configured() {
        // The clean slate: nothing configured → auto-detect a local engine.
        assert!(should_detect_local_chat(None, None, None));
        // A configured api key must SUPPRESS the swap — detection returns a
        // keyless config and would otherwise silently drop the key (401).
        assert!(!should_detect_local_chat(None, None, Some("sk-configured")));
        // An explicit URL or env URL also suppresses it (config precedence).
        assert!(!should_detect_local_chat(
            Some("http://cfg:1/v1"),
            None,
            None
        ));
        assert!(!should_detect_local_chat(
            None,
            Some("http://env:2/v1"),
            None
        ));
    }
}
