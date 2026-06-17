//! Chat LLM configuration: a pure precedence resolver plus a std-only TCP liveness probe.
//!
//! No HTTP client here — the Rig-built `AgentClient` (Phase 3)
//! lives in `agent.rs`. Keeping detection in the TUI crate preserves the
//! core's render/async-free boundary.

use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Probed-default endpoint: the local OpenAI-compatible serving endpoint the
/// dashboard already watches. Used only when no URL is configured anywhere.
pub const DEFAULT_CHAT_BASE_URL: &str = "http://127.0.0.1:8000";

/// Fallback model name when none is configured. Many local endpoints ignore
/// the model field or expose a single model, so a neutral default is safe.
pub const DEFAULT_CHAT_MODEL: &str = "local-model";

/// Short, one-shot probe budget so detection never stalls TUI startup.
pub const PROBE_TIMEOUT: Duration = Duration::from_millis(300);

/// Fully-resolved chat endpoint configuration. `api_key` is sourced from the
/// environment only — never from TOML/CLI/source (see `main.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    /// Custom auth header NAME (e.g. `Ocp-Apim-Subscription-Key`). When set, the
    /// `api_key` value is sent in this header instead of `Authorization: Bearer`.
    pub auth_header: Option<String>,
}

/// Resolve the chat endpoint by precedence: **CLI > config > env > probed
/// default**, returning `None` when nothing is available (no URL from any tier
/// and the default endpoint is not reachable).
///
/// Pure — all I/O (env reads, the TCP probe) happens at the call site and is
/// passed in. This is the single source of precedence truth and the unit-test
/// anchor. `cli_*` carries the CLI value already merged over config in the
/// `rocm` binary (so `cfg_*` is typically `None` at runtime); the separate
/// `cfg_*` params keep every tier independently testable.
#[allow(clippy::too_many_arguments)]
pub fn resolve_llm_config(
    cli_url: Option<&str>,
    cli_model: Option<&str>,
    cfg_url: Option<&str>,
    cfg_model: Option<&str>,
    env_key: Option<&str>,
    env_url: Option<&str>,
    auth_header: Option<&str>,
    probe_ok: bool,
) -> Option<LlmConfig> {
    // base_url precedence: CLI > config > env > probed-default (gated by probe).
    let base_url = cli_url
        .or(cfg_url)
        .or(env_url)
        .map(str::to_string)
        .or_else(|| probe_ok.then(|| DEFAULT_CHAT_BASE_URL.to_string()))?;

    // model precedence: CLI > config > built-in default.
    let model = cli_model
        .or(cfg_model)
        .map_or_else(|| DEFAULT_CHAT_MODEL.to_string(), str::to_string);

    Some(LlmConfig {
        base_url,
        model,
        api_key: env_key.map(str::to_string),
        auth_header: auth_header.map(str::to_string),
    })
}

/// Parse `host` and `port` out of an OpenAI-style base URL. Tolerates a
/// missing scheme and trailing path; defaults the port from the scheme
/// (`https` → 443, otherwise 80). Pure — no DNS, no connection.
pub fn parse_host_port(base_url: &str) -> Option<(String, u16)> {
    let trimmed = base_url.trim();
    let (scheme, rest) = match trimmed.split_once("://") {
        Some((s, r)) => (s, r),
        None => ("http", trimmed),
    };
    // Drop any path/query after the authority.
    let authority = rest.split(['/', '?']).next().unwrap_or(rest);
    if authority.is_empty() {
        return None;
    }
    let default_port = if scheme.eq_ignore_ascii_case("https") {
        443
    } else {
        80
    };
    match authority.rsplit_once(':') {
        Some((host, port_str)) if !host.is_empty() => {
            let port = port_str.parse::<u16>().ok()?;
            Some((host.to_string(), port))
        }
        _ => Some((authority.to_string(), default_port)),
    }
}

/// Best-effort TCP liveness probe for a candidate endpoint.
///
/// Returns `true` only if a connection to the parsed host:port succeeds within `timeout`.
/// Never panics; any parse/DNS/connect failure yields `false`.
pub fn probe_endpoint(base_url: &str, timeout: Duration) -> bool {
    let Some((host, port)) = parse_host_port(base_url) else {
        return false;
    };
    let Ok(addrs) = (host.as_str(), port).to_socket_addrs() else {
        return false;
    };
    for addr in addrs {
        if TcpStream::connect_timeout(&addr, timeout).is_ok() {
            return true;
        }
    }
    false
}

/// Probe the known local serving endpoints in priority order (Lemonade, then vLLM) and return the first reachable one.
///
/// Used by the TUI's in-app "detect a local engine" action so the user need not run the CLI skill first.
/// TCP-only (no HTTP); never blocks longer than [`PROBE_TIMEOUT`] per candidate.
pub fn detect_local_endpoint() -> Option<&'static str> {
    [
        crate::skills::LEMONADE_ENDPOINT,
        crate::skills::VLLM_ENDPOINT,
    ]
    .into_iter()
    .find(|ep| probe_endpoint(ep, PROBE_TIMEOUT))
}

/// Pick the first model id from an OpenAI-compatible `/v1/models` response (`{"data":[{"id":"…"}]}`).
///
/// Pure — no HTTP. `None` when the shape is missing,
/// empty, or malformed, so the caller can fall back to [`DEFAULT_CHAT_MODEL`].
pub fn pick_first_model(models_json: &serde_json::Value) -> Option<String> {
    models_json
        .get("data")?
        .as_array()?
        .iter()
        .find_map(|m| m.get("id").and_then(|id| id.as_str()))
        .map(str::to_string)
}

/// Build an [`LlmConfig`] for a locally-detected engine. Local OpenAI-compatible
/// servers (Lemonade / vLLM) need no gateway auth, so `api_key`/`auth_header`
/// are `None`. Pure.
pub fn detected_llm_config(base_url: &str, model: &str) -> LlmConfig {
    LlmConfig {
        base_url: base_url.to_string(),
        model: model.to_string(),
        api_key: None,
        auth_header: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_first_model_reads_data_array() {
        let j = serde_json::json!({"object":"list","data":[{"id":"Llama-3.2-3B"},{"id":"qwen"}]});
        assert_eq!(pick_first_model(&j).as_deref(), Some("Llama-3.2-3B"));
    }

    #[test]
    fn pick_first_model_none_on_empty_or_malformed() {
        assert_eq!(pick_first_model(&serde_json::json!({"data":[]})), None);
        assert_eq!(pick_first_model(&serde_json::json!({})), None);
        assert_eq!(pick_first_model(&serde_json::json!({"data":"nope"})), None);
        // entries without an "id" string are skipped
        assert_eq!(
            pick_first_model(&serde_json::json!({"data":[{"x":1}]})),
            None
        );
    }

    #[test]
    fn detected_llm_config_carries_no_auth() {
        let c = detected_llm_config("http://localhost:13305/v1", "m");
        assert_eq!(c.base_url, "http://localhost:13305/v1");
        assert_eq!(c.model, "m");
        assert_eq!(c.api_key, None);
        assert_eq!(c.auth_header, None);
    }

    #[test]
    fn cli_wins_over_every_other_tier() {
        let r = resolve_llm_config(
            Some("http://cli:1"),
            Some("cli-model"),
            Some("http://cfg:2"),
            Some("cfg-model"),
            Some("k"),
            Some("http://env:3"),
            Some("Ocp-Apim-Subscription-Key"),
            true,
        )
        .unwrap();
        assert_eq!(r.base_url, "http://cli:1");
        assert_eq!(r.model, "cli-model");
        assert_eq!(r.api_key.as_deref(), Some("k"));
        assert_eq!(r.auth_header.as_deref(), Some("Ocp-Apim-Subscription-Key"));
    }

    #[test]
    fn config_wins_when_no_cli() {
        let r = resolve_llm_config(
            None,
            None,
            Some("http://cfg:2"),
            Some("cfg-model"),
            None,
            Some("http://env:3"),
            None,
            true,
        )
        .unwrap();
        assert_eq!(r.base_url, "http://cfg:2");
        assert_eq!(r.model, "cfg-model");
        assert_eq!(r.api_key, None);
    }

    #[test]
    fn env_url_wins_when_no_cli_or_config() {
        let r = resolve_llm_config(
            None,
            None,
            None,
            None,
            Some("k"),
            Some("http://env:3"),
            None,
            false,
        )
        .unwrap();
        assert_eq!(r.base_url, "http://env:3");
        // No model anywhere → built-in default.
        assert_eq!(r.model, DEFAULT_CHAT_MODEL);
        assert_eq!(r.api_key.as_deref(), Some("k"));
    }

    #[test]
    fn probed_default_used_when_nothing_configured_but_reachable() {
        let r = resolve_llm_config(None, None, None, None, None, None, None, true).unwrap();
        assert_eq!(r.base_url, DEFAULT_CHAT_BASE_URL);
        assert_eq!(r.model, DEFAULT_CHAT_MODEL);
        assert_eq!(r.api_key, None);
    }

    #[test]
    fn none_when_nothing_available() {
        assert_eq!(
            resolve_llm_config(None, None, None, None, None, None, None, false),
            None
        );
    }

    #[test]
    fn parse_host_port_handles_scheme_path_and_defaults() {
        assert_eq!(
            parse_host_port("http://127.0.0.1:8000"),
            Some(("127.0.0.1".into(), 8000))
        );
        assert_eq!(
            parse_host_port("http://127.0.0.1:8000/v1"),
            Some(("127.0.0.1".into(), 8000))
        );
        assert_eq!(
            parse_host_port("localhost:1234"),
            Some(("localhost".into(), 1234))
        );
        assert_eq!(
            parse_host_port("https://api.example.com"),
            Some(("api.example.com".into(), 443))
        );
        assert_eq!(
            parse_host_port("http://example.com"),
            Some(("example.com".into(), 80))
        );
        assert_eq!(parse_host_port(""), None);
    }

    #[test]
    fn probe_unreachable_port_is_false_not_panic() {
        // Port 1 on localhost is essentially never open; must return false fast.
        assert!(!probe_endpoint(
            "http://127.0.0.1:1",
            Duration::from_millis(100)
        ));
        assert!(!probe_endpoint("not a url", Duration::from_millis(100)));
    }
}
