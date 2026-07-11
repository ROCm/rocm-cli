// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Chat LLM configuration: a pure precedence resolver plus a std-only TCP liveness probe.
//!
//! No HTTP client here — the Rig-built `AgentClient` (Phase 3)
//! lives in `agent.rs`. Keeping detection in the TUI crate preserves the
//! core's render/async-free boundary.

use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Probed-default endpoint: the local OpenAI-compatible serving endpoint the
/// dashboard already watches. Used only when no URL is configured anywhere.
///
/// The `/v1` suffix is required: the Rig OpenAI client appends `chat/completions`
/// directly to `base_url` (`base_url + "/" + path`), so a base without `/v1`
/// POSTs to `/chat/completions` and a vLLM/Lemonade server answers `404 Not
/// Found`. Matches the `/v1` convention of `VLLM_ENDPOINT` / `LEMONADE_ENDPOINT`
/// (those use `localhost`; `127.0.0.1` is the unambiguous IPv4 loopback and
/// matches the live endpoint asserted in `agent.rs`).
pub const DEFAULT_CHAT_BASE_URL: &str = "http://127.0.0.1:8000/v1";

/// Fallback model name when none is configured. Many local endpoints ignore
/// the model field or expose a single model, so a neutral default is safe.
pub const DEFAULT_CHAT_MODEL: &str = "local-model";

/// Short, one-shot probe budget so detection never stalls TUI startup.
pub const PROBE_TIMEOUT: Duration = Duration::from_millis(300);

/// Fully-resolved chat endpoint configuration. `api_key` is sourced from the
/// environment only — never from TOML/CLI/source (see `main.rs`).
#[derive(Clone, PartialEq, Eq)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    /// Custom auth header NAME (e.g. `Ocp-Apim-Subscription-Key`). When set, the
    /// `api_key` value is sent in this header instead of `Authorization: Bearer`.
    pub auth_header: Option<String>,
}

impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("auth_header", &self.auth_header)
            .finish()
    }
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

    // Loopback base_urls talk to a local server that needs no gateway auth;
    // never attach a cloud subscription-key header/api_key to them (prevents a
    // startup-configured local endpoint from leaking a cloud credential).
    let is_loopback = parse_host_port(&base_url).is_some_and(|(host, _)| is_loopback_host(&host));

    // Make the stripping debuggable: a user pointing at an authenticating local
    // proxy will otherwise see 401s with no on-host signal. Only warn when there
    // was actually a credential to discard.
    if is_loopback && (env_key.is_some() || auth_header.is_some()) {
        tracing::warn!(
            %base_url,
            "loopback endpoint: discarding configured api_key/auth_header (local servers use no gateway auth)"
        );
    }

    Some(LlmConfig {
        base_url,
        model,
        api_key: if is_loopback {
            None
        } else {
            env_key.map(str::to_string)
        },
        auth_header: if is_loopback {
            None
        } else {
            auth_header.map(str::to_string)
        },
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
    // Bracketed IPv6 authority: `[host]` or `[host]:port`. Must be handled
    // before the `rsplit_once(':')` fallback, which would otherwise split
    // *inside* the address (e.g. `[::1]` → host `[:`, port `1]`) and fail,
    // making a portless `[::1]` look like a non-loopback host.
    if let Some(after_open) = authority.strip_prefix('[') {
        let (host, after_close) = after_open.split_once(']')?;
        if host.is_empty() {
            return None;
        }
        let port = match after_close.strip_prefix(':') {
            Some(port_str) => port_str.parse::<u16>().ok()?,
            None if after_close.is_empty() => default_port,
            None => return None,
        };
        return Some((host.to_string(), port));
    }
    match authority.rsplit_once(':') {
        Some((host, port_str)) if !host.is_empty() => {
            let port = port_str.parse::<u16>().ok()?;
            Some((host.to_string(), port))
        }
        _ => Some((authority.to_string(), default_port)),
    }
}

/// True when `host` is a loopback address.
///
/// Expects a bare host as returned by [`parse_host_port`], which strips the
/// brackets from a bracketed IPv6 authority. Matches `localhost`
/// (case-insensitive), the IPv6 loopback (`::1`), and any `127.0.0.0/8` IPv4
/// address.
fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost") || host == "::1" || host.starts_with("127.")
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

/// Probe `candidates` (in priority order) concurrently and return the
/// highest-priority one that answered.
///
/// Split out from [`detect_local_endpoint`] so priority-order behavior is
/// testable against OS-assigned ephemeral ports instead of the real
/// well-known ports, which may be occupied by an unrelated ambient listener
/// on the machine running the test. Probing all candidates in parallel
/// (rather than sequentially) bounds the worst case (no server listening) to
/// one [`PROBE_TIMEOUT`] instead of one per candidate. TCP-only (no HTTP);
/// never blocks longer than [`PROBE_TIMEOUT`] total.
fn probe_first_reachable<'a>(candidates: &[&'a str]) -> Option<&'a str> {
    std::thread::scope(|scope| {
        // The collect is load-bearing, not needless: every probe must be
        // *spawned* before any is *joined*, or they run one at a time and the
        // whole point (bounding total latency to one PROBE_TIMEOUT) is lost.
        #[allow(clippy::needless_collect)]
        let handles: Vec<_> = candidates
            .iter()
            .map(|ep| (*ep, scope.spawn(|| probe_endpoint(ep, PROBE_TIMEOUT))))
            .collect();
        handles
            .into_iter()
            .find_map(|(ep, handle)| handle.join().unwrap_or(false).then_some(ep))
    })
}

/// Probe the known local serving endpoints (Lemonade, vLLM, then `rocm
/// serve`'s non-managed default) concurrently and return the highest-priority
/// one that answered.
///
/// Used by the TUI's in-app "detect a local engine" action, and by chat
/// startup, so the user need not run the CLI skill first.
pub fn detect_local_endpoint() -> Option<&'static str> {
    const CANDIDATES: [&str; 3] = [
        crate::skills::LEMONADE_ENDPOINT,
        crate::skills::VLLM_ENDPOINT,
        crate::skills::ROCM_SERVE_ENDPOINT,
    ];
    probe_first_reachable(&CANDIDATES)
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
        // Regression guard for the 404 bug: Rig appends `chat/completions` to
        // `base_url`, so the probed default must resolve to a `/v1` base or every
        // completion POSTs to `/chat/completions` and the server returns 404.
        // Asserting the *resolved* value (not the constant) fails on a revert of
        // the suffix rather than being tautological.
        assert!(
            r.base_url.ends_with("/v1"),
            "probed default base_url must end in /v1, got {}",
            r.base_url
        );
        assert_eq!(r.model, DEFAULT_CHAT_MODEL);
        assert_eq!(r.api_key, None);
    }

    #[test]
    fn loopback_base_url_strips_cloud_auth() {
        for url in [
            "http://127.0.0.1:8000/v1",
            "http://localhost:13305/v1",
            "http://[::1]:8000/v1",
            // Portless bracketed IPv6 previously bypassed stripping (parse
            // returned None → treated as remote). Must strip now.
            "http://[::1]/v1",
        ] {
            let r = resolve_llm_config(
                Some(url),
                None,
                None,
                None,
                Some("leaked-key"),
                None,
                Some("Ocp-Apim-Subscription-Key"),
                true,
            )
            .expect("config");
            assert_eq!(r.base_url, url);
            assert_eq!(r.api_key, None, "loopback strips api_key ({url})");
            assert_eq!(r.auth_header, None, "loopback strips auth_header ({url})");
        }
    }

    #[test]
    fn remote_base_url_keeps_cloud_auth() {
        let r = resolve_llm_config(
            Some("https://gw.example.com/openai"),
            None,
            None,
            None,
            Some("k"),
            None,
            Some("Ocp-Apim-Subscription-Key"),
            true,
        )
        .expect("config");
        assert_eq!(r.api_key.as_deref(), Some("k"));
        assert_eq!(r.auth_header.as_deref(), Some("Ocp-Apim-Subscription-Key"));
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
    fn parse_host_port_handles_bracketed_ipv6() {
        // Brackets are stripped so the host round-trips through `to_socket_addrs`
        // and matches `is_loopback_host`. Both the port-bearing and the portless
        // forms must parse (the portless form previously returned None).
        assert_eq!(
            parse_host_port("http://[::1]:8000/v1"),
            Some(("::1".into(), 8000))
        );
        assert_eq!(parse_host_port("http://[::1]/v1"), Some(("::1".into(), 80)));
        assert_eq!(
            parse_host_port("https://[2001:db8::1]/v1"),
            Some(("2001:db8::1".into(), 443))
        );
        assert_eq!(parse_host_port("http://[]/v1"), None);
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

    /// Binds an OS-assigned ephemeral port and returns its listener (kept
    /// alive so the port stays reachable) plus its `http://host:port` URL.
    ///
    /// Ephemeral ports are used instead of the real well-known ports
    /// (Lemonade/vLLM/rocm serve) so priority-order tests are deterministic:
    /// an ambient listener elsewhere on the machine (e.g. a real local
    /// engine, or — on some CI runners — a pre-installed one) can otherwise
    /// make well-known-port-based tests flake by answering a probe the test
    /// didn't intend to exercise.
    fn ephemeral_endpoint() -> (std::net::TcpListener, String) {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind ephemeral port");
        let url = format!("http://{}", listener.local_addr().expect("local_addr"));
        (listener, url)
    }

    #[test]
    fn probe_first_reachable_prefers_earlier_candidate_when_multiple_reachable() {
        // Two reachable candidates: priority order (first in the list) must
        // win even though probes run concurrently, not sequentially.
        let (_first, first_url) = ephemeral_endpoint();
        let (_second, second_url) = ephemeral_endpoint();
        assert_eq!(
            probe_first_reachable(&[first_url.as_str(), second_url.as_str()]),
            Some(first_url.as_str())
        );
    }

    #[test]
    fn probe_first_reachable_skips_unreachable_earlier_candidates() {
        // The first candidate (port 1) is essentially never open; the probe
        // must fall through to the second, reachable one.
        let (_listener, url) = ephemeral_endpoint();
        assert_eq!(
            probe_first_reachable(&["http://127.0.0.1:1", url.as_str()]),
            Some(url.as_str())
        );
    }

    #[test]
    fn probe_first_reachable_none_when_nothing_listening() {
        assert_eq!(probe_first_reachable(&["http://127.0.0.1:1"]), None);
    }

    #[test]
    fn detect_local_endpoint_bounded_latency_when_nothing_listening() {
        // Regression guard for the startup-latency fix: with no server
        // reachable, probing all 3 well-known ports must take roughly one
        // PROBE_TIMEOUT (parallel), not three (sequential). Generous margin
        // to avoid flaking on loaded CI machines.
        let start = std::time::Instant::now();
        let _ = detect_local_endpoint();
        assert!(
            start.elapsed() < PROBE_TIMEOUT * 2,
            "expected parallel probing to bound latency to ~1 PROBE_TIMEOUT, took {:?}",
            start.elapsed()
        );
    }
}
