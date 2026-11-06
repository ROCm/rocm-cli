// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The Anthropic-compatible `/v1/messages` surface of a managed service.
//!
//! An engine speaks the OpenAI API. A service additionally gets an Anthropic
//! Messages endpoint either because the engine serves `/v1/messages` itself
//! ([`AnthropicEndpointMode::Native`]) or because `rocm` fronts the engine with
//! a translation proxy ([`AnthropicEndpointMode::Proxy`]). The process that runs
//! the engine decides which, publishes the answer under the state-file keys
//! below, and [`crate::ManagedServiceRecord::refresh_from_engine_state`] adopts
//! it, so every reader (`rocm serve`, `rocm services list`, the readiness probe)
//! sees the same endpoint.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::{http_post_json_with_auth, openai_models_endpoint_has_model};

/// Engine state-file key naming the Anthropic-compatible endpoint's base URL.
pub const ANTHROPIC_ENDPOINT_URL_STATE_KEY: &str = "anthropic_endpoint_url";

/// Engine state-file key recording how the Anthropic endpoint is provided.
pub const ANTHROPIC_ENDPOINT_MODE_STATE_KEY: &str = "anthropic_endpoint_mode";

/// How a service provides its Anthropic-compatible endpoint.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AnthropicEndpointMode {
    /// The engine serves `/v1/messages` itself, on the same port as its OpenAI API.
    Native,
    /// A `rocm`-hosted proxy translates Anthropic requests to the engine's
    /// OpenAI API, on its own port.
    Proxy,
}

impl AnthropicEndpointMode {
    /// The value stored under [`ANTHROPIC_ENDPOINT_MODE_STATE_KEY`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Proxy => "proxy",
        }
    }
}

/// Whether the server at `endpoint_url` already serves `/v1/messages`.
///
/// Asks with an empty, invalid request instead of a real one, so detection never
/// spends a generation. A server that has the route refuses the body (400/422,
/// or 401 without a key); one that lacks it answers 404 or 405. Only a
/// connection failure is an error — it says nothing about the route.
pub fn anthropic_messages_native_probe(
    endpoint_url: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<bool> {
    let response = http_post_json_with_auth(
        endpoint_url,
        "/v1/messages",
        &serde_json::json!({}),
        endpoint_api_key,
        timeout,
    )?;
    Ok(!matches!(response.status, 404 | 405))
}

/// Whether the Anthropic endpoint at `anthropic_url` is up and fronts `model`.
///
/// Both modes answer `GET /v1/models`: a native engine from its own listing, the
/// proxy by forwarding to the engine. So the same cheap listing check that gates
/// the OpenAI endpoint also covers this one, without queuing a generation.
pub fn anthropic_endpoint_ready(
    anthropic_url: &str,
    model: Option<&str>,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> bool {
    openai_models_endpoint_has_model(anthropic_url, model, endpoint_api_key, timeout)
        .unwrap_or(false)
}

/// The Anthropic endpoint the serving process has published in `state_path`,
/// waiting up to `wait` for it to appear.
///
/// The endpoint is published a moment after the engine first lists its model, so
/// a caller that has just seen the service become ready can still find nothing
/// there. `Duration::ZERO` reads once.
pub fn wait_for_published_anthropic_endpoint(state_path: &Path, wait: Duration) -> Option<String> {
    let started = Instant::now();
    loop {
        let published = std::fs::read(state_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|state| {
                state
                    .get(ANTHROPIC_ENDPOINT_URL_STATE_KEY)
                    .and_then(serde_json::Value::as_str)
                    .filter(|url| !url.trim().is_empty())
                    .map(str::to_owned)
            });
        if published.is_some() || started.elapsed() >= wait {
            return published;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Serve one canned HTTP response, then stop.
    fn one_shot_server(status_line: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buffer = [0_u8; 2048];
                let _ = stream.read(&mut buffer);
                let body = "{}";
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        format!("http://127.0.0.1:{port}/v1")
    }

    #[test]
    fn native_probe_treats_a_refused_empty_request_as_support() {
        let url = one_shot_server("400 Bad Request");
        assert!(anthropic_messages_native_probe(&url, None, Duration::from_secs(5)).unwrap());
    }

    #[test]
    fn native_probe_treats_a_missing_route_as_no_support() {
        for status in ["404 Not Found", "405 Method Not Allowed"] {
            let url = one_shot_server(status);
            assert!(
                !anthropic_messages_native_probe(&url, None, Duration::from_secs(5)).unwrap(),
                "{status}"
            );
        }
    }

    #[test]
    fn native_probe_reports_an_unreachable_server_as_an_error() {
        let port = {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let url = format!("http://127.0.0.1:{port}/v1");
        assert!(anthropic_messages_native_probe(&url, None, Duration::from_secs(1)).is_err());
    }

    #[test]
    fn published_endpoint_is_read_from_state_and_absent_when_unpublished() {
        let dir = std::env::temp_dir().join(format!(
            "rocm-core-anthropic-published-{}-{}",
            std::process::id(),
            crate::unix_time_millis()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let state = dir.join("state.json");
        assert_eq!(
            wait_for_published_anthropic_endpoint(&state, Duration::ZERO),
            None
        );
        std::fs::write(&state, r#"{"status": "ready"}"#).unwrap();
        assert_eq!(
            wait_for_published_anthropic_endpoint(&state, Duration::ZERO),
            None
        );
        std::fs::write(
            &state,
            r#"{"anthropic_endpoint_url": "http://127.0.0.1:41000"}"#,
        )
        .unwrap();
        assert_eq!(
            wait_for_published_anthropic_endpoint(&state, Duration::ZERO).as_deref(),
            Some("http://127.0.0.1:41000")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn mode_strings_are_the_values_published_in_state() {
        assert_eq!(AnthropicEndpointMode::Native.as_str(), "native");
        assert_eq!(AnthropicEndpointMode::Proxy.as_str(), "proxy");
    }
}
