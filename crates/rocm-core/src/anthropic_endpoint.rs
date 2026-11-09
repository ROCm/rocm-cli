// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The Anthropic-compatible `/v1/messages` surface of a managed service.
//!
//! An engine speaks the OpenAI API. A service additionally gets an Anthropic
//! Messages endpoint because `rocm` fronts the engine with a translation proxy
//! ([`AnthropicEndpointMode::Proxy`]), even when the engine has a `/v1/messages`
//! route of its own: an engine's route only accepts the model name it serves,
//! while an Anthropic client names a Claude model. The process that runs the
//! engine publishes the proxy's address under the state-file keys below, and [`crate::ManagedServiceRecord::refresh_from_engine_state`] adopts
//! it, so every reader (`rocm serve`, `rocm services list`, the readiness probe)
//! sees the same endpoint.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::openai_models_endpoint_has_model;

/// Engine state-file key naming the Anthropic-compatible endpoint's base URL.
pub const ANTHROPIC_ENDPOINT_URL_STATE_KEY: &str = "anthropic_endpoint_url";

/// Engine state-file key recording how the Anthropic endpoint is provided.
pub const ANTHROPIC_ENDPOINT_MODE_STATE_KEY: &str = "anthropic_endpoint_mode";

/// How a service provides its Anthropic-compatible endpoint.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AnthropicEndpointMode {
    /// A `rocm`-hosted proxy translates Anthropic requests to the engine's
    /// OpenAI API, on its own port.
    Proxy,
}

impl AnthropicEndpointMode {
    /// The value stored under [`ANTHROPIC_ENDPOINT_MODE_STATE_KEY`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Proxy => "proxy",
        }
    }
}

/// Whether the Anthropic endpoint at `anthropic_url` is up and fronts `model`.
///
/// The proxy answers `GET /v1/models` by forwarding to the engine. So the same cheap listing check that gates
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
        assert_eq!(AnthropicEndpointMode::Proxy.as_str(), "proxy");
    }
}
