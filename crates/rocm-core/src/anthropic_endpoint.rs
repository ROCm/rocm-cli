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
//!
//! The publisher is a *thread of the engine's own process*, not a separate
//! writer: `rocm`'s `run_builtin_engine_serve_http` starts the proxy and then
//! runs the engine adapter in the same process. Both write the one engine state
//! file, and the engine adapters write it whole. [`publish_anthropic_endpoint`]
//! and `carry_published_anthropic_endpoint` are the two halves that keep the
//! whole-file write from dropping what the sibling thread published.
//!
//! The carry reaches a whole-file write only when that write goes through
//! [`crate::write_json_state_file`], which is
//! `carry_published_anthropic_endpoint`'s single caller. All three built-in
//! adapters' `write_state` do (`engines/lemonade`, `engines/vllm`,
//! `engines/aim`). An out-of-tree engine plugin that writes the state file
//! itself takes neither this nor the state directory's lock, and would drop
//! the publication.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use anyhow::Result;

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

/// What *this process* has published, keyed by the engine state file it went
/// into, so [`carry_published_anthropic_endpoint`] can put it back.
///
/// Deliberately process-local rather than read back off disk. A previous run of
/// the same service id left its own endpoint in that file, and that URL is dead
/// — its proxy died with the process that bound the port. The first state write
/// of a new run is what clears it (`ManagedServiceRecord::reset_for_restart`
/// relies on exactly that), so a disk-backed carry-over would resurrect a dead
/// endpoint on every restart. Only the publication this process made is ours to
/// preserve.
///
/// Keyed by path rather than held as a single value because the key is per
/// service: nothing today runs two engines in one process, but a map costs
/// nothing and a global would quietly cross-contaminate if one ever did (and
/// does cross-contaminate between tests sharing a process, which is the same
/// bug in miniature).
static PUBLISHED_ENDPOINTS: OnceLock<Mutex<HashMap<PathBuf, (String, String)>>> = OnceLock::new();

fn published_endpoints() -> &'static Mutex<HashMap<PathBuf, (String, String)>> {
    PUBLISHED_ENDPOINTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record `base_url` as this service's Anthropic endpoint in `state_path`, and
/// remember it for the rest of this process.
///
/// `base_url` carries no `/v1`: Anthropic SDKs append `/v1/messages` to it.
///
/// The in-memory note is taken *before* the file is touched, and the note's
/// mutex is released before the file lock is asked for. Both matter:
///
/// * noting first leaves no window — a whole-file rewrite that gets the file
///   lock first already sees the note and carries the endpoint, and one that
///   gets it after the merge below is carrying it too;
/// * releasing first keeps the two locks from ever being held in both orders,
///   since `carry_published_anthropic_endpoint` is called with the file lock
///   held.
///
/// Called more than once for one endpoint when the merge has to be retried
/// (`rocmd`'s proxy retries a state file caught mid-write); the note is the
/// same each time.
pub fn publish_anthropic_endpoint(
    state_path: &Path,
    base_url: &str,
    mode: AnthropicEndpointMode,
) -> Result<()> {
    {
        let mut published = published_endpoints()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        published.insert(
            state_path.to_path_buf(),
            (base_url.to_owned(), mode.as_str().to_owned()),
        );
    }
    crate::merge_json_state_file(
        state_path,
        &serde_json::json!({
            ANTHROPIC_ENDPOINT_URL_STATE_KEY: base_url,
            ANTHROPIC_ENDPOINT_MODE_STATE_KEY: mode.as_str(),
        }),
    )
}

/// `state`, with the Anthropic endpoint this process published for `state_path`
/// put back into it — or `None` when there is nothing to put back, so the
/// caller writes the document it already has and pays no clone.
///
/// [`crate::write_json_state_file`] is the only caller. A full rewrite replaces
/// every key, by design: it serializes a document its caller built rather than
/// patching the file, which is how a new run of a service clears what the
/// previous run left. The one thing it must not replace is a publication made
/// by *its own process*, because the publisher is a thread of that process
/// (`rocm`'s `run_builtin_engine_serve_http` starts the Anthropic proxy and
/// then runs the engine adapter) and nothing publishes a second time.
///
/// This is not a theoretical race. The engine adapter's post-spawn `running`
/// write happens immediately after the server process is started, and since
/// ROCMAI-550 it spends a host GPU probe (`crate::pinned_gpu_name`: `pnputil`
/// plus a PowerShell CIM query on Windows, seconds) building the document
/// before writing it. That is long enough for the server to come up, for the
/// proxy's poll to see a listed model, and for the publication to land inside
/// the window and be replaced — leaving a ready service that never shows an
/// `anthropic endpoint:` row.
///
/// The cross-process lock [`crate::merge_json_state_file`] and
/// [`crate::write_json_state_file`] share does not cover this. It orders the
/// two writers so neither sees a half-written file; it does not stop the
/// later one replacing what the earlier one recorded, which is exactly what a
/// full rewrite is for.
pub(crate) fn carry_published_anthropic_endpoint(
    state_path: &Path,
    state: &serde_json::Value,
) -> Option<serde_json::Value> {
    let (url, mode) = published_endpoints()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(state_path)
        .cloned()?;
    let mut carried = state.clone();
    let object = carried.as_object_mut()?;
    object.insert(
        ANTHROPIC_ENDPOINT_URL_STATE_KEY.to_owned(),
        serde_json::Value::String(url),
    );
    object.insert(
        ANTHROPIC_ENDPOINT_MODE_STATE_KEY.to_owned(),
        serde_json::Value::String(mode),
    );
    Some(carried)
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

    /// The regression ROCMAI-550's Windows lane hit, driven through the real
    /// writers: publish the endpoint, then have the engine adapter rewrite the
    /// state file whole — which is what `write_running_state` does right after
    /// it spawns the server, with a multi-second host GPU probe inside it.
    ///
    /// The rewritten document is deliberately *not* a copy of what is on disk:
    /// a "fix" that merely re-read the file before writing would not be what
    /// the adapters do, and would not satisfy this.
    #[test]
    fn a_whole_file_state_write_keeps_the_endpoint_this_process_published() {
        let dir = unique_dir("carry");
        let state_path = dir.join("state.json");
        crate::write_json_state_file(&state_path, &serde_json::json!({ "status": "starting" }))
            .unwrap();
        publish_anthropic_endpoint(
            &state_path,
            "http://127.0.0.1:41000",
            AnthropicEndpointMode::Proxy,
        )
        .unwrap();

        crate::write_json_state_file(
            &state_path,
            &serde_json::json!({ "status": "running", "server_pid": 7 }),
        )
        .unwrap();

        let state: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&state_path).unwrap()).unwrap();
        assert_eq!(
            state[ANTHROPIC_ENDPOINT_URL_STATE_KEY], "http://127.0.0.1:41000",
            "an engine rewriting the whole state file must not drop the \
             endpoint its own process published: nothing publishes twice, so \
             the `anthropic endpoint:` row never comes back"
        );
        assert_eq!(state[ANTHROPIC_ENDPOINT_MODE_STATE_KEY], "proxy");
        // The rewrite still replaces everything else, which is what it is for.
        assert_eq!(state["status"], "running");
        assert_eq!(state["server_pid"], 7);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The carry is scoped to what *this* process published for *this* file.
    ///
    /// Both halves matter. A previous run of the same service left a dead
    /// endpoint in that file — its proxy died with the process that bound the
    /// port — and the new run's first whole-file write is what clears it, which
    /// `ManagedServiceRecord::reset_for_restart` relies on; carrying anything
    /// read off disk would resurrect it. And a process that published for one
    /// service must not hand that endpoint to another.
    #[test]
    fn a_rewrite_still_clears_an_endpoint_this_process_did_not_publish() {
        let dir = unique_dir("unpublished");
        let published_to = dir.join("published.json");
        let other = dir.join("other.json");
        publish_anthropic_endpoint(
            &published_to,
            "http://127.0.0.1:41001",
            AnthropicEndpointMode::Proxy,
        )
        .unwrap();
        std::fs::write(
            &other,
            r#"{"anthropic_endpoint_url": "http://127.0.0.1:1", "anthropic_endpoint_mode": "proxy"}"#,
        )
        .unwrap();

        crate::write_json_state_file(&other, &serde_json::json!({ "status": "starting" })).unwrap();

        let state: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&other).unwrap()).unwrap();
        assert_eq!(
            state,
            serde_json::json!({ "status": "starting" }),
            "only this process's own publication for this exact state file may \
             survive a whole-file write; a dead endpoint from a previous run \
             must not be restored"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Publishing still writes the keys through to the file, which is what
    /// every out-of-process reader (`rocm services list`, `rocm serve`'s
    /// summary) actually looks at — the note is an addition, not a substitute.
    #[test]
    fn publishing_writes_the_endpoint_into_the_state_file_it_names() {
        let dir = unique_dir("publish");
        let state_path = dir.join("state.json");
        std::fs::write(&state_path, r#"{"status": "running"}"#).unwrap();
        publish_anthropic_endpoint(
            &state_path,
            "http://127.0.0.1:41002",
            AnthropicEndpointMode::Proxy,
        )
        .unwrap();
        assert_eq!(
            wait_for_published_anthropic_endpoint(&state_path, Duration::ZERO).as_deref(),
            Some("http://127.0.0.1:41002")
        );
        let state: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&state_path).unwrap()).unwrap();
        assert_eq!(state[ANTHROPIC_ENDPOINT_MODE_STATE_KEY], "proxy");
        // The merge keeps what was already recorded for the service.
        assert_eq!(state["status"], "running");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A directory no other test in this process shares: the publication note
    /// is keyed by path and lives for the whole test binary.
    fn unique_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rocm-core-anthropic-{tag}-{}-{}",
            std::process::id(),
            crate::unix_time_millis()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
