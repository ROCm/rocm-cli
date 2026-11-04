// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The live chat backend the tick loop owns: which [`AgentClient`] answers the
//! next chat turn, and the snapshot of the auto-detected local one that
//! `/provider local` restores. Split out of `app/event_loop.rs`, where the two
//! were loop locals threaded through the startup block and three drains.
//!
//! [`ChatRuntime::start`] resolves the backend once before the first frame, and
//! one `drain_*` method per one-shot edge on `AppState` (`provider_switch`,
//! `chat_endpoint_rebuild`, `chat_dispatch`) consumes that edge. The detect and
//! persist edges do not touch the backend and stay in `event_loop`.

use tokio::sync::mpsc;

use crate::agent::AgentClient;
use crate::client::ClientMsg;

use super::chat::{
    self, StartupChatOutcome, build_chat_agent, build_local_agent, detect_local_chat,
    discover_configured_chat_model, startup_chat_outcome,
};
use super::focus::should_skip_daemon;
use super::types::{ChatProvider, ChatTurn, NO_CHAT_BACKEND_MSG, ResolvedArgs};
use super::{AppState, ProviderSwitch};

type SharedAgent = std::sync::Arc<dyn AgentClient>;

pub(super) struct ChatRuntime {
    /// The backend the next submit is routed to; `None` when none could be
    /// built, in which case a submit surfaces an error turn.
    agent: Option<SharedAgent>,
    /// The auto-detected local backend, kept so `/provider local` can restore it
    /// after a switch to a remote provider. Without this, switching to OpenAI
    /// and back to local would leave `agent` pointing at the OpenAI backend
    /// (silent wrong-backend bug) — `build_chat_agent(Local)` returns `None` by
    /// design (Local is the backend built at startup), so the caller must restore the
    /// saved clone here. `Option<Arc<…>>` clone is a cheap Arc refcount bump.
    local_agent: Option<SharedAgent>,
}

impl ChatRuntime {
    /// Resolve the chat backend. `--chat-mock` short-circuits detection with a
    /// deterministic offline MockAgentClient (no live LLM, no network); otherwise
    /// we auto-detect the endpoint (the std-TCP probe runs once on a blocking
    /// thread before the first frame) and build the Rig backend.
    pub(super) async fn start(
        state: &mut AppState,
        args: &ResolvedArgs,
        chat_tx: &mpsc::UnboundedSender<ClientMsg>,
    ) -> Self {
        let agent: Option<SharedAgent> = if should_skip_daemon(args.focus) {
            // Focused host: Setup/Serve/Diagnose never chat. Skip endpoint detection
            // and backend construction entirely — no probe, no network, no OAuth
            // default. `chat_llm` stays `None` and the Chat tab is never drawn here.
            None
        } else if args.chat_mock {
            state.set_chat_config(
                Some(crate::llm::LlmConfig {
                    base_url: "mock://offline-demo".to_string(),
                    model: "mock-agent".to_string(),
                    api_key: None,
                    auth_header: None,
                }),
                true,
            );
            Some(std::sync::Arc::new(
                crate::agent::MockAgentClient::with_tool_call_and_approval_trigger(
                    "GPU-2 is running hot: 87% util, 71°C, drawing 250 W (90 GB/192 GB VRAM).",
                    "gpu_status",
                    "install the sdk",
                    crate::tool_exec::ApprovalIntent {
                        title: "Install TheRock ROCm SDK?".to_string(),
                        body: vec![
                            "install_sdk --channel release --format wheel --prefix ~/rocm-sdk"
                                .to_string(),
                        ],
                        name: "install_sdk".to_string(),
                        arguments: serde_json::json!({
                            "channel": "release",
                            "format": "wheel",
                            "prefix": "~/rocm-sdk",
                        }),
                    },
                    chat_tx.clone(),
                ),
            ) as SharedAgent)
        } else {
            // An endpoint we launched ourselves (managed-services registry) takes
            // priority over the well-known default port — this is how a tool-launched
            // engine on a non-default port (e.g. vLLM on :11435) is found. It does
            // NOT override an explicitly configured `chat_url`/env URL, so config
            // precedence is preserved (we only consult the registry when neither is
            // set, i.e. where the well-known default would otherwise be probed).
            // When neither an explicit URL (CLI/config) nor an env URL is set, run
            // the SAME full local-engine detection the manual 'd' path uses:
            // registry-first (an engine we launched ourselves, on whatever port it
            // bound), then a probe of the well-known Lemonade/vLLM/rocm-serve
            // ports (parallelized — see `llm::detect_local_endpoint` — so a cold
            // start with no server doesn't pay 3x the probe timeout), plus a
            // best-effort served-model fetch. This is what lets a local server win
            // over the ChatGPT cloud default at startup instead of only the single
            // well-known :8000 port that a bare `resolve_llm_config` probe covers.
            //
            // Gate on `chat_api_key.is_none()` too: local detection returns a
            // keyless `detected_llm_config` (api_key/auth_header forced to None),
            // so firing it when the user configured a key would SILENTLY DROP that
            // key and 401 at request time. A configured key means "use my
            // configured backend", so skip the swap and let `resolve_llm_config`
            // carry the key through its normal precedence.
            let detection_ran = chat::should_detect_local_chat(
                args.chat_url.as_deref(),
                args.chat_env_url.as_deref(),
                args.chat_api_key.as_deref(),
            );
            let detected = if detection_ran {
                detect_local_chat(state.tool_executor.clone()).await
            } else {
                None
            };
            let probe_target = args
                .chat_url
                .clone()
                .or_else(|| args.chat_env_url.clone())
                .unwrap_or_else(|| crate::llm::DEFAULT_CHAT_BASE_URL.to_string());
            // A detected endpoint (managed or probed) is already verified. When
            // detection ran and found nothing it already probed the well-known
            // vLLM :8000 port (== `DEFAULT_CHAT_BASE_URL`), so re-probing the same
            // fallback target here is redundant and just burns another probe
            // timeout on a cold start — treat that as unreachable directly.
            // Otherwise (an explicit URL/env/key path) TCP-probe the target.
            let startup_outcome = startup_chat_outcome(detection_ran, detected.is_some());
            let probe_ok = match startup_outcome {
                StartupChatOutcome::Local => true,
                StartupChatOutcome::OAuth => false,
                StartupChatOutcome::Configured => tokio::task::spawn_blocking(move || {
                    crate::llm::probe_endpoint(&probe_target, crate::llm::PROBE_TIMEOUT)
                })
                .await
                .unwrap_or(false),
            };
            let llm = detected.or_else(|| {
                crate::llm::resolve_llm_config(
                    args.chat_url.as_deref(),
                    args.chat_model.as_deref(),
                    None,
                    None,
                    args.chat_api_key.as_deref(),
                    args.chat_env_url.as_deref(),
                    args.chat_auth_header.as_deref(),
                    probe_ok,
                )
            });
            // A *configured* URL (CLI/env) with no explicit model resolves to the
            // `local-model` placeholder, which 404s on servers that register the
            // model under its real id. Only the `Configured` outcome needs this — the `Local` outcome already
            // carries a `/v1/models`-discovered model from `detect_local_chat`, and
            // `OAuth` has no config. Discovery is gated inside the helper on
            // `probe_ok` (an unreachable endpoint is never probed nor replaced) and
            // on the absence of an explicit model (config precedence wins).
            let llm = match llm {
                Some(cfg) if startup_outcome == StartupChatOutcome::Configured => Some(
                    discover_configured_chat_model(cfg, args.chat_model.as_deref(), probe_ok).await,
                ),
                other => other,
            };
            state.set_chat_config(llm, args.chat_auto_consent);
            // No reachable local endpoint AND no key/url configured → the no-key
            // ChatGPT OAuth default (device-code login surfaced in the chat tab).
            // This restores the no-key login the vendored Codex path provided; it
            // takes NO api_key (env-only invariant untouched — OAuth, not a key).
            let no_key_no_endpoint = startup_outcome == StartupChatOutcome::OAuth;
            if no_key_no_endpoint {
                let oauth_tx = chat_tx.clone();
                crate::agent::ChatGptAgentClient::new(
                    args.chat_model.clone(),
                    args.inference_params(),
                    move |url, code| {
                        let _ = oauth_tx.send(ClientMsg::ChatReply {
                            text: format!(
                                "To enable chat, sign in to ChatGPT: open {url} and enter the code {code}"
                            ),
                        });
                    },
                    state.tool_executor.clone(),
                    Some(chat_tx.clone()),
                )
                .ok()
                .map(|c| c.with_preamble(args.chat_system_prompt.clone()))
                .map(|c| std::sync::Arc::new(c) as SharedAgent)
            } else {
                // A build failure leaves `agent` None; a submit surfaces an error turn.
                match &state.chat_llm {
                    Some(cfg) => build_local_agent(
                        cfg.clone(),
                        args.inference_params(),
                        state.tool_executor.clone(),
                        chat_tx.clone(),
                        args.chat_system_prompt.clone(),
                    )
                    .ok(),
                    None => None,
                }
            }
        };

        let local_agent = agent.clone();
        Self { agent, local_agent }
    }

    /// Drain a `/provider` switch (Phase 8). Rebuild the live `agent` for the
    /// newly-selected backend. `Local` restores the backend `start` saved
    /// (it owns the auto-detect probe). `Openai`/`Anthropic` are built
    /// from `ResolvedArgs` keys (in-process seam, never argv). A build failure
    /// (e.g. missing key) leaves `agent` unchanged and surfaces an actionable
    /// error turn. Construction only — no network until the next submit.
    pub(super) fn drain_provider_switch(
        &mut self,
        state: &mut AppState,
        args: &ResolvedArgs,
        chat_tx: &mpsc::UnboundedSender<ClientMsg>,
    ) {
        if let Some(ProviderSwitch { previous, target }) = state.provider_switch.take() {
            match target {
                ChatProvider::Local => {
                    // Restore the auto-detected local backend saved in
                    // `ChatRuntime::start`. `build_chat_agent(Local)` returns None by
                    // design, so the restore must happen here — otherwise a prior
                    // `/provider openai` would leave requests routed to OpenAI.
                    self.agent = self.local_agent.clone();
                    state
                        .chat
                        .push(ChatTurn::system("switched to local".to_string()));
                }
                ChatProvider::Openai | ChatProvider::Anthropic => {
                    if let Some(new_agent) =
                        build_chat_agent(target, args, state.tool_executor.clone(), chat_tx.clone())
                    {
                        self.agent = Some(new_agent);
                        state
                            .chat
                            .push(ChatTurn::system(format!("switched to {}", target.label())));
                    } else {
                        // Revert the optimistic `active_provider` set by the slash
                        // handler back to the provider active BEFORE the switch
                        // attempt — not unconditionally Local — so the displayed
                        // provider stays honest (e.g. a failed openai→anthropic
                        // switch stays on openai). `agent` is never reassigned on a
                        // failed build, so it already matches `previous`; the two
                        // stay consistent (no stale-remote routing under a wrong
                        // label).
                        state.active_provider = previous;
                        let hint = if target == ChatProvider::Anthropic {
                            "anthropic requires ANTHROPIC_API_KEY in env or secure store"
                        } else {
                            "openai requires OPENAI_API_KEY in the environment"
                        };
                        state.chat.push(ChatTurn::error(format!(
                            "could not switch to {}: {hint}",
                            target.label()
                        )));
                    }
                }
            }
        }
    }

    /// Drain the endpoint-rebuild edge (Phase 8 sibling). An accepted
    /// detected-local offer must re-point the LIVE `agent` — and the
    /// `/provider local` restore snapshot — at the new local backend.
    /// `accept_detect_offer` swaps `chat_llm` to the auth-free local config
    /// but stays I/O-free, so without this the stale startup agent keeps
    /// routing chat to the cloud gateway (wrong-backend 401 bug). The edge
    /// carries the provider active BEFORE the optimistic switch to `Local`;
    /// on failure we revert `active_provider` to it (mirrors the
    /// `provider_switch` drain) so the tab never shows `Local` while `agent`
    /// still points elsewhere. Construction only — no network until submit.
    pub(super) fn drain_endpoint_rebuild(
        &mut self,
        state: &mut AppState,
        args: &ResolvedArgs,
        chat_tx: &mpsc::UnboundedSender<ClientMsg>,
    ) {
        if let Some(previous) = state.chat_endpoint_rebuild.take() {
            // `revert` restores the optimistic switch and surfaces an actionable
            // error turn so the tab does not sit on `Local` with the old agent.
            let revert = |state: &mut AppState, msg: String| {
                state.active_provider = previous;
                state.chat.push(ChatTurn::error(msg));
            };
            match state.chat_llm.clone() {
                Some(cfg) => {
                    match build_local_agent(
                        cfg,
                        args.inference_params(),
                        state.tool_executor.clone(),
                        chat_tx.clone(),
                        args.chat_system_prompt.clone(),
                    ) {
                        Ok(arc) => {
                            self.agent = Some(arc.clone());
                            // Refresh the restore snapshot so a later `/provider
                            // local` restores THIS accepted backend, not the
                            // stale startup one.
                            self.local_agent = Some(arc);
                            state
                                .chat
                                .push(ChatTurn::system("switched to local".to_string()));
                        }
                        Err(e) => revert(
                            state,
                            format!("could not switch to the detected local endpoint: {e}"),
                        ),
                    }
                }
                // Edge raised but `chat_llm` is None (shouldn't happen after a
                // real accept, but don't leave the tab stuck on `Local` with the
                // old agent and no feedback).
                None => revert(
                    state,
                    "could not switch to the detected local endpoint: no endpoint configured"
                        .to_string(),
                ),
            }
        }
    }

    /// Spawn the agent round-trip on the submit edge — keeps `apply_action`
    /// I/O-free. `chat_dispatch` is raised once by `submit_chat`; consume it
    /// so the in-flight request is spawned exactly once (not every tick).
    pub(super) fn drain_dispatch(
        &self,
        state: &mut AppState,
        chat_tx: &mpsc::UnboundedSender<ClientMsg>,
    ) {
        if state.chat_dispatch {
            state.chat_dispatch = false;
            match self.agent.clone() {
                Some(agent) => {
                    let history = state.chat.clone();
                    let snapshot = state.state_snapshot();
                    let reply_tx = chat_tx.clone();
                    tokio::spawn(async move {
                        let msg = match agent.complete(&history, snapshot).await {
                            Ok(text) => ClientMsg::ChatReply { text },
                            Err(e) => ClientMsg::ChatError {
                                message: e.to_string(),
                            },
                        };
                        let _ = reply_tx.send(msg);
                    });
                }
                None => state.on_chat_error(NO_CHAT_BACKEND_MSG.to_string()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::super::test_support::{args_with_anthropic_key, st};
    use super::super::types::Focus;
    use super::*;
    use crate::agent::MockAgentClient;
    use crate::app::{ChatConsent, ChatRole};

    fn mock(reply: &str) -> SharedAgent {
        Arc::new(MockAgentClient::new(reply))
    }

    fn is(slot: Option<&SharedAgent>, agent: &SharedAgent) -> bool {
        slot.is_some_and(|a| Arc::ptr_eq(a, agent))
    }

    fn channel() -> (
        mpsc::UnboundedSender<ClientMsg>,
        mpsc::UnboundedReceiver<ClientMsg>,
    ) {
        mpsc::unbounded_channel()
    }

    fn last_turn(state: &AppState) -> &ChatTurn {
        state.chat.last().expect("a transcript turn was pushed")
    }

    async fn next_msg(rx: &mut mpsc::UnboundedReceiver<ClientMsg>) -> ClientMsg {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("a chat message arrives within 5 s")
            .expect("the channel stays open")
    }

    // --- start ---------------------------------------------------------------

    #[tokio::test]
    async fn a_focused_host_builds_no_backend_and_leaves_chat_unconfigured() {
        let mut s = st();
        let (tx, _rx) = channel();
        let mut args = args_with_anthropic_key(None);
        args.focus = Some(Focus::Examine);
        // `chat_mock` would build an agent if the focus gate were skipped.
        args.chat_mock = true;

        let rt = ChatRuntime::start(&mut s, &args, &tx).await;

        assert!(rt.agent.is_none() && rt.local_agent.is_none());
        assert!(s.chat_llm.is_none());
        assert_eq!(s.chat_consent, ChatConsent::Unavailable);
    }

    #[tokio::test]
    async fn chat_mock_builds_the_offline_agent_with_consent_and_saves_it_as_the_local_one() {
        let mut s = st();
        let (tx, mut rx) = channel();
        let mut args = args_with_anthropic_key(None);
        args.chat_mock = true;

        let rt = ChatRuntime::start(&mut s, &args, &tx).await;

        let agent = rt.agent.clone().expect("the mock backend is built");
        assert!(
            is(rt.local_agent.as_ref(), &agent),
            "`/provider local` restores it"
        );
        let cfg = s.chat_llm.as_ref().expect("the mock config is installed");
        assert_eq!(cfg.base_url, "mock://offline-demo");
        assert_eq!(cfg.model, "mock-agent");
        assert_eq!(s.chat_consent, ChatConsent::Accepted);

        let reply = agent
            .complete(&[ChatTurn::user("hello")], s.state_snapshot())
            .await
            .unwrap();
        assert!(reply.contains("GPU-2 is running hot"), "got: {reply}");

        // The approval trigger is wired to the loop's chat channel.
        agent
            .complete(
                &[ChatTurn::user("Please install the SDK")],
                s.state_snapshot(),
            )
            .await
            .unwrap();
        match rx.try_recv() {
            Ok(ClientMsg::ChatApprovalRequired { intent }) => {
                assert_eq!(intent.name, "install_sdk");
            }
            _ => panic!("the approval request must reach the loop's chat channel"),
        }
    }

    #[tokio::test]
    async fn an_explicit_endpoint_is_used_without_local_detection() {
        // An unreachable loopback URL: the probe is refused at once, so nothing
        // is detected, and the configured URL is carried through as the backend.
        let mut s = st();
        let (tx, _rx) = channel();
        let mut args = args_with_anthropic_key(None);
        args.chat_url = Some("http://127.0.0.1:1/v1".to_string());
        args.chat_model = Some("my-model".to_string());

        let rt = ChatRuntime::start(&mut s, &args, &tx).await;

        let cfg = s
            .chat_llm
            .as_ref()
            .expect("the configured endpoint is kept");
        assert_eq!(cfg.base_url, "http://127.0.0.1:1/v1");
        assert_eq!(cfg.model, "my-model");
        assert_eq!(s.chat_consent, ChatConsent::Pending, "no --chat-yes");
        let agent = rt.agent.clone().expect("a backend is built for it");
        assert!(is(rt.local_agent.as_ref(), &agent));

        args.chat_auto_consent = true;
        let mut s2 = st();
        let _ = ChatRuntime::start(&mut s2, &args, &tx).await;
        assert_eq!(s2.chat_consent, ChatConsent::Accepted, "--chat-yes");
    }

    // --- drain_provider_switch -----------------------------------------------

    #[test]
    fn switching_to_a_keyed_remote_provider_replaces_only_the_live_agent() {
        let (tx, _rx) = channel();
        let local = mock("local");
        let mut rt = ChatRuntime {
            agent: Some(local.clone()),
            local_agent: Some(local.clone()),
        };
        let mut s = st();
        s.provider_switch = Some(ProviderSwitch {
            previous: ChatProvider::Local,
            target: ChatProvider::Anthropic,
        });
        let args = args_with_anthropic_key(Some("dummy-anthropic-key"));

        rt.drain_provider_switch(&mut s, &args, &tx);

        assert!(s.provider_switch.is_none(), "the edge is consumed");
        assert!(!is(rt.agent.as_ref(), &local), "the live agent was rebuilt");
        assert!(rt.agent.is_some());
        assert!(
            is(rt.local_agent.as_ref(), &local),
            "the restore snapshot is kept"
        );
        assert_eq!(last_turn(&s), &ChatTurn::system("switched to anthropic"));
    }

    #[test]
    fn a_failed_switch_keeps_the_agent_and_reverts_to_the_previous_provider() {
        let (tx, _rx) = channel();
        let current = mock("openai-backed");
        let mut rt = ChatRuntime {
            agent: Some(current.clone()),
            local_agent: None,
        };
        let mut s = st();
        // The slash handler optimistically set Anthropic; Openai was active.
        s.active_provider = ChatProvider::Anthropic;
        s.provider_switch = Some(ProviderSwitch {
            previous: ChatProvider::Openai,
            target: ChatProvider::Anthropic,
        });
        let args = args_with_anthropic_key(None);

        rt.drain_provider_switch(&mut s, &args, &tx);

        assert!(
            is(rt.agent.as_ref(), &current),
            "no stale-remote re-routing"
        );
        assert_eq!(
            s.active_provider,
            ChatProvider::Openai,
            "not forced to Local"
        );
        assert_eq!(last_turn(&s).role, ChatRole::Error);
        assert_eq!(
            last_turn(&s).content,
            "could not switch to anthropic: anthropic requires ANTHROPIC_API_KEY in env or secure store"
        );

        // The OpenAI hint names its own variable.
        s.provider_switch = Some(ProviderSwitch {
            previous: ChatProvider::Local,
            target: ChatProvider::Openai,
        });
        rt.drain_provider_switch(&mut s, &args, &tx);
        assert_eq!(
            last_turn(&s).content,
            "could not switch to openai: openai requires OPENAI_API_KEY in the environment"
        );
    }

    #[test]
    fn switching_back_to_local_restores_the_saved_local_agent() {
        let (tx, _rx) = channel();
        let local = mock("local");
        let remote = mock("remote");
        let mut rt = ChatRuntime {
            agent: Some(remote),
            local_agent: Some(local.clone()),
        };
        let mut s = st();
        s.provider_switch = Some(ProviderSwitch {
            previous: ChatProvider::Openai,
            target: ChatProvider::Local,
        });
        let args = args_with_anthropic_key(None);

        rt.drain_provider_switch(&mut s, &args, &tx);

        assert!(is(rt.agent.as_ref(), &local));
        assert_eq!(last_turn(&s), &ChatTurn::system("switched to local"));
    }

    #[test]
    fn a_provider_drain_with_no_switch_pending_changes_nothing() {
        let (tx, _rx) = channel();
        let local = mock("local");
        let mut rt = ChatRuntime {
            agent: Some(local.clone()),
            local_agent: Some(local.clone()),
        };
        let mut s = st();
        rt.drain_provider_switch(&mut s, &args_with_anthropic_key(None), &tx);
        assert!(is(rt.agent.as_ref(), &local));
        assert!(s.chat.is_empty());
    }

    // --- drain_endpoint_rebuild ----------------------------------------------

    #[test]
    fn an_accepted_local_endpoint_repoints_the_agent_and_the_restore_snapshot() {
        let (tx, _rx) = channel();
        let stale = mock("cloud");
        let mut rt = ChatRuntime {
            agent: Some(stale.clone()),
            local_agent: Some(stale.clone()),
        };
        let mut s = st();
        s.chat_llm = Some(crate::llm::LlmConfig {
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            model: "m".to_string(),
            api_key: None,
            auth_header: None,
        });
        s.chat_endpoint_rebuild = Some(ChatProvider::Openai);
        s.active_provider = ChatProvider::Local;
        let args = args_with_anthropic_key(None);

        rt.drain_endpoint_rebuild(&mut s, &args, &tx);

        assert!(s.chat_endpoint_rebuild.is_none(), "the edge is consumed");
        let rebuilt = rt.agent.clone().expect("a new local backend is built");
        assert!(
            !Arc::ptr_eq(&rebuilt, &stale),
            "not the stale startup agent"
        );
        assert!(
            is(rt.local_agent.as_ref(), &rebuilt),
            "snapshot refreshed too"
        );
        assert_eq!(
            s.active_provider,
            ChatProvider::Local,
            "no revert on success"
        );
        assert_eq!(last_turn(&s), &ChatTurn::system("switched to local"));
    }

    #[test]
    fn a_rebuild_with_no_endpoint_reverts_the_provider_and_keeps_the_agent() {
        let (tx, _rx) = channel();
        let stale = mock("cloud");
        let mut rt = ChatRuntime {
            agent: Some(stale.clone()),
            local_agent: Some(stale.clone()),
        };
        let mut s = st();
        s.chat_llm = None;
        s.chat_endpoint_rebuild = Some(ChatProvider::Anthropic);
        s.active_provider = ChatProvider::Local; // the optimistic switch
        let args = args_with_anthropic_key(None);

        rt.drain_endpoint_rebuild(&mut s, &args, &tx);

        assert!(s.chat_endpoint_rebuild.is_none());
        assert_eq!(s.active_provider, ChatProvider::Anthropic);
        assert!(is(rt.agent.as_ref(), &stale) && is(rt.local_agent.as_ref(), &stale));
        assert_eq!(last_turn(&s).role, ChatRole::Error);
        assert_eq!(
            last_turn(&s).content,
            "could not switch to the detected local endpoint: no endpoint configured"
        );
    }

    // --- drain_dispatch ------------------------------------------------------

    #[tokio::test]
    async fn a_submit_spawns_exactly_one_round_trip_and_posts_the_reply() {
        let (tx, mut rx) = channel();
        let rt = ChatRuntime {
            agent: Some(mock("pong")),
            local_agent: None,
        };
        let mut s = st();
        s.chat.push(ChatTurn::user("ping"));
        s.chat_dispatch = true;

        rt.drain_dispatch(&mut s, &tx);

        assert!(!s.chat_dispatch, "the edge is consumed");
        match next_msg(&mut rx).await {
            ClientMsg::ChatReply { text } => assert_eq!(text, "pong"),
            _ => panic!("expected the agent's reply"),
        }
        // No edge raised: a second drain must not spawn another request.
        rt.drain_dispatch(&mut s, &tx);
        assert!(
            tokio::time::timeout(Duration::from_millis(150), rx.recv())
                .await
                .is_err(),
            "no second round-trip"
        );
    }

    #[tokio::test]
    async fn a_failing_agent_posts_an_error_message() {
        let (tx, mut rx) = channel();
        let rt = ChatRuntime {
            agent: Some(Arc::new(MockAgentClient::failing())),
            local_agent: None,
        };
        let mut s = st();
        s.chat.push(ChatTurn::user("ping"));
        s.chat_dispatch = true;

        rt.drain_dispatch(&mut s, &tx);

        match next_msg(&mut rx).await {
            ClientMsg::ChatError { message } => {
                assert!(message.contains("mock failure"), "got: {message}");
            }
            _ => panic!("expected an error message"),
        }
    }

    #[tokio::test]
    async fn a_submit_with_no_backend_surfaces_the_no_backend_error_turn() {
        let (tx, mut rx) = channel();
        let rt = ChatRuntime {
            agent: None,
            local_agent: None,
        };
        let mut s = st();
        s.chat.push(ChatTurn::user("ping"));
        s.chat_sending = true;
        s.chat_dispatch = true;

        rt.drain_dispatch(&mut s, &tx);

        assert!(!s.chat_dispatch);
        assert!(!s.chat_sending, "the pending indicator is cleared");
        assert_eq!(last_turn(&s), &ChatTurn::error(NO_CHAT_BACKEND_MSG));
        assert!(rx.try_recv().is_err(), "nothing was spawned");
    }
}
