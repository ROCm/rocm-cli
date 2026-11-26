// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Chat-input and chat-consent state: accepting or declining the endpoint
//! prompt, the local-endpoint detection offer, submitting a turn, and applying
//! the agent's reply or error.
//!
//! A further `impl AppState` block, split out of `app/mod.rs` to keep the core
//! reducer focused. Distinct from `app/chat.rs`, which holds the chat-config
//! and agent-factory free functions this state calls into.

use rocm_dash_core::metrics::InstanceStatus;

use super::{AppState, ChatConsent, ChatEndpointHealth, ChatProvider, ChatTurn, SlashOutcome};

/// Shown instead of dispatching a message the backing endpoint cannot answer.
/// Names both recovery paths, since neither is discoverable from the tab.
pub(crate) const ENDPOINT_DOWN_MSG: &str = "the model backing this chat is no longer being served — start it again, or use \
     `/detect` to pick up another local engine (`/provider` switches to a cloud backend)";

impl AppState {
    /// The loopback port of the chat endpoint, when it is one a managed
    /// instance could be serving.
    ///
    /// `None` for a remote endpoint, and for any endpoint reached through a
    /// provider other than `Local` — `/provider anthropic` leaves `chat_llm`
    /// pointing at the local URL it replaced, so the provider has to be checked
    /// too or a stopped local service would block a cloud chat that never
    /// touched it.
    fn chat_endpoint_port(&self) -> Option<u16> {
        if self.active_provider != ChatProvider::Local {
            return None;
        }
        let base_url = &self.chat_llm.as_ref()?.base_url;
        // Reuse the crate's authority parser: it strips the brackets from an
        // IPv6 loopback (`http://[::1]:8000` → host `::1`) that a hand-rolled
        // `rsplit(':')` would mangle, and defaults the port from the scheme.
        let (host, port) = crate::llm::parse_host_port(base_url)?;
        crate::llm::is_loopback_host(&host).then_some(port)
    }

    /// Whether the instance backing the chat endpoint can answer a request.
    ///
    /// I/O-free, so the reducer and the renderer can share one liveness rule
    /// rather than each carrying its own.
    pub fn chat_endpoint_health(&self) -> ChatEndpointHealth {
        let Some(port) = self.chat_endpoint_port() else {
            return ChatEndpointHealth::Serving;
        };
        match self.instances.values().find(|i| i.port == Some(port)) {
            Some(inst) => match inst.status {
                InstanceStatus::Ready | InstanceStatus::Running | InstanceStatus::Unknown => {
                    ChatEndpointHealth::Serving
                }
                InstanceStatus::Starting { .. } => ChatEndpointHealth::Starting,
                InstanceStatus::Stopped => ChatEndpointHealth::Stopped,
                InstanceStatus::Error => ChatEndpointHealth::Failed,
            },
            // Nothing on that port. That is only a verdict for an endpoint we
            // once saw registered; otherwise it is an engine we never tracked.
            None if self.chat_endpoint_managed => ChatEndpointHealth::Stopped,
            None => ChatEndpointHealth::Serving,
        }
    }

    /// Record that the chat endpoint's port is served by a managed instance, so
    /// a later disappearance reads as "stopped" rather than "never watched".
    ///
    /// Called wherever `instances` changes. Latches on rather than tracking the
    /// instance both ways: the absence this exists to interpret is exactly the
    /// state that would otherwise clear it.
    pub(crate) fn refresh_chat_endpoint_health(&mut self) {
        let Some(port) = self.chat_endpoint_port() else {
            return;
        };
        if self.instances.values().any(|i| i.port == Some(port)) {
            self.chat_endpoint_managed = true;
        }
    }

    /// Accept the detected endpoint and enable chat. No-op when no endpoint is
    /// available. Focuses the input so the user can type immediately.
    pub const fn accept_chat_consent(&mut self) {
        if self.chat_llm.is_some() {
            self.chat_consent = ChatConsent::Accepted;
            self.chat_focused = true;
        }
    }

    /// Decline the detected endpoint. Chat stays off (re-enable with `y`).
    pub const fn decline_chat_consent(&mut self) {
        if self.chat_llm.is_some() {
            self.chat_consent = ChatConsent::Declined;
            self.chat_focused = false;
        }
    }

    /// Request an in-TUI local-engine probe. Raises the one-shot
    /// `chat_detect_dispatch` edge so `event_loop` runs the probe + `/v1/models`
    /// query off the reducer. No-op while a probe is already in flight or an
    /// offer is awaiting a decision. I/O-free.
    ///
    /// Reachable both from the pre-accept gate (`'d'` key) and, once
    /// `ChatConsent::Accepted`, from the `/detect` slash command (a focused
    /// `'d'` keypress is ordinary chat text at that point, so the gate key
    /// doesn't apply there — see `handle_slash_command`). When already
    /// accepted, echo the in-flight probe into the transcript since the
    /// pre-accept "detecting…" banner isn't drawn once chat is live.
    pub fn request_detect(&mut self) {
        if self.chat_detecting || self.chat_detect_offer.is_some() {
            return;
        }
        self.chat_detecting = true;
        self.chat_detect_msg = None;
        self.chat_detect_dispatch = true;
        if self.chat_consent == ChatConsent::Accepted {
            self.chat.push(ChatTurn::agent(
                "Detecting a local engine (Lemonade :13305 / vLLM :8000 / a rocm serve service)…"
                    .to_string(),
            ));
        }
    }

    /// Record the result of a detect attempt: `Some(cfg)` raises the offer
    /// prompt; `None` records a "nothing found" message. Clears the in-flight
    /// flag either way.
    ///
    /// Once `ChatConsent::Accepted`, the offer prompt is not drawn (the gate UI
    /// only renders pre-accept), so the result is also echoed into the
    /// transcript with the `/detect accept|save|dismiss` sub-commands needed to
    /// act on it.
    pub fn set_detect_result(&mut self, offer: Option<crate::llm::LlmConfig>) {
        self.chat_detecting = false;
        let accepted = self.chat_consent == ChatConsent::Accepted;
        if let Some(cfg) = offer {
            self.chat_detect_msg = None;
            if accepted {
                self.chat.push(ChatTurn::agent(format!(
                    "Detected a local engine: {}  (model: {}). Type `/detect accept` to \
                     switch now, `/detect save` to also persist it, or `/detect dismiss` \
                     to ignore.",
                    cfg.base_url, cfg.model
                )));
            }
            self.chat_detect_offer = Some(cfg);
        } else {
            self.chat_detect_offer = None;
            let msg = "no local engine found (Lemonade :13305 / vLLM :8000 / a rocm serve service)";
            if accepted {
                self.chat.push(ChatTurn::agent(msg.to_string()));
            } else {
                self.chat_detect_msg = Some(msg.to_string());
            }
        }
    }

    /// Accept the detected local endpoint for this session.
    ///
    /// Switches `chat_llm` to the offer and enables chat. The accepted endpoint
    /// is local, so this also selects the Local provider and raises the
    /// `chat_endpoint_rebuild` edge, marking the live agent for rebuild in
    /// `event_loop`. No-op when no offer is pending.
    pub fn accept_detect_offer(&mut self) {
        if let Some(cfg) = self.chat_detect_offer.take() {
            self.chat_llm = Some(cfg);
            self.chat_consent = ChatConsent::Accepted;
            self.chat_focused = true;
            // A new endpoint has its own liveness history; carrying the old
            // latch over would report the new one stopped before it is seen.
            self.chat_endpoint_managed = false;
            // The accepted endpoint is local; align the displayed provider and
            // raise the rebuild edge so `event_loop` swaps the live agent to it
            // (the startup agent may be a cloud gateway — see
            // `chat_endpoint_rebuild`). Capture the previous provider first so
            // the drain can revert the optimistic switch if the rebuild fails.
            let previous = self.active_provider;
            self.active_provider = ChatProvider::Local;
            self.chat_endpoint_rebuild = Some(previous);
        }
    }

    /// Dismiss the detected-endpoint offer, leaving the prior chat config and
    /// consent untouched.
    pub fn dismiss_detect_offer(&mut self) {
        self.chat_detect_offer = None;
    }

    /// Accept the detected endpoint **and** persist it.
    ///
    /// Same as [`accept_detect_offer`](Self::accept_detect_offer) (which selects
    /// the Local provider and raises the endpoint-rebuild edge), then raise the
    /// one-shot `chat_persist_dispatch` edge so `event_loop` writes
    /// `tui.chat_url`/`tui.chat_model` to the config file. No-op when no offer
    /// is pending.
    pub fn save_detect_offer(&mut self) {
        let had_offer = self.chat_detect_offer.is_some();
        self.accept_detect_offer();
        if had_offer {
            self.chat_persist_dispatch = true;
        }
    }

    /// Submit the current chat input. Empty / whitespace-only input is
    /// ignored. Pushes the user turn, marks the request in-flight, and raises
    /// the one-shot `chat_dispatch` edge so `event_loop` spawns the agent
    /// round-trip. Stays I/O-free (the spawn happens outside the reducer).
    pub fn submit_chat(&mut self) {
        // Ignore submits while a request is in flight — prevents a double-Enter
        // from spawning two racing agent tasks with desynced history.
        if self.chat_sending {
            return;
        }
        let text = self.chat_input.trim().to_string();
        if text.is_empty() {
            return;
        }
        // Slash commands are handled locally (nav/overlays/read-only tools) and
        // never reach the LLM. A `/`-prefixed line is ALWAYS consumed here, even
        // when unknown (it gets an error turn) — only non-slash text is sent on.
        if self.handle_slash_command(&text) == SlashOutcome::Handled {
            self.chat_input.clear();
            return;
        }
        // Refuse a message the endpoint is known not to answer. Dispatching it
        // only buys a raw transport error in the transcript; this says what
        // happened and how to get back. Deliberately below the slash branch —
        // `/detect` and `/provider` are the ways out, so they must still run.
        if self.chat_endpoint_health().blocks_send() {
            self.chat_input.clear();
            self.chat
                .push(ChatTurn::error(ENDPOINT_DOWN_MSG.to_string()));
            return;
        }
        self.chat.push(ChatTurn::user(text));
        self.chat_input.clear();
        self.chat_sending = true;
        self.chat_dispatch = true;
    }

    /// Capture a read-only telemetry snapshot for the chat tools. Plain owned
    /// clones — tools read this without touching the reducer or `&AppState`.
    pub fn state_snapshot(&self) -> crate::agent::StateSnapshot {
        crate::agent::StateSnapshot {
            latest: self.latest.clone(),
            instances: self.instances.values().cloned().collect(),
            bench_rows: self.bench_rows.iter().cloned().collect(),
        }
    }

    /// Handle a successful agent reply: append an `Agent` turn, clear the
    /// in-flight flag. Called from `event_loop` on `ClientMsg::ChatReply`.
    pub fn on_chat_reply(&mut self, text: String) {
        self.chat.push(ChatTurn::agent(text));
        self.chat_sending = false;
    }

    /// Handle an agent failure: append an `Error` turn, clear the in-flight
    /// flag. Called on `ClientMsg::ChatError` — never panics.
    pub fn on_chat_error(&mut self, message: String) {
        self.chat.push(ChatTurn::error(message));
        self.chat_sending = false;
    }
}

#[cfg(test)]
mod tests {
    use rocm_dash_core::metrics::{Instance, InstanceStatus};
    use rocm_dash_core::protocol::Event;

    use super::ENDPOINT_DOWN_MSG;
    use crate::app::test_support::st;
    use crate::app::types::NO_CHAT_BACKEND_MSG;
    use crate::app::{AppState, ChatConsent, ChatEndpointHealth, ChatProvider, ChatRole, ChatTurn};

    const SERVED_PORT: u16 = 8000;

    /// A chat accepted against a model served on `SERVED_PORT`, with the
    /// dashboard already having seen the instance serving it.
    ///
    /// Drives the real event path (`apply_event`) rather than writing
    /// `instances` directly — the wiring between an instance arriving and the
    /// chat endpoint noticing is exactly what is under test.
    fn chatting_with_a_served_model() -> AppState {
        let mut s = st();
        s.set_chat_config(
            Some(crate::llm::LlmConfig {
                base_url: format!("http://127.0.0.1:{SERVED_PORT}/v1"),
                model: "m".into(),
                api_key: None,
                auth_header: None,
            }),
            true,
        );
        s.apply_event(Event::InstanceDiscovered(served_instance()));
        assert_eq!(
            s.chat_endpoint_health(),
            ChatEndpointHealth::Serving,
            "precondition: the model is being served"
        );
        s
    }

    fn served_instance() -> Instance {
        Instance {
            container_id: "svc".into(),
            port: Some(SERVED_PORT),
            status: InstanceStatus::Ready,
            ..Default::default()
        }
    }

    /// Stopping a service removes its record, so the dashboard sees the
    /// instance vanish rather than turn `Stopped`.
    fn stop_serving(s: &mut AppState) {
        s.apply_event(Event::InstanceGone {
            container_id: "svc".into(),
        });
    }

    /// Scenario 1 — the chat tab reflects that the model is gone, and the
    /// earlier conversation survives.
    #[test]
    fn stopping_the_served_model_marks_the_chat_endpoint_down() {
        let mut s = chatting_with_a_served_model();
        s.chat.push(ChatTurn::user("earlier question"));
        s.chat.push(ChatTurn::agent("earlier answer"));

        stop_serving(&mut s);

        assert_eq!(s.chat_endpoint_health(), ChatEndpointHealth::Stopped);
        assert!(
            s.chat_endpoint_health().blocks_send(),
            "a stopped endpoint must not accept a message"
        );
        assert_eq!(s.chat.len(), 2, "the transcript is kept, not discarded");
    }

    /// Scenario 2 — sending to a stopped model explains the problem instead of
    /// dispatching a request that can only come back as a transport error.
    #[test]
    fn sending_to_a_stopped_model_explains_instead_of_dispatching() {
        let mut s = chatting_with_a_served_model();
        stop_serving(&mut s);

        s.chat_input = "are you there?".into();
        s.submit_chat();

        assert!(
            !s.chat_dispatch,
            "no request may be dispatched to a stopped endpoint"
        );
        assert!(!s.chat_sending, "and nothing may be left spinning");
        let last = s.chat.last().expect("a turn explaining the refusal");
        assert_eq!(last.role, ChatRole::Error);
        assert_eq!(last.content, ENDPOINT_DOWN_MSG);
        assert!(
            last.content.contains("/detect"),
            "the refusal must name a way back, got: {}",
            last.content
        );
    }

    /// Scenario 3 — the recovery commands are the way out of the stopped
    /// state, so they must still run while it holds.
    #[test]
    fn slash_commands_still_run_while_the_model_is_stopped() {
        let mut s = chatting_with_a_served_model();
        stop_serving(&mut s);

        s.chat_input = "/detect".into();
        s.submit_chat();

        assert!(
            s.chat_detect_dispatch,
            "/detect must still reach the detect flow when the endpoint is down"
        );
        assert!(
            s.chat.iter().all(|t| t.content != ENDPOINT_DOWN_MSG),
            "a slash command must not be refused as an unanswerable message"
        );
    }

    /// Scenario 4 — serving the model again restores chat without the user
    /// reconfiguring anything. A restarted service gets a fresh id, so this
    /// also pins that recovery is matched by port rather than by instance id.
    #[test]
    fn serving_the_model_again_restores_chat() {
        let mut s = chatting_with_a_served_model();
        stop_serving(&mut s);
        assert!(s.chat_endpoint_health().blocks_send());

        s.apply_event(Event::InstanceDiscovered(Instance {
            container_id: "svc-restarted".into(),
            ..served_instance()
        }));

        assert_eq!(s.chat_endpoint_health(), ChatEndpointHealth::Serving);
        s.chat_input = "back?".into();
        s.submit_chat();
        assert!(s.chat_dispatch, "chat accepts messages again");
    }

    /// Scenario 5 — a cloud provider is reached through its own backend, and
    /// `chat_llm` still holds the local URL it replaced. A locally served model
    /// stopping must not block a chat that never touched it.
    #[test]
    fn a_stopped_local_model_does_not_block_a_remote_provider() {
        let mut s = chatting_with_a_served_model();
        s.active_provider = ChatProvider::Anthropic;

        stop_serving(&mut s);

        assert_eq!(s.chat_endpoint_health(), ChatEndpointHealth::Serving);
        s.chat_input = "still fine".into();
        s.submit_chat();
        assert!(
            s.chat_dispatch,
            "the remote provider still accepts messages"
        );
    }

    /// A local engine the dashboard never saw registered (started by hand,
    /// found by `/detect`) has no instance to watch. Its absence from the map
    /// is the normal state, and must never be read as "stopped".
    #[test]
    fn an_engine_that_was_never_registered_is_not_reported_stopped() {
        let mut s = st();
        s.set_chat_config(
            Some(crate::llm::LlmConfig {
                base_url: format!("http://127.0.0.1:{SERVED_PORT}/v1"),
                model: "m".into(),
                api_key: None,
                auth_header: None,
            }),
            true,
        );

        assert_eq!(s.chat_endpoint_health(), ChatEndpointHealth::Serving);
        s.chat_input = "hello".into();
        s.submit_chat();
        assert!(s.chat_dispatch);
    }

    /// Pointing the chat at a different endpoint starts its liveness history
    /// over. Carrying the old one across would report a perfectly good new
    /// endpoint stopped before anything had been seen on it.
    #[test]
    fn switching_endpoints_does_not_inherit_the_old_ones_liveness() {
        let mut s = chatting_with_a_served_model();
        stop_serving(&mut s);
        assert!(s.chat_endpoint_health().blocks_send());

        s.set_chat_config(
            Some(crate::llm::LlmConfig {
                base_url: format!("http://127.0.0.1:{}/v1", SERVED_PORT + 7),
                model: "m".into(),
                api_key: None,
                auth_header: None,
            }),
            true,
        );

        assert_eq!(s.chat_endpoint_health(), ChatEndpointHealth::Serving);
    }

    /// An instance on a different port is not this chat's endpoint, so its
    /// coming and going says nothing about whether the model is served.
    #[test]
    fn an_instance_on_another_port_is_not_this_chat_endpoint() {
        let mut s = chatting_with_a_served_model();
        s.apply_event(Event::InstanceDiscovered(Instance {
            container_id: "other".into(),
            port: Some(SERVED_PORT + 1),
            status: InstanceStatus::Ready,
            ..Default::default()
        }));
        stop_serving(&mut s);

        assert_eq!(
            s.chat_endpoint_health(),
            ChatEndpointHealth::Stopped,
            "an unrelated instance must not stand in for the stopped one"
        );
    }

    #[test]
    fn detect_none_sets_message() {
        let mut s = st();
        s.request_detect();
        s.set_detect_result(None);
        assert!(!s.chat_detecting);
        assert!(s.chat_detect_offer.is_none());
        assert!(s.chat_detect_msg.is_some());
    }

    #[test]
    fn no_provider_no_key_chat_surfaces_actionable_message() {
        // Edge: agent is None (no endpoint, no provider key). Submitting chat
        // must surface a clear, ACTIONABLE message (the recovery affordances),
        // routed through `on_chat_error` as an error turn — not an error dump,
        // not a panic. This mirrors the event-loop None-agent branch, which
        // emits exactly `NO_CHAT_BACKEND_MSG`.
        let mut s = st();
        s.set_chat_config(None, false);
        assert_eq!(s.chat_consent, ChatConsent::Unavailable);
        // Drive the same surface the event loop uses for the None-agent case.
        s.on_chat_error(NO_CHAT_BACKEND_MSG.to_string());
        let last = s.chat.last().expect("an error turn was pushed");
        assert_eq!(last.role, ChatRole::Error);
        // Actionable: names both concrete recovery paths.
        assert!(
            last.content.contains("detect") && last.content.contains("/provider"),
            "empty-state must be actionable, got: {}",
            last.content
        );
        // Not a panic and not in-flight afterwards (sending cleared).
        assert!(!s.chat_sending);
    }
}
