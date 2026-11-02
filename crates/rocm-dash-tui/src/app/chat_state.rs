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

use super::{AppState, ChatConsent, ChatProvider, ChatTurn, SlashOutcome};

impl AppState {
    /// Install the resolved chat endpoint and set the initial consent state.
    /// `None` → `Unavailable`; `Some` → `Accepted` when pre-consented (e.g.
    /// `--chat-yes`), otherwise `Pending` (the one-time in-TUI prompt).
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
                "Detecting a local engine (Lemonade :13305 / vLLM :8000 / rocm serve :11435)…"
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
            let msg = "no local engine found (Lemonade :13305 / vLLM :8000 / rocm serve :11435)";
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
    use crate::app::test_support::st;
    use crate::app::types::NO_CHAT_BACKEND_MSG;
    use crate::app::{ChatConsent, ChatRole};

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
