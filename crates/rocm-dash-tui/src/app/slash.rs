// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Slash-command routing for the chat input.
//!
//! [`AppState::handle_slash_command`] classifies a `/`-prefixed chat line and
//! either mutates reducer state directly (nav / overlays), raises a one-shot
//! executor edge (`slash_tool` / `plan_request` / `provider_switch`) for the
//! event loop to drain off-thread, or pushes a usage/error turn. Stays I/O-free.
//! Split out of `app/mod.rs` to keep the core reducer focused. Of the
//! slash-command payload types it raises, `SlashOutcome` and
//! `SlashToolRequest` live in `types.rs`; `ProviderSwitch` stays in
//! `mod.rs`, alongside the `AppState` fields that carry them.

use super::{
    ActiveTab, AppState, ChatProvider, ChatTurn, Modal, ProviderSwitch, SlashOutcome,
    SlashToolRequest, request_quit,
};

impl AppState {
    /// Route a chat-input line that may be a slash command. Returns
    /// [`SlashOutcome::NotCommand`] for plain text (caller dispatches to the
    /// agent); otherwise handles it in-reducer and returns
    /// [`SlashOutcome::Handled`]. Stays I/O-free: executor-backed commands raise
    /// the `slash_tool` edge for the event loop to drain off-thread.
    pub(crate) fn handle_slash_command(&mut self, text: &str) -> SlashOutcome {
        /// Build a `rocm_command` slash-tool request from an argv slice and a
        /// chat-turn label. Centralizes the repeated `name: "rocm_command"` +
        /// `{"args": argv}` construction shared by the lifecycle read/mutate arms.
        fn rocm_cmd_request(argv: &[&str], label: impl Into<String>) -> SlashToolRequest {
            SlashToolRequest {
                name: "rocm_command".to_string(),
                args: serde_json::json!({ "args": argv }),
                label: label.into(),
            }
        }

        let trimmed = text.trim();
        let Some(rest) = trimmed.strip_prefix('/') else {
            return SlashOutcome::NotCommand;
        };
        // First whitespace-delimited word after '/', lowercased.
        let cmd = rest.split_whitespace().next().unwrap_or("").to_lowercase();

        match cmd.as_str() {
            // --- Group A: nav / session (deterministic, no executor) ---
            "home" => self.active_tab = ActiveTab::Home,
            "gpu" => self.active_tab = ActiveTab::Observe,
            "help" | "?" => self.modal = Modal::Help,
            "clear" => self.chat.clear(),
            // Shares `KeyAction::Quit`'s own gate (issue #145): a model still
            // being served gets the same confirm prompt the `q` key opens,
            // instead of a second, independently-decided quit path.
            "quit" | "exit" => self.should_quit = request_quit(self),
            // --- Group B: read-only overlays (mirror the keybind handlers) ---
            "doctor" => {
                self.close_overlays();
                self.examine_manager =
                    Some(crate::ui::examine_manager::ExamineManagerState::default());
            }
            "runtimes" => {
                self.close_overlays();
                self.runtime_manager =
                    Some(crate::ui::runtime_manager::RuntimeManagerState::default());
            }
            "config" => {
                self.close_overlays();
                self.config_manager =
                    Some(crate::ui::config_manager::ConfigManagerState::default());
            }
            "logs" => {
                self.close_overlays();
                self.logs_view = Some(crate::ui::logs_view::LogsViewState::default());
            }
            // --- Group B: read-only executor-backed (no overlay; off-thread) ---
            "model" => {
                self.slash_tool = Some(rocm_cmd_request(&["model"], "model"));
            }
            "daemon" => {
                self.slash_tool = Some(rocm_cmd_request(&["daemon", "status"], "daemon status"));
            }
            // --- Group D: mutating ops (approval-gated; surfaced via the modal) ---
            // Each raises a `slash_tool` request whose `execute()` returns
            // `ApprovalRequired`; the event loop opens the approval modal. The
            // safety validators run inside `execute()` (and again on the approved
            // replay), so an unsafe call surfaces an error instead of the modal.
            "install" => {
                // `/install <prefix>` — the install folder is required by the
                // validator (it never installs to a system path).
                match rest.split_whitespace().nth(1) {
                    Some(prefix) => {
                        self.slash_tool = Some(SlashToolRequest {
                            name: "install_sdk".to_string(),
                            args: serde_json::json!({
                                "channel": "release",
                                "format": "wheel",
                                "prefix": prefix,
                            }),
                            label: format!("install {prefix}"),
                        });
                    }
                    None => {
                        self.chat.push(ChatTurn::error(
                            "usage: /install <prefix>  (install folder, e.g. /install ~/rocm)"
                                .to_string(),
                        ));
                    }
                }
            }
            "engine" => {
                // `/engine <name>` — the engine name is required by the validator.
                match rest.split_whitespace().nth(1) {
                    Some(engine) => {
                        self.slash_tool = Some(SlashToolRequest {
                            name: "install_engine".to_string(),
                            args: serde_json::json!({ "engine": engine }),
                            label: format!("engine {engine}"),
                        });
                    }
                    None => {
                        self.chat.push(ChatTurn::error(
                            "usage: /engine <name> (e.g. /engine vllm)".to_string(),
                        ));
                    }
                }
            }
            "serve" => {
                // `/serve <model>` — loopback host only (validator rejects public).
                match rest.split_whitespace().nth(1) {
                    Some(model) => {
                        self.slash_tool = Some(SlashToolRequest {
                            name: "launch_server".to_string(),
                            args: serde_json::json!({ "model": model, "host": "127.0.0.1" }),
                            label: format!("serve {model}"),
                        });
                    }
                    None => {
                        self.chat.push(ChatTurn::error(
                            "usage: /serve <model> (e.g. /serve deepseek-r1)".to_string(),
                        ));
                    }
                }
            }
            "services" => {
                // `/services stop <id>` mutates (stop_server, approval-gated); a
                // bare `/services` is read-only and lists managed services.
                // `restart` is NOT yet wired through the chat seam, so it is
                // guided rather than silently running stop (a semantic lie).
                let mut words = rest.split_whitespace().skip(1);
                match words.next() {
                    Some("stop") => match words.next() {
                        Some(id) => {
                            self.slash_tool = Some(SlashToolRequest {
                                name: "stop_server".to_string(),
                                args: serde_json::json!({ "service_id": id }),
                                label: format!("services stop {id}"),
                            });
                        }
                        None => {
                            self.chat
                                .push(ChatTurn::error("usage: /services stop <id>".to_string()));
                        }
                    },
                    Some("restart") => {
                        self.chat.push(ChatTurn::error(
                            "services restart via chat is not supported yet; use /services stop <id> then /serve <model>"
                                .to_string(),
                        ));
                    }
                    Some(other) => {
                        self.chat.push(ChatTurn::error(format!(
                            "unknown /services action `{other}` (try stop, or /services to list)"
                        )));
                    }
                    None => {
                        self.slash_tool = Some(SlashToolRequest {
                            name: "services".to_string(),
                            args: serde_json::json!({}),
                            label: "services".to_string(),
                        });
                    }
                }
            }
            // --- Group D-rest: lifecycle ops (read/mutate split via rocm_command) ---
            // Read paths classify as ReadOnly in the bin and return a result turn;
            // mutating paths classify as ApprovalRequired and open the Phase-4 modal.
            "update" => {
                // `/update` reports available updates (read-only); `/update --apply`
                // applies them (approval-gated). Scan all tokens for the dash flag
                // (matching `/uninstall`) so the trigger isn't position-sensitive.
                let apply = rest.split_whitespace().skip(1).any(|tok| tok == "--apply");
                let argv: Vec<&str> = if apply {
                    vec!["update", "--apply"]
                } else {
                    vec!["update"]
                };
                self.slash_tool = Some(rocm_cmd_request(
                    &argv,
                    if apply { "update --apply" } else { "update" },
                ));
            }
            "comfyui" | "comfy" => {
                // Bare `/comfyui` is read-only status. status/logs read; install/
                // start/stop mutate (approval-gated).
                let sub = rest.split_whitespace().nth(1).map(str::to_lowercase);
                match sub.as_deref() {
                    None | Some("status") => {
                        self.slash_tool =
                            Some(rocm_cmd_request(&["comfyui", "status"], "comfyui status"));
                    }
                    Some(action @ ("logs" | "install" | "start" | "stop")) => {
                        self.slash_tool = Some(rocm_cmd_request(
                            &["comfyui", action],
                            format!("comfyui {action}"),
                        ));
                    }
                    Some(other) => {
                        self.chat.push(ChatTurn::error(format!(
                            "unknown /comfyui action `{other}` (try status, logs, install, start, stop)"
                        )));
                    }
                }
            }
            "uninstall" => {
                // SAFE default: bare `/uninstall` is a dry-run (read-only). A real
                // uninstall needs `/uninstall --apply` and is approval-gated (the
                // bin auto-adds --yes on approval).
                let flags: Vec<&str> = rest.split_whitespace().skip(1).collect();
                let saw_apply = flags.contains(&"--apply");
                let saw_dry_run = flags.contains(&"--dry-run");
                if saw_apply && saw_dry_run {
                    self.chat.push(ChatTurn::error(
                        "conflicting /uninstall flags: choose either --dry-run (safe) or --apply (real uninstall)"
                            .to_string(),
                    ));
                } else {
                    let real = saw_apply;
                    let argv: Vec<&str> = if real {
                        vec!["uninstall"]
                    } else {
                        vec!["uninstall", "--dry-run"]
                    };
                    self.slash_tool = Some(rocm_cmd_request(
                        &argv,
                        if real {
                            "uninstall"
                        } else {
                            "uninstall --dry-run"
                        },
                    ));
                }
            }
            "setup" => {
                // Bare `/setup` (or `/setup status`) reports first-time setup
                // (read-only); `/setup reset` clears the completion/dismissal
                // state (approval-gated) but does not reopen onboarding by
                // itself — that still needs the explicit `n` key on Observe.
                // The CLI only has status + reset — anything else is guided,
                // not run.
                let sub = rest.split_whitespace().nth(1).map(str::to_lowercase);
                match sub.as_deref() {
                    None | Some("status") => {
                        self.slash_tool =
                            Some(rocm_cmd_request(&["setup", "status"], "setup status"));
                    }
                    Some("reset") => {
                        self.slash_tool =
                            Some(rocm_cmd_request(&["setup", "reset"], "setup reset"));
                    }
                    Some(other) => {
                        self.chat.push(ChatTurn::error(format!(
                            "unknown /setup action `{other}` (try status or reset)"
                        )));
                    }
                }
            }
            // --- Group C: automations / reviews (read list; toggles + proposal
            // actions are approval-gated via the Phase-4 modal) ---
            "automations" => {
                // Bare `/automations` (or `list`) lists configured automations
                // (read-only). `enable`/`disable` toggle a watcher (approval-gated
                // via the watcher_enable/watcher_disable mutating tools).
                let mut words = rest.split_whitespace().skip(1);
                match words.next().map(str::to_lowercase).as_deref() {
                    None | Some("list") => {
                        self.slash_tool =
                            Some(rocm_cmd_request(&["automations", "list"], "automations"));
                    }
                    Some("enable") => match words.next() {
                        Some(watcher) => {
                            // Optional `--mode <m>`: when present, include it so the
                            // validator can accept observe|propose|contained.
                            let mode = match words.next() {
                                Some("--mode") => words.next(),
                                _ => None,
                            };
                            let args = match mode {
                                Some(m) => {
                                    serde_json::json!({ "watcher": watcher, "mode": m })
                                }
                                None => serde_json::json!({ "watcher": watcher }),
                            };
                            self.slash_tool = Some(SlashToolRequest {
                                name: "watcher_enable".to_string(),
                                args,
                                label: format!("automations enable {watcher}"),
                            });
                        }
                        None => {
                            self.chat.push(ChatTurn::error(
                                "usage: /automations enable <watcher> [--mode observe|propose|contained]"
                                    .to_string(),
                            ));
                        }
                    },
                    Some("disable") => match words.next() {
                        Some(watcher) => {
                            self.slash_tool = Some(SlashToolRequest {
                                name: "watcher_disable".to_string(),
                                args: serde_json::json!({ "watcher": watcher }),
                                label: format!("automations disable {watcher}"),
                            });
                        }
                        None => {
                            self.chat.push(ChatTurn::error(
                                "usage: /automations disable <watcher>".to_string(),
                            ));
                        }
                    },
                    Some(other) => {
                        self.chat.push(ChatTurn::error(format!(
                            "unknown /automations action `{other}` (try list, enable, disable)"
                        )));
                    }
                }
            }
            "reviews" => {
                // Bare `/reviews` lists pending reviews (read-only, via the
                // automations list). `/reviews <id>` shows one proposal's detail.
                match rest.split_whitespace().nth(1) {
                    None => {
                        self.slash_tool =
                            Some(rocm_cmd_request(&["automations", "list"], "reviews"));
                    }
                    Some(id) => {
                        self.slash_tool = Some(SlashToolRequest {
                            name: "proposal_action".to_string(),
                            args: serde_json::json!({ "proposal_id": id, "action": "show" }),
                            label: format!("reviews {id}"),
                        });
                    }
                }
            }
            "approve" => match rest.split_whitespace().nth(1) {
                Some(id) => {
                    self.slash_tool = Some(SlashToolRequest {
                        name: "proposal_action".to_string(),
                        args: serde_json::json!({ "proposal_id": id, "action": "approve" }),
                        label: format!("approve {id}"),
                    });
                }
                None => {
                    self.chat
                        .push(ChatTurn::error("usage: /approve <proposal-id>".to_string()));
                }
            },
            "reject" => match rest.split_whitespace().nth(1) {
                Some(id) => {
                    self.slash_tool = Some(SlashToolRequest {
                        name: "proposal_action".to_string(),
                        args: serde_json::json!({ "proposal_id": id, "action": "reject" }),
                        label: format!("reject {id}"),
                    });
                }
                None => {
                    self.chat
                        .push(ChatTurn::error("usage: /reject <proposal-id>".to_string()));
                }
            },
            "edit" => match rest.split_whitespace().nth(1) {
                Some(id) => {
                    // Editing a proposal's CONTENT isn't supported by the bin; show
                    // the proposal (read) so the operator can /approve or /reject.
                    self.slash_tool = Some(SlashToolRequest {
                        name: "proposal_action".to_string(),
                        args: serde_json::json!({ "proposal_id": id, "action": "show" }),
                        label: format!("review {id}"),
                    });
                    self.chat.push(ChatTurn::agent(format!(
                        "Editing a proposal's content isn't supported; showing {id}. Use /approve {id} or /reject {id}."
                    )));
                }
                None => {
                    self.chat
                        .push(ChatTurn::error("usage: /edit <proposal-id>".to_string()));
                }
            },
            // --- Group E: permissions (read status; escalation is approval-gated) ---
            "permissions" => {
                // Bare `/permissions` (or `status`) shows the current mode
                // (read-only via `config show`). `full-access`/`ask` change the
                // mode — escalation MUST route through the approval modal.
                let sub = rest.split_whitespace().nth(1).map(str::to_lowercase);
                match sub.as_deref() {
                    None | Some("status") => {
                        self.slash_tool =
                            Some(rocm_cmd_request(&["config", "show"], "permissions"));
                    }
                    Some("full-access" | "full_access") => {
                        self.slash_tool = Some(rocm_cmd_request(
                            &["config", "set-permissions", "full_access"],
                            "permissions full-access",
                        ));
                    }
                    Some("ask") => {
                        self.slash_tool = Some(rocm_cmd_request(
                            &["config", "set-permissions", "ask"],
                            "permissions ask",
                        ));
                    }
                    Some(other) => {
                        self.chat.push(ChatTurn::error(format!(
                            "unknown /permissions action `{other}` (try status, full-access, ask)"
                        )));
                    }
                }
            }
            // --- Group F: natural-language planner (Ask→Plan→Review→Run) ---
            // `/plan <request>` raises the `plan_request` edge; the event loop
            // calls the read-only `natural_language_plan` tool off-thread and
            // posts `PlanReady`. The plan is rendered for review; a complete
            // mutating action is then handed to the Phase-4 approval modal.
            "plan" => {
                // Split off the command word (`plan`, any case) and take the
                // free-form tail; case-insensitive, unlike `strip_prefix`.
                let request = rest
                    .split_once(char::is_whitespace)
                    .map(|(_, tail)| tail.trim())
                    .unwrap_or_default();
                if request.is_empty() {
                    self.chat.push(ChatTurn::agent(
                        "usage: /plan <request> (e.g. /plan install rocm into /opt/rocm)"
                            .to_string(),
                    ));
                } else {
                    self.plan_request = Some(request.to_string());
                }
            }
            // --- Group G: provider switch + chat entry (Phase 8) ---
            // `/provider [local|openai|anthropic]` switches the live chat
            // backend. A bare/unknown arg shows the current provider (or hints);
            // a valid one raises the `provider_switch` edge for the event loop to
            // rebuild the agent. Every backend calls the SAME ROCm tools.
            "provider" => match rest.split_whitespace().nth(1) {
                None => {
                    self.chat.push(ChatTurn::agent(format!(
                        "current provider: {} (usage: /provider [local|openai|anthropic])",
                        self.active_provider.label()
                    )));
                }
                Some(arg) => match ChatProvider::parse(arg) {
                    Some(p) => {
                        // Snapshot the prior provider BEFORE the optimistic set so
                        // a failed switch (missing key) can revert to it.
                        let previous = self.active_provider;
                        self.active_provider = p;
                        self.provider_switch = Some(ProviderSwitch {
                            previous,
                            target: p,
                        });
                    }
                    None => {
                        self.chat.push(ChatTurn::error(format!(
                            "unknown provider `{arg}` (try local, openai, or anthropic)"
                        )));
                    }
                },
            },
            // `/detect [accept|save|dismiss]` (EAI-7354): re-run local-engine
            // detection mid-session without colliding with focused text entry
            // (a bare `d` keypress while chat is focused is ordinary chat text,
            // and the pre-accept `'d'` gate key only exists before
            // `ChatConsent::Accepted`). A bare `/detect` raises the probe; the
            // result is echoed into the transcript (see `set_detect_result`)
            // with these same sub-commands to act on it. `accept`/`save` route
            // through the same reducer methods the pre-accept offer buttons
            // use, so a re-detect+accept mid-session raises the
            // `chat_endpoint_rebuild` edge exactly like the initial accept.
            "detect" => {
                // Lowercase the sub-command to match the case-insensitive
                // parsing of sibling commands (`/permissions`, `/provider`).
                let sub = rest.split_whitespace().nth(1).map(str::to_lowercase);
                match sub.as_deref() {
                    None => self.request_detect(),
                    // `accept`/`save` on nothing pending would be a silent
                    // no-op in the reducer; emit a short hint instead.
                    Some("accept" | "save") if self.chat_detect_offer.is_none() => {
                        self.chat.push(ChatTurn::system(
                            "no detected endpoint to act on yet — run `/detect` first to probe \
                             for a local engine"
                                .to_string(),
                        ));
                    }
                    Some("accept") => self.accept_detect_offer(),
                    Some("save") => self.save_detect_offer(),
                    Some("dismiss") => self.dismiss_detect_offer(),
                    Some(other) => {
                        self.chat.push(ChatTurn::error(format!(
                            "unknown /detect action `{other}` (try accept, save, dismiss, or bare /detect to probe)"
                        )));
                    }
                }
            }
            // `/chat [prompt]`: with a prompt, send it to the agent (passthrough,
            // exactly as a plain line would); bare `/chat` focuses the Chat tab.
            "chat" => {
                let prompt = rest
                    .split_once(char::is_whitespace)
                    .map(|(_, tail)| tail.trim())
                    .unwrap_or_default();
                if prompt.is_empty() {
                    self.active_tab = ActiveTab::Chat;
                    self.chat_focused = true;
                } else if !self.chat_sending {
                    // Mirror `submit_chat`'s dispatch tail (the slash already
                    // consumed `chat_input`); guarded so an in-flight request is
                    // never double-spawned.
                    self.chat.push(ChatTurn::user(prompt.to_string()));
                    self.chat_sending = true;
                    self.chat_dispatch = true;
                }
            }
            // Unknown slash command: an error turn, never sent to the LLM.
            other => {
                self.chat.push(ChatTurn::error(format!(
                    "unknown command: /{other} (try /help)"
                )));
            }
        }
        SlashOutcome::Handled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::test_support::st;
    use crate::app::{ChatConsent, ChatRole};

    /// EAI-7354: once `ChatConsent::Accepted`, a focused `'d'` keypress is
    /// ordinary chat text (see `handle_key`'s `ChatConsent::Accepted` arm), so
    /// re-detect must be reachable another way. `/detect` is the affordance —
    /// it runs through the same `request_detect`/`set_detect_result` edge as
    /// the pre-accept `'d'` key, and since the offer prompt isn't drawn once
    /// accepted (`ui::tabs::chat::draw` only renders the gate pre-accept), the
    /// result is echoed into the transcript instead.
    #[test]
    fn slash_detect_probes_and_echoes_result_while_accepted() {
        let mut s = st();
        s.set_chat_config(
            Some(crate::llm::LlmConfig {
                base_url: "http://localhost:8000/v1".into(),
                model: "m".into(),
                api_key: None,
                auth_header: None,
            }),
            true,
        );
        assert_eq!(s.chat_consent, ChatConsent::Accepted);

        assert_eq!(s.handle_slash_command("/detect"), SlashOutcome::Handled);
        assert!(s.chat_detecting && s.chat_detect_dispatch);
        assert!(
            s.chat.last().is_some(),
            "probing while accepted is echoed into the transcript"
        );

        let local = crate::llm::detected_llm_config("http://localhost:13305/v1", "Llama-3.2-3B");
        s.set_detect_result(Some(local.clone()));
        assert_eq!(s.chat_detect_offer.as_ref(), Some(&local));
        // No gate UI once accepted (draw_consent only renders pre-accept) — the
        // offer must be surfaced in the transcript instead.
        let last = s.chat.last().expect("echoed offer turn");
        assert!(last.content.contains("/detect accept"));
    }

    /// `/detect accept` mid-session must integrate with the same
    /// `chat_endpoint_rebuild` edge the initial pre-accept offer uses, or the
    /// live agent silently keeps talking to the stale endpoint.
    #[test]
    fn slash_detect_accept_raises_endpoint_rebuild_like_initial_accept() {
        let mut s = st();
        s.set_chat_config(
            Some(crate::llm::LlmConfig {
                base_url: "http://localhost:8000/v1".into(),
                model: "m".into(),
                api_key: None,
                auth_header: None,
            }),
            true,
        );
        s.active_provider = ChatProvider::Openai; // observe the realignment to Local
        let local = crate::llm::detected_llm_config("http://localhost:13305/v1", "Llama-3.2-3B");
        s.set_detect_result(Some(local.clone()));

        assert_eq!(
            s.handle_slash_command("/detect accept"),
            SlashOutcome::Handled
        );
        assert_eq!(s.chat_llm.as_ref(), Some(&local));
        assert_eq!(s.active_provider, ChatProvider::Local);
        assert_eq!(
            s.chat_endpoint_rebuild,
            Some(ChatProvider::Openai),
            "re-detect accept raises the rebuild edge exactly like the initial accept"
        );
    }

    #[test]
    fn slash_detect_dismiss_and_unknown_subcommand() {
        let mut s = st();
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:8000/v1",
            "m",
        )));
        assert_eq!(
            s.handle_slash_command("/detect dismiss"),
            SlashOutcome::Handled
        );
        assert!(s.chat_detect_offer.is_none());

        assert_eq!(
            s.handle_slash_command("/detect bogus"),
            SlashOutcome::Handled
        );
        let last = s.chat.last().expect("error turn");
        assert!(last.content.contains("unknown /detect action"));
    }

    /// Sub-commands are case-insensitive, matching `/permissions` / `/provider`.
    #[test]
    fn slash_detect_subcommand_is_case_insensitive() {
        let mut s = st();
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:8000/v1",
            "m",
        )));
        // `DISMISS` (upper) must act exactly like `dismiss`.
        assert_eq!(
            s.handle_slash_command("/detect DISMISS"),
            SlashOutcome::Handled
        );
        assert!(
            s.chat_detect_offer.is_none(),
            "uppercase sub-command is normalized and handled"
        );
    }

    /// `/detect accept` / `/detect save` with nothing pending emits a hint
    /// turn rather than a silent no-op.
    #[test]
    fn slash_detect_accept_without_offer_emits_hint() {
        let mut s = st();
        assert!(s.chat_detect_offer.is_none());
        assert_eq!(
            s.handle_slash_command("/detect accept"),
            SlashOutcome::Handled
        );
        // No endpoint was adopted; a hint explains why.
        assert_eq!(s.chat_endpoint_rebuild, None);
        let last = s.chat.last().expect("hint turn");
        assert!(last.content.contains("no detected endpoint"));
    }

    #[test]
    fn slash_help_opens_help_modal() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/help"), SlashOutcome::Handled);
        assert_eq!(s.modal, Modal::Help);
    }

    #[test]
    fn slash_question_mark_opens_help_modal() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/?"), SlashOutcome::Handled);
        assert_eq!(s.modal, Modal::Help);
    }

    #[test]
    fn slash_clear_empties_transcript() {
        let mut s = st();
        s.chat.push(ChatTurn::user("hi"));
        s.chat.push(ChatTurn::agent("hello"));
        assert_eq!(s.handle_slash_command("/clear"), SlashOutcome::Handled);
        assert!(s.chat.is_empty());
    }

    #[test]
    fn slash_quit_sets_should_quit() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/quit"), SlashOutcome::Handled);
        assert!(s.should_quit);
    }

    #[test]
    fn slash_exit_sets_should_quit() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/exit"), SlashOutcome::Handled);
        assert!(s.should_quit);
    }

    #[test]
    fn slash_quit_opens_confirm_prompt_instead_when_a_model_is_serving() {
        // Issue #145: `/quit` is gated the same way the `q` key is — it must
        // not bypass the confirm prompt just because it came from chat.
        let mut s = st();
        s.instances.insert(
            "vllm-1".into(),
            rocm_dash_core::metrics::Instance {
                status: rocm_dash_core::metrics::InstanceStatus::Ready,
                ..Default::default()
            },
        );
        assert_eq!(s.handle_slash_command("/quit"), SlashOutcome::Handled);
        assert!(!s.should_quit);
        assert!(s.quit_confirm.is_some());
    }

    #[test]
    fn slash_home_switches_to_overview() {
        let mut s = st();
        s.active_tab = ActiveTab::Observe;
        assert_eq!(s.handle_slash_command("/home"), SlashOutcome::Handled);
        assert_eq!(s.active_tab, ActiveTab::Home);
    }

    #[test]
    fn slash_gpu_switches_to_observe() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/gpu"), SlashOutcome::Handled);
        assert_eq!(s.active_tab, ActiveTab::Observe);
    }

    #[test]
    fn slash_doctor_opens_overlay() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/doctor"), SlashOutcome::Handled);
        assert!(s.examine_manager.is_some());
    }

    #[test]
    fn slash_runtimes_opens_overlay() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/runtimes"), SlashOutcome::Handled);
        assert!(s.runtime_manager.is_some());
    }

    #[test]
    fn slash_config_opens_overlay() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/config"), SlashOutcome::Handled);
        assert!(s.config_manager.is_some());
    }

    #[test]
    fn slash_logs_opens_overlay() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/logs"), SlashOutcome::Handled);
        assert!(s.logs_view.is_some());
    }

    #[test]
    fn slash_model_raises_executor_request() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/model"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("model raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(req.args, serde_json::json!({ "args": ["model"] }));
    }

    #[test]
    fn slash_daemon_raises_executor_request() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/daemon"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("daemon raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["daemon", "status"] })
        );
    }

    // --- Phase 4: mutating slash dispatch + approval modal flow ---

    #[test]
    fn slash_install_raises_install_sdk_request() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/install ~/rocm"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("install raises a slash_tool request");
        assert_eq!(req.name, "install_sdk");
        assert_eq!(req.args["channel"], "release");
        assert_eq!(req.args["format"], "wheel");
        // The validator REQUIRES a prefix; the slash path must supply one or the
        // modal never opens.
        assert_eq!(req.args["prefix"], "~/rocm");
    }

    #[test]
    fn slash_install_without_prefix_hints_not_dispatch() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/install"), SlashOutcome::Handled);
        assert!(
            s.slash_tool.is_none(),
            "no dispatch without an install folder"
        );
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn plan_request_set_by_slash() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/plan install rocm"),
            SlashOutcome::Handled
        );
        assert_eq!(s.plan_request.as_deref(), Some("install rocm"));
        // The command word is matched case-insensitively, and so must the arg
        // extraction be: `/Plan` (mixed case) dispatches just like `/plan`.
        let mut s_mixed = st();
        assert_eq!(
            s_mixed.handle_slash_command("/Plan install rocm"),
            SlashOutcome::Handled
        );
        assert_eq!(s_mixed.plan_request.as_deref(), Some("install rocm"));
        // Bare `/plan` hints with a usage turn and raises no plan edge.
        let mut s2 = st();
        assert_eq!(s2.handle_slash_command("/plan"), SlashOutcome::Handled);
        assert!(s2.plan_request.is_none(), "bare /plan must not dispatch");
        assert_eq!(s2.chat.last().unwrap().role, ChatRole::Agent);
        assert!(s2.chat.last().unwrap().content.contains("usage"));
    }

    #[test]
    fn slash_provider_switches_backend() {
        // /provider anthropic → active_provider set + edge raised.
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/provider anthropic"),
            SlashOutcome::Handled
        );
        assert_eq!(s.active_provider, ChatProvider::Anthropic);
        assert_eq!(
            s.provider_switch,
            Some(ProviderSwitch {
                previous: ChatProvider::Local,
                target: ChatProvider::Anthropic,
            })
        );
        // /provider openai → openai.
        let mut s2 = st();
        s2.handle_slash_command("/provider openai");
        assert_eq!(s2.active_provider, ChatProvider::Openai);
        assert_eq!(
            s2.provider_switch,
            Some(ProviderSwitch {
                previous: ChatProvider::Local,
                target: ChatProvider::Openai,
            })
        );
        // /provider local → local (matched case-insensitively).
        let mut s3 = st();
        s3.handle_slash_command("/Provider LOCAL");
        assert_eq!(s3.active_provider, ChatProvider::Local);
        assert_eq!(
            s3.provider_switch,
            Some(ProviderSwitch {
                previous: ChatProvider::Local,
                target: ChatProvider::Local,
            })
        );
    }

    #[test]
    fn slash_provider_switch_captures_previous_provider() {
        // (Phase-8 polish) A failed switch must revert to the provider that was
        // active BEFORE the attempt, not unconditionally to Local. Prove the
        // slash handler snapshots the prior provider in the edge: switch to
        // openai (optimistic), then attempt anthropic — the edge carries
        // previous=Openai so the drain can revert there on a build failure.
        let mut s = st();
        s.handle_slash_command("/provider openai");
        assert_eq!(s.active_provider, ChatProvider::Openai);
        s.handle_slash_command("/provider anthropic");
        assert_eq!(
            s.provider_switch,
            Some(ProviderSwitch {
                previous: ChatProvider::Openai,
                target: ChatProvider::Anthropic,
            }),
            "the failed-switch revert target is the prior provider, not Local"
        );
    }

    #[test]
    fn slash_provider_bare_shows_current_and_unknown_hints() {
        // Bare /provider shows the current backend and raises no edge.
        let mut s = st();
        s.handle_slash_command("/provider");
        assert!(s.provider_switch.is_none());
        let last = s.chat.last().unwrap();
        assert_eq!(last.role, ChatRole::Agent);
        assert!(last.content.contains("local"), "shows current provider");
        // Unknown provider hints (error turn), no edge, no switch.
        let mut s2 = st();
        s2.handle_slash_command("/provider grok");
        assert!(s2.provider_switch.is_none());
        assert_eq!(s2.active_provider, ChatProvider::Local);
        assert_eq!(s2.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_chat_passthrough_submits_prompt() {
        // /chat <prompt> pushes the user turn + raises the chat_dispatch edge.
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/chat what's GPU-2 doing?"),
            SlashOutcome::Handled
        );
        assert!(s.chat_dispatch, "passthrough raises the spawn edge");
        assert!(s.chat_sending);
        let last = s.chat.last().unwrap();
        assert_eq!(last.role, ChatRole::User);
        assert_eq!(last.content, "what's GPU-2 doing?");
    }

    #[test]
    fn slash_chat_bare_focuses_chat_tab() {
        // Bare /chat focuses the Chat tab, raises no dispatch edge.
        let mut s = st();
        s.active_tab = ActiveTab::Home;
        assert_eq!(s.handle_slash_command("/chat"), SlashOutcome::Handled);
        assert_eq!(s.active_tab, ActiveTab::Chat);
        assert!(s.chat_focused);
        assert!(!s.chat_dispatch, "bare /chat does not dispatch");
    }

    #[test]
    fn slash_engine_raises_install_engine_request() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/engine vllm"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("engine raises a slash_tool request");
        assert_eq!(req.name, "install_engine");
        assert_eq!(req.args, serde_json::json!({ "engine": "vllm" }));
    }

    #[test]
    fn slash_engine_without_name_hints_not_dispatch() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/engine"), SlashOutcome::Handled);
        assert!(s.slash_tool.is_none(), "no dispatch without an engine name");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_serve_raises_launch_server_request() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/serve deepseek-r1"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("serve raises a slash_tool request");
        assert_eq!(req.name, "launch_server");
        assert_eq!(req.args["model"], "deepseek-r1");
        // Loopback host is forced so the validator never rejects the slash path.
        assert_eq!(req.args["host"], "127.0.0.1");
    }

    #[test]
    fn slash_services_stop_raises_stop_server_request() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/services stop svc-1"),
            SlashOutcome::Handled
        );
        let req = s
            .slash_tool
            .expect("services stop raises a slash_tool request");
        assert_eq!(req.name, "stop_server");
        assert_eq!(req.args, serde_json::json!({ "service_id": "svc-1" }));
    }

    #[test]
    fn slash_services_restart_is_guided_not_stop() {
        // restart is NOT wired through the chat seam yet; it must guide the
        // operator instead of silently running stop_server (a semantic lie).
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/services restart svc-1"),
            SlashOutcome::Handled
        );
        assert!(
            s.slash_tool.is_none(),
            "restart must NOT dispatch a stop_server request"
        );
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_services_bare_is_read_only_list() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/services"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("bare services lists managed services");
        assert_eq!(req.name, "services");
    }

    // --- Phase 5: lifecycle slash dispatch (read/mutate split via rocm_command) ---

    #[test]
    fn slash_update_is_read_only_report() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/update"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("update raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(req.args, serde_json::json!({ "args": ["update"] }));
    }

    #[test]
    fn slash_update_apply_is_mutating() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/update --apply"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("update --apply raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["update", "--apply"] })
        );
    }

    #[test]
    fn slash_update_apply_is_position_independent() {
        // `--apply` anywhere in the args triggers the mutating path (matching
        // `/uninstall`'s all-tokens parse), not only as the immediate second token.
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/update --preview --apply"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("update --apply raises a request");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["update", "--apply"] }),
            "--apply past the second token must still apply"
        );
    }

    #[test]
    fn slash_comfyui_bare_is_status() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/comfyui"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("comfyui raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["comfyui", "status"] })
        );
    }

    #[test]
    fn slash_comfy_alias_is_status() {
        // `/comfy` is an alias for `/comfyui` and must map to the same
        // read-only status argv.
        let mut s = st();
        assert_eq!(s.handle_slash_command("/comfy"), SlashOutcome::Handled);
        let req = s
            .slash_tool
            .expect("comfy alias raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["comfyui", "status"] })
        );
    }

    #[test]
    fn slash_comfyui_start_is_mutating() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/comfyui start"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("comfyui start raises a request");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["comfyui", "start"] })
        );
    }

    #[test]
    fn slash_comfyui_logs_is_read_only() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/comfyui logs"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("comfyui logs raises a request");
        assert_eq!(req.args, serde_json::json!({ "args": ["comfyui", "logs"] }));
    }

    #[test]
    fn slash_uninstall_defaults_to_dry_run() {
        // SAFETY: a bare `/uninstall` must NEVER trigger a real uninstall.
        let mut s = st();
        assert_eq!(s.handle_slash_command("/uninstall"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("uninstall raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["uninstall", "--dry-run"] }),
            "bare /uninstall MUST default to a dry-run, not a real uninstall"
        );
        assert_eq!(
            req.label, "uninstall --dry-run",
            "the dry-run label is the safety-critical user-visible string"
        );
    }

    #[test]
    fn slash_uninstall_apply_is_real() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/uninstall --apply"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("uninstall --apply raises a request");
        assert_eq!(req.args, serde_json::json!({ "args": ["uninstall"] }));
        assert_eq!(
            req.label, "uninstall",
            "the real-uninstall label is the safety-critical user-visible string"
        );
    }

    #[test]
    fn slash_uninstall_conflicting_flags_is_guided() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/uninstall --apply --dry-run"),
            SlashOutcome::Handled
        );
        assert!(
            s.slash_tool.is_none(),
            "conflicting uninstall flags must NOT dispatch"
        );
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_setup_bare_is_status() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/setup"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("setup raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(req.args, serde_json::json!({ "args": ["setup", "status"] }));
    }

    #[test]
    fn slash_setup_reset_is_mutating() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/setup reset"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("setup reset raises a request");
        assert_eq!(req.args, serde_json::json!({ "args": ["setup", "reset"] }));
    }

    #[test]
    fn slash_setup_unknown_sub_is_guided() {
        // SetupCommand has only status + reset — `skip` must guide, not dispatch.
        let mut s = st();
        assert_eq!(s.handle_slash_command("/setup skip"), SlashOutcome::Handled);
        assert!(
            s.slash_tool.is_none(),
            "unsupported /setup sub must NOT dispatch"
        );
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    // --- Phase 6: automations / reviews / approve / reject / edit / permissions ---

    #[test]
    fn slash_automations_bare_lists() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations"),
            SlashOutcome::Handled
        );
        let req = s
            .slash_tool
            .expect("automations raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["automations", "list"] })
        );
    }

    #[test]
    fn slash_automations_enable_raises_watcher_enable() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations enable foo --mode observe"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("enable raises a request");
        assert_eq!(req.name, "watcher_enable");
        assert_eq!(
            req.args,
            serde_json::json!({ "watcher": "foo", "mode": "observe" })
        );
    }

    #[test]
    fn slash_automations_enable_without_mode_omits_field() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations enable foo"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("enable raises a request");
        assert_eq!(req.name, "watcher_enable");
        assert_eq!(req.args, serde_json::json!({ "watcher": "foo" }));
        assert!(req.args.get("mode").is_none(), "mode must be omitted");
    }

    #[test]
    fn slash_automations_disable_raises_watcher_disable() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations disable foo"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("disable raises a request");
        assert_eq!(req.name, "watcher_disable");
        assert_eq!(req.args, serde_json::json!({ "watcher": "foo" }));
    }

    #[test]
    fn slash_automations_enable_without_watcher_hints() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations enable"),
            SlashOutcome::Handled
        );
        assert!(s.slash_tool.is_none(), "no dispatch without a watcher");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_reviews_bare_lists() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/reviews"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("reviews raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["automations", "list"] })
        );
    }

    #[test]
    fn slash_reviews_id_shows_proposal() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/reviews p1"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("reviews <id> raises a request");
        assert_eq!(req.name, "proposal_action");
        assert_eq!(
            req.args,
            serde_json::json!({ "proposal_id": "p1", "action": "show" })
        );
    }

    #[test]
    fn slash_approve_id_raises_proposal_approve() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/approve p1"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("approve raises a request");
        assert_eq!(req.name, "proposal_action");
        assert_eq!(
            req.args,
            serde_json::json!({ "proposal_id": "p1", "action": "approve" })
        );
    }

    #[test]
    fn slash_approve_bare_hints() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/approve"), SlashOutcome::Handled);
        assert!(s.slash_tool.is_none(), "no dispatch without a proposal id");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_reject_id_raises_proposal_reject() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/reject p1"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("reject raises a request");
        assert_eq!(req.name, "proposal_action");
        assert_eq!(
            req.args,
            serde_json::json!({ "proposal_id": "p1", "action": "reject" })
        );
    }

    #[test]
    fn slash_edit_id_shows_proposal_with_note() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/edit p1"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("edit raises a request");
        assert_eq!(req.name, "proposal_action");
        assert_eq!(
            req.args,
            serde_json::json!({ "proposal_id": "p1", "action": "show" })
        );
        // A one-line note directs the operator to /approve or /reject.
        let last = s.chat.last().expect("edit pushes a note turn");
        assert_eq!(last.role, ChatRole::Agent);
        assert!(last.content.contains("/approve") && last.content.contains("/reject"));
    }

    #[test]
    fn slash_permissions_bare_is_config_show() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/permissions"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("permissions raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(req.args, serde_json::json!({ "args": ["config", "show"] }));
    }

    #[test]
    fn slash_permissions_full_access_is_mutating() {
        // SAFETY: permission escalation must go through the approval modal — it is
        // dispatched as an approval-classified rocm_command, never run inline.
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/permissions full-access"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("full-access raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["config", "set-permissions", "full_access"] })
        );
    }

    #[test]
    fn slash_permissions_ask_is_mutating() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/permissions ask"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("ask raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["config", "set-permissions", "ask"] })
        );
    }

    #[test]
    fn slash_tool_reply_does_not_disturb_chat_sending() {
        // The slash-tool reply path is decoupled from the agent state machine:
        // appending a slash-tool summary must NOT touch `chat_sending`, even if
        // an agent request happens to be in flight at the same time.
        let mut s = st();
        s.chat_sending = true;
        let before = s.chat.len();
        s.on_slash_tool_reply("x".into());
        assert_eq!(s.chat.len(), before + 1, "slash-tool turn appended");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(
            s.chat_sending,
            "chat_sending untouched by slash-tool reply (decoupled)"
        );
    }

    #[test]
    fn slash_unknown_appends_error_turn_and_is_handled() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/zzz"), SlashOutcome::Handled);
        assert_eq!(s.chat.len(), 1);
        assert_eq!(s.chat[0].role, ChatRole::Error);
        assert!(s.chat[0].content.contains("/zzz"));
    }

    #[test]
    fn plain_text_is_not_a_command() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("what's GPU-2 doing?"),
            SlashOutcome::NotCommand
        );
    }

    #[test]
    fn submit_routes_slash_command_away_from_the_agent() {
        // A slash line through submit_chat must NOT raise the agent dispatch edge.
        let mut s = st();
        s.chat_input = "/help".into();
        s.submit_chat();
        assert_eq!(s.modal, Modal::Help);
        assert!(
            !s.chat_dispatch,
            "slash command never dispatches to the LLM"
        );
        assert!(!s.chat_sending);
        assert!(s.chat_input.is_empty());
    }
}
