// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Slash-command routing for the chat input.
//!
//! [`AppState::handle_slash_command`] classifies a `/`-prefixed chat line and
//! either mutates reducer state directly (nav / overlays), raises a one-shot
//! executor edge (`slash_tool` / `plan_request` / `provider_switch`) for the
//! event loop to drain off-thread, or pushes a usage/error turn. Stays I/O-free.
//! Split out of `app/mod.rs` to keep the core reducer focused; the slash-command
//! payload types it raises (`SlashOutcome`, `SlashToolRequest`, `ProviderSwitch`)
//! stay in `mod.rs` alongside the `AppState` fields that carry them.

use super::{
    ActiveTab, AppState, ChatProvider, ChatTurn, Modal, ProviderSwitch, SlashOutcome,
    SlashToolRequest,
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
            "quit" | "exit" => self.should_quit = true,
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
                // (read-only); `/setup reset` re-arms it (approval-gated). The CLI
                // only has status + reset — anything else is guided, not run.
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
            "detect" => match rest.split_whitespace().nth(1) {
                None => self.request_detect(),
                Some("accept") => self.accept_detect_offer(),
                Some("save") => self.save_detect_offer(),
                Some("dismiss") => self.dismiss_detect_offer(),
                Some(other) => {
                    self.chat.push(ChatTurn::error(format!(
                        "unknown /detect action `{other}` (try accept, save, dismiss, or bare /detect to probe)"
                    )));
                }
            },
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
