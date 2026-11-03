// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The tool-approval flow: handing a completed plan's action to the approval
//! path, opening the approval modal for a surfaced mutating-tool intent, and
//! routing the operator's verdict.
//!
//! A further `impl AppState` block, split out of `app/mod.rs` to keep the core
//! reducer focused. Named `tool_approval` rather than `approval` so it does not
//! collide with the `ui::approval` render component.

use super::{AppState, ChatTurn, PendingApproval, PlannedAction, SlashToolRequest};

impl AppState {
    /// Handle a completed natural-language plan (Phase 7). Pushes the rendered
    /// plan as a chat turn (the review). If the plan's next action is a complete
    /// mutating action (`approval_required` AND NOT `has_placeholders`) that is
    /// NOT provider-assisted, hand its argv to the Phase-4 approval flow via the
    /// `rocm_command` slash-tool edge (execute → ApprovalRequired → modal). A
    /// placeholder/incomplete plan, a non-mutating one, or a provider-assisted
    /// one stays plan-only: no approval focus, no execution. Provider-assisted
    /// plans are review-only, mirroring the bin's
    /// `validate_freeform_execution_action` guard.
    pub(crate) fn on_plan_ready(&mut self, text: String, action: Option<PlannedAction>) {
        self.chat.push(ChatTurn::agent(text));
        if let Some(action) = action
            && action.approval_required
            && !action.has_placeholders
            && !action.provider_assisted
        {
            // `/plan` drains off-thread without setting `chat_sending`, so a slash
            // command issued while the plan was in flight may already have queued a
            // tool request or opened an approval. Don't clobber it — surface a
            // message and drop the plan's action (mirrors `open_approval`'s
            // single-in-flight guard).
            if self.slash_tool.is_some() || self.approval.is_some() {
                self.chat.push(ChatTurn::error(
                    "A command is already in progress; the planned action was discarded. Resolve it first, then re-run the plan.",
                ));
                return;
            }
            self.slash_tool = Some(SlashToolRequest {
                name: "rocm_command".to_string(),
                args: serde_json::json!({ "args": action.args }),
                label: "plan action".to_string(),
            });
        }
    }

    /// Handle an executor-backed slash-tool reply (`/model`, `/daemon`): append
    /// the summary as an agent-role turn WITHOUT touching `chat_sending`. The
    /// slash-tool path is independent of the agent in-flight state machine, so
    /// this must never clear/modify that flag (proven by
    /// `slash_tool_reply_does_not_disturb_chat_sending`).
    pub(crate) fn on_slash_tool_reply(&mut self, text: String) {
        self.chat.push(ChatTurn::agent(text));
    }

    /// Open the approval modal for a surfaced mutating-tool intent (Phase 4).
    /// Closes any operational overlay first so the modal owns focus alone.
    pub(crate) fn open_approval(&mut self, intent: crate::tool_exec::ApprovalIntent) {
        // Also refuse while a quit-confirm prompt is open (issue #145): it is
        // delivered asynchronously off a chat-agent event, independent of any
        // keypress, so it can otherwise arrive after the user already asked
        // to quit. Letting it through would silently race the two modals —
        // `quit_confirm` would still be what's drawn (rendered last), but the
        // approval would be what every keystroke actually resolves, so a `y`
        // meant to confirm the quit would instead approve an unreviewed
        // mutating tool call. The already-open, user-initiated quit decision
        // wins; the model already got its "surfaced for approval" tool
        // result regardless (`agent/tools.rs`), so discarding here — same as
        // the second-approval-while-one-pending case below — never leaves it
        // waiting on a reply that won't come.
        if self.approval.is_some() {
            self.chat.push(ChatTurn::error(
                "An action is already awaiting approval; the new request was discarded. Resolve the open approval first.",
            ));
            return;
        }
        if self.quit_confirm_pending() {
            self.chat.push(ChatTurn::error(
                "An action was proposed while quitting was being confirmed; the request was discarded.",
            ));
            return;
        }
        self.close_overlays();
        self.approval = Some(PendingApproval {
            req: crate::ui::approval::ApprovalRequest::new(intent.title, intent.body),
            // An unreviewed tool call the model wants to run defaults to Deny,
            // unlike the shared `ApprovalChoice` default (see its doc comment).
            choice: crate::ui::approval::ApprovalChoice::Deny,
            name: intent.name,
            arguments: intent.arguments,
        });
    }

    /// Route a key to the open approval modal: move the cursor and return a
    /// verdict if the key confirmed one. Pure w.r.t. I/O — the caller maps the
    /// verdict onto execution (Approve) or a declined turn (Deny/Cancel). No-op
    /// returning `None` when no modal is open.
    pub(crate) fn on_approval_key(
        &mut self,
        code: crossterm::event::KeyCode,
    ) -> Option<crate::ui::approval::ApprovalVerdict> {
        let pa = self.approval.as_mut()?;
        let (choice, verdict) = crate::ui::approval::approval_key(code, pa.choice);
        pa.choice = choice;
        verdict
    }

    /// Take the pending approval's `(name, arguments)` for off-thread execution,
    /// clearing the modal. Returns `None` if no modal is open.
    pub(crate) fn take_approval(&mut self) -> Option<(String, serde_json::Value)> {
        self.approval.take().map(|pa| (pa.name, pa.arguments))
    }

    /// Handle a Deny/Cancel verdict: clear the modal and append a declined turn.
    /// Nothing executes.
    pub(crate) fn on_approval_declined(&mut self) {
        self.approval = None;
        self.chat.push(ChatTurn::agent("Action declined."));
    }

    /// Append the approved-action result turn AND raise the one-shot
    /// `chat_dispatch` edge so the agent does EXACTLY ONE automatic follow-up
    /// turn that incorporates the result. The result is pushed as an agent turn
    /// (so `build_messages` sends it as conversational context); `chat_dispatch`
    /// is consumed once by the event loop, so this never loops. A further
    /// mutating request from that follow-up re-surfaces approval (user-gated),
    /// so there is no unbounded execution. Clears any open modal defensively.
    pub(crate) fn on_approval_result(&mut self, text: String) {
        self.approval = None;
        self.chat.push(ChatTurn::agent(text));
        // Exactly one follow-up: raise the edge once. `chat_sending` mirrors a
        // normal submit so the UI shows the in-flight state and a double key
        // can't race a second dispatch.
        self.chat_sending = true;
        self.chat_dispatch = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::actions::run_approved;
    use crate::app::test_support::st;
    use crate::app::{ActiveTab, ChatRole};

    /// Recording executor: mutating names surface `ApprovalRequired`; the
    /// approved replay records `(name, args)` and returns a success Result. Used
    /// to drive the approve/deny/follow-up tests offline (no real installs).
    #[derive(Debug)]
    struct RecordingExecutor {
        approved: std::sync::Arc<std::sync::Mutex<Vec<(String, serde_json::Value)>>>,
    }

    impl RecordingExecutor {
        fn new() -> Self {
            Self {
                approved: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }
    }

    impl crate::tool_exec::RocmToolExecutor for RecordingExecutor {
        fn execute(
            &self,
            name: &str,
            args: &serde_json::Value,
        ) -> crate::tool_exec::RocmToolOutcome {
            crate::tool_exec::RocmToolOutcome::ApprovalRequired(crate::tool_exec::ApprovalIntent {
                title: "T".to_string(),
                body: vec!["cmd".to_string()],
                name: name.to_string(),
                arguments: args.clone(),
            })
        }
        fn execute_approved(
            &self,
            name: &str,
            args: &serde_json::Value,
        ) -> crate::tool_exec::RocmToolOutcome {
            self.approved
                .lock()
                .unwrap()
                .push((name.to_string(), args.clone()));
            crate::tool_exec::RocmToolOutcome::Result(serde_json::json!({ "ok": true }))
        }
    }

    /// The leading sentences of the CLI refusal `rocm comfyui install` prints
    /// when two managed ROCm runtimes are ready and none is activated. This is a
    /// verbatim *prefix*, not the whole message: the real one continues with an
    /// `Available: <key>, <key>.` list and a trailing pointer to
    /// `rocm runtimes list`, both of which depend on the planted runtimes and
    /// neither of which this test inspects — it only pins what the seam does
    /// with the envelope it is handed, so the remedy-bearing prefix is the
    /// relevant part.
    const AMBIGUOUS_RUNTIME_REFUSAL: &str = "Multiple ROCm runtimes are ready. Pick one in `/runtimes`, set a default \
         with `rocm runtimes activate <key>`, or pass `--runtime-id <key>`.";

    /// An executor whose approved replay *replicates* what the real seam returns
    /// for a `rocm` subprocess that exited non-zero: `run_rocm_capture_for_paths`
    /// *captures* the failure, so `run_internal_mcp_call` returns `Ok` with an
    /// `isError: true` envelope and the stderr buried in `structuredContent` —
    /// it never returns `Err`, so the seam never builds `RocmToolOutcome::Error`.
    ///
    /// Replicates, not reaches: those producers live in the bin, which depends
    /// on this crate, so this crate cannot call them. The envelope below is
    /// hand-built to their shape and the test pins only what happens
    /// *downstream* of it. That the producers really do hand the seam an `Ok`
    /// envelope for a non-zero exit is pinned separately, against a real `rocm`
    /// subprocess, by `seam_execute_approved_captures_a_failing_command_as_a_result`
    /// in `apps/rocm/src/dash_seam.rs`.
    #[derive(Debug)]
    struct CapturedFailureExecutor;

    impl crate::tool_exec::RocmToolExecutor for CapturedFailureExecutor {
        fn execute(
            &self,
            name: &str,
            args: &serde_json::Value,
        ) -> crate::tool_exec::RocmToolOutcome {
            crate::tool_exec::RocmToolOutcome::ApprovalRequired(crate::tool_exec::ApprovalIntent {
                title: "Install ComfyUI".to_string(),
                body: vec!["rocm comfyui install".to_string()],
                name: name.to_string(),
                arguments: args.clone(),
            })
        }
        fn execute_approved(
            &self,
            _name: &str,
            _args: &serde_json::Value,
        ) -> crate::tool_exec::RocmToolOutcome {
            crate::tool_exec::RocmToolOutcome::Result(serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": format!("Ran `rocm` command.\n\nstderr:\n{AMBIGUOUS_RUNTIME_REFUSAL}"),
                }],
                "structuredContent": {
                    "argv": ["rocm", "comfyui", "install"],
                    "exit_status": 1,
                    "stdout": "",
                    "stderr": AMBIGUOUS_RUNTIME_REFUSAL,
                },
                "isError": true,
            }))
        }
    }

    #[test]
    fn on_plan_ready_renders_plan_text() {
        let mut s = st();
        s.on_plan_ready("planner: hybrid-parser-v1\nplan body".to_string(), None);
        // The review is appended as a chat turn…
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(s.chat.last().unwrap().content.contains("plan body"));
        // …and with no action there is nothing to approve or execute.
        assert!(s.slash_tool.is_none());
    }

    #[test]
    fn on_plan_ready_complete_mutating_hands_off_to_approval() {
        let mut s = st();
        let action = PlannedAction {
            args: vec![
                "install".to_string(),
                "sdk".to_string(),
                "--prefix".to_string(),
                "/x".to_string(),
            ],
            approval_required: true,
            has_placeholders: false,
            provider_assisted: false,
        };
        s.on_plan_ready("the plan".to_string(), Some(action));
        // The plan review is shown…
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        // …and the complete mutating action is forwarded as a rocm_command
        // slash-tool request (→ execute() → ApprovalRequired → modal).
        let req = s
            .slash_tool
            .expect("complete mutating plan hands off to approval");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args["args"],
            serde_json::json!(["install", "sdk", "--prefix", "/x"])
        );
    }

    #[test]
    fn on_plan_ready_guards_against_pending_slash_tool() {
        let mut s = st();
        // A slash command queued a tool request while the plan computed off-thread.
        s.slash_tool = Some(SlashToolRequest {
            name: "rocm_command".to_string(),
            args: serde_json::json!({ "args": ["services", "list"] }),
            label: "pending".to_string(),
        });
        let action = PlannedAction {
            args: vec!["update".to_string()],
            approval_required: true,
            has_placeholders: false,
            provider_assisted: false,
        };
        s.on_plan_ready("the plan".to_string(), Some(action));
        // The in-flight request is NOT clobbered…
        let req = s.slash_tool.as_ref().expect("pending request preserved");
        assert_eq!(req.label, "pending");
        assert_eq!(req.args["args"], serde_json::json!(["services", "list"]));
        // …and the user is told the planned action was discarded.
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn on_plan_ready_placeholder_stays_plan_only() {
        let mut s = st();
        let action = PlannedAction {
            args: vec![
                "install".to_string(),
                "sdk".to_string(),
                "--prefix".to_string(),
                "<PATH>".to_string(),
            ],
            approval_required: true,
            has_placeholders: true,
            provider_assisted: false,
        };
        s.on_plan_ready("the plan".to_string(), Some(action));
        // The plan is shown for review, but an incomplete (placeholder) plan
        // never focuses approval and never executes — mirrors the legacy
        // `natural_serve_with_missing_model_does_not_focus_approval` rule.
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(
            s.slash_tool.is_none(),
            "placeholder plan must stay plan-only (no approval, no execution)"
        );
        assert!(s.approval.is_none(), "no approval modal focus");
    }

    #[test]
    fn on_plan_ready_provider_assisted_stays_plan_only() {
        let mut s = st();
        // A complete (no placeholders) mutating action that a planner provider
        // produced. Even though approval_required && !has_placeholders, the
        // provider_assisted flag keeps it review-only — mirrors the bin's
        // `validate_freeform_execution_action` provider-assisted guard.
        let action = PlannedAction {
            args: vec![
                "serve".to_string(),
                "m".to_string(),
                "--managed".to_string(),
            ],
            approval_required: true,
            has_placeholders: false,
            provider_assisted: true,
        };
        s.on_plan_ready("the plan".to_string(), Some(action));
        // The plan text is rendered for review…
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        // …but the provider-assisted plan never focuses approval or executes:
        // the user runs the displayed command manually.
        assert!(
            s.slash_tool.is_none(),
            "provider-assisted plan must stay plan-only (no execution handoff)"
        );
        assert!(s.approval.is_none(), "no approval modal focus");
    }

    #[test]
    fn approval_required_opens_modal() {
        let mut s = st();
        let intent = crate::tool_exec::ApprovalIntent {
            title: "Install ROCm".to_string(),
            body: vec!["rocm install sdk".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({ "channel": "release" }),
        };
        s.open_approval(intent);
        let pa = s.approval.as_ref().expect("modal opened");
        assert_eq!(pa.name, "install_sdk");
        assert_eq!(pa.req.title, "Install ROCm");
    }

    #[test]
    fn second_approval_request_is_discarded_while_one_pending() {
        // Two mutating calls in one turn must not clobber: the operator could
        // otherwise approve args they never saw.
        let mut s = st();
        let first = crate::tool_exec::ApprovalIntent {
            title: "Install ROCm".to_string(),
            body: vec!["rocm install sdk".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({ "prefix": "~/rocm" }),
        };
        s.open_approval(first);
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "Stop server".to_string(),
            body: vec!["rocm services stop svc-1".to_string()],
            name: "stop_server".to_string(),
            arguments: serde_json::json!({ "service_id": "svc-1" }),
        });
        // The original intent survives intact; the second was discarded.
        let pa = s.approval.as_ref().expect("first approval still pending");
        assert_eq!(pa.name, "install_sdk");
        assert_eq!(pa.arguments["prefix"], "~/rocm");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn approval_request_is_discarded_while_quit_confirm_is_pending() {
        // Issue #145 regression: a chat-agent tool call surfacing for
        // approval must not land while the user is already mid-decision on
        // quitting — the quit-confirm prompt is what's drawn (rendered
        // last), but without this guard every keystroke would still resolve
        // the approval instead, so a `y` meant to confirm the quit would
        // silently approve an unreviewed mutating tool call.
        let mut s = st();
        s.open_quit_confirm();
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "Install ROCm".to_string(),
            body: vec!["rocm install sdk".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({}),
        });
        assert!(
            s.approval.is_none(),
            "approval must be discarded, not opened"
        );
        assert!(
            s.quit_confirm.is_some(),
            "quit-confirm must still own the screen"
        );
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn approve_path_runs_execute_approved_with_expected_args() {
        // (a) approve: a ChatApprovalRequired opens the modal; an Approve verdict
        // drives execute_approved with the exact name + args. We exercise the
        // same sync code path the spawn_blocking uses (`run_approved`).
        let exec = std::sync::Arc::new(RecordingExecutor::new());
        let recorded = exec.approved.clone();
        let shared: crate::tool_exec::SharedRocmToolExecutor = exec;

        let mut s = st();
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "T".to_string(),
            body: vec!["cmd".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({ "channel": "release", "format": "wheel" }),
        });
        // The modal defaults to Deny (item #16); move to Approve, then confirm.
        s.on_approval_key(crossterm::event::KeyCode::Tab);
        let verdict = s.on_approval_key(crossterm::event::KeyCode::Enter);
        assert_eq!(verdict, Some(crate::ui::approval::ApprovalVerdict::Approve));
        let (name, args) = s.take_approval().expect("approval taken on approve");
        assert!(s.approval.is_none(), "modal cleared after taking approval");

        let summary = run_approved(&shared, &name, &args);
        let log = recorded.lock().unwrap();
        assert_eq!(log.len(), 1, "execute_approved ran exactly once");
        assert_eq!(log[0].0, "install_sdk");
        assert_eq!(log[0].1["channel"], "release");
        assert_eq!(log[0].1["format"], "wheel");
        assert!(
            summary.contains("Approved"),
            "concise summary, not raw JSON"
        );
    }

    #[test]
    fn approved_command_failure_stays_a_collapsed_envelope() {
        // Pins the second half of the premise the ComfyUI e2e scenario's
        // CLI-only scope rests on (`tests/e2e-cucumber/features/comfyui.feature`):
        // *given* the captured `isError: true` envelope, `run_approved` takes
        // the `Result` arm and `summarize_json_value` collapses every field, so
        // the refusal text stays out of the chat — reading the `Error` arm
        // (`Approved · … failed: {e}`) as this path's renderer is wrong. The
        // first half — that a non-zero `rocm` exit really does arrive as that
        // envelope rather than as an `Err` — is pinned by the seam test named
        // on `CapturedFailureExecutor` above.
        let shared: crate::tool_exec::SharedRocmToolExecutor =
            std::sync::Arc::new(CapturedFailureExecutor);
        let summary = run_approved(
            &shared,
            "rocm_command",
            &serde_json::json!({ "args": ["comfyui", "install"] }),
        );
        assert!(
            summary.contains("content: [1 items]"),
            "the command envelope is collapsed, not inlined: {summary}"
        );
        assert!(
            summary.contains("structuredContent: {4 fields}"),
            "the captured stdout/stderr subtree is collapsed too: {summary}"
        );
        assert!(
            !summary.contains("Multiple ROCm runtimes are ready"),
            "the CLI refusal must not reach the chat: {summary}"
        );
        assert!(
            !summary.contains("failed:"),
            "a captured non-zero exit is not the Error arm: {summary}"
        );
    }

    #[test]
    fn deny_path_runs_nothing_and_appends_declined_turn() {
        // (b) deny: a Deny/Cancel verdict appends a declined turn and never
        // touches execute_approved.
        let exec = std::sync::Arc::new(RecordingExecutor::new());
        let recorded = exec.approved.clone();

        let mut s = st();
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "T".to_string(),
            body: vec!["cmd".to_string()],
            name: "launch_server".to_string(),
            arguments: serde_json::json!({ "model": "m" }),
        });
        // 'n' is a direct Deny verdict.
        let verdict = s.on_approval_key(crossterm::event::KeyCode::Char('n'));
        assert_eq!(verdict, Some(crate::ui::approval::ApprovalVerdict::Deny));
        s.on_approval_declined();
        assert!(s.approval.is_none(), "modal cleared on deny");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(s.chat.last().unwrap().content.contains("declined"));
        assert!(
            recorded.lock().unwrap().is_empty(),
            "deny must not execute the action"
        );

        // Esc also cancels (no execution).
        let mut s2 = st();
        s2.open_approval(crate::tool_exec::ApprovalIntent {
            title: "T".to_string(),
            body: vec!["cmd".to_string()],
            name: "stop_server".to_string(),
            arguments: serde_json::json!({ "service_id": "x" }),
        });
        assert_eq!(
            s2.on_approval_key(crossterm::event::KeyCode::Esc),
            Some(crate::ui::approval::ApprovalVerdict::Cancel)
        );
    }

    #[test]
    fn approval_modal_escape_is_not_a_focus_trap() {
        // Edge: the approval modal must be escapable — Esc and 'n' both yield a
        // closing verdict, and routing that verdict through the deny/cancel path
        // clears the modal (`approval` → None) without executing. The covered
        // active tab is preserved across open → escape (the modal overlays it).
        for key in [
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyCode::Char('n'),
        ] {
            let mut s = st();
            s.active_tab = ActiveTab::Observe;
            s.open_approval(crate::tool_exec::ApprovalIntent {
                title: "T".to_string(),
                body: vec!["cmd".to_string()],
                name: "stop_server".to_string(),
                arguments: serde_json::json!({ "service_id": "x" }),
            });
            assert!(s.approval.is_some(), "modal open before escape");
            let verdict = s.on_approval_key(key);
            // Esc → Cancel, 'n' → Deny; both are closing (non-Approve) verdicts.
            assert!(
                matches!(
                    verdict,
                    Some(
                        crate::ui::approval::ApprovalVerdict::Cancel
                            | crate::ui::approval::ApprovalVerdict::Deny
                    )
                ),
                "key {key:?} must yield a closing verdict, got {verdict:?}"
            );
            // The event loop routes Deny|Cancel through on_approval_declined.
            s.on_approval_declined();
            assert!(
                s.approval.is_none(),
                "escape must clear the modal (no focus trap) for {key:?}"
            );
            // The covered tab is preserved — the modal never navigated away.
            assert_eq!(s.active_tab, ActiveTab::Observe);
        }
    }

    #[test]
    fn approval_result_fires_exactly_one_follow_up_no_loop() {
        // (c) exactly one follow-up: on_approval_result appends the result turn
        // AND raises chat_dispatch exactly once; it must not re-trigger itself.
        let mut s = st();
        assert!(!s.chat_dispatch);
        s.on_approval_result("Approved · install_sdk: done".to_string());
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(s.chat_dispatch, "exactly one follow-up edge raised");
        assert!(s.chat_sending, "in-flight mirrors a normal submit");

        // Simulate the event loop consuming the edge once.
        s.chat_dispatch = false;
        // A subsequent tick must NOT re-raise it on its own (no self-loop): the
        // only thing that re-raises is another explicit result/submit.
        assert!(!s.chat_dispatch, "follow-up does not re-trigger itself");
    }

    #[test]
    fn approval_key_tab_moves_choice_without_verdict() {
        let mut s = st();
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "T".to_string(),
            body: vec!["cmd".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({}),
        });
        // Defaults to Deny (the safer default; see item #16).
        assert_eq!(
            s.approval.as_ref().unwrap().choice,
            crate::ui::approval::ApprovalChoice::Deny
        );
        // Tab toggles the cursor to Approve without producing a verdict.
        assert_eq!(s.on_approval_key(crossterm::event::KeyCode::Tab), None);
        assert_eq!(
            s.approval.as_ref().unwrap().choice,
            crate::ui::approval::ApprovalChoice::Approve
        );
        // Enter now confirms Approve.
        assert_eq!(
            s.on_approval_key(crossterm::event::KeyCode::Enter),
            Some(crate::ui::approval::ApprovalVerdict::Approve)
        );
    }
}
