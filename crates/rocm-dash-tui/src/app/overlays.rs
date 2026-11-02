// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Overlay bookkeeping and input gating: which manager, console, approval or
//! modal currently owns the keyboard, closing them, and the predicates the key
//! handlers and the renderer consult to decide whether a key reaches the
//! dashboard's global hotkeys or backs out of an overlay.
//!
//! A further `impl AppState` block, split out of `app/mod.rs` to keep the core
//! reducer focused.

use super::{ActiveTab, AppState, Focus, Modal};

impl AppState {
    /// Close every operational overlay. The overlays are mutually exclusive
    /// (only one is routed/drawn at a time), so opening any one first clears the
    /// rest — no open path can leave two `Some` at once.
    pub(super) fn close_overlays(&mut self) {
        self.services = None;
        self.serve_wizard = None;
        self.engine_manager = None;
        self.examine_manager = None;
        self.update_manager = None;
        self.install_manager = None;
        self.logs_view = None;
        self.runtime_manager = None;
        self.onboarding = None;
        self.automations_manager = None;
        self.command_screen = None;
        self.config_manager = None;
        self.bench_run = None;
        self.approval = None;
        self.quit_confirm = None;
        // A fresh overlay starts its console at the top.
        self.console_scroll = 0;
        self.console_hscroll = 0;
    }

    /// The job id of the manager that is currently showing its console (if any).
    /// Only one manager is open at a time, so at most one matches; `None` when no
    /// overlay is open or the open one is still on its form screen.
    pub(crate) fn active_job_id(&self) -> Option<&str> {
        self.services
            .as_ref()
            .and_then(|m| m.active_job.as_deref())
            .or_else(|| {
                self.serve_wizard
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.engine_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.examine_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.update_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.install_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.logs_view
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.runtime_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.onboarding
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.automations_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.command_screen
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.config_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
    }

    /// Whether a job console is currently displayed (a manager is open AND on its
    /// console sub-view). Gates scroll routing so wheel/PgUp-PgDn pan the log
    /// instead of moving the obscured Actions list.
    pub(crate) fn has_active_console(&self) -> bool {
        self.active_job_id().is_some()
    }

    /// Whether any operational manager overlay is open (approval excluded — it
    /// is the separate gating layer with its own routing). Used to decide inline
    /// vs. centered manager rendering and the ROCm/Serving `←`/Esc back-out.
    pub(crate) const fn has_open_overlay(&self) -> bool {
        self.services.is_some()
            || self.serve_wizard.is_some()
            || self.engine_manager.is_some()
            || self.examine_manager.is_some()
            || self.update_manager.is_some()
            || self.install_manager.is_some()
            || self.logs_view.is_some()
            || self.runtime_manager.is_some()
            || self.onboarding.is_some()
            || self.automations_manager.is_some()
            || self.command_screen.is_some()
            || self.config_manager.is_some()
            || self.bench_run.is_some()
    }

    /// Whether a chat tool-call approval is pending. Its own gating layer,
    /// separate from [`has_open_overlay`](Self::has_open_overlay) — a real
    /// keypress or click can never reach `OpenThemePicker`/`ToggleHelp`/`Quit`/
    /// a scrollbar/the pane body while this is `true`, so every input path
    /// that swallows for an open overlay must also check this, or a mouse
    /// gesture could bypass a gate no keypress ever could. Single source of
    /// truth for that check so the call sites can't drift apart.
    pub(crate) const fn approval_pending(&self) -> bool {
        self.approval.is_some()
    }

    /// Whether a modal that owns the body absolutely — no console, nothing to
    /// pan or fall through to — is open: the chat approval or the quit-confirm
    /// prompt. Unlike [`overlay_or_approval`](Self::overlay_or_approval), this
    /// deliberately excludes [`has_open_overlay`](Self::has_open_overlay): a
    /// manager overlay's job console is still a legitimate scroll/drag target,
    /// where these two never are.
    pub(crate) const fn blocks_body_absolutely(&self) -> bool {
        self.approval_pending() || self.quit_confirm_pending()
    }

    /// Whether *any* gating layer owns the screen: an open manager overlay, a
    /// pending chat approval, or a pending quit-confirm prompt. This exact
    /// `||` is what every input path that swallows for one must also swallow
    /// for the others — call sites wrote it out by hand before this existed,
    /// each with its own copy of this same reasoning; forgetting one would
    /// reopen the class of bug `approval_pending`'s own doc comment
    /// describes. Composed from [`has_open_overlay`](Self::has_open_overlay)
    /// and [`blocks_body_absolutely`](Self::blocks_body_absolutely) rather
    /// than repeating their two sub-conditions by hand, so a future third
    /// "owns-the-screen" gate only has to be added to one of the two
    /// functions, not to both. The name predates the quit-confirm prompt;
    /// kept as-is to avoid rippling a rename through every call site's
    /// comments.
    pub(crate) const fn overlay_or_approval(&self) -> bool {
        self.has_open_overlay() || self.blocks_body_absolutely()
    }

    /// Whether a global hotkey reaches its normal `handle_key` arm right now,
    /// or something else owns the keyboard first: a manager overlay or pending
    /// approval/quit-confirm owns every key, a plain `Modal::*` returns from
    /// its own arm, and focused chat text entry consumes every `Char`. Anything
    /// that dispatches a global hotkey action from a click (footer chips, the
    /// header warning badge) must gate on this, or the click can do what the
    /// key it stands for never could.
    pub(crate) const fn global_hotkeys_reachable(&self) -> bool {
        let chat_typing = matches!(self.active_tab, ActiveTab::Chat) && self.chat_focused;
        !(self.overlay_or_approval() || !matches!(self.modal, Modal::None) || chat_typing)
    }

    /// Number of warnings in the latest snapshot — what the header badge shows.
    pub(crate) fn warning_count(&self) -> usize {
        self.latest.as_ref().map_or(0, |s| s.warnings.len())
    }

    /// Focused-host exit gate: `true` when a `focus` is active AND its single
    /// overlay is closed (no manager is `Some`) AND no quit-confirm prompt is
    /// pending.
    ///
    /// [`has_open_overlay`](Self::has_open_overlay) stays `true` while a
    /// sub-popup (folder browser / model picker) is open, so this can't fire
    /// while the user is inside one of those. It does NOT by itself protect a
    /// running job console: the shared console maps `q` / running-`Esc` to
    /// "close overlay", which would null the manager mid-job. That case is
    /// handled upstream in `event_loop` by `focused_close_key_blocked`, which
    /// swallows those keys while the job is non-terminal — so by the time this
    /// gate is checked, a focused overlay only ever closed at its root (form
    /// screen or a terminal job). Always `false` for the normal
    /// (`focus == None`) dashboard, so its loop never self-exits. Pure read →
    /// unit-testable without a live terminal.
    ///
    /// The `quit_confirm_pending` exemption matters only if a focused host
    /// ever calls `actions::request_quit` (none does today): `open_quit_confirm`
    /// closes every manager overlay via `close_overlays`, which without this
    /// exemption would make this gate fire on the very next loop iteration —
    /// before `draw_focused` ever renders the prompt it draws for that case —
    /// exiting silently instead of showing it. Deliberately NOT folded into
    /// `has_open_overlay` itself: that predicate also gates unrelated reads
    /// (`should_pane_back_out`, `active_overlay_at_root`'s callers) where a
    /// pending quit-confirm must NOT count as an open manager overlay.
    pub(crate) const fn focused_should_exit(&self, focus: Option<Focus>) -> bool {
        focus.is_some() && !self.has_open_overlay() && !self.quit_confirm_pending()
    }

    /// Whether the open manager (if any) is at its TOP-LEVEL screen — no nested
    /// sub-popup (folder browser / model picker / import input), no pending
    /// gating approval, and no job console (running or terminal). Only one
    /// manager is open at a time, so this reflects that one; `true` when none is
    /// open. Gates the Esc back-out so Esc cancels the innermost layer first
    /// (and is ignored while a job runs) before it can eject the manager.
    pub(crate) fn active_overlay_at_root(&self) -> bool {
        self.serve_wizard.as_ref().is_none_or(|w| {
            w.browser.is_none()
                && w.picker.is_none()
                && w.approval.is_none()
                && w.active_job.is_none()
        }) && self
            .install_manager
            .as_ref()
            .is_none_or(|m| m.browser.is_none() && m.approval.is_none() && m.active_job.is_none())
            && self.onboarding.as_ref().is_none_or(|m| {
                m.browser.is_none()
                    && m.install_config.is_none()
                    && m.approval.is_none()
                    && m.active_job.is_none()
            })
            && self.runtime_manager.as_ref().is_none_or(|m| {
                m.browser.is_none()
                    && m.import_input.is_none()
                    && m.approval.is_none()
                    && m.active_job.is_none()
            })
            && self
                .engine_manager
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .services
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .update_manager
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .config_manager
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .command_screen
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .automations_manager
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .examine_manager
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .logs_view
                .as_ref()
                .is_none_or(|m| m.active_job.is_none())
        // bench_run is always at root when Some (no nested sub-popup or job).
    }

    /// Whether an `Esc` keypress should back out of an inline manager: true on
    /// any tab while a manager overlay is open AND that manager is at its root
    /// screen. The event loop closes the manager and returns focus to the
    /// Actions list when this holds. Pure read so it is unit-testable (the
    /// mutation lives in the event-loop arm).
    ///
    /// Not just ROCm/Serving: a manager can be opened from a non-domain tab
    /// (e.g. `examine_manager` from an Observe hotkey). This used to be gated
    /// on `active_tab == Rocm | Serving`, so on other tabs the manager's own
    /// event-loop arm handled root Esc directly (every overlay type already
    /// has a dedicated `Some(Ok(CtEvent::Key(k))) if state.<overlay>.is_some()`
    /// arm ahead of the generic handler, and each self-closes on root Esc
    /// regardless of `active_tab` — so there was no "Modal stays set but
    /// invisible" bug to fix here — true of every manager except onboarding,
    /// see below). Dropping the tab guard moves the close from the manager's
    /// own `on_key` to this shared path (`close_overlays()` + `pane_focus =
    /// Actions`) so a future manager doesn't need to duplicate that root-Esc
    /// handling. `pane_focus` is meaningless outside Rocm/Serving, so
    /// resetting it there is a harmless no-op.
    ///
    /// This generalization is only correct if `active_overlay_at_root`'s
    /// per-manager clause enumerates every nesting field the manager's state
    /// struct has — see the note on `OnboardingState` (and its sibling
    /// manager-state structs) about keeping that enumeration in sync when a
    /// new nested sub-view field is added. `active_overlay_at_root_enumeration_is_exhaustive`
    /// turns that into a build break instead of a silent drift: it destructures
    /// every one of those structs without `..`, so adding a field to any of
    /// them without updating both the test and this function fails to compile.
    ///
    /// When the manager has a sub-popup / approval / job console open, this is
    /// `false` so Esc falls through to the manager's own handler (cancel the
    /// sub-layer, dismiss a terminal console, or be ignored while a job runs) —
    /// it cannot eject the whole manager mid-flow.
    ///
    /// Only `Esc` backs out — `←` is left to the open manager (serve_wizard /
    /// install / config use it to cycle options). When NO manager is open, `←`
    /// returns focus from the Details preview to the Actions list via the normal
    /// `PaneFocusActions` key path.
    pub(crate) fn should_pane_back_out(&self, code: crossterm::event::KeyCode) -> bool {
        self.has_open_overlay()
            && self.active_overlay_at_root()
            && matches!(code, crossterm::event::KeyCode::Esc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::ChatConsent;
    use crate::app::test_support::st;

    #[test]
    fn global_hotkeys_are_unreachable_while_a_modal_overlay_or_prompt_owns_the_keyboard() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        assert!(s.global_hotkeys_reachable(), "idle dashboard");
        s.modal = Modal::Help;
        assert!(!s.global_hotkeys_reachable(), "a plain modal returns first");
        s.modal = Modal::None;
        s.services = Some(crate::ui::services_manager::ServicesManagerState::default());
        assert!(
            !s.global_hotkeys_reachable(),
            "a manager overlay owns every key"
        );
        s.services = None;
        s.open_quit_confirm();
        assert!(
            !s.global_hotkeys_reachable(),
            "the quit prompt owns every key"
        );
    }

    #[test]
    fn typing_in_chat_owns_the_keyboard_only_on_the_chat_tab_and_only_while_focused() {
        // `handle_key` hands every character to the chat input only once chat
        // is accepted and the input is focused.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_consent = ChatConsent::Accepted;
        s.active_tab = ActiveTab::Chat;
        s.chat_focused = false;
        assert!(s.global_hotkeys_reachable(), "chat tab, input not focused");
        s.chat_focused = true;
        assert!(!s.global_hotkeys_reachable(), "focused chat input");
        // `chat_focused` is not cleared when the user switches tab, so a stale
        // `true` on another tab must not be read as "typing".
        for tab in [
            ActiveTab::Home,
            ActiveTab::Rocm,
            ActiveTab::Serving,
            ActiveTab::Observe,
        ] {
            s.active_tab = tab;
            assert!(
                s.global_hotkeys_reachable(),
                "{tab:?} with a stale chat focus"
            );
        }
    }

    #[test]
    fn back_out_requires_an_open_manager_on_any_tab() {
        let mut s = st();
        // No manager open → never backs out, even on a domain tab.
        s.active_tab = ActiveTab::Rocm;
        assert!(!s.should_pane_back_out(crossterm::event::KeyCode::Esc));
        // Manager open on a non-domain tab (opened from Observe hotkey) →
        // Esc backs out uniformly regardless of tab, now that the
        // Rocm/Serving-only gate is gone. New coverage of the generalized
        // behavior — the manager's own event-loop arm already closed it on
        // this tab before the gate was removed, so this isn't a regression
        // test for a prior bug.
        s.active_tab = ActiveTab::Observe;
        s.examine_manager = Some(crate::ui::examine_manager::ExamineManagerState::default());
        assert!(s.has_open_overlay());
        assert!(s.should_pane_back_out(crossterm::event::KeyCode::Esc));
    }

    #[test]
    fn esc_defers_to_manager_when_a_subscreen_is_open() {
        // With a job console (or sub-popup / approval) open inside an inline
        // manager, Esc must reach the manager (cancel the inner layer / be
        // ignored while running), NOT eject the whole manager.
        let mut s = st();
        s.active_tab = ActiveTab::Rocm;
        s.install_manager = Some(crate::ui::install_manager::InstallManagerState {
            active_job: Some("install-job".into()), // a console is up
            ..Default::default()
        });
        assert!(s.has_open_overlay());
        assert!(
            !s.should_pane_back_out(crossterm::event::KeyCode::Esc),
            "Esc must defer to the manager while a job console is open"
        );
        // Once the console is dismissed (back at root), Esc backs out.
        s.install_manager.as_mut().unwrap().active_job = None;
        assert!(s.should_pane_back_out(crossterm::event::KeyCode::Esc));
    }

    #[test]
    fn esc_defers_to_onboarding_install_config_subview() {
        // Regression coverage for the `install_config` nesting field: the
        // onboarding wizard's Configure sub-view is a nested sub-view just
        // like a manager's job console, so root Esc must defer to it instead
        // of ejecting the whole wizard.
        let mut s = st();
        s.active_tab = ActiveTab::Rocm;
        s.onboarding = Some(crate::ui::onboarding::OnboardingState {
            install_config: Some(crate::ui::onboarding::InstallConfig::default()),
            ..Default::default()
        });
        assert!(s.has_open_overlay());
        assert!(
            !s.should_pane_back_out(crossterm::event::KeyCode::Esc),
            "Esc must defer to onboarding while the Configure sub-view is open"
        );
        // Once the sub-view is closed (back at root), Esc backs out again.
        s.onboarding.as_mut().unwrap().install_config = None;
        assert!(s.should_pane_back_out(crossterm::event::KeyCode::Esc));
    }

    // ---------- T13: bench_run overlay invariants ----------

    #[test]
    fn t13_bench_run_in_has_open_overlay() {
        let mut s = st();
        assert!(!s.has_open_overlay(), "no overlay initially");
        s.bench_run = Some(crate::ui::bench_run::BenchRunState::new(None));
        assert!(
            s.has_open_overlay(),
            "bench_run must be in has_open_overlay"
        );
    }

    #[test]
    fn t13_bench_run_cleared_by_close_overlays() {
        let mut s = st();
        s.bench_run = Some(crate::ui::bench_run::BenchRunState::new(None));
        s.close_overlays();
        assert!(s.bench_run.is_none(), "close_overlays must clear bench_run");
    }

    #[test]
    fn t13_bench_run_at_root() {
        let mut s = st();
        assert!(s.active_overlay_at_root(), "nothing open → at root");
        s.bench_run = Some(crate::ui::bench_run::BenchRunState::new(None));
        assert!(
            s.active_overlay_at_root(),
            "bench_run (no sub-popup/job) is always at root"
        );
    }

    /// `active_overlay_at_root`'s per-manager clauses are a hand-maintained
    /// enumeration of each manager's nested sub-view fields (documented on
    /// `OnboardingState`, which lists every struct this covers). Nothing stops
    /// a future field — a new sub-popup, picker, or prompt — from being added
    /// to one of these structs without a matching update there, which would
    /// silently let Esc eject the whole manager instead of deferring to the
    /// new sub-view.
    ///
    /// This exhaustively destructures every one of those structs (no `..`),
    /// naming every field. Adding a field to any of them without updating
    /// this test — and, in step, `active_overlay_at_root` — fails to compile
    /// (E0027), turning the silent-drift risk into a build break.
    #[test]
    fn active_overlay_at_root_enumeration_is_exhaustive() {
        use crate::ui::automations_manager::AutomationsManagerState;
        use crate::ui::command_screen::CommandScreenState;
        use crate::ui::config_manager::ConfigManagerState;
        use crate::ui::engine_manager::EngineManagerState;
        use crate::ui::examine_manager::ExamineManagerState;
        use crate::ui::install_manager::InstallManagerState;
        use crate::ui::logs_view::LogsViewState;
        use crate::ui::onboarding::OnboardingState;
        use crate::ui::runtime_manager::RuntimeManagerState;
        use crate::ui::serve_wizard::ServeWizardState;
        use crate::ui::services_manager::ServicesManagerState;
        use crate::ui::update_manager::UpdateManagerState;

        let ServeWizardState {
            field: _,
            model: _,
            engine_idx: _,
            device_idx: _,
            host: _,
            port: _,
            managed: _,
            browser,
            picker,
            approval,
            active_job,
            message: _,
        } = ServeWizardState::default();
        assert!(
            browser.is_none() && picker.is_none() && approval.is_none() && active_job.is_none()
        );

        let InstallManagerState {
            field: _,
            channel: _,
            format_idx: _,
            prefix: _,
            dry_run: _,
            browser,
            approval,
            active_job,
            message: _,
        } = InstallManagerState::default();
        assert!(browser.is_none() && approval.is_none() && active_job.is_none());

        let OnboardingState {
            step: _,
            choice: _,
            browser,
            install_config,
            approval,
            active_job,
            message: _,
        } = OnboardingState::default();
        assert!(
            browser.is_none()
                && install_config.is_none()
                && approval.is_none()
                && active_job.is_none()
        );

        let RuntimeManagerState {
            selected: _,
            browser,
            import_input,
            approval,
            active_job,
            message: _,
        } = RuntimeManagerState::default();
        assert!(
            browser.is_none()
                && import_input.is_none()
                && approval.is_none()
                && active_job.is_none()
        );

        let EngineManagerState {
            selected: _,
            approval,
            active_job,
            message: _,
        } = EngineManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let ServicesManagerState {
            selected: _,
            approval,
            active_job,
        } = ServicesManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let UpdateManagerState {
            selected: _,
            approval,
            active_job,
            message: _,
        } = UpdateManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let ConfigManagerState {
            action_sel: _,
            provider_sel: _,
            approval,
            active_job,
            message: _,
        } = ConfigManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let CommandScreenState {
            input: _,
            approval,
            active_job,
            message: _,
        } = CommandScreenState::default();
        assert!(approval.is_none() && active_job.is_none());

        let AutomationsManagerState {
            selected: _,
            approval,
            active_job,
            message: _,
        } = AutomationsManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let ExamineManagerState {
            active_job,
            report: _,
            approval,
            message: _,
        } = ExamineManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let LogsViewState {
            query: _,
            active_job,
        } = LogsViewState::default();
        assert!(active_job.is_none());
    }

    #[test]
    fn close_overlays_clears_pending_approval() {
        // A stale approval modal must not survive close_overlays (focus trap).
        let mut s = st();
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "Install ROCm".to_string(),
            body: vec!["rocm install sdk".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({}),
        });
        assert!(s.approval.is_some(), "approval pending before close");
        s.close_overlays();
        assert!(s.approval.is_none(), "close_overlays must clear the modal");
    }

    #[test]
    fn close_overlays_clears_pending_quit_confirm() {
        // Mirrors close_overlays_clears_pending_approval: quit_confirm is the
        // sibling gating layer and must be cleared the same way, or a future
        // path that reaches close_overlays() while it's open would leave a
        // stale, undismissable prompt.
        let mut s = st();
        s.open_quit_confirm();
        assert!(
            s.quit_confirm.is_some(),
            "quit_confirm pending before close"
        );
        s.close_overlays();
        assert!(
            s.quit_confirm.is_none(),
            "close_overlays must clear quit_confirm"
        );
    }
}
