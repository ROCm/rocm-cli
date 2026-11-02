// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Focused-host policy: what a dashboard launched with an explicit `Focus`
//! (Setup, Serve or Examine) does differently from the full dashboard — skip
//! the daemon client and chat backend, open its one overlay at startup, and
//! swallow the console keys that would tear down a running job.
//!
//! A leaf module, split out of `app/event_loop.rs`, whose `run` and event loop
//! call these helpers.

use crossterm::event::KeyCode;

use crate::ui;

use super::{AppState, Focus};

/// Whether the event loop should skip the embedded daemon client AND the chat
/// backend resolution. True exactly when a [`Focus`] is set: a focused host runs
/// one overlay that streams its own job through the job-bridge, so it needs
/// neither live telemetry nor an LLM. `focus == None` (the dashboard) keeps both.
/// Pure predicate → unit-testable without a runtime; also names the render branch
/// (`draw_focused` when true, `draw` when false).
pub(super) const fn should_skip_daemon(focus: Option<Focus>) -> bool {
    focus.is_some()
}

/// In a focused host, whether a console "close" key must be SWALLOWED because
/// the active job is still running.
///
/// In the dashboard, `q` / running-`Esc` detach the console and leave the job
/// running in the background (the app persists). A focused host has no
/// background: closing the overlay trips the [`AppState::focused_should_exit`]
/// gate and returns from `event_loop`, tearing down the runtime and killing the
/// child via `kill_on_drop` — truncating a mutating install/serve mid-write. So
/// while the job is non-terminal we swallow those keys; the user stops a job
/// explicitly with `Ctrl+C` (never blocked here), and once it is terminal `q` /
/// `Esc` exit normally. Always `false` for the dashboard (`focus == None`).
pub(super) fn focused_close_key_blocked(
    state: &AppState,
    focus: Option<Focus>,
    code: KeyCode,
) -> bool {
    if !should_skip_daemon(focus) {
        return false;
    }
    let running = state
        .active_job_id()
        .is_some_and(|id| ui::job_console::console_esc_closes(state.jobs.job(id)));
    running && matches!(code, KeyCode::Char('q') | KeyCode::Esc)
}

/// Open the single overlay a focused host should host, returning any initial
/// job-bridge side effects to pump (Examine auto-runs `rocm examine` on open;
/// Setup/Serve open their form and wait for input). Clears any other overlay
/// first (mutually-exclusive invariant). Pure w.r.t. process I/O — the caller
/// runs the returned effects through [`crate::jobs::run_effects`].
pub(super) fn open_overlay_for_focus(
    state: &mut AppState,
    focus: Focus,
) -> Vec<rocm_dash_core::state::SideEffect> {
    state.close_overlays();
    match focus {
        Focus::Setup => {
            state.onboarding = Some(crate::ui::onboarding::OnboardingState::default());
            Vec::new()
        }
        Focus::Serve => {
            state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState::default());
            Vec::new()
        }
        Focus::Examine => {
            let (mgr, fx) = crate::ui::examine_manager::open_running(&mut state.jobs);
            state.examine_manager = Some(mgr);
            fx
        }
    }
}

/// The dashboard's entire startup gate: an overlay opens only when an
/// explicit `Focus` was resolved from the command line. `event_loop` calls
/// this directly (rather than inlining the `match`) so a regression test can
/// exercise the actual gate instead of `AppState::new`, which takes no focus
/// argument and can't observe it either way.
pub(super) fn apply_startup_focus(
    state: &mut AppState,
    focus: Option<Focus>,
) -> Vec<rocm_dash_core::state::SideEffect> {
    match focus {
        Some(focus) => open_overlay_for_focus(state, focus),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::test_support::{press, st};

    #[test]
    fn startup_focus_gate_only_opens_onboarding_for_explicit_setup_focus() {
        // Regression guard: this calls `apply_startup_focus`, the same gate
        // `event_loop` uses, not just `AppState::new` — which takes no focus
        // argument and hardcodes `onboarding: None` regardless, so it cannot
        // exhibit an auto-open regression either way.
        let mut s = st();
        assert!(apply_startup_focus(&mut s, None).is_empty());
        assert!(
            s.onboarding.is_none(),
            "no --focus flag must not open onboarding"
        );

        let mut s = st();
        assert!(apply_startup_focus(&mut s, Some(Focus::Setup)).is_empty());
        assert!(
            s.onboarding.is_some(),
            "an explicit Focus::Setup must open onboarding"
        );
    }

    #[test]
    fn should_skip_daemon_predicate_matches_focus() {
        // The dashboard (focus=None) keeps the daemon client + chat backend; any
        // focus skips both. The render branch reuses this same predicate.
        assert!(!should_skip_daemon(None));
        assert!(should_skip_daemon(Some(Focus::Setup)));
        assert!(should_skip_daemon(Some(Focus::Serve)));
        assert!(should_skip_daemon(Some(Focus::Examine)));
        // focus=None never self-exits — the dash loop only breaks on Quit/EOF.
        assert!(!st().focused_should_exit(None));
    }

    #[test]
    fn open_overlay_for_focus_opens_the_right_overlay() {
        let mut s = st();
        assert!(open_overlay_for_focus(&mut s, Focus::Setup).is_empty());
        assert!(s.onboarding.is_some());
        assert!(s.serve_wizard.is_none() && s.examine_manager.is_none());

        let mut s = st();
        assert!(open_overlay_for_focus(&mut s, Focus::Serve).is_empty());
        assert!(s.serve_wizard.is_some());
        assert!(s.onboarding.is_none() && s.examine_manager.is_none());

        let mut s = st();
        let fx = open_overlay_for_focus(&mut s, Focus::Examine);
        assert!(s.examine_manager.is_some());
        assert!(s.onboarding.is_none() && s.serve_wizard.is_none());
        assert_eq!(fx.len(), 1, "examine auto-runs on open");
    }

    #[test]
    fn focused_examine_auto_runs_rocm_examine() {
        let mut s = st();
        let fx = open_overlay_for_focus(&mut s, Focus::Examine);
        assert_eq!(fx.len(), 1, "exactly one spawn side effect on open");
        match &fx[0] {
            rocm_dash_core::state::SideEffect::SpawnJob { cmd, args, .. } => {
                assert!(cmd.contains("rocm"), "cmd resolves to the rocm exe: {cmd}");
                assert!(
                    args.iter().any(|a| a == "examine"),
                    "examine in args: {args:?}"
                );
            }
            other => panic!("expected SpawnJob, got {other:?}"),
        }
        assert_eq!(
            s.examine_manager.as_ref().unwrap().active_job.as_deref(),
            Some("examine"),
            "the auto-run wires the active job"
        );
    }

    #[test]
    fn focused_exit_gate_holds_until_examine_closed_at_root() {
        let mut s = st();
        // Focused Diagnose: examine opens AND auto-runs → a job-console sub-state.
        let _ = open_overlay_for_focus(&mut s, Focus::Examine);
        assert!(s.examine_manager.as_ref().unwrap().active_job.is_some());
        assert!(
            !s.focused_should_exit(Some(Focus::Examine)),
            "a running job keeps the launcher out"
        );

        // Job terminal → first Esc dismisses the console back to the intro card;
        // the overlay is still open, so the gate stays shut.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        let _ = crate::ui::examine_manager::on_key(
            &mut s.examine_manager,
            &mut s.jobs,
            press(KeyCode::Esc),
        );
        assert!(
            s.examine_manager.is_some(),
            "console dismissed, overlay stays"
        );
        assert!(
            !s.focused_should_exit(Some(Focus::Examine)),
            "at the intro (not root-closed) the gate is still shut"
        );

        // Second Esc at the intro (root) closes the overlay → now exit to menu.
        let _ = crate::ui::examine_manager::on_key(
            &mut s.examine_manager,
            &mut s.jobs,
            press(KeyCode::Esc),
        );
        assert!(s.examine_manager.is_none(), "root Esc closes the overlay");
        assert!(
            s.focused_should_exit(Some(Focus::Examine)),
            "closed at root → return to the launcher"
        );
    }

    #[test]
    fn focused_should_exit_stays_shut_while_quit_confirm_is_pending() {
        // `open_quit_confirm` closes every manager overlay via `close_overlays`,
        // so `has_open_overlay()` goes false the instant it is called. Without
        // `focused_should_exit` also checking `quit_confirm_pending()`, a
        // focused host reaching `actions::request_quit` would hit this gate on
        // the very next loop iteration and exit before `draw_focused` ever drew
        // the prompt — silently, instead of showing it.
        let mut s = st();
        let _ = open_overlay_for_focus(&mut s, Focus::Serve);
        assert!(s.has_open_overlay());

        s.open_quit_confirm();
        assert!(
            !s.has_open_overlay(),
            "open_quit_confirm closes every manager overlay"
        );
        assert!(
            !s.focused_should_exit(Some(Focus::Serve)),
            "a pending quit-confirm must hold the focused-exit gate shut"
        );

        // Decline (Esc cancels) resolves the prompt without exiting.
        assert_eq!(
            s.resolve_quit_confirm_key(KeyCode::Esc),
            Some(false),
            "Esc cancels the quit-confirm prompt rather than exiting"
        );
        assert!(
            s.focused_should_exit(Some(Focus::Serve)),
            "once the prompt is resolved and no overlay is open, the gate opens again"
        );
    }

    #[test]
    fn focused_close_keys_swallowed_while_job_runs() {
        // Regression for the mid-job ejection defect: `q` and running-`Esc` must
        // be swallowed by the focused host while the job is non-terminal, so the
        // overlay is never nulled (which would tear the runtime down and kill the
        // child via kill_on_drop mid-write).
        let mut s = st();
        let _ = open_overlay_for_focus(&mut s, Focus::Examine); // auto-runs a job
        assert!(s.has_active_console(), "examine console is live");
        // Running job → q and Esc are blocked; Ctrl+C ('c') is NOT (it cancels).
        assert!(focused_close_key_blocked(
            &s,
            Some(Focus::Examine),
            KeyCode::Char('q')
        ));
        assert!(focused_close_key_blocked(
            &s,
            Some(Focus::Examine),
            KeyCode::Esc
        ));
        assert!(!focused_close_key_blocked(
            &s,
            Some(Focus::Examine),
            KeyCode::Char('c')
        ));
        // The dashboard (focus=None) never blocks — behavior is unchanged there.
        assert!(!focused_close_key_blocked(&s, None, KeyCode::Char('q')));

        // Because those keys are swallowed (never routed to the manager), the
        // overlay stays open and the exit gate stays shut mid-job.
        assert!(s.examine_manager.is_some());
        assert!(!s.focused_should_exit(Some(Focus::Examine)));

        // Once the job is terminal, close keys are allowed again → normal exit.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        assert!(
            !focused_close_key_blocked(&s, Some(Focus::Examine), KeyCode::Char('q')),
            "a terminal job no longer blocks exit (the child already exited)"
        );
    }

    #[test]
    fn focused_gate_shut_across_serve_sub_states() {
        // Exit-at-root (b)+(c): the focused gate stays shut while a folder
        // browser / model picker / approval is open — it only opens at root.
        let recipes: Vec<crate::ui::model_picker::ModelRecipeSummary> = Vec::new();
        let mut s = st();
        let _ = open_overlay_for_focus(&mut s, Focus::Serve);

        // (b) Tab on the Model field opens the folder-browser sub-popup.
        let _ = crate::ui::serve_wizard::on_key(
            &mut s.serve_wizard,
            &mut s.jobs,
            &recipes,
            press(KeyCode::Tab),
        );
        assert!(s.serve_wizard.as_ref().unwrap().browser.is_some());
        assert!(
            !s.focused_should_exit(Some(Focus::Serve)),
            "gate shut while the folder browser is open"
        );
        // Esc closes the sub-popup, not the wizard → still shut.
        let _ = crate::ui::serve_wizard::on_key(
            &mut s.serve_wizard,
            &mut s.jobs,
            &recipes,
            press(KeyCode::Esc),
        );
        assert!(s.serve_wizard.as_ref().unwrap().browser.is_none());
        assert!(s.serve_wizard.is_some());
        assert!(!s.focused_should_exit(Some(Focus::Serve)));

        // (c) Stage an approval (valid model, Launch field, Enter).
        {
            let w = s.serve_wizard.as_mut().unwrap();
            w.model = "org/model".to_string();
            w.field = crate::ui::serve_wizard::FIELDS.len() - 1; // Launch
        }
        let _ = crate::ui::serve_wizard::on_key(
            &mut s.serve_wizard,
            &mut s.jobs,
            &recipes,
            press(KeyCode::Enter),
        );
        assert!(
            s.serve_wizard.as_ref().unwrap().approval.is_some(),
            "a launch approval is pending"
        );
        assert!(
            !s.focused_should_exit(Some(Focus::Serve)),
            "gate shut while an approval is pending"
        );

        // Only a root close (wizard → None) opens the gate.
        s.serve_wizard = None;
        assert!(s.focused_should_exit(Some(Focus::Serve)));
    }
}
