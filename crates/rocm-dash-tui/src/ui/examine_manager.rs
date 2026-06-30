// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Examine overlay (Phase 3 Wave 2).
//!
//! Runs `rocm examine` — a read-only environment check — through the job-bridge
//! and shows its streamed output in the shared job console. Read-only, so it
//! needs no approval gate (the gate is for mutating actions only). This is the
//! read-only-report archetype every diagnostic screen reuses.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::state::{SideEffect, State, StateEvent};

use crate::ui::exec::resolve_exe;
use crate::ui::job_console::{ConsoleOutcome, on_console_key};
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone, Default)]
pub struct ExamineManagerState {
    /// In-flight (or just-finished) `rocm examine` job id.
    pub active_job: Option<String>,
}

/// Handle a key while the overlay is open. Mirrors the operational-screen seam.
pub fn on_key(
    examine: &mut Option<ExamineManagerState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(d) = examine.as_mut() else {
        return Vec::new();
    };

    if let Some(job_id) = d.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *examine = None,
            ConsoleOutcome::Dismissed => d.active_job = None,
            ConsoleOutcome::Unhandled => {
                // `r` re-runs after a terminal result.
                if key.code == KeyCode::Char('r')
                    && jobs
                        .job(&job_id)
                        .is_none_or(rocm_dash_core::state::JobState::is_terminal)
                {
                    d.active_job = None;
                    return run_examine(d, jobs);
                }
            }
        }
        return Vec::new();
    }

    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => *examine = None,
        KeyCode::Enter | KeyCode::Char('r') => return run_examine(d, jobs),
        _ => {}
    }
    Vec::new()
}

/// Spawn `rocm examine` (read-only). A stable id replaces any prior console.
fn run_examine(d: &mut ExamineManagerState, jobs: &mut State) -> Vec<SideEffect> {
    let cmd = resolve_exe();
    let id = "examine".to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd,
        args: vec!["examine".to_string()],
    });
    // Examine uses a single stable id, so a no-op (a prior run still going)
    // means re-attach to that same console — intentional, unlike the
    // distinct-id screens (serve/engine/update) where a no-op surfaces an
    // "already running" message instead. Either way `active_job` points at the
    // live job, never a stale one.
    d.active_job = Some(id);
    fx
}

/// Render the overlay (intro card, or the job console once running).
pub fn draw_examine_manager(
    f: &mut Frame,
    area: Rect,
    _d: &ExamineManagerState,
    _jobs: &State,
    theme: &Theme,
) {
    let inner = panel::bento(
        f,
        area,
        Some("Examine — environment check"),
        BoxRole::Primary,
        false,
        theme,
    );
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "Checks this machine's GPU, ROCm install, engines, and folders.",
                Style::default().fg(theme.fg),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Read-only — nothing is changed.",
                Style::default().fg(theme.muted),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  [ Enter: run `rocm examine` ]  ",
                Style::default()
                    .bg(theme.accent)
                    .fg(theme.bg)
                    .add_modifier(Modifier::BOLD),
            )),
        ]),
        rows[0],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Enter/r run · Esc close",
            Style::default().fg(theme.muted),
        ))),
        rows[1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    #[test]
    fn enter_runs_examine_read_only_no_approval() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1, "spawns one job, no approval step");
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        assert_eq!(d.as_ref().unwrap().active_job.as_deref(), Some("examine"));
    }

    #[test]
    fn esc_closes_when_idle() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        assert!(d.is_none());
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        on_key(&mut d, &mut jobs, key(KeyCode::Char('q')));
        assert!(d.is_none());
    }

    #[test]
    fn r_reruns_after_a_terminal_result() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter)); // first run
        jobs.apply(StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        // `r` on a terminal job re-runs (spawns again).
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('r')));
        assert_eq!(fx.len(), 1, "r re-runs after a terminal result");
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        assert_eq!(d.as_ref().unwrap().active_job.as_deref(), Some("examine"));
    }

    #[test]
    fn r_at_idle_runs_examine() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Char('r')));
        assert_eq!(fx.len(), 1);
        assert_eq!(d.as_ref().unwrap().active_job.as_deref(), Some("examine"));
    }

    #[test]
    fn esc_dismisses_console_only_when_terminal() {
        let mut d = Some(ExamineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        // Running: Esc does not dismiss.
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        assert!(d.as_ref().unwrap().active_job.is_some());
        // Terminal: Esc returns to the intro card.
        jobs.apply(StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        assert!(d.as_ref().unwrap().active_job.is_none());
        assert!(d.is_some());
    }

    #[test]
    fn snapshot_intro_then_console() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(100, 20);
        let mut term = Terminal::new(backend).unwrap();
        let d = ExamineManagerState::default();
        let jobs = State::default();
        term.draw(|f| draw_examine_manager(f, f.area(), &d, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Examine"));
        assert!(out.contains("rocm examine"));
    }
}
