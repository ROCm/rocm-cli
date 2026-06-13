//! Doctor overlay (Phase 3 Wave 2).
//!
//! Runs `rocm doctor` — a read-only environment check — through the job-bridge
//! and shows its streamed output in the shared job console. Read-only, so it
//! needs no approval gate (the gate is for mutating actions only). This is the
//! read-only-report archetype every diagnostic screen reuses.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::state::{SideEffect, State, StateEvent};

use crate::ui::exec::resolve_exe;
use crate::ui::job_console::draw_job_console;
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone, Default)]
pub struct DoctorManagerState {
    /// In-flight (or just-finished) `rocm doctor` job id.
    pub active_job: Option<String>,
}

/// Handle a key while the overlay is open. Mirrors the operational-screen seam.
pub fn on_key(
    doctor: &mut Option<DoctorManagerState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(d) = doctor.as_mut() else {
        return Vec::new();
    };

    if let Some(job_id) = d.active_job.clone() {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return jobs.apply(StateEvent::CancelJob(job_id));
            }
            KeyCode::Char('q') => *doctor = None,
            KeyCode::Esc | KeyCode::Enter
                if jobs.job(&job_id).map(|j| j.is_terminal()).unwrap_or(true) =>
            {
                d.active_job = None;
            }
            // `r` re-runs after a terminal result.
            KeyCode::Char('r') if jobs.job(&job_id).map(|j| j.is_terminal()).unwrap_or(true) => {
                d.active_job = None;
                return run_doctor(d, jobs);
            }
            _ => {}
        }
        return Vec::new();
    }

    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => *doctor = None,
        KeyCode::Enter | KeyCode::Char('r') => return run_doctor(d, jobs),
        _ => {}
    }
    Vec::new()
}

/// Spawn `rocm doctor` (read-only). A stable id replaces any prior console.
fn run_doctor(d: &mut DoctorManagerState, jobs: &mut State) -> Vec<SideEffect> {
    let cmd = resolve_exe();
    let id = "doctor".to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd,
        args: vec!["doctor".to_string()],
    });
    if fx.is_empty() {
        // A prior doctor run is still going; keep showing it.
        d.active_job = Some(id);
        return fx;
    }
    d.active_job = Some(id);
    fx
}

/// Render the overlay (intro card, or the job console once running).
pub fn draw_doctor_manager(
    f: &mut Frame,
    area: Rect,
    d: &DoctorManagerState,
    jobs: &State,
    theme: &Theme,
) {
    if let Some(job_id) = &d.active_job
        && let Some(job) = jobs.job(job_id)
    {
        draw_job_console(f, area, job, 0, theme);
        return;
    }

    let popup = centered_rect(70, 50, 90, 14, area);
    let inner = draw_popup_frame(f, popup, "Doctor — environment check", theme);
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
                "  [ Enter: run `rocm doctor` ]  ",
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
            "Enter run · Esc close",
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
    fn enter_runs_doctor_read_only_no_approval() {
        let mut d = Some(DoctorManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1, "spawns one job, no approval step");
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        assert_eq!(d.as_ref().unwrap().active_job.as_deref(), Some("doctor"));
    }

    #[test]
    fn esc_closes_when_idle() {
        let mut d = Some(DoctorManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        assert!(d.is_none());
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut d = Some(DoctorManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        on_key(&mut d, &mut jobs, key(KeyCode::Char('q')));
        assert!(d.is_none());
    }

    #[test]
    fn esc_dismisses_console_only_when_terminal() {
        let mut d = Some(DoctorManagerState::default());
        let mut jobs = State::default();
        on_key(&mut d, &mut jobs, key(KeyCode::Enter));
        // Running: Esc does not dismiss.
        on_key(&mut d, &mut jobs, key(KeyCode::Esc));
        assert!(d.as_ref().unwrap().active_job.is_some());
        // Terminal: Esc returns to the intro card.
        jobs.apply(StateEvent::JobDone {
            id: "doctor".into(),
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
        let d = DoctorManagerState::default();
        let jobs = State::default();
        term.draw(|f| draw_doctor_manager(f, f.area(), &d, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(out.contains("Doctor"));
        assert!(out.contains("rocm doctor"));
    }
}
