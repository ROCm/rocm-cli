// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Job console widget (Phase 3 Wave 0).
//!
//! Renders a [`JobState`] from the reducer: a status header, the streamed
//! output ring, and key hints. This is the shared "running job" surface every
//! operational screen reuses instead of the frozen rocm-cli `running_job`
//! modal. The widget is read-only over the reducer's job model.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::state::{JobState, JobStatus, SideEffect, State, StateEvent};

use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// What a console keypress means to the owning screen.
///
/// The shared seam every operational overlay routes its `active_job` keys through, so the
/// Ctrl+C/`q`/Esc-Enter behavior is defined once instead of per screen.
#[derive(Debug)]
pub enum ConsoleOutcome {
    /// Ctrl+C cancelled the running job — the caller runs these effects.
    Cancelled(Vec<SideEffect>),
    /// `q` — the caller closes the whole overlay.
    Closed,
    /// Esc/Enter on a terminal job — the caller dismisses the console (returns
    /// to the screen body), and may clear any transient message.
    Dismissed,
    /// Not a console key — the caller may handle it (e.g. a screen-specific
    /// re-run shortcut).
    Unhandled,
}

/// Interpret a key while a job console is showing `job_id`. Pure except for the
/// `CancelJob` reducer apply (which only mutates the in-memory job model).
pub fn on_console_key(job_id: &str, jobs: &mut State, key: KeyEvent) -> ConsoleOutcome {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            ConsoleOutcome::Cancelled(jobs.apply(StateEvent::CancelJob(job_id.to_string())))
        }
        // `q` always closes the overlay so the user is never trapped mid-job
        // (the job keeps running in the background).
        KeyCode::Char('q') => ConsoleOutcome::Closed,
        KeyCode::Esc | KeyCode::Enter
            if jobs
                .job(job_id)
                .is_none_or(rocm_dash_core::state::JobState::is_terminal) =>
        {
            ConsoleOutcome::Dismissed
        }
        _ => ConsoleOutcome::Unhandled,
    }
}

/// Human-readable status label + the color it should render in.
pub fn status_label(job: &JobState, theme: &Theme) -> (String, ratatui::style::Color) {
    match &job.status {
        JobStatus::Running => ("running".to_string(), theme.accent),
        JobStatus::Done { code: 0 } => ("done".to_string(), theme.ok),
        JobStatus::Done { code } => (format!("exited ({code})"), theme.warn),
        JobStatus::Failed { message } => (format!("failed: {message}"), theme.err),
        JobStatus::Cancelled => ("cancelled".to_string(), theme.muted),
    }
}

/// Render the job console centered over `area`. `scroll` is the first visible
/// output line offset (the caller clamps it).
pub fn draw_job_console(f: &mut Frame, area: Rect, job: &JobState, scroll: u16, theme: &Theme) {
    let popup = centered_rect(90, 84, 140, 40, area);
    let title = format!("{} {}", job.cmd, job.args.join(" "));
    let inner = draw_popup_frame(f, popup, title.trim(), theme);
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);

    // Header: status badge.
    let (label, color) = status_label(job, theme);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " status ",
                Style::default()
                    .fg(theme.bg)
                    .bg(color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                label,
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
        ])),
        rows[0],
    );

    // Body: streamed output lines.
    let lines: Vec<Line> = job
        .output
        .iter()
        .map(|l| Line::from(Span::styled(l.clone(), Style::default().fg(theme.fg))))
        .collect();
    f.render_widget(Paragraph::new(lines).scroll((scroll, 0)), rows[1]);

    // Footer: key hints (cancel only while running).
    let hints = if matches!(job.status, JobStatus::Running) {
        "Ctrl+C cancel · PgUp/PgDn scroll"
    } else {
        "Enter/Esc close · PgUp/PgDn scroll"
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            hints,
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_dash_core::state::{State, StateEvent};

    fn theme() -> Theme {
        Theme::from_name("default")
    }

    fn k(code: crossterm::event::KeyCode) -> crossterm::event::KeyEvent {
        crossterm::event::KeyEvent::new(code, crossterm::event::KeyModifiers::NONE)
    }

    #[test]
    fn console_key_outcomes() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut s = State::default();
        s.apply(StateEvent::StartJob {
            id: "j".into(),
            cmd: "sleep".into(),
            args: vec!["1".into()],
        });
        // Ctrl+C cancels (returns effects).
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(matches!(
            on_console_key("j", &mut s, ctrl_c),
            ConsoleOutcome::Cancelled(_)
        ));
        // `q` closes regardless of job state.
        assert!(matches!(
            on_console_key("j", &mut s, k(KeyCode::Char('q'))),
            ConsoleOutcome::Closed
        ));
        // Esc on a RUNNING job is not a dismiss (Unhandled).
        let mut s2 = State::default();
        s2.apply(StateEvent::StartJob {
            id: "j".into(),
            cmd: "x".into(),
            args: vec![],
        });
        assert!(matches!(
            on_console_key("j", &mut s2, k(KeyCode::Esc)),
            ConsoleOutcome::Unhandled
        ));
        // Esc on a TERMINAL job dismisses.
        s2.apply(StateEvent::JobDone {
            id: "j".into(),
            code: 0,
        });
        assert!(matches!(
            on_console_key("j", &mut s2, k(KeyCode::Esc)),
            ConsoleOutcome::Dismissed
        ));
        // A missing job id is treated as terminal → Esc dismisses.
        assert!(matches!(
            on_console_key("gone", &mut s2, k(KeyCode::Enter)),
            ConsoleOutcome::Dismissed
        ));
    }

    #[test]
    fn status_labels_track_lifecycle() {
        let mut s = State::default();
        s.apply(StateEvent::StartJob {
            id: "j".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        });
        let t = theme();
        assert_eq!(status_label(s.job("j").unwrap(), &t).0, "running");
        s.apply(StateEvent::JobDone {
            id: "j".into(),
            code: 0,
        });
        assert_eq!(status_label(s.job("j").unwrap(), &t).0, "done");
    }
}
