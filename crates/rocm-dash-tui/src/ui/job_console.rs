//! Job console widget (Phase 3 Wave 0).
//!
//! Renders a [`JobState`] from the reducer: a status header, the streamed
//! output ring, and key hints. This is the shared "running job" surface every
//! operational screen reuses instead of the frozen rocm-cli `running_job`
//! modal. The widget is read-only over the reducer's job model.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::state::{JobState, JobStatus};

use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

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
