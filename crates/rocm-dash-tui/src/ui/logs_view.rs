//! Logs overlay (Phase 3 Wave 3).
//!
//! Browses recent ROCm CLI logs via `rocm logs [--search WORDS]` — read-only, so
//! no approval gate. An optional search box filters before running. Output
//! streams into the shared job console. Read-only-with-input archetype.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::state::{SideEffect, State, StateEvent};

use crate::ui::exec::resolve_exe;
use crate::ui::job_console::{ConsoleOutcome, draw_job_console, on_console_key};
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone, Default)]
pub struct LogsViewState {
    /// Optional space-separated search terms.
    pub query: String,
    /// In-flight (or just-finished) `rocm logs` job id.
    pub active_job: Option<String>,
}

/// Handle a key while the overlay is open.
pub fn on_key(
    logs: &mut Option<LogsViewState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(l) = logs.as_mut() else {
        return Vec::new();
    };

    if let Some(job_id) = l.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *logs = None,
            ConsoleOutcome::Dismissed => l.active_job = None,
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // Search box (read-only-with-input archetype): Esc closes; printable chars
    // — including `q` — are captured as search text, not treated as a close key
    // (the same form-field tradeoff as serve_wizard's Model field). From the
    // running console, `q` closes via on_console_key above.
    match key.code {
        KeyCode::Esc => *logs = None,
        KeyCode::Enter => return run_logs(l, jobs),
        KeyCode::Backspace => {
            l.query.pop();
        }
        KeyCode::Char(c) => l.query.push(c),
        _ => {}
    }
    Vec::new()
}

/// The `rocm logs` argv for the current query (read-only).
fn logs_args(query: &str) -> Vec<String> {
    let mut args = vec!["logs".to_string()];
    let terms: Vec<&str> = query.split_whitespace().collect();
    if !terms.is_empty() {
        args.push("--search".to_string());
        args.extend(terms.into_iter().map(str::to_string));
    }
    args
}

/// Spawn `rocm logs` (read-only). A stable id replaces any prior console.
fn run_logs(l: &mut LogsViewState, jobs: &mut State) -> Vec<SideEffect> {
    let cmd = resolve_exe();
    let id = "logs".to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd,
        args: logs_args(&l.query),
    });
    // Single stable id → a no-op means re-attach to the running console.
    l.active_job = Some(id);
    fx
}

/// Render the overlay (search box, or the job console once running).
pub fn draw_logs_view(f: &mut Frame, area: Rect, l: &LogsViewState, jobs: &State, theme: &Theme) {
    if let Some(job_id) = &l.active_job
        && let Some(job) = jobs.job(job_id)
    {
        draw_job_console(f, area, job, 0, theme);
        return;
    }

    let popup = centered_rect(70, 40, 88, 10, area);
    let inner = draw_popup_frame(f, popup, "Logs — recent ROCm CLI activity", theme);
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    let query_display = if l.query.is_empty() {
        "(optional: type words to search, or just Enter for recent)".to_string()
    } else {
        l.query.clone()
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("search: ", Style::default().fg(theme.muted)),
            Span::styled(
                query_display,
                Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
            ),
        ])),
        rows[0],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "type to search · Enter view · Esc close",
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
    fn logs_args_recent_when_no_query() {
        assert_eq!(logs_args(""), vec!["logs"]);
        assert_eq!(logs_args("   "), vec!["logs"]);
    }

    #[test]
    fn logs_args_search_splits_terms() {
        assert_eq!(
            logs_args("error  serve"),
            vec!["logs", "--search", "error", "serve"]
        );
    }

    #[test]
    fn typing_builds_query_then_enter_runs_read_only() {
        let mut l = Some(LogsViewState::default());
        let mut jobs = State::default();
        for c in "vllm".chars() {
            on_key(&mut l, &mut jobs, key(KeyCode::Char(c)));
        }
        assert_eq!(l.as_ref().unwrap().query, "vllm");
        let fx = on_key(&mut l, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1, "read-only spawn, no approval");
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        assert_eq!(l.as_ref().unwrap().active_job.as_deref(), Some("logs"));
    }

    #[test]
    fn esc_closes_when_idle_and_q_types_into_query() {
        let mut l = Some(LogsViewState::default());
        let mut jobs = State::default();
        // `q` is a search character here, not a close key (Esc closes).
        on_key(&mut l, &mut jobs, key(KeyCode::Char('q')));
        assert_eq!(l.as_ref().unwrap().query, "q");
        on_key(&mut l, &mut jobs, key(KeyCode::Esc));
        assert!(l.is_none());
    }

    #[test]
    fn q_closes_overlay_while_job_runs() {
        let mut l = Some(LogsViewState::default());
        let mut jobs = State::default();
        on_key(&mut l, &mut jobs, key(KeyCode::Enter));
        on_key(&mut l, &mut jobs, key(KeyCode::Char('q')));
        assert!(l.is_none(), "q closes the overlay from the console view");
    }

    #[test]
    fn second_enter_while_running_reattaches_to_console() {
        // Single stable id: a second view while the prior job still runs re-uses
        // the same console (read-only re-attach, like examine) — never an error.
        let mut l = Some(LogsViewState::default());
        let mut jobs = State::default();
        on_key(&mut l, &mut jobs, key(KeyCode::Enter));
        assert_eq!(l.as_ref().unwrap().active_job.as_deref(), Some("logs"));
        // Dismiss is unreachable while running; simulate re-entry from a fresh
        // overlay against the same still-running job.
        let mut l2 = Some(LogsViewState::default());
        let fx = on_key(&mut l2, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty(), "no second spawn for the running id");
        assert_eq!(
            l2.as_ref().unwrap().active_job.as_deref(),
            Some("logs"),
            "re-attaches to the live console"
        );
        assert_eq!(jobs.jobs.len(), 1);
    }

    #[test]
    fn snapshot_shows_search_box() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(94, 14);
        let mut term = Terminal::new(backend).unwrap();
        let l = LogsViewState::default();
        let jobs = State::default();
        term.draw(|f| draw_logs_view(f, f.area(), &l, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Logs"));
        assert!(out.contains("search"));
    }
}
