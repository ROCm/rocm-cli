// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Job console widget.
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

use crate::app::{ScrollTarget, ScrollbarHandle};
use crate::ui::exec::display_args;
use crate::ui::panel::{centered_rect, draw_popup_frame};
use crate::ui::theme::{Theme, readable_text_on};

/// What a console keypress means to the owning screen.
///
/// The shared seam every operational overlay routes its `active_job` keys through, so the
/// Ctrl+C/`q`/Esc-Enter behavior is defined once instead of per screen.
#[derive(Debug)]
pub enum ConsoleOutcome {
    /// Ctrl+C cancelled the running job — the caller runs these effects.
    Cancelled(Vec<SideEffect>),
    /// `q` (on any job) or Esc (on a still-running one) — the caller leaves
    /// the console. What leaving *does* is the caller's call; the two
    /// possibilities are named by [`ConsoleExit`].
    Closed,
    /// Esc/Enter on a terminal job — the caller dismisses the console (returns
    /// to the screen body), and may clear any transient message.
    Dismissed,
    /// Not a console key — the caller may handle it (e.g. a screen-specific
    /// re-run shortcut).
    Unhandled,
}

/// What an overlay does with a [`ConsoleOutcome::Closed`] on a running job.
///
/// This is the one thing that is NOT uniform across the managers, named here
/// so the hints can say which applies.
///
/// `on_console_key` itself is uniform: `q` and Esc never get swallowed. This
/// only records what the owner does next, and it only drives wording — the
/// console footer hint below and the dashboard footer's Esc chip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleExit {
    /// Leaving is immediate and the job keeps running in the background. Every
    /// manager except the one below.
    Immediate,
    /// Leaving opens the owner's confirmation modal first, because leaving a
    /// launch that is still starting can hide a near-immediate failure. Only
    /// `ui::serve_wizard`'s running launch console does this.
    Confirmed,
}

/// Whether Esc on `job` leaves the console entirely (a
/// [`ConsoleOutcome::Closed`]), rather than merely dismissing it back to the
/// screen body.
///
/// `true` when the job is still running; `false` once it's finished or
/// missing. Shared by [`on_console_key`], the dashboard footer's Esc-chip
/// label and `AppState::console_exit` so they can't drift out of sync. Note
/// that "leaves the console" is not the same as "closes the overlay" for a
/// [`ConsoleExit::Confirmed`] owner, which asks first.
pub fn console_esc_closes(job: Option<&JobState>) -> bool {
    job.is_some_and(|j| !j.is_terminal())
}

/// Interpret a key while a job console is showing `job_id`. Pure except for the
/// `CancelJob` reducer apply (which only mutates the in-memory job model).
pub fn on_console_key(job_id: &str, jobs: &mut State, key: KeyEvent) -> ConsoleOutcome {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            ConsoleOutcome::Cancelled(jobs.apply(StateEvent::CancelJob(job_id.to_string())))
        }
        // `q` is always answered with `Closed`, never swallowed, so the user is
        // never trapped mid-job. The invariant is about the key reaching the
        // owner, not about what the owner then does: a `ConsoleExit::Immediate`
        // owner leaves at once and the job keeps running in the background,
        // while the one `ConsoleExit::Confirmed` owner (`ui::serve_wizard`'s
        // still-starting launch) turns this into its detach-confirm modal, from
        // which one more key leaves. Nothing here is conditional on the owner —
        // that asymmetry lives entirely in the caller.
        KeyCode::Char('q') => ConsoleOutcome::Closed,
        // Esc on a still-running job means the same as `q` above — the
        // conventional "get me out" key, so the user is never trapped during a
        // long step (e.g. a managed serve readiness wait).
        KeyCode::Esc if console_esc_closes(jobs.job(job_id)) => ConsoleOutcome::Closed,
        // On a finished (or vanished) job, Esc/Enter dismiss the console back to
        // the screen body.
        KeyCode::Esc | KeyCode::Enter if !console_esc_closes(jobs.job(job_id)) => {
            ConsoleOutcome::Dismissed
        }
        _ => ConsoleOutcome::Unhandled,
    }
}

/// Human-readable status label + the color it should render in.
fn status_label(job: &JobState, theme: &Theme) -> (String, ratatui::style::Color) {
    (job.status.label(), theme.job_status_color(&job.status))
}

/// Render the job console centered over `area`.
///
/// `scroll` is `(vertical_line, horizontal_col)` of the first visible cell (the
/// caller clamps it). `tick_count` drives the running-job progress spinner.
/// `exit` is what the owning overlay does when the user leaves, which only the
/// footer hint depends on. Returns the scrollbar handles drawn this frame so
/// the caller can record them for mouse hit-testing.
pub fn draw_job_console(
    f: &mut Frame,
    area: Rect,
    job: &JobState,
    scroll: (u16, u16),
    tick_count: u64,
    exit: ConsoleExit,
    theme: &Theme,
) -> Vec<ScrollbarHandle> {
    let popup = centered_rect(90, 84, 140, 40, area);
    let title = format!("{} {}", job.cmd, display_args(&job.args));
    let inner = draw_popup_frame(f, popup, title.trim(), theme);
    if inner.height == 0 {
        return Vec::new();
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);

    // Header: while running, a compact chip + animated braille spinner + parsed
    // percentage (already visually distinct via motion). Once the job reaches a
    // terminal state, render a full-width colored banner instead — a small chip
    // is easy to miss against a whole popup of scrolled output.
    let (label, color) = status_label(job, theme);
    let is_terminal = job.is_terminal();
    let mut header = Vec::new();
    if is_terminal {
        // Redundant by construction now that both branches share `is_terminal()` —
        // kept as a cheap regression guard against a future edit splitting the two
        // checks back apart, not because this can fail today.
        debug_assert!(
            job.is_terminal(),
            "terminal banner only renders for finished jobs"
        );
        let glyph = job.status.glyph();
        header.push(Span::styled(
            format!(" {glyph} {label} "),
            Style::default()
                .fg(readable_text_on(color))
                .add_modifier(Modifier::BOLD),
        ));
    } else {
        header.push(Span::styled(
            " status ",
            Style::default()
                .fg(readable_text_on(color))
                .bg(color)
                .add_modifier(Modifier::BOLD),
        ));
        header.push(Span::raw(" "));
        header.push(Span::styled(
            format!("{} ", crate::ui::spinner::spinner_frame(tick_count)),
            Style::default().fg(theme.accent),
        ));
        header.push(Span::styled(
            label,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
        // The most recent line carrying a percentage wins (later output is more
        // current than earlier); silently omit the figure when none is present.
        if let Some(pct) = job
            .output
            .iter()
            .rev()
            .find_map(|l| crate::ui::spinner::parse_progress_pct(l))
        {
            header.push(Span::styled(
                format!("  {pct}%"),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ));
        }
    }
    let header_para = Paragraph::new(Line::from(header));
    let header_para = if is_terminal {
        header_para.style(Style::default().bg(color))
    } else {
        header_para
    };
    f.render_widget(header_para, rows[0]);

    // Body: streamed output lines, with a scrollbar when the ring overflows the
    // viewport so the user can see there's more above/below.
    let lines: Vec<Line> = job
        .output
        .iter()
        .map(|l| Line::from(Span::styled(l.clone(), Style::default().fg(theme.fg))))
        .collect();
    let content_w = job
        .output
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0);
    let vbody = crate::ui::panel::vertical_scrollbar(
        f,
        rows[1],
        job.output.len(),
        rows[1].height as usize,
        scroll.0 as usize,
        theme,
    );
    let body = crate::ui::panel::horizontal_scrollbar(
        f,
        vbody,
        content_w,
        vbody.width as usize,
        scroll.1 as usize,
        theme,
    );
    f.render_widget(Paragraph::new(lines).scroll(scroll), body);

    // Report the drawn bars so the caller can hit-test mouse clicks/drags.
    let handles = [
        ScrollbarHandle::new(
            rows[1],
            vbody,
            false,
            job.output.len(),
            rows[1].height as usize,
            ScrollTarget::Console,
        ),
        ScrollbarHandle::new(
            vbody,
            body,
            true,
            content_w,
            vbody.width as usize,
            ScrollTarget::ConsoleH,
        ),
    ];

    // Footer: key hints. While running, Esc leaves the console and Ctrl+C
    // cancels — advertise both so the user is never left feeling trapped during
    // a long step (e.g. a managed serve's readiness wait). What leaving does
    // differs by owner, so the hint is picked from `exit` rather than hard-coded
    // to the common case: saying "keeps running" over a console whose Esc opens
    // a confirmation would describe behaviour the key no longer has.
    let hints = match (matches!(job.status, JobStatus::Running), exit) {
        (true, ConsoleExit::Immediate) => {
            "Esc close (keeps running) · Ctrl+C cancel · wheel / PgUp·PgDn scroll"
        }
        (true, ConsoleExit::Confirmed) => {
            "Esc leave (asks first) · Ctrl+C cancel · wheel / PgUp·PgDn scroll"
        }
        (false, _) => "Enter/Esc dismiss · wheel / PgUp·PgDn scroll",
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            hints,
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );

    handles.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::test_support::press as k;
    use rocm_dash_core::state::{State, StateEvent};

    fn theme() -> Theme {
        Theme::from_name("default")
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
        // `q` yields `Closed` regardless of job state. What the owning overlay
        // then does is its own choice (see `ConsoleExit`): all but one leave at
        // once, the serve wizard's still-starting launch asks first.
        assert!(matches!(
            on_console_key("j", &mut s, k(KeyCode::Char('q'))),
            ConsoleOutcome::Closed
        ));
        // Esc on a RUNNING job yields `Closed` too — same handoff as `q`.
        let mut s2 = State::default();
        s2.apply(StateEvent::StartJob {
            id: "j".into(),
            cmd: "x".into(),
            args: vec![],
        });
        assert!(matches!(
            on_console_key("j", &mut s2, k(KeyCode::Esc)),
            ConsoleOutcome::Closed
        ));
        // Enter on a RUNNING job stays Unhandled (avoids accidental dismissal).
        assert!(matches!(
            on_console_key("j", &mut s2, k(KeyCode::Enter)),
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
    fn console_esc_closes_tracks_job_terminality() {
        // Direct coverage for the seam itself: `on_console_key` and the
        // footer's Esc-chip label both call through `console_esc_closes`, so
        // a regression here would silently desync the two without this test.
        let mut s = State::default();
        s.apply(StateEvent::StartJob {
            id: "j".into(),
            cmd: "x".into(),
            args: vec![],
        });
        assert!(console_esc_closes(s.job("j")));
        s.apply(StateEvent::JobDone {
            id: "j".into(),
            code: 0,
        });
        assert!(!console_esc_closes(s.job("j")));
        assert!(!console_esc_closes(None));
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

    #[test]
    fn status_label_delegates_to_shared_job_status_helpers() {
        // Guards `job_console::status_label` specifically: it must return the
        // literal label/color the shared `JobStatus`/`Theme` helpers define,
        // not a hand-rolled match of its own. This does not exercise
        // `tabs/home` or `dock` — those have their own render-level
        // regression tests (`activity_feed_colors_match_shared_job_status_color`,
        // `logs_dock_tints_nonzero_exit_as_warn_not_ok`).
        let t = theme();
        let cases: [(JobStatus, &str, ratatui::style::Color); 5] = [
            (JobStatus::Running, "running", t.accent),
            (JobStatus::Done { code: 0 }, "done", t.ok),
            (JobStatus::Done { code: 7 }, "exited (7)", t.warn),
            (
                JobStatus::Failed {
                    message: "boom".into(),
                },
                "failed: boom",
                t.err,
            ),
            (JobStatus::Cancelled, "cancelled", t.muted),
        ];
        for (status, expected_label, expected_color) in cases {
            let job = JobState {
                cmd: "x".into(),
                args: Vec::new(),
                status,
                output: std::collections::VecDeque::default(),
                cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            };
            let (label, color) = status_label(&job, &t);
            assert_eq!(label, expected_label);
            assert_eq!(color, expected_color);
        }
    }

    #[test]
    fn terminal_status_fills_full_width_banner() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let t = theme();
        let mut s = State::default();
        s.apply(StateEvent::StartJob {
            id: "j".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        });
        s.apply(StateEvent::JobDone {
            id: "j".into(),
            code: 0,
        });
        let job = s.job("j").unwrap();

        let backend = TestBackend::new(100, 30);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            draw_job_console(f, f.area(), job, (0, 0), 0, ConsoleExit::Immediate, &t);
        })
        .unwrap();
        let buf = term.backend().buffer().clone();
        let filled = buf
            .content()
            .iter()
            .filter(|c| c.style().bg == Some(t.ok))
            .count();
        assert!(
            filled > 50,
            "expected a done job to paint a full-width header banner, only {filled} cells colored"
        );
        let out: String = buf
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains('✓'), "done banner missing glyph: {out:?}");
    }

    #[test]
    fn running_console_hint_names_the_owners_exit_not_a_fixed_one() {
        // The hint is the console's own promise about Esc. For an
        // `Immediate` owner that promise is "close (keeps running)"; for the
        // one `Confirmed` owner it must NOT say that, because Esc there opens
        // a question instead. Both wordings are checked from the same job, so
        // a hint hard-coded back to either case fails on the other row.
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let t = theme();
        let mut s = State::default();
        s.apply(StateEvent::StartJob {
            id: "j".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        });
        let job = s.job("j").unwrap();

        let render = |exit: ConsoleExit| {
            let backend = TestBackend::new(100, 30);
            let mut term = Terminal::new(backend).unwrap();
            term.draw(|f| {
                draw_job_console(f, f.area(), job, (0, 0), 0, exit, &t);
            })
            .unwrap();
            let out: String = term
                .backend()
                .buffer()
                .content()
                .iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect();
            out
        };

        let immediate = render(ConsoleExit::Immediate);
        assert!(
            immediate.contains("Esc close (keeps running)"),
            "an immediate owner still promises a plain close: {immediate:?}"
        );

        let confirmed = render(ConsoleExit::Confirmed);
        assert!(
            confirmed.contains("Esc leave (asks first)"),
            "a confirming owner must advertise the question: {confirmed:?}"
        );
        assert!(
            !confirmed.contains("keeps running"),
            "the stale 'close (keeps running)' wording must not survive here: {confirmed:?}"
        );
    }

    #[test]
    fn running_status_keeps_compact_chip() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let t = theme();
        let mut s = State::default();
        s.apply(StateEvent::StartJob {
            id: "j".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        });
        let job = s.job("j").unwrap();

        let backend = TestBackend::new(100, 30);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            draw_job_console(f, f.area(), job, (0, 0), 0, ConsoleExit::Immediate, &t);
        })
        .unwrap();
        let buf = term.backend().buffer().clone();
        let filled = buf
            .content()
            .iter()
            .filter(|c| c.style().bg == Some(t.accent))
            .count();
        assert!(
            filled < 20,
            "a running job should keep the compact status chip, not a full-width banner: {filled} cells colored"
        );
    }

    #[test]
    fn nonzero_exit_gets_a_distinct_glyph_from_success() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let t = theme();
        let mut s = State::default();
        s.apply(StateEvent::StartJob {
            id: "j".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        });
        s.apply(StateEvent::JobDone {
            id: "j".into(),
            code: 1,
        });
        let job = s.job("j").unwrap();

        let backend = TestBackend::new(100, 30);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            draw_job_console(f, f.area(), job, (0, 0), 0, ConsoleExit::Immediate, &t);
        })
        .unwrap();
        let buf = term.backend().buffer().clone();
        let out: String = buf
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(
            !out.contains('✓'),
            "a nonzero exit banner should not reuse the success glyph: {out:?}"
        );
        assert!(
            out.contains('!'),
            "a nonzero exit banner should carry a distinct glyph: {out:?}"
        );
    }
}
