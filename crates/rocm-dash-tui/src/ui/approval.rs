// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Approval-gate seam (Phase 3 Wave 0).
//!
//! This is a **render + event seam ONLY**. The decision *logic* — what an
//! approval actually does (run a CLI command, enable full access, approve a
//! proposal) — stays CLI-side per the working agreement
//! (`rocm-cli-unification-working-agreements.md` §1). This module renders an
//! [`ApprovalRequest`] and reports the user's [`ApprovalVerdict`]; the caller
//! maps the verdict onto a CLI-side action.
//!
//! It must never gain a mutating capability and never touches the read-only
//! chat seam ([`crate::agent::AgentClient`]).
//!
//! Keymap mirrors the frozen rocm-cli `pending_approval` screen: Up/Down/Tab
//! move the cursor; `y` approves; `n` denies; Esc cancels; Enter performs the
//! highlighted choice.

use crossterm::event::KeyCode;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// A request to approve or deny a CLI-side action. Carries display data only;
/// the actionable payload lives CLI-side, keyed by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    pub title: String,
    /// Lines describing what will run (the command, an explanation, a diff).
    pub body: Vec<String>,
}

impl ApprovalRequest {
    pub fn new(title: impl Into<String>, body: Vec<String>) -> Self {
        Self {
            title: title.into(),
            body,
        }
    }
}

/// Which button the cursor is on.
///
/// `Approve` is the shared default — it governs the ten manager-embedded
/// approval flows (install/stop/reinstall/etc., each via their own
/// `ApprovalChoice::default()` call) where the user just explicitly
/// requested the action being confirmed. The chat/tool-call approval flow
/// (`AppState::open_approval`) is a different case — an unreviewed tool
/// call the model wants to run — and opts into `Deny` explicitly there
/// instead of moving this shared default.
///
/// The Doctor report send (`examine_manager`) is a second `Deny` flow, for a
/// different reason. The user pressed `s` to review the report, not to send it,
/// and approving moves machine data off the screen, so the safe button stays the
/// default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApprovalChoice {
    #[default]
    Approve,
    Deny,
}

impl ApprovalChoice {
    #[must_use]
    pub const fn toggle(self) -> Self {
        match self {
            Self::Approve => Self::Deny,
            Self::Deny => Self::Approve,
        }
    }
}

/// The user's decision. The caller decides what each verdict *means*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalVerdict {
    Approve,
    Deny,
    Cancel,
}

/// Pure key handler. Returns the (possibly moved) cursor and an optional
/// verdict. No side effects — the caller routes the verdict CLI-side.
pub const fn approval_key(
    key: KeyCode,
    choice: ApprovalChoice,
) -> (ApprovalChoice, Option<ApprovalVerdict>) {
    match key {
        KeyCode::Up | KeyCode::Down | KeyCode::Tab | KeyCode::BackTab => (choice.toggle(), None),
        KeyCode::Char('y' | 'Y') => (choice, Some(ApprovalVerdict::Approve)),
        KeyCode::Char('n' | 'N') => (choice, Some(ApprovalVerdict::Deny)),
        // Esc and `q` both cancel — `q` must never be silently swallowed while a
        // modal is up, so the user is never trapped on any screen.
        KeyCode::Esc | KeyCode::Char('q' | 'Q') => (choice, Some(ApprovalVerdict::Cancel)),
        KeyCode::Enter => {
            let verdict = match choice {
                ApprovalChoice::Approve => ApprovalVerdict::Approve,
                ApprovalChoice::Deny => ApprovalVerdict::Deny,
            };
            (choice, Some(verdict))
        }
        _ => (choice, None),
    }
}

/// Render the approval modal over `area`.
pub fn draw_approval(
    f: &mut Frame,
    area: Rect,
    req: &ApprovalRequest,
    choice: ApprovalChoice,
    theme: &Theme,
) {
    draw_approval_inner(f, area, req, choice, None, theme);
}

/// Render the approval modal with the body scrolled down by `scroll` rows.
///
/// Only a caller that handles the scroll keys may use this. [`draw_approval`] is
/// for every other dialog and never promises keys it does not handle.
///
/// A body taller than the popup would otherwise be cut off at the bottom, which
/// hides the very text a user is asked to read before approving. When the body
/// does not fit, the row under the buttons says so and names the keys. Returns
/// how far the body can scroll, so the caller can keep its offset in range.
/// `scroll` is clamped to that range here, so a stale offset never blanks the body.
pub fn draw_approval_scrolled(
    f: &mut Frame,
    area: Rect,
    req: &ApprovalRequest,
    choice: ApprovalChoice,
    scroll: u16,
    theme: &Theme,
) -> u16 {
    draw_approval_inner(f, area, req, choice, Some(scroll), theme)
}

/// Shared drawing. `scroll` is `Some` only when the caller handles the scroll
/// keys, and only then does the dialog show the "More to read" row.
fn draw_approval_inner(
    f: &mut Frame,
    area: Rect,
    req: &ApprovalRequest,
    choice: ApprovalChoice,
    scroll: Option<u16>,
    theme: &Theme,
) -> u16 {
    let popup = centered_rect(72, 70, 96, 24, area);
    let inner = draw_popup_frame(f, popup, &format!("Review: {}", req.title), theme);
    if inner.height == 0 {
        return 0;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(2)])
        .split(inner);

    let body: Vec<Line> = req
        .body
        .iter()
        .map(|l| Line::from(Span::styled(l.clone(), Style::default().fg(theme.fg))))
        .collect();
    let paragraph = Paragraph::new(body).wrap(Wrap { trim: false });
    let total = paragraph.line_count(rows[0].width);
    let visible = usize::from(rows[0].height);
    let max_scroll = u16::try_from(total.saturating_sub(visible)).unwrap_or(u16::MAX);
    let offers_scroll = scroll.is_some();
    let scroll = scroll.unwrap_or(0).min(max_scroll);
    f.render_widget(paragraph.scroll((scroll, 0)), rows[0]);

    let footer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(rows[1]);
    f.render_widget(buttons_line(choice, theme), footer[0]);
    if max_scroll > 0 && offers_scroll {
        let last_shown = usize::from(scroll) + visible;
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(
                    "More to read: PgUp/PgDn scroll · Home/End jump · row {} of {}",
                    last_shown.min(total),
                    total
                ),
                Style::default().fg(theme.muted),
            ))),
            footer[1],
        );
    }
    max_scroll
}

fn buttons_line<'a>(choice: ApprovalChoice, theme: &Theme) -> Paragraph<'a> {
    let approve = button_span(
        " Approve (y) ",
        choice == ApprovalChoice::Approve,
        theme.ok,
        theme,
    );
    let deny = button_span(
        " Deny (n) ",
        choice == ApprovalChoice::Deny,
        theme.err,
        theme,
    );
    let help = Span::styled(
        "   Tab move · Enter confirm · Esc/q cancel",
        Style::default().fg(theme.muted),
    );
    Paragraph::new(Line::from(vec![approve, Span::raw("  "), deny, help]))
}

fn button_span<'a>(
    label: &'a str,
    selected: bool,
    accent: ratatui::style::Color,
    theme: &Theme,
) -> Span<'a> {
    if selected {
        Span::styled(
            label,
            Style::default()
                .bg(accent)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(label, Style::default().fg(theme.muted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab_toggles_choice_without_verdict() {
        let (c, v) = approval_key(KeyCode::Tab, ApprovalChoice::Approve);
        assert_eq!(c, ApprovalChoice::Deny);
        assert!(v.is_none());
        let (c, v) = approval_key(KeyCode::Up, c);
        assert_eq!(c, ApprovalChoice::Approve);
        assert!(v.is_none());
    }

    #[test]
    fn y_and_n_are_direct_verdicts() {
        assert_eq!(
            approval_key(KeyCode::Char('y'), ApprovalChoice::Deny).1,
            Some(ApprovalVerdict::Approve)
        );
        assert_eq!(
            approval_key(KeyCode::Char('n'), ApprovalChoice::Approve).1,
            Some(ApprovalVerdict::Deny)
        );
    }

    #[test]
    fn enter_confirms_highlighted_choice() {
        assert_eq!(
            approval_key(KeyCode::Enter, ApprovalChoice::Approve).1,
            Some(ApprovalVerdict::Approve)
        );
        assert_eq!(
            approval_key(KeyCode::Enter, ApprovalChoice::Deny).1,
            Some(ApprovalVerdict::Deny)
        );
    }

    #[test]
    fn esc_and_q_cancel() {
        assert_eq!(
            approval_key(KeyCode::Esc, ApprovalChoice::Approve).1,
            Some(ApprovalVerdict::Cancel)
        );
        // `q` must cancel too — never silently swallowed while the modal is up.
        assert_eq!(
            approval_key(KeyCode::Char('q'), ApprovalChoice::Approve).1,
            Some(ApprovalVerdict::Cancel)
        );
        assert_eq!(
            approval_key(KeyCode::Char('Q'), ApprovalChoice::Deny).1,
            Some(ApprovalVerdict::Cancel)
        );
    }

    fn long_request(lines: usize) -> ApprovalRequest {
        ApprovalRequest::new(
            "Long body",
            (0..lines).map(|i| format!("line {i}")).collect(),
        )
    }

    /// Draw the dialog at 110x30 and return the screen text and the scroll range.
    fn draw(req: &ApprovalRequest, scroll: Option<u16>) -> (String, u16) {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        let mut max = 0;
        term.draw(|f| {
            max = if let Some(s) = scroll {
                draw_approval_scrolled(f, f.area(), req, ApprovalChoice::Deny, s, &theme)
            } else {
                draw_approval(f, f.area(), req, ApprovalChoice::Deny, &theme);
                0
            };
        })
        .unwrap();
        let text: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        (text, max)
    }

    #[test]
    fn a_plain_approval_never_promises_scroll_keys() {
        // Only the callers that handle PgUp and PgDn opt in. Every other dialog
        // uses `draw_approval`, and a hint for keys that do nothing is false.
        let (out, _) = draw(&long_request(60), None);
        assert!(out.contains("Approve (y)"), "{out}");
        assert!(!out.contains("More to read"), "{out}");
        assert!(!out.contains("PgDn"), "{out}");
    }

    #[test]
    fn the_scrolled_approval_says_there_is_more_when_the_body_overflows() {
        let (out, max) = draw(&long_request(60), Some(0));
        assert!(max > 0);
        assert!(out.contains("More to read: PgUp/PgDn scroll"), "{out}");
        let (out, max) = draw(&long_request(2), Some(0));
        assert_eq!(max, 0);
        assert!(
            !out.contains("More to read"),
            "a body that fits needs no hint"
        );
    }

    #[test]
    fn an_out_of_range_scroll_still_shows_the_body() {
        let (out, max) = draw(&long_request(60), Some(0));
        let (stale, stale_max) = draw(&long_request(60), Some(u16::MAX));
        assert_eq!(stale_max, max);
        // Unclamped, the offset would scroll past every row and blank the body.
        assert!(stale.contains("line 59"), "{stale}");
        assert!(out.contains("line 0"), "{out}");
    }

    #[test]
    fn the_hint_names_the_last_row_shown() {
        let (out, max) = draw(&long_request(60), Some(0));
        let visible = 60 - usize::from(max);
        assert!(out.contains(&format!("row {visible} of 60")), "{out}");
        let (out, _) = draw(&long_request(60), Some(max));
        assert!(out.contains("row 60 of 60"), "{out}");
        let (out, _) = draw(&long_request(60), Some(1));
        assert!(out.contains(&format!("row {} of 60", visible + 1)), "{out}");
    }
}
