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
//! chat seam (`agent.rs:59-62`).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApprovalChoice {
    #[default]
    Approve,
    Deny,
}

impl ApprovalChoice {
    pub fn toggle(self) -> Self {
        match self {
            ApprovalChoice::Approve => ApprovalChoice::Deny,
            ApprovalChoice::Deny => ApprovalChoice::Approve,
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
pub fn approval_key(
    key: KeyCode,
    choice: ApprovalChoice,
) -> (ApprovalChoice, Option<ApprovalVerdict>) {
    match key {
        KeyCode::Up | KeyCode::Down | KeyCode::Tab | KeyCode::BackTab => (choice.toggle(), None),
        KeyCode::Char('y') | KeyCode::Char('Y') => (choice, Some(ApprovalVerdict::Approve)),
        KeyCode::Char('n') | KeyCode::Char('N') => (choice, Some(ApprovalVerdict::Deny)),
        KeyCode::Esc => (choice, Some(ApprovalVerdict::Cancel)),
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
    let popup = centered_rect(72, 70, 96, 24, area);
    let inner = draw_popup_frame(f, popup, &format!("Review: {}", req.title), theme);
    if inner.height == 0 {
        return;
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
    f.render_widget(Paragraph::new(body).wrap(Wrap { trim: false }), rows[0]);

    f.render_widget(buttons_line(choice, theme), rows[1]);
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
        "   Tab move · Enter confirm · Esc cancel",
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
    fn esc_cancels() {
        assert_eq!(
            approval_key(KeyCode::Esc, ApprovalChoice::Approve).1,
            Some(ApprovalVerdict::Cancel)
        );
    }
}
