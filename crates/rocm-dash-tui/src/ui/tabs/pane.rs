// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Shared Actions + Details pane used by the ROCm and Serving tabs.
//!
//! Each domain tab is a left **Actions** bento (its locked verb list) plus a
//! right **`Details: <verb>`** bento that previews the selected operation. The
//! verb rows open EXISTING managers through the EXISTING execution seam
//! (`KeyAction::Open*` → `apply_action` → `RocmToolExecutor` / `ui/approval.rs`)
//! — no second approval path, no reimplementation of the managers.
//!
//! `rocm.rs` and `serving.rs` only declare their verb tables; all rendering,
//! hit-testing, and the list|detail geometry live here so the two tabs cannot
//! drift.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::app::{KeyAction, PaneFocus};
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

/// One row in a domain tab's Actions list, with its inline Details preview.
///
/// Every field is grounded in the real manager flow (no invented data):
/// `summary` = one line on what it does, `steps` = the actual stages the manager
/// walks, `cmd` = the underlying CLI it drives. `action` is the seam verb the row
/// opens — `KeyAction::Nothing` marks a display-only row (Uninstall / Optimize),
/// which renders dimmed with `badge` and is a safe no-op when activated.
pub struct Verb {
    pub icon: &'static str,
    pub label: &'static str,
    pub action: KeyAction,
    pub summary: &'static str,
    pub steps: &'static [&'static str],
    pub cmd: &'static str,
    pub read_only: bool,
    /// Small status pill shown after a display-only row's label (e.g. `soon`).
    pub badge: Option<&'static str>,
}

impl Verb {
    /// A row is actionable when it maps to a real seam verb (not `Nothing`).
    const fn is_actionable(&self) -> bool {
        !matches!(self.action, KeyAction::Nothing)
    }
}

/// Resolve the seam action for the verb at `sel` (clamped). Used by
/// `apply_action` when Enter is pressed on a ROCm/Serving tab.
#[must_use]
pub fn verb_action(verbs: &[Verb], sel: usize) -> KeyAction {
    verbs.get(sel).map_or(KeyAction::Nothing, |v| v.action)
}

/// The 46/54 list|detail split. Single source so [`hit_test`] reconstructs
/// exactly what [`draw`] painted.
fn split_columns(area: Rect) -> std::rc::Rc<[Rect]> {
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area)
}

/// The Actions list rect (left column) for a domain tab body `area`.
///
/// Used by the mouse-scroll router to gate wheel-driven selection moves to when
/// the pointer is actually over the Actions list (not the Details pane).
#[must_use]
pub fn actions_rect(area: Rect) -> Rect {
    split_columns(area)[0]
}

/// Draw the Actions list (left) + Details preview (right) for one domain tab.
pub fn draw(
    f: &mut Frame,
    area: Rect,
    list_title: &str,
    verbs: &[Verb],
    sel: usize,
    focus: PaneFocus,
    theme: &Theme,
) {
    let cols = split_columns(area);
    draw_verb_list(f, cols[0], list_title, verbs, sel, focus, theme);
    draw_detail(f, cols[1], verbs, sel, focus, theme);
}

/// Map a left-click in a domain tab's body to an action.
///
/// A verb row selects that verb; anywhere in the detail pane activates (steps
/// in, or opens once focused) — mirroring the keyboard model so the visible
/// Start affordance is honest.
#[must_use]
pub fn hit_test(area: Rect, x: u16, y: u16, verb_count: usize) -> Option<KeyAction> {
    let cols = split_columns(area);
    let (list, detail) = (cols[0], cols[1]);
    let hit = |r: Rect| x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height;

    if hit(detail) {
        return Some(KeyAction::PaneActivate);
    }
    if hit(list) {
        // Reconstruct bento's inner rect (rounded border + adaptive padding).
        let inner = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .padding(panel::padding_for(list))
            .inner(list);
        if x >= inner.x && x < inner.x + inner.width && y >= inner.y {
            // Each verb occupies two rows (label + blank); blanks don't select.
            let row = (y - inner.y) as usize;
            let idx = row / 2;
            if row.is_multiple_of(2) && idx < verb_count {
                return Some(KeyAction::PaneSelect(idx));
            }
        }
    }
    None
}

fn draw_verb_list(
    f: &mut Frame,
    area: Rect,
    title: &str,
    verbs: &[Verb],
    sel: usize,
    focus: PaneFocus,
    theme: &Theme,
) {
    // The list owns focus until the user steps right into the detail pane.
    let list_focused = focus == PaneFocus::Actions;
    let inner = panel::bento(f, area, Some(title), BoxRole::Primary, list_focused, theme);
    if inner.height == 0 {
        return;
    }

    let sel = sel.min(verbs.len().saturating_sub(1));
    // When focus has moved into the detail pane, dim the list cursor so it reads
    // as "parked here" rather than active.
    let cursor_color = if list_focused {
        theme.accent
    } else {
        theme.muted
    };
    let mut lines: Vec<Line> = Vec::new();
    for (i, v) in verbs.iter().enumerate() {
        let selected = i == sel;
        let actionable = v.is_actionable();
        let cur = if selected { "▸ " } else { "  " };
        // Display-only rows read muted; actionable rows use the foreground.
        let label_color = if actionable { theme.fg } else { theme.muted };
        let label_style = if selected {
            Style::default()
                .fg(if actionable {
                    cursor_color
                } else {
                    theme.muted
                })
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(label_color)
        };
        let icon_color = if actionable {
            theme.accent_2
        } else {
            theme.muted
        };
        let mut spans = vec![
            Span::styled(cur, Style::default().fg(cursor_color)),
            Span::styled(format!("{}  ", v.icon), Style::default().fg(icon_color)),
            Span::styled(v.label, label_style),
        ];
        if let Some(badge) = v.badge {
            spans.push(Span::raw("  "));
            spans.push(Span::styled(
                format!(" {badge} "),
                Style::default()
                    .bg(theme.warn)
                    .fg(theme.bg)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::from(spans));
        lines.push(Line::from(""));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_detail(
    f: &mut Frame,
    area: Rect,
    verbs: &[Verb],
    sel: usize,
    focus: PaneFocus,
    theme: &Theme,
) {
    let sel = sel.min(verbs.len().saturating_sub(1));
    let Some(v) = verbs.get(sel) else {
        return;
    };
    let actionable = v.is_actionable();
    // The detail pane lights up once the user steps into it with → or Enter.
    let focused = focus == PaneFocus::Detail;

    let title = format!("{} {}", v.icon, v.label);
    let inner = panel::bento(f, area, Some(&title), BoxRole::Secondary, focused, theme);
    if inner.height == 0 {
        return;
    }

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(v.summary, Style::default().fg(theme.fg))),
        Line::from(""),
        Line::from(Span::styled(
            "What you'll do",
            Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD),
        )),
    ];
    for (i, step) in v.steps.iter().enumerate() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {}. ", i + 1),
                Style::default().fg(theme.accent_2),
            ),
            Span::styled(*step, Style::default().fg(theme.fg)),
        ]));
    }
    lines.push(Line::from(""));

    if actionable {
        // Start affordance: filled accent when focused, outlined hint otherwise.
        let (start_style, hint) = if focused {
            (
                Style::default()
                    .fg(theme.bg)
                    .bg(theme.accent)
                    .add_modifier(Modifier::BOLD),
                "   Enter — opens the guided manager",
            )
        } else {
            (
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
                "   → or Enter to begin",
            )
        };
        lines.push(Line::from(vec![
            Span::styled("  ▸ Start ", start_style),
            Span::styled(hint, Style::default().fg(theme.muted)),
        ]));
        lines.push(Line::from(Span::styled(
            format!("  Runs: {}", v.cmd),
            Style::default().fg(theme.muted),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            if v.read_only {
                "Read-only — nothing is changed."
            } else {
                "Mutating actions ask before they run. Default mode: ask."
            },
            Style::default().fg(theme.muted),
        )));
    } else {
        // Display-only row: no Start button, an honest "not available" note.
        lines.push(Line::from(Span::styled(
            v.badge.map_or_else(
                || "Not available yet.".to_string(),
                |b| format!("Marked “{b}” — planned, not yet built."),
            ),
            Style::default().fg(theme.muted),
        )));
    }

    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}
