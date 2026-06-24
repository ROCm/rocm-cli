// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Tabbed view modes. Each tab module exposes a single
//! `pub fn draw(f, area, state, theme)` so they can be implemented in parallel
//! without touching shared files.

pub mod action;
pub mod bench;
pub mod chat;
pub mod hardware;
pub mod home;
pub mod instances;
pub mod observe;
pub mod overview;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::ActiveTab;
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

pub const TAB_LABELS: [(ActiveTab, &str, char); 4] = [
    (ActiveTab::Home, "Home", '1'),
    (ActiveTab::Action, "Action", '2'),
    (ActiveTab::Observe, "Observe", '3'),
    (ActiveTab::Chat, "Chat", '4'),
];

/// The four tab labels in display order (for the outlined panel renderer).
#[must_use]
pub const fn tab_labels() -> [&'static str; 4] {
    [
        TAB_LABELS[0].1,
        TAB_LABELS[1].1,
        TAB_LABELS[2].1,
        TAB_LABELS[3].1,
    ]
}

/// Index of `tab` within [`TAB_LABELS`] (the active-folder index).
#[must_use]
pub fn active_index(tab: ActiveTab) -> usize {
    TAB_LABELS
        .iter()
        .position(|(t, _, _)| *t == tab)
        .unwrap_or(0)
}

/// A single tab chip's screen extent. `x_start..x_end` are absolute columns
/// (end-exclusive) on the tab-bar row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabChip {
    pub tab: ActiveTab,
    pub x_start: u16,
    pub x_end: u16,
}

/// Outlined-tab chip clickable spans starting at `origin_x`, one `(start, end)`
/// (end-exclusive) per label. This is the SINGLE SOURCE of tab geometry: both
/// [`draw_tab_panel`] (rendering) and [`compute_chip_layout`] (hit-testing)
/// route through it so they cannot drift. The active marker (`●`) and the
/// 1-based index are both a single cell, so widths are active-independent.
fn outlined_chip_spans(origin_x: u16, labels: &[&str]) -> Vec<(u16, u16)> {
    let mut out = Vec::with_capacity(labels.len());
    let mut sx = origin_x;
    for lab in labels {
        let cw = lab.chars().count() as u16 + 4; // " X label "
        let ex = sx + cw + 1; // right border column (inclusive)
        out.push((sx, ex + 1)); // clickable span [sx, ex] → end-exclusive
        sx = ex + 2; // 1-col gap between folders
    }
    out
}

/// Per-chip bounding boxes for the live tab panel.
///
/// The first folder's left border is at `bar_x`. Mirrors [`draw_tab_panel`]'s
/// outlined-folder geometry exactly (via [`outlined_chip_spans`]) so a click
/// resolves to the tab drawn at that column.
pub fn compute_chip_layout(bar_x: u16) -> [TabChip; 4] {
    let labels = [
        TAB_LABELS[0].1,
        TAB_LABELS[1].1,
        TAB_LABELS[2].1,
        TAB_LABELS[3].1,
    ];
    let spans = outlined_chip_spans(bar_x, &labels);
    let mut out = [TabChip {
        tab: ActiveTab::Home,
        x_start: 0,
        x_end: 0,
    }; 4];
    for (i, (start, end)) in spans.iter().enumerate() {
        out[i] = TabChip {
            tab: TAB_LABELS[i].0,
            x_start: *start,
            x_end: *end,
        };
    }
    out
}

/// Stamp a string at `(x, y)` if the row is on-screen. Mirror of the
/// `gen_mockups` low-level `put` so the ported chrome paints identically.
fn put(f: &mut Frame, x: u16, y: u16, s: &str, style: Style) {
    if y < f.area().height {
        f.buffer_mut().set_string(x, y, s, style);
    }
}

/// Draw a horizontal run of `ch` from `x0` to `x1` inclusive on row `y`.
fn hline(f: &mut Frame, x0: u16, x1: u16, y: u16, ch: &str, style: Style) {
    for x in x0..=x1 {
        put(f, x, y, ch, style);
    }
}

/// Outlined folder-tab panel (port of `gen_mockups.rs` `tab_panel`).
///
/// A bordered body box whose top edge carries raised, outlined tab "folders";
/// the active tab opens into the body (its bottom edge is erased). Returns the
/// inner content rect so the caller composes the tab body inside the frame.
///
/// `outer` must be ≥3 rows tall (top edge / label row / panel line) + body.
///
/// This is the live tab chrome (`ui::draw`). Tab hit-testing routes through
/// the same [`outlined_chip_spans`] geometry via [`compute_chip_layout`], so a
/// click resolves to the folder painted at that column.
#[must_use]
pub fn draw_tab_panel(
    f: &mut Frame,
    outer: Rect,
    labels: &[&str],
    active: usize,
    theme: &Theme,
) -> Rect {
    let border = Style::default().fg(theme.border);
    let acc = Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD);

    if outer.width < 4 || outer.height < 4 {
        return outer;
    }

    let x0 = outer.x;
    let x1 = outer.x + outer.width - 1;
    let y_top = outer.y;
    let y_lab = outer.y + 1;
    let y_line = outer.y + 2;
    let y_bot = outer.y + outer.height - 1;

    // Panel box (the body frame), rounded to match the bento boxes. The tab
    // folders are stamped on top after.
    hline(f, x0, x1, y_line, "─", border);
    put(f, x0, y_line, "╭", border);
    put(f, x1, y_line, "╮", border);
    for y in (y_line + 1)..y_bot {
        put(f, x0, y, "│", border);
        put(f, x1, y, "│", border);
    }
    hline(f, x0, x1, y_bot, "─", border);
    put(f, x0, y_bot, "╰", border);
    put(f, x1, y_bot, "╯", border);

    // Tab folders along the top edge — geometry from the shared single source.
    let spans = outlined_chip_spans(x0 + 2, labels);
    for (i, lab) in labels.iter().enumerate() {
        let is_active = i == active;
        let content = if is_active {
            format!(" ● {lab} ")
        } else {
            format!(" {} {lab} ", i + 1)
        };
        let (sx, end) = spans[i];
        let ex = end.saturating_sub(1); // right border column (inclusive)
        if ex >= x1 {
            break; // out of horizontal room — stop cleanly rather than overflow
        }
        // Inactive folders share the frame's border color so they read as one
        // piece; only the active folder is accented.
        let style = if is_active { acc } else { border };

        // Rounded folder top, matching the frame.
        put(f, sx, y_top, "╭", style);
        hline(f, sx + 1, ex - 1, y_top, "─", style);
        put(f, ex, y_top, "╮", style);
        put(f, sx, y_lab, "│", style);
        put(f, sx + 1, y_lab, &content, style);
        put(f, ex, y_lab, "│", style);
        if is_active {
            // Open the active folder into the body: erase the panel line under
            // it. The bottom corners are accent so the active color continues
            // down the sides and curves outward into the frame; the rounded arc
            // (not a straight horizontal run) is the whole flourish — the frame
            // line beyond the corners stays the border color.
            put(f, sx, y_line, "╯", acc);
            for x in (sx + 1)..ex {
                put(f, x, y_line, " ", Style::default().bg(theme.bg));
            }
            put(f, ex, y_line, "╰", acc);
        } else {
            put(f, sx, y_line, "┴", style);
            put(f, ex, y_line, "┴", style);
        }
    }

    Rect::new(
        x0 + 2,
        y_line + 1,
        outer.width.saturating_sub(4),
        y_bot.saturating_sub(y_line + 1),
    )
}

/// Common stub renderer used by tabs that are not yet implemented.
pub fn draw_placeholder(f: &mut Frame, area: Rect, title: &str, body: &str, theme: &Theme) {
    let inner = panel::bento(f, area, Some(title), BoxRole::Muted, false, theme);
    let p = Paragraph::new(Line::from(Span::styled(
        body,
        Style::default().fg(theme.muted),
    )));
    f.render_widget(p, inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chip_layout_matches_draw_widths() {
        // Outlined folders: chip width = label.len()+6 (┌ + " X label " + ┐),
        // 1-col gap between. Home(4)/Action(6)/Observe(7)/Chat(4).
        let chips = compute_chip_layout(0);
        // " ● Home " (8) inside a box (10) → 0..10
        assert_eq!(chips[0].x_start, 0);
        assert_eq!(chips[0].x_end, 10);
        assert_eq!(chips[0].tab, ActiveTab::Home);
        // gap then " 2 Action " (10) box (12) → 11..23
        assert_eq!(chips[1].x_start, 11);
        assert_eq!(chips[1].x_end, 23);
        // " 3 Observe " (11) box (13) → 24..37
        assert_eq!(chips[2].x_start, 24);
        assert_eq!(chips[2].x_end, 37);
        // " 4 Chat " (8) box (10) → 38..48
        assert_eq!(chips[3].x_start, 38);
        assert_eq!(chips[3].x_end, 48);
        assert_eq!(chips[3].tab, ActiveTab::Chat);
    }

    #[test]
    fn chip_layout_honors_bar_x_offset() {
        let chips = compute_chip_layout(100);
        assert_eq!(chips[0].x_start, 100);
        // Full panel (Home start → Chat end) spans 48 columns.
        assert_eq!(chips[3].x_end - chips[0].x_start, 48);
    }

    #[test]
    fn outlined_tab_panel_paints_labels_and_active_marker() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(80, 12);
        let mut term = Terminal::new(backend).unwrap();
        let inner = std::cell::Cell::new(Rect::new(0, 0, 0, 0));
        term.draw(|f| {
            let r = draw_tab_panel(
                f,
                f.area(),
                &["Home", "Action", "Observe", "Chat"],
                0,
                &theme,
            );
            inner.set(r);
        })
        .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        // Inactive tabs show their 1-based index; the active tab shows ●.
        assert!(out.contains("● Home"), "active marker missing: {out:?}");
        assert!(out.contains("Action"), "tab label missing: {out:?}");
        // Frame + folders use rounded corners now.
        assert!(
            out.contains('╮') && out.contains('╰'),
            "no rounded panel frame"
        );
        // Inner content rect is strictly inside the outer frame.
        let r = inner.get();
        assert!(r.width > 0 && r.height > 0 && r.y >= 3);
    }

    #[test]
    fn outlined_tab_panel_survives_tiny_area() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        for (w, h) in [(2u16, 2u16), (4, 4), (10, 3)] {
            let backend = TestBackend::new(w.max(1), h.max(1));
            let mut term = Terminal::new(backend).unwrap();
            term.draw(|f| {
                let _ = draw_tab_panel(f, f.area(), &["Home", "Chat"], 1, &theme);
            })
            .unwrap();
        }
    }
}
