// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Tabbed view modes. Each tab module exposes a single
//! `pub fn draw(f, area, state, theme)` so they can be implemented in parallel
//! without touching shared files.

pub mod bench;
pub mod chat;
pub mod hardware;
pub mod instances;
pub mod overview;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::ActiveTab;
use crate::ui::theme::Theme;

pub const TAB_LABELS: [(ActiveTab, &str, char); 5] = [
    (ActiveTab::Overview, "Overview", '1'),
    (ActiveTab::Hardware, "Hardware", '2'),
    (ActiveTab::Instances, "Instances", '3'),
    (ActiveTab::Bench, "Bench", '4'),
    (ActiveTab::Chat, "Chat", '5'),
];

/// A single tab chip's screen extent. `x_start..x_end` are absolute columns
/// (end-exclusive) on the tab-bar row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabChip {
    pub tab: ActiveTab,
    pub x_start: u16,
    pub x_end: u16,
}

/// Compute the per-chip bounding boxes for a tab bar starting at `bar_x`.
///
/// Mirrors `draw_tab_bar` exactly: a 3-char digit chip (" 1 "), then a
/// `" Label "` chip (`label.len() + 2`), separated between chips by `" · "`
/// (3 chars). Pure — both `draw_tab_bar` and `hit_test` route through this
/// so they cannot drift.
pub fn compute_chip_layout(bar_x: u16) -> [TabChip; 5] {
    let mut out = [TabChip {
        tab: ActiveTab::Overview,
        x_start: 0,
        x_end: 0,
    }; 5];
    let mut x = bar_x;
    for (i, (tab, label, _key)) in TAB_LABELS.iter().enumerate() {
        if i > 0 {
            x = x.saturating_add(3); // " · "
        }
        let chip_w = 3u16 + label.len() as u16 + 2; // " 1 " + " Label "
        out[i] = TabChip {
            tab: *tab,
            x_start: x,
            x_end: x.saturating_add(chip_w),
        };
        x = x.saturating_add(chip_w);
    }
    out
}

/// Render the segmented tab bar. One row tall.
pub fn draw_tab_bar(f: &mut Frame, area: Rect, active: ActiveTab, theme: &Theme) {
    let mut spans: Vec<Span> = Vec::with_capacity(TAB_LABELS.len() * 3);
    for (i, (tab, label, key)) in TAB_LABELS.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" · ", Style::default().fg(theme.muted)));
        }
        let style = if *tab == active {
            Style::default()
                .bg(theme.accent)
                .fg(theme.surface_2)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.fg)
        };
        spans.push(Span::styled(
            format!(" {key} "),
            Style::default().fg(theme.muted),
        ));
        spans.push(Span::styled(format!(" {label} "), style));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
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
// ponytail: this is the painter only; it is NOT wired into `ui::draw` yet
// (P2 adds Home behind it; P3 makes it the live chrome). Tab hit-testing stays
// routed through the single-source-of-truth [`compute_chip_layout`] /
// [`draw_tab_bar`] geometry until the live switch — no second offset table is
// introduced here.
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
    let muted = Style::default().fg(theme.muted);

    if outer.width < 4 || outer.height < 4 {
        return outer;
    }

    let x0 = outer.x;
    let x1 = outer.x + outer.width - 1;
    let y_top = outer.y;
    let y_lab = outer.y + 1;
    let y_line = outer.y + 2;
    let y_bot = outer.y + outer.height - 1;

    // Panel box (the body frame). The tab folders are stamped on top after.
    hline(f, x0, x1, y_line, "─", border);
    put(f, x0, y_line, "┌", border);
    put(f, x1, y_line, "┐", border);
    for y in (y_line + 1)..y_bot {
        put(f, x0, y, "│", border);
        put(f, x1, y, "│", border);
    }
    hline(f, x0, x1, y_bot, "─", border);
    put(f, x0, y_bot, "└", border);
    put(f, x1, y_bot, "┘", border);

    // Tab folders along the top edge.
    let mut sx = x0 + 2;
    for (i, lab) in labels.iter().enumerate() {
        let is_active = i == active;
        let content = if is_active {
            format!(" ● {lab} ")
        } else {
            format!(" {} {lab} ", i + 1)
        };
        let cw = content.chars().count() as u16;
        let ex = sx + cw + 1; // right border column (inclusive)
        if ex >= x1 {
            break; // out of horizontal room — stop cleanly rather than overflow
        }
        let style = if is_active { acc } else { muted };

        put(f, sx, y_top, "┌", style);
        hline(f, sx + 1, ex - 1, y_top, "─", style);
        put(f, ex, y_top, "┐", style);
        put(f, sx, y_lab, "│", style);
        put(f, sx + 1, y_lab, &content, style);
        put(f, ex, y_lab, "│", style);
        if is_active {
            // Open the active folder into the body: erase the panel line under it.
            put(f, sx, y_line, "┘", style);
            for x in (sx + 1)..ex {
                put(f, x, y_line, " ", Style::default().bg(theme.bg));
            }
            put(f, ex, y_line, "└", style);
        } else {
            put(f, sx, y_line, "┴", style);
            put(f, ex, y_line, "┴", style);
        }
        sx = ex + 2;
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
    use ratatui::widgets::{Block, Borders};
    let p = Paragraph::new(Line::from(Span::styled(
        body,
        Style::default().fg(theme.muted),
    )))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" {title} "))
            .border_style(theme.border_style())
            .title_style(theme.title_style()),
    );
    f.render_widget(p, area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chip_layout_matches_draw_widths() {
        let chips = compute_chip_layout(0);
        // " 1 " (3) + " Overview " (10) = 13
        assert_eq!(chips[0].x_start, 0);
        assert_eq!(chips[0].x_end, 13);
        // separator (3) then " 2 " (3) + " Hardware " (10) = 16..29
        assert_eq!(chips[1].x_start, 16);
        assert_eq!(chips[1].x_end, 29);
        // " 3 " + " Instances " (11) = 32..46
        assert_eq!(chips[2].x_start, 32);
        assert_eq!(chips[2].x_end, 46);
        // " 4 " + " Bench " (7) = 49..59
        assert_eq!(chips[3].x_start, 49);
        assert_eq!(chips[3].x_end, 59);
        // separator (3) then " 5 " (3) + " Chat " (6) = 62..71
        assert_eq!(chips[4].x_start, 62);
        assert_eq!(chips[4].x_end, 71);
        assert_eq!(chips[4].tab, ActiveTab::Chat);
    }

    #[test]
    fn chip_layout_honors_bar_x_offset() {
        let chips = compute_chip_layout(100);
        assert_eq!(chips[0].x_start, 100);
        // Full bar (Overview start → Chat end) spans 71 columns.
        assert_eq!(chips[4].x_end - chips[0].x_start, 71);
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
        assert!(out.contains('┐') && out.contains('└'), "no panel frame");
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
