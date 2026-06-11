//! Tabbed view modes. Each tab module exposes a single
//! `pub fn draw(f, area, state, theme)` so they can be implemented in parallel
//! without touching shared files.

pub mod bench;
pub mod chat;
pub mod hardware;
pub mod instances;
pub mod overview;

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

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
}
