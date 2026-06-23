//! Action tab — the guided list of mutating verbs.
//!
//! Each actionable row opens an EXISTING manager overlay through the EXISTING
//! execution seam (`KeyAction::Open*` → `apply_action` → `RocmToolExecutor` /
//! `ui/approval.rs`). There is no second approval path and no reimplementation
//! of the managers — composition only.
//!
//! Per the latest mocks: there is no image-generation verb row; "Optimize a
//! model" is shown dimmed with a `soon` marker; "Uninstall" is a display-only
//! row (no uninstall manager is wired this run — ponytail).

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::app::{AppState, KeyAction};
use crate::ui::theme::Theme;

/// The actionable verbs, in display order. Each maps to an existing seam action.
/// `(icon, label, KeyAction)`.
const VERBS: &[(&str, &str, KeyAction)] = &[
    ("◆", "Serve a model", KeyAction::OpenServeWizard),
    ("⚙", "Set up / Install ROCm", KeyAction::OpenInstall),
    ("⌬", "Engines", KeyAction::OpenEngineManager),
    ("⚕", "Diagnose & fix  (doctor)", KeyAction::OpenDoctor),
    ("⇲", "Check for updates", KeyAction::OpenUpdate),
    ("⮌", "Manage providers & keys", KeyAction::OpenConfig),
];

/// Number of selectable (actionable) verbs — the Action tab's selection length.
pub const VERB_COUNT: usize = VERBS.len();

/// Resolve the seam action for the verb at `sel` (clamped). Used by
/// `apply_action` when Enter is pressed on the Action tab.
#[must_use]
pub fn verb_action(sel: usize) -> KeyAction {
    VERBS.get(sel).map_or(KeyAction::Nothing, |(_, _, a)| *a)
}

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area);

    draw_verb_list(f, cols[0], state, theme);
    draw_detail(f, cols[1], state, theme);
}

fn draw_verb_list(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Actions ")
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height == 0 {
        return;
    }

    let sel = state.action_sel.min(VERB_COUNT.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    for (i, (icon, label, _)) in VERBS.iter().enumerate() {
        let focused = i == sel;
        let (cur, c) = if focused {
            ("▸ ", theme.accent)
        } else {
            ("  ", theme.fg)
        };
        let style = if focused {
            Style::default().fg(c).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(c)
        };
        lines.push(Line::from(vec![
            Span::styled(cur, Style::default().fg(theme.accent)),
            Span::styled(format!("{icon}  "), Style::default().fg(theme.accent_2)),
            Span::styled(*label, style),
        ]));
        lines.push(Line::from(""));
    }
    // Display-only rows: Optimize (soon) and Uninstall (no manager wired).
    lines.push(Line::from(vec![
        Span::styled("  ⚡  ", Style::default().fg(theme.muted)),
        Span::styled("Optimize a model", Style::default().fg(theme.muted)),
        Span::raw("  "),
        Span::styled(
            " soon ",
            Style::default()
                .bg(theme.warn)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  ⊘  ", Style::default().fg(theme.muted)),
        Span::styled("Uninstall", Style::default().fg(theme.muted)),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "soon = planned, not yet built",
        Style::default().fg(theme.muted),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_detail(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let sel = state.action_sel.min(VERB_COUNT.saturating_sub(1));
    let (icon, label, _) = VERBS[sel];
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {icon} {label} "))
        .border_style(Style::default().fg(theme.accent))
        .title_style(theme.title_style());
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height == 0 {
        return;
    }
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "Opens the guided manager for this action.",
                Style::default().fg(theme.fg),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    "  ▸ Start ",
                    Style::default()
                        .fg(theme.bg)
                        .bg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Enter", Style::default().fg(theme.muted)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Mutating actions ask before they run. Default mode: ask.",
                Style::default().fg(theme.muted),
            )),
        ]),
        inner,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::ActiveTab;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn render(state: &AppState, cols: u16, rows: u16) -> String {
        let backend = TestBackend::new(cols, rows);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, f.area(), state, &state.theme))
            .unwrap();
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn action_renders_verbs_and_soon_marker() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Action;
        let out = render(&s, 120, 24);
        assert!(out.contains("Serve a model"), "serve row missing: {out:?}");
        assert!(out.contains("Engines"), "engines row missing");
        assert!(out.contains("soon"), "optimize soon marker missing");
        // Per the latest mocks, no image-generation verb anywhere.
        let needle = ["generate", "an", "image"].join(" ");
        assert!(
            !out.to_lowercase().contains(&needle),
            "image row must not appear"
        );
    }

    #[test]
    fn verb_action_maps_selection_to_seam() {
        assert_eq!(verb_action(0), KeyAction::OpenServeWizard);
        assert_eq!(verb_action(2), KeyAction::OpenEngineManager);
        // Out-of-range is a safe no-op.
        assert_eq!(verb_action(99), KeyAction::Nothing);
    }

    #[test]
    fn action_does_not_panic_when_squeezed() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Action;
        for h in [1u16, 2, 3, 5, 10] {
            let _ = render(&s, 60, h);
        }
    }
}
