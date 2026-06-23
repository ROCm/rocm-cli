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
use ratatui::widgets::{Paragraph, Wrap};

use crate::app::{ActionFocus, ActiveTab, AppState, KeyAction};
use crate::ui::panel::{self, BoxRole};
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

/// Inline preview of an operation, shown in the detail pane beside the verb
/// list. Each field is grounded in the real manager flow (no invented data):
/// `summary` = one line on what it does, `steps` = the actual stages the manager
/// walks, `cmd` = the underlying CLI it drives, `read_only` flips the safety
/// footnote (doctor changes nothing; the rest ask before mutating).
struct VerbDetail {
    summary: &'static str,
    steps: &'static [&'static str],
    cmd: &'static str,
    read_only: bool,
}

/// Per-verb detail, aligned 1:1 with [`VERBS`].
const DETAILS: [VerbDetail; VERB_COUNT] = [
    VerbDetail {
        summary: "Launch a model on a serving engine and expose an OpenAI-style endpoint.",
        steps: &[
            "Pick a model",
            "Choose an engine — vLLM · SGLang · llama.cpp · PyTorch",
            "Set GPU placement (required / preferred / CPU-only)",
            "Launch on 127.0.0.1:11435 and watch it come up",
        ],
        cmd: "rocm serve --engine …",
        read_only: false,
    },
    VerbDetail {
        summary: "Install or repair the ROCm SDK (TheRock).",
        steps: &[
            "Pick a channel (e.g. release)",
            "Choose a format — wheel or tarball",
            "Pick an install folder (prefix)",
            "Dry-run to preview, then apply",
        ],
        cmd: "rocm install --channel … --format …",
        read_only: false,
    },
    VerbDetail {
        summary: "Install, reinstall, or configure serving engines.",
        steps: &[
            "Browse engines — vLLM · SGLang · llama.cpp · PyTorch · lemonade",
            "See install status per engine",
            "Install / reinstall an engine",
            "Adjust engine config",
        ],
        cmd: "rocm engines …",
        read_only: false,
    },
    VerbDetail {
        summary: "Read-only environment check that flags what needs fixing.",
        steps: &[
            "Run `rocm doctor` (one job, no approval)",
            "Review runtime / driver / permission checks",
            "Re-run after fixes with r",
        ],
        cmd: "rocm doctor",
        read_only: true,
    },
    VerbDetail {
        summary: "Check, preview, and apply ROCm package updates.",
        steps: &[
            "Check for updates",
            "Preview the update (dry-run)",
            "Apply the update and activate it",
        ],
        cmd: "rocm update --check",
        read_only: false,
    },
    VerbDetail {
        summary: "Manage AI providers and API keys.",
        steps: &[
            "Show saved config",
            "Enable a provider — local · anthropic · openai",
            "Disable a provider",
            "Keys stay in your local config",
        ],
        cmd: "rocm config …",
        read_only: false,
    },
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
    let cols = split_columns(area);
    draw_verb_list(f, cols[0], state, theme);
    draw_detail(f, cols[1], state, theme);
}

/// The Action tab's 46/54 list|detail split. Single source so [`hit_test`]
/// reconstructs exactly what [`draw`] painted.
fn split_columns(area: Rect) -> std::rc::Rc<[Rect]> {
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area)
}

/// Map a left-click in the Action body to an action.
///
/// A verb row selects that verb; anywhere in the detail pane activates (steps
/// in, or opens once focused) — mirroring the keyboard model so the visible
/// Start affordance is honest.
#[must_use]
pub fn hit_test(area: Rect, x: u16, y: u16) -> Option<KeyAction> {
    let cols = split_columns(area);
    let (list, detail) = (cols[0], cols[1]);
    let hit = |r: Rect| x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height;

    if hit(detail) {
        return Some(KeyAction::ActionActivate);
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
            if row.is_multiple_of(2) && idx < VERB_COUNT {
                return Some(KeyAction::ActionSelect(idx));
            }
        }
    }
    None
}

fn draw_verb_list(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    // The list owns focus until the user steps right into the detail pane.
    let list_focused =
        state.active_tab == ActiveTab::Action && state.action_focus == ActionFocus::List;
    let inner = panel::bento(
        f,
        area,
        Some("Actions"),
        BoxRole::Primary,
        list_focused,
        theme,
    );
    if inner.height == 0 {
        return;
    }

    let sel = state.action_sel.min(VERB_COUNT.saturating_sub(1));
    // When focus has moved into the detail pane, dim the list cursor so it reads
    // as "parked here" rather than active.
    let cursor_color = if list_focused {
        theme.accent
    } else {
        theme.muted
    };
    let mut lines: Vec<Line> = Vec::new();
    for (i, (icon, label, _)) in VERBS.iter().enumerate() {
        let focused = i == sel;
        let (cur, c) = if focused {
            ("▸ ", cursor_color)
        } else {
            ("  ", theme.fg)
        };
        let style = if focused {
            Style::default().fg(c).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(c)
        };
        lines.push(Line::from(vec![
            Span::styled(cur, Style::default().fg(cursor_color)),
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
    let d = &DETAILS[sel];
    // The detail pane lights up (bold, brightened border + highlighted Start)
    // once the user steps into it with → or Enter.
    let focused =
        state.active_tab == ActiveTab::Action && state.action_focus == ActionFocus::Detail;

    let title = format!("{icon} {label}");
    let inner = panel::bento(f, area, Some(&title), BoxRole::Secondary, focused, theme);
    if inner.height == 0 {
        return;
    }

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(d.summary, Style::default().fg(theme.fg))),
        Line::from(""),
        Line::from(Span::styled(
            "What you'll do",
            Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD),
        )),
    ];
    for (i, step) in d.steps.iter().enumerate() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {}. ", i + 1),
                Style::default().fg(theme.accent_2),
            ),
            Span::styled(*step, Style::default().fg(theme.fg)),
        ]));
    }
    lines.push(Line::from(""));

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
        format!("  Runs: {}", d.cmd),
        Style::default().fg(theme.muted),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        if d.read_only {
            "Read-only — nothing is changed."
        } else {
            "Mutating actions ask before they run. Default mode: ask."
        },
        Style::default().fg(theme.muted),
    )));

    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
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

    #[test]
    fn details_align_one_to_one_with_verbs() {
        // The inline previews must stay aligned with the verb list, and doctor
        // (the only read-only verb) must be the one flagged read_only.
        assert_eq!(DETAILS.len(), VERB_COUNT);
        for (i, (_, label, _)) in VERBS.iter().enumerate() {
            let is_doctor = label.contains("doctor");
            assert_eq!(
                DETAILS[i].read_only, is_doctor,
                "read_only flag mismatched at {label}"
            );
        }
    }

    #[test]
    fn hit_test_selects_verb_rows_and_activates_detail() {
        let area = Rect::new(0, 0, 100, 24);
        let cols = split_columns(area);
        let list = cols[0];
        let inner = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .padding(panel::padding_for(list))
            .inner(list);
        // First verb row (label row) selects verb 0.
        assert_eq!(
            hit_test(area, inner.x + 1, inner.y),
            Some(KeyAction::ActionSelect(0))
        );
        // The blank row between verbs selects nothing.
        assert_eq!(hit_test(area, inner.x + 1, inner.y + 1), None);
        // Second verb row selects verb 1.
        assert_eq!(
            hit_test(area, inner.x + 1, inner.y + 2),
            Some(KeyAction::ActionSelect(1))
        );
        // A click anywhere in the detail pane activates (step-in / open).
        assert_eq!(
            hit_test(area, cols[1].x + 3, cols[1].y + 3),
            Some(KeyAction::ActionActivate)
        );
    }
}
