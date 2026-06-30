// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Serving tab — install, configure, and manage serving engines + instances.
//!
//! Left Actions bento (the locked Serving verb list) + right `Details: <verb>`
//! bento. Each verb opens an EXISTING manager through the EXISTING seam
//! (`KeyAction::Open*`); the shared renderer lives in [`super::pane`] so ROCm
//! and Serving cannot drift. "Optimize a model" is a display-only `soon` row.

use ratatui::Frame;
use ratatui::layout::Rect;

use super::pane::{self, Verb};
use crate::app::{AppState, KeyAction};
use crate::ui::theme::Theme;

/// The Serving Actions list, in display order. The last row is display-only.
pub const VERBS: &[Verb] = &[
    Verb {
        icon: "◆",
        label: "Serve a model",
        action: KeyAction::OpenServeWizard,
        summary: "Launch a model on a serving engine and expose an OpenAI-style endpoint.",
        steps: &[
            "Pick a model",
            "Choose an engine — vLLM · SGLang · llama.cpp · PyTorch",
            "Set GPU placement (required / preferred / CPU-only)",
            "Launch on 127.0.0.1:11435 and watch it come up",
        ],
        cmd: "rocm serve --engine …",
        read_only: false,
        badge: None,
    },
    Verb {
        icon: "⌬",
        label: "Engines",
        action: KeyAction::OpenEngineManager,
        summary: "Install, reinstall, or configure serving engines.",
        steps: &[
            "Browse engines — vLLM · SGLang · llama.cpp · PyTorch · lemonade",
            "See install status per engine",
            "Install / reinstall an engine",
            "Adjust engine config",
        ],
        cmd: "rocm engines …",
        read_only: false,
        badge: None,
    },
    Verb {
        icon: "▤",
        label: "Running instances/services",
        action: KeyAction::OpenServices,
        summary: "See managed inference servers and stop or restart them.",
        steps: &[
            "List managed services — model · port · status · tok/s",
            "Select a running service",
            "Stop or restart it (asks first)",
            "Watch the lifecycle job complete",
        ],
        cmd: "rocm services …",
        read_only: false,
        badge: None,
    },
    Verb {
        icon: "⮌",
        label: "Providers & keys",
        action: KeyAction::OpenConfig,
        summary: "Manage AI providers and API keys.",
        steps: &[
            "Show saved config",
            "Enable a provider — local · anthropic · openai",
            "Disable a provider",
            "Keys stay in your local config",
        ],
        cmd: "rocm config …",
        read_only: false,
        badge: None,
    },
    Verb {
        icon: "☰",
        label: "Logs",
        action: KeyAction::OpenLogs,
        summary: "Browse recent ROCm CLI logs (read-only).",
        steps: &[
            "Optionally type search terms",
            "Run `rocm logs [--search …]`",
            "Scroll the streamed output",
        ],
        cmd: "rocm logs",
        read_only: true,
        badge: None,
    },
    Verb {
        icon: "⚡",
        label: "Optimize a model",
        action: KeyAction::Nothing,
        summary: "Tune a served model for throughput / latency.",
        steps: &[
            "Pick a served model",
            "Choose an optimization profile",
            "Apply and re-benchmark",
        ],
        cmd: "",
        read_only: false,
        badge: Some("soon"),
    },
];

/// Number of rows in the Serving Actions list (its selection length).
pub const VERB_COUNT: usize = VERBS.len();

/// Resolve the seam action for the verb at `sel` (clamped).
#[must_use]
pub fn verb_action(sel: usize) -> KeyAction {
    pane::verb_action(VERBS, sel)
}

/// Map a left-click in the Serving body to an action.
#[must_use]
pub fn hit_test(area: Rect, x: u16, y: u16) -> Option<KeyAction> {
    pane::hit_test(area, x, y, VERB_COUNT)
}

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    pane::draw(
        f,
        area,
        "Serving actions",
        VERBS,
        state.serving_sel,
        state.pane_focus,
        theme,
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
    fn serving_renders_its_six_rows() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Serving;
        let out = render(&s, 120, 28);
        for label in [
            "Serve a model",
            "Engines",
            "Running instances/services",
            "Providers & keys",
            "Logs",
            "Optimize a model",
        ] {
            assert!(
                out.contains(label),
                "Serving row {label:?} missing: {out:?}"
            );
        }
        // The display-only row carries its `soon` badge.
        assert!(out.contains("soon"), "optimize soon badge missing: {out:?}");
    }

    #[test]
    fn serving_verb_action_maps_selection_to_seam() {
        assert_eq!(verb_action(0), KeyAction::OpenServeWizard);
        assert_eq!(verb_action(1), KeyAction::OpenEngineManager);
        // Display-only Optimize is a safe no-op.
        assert_eq!(verb_action(5), KeyAction::Nothing);
        assert_eq!(verb_action(99), KeyAction::Nothing);
    }

    #[test]
    fn serving_does_not_panic_when_squeezed() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Serving;
        for h in [1u16, 2, 3, 5, 10] {
            let _ = render(&s, 60, h);
        }
    }

    #[test]
    fn serving_hit_test_selects_rows_and_activates_detail() {
        let area = Rect::new(0, 0, 100, 24);
        let cols = ratatui::layout::Layout::default()
            .direction(ratatui::layout::Direction::Horizontal)
            .constraints([
                ratatui::layout::Constraint::Percentage(46),
                ratatui::layout::Constraint::Percentage(54),
            ])
            .split(area);
        let list = cols[0];
        let inner = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .padding(crate::ui::panel::padding_for(list))
            .inner(list);
        assert_eq!(
            hit_test(area, inner.x + 1, inner.y),
            Some(KeyAction::PaneSelect(0))
        );
        assert_eq!(hit_test(area, inner.x + 1, inner.y + 1), None);
        assert_eq!(
            hit_test(area, cols[1].x + 3, cols[1].y + 3),
            Some(KeyAction::PaneActivate)
        );
    }
}
