// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! ROCm tab — install, configure, and manage the ROCm platform.
//!
//! Left Actions bento (the locked ROCm verb list) + right `Details: <verb>`
//! bento. Each verb opens an EXISTING manager through the EXISTING seam
//! (`KeyAction::Open*`); the shared renderer lives in [`super::pane`] so ROCm
//! and Serving cannot drift. "Uninstall" is a display-only row (no manager is
//! wired this run — use the CLI).

use ratatui::Frame;
use ratatui::layout::Rect;

use super::pane::{self, Verb};
use crate::app::{AppState, KeyAction};
use crate::ui::theme::Theme;

/// The ROCm Actions list, in display order. The last row is display-only.
pub const VERBS: &[Verb] = &[
    Verb {
        icon: "⚙",
        label: "Set up / Install ROCm",
        action: KeyAction::OpenInstall,
        summary: "Install or repair the ROCm SDK (TheRock).",
        steps: &[
            "Pick a channel (e.g. release)",
            "Choose a format — wheel or tarball",
            "Pick an install folder (prefix)",
            "Dry-run to preview, then apply",
        ],
        cmd: "rocm install --channel … --format …",
        read_only: false,
        badge: None,
    },
    Verb {
        icon: "⇲",
        label: "Check for updates",
        action: KeyAction::OpenUpdate,
        summary: "Check, preview, and apply ROCm package updates.",
        steps: &[
            "Check for updates",
            "Preview the update (dry-run)",
            "Apply the update and activate it",
        ],
        cmd: "rocm update --check",
        read_only: false,
        badge: None,
    },
    Verb {
        icon: "⚕",
        label: "Diagnose & fix  (doctor)",
        action: KeyAction::OpenExamine,
        summary: "Read-only environment check that flags what needs fixing.",
        steps: &[
            "Run `rocm doctor` (one job, no approval)",
            "Review runtime / driver / permission checks",
            "Re-run after fixes with r",
        ],
        cmd: "rocm doctor",
        read_only: true,
        badge: None,
    },
    Verb {
        icon: "⟲",
        label: "Runtimes",
        action: KeyAction::OpenRuntimes,
        summary: "List and manage registered ROCm runtime installs.",
        steps: &[
            "Refresh the registered runtimes (rocm runtimes list)",
            "Activate or roll back a runtime",
            "Adopt an existing ROCm env folder",
            "Import a runtime manifest",
        ],
        cmd: "rocm runtimes …",
        read_only: false,
        badge: None,
    },
    Verb {
        icon: "⌘",
        label: "Command runner",
        action: KeyAction::OpenCommand,
        summary: "Run any rocm subcommand through the approval gate.",
        steps: &[
            "Type a `rocm …` subcommand",
            "Review the exact argv",
            "Approve to run it as a background job",
            "Watch the output stream in the console",
        ],
        cmd: "rocm …",
        read_only: false,
        badge: None,
    },
    Verb {
        icon: "⊘",
        label: "Uninstall",
        action: KeyAction::Nothing,
        summary: "Remove a ROCm install. Not wired into the dashboard yet — use the CLI.",
        steps: &[
            "Run `rocm uninstall --dry-run` to preview",
            "Run `rocm uninstall` to apply",
        ],
        cmd: "",
        read_only: false,
        badge: None,
    },
];

/// Number of rows in the ROCm Actions list (its selection length).
pub const VERB_COUNT: usize = VERBS.len();

/// Resolve the seam action for the verb at `sel` (clamped).
#[must_use]
pub fn verb_action(sel: usize) -> KeyAction {
    pane::verb_action(VERBS, sel)
}

/// Map a left-click in the ROCm body to an action.
#[must_use]
pub fn hit_test(area: Rect, x: u16, y: u16) -> Option<KeyAction> {
    pane::hit_test(area, x, y, VERB_COUNT)
}

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    pane::draw(f, area, "ROCm actions", VERBS, state.rocm_sel, state, theme);
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
    fn rocm_renders_its_six_rows() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        let out = render(&s, 120, 28);
        for label in [
            "Set up / Install ROCm",
            "Check for updates",
            "doctor",
            "Runtimes",
            "Command runner",
            "Uninstall",
        ] {
            assert!(out.contains(label), "ROCm row {label:?} missing: {out:?}");
        }
    }

    #[test]
    fn rocm_detail_shows_installed_rocm_version() {
        use rocm_dash_core::metrics::{GpuSystemInfo, Snapshot};
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.rocm_sel = 0; // "Set up / Install ROCm" → OpenInstall
        s.latest = Some(Snapshot {
            gpu_system_info: Some(GpuSystemInfo {
                rocm_version: Some("6.4.1".into()),
                ..Default::default()
            }),
            ..Default::default()
        });
        let out = render(&s, 120, 28);
        assert!(
            out.contains("Installed now"),
            "live install header missing: {out:?}"
        );
        assert!(
            out.contains("6.4.1"),
            "detected ROCm version not shown: {out:?}"
        );
    }

    #[test]
    fn rocm_verb_action_maps_selection_to_seam() {
        assert_eq!(verb_action(0), KeyAction::OpenInstall);
        assert_eq!(verb_action(2), KeyAction::OpenExamine);
        // Display-only Uninstall is a safe no-op.
        assert_eq!(verb_action(5), KeyAction::Nothing);
        // Out-of-range is a safe no-op.
        assert_eq!(verb_action(99), KeyAction::Nothing);
    }

    #[test]
    fn rocm_does_not_panic_when_squeezed() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        for h in [1u16, 2, 3, 5, 10] {
            let _ = render(&s, 60, h);
        }
    }

    #[test]
    fn rocm_hit_test_selects_rows_and_activates_detail() {
        let area = Rect::new(0, 0, 100, 24);
        // Reconstruct the Actions list inner rect (rounded border + padding).
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
        // First verb row selects row 0; the blank row between selects nothing.
        assert_eq!(
            hit_test(area, inner.x + 1, inner.y),
            Some(KeyAction::PaneSelect(0))
        );
        assert_eq!(hit_test(area, inner.x + 1, inner.y + 1), None);
        // A click in the detail column activates.
        assert_eq!(
            hit_test(area, cols[1].x + 3, cols[1].y + 3),
            Some(KeyAction::PaneActivate)
        );
    }
}
