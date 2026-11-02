// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The Help overlay and the other `Modal::*` renderers.
//!
//! Modal content is rendered by the active tab (`detail_modal` from the tab
//! module) or by `draw_help` here. The popup geometry and the
//! Clear-then-block pattern (`centered_rect`, `centered_width`,
//! `draw_popup_frame`, `grey_overlay`) live in `ui/panel.rs`, so the underlying
//! body shows through the gaps; this module imports them from there, and
//! re-exports those four helpers so their pre-split `ui::modal` paths still
//! resolve.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Wrap};

use crate::app::{AppState, OPTIONS_TABS, PALETTE_DESTS};
use crate::ui::help_modal::popup_height_for;
use crate::ui::panel::{self, BoxRole};
// Re-exported, not just imported: these four were public at this path before
// the geometry moved into `panel`, so the pre-split public paths stay valid.
pub use crate::ui::panel::{centered_rect, centered_width, draw_popup_frame, grey_overlay};
use crate::ui::theme::Theme;

/// Lines of text a multi-line warning is cut to once it is longer than this plus
/// one. A failed amd-smi call embeds its raw stderr, which can run to dozens of
/// lines; uncapped it would push every warning listed after it off a popup that
/// does not scroll. A warning only one line over the cap is shown whole, since
/// "1 more line" costs the row it saves.
///
/// The cap counts lines of the warning's text, not the rows they wrap to, so one
/// very long line can still take several rows. A popup that runs out of room
/// says so with its truncation marker (see [`draw_warnings`]).
const WARNING_MAX_LINES: usize = 4;

/// One warning as popup lines: a bullet on the first line, an indent on the
/// continuation lines of a multi-line one, and a count of what the cap cut.
fn warning_lines<'a>(warning: &'a str, theme: &Theme) -> Vec<Line<'a>> {
    let bullet = Style::default().fg(theme.warn);
    let text = Style::default().fg(theme.fg);
    let all: Vec<&str> = warning
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .collect();
    if all.is_empty() {
        return vec![Line::from(Span::styled("• ", bullet))];
    }
    let shown = if all.len() <= WARNING_MAX_LINES + 1 {
        all.len()
    } else {
        WARNING_MAX_LINES
    };
    let mut out: Vec<Line> = all[..shown]
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let lead = if i == 0 { "• " } else { "  " };
            Line::from(vec![Span::styled(lead, bullet), Span::styled(*l, text)])
        })
        .collect();
    if all.len() > shown {
        out.push(Line::from(Span::styled(
            format!("  … {} more line(s)", all.len() - shown),
            Style::default().fg(theme.muted),
        )));
    }
    out
}

/// Render the Warnings modal: the text behind the header `⚠ N` badge.
///
/// It lists the warnings the daemon currently reports. A multi-line one longer
/// than [`WARNING_MAX_LINES`] + 1 lines is cut to [`WARNING_MAX_LINES`] lines of
/// text with a count of the rest.
///
/// Content-sized like [`draw_help`] — the warnings are few and short-lived, so
/// the popup is sized to them rather than growing a scroll position of its own.
/// On a terminal too short for the close hint, the hint goes first (the footer
/// still shows Esc); only when the warnings themselves do not fit does the last
/// row say so. That marker is in the body, not the border title as in
/// [`draw_help`]: a title marker needs a popup at least ~38 columns wide and is
/// silently dropped on a narrower one, which would leave cut-off warning text
/// unmarked.
pub fn draw_warnings(f: &mut Frame, area: Rect, warnings: &[String], theme: &Theme) {
    grey_overlay(f);
    let body: Vec<Line> = if warnings.is_empty() {
        // The daemon can clear its warnings while the popup is open.
        vec![Line::from(Span::styled(
            "No active warnings.",
            Style::default().fg(theme.muted),
        ))]
    } else {
        warnings
            .iter()
            .flat_map(|w| warning_lines(w, theme))
            .collect()
    };
    let mut with_hint = body.clone();
    with_hint.push(Line::raw(""));
    with_hint.push(Line::from(Span::styled(
        "Esc / Enter / ! close",
        Style::default().fg(theme.muted),
    )));

    let width = centered_width(70, 80, area);
    let rows_for = |lines: &[Line]| {
        let p = Paragraph::new(lines.to_vec()).wrap(Wrap { trim: false });
        popup_height_for(p.line_count(width.saturating_sub(2)))
    };
    let needed = rows_for(&with_hint);
    let popup = centered_rect(70, 100, 80, needed, area);
    let (lines, popup, truncated) = if needed <= popup.height {
        (with_hint, popup, false)
    } else {
        let needed = rows_for(&body);
        let popup = centered_rect(70, 100, 80, needed, area);
        (body, popup, needed > popup.height)
    };

    let title = if warnings.is_empty() {
        "Warnings".to_string()
    } else {
        format!("Warnings ({})", warnings.len())
    };
    let inner = draw_popup_frame(f, popup, &title, theme);
    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    if truncated && inner.height > 0 {
        let body_area = Rect::new(inner.x, inner.y, inner.width, inner.height - 1);
        f.render_widget(p, body_area);
        let marker = Paragraph::new(Line::from(Span::styled(
            "… truncated — resize",
            Style::default().fg(theme.warn),
        )));
        let marker_row = Rect::new(inner.x, inner.y + body_area.height, inner.width, 1);
        f.render_widget(marker, marker_row);
    } else {
        f.render_widget(p, inner);
    }
}

// ===========================================================================
// P4 overlays: Esc menu, command palette, tabbed Options, global Help. They
// compose the Phase-1 chrome helpers (grey_overlay / draw_logo / opt_row /
// draw_tab_panel) over the stable 4-tab body.
// ===========================================================================

/// Row (relative to `inner.y`) where the Esc menu's selectable items begin,
/// just below the 5-row logo (rows 1..=5).
const MENU_ITEMS_Y: u16 = 8;
/// Number of selectable rows in the Esc menu (Options / Help / Quit).
const MENU_ITEM_COUNT: u16 = 3;
/// Minimum inner width the logo needs before the menu will render.
const MENU_MIN_WIDTH: u16 = 31;

/// Whether the Esc menu's inner box is tall and wide enough to show the logo
/// AND every selectable item. The old guard only checked the logo
/// (`inner.height < 6`), so inner heights 6..=10 painted the logo with no
/// reachable Options/Help/Quit. Items occupy rows
/// `MENU_ITEMS_Y .. MENU_ITEMS_Y + MENU_ITEM_COUNT`, so the box must be at least
/// `MENU_ITEMS_Y + MENU_ITEM_COUNT` rows tall (11) before the menu is drawn.
const fn menu_fits(inner_height: u16, inner_width: u16) -> bool {
    inner_width >= MENU_MIN_WIDTH && inner_height >= MENU_ITEMS_Y + MENU_ITEM_COUNT
}

/// Esc main menu: Home backdrop dimmed by `grey_overlay`, a double-border modal
/// with the btop `draw_logo` and the Options/Help/Quit list.
pub fn draw_menu(f: &mut Frame, area: Rect, sel: usize, theme: &Theme) {
    grey_overlay(f);
    let modal = centered_rect(50, 70, 60, 17, area);
    f.render_widget(Clear, modal);
    let inner = panel::bento(f, modal, None, BoxRole::Primary, false, theme);
    if !menu_fits(inner.height, inner.width) {
        return;
    }
    let logo_w = 31u16;
    let cx = inner.x + inner.width.saturating_sub(logo_w) / 2;
    draw_logo(f, cx, inner.y + 1, theme);

    let items = ["Options", "Help", "Quit"];
    let mx = inner.x + 4;
    for (i, label) in items.iter().enumerate() {
        let y = inner.y + MENU_ITEMS_Y + i as u16;
        if y >= inner.y + inner.height {
            break;
        }
        let focused = i == sel;
        let (cur, st) = if focused {
            (
                "▸ ",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            ("  ", Style::default().fg(theme.fg))
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(cur, Style::default().fg(theme.accent)),
                Span::styled(*label, st),
            ])),
            Rect::new(mx, y, inner.width.saturating_sub(4), 1),
        );
    }
}

/// Command palette: dimmed backdrop + centered "Go to…" card with a `:` filter
/// line and the destination rows.
pub fn draw_palette(f: &mut Frame, area: Rect, sel: usize, theme: &Theme) {
    grey_overlay(f);
    let modal = centered_rect(50, 60, 54, 12, area);
    let inner = draw_popup_frame(f, modal, "Go to…", theme);
    if inner.height == 0 {
        return;
    }
    f.render_widget(Clear, inner);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            ": type to filter",
            Style::default().fg(theme.muted),
        ))),
        Rect::new(inner.x, inner.y, inner.width, 1),
    );
    for (i, (label, _)) in PALETTE_DESTS.iter().enumerate() {
        let y = inner.y + 2 + i as u16;
        if y >= inner.y + inner.height {
            break;
        }
        let focused = i == sel;
        let (cur, st) = if focused {
            (
                "▸ ",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            ("  ", Style::default().fg(theme.fg))
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(cur, Style::default().fg(theme.accent)),
                Span::styled(*label, st),
            ])),
            Rect::new(inner.x, y, inner.width, 1),
        );
    }
}

/// Tabbed Options panel reusing the outlined tab renderer + `opt_row`.
pub fn draw_options(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    grey_overlay(f);
    let modal = centered_rect(80, 80, 112, 26, area);
    f.render_widget(Clear, modal);
    let _ = panel::bento(f, modal, None, BoxRole::Neutral, false, theme);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            " ⚙  Options",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))),
        Rect::new(modal.x + 2, modal.y, modal.width.saturating_sub(2), 1),
    );
    let panel = Rect::new(
        modal.x + 1,
        modal.y + 1,
        modal.width.saturating_sub(2),
        modal.height.saturating_sub(2),
    );
    let inner = crate::ui::tabs::draw_tab_panel(f, panel, OPTIONS_TABS, state.options_tab, theme);
    if inner.height == 0 {
        return;
    }
    // Render a few representative rows for the active settings tab. Real config
    // is wired where it exists; net-new toggles are display-with-intent.
    let rows: &[(&str, String, &str)] = match state.options_tab {
        0 => &[
            ("Theme", state.theme_name.clone(), "◂ t ▸"),
            // ponytail: telemetry/refresh toggles are display-with-intent — no
            // new persisted config store is invented this run.
            ("Telemetry", "local-only".to_string(), "—"),
            ("Refresh", "1s".to_string(), "—"),
        ],
        1 => &[("Per-core bars", "on".to_string(), "—")],
        2 => &[("Gradient gauges", "on".to_string(), "—")],
        _ => &[("Default engine", "auto".to_string(), "—")],
    };
    for (i, (label, value, control)) in rows.iter().enumerate() {
        let y = inner.y + 1 + i as u16 * 2;
        if y >= inner.y + inner.height {
            break;
        }
        opt_row(
            f,
            Rect::new(inner.x + 2, y, inner.width.saturating_sub(4), 1),
            label,
            value,
            control,
            i == 0,
            theme,
        );
    }
}

// ===========================================================================
// btop-style chrome helpers. Pure draw fns composed by the P4 overlays above.
// ===========================================================================

/// Stamp a string at `(x, y)` if the row is on-screen.
fn put(f: &mut Frame, x: u16, y: u16, s: &str, style: Style) {
    if y < f.area().height {
        f.buffer_mut().set_string(x, y, s, style);
    }
}

/// Big block "ROCm" wordmark with a horizontal accent→cyan gradient sweep.
///
/// `cx` is the left column of the 31-wide logo; it occupies 5 rows from `y`.
/// Reuses the crate's [`gradient::lerp3_t`] ramp rather than a local lerp.
pub fn draw_logo(f: &mut Frame, cx: u16, y: u16, theme: &Theme) {
    use ratatui::style::Color;
    const R: [&str; 5] = ["██████ ", "██   ██", "██████ ", "██   ██", "██   ██"];
    const O: [&str; 5] = [" █████ ", "██   ██", "██   ██", "██   ██", " █████ "];
    const C: [&str; 5] = [" ██████", "██     ", "██     ", "██     ", " ██████"];
    // lowercase "m": blank top row, sits on the baseline like R/O/C bottoms.
    const M: [&str; 5] = ["       ", "██████ ", "██ █ ██", "██ █ ██", "██ █ ██"];
    // ponytail: btop gradient is accent_2 → accent → bright cyan; the bright
    // stop is a fixed light cyan (matches the mock) rather than a theme token.
    let light = Color::Rgb(0xc4, 0xf2, 0xff);
    let stops = [theme.accent_2, theme.accent, light];
    for i in 0..5 {
        let line = format!("{} {} {} {}", R[i], O[i], C[i], M[i]);
        let n = line.chars().count().max(2);
        for (j, ch) in line.chars().enumerate() {
            if ch != ' ' {
                let t = j as f64 / (n - 1) as f64;
                put(
                    f,
                    cx + j as u16,
                    y + i as u16,
                    &ch.to_string(),
                    Style::default().fg(crate::ui::gradient::lerp3_t(stops, t)),
                );
            }
        }
    }
}

/// One settings row for the Options panel: focusable label on the left, value +
/// control hint right-aligned.
pub fn opt_row(
    f: &mut Frame,
    area: Rect,
    label: &str,
    value: &str,
    control: &str,
    focused: bool,
    theme: &Theme,
) {
    let (cur, lc) = if focused {
        ("▸ ", theme.accent)
    } else {
        ("  ", theme.fg)
    };
    let lstyle = if focused {
        Style::default().fg(lc).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(lc)
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(cur, Style::default().fg(theme.accent)),
            Span::styled(label, lstyle),
        ])),
        Rect::new(area.x, area.y, area.width, 1),
    );
    let val_w = (value.chars().count() + control.chars().count() + 2) as u16;
    let vx = area.x + area.width.saturating_sub(val_w);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(value, Style::default().fg(theme.accent)),
            Span::raw("  "),
            Span::styled(control, Style::default().fg(theme.muted)),
        ])),
        Rect::new(vx, area.y, val_w, 1),
    );
}

#[cfg(test)]
mod ported_chrome_tests {
    use super::{draw_logo, grey_overlay, opt_row};
    use crate::ui::tests::{bottom_border_row, flat, rows};
    use crate::ui::theme::Theme;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    #[test]
    fn esc_menu_requires_room_for_all_items() {
        use super::{MENU_MIN_WIDTH, menu_fits};
        // Logo occupies rows 1..=5; the three items render at rows 8,9,10. The
        // old `inner.height < 6` guard let inner heights 6,7,8 paint the logo
        // with NO reachable Options/Help/Quit, and heights 9,10 cut items off.
        // `menu_fits` must reject the whole broken range and accept only once
        // all three items fit (inner.height >= 11). This test FAILS against the
        // old `< 6` guard (which accepted 6..=10).
        let w = MENU_MIN_WIDTH;
        for h in [6u16, 7, 8, 9, 10] {
            assert!(
                !menu_fits(h, w),
                "inner height {h} must not render the logo without all items"
            );
        }
        assert!(menu_fits(11, w), "height 11 must fit Options/Help/Quit");
        assert!(menu_fits(12, w), "height 12 must fit the menu");
        // Width guard preserved: a too-narrow box never renders the menu.
        assert!(
            !menu_fits(20, MENU_MIN_WIDTH - 1),
            "a box narrower than the logo must not render"
        );
    }

    #[test]
    fn draw_logo_paints_block_wordmark() {
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(40, 6);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw_logo(f, 1, 0, &theme)).unwrap();
        let out = flat(&term);
        assert!(out.contains('█'), "logo block glyphs missing: {out:?}");
    }

    #[test]
    fn grey_overlay_dims_every_cell() {
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(10, 3);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            draw_logo(f, 0, 0, &theme);
            grey_overlay(f);
        })
        .unwrap();
        // Every cell should carry the wash bg after the overlay.
        let wash = ratatui::style::Color::Rgb(0x1c, 0x1e, 0x22);
        let buf = term.backend().buffer();
        assert!(
            buf.content().iter().all(|c| c.style().bg == Some(wash)),
            "overlay did not wash every cell"
        );
    }

    #[test]
    fn p4_overlays_render_key_content() {
        use crate::app::{ActiveTab, AppState};
        let theme = Theme::from_name("default-dark");
        let area = ratatui::layout::Rect::new(0, 0, 120, 30);

        let render = |draw: &dyn Fn(&mut ratatui::Frame)| -> String {
            let backend = TestBackend::new(120, 30);
            let mut term = Terminal::new(backend).unwrap();
            term.draw(|f| draw(f)).unwrap();
            term.backend()
                .buffer()
                .content()
                .iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect()
        };

        let menu = render(&|f| super::draw_menu(f, area, 0, &theme));
        assert!(menu.contains("Options"), "menu missing Options: {menu:?}");
        assert!(menu.contains("Quit"), "menu missing Quit");

        let palette = render(&|f| super::draw_palette(f, area, 0, &theme));
        assert!(
            palette.contains("Go to"),
            "palette missing Go to: {palette:?}"
        );

        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Home;
        let options = render(&|f| super::draw_options(f, area, &s, &theme));
        assert!(
            options.contains("Options"),
            "options missing title: {options:?}"
        );
        assert!(options.contains("General"), "options missing tab label");

        let help = render(&|f| {
            crate::ui::help_modal::draw_global_help(f, area, &theme);
        });
        assert!(
            help.contains("Keyboard"),
            "global help missing title: {help:?}"
        );
        assert!(help.contains("NAVIGATE"), "global help missing group");
    }

    #[test]
    fn opt_row_renders_label_value_control() {
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(40, 1);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            opt_row(
                f,
                Rect::new(0, 0, 40, 1),
                "Theme",
                "tokyo",
                "▸◂",
                true,
                &theme,
            );
        })
        .unwrap();
        let out = flat(&term);
        assert!(out.contains("Theme"), "label missing: {out:?}");
        assert!(out.contains("tokyo"), "value missing: {out:?}");
        assert!(out.contains('▸'), "focus/control marker missing: {out:?}");
    }

    #[test]
    fn warnings_modal_lists_every_warning_with_a_count() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec![
            "amd-smi unavailable (alpha)".to_string(),
            "vram attribution: beta".to_string(),
            "vllm scrape: gamma".to_string(),
        ];
        let area = Rect::new(0, 0, 80, 30);
        let painted = rows(80, 30, |f| {
            super::draw_warnings(f, area, &warnings, &theme);
        })
        .join("\n");
        for w in &warnings {
            assert!(painted.contains(w.as_str()), "missing {w:?}:\n{painted}");
        }
        assert!(painted.contains("Warnings (3)"), "{painted}");
    }

    #[test]
    fn warnings_modal_does_not_clip_a_long_warning_on_a_narrow_terminal() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec![
            "amd-smi is missing, unresolvable, or failed to run, even though a GPU was \
             detected by other means - it may not be supported on this platform yet"
                .to_string(),
        ];
        let area = Rect::new(0, 0, 40, 30);
        let painted = rows(40, 30, |f| {
            super::draw_warnings(f, area, &warnings, &theme);
        })
        .join("\n");
        assert!(
            painted.contains("platform yet"),
            "the tail of the warning must survive wrapping:\n{painted}"
        );
    }

    #[test]
    fn warnings_modal_says_so_when_the_warnings_have_cleared() {
        let theme = Theme::from_name("default-dark");
        let area = Rect::new(0, 0, 80, 30);
        let painted = rows(80, 30, |f| super::draw_warnings(f, area, &[], &theme)).join("\n");
        assert!(painted.contains("No active warnings."), "{painted}");
    }

    #[test]
    fn warnings_modal_marks_truncation_when_the_terminal_is_too_short() {
        let theme = Theme::from_name("default-dark");
        let warnings: Vec<String> = (0..12).map(|i| format!("warning number {i}")).collect();
        // 80 columns is wide enough for a title marker; 40 is not. The cut must
        // be marked either way, since the warnings are what the popup is for.
        for (w, h) in [(80u16, 8u16), (40, 8)] {
            let area = Rect::new(0, 0, w, h);
            let painted = rows(w, h, |f| super::draw_warnings(f, area, &warnings, &theme));
            let bottom = bottom_border_row(&painted);
            assert!(
                painted[bottom - 1].contains("truncated"),
                "{w}x{h}: the marker takes the last inner row:\n{}",
                painted.join("\n")
            );
            assert!(
                painted[bottom - 2].contains("warning number"),
                "{w}x{h}: the marker must not overwrite the last body row:\n{}",
                painted.join("\n")
            );
        }
    }

    #[test]
    fn warnings_modal_does_not_mark_truncation_when_everything_fits() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec!["only one".to_string()];
        let area = Rect::new(0, 0, 80, 30);
        let painted = rows(80, 30, |f| super::draw_warnings(f, area, &warnings, &theme)).join("\n");
        assert!(!painted.contains("truncated"), "{painted}");
        assert!(painted.contains("Esc / Enter / ! close"), "{painted}");
    }

    #[test]
    fn warnings_modal_drops_the_close_hint_before_it_cuts_a_warning() {
        let theme = Theme::from_name("default-dark");
        let warnings: Vec<String> = (1..=3).map(|i| format!("warning {i}")).collect();
        // Five rows: three of warnings plus the borders. The blank row and the
        // close hint do not fit, and are the only things that should go.
        let area = Rect::new(0, 0, 80, 5);
        let painted = rows(80, 5, |f| super::draw_warnings(f, area, &warnings, &theme)).join("\n");
        for w in &warnings {
            assert!(painted.contains(w.as_str()), "{painted}");
        }
        assert!(!painted.contains("Esc / Enter"), "{painted}");
        assert!(!painted.contains("truncated"), "nothing was cut: {painted}");
    }

    /// Paint one warning of `n` lines (`row 1` .. `row n`, after a first line) at a
    /// size with room to spare, followed by a second warning.
    fn painted_multi_line_warning(n: usize) -> String {
        let theme = Theme::from_name("default-dark");
        let mut lines = vec!["first line".to_string()];
        lines.extend((1..n).map(|i| format!("row {i}")));
        let warnings = vec![lines.join("\n"), "the warning after it".to_string()];
        let area = Rect::new(0, 0, 80, 40);
        rows(80, 40, |f| super::draw_warnings(f, area, &warnings, &theme)).join("\n")
    }

    #[test]
    fn warnings_modal_cuts_a_long_multi_line_warning_to_four_lines_and_counts_the_rest() {
        // 31 lines: the first plus 30 of stderr.
        let painted = painted_multi_line_warning(31);
        for shown in ["• first line", "  row 1", "  row 2", "  row 3"] {
            assert!(painted.contains(shown), "{shown:?}:\n{painted}");
        }
        assert!(!painted.contains("row 4"), "{painted}");
        assert!(painted.contains("… 27 more line(s)"), "{painted}");
        assert!(painted.contains("the warning after it"), "{painted}");
    }

    #[test]
    fn warnings_modal_shows_a_warning_of_up_to_five_lines_whole() {
        // Four lines fit the cap; five would hide one line behind a row that
        // costs the same, so it is shown whole too.
        for n in [4usize, 5] {
            let painted = painted_multi_line_warning(n);
            assert!(
                painted.contains(&format!("row {}", n - 1)),
                "{n}:\n{painted}"
            );
            assert!(!painted.contains("more line(s)"), "{n}:\n{painted}");
        }
        let six = painted_multi_line_warning(6);
        assert!(!six.contains("row 4"), "{six}");
        assert!(six.contains("… 2 more line(s)"), "{six}");
    }

    #[test]
    fn warnings_modal_lists_a_blank_warning_as_a_bare_bullet() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec![String::new(), "second".to_string()];
        let area = Rect::new(0, 0, 80, 20);
        let painted = rows(80, 20, |f| super::draw_warnings(f, area, &warnings, &theme));
        let bullets = painted.iter().filter(|r| r.contains('•')).count();
        assert_eq!(
            bullets,
            2,
            "one bullet per warning:\n{}",
            painted.join("\n")
        );
    }

    // Warning text is often raw amd-smi stderr. Nothing in this crate strips it:
    // ratatui drops every grapheme holding a control character when it writes the
    // buffer, so an escape sequence cannot reach the terminal. This pins that.
    #[test]
    fn warnings_modal_paints_no_control_character_from_a_warning() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec![
            "\x1b[31mred\x1b[0m\x07 bell\ttab\rreturn\x1b]0;title\x07\ndel\x7f after".to_string(),
        ];
        let area = Rect::new(0, 0, 80, 20);
        let backend = TestBackend::new(80, 20);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| super::draw_warnings(f, area, &warnings, &theme))
            .unwrap();
        let buf = term.backend().buffer();
        let painted: String = (0..20)
            .flat_map(|y| (0..80).map(move |x| (x, y)))
            .map(|pos| buf[pos].symbol().to_string())
            .collect();
        assert!(
            !painted.chars().any(char::is_control),
            "a control character reached the buffer: {painted:?}"
        );
        assert!(painted.contains("red"), "{painted:?}");
        assert!(
            painted.contains("after"),
            "the second line must survive: {painted:?}"
        );
    }
}
