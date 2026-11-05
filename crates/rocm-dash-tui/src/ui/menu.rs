// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The Esc main menu, the command palette and the tabbed Options panel
//! (`Modal::Menu`, `Modal::Palette`, `Modal::Options`), with the block-letter
//! "ROCm" logo (`draw_logo`) and the settings row (`opt_row`) they compose.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};

use crate::app::{AppState, MENU_ITEMS, OPTIONS_TABS, PALETTE_DESTS};
use crate::ui::panel::{self, BoxRole, centered_rect, draw_popup_frame, grey_overlay};
use crate::ui::theme::Theme;

// ===========================================================================
// P4 overlays: Esc menu, command palette and tabbed Options. They
// compose the Phase-1 chrome helpers (grey_overlay / draw_logo / opt_row /
// draw_tab_panel) over the stable 4-tab body.
// ===========================================================================

/// Row (relative to `inner.y`) where the Esc menu's selectable items begin,
/// just below the 5-row logo (rows 1..=5).
const MENU_ITEMS_Y: u16 = 8;
/// Number of selectable rows in the Esc menu, counted from the one list
/// ([`MENU_ITEMS`]) the key handling and the drawn rows also use.
const MENU_ITEM_COUNT: u16 = MENU_ITEMS.len() as u16;
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

    let mx = inner.x + 4;
    for (i, item) in MENU_ITEMS.iter().enumerate() {
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
                Span::styled(item.label(), st),
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
mod tests {
    use super::{MENU_ITEMS, draw_logo, grey_overlay, opt_row};
    use crate::ui::tests::flat;
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

    /// The text between the menu box's side borders, outer blanks trimmed.
    fn inside(row: &str) -> &str {
        row.trim().trim_matches('│').trim()
    }

    /// Render `draw_menu` on a `w`x`h` screen with `sel` focused and return the
    /// rows below the logo that carry menu text inside the box, as
    /// `(screen row, text inside the borders)`. The bottom border does not
    /// start with a side border, so it is excluded.
    fn drawn_item_rows(w: u16, h: u16, sel: usize) -> Vec<(usize, String)> {
        let theme = Theme::from_name("default-dark");
        let area = Rect::new(0, 0, w, h);
        let rows = crate::ui::tests::rows(w, h, |f| super::draw_menu(f, area, sel, &theme));
        let logo_end = rows
            .iter()
            .rposition(|r| r.contains('█'))
            .expect("the logo is drawn when the menu fits");
        rows.iter()
            .enumerate()
            .skip(logo_end + 1)
            .filter(|(_, r)| r.trim_start().starts_with('│') && !inside(r).is_empty())
            .map(|(y, r)| (y, inside(r).to_string()))
            .collect()
    }

    #[test]
    fn esc_menu_draws_options_help_quit_in_order_with_only_the_selected_row_marked() {
        // Characterisation of the drawn rows: the same three labels in the same
        // order, one "▸ " cursor on the selected row and a blank cursor column
        // on the others.
        for (sel, expected) in [
            (0usize, ["▸ Options", "Help", "Quit"]),
            (1, ["Options", "▸ Help", "Quit"]),
            (2, ["Options", "Help", "▸ Quit"]),
        ] {
            let drawn = drawn_item_rows(120, 30, sel);
            let texts: Vec<&str> = drawn.iter().map(|(_, t)| t.as_str()).collect();
            assert_eq!(texts, expected, "rows drawn with sel={sel}");
            // The three items are on consecutive screen rows.
            assert_eq!(drawn[1].0, drawn[0].0 + 1, "sel={sel}");
            assert_eq!(drawn[2].0, drawn[1].0 + 1, "sel={sel}");
        }
    }

    #[test]
    fn esc_menu_item_rows_sit_at_a_fixed_column_with_focus_styling() {
        use ratatui::style::Modifier;
        let theme = Theme::from_name("default-dark");
        let area = Rect::new(0, 0, 120, 30);
        let mut term = Terminal::new(TestBackend::new(120, 30)).unwrap();
        term.draw(|f| super::draw_menu(f, area, 1, &theme)).unwrap();
        // One render: the row text and the cell styles both come from this buffer.
        let buf = term.backend().buffer();
        let rows: Vec<String> = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect();
        let y_of = |label: &str| -> u16 {
            rows.iter()
                .position(|r| inside(r) == label || inside(r) == format!("▸ {label}"))
                .unwrap_or_else(|| panic!("no row for {label}")) as u16
        };
        // Locate the box's left border on the Help row instead of hardcoding
        // where the popup sits, so a change to the popup's size does not
        // break this test. The cursor column is 7 cells right of that border
        // and the label 2 more.
        let help_y = y_of("Help");
        let border_x = (0..buf.area.width)
            .find(|&x| buf[(x, help_y)].symbol() == "│")
            .expect("the Help row starts inside a bordered box");
        let (cursor_x, label_x) = (border_x + 7, border_x + 9);
        assert_eq!(buf[(cursor_x, help_y)].symbol(), "▸");
        assert_eq!(buf[(cursor_x, help_y)].style().fg, Some(theme.accent));
        assert_eq!(buf[(label_x, help_y)].symbol(), "H");
        assert_eq!(buf[(label_x, help_y)].style().fg, Some(theme.accent));
        assert!(
            buf[(label_x, help_y)]
                .style()
                .add_modifier
                .contains(Modifier::BOLD),
            "the selected label is bold"
        );
        for (label, first) in [("Options", "O"), ("Quit", "Q")] {
            let y = y_of(label);
            assert_eq!(buf[(label_x, y)].symbol(), first, "{label}");
            assert_eq!(buf[(label_x, y)].style().fg, Some(theme.fg), "{label}");
            assert!(
                !buf[(label_x, y)]
                    .style()
                    .add_modifier
                    .contains(Modifier::BOLD),
                "{label} is not bold when unselected"
            );
            assert_eq!(buf[(cursor_x, y)].symbol(), " ", "{label} has no cursor");
        }
    }

    #[test]
    fn esc_menu_is_drawn_only_when_the_box_is_tall_enough_for_every_item() {
        // End-to-end counterpart of `esc_menu_requires_room_for_all_items`: on
        // a 120-column screen the box has 13 rows at height 19 (the logo would
        // show but not every item) and 14 at height 20, the smallest screen on
        // which Options, Help and Quit all fit.
        let theme = Theme::from_name("default-dark");
        let render = |h: u16| {
            let area = Rect::new(0, 0, 120, h);
            crate::ui::tests::rows(120, h, |f| super::draw_menu(f, area, 0, &theme)).join("\n")
        };
        let short = render(19);
        assert!(
            !short.contains('█')
                && !short.contains("Options")
                && !short.contains("Help")
                && !short.contains("Quit"),
            "height 19 must draw an empty frame:\n{short}"
        );
        let just_fits = render(20);
        assert!(just_fits.contains('█'), "logo at height 20");
        for label in ["Options", "Help", "Quit"] {
            assert!(just_fits.contains(label), "{label} at height 20");
        }
    }

    #[test]
    fn esc_menu_draws_the_rows_of_the_list_the_selection_walks() {
        // The drawn rows and the wrap bound of `KeyAction::MenuMove` both come
        // from `MENU_ITEMS` (`app/types.rs`), so what is drawn is exactly what
        // the selection can reach, label for label.
        let drawn = drawn_item_rows(120, 30, 0);
        let drawn: Vec<&str> = drawn
            .iter()
            .map(|(_, t)| t.trim_start_matches("▸ "))
            .collect();
        let listed: Vec<&str> = MENU_ITEMS.iter().map(|i| i.label()).collect();
        assert_eq!(drawn, listed);
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
}
