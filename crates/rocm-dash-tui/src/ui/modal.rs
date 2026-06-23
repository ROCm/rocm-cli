// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Centered popup helpers + the Help overlay.
//!
//! Modal content is rendered by the active tab (`detail_modal` from the tab
//! module) or by `draw_help` here. This module owns the geometry and the
//! Clear-then-block pattern so the underlying body shows through the gaps.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use crate::app::ActiveTab;
use crate::ui::gradient::GradientGauge;
use crate::ui::sparkline::BrailleSparkline;
use crate::ui::theme::{self, Theme};

/// Centered rectangle taking `pct_x`% width and `pct_y`% height of `area`,
/// clamped to a maximum so it doesn't drown the screen on big terminals.
pub fn centered_rect(pct_x: u16, pct_y: u16, max_w: u16, max_h: u16, area: Rect) -> Rect {
    let h_pct = (area.height * pct_y / 100).min(max_h).max(5);
    let v_pad = (area.height.saturating_sub(h_pct)) / 2;
    let vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(v_pad),
            Constraint::Length(h_pct),
            Constraint::Min(0),
        ])
        .split(area);

    let w_pct = (area.width * pct_x / 100).min(max_w).max(20);
    let h_pad = (area.width.saturating_sub(w_pct)) / 2;
    let horiz = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(h_pad),
            Constraint::Length(w_pct),
            Constraint::Min(0),
        ])
        .split(vert[1]);

    horiz[1]
}

/// Render a bordered block with `title` over `area` after clearing it,
/// returning the inner area so the caller can render content into it.
pub fn draw_popup_frame(f: &mut Frame, area: Rect, title: &str, theme: &Theme) -> Rect {
    f.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {title} "))
        .border_style(Style::default().fg(theme.accent))
        .title_style(
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);
    inner
}

/// Shared chrome: a titled popup whose body is a scrollable block of `lines`.
///
/// Centralizes the `draw_modal_*` pattern so operational screens don't rebuild
/// it (Phase 3 Wave 0). `scroll` is the first visible line offset.
pub fn draw_scrollable_lines(
    f: &mut Frame,
    area: Rect,
    title: &str,
    lines: Vec<Line>,
    scroll: u16,
    theme: &Theme,
) {
    let inner = draw_popup_frame(f, area, title, theme);
    if inner.height == 0 {
        return;
    }
    let p = Paragraph::new(lines)
        .scroll((scroll, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(p, inner);
}

/// Render the Help modal for the active tab.
pub fn draw_help(f: &mut Frame, area: Rect, tab: ActiveTab, theme: &Theme) {
    let popup = centered_rect(70, 70, 80, 22, area);
    let inner = draw_popup_frame(f, popup, "Help", theme);

    let mut lines: Vec<Line> = vec![
        key_line("q", "quit", theme),
        key_line("?", "toggle this help", theme),
        key_line("Tab / Shift-Tab", "next / previous tab", theme),
        key_line("1 .. 5", "jump to tab", theme),
        key_line("t", "open theme picker", theme),
        key_line("Space", "pause / resume (replay only)", theme),
        key_line("+ / -", "speed up / slow down (replay only)", theme),
        key_line("[ / ]", "jump ±10s (replay only)", theme),
        key_line("{ / }", "jump ±60s (replay only)", theme),
        Line::raw(""),
    ];
    let tab_help: &[(&str, &str)] = match tab {
        ActiveTab::Overview => &[("(no tab-specific keys)", "")],
        ActiveTab::Hardware => &[
            ("j / Down", "select next GPU (scrolls when list overflows)"),
            (
                "k / Up",
                "select previous GPU (scrolls when list overflows)",
            ),
            ("g / Home", "first GPU"),
            ("G / End", "last GPU"),
            ("Enter", "open GPU detail"),
            ("Esc / Enter", "close any modal"),
        ],
        ActiveTab::Instances => &[
            ("j / Down", "select next instance"),
            ("k / Up", "select previous instance"),
            ("g / Home", "first instance"),
            ("G / End", "last instance"),
            ("Enter", "open instance detail"),
            ("Esc / Enter", "close any modal"),
        ],
        ActiveTab::Bench => &[
            ("j / Down", "select next bench row"),
            ("k / Up", "select previous bench row"),
            ("g / Home", "first row"),
            ("G / End", "last row (newest)"),
            ("Enter", "open row detail"),
            ("Esc / Enter", "close any modal"),
        ],
        ActiveTab::Chat => &[
            ("y / Enter", "accept the detected endpoint (consent prompt)"),
            ("n", "decline / disable chat"),
            ("d", "detect a local engine (gate)"),
            (
                "y / s / n",
                "detected engine: use now / use & save / dismiss",
            ),
            ("i / Enter", "focus the input (insert mode, once enabled)"),
            ("Esc", "leave insert mode"),
            ("Enter", "send the message (while focused)"),
            ("Backspace", "delete a character (while focused)"),
        ],
    };
    lines.push(Line::from(Span::styled(
        format!("— {tab:?} tab —"),
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    )));
    for (k, desc) in tab_help {
        lines.push(key_line(k, desc, theme));
    }

    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(p, inner);
}

fn key_line<'a>(key: &'a str, desc: &'a str, theme: &Theme) -> Line<'a> {
    Line::from(vec![
        Span::styled(
            format!("  {key:<18} "),
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(desc, Style::default().fg(theme.fg)),
    ])
}

/// Theme picker modal. Renders the registered themes as a scrollable list
/// with a five-color swatch preview per entry. The cursor row is highlighted.
///
/// `sel` is the picker cursor; clamped against `theme_names().len()`.
/// `current_name` is the currently-active theme name (rendered with a marker).
pub fn draw_theme_picker(
    f: &mut Frame,
    area: Rect,
    sel: usize,
    current_name: &str,
    active_theme: &Theme,
) {
    let popup = centered_rect(80, 80, 110, 30, area);
    let inner = draw_popup_frame(
        f,
        popup,
        "Theme — j/k select, Enter apply, Esc cancel",
        active_theme,
    );
    if inner.height == 0 {
        return;
    }

    // Split into list (left) + live preview (right). When the popup is too
    // narrow for both, fall back to list-only.
    let split = if inner.width >= 60 {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(38), Constraint::Min(20)])
            .split(inner)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(0)])
            .split(inner)
    };

    draw_theme_list(f, split[0], sel, current_name, active_theme);
    if split.len() == 2 {
        let names = theme::theme_names();
        if let Some(name) = names.get(sel) {
            let preview_theme = Theme::from_name(name);
            draw_theme_preview(f, split[1], &preview_theme, active_theme);
        }
    }
}

fn draw_theme_list(
    f: &mut Frame,
    inner: Rect,
    sel: usize,
    current_name: &str,
    active_theme: &Theme,
) {
    let names = theme::theme_names();
    let visible = inner.height as usize;
    let start = if sel >= visible {
        sel.saturating_sub(visible - 1)
    } else {
        0
    };
    let end = (start + visible).min(names.len());

    let mut lines: Vec<Line> = Vec::with_capacity(visible);
    for (i, name) in names[start..end].iter().enumerate() {
        let idx = start + i;
        let theme = Theme::from_name(name);
        let marker = if name == &current_name { "●" } else { " " };
        let selected = idx == sel;

        // Five-color swatch: bg / accent / ok / warn / err.
        let swatch = vec![
            Span::styled(" ██ ", Style::default().fg(theme.bg)),
            Span::styled("██ ", Style::default().fg(theme.accent)),
            Span::styled("██ ", Style::default().fg(theme.ok)),
            Span::styled("██ ", Style::default().fg(theme.warn)),
            Span::styled("██ ", Style::default().fg(theme.err)),
        ];

        let label_style = if selected {
            Style::default()
                .bg(active_theme.surface_2)
                .fg(active_theme.fg)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(active_theme.fg)
        };
        let marker_style = Style::default()
            .fg(active_theme.accent)
            .add_modifier(Modifier::BOLD);

        let mut spans: Vec<Span> = Vec::with_capacity(8);
        spans.push(Span::styled(format!(" {marker} "), marker_style));
        spans.extend(swatch);
        spans.push(Span::styled(format!(" {name}"), label_style));
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// Live preview of a candidate theme. Renders a compact composition that
/// exercises the colors most-affected by a theme switch: bg/fg contrast,
/// accent, the ok/warn/err triple, and the gradient ramp.
///
/// `preview_theme` is the theme being previewed (the one the cursor is on).
/// `active_theme` is the currently-applied theme — used only for the inner
/// title border / label color so the preview frame stays consistent with
/// the surrounding modal even when the previewed bg is light/dark inverse.
pub fn draw_theme_preview(f: &mut Frame, area: Rect, preview_theme: &Theme, active_theme: &Theme) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" preview ")
        .border_style(Style::default().fg(active_theme.muted))
        .title_style(
            Style::default()
                .fg(active_theme.muted)
                .add_modifier(Modifier::BOLD),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    // Paint the preview canvas with the candidate theme's bg so contrast
    // against the rest of the modal is visible at a glance.
    f.render_widget(Clear, inner);
    let bg_fill = Paragraph::new("").style(Style::default().bg(preview_theme.bg));
    f.render_widget(bg_fill, inner);

    // Stacked rows: header, gauge, sparkline, badges.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);

    // Row 0: mock header line.
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "rocm.ai",
            Style::default()
                .fg(preview_theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("   → connected", Style::default().fg(preview_theme.muted)),
    ]))
    .style(Style::default().bg(preview_theme.bg));
    f.render_widget(header, rows[0]);

    // Row 1: gradient memory gauge at 73%.
    let label = "73.0%";
    let gauge = GradientGauge::new(0.73)
        .stops(preview_theme.ok, preview_theme.warn, preview_theme.err)
        .track_bg(preview_theme.surface_2)
        .label(label)
        .label_fg(preview_theme.fg);
    f.render_widget(gauge, rows[1]);

    // Row 2: gradient sparkline over a deterministic sine-ish series so the
    // preview is visually rich and stable.
    let data: Vec<u64> = (0..rows[2].width as usize)
        .map(|i| {
            let t = i as f64 / f64::from(rows[2].width.max(1));
            // Two-bump curve so the gradient sweeps through all three stops.
            let v = (t * std::f64::consts::PI * 2.0)
                .sin()
                .mul_add(40.0, 60.0)
                .max(2.0);
            v as u64
        })
        .collect();
    let spark = BrailleSparkline::new(&data)
        .max(100)
        .style(Style::default().fg(preview_theme.accent))
        .gradient(preview_theme.ok, preview_theme.warn, preview_theme.err);
    f.render_widget(spark, rows[2]);

    // Row 3 (flexible): three status badges + a footer-style accent_2 span.
    let badges = Paragraph::new(vec![
        Line::from(vec![
            badge(" OK ", preview_theme.ok, preview_theme),
            Span::raw(" "),
            badge(" WARN ", preview_theme.warn, preview_theme),
            Span::raw(" "),
            badge(" ERR ", preview_theme.err, preview_theme),
        ]),
        Line::from(Span::styled(
            "  info",
            Style::default().fg(preview_theme.accent_2),
        )),
        Line::from(Span::styled(
            "  muted text reads here",
            Style::default().fg(preview_theme.muted),
        )),
    ])
    .style(Style::default().bg(preview_theme.bg));
    f.render_widget(badges, rows[3]);

    // Bottom row: theme name in the previewed fg so you see fg/bg contrast.
    let footer = Paragraph::new(Line::from(Span::styled(
        " preview rendered with the highlighted theme ",
        Style::default()
            .fg(preview_theme.fg)
            .bg(preview_theme.surface_2),
    )));
    f.render_widget(footer, rows[4]);
}

fn badge<'a>(label: &'a str, bg: ratatui::style::Color, preview_theme: &Theme) -> Span<'a> {
    Span::styled(
        label,
        Style::default()
            .bg(bg)
            .fg(preview_theme.bg)
            .add_modifier(Modifier::BOLD),
    )
}

// ===========================================================================
// btop-style chrome helpers ported from `examples/gen_mockups.rs` (Phase 1).
// Pure draw fns, no wiring this phase — the Esc menu / Options overlays compose
// them in Phase 4. Hit-testing / state are added with the live wiring.
// ===========================================================================

/// Stamp a string at `(x, y)` if the row is on-screen. Mirror of the
/// `gen_mockups` low-level `put`.
fn put(f: &mut Frame, x: u16, y: u16, s: &str, style: Style) {
    if y < f.area().height {
        f.buffer_mut().set_string(x, y, s, style);
    }
}

/// Dim the entire frame to a cool grey wash so a centered modal reads as the
/// foreground (port of `gen_mockups.rs` `grey_overlay`). Call before drawing
/// the modal box on top.
pub fn grey_overlay(f: &mut Frame) {
    use ratatui::style::Color;
    let area = f.area();
    let wash = Color::Rgb(0x1c, 0x1e, 0x22);
    let dim = Color::Rgb(0x4c, 0x50, 0x57);
    let buf = f.buffer_mut();
    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            if let Some(c) = buf.cell_mut((x, y)) {
                c.set_style(Style::default().fg(dim).bg(wash));
            }
        }
    }
}

/// Big block "ROCm" wordmark with a horizontal accent→cyan gradient sweep.
///
/// Port of `gen_mockups.rs` `draw_logo`. `cx` is the left column of the
/// 31-wide logo; it occupies 5 rows from `y`. The lowercase `m` matches the
/// final mock. Reuses the crate's [`gradient::lerp3_t`] ramp rather than a
/// local lerp.
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
/// control hint right-aligned (port of `gen_mockups.rs` `opt_row`).
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
    use crate::ui::theme::Theme;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    fn flat(term: &Terminal<TestBackend>) -> String {
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
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
