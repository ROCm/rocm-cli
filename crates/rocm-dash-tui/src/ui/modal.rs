// Copyright © Advanced Micro Devices, Inc., or its affiliates.
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
use ratatui::widgets::{Clear, Paragraph, Wrap};

use crate::app::{ActiveTab, AppState};
use crate::ui::gradient::GradientGauge;
use crate::ui::panel::{self, BoxRole};
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
    // Compact frame: rounded + notch title but the classic 1-cell inset, so the
    // many manager overlays keep their original content geometry.
    panel::popup(f, area, title, theme)
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
        ActiveTab::Home => &[("(no tab-specific keys — see the ROCm / Serving tabs)", "")],
        ActiveTab::Rocm | ActiveTab::Serving => &[
            ("j / k  ↑ / ↓", "select an action"),
            ("→ / Enter", "open it in Details (asks before mutating)"),
            ("←", "Details preview → Actions list"),
            ("Esc", "close an open manager (back to Actions)"),
        ],
        ActiveTab::Observe => &[
            ("j / Down", "select next instance"),
            ("k / Up", "select previous instance"),
            ("g / Home", "first instance"),
            ("G / End", "last instance"),
            ("Enter", "open instance detail"),
            ("s", "services manager"),
            (
                "w / e / d / u / i / l",
                "serve / engines / doctor / update / install / logs",
            ),
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
    let inner = panel::bento(
        f,
        area,
        Some("preview"),
        BoxRole::Secondary,
        false,
        active_theme,
    );
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
// P4 overlays: Esc menu, command palette, tabbed Options, global Help. They
// compose the Phase-1 chrome helpers (grey_overlay / draw_logo / opt_row /
// draw_tab_panel) over the stable 4-tab body.
// ===========================================================================

/// Esc-menu rows: Options / Help / Quit.
pub const MENU_ITEMS: usize = 3;

/// Command-palette destinations (label, tab).
pub const PALETTE_DESTS: &[(&str, ActiveTab)] = &[
    ("Home", ActiveTab::Home),
    ("ROCm", ActiveTab::Rocm),
    ("Serving", ActiveTab::Serving),
    ("Observe", ActiveTab::Observe),
    ("Chat", ActiveTab::Chat),
];

/// Options panel tabs.
pub const OPTIONS_TABS: &[&str] = &["General", "CPU", "GPU", "Engines"];

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

/// Global 2-column keyboard reference (NAVIGATE / OVERLAYS / ACTIONS / CHAT /
/// GLOBAL). Distinct from the contextual per-tab `?` help (`draw_help`).
pub fn draw_global_help(f: &mut Frame, area: Rect, theme: &Theme) {
    grey_overlay(f);
    let modal = centered_rect(80, 80, 100, 26, area);
    let inner = draw_popup_frame(f, modal, "Keyboard", theme);
    if inner.height == 0 {
        return;
    }
    f.render_widget(Clear, inner);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(inner);
    let left: &[(&str, &[(&str, &str)])] = &[
        (
            "NAVIGATE",
            &[
                ("Tab / ⇧Tab", "next / prev tab"),
                ("1 .. 5", "jump to tab"),
                ("j / k", "select"),
            ],
        ),
        (
            "OVERLAYS",
            &[
                ("Esc", "main menu"),
                (":", "command palette"),
                ("? ", "this help"),
                ("t", "theme picker"),
            ],
        ),
    ];
    let right: &[(&str, &[(&str, &str)])] = &[
        (
            "ACTIONS",
            &[
                ("w / e / d", "serve / engines / doctor"),
                ("u / i / l", "update / install / logs"),
                ("Enter", "open / detail"),
            ],
        ),
        (
            "CHAT / GLOBAL",
            &[("i / Enter", "focus chat input"), ("q", "quit")],
        ),
    ];
    render_help_groups(f, cols[0], left, theme);
    render_help_groups(f, cols[1], right, theme);
}

fn render_help_groups(
    f: &mut Frame,
    area: Rect,
    groups: &[(&str, &[(&str, &str)])],
    theme: &Theme,
) {
    let mut lines: Vec<Line> = Vec::new();
    for (title, rows) in groups {
        lines.push(Line::from(Span::styled(
            *title,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        for (k, desc) in *rows {
            lines.push(key_line(k, desc, theme));
        }
        lines.push(Line::raw(""));
    }
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
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

/// Dim the entire frame to a cool grey wash so a centered modal reads as the
/// foreground. Call before drawing the modal box on top.
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

        let help = render(&|f| super::draw_global_help(f, area, &theme));
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
