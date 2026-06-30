// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Shared "bento" box — the single source of box chrome for the dash.
//!
//! One helper so every visible box reads as part of one design system:
//! - rounded corners (`╭╮╰╯`);
//! - a btop-style inline title whose border dips **down** on each side of the
//!   label (a short descender), so the title reads as set into the frame
//!   (see `aristocratos/btop` box titles);
//! - a [`BoxRole`]-driven border color so color means something (telemetry vs
//!   action vs warning) and adjacent boxes differ;
//! - a faint per-role surface tint so boxes lift off the background with a hint
//!   of their role hue, anchored to the theme bg so it stays legible in light
//!   and dark themes;
//! - generous left/top padding (backed off on tiny boxes).
//!
//! The 4-tab folder panel ([`crate::ui::tabs::draw_tab_panel`]) is intentionally
//! NOT drawn through here — it keeps its centered-label folder chrome.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, BorderType, Borders, Padding};

use crate::ui::theme::Theme;

/// Semantic role of a box. Drives the border color + tint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxRole {
    /// Primary actionable surface (accent).
    Primary,
    /// Telemetry / observation (accent_2).
    Secondary,
    /// Healthy / positive (ok).
    Success,
    /// Caution (warn).
    Warning,
    /// Error / danger (err).
    Danger,
    /// Low-emphasis: logs, context, footnotes (muted).
    Muted,
    /// Generic chrome (border).
    Neutral,
}

impl BoxRole {
    /// Base border color for this role.
    #[must_use]
    pub const fn color(self, theme: &Theme) -> Color {
        match self {
            Self::Primary => theme.accent,
            Self::Secondary => theme.accent_2,
            Self::Success => theme.ok,
            Self::Warning => theme.warn,
            Self::Danger => theme.err,
            Self::Muted => theme.muted,
            Self::Neutral => theme.border,
        }
    }
}

/// Linear blend of two colors at `t` (0 = `a`, 1 = `b`).
///
/// Non-RGB colors fall back to `a` — we can't interpolate indexed/named colors
/// meaningfully, and the dash themes are all RGB so this is a safe degradation.
#[must_use]
pub fn blend(a: Color, b: Color, t: f32) -> Color {
    match (a, b) {
        (Color::Rgb(ar, ag, ab), Color::Rgb(br, bg, bb)) => {
            let t = t.clamp(0.0, 1.0);
            let mix = |x: u8, y: u8| {
                (f32::from(y) - f32::from(x))
                    .mul_add(t, f32::from(x))
                    .round() as u8
            };
            Color::Rgb(mix(ar, br), mix(ag, bg), mix(ab, bb))
        }
        _ => a,
    }
}

/// Border color for a box, brightened toward `fg` when focused.
#[must_use]
fn border_color(role: BoxRole, focused: bool, theme: &Theme) -> Color {
    let base = role.color(theme);
    if focused {
        blend(base, theme.fg, 0.30)
    } else {
        base
    }
}

/// Adaptive padding: heavier on the left and top (per the design), backing off
/// on short/narrow boxes so tiny panels still render their content.
pub(crate) fn padding_for(area: Rect) -> Padding {
    let left = if area.width >= 8 {
        2
    } else {
        u16::from(area.width >= 3)
    };
    let top = u16::from(area.height >= 5);
    Padding::new(left, 1, top, 0)
}

/// Stamp a string at `(x, y)` if on-screen. Local mirror of the chrome `put`.
fn put(f: &mut Frame, x: u16, y: u16, text: &str, style: Style) {
    let area = f.area();
    if x < area.x + area.width && y < area.y + area.height {
        f.buffer_mut().set_string(x, y, text, style);
    }
}

/// Overlay the btop-style title onto the already-drawn rounded top border.
///
/// Layout on the top row: `╭─╮ Title ╭───────╮` — the inner `╮`/`╭` are the label
/// brackets whose arcs turn the line downward, framing the title in a shallow
/// notch. The bracket glyphs take the role `border` color; the label text takes
/// `text_fg` (a legible foreground) so the title stays readable over the faint
/// tint on every theme.
fn stamp_title(f: &mut Frame, area: Rect, title: &str, border: Color, bg: Color, text_fg: Color) {
    let title = title.trim();
    if title.is_empty() {
        return;
    }
    let label_w = title.chars().count() as u16;
    // corner(x0) + at least one dash, then the left bracket.
    let lb = area.x + 2;
    let rb = lb + 1 + label_w; // right bracket column
    let x1 = area.x + area.width - 1;
    if rb >= x1 {
        return; // not enough room — leave the plain rounded top edge
    }
    let y0 = area.y;
    let bracket = Style::default().fg(border).bg(bg);
    let text = Style::default()
        .fg(text_fg)
        .bg(bg)
        .add_modifier(Modifier::BOLD);

    // Brackets hug the label directly (no inner spaces); their own downward arc
    // is the whole notch — a short half-cell turn, not a full descender.
    put(f, lb, y0, "╮", bracket);
    put(f, lb + 1, y0, title, text);
    put(f, rb, y0, "╭", bracket);
}

/// Core box renderer shared by [`bento`] and [`popup`].
///
/// `compact` selects the popup geometry: no extra padding (inner = classic
/// 1-cell inset) and no title descenders. Otherwise the full bento treatment
/// (adaptive left/top padding + downward-notch descenders) applies.
fn render_box(
    f: &mut Frame,
    area: Rect,
    title: Option<&str>,
    role: BoxRole,
    focused: bool,
    theme: &Theme,
    compact: bool,
) -> Rect {
    if area.width == 0 || area.height == 0 {
        return area;
    }
    let border = border_color(role, focused, theme);
    let border_style = if focused {
        Style::default().fg(border).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(border)
    };
    let padding = if compact {
        Padding::ZERO
    } else {
        padding_for(area)
    };

    // One background — the theme bg — for every box. Only borders and the
    // elements inside carry color; boxes are distinguished by border, not fill.
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style)
        .style(Style::default().bg(theme.bg))
        .padding(padding);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if let Some(t) = title {
        stamp_title(f, area, t, border, theme.bg, theme.fg);
    }
    inner
}

/// Rounded bento box with the btop downward-notch title.
///
/// Adds role-based border color, a faint tint, and generous left/top padding,
/// and returns the inner content rect. `focused` brightens + bolds the border.
#[must_use]
pub fn bento(
    f: &mut Frame,
    area: Rect,
    title: Option<&str>,
    role: BoxRole,
    focused: bool,
    theme: &Theme,
) -> Rect {
    render_box(f, area, title, role, focused, theme, false)
}

/// Compact rounded frame for centered popups / manager overlays.
///
/// Same rounded border + inline notch title + tint as [`bento`], but with the
/// classic 1-cell inset (so existing manager layouts are unchanged) and no
/// descenders. Always the Primary (accent) role.
#[must_use]
pub fn popup(f: &mut Frame, area: Rect, title: &str, theme: &Theme) -> Rect {
    render_box(f, area, Some(title), BoxRole::Primary, false, theme, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn render<F: Fn(&mut Frame)>(w: u16, h: u16, draw: F) -> String {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f)).unwrap();
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn bento_draws_rounded_corners_and_title() {
        let theme = Theme::default_dark();
        let out = render(40, 8, |f| {
            let _ = bento(
                f,
                f.area(),
                Some("Actions"),
                BoxRole::Primary,
                false,
                &theme,
            );
        });
        // Rounded corners present, sharp corners absent on the frame.
        assert!(out.contains('╭'), "missing rounded top-left: {out:?}");
        assert!(out.contains('╮'), "missing rounded top-right/bracket");
        assert!(
            out.contains('╰') && out.contains('╯'),
            "missing rounded bottoms"
        );
        assert!(!out.contains('┌'), "sharp corner leaked: {out:?}");
        assert!(out.contains("Actions"), "title text missing");
    }

    #[test]
    fn bento_title_notch_brackets_frame_the_label() {
        let theme = Theme::default_dark();
        let backend = TestBackend::new(40, 8);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            let _ = bento(f, f.area(), Some("Logs"), BoxRole::Muted, false, &theme);
        })
        .unwrap();
        let buf = term.backend().buffer();
        // The downward-turn notch lives on the top border row (the brackets'
        // own half-cell arc) — not as a full descender hanging into the box.
        let top: String = (0..40)
            .map(|x| buf.cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            top.contains("╮Logs╭"),
            "title notch missing on top row: {top:?}"
        );
        // Row 1 is interior content only — just the two side walls, no descender.
        let row1: Vec<&str> = (0..40)
            .map(|x| buf.cell((x, 1)).unwrap().symbol())
            .collect();
        let verticals = row1.iter().filter(|s| **s == "│").count();
        assert_eq!(
            verticals, 2,
            "expected only the side walls on row 1, got {row1:?}"
        );
    }

    #[test]
    fn bento_roles_pick_distinct_border_colors() {
        let theme = Theme::default_dark();
        assert_ne!(
            BoxRole::Primary.color(&theme),
            BoxRole::Secondary.color(&theme)
        );
        assert_ne!(BoxRole::Warning.color(&theme), BoxRole::Muted.color(&theme));
        assert_eq!(BoxRole::Primary.color(&theme), theme.accent);
    }

    #[test]
    fn focused_border_differs_from_unfocused() {
        let theme = Theme::default_dark();
        assert_ne!(
            border_color(BoxRole::Primary, true, &theme),
            border_color(BoxRole::Primary, false, &theme)
        );
    }

    #[test]
    fn bento_survives_tiny_areas() {
        let theme = Theme::default_dark();
        for (w, h) in [(1u16, 1u16), (2, 2), (4, 3), (6, 2), (40, 1)] {
            let _ = render(w.max(1), h.max(1), |f| {
                let _ = bento(f, f.area(), Some("x"), BoxRole::Neutral, false, &theme);
            });
        }
    }

    #[test]
    fn blend_endpoints_and_midpoint() {
        let a = Color::Rgb(0, 0, 0);
        let b = Color::Rgb(100, 200, 50);
        assert_eq!(blend(a, b, 0.0), a);
        assert_eq!(blend(a, b, 1.0), b);
        assert_eq!(blend(a, b, 0.5), Color::Rgb(50, 100, 25));
        // Non-RGB falls back to a.
        assert_eq!(blend(Color::Red, b, 0.5), Color::Red);
    }
}
