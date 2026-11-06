// Copyright © Advanced Micro Devices, Inc., or its affiliates.
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
//!
//! Popup geometry and framing live here too, beside [`popup`]: [`centered_rect`]
//! (with [`centered_width`]), [`draw_popup_frame`] and [`grey_overlay`], shared by
//! `ui/approval.rs`, `ui/job_console.rs` and the other overlays, and by the
//! Modal renderers (`ui/help_modal.rs`, `ui/warnings_modal.rs`,
//! `ui/theme_picker.rs` and `ui/menu.rs`).

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Padding, Scrollbar, ScrollbarOrientation, ScrollbarState,
};

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

/// Whether [`stamp_title`] has room to draw `title` on a box `area_width`
/// columns wide.
///
/// The top row spends one column on each rounded corner, one on the dash before
/// the label, and one on each label bracket, so the label itself needs
/// `area_width - 4` columns. A title that does not fit is dropped rather than
/// overflowing the border.
///
/// Public so a caller choosing between a long and a short title — the help
/// modals, which append a truncation marker only when they have to — can *ask*
/// instead of re-deriving this arithmetic and drifting from it. Without it the
/// longer form would silently erase the title it was meant to annotate.
#[must_use]
pub fn title_fits(area_width: u16, title: &str) -> bool {
    let label_w = title.trim().chars().count();
    label_w > 0 && label_w.saturating_add(4) < usize::from(area_width)
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
    if !title_fits(area.width, title) {
        return; // empty, or not enough room — leave the plain rounded top edge
    }
    let label_w = title.chars().count() as u16;
    // corner(x0) + at least one dash, then the left bracket.
    let lb = area.x + 2;
    let rb = lb + 1 + label_w; // right bracket column
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

/// Draw a vertical scrollbar on the right edge of `area` when content overflows.
///
/// Returns the content rect: `area` shrunk one column on the right when a bar is
/// drawn, or `area` unchanged when everything fits. Pure draw — the caller owns
/// `position` (first visible line) and clamps it. `content_len` is the total
/// scrollable units, `viewport_len` the visible units.
#[must_use]
pub fn vertical_scrollbar(
    f: &mut Frame,
    area: Rect,
    content_len: usize,
    viewport_len: usize,
    position: usize,
    theme: &Theme,
) -> Rect {
    vertical_scrollbar_impl(f, area, content_len, viewport_len, position, theme, false)
}

/// Like [`vertical_scrollbar`], but always reserves and draws the column even
/// when `content_len` alone would not warrant one.
///
/// For a pair of panes that share one scroll position (e.g. two side-by-side
/// panels scrolled in lockstep), gating each pane's column on its *own*
/// `content_len` lets one reserve a column while its sibling doesn't, so the
/// two end up different widths purely because one has slightly less content —
/// a layout wobble with no functional meaning. Callers that need the pair to
/// stay the same width decide reservation once (e.g. "either pane
/// overflows") and pass that decision in here for both, rather than letting
/// each pane re-derive its own answer.
#[must_use]
pub fn vertical_scrollbar_forced(
    f: &mut Frame,
    area: Rect,
    content_len: usize,
    viewport_len: usize,
    position: usize,
    theme: &Theme,
) -> Rect {
    vertical_scrollbar_impl(f, area, content_len, viewport_len, position, theme, true)
}

fn vertical_scrollbar_impl(
    f: &mut Frame,
    area: Rect,
    content_len: usize,
    viewport_len: usize,
    position: usize,
    theme: &Theme,
    force: bool,
) -> Rect {
    if area.width < 2 || area.height == 0 || (!force && content_len <= viewport_len) {
        return area;
    }
    // A forced bar can be asked to represent content that doesn't actually
    // overflow (see `vertical_scrollbar_forced`). Normalize to a single unit
    // of content/viewport in that case: ratatui's thumb-size formula treats
    // `content_len` and `viewport_len` as real proportions, so feeding it
    // `content_len <= viewport_len` as-is renders a partial thumb sized by
    // that (meaningless) ratio rather than the "fully visible, nothing to
    // scroll" full-track thumb this case should show.
    let (content_len, viewport_len, position) = if content_len <= viewport_len {
        (1, 1, 0)
    } else {
        (content_len, viewport_len, position)
    };
    let max_position = content_len.saturating_sub(viewport_len);
    let rendered_position =
        position.min(max_position) * content_len.saturating_sub(1) / max_position.max(1);
    let mut sb = ScrollbarState::new(content_len)
        .viewport_content_length(viewport_len)
        .position(rendered_position);
    let bar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(None)
        .end_symbol(None)
        .thumb_style(Style::default().fg(theme.accent))
        .track_style(Style::default().fg(theme.border));
    f.render_stateful_widget(bar, area, &mut sb);
    Rect {
        width: area.width - 1,
        ..area
    }
}

/// Draw a horizontal scrollbar on the bottom edge of `area` when content is
/// wider than the viewport.
///
/// Returns the content rect: `area` shrunk one row at the bottom when a bar is
/// drawn, or `area` unchanged when everything fits. `content_len` is the widest
/// content column count, `viewport_len` the visible column count.
#[must_use]
pub fn horizontal_scrollbar(
    f: &mut Frame,
    area: Rect,
    content_len: usize,
    viewport_len: usize,
    position: usize,
    theme: &Theme,
) -> Rect {
    if content_len <= viewport_len || area.height < 2 || area.width == 0 {
        return area;
    }
    let max_position = content_len.saturating_sub(viewport_len);
    let rendered_position =
        position.min(max_position) * content_len.saturating_sub(1) / max_position.max(1);
    let mut sb = ScrollbarState::new(content_len)
        .viewport_content_length(viewport_len)
        .position(rendered_position);
    let bar = Scrollbar::new(ScrollbarOrientation::HorizontalBottom)
        .begin_symbol(None)
        .end_symbol(None)
        .thumb_style(Style::default().fg(theme.accent))
        .track_style(Style::default().fg(theme.border));
    f.render_stateful_widget(bar, area, &mut sb);
    Rect {
        height: area.height - 1,
        ..area
    }
}

/// `extent * pct / 100`, computed in `u32` so the intermediate product cannot
/// overflow a `u16`.
///
/// `area.height * pct_y` overflows above 655 rows once `pct_y` is 100 — which
/// [`draw_help`](crate::ui::help_modal::draw_help) now passes, to mean "the whole area when the content is taller
/// than it". That is a debug-build panic on a terminal tall enough (tmux panes
/// and some GUI terminals report large synthetic sizes), and a wrap-around in
/// release. Widening here removes the ceiling for every caller rather than
/// leaving each one to know where it is.
fn scale_pct(extent: u16, pct: u16) -> u16 {
    u16::try_from(u32::from(extent) * u32::from(pct) / 100).unwrap_or(u16::MAX)
}

/// Centered rectangle taking `pct_x`% width and `pct_y`% height of `area`,
/// clamped to a maximum so it doesn't drown the screen on big terminals.
pub fn centered_rect(pct_x: u16, pct_y: u16, max_w: u16, max_h: u16, area: Rect) -> Rect {
    let h_pct = centered_height(pct_y, max_h, area);
    let v_pad = (area.height.saturating_sub(h_pct)) / 2;
    let w_pct = centered_width(pct_x, max_w, area);
    let vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(v_pad),
            Constraint::Length(h_pct),
            Constraint::Min(0),
        ])
        .split(area);

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

/// Narrowest a popup is allowed to be shrunk to by the percentage, so a modal
/// on a merely small terminal is still wide enough to read.
const MIN_POPUP_WIDTH: u16 = 20;

/// Width [`centered_rect`] will give a popup, split out so a caller that needs
/// to lay its content out *before* the popup exists (to size the popup to that
/// content) cannot drift from the real geometry.
///
/// The [`MIN_POPUP_WIDTH`] floor is itself clamped to `area.width`, which is the
/// difference between a number the popup might get and the number it *will*
/// get. An unclamped floor returns 20 on a terminal narrower than 20 — a width
/// the popup can never have, because [`centered_rect`]'s `Layout::split`
/// truncates the segment to the area. That is harmless for a caller that only
/// renders, but both help modals now *measure* their wrapped content against
/// this width before the popup exists: measuring the wrap at 20 columns and
/// rendering it into 12 under-counts the rows the content needs and cuts the
/// tail off — the precise silent truncation content-sizing was added to remove.
pub fn centered_width(pct_x: u16, max_w: u16, area: Rect) -> u16 {
    let floor = MIN_POPUP_WIDTH.min(area.width);
    scale_pct(area.width, pct_x).min(max_w).max(floor)
}

/// Shortest a popup is allowed to be shrunk to by the percentage: a border pair
/// plus enough body rows to be worth opening.
const MIN_POPUP_HEIGHT: u16 = 5;

/// Height [`centered_rect`] will give a popup — the exact counterpart of
/// [`centered_width`], and split out for the same reason: the number this
/// computes has to be the number the popup gets.
///
/// The [`MIN_POPUP_HEIGHT`] floor is clamped to `area.height` for the reason
/// spelled out on [`centered_width`]. An unclamped floor asks for 5 rows on an
/// area shorter than 5, which the `Layout::split` below then truncates, so the
/// requested geometry and the rendered geometry disagree. Nothing measures
/// against the height *today* — the callers measure their content and pass the
/// answer in as `max_h` — so this is a latent form of the defect that had gone
/// live on the width. Keeping the two sides identical is what stops it going
/// live here the first time a caller needs the height before the popup exists.
fn centered_height(pct_y: u16, max_h: u16, area: Rect) -> u16 {
    let floor = MIN_POPUP_HEIGHT.min(area.height);
    scale_pct(area.height, pct_y).min(max_h).max(floor)
}

/// Render a bordered block with `title` over `area` after clearing it,
/// returning the inner area so the caller can render content into it.
pub fn draw_popup_frame(f: &mut Frame, area: Rect, title: &str, theme: &Theme) -> Rect {
    f.render_widget(Clear, area);
    // Compact frame: rounded + notch title but the classic 1-cell inset, so the
    // many manager overlays keep their original content geometry.
    popup(f, area, title, theme)
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

    fn draw_scrollbar<F: FnOnce(&mut Frame) -> Rect>(w: u16, h: u16, f: F) -> Rect {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).unwrap();
        let mut got = Rect::new(0, 0, 0, 0);
        term.draw(|frame| got = f(frame)).unwrap();
        got
    }

    #[test]
    fn vertical_scrollbar_noops_when_content_fits() {
        let theme = Theme::default_dark();
        let area = Rect::new(0, 0, 20, 10);
        // 8 lines into a 10-row viewport: everything fits, no bar, rect intact.
        let got = draw_scrollbar(20, 10, |f| vertical_scrollbar(f, area, 8, 10, 0, &theme));
        assert_eq!(got, area, "no scrollbar → content rect unchanged");
    }

    #[test]
    fn vertical_scrollbar_shrinks_when_overflowing() {
        let theme = Theme::default_dark();
        let area = Rect::new(0, 0, 20, 10);
        // 50 lines into a 10-row viewport → bar drawn, rect loses a column.
        let got = draw_scrollbar(20, 10, |f| vertical_scrollbar(f, area, 50, 10, 0, &theme));
        assert_eq!(
            got.width, 19,
            "content rect shrinks by the scrollbar column"
        );
        assert_eq!(got.height, area.height, "height unchanged for vertical bar");
    }

    #[test]
    fn vertical_scrollbar_forced_reserves_column_even_when_content_fits() {
        let theme = Theme::default_dark();
        let area = Rect::new(0, 0, 20, 10);
        // Plain `vertical_scrollbar` would no-op here (8 <= 10) — `_forced`
        // must reserve the column anyway, for a pane whose sibling overflows
        // and needs both panes to share one width.
        let got = draw_scrollbar(20, 10, |f| {
            vertical_scrollbar_forced(f, area, 8, 10, 0, &theme)
        });
        assert_eq!(
            got.width, 19,
            "forced bar reserves the column regardless of content_len"
        );
    }

    #[test]
    fn vertical_scrollbar_forced_fills_the_track_when_content_fits() {
        let theme = Theme::default_dark();
        let backend = TestBackend::new(2, 4);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            let _ = vertical_scrollbar_forced(f, f.area(), 2, 4, 0, &theme);
        })
        .unwrap();
        let buf = term.backend().buffer();
        // content_len (2) < viewport_len (4): the thumb must fill the whole
        // track ("fully visible, nothing to scroll") rather than computing a
        // bogus partial thumb from a max_position that would underflow to 0
        // by coincidence rather than by the fits-entirely case being handled.
        let cells: Vec<&str> = (0..4).map(|y| buf.cell((1, y)).unwrap().symbol()).collect();
        assert_eq!(cells, ["█", "█", "█", "█"]);
    }

    #[test]
    fn horizontal_scrollbar_shrinks_height_when_overflowing() {
        let theme = Theme::default_dark();
        let area = Rect::new(0, 0, 20, 10);
        let got = draw_scrollbar(20, 10, |f| {
            horizontal_scrollbar(f, area, 200, 20, 0, &theme)
        });
        assert_eq!(got.height, 9, "content rect shrinks by the scrollbar row");
        assert_eq!(got.width, area.width, "width unchanged for horizontal bar");
    }

    #[test]
    fn vertical_scrollbar_renders_expected_thumb_cells() {
        let theme = Theme::default_dark();
        let backend = TestBackend::new(2, 4);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            let _ = vertical_scrollbar(f, f.area(), 4, 2, 1, &theme);
        })
        .unwrap();
        let buf = term.backend().buffer();
        let cells: Vec<&str> = (0..4).map(|y| buf.cell((1, y)).unwrap().symbol()).collect();
        assert_eq!(cells, ["║", "█", "█", "║"]);
    }

    #[test]
    fn horizontal_scrollbar_renders_expected_thumb_cells() {
        let theme = Theme::default_dark();
        let backend = TestBackend::new(4, 2);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            let _ = horizontal_scrollbar(f, f.area(), 4, 2, 1, &theme);
        })
        .unwrap();
        let buf = term.backend().buffer();
        let cells: Vec<&str> = (0..4).map(|x| buf.cell((x, 1)).unwrap().symbol()).collect();
        assert_eq!(cells, ["═", "█", "█", "═"]);
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

    /// The width the content-sized modals *measure* their wrapped content
    /// against must be the width the popup actually gets.
    ///
    /// `centered_width`'s minimum-width floor is what can break that. On a
    /// terminal narrower than the floor, an unclamped floor hands back a number
    /// `centered_rect` then truncates (`Layout::split` cannot emit a segment
    /// wider than its input), so `draw_help` / `draw_global_help` count the rows
    /// their content wraps to at one width and render it at a narrower one —
    /// under-counting the rows needed and cutting the tail off. That is the
    /// silent truncation content-sizing exists to remove, re-introduced at a
    /// geometry nothing else in this module exercises.
    ///
    /// Both live `(pct_x, max_w)` pairs are checked: `draw_help`'s 70/80 and
    /// `draw_global_help`'s 80/100.
    #[test]
    fn popup_width_is_never_wider_than_a_narrow_area() {
        for w in 0..=24u16 {
            let area = Rect::new(0, 0, w, 40);
            for (pct_x, max_w) in [(70u16, 80u16), (80, 100)] {
                let measured = super::centered_width(pct_x, max_w, area);
                assert!(
                    measured <= w,
                    "centered_width({pct_x}, {max_w}) returned {measured} on a \
                     {w}-column area: the modals would measure their content at \
                     a width the popup cannot have"
                );
                assert_eq!(
                    super::centered_rect(pct_x, 100, max_w, 10, area).width,
                    measured,
                    "the popup's rendered width must equal the width its \
                     content was measured against ({w}-column area)"
                );
            }
        }
    }

    /// The height counterpart of the sweep above, pinned for the same reason.
    ///
    /// Unlike the width, nothing measures its content against this number yet,
    /// so the defect is latent rather than live: `centered_rect` returns a rect
    /// `Layout::split` has already truncated to the area, and *that* is why this
    /// test asserts on `centered_height` rather than on `centered_rect(..).
    /// height`. A test written against the returned rect could not fail — it
    /// would be re-asserting ratatui's own invariant, which is precisely the
    /// mistake `help_modal_is_clamped_when_the_content_cannot_fit` made before
    /// it was deleted. The property this crate owns is that the height it *asks*
    /// for is the height it gets.
    ///
    /// Both live `(pct_y, max_h)` shapes are swept: the content-sized help
    /// modals' `pct_y = 100`, and a fixed-height overlay's 80/30.
    #[test]
    fn popup_height_is_never_taller_than_a_short_area() {
        for h in 0..=8u16 {
            let area = Rect::new(0, 0, 80, h);
            for (pct_y, max_h) in [(100u16, 12u16), (80, 30)] {
                let asked = super::centered_height(pct_y, max_h, area);
                assert!(
                    asked <= h,
                    "centered_height({pct_y}, {max_h}) returned {asked} on a \
                     {h}-row area: the popup would be laid out to a height it \
                     cannot have"
                );
                assert_eq!(
                    super::centered_rect(100, pct_y, 80, max_h, area).height,
                    asked,
                    "the popup's rendered height must equal the height it was \
                     laid out to ({h}-row area)"
                );
            }
        }
    }

    /// `centered_rect` scales a percentage of the area, and `area.height` is a
    /// `u16`. Since `draw_help` passes `pct_y = 100`, a `u16` product overflows
    /// above 655 rows — a debug-build panic on a tall terminal. The arithmetic
    /// widens to `u32` so no caller has to know where the ceiling is.
    #[test]
    fn centered_rect_does_not_overflow_on_a_very_tall_or_wide_area() {
        for (w, h) in [(2000u16, 2000u16), (u16::MAX, u16::MAX)] {
            let area = Rect::new(0, 0, w, h);
            let r = super::centered_rect(100, 100, u16::MAX, u16::MAX, area);
            assert!(
                r.height <= h && r.width <= w,
                "centered_rect({w}x{h}) escaped its area: {r:?}"
            );
            assert_eq!(super::centered_width(100, u16::MAX, area), w);
        }
    }
}
