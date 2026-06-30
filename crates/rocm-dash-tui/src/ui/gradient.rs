// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Three-stop horizontal gradient gauge (ok → warn → err by default).
//!
//! ratatui's built-in `Gauge` only supports a single color. This widget
//! renders the filled portion by setting each cell's background to a color
//! interpolated across the bar's width — so a 100% bar reads as a smooth
//! green-yellow-red sweep, and the fill *length* itself still tracks the
//! ratio. The unfilled portion keeps the panel's surface color.
//!
//! Inspired by btop's mem/cpu meters.

use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::Widget;

#[derive(Debug, Clone, Copy)]
pub struct GradientGauge<'a> {
    ratio: f64,
    stops: [Color; 3],
    track_bg: Color,
    label: Option<&'a str>,
    label_fg: Color,
}

impl<'a> GradientGauge<'a> {
    pub const fn new(ratio: f64) -> Self {
        Self {
            ratio: ratio.clamp(0.0, 1.0),
            stops: [
                Color::Rgb(0x1a, 0xa0, 0x1a),
                Color::Rgb(0xf5, 0x9e, 0x0b),
                Color::Rgb(0xed, 0x1c, 0x24),
            ],
            track_bg: Color::Reset,
            label: None,
            label_fg: Color::White,
        }
    }

    #[must_use]
    pub const fn stops(mut self, start: Color, mid: Color, end: Color) -> Self {
        self.stops = [start, mid, end];
        self
    }

    #[must_use]
    pub const fn track_bg(mut self, c: Color) -> Self {
        self.track_bg = c;
        self
    }

    #[must_use]
    pub const fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    #[must_use]
    pub const fn label_fg(mut self, c: Color) -> Self {
        self.label_fg = c;
        self
    }
}

impl Widget for GradientGauge<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let width = area.width as usize;
        // Filled cell count (floor — partial cells aren't subdividable in TTY).
        let filled = (self.ratio * width as f64).round() as usize;
        let filled = filled.min(width);

        for cy in 0..area.height {
            for cx in 0..width {
                let bg = if cx < filled {
                    lerp3(self.stops, cx, width.saturating_sub(1).max(1))
                } else {
                    self.track_bg
                };
                if let Some(cell) = buf.cell_mut((area.x + cx as u16, area.y + cy)) {
                    cell.set_char(' ');
                    cell.set_style(Style::default().bg(bg));
                }
            }
        }

        if let Some(label) = self.label {
            // Center label on the middle row. Read the per-cell bg under the
            // label so the text foreground stays legible against whatever stop
            // it lands on (bg from the gauge below, fg from `label_fg`).
            let mid_y = area.y + area.height / 2;
            let len = label.chars().count() as u16;
            if len <= area.width {
                let start_x = area.x + (area.width - len) / 2;
                buf.set_string(
                    start_x,
                    mid_y,
                    label,
                    Style::default()
                        .fg(self.label_fg)
                        .add_modifier(Modifier::BOLD),
                );
            }
        }
        let _ = Alignment::Center;
    }
}

/// 3-stop linear color interpolation by cell index across `last+1` cells.
/// Wraps `lerp3_t`; kept for callers that think in column indices.
fn lerp3(stops: [Color; 3], x: usize, last: usize) -> Color {
    if last == 0 {
        return stops[2];
    }
    lerp3_t(stops, x as f64 / last as f64)
}

/// 3-stop linear color interpolation by a unit-interval parameter `t`
/// (clamped to `[0, 1]`). The first half lerps stops[0]→stops[1]; the
/// second half lerps stops[1]→stops[2].
///
/// Exposed for widgets (sparklines, core bars) that need to color samples
/// by their *value*, not their position.
pub fn lerp3_t(stops: [Color; 3], t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    if t <= 0.5 {
        lerp2(stops[0], stops[1], t * 2.0)
    } else {
        lerp2(stops[1], stops[2], (t - 0.5) * 2.0)
    }
}

fn lerp2(a: Color, b: Color, t: f64) -> Color {
    let (ar, ag, ab) = rgb_of(a);
    let (br, bg, bb) = rgb_of(b);
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: u8, y: u8| -> u8 {
        let v = (f64::from(y) - f64::from(x)).mul_add(t, f64::from(x));
        v.round().clamp(0.0, 255.0) as u8
    };
    Color::Rgb(lerp(ar, br), lerp(ag, bg), lerp(ab, bb))
}

const fn rgb_of(c: Color) -> (u8, u8, u8) {
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        // Fallback: indexed / named colors aren't blendable without a
        // palette table; treat as mid-grey so the gauge still renders.
        _ => (128, 128, 128),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp2_endpoints_are_exact() {
        let a = Color::Rgb(0, 0, 0);
        let b = Color::Rgb(200, 100, 50);
        assert_eq!(lerp2(a, b, 0.0), a);
        assert_eq!(lerp2(a, b, 1.0), b);
    }

    #[test]
    fn lerp2_midpoint_is_arithmetic_mean() {
        let a = Color::Rgb(0, 0, 0);
        let b = Color::Rgb(100, 200, 50);
        let mid = lerp2(a, b, 0.5);
        // round(50), round(100), round(25)
        assert_eq!(mid, Color::Rgb(50, 100, 25));
    }

    #[test]
    fn lerp3_passes_through_each_stop() {
        let stops = [
            Color::Rgb(10, 0, 0),
            Color::Rgb(0, 20, 0),
            Color::Rgb(0, 0, 30),
        ];
        assert_eq!(lerp3(stops, 0, 10), stops[0]);
        assert_eq!(lerp3(stops, 5, 10), stops[1]);
        assert_eq!(lerp3(stops, 10, 10), stops[2]);
    }

    #[test]
    fn lerp3_handles_zero_width_gauge() {
        let stops = [
            Color::Rgb(10, 0, 0),
            Color::Rgb(0, 20, 0),
            Color::Rgb(0, 0, 30),
        ];
        assert_eq!(lerp3(stops, 0, 0), stops[2]);
    }

    #[test]
    fn render_fills_expected_cells_per_ratio() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 10, 1));
        let gauge = GradientGauge::new(0.5);
        gauge.render(buf.area, &mut buf);
        // 5 cells should have a non-Reset bg, 5 should be Reset (track).
        let mut filled = 0;
        for x in 0..10 {
            let cell = buf.cell((x, 0)).unwrap();
            if !matches!(cell.style().bg, Some(Color::Reset) | None) {
                filled += 1;
            }
        }
        assert_eq!(filled, 5);
    }

    #[test]
    fn render_zero_ratio_paints_no_fill() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 8, 1));
        GradientGauge::new(0.0).render(buf.area, &mut buf);
        for x in 0..8 {
            let cell = buf.cell((x, 0)).unwrap();
            assert!(matches!(cell.style().bg, Some(Color::Reset) | None));
        }
    }

    #[test]
    fn render_full_ratio_paints_every_cell() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 8, 1));
        GradientGauge::new(1.0).render(buf.area, &mut buf);
        for x in 0..8 {
            let cell = buf.cell((x, 0)).unwrap();
            assert!(!matches!(cell.style().bg, Some(Color::Reset) | None));
        }
    }

    #[test]
    fn label_renders_at_horizontal_center() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 10, 1));
        GradientGauge::new(0.5)
            .label("50%")
            .render(buf.area, &mut buf);
        // Label starts at (10 - 3) / 2 = 3
        let s: String = (3..6)
            .map(|x| {
                buf.cell((x, 0))
                    .unwrap()
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' ')
            })
            .collect();
        assert_eq!(s, "50%");
    }
}
