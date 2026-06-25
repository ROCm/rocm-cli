// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Braille sparkline widget.
//!
//! One character cell = 2 cols × 4 rows of dots. We pack two consecutive
//! samples into each character (left half + right half), so an `N`-wide area
//! renders `2N` x-bins at `4 * height` vertical resolution.
//!
//! Inspired by btop's CPU graph and the matching widget in ctux.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

use crate::ui::gradient::lerp3_t;

pub struct BrailleSparkline<'a> {
    data: &'a [u64],
    max: u64,
    style: Style,
    gradient: Option<[Color; 3]>,
}

impl<'a> BrailleSparkline<'a> {
    pub fn new(data: &'a [u64]) -> Self {
        Self {
            data,
            max: 1,
            style: Style::default(),
            gradient: None,
        }
    }

    #[must_use]
    pub fn max(mut self, m: u64) -> Self {
        self.max = m.max(1);
        self
    }

    #[must_use]
    pub const fn style(mut self, s: Style) -> Self {
        self.style = s;
        self
    }

    /// Color each character cell by the larger of its two samples, mapped
    /// across the gradient (low value = stops[0], peak = stops[2]). When
    /// set, overrides `style.fg` per cell. Leaves `style.fg` as the fallback
    /// for empty/zero cells.
    #[must_use]
    pub const fn gradient(mut self, start: Color, mid: Color, end: Color) -> Self {
        self.gradient = Some([start, mid, end]);
        self
    }
}

impl Widget for BrailleSparkline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let cols = area.width as usize;
        let rows = area.height as usize;
        let total_dot_rows = rows * 4;

        // Right-align: show the most recent `cols * 2` samples.
        let bins = cols * 2;
        let start = self.data.len().saturating_sub(bins);
        let slice = &self.data[start..];

        for cx in 0..cols {
            let left = slice.get(cx * 2).copied();
            let right = slice.get(cx * 2 + 1).copied();
            // Color the whole character column by the larger of its two
            // samples — peaks render with the gradient end color even when
            // their neighbor is low.
            let cell_value = left.unwrap_or(0).max(right.unwrap_or(0));
            let style = match self.gradient {
                Some(stops) if cell_value > 0 => {
                    let t = cell_value as f64 / self.max as f64;
                    self.style.fg(lerp3_t(stops, t))
                }
                _ => self.style,
            };
            for cy in 0..rows {
                let row_top = cy * 4;
                let l = lit_dots(left, self.max, total_dot_rows, row_top);
                let r = lit_dots(right, self.max, total_dot_rows, row_top);
                if l == 0 && r == 0 {
                    continue;
                }
                let s = braille_char(l, r).to_string();
                buf.set_string(area.x + cx as u16, area.y + cy as u16, &s, style);
            }
        }
    }
}

/// Returns a 4-bit mask of which dots in this character row are lit
/// (bit 0 = top dot in the cell, bit 3 = bottom).
fn lit_dots(value: Option<u64>, max: u64, total_dot_rows: usize, row_top: usize) -> u8 {
    let Some(v) = value else {
        return 0;
    };
    let v = v.min(max);
    // Fill from the bottom. A value at `max` lights every dot row.
    let lit = ((v as f64 / max as f64) * total_dot_rows as f64).round() as usize;
    let first_lit_row = total_dot_rows.saturating_sub(lit);

    let mut mask = 0u8;
    for i in 0..4 {
        let dot_row = row_top + i;
        if dot_row >= first_lit_row && dot_row < total_dot_rows {
            mask |= 1 << i;
        }
    }
    mask
}

/// Compose left + right 4-bit dot columns into a Unicode braille code point.
///
/// Braille (U+2800..U+28FF) bit layout:
///
/// ```text
///   left col       right col
///     1 ●            4 ●        bits 0x01 / 0x08
///     2 ●            5 ●        bits 0x02 / 0x10
///     3 ●            6 ●        bits 0x04 / 0x20
///     7 ●            8 ●        bits 0x40 / 0x80
/// ```
const fn braille_char(left: u8, right: u8) -> char {
    let mut code = 0u32;
    if left & 0b0001 != 0 {
        code |= 0x01;
    }
    if left & 0b0010 != 0 {
        code |= 0x02;
    }
    if left & 0b0100 != 0 {
        code |= 0x04;
    }
    if left & 0b1000 != 0 {
        code |= 0x40;
    }
    if right & 0b0001 != 0 {
        code |= 0x08;
    }
    if right & 0b0010 != 0 {
        code |= 0x10;
    }
    if right & 0b0100 != 0 {
        code |= 0x20;
    }
    if right & 0b1000 != 0 {
        code |= 0x80;
    }
    char::from_u32(0x2800 + code).expect("valid braille code point")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_blank_braille() {
        assert_eq!(braille_char(0, 0), '\u{2800}');
    }

    #[test]
    fn all_dots_is_full_braille_block() {
        assert_eq!(braille_char(0b1111, 0b1111), '⣿');
    }

    #[test]
    fn bottom_only_lights_bottom_row_of_dots() {
        // dot 7 (left bottom) + dot 8 (right bottom) = 0x40 | 0x80
        let c = braille_char(0b1000, 0b1000);
        assert_eq!(c as u32 - 0x2800, 0xC0);
    }

    #[test]
    fn full_value_fills_all_rows() {
        // 1 char row tall = 4 dot rows; full value should light all 4 dots
        let m = lit_dots(Some(100), 100, 4, 0);
        assert_eq!(m, 0b1111);
    }

    #[test]
    fn zero_value_lights_nothing() {
        let m = lit_dots(Some(0), 100, 4, 0);
        assert_eq!(m, 0);
    }

    #[test]
    fn half_value_in_2_row_cell_fills_bottom_row_only() {
        // 2 rows = 8 dot rows; half value (50) → 4 dots from the bottom
        // top char row (row_top=0): nothing lit (top dots are above the fill)
        // bottom char row (row_top=4): all 4 dots lit
        assert_eq!(lit_dots(Some(50), 100, 8, 0), 0);
        assert_eq!(lit_dots(Some(50), 100, 8, 4), 0b1111);
    }

    #[test]
    fn gradient_colors_peak_cell_with_end_stop() {
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        let data = vec![100u64]; // single peak value at max
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
        let stops = (
            Color::Rgb(10, 0, 0),
            Color::Rgb(0, 20, 0),
            Color::Rgb(0, 0, 30),
        );
        BrailleSparkline::new(&data)
            .max(100)
            .gradient(stops.0, stops.1, stops.2)
            .render(buf.area, &mut buf);
        let fg = buf.cell((0, 0)).unwrap().style().fg;
        assert_eq!(fg, Some(stops.2));
    }

    #[test]
    fn gradient_disabled_uses_flat_style() {
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        let data = vec![100u64];
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
        let flat = Color::Rgb(123, 45, 67);
        BrailleSparkline::new(&data)
            .max(100)
            .style(Style::default().fg(flat))
            .render(buf.area, &mut buf);
        let fg = buf.cell((0, 0)).unwrap().style().fg;
        assert_eq!(fg, Some(flat));
    }
}
