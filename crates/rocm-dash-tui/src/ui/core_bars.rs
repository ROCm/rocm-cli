//! Per-CPU-core vertical braille bars.
//!
//! Each core renders as one character column (2 dots wide) showing its current
//! utilization as a thermometer fill from the bottom. Designed for ≤ ~200 cores
//! visible in a terminal that is at least that wide. Cores that don't fit on
//! screen are dropped from the right edge.
//!
//! Inspired by btop's per-core mini bars.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

use crate::ui::gradient::lerp3_t;

pub struct CoreBars<'a> {
    values: &'a [f32],
    max: f32,
    style: Style,
    gradient: Option<[Color; 3]>,
}

impl<'a> CoreBars<'a> {
    pub fn new(values: &'a [f32]) -> Self {
        Self {
            values,
            max: 100.0,
            style: Style::default(),
            gradient: None,
        }
    }

    #[must_use]
    pub fn max(mut self, m: f32) -> Self {
        self.max = if m > 0.0 { m } else { 1.0 };
        self
    }

    #[must_use]
    pub const fn style(mut self, s: Style) -> Self {
        self.style = s;
        self
    }

    /// Color each bar by its fill ratio (low → stops[0], full → stops[2]).
    /// When set, overrides `style.fg` per column.
    #[must_use]
    pub const fn gradient(mut self, start: Color, mid: Color, end: Color) -> Self {
        self.gradient = Some([start, mid, end]);
        self
    }
}

impl Widget for CoreBars<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 || self.values.is_empty() {
            return;
        }
        let cols = (area.width as usize).min(self.values.len());
        let rows = area.height as usize;
        let total_dot_rows = rows * 4;

        for cx in 0..cols {
            let v = self.values[cx].clamp(0.0, self.max);
            let lit = ((v / self.max) * total_dot_rows as f32).round() as usize;
            let first_lit = total_dot_rows.saturating_sub(lit);
            let style = match self.gradient {
                Some(stops) if v > 0.0 => self.style.fg(lerp3_t(stops, f64::from(v / self.max))),
                _ => self.style,
            };
            for cy in 0..rows {
                let row_top = cy * 4;
                let mut mask = 0u8;
                for i in 0..4 {
                    let dot_row = row_top + i;
                    if dot_row >= first_lit && dot_row < total_dot_rows {
                        mask |= 1 << i;
                    }
                }
                if mask == 0 {
                    continue;
                }
                // Use both left and right dot columns for a fuller bar look.
                let s = braille_both_cols(mask).to_string();
                buf.set_string(area.x + cx as u16, area.y + cy as u16, &s, style);
            }
        }
    }
}

const fn braille_both_cols(mask: u8) -> char {
    let mut code = 0u32;
    if mask & 0b0001 != 0 {
        code |= 0x01 | 0x08;
    }
    if mask & 0b0010 != 0 {
        code |= 0x02 | 0x10;
    }
    if mask & 0b0100 != 0 {
        code |= 0x04 | 0x20;
    }
    if mask & 0b1000 != 0 {
        code |= 0x40 | 0x80;
    }
    char::from_u32(0x2800 + code).expect("valid braille code point")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_value_fills_block() {
        assert_eq!(braille_both_cols(0b1111), '⣿');
    }

    #[test]
    fn zero_is_blank() {
        assert_eq!(braille_both_cols(0), '\u{2800}');
    }

    #[test]
    fn bottom_only_fills_bottom_dot_row() {
        let c = braille_both_cols(0b1000);
        assert_eq!(c as u32 - 0x2800, 0x40 | 0x80);
    }
}
