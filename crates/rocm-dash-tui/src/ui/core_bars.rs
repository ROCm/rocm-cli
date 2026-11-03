// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Per-CPU-core vertical braille bars.
//!
//! Each core renders as a thermometer fill from the bottom. The cores tile the
//! width of the area rather than taking one column each: core `i` owns columns
//! `[i*w/n, (i+1)*w/n)`, so the strip spans its box instead of ending in a
//! short stub on the left. When every slot can spare a column, each gives up
//! its last one as a gutter, which leaves the box's rightmost column blank.
//!
//! The gutter is all-or-nothing across the strip. Slot widths differ by up to
//! one column, so taking a gutter only where a slot could spare one would let a
//! 1-wide bar sit flush against its neighbour and read as a single wider bar.
//!
//! With more cores than columns a slot comes out empty and that core is not
//! drawn, so a very wide machine in a narrow pane shows a sample spread across
//! the whole machine rather than the leftmost run of cores. The sample favours
//! odd indices: core 0's slot is the one that collapses first.
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
        let width = area.width as usize;
        let n = self.values.len();
        let rows = area.height as usize;
        let total_dot_rows = rows * 4;

        for (core, &value) in self.values.iter().enumerate() {
            let (start, end) = slot(core, n, width);
            if start >= end {
                continue; // more cores than columns: this one has no slot
            }
            let v = value.clamp(0.0, self.max);
            // Every core draws at least the bottom dot row: an idle core is a
            // baseline, not a gap, otherwise a few percent of `max` rounds to
            // zero dots and the bar strip breaks into fragments.
            let lit = ((v / self.max) * total_dot_rows as f32).round().max(1.0) as usize;
            let first_lit = total_dot_rows.saturating_sub(lit);
            let style = match self.gradient {
                Some(stops) => self.style.fg(lerp3_t(stops, f64::from(v / self.max))),
                None => self.style,
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
                for cx in start..end {
                    buf.set_string(area.x + cx as u16, area.y + cy as u16, &s, style);
                }
            }
        }
    }
}

/// Columns `[start, end)` that core `core` of `n` draws into, across `width`.
///
/// Slots tile the width exactly, with widths differing by at most one column.
/// `n` must be non-zero; `render` guarantees that by returning early on an
/// empty slice.
const fn slot(core: usize, n: usize, width: usize) -> (usize, usize) {
    debug_assert!(n > 0, "slot() divides by the core count");
    let start = core * width / n;
    let end = (core + 1) * width / n;
    // One gutter rule for the whole strip: take a column only when every slot
    // can spare one. Deciding per slot fuses a 1-wide bar onto its neighbour,
    // and two cores reading as one bar misstates how many the host has.
    let gutter = if width / n >= 2 { 1 } else { 0 };
    (start, end - gutter)
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

    #[test]
    fn slots_tile_the_full_width_with_gutters() {
        // 32 cores across 170 columns. Asserted on what `slot` returns, not on
        // the slot arithmetic recomputed here, so the test still fails if the
        // drawn range stops matching the tiling.
        let (n, width) = (32, 170);
        let slots: Vec<(usize, usize)> = (0..n).map(|c| slot(c, n, width)).collect();
        assert_eq!(slots[0].0, 0, "strip does not start at the left edge");
        assert_eq!(
            slots[n - 1].1,
            width - 1,
            "strip should reach the right edge bar its own gutter column"
        );
        for (core, pair) in slots.windows(2).enumerate() {
            let ((start, end), (next_start, _)) = (pair[0], pair[1]);
            assert!(end > start, "core {core} has no bar");
            assert_eq!(next_start, end + 1, "core {core} has no gutter");
        }
    }

    #[test]
    fn gutters_are_all_or_nothing_across_the_strip() {
        // 32 cores into 48 columns: slots alternate 1 and 2 wide. Taking a
        // gutter only from the 2-wide ones would leave pairs of bars touching
        // and reading as 16 wider bars.
        let (n, width) = (32, 48);
        let widths: Vec<usize> = (0..n)
            .map(|c| {
                let (s, e) = slot(c, n, width);
                e - s
            })
            .collect();
        assert!(widths.iter().all(|&w| w >= 1), "a core lost its bar");
        let touching = (0..n - 1)
            .filter(|&c| slot(c + 1, n, width).0 == slot(c, n, width).1)
            .count();
        assert_eq!(touching, n - 1, "gutters applied unevenly: {widths:?}");
    }

    #[test]
    fn narrow_area_degrades_to_one_column_per_core() {
        // One column per core leaves nothing to spare, so the gutter goes
        // rather than the bar.
        for core in 0..8 {
            assert_eq!(slot(core, 8, 8), (core, core + 1));
        }
    }

    #[test]
    fn more_cores_than_columns_samples_evenly() {
        // 8 cores into 4 columns: every other core gets a slot, so the strip
        // samples the whole machine instead of showing the leftmost four and
        // nothing from the rest of it.
        let drawn: Vec<usize> = (0..8)
            .filter(|&c| slot(c, 8, 4).0 < slot(c, 8, 4).1)
            .collect();
        assert_eq!(drawn, [1, 3, 5, 7]);
    }

    #[test]
    fn bars_fill_their_slot_and_idle_cores_draw_a_baseline() {
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        // Two cores across 10 columns: one busy, one idle at a fraction of a
        // dot row. Both must be visible, each 4 columns wide plus a gutter.
        let values = [90.0_f32, 0.4];
        let mut buf = Buffer::empty(Rect::new(0, 0, 10, 1));
        CoreBars::new(&values).max(100.0).render(buf.area, &mut buf);
        // An untouched cell keeps the buffer's blank, not a blank braille cell.
        let drawn: Vec<bool> = (0..10)
            .map(|x| buf.cell((x, 0)).unwrap().symbol() != " ")
            .collect();
        assert_eq!(
            drawn,
            [true, true, true, true, false, true, true, true, true, false]
        );
        // The busy core fills its slot; the idle one is a baseline, not a bar.
        assert_eq!(buf.cell((0, 0)).unwrap().symbol(), "\u{28ff}");
        assert_eq!(buf.cell((5, 0)).unwrap().symbol(), "\u{28c0}");
    }
}
