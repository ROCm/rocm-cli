// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Per-CPU-core vertical braille bars.
//!
//! Each core renders as a thermometer fill from the bottom. The cores tile the
//! width of the area rather than taking one column each, so the strip spans its
//! box instead of ending in a short stub on the left. The strip has three
//! shapes, by how much width each core can have:
//!
//! - **Two or more columns each**: every bar is the same width and separated
//!   from its neighbour, with the left-over columns widening the *gaps* rather
//!   than the bars. Two cores under equal load must draw the same size, and
//!   when bars are separated a one-column difference is plain to see.
//! - **About one column each**: bars tile proportionally and touch, so the
//!   strip reads as a continuous profile across the cores. Widths differ by up
//!   to a column here, which is not a signal when nothing is separated, and
//!   gutters are dropped entirely rather than applied to some bars and not
//!   others.
//! - **More cores than columns**: each column covers a group of cores and
//!   draws the busiest of them. Every core reaches a column, so a single
//!   pegged core on a many-core host cannot be hidden by its quiet neighbours.
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

impl CoreBars<'_> {
    /// Paint one bar of `value` across columns `[start, end)` of `area`.
    fn draw_bar(&self, buf: &mut Buffer, area: Rect, start: usize, end: usize, value: f32) {
        let rows = area.height as usize;
        let total_dot_rows = rows * 4;
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

impl Widget for CoreBars<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 || self.values.is_empty() {
            return;
        }
        let width = area.width as usize;
        let n = self.values.len();

        if n > width {
            // One column per group of cores, showing the busiest of the group.
            // Giving the column to a single representative core instead would
            // let a pegged core vanish behind idle neighbours.
            for cx in 0..width {
                let (lo, hi) = core_group(cx, n, width);
                let busiest = self.values[lo..hi]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                self.draw_bar(buf, area, cx, cx + 1, busiest);
            }
            return;
        }
        for (core, &value) in self.values.iter().enumerate() {
            let (start, end) = slot(core, n, width);
            self.draw_bar(buf, area, start, end, value);
        }
    }
}

/// Cores `[lo, hi)` sharing column `cx` when cores outnumber columns.
///
/// The groups tile `0..n`, so every core lands in exactly one column.
const fn core_group(cx: usize, n: usize, width: usize) -> (usize, usize) {
    let lo = cx * n / width;
    let hi = (cx + 1) * n / width;
    (lo, if hi > lo { hi } else { lo + 1 })
}

/// Columns `[start, end)` that core `core` of `n` draws into, across `width`.
///
/// Requires `width >= n >= 1`; `render` routes the other case to `core_group`.
///
/// With two or more columns per core every bar is `width / n - 1` wide and the
/// `width % n` left-over columns go into the gaps, so bars stay uniform and
/// only the spacing varies. With about one column per core the bars tile
/// proportionally and touch: a gutter there would have to come out of some
/// bars and not others, and a 1-wide bar flush against its neighbour reads as
/// one wider bar, which misstates the core count.
const fn slot(core: usize, n: usize, width: usize) -> (usize, usize) {
    debug_assert!(n > 0, "slot() divides by the core count");
    let slot_w = width / n;
    if slot_w < 2 {
        return (core * width / n, (core + 1) * width / n);
    }
    // Proportional start, uniform width. Starting each bar at its exact share
    // of the width spreads the spare columns evenly down the strip; handing
    // them to the first `width % n` cores instead would bunch every wide gap
    // into the left half and leave the right half visibly denser.
    let bar_w = slot_w - 1;
    let start = core * width / n;
    (start, start + bar_w)
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
    fn separated_bars_are_all_the_same_width() {
        // 32 cores across 170 columns: 5 columns each with 10 left over. The
        // spare columns must widen the gaps, not ten of the bars — two cores
        // under equal load have to draw the same size.
        let (n, width) = (32, 170);
        let slots: Vec<(usize, usize)> = (0..n).map(|c| slot(c, n, width)).collect();
        let widths: Vec<usize> = slots.iter().map(|&(s, e)| e - s).collect();
        assert!(
            widths.iter().all(|&w| w == widths[0]),
            "bars differ in width: {widths:?}"
        );
        assert_eq!(slots[0].0, 0, "strip does not start at the left edge");
        // The last bar ends within one slot of the right edge: its own gutter,
        // plus whatever the proportional start rounded away.
        assert!(
            slots[n - 1].1 + width / n >= width,
            "strip stops short of the right edge: ends at {}",
            slots[n - 1].1
        );
        for (core, pair) in slots.windows(2).enumerate() {
            let ((_, end), (next_start, _)) = (pair[0], pair[1]);
            assert!(next_start > end, "core {core} has no gutter");
        }
        // Gaps stay within a column of each other, so the strip has an even
        // rhythm rather than a sparse half and a dense half.
        let gaps: Vec<usize> = slots.windows(2).map(|p| p[1].0 - p[0].1).collect();
        let (lo, hi) = (gaps.iter().min().unwrap(), gaps.iter().max().unwrap());
        assert!(hi - lo <= 1, "gaps vary too much: {gaps:?}");
    }

    /// How many neighbouring core pairs have no blank column between them.
    fn touching_pairs(n: usize, width: usize) -> usize {
        (0..n - 1)
            .filter(|&c| slot(c + 1, n, width).0 == slot(c, n, width).1)
            .count()
    }

    #[test]
    fn gutters_are_all_or_nothing_across_the_strip() {
        // The strip is one of two shapes, never a mixture.
        //
        // At 2+ columns per core every bar is separated. At fewer, slots
        // alternate 1 and 2 columns wide, and taking a gutter only from the
        // 2-wide ones would leave *some* pairs touching, so 32 cores would
        // read as 16 wider bars. All touching is a contiguous histogram and
        // is honest about the core count; a mixture is not.
        assert_eq!(touching_pairs(32, 170), 0, "wide strip should be separated");
        assert_eq!(
            touching_pairs(32, 48),
            31,
            "narrow strip should be contiguous, not partly gapped"
        );
    }

    #[test]
    fn every_core_keeps_a_bar_in_the_contiguous_regime() {
        let (n, width) = (32, 48);
        let widths: Vec<usize> = (0..n)
            .map(|c| {
                let (s, e) = slot(c, n, width);
                e - s
            })
            .collect();
        assert!(widths.iter().all(|&w| w >= 1), "a core lost its bar");
    }

    #[test]
    fn the_baseline_bar_takes_the_gradient_start_color() {
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        // Same contract as the sparkline: an idle core's baseline is part of
        // the chart, so it carries the gradient's low color. A `v > 0.0`
        // guard on the gradient would pass without this.
        let values = [0.0_f32];
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
        let (low, mid, high) = (
            Color::Rgb(10, 0, 0),
            Color::Rgb(0, 20, 0),
            Color::Rgb(0, 0, 30),
        );
        CoreBars::new(&values)
            .max(100.0)
            .style(Style::default().fg(Color::Rgb(99, 99, 99)))
            .gradient(low, mid, high)
            .render(buf.area, &mut buf);
        assert_eq!(buf.cell((0, 0)).unwrap().style().fg, Some(low));
        assert_eq!(buf.cell((0, 0)).unwrap().symbol(), "\u{28c0}");
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
    fn core_groups_cover_every_core_exactly_once() {
        // 8 cores into 4 columns. The groups must tile 0..n: a core that falls
        // in no group is a core the user cannot see.
        let (n, width) = (8, 4);
        let mut next = 0;
        for cx in 0..width {
            let (lo, hi) = core_group(cx, n, width);
            assert_eq!(lo, next, "column {cx} skips a core");
            assert!(hi > lo, "column {cx} covers no core");
            next = hi;
        }
        assert_eq!(next, n, "the last cores reach no column");
    }

    #[test]
    fn a_pegged_core_is_visible_even_when_cores_outnumber_columns() {
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        // Core 0 at full load with seven idle neighbours, in four columns.
        // Showing one representative core per column used to drop core 0
        // entirely, hiding the only core doing work.
        let mut values = [0.0_f32; 8];
        values[0] = 100.0;
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 1));
        CoreBars::new(&values).max(100.0).render(buf.area, &mut buf);
        assert_eq!(
            buf.cell((0, 0)).unwrap().symbol(),
            "\u{28ff}",
            "the pegged core did not reach the strip"
        );
        // Its idle neighbours still read as a baseline, not a full bar.
        assert_eq!(buf.cell((1, 0)).unwrap().symbol(), "\u{28c0}");
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
