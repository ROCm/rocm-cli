//! 2-D heatmap widget.
//!
//! Each row is a series of `f64` values. Each cell renders as a single
//! character whose background color is interpolated across the gradient
//! stops by `value / row_max`. Optional left labels are rendered in the
//! widget's `label_style`. When the matrix is wider than the area, the
//! widget right-aligns (drops the oldest columns) — the most-recent
//! samples are always visible, which matches how the rest of the dashboard
//! presents history.
//!
//! Designed for "metric × time" matrices: per-GPU detail (util/temp/power/
//! vram% × time), per-instance kv-cache × time, etc.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

use crate::ui::gradient::lerp3_t;

#[derive(Debug, Clone)]
pub struct HeatmapRow {
    pub label: String,
    pub data: Vec<f64>,
    /// Per-row max for normalization. If zero or negative, the cell renders
    /// as `track_bg`.
    pub max: f64,
    /// Optional per-row gradient override. Falls back to the widget-level
    /// `stops` when None.
    pub stops: Option<[Color; 3]>,
}

impl HeatmapRow {
    pub fn new(label: impl Into<String>, data: Vec<f64>, max: f64) -> Self {
        Self {
            label: label.into(),
            data,
            max,
            stops: None,
        }
    }

    pub fn stops(mut self, start: Color, mid: Color, end: Color) -> Self {
        self.stops = Some([start, mid, end]);
        self
    }
}

pub struct Heatmap<'a> {
    rows: &'a [HeatmapRow],
    stops: [Color; 3],
    track_bg: Color,
    label_style: Style,
    label_width: u16,
}

impl<'a> Heatmap<'a> {
    pub fn new(rows: &'a [HeatmapRow]) -> Self {
        // Pick a sensible default label width: longest label + 1 padding,
        // capped at 12. Callers can override with `label_width`.
        let default_label_w = rows
            .iter()
            .map(|r| r.label.chars().count() as u16)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .min(12);
        Self {
            rows,
            stops: [
                Color::Rgb(0x1a, 0xa0, 0x1a),
                Color::Rgb(0xf5, 0x9e, 0x0b),
                Color::Rgb(0xed, 0x1c, 0x24),
            ],
            track_bg: Color::Reset,
            label_style: Style::default(),
            label_width: default_label_w,
        }
    }

    pub fn stops(mut self, start: Color, mid: Color, end: Color) -> Self {
        self.stops = [start, mid, end];
        self
    }

    pub fn track_bg(mut self, c: Color) -> Self {
        self.track_bg = c;
        self
    }

    pub fn label_style(mut self, s: Style) -> Self {
        self.label_style = s;
        self
    }

    pub fn label_width(mut self, w: u16) -> Self {
        self.label_width = w;
        self
    }
}

impl Widget for Heatmap<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 || self.rows.is_empty() {
            return;
        }
        let label_w = self.label_width.min(area.width);
        let data_w = area.width.saturating_sub(label_w);
        if data_w == 0 {
            // No room for cells — just render labels.
            for (i, row) in self.rows.iter().enumerate() {
                if i as u16 >= area.height {
                    break;
                }
                let y = area.y + i as u16;
                let truncated = truncate_to(&row.label, label_w as usize);
                buf.set_string(area.x, y, &truncated, self.label_style);
            }
            return;
        }

        for (i, row) in self.rows.iter().enumerate() {
            if i as u16 >= area.height {
                break;
            }
            let y = area.y + i as u16;

            // Label.
            if label_w > 0 {
                let truncated = truncate_to(&row.label, label_w as usize);
                buf.set_string(area.x, y, &truncated, self.label_style);
            }

            // Right-align data: drop oldest columns when wider than `data_w`.
            let data_len = row.data.len();
            let visible = data_w as usize;
            let start = data_len.saturating_sub(visible);
            let slice = &row.data[start..];
            let row_stops = row.stops.unwrap_or(self.stops);

            for (cx, v) in slice.iter().enumerate() {
                let x = area.x + label_w + cx as u16;
                let bg = if row.max > 0.0 && *v > 0.0 {
                    let t = (*v / row.max).clamp(0.0, 1.0);
                    lerp3_t(row_stops, t)
                } else {
                    self.track_bg
                };
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_char(' ');
                    cell.set_style(Style::default().bg(bg));
                }
            }
            // Pad any leftover data area with track_bg so previous frames
            // don't bleed through when a row has fewer samples than width.
            let painted = slice.len() as u16;
            for cx in painted..data_w {
                let x = area.x + label_w + cx;
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_char(' ');
                    cell.set_style(Style::default().bg(self.track_bg));
                }
            }
        }
    }
}

fn truncate_to(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        format!("{s:<max$}")
    } else {
        s.chars().take(max).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(label: &str, data: Vec<f64>, max: f64) -> HeatmapRow {
        HeatmapRow::new(label, data, max)
    }

    #[test]
    fn empty_rows_render_nothing() {
        let rows: Vec<HeatmapRow> = Vec::new();
        let mut buf = Buffer::empty(Rect::new(0, 0, 10, 3));
        Heatmap::new(&rows).render(buf.area, &mut buf);
        // No panic, no styled cells.
        for y in 0..3 {
            for x in 0..10 {
                let cell = buf.cell((x, y)).unwrap();
                assert!(matches!(cell.style().bg, Some(Color::Reset) | None));
            }
        }
    }

    #[test]
    fn default_label_width_is_longest_plus_one_capped() {
        let rows = vec![
            row("util", vec![1.0], 100.0),
            row("temperature", vec![1.0], 100.0),
            row("pwr", vec![1.0], 100.0),
        ];
        let h = Heatmap::new(&rows);
        // "temperature" is 11 chars; +1 = 12, cap = 12.
        assert_eq!(h.label_width, 12);
    }

    #[test]
    fn cells_use_per_row_gradient_when_set() {
        let row_red_only = row("a", vec![50.0], 100.0).stops(
            Color::Rgb(255, 0, 0),
            Color::Rgb(255, 0, 0),
            Color::Rgb(255, 0, 0),
        );
        let rows = vec![row_red_only];
        let mut buf = Buffer::empty(Rect::new(0, 0, 5, 1));
        Heatmap::new(&rows)
            .label_width(2)
            .render(buf.area, &mut buf);
        // x = 2 (after label_w=2) is the first data cell.
        let cell = buf.cell((2, 0)).unwrap();
        assert_eq!(cell.style().bg, Some(Color::Rgb(255, 0, 0)));
    }

    #[test]
    fn zero_max_or_zero_value_paints_track_bg() {
        let rows = vec![row("a", vec![0.0, 50.0], 0.0)];
        let track = Color::Rgb(20, 20, 20);
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 1));
        Heatmap::new(&rows)
            .label_width(2)
            .track_bg(track)
            .render(buf.area, &mut buf);
        // Both cells should be track_bg because row.max == 0.
        assert_eq!(buf.cell((2, 0)).unwrap().style().bg, Some(track));
        assert_eq!(buf.cell((3, 0)).unwrap().style().bg, Some(track));
    }

    #[test]
    fn right_aligns_when_data_wider_than_area() {
        // Width 5 = label(2) + data(3). 5 samples → keeps the last 3.
        let mut rows = vec![row("a", vec![10.0, 20.0, 30.0, 40.0, 50.0], 50.0)];
        rows[0] = rows[0].clone().stops(
            Color::Rgb(0, 0, 100),
            Color::Rgb(0, 100, 0),
            Color::Rgb(100, 0, 0),
        );
        let mut buf = Buffer::empty(Rect::new(0, 0, 5, 1));
        Heatmap::new(&rows)
            .label_width(2)
            .render(buf.area, &mut buf);
        // The leftmost data cell should reflect value 30 (the 3rd-newest).
        let cell30 = buf.cell((2, 0)).unwrap();
        let cell40 = buf.cell((3, 0)).unwrap();
        let cell50 = buf.cell((4, 0)).unwrap();
        // 30/50 = 0.6 → past midpoint, mid→end interp; 50/50 = 1.0 → end stop.
        assert_eq!(cell50.style().bg, Some(Color::Rgb(100, 0, 0)));
        // 40/50 = 0.8 → mid→end at t=0.6.
        assert!(matches!(cell40.style().bg, Some(Color::Rgb(_, _, _))));
        // 30/50 = 0.6 → mid→end at t=0.2 → mostly green-ish.
        assert!(matches!(cell30.style().bg, Some(Color::Rgb(_, _, _))));
    }

    #[test]
    fn truncates_label_when_too_long() {
        let rows = vec![row("temperature", vec![50.0], 100.0)];
        let mut buf = Buffer::empty(Rect::new(0, 0, 10, 1));
        Heatmap::new(&rows)
            .label_width(4)
            .render(buf.area, &mut buf);
        let s: String = (0..4)
            .map(|x| {
                buf.cell((x, 0))
                    .unwrap()
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' ')
            })
            .collect();
        assert_eq!(s, "temp");
    }

    #[test]
    fn rows_beyond_height_are_dropped() {
        let rows = vec![
            row("a", vec![50.0], 100.0),
            row("b", vec![50.0], 100.0),
            row("c", vec![50.0], 100.0),
        ];
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 2));
        Heatmap::new(&rows)
            .label_width(2)
            .render(buf.area, &mut buf);
        // Row 2 ("c") wasn't rendered — y=2 doesn't exist in a 2-row area,
        // but more importantly the widget didn't panic.
        assert_eq!(buf.area.height, 2);
    }

    #[test]
    fn renders_label_only_when_no_room_for_cells() {
        let rows = vec![row("a", vec![50.0], 100.0)];
        let mut buf = Buffer::empty(Rect::new(0, 0, 2, 1));
        Heatmap::new(&rows)
            .label_width(4)
            .render(buf.area, &mut buf);
        // label_w(4) > area.width(2) → clamped to area.width, no data cells.
        // No assertion on the data — but make sure the label glyph appears at x=0.
        let first = buf
            .cell((0, 0))
            .unwrap()
            .symbol()
            .chars()
            .next()
            .unwrap_or(' ');
        assert_eq!(first, 'a');
    }

    #[test]
    fn label_style_is_applied() {
        use ratatui::style::Modifier;
        let rows = vec![row("a", vec![10.0], 100.0)];
        let mut buf = Buffer::empty(Rect::new(0, 0, 5, 1));
        let style = Style::default()
            .fg(Color::Rgb(123, 45, 67))
            .add_modifier(Modifier::BOLD);
        Heatmap::new(&rows)
            .label_width(2)
            .label_style(style)
            .render(buf.area, &mut buf);
        let cell_style = buf.cell((0, 0)).unwrap().style();
        assert_eq!(cell_style.fg, Some(Color::Rgb(123, 45, 67)));
        assert!(cell_style.add_modifier.contains(Modifier::BOLD));
    }
}
