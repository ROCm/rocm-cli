// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Bench Observe sub-panel — full-screen bench browser with Pass^N / Pass@N rollups + sparkline.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::ui::panel::{self, BoxRole};

use rocm_dash_core::bench_rollup::{PassNRollup, rollup_pass_n, row_verdict};
use rocm_dash_core::bench_schema::{BenchmarkRow, PassFail};

use crate::app::AppState;
use crate::ui::format;
use crate::ui::sparkline::BrailleSparkline;
use crate::ui::theme::Theme;

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    if state.bench_rows.is_empty() {
        let inner = panel::bento(f, area, Some("Bench"), BoxRole::Neutral, false, theme);
        let p = Paragraph::new(Line::from(Span::styled(
            "no rows · run `rocm bench load --endpoint <url>` to populate the daemon-tailed bench results file · press b to run a sweep",
            Style::default().fg(theme.muted),
        )));
        f.render_widget(p, inner);
        return;
    }

    let rollup_rows = rollup_pass_n(state.bench_rows.iter());
    let rollup_height = compute_rollup_height(rollup_rows.len());

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(rollup_height),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(area);

    draw_rollup(f, chunks[0], &rollup_rows, theme);
    draw_rows_table(f, chunks[1], state, theme);
    draw_sparkline(f, chunks[2], state, theme);
}

// ---------- rollup ----------

fn compute_rollup_height(n_groups: usize) -> u16 {
    // 2 for borders + 1 for header + up to 8 data rows
    let data_rows = n_groups.min(8) as u16;
    (2 + 1 + data_rows).max(4)
}

/// Compact `tp·dtype` config token, e.g. `4·fp8`. `-` for missing parts.
fn cfg_token(r: &PassNRollup) -> String {
    let tp = r.tp.map_or_else(|| "-".into(), |v| v.to_string());
    let dtype = r.dtype.as_deref().unwrap_or("-");
    format!("{tp}·{dtype}")
}

/// `✓`/`✗` verdict span, green (`ok`) when `pass`, red (`err`) otherwise.
fn verdict_mark(pass: bool, theme: &Theme) -> Span<'static> {
    let (mark, color) = if pass {
        ("✓", theme.ok)
    } else {
        ("✗", theme.err)
    };
    Span::styled(
        mark,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

/// Pass@N mark. Pass@N is only a *distinct* signal when N > 1; for a
/// single-trial group it is identical to Pass^N, so render a muted dash
/// instead of a redundant second tick/cross.
fn at_n_mark(r: &PassNRollup, theme: &Theme) -> Span<'static> {
    if r.n_trials <= 1 {
        Span::styled("—", Style::default().fg(theme.muted))
    } else {
        verdict_mark(r.pass_at_n, theme)
    }
}

fn draw_rollup(f: &mut Frame, area: Rect, rows: &[PassNRollup], theme: &Theme) {
    let title = format!("Rollup · {} groups", rows.len());
    let inner = panel::bento(f, area, Some(&title), BoxRole::Secondary, false, theme);
    if inner.height == 0 {
        return;
    }

    let header = Line::from(vec![Span::styled(
        format!(
            "{:<12} {:<16} {:<10} {:>3} {:>6} {:>6} {:>14} {:>14}",
            "cell", "model", "cfg", "N", "Pass^N", "Pass@N", "meanPTPS", "meanGTPS"
        ),
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    )]);

    let max_rows = (inner.height as usize).saturating_sub(1);
    let shown = rows.iter().take(max_rows.min(8));
    let mut lines: Vec<Line> = Vec::with_capacity(max_rows + 1);
    lines.push(header);

    for r in shown {
        let cell = trunc_str(&r.cell, 12);
        let model = trunc_str(r.model.as_deref().unwrap_or("?"), 16);
        let cfg = trunc_str(&cfg_token(r), 10);
        lines.push(Line::from(vec![
            Span::styled(format!("{cell:<12} "), Style::default().fg(theme.fg)),
            Span::styled(format!("{model:<16} "), Style::default().fg(theme.accent)),
            Span::styled(format!("{cfg:<10} "), Style::default().fg(theme.muted)),
            Span::styled(
                format!("{:>3} ", r.n_trials),
                Style::default().fg(theme.muted),
            ),
            verdict_mark(r.pass_n_of_n, theme),
            Span::styled("     ", Style::default()),
            at_n_mark(r, theme),
            Span::styled("     ", Style::default()),
            Span::styled(
                format!("{:>14} ", format::tps_opt(r.mean_prompt_tps)),
                Style::default().fg(theme.fg),
            ),
            Span::styled(
                format!("{:>14}", format::tps_opt(r.mean_gen_tps)),
                Style::default().fg(theme.fg),
            ),
        ]));
    }

    f.render_widget(Paragraph::new(lines), inner);
}

// ---------- wide rows table ----------

const fn verdict_label(r: &BenchmarkRow) -> &'static str {
    match row_verdict(r) {
        PassFail::Pass => "Pass",
        PassFail::Fail => "Fail",
        PassFail::Unknown => "Unknown",
    }
}

const fn verdict_color(r: &BenchmarkRow, theme: &Theme) -> Color {
    match row_verdict(r) {
        PassFail::Pass => theme.ok,
        PassFail::Fail => theme.err,
        PassFail::Unknown => theme.muted,
    }
}

fn trunc_str(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

/// Compute the visible window `[start, end)` over `total` rows so that
/// `sel` is visible, biasing toward keeping the newest (highest index) rows
/// in view. `visible_rows` is the number of data rows that fit.
///
/// Returns `(start, end)` with `end - start <= visible_rows` and
/// `start <= sel < end` whenever `total > 0` and `sel < total`.
fn visible_window(total: usize, visible_rows: usize, sel: usize) -> (usize, usize) {
    if total == 0 || visible_rows == 0 {
        return (0, 0);
    }
    let cap = visible_rows.min(total);
    // Default window: anchor to the newest rows (tail).
    let mut start = total - cap;
    let mut end = total;
    if sel < start {
        // Scroll up: put sel at the top of the window.
        start = sel;
        end = (start + cap).min(total);
    } else if sel >= end {
        // Scroll down: put sel at the bottom of the window.
        end = (sel + 1).min(total);
        start = end - cap;
    }
    (start, end)
}

fn draw_rows_table(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let total = state.bench_rows.len();
    let inner_height_estimate = area.height.saturating_sub(2);
    let avail_estimate = (inner_height_estimate as usize).saturating_sub(1);
    let sel_display = if total == 0 {
        0
    } else {
        state.bench_sel.min(total - 1) + 1
    };
    let title = format!(
        "Bench rows · {total} total · row {sel_display}/{total} · showing {}",
        avail_estimate.min(total)
    );
    let inner = panel::bento(f, area, Some(&title), BoxRole::Primary, false, theme);
    if inner.height == 0 {
        return;
    }

    let header = Line::from(vec![Span::styled(
        format!(
            "{:<10} {:>4} {:<20} {:>3} {:>6} {:>10} {:>13} {:>13} {:>5} {:<8}",
            "cell", "run", "model", "tp", "dtype", "wall", "pTPS", "gTPS", "m_run", "verdict",
        ),
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    )]);

    let avail = (inner.height as usize).saturating_sub(1);
    let sel = state.bench_sel.min(total.saturating_sub(1));
    let (start, end) = visible_window(total, avail, sel);

    let mut lines: Vec<Line> = Vec::with_capacity(end - start + 1);
    lines.push(header);

    for (idx, r) in state
        .bench_rows
        .iter()
        .enumerate()
        .skip(start)
        .take(end - start)
    {
        let cell = trunc_str(&r.cell, 10);
        let model = trunc_str(r.model.as_deref().unwrap_or("?"), 20);
        let tp = r.tp.map_or_else(|| "-".into(), |v| v.to_string());
        let dtype = trunc_str(r.dtype.as_deref().unwrap_or("-"), 6);
        let wall = match r.wall_s {
            Some(v) => format::duration(v),
            None => "-".into(),
        };
        let ptps = format::tps_opt(r.prompt_tps);
        let gtps = format::tps_opt(r.gen_tps);
        let mrun = r
            .max_running_reqs
            .map_or_else(|| "-".into(), |v| v.to_string());
        let v_text = verdict_label(r);
        let v_color = verdict_color(r, theme);

        let is_sel = idx == sel;
        let row_bg = if is_sel { Some(theme.surface_2) } else { None };
        let apply_bg = |s: Style| match row_bg {
            Some(bg) => s.bg(bg).add_modifier(Modifier::BOLD),
            None => s,
        };

        lines.push(Line::from(vec![
            Span::styled(
                format!("{cell:<10} "),
                apply_bg(Style::default().fg(theme.accent)),
            ),
            Span::styled(
                format!("{:>4} ", r.run),
                apply_bg(Style::default().fg(theme.muted)),
            ),
            Span::styled(
                format!("{model:<20} "),
                apply_bg(Style::default().fg(theme.fg)),
            ),
            Span::styled(
                format!("{tp:>3} "),
                apply_bg(Style::default().fg(theme.muted)),
            ),
            Span::styled(
                format!("{dtype:>6} "),
                apply_bg(Style::default().fg(theme.muted)),
            ),
            Span::styled(
                format!("{wall:>10} "),
                apply_bg(Style::default().fg(theme.fg)),
            ),
            Span::styled(
                format!("{ptps:>13} "),
                apply_bg(Style::default().fg(theme.fg)),
            ),
            Span::styled(
                format!("{gtps:>13} "),
                apply_bg(Style::default().fg(theme.fg)),
            ),
            Span::styled(
                format!("{mrun:>5} "),
                apply_bg(Style::default().fg(theme.muted)),
            ),
            Span::styled(
                format!("{v_text:<8}"),
                apply_bg(Style::default().fg(v_color).add_modifier(Modifier::BOLD)),
            ),
        ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

// ---------- sparkline ----------

fn sparkline_max(data: &[u64]) -> u64 {
    let peak = data.iter().copied().max().unwrap_or(0);
    if peak == 0 {
        100
    } else {
        // round up to nearest 500
        let rounded = ((peak / 500) + 1) * 500;
        rounded.max(peak + peak / 10)
    }
}

fn draw_sparkline(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let n = state.bench_rows.len();
    let title = format!("prompt_tps · last {n} rows");
    let inner = panel::bento(f, area, Some(&title), BoxRole::Success, false, theme);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let data: Vec<u64> = state
        .bench_rows
        .iter()
        .map(|r| r.prompt_tps.unwrap_or(0.0).max(0.0) as u64)
        .collect();
    let max = sparkline_max(&data);
    // Higher prompt_tps is better, so use a "cool" gradient that ramps from
    // muted accent up through bright accent into ok-green for peak values —
    // visually rewards high throughput rather than flagging it.
    let spark = BrailleSparkline::new(&data)
        .max(max)
        .style(Style::default().fg(theme.accent))
        .gradient(theme.accent_2, theme.accent, theme.ok);
    f.render_widget(spark, inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(cell: &str, tp: Option<u32>, dtype: Option<&str>) -> PassNRollup {
        PassNRollup {
            cell: cell.to_string(),
            model: None,
            engine: None,
            tp,
            dtype: dtype.map(std::string::ToString::to_string),
            concurrency: None,
            n_trials: 0,
            n_passed: 0,
            pass_n_of_n: false,
            pass_at_n: false,
            mean_prompt_tps: None,
            mean_gen_tps: None,
        }
    }

    #[test]
    fn cfg_token_renders_tp_and_dtype() {
        assert_eq!(cfg_token(&group("A", Some(4), Some("fp8"))), "4·fp8");
        assert_eq!(cfg_token(&group("A", None, Some("fp16"))), "-·fp16");
        assert_eq!(cfg_token(&group("A", Some(8), None)), "8·-");
        assert_eq!(cfg_token(&group("A", None, None)), "-·-");
    }

    #[test]
    fn verdict_mark_colors_pass_and_fail() {
        let theme = Theme::default_dark();
        let ok = verdict_mark(true, &theme);
        assert_eq!(ok.content, "✓");
        assert_eq!(ok.style.fg, Some(theme.ok));
        let err = verdict_mark(false, &theme);
        assert_eq!(err.content, "✗");
        assert_eq!(err.style.fg, Some(theme.err));
    }

    #[test]
    fn sparkline_max_handles_empty_and_zero() {
        assert_eq!(sparkline_max(&[]), 100);
        assert_eq!(sparkline_max(&[0, 0]), 100);
        let m = sparkline_max(&[100, 250]);
        assert!(m >= 500);
        let m2 = sparkline_max(&[600]);
        assert!(m2 >= 1000);
    }

    #[test]
    fn visible_window_anchors_to_tail_when_sel_in_tail() {
        // 50 rows, 10 visible, selecting newest -> window [40, 50).
        assert_eq!(visible_window(50, 10, 49), (40, 50));
        // Selecting somewhere inside the default tail window stays anchored.
        assert_eq!(visible_window(50, 10, 45), (40, 50));
    }

    #[test]
    fn visible_window_scrolls_up_when_sel_above_tail() {
        // sel=5 is well above the default [40, 50) tail; window should shift.
        let (start, end) = visible_window(50, 10, 5);
        assert_eq!(start, 5);
        assert_eq!(end, 15);
        assert!(start <= 5 && 5 < end);
    }

    #[test]
    fn visible_window_handles_empty_and_zero_height() {
        assert_eq!(visible_window(0, 10, 0), (0, 0));
        assert_eq!(visible_window(10, 0, 5), (0, 0));
    }

    #[test]
    fn visible_window_keeps_sel_visible_when_total_smaller_than_capacity() {
        // total < visible_rows: show everything.
        assert_eq!(visible_window(3, 10, 0), (0, 3));
        assert_eq!(visible_window(3, 10, 2), (0, 3));
    }

    #[test]
    fn visible_window_scrolls_down_when_sel_below_default_window() {
        // total=20, visible=5: default window is [15, 20). sel=18 stays inside.
        assert_eq!(visible_window(20, 5, 18), (15, 20));
        // Smaller window: visible=3, default [17,20). sel=10 needs scroll.
        let (s, e) = visible_window(20, 3, 10);
        assert!(s <= 10 && 10 < e);
        assert_eq!(e - s, 3);
    }
}
