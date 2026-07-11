// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Bench Observe sub-panel — full-screen bench browser with Pass^N / Pass@N rollups + sparkline.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::ui::panel::{self, BoxRole};

use rocm_dash_core::bench_rollup::{PassNRollup, rollup_pass_n, row_verdict};
use rocm_dash_core::bench_schema::{BenchmarkRow, PassFail};

use crate::app::{AppState, KeyAction};
use crate::ui::format;
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::sparkline::BrailleSparkline;
use crate::ui::theme::Theme;

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    if state.bench_rows.is_empty() {
        let inner = panel::bento(f, area, Some("Bench"), BoxRole::Neutral, false, theme);
        let p = Paragraph::new(Line::from(Span::styled(
            "no rows · run `rocm bench load --endpoint <url>` to populate the daemon-tailed bench directory · press b to run a sweep",
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

/// Pure helper: given the rows-table's *inner* (post-border) area and the
/// currently visible window `[start, end)`, resolve a click at `(x, y)` to
/// a bench-row index, or `None` if the click misses a data line.
///
/// Row 0 of `rows_table_inner` is the header; rows 1..=visible are data
/// lines mapped to `[start, end)` in order.
const fn row_hit(
    rows_table_inner: Rect,
    start: usize,
    end: usize,
    x: u16,
    y: u16,
) -> Option<usize> {
    if rows_table_inner.width == 0 || rows_table_inner.height == 0 {
        return None;
    }
    if x < rows_table_inner.x || x >= rows_table_inner.x + rows_table_inner.width {
        return None;
    }
    if y < rows_table_inner.y || y >= rows_table_inner.y + rows_table_inner.height {
        return None;
    }
    let row_offset = y - rows_table_inner.y;
    if row_offset == 0 {
        // Header line.
        return None;
    }
    let visible = end.saturating_sub(start);
    let data_idx = (row_offset - 1) as usize;
    if data_idx >= visible {
        return None;
    }
    Some(start + data_idx)
}

/// Resolve a click at `(x, y)` inside the Bench Observe sub-panel body. Returns a
/// `KeyAction` to dispatch, or `None` when the click misses everything
/// actionable.
pub fn hit_test(area: Rect, x: u16, y: u16, state: &AppState) -> Option<KeyAction> {
    if state.bench_rows.is_empty() {
        return None;
    }
    // Recompute the same vertical layout as `draw`.
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
    let rows_outer = chunks[1];
    // Mirror panel::bento's inner rect: rounded full border + the same adaptive
    // padding it applies, so click mapping matches the drawn table exactly.
    let rows_inner = Block::default()
        .borders(Borders::ALL)
        .padding(panel::padding_for(rows_outer))
        .inner(rows_outer);

    let total = state.bench_rows.len();
    let avail = (rows_inner.height as usize).saturating_sub(1);
    if avail == 0 {
        return None;
    }
    let sel = state.bench_sel.min(total.saturating_sub(1));
    let (start, end) = visible_window(total, avail, sel);

    let target = row_hit(rows_inner, start, end, x, y)?;
    if target == state.bench_sel {
        Some(KeyAction::OpenDetail)
    } else {
        let delta = target.cast_signed() - state.bench_sel.cast_signed();
        Some(KeyAction::Move(delta))
    }
}

pub fn draw_detail(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let Some(row) = state.bench_rows.get(state.bench_sel) else {
        let popup = centered_rect(60, 30, 80, 10, area);
        let inner = draw_popup_frame(f, popup, "Bench row detail", theme);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "no selection",
                Style::default().fg(theme.muted),
            ))),
            inner,
        );
        return;
    };

    let title = format!("Row · {} run {}", row.cell, row.run);
    let popup = centered_rect(85, 85, 130, 36, area);
    let inner = draw_popup_frame(f, popup, &title, theme);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let lines = build_detail_lines(row, theme);
    let max_scroll = lines.len().saturating_sub(1) as u16;
    let scroll = state.bench_detail_scroll.min(max_scroll);

    // Reserve the last inner row for a footer hint; body gets the rest.
    let (body_area, footer_area) = if inner.height >= 2 {
        let body = Rect::new(inner.x, inner.y, inner.width, inner.height - 1);
        let footer = Rect::new(inner.x, inner.y + inner.height - 1, inner.width, 1);
        (body, Some(footer))
    } else {
        (inner, None)
    };

    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    f.render_widget(p, body_area);

    if let Some(footer) = footer_area {
        let hint = Paragraph::new(Line::from(Span::styled(
            "j/k or ↑/↓ scroll · PgUp/PgDn jump · Esc close",
            Style::default().fg(theme.muted),
        )));
        f.render_widget(hint, footer);
    }
}

// ---------- detail body ----------

#[allow(clippy::ref_option)]
fn fmt_opt<T: std::fmt::Display>(v: &Option<T>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "-".to_string(),
    }
}

/// SI-formatted optional `u32` counter (`-` when None).
fn fmt_opt_u32_si(v: Option<u32>) -> String {
    match v {
        Some(x) => format::si(f64::from(x)),
        None => "-".to_string(),
    }
}

/// SI-formatted optional `u64` counter (`-` when None).
fn fmt_opt_u64_si(v: Option<u64>) -> String {
    match v {
        Some(x) => format::si(x as f64),
        None => "-".to_string(),
    }
}

fn fmt_opt_f32_4(v: Option<f32>) -> String {
    match v {
        Some(x) => format!("{x:.4}"),
        None => "-".to_string(),
    }
}

const fn fmt_opt_bool(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "true",
        Some(false) => "false",
        None => "-",
    }
}

fn verdict_span(v: PassFail, theme: &Theme) -> Span<'static> {
    let (label, color) = match v {
        PassFail::Pass => ("Pass", theme.ok),
        PassFail::Fail => ("Fail", theme.err),
        PassFail::Unknown => ("Unknown", theme.muted),
    };
    Span::styled(
        label,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

fn section_header(title: &str, theme: &Theme) -> Line<'static> {
    Line::from(Span::styled(
        format!("— {title} —"),
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    ))
}

fn kv_line(key: &str, value: String, theme: &Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key:<22} "), Style::default().fg(theme.accent)),
        Span::styled(value, Style::default().fg(theme.fg)),
    ])
}

fn kv_span_line(key: &str, value: Span<'static>, theme: &Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key:<22} "), Style::default().fg(theme.accent)),
        value,
    ])
}

fn build_detail_lines(row: &BenchmarkRow, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::with_capacity(48);
    lines.push(section_header("identity", theme));
    lines.push(kv_line("cell", row.cell.clone(), theme));
    lines.push(kv_line("run", row.run.to_string(), theme));
    lines.push(kv_line("model", fmt_opt(&row.model), theme));
    lines.push(kv_line("endpoint", fmt_opt(&row.endpoint), theme));
    lines.push(kv_line("judge_model", fmt_opt(&row.judge_model), theme));
    lines.push(Line::raw(""));

    // config
    lines.push(section_header("config", theme));
    lines.push(kv_line("tp", fmt_opt(&row.tp), theme));
    lines.push(kv_line("pp", fmt_opt(&row.pp), theme));
    lines.push(kv_line("dtype", fmt_opt(&row.dtype), theme));
    lines.push(kv_line(
        "attention_backend",
        fmt_opt(&row.attention_backend),
        theme,
    ));
    lines.push(kv_line("max_num_seqs", fmt_opt(&row.max_num_seqs), theme));
    lines.push(kv_line("concurrency", fmt_opt(&row.concurrency), theme));
    lines.push(kv_line("extra_args", fmt_opt(&row.extra_args), theme));
    lines.push(Line::raw(""));

    // performance
    lines.push(section_header("performance", theme));
    lines.push(kv_line(
        "wall_s",
        row.wall_s.map_or_else(|| "-".into(), format::duration),
        theme,
    ));
    lines.push(kv_line("n_requests", fmt_opt_u32_si(row.n_requests), theme));
    lines.push(kv_line(
        "prompt_tokens",
        fmt_opt_u64_si(row.prompt_tokens),
        theme,
    ));
    lines.push(kv_line(
        "prompt_tps",
        format::tps_opt(row.prompt_tps),
        theme,
    ));
    lines.push(kv_line(
        "completion_tokens",
        fmt_opt_u64_si(row.completion_tokens),
        theme,
    ));
    lines.push(kv_line("gen_tps", format::tps_opt(row.gen_tps), theme));
    lines.push(kv_line(
        "max_running_reqs",
        fmt_opt_u32_si(row.max_running_reqs),
        theme,
    ));
    lines.push(kv_line(
        "max_waiting_reqs",
        fmt_opt_u32_si(row.max_waiting_reqs),
        theme,
    ));
    lines.push(kv_line("ttft_ms", fmt_opt(&row.ttft_ms), theme));
    lines.push(kv_line("tpot_ms", fmt_opt(&row.tpot_ms), theme));
    lines.push(kv_line("out_chars", fmt_opt_u64_si(row.out_chars), theme));
    lines.push(Line::raw(""));

    // verdict
    lines.push(section_header("verdict", theme));
    lines.push(kv_line("rc", fmt_opt(&row.rc), theme));
    lines.push(kv_span_line(
        "pass_fail",
        verdict_span(row.pass_fail, theme),
        theme,
    ));
    lines.push(kv_span_line(
        "judge_pass_fail",
        verdict_span(row.judge_pass_fail, theme),
        theme,
    ));
    lines.push(kv_line(
        "assertion_pass",
        fmt_opt_bool(row.assertion_pass).to_string(),
        theme,
    ));
    lines.push(kv_line(
        "assertion_fail_count",
        fmt_opt(&row.assertion_fail_count),
        theme,
    ));
    lines.push(kv_line(
        "assertion_summary",
        fmt_opt(&row.assertion_summary),
        theme,
    ));
    lines.push(kv_line(
        "quality_score",
        fmt_opt_f32_4(row.quality_score),
        theme,
    ));
    lines.push(kv_line(
        "safety_pass",
        fmt_opt_bool(row.safety_pass).to_string(),
        theme,
    ));
    lines.push(kv_line(
        "safety_violations",
        fmt_opt(&row.safety_violations),
        theme,
    ));

    lines
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

    #[test]
    fn row_hit_returns_none_for_header_or_out_of_bounds() {
        // 30 cols wide, 10 rows tall, anchored at (5, 2).
        let inner = Rect::new(5, 2, 30, 10);
        // Header row at y=2.
        assert_eq!(row_hit(inner, 0, 5, 10, 2), None);
        // Outside x range.
        assert_eq!(row_hit(inner, 0, 5, 4, 3), None);
        assert_eq!(row_hit(inner, 0, 5, 35, 3), None);
        // Outside y range.
        assert_eq!(row_hit(inner, 0, 5, 10, 1), None);
        assert_eq!(row_hit(inner, 0, 5, 10, 12), None);
    }

    #[test]
    fn row_hit_maps_data_lines_to_window_indices() {
        let inner = Rect::new(0, 0, 20, 10);
        // Window [10, 15): 5 data rows starting at y=1.
        assert_eq!(row_hit(inner, 10, 15, 5, 1), Some(10));
        assert_eq!(row_hit(inner, 10, 15, 5, 2), Some(11));
        assert_eq!(row_hit(inner, 10, 15, 5, 5), Some(14));
        // y=6 lands past the visible window (only 5 data rows shown).
        assert_eq!(row_hit(inner, 10, 15, 5, 6), None);
    }

    #[test]
    fn row_hit_handles_zero_dim_area() {
        let zero_w = Rect::new(0, 0, 0, 10);
        assert_eq!(row_hit(zero_w, 0, 5, 0, 1), None);
        let zero_h = Rect::new(0, 0, 10, 0);
        assert_eq!(row_hit(zero_h, 0, 5, 0, 0), None);
    }

    #[test]
    fn row_hit_handles_empty_window() {
        let inner = Rect::new(0, 0, 10, 5);
        // start == end → no data lines.
        assert_eq!(row_hit(inner, 3, 3, 5, 1), None);
    }
}
