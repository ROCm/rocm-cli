//! Overview tab — the original 60/40 split layout. Preserved verbatim so
//! switching tabs always lands back on the familiar view.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::app::{AppState, ConnState};
use crate::ui::core_bars::CoreBars;
use crate::ui::format;
use crate::ui::gradient::GradientGauge;
use crate::ui::sparkline::BrailleSparkline;
use crate::ui::theme::Theme;
use crate::ui::widgets::{gpu_stats_line, trunc};

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),
            Constraint::Length(4),
            Constraint::Min(0),
        ])
        .split(cols[0]);

    draw_cpu(f, left[0], state, theme);
    draw_memory(f, left[1], state, theme);
    draw_host(f, left[2], state, theme);

    let n_gpus = state
        .latest
        .as_ref()
        .map_or(1, |s| s.gpus.len().max(1));
    let gpu_height = (n_gpus as u16 * 4 + 2).clamp(8, 20);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(gpu_height),
            Constraint::Min(6),
            Constraint::Length(10),
        ])
        .split(cols[1]);

    draw_gpu(f, right[0], state, theme);
    draw_instances(f, right[1], state, theme);
    draw_bench(f, right[2], state, theme);
}

fn draw_cpu(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let n_cores = state
        .latest
        .as_ref()
        .map_or(0, |s| s.host.cpu_per_core_pct.len());
    let title = match state.latest.as_ref() {
        Some(s) => format!(
            " CPU · {} · {} cores ",
            format::pct(s.host.cpu_overall_pct),
            n_cores
        ),
        None => " CPU ".to_string(),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height == 0 {
        return;
    }

    let split = if n_cores == 0 || inner.height < 4 {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1)])
            .split(inner)
    } else {
        let agg = inner.height.saturating_sub(2).max(2);
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(agg), Constraint::Min(1)])
            .split(inner)
    };

    let data: Vec<u64> = state
        .history
        .iter()
        .map(|s| s.host.cpu_overall_pct.clamp(0.0, 100.0) as u64)
        .collect();
    let spark = BrailleSparkline::new(&data)
        .max(100)
        .style(Style::default().fg(theme.accent))
        .gradient(theme.ok, theme.warn, theme.err);
    f.render_widget(spark, split[0]);

    if split.len() == 2
        && let Some(s) = state.latest.as_ref()
    {
        let bars = CoreBars::new(&s.host.cpu_per_core_pct)
            .max(100.0)
            .style(Style::default().fg(theme.ok))
            .gradient(theme.ok, theme.warn, theme.err);
        f.render_widget(bars, split[1]);
    }
}

fn draw_memory(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let (used, total, ratio) = match state.latest.as_ref() {
        Some(s) => {
            let used = s.host.memory_used_mb;
            let total = s.host.memory_total_mb.max(1);
            (used, total, (used as f64 / total as f64).clamp(0.0, 1.0))
        }
        None => (0, 1, 0.0),
    };
    let title = format!(" Memory · {} ", format::mib_pair(used, total));
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let inner = block.inner(area);
    f.render_widget(block, area);

    let label = format::pct((ratio * 100.0) as f32);
    let gauge = GradientGauge::new(ratio)
        .stops(theme.ok, theme.warn, theme.err)
        .track_bg(theme.surface_2)
        .label(&label)
        .label_fg(theme.fg);
    f.render_widget(gauge, inner);
}

fn draw_host(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let lines: Vec<Line> = match state.latest.as_ref() {
        Some(s) => vec![
            Line::from(Span::styled(
                format!("swap_used_mb       {}", format::mib(s.host.swap_used_mb)),
                Style::default().fg(theme.fg),
            )),
            Line::from(Span::styled(
                format!("cpu_per_core_pct   [{}]", s.host.cpu_per_core_pct.len()),
                Style::default().fg(theme.muted),
            )),
            Line::from(Span::styled(
                format!("snapshots_in_buf   {}", state.history.len()),
                Style::default().fg(theme.muted),
            )),
            Line::from(Span::styled(
                format!("ts_latest          {}", s.timestamp.format("%H:%M:%S UTC")),
                Style::default().fg(theme.muted),
            )),
        ],
        None => vec![Line::from(Span::styled(
            "waiting for first snapshot…",
            Style::default().fg(theme.muted),
        ))],
    };
    let p = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Host ")
            .border_style(theme.border_style())
            .title_style(theme.title_style()),
    );
    f.render_widget(p, area);
}

fn draw_gpu(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let title = match state
        .latest
        .as_ref()
        .and_then(|s| s.gpu_system_info.as_ref())
    {
        Some(si) => format!(
            " GPU · {} × {} · ROCm {} ",
            si.physical_gpu_count,
            si.gpu_model,
            si.rocm_version.as_deref().unwrap_or("?")
        ),
        None => " GPU ".to_string(),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height == 0 {
        return;
    }

    let Some(snap) = state.latest.as_ref() else {
        let p = Paragraph::new(Line::from(Span::styled(
            "waiting for first snapshot…",
            Style::default().fg(theme.muted),
        )));
        f.render_widget(p, inner);
        return;
    };

    if snap.gpus.is_empty() {
        let lines: Vec<Line> = if snap.warnings.is_empty() {
            vec![Line::from(Span::styled(
                "no GPUs reported",
                Style::default().fg(theme.muted),
            ))]
        } else {
            snap.warnings
                .iter()
                .map(|w| Line::from(Span::styled(w.clone(), Style::default().fg(theme.warn))))
                .collect()
        };
        f.render_widget(Paragraph::new(lines), inner);
        return;
    }

    let n = snap.gpus.len() as u16;
    let per_gpu = (inner.height / n).max(1);
    let constraints: Vec<Constraint> = (0..n).map(|_| Constraint::Length(per_gpu)).collect();
    let slots = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    for (i, g) in snap.gpus.iter().enumerate() {
        let slot = slots[i];
        if slot.height == 0 {
            continue;
        }
        let stats_line = gpu_stats_line(g, theme);

        if slot.height < 2 {
            f.render_widget(Paragraph::new(stats_line), slot);
            continue;
        }

        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(slot);

        f.render_widget(Paragraph::new(stats_line), split[0]);

        let history: Vec<u64> = state
            .history
            .iter()
            .filter_map(|s| {
                s.gpus
                    .get(i)
                    .map(|g| g.gpu_utilization_pct.clamp(0.0, 100.0) as u64)
            })
            .collect();
        let spark = BrailleSparkline::new(&history)
            .max(100)
            .style(Style::default().fg(theme.accent))
            .gradient(theme.ok, theme.warn, theme.err);
        f.render_widget(spark, split[1]);
    }
}

fn draw_instances(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let count = state.instances.len();
    let title = format!(" Instances · {count} ");
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.instances.is_empty() {
        let body = match &state.conn {
            ConnState::Connected { .. } => {
                "no instances · start daemon with --enable-docker on a host with vLLM containers"
            }
            _ => "waiting for daemon…",
        };
        let p = Paragraph::new(Line::from(Span::styled(
            body,
            Style::default().fg(theme.muted),
        )));
        f.render_widget(p, inner);
        return;
    }

    let header = Line::from(vec![Span::styled(
        format!(
            "{:<18} {:<20} {:>5} {:>3} {:<10}",
            "name", "model", "port", "tp", "gpus"
        ),
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    )]);

    let mut instances: Vec<&rocm_dash_core::metrics::Instance> = state.instances.values().collect();
    instances.sort_by(|a, b| a.container_name.cmp(&b.container_name));

    let rows_shown = (inner.height as usize).saturating_sub(1);
    let mut lines: Vec<Line> = Vec::with_capacity(rows_shown + 1);
    lines.push(header);
    for inst in instances.into_iter().take(rows_shown) {
        let status_color = match inst.status {
            rocm_dash_core::metrics::InstanceStatus::Running => theme.ok,
            rocm_dash_core::metrics::InstanceStatus::Starting => theme.warn,
            rocm_dash_core::metrics::InstanceStatus::Stopped
            | rocm_dash_core::metrics::InstanceStatus::Error => theme.err,
            rocm_dash_core::metrics::InstanceStatus::Unknown => theme.muted,
        };
        let name = trunc(&inst.container_name, 18);
        let model = trunc(&inst.model_name, 20);
        let port = inst
            .port.map_or_else(|| "-".into(), |p| p.to_string());
        let gpus = if inst.gpu_ids.is_empty() {
            "-".to_string()
        } else {
            inst.gpu_ids.join(",")
        };
        let gpus = trunc(&gpus, 10);
        lines.push(Line::from(vec![
            Span::styled(format!("{name:<18} "), Style::default().fg(status_color)),
            Span::styled(format!("{model:<20} "), Style::default().fg(theme.fg)),
            Span::styled(
                format!("{port:>5} {:>3} ", inst.tensor_parallel_size),
                Style::default().fg(theme.muted),
            ),
            Span::styled(format!("{gpus:<10}"), Style::default().fg(theme.accent)),
        ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_bench(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let count = state.bench_rows.len();
    let title = format!(" Bench rows · {count} ");
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.bench_rows.is_empty() {
        let p = Paragraph::new(Line::from(Span::styled(
            "no rows · start daemon with --bench-csv <path>",
            Style::default().fg(theme.muted),
        )));
        f.render_widget(p, inner);
        return;
    }

    let header = Line::from(vec![Span::styled(
        format!(
            "{:<10} {:>3} {:<20} {:>12} {:>12} {:<8}",
            "cell", "run", "model", "pTPS", "gTPS", "verdict"
        ),
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    )]);

    let rows_shown = (inner.height as usize).saturating_sub(1);
    let start = state.bench_rows.len().saturating_sub(rows_shown);
    let mut lines: Vec<Line> = Vec::with_capacity(rows_shown + 1);
    lines.push(header);
    for r in state.bench_rows.iter().skip(start) {
        let verdict_color = match r.pass_fail {
            rocm_dash_core::bench_schema::PassFail::Pass => theme.ok,
            rocm_dash_core::bench_schema::PassFail::Fail => theme.err,
            rocm_dash_core::bench_schema::PassFail::Unknown => match r.judge_pass_fail {
                rocm_dash_core::bench_schema::PassFail::Pass => theme.ok,
                rocm_dash_core::bench_schema::PassFail::Fail => theme.err,
                rocm_dash_core::bench_schema::PassFail::Unknown => theme.muted,
            },
        };
        let verdict_text = match (r.pass_fail, r.judge_pass_fail) {
            (rocm_dash_core::bench_schema::PassFail::Unknown, j) => format!("{j:?}"),
            (p, _) => format!("{p:?}"),
        };
        let model = r.model.as_deref().unwrap_or("?");
        let model_trunc = if model.len() > 20 {
            &model[..20]
        } else {
            model
        };
        let cell = if r.cell.len() > 10 {
            &r.cell[..10]
        } else {
            r.cell.as_str()
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("{cell:<10} {:>3} {model_trunc:<20} ", r.run),
                Style::default().fg(theme.fg),
            ),
            Span::styled(
                format!(
                    "{:>12} {:>12} ",
                    format::tps_opt(r.prompt_tps),
                    format::tps_opt(r.gen_tps)
                ),
                Style::default().fg(theme.muted),
            ),
            Span::styled(verdict_text, Style::default().fg(verdict_color)),
        ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
}
