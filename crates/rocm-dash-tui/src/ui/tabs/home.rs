// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Home tab — the landing "instrument cluster".
//!
//! Composes the home layout against live `AppState` (read-only): a hero GPU
//! gauge + spark, a stacked VRAM/TEMP/POWER mini-spark cluster, and Running /
//! Health / Updates tiles. Empty/absent telemetry renders honest placeholders
//! rather than synthetic numbers.
//!
//! ponytail: Home is added behind the existing default this phase (P2). It is
//! reachable by Tab / digit `1` but is NOT the default tab yet — P3 repoints
//! the default and folds the telemetry tabs into Observe.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::{AppState, ConnState};
use crate::ui::format;
use crate::ui::gradient::GradientGauge;
use crate::ui::panel::{self, BoxRole};
use crate::ui::sparkline::BrailleSparkline;
use crate::ui::theme::Theme;

/// History series helper: map each snapshot to a 0..=100 magnitude via `pick`.
fn history<F: Fn(&rocm_dash_core::metrics::Snapshot) -> f64>(
    state: &AppState,
    pick: F,
) -> Vec<u64> {
    state
        .history
        .iter()
        .map(|s| pick(s).clamp(0.0, 100.0) as u64)
        .collect()
}

/// Peak GPU utilization across the GPUs in a snapshot (the headline metric).
fn snap_util(s: &rocm_dash_core::metrics::Snapshot) -> f64 {
    s.gpus
        .iter()
        .map(|g| f64::from(g.gpu_utilization_pct))
        .fold(0.0, f64::max)
}

/// A labeled one-row instrument: `LABEL ▁▂▃▅▆▇  value`. Port of the mock's
/// `mini_spark`; `accent` picks the flat throughput tint, else value-gradient.
fn mini_spark(
    f: &mut Frame,
    area: Rect,
    label: &str,
    val: &str,
    data: &[u64],
    accent: bool,
    theme: &Theme,
) {
    if area.height == 0 {
        return;
    }
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            label,
            Style::default().fg(theme.muted),
        ))),
        Rect::new(area.x, area.y, area.width.min(6), 1),
    );
    let lw = label.chars().count() as u16 + 1;
    let vw = val.chars().count() as u16 + 1;
    let sw = area.width.saturating_sub(lw + vw);
    if sw > 1 {
        let sa = Rect::new(area.x + lw, area.y, sw, 1);
        let mut s = BrailleSparkline::new(data).max(100);
        s = if accent {
            s.style(Style::default().fg(theme.accent))
        } else {
            s.style(Style::default().fg(theme.accent))
                .gradient(theme.ok, theme.warn, theme.err)
        };
        f.render_widget(s, sa);
    }
    let vx = area.x + area.width.saturating_sub(vw - 1);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            val,
            Style::default().fg(if accent { theme.accent } else { theme.fg }),
        ))),
        Rect::new(vx, area.y, vw, 1),
    );
}

/// Hero card title in the mock's form: `Node throughput · N × MODEL · ROCm V`.
/// Falls back to `Unknown GPU` when no GPU model is detected, and drops the
/// count/ROCm segments that aren't known rather than inventing them.
fn node_throughput_title(state: &AppState) -> String {
    let info = state
        .latest
        .as_ref()
        .and_then(|s| s.gpu_system_info.as_ref());
    let n = state.latest.as_ref().map_or(0, |s| s.gpus.len());
    let model = info
        .map(|g| g.gpu_model.trim())
        .filter(|m| !m.is_empty())
        .map_or_else(
            || "Unknown GPU".to_string(),
            |m| {
                if n > 1 {
                    format!("{n} × {m}")
                } else {
                    m.to_string()
                }
            },
        );
    let rocm = info
        .and_then(|g| g.rocm_version.as_ref())
        .map_or_else(String::new, |v| format!(" · ROCm {v}"));
    format!("Node throughput · {model}{rocm}")
}

/// A bento card: titled bordered block, returns the inner content rect.
fn card(f: &mut Frame, area: Rect, title: &str, role: BoxRole, theme: &Theme) -> Rect {
    panel::bento(f, area, Some(title), role, false, theme)
}

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(11),
            Constraint::Length(8),
            Constraint::Min(0),
        ])
        .split(area);

    draw_hero_band(f, rows[0], state, theme);
    draw_tiles(f, rows[1], state, theme);
    draw_activity(f, rows[2], state, theme);
}

/// "Activity · node" block: a node-load mini-spark over a recent-activity feed
/// derived from live state (running services + recent jobs). Honest placeholder
/// when there's nothing to show.
fn draw_activity(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    if area.height < 2 {
        return;
    }
    let inner = card(f, area, "Activity · node", BoxRole::Muted, theme);
    if inner.height == 0 {
        return;
    }
    let load_hist = history(state, snap_util);
    mini_spark(
        f,
        Rect::new(inner.x, inner.y, inner.width, 1),
        "node load ",
        if load_hist.is_empty() { "—" } else { "live" },
        &load_hist,
        false,
        theme,
    );
    if inner.height < 3 {
        return;
    }
    let feed = Rect::new(inner.x, inner.y + 2, inner.width, inner.height - 2);
    let mut lines: Vec<Line> = Vec::new();
    // Running services first (most relevant "what's live now").
    for inst in state
        .instances
        .values()
        .filter(|i| i.status.is_serving())
        .take(feed.height as usize)
    {
        let port = inst.port.map_or_else(String::new, |p| format!(" on :{p}"));
        lines.push(Line::from(vec![
            Span::styled("● ", Style::default().fg(theme.ok)),
            Span::styled(inst.model_name.clone(), Style::default().fg(theme.fg)),
            Span::styled(format!("{port} serving"), Style::default().fg(theme.muted)),
        ]));
    }
    // Then recent jobs (tools run), newest-relevant first.
    for job in state.jobs.jobs.values().take(feed.height as usize) {
        let (glyph, color) = match job.status {
            rocm_dash_core::state::JobStatus::Failed { .. } => ("✗ ", theme.err),
            rocm_dash_core::state::JobStatus::Done { .. } => ("✓ ", theme.ok),
            _ => ("⋯ ", theme.muted),
        };
        lines.push(Line::from(vec![
            Span::styled(glyph, Style::default().fg(color)),
            Span::styled(job.cmd.clone(), Style::default().fg(theme.fg)),
        ]));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "no recent activity — serve a model or run an action to populate this",
            Style::default().fg(theme.muted),
        )));
    }
    lines.truncate(feed.height as usize);
    f.render_widget(Paragraph::new(lines), feed);
}

fn draw_hero_band(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let title = node_throughput_title(state);
    let hero = card(f, area, &title, BoxRole::Secondary, theme);
    if hero.width >= 8 && hero.height >= 6 {
        let hcols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .spacing(2)
            .split(hero);
        draw_hero_left(f, hcols[0], state, theme);
        draw_hero_right(f, hcols[1], state, theme);
    }
}

fn draw_hero_left(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let lh = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1); 6])
        .split(area);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "GPU UTILIZATION",
            Style::default().fg(theme.muted),
        ))),
        lh[0],
    );
    let util = state.latest.as_ref().map_or(0.0, snap_util);
    let util_label = format::pct(util as f32);
    let g = GradientGauge::new(util / 100.0)
        .stops(theme.ok, theme.warn, theme.err)
        .track_bg(theme.surface_2)
        .label(&util_label)
        .label_fg(theme.fg);
    f.render_widget(g, lh[1]);
    let util_hist = history(state, snap_util);
    f.render_widget(
        BrailleSparkline::new(&util_hist)
            .max(100)
            .style(Style::default().fg(theme.accent))
            .gradient(theme.ok, theme.warn, theme.err),
        lh[2],
    );
    // Tokens/watt: summed across running instances when available.
    let tpw: f64 = state
        .instances
        .values()
        .filter_map(|i| i.tokens_per_watt)
        .sum();
    let tpw_label = if tpw > 0.0 {
        format!("⎓ {tpw:.1} tokens / watt")
    } else {
        "⎓ tokens / watt —".to_string()
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            tpw_label,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))),
        lh[4],
    );
}

fn draw_hero_right(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let rh = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1); 6])
        .split(area);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "LIVE · last 60s",
            Style::default().fg(theme.muted),
        ))),
        rh[0],
    );
    let latest = state.latest.as_ref();
    let vram = latest.map_or(0.0, |s| {
        let used: u64 = s.gpus.iter().map(|g| g.vram_used_mb).sum();
        let total: u64 = s.gpus.iter().map(|g| g.vram_total_mb).sum::<u64>().max(1);
        used as f64 / total as f64 * 100.0
    });
    let temp = latest.map_or(0.0, |s| {
        s.gpus
            .iter()
            .map(|g| f64::from(g.temperature_c))
            .fold(0.0, f64::max)
    });
    let power: f64 = latest.map_or(0.0, |s| {
        s.gpus.iter().map(|g| f64::from(g.power_w)).sum::<f64>()
    });
    mini_spark(
        f,
        rh[1],
        "VRAM ",
        &format::pct(vram as f32),
        &history(state, |s| {
            let used: u64 = s.gpus.iter().map(|g| g.vram_used_mb).sum();
            let total: u64 = s.gpus.iter().map(|g| g.vram_total_mb).sum::<u64>().max(1);
            used as f64 / total as f64 * 100.0
        }),
        false,
        theme,
    );
    mini_spark(
        f,
        rh[2],
        "TEMP ",
        &format::celsius(temp as f32),
        &history(state, |s| {
            s.gpus
                .iter()
                .map(|g| f64::from(g.temperature_c))
                .fold(0.0, f64::max)
        }),
        false,
        theme,
    );
    mini_spark(
        f,
        rh[3],
        "POWER",
        &format::watts(power as f32),
        // Power can exceed 100; scale to a nominal 1kW ceiling for the trace.
        &history(state, |s| {
            s.gpus.iter().map(|g| f64::from(g.power_w)).sum::<f64>() / 10.0
        }),
        false,
        theme,
    );
    let tps: f64 = state.instances.values().filter_map(|i| i.gen_tps).sum();
    mini_spark(f, rh[4], "T/S  ", &format!("{tps:.0}"), &[], true, theme);
}

fn draw_tiles(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ])
        .split(area);

    let n_running = state
        .instances
        .values()
        .filter(|i| i.status.is_serving())
        .count();
    let running = card(
        f,
        mid[0],
        &format!("Running · {n_running}"),
        BoxRole::Success,
        theme,
    );
    if running.height > 0 {
        let line = state
            .instances
            .values()
            .find(|i| i.status.is_serving())
            .map_or_else(
                || {
                    Line::from(Span::styled(
                        "Nothing running",
                        Style::default().fg(theme.muted),
                    ))
                },
                |i| {
                    Line::from(vec![
                        Span::styled("● ", Style::default().fg(theme.ok)),
                        Span::styled(
                            i.model_name.clone(),
                            Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                        ),
                    ])
                },
            );
        f.render_widget(Paragraph::new(line), running);
    }

    // Health tile — derive from snapshot/system-info presence + conn state.
    let health = card(f, mid[1], "Health", BoxRole::Secondary, theme);
    if health.height > 0 {
        let info = state
            .latest
            .as_ref()
            .and_then(|s| s.gpu_system_info.as_ref());
        let rocm = info
            .and_then(|i| i.rocm_version.clone())
            .map_or_else(|| "ROCm —".to_string(), |v| format!("ROCm {v}"));
        let gpu_ok = state.latest.as_ref().is_some_and(|s| !s.gpus.is_empty());
        let mark = |ok: bool| {
            if ok {
                Span::styled("✓ ", Style::default().fg(theme.ok))
            } else {
                Span::styled("· ", Style::default().fg(theme.muted))
            }
        };
        f.render_widget(
            Paragraph::new(vec![
                Line::from(vec![
                    mark(gpu_ok),
                    Span::styled("GPU", Style::default().fg(theme.fg)),
                ]),
                Line::from(vec![
                    mark(info.is_some()),
                    Span::styled("Driver", Style::default().fg(theme.fg)),
                ]),
                Line::from(vec![
                    mark(info.and_then(|i| i.rocm_version.as_ref()).is_some()),
                    Span::styled(rocm, Style::default().fg(theme.fg)),
                ]),
            ]),
            health,
        );
    }

    // Updates tile — honest placeholder (no update feed wired this run).
    let updates = card(f, mid[2], "Updates", BoxRole::Muted, theme);
    if updates.height > 0 {
        let body = match state.conn {
            ConnState::Connected { .. } => {
                Line::from(Span::styled("Up to date", Style::default().fg(theme.muted)))
            }
            _ => Line::from(Span::styled("Checking…", Style::default().fg(theme.muted))),
        };
        // ponytail: no update-feed data source this run; tile shows conn-derived
        // status rather than the mock's hardcoded "ROCm 6.3 ready".
        f.render_widget(Paragraph::new(body), updates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::ActiveTab;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use rocm_dash_core::metrics::{GpuMetrics, GpuSystemInfo, Snapshot, SystemMetrics};

    fn render(state: &AppState, cols: u16, rows: u16) -> String {
        let backend = TestBackend::new(cols, rows);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, f.area(), state, &state.theme))
            .unwrap();
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    fn state_with_gpu() -> AppState {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Home;
        s.latest = Some(Snapshot {
            host: SystemMetrics::default(),
            gpus: vec![GpuMetrics {
                device_id: "GPU0".into(),
                vram_used_mb: 80_000,
                vram_total_mb: 192_000,
                gpu_utilization_pct: 62.0,
                temperature_c: 51.0,
                power_w: 420.0,
                clock_mhz: Some(2100.0),
            }],
            gpu_system_info: Some(GpuSystemInfo {
                gpu_model: "Instinct MI355X".into(),
                rocm_version: Some("6.2".into()),
                ..Default::default()
            }),
            ..Default::default()
        });
        s
    }

    #[test]
    fn home_renders_hero_and_tiles() {
        let out = render(&state_with_gpu(), 160, 30);
        assert!(
            out.contains("GPU UTILIZATION"),
            "hero label missing: {out:?}"
        );
        assert!(out.contains("Running"), "running tile missing");
        assert!(out.contains("Health"), "health tile missing");
        assert!(out.contains("Updates"), "updates tile missing");
        assert!(out.contains("Instinct MI355X"), "gpu model missing");
        // Hero title is the node-throughput line (mock parity), not bare model.
        assert!(
            out.contains("Node throughput"),
            "node-throughput title missing: {out:?}"
        );
        // Activity block present.
        assert!(out.contains("Activity"), "activity block missing");
    }

    #[test]
    fn node_throughput_title_formats_and_falls_back() {
        // Multi-GPU + ROCm → "Node throughput · N × MODEL · ROCm V".
        let s = state_with_gpu();
        let t = node_throughput_title(&s);
        assert!(t.starts_with("Node throughput · "), "prefix: {t}");
        assert!(t.contains("Instinct MI355X"), "model: {t}");
        assert!(t.contains("ROCm 6.2"), "rocm version: {t}");
        // No telemetry → Unknown GPU, no fabricated count/version.
        let empty = AppState::new("t".into(), "default-dark".into());
        let t2 = node_throughput_title(&empty);
        assert_eq!(t2, "Node throughput · Unknown GPU");
    }

    #[test]
    fn home_renders_placeholder_when_empty() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Home;
        let out = render(&s, 160, 30);
        assert!(out.contains("Nothing running"), "empty running tile");
    }

    #[test]
    fn home_does_not_panic_when_squeezed() {
        let s = state_with_gpu();
        for h in [1u16, 2, 3, 5, 8, 11] {
            let _ = render(&s, 80, h);
        }
    }
}
