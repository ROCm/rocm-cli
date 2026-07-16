// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Observe tab — the AI-serving telemetry surface.
//!
//! P5 reorients Observe around AI-serving efficiency/throughput: a top band of
//! two hero panels — **Node efficiency** (tok/watt + trend) and **Node
//! throughput** (Σ gen_tps + total power W) — over an AI per-instance table
//! (model · tok/s · tok/watt · TTFT · TPOT · power · queue · kv%). The deep
//! hardware (GPU/host) cluster and the bench rollup are kept reachable below so
//! nothing is lost. The selectable list is the instances table (wired via the
//! main selection model in `AppState`).
//!
//! The amber no-live-data banner is shown iff live daemon telemetry is
//! absent — driven by `ConnState` / snapshot presence, never a hardcoded flag.
//! Simulated (`--demo`/`--replay`) sessions are marked separately by the global
//! SIMULATED DATA header chip; see `AppState::simulated`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::{AppState, ConnState};
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;
use crate::ui::{format, widgets};

/// True when we have no live daemon telemetry to show — i.e. not connected to a
/// running daemon, or connected but no snapshot has arrived yet. The
/// no-live-data banner is shown exactly in this case (honesty: numbers may be
/// placeholders).
const fn telemetry_absent(state: &AppState) -> bool {
    !matches!(state.conn, ConnState::Connected { .. }) || state.latest.is_none()
}

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let body = if telemetry_absent(state) {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .split(area);
        draw_no_live_data_banner(f, split[0], theme);
        split[1]
    } else {
        area
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),      // hero band (two panels)
            Constraint::Percentage(40), // AI per-instance table
            Constraint::Percentage(35), // deep hardware/GPU cluster
            Constraint::Min(6),         // bench rollup
        ])
        .split(body);

    draw_hero_band(f, rows[0], state, theme);
    // AI per-instance table (tok/watt, TTFT, TPOT, queue, kv%).
    super::instances::draw_table(f, rows[1], state, theme);
    // Deep hardware + bench detail kept reachable below the AI surface.
    super::hardware::draw(f, rows[2], state, theme);
    super::bench::draw(f, rows[3], state, theme);
}

/// Two hero panels: Node efficiency (tok/watt + trend) and Node throughput
/// (Σ gen_tps + total power W). Reuses `widgets::node_efficiency` and the
/// braille sparkline; honest `—` when a metric is unavailable (color is never
/// the only signal — every value carries a text label + unit).
fn draw_hero_band(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
        .split(area);
    draw_efficiency_hero(f, cols[0], state, theme);
    draw_throughput_hero(f, cols[1], state, theme);
}

/// Node-efficiency hero: the big tok/watt number + a trend sparkline over the
/// snapshot history (the redesign's headline AI-serving efficiency metric).
fn draw_efficiency_hero(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let inner = panel::bento(
        f,
        area,
        Some("Node efficiency"),
        BoxRole::Primary,
        false,
        theme,
    );
    if inner.height == 0 {
        return;
    }
    let eff = state.latest.as_ref().and_then(widgets::node_efficiency);
    let value = eff.map_or_else(|| "—".to_string(), |v| format!("{v:.2} tok/W"));
    // Trend: node efficiency across recent snapshots (scaled to milli-tok/W so
    // the integer sparkline keeps resolution); empty when no history yet.
    let trend: Vec<u64> = state
        .history
        .iter()
        .filter_map(widgets::node_efficiency)
        .map(|v| (v * 1000.0).round().max(0.0) as u64)
        .collect();

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            value,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))),
        rows[0],
    );
    if !trend.is_empty() && rows[1].height > 0 {
        f.render_widget(
            crate::ui::sparkline::BrailleSparkline::new(&trend)
                .style(Style::default().fg(theme.accent_2)),
            rows[1],
        );
    }
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "tokens per watt · whole node",
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );
}

/// Node-throughput hero: Σ generation tok/s across instances + total board
/// power (W). Both honest `—` when absent.
fn draw_throughput_hero(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let inner = panel::bento(
        f,
        area,
        Some("Node throughput"),
        BoxRole::Secondary,
        false,
        theme,
    );
    if inner.height == 0 {
        return;
    }
    let (tps, has_tps, power, has_power) =
        state
            .latest
            .as_ref()
            .map_or((0.0, false, 0.0_f32, false), |snap| {
                let mut tps = 0.0;
                let mut has_tps = false;
                for inst in &snap.instances {
                    if let Some(v) = inst.gen_tps {
                        tps += v;
                        has_tps = true;
                    }
                }
                let power: f32 = snap.gpus.iter().map(|g| g.power_w).sum();
                (tps, has_tps, power, !snap.gpus.is_empty())
            });
    let tps_str = if has_tps {
        format::tps_opt(Some(tps))
    } else {
        "—".to_string()
    };
    let power_str = if has_power {
        format::watts(power)
    } else {
        "—".to_string()
    };
    let lines = vec![
        Line::from(Span::styled(
            tps_str,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("power ", Style::default().fg(theme.muted)),
            Span::styled(power_str, Style::default().fg(theme.fg)),
        ]),
        Line::from(Span::styled(
            "Σ generation throughput · all instances",
            Style::default().fg(theme.muted),
        )),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_no_live_data_banner(f: &mut Frame, area: Rect, theme: &Theme) {
    let banner = Paragraph::new(Line::from(vec![
        Span::styled(
            " ⚠ no live data ",
            Style::default()
                .bg(theme.warn)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "  not connected to a ROCm daemon — showing placeholders / last-known",
            Style::default().fg(theme.warn),
        ),
    ]));
    f.render_widget(banner, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::ActiveTab;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use rocm_dash_core::metrics::{GpuMetrics, Instance, InstanceStatus, Snapshot, SystemMetrics};

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

    fn connected_with_snapshot() -> AppState {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Observe;
        s.conn = ConnState::Connected {
            host: "localhost".into(),
            version: "1.0".into(),
        };
        s.latest = Some(Snapshot {
            host: SystemMetrics::default(),
            gpus: vec![GpuMetrics {
                device_id: "GPU0".into(),
                vram_used_mb: 40_000,
                vram_total_mb: 192_000,
                gpu_utilization_pct: 50.0,
                temperature_c: 55.0,
                power_w: 300.0,
                clock_mhz: Some(2000.0),
            }],
            ..Default::default()
        });
        s
    }

    fn instance(model: &str, gen_tps: Option<f64>, ttft: Option<f64>) -> Instance {
        Instance {
            container_id: model.into(),
            container_name: model.into(),
            status: InstanceStatus::Running,
            model_name: model.into(),
            gpu_ids: vec!["0".into()],
            gen_tps,
            tokens_per_watt: gen_tps.map(|t| t / 300.0),
            ttft_ms: ttft,
            tpot_ms: ttft.map(|_| 22.0),
            running_reqs: gen_tps.map(|_| 3),
            waiting_reqs: gen_tps.map(|_| 0),
            kv_cache_usage_pct: gen_tps.map(|_| 42.0),
            ..Default::default()
        }
    }

    fn connected_with_instances(insts: Vec<Instance>) -> AppState {
        let mut s = connected_with_snapshot();
        if let Some(snap) = s.latest.as_mut() {
            snap.instances = insts.clone();
        }
        for i in insts {
            s.instances.insert(i.container_id.clone(), i);
        }
        s
    }

    #[test]
    fn observe_folds_telemetry_surfaces() {
        // Render tall so the deep hardware/bench detail below the AI band is
        // fully reachable (GPU cluster included).
        let out = render(&connected_with_snapshot(), 160, 64);
        assert!(
            out.contains("GPU0"),
            "deep GPU cluster not reachable: {out:?}"
        );
        assert!(out.contains("Instances"), "instances surface missing");
        assert!(out.contains("Bench"), "bench surface missing");
    }

    #[test]
    fn observe_renders_two_hero_panels() {
        let out = render(
            &connected_with_instances(vec![instance("m", Some(180.0), Some(120.0))]),
            160,
            50,
        );
        assert!(
            out.contains("Node efficiency"),
            "efficiency hero missing: {out:?}"
        );
        assert!(
            out.contains("Node throughput"),
            "throughput hero missing: {out:?}"
        );
        // Throughput hero surfaces total power and Σ tok/s.
        assert!(out.contains("tok/W"), "tok/watt headline missing");
        assert!(out.contains("power"), "node power missing");
    }

    #[test]
    fn observe_table_shows_ai_columns_with_values() {
        let out = render(
            &connected_with_instances(vec![instance("llama", Some(200.0), Some(150.0))]),
            160,
            50,
        );
        for header in [
            "MODEL", "TOK/S", "TOK/W", "TTFT", "TPOT", "POWER", "QUEUE", "KV%",
        ] {
            assert!(
                out.contains(header),
                "table header {header:?} missing: {out:?}"
            );
        }
        // A sample row's live values render (TTFT + TPOT in ms, queue r/w).
        assert!(out.contains("150ms"), "TTFT value missing: {out:?}");
        // Fixture sets tpot_ms = 22.0 (observe.rs `instance`); assert it too.
        assert!(out.contains("22ms"), "TPOT value missing: {out:?}");
        assert!(
            out.contains("3/0"),
            "queue running/waiting missing: {out:?}"
        );
    }

    #[test]
    fn observe_missing_metrics_render_em_dash() {
        // An instance with no live metrics renders honest `—` placeholders and
        // NO fabricated numbers for its AI columns.
        let out = render(
            &connected_with_instances(vec![instance("empty", None, None)]),
            160,
            50,
        );
        assert!(out.contains("—"), "honest placeholder missing: {out:?}");
        // No fabricated throughput/latency for the empty instance.
        assert!(!out.contains("0ms"), "fabricated 0ms latency: {out:?}");
        assert!(!out.contains("0.00"), "fabricated 0.00 tok/W: {out:?}");
    }

    #[test]
    fn no_live_data_banner_absent_when_connected_with_telemetry() {
        let out = render(&connected_with_snapshot(), 160, 44);
        assert!(
            !out.contains("no live data"),
            "no-live-data banner must be hidden when connected with telemetry: {out:?}"
        );
    }

    #[test]
    fn no_live_data_banner_present_when_telemetry_absent() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Observe;
        // Initial conn, no snapshot → telemetry absent.
        let out = render(&s, 160, 44);
        assert!(
            out.contains("no live data"),
            "no-live-data banner expected: {out:?}"
        );
    }

    #[test]
    fn observe_does_not_panic_when_squeezed() {
        let s = connected_with_snapshot();
        for h in [1u16, 2, 3, 5, 8, 12] {
            let _ = render(&s, 80, h);
        }
    }
}
