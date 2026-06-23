//! Observe tab — the folded telemetry surface.
//!
//! P3 collapses the former Overview / Hardware / Instances / Bench tabs into one
//! instrument-cluster view. Rather than rewrite their renderers, Observe
//! composes the EXISTING per-surface draw fns into stacked regions (ponytail):
//! GPU/hardware cluster up top, the instances table in the middle, the bench
//! rollup at the bottom. The selectable list is the instances table (wired via
//! the main selection model in `AppState`).
//!
//! The amber demo-data banner (F151) is shown iff live daemon telemetry is
//! absent — driven by `ConnState` / snapshot presence, never a hardcoded flag.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::{AppState, ConnState};
use crate::ui::theme::Theme;

/// True when we have no live daemon telemetry to show — i.e. not connected to a
/// running daemon, or connected but no snapshot has arrived yet. The demo-data
/// banner is shown exactly in this case (honesty: numbers may be placeholders).
const fn telemetry_absent(state: &AppState) -> bool {
    !matches!(state.conn, ConnState::Connected { .. }) || state.latest.is_none()
}

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let body = if telemetry_absent(state) {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .split(area);
        draw_demo_banner(f, split[0], theme);
        split[1]
    } else {
        area
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(35),
            Constraint::Percentage(25),
        ])
        .split(body);

    // Compose the existing renderers — no reimplementation.
    super::hardware::draw(f, rows[0], state, theme);
    super::instances::draw(f, rows[1], state, theme);
    super::bench::draw(f, rows[2], state, theme);
}

fn draw_demo_banner(f: &mut Frame, area: Rect, theme: &Theme) {
    let banner = Paragraph::new(Line::from(vec![
        Span::styled(
            " ⚠ demo data ",
            Style::default()
                .bg(theme.warn)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "  no live ROCm daemon — telemetry shown is synthetic / last-known",
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
    use rocm_dash_core::metrics::{GpuMetrics, Snapshot, SystemMetrics};

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

    #[test]
    fn observe_folds_telemetry_surfaces() {
        let out = render(&connected_with_snapshot(), 160, 44);
        // GPU cluster + instances + bench rollup all present in one tab.
        assert!(out.contains("GPU0"), "gpu cluster missing: {out:?}");
        assert!(out.contains("Instances"), "instances surface missing");
        assert!(out.contains("Bench"), "bench surface missing");
    }

    #[test]
    fn demo_banner_absent_when_connected_with_telemetry() {
        let out = render(&connected_with_snapshot(), 160, 44);
        assert!(
            !out.contains("demo data"),
            "demo banner must be hidden when connected with telemetry: {out:?}"
        );
    }

    #[test]
    fn demo_banner_present_when_telemetry_absent() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Observe;
        // Initial conn, no snapshot → telemetry absent.
        let out = render(&s, 160, 44);
        assert!(out.contains("demo data"), "demo banner expected: {out:?}");
    }

    #[test]
    fn observe_does_not_panic_when_squeezed() {
        let s = connected_with_snapshot();
        for h in [1u16, 2, 3, 5, 8, 12] {
            let _ = render(&s, 80, h);
        }
    }
}
