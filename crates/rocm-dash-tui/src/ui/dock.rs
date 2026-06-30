// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Wide-layout triptych rails (Phase 6).
//!
//! When the terminal is wide enough, `ui::draw` splits the body into a left GPU
//! wall, the existing center tab body, and a persistent right dock. The right
//! dock is LOGS or CONTEXT/DETAIL — NEVER a chat composer (user directive). The
//! mock's interactive-composer rail is deliberately not ported; only the
//! read-only LOGS and CONTEXT rails are.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::metrics::InstanceStatus;
use rocm_dash_core::state::JobStatus;

use crate::app::{ActiveTab, AppState};
use crate::ui::format;
use crate::ui::gradient::GradientGauge;
use crate::ui::panel::{self, BoxRole};
use crate::ui::sparkline::BrailleSparkline;
use crate::ui::theme::Theme;

/// Minimum terminal size for the wide triptych. Below this the dash stays
/// single-column (byte-for-byte the pre-Phase-6 layout).
pub const WIDE_COLS: u16 = 180;
pub const WIDE_ROWS: u16 = 45;
/// Left GPU-wall width and right-dock width in the wide layout.
pub const RAIL_W: u16 = 40;
pub const DOCK_W: u16 = 52;

/// Whether `area` qualifies for the wide triptych.
#[must_use]
pub const fn is_wide(cols: u16, rows: u16) -> bool {
    cols >= WIDE_COLS && rows >= WIDE_ROWS
}

/// Split a body rect into (left rail, center, right dock) for the wide layout.
/// Returns `None` when the body is too narrow to host both rails + a center.
#[must_use]
pub const fn triptych(body: Rect) -> Option<(Rect, Rect, Rect)> {
    let min_center = 40u16;
    if body.width < RAIL_W + DOCK_W + min_center {
        return None;
    }
    let left = Rect::new(body.x, body.y, RAIL_W, body.height);
    let center_w = body.width - RAIL_W - DOCK_W;
    let center = Rect::new(body.x + RAIL_W, body.y, center_w, body.height);
    let right = Rect::new(body.x + RAIL_W + center_w, body.y, DOCK_W, body.height);
    Some((left, center, right))
}

/// Draw the right dock appropriate for `tab`: a live LOGS stream for the
/// operational tabs (Observe/ROCm/Serving), a CONTEXT rail for Home/Chat.
pub fn draw_right_dock(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    match state.active_tab {
        ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving => {
            logs_dock(f, area, state, theme);
        }
        ActiveTab::Home | ActiveTab::Chat => context_rail(f, area, state, theme),
    }
}

/// The persistent left rail: a wall of GPU instrument cards from live snapshot
/// telemetry (port of the mock `gpu_wall`, read-only).
pub fn gpu_wall(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let gpus = state.latest.as_ref().map_or(&[][..], |s| s.gpus.as_slice());
    let title = format!("GPUs · {}", gpus.len());
    let inner = panel::bento(f, area, Some(&title), BoxRole::Secondary, false, theme);
    if inner.height == 0 {
        return;
    }
    if gpus.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "no GPU telemetry",
                Style::default().fg(theme.muted),
            ))),
            inner,
        );
        return;
    }
    let per = 4u16; // 3 content rows + 1 gap
    for (i, g) in gpus.iter().enumerate() {
        let y = inner.y + i as u16 * per;
        if y + 3 > inner.y + inner.height {
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    format!("… +{} more", gpus.len() - i),
                    Style::default().fg(theme.muted),
                ))),
                Rect::new(inner.x, y, inner.width, 1),
            );
            break;
        }
        let util = f64::from(g.gpu_utilization_pct);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("● GPU{i}"),
                Style::default().fg(theme.ok),
            ))),
            Rect::new(inner.x, y, inner.width, 1),
        );
        f.render_widget(
            GradientGauge::new(util / 100.0)
                .stops(theme.ok, theme.warn, theme.err)
                .track_bg(theme.surface_2)
                .label(&format::pct(util as f32))
                .label_fg(theme.fg),
            Rect::new(inner.x, y + 1, inner.width, 1),
        );
        let vram = if g.vram_total_mb > 0 {
            g.vram_used_mb as f64 / g.vram_total_mb as f64 * 100.0
        } else {
            0.0
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("VRAM {}", format::pct(vram as f32)),
                Style::default().fg(theme.muted),
            ))),
            Rect::new(inner.x, y + 2, inner.width, 1),
        );
    }
}

/// Live LOGS dock: the most recent job-console lines across `AppState.jobs`
/// (reuses the existing job output ring — no new data source).
pub fn logs_dock(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let title = if state.dock_logs_scroll > 0 {
        format!("LOGS · ↑{}", state.dock_logs_scroll)
    } else {
        "LOGS".to_string()
    };
    let inner = panel::bento(f, area, Some(&title), BoxRole::Muted, false, theme);
    if inner.height == 0 {
        return;
    }
    // Full chronological buffer across jobs (jobs sorted by id for a stable
    // order so the scroll offset is deterministic frame-to-frame).
    let mut jobs: Vec<_> = state.jobs.jobs.iter().collect();
    jobs.sort_by(|a, b| a.0.cmp(b.0));
    let mut lines: Vec<Line> = Vec::new();
    for (_, job) in jobs {
        let color = match job.status {
            JobStatus::Failed { .. } => theme.err,
            JobStatus::Done { .. } => theme.ok,
            _ => theme.fg,
        };
        for l in &job.output {
            lines.push(Line::from(Span::styled(
                l.clone(),
                Style::default().fg(color),
            )));
        }
    }
    if lines.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "no job output yet — start an action to stream logs",
                Style::default().fg(theme.muted),
            ))),
            inner,
        );
        return;
    }

    // Window the buffer to the dock height, anchored to the tail. `dock_logs_scroll`
    // counts lines UP from the newest; clamp it so we never scroll past the top.
    let cap = inner.height as usize;
    let total = lines.len();
    let max_scroll = total.saturating_sub(cap);
    let scroll = (state.dock_logs_scroll as usize).min(max_scroll);
    let end = total - scroll;
    let start = end.saturating_sub(cap);
    let window: Vec<Line> = lines[start..end].to_vec();
    f.render_widget(Paragraph::new(window), inner);
}

/// CONTEXT rail: RUNNING SERVICES / GPU STATE / RECENT TOOLS read from
/// instances / snapshot / jobs (read-only). This is the agent-grounding pane —
/// not a composer.
pub fn context_rail(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let inner = panel::bento(f, area, Some("CONTEXT"), BoxRole::Primary, false, theme);
    if inner.height == 0 {
        return;
    }
    let header = |t: &'static str| {
        Line::from(Span::styled(
            t,
            Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD),
        ))
    };
    let mut lines: Vec<Line> = vec![header("RUNNING SERVICES")];
    let running: Vec<_> = state
        .instances
        .values()
        .filter(|i| i.status == InstanceStatus::Running)
        .collect();
    if running.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default().fg(theme.muted),
        )));
    } else {
        for i in running.iter().take(4) {
            let port = i.port.map_or_else(String::new, |p| format!(" :{p}"));
            lines.push(Line::from(vec![
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled(i.model_name.clone(), Style::default().fg(theme.fg)),
                Span::styled(port, Style::default().fg(theme.muted)),
            ]));
        }
    }
    lines.push(Line::raw(""));
    lines.push(header("GPU STATE"));
    if let Some(s) = state.latest.as_ref() {
        let util = s
            .gpus
            .iter()
            .map(|g| f64::from(g.gpu_utilization_pct))
            .fold(0.0, f64::max);
        lines.push(Line::from(Span::styled(
            format!("  peak util {}", format::pct(util as f32)),
            Style::default().fg(theme.fg),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "  no telemetry",
            Style::default().fg(theme.muted),
        )));
    }
    lines.push(Line::raw(""));
    lines.push(header("RECENT TOOLS"));
    let mut tools = state
        .jobs
        .jobs
        .values()
        .map(|j| j.cmd.clone())
        .collect::<Vec<_>>();
    tools.truncate(4);
    if tools.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none yet",
            Style::default().fg(theme.muted),
        )));
    } else {
        for t in tools {
            lines.push(Line::from(Span::styled(
                format!("  {t}"),
                Style::default().fg(theme.fg),
            )));
        }
    }
    // A small live trace so the rail has motion when telemetry is present.
    let util_hist: Vec<u64> = state
        .history
        .iter()
        .map(|s| {
            s.gpus
                .iter()
                .map(|g| f64::from(g.gpu_utilization_pct))
                .fold(0.0, f64::max) as u64
        })
        .collect();
    let body = inner.height.saturating_sub(lines.len() as u16);
    f.render_widget(Paragraph::new(lines), inner);
    if body >= 1 && !util_hist.is_empty() {
        f.render_widget(
            BrailleSparkline::new(&util_hist)
                .max(100)
                .style(Style::default().fg(theme.accent))
                .gradient(theme.ok, theme.warn, theme.err),
            Rect::new(inner.x, inner.y + inner.height - 1, inner.width, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use rocm_dash_core::metrics::{GpuMetrics, Instance, Snapshot};

    fn flat(term: &Terminal<TestBackend>) -> String {
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn is_wide_and_triptych_extents() {
        assert!(!is_wide(160, 44));
        assert!(!is_wide(200, 40));
        assert!(is_wide(180, 45));
        let (l, c, r) = triptych(Rect::new(0, 0, 200, 50)).unwrap();
        assert_eq!(l.width, RAIL_W);
        assert_eq!(r.width, DOCK_W);
        assert_eq!(l.width + c.width + r.width, 200);
        // A narrow body yields no triptych.
        assert!(triptych(Rect::new(0, 0, 100, 50)).is_none());
    }

    #[test]
    fn logs_dock_shows_logs_and_no_composer() {
        let s = AppState::new("t".into(), "default-dark".into());
        let backend = TestBackend::new(DOCK_W, 20);
        let mut term = Terminal::new(backend).unwrap();
        let theme = s.theme;
        term.draw(|f| logs_dock(f, f.area(), &s, &theme)).unwrap();
        let out = flat(&term);
        assert!(out.contains("LOGS"), "logs title missing: {out:?}");
        // The dock must never be a chat composer.
        assert!(!out.contains("Message"), "composer leaked into dock");
        assert!(!out.contains("press i to type"), "composer prompt in dock");
    }

    #[test]
    fn logs_dock_scroll_windows_the_buffer() {
        use rocm_dash_core::state::StateEvent;
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.jobs.apply(StateEvent::StartJob {
            id: "logs".into(),
            cmd: "rocm".into(),
            args: vec!["logs".into()],
        });
        for i in 0..30 {
            s.jobs.apply(StateEvent::JobLine {
                id: "logs".into(),
                line: format!("LINE{i:02}"),
            });
        }
        let theme = s.theme;
        let render = |st: &AppState| {
            let backend = TestBackend::new(DOCK_W, 12);
            let mut term = Terminal::new(backend).unwrap();
            term.draw(|f| logs_dock(f, f.area(), st, &theme)).unwrap();
            flat(&term)
        };
        // Pinned to the tail: the newest line shows, an early line does not.
        let tail = render(&s);
        assert!(tail.contains("LINE29"), "tail shows newest: {tail:?}");
        assert!(!tail.contains("LINE00"), "tail hides oldest");
        // Scrolled all the way up: the oldest line comes into view, newest off.
        s.last_dock_area = Some(Rect::new(0, 0, DOCK_W, 12));
        s.scroll_dock(100);
        let up = render(&s);
        assert!(up.contains("LINE00"), "scroll-up reveals oldest: {up:?}");
        assert!(!up.contains("LINE29"), "newest scrolled off");
        assert!(up.contains('↑'), "title shows the scroll offset");
    }

    #[test]
    fn context_rail_shows_sections() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.instances.insert(
            "id".into(),
            Instance {
                container_id: "id".into(),
                model_name: "Qwen3-72B".into(),
                status: InstanceStatus::Running,
                port: Some(8000),
                ..Default::default()
            },
        );
        let backend = TestBackend::new(DOCK_W, 24);
        let mut term = Terminal::new(backend).unwrap();
        let theme = s.theme;
        term.draw(|f| context_rail(f, f.area(), &s, &theme))
            .unwrap();
        let out = flat(&term);
        assert!(
            out.contains("RUNNING SERVICES"),
            "missing services: {out:?}"
        );
        assert!(out.contains("GPU STATE"), "missing gpu state");
        assert!(out.contains("Qwen3-72B"), "running service not listed");
        assert!(!out.contains("Message"), "composer leaked into dock");
    }

    #[test]
    fn gpu_wall_renders_or_empty() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.latest = Some(Snapshot {
            gpus: vec![GpuMetrics {
                device_id: "GPU0".into(),
                gpu_utilization_pct: 50.0,
                vram_used_mb: 40_000,
                vram_total_mb: 192_000,
                ..Default::default()
            }],
            ..Default::default()
        });
        let backend = TestBackend::new(RAIL_W, 20);
        let mut term = Terminal::new(backend).unwrap();
        let theme = s.theme;
        term.draw(|f| gpu_wall(f, f.area(), &s, &theme)).unwrap();
        let out = flat(&term);
        assert!(out.contains("GPUs"), "gpu wall title missing: {out:?}");
    }
}
