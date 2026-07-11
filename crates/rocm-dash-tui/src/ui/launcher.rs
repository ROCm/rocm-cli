// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Minimal launcher — the bare-`rocm` front door.
//!
//! A small pre-dash screen: a live status strip (GPU + serving) over an icon
//! menu of the headline verbs. Escalating ("Open full dashboard" / `d`) falls
//! through to the existing dash run loop (ponytail: the launcher is a thin
//! pre-screen, not a parallel event loop). Per the latest mocks there is no
//! image-generation row; "Optimize a model" is display-only with a `soon` badge.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

#[cfg(test)]
use rocm_dash_core::metrics::InstanceStatus;

use crate::app::AppState;
use crate::ui::format;
use crate::ui::theme::Theme;

/// Where a selected launcher row leads. The runtime maps these to the existing
/// dash entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LauncherChoice {
    Serve,
    SetUp,
    Diagnose,
    Chat,
    OpenDashboard,
}

/// The selectable rows, in display order (icon, label, desc, choice). The
/// display-only "Optimize a model" (soon) row is rendered separately and is not
/// part of this list.
pub const ROWS: &[(&str, &str, &str, LauncherChoice)] = &[
    (
        "⚙",
        "Set up this system",
        "install / update ROCm",
        LauncherChoice::SetUp,
    ),
    (
        "◆",
        "Serve a model",
        "run a model on your GPU",
        LauncherChoice::Serve,
    ),
    (
        "⚕",
        "Diagnose & fix",
        "check GPU, driver & ROCm",
        LauncherChoice::Diagnose,
    ),
    (
        "◷",
        "Chat",
        "talk to a local or API model",
        LauncherChoice::Chat,
    ),
    (
        "▣",
        "Open full dashboard  →",
        "live instruments & every action",
        LauncherChoice::OpenDashboard,
    ),
];

/// Number of selectable rows.
#[must_use]
pub const fn row_count() -> usize {
    ROWS.len()
}

/// Resolve the choice for cursor `sel` (clamped).
#[must_use]
pub fn choice_for(sel: usize) -> LauncherChoice {
    ROWS.get(sel)
        .map_or(LauncherChoice::OpenDashboard, |(_, _, _, c)| *c)
}

/// True when a model is actively serving (drives the running vs idle variant).
fn is_running(state: &AppState) -> bool {
    state.instances.values().any(|i| i.status.is_serving())
}

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, sel: usize, theme: &Theme) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // title
            Constraint::Length(3), // status strip
            Constraint::Length(1), // spacer
            Constraint::Length(1), // prompt
            Constraint::Min(0),    // menu
        ])
        .margin(1)
        .split(area);

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "rocm.ai",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  local AI control room", Style::default().fg(theme.muted)),
        ])),
        rows[0],
    );

    draw_status_strip(f, rows[1], state, theme);

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "What would you like to do?",
            Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
        ))),
        rows[3],
    );

    draw_menu(f, rows[4], state, sel, theme);
}

fn draw_status_strip(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    if area.height == 0 {
        return;
    }
    let running = is_running(state);
    let gpu_line = state.latest.as_ref().map_or_else(
        || {
            Line::from(Span::styled(
                "GPU  —  no live telemetry",
                Style::default().fg(theme.muted),
            ))
        },
        |s| {
            let util = s
                .gpus
                .iter()
                .map(|g| f64::from(g.gpu_utilization_pct))
                .fold(0.0, f64::max);
            let model = s
                .gpu_system_info
                .as_ref()
                .map_or("GPU", |g| g.gpu_model.as_str());
            Line::from(vec![
                Span::styled("GPU ", Style::default().fg(theme.muted)),
                Span::styled(model.to_string(), Style::default().fg(theme.fg)),
                Span::styled("  │  ", Style::default().fg(theme.border)),
                Span::styled(
                    format!("Util {}", format::pct(util as f32)),
                    Style::default().fg(theme.ok),
                ),
            ])
        },
    );
    let serve_line = if running {
        let inst = state.instances.values().find(|i| i.status.is_serving());
        let (name, port) = inst.map_or(("model", String::new()), |i| {
            (
                i.model_name.as_str(),
                i.port.map_or_else(String::new, |p| format!(" on :{p}")),
            )
        });
        Line::from(vec![
            Span::styled("● ", Style::default().fg(theme.ok)),
            Span::styled("Serving ", Style::default().fg(theme.muted)),
            Span::styled(
                name.to_string(),
                Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
            ),
            Span::styled(port, Style::default().fg(theme.muted)),
            Span::styled("  │  ", Style::default().fg(theme.border)),
            Span::styled("✓ healthy", Style::default().fg(theme.ok)),
        ])
    } else {
        Line::from(vec![
            Span::styled("○ ", Style::default().fg(theme.muted)),
            Span::styled("Idle — nothing serving", Style::default().fg(theme.muted)),
        ])
    };
    f.render_widget(Paragraph::new(vec![gpu_line, serve_line]), area);
}

fn draw_menu(f: &mut Frame, area: Rect, state: &AppState, sel: usize, theme: &Theme) {
    let running = is_running(state);
    let mut lines: Vec<Line> = Vec::new();
    for (i, (icon, label, desc, choice)) in ROWS.iter().enumerate() {
        // Idle greys the "Chat" row when nothing is serving and no API model is
        // configured — but keep it selectable; just dim the descriptor.
        let dim_desc = !running && *choice == LauncherChoice::Chat;
        let focused = i == sel;
        let (cur, lstyle) = if focused {
            (
                "▸ ",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            ("  ", Style::default().fg(theme.fg))
        };
        lines.push(Line::from(vec![
            Span::styled(cur, Style::default().fg(theme.accent)),
            Span::styled(format!("{icon}  "), Style::default().fg(theme.accent_2)),
            Span::styled(*label, lstyle),
            Span::styled(
                format!("   {desc}"),
                Style::default().fg(if dim_desc { theme.border } else { theme.muted }),
            ),
        ]));
    }
    // Display-only "Optimize a model" row with a soon badge.
    lines.push(Line::from(vec![
        Span::styled("  ⚡  ", Style::default().fg(theme.muted)),
        Span::styled("Optimize a model", Style::default().fg(theme.muted)),
        Span::raw("  "),
        Span::styled(
            " soon ",
            Style::default()
                .bg(theme.warn)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "↑↓ move   Enter select   d dashboard   q quit",
        Style::default().fg(theme.muted),
    )));
    f.render_widget(Paragraph::new(lines), area);
}

/// Run the launcher as a synchronous pre-dash screen.
///
/// Returns the chosen destination, or `None` when the user quits. On success the
/// caller escalates into the existing dash entry points (ponytail: no parallel
/// async loop here).
///
/// Renders the idle variant from a fresh `AppState` (no live daemon is started
/// just for the front door); live telemetry appears once the dash is opened.
///
/// # Errors
/// Propagates terminal setup / event-read I/O errors.
pub fn run_launcher(theme_name: &str) -> std::io::Result<Option<LauncherChoice>> {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind};
    use crossterm::terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    };
    use ratatui::Terminal;
    use ratatui::backend::CrosstermBackend;

    let state = AppState::new(String::new(), theme_name.to_string());
    let theme = state.theme;

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut sel = 0usize;
    let result = loop {
        terminal.draw(|f| draw(f, f.area(), &state, sel, &theme))?;
        let Event::Key(k) = event::read()? else {
            continue;
        };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        match k.code {
            KeyCode::Char('q') | KeyCode::Esc => break None,
            KeyCode::Char('d') => break Some(LauncherChoice::OpenDashboard),
            KeyCode::Enter => break Some(choice_for(sel)),
            KeyCode::Down | KeyCode::Char('j') => sel = (sel + 1) % row_count(),
            KeyCode::Up | KeyCode::Char('k') => {
                sel = (sel + row_count() - 1) % row_count();
            }
            _ => {}
        }
    };

    disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use rocm_dash_core::metrics::{GpuMetrics, GpuSystemInfo, Instance, Snapshot};

    fn render(state: &AppState, sel: usize, cols: u16, rows: u16) -> String {
        let backend = TestBackend::new(cols, rows);
        let mut term = Terminal::new(backend).unwrap();
        let theme = state.theme;
        term.draw(|f| draw(f, f.area(), state, sel, &theme))
            .unwrap();
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    fn base() -> AppState {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.latest = Some(Snapshot {
            gpus: vec![GpuMetrics {
                device_id: "GPU0".into(),
                gpu_utilization_pct: 34.0,
                ..Default::default()
            }],
            gpu_system_info: Some(GpuSystemInfo {
                gpu_model: "Radeon 8060S".into(),
                ..Default::default()
            }),
            ..Default::default()
        });
        s
    }

    #[test]
    fn launcher_running_shows_status_strip_and_rows() {
        let mut s = base();
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
        let out = render(&s, 0, 100, 24);
        assert!(out.contains("GPU"), "status strip GPU missing: {out:?}");
        assert!(out.contains("Serving"), "running serve line missing");
        assert!(out.contains("Qwen3-72B"), "served model missing");
        // ≥4 menu rows present.
        for label in [
            "Serve a model",
            "Set up this system",
            "Diagnose & fix",
            "Chat",
        ] {
            assert!(out.contains(label), "menu row {label:?} missing");
        }
        assert!(out.contains("Open full dashboard"), "dashboard row missing");
        assert!(out.contains("soon"), "optimize soon badge missing");
        let needle = ["generate", "an", "image"].join(" ");
        assert!(!out.to_lowercase().contains(&needle), "no image verb");
    }

    #[test]
    fn launcher_idle_variant_renders() {
        let out = render(&base(), 0, 100, 24);
        assert!(out.contains("Idle"), "idle status line missing: {out:?}");
        assert!(out.contains("Serve a model"), "menu missing in idle");
    }

    #[test]
    fn choice_mapping_and_counts() {
        assert_eq!(row_count(), 5);
        // Row order: 0=Set up, 1=Serve, 2=Diagnose, 3=Chat, 4=Open dashboard.
        assert_eq!(choice_for(0), LauncherChoice::SetUp);
        assert_eq!(choice_for(1), LauncherChoice::Serve);
        assert_eq!(choice_for(4), LauncherChoice::OpenDashboard);
        assert_eq!(choice_for(99), LauncherChoice::OpenDashboard);
        // The first two rows carry the expected labels in the new order.
        assert_eq!(ROWS[0].1, "Set up this system");
        assert_eq!(ROWS[0].3, LauncherChoice::SetUp);
        assert_eq!(ROWS[1].1, "Serve a model");
        assert_eq!(ROWS[1].3, LauncherChoice::Serve);
    }

    #[test]
    fn launcher_does_not_panic_when_squeezed() {
        let s = base();
        for h in [1u16, 2, 3, 5, 8] {
            let _ = render(&s, 0, 60, h);
        }
    }
}
