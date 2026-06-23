// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

pub mod approval;
pub mod automations_manager;
pub mod bench;
pub mod command_screen;
pub mod config_manager;
pub mod core_bars;
pub mod dock;
pub mod engine_manager;
pub mod examine_manager;
pub mod exec;
pub mod folder_browser;
pub mod format;
pub mod gradient;
pub mod heatmap;
pub mod install_manager;
pub mod instance_list;
pub mod job_console;
pub mod launcher;
pub mod logs_view;
pub mod modal;
pub mod model_picker;
pub mod onboarding;
pub mod runtime_manager;
pub mod serve_wizard;
pub mod services_manager;
pub mod sparkline;
pub mod tabs;
pub mod theme;
pub mod update_manager;
pub mod widgets;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::app::{ActiveTab, AppState, ConnState, Modal};
use crate::ui::theme::Theme;

pub fn draw(f: &mut Frame, state: &mut AppState) {
    let theme = state.theme;
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(f.area());
    let body = outer[1];
    let footer_area = outer[2];

    draw_header(f, outer[0], state, &theme);

    // The body is framed by the outlined folder-tab panel (the live chrome). On
    // a wide screen the panel wraps only the CENTER column so the tabs read as
    // belonging to it, with the GPU wall / dock rails beside it; below the
    // threshold the panel spans the full width (single-column fallback).
    let active_idx = tabs::active_index(state.active_tab);
    let labels = tabs::tab_labels();
    let area = f.area();
    let (panel_outer, chip_origin_x) = if dock::is_wide(area.width, area.height) {
        if let Some((left, center_outer, right)) = wide_triptych(body) {
            dock::gpu_wall(f, left, state, &theme);
            dock::draw_right_dock(f, right, state, &theme);
            (center_outer, center_outer.x + 2)
        } else {
            (body, body.x + 2)
        }
    } else {
        (body, body.x + 2)
    };
    let center_inner = tabs::draw_tab_panel(f, panel_outer, &labels, active_idx, &theme);

    // Tab hit-testing: the clickable folder row is the panel's label row, with
    // chips starting at `chip_origin_x` (shared geometry with the renderer).
    state.last_tab_bar_area = Some(Rect::new(
        chip_origin_x,
        panel_outer.y + 1,
        panel_outer.width,
        1,
    ));
    state.last_body_area = Some(center_inner);

    match state.active_tab {
        ActiveTab::Home => tabs::home::draw(f, center_inner, state, &theme),
        ActiveTab::Action => tabs::action::draw(f, center_inner, state, &theme),
        ActiveTab::Observe => tabs::observe::draw(f, center_inner, state, &theme),
        ActiveTab::Chat => tabs::chat::draw(f, center_inner, state, &theme),
    }
    draw_footer(f, footer_area, state, &theme);

    // Modal overlay (rendered last so it sits on top of the body).
    match state.modal {
        Modal::None => {}
        Modal::Help => modal::draw_help(f, body, state.active_tab, &theme),
        // Observe folds the telemetry tabs; its detail modal is the instance
        // detail (the selectable list on that surface).
        Modal::Detail => {
            if state.active_tab == ActiveTab::Observe {
                tabs::instances::draw_detail(f, body, state, &theme);
            }
        }
        Modal::ThemePicker => {
            modal::draw_theme_picker(f, body, state.theme_picker_sel, &state.theme_name, &theme);
        }
        Modal::Menu => modal::draw_menu(f, body, state.menu_sel, &theme),
        Modal::Palette => modal::draw_palette(f, body, state.palette_sel, &theme),
        Modal::Options => modal::draw_options(f, body, state, &theme),
        Modal::GlobalHelp => modal::draw_global_help(f, body, &theme),
    }

    // Operational overlays (Phase 3 Wave 1): only one is open at a time. They
    // sit above the tab body + modals when open.
    if let Some(sm) = &state.services {
        services_manager::draw_services_manager(f, body, sm, &state.instances, &state.jobs, &theme);
    } else if let Some(w) = &state.serve_wizard {
        serve_wizard::draw_serve_wizard(f, body, w, &state.jobs, &state.model_recipes, &theme);
    } else if let Some(em) = &state.engine_manager {
        engine_manager::draw_engine_manager(f, body, em, &state.jobs, &theme);
    } else if let Some(d) = &state.examine_manager {
        examine_manager::draw_examine_manager(f, body, d, &state.jobs, &theme);
    } else if let Some(u) = &state.update_manager {
        update_manager::draw_update_manager(f, body, u, &state.jobs, &theme);
    } else if let Some(im) = &state.install_manager {
        install_manager::draw_install_manager(f, body, im, &state.jobs, &theme);
    } else if let Some(lv) = &state.logs_view {
        logs_view::draw_logs_view(f, body, lv, &state.jobs, &theme);
    } else if let Some(rm) = &state.runtime_manager {
        runtime_manager::draw_runtime_manager(f, body, rm, &state.runtimes, &state.jobs, &theme);
    } else if let Some(o) = &state.onboarding {
        onboarding::draw_onboarding(f, body, o, &state.jobs, &theme);
    } else if let Some(am) = &state.automations_manager {
        automations_manager::draw_automations_manager(
            f,
            body,
            am,
            &state.automations,
            &state.jobs,
            &theme,
        );
    } else if let Some(c) = &state.command_screen {
        command_screen::draw_command_screen(f, body, c, &state.jobs, &theme);
    } else if let Some(cm) = &state.config_manager {
        config_manager::draw_config_manager(f, body, cm, &state.jobs, &theme);
    }

    // Approval modal (Phase 4): drawn LAST so it sits on top of every overlay
    // and owns the screen while a mutating-tool approval is pending.
    if let Some(pa) = &state.approval {
        approval::draw_approval(f, body, &pa.req, pa.choice, &theme);
    }
}

/// Wide-layout geometry: GPU wall (left) / outlined-tab center panel / dock
/// (right). The rails align with the center *content* panel — they start 2 rows
/// below the tab tops — so the tab band reads as belonging to the center column.
/// Returns `None` when the body is too narrow for both rails plus a usable
/// center (single-column fallback).
const fn wide_triptych(body: Rect) -> Option<(Rect, Rect, Rect)> {
    let lw = dock::RAIL_W;
    let rw = dock::DOCK_W;
    let min_center = 44u16;
    if body.width < lw + rw + min_center || body.height < 5 {
        return None;
    }
    let rail_y = body.y + 2;
    let rail_h = body.height - 2;
    let left = Rect::new(body.x, rail_y, lw, rail_h);
    let right = Rect::new(body.x + body.width - rw, rail_y, rw, rail_h);
    let center_x = body.x + lw + 1;
    let center_w = right.x - center_x - 1;
    let center_outer = Rect::new(center_x, body.y, center_w, body.height);
    Some((left, center_outer, right))
}

fn draw_header(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let (status_text, status_color) = match &state.conn {
        ConnState::Initial => ("starting".to_string(), theme.muted),
        ConnState::Connecting => ("connecting…".to_string(), theme.warn),
        ConnState::Connected { host, version } => (
            format!("connected · {host} · rocm daemon {version}"),
            theme.ok,
        ),
        ConnState::Disconnected { reason } => (format!("disconnected · {reason}"), theme.err),
    };

    let mut spans: Vec<Span> = vec![
        Span::styled(
            "rocm.ai",
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(theme.accent),
        ),
        Span::raw("  "),
        Span::styled(
            format!("→ {}", state.connect),
            Style::default().fg(theme.muted),
        ),
        Span::raw("   "),
        Span::styled(status_text, Style::default().fg(status_color)),
    ];
    let warning_count = state.latest.as_ref().map_or(0, |s| s.warnings.len());
    if warning_count > 0 {
        spans.push(Span::raw("   "));
        spans.push(Span::styled(
            format!(" ⚠ {warning_count} "),
            Style::default()
                .bg(theme.warn)
                .fg(theme.surface_2)
                .add_modifier(Modifier::BOLD),
        ));
    }
    if let Some(r) = state.replay.as_ref() {
        spans.push(Span::raw("   "));
        let (icon, fg) = if r.paused {
            ("⏸", theme.warn)
        } else {
            ("▶", theme.ok)
        };
        spans.push(Span::styled(
            format!(" {icon} {:.2}× ", r.speed),
            Style::default()
                .bg(theme.surface_2)
                .fg(fg)
                .add_modifier(Modifier::BOLD),
        ));
        if r.total_s > 0 {
            spans.push(Span::raw(" "));
            spans.push(Span::styled(
                format!(
                    "{} / {}",
                    crate::app::format_mmss(r.elapsed_s),
                    crate::app::format_mmss(r.total_s)
                ),
                Style::default().fg(theme.muted),
            ));
        }
    }
    spans.push(Span::raw("   "));
    spans.push(Span::styled(
        format!("theme: {}", state.theme_name),
        Style::default().fg(theme.muted),
    ));
    // Headline chrome hint (per the mocks): Esc menu · t theme · ? help.
    spans.push(Span::raw("   "));
    spans.push(Span::styled(
        "Esc menu · t theme · ? help",
        Style::default().fg(theme.muted),
    ));
    let header = Paragraph::new(vec![Line::from(spans)]).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style()),
    );
    f.render_widget(header, area);
}

fn draw_footer(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let chip = |k: &str| {
        Span::styled(
            format!(" {k} "),
            Style::default().bg(theme.surface_2).fg(theme.fg),
        )
    };
    let mut spans: Vec<Span> = vec![
        chip("Tab"),
        Span::raw(" next  "),
        chip("1–4"),
        Span::raw(" jump  "),
    ];
    if matches!(state.active_tab, ActiveTab::Observe | ActiveTab::Action) {
        spans.push(chip("j/k"));
        spans.push(Span::raw(" select  "));
        spans.push(chip("Enter"));
        spans.push(Span::raw(if state.active_tab == ActiveTab::Action {
            " open  "
        } else {
            " detail  "
        }));
    }
    // Guided-action entry points — Observe (telemetry) + Action (verb list).
    if matches!(state.active_tab, ActiveTab::Observe | ActiveTab::Action) {
        spans.push(chip("w"));
        spans.push(Span::raw(" serve  "));
        spans.push(chip("e"));
        spans.push(Span::raw(" engines  "));
        spans.push(chip("d"));
        spans.push(Span::raw(" examine  "));
        spans.push(chip("u"));
        spans.push(Span::raw(" update  "));
        spans.push(chip("i"));
        spans.push(Span::raw(" install  "));
        spans.push(chip("l"));
        spans.push(Span::raw(" logs  "));
    }
    if state.active_tab == ActiveTab::Observe {
        spans.push(chip("s"));
        spans.push(Span::raw(" services  "));
    }
    if state.replay.is_some() {
        spans.push(chip("Space"));
        spans.push(Span::raw(" pause  "));
        spans.push(chip("+/-"));
        spans.push(Span::raw(" speed  "));
    }
    spans.push(chip("t"));
    spans.push(Span::raw(" theme  "));
    spans.push(chip("?"));
    spans.push(Span::raw(" help  "));
    spans.push(chip("q"));
    spans.push(Span::raw(" quit"));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_triptych_aligns_tabs_with_center_not_far_left() {
        // A wide body: tabs (over center_outer) must start to the RIGHT of the
        // GPU wall, left-aligned with the center column — not at the far left.
        let body = Rect::new(0, 3, 200, 47);
        let (left, center_outer, right) = wide_triptych(body).expect("wide body splits");
        assert_eq!(left.x, 0, "GPU wall hugs the left edge");
        assert_eq!(left.width, dock::RAIL_W);
        // Center (and thus the outlined tabs) begins past the left rail.
        assert!(
            center_outer.x >= dock::RAIL_W,
            "tabs must align with center, got x={}",
            center_outer.x
        );
        assert!(center_outer.x < right.x, "center sits between the rails");
        // Rails align with the center content panel (2 rows below the tab tops).
        assert_eq!(left.y, body.y + 2);
        assert_eq!(center_outer.y, body.y);
    }

    #[test]
    fn narrow_body_has_no_triptych() {
        assert!(wide_triptych(Rect::new(0, 0, 100, 40)).is_none());
    }
}
