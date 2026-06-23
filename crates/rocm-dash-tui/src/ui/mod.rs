// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

pub mod approval;
pub mod automations_manager;
pub mod bench;
pub mod command_screen;
pub mod config_manager;
pub mod core_bars;
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
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(f.area());

    state.last_tab_bar_area = Some(outer[1]);
    state.last_body_area = Some(outer[2]);

    draw_header(f, outer[0], state, &theme);
    tabs::draw_tab_bar(f, outer[1], state.active_tab, &theme);
    match state.active_tab {
        ActiveTab::Home => tabs::home::draw(f, outer[2], state, &theme),
        ActiveTab::Overview => tabs::overview::draw(f, outer[2], state, &theme),
        ActiveTab::Hardware => tabs::hardware::draw(f, outer[2], state, &theme),
        ActiveTab::Instances => tabs::instances::draw(f, outer[2], state, &theme),
        ActiveTab::Bench => tabs::bench::draw(f, outer[2], state, &theme),
        ActiveTab::Chat => tabs::chat::draw(f, outer[2], state, &theme),
    }
    draw_footer(f, outer[3], state, &theme);

    // Modal overlay (rendered last so it sits on top of the body).
    match state.modal {
        Modal::None => {}
        Modal::Help => modal::draw_help(f, outer[2], state.active_tab, &theme),
        Modal::Detail => match state.active_tab {
            ActiveTab::Instances => tabs::instances::draw_detail(f, outer[2], state, &theme),
            ActiveTab::Bench => tabs::bench::draw_detail(f, outer[2], state, &theme),
            ActiveTab::Hardware => tabs::hardware::draw_detail(f, outer[2], state, &theme),
            _ => {}
        },
        Modal::ThemePicker => modal::draw_theme_picker(
            f,
            outer[2],
            state.theme_picker_sel,
            &state.theme_name,
            &theme,
        ),
    }

    // Operational overlays (Phase 3 Wave 1): only one is open at a time. They
    // sit above the tab body + modals when open.
    if let Some(sm) = &state.services {
        services_manager::draw_services_manager(
            f,
            outer[2],
            sm,
            &state.instances,
            &state.jobs,
            &theme,
        );
    } else if let Some(w) = &state.serve_wizard {
        serve_wizard::draw_serve_wizard(f, outer[2], w, &state.jobs, &state.model_recipes, &theme);
    } else if let Some(em) = &state.engine_manager {
        engine_manager::draw_engine_manager(f, outer[2], em, &state.jobs, &theme);
    } else if let Some(d) = &state.examine_manager {
        examine_manager::draw_examine_manager(f, outer[2], d, &state.jobs, &theme);
    } else if let Some(u) = &state.update_manager {
        update_manager::draw_update_manager(f, outer[2], u, &state.jobs, &theme);
    } else if let Some(im) = &state.install_manager {
        install_manager::draw_install_manager(f, outer[2], im, &state.jobs, &theme);
    } else if let Some(lv) = &state.logs_view {
        logs_view::draw_logs_view(f, outer[2], lv, &state.jobs, &theme);
    } else if let Some(rm) = &state.runtime_manager {
        runtime_manager::draw_runtime_manager(
            f,
            outer[2],
            rm,
            &state.runtimes,
            &state.jobs,
            &theme,
        );
    } else if let Some(o) = &state.onboarding {
        onboarding::draw_onboarding(f, outer[2], o, &state.jobs, &theme);
    } else if let Some(am) = &state.automations_manager {
        automations_manager::draw_automations_manager(
            f,
            outer[2],
            am,
            &state.automations,
            &state.jobs,
            &theme,
        );
    } else if let Some(c) = &state.command_screen {
        command_screen::draw_command_screen(f, outer[2], c, &state.jobs, &theme);
    } else if let Some(cm) = &state.config_manager {
        config_manager::draw_config_manager(f, outer[2], cm, &state.jobs, &theme);
    }

    // Approval modal (Phase 4): drawn LAST so it sits on top of every overlay
    // and owns the screen while a mutating-tool approval is pending.
    if let Some(pa) = &state.approval {
        approval::draw_approval(f, outer[2], &pa.req, pa.choice, &theme);
    }
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
        chip("1–6"),
        Span::raw(" jump  "),
    ];
    if matches!(
        state.active_tab,
        ActiveTab::Hardware | ActiveTab::Instances | ActiveTab::Bench
    ) {
        spans.push(chip("j/k"));
        spans.push(Span::raw(" select  "));
        spans.push(chip("Enter"));
        spans.push(Span::raw(" detail  "));
    }
    // Operational-screen entry points (Phase 3 Wave 1).
    if matches!(state.active_tab, ActiveTab::Overview | ActiveTab::Instances) {
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
    if state.active_tab == ActiveTab::Instances {
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
