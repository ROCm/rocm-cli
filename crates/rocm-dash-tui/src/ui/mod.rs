// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

pub mod approval;
pub mod automations_manager;
pub mod bench;
pub mod bench_run;
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
pub mod panel;
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
use ratatui::widgets::{Block, Clear, Paragraph};

use crate::app::{ActiveTab, AppState, ConnState, FooterChip, KeyAction, Modal};
use crate::ui::theme::Theme;

pub fn draw(f: &mut Frame, state: &mut AppState) {
    let theme = state.theme;
    // Paint the whole frame with the theme background first, so every cell of
    // empty space matches the bg instead of showing the terminal default (which
    // read as a stray black box beneath the tabs).
    f.render_widget(
        Block::default().style(Style::default().bg(theme.bg)),
        f.area(),
    );
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
    // The right dock only hosts a scrollable LOGS stream on the operational tabs;
    // record its rect for wheel hit-testing (cleared otherwise).
    let mut dock_logs_rect: Option<Rect> = None;
    let (panel_outer, chip_origin_x) = if dock::is_wide(area.width, area.height) {
        if let Some((left, center_outer, right)) = wide_triptych(body) {
            dock::gpu_wall(f, left, state, &theme);
            dock::draw_right_dock(f, right, state, &theme);
            if matches!(
                state.active_tab,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) {
                dock_logs_rect = Some(right);
            }
            (center_outer, center_outer.x + 2)
        } else {
            (body, body.x + 2)
        }
    } else {
        (body, body.x + 2)
    };
    state.last_dock_area = dock_logs_rect;
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
        ActiveTab::Rocm => tabs::rocm::draw(f, center_inner, state, &theme),
        ActiveTab::Serving => tabs::serving::draw(f, center_inner, state, &theme),
        ActiveTab::Observe => tabs::observe::draw(f, center_inner, state, &theme),
        ActiveTab::Chat => tabs::chat::draw(f, center_inner, state, &theme),
    }
    let footer_chips = draw_footer(f, footer_area, state, &theme);
    state.last_footer_chips = footer_chips;

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

    // Operational managers render as a centered MODAL on every tab. The
    // ROCm/Serving Details bento keeps its summary + Start affordance; activating
    // Start opens the manager here, on top of the body, consistent regardless of
    // which tab is active (an inline manager became an orphaned floating box when
    // you switched tabs). Only one is open at a time (mutually exclusive via
    // `close_overlays`).
    // Dim the whole frame behind an open manager so the modal reads as the
    // foreground (and the body underneath is visibly inert).
    if state.has_open_overlay() {
        modal::grey_overlay(f);
    }
    let manager_rect = modal::centered_rect(82, 80, 130, 34, body);
    draw_active_manager(f, manager_rect, state, &theme);

    // Approval modal (Phase 4): drawn LAST so it sits on top of every overlay
    // and owns the screen while a mutating-tool approval is pending. This is the
    // sole approval renderer (the single gating path). It dims the backdrop too.
    if let Some(pa) = &state.approval {
        modal::grey_overlay(f);
        approval::draw_approval(f, body, &pa.req, pa.choice, &theme);
    }
}

/// Render a *focused-host* frame: theme background plus exactly ONE overlay.
///
/// A single hint line sits below it — no header, tab shell, dock, or footer
/// legend. Used by the bare-`rocm` launcher's in-place flows (Set up / Serve /
/// Diagnose), where the full dashboard chrome would be misleading. The overlay
/// is drawn through the same [`draw_active_manager`] path the dashboard uses (so
/// the approval / job-console layering is identical). Falls back to a centered
/// "closing…" note when no overlay is open — defensive; the event loop breaks at
/// that point and hands control back to the launcher.
pub fn draw_focused(f: &mut Frame, state: &mut AppState) {
    let theme = state.theme;
    f.render_widget(
        Block::default().style(Style::default().bg(theme.bg)),
        f.area(),
    );
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(f.area());
    let body = outer[0];
    let footer_area = outer[1];

    if state.has_open_overlay() {
        let manager_rect = modal::centered_rect(82, 80, 130, 34, body);
        draw_active_manager(f, manager_rect, state, &theme);
    } else {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "closing…",
                Style::default().fg(theme.muted),
            )))
            .alignment(ratatui::layout::Alignment::Center),
            body,
        );
    }

    // One honest hint: focused overlays return to the launcher menu (not the
    // dash tab shell — hence no "1–5" tab legend).
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Esc  back to menu",
            Style::default().fg(theme.muted),
        ))),
        footer_area,
    );
}

/// Draw whichever operational manager is open into `rect`. The managers are
/// mutually exclusive (only one `Some` at a time), so this draws at most one.
/// `rect` is the ROCm/Serving Details pane when inline, or a centered overlay
/// rect when a manager was opened from a non-domain tab. No-op when none open.
fn draw_active_manager(f: &mut Frame, rect: Rect, state: &AppState, theme: &Theme) {
    if !state.has_open_overlay() {
        return;
    }
    // Opaque modal card: clear whatever is behind, then back the whole card with
    // the theme bg. Without the `Clear`, the box only recolors the cells it
    // covers and body glyphs bleed through the gaps; the bg fill also makes the
    // ring around the (slightly inset) job console solid rather than see-through.
    f.render_widget(Clear, rect);
    f.render_widget(Block::default().style(Style::default().bg(theme.bg)), rect);

    // Console sub-view: whichever manager is streaming a job shows the shared
    // job console here, panned by the single `console_scroll`/`console_hscroll`
    // source. Centralized so the 13 managers don't each duplicate the branch.
    if let Some(job) = state.active_job_id().and_then(|id| state.jobs.job(id)) {
        job_console::draw_job_console(
            f,
            rect,
            job,
            (state.console_scroll, state.console_hscroll),
            theme,
        );
        return;
    }
    if let Some(sm) = &state.services {
        services_manager::draw_services_manager(f, rect, sm, &state.instances, &state.jobs, theme);
    } else if let Some(w) = &state.serve_wizard {
        serve_wizard::draw_serve_wizard(f, rect, w, &state.jobs, &state.model_recipes, theme);
    } else if let Some(em) = &state.engine_manager {
        engine_manager::draw_engine_manager(f, rect, em, &state.jobs, theme);
    } else if let Some(d) = &state.examine_manager {
        examine_manager::draw_examine_manager(f, rect, d, &state.jobs, theme);
    } else if let Some(u) = &state.update_manager {
        update_manager::draw_update_manager(f, rect, u, &state.jobs, theme);
    } else if let Some(im) = &state.install_manager {
        install_manager::draw_install_manager(f, rect, im, &state.jobs, theme);
    } else if let Some(lv) = &state.logs_view {
        logs_view::draw_logs_view(f, rect, lv, &state.jobs, theme);
    } else if let Some(rm) = &state.runtime_manager {
        runtime_manager::draw_runtime_manager(f, rect, rm, &state.runtimes, &state.jobs, theme);
    } else if let Some(o) = &state.onboarding {
        onboarding::draw_onboarding(f, rect, o, &state.jobs, theme);
    } else if let Some(am) = &state.automations_manager {
        automations_manager::draw_automations_manager(
            f,
            rect,
            am,
            &state.automations,
            &state.jobs,
            theme,
        );
    } else if let Some(c) = &state.command_screen {
        command_screen::draw_command_screen(f, rect, c, &state.jobs, theme);
    } else if let Some(cm) = &state.config_manager {
        config_manager::draw_config_manager(f, rect, cm, &state.jobs, theme);
    } else if let Some(br) = &state.bench_run {
        bench_run::draw_bench_run(f, rect, br, theme);
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
    let inner = panel::bento(f, area, None, panel::BoxRole::Neutral, false, theme);
    f.render_widget(Paragraph::new(vec![Line::from(spans)]), inner);
}

/// One footer-legend segment: a key chip (optionally clickable) or plain text.
enum Seg {
    /// A keycap. `Some(action)` => left-clicking it dispatches that action and
    /// the cap is highlighted to signal it is interactive; `None` => a display
    /// keycap (e.g. the `1–4` range) that has no single click target.
    Key(&'static str, Option<KeyAction>),
    /// A plain text separator / label.
    Sep(&'static str),
}

/// Draw the footer legend and return the clickable chip geometry for hit-testing.
fn draw_footer(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) -> Vec<FooterChip> {
    let is_action_tab = matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving);
    let enter_action = if is_action_tab {
        KeyAction::PaneActivate
    } else {
        KeyAction::OpenDetail
    };
    let mut segs: Vec<Seg> = vec![
        Seg::Key("Tab", Some(KeyAction::SwitchTab(state.active_tab.next()))),
        Seg::Sep(" next  "),
        Seg::Key("1–5", None),
        Seg::Sep(" jump  "),
    ];
    // On a domain tab with a manager open inline, the pane keys route to the
    // manager — advertise the back-out instead of the (now wrong) select/open.
    if is_action_tab && state.has_open_overlay() {
        segs.push(Seg::Key("Esc", None));
        segs.push(Seg::Sep(" back out  "));
    } else if matches!(
        state.active_tab,
        ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
    ) {
        segs.push(Seg::Key("j/k", Some(KeyAction::Move(1))));
        segs.push(Seg::Sep(" select  "));
        segs.push(Seg::Key("Enter", Some(enter_action)));
        segs.push(Seg::Sep(if is_action_tab {
            " open  "
        } else {
            " detail  "
        }));
    }
    // Guided-action letter hotkeys — Observe only (telemetry quick-jumps). On
    // ROCm/Serving the Actions list is the single path, so no letter chips.
    if state.active_tab == ActiveTab::Observe {
        segs.push(Seg::Key("w", Some(KeyAction::OpenServeWizard)));
        segs.push(Seg::Sep(" serve  "));
        segs.push(Seg::Key("e", Some(KeyAction::OpenEngineManager)));
        segs.push(Seg::Sep(" engines  "));
        segs.push(Seg::Key("d", Some(KeyAction::OpenExamine)));
        segs.push(Seg::Sep(" examine  "));
        segs.push(Seg::Key("u", Some(KeyAction::OpenUpdate)));
        segs.push(Seg::Sep(" update  "));
        segs.push(Seg::Key("i", Some(KeyAction::OpenInstall)));
        segs.push(Seg::Sep(" install  "));
        segs.push(Seg::Key("l", Some(KeyAction::OpenLogs)));
        segs.push(Seg::Sep(" logs  "));
        segs.push(Seg::Key("s", Some(KeyAction::OpenServices)));
        segs.push(Seg::Sep(" services  "));
        segs.push(Seg::Key("b", Some(KeyAction::OpenBenchRun)));
        segs.push(Seg::Sep(" bench  "));
    }
    if state.replay.is_some() {
        segs.push(Seg::Key("Space", Some(KeyAction::ReplayTogglePause)));
        segs.push(Seg::Sep(" pause  "));
        segs.push(Seg::Key("+/-", Some(KeyAction::ReplaySpeedUp)));
        segs.push(Seg::Sep(" speed  "));
    }
    segs.push(Seg::Key("t", Some(KeyAction::OpenThemePicker)));
    segs.push(Seg::Sep(" theme  "));
    segs.push(Seg::Key("?", Some(KeyAction::ToggleHelp)));
    segs.push(Seg::Sep(" help  "));
    segs.push(Seg::Key("q", Some(KeyAction::Quit)));
    segs.push(Seg::Sep(" quit"));

    // Lay out left-to-right, rendering each segment in its own cell span so the
    // recorded chip geometry matches the painted columns exactly.
    let mut cx = area.x;
    let row = area.y;
    let max_x = area.x.saturating_add(area.width);
    let mut chips: Vec<FooterChip> = Vec::new();
    // Keycap surface tracks the theme's luminance (a small nudge off the bg)
    // rather than surface_2 (= br_black), which collides with text on light
    // themes. Clickable caps add UNDERLINE as a non-color affordance so
    // interactivity does not rely on hue alone.
    let cap_bg = panel::blend(theme.bg, theme.fg, 0.12);
    for seg in &segs {
        if cx >= max_x {
            break;
        }
        let (text, key, action): (&str, bool, Option<KeyAction>) = match seg {
            Seg::Key(cap, act) => (cap, true, *act),
            Seg::Sep(sep) => (sep, false, None),
        };
        let body = if key {
            format!(" {text} ")
        } else {
            text.to_string()
        };
        let width = body.chars().count() as u16;
        let draw_w = width.min(max_x - cx);
        let style = match (key, action.is_some()) {
            // Clickable keycap: accent + bold + underline (color-independent cue).
            (true, true) => Style::default()
                .bg(cap_bg)
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            // Display-only keycap (e.g. 1–4): same surface, muted, no underline.
            (true, false) => Style::default().bg(cap_bg).fg(theme.muted),
            // Separator / label.
            _ => Style::default().fg(theme.muted),
        };
        f.render_widget(
            Paragraph::new(Span::styled(body, style)),
            Rect::new(cx, row, draw_w, 1),
        );
        if let Some(act) = action {
            chips.push(FooterChip {
                x0: cx,
                x1: cx + width,
                y: row,
                action: act,
            });
        }
        cx = cx.saturating_add(width);
    }
    chips
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
