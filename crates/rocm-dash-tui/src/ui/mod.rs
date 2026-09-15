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
pub mod spinner;
pub mod tabs;
pub mod theme;
pub mod update_manager;
pub mod widgets;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph};

use crate::app::{
    ActiveTab, AppState, ChatConsent, ConnState, FooterChip, KeyAction, Modal, PaneFocus,
};
use crate::ui::theme::Theme;

pub fn draw(f: &mut Frame, state: &mut AppState) {
    let theme = state.theme;
    // Scrollbar hit-test registry is rebuilt every frame from what's actually
    // drawn, so mouse clicks resolve against the current layout.
    state.scrollbars.borrow_mut().clear();
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
        Modal::Help => {
            state.help_max_scroll =
                modal::draw_help(f, body, state.active_tab, &theme, state.help_scroll);
        }
        // Observe folds the telemetry tabs; its detail modal is the instance
        // detail (the selectable list on that surface).
        Modal::Detail => {
            if state.active_tab == ActiveTab::Observe {
                let max_scroll = tabs::instances::draw_detail(f, body, state, &theme);
                state.instance_detail_max_scroll = max_scroll;
            }
        }
        Modal::ThemePicker => {
            modal::draw_theme_picker(f, body, state.theme_picker_sel, &state.theme_name, &theme);
        }
        Modal::Menu => modal::draw_menu(f, body, state.menu_sel, &theme),
        Modal::Palette => modal::draw_palette(f, body, state.palette_sel, &theme),
        Modal::Options => modal::draw_options(f, body, state, &theme),
        Modal::GlobalHelp => {
            state.help_max_scroll = modal::draw_global_help(f, body, &theme, state.help_scroll);
        }
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
/// (and any job console nested inside it) is drawn through the same
/// [`draw_active_manager`] path [`draw`] uses, and the same dimmed-backdrop
/// wash is applied behind it. It is NOT identical to [`draw`] in two ways:
/// there is no approval layer here (a focused-host session has no chat, so no
/// tool call can ever be pending), and the "Esc back to menu" hint below is
/// rendered with a foreground-only `Style` (no `bg`) — `ratatui::Style::patch`
/// leaves an unset field untouched rather than clearing it, so the hint
/// inherits the wash's background from the cells underneath it rather than
/// reverting to plain theme bg, exactly like `draw`'s footer. Falls back to a
/// centered "closing…" note when no overlay is open — defensive; the event
/// loop breaks at that point and hands control back to the launcher.
pub fn draw_focused(f: &mut Frame, state: &mut AppState) {
    let theme = state.theme;
    state.scrollbars.borrow_mut().clear();
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
        // Dim the periphery behind the modal, matching `draw()`'s treatment so
        // the overlay reads as the foreground here too (previously this path
        // skipped the wash entirely, leaving the area outside the manager card
        // at plain theme background instead of dimmed).
        modal::grey_overlay(f);
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
        let handles = job_console::draw_job_console(
            f,
            rect,
            job,
            (state.console_scroll, state.console_hscroll),
            state.tick_count,
            theme,
        );
        state.scrollbars.borrow_mut().extend(handles);
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
    // In demo/replay the data is not live, so never present the session as
    // "connected" to a real daemon — the Connected case shows a simulated label
    // instead, and the SIMULATED DATA chip below makes the state unmistakable.
    // Other states (e.g. Disconnected on end-of-recording or a failed replay
    // file) still surface their reason so playback status is not masked.
    let (status_text, status_color) = match &state.conn {
        ConnState::Initial => ("starting".to_string(), theme.muted),
        ConnState::Connecting => ("connecting…".to_string(), theme.warn),
        ConnState::Connected { .. } if state.simulated => {
            ("simulated — not live".to_string(), theme.warn)
        }
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
    ];
    if state.simulated {
        spans.push(Span::styled(
            " SIMULATED DATA ",
            Style::default()
                .bg(theme.warn)
                .fg(theme.surface_2)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw("  "));
    }
    spans.push(Span::styled(
        format!("→ {}", state.connect),
        Style::default().fg(theme.muted),
    ));
    spans.push(Span::raw("   "));
    spans.push(Span::styled(status_text, Style::default().fg(status_color)));
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
    // Exactly one Esc chip is shown at all times, and it must match what Esc
    // actually does — the real routing priority (highest first) is: a pending
    // chat approval owns every key; then an open manager overlay backs itself
    // out; then a `Modal::*` overlay closes; then a focused/gating Chat tab
    // absorbs Esc; only once none of those apply does Esc fall through to the
    // uniform "menu" fallback (item #35). Mirror that order here so the chip
    // never advertises `menu` while a click on it would actually do something
    // else.
    if state.approval.is_some() {
        segs.push(Seg::Key("Esc", None));
        segs.push(Seg::Sep(" cancel  "));
    } else if state.has_open_overlay() && state.active_overlay_at_root() {
        segs.push(Seg::Key("Esc", None));
        segs.push(Seg::Sep(" back out  "));
    } else if state.has_open_overlay()
        && state
            .active_job_id()
            .is_some_and(|id| crate::ui::job_console::console_esc_closes(state.jobs.job(id)))
    {
        // A manager's job console is showing a still-running job — Esc fully
        // closes the overlay there (the job keeps running in the background),
        // matching the console's own footer hint ("Esc close (keeps
        // running)"), not the generic sub-popup "cancel" below. Once the job
        // finishes, `on_console_key` only dismisses the console back to the
        // screen body (the overlay stays open), so that case falls through to
        // the "cancel" arm below, which already describes it correctly. Shares
        // `console_esc_closes` with `on_console_key` so the two can't drift.
        segs.push(Seg::Key("Esc", None));
        segs.push(Seg::Sep(" close  "));
    } else if state.has_open_overlay() {
        // A manager is open but not at its root layer (sub-popup, picker, or
        // approval) — Esc is handled by that layer's own event-loop arm, not
        // by `should_pane_back_out`/`OpenMenu`. `None` keeps the chip
        // non-clickable so it can't dispatch the wrong action.
        segs.push(Seg::Key("Esc", None));
        segs.push(Seg::Sep(" cancel  "));
    } else if state.modal != Modal::None {
        segs.push(Seg::Key("Esc", Some(KeyAction::CloseModal)));
        segs.push(Seg::Sep(" close  "));
    } else if state.active_tab == ActiveTab::Chat
        && state.chat_detect_offer.is_some()
        && state.chat_consent != ChatConsent::Accepted
    {
        segs.push(Seg::Key("Esc", Some(KeyAction::ChatDetectDismiss)));
        segs.push(Seg::Sep(" dismiss  "));
    } else if state.active_tab == ActiveTab::Chat && state.chat_focused {
        segs.push(Seg::Key("Esc", Some(KeyAction::ChatBlur)));
        segs.push(Seg::Sep(" unfocus  "));
    } else {
        // Dispatch `PaneEscape`, not a hardcoded `OpenMenu` — `apply_action`
        // resolves `PaneEscape` against `pane_focus` exactly as a real
        // keypress does (Details → Actions on Rocm/Serving, else the menu),
        // so the chip can't promise "menu" when the key would actually just
        // step the pane back out.
        let steps_out_of_detail = is_action_tab && state.pane_focus == PaneFocus::Detail;
        segs.push(Seg::Key("Esc", Some(KeyAction::PaneEscape)));
        segs.push(Seg::Sep(if steps_out_of_detail {
            " back  "
        } else {
            " menu  "
        }));
        if matches!(
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
    if state.has_open_overlay() {
        // While a manager overlay is open it owns every key (the event loop
        // routes each keypress to its `on_key`, never falling through to
        // `apply_action`), so a real `q` press can't reach `KeyAction::Quit`
        // there — it cancels the approval, closes the job console, or backs
        // the manager out instead, but it never tears down the app or kills
        // a job the way `Quit` does. `None` keeps the chip non-clickable so a
        // click can't do something the key never would.
        segs.push(Seg::Key("q", None));
        segs.push(Seg::Sep(" close"));
    } else {
        segs.push(Seg::Key("q", Some(KeyAction::Quit)));
        segs.push(Seg::Sep(" quit"));
    }

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

    #[test]
    fn footer_shows_esc_menu_chip_when_no_overlay_open() {
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let state = AppState::new("t".into(), "default-dark".into());
        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        // The chip dispatches `PaneEscape`, matching what a real Esc keypress
        // resolves to via `handle_key`'s catch-all — not a hardcoded
        // `OpenMenu` that would diverge from `apply_action`'s `pane_focus`
        // handling on Rocm/Serving.
        let _ = chips
            .iter()
            .find(|c| c.action == KeyAction::PaneEscape)
            .expect("a fallback Esc chip stepping the pane back out must always be present");
    }

    #[test]
    fn footer_esc_chip_dispatches_pane_escape_when_detail_focused() {
        // Regression: on Rocm/Serving with `pane_focus == Detail`, the real
        // Esc key steps Details → Actions first (`PaneEscape` in
        // `apply_action`); it does not open the menu. The chip must dispatch
        // the same `PaneEscape` action (not `OpenMenu`) so a click matches
        // the keypress, and its label must say so.
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.active_tab = ActiveTab::Rocm;
        state.pane_focus = PaneFocus::Detail;

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        assert!(
            chips.iter().any(|c| c.action == KeyAction::PaneEscape),
            "Esc chip must dispatch PaneEscape, matching the real key, while Detail is focused"
        );
        assert!(
            !chips.iter().any(|c| c.action == KeyAction::OpenMenu),
            "no chip may claim OpenMenu while Esc would actually step Detail back to Actions"
        );
        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("back"),
            "chip label should say the Esc key steps back out of Detail: {row:?}"
        );
    }

    #[test]
    fn footer_esc_chip_is_not_clickable_menu_when_overlay_has_a_sub_popup_open() {
        // Regression: with a manager open but not at its root layer (here, a
        // folder browser sub-popup), `has_open_overlay()` is true but
        // `active_overlay_at_root()` is false. The chip must not fall through
        // to the generic `OpenMenu` arm — that key is actually consumed by the
        // manager's own event-loop arm, which cancels the sub-layer, not the
        // menu. Any chip shown here must be non-clickable (`action == None`)
        // and labeled "cancel" (this sub-popup has no "close means job keeps
        // running" nuance, unlike a job console — see the "close" test below).
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState {
            browser: Some(crate::ui::folder_browser::FolderBrowser::new(
                "t",
                std::env::temp_dir(),
            )),
            ..Default::default()
        });
        assert!(state.has_open_overlay());
        assert!(!state.active_overlay_at_root());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        for chip in &chips {
            assert_ne!(
                chip.action,
                KeyAction::OpenMenu,
                "no chip may dispatch OpenMenu while a sub-popup owns Esc"
            );
        }
        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("Esc  cancel"),
            "sub-popup Esc chip should say cancel: {row:?}"
        );
        // Note: "close" legitimately appears elsewhere in this row (the `q`
        // chip always says "close" while any overlay is open, root or not —
        // see `footer_q_chip_is_not_clickable_quit_when_a_manager_overlay_is_open`),
        // so the Esc chip's own label must be checked specifically rather
        // than scanning the whole row for the substring.
        assert!(
            !row.contains("Esc  close"),
            "sub-popup Esc chip should not say close: {row:?}"
        );
    }

    #[test]
    fn footer_esc_chip_labels_close_when_job_console_is_open() {
        // A manager's job console is showing a still-running job — Esc fully
        // closes the overlay there (matching the console's own "Esc close
        // (keeps running)" footer hint), so the dashboard footer chip must say
        // "close", not the generic sub-popup "cancel".
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use rocm_dash_core::state::StateEvent;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.jobs.apply(StateEvent::StartJob {
            id: "job".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        });
        state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState {
            active_job: Some("job".into()),
            ..Default::default()
        });
        assert!(state.has_open_overlay());
        assert!(!state.active_overlay_at_root());
        assert!(state.active_job_id().is_some());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            let _ = draw_footer(f, f.area(), &state, &theme);
        })
        .unwrap();

        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("close"),
            "job console Esc chip should say close: {row:?}"
        );
        assert!(
            !row.contains("cancel"),
            "job console Esc chip should not say cancel: {row:?}"
        );
    }

    #[test]
    fn footer_esc_chip_labels_cancel_when_job_console_shows_a_finished_job() {
        // Once the job console's job has finished, Esc only dismisses the
        // console back to the screen body (the overlay itself stays open) —
        // `on_console_key` never returns `Closed` for a terminal job. The
        // footer chip must not claim "close" here; it falls through to the
        // generic sub-popup "cancel" label, which already describes this
        // case correctly.
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use rocm_dash_core::state::StateEvent;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.jobs.apply(StateEvent::StartJob {
            id: "job".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        });
        state.jobs.apply(StateEvent::JobDone {
            id: "job".into(),
            code: 0,
        });
        state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState {
            active_job: Some("job".into()),
            ..Default::default()
        });
        assert!(state.has_open_overlay());
        assert!(!state.active_overlay_at_root());
        assert!(state.active_job_id().is_some());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            let _ = draw_footer(f, f.area(), &state, &theme);
        })
        .unwrap();

        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("Esc  cancel"),
            "finished-job console Esc chip should say cancel: {row:?}"
        );
        // Note: "close" legitimately appears elsewhere in this row (the `q`
        // chip always says "close" while any overlay is open — see
        // `footer_q_chip_is_not_clickable_quit_when_a_manager_overlay_is_open`),
        // so the Esc chip's own label must be checked specifically rather
        // than scanning the whole row for the substring.
        assert!(
            !row.contains("Esc  close"),
            "finished-job console Esc chip should not say close: {row:?}"
        );
    }

    #[test]
    fn footer_q_chip_is_not_clickable_quit_when_a_manager_overlay_is_open() {
        // Regression: while any manager overlay is open it owns every key
        // (the event loop routes each keypress to the manager's own `on_key`,
        // never falling through to `apply_action`), so a real `q` press can
        // never reach `KeyAction::Quit` there — it only cancels/closes the
        // overlay. A click on the footer chip must not diverge from that and
        // tear down the app (killing a still-running job via `kill_on_drop`)
        // when the key itself never would.
        use crate::ui::services_manager::ServicesManagerState;
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.services = Some(ServicesManagerState::default());
        assert!(state.has_open_overlay());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        for chip in &chips {
            assert_ne!(
                chip.action,
                KeyAction::Quit,
                "no chip may dispatch Quit while a manager overlay owns `q`"
            );
        }
        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("close"),
            "q chip should say close while an overlay is open: {row:?}"
        );
        assert!(
            !row.contains("quit"),
            "q chip should not say quit while an overlay is open: {row:?}"
        );
    }

    /// The wash `grey_overlay` paints behind an open overlay (see
    /// `modal::grey_overlay`'s own `grey_overlay_dims_every_cell` test for the
    /// exact color); these tests only check that `draw`/`draw_focused` actually
    /// invoke it at their three call sites, not the wash's own correctness.
    const OVERLAY_WASH: ratatui::style::Color = ratatui::style::Color::Rgb(0x1c, 0x1e, 0x22);

    #[test]
    fn draw_dims_periphery_with_grey_overlay_behind_manager_overlay() {
        // Covers `draw`'s manager-overlay `grey_overlay(f)` call (the
        // `has_open_overlay()` branch, not the approval one).
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut state = AppState::new("t".into(), "default-dark".into());
        state.services = Some(crate::ui::services_manager::ServicesManagerState::default());
        assert!(state.has_open_overlay());

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| draw(f, &mut state)).unwrap();
        let corner = term.backend().buffer().cell((0, 0)).unwrap();
        assert_eq!(
            corner.style().bg,
            Some(OVERLAY_WASH),
            "corner cell must carry grey_overlay's wash bg while a manager overlay is open"
        );
    }

    #[test]
    fn draw_dims_periphery_with_grey_overlay_behind_approval_modal() {
        // Covers `draw`'s approval-modal `grey_overlay(f)` call, distinct from
        // the manager-overlay one above (`state.approval` is `Some` with no
        // manager open).
        use crate::app::PendingApproval;
        use crate::ui::approval::{ApprovalChoice, ApprovalRequest};
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut state = AppState::new("t".into(), "default-dark".into());
        state.approval = Some(PendingApproval {
            req: ApprovalRequest::new("run it", vec!["echo hi".into()]),
            choice: ApprovalChoice::default(),
            name: "tool".into(),
            arguments: serde_json::Value::Null,
        });
        assert!(!state.has_open_overlay());

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| draw(f, &mut state)).unwrap();
        let corner = term.backend().buffer().cell((0, 0)).unwrap();
        assert_eq!(
            corner.style().bg,
            Some(OVERLAY_WASH),
            "corner cell must carry grey_overlay's wash bg while the approval modal is open"
        );
    }

    #[test]
    fn draw_focused_dims_periphery_with_grey_overlay_behind_manager_overlay() {
        // Covers `draw_focused`'s own `grey_overlay(f)` call — a separate
        // renderer from `draw`, previously skipping the wash entirely for the
        // bare-launcher focused-host path.
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut state = AppState::new("t".into(), "default-dark".into());
        state.services = Some(crate::ui::services_manager::ServicesManagerState::default());
        assert!(state.has_open_overlay());

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| draw_focused(f, &mut state)).unwrap();
        let corner = term.backend().buffer().cell((0, 0)).unwrap();
        assert_eq!(
            corner.style().bg,
            Some(OVERLAY_WASH),
            "corner cell must carry grey_overlay's wash bg in the focused-host renderer too"
        );
    }
}
