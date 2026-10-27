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
mod footer;
pub mod format;
pub mod gradient;
mod header;
mod help_modal;
pub mod install_manager;
mod instance_detail;
pub mod instance_list;
pub mod job_console;
pub mod launcher;
pub mod logs_view;
mod menu;
pub mod modal;
pub mod model_picker;
pub mod onboarding;
pub mod panel;
mod quit_confirm;
pub mod runtime_manager;
pub mod serve_wizard;
pub mod services_manager;
pub mod sparkline;
pub mod spinner;
pub mod tabs;
pub mod theme;
mod theme_picker;
pub mod update_manager;
mod warnings_modal;
pub mod widgets;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph};

use crate::app::{ActiveTab, AppState, Modal};
use crate::ui::theme::Theme;

use footer::draw_footer;
use header::draw_header;
pub(crate) use quit_confirm::quit_confirm_request;

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

    state.last_header_badge = draw_header(f, outer[0], state, &theme);

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
        Modal::Help => help_modal::draw_help(f, body, state.active_tab, &theme),
        // Observe folds the telemetry tabs; its detail modal is the instance
        // detail (the selectable list on that surface).
        Modal::Detail => {
            if state.active_tab == ActiveTab::Observe {
                let max_scroll = instance_detail::draw_detail(f, body, state, &theme);
                state.instance_detail_max_scroll = max_scroll;
            }
        }
        Modal::ThemePicker => {
            theme_picker::draw_theme_picker(
                f,
                body,
                state.theme_picker_sel,
                &state.theme_name,
                &theme,
            );
        }
        Modal::Menu => menu::draw_menu(f, body, state.menu_sel, &theme),
        Modal::Palette => menu::draw_palette(f, body, state.palette_sel, &theme),
        Modal::Options => menu::draw_options(f, body, state, &theme),
        Modal::GlobalHelp => help_modal::draw_global_help(f, body, &theme),
        Modal::Warnings => {
            let warnings = state.latest.as_ref().map_or(&[][..], |s| &s.warnings[..]);
            warnings_modal::draw_warnings(f, body, warnings, &theme);
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
        panel::grey_overlay(f);
    }
    let manager_rect = manager_rect(body);
    draw_active_manager(f, manager_rect, state, &theme);

    // Approval modal (Phase 4): drawn LAST so it sits on top of every overlay
    // and owns the screen while a mutating-tool approval is pending. This is the
    // sole approval renderer (the single gating path). It dims the backdrop too.
    if let Some(pa) = &state.approval {
        panel::grey_overlay(f);
        approval::draw_approval(f, body, &pa.req, pa.choice, &theme);
    }

    // Quit-confirm prompt (issue #145): a quit attempted while a model is
    // still serving. Same reuse of the approval seam, same "drawn last, owns
    // the screen" rule — the two are never open at the same time, so render
    // order between them never matters: `open_approval` refuses outright
    // while `quit_confirm` is open, and `open_quit_confirm` closes any open
    // approval the same way it closes every other overlay (see their doc
    // comments on `AppState`, not a doc comment on `quit_confirm` itself).
    if let Some(choice) = state.quit_confirm {
        panel::grey_overlay(f);
        // `true`: the dashboard's own `s` key (`quit_confirm_open_services_key`
        // in `event_loop`) can act on the hint this renders.
        approval::draw_approval(f, body, &quit_confirm_request(state, true), choice, &theme);
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
///
/// Also renders the quit-confirm prompt (issue #145) when open, same as
/// `draw`'s own arm — dead last, "owns the screen" the same way. No focused
/// path actually opens it today (`actions::request_quit`'s own doc comment:
/// unreachable in practice), but a future one that does needs two things
/// together, not this arm alone: `open_quit_confirm` closes every manager
/// overlay, which would otherwise make `AppState::focused_should_exit` fire
/// on the very next loop iteration and exit silently before this ever drew —
/// so that gate also exempts a pending quit-confirm (see its own doc
/// comment). Drawing the prompt here is what makes that exemption show
/// something instead of merely not exiting.
/// `offer_stop_action: false` — same reasoning as the launcher's own copy:
/// a focused host has no live Services overlay of its own to jump to.
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
        panel::grey_overlay(f);
        let manager_rect = manager_rect(body);
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

    if let Some(choice) = state.quit_confirm {
        panel::grey_overlay(f);
        approval::draw_approval(f, body, &quit_confirm_request(state, false), choice, &theme);
    }
}

/// Where an operational manager overlay is drawn inside the dashboard `body`.
/// One place, so the draw paths and the tests that check what fits agree.
pub(crate) fn manager_rect(body: Rect) -> Rect {
    panel::centered_rect(82, 80, 130, 34, body)
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
        services_manager::draw_services_manager(
            f,
            rect,
            sm,
            &state.instances,
            &state.jobs,
            state.services_past_attempts,
            theme,
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    // Shared with `app`'s own test module rather than each carrying its own
    // byte-identical copy — see its doc comment.
    use crate::app::test_support::{st, with_registry_probe};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    pub(in crate::ui) fn flat(term: &Terminal<TestBackend>) -> String {
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    /// Render `draw` onto a fresh `w`x`h` backend and return it row by row,
    /// trailing blanks trimmed. Row structure is what the sizing tests below
    /// assert on — `flat` throws it away.
    pub(in crate::ui) fn rows(w: u16, h: u16, draw: impl Fn(&mut ratatui::Frame)) -> Vec<String> {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f)).unwrap();
        let buf = term.backend().buffer();
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    /// Index of the popup's bottom border row (the rounded bottom-left corner).
    pub(in crate::ui) fn bottom_border_row(rows: &[String]) -> usize {
        rows.iter()
            .position(|r| r.contains('╰'))
            .unwrap_or_else(|| panic!("no popup bottom border was painted:\n{}", rows.join("\n")))
    }

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
    fn draw_renders_the_press_s_hint_for_a_stoppable_managed_instance() {
        // Regression guard: `quit_confirm_request_offers_the_stop_hint_only_for_the_dashboard_with_something_stoppable`
        // already pins the request-building logic, but nothing rendered the
        // actual frame through the top-level `draw` entry point a real user
        // sees — a wiring break between `draw`'s `offer_stop_action: true` arm
        // (`ui/mod.rs`'s own render match on `state.quit_confirm`) and the
        // request builder could pass that test while never reaching the
        // screen.
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use rocm_dash_core::metrics::{Instance, InstanceStatus};

        let mut state = AppState::new("t".into(), "default-dark".into());
        state.instances.insert(
            "svc-managed".to_string(),
            Instance {
                container_id: "svc-managed".to_string(),
                managed: true,
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        with_registry_probe(&mut state, true, &["svc-managed"]);
        state.open_quit_confirm();

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| draw(f, &mut state)).unwrap();
        let painted: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(
            painted.contains("Press s to open Services"),
            "the dashboard's rendered quit-confirm prompt must offer the \
             stop hint for a stoppable managed instance: {painted:?}"
        );
    }

    /// A state whose latest snapshot has a working GPU and these warnings — the
    /// case where the warning text was shown nowhere at all.
    pub(in crate::ui) fn state_with_warnings(warnings: &[&str]) -> AppState {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.latest = Some(rocm_dash_core::metrics::Snapshot {
            gpus: vec![rocm_dash_core::metrics::GpuMetrics {
                device_id: "gpu-0".into(),
                vram_total_mb: 1024,
                ..Default::default()
            }],
            warnings: warnings.iter().map(|w| (*w).to_string()).collect(),
            ..Default::default()
        });
        s
    }

    /// Draw a whole frame and return its rows as text.
    pub(in crate::ui) fn draw_frame(state: &mut AppState, w: u16, h: u16) -> Vec<String> {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| draw(f, state)).unwrap();
        let buf = term.backend().buffer();
        (0..h)
            .map(|y| (0..w).map(|x| buf.cell((x, y)).unwrap().symbol()).collect())
            .collect()
    }

    #[test]
    fn warnings_popup_shows_the_warning_text_while_a_gpu_is_reported() {
        let mut s = state_with_warnings(&["vram attribution: first", "vllm scrape: second"]);
        s.modal = Modal::Warnings;
        let frame = draw_frame(&mut s, 120, 40).join("\n");
        assert!(frame.contains("vram attribution: first"), "{frame}");
        assert!(frame.contains("vllm scrape: second"), "{frame}");
        assert!(frame.contains("Warnings (2)"), "{frame}");
    }

    /// The wash `grey_overlay` paints behind an open overlay (see the
    /// `grey_overlay_dims_every_cell` test in `menu.rs` for the exact color:
    /// the function itself is `panel::grey_overlay`, and the test stays beside
    /// `draw_logo`, which it composes); these tests only check that
    /// `draw`/`draw_focused` actually invoke it at their three call sites, not
    /// the wash's own correctness.
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

    #[test]
    fn draw_focused_renders_the_quit_confirm_prompt_when_open() {
        // No focused path opens `quit_confirm` today (`actions::request_quit`'s
        // own doc comment), but `draw_focused` must still render it if one
        // ever does — otherwise a future focused path would open an
        // invisible, key-eating prompt instead of a visible one. Paired with
        // `focused_should_exit_stays_shut_while_quit_confirm_is_pending`
        // (`app::event_loop` tests), which covers the exit-gate half of the
        // same guarantee. Mirrors `launcher_renders_the_quit_confirm_prompt_when_open`
        // in `ui::launcher`, the same regression for that renderer.
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut state = AppState::new("t".into(), "default-dark".into());
        state.open_quit_confirm();

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| draw_focused(f, &mut state)).unwrap();
        let painted: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(
            painted.contains("quit"),
            "the focused-host renderer must render the quit-confirm prompt: {painted:?}"
        );
    }

    #[test]
    fn draw_focused_shows_overlay_without_tab_chrome() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let mut s = st();
        // Intro card (no auto-run) → deterministic overlay content to assert on.
        s.examine_manager = Some(crate::ui::examine_manager::ExamineManagerState::default());
        let backend = TestBackend::new(120, 32);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| crate::ui::draw_focused(f, &mut s)).unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Examine"), "overlay content present: {out:?}");
        assert!(
            !out.contains("1–5"),
            "no dash tab-shell hint in focused mode"
        );
        assert!(
            out.contains("Esc"),
            "focused hint carries an Esc affordance"
        );
        // Periphery must carry the same grey_overlay wash `draw()` uses behind
        // every dashboard modal — text-only assertions above would still pass
        // if the `grey_overlay` call in `draw_focused` were dropped, since the
        // corner is plain theme bg either way in terms of glyphs (it's blank).
        let wash = ratatui::style::Color::Rgb(0x1c, 0x1e, 0x22);
        let corner = term.backend().buffer().cell((0, 0)).unwrap();
        assert_eq!(
            corner.style().bg,
            Some(wash),
            "corner cell must carry grey_overlay's wash bg, not plain theme bg"
        );
        // The "Esc back to menu" hint is rendered with a foreground-only
        // style (no explicit bg), and `ratatui::Style::patch` leaves an
        // unset field alone rather than clearing it — so the hint inherits
        // grey_overlay's wash bg from the cells underneath it, exactly like
        // `draw()`'s footer. Assert on the cell directly (not just its
        // text), so this fails if the hint's style ever gains an explicit
        // `bg` that would revert it to plain theme background.
        let hint_row_y = term.backend().buffer().area().height - 1;
        let hint_cell = term.backend().buffer().cell((0, hint_row_y)).unwrap();
        assert_eq!(
            hint_cell.symbol(),
            "E",
            "hint row should start with the Esc affordance"
        );
        assert_eq!(
            hint_cell.style().bg,
            Some(wash),
            "the Esc hint inherits grey_overlay's wash bg, same as draw()'s footer"
        );
    }

    #[test]
    fn draw_focused_serve_with_empty_recipes_does_not_panic() {
        // Edge input: the serve wizard must render (and the focused host must
        // paint it) with NO built-in recipes — the user can still type a path.
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let mut s = st();
        s.model_recipes = Vec::new();
        s.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState::default());
        let backend = TestBackend::new(120, 32);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| crate::ui::draw_focused(f, &mut s)).unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        // No panic == pass; the focused hint is present regardless of recipes.
        assert!(
            out.contains("Esc"),
            "focused hint renders with empty recipes"
        );
    }
}
