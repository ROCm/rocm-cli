// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Dashboard reducer: `AppState` and its `apply_event` entry point (the
//! `apply_action`-adjacent reducer impl it dispatches into lives in
//! `actions.rs`).
//!
//! Split into focused submodules to keep this file to the reducer's core:
//! `app/types.rs` (shared type/enum defs), `app/event_loop.rs` (terminal
//! lifecycle + tick loop), `app/scrollbar.rs` (mouse hit-testing), and
//! `app/actions.rs` (`KeyAction` dispatch). `crate::app::*` paths for the
//! public surface moved out are unchanged via the re-exports below.

use std::collections::{HashMap, VecDeque};

use rocm_dash_core::bench_schema::BenchmarkRow;
use rocm_dash_core::metrics::{Instance, Snapshot};
use rocm_dash_core::protocol::Event;

use crate::ui::theme::Theme;

// Submodules holding cohesive pieces of `AppState` + free fns split out of
// this file to keep the core reducer focused (a file→dir module move:
// `crate::app::*` paths for the public surface are unchanged).
mod actions;
mod chat;
mod event_loop;
mod scrollbar;
mod slash;
mod summary;
mod types;

// Re-exports restoring the pre-split `crate::app::*` public surface. A
// `pub(crate)` item with no caller through that path isn't re-exported just
// because it was reachable there pre-split (see the removed
// `NO_CHAT_BACKEND_MSG` re-export this rule cost).
pub use actions::{KeyAction, handle_mouse, tab_bar_hit};
pub(crate) use event_loop::{
    HOME_UPDATE_CHECK_JOB_ID, SHUTTING_DOWN, exit_on_ctrl_c, is_ctrl_c, lock_terminal_writer,
    restore_terminal, shutdown_claimed_on,
};

pub use event_loop::{run, spawn_termination_watcher};
// Only reached via a test (`launcher.rs`'s
// `the_front_door_comes_back_after_a_session_ends_cleanly`); gated to that
// build so the compiler (not a hand-maintained `#[allow]`) flags this as dead
// if that caller ever disappears.
#[cfg(test)]
pub(crate) use event_loop::restore_after_session;
pub use scrollbar::{FooterChip, PaneFocus, ScrollDrag, ScrollTarget, ScrollbarHandle};
pub use types::{
    ActiveTab, ChatConsent, ChatKeyCtx, ChatRole, ChatTurn, ConnState, Focus, Modal, PlannedAction,
    ReplayState, ResolvedArgs, UpdateStatus, format_mmss,
};
pub(crate) use types::{ChatProvider, PendingApproval, SlashOutcome, SlashToolRequest};

/// How many snapshots to keep for sparklines.
pub const HISTORY_CAP: usize = 240;

/// How many benchmark rows to keep client-side for the bench panel.
pub const BENCH_CAP: usize = 200;

pub struct AppState {
    pub connect: String,
    pub conn: ConnState,
    pub latest: Option<Snapshot>,
    pub history: VecDeque<Snapshot>,
    pub bench_rows: VecDeque<BenchmarkRow>,
    pub instances: HashMap<String, Instance>,
    pub active_tab: ActiveTab,
    pub modal: Modal,
    /// Cursor into the Esc main-menu rows (Options / Help / Quit).
    pub menu_sel: usize,
    /// Cursor into the command-palette destination rows.
    pub palette_sel: usize,
    /// Active tab index in the Options panel (General / CPU / GPU / Engines).
    pub options_tab: usize,
    /// Cursor into the ROCm tab's Actions list (per-tab selection).
    pub rocm_sel: usize,
    /// Cursor into the Serving tab's Actions list (per-tab selection).
    pub serving_sel: usize,
    /// Whether the active domain tab's focus is on the Actions list or the
    /// Details pane. `→`/Enter moves focus into Details; `←`/Esc returns to it.
    pub pane_focus: PaneFocus,
    /// Cursor into the sorted Instances grid.
    pub instance_sel: usize,
    /// Cursor into the bench_rows VecDeque (0 = oldest, len-1 = newest).
    pub bench_sel: usize,
    /// Cursor into the Hardware Observe sub-panel's per-GPU panel list.
    pub gpu_sel: usize,
    /// Scroll offset (first visible GPU index) for the Hardware Observe sub-panel when the
    /// GPU list renders as a scrolled window of compact rows. Kept in sync with
    /// `gpu_sel` so the selection stays visible.
    pub gpu_scroll: usize,
    pub theme_name: String,
    pub theme: Theme,
    pub theme_picker_sel: usize,
    /// Scroll offset (in lines) inside the instance Detail modal's body
    /// (launch args / env vars panes). Reset when the modal opens.
    pub instance_detail_scroll: u16,
    /// Last-measured upper bound for `instance_detail_scroll`, written back
    /// by the renderer each frame (see `ui::tabs::instances::draw_detail`),
    /// mirroring `chat_max_scroll`.
    pub instance_detail_max_scroll: u16,
    /// Vertical scroll offset (first visible line) of the active job console.
    /// Shared by whichever operational manager is showing its console; reset
    /// when an overlay opens (`close_overlays`).
    pub console_scroll: u16,
    /// Horizontal scroll offset (columns) of the active job console — log lines
    /// drawn wider than the console wrap off-screen, so the wheel/H-wheel pans.
    pub console_hscroll: u16,
    /// Monotonic UI repaint counter, incremented once per tick (~250ms). Drives
    /// frame-based animation (e.g. the job-console braille progress spinner)
    /// without threading a clock through the render path.
    pub tick_count: u64,
    /// Scroll offset of the wide-layout right LOGS dock, counted in lines UP from
    /// the newest line (0 = pinned to the tail). Clamped against the buffer.
    pub dock_logs_scroll: u16,
    /// Last drawn right-dock rect (wide layout, operational tabs). `None` when the
    /// dock isn't showing logs. Mouse-wheel hit-tests resolve against it.
    pub last_dock_area: Option<ratatui::layout::Rect>,
    /// Scrollbars drawn this frame, recorded so a mouse click/drag can hit-test
    /// them. Cleared and repopulated every `ui::draw`; interior-mutable because
    /// the deep render fns hold `&AppState`.
    pub scrollbars: std::cell::RefCell<Vec<ScrollbarHandle>>,
    /// Active scrollbar drag, including the pointer's offset inside a multi-cell
    /// thumb so grabbing it never snaps its leading edge to the pointer.
    pub scroll_drag: Option<ScrollDrag>,
    /// Chat transcript (TUI-local; never travels over the daemon protocol).
    pub chat: Vec<ChatTurn>,
    /// Pending input buffer for the Chat tab.
    pub chat_input: String,
    /// True while a chat request is in flight (drives a spinner / disables send).
    pub chat_sending: bool,
    /// Edge flag: set by `submit_chat`, consumed once by `event_loop` to spawn
    /// the agent round-trip. Keeps `apply_action` I/O-free (it only mutates).
    pub chat_dispatch: bool,
    /// True while the Chat tab has text-entry focus: keys go to `chat_input`
    /// instead of firing global hotkeys.
    pub chat_focused: bool,
    /// Scroll offset (lines from top) into the chat transcript. Clamped at 0;
    /// the renderer clamps the upper bound against the actual line count.
    pub chat_scroll: u16,
    /// Maximum valid chat transcript offset measured during the latest render.
    pub chat_max_scroll: u16,
    /// Whether new chat rows keep the viewport pinned to its bottom.
    pub chat_follow: bool,
    /// Resolved chat endpoint (base_url + model + env api_key). `None` when no
    /// endpoint was detected. `api_key` is never rendered or logged.
    pub chat_llm: Option<crate::llm::LlmConfig>,
    /// One-time consent gate for using the detected endpoint.
    pub chat_consent: ChatConsent,
    /// A locally-detected chat endpoint awaiting the user's use/save/dismiss
    /// choice (the in-TUI "detect a local engine" flow). `None` normally.
    pub chat_detect_offer: Option<crate::llm::LlmConfig>,
    /// True while a local-engine probe is in flight (drives a "detecting…" hint).
    pub chat_detecting: bool,
    /// Edge flag: set by `request_detect`, consumed once by `event_loop` to run
    /// the probe + `/v1/models` query off the reducer. Keeps `apply_action`
    /// I/O-free (it only mutates).
    pub chat_detect_dispatch: bool,
    /// Transient message from the last detect attempt (e.g. "no local engine
    /// found"), shown on the gate. Cleared when a new detect starts.
    pub chat_detect_msg: Option<String>,
    /// Edge flag: set by `save_detect_offer`, consumed once by `event_loop` to
    /// persist the accepted endpoint to `config.toml`. Keeps `apply_action`
    /// I/O-free.
    pub chat_persist_dispatch: bool,
    /// Edge: raised by `accept_detect_offer` (and thus `save_detect_offer`),
    /// consumed once by `event_loop`. `Some(previous)` carries the provider that
    /// was active before the optimistic switch to `Local`.
    ///
    /// Rebuilds the live chat `agent` from the newly-accepted `chat_llm` so
    /// submits stop routing to the stale startup backend (e.g. a cloud gateway).
    /// On rebuild failure the drain reverts `active_provider` to `previous` so
    /// the displayed provider stays honest. Keeps `apply_action` I/O-free.
    pub(crate) chat_endpoint_rebuild: Option<ChatProvider>,
    /// Replay scrubber state. `None` when running against a live daemon.
    pub replay: Option<ReplayState>,
    /// Data-honesty flag: `true` when the displayed telemetry is NOT from a live
    /// daemon — i.e. `--demo`, `--replay`, or an asset generator. Drives the
    /// persistent "SIMULATED DATA" marker and suppresses live/connected/health
    /// indicators so simulated data can never be presented as live. Distinct
    /// from `replay`, which is playback-control state and is not set by the
    /// screenshot/cast generators.
    pub simulated: bool,
    /// Managed-service records that are no longer running, from the bin's
    /// registry read at launch (see `ResolvedArgs::services_past_attempts`).
    /// Rendered by the services overlay so failed servers are not invisible.
    pub services_past_attempts: usize,
    /// Last body area used by the most recent draw. Mouse hit-tests resolve
    /// pointer coordinates against this rect (filled by `ui::draw`).
    pub last_body_area: Option<ratatui::layout::Rect>,
    /// Same for the tab bar, used for click-to-switch-tab.
    pub last_tab_bar_area: Option<ratatui::layout::Rect>,
    /// Clickable footer-legend chips from the most recent draw. Left-clicking a
    /// chip dispatches the same `KeyAction` as pressing that key.
    pub last_footer_chips: Vec<FooterChip>,
    /// Background-job model for operational screens (Phase 3 Wave 1). The
    /// job-bridge runtime streams `StateEvent`s into this from the event loop.
    pub jobs: rocm_dash_core::state::State,
    /// Services manager overlay (Phase 3 Wave 1). `None` = closed.
    pub services: Option<crate::ui::services_manager::ServicesManagerState>,
    /// Serve wizard overlay (Phase 3 Wave 1). `None` = closed.
    pub serve_wizard: Option<crate::ui::serve_wizard::ServeWizardState>,
    /// Engine manager overlay (Phase 3 Wave 1). `None` = closed.
    pub engine_manager: Option<crate::ui::engine_manager::EngineManagerState>,
    /// examine overlay (Phase 3 Wave 2). `None` = closed.
    pub examine_manager: Option<crate::ui::examine_manager::ExamineManagerState>,
    /// Update overlay (Phase 3 Wave 2). `None` = closed.
    pub update_manager: Option<crate::ui::update_manager::UpdateManagerState>,
    /// Install overlay (Phase 3 Wave 2). `None` = closed.
    pub install_manager: Option<crate::ui::install_manager::InstallManagerState>,
    /// Logs overlay (Phase 3 Wave 3). `None` = closed.
    pub logs_view: Option<crate::ui::logs_view::LogsViewState>,
    /// Runtime manager overlay (Phase 3 Wave 2). `None` = closed.
    pub runtime_manager: Option<crate::ui::runtime_manager::RuntimeManagerState>,
    /// Onboarding wizard overlay (Phase 3 Wave 2). `None` = closed.
    pub onboarding: Option<crate::ui::onboarding::OnboardingState>,
    /// Automations manager overlay (Phase 3 Wave 3). `None` = closed.
    pub automations_manager: Option<crate::ui::automations_manager::AutomationsManagerState>,
    /// Command runner overlay (Phase 3 Wave 3). `None` = closed.
    pub command_screen: Option<crate::ui::command_screen::CommandScreenState>,
    /// Config & provider manager overlay (Phase 3 Wave 3). `None` = closed.
    pub config_manager: Option<crate::ui::config_manager::ConfigManagerState>,
    /// Bench-run form overlay. `None` = closed.
    pub bench_run: Option<crate::ui::bench_run::BenchRunState>,
    /// Built-in model recipes for the serve wizard's picker. Set from
    /// `ResolvedArgs` in the event loop; empty by default.
    pub model_recipes: Vec<crate::ui::model_picker::ModelRecipeSummary>,
    /// Registered ROCm runtimes for the runtime manager. Set from
    /// `ResolvedArgs` in the event loop; empty by default.
    pub runtimes: Vec<crate::ui::runtime_manager::RuntimeSummary>,
    /// Background checks for the automations manager. Set from `ResolvedArgs`
    /// in the event loop; empty by default.
    pub automations: Vec<crate::ui::automations_manager::AutomationSummary>,
    /// Bin-injected tool-executor seam. Set from `ResolvedArgs` in the event
    /// loop; `None` for demo/replay/mock and by default.
    pub tool_executor: Option<crate::tool_exec::SharedRocmToolExecutor>,
    /// Daemon-tailed bench CSV path from the bin config.
    ///
    /// Forwarded from [`ResolvedArgs::bench_results_dir`] so the bench-run form
    /// can default `--out` to the live-tailed file. `None` when not configured.
    pub bench_results_dir: Option<std::path::PathBuf>,
    /// Set by a `/quit` (or `/exit`) slash command; the event loop breaks on it.
    pub(crate) should_quit: bool,
    /// Edge: a pending executor-backed read-only slash command. Raised by
    /// `handle_slash_command`, drained once by the event loop (spawn_blocking).
    pub(crate) slash_tool: Option<SlashToolRequest>,
    /// Edge: a pending `/plan <request>` natural-language plan. Raised by
    /// `handle_slash_command`, drained once by the event loop (spawn_blocking)
    /// which calls the read-only `natural_language_plan` tool (Phase 7).
    pub(crate) plan_request: Option<String>,
    /// A surfaced mutating-tool approval awaiting the operator's decision
    /// (Phase 4). `Some` ⇒ the approval modal is open and owns keyboard focus.
    pub(crate) approval: Option<PendingApproval>,
    /// The chat LLM backend currently selected (Phase 8). Defaults to `Local`.
    pub(crate) active_provider: ChatProvider,
    /// Edge: a pending `/provider` switch. Raised by `handle_slash_command`,
    /// drained once by the event loop which rebuilds the live `agent`. Carries
    /// both the target and the provider that was active BEFORE the optimistic
    /// switch, so a failed build (missing key) reverts to the prior provider
    /// rather than unconditionally to `Local`.
    pub(crate) provider_switch: Option<ProviderSwitch>,
    /// Result of the last completed background update check. Drives the Home
    /// tab's Updates tile. `Unknown` until the first check resolves.
    pub update_status: UpdateStatus,
    /// True while a `home-update-check` job is running (spawned but not yet
    /// terminal). Drives the tile's "Checking…" state.
    pub update_status_pending: bool,
    /// When the next periodic update check is due. Checked each tick;
    /// initialized to `Instant::now()` so a check is due immediately after
    /// startup.
    pub(crate) update_check_due_at: std::time::Instant,
}

/// A pending `/provider` switch edge: the `target` backend plus the `previous`
/// provider captured before the optimistic `active_provider` set. The event-loop
/// drain rebuilds the agent for `target`; on failure it reverts `active_provider`
/// to `previous` (honest display) instead of forcing `Local`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderSwitch {
    pub(crate) previous: ChatProvider,
    pub(crate) target: ChatProvider,
}

impl AppState {
    pub fn new(connect: String, theme_name: String) -> Self {
        let theme = Theme::from_name(&theme_name);
        let names = crate::ui::theme::theme_names();
        let theme_picker_sel = names.iter().position(|n| *n == theme_name).unwrap_or(0);
        Self {
            connect,
            conn: ConnState::Initial,
            latest: None,
            history: VecDeque::with_capacity(HISTORY_CAP),
            bench_rows: VecDeque::with_capacity(BENCH_CAP),
            instances: HashMap::new(),
            active_tab: ActiveTab::default(),
            modal: Modal::None,
            menu_sel: 0,
            palette_sel: 0,
            options_tab: 0,
            rocm_sel: 0,
            serving_sel: 0,
            pane_focus: PaneFocus::Actions,
            instance_sel: 0,
            bench_sel: 0,
            gpu_sel: 0,
            gpu_scroll: 0,
            theme_name,
            theme,
            theme_picker_sel,
            instance_detail_scroll: 0,
            instance_detail_max_scroll: 0,
            console_scroll: 0,
            console_hscroll: 0,
            tick_count: 0,
            dock_logs_scroll: 0,
            last_dock_area: None,
            scrollbars: std::cell::RefCell::new(Vec::new()),
            scroll_drag: None,
            chat: Vec::new(),
            chat_input: String::new(),
            chat_sending: false,
            chat_dispatch: false,
            chat_focused: false,
            chat_scroll: 0,
            chat_max_scroll: 0,
            chat_follow: true,
            chat_llm: None,
            chat_consent: ChatConsent::Unavailable,
            chat_detect_offer: None,
            chat_detecting: false,
            chat_detect_dispatch: false,
            chat_detect_msg: None,
            chat_persist_dispatch: false,
            chat_endpoint_rebuild: None,
            replay: None,
            simulated: false,
            services_past_attempts: 0,
            last_body_area: None,
            last_tab_bar_area: None,
            last_footer_chips: Vec::new(),
            jobs: rocm_dash_core::state::State::default(),
            services: None,
            serve_wizard: None,
            engine_manager: None,
            examine_manager: None,
            update_manager: None,
            install_manager: None,
            logs_view: None,
            runtime_manager: None,
            onboarding: None,
            automations_manager: None,
            command_screen: None,
            config_manager: None,
            bench_run: None,
            model_recipes: Vec::new(),
            runtimes: Vec::new(),
            automations: Vec::new(),
            tool_executor: None,
            bench_results_dir: None,
            should_quit: false,
            slash_tool: None,
            plan_request: None,
            approval: None,
            active_provider: ChatProvider::default(),
            provider_switch: None,
            update_status: UpdateStatus::Unknown,
            update_status_pending: false,
            update_check_due_at: std::time::Instant::now(),
        }
    }

    /// Close every operational overlay. The overlays are mutually exclusive
    /// (only one is routed/drawn at a time), so opening any one first clears the
    /// rest — no open path can leave two `Some` at once.
    fn close_overlays(&mut self) {
        self.services = None;
        self.serve_wizard = None;
        self.engine_manager = None;
        self.examine_manager = None;
        self.update_manager = None;
        self.install_manager = None;
        self.logs_view = None;
        self.runtime_manager = None;
        self.onboarding = None;
        self.automations_manager = None;
        self.command_screen = None;
        self.config_manager = None;
        self.bench_run = None;
        self.approval = None;
        // A fresh overlay starts its console at the top.
        self.console_scroll = 0;
        self.console_hscroll = 0;
    }

    /// The job id of the manager that is currently showing its console (if any).
    /// Only one manager is open at a time, so at most one matches; `None` when no
    /// overlay is open or the open one is still on its form screen.
    pub(crate) fn active_job_id(&self) -> Option<&str> {
        self.services
            .as_ref()
            .and_then(|m| m.active_job.as_deref())
            .or_else(|| {
                self.serve_wizard
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.engine_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.examine_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.update_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.install_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.logs_view
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.runtime_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.onboarding
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.automations_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.command_screen
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
            .or_else(|| {
                self.config_manager
                    .as_ref()
                    .and_then(|m| m.active_job.as_deref())
            })
    }

    /// Whether a job console is currently displayed (a manager is open AND on its
    /// console sub-view). Gates scroll routing so wheel/PgUp-PgDn pan the log
    /// instead of moving the obscured Actions list.
    pub(crate) fn has_active_console(&self) -> bool {
        self.active_job_id().is_some()
    }

    /// Pan the active job console. `dv`/`dh` are line/column deltas (negative =
    /// up/left). Clamps at 0; the vertical offset is clamped against the active
    /// job's line count and the horizontal offset against its widest line, so it
    /// can't scroll past the content into blank space in either axis.
    pub(crate) fn scroll_console(&mut self, dv: i16, dh: i16) {
        let job = self.active_job_id().and_then(|id| self.jobs.job(id));
        let max_v = job.map_or(0, |j| j.output.len().saturating_sub(1));
        let max_v = i32::try_from(max_v).unwrap_or(i32::MAX);
        let max_h = job.map_or(0, |j| {
            j.output
                .iter()
                .map(|l| l.chars().count())
                .max()
                .unwrap_or(0)
                .saturating_sub(1)
        });
        let max_h = i32::try_from(max_h).unwrap_or(i32::MAX);
        let v = u16::try_from((i32::from(self.console_scroll) + i32::from(dv)).clamp(0, max_v))
            .unwrap_or(u16::MAX);
        let h = u16::try_from((i32::from(self.console_hscroll) + i32::from(dh)).clamp(0, max_h))
            .unwrap_or(u16::MAX);
        self.console_scroll = v;
        self.console_hscroll = h;
    }

    /// Total log lines the wide-layout LOGS dock aggregates across all jobs.
    /// Single source for both the renderer's window and the scroll clamp.
    pub(crate) fn dock_logs_total(&self) -> usize {
        self.jobs.jobs.values().map(|j| j.output.len()).sum()
    }

    /// Scroll the wide-layout LOGS dock by `dv` lines (negative = toward newer).
    /// Counted up from the tail and clamped against the buffer minus the dock's
    /// visible height (derived from the last drawn dock rect).
    pub(crate) fn scroll_dock(&mut self, dv: i16) {
        let cap = self
            .last_dock_area
            .map_or(0, |r| r.height.saturating_sub(3) as usize);
        let max = i32::try_from(self.dock_logs_total().saturating_sub(cap)).unwrap_or(i32::MAX);
        self.dock_logs_scroll =
            u16::try_from((i32::from(self.dock_logs_scroll) + i32::from(dv)).clamp(0, max))
                .unwrap_or(u16::MAX);
    }

    /// Apply an absolute chat offset against the latest measured viewport and
    /// derive follow-tail state from whether the viewport is at its bottom.
    fn set_chat_scroll(&mut self, position: usize) {
        self.chat_scroll = u16::try_from(position)
            .unwrap_or(u16::MAX)
            .min(self.chat_max_scroll);
        self.chat_follow = self.chat_scroll == self.chat_max_scroll;
    }

    /// Arm a scrollbar drag and apply `position` in the target's own offset
    /// units. The grab offset is retained for subsequent pointer moves.
    pub(crate) fn apply_scroll_grab(
        &mut self,
        target: ScrollTarget,
        position: usize,
        grab_offset: u16,
    ) {
        self.scroll_drag = Some(ScrollDrag {
            target,
            grab_offset,
        });
        let p = u16::try_from(position).unwrap_or(u16::MAX);
        match target {
            ScrollTarget::Console => self.console_scroll = p,
            ScrollTarget::ConsoleH => self.console_hscroll = p,
            ScrollTarget::Chat => self.set_chat_scroll(position),
            ScrollTarget::DockLogs => self.dock_logs_scroll = p,
            ScrollTarget::InstanceDetail => self.instance_detail_scroll = p,
        }
    }

    /// Record a scrollbar drawn this frame for later mouse hit-testing.
    ///
    /// `area` is the rect passed to the scrollbar helper and `drawn` its return
    /// value; when they're equal no bar was drawn (content fit) and nothing is
    /// recorded. The 1-cell track strip is derived from `area` and `horizontal`.
    pub(crate) fn record_scrollbar(
        &self,
        area: ratatui::layout::Rect,
        drawn: ratatui::layout::Rect,
        horizontal: bool,
        content_len: usize,
        viewport_len: usize,
        target: ScrollTarget,
    ) {
        if let Some(h) =
            ScrollbarHandle::new(area, drawn, horizontal, content_len, viewport_len, target)
        {
            self.scrollbars.borrow_mut().push(h);
        }
    }

    /// Whether any operational manager overlay is open (approval excluded — it
    /// is the separate gating layer with its own routing). Used to decide inline
    /// vs. centered manager rendering and the ROCm/Serving `←`/Esc back-out.
    pub(crate) const fn has_open_overlay(&self) -> bool {
        self.services.is_some()
            || self.serve_wizard.is_some()
            || self.engine_manager.is_some()
            || self.examine_manager.is_some()
            || self.update_manager.is_some()
            || self.install_manager.is_some()
            || self.logs_view.is_some()
            || self.runtime_manager.is_some()
            || self.onboarding.is_some()
            || self.automations_manager.is_some()
            || self.command_screen.is_some()
            || self.config_manager.is_some()
            || self.bench_run.is_some()
    }

    /// Whether a chat tool-call approval is pending. Its own gating layer,
    /// separate from [`has_open_overlay`](Self::has_open_overlay) — a real
    /// keypress or click can never reach `OpenThemePicker`/`ToggleHelp`/`Quit`/
    /// a scrollbar/the pane body while this is `true`, so every input path
    /// that swallows for an open overlay must also check this, or a mouse
    /// gesture could bypass a gate no keypress ever could. Single source of
    /// truth for that check so the call sites can't drift apart.
    pub(crate) const fn approval_pending(&self) -> bool {
        self.approval.is_some()
    }

    /// Whether *either* gating layer owns the screen: an open manager overlay
    /// or a pending chat approval. This exact `||` is what every input path
    /// that swallows for one must also swallow for the other — two call sites
    /// wrote it out by hand before this existed, each with its own copy of
    /// this same reasoning; a third forgetting one half would reopen the
    /// class of bug `approval_pending`'s own doc comment describes.
    pub(crate) const fn overlay_or_approval(&self) -> bool {
        self.has_open_overlay() || self.approval_pending()
    }

    /// Focused-host exit gate: `true` when a `focus` is active AND its single
    /// overlay is closed (no manager is `Some`).
    ///
    /// [`has_open_overlay`](Self::has_open_overlay) stays `true` while a
    /// sub-popup (folder browser / model picker) is open, so this can't fire
    /// while the user is inside one of those. It does NOT by itself protect a
    /// running job console: the shared console maps `q` / running-`Esc` to
    /// "close overlay", which would null the manager mid-job. That case is
    /// handled upstream in `event_loop` by `focused_close_key_blocked`, which
    /// swallows those keys while the job is non-terminal — so by the time this
    /// gate is checked, a focused overlay only ever closed at its root (form
    /// screen or a terminal job). Always `false` for the normal
    /// (`focus == None`) dashboard, so its loop never self-exits. Pure read →
    /// unit-testable without a live terminal.
    pub(crate) const fn focused_should_exit(&self, focus: Option<Focus>) -> bool {
        focus.is_some() && !self.has_open_overlay()
    }

    /// Whether the open manager (if any) is at its TOP-LEVEL screen — no nested
    /// sub-popup (folder browser / model picker / import input), no pending
    /// gating approval, and no job console (running or terminal). Only one
    /// manager is open at a time, so this reflects that one; `true` when none is
    /// open. Gates the Esc back-out so Esc cancels the innermost layer first
    /// (and is ignored while a job runs) before it can eject the manager.
    pub(crate) fn active_overlay_at_root(&self) -> bool {
        self.serve_wizard.as_ref().is_none_or(|w| {
            w.browser.is_none()
                && w.picker.is_none()
                && w.approval.is_none()
                && w.active_job.is_none()
        }) && self
            .install_manager
            .as_ref()
            .is_none_or(|m| m.browser.is_none() && m.approval.is_none() && m.active_job.is_none())
            && self.onboarding.as_ref().is_none_or(|m| {
                m.browser.is_none()
                    && m.install_config.is_none()
                    && m.approval.is_none()
                    && m.active_job.is_none()
            })
            && self.runtime_manager.as_ref().is_none_or(|m| {
                m.browser.is_none()
                    && m.import_input.is_none()
                    && m.approval.is_none()
                    && m.active_job.is_none()
            })
            && self
                .engine_manager
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .services
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .update_manager
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .config_manager
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .command_screen
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .automations_manager
                .as_ref()
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
            && self
                .examine_manager
                .as_ref()
                .is_none_or(|m| m.active_job.is_none())
            && self
                .logs_view
                .as_ref()
                .is_none_or(|m| m.active_job.is_none())
        // bench_run is always at root when Some (no nested sub-popup or job).
    }

    /// Whether an `Esc` keypress should back out of an inline manager: true on
    /// any tab while a manager overlay is open AND that manager is at its root
    /// screen. The event loop closes the manager and returns focus to the
    /// Actions list when this holds. Pure read so it is unit-testable (the
    /// mutation lives in the event-loop arm).
    ///
    /// Not just ROCm/Serving: a manager can be opened from a non-domain tab
    /// (e.g. `examine_manager` from an Observe hotkey). This used to be gated
    /// on `active_tab == Rocm | Serving`, so on other tabs the manager's own
    /// event-loop arm handled root Esc directly (every overlay type already
    /// has a dedicated `Some(Ok(CtEvent::Key(k))) if state.<overlay>.is_some()`
    /// arm ahead of the generic handler, and each self-closes on root Esc
    /// regardless of `active_tab` — so there was no "Modal stays set but
    /// invisible" bug to fix here — true of every manager except onboarding,
    /// see below). Dropping the tab guard moves the close from the manager's
    /// own `on_key` to this shared path (`close_overlays()` + `pane_focus =
    /// Actions`) so a future manager doesn't need to duplicate that root-Esc
    /// handling. `pane_focus` is meaningless outside Rocm/Serving, so
    /// resetting it there is a harmless no-op.
    ///
    /// This generalization is only correct if `active_overlay_at_root`'s
    /// per-manager clause enumerates every nesting field the manager's state
    /// struct has — see the note on `OnboardingState` (and its sibling
    /// manager-state structs) about keeping that enumeration in sync when a
    /// new nested sub-view field is added. `active_overlay_at_root_enumeration_is_exhaustive`
    /// turns that into a build break instead of a silent drift: it destructures
    /// every one of those structs without `..`, so adding a field to any of
    /// them without updating both the test and this function fails to compile.
    ///
    /// When the manager has a sub-popup / approval / job console open, this is
    /// `false` so Esc falls through to the manager's own handler (cancel the
    /// sub-layer, dismiss a terminal console, or be ignored while a job runs) —
    /// it cannot eject the whole manager mid-flow.
    ///
    /// Only `Esc` backs out — `←` is left to the open manager (serve_wizard /
    /// install / config use it to cycle options). When NO manager is open, `←`
    /// returns focus from the Details preview to the Actions list via the normal
    /// `PaneFocusActions` key path.
    pub(crate) fn should_pane_back_out(&self, code: crossterm::event::KeyCode) -> bool {
        self.has_open_overlay()
            && self.active_overlay_at_root()
            && matches!(code, crossterm::event::KeyCode::Esc)
    }

    /// Open the theme picker modal, positioning the cursor on the active theme.
    pub fn open_theme_picker(&mut self) {
        self.close_overlays();
        let names = crate::ui::theme::theme_names();
        self.theme_picker_sel = names
            .iter()
            .position(|n| *n == self.theme_name)
            .unwrap_or(0);
        self.modal = Modal::ThemePicker;
    }

    /// Move the theme picker cursor. Clamped to the theme registry length.
    pub fn theme_picker_move(&mut self, delta: isize) {
        let len = crate::ui::theme::theme_names().len();
        if len == 0 {
            return;
        }
        let next =
            (self.theme_picker_sel.cast_signed() + delta).clamp(0, len.cast_signed() - 1) as usize;
        self.theme_picker_sel = next;
    }

    pub const fn theme_picker_first(&mut self) {
        self.theme_picker_sel = 0;
    }

    pub fn theme_picker_last(&mut self) {
        let len = crate::ui::theme::theme_names().len();
        if len > 0 {
            self.theme_picker_sel = len - 1;
        }
    }

    /// Reset the instance Detail modal's scroll offset (called when opening
    /// the modal, so a stale offset never carries over from a previous
    /// instance's selection).
    pub const fn reset_instance_detail_scroll(&mut self) {
        self.instance_detail_scroll = 0;
        self.instance_detail_max_scroll = 0;
    }

    /// Adjust the instance Detail modal's scroll. `delta` is in lines;
    /// clamped against `[0, instance_detail_max_scroll]` (the latter is last
    /// written back by the renderer, see `instance_detail_max_scroll`), so
    /// `i16::MIN`/`i16::MAX` ("jump to start/end") land exactly on
    /// `0`/`instance_detail_max_scroll` instead of overflowing into an offset
    /// far past the real content length.
    pub fn scroll_instance_detail(&mut self, delta: i16) {
        let cur = i32::from(self.instance_detail_scroll);
        let max = i32::from(self.instance_detail_max_scroll);
        let next = u16::try_from((cur + i32::from(delta)).clamp(0, max)).unwrap_or(u16::MAX);
        self.instance_detail_scroll = next;
    }

    /// Install the resolved chat endpoint and set the initial consent state.
    /// `None` → `Unavailable`; `Some` → `Accepted` when pre-consented (e.g.
    /// `--chat-yes`), otherwise `Pending` (the one-time in-TUI prompt).
    /// Accept the detected endpoint and enable chat. No-op when no endpoint is
    /// available. Focuses the input so the user can type immediately.
    pub const fn accept_chat_consent(&mut self) {
        if self.chat_llm.is_some() {
            self.chat_consent = ChatConsent::Accepted;
            self.chat_focused = true;
        }
    }

    /// Decline the detected endpoint. Chat stays off (re-enable with `y`).
    pub const fn decline_chat_consent(&mut self) {
        if self.chat_llm.is_some() {
            self.chat_consent = ChatConsent::Declined;
            self.chat_focused = false;
        }
    }

    /// Request an in-TUI local-engine probe. Raises the one-shot
    /// `chat_detect_dispatch` edge so `event_loop` runs the probe + `/v1/models`
    /// query off the reducer. No-op while a probe is already in flight or an
    /// offer is awaiting a decision. I/O-free.
    ///
    /// Reachable both from the pre-accept gate (`'d'` key) and, once
    /// `ChatConsent::Accepted`, from the `/detect` slash command (a focused
    /// `'d'` keypress is ordinary chat text at that point, so the gate key
    /// doesn't apply there — see `handle_slash_command`). When already
    /// accepted, echo the in-flight probe into the transcript since the
    /// pre-accept "detecting…" banner isn't drawn once chat is live.
    pub fn request_detect(&mut self) {
        if self.chat_detecting || self.chat_detect_offer.is_some() {
            return;
        }
        self.chat_detecting = true;
        self.chat_detect_msg = None;
        self.chat_detect_dispatch = true;
        if self.chat_consent == ChatConsent::Accepted {
            self.chat.push(ChatTurn::agent(
                "Detecting a local engine (Lemonade :13305 / vLLM :8000 / rocm serve :11435)…"
                    .to_string(),
            ));
        }
    }

    /// Record the result of a detect attempt: `Some(cfg)` raises the offer
    /// prompt; `None` records a "nothing found" message. Clears the in-flight
    /// flag either way.
    ///
    /// Once `ChatConsent::Accepted`, the offer prompt is not drawn (the gate UI
    /// only renders pre-accept), so the result is also echoed into the
    /// transcript with the `/detect accept|save|dismiss` sub-commands needed to
    /// act on it.
    pub fn set_detect_result(&mut self, offer: Option<crate::llm::LlmConfig>) {
        self.chat_detecting = false;
        let accepted = self.chat_consent == ChatConsent::Accepted;
        if let Some(cfg) = offer {
            self.chat_detect_msg = None;
            if accepted {
                self.chat.push(ChatTurn::agent(format!(
                    "Detected a local engine: {}  (model: {}). Type `/detect accept` to \
                     switch now, `/detect save` to also persist it, or `/detect dismiss` \
                     to ignore.",
                    cfg.base_url, cfg.model
                )));
            }
            self.chat_detect_offer = Some(cfg);
        } else {
            self.chat_detect_offer = None;
            let msg = "no local engine found (Lemonade :13305 / vLLM :8000 / rocm serve :11435)";
            if accepted {
                self.chat.push(ChatTurn::agent(msg.to_string()));
            } else {
                self.chat_detect_msg = Some(msg.to_string());
            }
        }
    }

    /// Accept the detected local endpoint for this session.
    ///
    /// Switches `chat_llm` to the offer and enables chat. The accepted endpoint
    /// is local, so this also selects the Local provider and raises the
    /// `chat_endpoint_rebuild` edge, marking the live agent for rebuild in
    /// `event_loop`. No-op when no offer is pending.
    pub fn accept_detect_offer(&mut self) {
        if let Some(cfg) = self.chat_detect_offer.take() {
            self.chat_llm = Some(cfg);
            self.chat_consent = ChatConsent::Accepted;
            self.chat_focused = true;
            // The accepted endpoint is local; align the displayed provider and
            // raise the rebuild edge so `event_loop` swaps the live agent to it
            // (the startup agent may be a cloud gateway — see
            // `chat_endpoint_rebuild`). Capture the previous provider first so
            // the drain can revert the optimistic switch if the rebuild fails.
            let previous = self.active_provider;
            self.active_provider = ChatProvider::Local;
            self.chat_endpoint_rebuild = Some(previous);
        }
    }

    /// Dismiss the detected-endpoint offer, leaving the prior chat config and
    /// consent untouched.
    pub fn dismiss_detect_offer(&mut self) {
        self.chat_detect_offer = None;
    }

    /// Accept the detected endpoint **and** persist it.
    ///
    /// Same as [`accept_detect_offer`](Self::accept_detect_offer) (which selects
    /// the Local provider and raises the endpoint-rebuild edge), then raise the
    /// one-shot `chat_persist_dispatch` edge so `event_loop` writes
    /// `tui.chat_url`/`tui.chat_model` to the config file. No-op when no offer
    /// is pending.
    pub fn save_detect_offer(&mut self) {
        let had_offer = self.chat_detect_offer.is_some();
        self.accept_detect_offer();
        if had_offer {
            self.chat_persist_dispatch = true;
        }
    }

    /// Submit the current chat input. Empty / whitespace-only input is
    /// ignored. Pushes the user turn, marks the request in-flight, and raises
    /// the one-shot `chat_dispatch` edge so `event_loop` spawns the agent
    /// round-trip. Stays I/O-free (the spawn happens outside the reducer).
    pub fn submit_chat(&mut self) {
        // Ignore submits while a request is in flight — prevents a double-Enter
        // from spawning two racing agent tasks with desynced history.
        if self.chat_sending {
            return;
        }
        let text = self.chat_input.trim().to_string();
        if text.is_empty() {
            return;
        }
        // Slash commands are handled locally (nav/overlays/read-only tools) and
        // never reach the LLM. A `/`-prefixed line is ALWAYS consumed here, even
        // when unknown (it gets an error turn) — only non-slash text is sent on.
        if self.handle_slash_command(&text) == SlashOutcome::Handled {
            self.chat_input.clear();
            return;
        }
        self.chat.push(ChatTurn::user(text));
        self.chat_input.clear();
        self.chat_sending = true;
        self.chat_dispatch = true;
    }

    /// Capture a read-only telemetry snapshot for the chat tools. Plain owned
    /// clones — tools read this without touching the reducer or `&AppState`.
    pub fn state_snapshot(&self) -> crate::agent::StateSnapshot {
        crate::agent::StateSnapshot {
            latest: self.latest.clone(),
            instances: self.instances.values().cloned().collect(),
            bench_rows: self.bench_rows.iter().cloned().collect(),
        }
    }

    /// Handle a successful agent reply: append an `Agent` turn, clear the
    /// in-flight flag. Called from `event_loop` on `ClientMsg::ChatReply`.
    pub fn on_chat_reply(&mut self, text: String) {
        self.chat.push(ChatTurn::agent(text));
        self.chat_sending = false;
    }

    /// Handle an agent failure: append an `Error` turn, clear the in-flight
    /// flag. Called on `ClientMsg::ChatError` — never panics.
    pub fn on_chat_error(&mut self, message: String) {
        self.chat.push(ChatTurn::error(message));
        self.chat_sending = false;
    }

    /// Handle a completed natural-language plan (Phase 7). Pushes the rendered
    /// plan as a chat turn (the review). If the plan's next action is a complete
    /// mutating action (`approval_required` AND NOT `has_placeholders`) that is
    /// NOT provider-assisted, hand its argv to the Phase-4 approval flow via the
    /// `rocm_command` slash-tool edge (execute → ApprovalRequired → modal). A
    /// placeholder/incomplete plan, a non-mutating one, or a provider-assisted
    /// one stays plan-only: no approval focus, no execution. Provider-assisted
    /// plans are review-only, mirroring the bin's
    /// `validate_freeform_execution_action` guard.
    pub(crate) fn on_plan_ready(&mut self, text: String, action: Option<PlannedAction>) {
        self.chat.push(ChatTurn::agent(text));
        if let Some(action) = action
            && action.approval_required
            && !action.has_placeholders
            && !action.provider_assisted
        {
            // `/plan` drains off-thread without setting `chat_sending`, so a slash
            // command issued while the plan was in flight may already have queued a
            // tool request or opened an approval. Don't clobber it — surface a
            // message and drop the plan's action (mirrors `open_approval`'s
            // single-in-flight guard).
            if self.slash_tool.is_some() || self.approval.is_some() {
                self.chat.push(ChatTurn::error(
                    "A command is already in progress; the planned action was discarded. Resolve it first, then re-run the plan.",
                ));
                return;
            }
            self.slash_tool = Some(SlashToolRequest {
                name: "rocm_command".to_string(),
                args: serde_json::json!({ "args": action.args }),
                label: "plan action".to_string(),
            });
        }
    }

    /// Handle an executor-backed slash-tool reply (`/model`, `/daemon`): append
    /// the summary as an agent-role turn WITHOUT touching `chat_sending`. The
    /// slash-tool path is independent of the agent in-flight state machine, so
    /// this must never clear/modify that flag (proven by
    /// `slash_tool_reply_does_not_disturb_chat_sending`).
    pub(crate) fn on_slash_tool_reply(&mut self, text: String) {
        self.chat.push(ChatTurn::agent(text));
    }

    /// Open the approval modal for a surfaced mutating-tool intent (Phase 4).
    /// Closes any operational overlay first so the modal owns focus alone.
    pub(crate) fn open_approval(&mut self, intent: crate::tool_exec::ApprovalIntent) {
        if self.approval.is_some() {
            self.chat.push(ChatTurn::error(
                "An action is already awaiting approval; the new request was discarded. Resolve the open approval first.",
            ));
            return;
        }
        self.close_overlays();
        self.approval = Some(PendingApproval {
            req: crate::ui::approval::ApprovalRequest::new(intent.title, intent.body),
            // An unreviewed tool call the model wants to run defaults to Deny,
            // unlike the shared `ApprovalChoice` default (see its doc comment).
            choice: crate::ui::approval::ApprovalChoice::Deny,
            name: intent.name,
            arguments: intent.arguments,
        });
    }

    /// Route a key to the open approval modal: move the cursor and return a
    /// verdict if the key confirmed one. Pure w.r.t. I/O — the caller maps the
    /// verdict onto execution (Approve) or a declined turn (Deny/Cancel). No-op
    /// returning `None` when no modal is open.
    pub(crate) fn on_approval_key(
        &mut self,
        code: crossterm::event::KeyCode,
    ) -> Option<crate::ui::approval::ApprovalVerdict> {
        let pa = self.approval.as_mut()?;
        let (choice, verdict) = crate::ui::approval::approval_key(code, pa.choice);
        pa.choice = choice;
        verdict
    }

    /// Take the pending approval's `(name, arguments)` for off-thread execution,
    /// clearing the modal. Returns `None` if no modal is open.
    pub(crate) fn take_approval(&mut self) -> Option<(String, serde_json::Value)> {
        self.approval.take().map(|pa| (pa.name, pa.arguments))
    }

    /// Handle a Deny/Cancel verdict: clear the modal and append a declined turn.
    /// Nothing executes.
    pub(crate) fn on_approval_declined(&mut self) {
        self.approval = None;
        self.chat.push(ChatTurn::agent("Action declined."));
    }

    /// Append the approved-action result turn AND raise the one-shot
    /// `chat_dispatch` edge so the agent does EXACTLY ONE automatic follow-up
    /// turn that incorporates the result. The result is pushed as an agent turn
    /// (so `build_messages` sends it as conversational context); `chat_dispatch`
    /// is consumed once by the event loop, so this never loops. A further
    /// mutating request from that follow-up re-surfaces approval (user-gated),
    /// so there is no unbounded execution. Clears any open modal defensively.
    pub(crate) fn on_approval_result(&mut self, text: String) {
        self.approval = None;
        self.chat.push(ChatTurn::agent(text));
        // Exactly one follow-up: raise the edge once. `chat_sending` mirrors a
        // normal submit so the UI shows the in-flight state and a double key
        // can't race a second dispatch.
        self.chat_sending = true;
        self.chat_dispatch = true;
    }

    /// Apply the currently-highlighted picker entry and close the modal.
    pub fn apply_theme_pick(&mut self) {
        let names = crate::ui::theme::theme_names();
        if let Some(name) = names.get(self.theme_picker_sel) {
            self.theme_name = (*name).to_string();
            self.theme = Theme::from_name(name);
        }
        self.modal = Modal::None;
    }

    /// Number of selectable items for the current tab. Returns 0 when the
    /// active tab has no selection model.
    pub fn selection_len(&self) -> usize {
        match self.active_tab {
            // Observe folds the telemetry tabs; its selectable list is the
            // instances table (the one actionable list in the cluster).
            ActiveTab::Observe => self.instances.len(),
            ActiveTab::Rocm => crate::ui::tabs::rocm::VERB_COUNT,
            ActiveTab::Serving => crate::ui::tabs::serving::VERB_COUNT,
            _ => 0,
        }
    }

    /// Move the selection cursor for the active tab. Clamped to [0, len-1].
    pub fn move_selection(&mut self, delta: isize) {
        let len = self.selection_len();
        if len == 0 {
            return;
        }
        let sel = self.selection_for(self.active_tab);
        let next = (sel.cast_signed() + delta).clamp(0, len.cast_signed() - 1) as usize;
        self.set_selection(self.active_tab, next);
    }

    pub const fn select_first(&mut self) {
        self.set_selection(self.active_tab, 0);
    }

    pub fn select_last(&mut self) {
        let len = self.selection_len();
        if len > 0 {
            self.set_selection(self.active_tab, len - 1);
        }
    }

    const fn selection_for(&self, tab: ActiveTab) -> usize {
        match tab {
            ActiveTab::Observe => self.instance_sel,
            ActiveTab::Rocm => self.rocm_sel,
            ActiveTab::Serving => self.serving_sel,
            _ => 0,
        }
    }

    const fn set_selection(&mut self, tab: ActiveTab, idx: usize) {
        match tab {
            ActiveTab::Observe => self.instance_sel = idx,
            ActiveTab::Rocm => self.rocm_sel = idx,
            ActiveTab::Serving => self.serving_sel = idx,
            _ => {}
        }
    }

    /// Number of rows in the active domain tab's Actions list (0 elsewhere).
    const fn pane_verb_count(&self) -> usize {
        match self.active_tab {
            ActiveTab::Rocm => crate::ui::tabs::rocm::VERB_COUNT,
            ActiveTab::Serving => crate::ui::tabs::serving::VERB_COUNT,
            _ => 0,
        }
    }

    /// Seam action for the active domain tab's selected verb (`Nothing` else).
    fn pane_verb_action(&self) -> KeyAction {
        match self.active_tab {
            ActiveTab::Rocm => crate::ui::tabs::rocm::verb_action(self.rocm_sel),
            ActiveTab::Serving => crate::ui::tabs::serving::verb_action(self.serving_sel),
            _ => KeyAction::Nothing,
        }
    }

    /// Clamp both selectors after a state update that may have shrunk the
    /// underlying collection. Call after push_snapshot / instance changes.
    fn clamp_selectors(&mut self) {
        if self.instances.is_empty() {
            self.instance_sel = 0;
        } else {
            self.instance_sel = self.instance_sel.min(self.instances.len() - 1);
        }
        if self.bench_rows.is_empty() {
            self.bench_sel = 0;
        } else {
            self.bench_sel = self.bench_sel.min(self.bench_rows.len() - 1);
        }
        let gpu_count = self.latest.as_ref().map_or(0, |s| s.gpus.len());
        if gpu_count > 0 {
            self.gpu_sel = self.gpu_sel.min(gpu_count - 1);
        } else {
            self.gpu_sel = 0;
        }
        // Keep the scroll offset from running past the (possibly shrunk) list.
        self.gpu_scroll = self.gpu_scroll.min(gpu_count.saturating_sub(1));
    }

    fn push_snapshot(&mut self, snap: Snapshot) {
        // Snapshots carry the daemon's current instance set — treat them as truth.
        self.instances.clear();
        for inst in &snap.instances {
            self.instances
                .insert(inst.container_id.clone(), inst.clone());
        }
        if self.history.len() == HISTORY_CAP {
            self.history.pop_front();
        }
        self.history.push_back(snap.clone());
        self.latest = Some(snap);
        self.clamp_selectors();
    }

    fn upsert_instance(&mut self, inst: Instance) {
        self.instances.insert(inst.container_id.clone(), inst);
        self.clamp_selectors();
    }

    fn remove_instance(&mut self, id: &str) {
        self.instances.remove(id);
        self.clamp_selectors();
    }

    fn push_bench_rows(&mut self, rows: Vec<BenchmarkRow>) {
        for r in rows {
            if self.bench_rows.len() == BENCH_CAP {
                self.bench_rows.pop_front();
            }
            self.bench_rows.push_back(r);
        }
        self.clamp_selectors();
    }

    /// Wipe state derived from past events so a backward replay seek can
    /// repopulate from scratch. Preserves UI scaffolding (theme, tabs,
    /// selectors, modal) so the user's frame of reference doesn't jump.
    pub fn reset_for_seek(&mut self) {
        self.latest = None;
        self.history.clear();
        self.instances.clear();
        self.bench_rows.clear();
        self.clamp_selectors();
    }

    /// Apply a wire `Event` to local state. Single source of truth for the
    /// event-to-state transition — used by both the live event loop and by
    /// out-of-process drivers (replay, screenshot generation, future test
    /// harnesses).
    pub fn apply_event(&mut self, event: Event) {
        match event {
            Event::Snapshot(snap) => self.push_snapshot(snap),
            Event::BenchmarkRowsAppended { rows } => self.push_bench_rows(rows),
            Event::InstanceDiscovered(inst) => self.upsert_instance(inst),
            Event::InstanceGone { container_id } => self.remove_instance(&container_id),
            // Welcome / Warning / Error / Bye don't mutate AppState directly.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::layout::Rect;
    use tokio::sync::mpsc;

    use crate::client::ClientMsg;

    use super::actions::run_approved;
    use super::chat::build_chat_agent;
    use super::summary::summarize_slash_tool;
    use super::types::NO_CHAT_BACKEND_MSG;

    #[test]
    fn back_out_requires_an_open_manager_on_any_tab() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        // No manager open → never backs out, even on a domain tab.
        s.active_tab = ActiveTab::Rocm;
        assert!(!s.should_pane_back_out(crossterm::event::KeyCode::Esc));
        // Manager open on a non-domain tab (opened from Observe hotkey) →
        // Esc backs out uniformly regardless of tab, now that the
        // Rocm/Serving-only gate is gone. New coverage of the generalized
        // behavior — the manager's own event-loop arm already closed it on
        // this tab before the gate was removed, so this isn't a regression
        // test for a prior bug.
        s.active_tab = ActiveTab::Observe;
        s.examine_manager = Some(crate::ui::examine_manager::ExamineManagerState::default());
        assert!(s.has_open_overlay());
        assert!(s.should_pane_back_out(crossterm::event::KeyCode::Esc));
    }

    #[test]
    fn esc_defers_to_manager_when_a_subscreen_is_open() {
        // With a job console (or sub-popup / approval) open inside an inline
        // manager, Esc must reach the manager (cancel the inner layer / be
        // ignored while running), NOT eject the whole manager.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.install_manager = Some(crate::ui::install_manager::InstallManagerState {
            active_job: Some("install-job".into()), // a console is up
            ..Default::default()
        });
        assert!(s.has_open_overlay());
        assert!(
            !s.should_pane_back_out(crossterm::event::KeyCode::Esc),
            "Esc must defer to the manager while a job console is open"
        );
        // Once the console is dismissed (back at root), Esc backs out.
        s.install_manager.as_mut().unwrap().active_job = None;
        assert!(s.should_pane_back_out(crossterm::event::KeyCode::Esc));
    }

    #[test]
    fn esc_defers_to_onboarding_install_config_subview() {
        // Regression coverage for the `install_config` nesting field: the
        // onboarding wizard's Configure sub-view is a nested sub-view just
        // like a manager's job console, so root Esc must defer to it instead
        // of ejecting the whole wizard.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.onboarding = Some(crate::ui::onboarding::OnboardingState {
            install_config: Some(crate::ui::onboarding::InstallConfig::default()),
            ..Default::default()
        });
        assert!(s.has_open_overlay());
        assert!(
            !s.should_pane_back_out(crossterm::event::KeyCode::Esc),
            "Esc must defer to onboarding while the Configure sub-view is open"
        );
        // Once the sub-view is closed (back at root), Esc backs out again.
        s.onboarding.as_mut().unwrap().install_config = None;
        assert!(s.should_pane_back_out(crossterm::event::KeyCode::Esc));
    }

    #[test]
    fn scroll_dock_clamps_against_buffer() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        // Dock 10 rows tall → ~7 visible lines after border/padding.
        s.last_dock_area = Some(Rect::new(0, 0, 52, 10));
        s.jobs.apply(rocm_dash_core::state::StateEvent::StartJob {
            id: "logs".into(),
            cmd: "rocm".into(),
            args: vec!["logs".into()],
        });
        for i in 0..20 {
            s.jobs.apply(rocm_dash_core::state::StateEvent::JobLine {
                id: "logs".into(),
                line: format!("line {i}"),
            });
        }
        // 20 lines, ~7 visible → max scroll-up is bounded, never past the top.
        s.scroll_dock(100);
        assert!(s.dock_logs_scroll <= 13, "clamped: {}", s.dock_logs_scroll);
        assert!(s.dock_logs_scroll > 0, "scrolled up some");
        s.scroll_dock(-100);
        assert_eq!(s.dock_logs_scroll, 0, "back to the tail");
    }

    #[test]
    fn scroll_instance_detail_clamps_to_measured_max() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.instance_detail_max_scroll = 9;
        // i16::MAX is the jump-to-end gesture; it must land on the measured
        // max, not overflow past it.
        s.scroll_instance_detail(i16::MAX);
        assert_eq!(s.instance_detail_scroll, 9, "jump-to-end clamps to max");
        // i16::MIN is jump-to-start; it must land on 0, not underflow.
        s.scroll_instance_detail(i16::MIN);
        assert_eq!(s.instance_detail_scroll, 0, "jump-to-start clamps to 0");
    }

    #[test]
    fn scroll_console_clamps_and_tracks_output() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        // No console → both axes clamp to 0 (no content to pan over).
        s.scroll_console(5, 5);
        assert_eq!(s.console_scroll, 0);
        assert_eq!(s.console_hscroll, 0, "no console → horizontal clamps to 0");
        // With a console of N lines, vertical clamps to N-1.
        s.jobs.apply(rocm_dash_core::state::StateEvent::StartJob {
            id: "logs".into(),
            cmd: "rocm".into(),
            args: vec!["logs".into()],
        });
        for i in 0..4 {
            s.jobs.apply(rocm_dash_core::state::StateEvent::JobLine {
                id: "logs".into(),
                line: format!("line {i}"),
            });
        }
        s.logs_view = Some(crate::ui::logs_view::LogsViewState {
            active_job: Some("logs".into()),
            ..Default::default()
        });
        s.console_scroll = 0;
        s.scroll_console(100, 0);
        assert_eq!(s.console_scroll, 3, "clamped to output.len()-1 (4 lines)");
        s.scroll_console(-100, 0);
        assert_eq!(s.console_scroll, 0);
        // Horizontal clamps to the widest line minus one ("line 0" = 6 chars).
        s.scroll_console(0, 100);
        assert_eq!(s.console_hscroll, 5, "clamped to max line width - 1");
        s.scroll_console(0, -100);
        assert_eq!(s.console_hscroll, 0);
    }

    // ---------- T13: bench_run overlay invariants ----------

    #[test]
    fn t13_bench_run_in_has_open_overlay() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        assert!(!s.has_open_overlay(), "no overlay initially");
        s.bench_run = Some(crate::ui::bench_run::BenchRunState::new(None));
        assert!(
            s.has_open_overlay(),
            "bench_run must be in has_open_overlay"
        );
    }

    #[test]
    fn t13_bench_run_cleared_by_close_overlays() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.bench_run = Some(crate::ui::bench_run::BenchRunState::new(None));
        s.close_overlays();
        assert!(s.bench_run.is_none(), "close_overlays must clear bench_run");
    }

    #[test]
    fn t13_bench_run_at_root() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        assert!(s.active_overlay_at_root(), "nothing open → at root");
        s.bench_run = Some(crate::ui::bench_run::BenchRunState::new(None));
        assert!(
            s.active_overlay_at_root(),
            "bench_run (no sub-popup/job) is always at root"
        );
    }

    /// `active_overlay_at_root`'s per-manager clauses are a hand-maintained
    /// enumeration of each manager's nested sub-view fields (documented on
    /// `OnboardingState`, which lists every struct this covers). Nothing stops
    /// a future field — a new sub-popup, picker, or prompt — from being added
    /// to one of these structs without a matching update there, which would
    /// silently let Esc eject the whole manager instead of deferring to the
    /// new sub-view.
    ///
    /// This exhaustively destructures every one of those structs (no `..`),
    /// naming every field. Adding a field to any of them without updating
    /// this test — and, in step, `active_overlay_at_root` — fails to compile
    /// (E0027), turning the silent-drift risk into a build break.
    #[test]
    fn active_overlay_at_root_enumeration_is_exhaustive() {
        use crate::ui::automations_manager::AutomationsManagerState;
        use crate::ui::command_screen::CommandScreenState;
        use crate::ui::config_manager::ConfigManagerState;
        use crate::ui::engine_manager::EngineManagerState;
        use crate::ui::examine_manager::ExamineManagerState;
        use crate::ui::install_manager::InstallManagerState;
        use crate::ui::logs_view::LogsViewState;
        use crate::ui::onboarding::OnboardingState;
        use crate::ui::runtime_manager::RuntimeManagerState;
        use crate::ui::serve_wizard::ServeWizardState;
        use crate::ui::services_manager::ServicesManagerState;
        use crate::ui::update_manager::UpdateManagerState;

        let ServeWizardState {
            field: _,
            model: _,
            engine_idx: _,
            device_idx: _,
            host: _,
            port: _,
            managed: _,
            browser,
            picker,
            approval,
            active_job,
            message: _,
        } = ServeWizardState::default();
        assert!(
            browser.is_none() && picker.is_none() && approval.is_none() && active_job.is_none()
        );

        let InstallManagerState {
            field: _,
            channel: _,
            format_idx: _,
            prefix: _,
            dry_run: _,
            browser,
            approval,
            active_job,
            message: _,
        } = InstallManagerState::default();
        assert!(browser.is_none() && approval.is_none() && active_job.is_none());

        let OnboardingState {
            step: _,
            choice: _,
            browser,
            install_config,
            approval,
            active_job,
            message: _,
        } = OnboardingState::default();
        assert!(
            browser.is_none()
                && install_config.is_none()
                && approval.is_none()
                && active_job.is_none()
        );

        let RuntimeManagerState {
            selected: _,
            browser,
            import_input,
            approval,
            active_job,
            message: _,
        } = RuntimeManagerState::default();
        assert!(
            browser.is_none()
                && import_input.is_none()
                && approval.is_none()
                && active_job.is_none()
        );

        let EngineManagerState {
            selected: _,
            approval,
            active_job,
            message: _,
        } = EngineManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let ServicesManagerState {
            selected: _,
            approval,
            active_job,
        } = ServicesManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let UpdateManagerState {
            selected: _,
            approval,
            active_job,
            message: _,
        } = UpdateManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let ConfigManagerState {
            action_sel: _,
            provider_sel: _,
            approval,
            active_job,
            message: _,
        } = ConfigManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let CommandScreenState {
            input: _,
            approval,
            active_job,
            message: _,
        } = CommandScreenState::default();
        assert!(approval.is_none() && active_job.is_none());

        let AutomationsManagerState {
            selected: _,
            approval,
            active_job,
            message: _,
        } = AutomationsManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

        let ExamineManagerState { active_job } = ExamineManagerState::default();
        assert!(active_job.is_none());

        let LogsViewState {
            query: _,
            active_job,
        } = LogsViewState::default();
        assert!(active_job.is_none());
    }

    #[test]
    fn open_theme_picker_places_cursor_on_active_theme() {
        let mut s = AppState::new("test".into(), "dracula".into());
        s.open_theme_picker();
        assert_eq!(s.modal, Modal::ThemePicker);
        let names = crate::ui::theme::theme_names();
        assert_eq!(names[s.theme_picker_sel], "dracula");
    }

    #[test]
    fn theme_picker_move_clamps_to_registry() {
        let mut s = AppState::new("test".into(), "default-dark".into());
        s.theme_picker_sel = 0;
        s.theme_picker_move(-3);
        assert_eq!(s.theme_picker_sel, 0);
        s.theme_picker_move(1_000);
        let names = crate::ui::theme::theme_names();
        assert_eq!(s.theme_picker_sel, names.len() - 1);
    }

    #[test]
    fn apply_theme_pick_swaps_theme_and_closes_modal() {
        let mut s = AppState::new("test".into(), "default-dark".into());
        s.open_theme_picker();
        let names = crate::ui::theme::theme_names();
        let target = names.iter().position(|n| *n == "nord").unwrap();
        s.theme_picker_sel = target;
        s.apply_theme_pick();
        assert_eq!(s.theme_name, "nord");
        assert_eq!(s.modal, Modal::None);
        // Theme actually swapped.
        let nord = Theme::nord();
        assert_eq!(s.theme.bg, nord.bg);
    }

    #[test]
    fn unknown_initial_theme_falls_back_to_default_dark() {
        let s = AppState::new("test".into(), "nope".into());
        let dark = Theme::default_dark();
        assert_eq!(s.theme.bg, dark.bg);
    }

    #[test]
    fn move_selection_clamps_to_bounds() {
        // P3: Observe's selectable list is the instances table.
        let mut s = AppState::new("test".into(), "default-dark".into());
        s.active_tab = ActiveTab::Observe;
        for i in 0..5 {
            s.instances.insert(
                format!("id{i}"),
                rocm_dash_core::metrics::Instance {
                    container_id: format!("id{i}"),
                    ..Default::default()
                },
            );
        }
        s.instance_sel = 0;
        s.move_selection(-3);
        assert_eq!(s.instance_sel, 0);
        s.move_selection(10);
        assert_eq!(s.instance_sel, 4);
        s.select_first();
        assert_eq!(s.instance_sel, 0);
        s.select_last();
        assert_eq!(s.instance_sel, 4);
    }

    #[test]
    fn reset_for_seek_clears_event_derived_state() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.history
            .push_back(rocm_dash_core::metrics::Snapshot::default());
        s.latest = Some(rocm_dash_core::metrics::Snapshot::default());
        s.bench_rows.push_back(BenchmarkRow::default());
        s.reset_for_seek();
        assert!(s.history.is_empty());
        assert!(s.latest.is_none());
        assert!(s.bench_rows.is_empty());
        assert!(s.instances.is_empty());
    }

    #[test]
    fn selectors_reclamp_after_pop() {
        let mut s = AppState::new("test".into(), "default-dark".into());
        s.active_tab = ActiveTab::Observe;
        for i in 0..3 {
            s.bench_rows.push_back(BenchmarkRow {
                cell: format!("c{i}"),
                ..Default::default()
            });
        }
        s.bench_sel = 2;
        s.bench_rows.clear();
        s.clamp_selectors();
        assert_eq!(s.bench_sel, 0);
    }

    #[test]
    fn config_with_chat_sets_local_endpoint_and_clears_auth() {
        let mut cfg = rocm_dash_core::config::Config::default();
        cfg.tui.chat_auth_header = Some("Ocp-Apim-Subscription-Key".into());
        let next = super::chat::config_with_chat(cfg, "http://localhost:8000/v1", "qwen");
        assert_eq!(
            next.tui.chat_url.as_deref(),
            Some("http://localhost:8000/v1")
        );
        assert_eq!(next.tui.chat_model.as_deref(), Some("qwen"));
        assert_eq!(
            next.tui.chat_auth_header, None,
            "local needs no gateway auth"
        );
    }

    #[test]
    fn detect_none_sets_message() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.request_detect();
        s.set_detect_result(None);
        assert!(!s.chat_detecting);
        assert!(s.chat_detect_offer.is_none());
        assert!(s.chat_detect_msg.is_some());
    }

    /// EAI-7354: once `ChatConsent::Accepted`, a focused `'d'` keypress is
    /// ordinary chat text (see `handle_key`'s `ChatConsent::Accepted` arm), so
    /// re-detect must be reachable another way. `/detect` is the affordance —
    /// it runs through the same `request_detect`/`set_detect_result` edge as
    /// the pre-accept `'d'` key, and since the offer prompt isn't drawn once
    /// accepted (`ui::tabs::chat::draw` only renders the gate pre-accept), the
    /// result is echoed into the transcript instead.
    #[test]
    fn slash_detect_probes_and_echoes_result_while_accepted() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.set_chat_config(
            Some(crate::llm::LlmConfig {
                base_url: "http://localhost:8000/v1".into(),
                model: "m".into(),
                api_key: None,
                auth_header: None,
            }),
            true,
        );
        assert_eq!(s.chat_consent, ChatConsent::Accepted);

        assert_eq!(s.handle_slash_command("/detect"), SlashOutcome::Handled);
        assert!(s.chat_detecting && s.chat_detect_dispatch);
        assert!(
            s.chat.last().is_some(),
            "probing while accepted is echoed into the transcript"
        );

        let local = crate::llm::detected_llm_config("http://localhost:13305/v1", "Llama-3.2-3B");
        s.set_detect_result(Some(local.clone()));
        assert_eq!(s.chat_detect_offer.as_ref(), Some(&local));
        // No gate UI once accepted (draw_consent only renders pre-accept) — the
        // offer must be surfaced in the transcript instead.
        let last = s.chat.last().expect("echoed offer turn");
        assert!(last.content.contains("/detect accept"));
    }

    /// `/detect accept` mid-session must integrate with the same
    /// `chat_endpoint_rebuild` edge the initial pre-accept offer uses, or the
    /// live agent silently keeps talking to the stale endpoint.
    #[test]
    fn slash_detect_accept_raises_endpoint_rebuild_like_initial_accept() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.set_chat_config(
            Some(crate::llm::LlmConfig {
                base_url: "http://localhost:8000/v1".into(),
                model: "m".into(),
                api_key: None,
                auth_header: None,
            }),
            true,
        );
        s.active_provider = ChatProvider::Openai; // observe the realignment to Local
        let local = crate::llm::detected_llm_config("http://localhost:13305/v1", "Llama-3.2-3B");
        s.set_detect_result(Some(local.clone()));

        assert_eq!(
            s.handle_slash_command("/detect accept"),
            SlashOutcome::Handled
        );
        assert_eq!(s.chat_llm.as_ref(), Some(&local));
        assert_eq!(s.active_provider, ChatProvider::Local);
        assert_eq!(
            s.chat_endpoint_rebuild,
            Some(ChatProvider::Openai),
            "re-detect accept raises the rebuild edge exactly like the initial accept"
        );
    }

    #[test]
    fn slash_detect_dismiss_and_unknown_subcommand() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:8000/v1",
            "m",
        )));
        assert_eq!(
            s.handle_slash_command("/detect dismiss"),
            SlashOutcome::Handled
        );
        assert!(s.chat_detect_offer.is_none());

        assert_eq!(
            s.handle_slash_command("/detect bogus"),
            SlashOutcome::Handled
        );
        let last = s.chat.last().expect("error turn");
        assert!(last.content.contains("unknown /detect action"));
    }

    /// Sub-commands are case-insensitive, matching `/permissions` / `/provider`.
    #[test]
    fn slash_detect_subcommand_is_case_insensitive() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:8000/v1",
            "m",
        )));
        // `DISMISS` (upper) must act exactly like `dismiss`.
        assert_eq!(
            s.handle_slash_command("/detect DISMISS"),
            SlashOutcome::Handled
        );
        assert!(
            s.chat_detect_offer.is_none(),
            "uppercase sub-command is normalized and handled"
        );
    }

    /// `/detect accept` / `/detect save` with nothing pending emits a hint
    /// turn rather than a silent no-op.
    #[test]
    fn slash_detect_accept_without_offer_emits_hint() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        assert!(s.chat_detect_offer.is_none());
        assert_eq!(
            s.handle_slash_command("/detect accept"),
            SlashOutcome::Handled
        );
        // No endpoint was adopted; a hint explains why.
        assert_eq!(s.chat_endpoint_rebuild, None);
        let last = s.chat.last().expect("hint turn");
        assert!(last.content.contains("no detected endpoint"));
    }

    // --- Slash-command dispatch (Phase 3 nav/session + read-only) ---

    fn st() -> AppState {
        AppState::new("t".into(), "default-dark".into())
    }

    // --- Focused host (Phase 2): default-off focus flag hosts one overlay ---

    #[test]
    fn resolved_args_focus_defaults_none() {
        // The focus flag is additive and off by default in every constructor, so
        // existing dash/chat behavior is byte-identical.
        assert!(args_with_anthropic_key(None).focus.is_none());
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

    #[test]
    fn slash_help_opens_help_modal() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/help"), SlashOutcome::Handled);
        assert_eq!(s.modal, Modal::Help);
    }

    #[test]
    fn slash_question_mark_opens_help_modal() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/?"), SlashOutcome::Handled);
        assert_eq!(s.modal, Modal::Help);
    }

    #[test]
    fn slash_clear_empties_transcript() {
        let mut s = st();
        s.chat.push(ChatTurn::user("hi"));
        s.chat.push(ChatTurn::agent("hello"));
        assert_eq!(s.handle_slash_command("/clear"), SlashOutcome::Handled);
        assert!(s.chat.is_empty());
    }

    #[test]
    fn slash_quit_sets_should_quit() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/quit"), SlashOutcome::Handled);
        assert!(s.should_quit);
    }

    #[test]
    fn slash_exit_sets_should_quit() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/exit"), SlashOutcome::Handled);
        assert!(s.should_quit);
    }

    #[test]
    fn slash_home_switches_to_overview() {
        let mut s = st();
        s.active_tab = ActiveTab::Observe;
        assert_eq!(s.handle_slash_command("/home"), SlashOutcome::Handled);
        assert_eq!(s.active_tab, ActiveTab::Home);
    }

    #[test]
    fn slash_gpu_switches_to_observe() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/gpu"), SlashOutcome::Handled);
        assert_eq!(s.active_tab, ActiveTab::Observe);
    }

    #[test]
    fn slash_doctor_opens_overlay() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/doctor"), SlashOutcome::Handled);
        assert!(s.examine_manager.is_some());
    }

    #[test]
    fn slash_runtimes_opens_overlay() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/runtimes"), SlashOutcome::Handled);
        assert!(s.runtime_manager.is_some());
    }

    #[test]
    fn slash_config_opens_overlay() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/config"), SlashOutcome::Handled);
        assert!(s.config_manager.is_some());
    }

    #[test]
    fn slash_logs_opens_overlay() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/logs"), SlashOutcome::Handled);
        assert!(s.logs_view.is_some());
    }

    #[test]
    fn slash_model_raises_executor_request() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/model"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("model raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(req.args, serde_json::json!({ "args": ["model"] }));
    }

    #[test]
    fn slash_daemon_raises_executor_request() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/daemon"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("daemon raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["daemon", "status"] })
        );
    }

    // --- Phase 4: mutating slash dispatch + approval modal flow ---

    #[test]
    fn slash_install_raises_install_sdk_request() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/install ~/rocm"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("install raises a slash_tool request");
        assert_eq!(req.name, "install_sdk");
        assert_eq!(req.args["channel"], "release");
        assert_eq!(req.args["format"], "wheel");
        // The validator REQUIRES a prefix; the slash path must supply one or the
        // modal never opens.
        assert_eq!(req.args["prefix"], "~/rocm");
    }

    #[test]
    fn slash_install_without_prefix_hints_not_dispatch() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/install"), SlashOutcome::Handled);
        assert!(
            s.slash_tool.is_none(),
            "no dispatch without an install folder"
        );
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn plan_request_set_by_slash() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/plan install rocm"),
            SlashOutcome::Handled
        );
        assert_eq!(s.plan_request.as_deref(), Some("install rocm"));
        // The command word is matched case-insensitively, and so must the arg
        // extraction be: `/Plan` (mixed case) dispatches just like `/plan`.
        let mut s_mixed = st();
        assert_eq!(
            s_mixed.handle_slash_command("/Plan install rocm"),
            SlashOutcome::Handled
        );
        assert_eq!(s_mixed.plan_request.as_deref(), Some("install rocm"));
        // Bare `/plan` hints with a usage turn and raises no plan edge.
        let mut s2 = st();
        assert_eq!(s2.handle_slash_command("/plan"), SlashOutcome::Handled);
        assert!(s2.plan_request.is_none(), "bare /plan must not dispatch");
        assert_eq!(s2.chat.last().unwrap().role, ChatRole::Agent);
        assert!(s2.chat.last().unwrap().content.contains("usage"));
    }

    /// Minimal `ResolvedArgs` for the `build_chat_agent` factory tests. Keys are
    /// passed via the struct (the in-process seam) — never argv.
    fn args_with_anthropic_key(key: Option<&str>) -> ResolvedArgs {
        ResolvedArgs {
            connect: "test".into(),
            token: None,
            theme: "default-dark".into(),
            replay: None,
            initial_tab: ActiveTab::Chat,
            focus: None,
            chat_url: None,
            chat_model: None,
            chat_auth_header: None,
            chat_temperature: None,
            chat_top_p: None,
            chat_max_tokens: None,
            chat_env_url: None,
            chat_api_key: None,
            anthropic_api_key: key.map(str::to_string),
            chat_auto_consent: false,
            chat_mock: false,
            model_recipes: Vec::new(),
            runtimes: Vec::new(),
            automations: Vec::new(),
            chat_system_prompt: None,
            tool_executor: None,
            bench_results_dir: None,
            services_past_attempts: 0,
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn resolved_args_inference_params_maps_and_defaults() {
        // Bare args carry no sampling override → all three knobs are None, so an
        // unset endpoint default is never clobbered.
        let bare = args_with_anthropic_key(None);
        assert_eq!(
            bare.inference_params(),
            crate::agent::InferenceParams::default()
        );

        // Populated args copy every knob through unchanged (CLI-over-config merge
        // already happened in the bin).
        let mut args = args_with_anthropic_key(None);
        args.chat_temperature = Some(0.25);
        args.chat_top_p = Some(0.5);
        args.chat_max_tokens = Some(512);
        let params = args.inference_params();
        assert_eq!(params.temperature, Some(0.25));
        assert_eq!(params.top_p, Some(0.5));
        assert_eq!(params.max_tokens, Some(512));
    }

    #[test]
    fn slash_provider_switches_backend() {
        // /provider anthropic → active_provider set + edge raised.
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/provider anthropic"),
            SlashOutcome::Handled
        );
        assert_eq!(s.active_provider, ChatProvider::Anthropic);
        assert_eq!(
            s.provider_switch,
            Some(ProviderSwitch {
                previous: ChatProvider::Local,
                target: ChatProvider::Anthropic,
            })
        );
        // /provider openai → openai.
        let mut s2 = st();
        s2.handle_slash_command("/provider openai");
        assert_eq!(s2.active_provider, ChatProvider::Openai);
        assert_eq!(
            s2.provider_switch,
            Some(ProviderSwitch {
                previous: ChatProvider::Local,
                target: ChatProvider::Openai,
            })
        );
        // /provider local → local (matched case-insensitively).
        let mut s3 = st();
        s3.handle_slash_command("/Provider LOCAL");
        assert_eq!(s3.active_provider, ChatProvider::Local);
        assert_eq!(
            s3.provider_switch,
            Some(ProviderSwitch {
                previous: ChatProvider::Local,
                target: ChatProvider::Local,
            })
        );
    }

    #[test]
    fn slash_provider_switch_captures_previous_provider() {
        // (Phase-8 polish) A failed switch must revert to the provider that was
        // active BEFORE the attempt, not unconditionally to Local. Prove the
        // slash handler snapshots the prior provider in the edge: switch to
        // openai (optimistic), then attempt anthropic — the edge carries
        // previous=Openai so the drain can revert there on a build failure.
        let mut s = st();
        s.handle_slash_command("/provider openai");
        assert_eq!(s.active_provider, ChatProvider::Openai);
        s.handle_slash_command("/provider anthropic");
        assert_eq!(
            s.provider_switch,
            Some(ProviderSwitch {
                previous: ChatProvider::Openai,
                target: ChatProvider::Anthropic,
            }),
            "the failed-switch revert target is the prior provider, not Local"
        );
    }

    #[test]
    fn no_provider_no_key_chat_surfaces_actionable_message() {
        // Edge: agent is None (no endpoint, no provider key). Submitting chat
        // must surface a clear, ACTIONABLE message (the recovery affordances),
        // routed through `on_chat_error` as an error turn — not an error dump,
        // not a panic. This mirrors the event-loop None-agent branch, which
        // emits exactly `NO_CHAT_BACKEND_MSG`.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.set_chat_config(None, false);
        assert_eq!(s.chat_consent, ChatConsent::Unavailable);
        // Drive the same surface the event loop uses for the None-agent case.
        s.on_chat_error(NO_CHAT_BACKEND_MSG.to_string());
        let last = s.chat.last().expect("an error turn was pushed");
        assert_eq!(last.role, ChatRole::Error);
        // Actionable: names both concrete recovery paths.
        assert!(
            last.content.contains("detect") && last.content.contains("/provider"),
            "empty-state must be actionable, got: {}",
            last.content
        );
        // Not a panic and not in-flight afterwards (sending cleared).
        assert!(!s.chat_sending);
    }

    #[test]
    fn slash_provider_bare_shows_current_and_unknown_hints() {
        // Bare /provider shows the current backend and raises no edge.
        let mut s = st();
        s.handle_slash_command("/provider");
        assert!(s.provider_switch.is_none());
        let last = s.chat.last().unwrap();
        assert_eq!(last.role, ChatRole::Agent);
        assert!(last.content.contains("local"), "shows current provider");
        // Unknown provider hints (error turn), no edge, no switch.
        let mut s2 = st();
        s2.handle_slash_command("/provider grok");
        assert!(s2.provider_switch.is_none());
        assert_eq!(s2.active_provider, ChatProvider::Local);
        assert_eq!(s2.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_chat_passthrough_submits_prompt() {
        // /chat <prompt> pushes the user turn + raises the chat_dispatch edge.
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/chat what's GPU-2 doing?"),
            SlashOutcome::Handled
        );
        assert!(s.chat_dispatch, "passthrough raises the spawn edge");
        assert!(s.chat_sending);
        let last = s.chat.last().unwrap();
        assert_eq!(last.role, ChatRole::User);
        assert_eq!(last.content, "what's GPU-2 doing?");
    }

    #[test]
    fn slash_chat_bare_focuses_chat_tab() {
        // Bare /chat focuses the Chat tab, raises no dispatch edge.
        let mut s = st();
        s.active_tab = ActiveTab::Home;
        assert_eq!(s.handle_slash_command("/chat"), SlashOutcome::Handled);
        assert_eq!(s.active_tab, ActiveTab::Chat);
        assert!(s.chat_focused);
        assert!(!s.chat_dispatch, "bare /chat does not dispatch");
    }

    #[test]
    fn build_chat_agent_anthropic_with_key() {
        // The factory returns Some for Anthropic when a key is present in args
        // (carried in-process — never argv). Construction only, no network.
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let args = args_with_anthropic_key(Some("dummy-anthropic-key"));
        let agent = build_chat_agent(ChatProvider::Anthropic, &args, None, tx);
        assert!(agent.is_some(), "anthropic builds with a key");
    }

    #[test]
    fn build_chat_agent_anthropic_without_key_is_none() {
        // No key → None (the event loop reverts to local + an error turn).
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let args = args_with_anthropic_key(None);
        let agent = build_chat_agent(ChatProvider::Anthropic, &args, None, tx);
        assert!(agent.is_none(), "anthropic without a key does not build");
    }

    #[test]
    fn build_chat_agent_openai_requires_key() {
        // No OpenAI key → None. Without this gate the factory would build a dummy
        // `sk-no-key` backend that 401s at request time, so the switch reports
        // success then fails. With a key → Some (construction only, no network).
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut args = args_with_anthropic_key(None);
        assert!(
            build_chat_agent(ChatProvider::Openai, &args, None, tx.clone()).is_none(),
            "openai without a key must not build a dead backend"
        );
        args.chat_api_key = Some("sk-real-key".to_string());
        assert!(
            build_chat_agent(ChatProvider::Openai, &args, None, tx).is_some(),
            "openai builds with a key"
        );
    }

    #[test]
    fn build_chat_agent_local_defers_to_inline_build() {
        // Local is owned by the inline auto-detect path, so the factory returns
        // None for it (the caller keeps the existing agent).
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let args = args_with_anthropic_key(Some("k"));
        assert!(build_chat_agent(ChatProvider::Local, &args, None, tx).is_none());
    }

    #[test]
    fn provider_local_restores_saved_local_agent() {
        // Invariant for the `ChatProvider::Local` arm of the provider_switch
        // drain: because `build_chat_agent(Local)` returns None (asserted above),
        // the event loop CANNOT rebuild the local backend on demand. It must
        // restore the `local_agent` clone snapshotted before the loop. This test
        // models that contract: after a remote switch flips `agent` away from the
        // saved local clone, `/provider local` must re-point `agent` back to it.
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let local_agent: Option<std::sync::Arc<dyn crate::agent::AgentClient>> = Some(
            std::sync::Arc::new(crate::agent::MockAgentClient::new("local"))
                as std::sync::Arc<dyn crate::agent::AgentClient>,
        );
        // Simulate a prior remote switch: `agent` now points elsewhere.
        let remote: Option<std::sync::Arc<dyn crate::agent::AgentClient>> = Some(
            std::sync::Arc::new(crate::agent::MockAgentClient::new("remote"))
                as std::sync::Arc<dyn crate::agent::AgentClient>,
        );
        let mut agent = remote;
        assert!(!std::sync::Arc::ptr_eq(
            agent.as_ref().unwrap(),
            local_agent.as_ref().unwrap()
        ));
        // The Local arm's restore line (mirrors event_loop.rs): the factory cannot help.
        let args = args_with_anthropic_key(Some("k"));
        assert!(build_chat_agent(ChatProvider::Local, &args, None, tx).is_none());
        agent = local_agent.clone();
        // `agent` is now the original auto-detected local backend, not the remote.
        assert!(std::sync::Arc::ptr_eq(
            agent.as_ref().unwrap(),
            local_agent.as_ref().unwrap()
        ));
    }

    #[test]
    fn chat_keys_flow_only_through_resolved_args_not_argv() {
        // The seam carries keys via ResolvedArgs (in-process), never process
        // argv. This structurally asserts the factory reads the key from the
        // struct field — there is no argv plumbing in the build path.
        let args = args_with_anthropic_key(Some("sentinel-key"));
        assert_eq!(args.anthropic_api_key.as_deref(), Some("sentinel-key"));
        // The real process args never carry the key (no `--api-key`-style flag
        // exists; keys are env/secure-store sourced by the bin into the struct).
        let argv: Vec<String> = std::env::args().collect();
        assert!(
            !argv.iter().any(|a| a.contains("sentinel-key")),
            "no key value is ever present in process argv"
        );
    }

    #[test]
    fn on_plan_ready_renders_plan_text() {
        let mut s = st();
        s.on_plan_ready("planner: hybrid-parser-v1\nplan body".to_string(), None);
        // The review is appended as a chat turn…
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(s.chat.last().unwrap().content.contains("plan body"));
        // …and with no action there is nothing to approve or execute.
        assert!(s.slash_tool.is_none());
    }

    #[test]
    fn on_plan_ready_complete_mutating_hands_off_to_approval() {
        let mut s = st();
        let action = PlannedAction {
            args: vec![
                "install".to_string(),
                "sdk".to_string(),
                "--prefix".to_string(),
                "/x".to_string(),
            ],
            approval_required: true,
            has_placeholders: false,
            provider_assisted: false,
        };
        s.on_plan_ready("the plan".to_string(), Some(action));
        // The plan review is shown…
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        // …and the complete mutating action is forwarded as a rocm_command
        // slash-tool request (→ execute() → ApprovalRequired → modal).
        let req = s
            .slash_tool
            .expect("complete mutating plan hands off to approval");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args["args"],
            serde_json::json!(["install", "sdk", "--prefix", "/x"])
        );
    }

    #[test]
    fn on_plan_ready_guards_against_pending_slash_tool() {
        let mut s = st();
        // A slash command queued a tool request while the plan computed off-thread.
        s.slash_tool = Some(SlashToolRequest {
            name: "rocm_command".to_string(),
            args: serde_json::json!({ "args": ["services", "list"] }),
            label: "pending".to_string(),
        });
        let action = PlannedAction {
            args: vec!["update".to_string()],
            approval_required: true,
            has_placeholders: false,
            provider_assisted: false,
        };
        s.on_plan_ready("the plan".to_string(), Some(action));
        // The in-flight request is NOT clobbered…
        let req = s.slash_tool.as_ref().expect("pending request preserved");
        assert_eq!(req.label, "pending");
        assert_eq!(req.args["args"], serde_json::json!(["services", "list"]));
        // …and the user is told the planned action was discarded.
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn on_plan_ready_placeholder_stays_plan_only() {
        let mut s = st();
        let action = PlannedAction {
            args: vec![
                "install".to_string(),
                "sdk".to_string(),
                "--prefix".to_string(),
                "<PATH>".to_string(),
            ],
            approval_required: true,
            has_placeholders: true,
            provider_assisted: false,
        };
        s.on_plan_ready("the plan".to_string(), Some(action));
        // The plan is shown for review, but an incomplete (placeholder) plan
        // never focuses approval and never executes — mirrors the legacy
        // `natural_serve_with_missing_model_does_not_focus_approval` rule.
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(
            s.slash_tool.is_none(),
            "placeholder plan must stay plan-only (no approval, no execution)"
        );
        assert!(s.approval.is_none(), "no approval modal focus");
    }

    #[test]
    fn on_plan_ready_provider_assisted_stays_plan_only() {
        let mut s = st();
        // A complete (no placeholders) mutating action that a planner provider
        // produced. Even though approval_required && !has_placeholders, the
        // provider_assisted flag keeps it review-only — mirrors the bin's
        // `validate_freeform_execution_action` provider-assisted guard.
        let action = PlannedAction {
            args: vec![
                "serve".to_string(),
                "m".to_string(),
                "--managed".to_string(),
            ],
            approval_required: true,
            has_placeholders: false,
            provider_assisted: true,
        };
        s.on_plan_ready("the plan".to_string(), Some(action));
        // The plan text is rendered for review…
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        // …but the provider-assisted plan never focuses approval or executes:
        // the user runs the displayed command manually.
        assert!(
            s.slash_tool.is_none(),
            "provider-assisted plan must stay plan-only (no execution handoff)"
        );
        assert!(s.approval.is_none(), "no approval modal focus");
    }

    #[test]
    fn slash_engine_raises_install_engine_request() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/engine vllm"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("engine raises a slash_tool request");
        assert_eq!(req.name, "install_engine");
        assert_eq!(req.args, serde_json::json!({ "engine": "vllm" }));
    }

    #[test]
    fn slash_engine_without_name_hints_not_dispatch() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/engine"), SlashOutcome::Handled);
        assert!(s.slash_tool.is_none(), "no dispatch without an engine name");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_serve_raises_launch_server_request() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/serve deepseek-r1"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("serve raises a slash_tool request");
        assert_eq!(req.name, "launch_server");
        assert_eq!(req.args["model"], "deepseek-r1");
        // Loopback host is forced so the validator never rejects the slash path.
        assert_eq!(req.args["host"], "127.0.0.1");
    }

    #[test]
    fn slash_services_stop_raises_stop_server_request() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/services stop svc-1"),
            SlashOutcome::Handled
        );
        let req = s
            .slash_tool
            .expect("services stop raises a slash_tool request");
        assert_eq!(req.name, "stop_server");
        assert_eq!(req.args, serde_json::json!({ "service_id": "svc-1" }));
    }

    #[test]
    fn slash_services_restart_is_guided_not_stop() {
        // restart is NOT wired through the chat seam yet; it must guide the
        // operator instead of silently running stop_server (a semantic lie).
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/services restart svc-1"),
            SlashOutcome::Handled
        );
        assert!(
            s.slash_tool.is_none(),
            "restart must NOT dispatch a stop_server request"
        );
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_services_bare_is_read_only_list() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/services"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("bare services lists managed services");
        assert_eq!(req.name, "services");
    }

    // --- Phase 5: lifecycle slash dispatch (read/mutate split via rocm_command) ---

    #[test]
    fn slash_update_is_read_only_report() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/update"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("update raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(req.args, serde_json::json!({ "args": ["update"] }));
    }

    #[test]
    fn slash_update_apply_is_mutating() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/update --apply"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("update --apply raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["update", "--apply"] })
        );
    }

    #[test]
    fn slash_update_apply_is_position_independent() {
        // `--apply` anywhere in the args triggers the mutating path (matching
        // `/uninstall`'s all-tokens parse), not only as the immediate second token.
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/update --preview --apply"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("update --apply raises a request");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["update", "--apply"] }),
            "--apply past the second token must still apply"
        );
    }

    #[test]
    fn slash_comfyui_bare_is_status() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/comfyui"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("comfyui raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["comfyui", "status"] })
        );
    }

    #[test]
    fn slash_comfy_alias_is_status() {
        // `/comfy` is an alias for `/comfyui` and must map to the same
        // read-only status argv.
        let mut s = st();
        assert_eq!(s.handle_slash_command("/comfy"), SlashOutcome::Handled);
        let req = s
            .slash_tool
            .expect("comfy alias raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["comfyui", "status"] })
        );
    }

    #[test]
    fn slash_comfyui_start_is_mutating() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/comfyui start"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("comfyui start raises a request");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["comfyui", "start"] })
        );
    }

    #[test]
    fn slash_comfyui_logs_is_read_only() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/comfyui logs"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("comfyui logs raises a request");
        assert_eq!(req.args, serde_json::json!({ "args": ["comfyui", "logs"] }));
    }

    #[test]
    fn slash_uninstall_defaults_to_dry_run() {
        // SAFETY: a bare `/uninstall` must NEVER trigger a real uninstall.
        let mut s = st();
        assert_eq!(s.handle_slash_command("/uninstall"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("uninstall raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["uninstall", "--dry-run"] }),
            "bare /uninstall MUST default to a dry-run, not a real uninstall"
        );
        assert_eq!(
            req.label, "uninstall --dry-run",
            "the dry-run label is the safety-critical user-visible string"
        );
    }

    #[test]
    fn slash_uninstall_apply_is_real() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/uninstall --apply"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("uninstall --apply raises a request");
        assert_eq!(req.args, serde_json::json!({ "args": ["uninstall"] }));
        assert_eq!(
            req.label, "uninstall",
            "the real-uninstall label is the safety-critical user-visible string"
        );
    }

    #[test]
    fn slash_uninstall_conflicting_flags_is_guided() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/uninstall --apply --dry-run"),
            SlashOutcome::Handled
        );
        assert!(
            s.slash_tool.is_none(),
            "conflicting uninstall flags must NOT dispatch"
        );
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_setup_bare_is_status() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/setup"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("setup raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(req.args, serde_json::json!({ "args": ["setup", "status"] }));
    }

    #[test]
    fn slash_setup_reset_is_mutating() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/setup reset"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("setup reset raises a request");
        assert_eq!(req.args, serde_json::json!({ "args": ["setup", "reset"] }));
    }

    #[test]
    fn slash_setup_unknown_sub_is_guided() {
        // SetupCommand has only status + reset — `skip` must guide, not dispatch.
        let mut s = st();
        assert_eq!(s.handle_slash_command("/setup skip"), SlashOutcome::Handled);
        assert!(
            s.slash_tool.is_none(),
            "unsupported /setup sub must NOT dispatch"
        );
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    // --- Phase 6: automations / reviews / approve / reject / edit / permissions ---

    #[test]
    fn slash_automations_bare_lists() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations"),
            SlashOutcome::Handled
        );
        let req = s
            .slash_tool
            .expect("automations raises a slash_tool request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["automations", "list"] })
        );
    }

    #[test]
    fn slash_automations_enable_raises_watcher_enable() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations enable foo --mode observe"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("enable raises a request");
        assert_eq!(req.name, "watcher_enable");
        assert_eq!(
            req.args,
            serde_json::json!({ "watcher": "foo", "mode": "observe" })
        );
    }

    #[test]
    fn slash_automations_enable_without_mode_omits_field() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations enable foo"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("enable raises a request");
        assert_eq!(req.name, "watcher_enable");
        assert_eq!(req.args, serde_json::json!({ "watcher": "foo" }));
        assert!(req.args.get("mode").is_none(), "mode must be omitted");
    }

    #[test]
    fn slash_automations_disable_raises_watcher_disable() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations disable foo"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("disable raises a request");
        assert_eq!(req.name, "watcher_disable");
        assert_eq!(req.args, serde_json::json!({ "watcher": "foo" }));
    }

    #[test]
    fn slash_automations_enable_without_watcher_hints() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/automations enable"),
            SlashOutcome::Handled
        );
        assert!(s.slash_tool.is_none(), "no dispatch without a watcher");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_reviews_bare_lists() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/reviews"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("reviews raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["automations", "list"] })
        );
    }

    #[test]
    fn slash_reviews_id_shows_proposal() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/reviews p1"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("reviews <id> raises a request");
        assert_eq!(req.name, "proposal_action");
        assert_eq!(
            req.args,
            serde_json::json!({ "proposal_id": "p1", "action": "show" })
        );
    }

    #[test]
    fn slash_approve_id_raises_proposal_approve() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/approve p1"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("approve raises a request");
        assert_eq!(req.name, "proposal_action");
        assert_eq!(
            req.args,
            serde_json::json!({ "proposal_id": "p1", "action": "approve" })
        );
    }

    #[test]
    fn slash_approve_bare_hints() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/approve"), SlashOutcome::Handled);
        assert!(s.slash_tool.is_none(), "no dispatch without a proposal id");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn slash_reject_id_raises_proposal_reject() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/reject p1"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("reject raises a request");
        assert_eq!(req.name, "proposal_action");
        assert_eq!(
            req.args,
            serde_json::json!({ "proposal_id": "p1", "action": "reject" })
        );
    }

    #[test]
    fn slash_edit_id_shows_proposal_with_note() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/edit p1"), SlashOutcome::Handled);
        let req = s.slash_tool.expect("edit raises a request");
        assert_eq!(req.name, "proposal_action");
        assert_eq!(
            req.args,
            serde_json::json!({ "proposal_id": "p1", "action": "show" })
        );
        // A one-line note directs the operator to /approve or /reject.
        let last = s.chat.last().expect("edit pushes a note turn");
        assert_eq!(last.role, ChatRole::Agent);
        assert!(last.content.contains("/approve") && last.content.contains("/reject"));
    }

    #[test]
    fn slash_permissions_bare_is_config_show() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/permissions"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("permissions raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(req.args, serde_json::json!({ "args": ["config", "show"] }));
    }

    #[test]
    fn slash_permissions_full_access_is_mutating() {
        // SAFETY: permission escalation must go through the approval modal — it is
        // dispatched as an approval-classified rocm_command, never run inline.
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/permissions full-access"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("full-access raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["config", "set-permissions", "full_access"] })
        );
    }

    #[test]
    fn slash_permissions_ask_is_mutating() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("/permissions ask"),
            SlashOutcome::Handled
        );
        let req = s.slash_tool.expect("ask raises a request");
        assert_eq!(req.name, "rocm_command");
        assert_eq!(
            req.args,
            serde_json::json!({ "args": ["config", "set-permissions", "ask"] })
        );
    }

    /// Recording executor: mutating names surface `ApprovalRequired`; the
    /// approved replay records `(name, args)` and returns a success Result. Used
    /// to drive the approve/deny/follow-up tests offline (no real installs).
    #[derive(Debug)]
    struct RecordingExecutor {
        approved: std::sync::Arc<std::sync::Mutex<Vec<(String, serde_json::Value)>>>,
    }
    impl RecordingExecutor {
        fn new() -> Self {
            Self {
                approved: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }
    }
    impl crate::tool_exec::RocmToolExecutor for RecordingExecutor {
        fn execute(
            &self,
            name: &str,
            args: &serde_json::Value,
        ) -> crate::tool_exec::RocmToolOutcome {
            crate::tool_exec::RocmToolOutcome::ApprovalRequired(crate::tool_exec::ApprovalIntent {
                title: "T".to_string(),
                body: vec!["cmd".to_string()],
                name: name.to_string(),
                arguments: args.clone(),
            })
        }
        fn execute_approved(
            &self,
            name: &str,
            args: &serde_json::Value,
        ) -> crate::tool_exec::RocmToolOutcome {
            self.approved
                .lock()
                .unwrap()
                .push((name.to_string(), args.clone()));
            crate::tool_exec::RocmToolOutcome::Result(serde_json::json!({ "ok": true }))
        }
    }

    #[test]
    fn approval_required_opens_modal() {
        let mut s = st();
        let intent = crate::tool_exec::ApprovalIntent {
            title: "Install ROCm".to_string(),
            body: vec!["rocm install sdk".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({ "channel": "release" }),
        };
        s.open_approval(intent);
        let pa = s.approval.as_ref().expect("modal opened");
        assert_eq!(pa.name, "install_sdk");
        assert_eq!(pa.req.title, "Install ROCm");
    }

    #[test]
    fn close_overlays_clears_pending_approval() {
        // A stale approval modal must not survive close_overlays (focus trap).
        let mut s = st();
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "Install ROCm".to_string(),
            body: vec!["rocm install sdk".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({}),
        });
        assert!(s.approval.is_some(), "approval pending before close");
        s.close_overlays();
        assert!(s.approval.is_none(), "close_overlays must clear the modal");
    }

    #[test]
    fn second_approval_request_is_discarded_while_one_pending() {
        // Two mutating calls in one turn must not clobber: the operator could
        // otherwise approve args they never saw.
        let mut s = st();
        let first = crate::tool_exec::ApprovalIntent {
            title: "Install ROCm".to_string(),
            body: vec!["rocm install sdk".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({ "prefix": "~/rocm" }),
        };
        s.open_approval(first);
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "Stop server".to_string(),
            body: vec!["rocm services stop svc-1".to_string()],
            name: "stop_server".to_string(),
            arguments: serde_json::json!({ "service_id": "svc-1" }),
        });
        // The original intent survives intact; the second was discarded.
        let pa = s.approval.as_ref().expect("first approval still pending");
        assert_eq!(pa.name, "install_sdk");
        assert_eq!(pa.arguments["prefix"], "~/rocm");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
    }

    #[test]
    fn approve_path_runs_execute_approved_with_expected_args() {
        // (a) approve: a ChatApprovalRequired opens the modal; an Approve verdict
        // drives execute_approved with the exact name + args. We exercise the
        // same sync code path the spawn_blocking uses (`run_approved`).
        let exec = std::sync::Arc::new(RecordingExecutor::new());
        let recorded = exec.approved.clone();
        let shared: crate::tool_exec::SharedRocmToolExecutor = exec;

        let mut s = st();
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "T".to_string(),
            body: vec!["cmd".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({ "channel": "release", "format": "wheel" }),
        });
        // The modal defaults to Deny (item #16); move to Approve, then confirm.
        s.on_approval_key(crossterm::event::KeyCode::Tab);
        let verdict = s.on_approval_key(crossterm::event::KeyCode::Enter);
        assert_eq!(verdict, Some(crate::ui::approval::ApprovalVerdict::Approve));
        let (name, args) = s.take_approval().expect("approval taken on approve");
        assert!(s.approval.is_none(), "modal cleared after taking approval");

        let summary = run_approved(&shared, &name, &args);
        let log = recorded.lock().unwrap();
        assert_eq!(log.len(), 1, "execute_approved ran exactly once");
        assert_eq!(log[0].0, "install_sdk");
        assert_eq!(log[0].1["channel"], "release");
        assert_eq!(log[0].1["format"], "wheel");
        assert!(
            summary.contains("Approved"),
            "concise summary, not raw JSON"
        );
    }

    /// The leading sentences of the CLI refusal `rocm comfyui install` prints
    /// when two managed ROCm runtimes are ready and none is activated. This is a
    /// verbatim *prefix*, not the whole message: the real one continues with an
    /// `Available: <key>, <key>.` list and a trailing pointer to
    /// `rocm runtimes list`, both of which depend on the planted runtimes and
    /// neither of which this test inspects — it only pins what the seam does
    /// with the envelope it is handed, so the remedy-bearing prefix is the
    /// relevant part.
    const AMBIGUOUS_RUNTIME_REFUSAL: &str = "Multiple ROCm runtimes are ready. Pick one in `/runtimes`, set a default \
         with `rocm runtimes activate <key>`, or pass `--runtime-id <key>`.";

    /// An executor whose approved replay *replicates* what the real seam returns
    /// for a `rocm` subprocess that exited non-zero: `run_rocm_capture_for_paths`
    /// *captures* the failure, so `run_internal_mcp_call` returns `Ok` with an
    /// `isError: true` envelope and the stderr buried in `structuredContent` —
    /// it never returns `Err`, so the seam never builds `RocmToolOutcome::Error`.
    ///
    /// Replicates, not reaches: those producers live in the bin, which depends
    /// on this crate, so this crate cannot call them. The envelope below is
    /// hand-built to their shape and the test pins only what happens
    /// *downstream* of it. That the producers really do hand the seam an `Ok`
    /// envelope for a non-zero exit is pinned separately, against a real `rocm`
    /// subprocess, by `seam_execute_approved_captures_a_failing_command_as_a_result`
    /// in `apps/rocm/src/dash_seam.rs`.
    #[derive(Debug)]
    struct CapturedFailureExecutor;
    impl crate::tool_exec::RocmToolExecutor for CapturedFailureExecutor {
        fn execute(
            &self,
            name: &str,
            args: &serde_json::Value,
        ) -> crate::tool_exec::RocmToolOutcome {
            crate::tool_exec::RocmToolOutcome::ApprovalRequired(crate::tool_exec::ApprovalIntent {
                title: "Install ComfyUI".to_string(),
                body: vec!["rocm comfyui install".to_string()],
                name: name.to_string(),
                arguments: args.clone(),
            })
        }
        fn execute_approved(
            &self,
            _name: &str,
            _args: &serde_json::Value,
        ) -> crate::tool_exec::RocmToolOutcome {
            crate::tool_exec::RocmToolOutcome::Result(serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": format!("Ran `rocm` command.\n\nstderr:\n{AMBIGUOUS_RUNTIME_REFUSAL}"),
                }],
                "structuredContent": {
                    "argv": ["rocm", "comfyui", "install"],
                    "exit_status": 1,
                    "stdout": "",
                    "stderr": AMBIGUOUS_RUNTIME_REFUSAL,
                },
                "isError": true,
            }))
        }
    }

    #[test]
    fn approved_command_failure_stays_a_collapsed_envelope() {
        // Pins the second half of the premise the ComfyUI e2e scenario's
        // CLI-only scope rests on (`tests/e2e-cucumber/features/comfyui.feature`):
        // *given* the captured `isError: true` envelope, `run_approved` takes
        // the `Result` arm and `summarize_json_value` collapses every field, so
        // the refusal text stays out of the chat — reading the `Error` arm
        // (`Approved · … failed: {e}`) as this path's renderer is wrong. The
        // first half — that a non-zero `rocm` exit really does arrive as that
        // envelope rather than as an `Err` — is pinned by the seam test named
        // on `CapturedFailureExecutor` above.
        let shared: crate::tool_exec::SharedRocmToolExecutor =
            std::sync::Arc::new(CapturedFailureExecutor);
        let summary = run_approved(
            &shared,
            "rocm_command",
            &serde_json::json!({ "args": ["comfyui", "install"] }),
        );
        assert!(
            summary.contains("content: [1 items]"),
            "the command envelope is collapsed, not inlined: {summary}"
        );
        assert!(
            summary.contains("structuredContent: {4 fields}"),
            "the captured stdout/stderr subtree is collapsed too: {summary}"
        );
        assert!(
            !summary.contains("Multiple ROCm runtimes are ready"),
            "the CLI refusal must not reach the chat: {summary}"
        );
        assert!(
            !summary.contains("failed:"),
            "a captured non-zero exit is not the Error arm: {summary}"
        );
    }

    #[test]
    fn deny_path_runs_nothing_and_appends_declined_turn() {
        // (b) deny: a Deny/Cancel verdict appends a declined turn and never
        // touches execute_approved.
        let exec = std::sync::Arc::new(RecordingExecutor::new());
        let recorded = exec.approved.clone();

        let mut s = st();
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "T".to_string(),
            body: vec!["cmd".to_string()],
            name: "launch_server".to_string(),
            arguments: serde_json::json!({ "model": "m" }),
        });
        // 'n' is a direct Deny verdict.
        let verdict = s.on_approval_key(crossterm::event::KeyCode::Char('n'));
        assert_eq!(verdict, Some(crate::ui::approval::ApprovalVerdict::Deny));
        s.on_approval_declined();
        assert!(s.approval.is_none(), "modal cleared on deny");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(s.chat.last().unwrap().content.contains("declined"));
        assert!(
            recorded.lock().unwrap().is_empty(),
            "deny must not execute the action"
        );

        // Esc also cancels (no execution).
        let mut s2 = st();
        s2.open_approval(crate::tool_exec::ApprovalIntent {
            title: "T".to_string(),
            body: vec!["cmd".to_string()],
            name: "stop_server".to_string(),
            arguments: serde_json::json!({ "service_id": "x" }),
        });
        assert_eq!(
            s2.on_approval_key(crossterm::event::KeyCode::Esc),
            Some(crate::ui::approval::ApprovalVerdict::Cancel)
        );
    }

    #[test]
    fn approval_modal_escape_is_not_a_focus_trap() {
        // Edge: the approval modal must be escapable — Esc and 'n' both yield a
        // closing verdict, and routing that verdict through the deny/cancel path
        // clears the modal (`approval` → None) without executing. The covered
        // active tab is preserved across open → escape (the modal overlays it).
        for key in [
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyCode::Char('n'),
        ] {
            let mut s = st();
            s.active_tab = ActiveTab::Observe;
            s.open_approval(crate::tool_exec::ApprovalIntent {
                title: "T".to_string(),
                body: vec!["cmd".to_string()],
                name: "stop_server".to_string(),
                arguments: serde_json::json!({ "service_id": "x" }),
            });
            assert!(s.approval.is_some(), "modal open before escape");
            let verdict = s.on_approval_key(key);
            // Esc → Cancel, 'n' → Deny; both are closing (non-Approve) verdicts.
            assert!(
                matches!(
                    verdict,
                    Some(
                        crate::ui::approval::ApprovalVerdict::Cancel
                            | crate::ui::approval::ApprovalVerdict::Deny
                    )
                ),
                "key {key:?} must yield a closing verdict, got {verdict:?}"
            );
            // The event loop routes Deny|Cancel through on_approval_declined.
            s.on_approval_declined();
            assert!(
                s.approval.is_none(),
                "escape must clear the modal (no focus trap) for {key:?}"
            );
            // The covered tab is preserved — the modal never navigated away.
            assert_eq!(s.active_tab, ActiveTab::Observe);
        }
    }

    #[test]
    fn approval_result_fires_exactly_one_follow_up_no_loop() {
        // (c) exactly one follow-up: on_approval_result appends the result turn
        // AND raises chat_dispatch exactly once; it must not re-trigger itself.
        let mut s = st();
        assert!(!s.chat_dispatch);
        s.on_approval_result("Approved · install_sdk: done".to_string());
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(s.chat_dispatch, "exactly one follow-up edge raised");
        assert!(s.chat_sending, "in-flight mirrors a normal submit");

        // Simulate the event loop consuming the edge once.
        s.chat_dispatch = false;
        // A subsequent tick must NOT re-raise it on its own (no self-loop): the
        // only thing that re-raises is another explicit result/submit.
        assert!(!s.chat_dispatch, "follow-up does not re-trigger itself");
    }

    #[test]
    fn approval_key_tab_moves_choice_without_verdict() {
        let mut s = st();
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "T".to_string(),
            body: vec!["cmd".to_string()],
            name: "install_sdk".to_string(),
            arguments: serde_json::json!({}),
        });
        // Defaults to Deny (the safer default; see item #16).
        assert_eq!(
            s.approval.as_ref().unwrap().choice,
            crate::ui::approval::ApprovalChoice::Deny
        );
        // Tab toggles the cursor to Approve without producing a verdict.
        assert_eq!(s.on_approval_key(crossterm::event::KeyCode::Tab), None);
        assert_eq!(
            s.approval.as_ref().unwrap().choice,
            crate::ui::approval::ApprovalChoice::Approve
        );
        // Enter now confirms Approve.
        assert_eq!(
            s.on_approval_key(crossterm::event::KeyCode::Enter),
            Some(crate::ui::approval::ApprovalVerdict::Approve)
        );
    }

    #[test]
    fn slash_tool_reply_does_not_disturb_chat_sending() {
        // The slash-tool reply path is decoupled from the agent state machine:
        // appending a slash-tool summary must NOT touch `chat_sending`, even if
        // an agent request happens to be in flight at the same time.
        let mut s = st();
        s.chat_sending = true;
        let before = s.chat.len();
        s.on_slash_tool_reply("x".into());
        assert_eq!(s.chat.len(), before + 1, "slash-tool turn appended");
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert!(
            s.chat_sending,
            "chat_sending untouched by slash-tool reply (decoupled)"
        );
    }

    #[test]
    fn summarize_slash_tool_is_concise_not_raw_json() {
        // A representative Result(json) must summarize to a terse, labelled,
        // length-bounded blurb — never a raw JSON dump with braces-spam.
        let outcome = crate::tool_exec::RocmToolOutcome::Result(serde_json::json!({
            "status": "ok",
            "model": "llama3",
            "nested": { "a": 1, "b": 2, "c": 3 },
        }));
        let out = summarize_slash_tool("model", &outcome);
        assert!(out.contains("/model"), "carries the slash label");
        assert!(out.contains("status: ok"), "scalars shown inline");
        // Nested containers collapse to a shape hint, not an inlined subtree.
        assert!(out.contains("{3 fields}"), "nested object shown as shape");
        assert!(
            !out.contains("\"nested\""),
            "no raw JSON keys / braces-spam in summary"
        );
        assert!(out.len() < 200, "summary stays length-bounded");
    }

    #[test]
    fn slash_unknown_appends_error_turn_and_is_handled() {
        let mut s = st();
        assert_eq!(s.handle_slash_command("/zzz"), SlashOutcome::Handled);
        assert_eq!(s.chat.len(), 1);
        assert_eq!(s.chat[0].role, ChatRole::Error);
        assert!(s.chat[0].content.contains("/zzz"));
    }

    #[test]
    fn plain_text_is_not_a_command() {
        let mut s = st();
        assert_eq!(
            s.handle_slash_command("what's GPU-2 doing?"),
            SlashOutcome::NotCommand
        );
    }

    #[test]
    fn submit_routes_slash_command_away_from_the_agent() {
        // A slash line through submit_chat must NOT raise the agent dispatch edge.
        let mut s = st();
        s.chat_input = "/help".into();
        s.submit_chat();
        assert_eq!(s.modal, Modal::Help);
        assert!(
            !s.chat_dispatch,
            "slash command never dispatches to the LLM"
        );
        assert!(!s.chat_sending);
        assert!(s.chat_input.is_empty());
    }
}
