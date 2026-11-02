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
mod quit_confirm;
mod scrollbar;
mod slash;
mod summary;
#[cfg(test)]
pub(crate) mod test_support;
mod tool_approval;
mod types;

// Re-exports restoring the pre-split `crate::app::*` public surface. A
// `pub(crate)` item with no caller through that path isn't re-exported just
// because it was reachable there pre-split (see the removed
// `NO_CHAT_BACKEND_MSG` re-export this rule cost).
pub(crate) use actions::request_quit;
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
pub use scrollbar::{FooterChip, ScrollDrag, ScrollbarHandle};
pub use types::{
    ActiveTab, ChatConsent, ChatKeyCtx, ChatRole, ChatTurn, ConnState, Focus, MENU_ITEMS, Modal,
    OPTIONS_TABS, PALETTE_DESTS, PaneFocus, PlannedAction, ReplayState, ResolvedArgs, ScrollTarget,
    UpdateStatus, format_mmss,
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
    /// Set once the daemon's first instance snapshot has landed (see
    /// `push_snapshot`). Until then, `instances` being empty says nothing
    /// about whether anything is actually serving — it may simply not have
    /// arrived yet — which is what `has_live_instance` consults a live
    /// registry read for (issue #145).
    pub(crate) has_received_snapshot: bool,
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
    /// Set from `ResolvedArgs::managed_registry_probe`: the bin-injected live
    /// registry-read seam `has_live_instance` / `has_stoppable_managed_instance_with`
    /// / `quit_confirm_open_services_key` consult, rather than a snapshot
    /// frozen at launch (issue #145). `Some` on every bin path — this is
    /// never `None`, including under `--demo`/`--replay`/`--chat-mock`.
    /// `--demo`/`--replay` are made safe by `simulated` (see
    /// `has_live_instance`), not by this being absent; `--chat-mock` is NOT
    /// `simulated` and deliberately does read this live (see dash-29). See
    /// `live_registry_snapshot` and `registry_probe::ManagedRegistryProbe`.
    pub managed_registry_probe: Option<crate::registry_probe::SharedManagedRegistryProbe>,
    /// Set from `ResolvedArgs::launcher_hosted`: `true` when this session was
    /// opened from the pre-dashboard launcher, which owns the quit decision
    /// that actually exits the process. See the field doc on
    /// `ResolvedArgs::launcher_hosted` and `actions::request_quit`.
    pub(crate) launcher_hosted: bool,
    /// Last body area used by the most recent draw. Mouse hit-tests resolve
    /// pointer coordinates against this rect (filled by `ui::draw`).
    pub last_body_area: Option<ratatui::layout::Rect>,
    /// Same for the tab bar, used for click-to-switch-tab.
    pub last_tab_bar_area: Option<ratatui::layout::Rect>,
    /// Header `⚠ N` badge from the most recent draw (`None` when no warning is
    /// shown). Left-clicking it opens the warnings popup, as `!` does, under the
    /// same gating.
    pub last_header_badge: Option<ratatui::layout::Rect>,
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
    /// A quit attempted while an instance was still serving (this host's own
    /// managed instance, or one Docker-/Lemonade-discovered from an external
    /// daemon), from any of `request_quit`'s entry points: `q`, the Esc main
    /// menu's `Quit` row,
    /// `/quit`/`/exit` in chat, or the pre-dashboard launcher's own `q`/Esc.
    /// `Some` ⇒ the confirm-before-quit prompt is open and owns keyboard
    /// focus, the same way `approval` does.
    pub(crate) quit_confirm: Option<crate::ui::approval::ApprovalChoice>,
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
            has_received_snapshot: false,
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
            managed_registry_probe: None,
            launcher_hosted: false,
            last_body_area: None,
            last_tab_bar_area: None,
            last_header_badge: None,
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
            quit_confirm: None,
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
        self.quit_confirm = None;
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

    /// Whether a modal that owns the body absolutely — no console, nothing to
    /// pan or fall through to — is open: the chat approval or the quit-confirm
    /// prompt. Unlike [`overlay_or_approval`](Self::overlay_or_approval), this
    /// deliberately excludes [`has_open_overlay`](Self::has_open_overlay): a
    /// manager overlay's job console is still a legitimate scroll/drag target,
    /// where these two never are.
    pub(crate) const fn blocks_body_absolutely(&self) -> bool {
        self.approval_pending() || self.quit_confirm_pending()
    }

    /// Whether *any* gating layer owns the screen: an open manager overlay, a
    /// pending chat approval, or a pending quit-confirm prompt. This exact
    /// `||` is what every input path that swallows for one must also swallow
    /// for the others — call sites wrote it out by hand before this existed,
    /// each with its own copy of this same reasoning; forgetting one would
    /// reopen the class of bug `approval_pending`'s own doc comment
    /// describes. Composed from [`has_open_overlay`](Self::has_open_overlay)
    /// and [`blocks_body_absolutely`](Self::blocks_body_absolutely) rather
    /// than repeating their two sub-conditions by hand, so a future third
    /// "owns-the-screen" gate only has to be added to one of the two
    /// functions, not to both. The name predates the quit-confirm prompt;
    /// kept as-is to avoid rippling a rename through every call site's
    /// comments.
    pub(crate) const fn overlay_or_approval(&self) -> bool {
        self.has_open_overlay() || self.blocks_body_absolutely()
    }

    /// Whether a global hotkey reaches its normal `handle_key` arm right now,
    /// or something else owns the keyboard first: a manager overlay or pending
    /// approval/quit-confirm owns every key, a plain `Modal::*` returns from
    /// its own arm, and focused chat text entry consumes every `Char`. Anything
    /// that dispatches a global hotkey action from a click (footer chips, the
    /// header warning badge) must gate on this, or the click can do what the
    /// key it stands for never could.
    pub(crate) const fn global_hotkeys_reachable(&self) -> bool {
        let chat_typing = matches!(self.active_tab, ActiveTab::Chat) && self.chat_focused;
        !(self.overlay_or_approval() || !matches!(self.modal, Modal::None) || chat_typing)
    }

    /// Number of warnings in the latest snapshot — what the header badge shows.
    pub(crate) fn warning_count(&self) -> usize {
        self.latest.as_ref().map_or(0, |s| s.warnings.len())
    }

    /// Focused-host exit gate: `true` when a `focus` is active AND its single
    /// overlay is closed (no manager is `Some`) AND no quit-confirm prompt is
    /// pending.
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
    ///
    /// The `quit_confirm_pending` exemption matters only if a focused host
    /// ever calls `actions::request_quit` (none does today): `open_quit_confirm`
    /// closes every manager overlay via `close_overlays`, which without this
    /// exemption would make this gate fire on the very next loop iteration —
    /// before `draw_focused` ever renders the prompt it draws for that case —
    /// exiting silently instead of showing it. Deliberately NOT folded into
    /// `has_open_overlay` itself: that predicate also gates unrelated reads
    /// (`should_pane_back_out`, `active_overlay_at_root`'s callers) where a
    /// pending quit-confirm must NOT count as an open manager overlay.
    pub(crate) const fn focused_should_exit(&self, focus: Option<Focus>) -> bool {
        focus.is_some() && !self.has_open_overlay() && !self.quit_confirm_pending()
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
                .is_none_or(|m| m.approval.is_none() && m.active_job.is_none())
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
        // Snapshots carry the daemon's current instance set — treat them as
        // truth, and from here on `instances` alone is authoritative for
        // `has_live_instance` (issue #145's startup-race fallback stops
        // mattering the moment real data exists).
        self.has_received_snapshot = true;
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

    #[test]
    fn global_hotkeys_are_unreachable_while_a_modal_overlay_or_prompt_owns_the_keyboard() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        assert!(s.global_hotkeys_reachable(), "idle dashboard");
        s.modal = Modal::Help;
        assert!(!s.global_hotkeys_reachable(), "a plain modal returns first");
        s.modal = Modal::None;
        s.services = Some(crate::ui::services_manager::ServicesManagerState::default());
        assert!(
            !s.global_hotkeys_reachable(),
            "a manager overlay owns every key"
        );
        s.services = None;
        s.open_quit_confirm();
        assert!(
            !s.global_hotkeys_reachable(),
            "the quit prompt owns every key"
        );
    }

    #[test]
    fn typing_in_chat_owns_the_keyboard_only_on_the_chat_tab_and_only_while_focused() {
        // `handle_key` hands every character to the chat input only once chat
        // is accepted and the input is focused.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_consent = ChatConsent::Accepted;
        s.active_tab = ActiveTab::Chat;
        s.chat_focused = false;
        assert!(s.global_hotkeys_reachable(), "chat tab, input not focused");
        s.chat_focused = true;
        assert!(!s.global_hotkeys_reachable(), "focused chat input");
        // `chat_focused` is not cleared when the user switches tab, so a stale
        // `true` on another tab must not be read as "typing".
        for tab in [
            ActiveTab::Home,
            ActiveTab::Rocm,
            ActiveTab::Serving,
            ActiveTab::Observe,
        ] {
            s.active_tab = tab;
            assert!(
                s.global_hotkeys_reachable(),
                "{tab:?} with a stale chat focus"
            );
        }
    }

    use crate::client::ClientMsg;

    use super::chat::build_chat_agent;
    use super::test_support::{args_with_anthropic_key, st};
    use super::types::NO_CHAT_BACKEND_MSG;

    #[test]
    fn actions_does_not_import_from_scrollbar() {
        // scrollbar.rs depends on actions.rs (KeyAction/handle_mouse/tab_bar_hit,
        // plus apply_action under cfg(test)); actions.rs must never depend
        // back on scrollbar.rs -- directly, through `crate::app::scrollbar`,
        // or through one of the names app/mod.rs re-exports from it
        // (ScrollbarHandle, ScrollDrag, FooterChip) -- or the direct
        // `use`-level cycle that moving PaneFocus/ScrollTarget into types.rs
        // removed would come back. The compiler accepts such a cycle, so
        // nothing else catches it.
        //
        // Parsing `use` items can't guard that on its own: a glob import
        // (`use super::*;`) brings a name in without ever writing a `use`
        // that names it, and a fully-qualified path used inline needs no
        // `use` statement at all. actions.rs has no legitimate reason to
        // write any of these identifiers anywhere, by any means, so this
        // scans its whole non-comment source for a whole-word match
        // instead of parsing `use` syntax specifically -- a stronger check
        // that is also simpler. Matching whole identifier tokens (not a
        // bare substring) means `vertical_scrollbar` doesn't false-positive
        // on `scrollbar`.
        //
        // Known gaps: a name written only inside a `/* */` block comment
        // (unlike `//`, not stripped below) would false-positive; a
        // reference whose type is fully inferred, with the identifier
        // never written out as text anywhere, can't be caught by scanning
        // text rather than parsing it. Unlike those two, which only
        // over-fire or miss safely, the line-stripping below has no
        // string-literal awareness: a name placed after a `//` that's
        // actually inside a string literal (e.g. on a line with a
        // `"http://..."` value) would be silently dropped from the scan
        // and let a real violation pass. Currently harmless -- no line
        // here combines the two -- but worth knowing this guard doesn't
        // cover.
        const SCROLLBAR_NAMES: [&str; 4] =
            ["scrollbar", "ScrollbarHandle", "ScrollDrag", "FooterChip"];

        fn tokens(s: &str) -> impl Iterator<Item = &str> {
            s.split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|t| !t.is_empty())
        }

        fn offenders_in(src: &str) -> Vec<&str> {
            tokens(src)
                .filter(|t| SCROLLBAR_NAMES.contains(t))
                .collect()
        }

        // Positive controls: prove the detector actually flags every name
        // it claims to guard -- including the re-export names, not just the
        // module name -- and specifically a glob import that brings a name
        // in without any `use` naming it directly, which a `use`-parsing
        // check would miss.
        for name in SCROLLBAR_NAMES {
            let known_bad = format!("use super::*;\nfn f() -> {name} {{ todo!() }}");
            assert!(
                !offenders_in(&known_bad).is_empty(),
                "sanity check failed: the detector doesn't flag {name}"
            );
        }
        assert!(
            offenders_in("let vertical_scrollbar = 1;").is_empty(),
            "sanity check failed: the detector false-positives on a substring match"
        );

        let without_comments: String = include_str!("actions.rs")
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");

        let offenders = offenders_in(&without_comments);
        assert!(
            offenders.is_empty(),
            "actions.rs must not reference scrollbar.rs or its re-exported types \
             (ScrollbarHandle, ScrollDrag, FooterChip), by any means: {offenders:#?}"
        );
    }

    #[test]
    fn back_out_requires_an_open_manager_on_any_tab() {
        let mut s = st();
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
        let mut s = st();
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
        let mut s = st();
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
        let mut s = st();
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
        let mut s = st();
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
        let mut s = st();
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
        let mut s = st();
        assert!(!s.has_open_overlay(), "no overlay initially");
        s.bench_run = Some(crate::ui::bench_run::BenchRunState::new(None));
        assert!(
            s.has_open_overlay(),
            "bench_run must be in has_open_overlay"
        );
    }

    #[test]
    fn t13_bench_run_cleared_by_close_overlays() {
        let mut s = st();
        s.bench_run = Some(crate::ui::bench_run::BenchRunState::new(None));
        s.close_overlays();
        assert!(s.bench_run.is_none(), "close_overlays must clear bench_run");
    }

    #[test]
    fn t13_bench_run_at_root() {
        let mut s = st();
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

        let ExamineManagerState {
            active_job,
            report: _,
            approval,
            message: _,
        } = ExamineManagerState::default();
        assert!(approval.is_none() && active_job.is_none());

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
        let mut s = st();
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
        let mut s = st();
        s.request_detect();
        s.set_detect_result(None);
        assert!(!s.chat_detecting);
        assert!(s.chat_detect_offer.is_none());
        assert!(s.chat_detect_msg.is_some());
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
    fn no_provider_no_key_chat_surfaces_actionable_message() {
        // Edge: agent is None (no endpoint, no provider key). Submitting chat
        // must surface a clear, ACTIONABLE message (the recovery affordances),
        // routed through `on_chat_error` as an error turn — not an error dump,
        // not a panic. This mirrors the event-loop None-agent branch, which
        // emits exactly `NO_CHAT_BACKEND_MSG`.
        let mut s = st();
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
    fn close_overlays_clears_pending_quit_confirm() {
        // Mirrors close_overlays_clears_pending_approval: quit_confirm is the
        // sibling gating layer and must be cleared the same way, or a future
        // path that reaches close_overlays() while it's open would leave a
        // stale, undismissable prompt.
        let mut s = st();
        s.open_quit_confirm();
        assert!(
            s.quit_confirm.is_some(),
            "quit_confirm pending before close"
        );
        s.close_overlays();
        assert!(
            s.quit_confirm.is_none(),
            "close_overlays must clear quit_confirm"
        );
    }
}
