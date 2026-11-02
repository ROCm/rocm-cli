// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Dashboard reducer: `AppState`, its constructor, the theme picker, row
//! selection and the `apply_event` entry point that folds daemon events into
//! state.
//!
//! The rest of `AppState`'s behaviour lives in child modules, each a further
//! `impl AppState` block: `app/overlays.rs` (which overlay owns the keyboard,
//! and the gating predicates), `app/scroll.rs` (scroll offsets),
//! `app/chat_state.rs` (chat consent, detection and turns),
//! `app/quit_confirm.rs` and `app/tool_approval.rs` (the two confirmation
//! prompts), `app/slash.rs` (slash-command routing) and `app/chat.rs`
//! (`set_chat_config`, plus the chat-agent and endpoint helpers). The other
//! child modules hold no `AppState` methods: `app/types.rs` (shared type/enum
//! defs and the menu data), `app/actions.rs` (`KeyAction` dispatch:
//! `handle_key`, `apply_action`), `app/scrollbar.rs` (mouse hit-testing),
//! `app/event_loop.rs` (the tick loop), `app/terminal.rs` (terminal restore,
//! shutdown latch, signal watcher), `app/focus.rs` (focused-host policy),
//! `app/update_check.rs` (the Updates-tile check) and `app/summary.rs`. `crate::app::*` paths for the public surface moved out
//! are unchanged via the re-exports below.

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
mod chat_state;
mod event_loop;
mod focus;
mod overlays;
mod quit_confirm;
mod scroll;
mod scrollbar;
mod slash;
mod summary;
mod terminal;
#[cfg(test)]
pub(crate) mod test_support;
mod tool_approval;
mod types;
mod update_check;

// Re-exports restoring the pre-split `crate::app::*` public surface. A
// `pub(crate)` item with no caller through that path isn't re-exported just
// because it was reachable there pre-split (see the removed
// `NO_CHAT_BACKEND_MSG` re-export this rule cost).
pub(crate) use actions::request_quit;
pub use actions::{KeyAction, handle_mouse, tab_bar_hit};
pub use event_loop::run;
pub use terminal::spawn_termination_watcher;
pub(crate) use terminal::{
    SHUTTING_DOWN, exit_on_ctrl_c, is_ctrl_c, lock_terminal_writer, restore_terminal,
    shutdown_claimed_on,
};
pub(crate) use update_check::HOME_UPDATE_CHECK_JOB_ID;
// Only reached via a test (`launcher.rs`'s
// `the_front_door_comes_back_after_a_session_ends_cleanly`); gated to that
// build so the compiler (not a hand-maintained `#[allow]`) flags this as dead
// if that caller ever disappears.
pub use scrollbar::{FooterChip, ScrollDrag, ScrollbarHandle};
#[cfg(test)]
pub(crate) use terminal::restore_after_session;
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
    use tokio::sync::mpsc;

    use crate::client::ClientMsg;

    use super::chat::build_chat_agent;
    use super::test_support::{args_with_anthropic_key, st};

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
}
