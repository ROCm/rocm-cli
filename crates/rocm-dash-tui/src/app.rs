//! Event loop. Sets up the terminal, spawns the client task, drives renders.

use std::collections::VecDeque;
use std::io;
use std::time::Duration;

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, Event as CtEvent, EventStream, KeyCode, KeyEvent,
    KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::collections::HashMap;

use rocm_dash_core::bench_schema::BenchmarkRow;
use rocm_dash_core::metrics::{Instance, Snapshot};
use rocm_dash_core::protocol::Event;
use tokio::sync::mpsc;
use tokio::time::interval;

use crate::client::{self, ClientMsg};
use crate::ui;
use crate::ui::theme::Theme;

/// Args after CLI + config resolution. Consumed by `run`.
#[derive(Debug, Clone)]
pub struct ResolvedArgs {
    pub connect: String,
    pub token: Option<String>,
    pub theme: String,
    /// When `Some`, replay events from a file instead of connecting to a
    /// live daemon. Mutually exclusive with `connect` (enforced by clap).
    pub replay: Option<std::path::PathBuf>,
    /// Which tab is active when the TUI opens. `Chat` for the chat-first launch
    /// (bare `rocm` / `rocm chat`); `Overview` for the dashboard (`rocm dash`).
    pub initial_tab: ActiveTab,
    /// Chat endpoint base URL, CLI-flag value already merged over config.
    pub chat_url: Option<String>,
    /// Chat model, CLI-flag value already merged over config.
    pub chat_model: Option<String>,
    /// Custom auth header NAME (CLI-flag value merged over config), e.g.
    /// `Ocp-Apim-Subscription-Key` for Azure APIM gateways.
    pub chat_auth_header: Option<String>,
    /// Chat endpoint base URL from the environment (`OPENAI_BASE_URL`).
    /// A separate, lower-precedence tier than `chat_url`.
    pub chat_env_url: Option<String>,
    /// Chat api key, sourced from the environment ONLY (never TOML/CLI/source).
    pub chat_api_key: Option<String>,
    /// Pre-consent to using the detected endpoint (`--chat-yes`), skipping the
    /// one-time in-TUI prompt for the demo.
    pub chat_auto_consent: bool,
    /// Use the offline `MockAgentClient` for chat (`--chat-mock`) — a
    /// deterministic, fully-offline demo with no live LLM.
    pub chat_mock: bool,
    /// Built-in model recipes for the serve wizard's picker (Phase 3 Wave 1).
    /// Adapted by the bin (`apps/rocm`, which has `rocm-core`) so this crate
    /// needs no `rocm-core` dep. Empty when none are available.
    pub model_recipes: Vec<crate::ui::model_picker::ModelRecipeSummary>,
    /// Registered ROCm runtimes for the runtime manager (Phase 3 Wave 2).
    /// Adapted by the bin (`apps/rocm`, which has `rocm-core`) so this crate
    /// needs no `rocm-core` dep. Empty when none are available.
    pub runtimes: Vec<crate::ui::runtime_manager::RuntimeSummary>,
    /// Background checks for the automations manager (Phase 3 Wave 3). Adapted
    /// by the bin. Empty when none are available.
    pub automations: Vec<crate::ui::automations_manager::AutomationSummary>,
}

type Tui = Terminal<CrosstermBackend<io::Stdout>>;

/// How many snapshots to keep for sparklines.
pub const HISTORY_CAP: usize = 240;

/// How many benchmark rows to keep client-side for the bench panel.
pub const BENCH_CAP: usize = 200;

/// Lines per PageUp/PageDown step in the chat transcript.
const CHAT_SCROLL_STEP: i16 = 5;

#[derive(Debug, Clone, Default)]
pub enum ConnState {
    #[default]
    Initial,
    Connecting,
    Connected {
        host: String,
        version: String,
    },
    Disconnected {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ActiveTab {
    #[default]
    Overview,
    Hardware,
    Instances,
    Bench,
    Chat,
}

impl ActiveTab {
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Overview => Self::Hardware,
            Self::Hardware => Self::Instances,
            Self::Instances => Self::Bench,
            Self::Bench => Self::Chat,
            Self::Chat => Self::Overview,
        }
    }
    #[must_use]
    pub const fn prev(self) -> Self {
        match self {
            Self::Overview => Self::Chat,
            Self::Hardware => Self::Overview,
            Self::Instances => Self::Hardware,
            Self::Bench => Self::Instances,
            Self::Chat => Self::Bench,
        }
    }
    pub const fn from_digit(d: char) -> Option<Self> {
        match d {
            '1' => Some(Self::Overview),
            '2' => Some(Self::Hardware),
            '3' => Some(Self::Instances),
            '4' => Some(Self::Bench),
            '5' => Some(Self::Chat),
            _ => None,
        }
    }
}

/// Who authored a chat turn. Plain TUI-local data — `rocm-dash-core` carries
/// no chat types; chat is owned by the TUI crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Agent,
    Error,
}

/// One line in the chat transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub content: String,
}

impl ChatTurn {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }
    pub fn agent(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Agent,
            content: content.into(),
        }
    }
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Error,
            content: content.into(),
        }
    }
}

/// Consent state for using the auto-detected LLM endpoint. The chat surface
/// asks once before any request leaves the machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatConsent {
    /// No endpoint detected from any source — actionable empty-state.
    #[default]
    Unavailable,
    /// Endpoint detected; awaiting the user's one-time accept/decline.
    Pending,
    /// User accepted — chat is enabled.
    Accepted,
    /// User declined — chat stays off until re-enabled.
    Declined,
}

/// Inputs `handle_key` needs to interpret keys on the Chat tab without holding
/// `&AppState` (keeps the function pure and unit-testable).
#[derive(Debug, Clone, Copy)]
pub struct ChatKeyCtx {
    pub focused: bool,
    pub consent: ChatConsent,
    /// A locally-detected endpoint is awaiting use/dismiss — its keys take
    /// precedence over the normal consent prompt.
    pub offer_pending: bool,
}

impl Default for ChatKeyCtx {
    fn default() -> Self {
        // Default to a usable, unfocused surface for tests that don't exercise
        // consent/insert specifics.
        Self {
            focused: false,
            consent: ChatConsent::Accepted,
            offer_pending: false,
        }
    }
}

/// Replay scrubber state. Only present when `--replay` was given.
#[derive(Debug, Clone)]
pub struct ReplayState {
    pub controller: crate::replay::ReplayController,
    pub paused: bool,
    pub speed: f64,
    /// Current playhead in seconds since the start of the recording.
    pub elapsed_s: u64,
    /// Total length of the recording in seconds.
    pub total_s: u64,
}

impl ReplayState {
    pub const fn new(controller: crate::replay::ReplayController) -> Self {
        Self {
            controller,
            paused: false,
            speed: 1.0,
            elapsed_s: 0,
            total_s: 0,
        }
    }
}

/// Format a duration in seconds as `M:SS` (or `H:MM:SS` past an hour).
pub fn format_mmss(secs: u64) -> String {
    if secs >= 3600 {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        format!("{h}:{m:02}:{s:02}")
    } else {
        let m = secs / 60;
        let s = secs % 60;
        format!("{m}:{s:02}")
    }
}

/// Modal overlays. Only one is shown at a time, on top of the active tab body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Modal {
    #[default]
    None,
    Help,
    Detail,
    ThemePicker,
}

pub struct AppState {
    pub connect: String,
    pub conn: ConnState,
    pub latest: Option<Snapshot>,
    pub history: VecDeque<Snapshot>,
    pub bench_rows: VecDeque<BenchmarkRow>,
    pub instances: HashMap<String, Instance>,
    pub active_tab: ActiveTab,
    pub modal: Modal,
    /// Cursor into the sorted Instances grid.
    pub instance_sel: usize,
    /// Cursor into the bench_rows VecDeque (0 = oldest, len-1 = newest).
    pub bench_sel: usize,
    /// Cursor into the Hardware tab's per-GPU panel list.
    pub gpu_sel: usize,
    /// Scroll offset (first visible GPU index) for the Hardware tab when the
    /// GPU list renders as a scrolled window of compact rows. Kept in sync with
    /// `gpu_sel` so the selection stays visible.
    pub gpu_scroll: usize,
    pub theme_name: String,
    pub theme: Theme,
    pub theme_picker_sel: usize,
    /// Scroll offset (in lines) inside the Bench Detail modal. Reset on Open.
    pub bench_detail_scroll: u16,
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
    /// Replay scrubber state. `None` when running against a live daemon.
    pub replay: Option<ReplayState>,
    /// Last body area used by the most recent draw. Mouse hit-tests resolve
    /// pointer coordinates against this rect (filled by `ui::draw`).
    pub last_body_area: Option<ratatui::layout::Rect>,
    /// Same for the tab bar, used for click-to-switch-tab.
    pub last_tab_bar_area: Option<ratatui::layout::Rect>,
    /// Background-job model for operational screens (Phase 3 Wave 1). The
    /// job-bridge runtime streams `StateEvent`s into this from the event loop.
    pub jobs: rocm_dash_core::state::State,
    /// Services manager overlay (Phase 3 Wave 1). `None` = closed.
    pub services: Option<crate::ui::services_manager::ServicesManagerState>,
    /// Serve wizard overlay (Phase 3 Wave 1). `None` = closed.
    pub serve_wizard: Option<crate::ui::serve_wizard::ServeWizardState>,
    /// Engine manager overlay (Phase 3 Wave 1). `None` = closed.
    pub engine_manager: Option<crate::ui::engine_manager::EngineManagerState>,
    /// Examine overlay (Phase 3 Wave 2). `None` = closed.
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
    /// Built-in model recipes for the serve wizard's picker. Set from
    /// `ResolvedArgs` in the event loop; empty by default.
    pub model_recipes: Vec<crate::ui::model_picker::ModelRecipeSummary>,
    /// Registered ROCm runtimes for the runtime manager. Set from
    /// `ResolvedArgs` in the event loop; empty by default.
    pub runtimes: Vec<crate::ui::runtime_manager::RuntimeSummary>,
    /// Background checks for the automations manager. Set from `ResolvedArgs`
    /// in the event loop; empty by default.
    pub automations: Vec<crate::ui::automations_manager::AutomationSummary>,
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
            instance_sel: 0,
            bench_sel: 0,
            gpu_sel: 0,
            gpu_scroll: 0,
            theme_name,
            theme,
            theme_picker_sel,
            bench_detail_scroll: 0,
            chat: Vec::new(),
            chat_input: String::new(),
            chat_sending: false,
            chat_dispatch: false,
            chat_focused: false,
            chat_scroll: 0,
            chat_llm: None,
            chat_consent: ChatConsent::Unavailable,
            chat_detect_offer: None,
            chat_detecting: false,
            chat_detect_dispatch: false,
            chat_detect_msg: None,
            chat_persist_dispatch: false,
            replay: None,
            last_body_area: None,
            last_tab_bar_area: None,
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
            model_recipes: Vec::new(),
            runtimes: Vec::new(),
            automations: Vec::new(),
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
    }

    /// Open the theme picker modal, positioning the cursor on the active theme.
    pub fn open_theme_picker(&mut self) {
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

    /// Reset the bench-detail scroll offset (called when opening the modal).
    pub const fn reset_bench_detail_scroll(&mut self) {
        self.bench_detail_scroll = 0;
    }

    /// Adjust the bench-detail scroll. `delta` is in lines; clamped at 0
    /// (no upper bound — the renderer clamps against the actual line count).
    pub fn scroll_bench_detail(&mut self, delta: i16) {
        let cur = i32::from(self.bench_detail_scroll);
        let next = (cur + i32::from(delta)).max(0) as u16;
        self.bench_detail_scroll = next;
    }

    /// Install the resolved chat endpoint and set the initial consent state.
    /// `None` → `Unavailable`; `Some` → `Accepted` when pre-consented (e.g.
    /// `--chat-yes`), otherwise `Pending` (the one-time in-TUI prompt).
    pub fn set_chat_config(&mut self, llm: Option<crate::llm::LlmConfig>, pre_consent: bool) {
        self.chat_consent = match (&llm, pre_consent) {
            (None, _) => ChatConsent::Unavailable,
            (Some(_), true) => ChatConsent::Accepted,
            (Some(_), false) => ChatConsent::Pending,
        };
        self.chat_llm = llm;
    }

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
    pub fn request_detect(&mut self) {
        if self.chat_detecting || self.chat_detect_offer.is_some() {
            return;
        }
        self.chat_detecting = true;
        self.chat_detect_msg = None;
        self.chat_detect_dispatch = true;
    }

    /// Record the result of a detect attempt: `Some(cfg)` raises the offer
    /// prompt; `None` records a "nothing found" message. Clears the in-flight
    /// flag either way.
    pub fn set_detect_result(&mut self, offer: Option<crate::llm::LlmConfig>) {
        self.chat_detecting = false;
        if let Some(cfg) = offer {
            self.chat_detect_msg = None;
            self.chat_detect_offer = Some(cfg);
        } else {
            self.chat_detect_offer = None;
            self.chat_detect_msg =
                Some("no local engine found (Lemonade :13305 / vLLM :8000)".into());
        }
    }

    /// Accept the detected local endpoint for this session: switch `chat_llm`
    /// to the offer and enable chat. No-op when no offer is pending.
    pub fn accept_detect_offer(&mut self) {
        if let Some(cfg) = self.chat_detect_offer.take() {
            self.chat_llm = Some(cfg);
            self.chat_consent = ChatConsent::Accepted;
            self.chat_focused = true;
        }
    }

    /// Dismiss the detected-endpoint offer, leaving the prior chat config and
    /// consent untouched.
    pub fn dismiss_detect_offer(&mut self) {
        self.chat_detect_offer = None;
    }

    /// Accept the detected endpoint **and** persist it: same as
    /// [`accept_detect_offer`](Self::accept_detect_offer), then raise the
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
            ActiveTab::Instances => self.instances.len(),
            ActiveTab::Bench => self.bench_rows.len(),
            ActiveTab::Hardware => self.latest.as_ref().map_or(0, |s| s.gpus.len()),
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

    pub fn select_first(&mut self) {
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
            ActiveTab::Instances => self.instance_sel,
            ActiveTab::Bench => self.bench_sel,
            ActiveTab::Hardware => self.gpu_sel,
            _ => 0,
        }
    }

    fn set_selection(&mut self, tab: ActiveTab, idx: usize) {
        match tab {
            ActiveTab::Instances => self.instance_sel = idx,
            ActiveTab::Bench => self.bench_sel = idx,
            ActiveTab::Hardware => {
                self.gpu_sel = idx;
                self.sync_gpu_scroll();
            }
            _ => {}
        }
    }

    /// Advance `gpu_scroll` so the selected GPU stays within the visible window.
    /// Derives the visible row count from the last rendered body area; with no
    /// prior draw it is a no-op (the renderer self-corrects on the next frame).
    fn sync_gpu_scroll(&mut self) {
        let n = self.latest.as_ref().map_or(0, |s| s.gpus.len());
        let body_h = self.last_body_area.map_or(0, |r| r.height);
        let visible = crate::ui::tabs::hardware::gpu_visible_count(body_h, n);
        self.gpu_scroll =
            crate::ui::tabs::hardware::scroll_to_show(self.gpu_sel, self.gpu_scroll, visible);
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

pub async fn run(args: ResolvedArgs) -> color_eyre::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = event_loop(&mut terminal, &args).await;

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    res
}

async fn event_loop(terminal: &mut Tui, args: &ResolvedArgs) -> color_eyre::Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
    // Job-bridge channel (Phase 3 Wave 1): the async runtime streams
    // `StateEvent`s (JobLine/JobDone/JobErr) for operational screens here.
    let (job_tx, mut job_rx) = mpsc::unbounded_channel::<rocm_dash_core::state::StateEvent>();
    // Retain a sender for chat replies BEFORE `tx` is moved into the client /
    // replay task below — the spawned agent task feeds replies back through the
    // same `rx.recv()` arm the daemon events already use (no new plumbing).
    let chat_tx = tx.clone();
    let replay_controller = if let Some(path) = args.replay.clone() {
        Some(crate::replay::spawn(path, tx))
    } else {
        client::spawn(args.connect.clone(), tx);
        None
    };

    let mut events = EventStream::new();
    let mut tick = interval(Duration::from_millis(250));
    let connect_label = match &args.replay {
        Some(p) => format!(
            "replay:{}",
            p.file_name().and_then(|n| n.to_str()).unwrap_or("?")
        ),
        None => args.connect.clone(),
    };
    let mut state = AppState::new(connect_label, args.theme.clone());
    // Honor the chat-first vs dashboard launch choice (rocm-cli semantics).
    state.active_tab = args.initial_tab;
    // Serve-wizard recipe picker source (Phase 3 Wave 1), adapted by the bin.
    state.model_recipes = args.model_recipes.clone();
    // Runtime manager source (Phase 3 Wave 2), adapted by the bin.
    state.runtimes = args.runtimes.clone();
    // Automations manager source (Phase 3 Wave 3), adapted by the bin.
    state.automations = args.automations.clone();
    state.replay = replay_controller.map(ReplayState::new);

    // Resolve the chat backend. `--chat-mock` short-circuits detection with a
    // deterministic offline MockAgentClient (no live LLM, no network); otherwise
    // we auto-detect the endpoint (the std-TCP probe runs once on a blocking
    // thread before the first frame) and build the Rig backend.
    let agent: Option<std::sync::Arc<dyn crate::agent::AgentClient>> = if args.chat_mock {
        state.set_chat_config(
            Some(crate::llm::LlmConfig {
                base_url: "mock://offline-demo".to_string(),
                model: "mock-agent".to_string(),
                api_key: None,
                auth_header: None,
            }),
            true,
        );
        Some(
            std::sync::Arc::new(crate::agent::MockAgentClient::with_tool_call(
                "GPU-2 is running hot: 87% util, 71°C, drawing 250 W (90 GB/192 GB VRAM).",
                "gpu_status",
            )) as std::sync::Arc<dyn crate::agent::AgentClient>,
        )
    } else {
        let probe_target = args
            .chat_url
            .clone()
            .or_else(|| args.chat_env_url.clone())
            .unwrap_or_else(|| crate::llm::DEFAULT_CHAT_BASE_URL.to_string());
        let probe_ok = tokio::task::spawn_blocking(move || {
            crate::llm::probe_endpoint(&probe_target, crate::llm::PROBE_TIMEOUT)
        })
        .await
        .unwrap_or(false);
        let llm = crate::llm::resolve_llm_config(
            args.chat_url.as_deref(),
            args.chat_model.as_deref(),
            None,
            None,
            args.chat_api_key.as_deref(),
            args.chat_env_url.as_deref(),
            args.chat_auth_header.as_deref(),
            probe_ok,
        );
        state.set_chat_config(llm, args.chat_auto_consent);
        // No reachable local endpoint AND no key/url configured → the no-key
        // ChatGPT OAuth default (device-code login surfaced in the chat tab).
        // This restores the no-key login the vendored Codex path provided; it
        // takes NO api_key (env-only invariant untouched — OAuth, not a key).
        let no_key_no_endpoint = !probe_ok
            && args.chat_api_key.is_none()
            && args.chat_url.is_none()
            && args.chat_env_url.is_none();
        if no_key_no_endpoint {
            let oauth_tx = chat_tx.clone();
            crate::agent::ChatGptAgentClient::new(args.chat_model.clone(), move |url, code| {
                let _ = oauth_tx.send(ClientMsg::ChatReply {
                    text: format!(
                        "To enable chat, sign in to ChatGPT: open {url} and enter the code {code}"
                    ),
                });
            })
            .ok()
            .map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn crate::agent::AgentClient>)
        } else {
            // A build failure leaves `agent` None; a submit surfaces an error turn.
            match &state.chat_llm {
                Some(cfg) => crate::agent::RigAgentClient::new(cfg.clone())
                    .ok()
                    .map(|c| {
                        std::sync::Arc::new(c) as std::sync::Arc<dyn crate::agent::AgentClient>
                    }),
                None => None,
            }
        }
    };

    loop {
        terminal.draw(|f| ui::draw(f, &mut state))?;
        tokio::select! {
            _ = tick.tick() => { /* repaint */ }
            maybe_msg = rx.recv() => {
                match maybe_msg {
                    Some(ClientMsg::Connecting) => state.conn = ConnState::Connecting,
                    Some(ClientMsg::Connected { host, daemon_version }) => {
                        state.conn = ConnState::Connected { host, version: daemon_version };
                    }
                    Some(ClientMsg::Disconnected { reason }) => {
                        state.conn = ConnState::Disconnected { reason };
                        state.latest = None;
                    }
                    Some(ClientMsg::Event(ev)) => state.apply_event(*ev),
                    Some(ClientMsg::ReplaySeek) => state.reset_for_seek(),
                    Some(ClientMsg::ReplayPosition { elapsed_s, total_s }) => {
                        if let Some(r) = state.replay.as_mut() {
                            r.elapsed_s = elapsed_s;
                            r.total_s = total_s;
                        }
                    }
                    Some(ClientMsg::ChatReply { text }) => state.on_chat_reply(text),
                    Some(ClientMsg::ChatError { message }) => state.on_chat_error(message),
                    Some(ClientMsg::ChatDetectResult { offer }) => state.set_detect_result(offer),
                    None => break,
                }
            }
            // Job-bridge events feed the operational-screen job model (Wave 1).
            maybe_job = job_rx.recv() => {
                if let Some(ev) = maybe_job {
                    let fx = state.jobs.apply(ev);
                    crate::jobs::run_effects(fx, &job_tx);
                }
            }
            maybe_ev = events.next() => {
                match maybe_ev {
                    // The services-manager overlay, when open, owns all keys
                    // (and may spawn lifecycle jobs through the job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.services.is_some() => {
                        let fx = crate::ui::services_manager::on_key(
                            &mut state.services,
                            &mut state.jobs,
                            &state.instances,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The serve-wizard overlay, when open, owns all keys (and may
                    // spawn a launch job through the job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.serve_wizard.is_some() => {
                        let fx = crate::ui::serve_wizard::on_key(
                            &mut state.serve_wizard,
                            &mut state.jobs,
                            &state.model_recipes,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The engine-manager overlay, when open, owns all keys (and
                    // may stream an install job through the job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.engine_manager.is_some() => {
                        let fx = crate::ui::engine_manager::on_key(
                            &mut state.engine_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The examine overlay, when open, owns all keys (read-only
                    // `rocm examine` job through the job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.examine_manager.is_some() => {
                        let fx = crate::ui::examine_manager::on_key(
                            &mut state.examine_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The update overlay, when open, owns all keys (check/preview
                    // read-only; apply gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.update_manager.is_some() => {
                        let fx = crate::ui::update_manager::on_key(
                            &mut state.update_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The install overlay, when open, owns all keys (dry-run
                    // read-only; install gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.install_manager.is_some() => {
                        let fx = crate::ui::install_manager::on_key(
                            &mut state.install_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The logs overlay, when open, owns all keys (read-only
                    // `rocm logs` through the job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.logs_view.is_some() => {
                        let fx = crate::ui::logs_view::on_key(
                            &mut state.logs_view,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The runtime manager, when open, owns all keys (refresh
                    // read-only; activate/rollback/uninstall/adopt/import gated).
                    Some(Ok(CtEvent::Key(k))) if state.runtime_manager.is_some() => {
                        let fx = crate::ui::runtime_manager::on_key(
                            &mut state.runtime_manager,
                            &state.runtimes,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The onboarding wizard, when open, owns all keys (install /
                    // adopt gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.onboarding.is_some() => {
                        let fx = crate::ui::onboarding::on_key(
                            &mut state.onboarding,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The automations manager, when open, owns all keys (refresh
                    // read-only; enable/disable gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.automations_manager.is_some() => {
                        let fx = crate::ui::automations_manager::on_key(
                            &mut state.automations_manager,
                            &state.automations,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The command runner, when open, owns all keys (every
                    // command gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.command_screen.is_some() => {
                        let fx = crate::ui::command_screen::on_key(
                            &mut state.command_screen,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The config & provider manager, when open, owns all keys
                    // (show read-only; provider toggles gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.config_manager.is_some() => {
                        let fx = crate::ui::config_manager::on_key(
                            &mut state.config_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    Some(Ok(CtEvent::Key(k))) => {
                        let chat_ctx = ChatKeyCtx {
                            focused: state.chat_focused,
                            consent: state.chat_consent,
                            offer_pending: state.chat_detect_offer.is_some(),
                        };
                        let action = handle_key(k, state.active_tab, &state.modal, chat_ctx);
                        if apply_action(&mut state, action) {
                            break;
                        }
                    }
                    Some(Ok(CtEvent::Mouse(me))) => {
                        let action = resolve_mouse(me, &state);
                        if apply_action(&mut state, action) {
                            break;
                        }
                    }
                    Some(Ok(CtEvent::Resize(_, _))) => { /* repaint */ }
                    Some(Err(e)) => return Err(e.into()),
                    None => break,
                    _ => {}
                }
            }
        }

        // Spawn the agent round-trip on the submit edge — keeps `apply_action`
        // I/O-free. `chat_dispatch` is raised once by `submit_chat`; consume it
        // so the in-flight request is spawned exactly once (not every tick).
        if state.chat_dispatch {
            state.chat_dispatch = false;
            match agent.clone() {
                Some(agent) => {
                    let history = state.chat.clone();
                    let snapshot = state.state_snapshot();
                    let reply_tx = chat_tx.clone();
                    tokio::spawn(async move {
                        let msg = match agent.complete(&history, snapshot).await {
                            Ok(text) => ClientMsg::ChatReply { text },
                            Err(e) => ClientMsg::ChatError {
                                message: e.to_string(),
                            },
                        };
                        let _ = reply_tx.send(msg);
                    });
                }
                None => state.on_chat_error("chat backend unavailable".to_string()),
            }
        }

        // Run the local-engine probe + `/v1/models` query on the detect edge,
        // off the reducer. Raised once by `request_detect`; result returns via
        // `ClientMsg::ChatDetectResult`.
        if state.chat_detect_dispatch {
            state.chat_detect_dispatch = false;
            let reply_tx = chat_tx.clone();
            tokio::spawn(async move {
                let offer = detect_local_chat().await;
                let _ = reply_tx.send(ClientMsg::ChatDetectResult { offer });
            });
        }

        // Persist the accepted endpoint on the save edge (a small synchronous
        // file write; the message surfaces success/failure on the gate is not
        // shown once Accepted, so we keep it terse via tracing + chat_detect_msg).
        if state.chat_persist_dispatch {
            state.chat_persist_dispatch = false;
            if let Some(cfg) = state.chat_llm.clone() {
                match persist_chat_endpoint(&cfg.base_url, &cfg.model) {
                    Ok(path) => {
                        tracing::info!(?path, "saved chat endpoint to config");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to save chat endpoint");
                        state.chat_detect_msg = Some(format!("could not save config: {e}"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Probe for a local engine (TCP), then query its `/v1/models` to choose a
/// model. Returns a ready-to-use [`LlmConfig`], or `None` when nothing is
/// reachable. All I/O lives here, outside the reducer.
async fn detect_local_chat() -> Option<crate::llm::LlmConfig> {
    // TCP probe is blocking; keep it off the async reactor.
    let base = tokio::task::spawn_blocking(crate::llm::detect_local_endpoint)
        .await
        .ok()
        .flatten()?;

    // Best-effort model query; fall back to the neutral default on any failure.
    let model = fetch_first_model(base)
        .await
        .unwrap_or_else(|| crate::llm::DEFAULT_CHAT_MODEL.to_string());
    Some(crate::llm::detected_llm_config(base, &model))
}

/// Persist an accepted local endpoint to the user's `config.toml`: load the
/// existing config (or defaults), set `tui.chat_url`/`tui.chat_model`, and write
/// it back. Best-effort — returns a human error string on failure.
///
/// Uses [`default_config_path`] (a `--config` override is not honored by this
/// in-TUI save; that's a documented limitation). All I/O lives here.
fn persist_chat_endpoint(base_url: &str, model: &str) -> Result<std::path::PathBuf, String> {
    use rocm_dash_core::config::{Config, default_config_path};
    let path = default_config_path().ok_or_else(|| "no config path available".to_string())?;
    let cfg = Config::load(&path).unwrap_or_default();
    let next = config_with_chat(cfg, base_url, model);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let toml = toml::to_string_pretty(&next).map_err(|e| e.to_string())?;
    std::fs::write(&path, toml).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Pure immutable transform: return a copy of `cfg` with the chat endpoint set
/// to a local engine (base_url + model), clearing any gateway auth header since
/// local engines need none.
fn config_with_chat(
    mut cfg: rocm_dash_core::config::Config,
    base_url: &str,
    model: &str,
) -> rocm_dash_core::config::Config {
    cfg.tui.chat_url = Some(base_url.to_string());
    cfg.tui.chat_model = Some(model.to_string());
    cfg.tui.chat_auth_header = None;
    cfg
}

/// GET `{base}/models` and return the first served model id, or `None`.
async fn fetch_first_model(base_url: &str) -> Option<String> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok()?;
    let resp = client.get(&url).send().await.ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    crate::llm::pick_first_model(&json)
}

/// Apply a `KeyAction` to mutable state. Returns `true` when the action
/// requests application exit (Quit).
fn apply_action(state: &mut AppState, action: KeyAction) -> bool {
    match action {
        KeyAction::Quit => return true,
        KeyAction::SwitchTab(t) => {
            state.active_tab = t;
            state.modal = Modal::None;
        }
        KeyAction::Move(d) => {
            if state.modal == Modal::ThemePicker {
                state.theme_picker_move(d);
            } else {
                state.move_selection(d);
            }
        }
        KeyAction::SelectFirst => {
            if state.modal == Modal::ThemePicker {
                state.theme_picker_first();
            } else {
                state.select_first();
            }
        }
        KeyAction::SelectLast => {
            if state.modal == Modal::ThemePicker {
                state.theme_picker_last();
            } else {
                state.select_last();
            }
        }
        KeyAction::OpenDetail => {
            if state.selection_len() > 0 {
                state.modal = Modal::Detail;
                if state.active_tab == ActiveTab::Bench {
                    state.reset_bench_detail_scroll();
                }
            }
        }
        KeyAction::ToggleHelp => {
            state.modal = if state.modal == Modal::Help {
                Modal::None
            } else {
                Modal::Help
            };
        }
        KeyAction::CloseModal => state.modal = Modal::None,
        // The operational overlays are mutually exclusive: opening any one first
        // closes the rest (see `close_overlays`), so no open path — key, mouse,
        // or effect — can ever leave two `Some` at once.
        KeyAction::OpenServices => {
            state.close_overlays();
            state.services = Some(crate::ui::services_manager::ServicesManagerState::default());
        }
        KeyAction::OpenServeWizard => {
            state.close_overlays();
            state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState::default());
        }
        KeyAction::OpenEngineManager => {
            state.close_overlays();
            state.engine_manager = Some(crate::ui::engine_manager::EngineManagerState::default());
        }
        KeyAction::OpenExamine => {
            state.close_overlays();
            state.examine_manager =
                Some(crate::ui::examine_manager::ExamineManagerState::default());
        }
        KeyAction::OpenUpdate => {
            state.close_overlays();
            state.update_manager = Some(crate::ui::update_manager::UpdateManagerState::default());
        }
        KeyAction::OpenInstall => {
            state.close_overlays();
            state.install_manager =
                Some(crate::ui::install_manager::InstallManagerState::default());
        }
        KeyAction::OpenLogs => {
            state.close_overlays();
            state.logs_view = Some(crate::ui::logs_view::LogsViewState::default());
        }
        KeyAction::OpenRuntimes => {
            state.close_overlays();
            state.runtime_manager =
                Some(crate::ui::runtime_manager::RuntimeManagerState::default());
        }
        KeyAction::OpenOnboarding => {
            state.close_overlays();
            state.onboarding = Some(crate::ui::onboarding::OnboardingState::default());
        }
        KeyAction::OpenAutomations => {
            state.close_overlays();
            state.automations_manager =
                Some(crate::ui::automations_manager::AutomationsManagerState::default());
        }
        KeyAction::OpenCommand => {
            state.close_overlays();
            state.command_screen = Some(crate::ui::command_screen::CommandScreenState::default());
        }
        KeyAction::OpenConfig => {
            state.close_overlays();
            state.config_manager = Some(crate::ui::config_manager::ConfigManagerState::default());
        }
        KeyAction::OpenThemePicker => state.open_theme_picker(),
        KeyAction::ApplyThemePick => state.apply_theme_pick(),
        KeyAction::ScrollModal(d) => {
            if state.active_tab == ActiveTab::Bench && state.modal == Modal::Detail {
                if d == i16::MIN {
                    state.bench_detail_scroll = 0;
                } else if d == i16::MAX {
                    state.bench_detail_scroll = u16::MAX;
                } else {
                    state.scroll_bench_detail(d);
                }
            }
        }
        KeyAction::ReplayTogglePause => {
            if let Some(r) = state.replay.as_mut() {
                r.paused = !r.paused;
                if r.paused {
                    r.controller.pause();
                } else {
                    r.controller.resume();
                }
            }
        }
        KeyAction::ReplaySpeedUp => {
            if let Some(r) = state.replay.as_mut() {
                r.speed = crate::replay::next_speed(r.speed);
                r.controller.set_speed(r.speed);
            }
        }
        KeyAction::ReplaySpeedDown => {
            if let Some(r) = state.replay.as_mut() {
                r.speed = crate::replay::prev_speed(r.speed);
                r.controller.set_speed(r.speed);
            }
        }
        KeyAction::ReplayJump(delta_s) => {
            if let Some(r) = state.replay.as_ref() {
                r.controller.jump(delta_s);
            }
        }
        KeyAction::ChatInput(c) => state.chat_input.push(c),
        KeyAction::ChatBackspace => {
            state.chat_input.pop();
        }
        KeyAction::ChatSubmit => state.submit_chat(),
        KeyAction::ChatFocus => state.chat_focused = true,
        KeyAction::ChatBlur => state.chat_focused = false,
        KeyAction::ChatConsentAccept => state.accept_chat_consent(),
        KeyAction::ChatConsentDecline => state.decline_chat_consent(),
        KeyAction::ChatDetect => state.request_detect(),
        KeyAction::ChatDetectAccept => state.accept_detect_offer(),
        KeyAction::ChatDetectSave => state.save_detect_offer(),
        KeyAction::ChatDetectDismiss => state.dismiss_detect_offer(),
        KeyAction::ChatScroll(d) => {
            let next = (i32::from(state.chat_scroll) + i32::from(d)).max(0) as u16;
            state.chat_scroll = next;
        }
        KeyAction::Nothing => {}
    }
    false
}

/// Translate a mouse event into a `KeyAction` using whatever state context
/// is needed (last drawn areas, active tab, current modal).
fn resolve_mouse(me: MouseEvent, state: &AppState) -> KeyAction {
    if me.kind == MouseEventKind::Down(MouseButton::Left) {
        if let Some(area) = state.last_tab_bar_area
            && let Some(tab) = tab_bar_hit(area, me.column, me.row)
        {
            return KeyAction::SwitchTab(tab);
        }
        if state.modal == Modal::None
            && let Some(area) = state.last_body_area
        {
            let action = match state.active_tab {
                ActiveTab::Instances => {
                    ui::tabs::instances::hit_test(area, me.column, me.row, state)
                }
                ActiveTab::Bench => ui::tabs::bench::hit_test(area, me.column, me.row, state),
                _ => None,
            };
            if let Some(a) = action {
                return a;
            }
        }
        return KeyAction::Nothing;
    }
    handle_mouse(me, &state.modal, state.active_tab)
}

#[derive(Debug, PartialEq, Eq)]
pub enum KeyAction {
    Nothing,
    Quit,
    SwitchTab(ActiveTab),
    Move(isize),
    SelectFirst,
    SelectLast,
    OpenDetail,
    ToggleHelp,
    CloseModal,
    OpenThemePicker,
    ApplyThemePick,
    /// Vertical scroll inside the active modal body (positive = down).
    ScrollModal(i16),
    /// Toggle replay pause / resume. No-op when not replaying.
    ReplayTogglePause,
    /// Step replay speed up or down (clamped). No-op when not replaying.
    ReplaySpeedUp,
    ReplaySpeedDown,
    /// Move the replay playhead by `delta_s` seconds (negative = rewind).
    ReplayJump(i64),
    /// Chat insert-mode: append a character to `chat_input`.
    ChatInput(char),
    /// Chat insert-mode: pop the last character from `chat_input`.
    ChatBackspace,
    /// Chat: submit the current input buffer as a user turn.
    ChatSubmit,
    /// Chat: enter text-entry focus.
    ChatFocus,
    /// Chat: leave text-entry focus.
    ChatBlur,
    /// Chat: accept the detected endpoint (one-time consent).
    ChatConsentAccept,
    /// Chat: decline the detected endpoint.
    ChatConsentDecline,
    /// Chat: probe for a local engine and offer it (in-TUI auto-detect).
    ChatDetect,
    /// Chat: accept the detected local endpoint for this session.
    ChatDetectAccept,
    /// Chat: accept the detected endpoint and persist it to config.
    ChatDetectSave,
    /// Chat: dismiss the detected-endpoint offer, keeping the prior config.
    ChatDetectDismiss,
    /// Chat: scroll the transcript by N lines (positive = down).
    ChatScroll(i16),
    /// Open the services-manager overlay (Phase 3 Wave 1).
    OpenServices,
    /// Open the serve-wizard overlay (Phase 3 Wave 1).
    OpenServeWizard,
    /// Open the engine-manager overlay (Phase 3 Wave 1).
    OpenEngineManager,
    /// Open the examine overlay (Phase 3 Wave 2).
    OpenExamine,
    /// Open the update overlay (Phase 3 Wave 2).
    OpenUpdate,
    /// Open the install overlay (Phase 3 Wave 2).
    OpenInstall,
    /// Open the runtime manager overlay.
    OpenRuntimes,
    /// Open the onboarding wizard overlay.
    OpenOnboarding,
    /// Open the automations manager overlay.
    OpenAutomations,
    /// Open the command runner overlay.
    OpenCommand,
    /// Open the config & provider manager overlay.
    OpenConfig,
    /// Open the logs overlay (Phase 3 Wave 3).
    OpenLogs,
}

fn handle_key(k: KeyEvent, current: ActiveTab, modal: &Modal, chat: ChatKeyCtx) -> KeyAction {
    if k.kind != KeyEventKind::Press {
        return KeyAction::Nothing;
    }
    // Chat tab key handling, placed BEFORE the global hotkey match so focused
    // text entry and the consent prompt absorb keys (the short-circuit that
    // stops `q`, `1`–`5`, etc. from firing while typing / deciding consent).
    if current == ActiveTab::Chat && *modal == Modal::None {
        // A detected-endpoint offer (gate-only) absorbs its decision keys before
        // the normal consent prompt: y use now, n/Esc dismiss. ([s] use & save
        // is wired with persistence.) Other keys fall through to the globals.
        if chat.offer_pending && chat.consent != ChatConsent::Accepted {
            match k.code {
                KeyCode::Char('y' | 'Y') | KeyCode::Enter => {
                    return KeyAction::ChatDetectAccept;
                }
                KeyCode::Char('s' | 'S') => {
                    return KeyAction::ChatDetectSave;
                }
                KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                    return KeyAction::ChatDetectDismiss;
                }
                _ => {}
            }
        }
        match chat.consent {
            ChatConsent::Accepted => {
                // History scroll works whether or not the input is focused.
                match k.code {
                    KeyCode::PageUp => return KeyAction::ChatScroll(-CHAT_SCROLL_STEP),
                    KeyCode::PageDown => return KeyAction::ChatScroll(CHAT_SCROLL_STEP),
                    _ => {}
                }
                if chat.focused {
                    return match k.code {
                        KeyCode::Esc => KeyAction::ChatBlur,
                        KeyCode::Enter => KeyAction::ChatSubmit,
                        KeyCode::Backspace => KeyAction::ChatBackspace,
                        KeyCode::Char(c) => KeyAction::ChatInput(c),
                        _ => KeyAction::Nothing,
                    };
                }
                // Not focused: `i`/`Enter` enter insert mode; other keys fall
                // through to the global hotkeys below.
                if let KeyCode::Char('i') | KeyCode::Enter = k.code {
                    return KeyAction::ChatFocus;
                }
            }
            ChatConsent::Pending | ChatConsent::Declined => {
                // Consent gate: y/Enter accept, n decline, d detect a local
                // engine. Other keys (q, digits, Tab, ?) fall through to the
                // globals so the user isn't trapped.
                match k.code {
                    KeyCode::Char('y' | 'Y') | KeyCode::Enter => {
                        return KeyAction::ChatConsentAccept;
                    }
                    KeyCode::Char('n' | 'N') => {
                        return KeyAction::ChatConsentDecline;
                    }
                    KeyCode::Char('d' | 'D') => {
                        return KeyAction::ChatDetect;
                    }
                    _ => {}
                }
            }
            // No endpoint configured: the only gate action is to detect one.
            ChatConsent::Unavailable => {
                if let KeyCode::Char('d' | 'D') = k.code {
                    return KeyAction::ChatDetect;
                }
            }
        }
    }
    // ThemePicker is a navigable modal — j/k/g/G move the cursor, Enter applies.
    if *modal == Modal::ThemePicker {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc | KeyCode::Char('t') => KeyAction::CloseModal,
            KeyCode::Enter => KeyAction::ApplyThemePick,
            KeyCode::Char('j') | KeyCode::Down => KeyAction::Move(1),
            KeyCode::Char('k') | KeyCode::Up => KeyAction::Move(-1),
            KeyCode::Char('g') | KeyCode::Home => KeyAction::SelectFirst,
            KeyCode::Char('G') | KeyCode::End => KeyAction::SelectLast,
            _ => KeyAction::Nothing,
        };
    }
    // Detail modal: vertical scroll keys, plus quit/close.
    if *modal == Modal::Detail {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('?') => KeyAction::CloseModal,
            KeyCode::Char('j') | KeyCode::Down => KeyAction::ScrollModal(1),
            KeyCode::Char('k') | KeyCode::Up => KeyAction::ScrollModal(-1),
            KeyCode::PageDown => KeyAction::ScrollModal(10),
            KeyCode::PageUp => KeyAction::ScrollModal(-10),
            KeyCode::Char('g') | KeyCode::Home => KeyAction::ScrollModal(i16::MIN),
            KeyCode::Char('G') | KeyCode::End => KeyAction::ScrollModal(i16::MAX),
            _ => KeyAction::Nothing,
        };
    }
    // Help absorbs everything except quit / close / ? toggle.
    if *modal == Modal::Help {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('?') => KeyAction::CloseModal,
            _ => KeyAction::Nothing,
        };
    }
    match k.code {
        KeyCode::Char('q') => KeyAction::Quit,
        KeyCode::Esc => KeyAction::Nothing, // Esc no longer quits; it closes modals.
        KeyCode::Char('?') => KeyAction::ToggleHelp,
        KeyCode::Char('t') => KeyAction::OpenThemePicker,
        KeyCode::BackTab => KeyAction::SwitchTab(current.prev()),
        KeyCode::Tab => {
            if k.modifiers.contains(KeyModifiers::SHIFT) {
                KeyAction::SwitchTab(current.prev())
            } else {
                KeyAction::SwitchTab(current.next())
            }
        }
        KeyCode::Char(c @ '1'..='5') => match ActiveTab::from_digit(c) {
            Some(t) => KeyAction::SwitchTab(t),
            None => KeyAction::Nothing,
        },
        KeyCode::PageDown => KeyAction::Move(10),
        KeyCode::PageUp => KeyAction::Move(-10),
        KeyCode::Char(' ') => KeyAction::ReplayTogglePause,
        KeyCode::Char('+' | '=') => KeyAction::ReplaySpeedUp,
        KeyCode::Char('-' | '_') => KeyAction::ReplaySpeedDown,
        KeyCode::Char('[') => KeyAction::ReplayJump(-10),
        KeyCode::Char(']') => KeyAction::ReplayJump(10),
        KeyCode::Char('{') => KeyAction::ReplayJump(-60),
        KeyCode::Char('}') => KeyAction::ReplayJump(60),
        KeyCode::Char('j') | KeyCode::Down => KeyAction::Move(1),
        KeyCode::Char('k') | KeyCode::Up => KeyAction::Move(-1),
        KeyCode::Char('g') | KeyCode::Home => KeyAction::SelectFirst,
        KeyCode::Char('G') | KeyCode::End => KeyAction::SelectLast,
        // Services manager: open from the Instances tab (where servers live).
        KeyCode::Char('s') if current == ActiveTab::Instances => KeyAction::OpenServices,
        // Serve wizard: launch a model from the Overview or Instances tab.
        KeyCode::Char('w') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenServeWizard
        }
        // Engine manager: use/install/reinstall serving engines.
        KeyCode::Char('e') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenEngineManager
        }
        // Examine: read-only environment check.
        KeyCode::Char('d') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenExamine
        }
        // Update: check/preview/apply ROCm package updates.
        KeyCode::Char('u') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenUpdate
        }
        // Install: ROCm SDK (TheRock) install / dry-run.
        KeyCode::Char('i') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenInstall
        }
        // Logs: browse recent ROCm CLI logs.
        KeyCode::Char('l') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenLogs
        }
        // Runtimes: list/activate/adopt/import ROCm runtimes.
        KeyCode::Char('r') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenRuntimes
        }
        // Onboarding: first-run setup wizard (install / adopt).
        KeyCode::Char('n') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenOnboarding
        }
        // Automations: list/enable/disable background checks.
        KeyCode::Char('a') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenAutomations
        }
        // Command runner: run any ROCm CLI subcommand (gated).
        KeyCode::Char('c') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenCommand
        }
        // Config & providers.
        KeyCode::Char('p') if matches!(current, ActiveTab::Overview | ActiveTab::Instances) => {
            KeyAction::OpenConfig
        }
        KeyCode::Enter => KeyAction::OpenDetail,
        _ => KeyAction::Nothing,
    }
}

/// Translate a `MouseEvent` into the existing `KeyAction` vocabulary.
///
/// The caller is responsible for the surrounding state context:
/// - `last_tab_bar_area` / `last_body_area` are read off `AppState` by the
///   event loop so this function stays pure on the input event.
/// - Per-tab body clicks are dispatched to the active tab module's
///   `hit_test` from the event loop.
///
/// We only translate the parts of mouse handling that are tab-agnostic:
/// the scroll wheel, and (in the event loop) the tab-bar click. Per-tab
/// click is handled in tab modules.
pub fn handle_mouse(ev: MouseEvent, modal: &Modal, tab: ActiveTab) -> KeyAction {
    let delta: i16 = match ev.kind {
        MouseEventKind::ScrollDown => 3,
        MouseEventKind::ScrollUp => -3,
        _ => return KeyAction::Nothing,
    };
    if *modal == Modal::Detail {
        KeyAction::ScrollModal(delta)
    } else if *modal == Modal::ThemePicker
        || (*modal == Modal::None && matches!(tab, ActiveTab::Instances | ActiveTab::Bench))
    {
        KeyAction::Move(delta as isize)
    } else {
        KeyAction::Nothing
    }
}

/// Resolve a left-click at `(x, y)` against `tab_bar_area`. Returns the tab
/// to switch to, or `None` if the click is outside or doesn't land on a chip.
///
/// Uses [`ui::tabs::compute_chip_layout`] so the hit-test geometry exactly
/// mirrors what `draw_tab_bar` rendered. Separator gaps (` · `) between
/// chips are intentional dead zones — clicking the dot does nothing.
pub fn tab_bar_hit(tab_bar_area: ratatui::layout::Rect, x: u16, y: u16) -> Option<ActiveTab> {
    if y != tab_bar_area.y {
        return None;
    }
    let chips = ui::tabs::compute_chip_layout(tab_bar_area.x);
    let bar_right = tab_bar_area.x.saturating_add(tab_bar_area.width);
    for chip in chips {
        if chip.x_end > bar_right {
            // Chip overflows the bar — terminal too narrow to show it; skip.
            continue;
        }
        if x >= chip.x_start && x < chip.x_end {
            return Some(chip.tab);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::layout::Rect;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn hk(c: KeyCode, tab: ActiveTab) -> KeyAction {
        handle_key(press(c), tab, &Modal::None, ChatKeyCtx::default())
    }

    #[test]
    fn q_quits_esc_does_not() {
        assert_eq!(hk(KeyCode::Char('q'), ActiveTab::Overview), KeyAction::Quit);
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Bench), KeyAction::Nothing);
    }

    #[test]
    fn tab_cycles_forward_and_wraps() {
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Overview),
            KeyAction::SwitchTab(ActiveTab::Hardware)
        );
        // Bench now precedes Chat; Chat wraps back to Overview.
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Bench),
            KeyAction::SwitchTab(ActiveTab::Chat)
        );
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Chat),
            KeyAction::SwitchTab(ActiveTab::Overview)
        );
    }

    #[test]
    fn back_tab_and_shift_tab_both_cycle_backward() {
        assert_eq!(
            hk(KeyCode::BackTab, ActiveTab::Hardware),
            KeyAction::SwitchTab(ActiveTab::Overview)
        );
        let shift_tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT);
        // Overview's previous tab is now Chat (the new last tab).
        assert_eq!(
            handle_key(
                shift_tab,
                ActiveTab::Overview,
                &Modal::None,
                ChatKeyCtx::default()
            ),
            KeyAction::SwitchTab(ActiveTab::Chat)
        );
    }

    #[test]
    fn number_keys_jump_to_tab() {
        assert_eq!(
            hk(KeyCode::Char('3'), ActiveTab::Overview),
            KeyAction::SwitchTab(ActiveTab::Instances)
        );
        // `5` now reaches the Chat tab (digit guard widened to '1'..='5').
        assert_eq!(
            hk(KeyCode::Char('5'), ActiveTab::Overview),
            KeyAction::SwitchTab(ActiveTab::Chat)
        );
        assert_eq!(
            hk(KeyCode::Char('6'), ActiveTab::Overview),
            KeyAction::Nothing
        );
    }

    #[test]
    fn from_digit_maps_five_to_chat() {
        assert_eq!(ActiveTab::from_digit('5'), Some(ActiveTab::Chat));
        assert_eq!(ActiveTab::from_digit('6'), None);
    }

    #[test]
    fn release_events_are_ignored() {
        let release = KeyEvent::new_with_kind(
            KeyCode::Char('q'),
            KeyModifiers::NONE,
            KeyEventKind::Release,
        );
        assert_eq!(
            handle_key(
                release,
                ActiveTab::Overview,
                &Modal::None,
                ChatKeyCtx::default()
            ),
            KeyAction::Nothing
        );
    }

    #[test]
    fn jk_arrows_and_g_drive_selection() {
        assert_eq!(
            hk(KeyCode::Char('j'), ActiveTab::Instances),
            KeyAction::Move(1)
        );
        assert_eq!(
            hk(KeyCode::Char('k'), ActiveTab::Instances),
            KeyAction::Move(-1)
        );
        assert_eq!(hk(KeyCode::Down, ActiveTab::Bench), KeyAction::Move(1));
        assert_eq!(hk(KeyCode::Up, ActiveTab::Bench), KeyAction::Move(-1));
        assert_eq!(
            hk(KeyCode::Char('g'), ActiveTab::Bench),
            KeyAction::SelectFirst
        );
        assert_eq!(
            hk(KeyCode::Char('G'), ActiveTab::Bench),
            KeyAction::SelectLast
        );
        assert_eq!(hk(KeyCode::Enter, ActiveTab::Bench), KeyAction::OpenDetail);
    }

    #[test]
    fn operational_open_keys_are_tab_scoped() {
        // `s` opens services only on Instances; Nothing elsewhere.
        assert_eq!(
            hk(KeyCode::Char('s'), ActiveTab::Instances),
            KeyAction::OpenServices
        );
        assert_eq!(
            hk(KeyCode::Char('s'), ActiveTab::Overview),
            KeyAction::Nothing
        );
        // `w` / `e` open from Overview + Instances; Nothing on other tabs.
        assert_eq!(
            hk(KeyCode::Char('w'), ActiveTab::Overview),
            KeyAction::OpenServeWizard
        );
        assert_eq!(
            hk(KeyCode::Char('e'), ActiveTab::Instances),
            KeyAction::OpenEngineManager
        );
        assert_eq!(
            hk(KeyCode::Char('w'), ActiveTab::Hardware),
            KeyAction::Nothing
        );
        assert_eq!(hk(KeyCode::Char('e'), ActiveTab::Bench), KeyAction::Nothing);
        // Examine / update open from Overview + Instances.
        assert_eq!(
            hk(KeyCode::Char('d'), ActiveTab::Overview),
            KeyAction::OpenExamine
        );
        assert_eq!(
            hk(KeyCode::Char('u'), ActiveTab::Instances),
            KeyAction::OpenUpdate
        );
        assert_eq!(hk(KeyCode::Char('d'), ActiveTab::Bench), KeyAction::Nothing);
        // Install / logs open from Overview + Instances.
        assert_eq!(
            hk(KeyCode::Char('i'), ActiveTab::Overview),
            KeyAction::OpenInstall
        );
        assert_eq!(
            hk(KeyCode::Char('l'), ActiveTab::Instances),
            KeyAction::OpenLogs
        );
        // On the Chat tab none of these open an overlay (the operational keys
        // are guarded to Overview/Instances). `i` is the one that means
        // something else on Chat — insert mode — never OpenInstall.
        for c in ['w', 'e', 'd', 'u', 'l'] {
            assert_eq!(
                hk(KeyCode::Char(c), ActiveTab::Chat),
                KeyAction::Nothing,
                "key {c} must not open an overlay from Chat"
            );
        }
        assert_eq!(
            hk(KeyCode::Char('i'), ActiveTab::Chat),
            KeyAction::ChatFocus,
            "i is chat-insert on Chat, never OpenInstall"
        );
    }

    #[test]
    fn opening_an_overlay_closes_the_others() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s, KeyAction::OpenServices);
        assert!(s.services.is_some() && s.serve_wizard.is_none() && s.engine_manager.is_none());
        // Opening another overlay (defensive path) clears the prior one.
        apply_action(&mut s, KeyAction::OpenServeWizard);
        assert!(s.serve_wizard.is_some() && s.services.is_none() && s.engine_manager.is_none());
        apply_action(&mut s, KeyAction::OpenEngineManager);
        assert!(s.engine_manager.is_some() && s.services.is_none() && s.serve_wizard.is_none());
        // Wave 2/3 overlays join the mutual-exclusion set.
        apply_action(&mut s, KeyAction::OpenExamine);
        assert!(s.examine_manager.is_some() && s.engine_manager.is_none());
        apply_action(&mut s, KeyAction::OpenUpdate);
        assert!(s.update_manager.is_some() && s.examine_manager.is_none());
        apply_action(&mut s, KeyAction::OpenInstall);
        assert!(s.install_manager.is_some() && s.update_manager.is_none());
        apply_action(&mut s, KeyAction::OpenLogs);
        assert!(s.logs_view.is_some() && s.install_manager.is_none());
        apply_action(&mut s, KeyAction::OpenRuntimes);
        assert!(s.runtime_manager.is_some() && s.logs_view.is_none());
        apply_action(&mut s, KeyAction::OpenOnboarding);
        assert!(s.onboarding.is_some() && s.runtime_manager.is_none());
        apply_action(&mut s, KeyAction::OpenAutomations);
        assert!(s.automations_manager.is_some() && s.onboarding.is_none());
        apply_action(&mut s, KeyAction::OpenCommand);
        assert!(s.command_screen.is_some() && s.automations_manager.is_none());
        apply_action(&mut s, KeyAction::OpenConfig);
        assert!(s.config_manager.is_some() && s.command_screen.is_none());
    }

    #[test]
    fn question_mark_toggles_help() {
        assert_eq!(
            hk(KeyCode::Char('?'), ActiveTab::Overview),
            KeyAction::ToggleHelp
        );
    }

    #[test]
    fn t_opens_theme_picker() {
        assert_eq!(
            hk(KeyCode::Char('t'), ActiveTab::Overview),
            KeyAction::OpenThemePicker
        );
    }

    #[test]
    fn theme_picker_absorbs_navigation_keys() {
        let with_picker = |c| {
            handle_key(
                press(c),
                ActiveTab::Overview,
                &Modal::ThemePicker,
                ChatKeyCtx::default(),
            )
        };
        assert_eq!(with_picker(KeyCode::Char('j')), KeyAction::Move(1));
        assert_eq!(with_picker(KeyCode::Char('k')), KeyAction::Move(-1));
        assert_eq!(with_picker(KeyCode::Enter), KeyAction::ApplyThemePick);
        assert_eq!(with_picker(KeyCode::Esc), KeyAction::CloseModal);
        assert_eq!(with_picker(KeyCode::Char('t')), KeyAction::CloseModal);
        assert_eq!(with_picker(KeyCode::Char('q')), KeyAction::Quit);
        assert_eq!(with_picker(KeyCode::Char('1')), KeyAction::Nothing);
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
    fn tab_bar_hit_matches_per_chip_extents() {
        // Bar wide enough to fit every chip (59 chars wide minimum).
        let bar = Rect::new(0, 0, 80, 1);
        // Inside each chip → its tab.
        assert_eq!(tab_bar_hit(bar, 5, 0), Some(ActiveTab::Overview));
        assert_eq!(tab_bar_hit(bar, 20, 0), Some(ActiveTab::Hardware));
        assert_eq!(tab_bar_hit(bar, 40, 0), Some(ActiveTab::Instances));
        assert_eq!(tab_bar_hit(bar, 55, 0), Some(ActiveTab::Bench));
        // Separator " · " between chips 0 and 1 (x in 13..16) is a dead zone.
        assert_eq!(tab_bar_hit(bar, 14, 0), None);
        // Past the last chip (x >= 59) → None.
        assert_eq!(tab_bar_hit(bar, 60, 0), None);
        // Wrong row.
        assert_eq!(tab_bar_hit(bar, 5, 2), None);
    }

    #[test]
    fn tab_bar_hit_skips_chips_that_overflow_a_narrow_bar() {
        // Bar can only fit the first two chips (29 chars).
        let bar = Rect::new(0, 0, 30, 1);
        assert_eq!(tab_bar_hit(bar, 5, 0), Some(ActiveTab::Overview));
        assert_eq!(tab_bar_hit(bar, 20, 0), Some(ActiveTab::Hardware));
        // Instances chip would be at 32..46 — outside bar → None.
        assert_eq!(tab_bar_hit(bar, 40, 0), None);
    }

    #[test]
    fn tab_bar_hit_honors_x_offset() {
        // Bar offset 10 columns to the right.
        let bar = Rect::new(10, 0, 80, 1);
        assert_eq!(tab_bar_hit(bar, 15, 0), Some(ActiveTab::Overview));
        // Equivalent absolute x of the previous "inside chip 0" test.
        assert_eq!(tab_bar_hit(bar, 5, 0), None);
    }

    #[test]
    fn handle_mouse_routes_scroll_by_modal_and_tab() {
        let scroll_down = MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        // No modal, non-interactive tab → Nothing
        assert_eq!(
            handle_mouse(scroll_down, &Modal::None, ActiveTab::Overview),
            KeyAction::Nothing
        );
        // No modal, Instances → Move
        assert_eq!(
            handle_mouse(scroll_down, &Modal::None, ActiveTab::Instances),
            KeyAction::Move(3)
        );
        // Detail modal → ScrollModal
        assert_eq!(
            handle_mouse(scroll_down, &Modal::Detail, ActiveTab::Bench),
            KeyAction::ScrollModal(3)
        );
        // ThemePicker → Move (drives picker cursor)
        assert_eq!(
            handle_mouse(scroll_down, &Modal::ThemePicker, ActiveTab::Overview),
            KeyAction::Move(3)
        );
    }

    #[test]
    fn scroll_bench_detail_clamps_at_zero() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.bench_detail_scroll = 5;
        s.scroll_bench_detail(-100);
        assert_eq!(s.bench_detail_scroll, 0);
        s.scroll_bench_detail(7);
        assert_eq!(s.bench_detail_scroll, 7);
    }

    #[test]
    fn detail_modal_j_k_emit_scroll() {
        let with_detail = |c| {
            handle_key(
                press(c),
                ActiveTab::Bench,
                &Modal::Detail,
                ChatKeyCtx::default(),
            )
        };
        assert_eq!(with_detail(KeyCode::Char('j')), KeyAction::ScrollModal(1));
        assert_eq!(with_detail(KeyCode::Char('k')), KeyAction::ScrollModal(-1));
        assert_eq!(with_detail(KeyCode::PageDown), KeyAction::ScrollModal(10));
        assert_eq!(
            with_detail(KeyCode::Char('g')),
            KeyAction::ScrollModal(i16::MIN)
        );
        assert_eq!(
            with_detail(KeyCode::Char('G')),
            KeyAction::ScrollModal(i16::MAX)
        );
        assert_eq!(with_detail(KeyCode::Esc), KeyAction::CloseModal);
    }

    #[test]
    fn unknown_initial_theme_falls_back_to_default_dark() {
        let s = AppState::new("test".into(), "nope".into());
        let dark = Theme::default_dark();
        assert_eq!(s.theme.bg, dark.bg);
    }

    #[test]
    fn help_modal_absorbs_navigation() {
        let with_help = |c| {
            handle_key(
                press(c),
                ActiveTab::Bench,
                &Modal::Help,
                ChatKeyCtx::default(),
            )
        };
        // j/k inside Help do nothing (Help has no scrollable body today).
        assert_eq!(with_help(KeyCode::Char('j')), KeyAction::Nothing);
        assert_eq!(with_help(KeyCode::Tab), KeyAction::Nothing);
        assert_eq!(with_help(KeyCode::Esc), KeyAction::CloseModal);
        assert_eq!(with_help(KeyCode::Enter), KeyAction::CloseModal);
        assert_eq!(with_help(KeyCode::Char('q')), KeyAction::Quit);
    }

    #[test]
    fn move_selection_clamps_to_bounds() {
        let mut s = AppState::new("test".into(), "default-dark".into());
        s.active_tab = ActiveTab::Bench;
        for i in 0..5 {
            s.bench_rows.push_back(BenchmarkRow {
                cell: format!("c{i}"),
                ..Default::default()
            });
        }
        s.bench_sel = 0;
        s.move_selection(-3);
        assert_eq!(s.bench_sel, 0);
        s.move_selection(10);
        assert_eq!(s.bench_sel, 4);
        s.select_first();
        assert_eq!(s.bench_sel, 0);
        s.select_last();
        assert_eq!(s.bench_sel, 4);
    }

    #[test]
    fn format_mmss_renders_minutes_and_hours() {
        assert_eq!(format_mmss(0), "0:00");
        assert_eq!(format_mmss(7), "0:07");
        assert_eq!(format_mmss(65), "1:05");
        assert_eq!(format_mmss(599), "9:59");
        assert_eq!(format_mmss(3600), "1:00:00");
        assert_eq!(format_mmss(3661), "1:01:01");
    }

    #[test]
    fn bracket_keys_emit_replay_jump() {
        assert_eq!(
            hk(KeyCode::Char('['), ActiveTab::Overview),
            KeyAction::ReplayJump(-10)
        );
        assert_eq!(
            hk(KeyCode::Char(']'), ActiveTab::Overview),
            KeyAction::ReplayJump(10)
        );
        assert_eq!(
            hk(KeyCode::Char('{'), ActiveTab::Overview),
            KeyAction::ReplayJump(-60)
        );
        assert_eq!(
            hk(KeyCode::Char('}'), ActiveTab::Overview),
            KeyAction::ReplayJump(60)
        );
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
        s.active_tab = ActiveTab::Bench;
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
    fn chat_insert_mode_captures_text_and_shortcircuits_hotkeys() {
        let accepted_focused = ChatKeyCtx {
            focused: true,
            consent: ChatConsent::Accepted,
            offer_pending: false,
        };
        let focused = |c| handle_key(press(c), ActiveTab::Chat, &Modal::None, accepted_focused);
        // Printable chars become input, including ones that are global hotkeys.
        assert_eq!(focused(KeyCode::Char('h')), KeyAction::ChatInput('h'));
        assert_eq!(focused(KeyCode::Char('q')), KeyAction::ChatInput('q'));
        assert_eq!(focused(KeyCode::Char('5')), KeyAction::ChatInput('5'));
        assert_eq!(focused(KeyCode::Backspace), KeyAction::ChatBackspace);
        assert_eq!(focused(KeyCode::Enter), KeyAction::ChatSubmit);
        assert_eq!(focused(KeyCode::Esc), KeyAction::ChatBlur);

        // Accepted but NOT focused: `q` still quits and `i`/Enter enter insert mode.
        let accepted = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Accepted,
            offer_pending: false,
        };
        let unfocused = |c| handle_key(press(c), ActiveTab::Chat, &Modal::None, accepted);
        assert_eq!(unfocused(KeyCode::Char('q')), KeyAction::Quit);
        assert_eq!(unfocused(KeyCode::Char('i')), KeyAction::ChatFocus);
        assert_eq!(unfocused(KeyCode::Enter), KeyAction::ChatFocus);
        assert_eq!(
            unfocused(KeyCode::Char('1')),
            KeyAction::SwitchTab(ActiveTab::Overview)
        );
    }

    #[test]
    fn chat_consent_gate_maps_keys_and_lets_globals_through() {
        let pending = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Pending,
            offer_pending: false,
        };
        let gate = |c| handle_key(press(c), ActiveTab::Chat, &Modal::None, pending);
        // y / Y / Enter accept; n / N decline.
        assert_eq!(gate(KeyCode::Char('y')), KeyAction::ChatConsentAccept);
        assert_eq!(gate(KeyCode::Enter), KeyAction::ChatConsentAccept);
        assert_eq!(gate(KeyCode::Char('n')), KeyAction::ChatConsentDecline);
        // Globals not trapped by the gate: q quits, digit switches tab.
        assert_eq!(gate(KeyCode::Char('q')), KeyAction::Quit);
        assert_eq!(
            gate(KeyCode::Char('2')),
            KeyAction::SwitchTab(ActiveTab::Hardware)
        );
    }

    #[test]
    fn chat_consent_accept_and_decline_transition_state() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        // No endpoint → Unavailable; accept/decline are no-ops.
        s.set_chat_config(None, false);
        assert_eq!(s.chat_consent, ChatConsent::Unavailable);
        apply_action(&mut s, KeyAction::ChatConsentAccept);
        assert_eq!(s.chat_consent, ChatConsent::Unavailable);

        // Endpoint present, no pre-consent → Pending.
        let llm = crate::llm::LlmConfig {
            base_url: "http://127.0.0.1:8000".into(),
            model: "m".into(),
            api_key: None,
            auth_header: None,
        };
        s.set_chat_config(Some(llm.clone()), false);
        assert_eq!(s.chat_consent, ChatConsent::Pending);
        // Accept → Accepted + focused.
        apply_action(&mut s, KeyAction::ChatConsentAccept);
        assert_eq!(s.chat_consent, ChatConsent::Accepted);
        assert!(s.chat_focused);
        // Decline → Declined + unfocused.
        apply_action(&mut s, KeyAction::ChatConsentDecline);
        assert_eq!(s.chat_consent, ChatConsent::Declined);
        assert!(!s.chat_focused);

        // Pre-consent → Accepted immediately.
        s.set_chat_config(Some(llm), true);
        assert_eq!(s.chat_consent, ChatConsent::Accepted);
    }

    #[test]
    fn detect_offer_lifecycle_accept_switches_chat() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Chat;
        // Gateway-configured chat, pending consent.
        let gw = crate::llm::LlmConfig {
            base_url: "https://gw/OpenAI".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("k".into()),
            auth_header: Some("Ocp-Apim-Subscription-Key".into()),
        };
        s.set_chat_config(Some(gw), false);

        // request_detect raises the dispatch edge + detecting flag.
        apply_action(&mut s, KeyAction::ChatDetect);
        assert!(s.chat_detecting && s.chat_detect_dispatch);

        // event_loop reports a detected local engine.
        let local = crate::llm::detected_llm_config("http://localhost:13305/v1", "Llama-3.2-3B");
        s.set_detect_result(Some(local.clone()));
        assert!(!s.chat_detecting);
        assert_eq!(s.chat_detect_offer.as_ref(), Some(&local));

        // Accept the offer → chat switches to the local endpoint + enabled.
        apply_action(&mut s, KeyAction::ChatDetectAccept);
        assert_eq!(s.chat_consent, ChatConsent::Accepted);
        assert_eq!(s.chat_llm.as_ref(), Some(&local));
        assert!(s.chat_detect_offer.is_none());
    }

    #[test]
    fn detect_offer_dismiss_keeps_prior_config() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        let gw = crate::llm::LlmConfig {
            base_url: "https://gw/OpenAI".into(),
            model: "gpt-4o-mini".into(),
            api_key: None,
            auth_header: None,
        };
        s.set_chat_config(Some(gw.clone()), false);
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:8000/v1",
            "x",
        )));
        // Dismiss → offer gone, gateway config + Pending consent intact.
        apply_action(&mut s, KeyAction::ChatDetectDismiss);
        assert!(s.chat_detect_offer.is_none());
        assert_eq!(s.chat_llm.as_ref(), Some(&gw));
        assert_eq!(s.chat_consent, ChatConsent::Pending);
    }

    #[test]
    fn save_detect_offer_accepts_and_raises_persist_edge() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:13305/v1",
            "Llama-3.2-3B",
        )));
        apply_action(&mut s, KeyAction::ChatDetectSave);
        assert_eq!(s.chat_consent, ChatConsent::Accepted);
        assert!(s.chat_persist_dispatch, "save raises the persist edge");
        assert_eq!(
            s.chat_llm.as_ref().map(|c| c.base_url.as_str()),
            Some("http://localhost:13305/v1")
        );
        // No offer → save is a no-op (no edge).
        let mut s2 = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s2, KeyAction::ChatDetectSave);
        assert!(!s2.chat_persist_dispatch);
    }

    #[test]
    fn config_with_chat_sets_local_endpoint_and_clears_auth() {
        let mut cfg = rocm_dash_core::config::Config::default();
        cfg.tui.chat_auth_header = Some("Ocp-Apim-Subscription-Key".into());
        let next = config_with_chat(cfg, "http://localhost:8000/v1", "qwen");
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

    #[test]
    fn detect_key_available_on_gate_and_offer_keys_take_precedence() {
        // `d` triggers detect from the Unavailable empty-state.
        let unavail = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Unavailable,
            offer_pending: false,
        };
        assert_eq!(
            handle_key(
                press(KeyCode::Char('d')),
                ActiveTab::Chat,
                &Modal::None,
                unavail
            ),
            KeyAction::ChatDetect
        );
        // With an offer pending, y/n map to the offer (not consent).
        let offering = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Pending,
            offer_pending: true,
        };
        assert_eq!(
            handle_key(
                press(KeyCode::Char('y')),
                ActiveTab::Chat,
                &Modal::None,
                offering
            ),
            KeyAction::ChatDetectAccept
        );
        assert_eq!(
            handle_key(
                press(KeyCode::Char('n')),
                ActiveTab::Chat,
                &Modal::None,
                offering
            ),
            KeyAction::ChatDetectDismiss
        );
    }

    #[test]
    fn chat_input_actions_mutate_buffer() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Chat;
        s.chat_focused = true;
        apply_action(&mut s, KeyAction::ChatInput('h'));
        apply_action(&mut s, KeyAction::ChatInput('i'));
        assert_eq!(s.chat_input, "hi");
        apply_action(&mut s, KeyAction::ChatBackspace);
        assert_eq!(s.chat_input, "h");
        apply_action(&mut s, KeyAction::ChatBlur);
        assert!(!s.chat_focused);
        apply_action(&mut s, KeyAction::ChatFocus);
        assert!(s.chat_focused);
    }

    #[test]
    fn chat_submit_pushes_user_turn_and_raises_dispatch() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "what's GPU-2 doing?".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        // Only the user turn is pushed; the agent reply arrives async.
        assert_eq!(s.chat.len(), 1);
        assert_eq!(s.chat[0].role, ChatRole::User);
        assert_eq!(s.chat[0].content, "what's GPU-2 doing?");
        assert!(s.chat_input.is_empty());
        assert!(s.chat_sending, "submit marks the request in flight");
        assert!(s.chat_dispatch, "submit raises the spawn edge");
    }

    #[test]
    fn chat_submit_ignores_empty_input() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "   ".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert!(s.chat.is_empty());
        assert!(!s.chat_sending);
        assert!(!s.chat_dispatch);
    }

    #[test]
    fn chat_submit_ignored_while_request_in_flight() {
        // A second submit before the first reply lands must be a no-op — no
        // second user turn, no second spawn (prevents a racing double request).
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "first".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert!(s.chat_sending);
        assert_eq!(s.chat.len(), 1);
        s.chat_dispatch = false; // simulate event_loop consuming the edge
        s.chat_input = "second".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert_eq!(s.chat.len(), 1, "second submit ignored while in flight");
        assert!(!s.chat_dispatch, "no second dispatch edge raised");
        // After the reply clears the flag, submits work again.
        s.on_chat_reply("done".into());
        assert!(!s.chat_sending);
        s.chat_input = "third".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert!(s.chat_dispatch);
    }

    #[tokio::test]
    async fn chat_reply_path_appends_agent_turn_and_clears_sending() {
        // The wired ChatSubmit→reply path using the MockAgentClient (no LLM).
        let agent = crate::agent::MockAgentClient::new("GPU-2: 87% util, 71°C");
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "what's GPU-2 doing?".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert!(s.chat_sending);
        // Simulate event_loop: run the agent over the history, deliver the reply.
        let snapshot = s.state_snapshot();
        let reply = crate::agent::AgentClient::complete(&agent, &s.chat, snapshot)
            .await
            .expect("mock reply");
        s.on_chat_reply(reply);
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert_eq!(s.chat.last().unwrap().content, "GPU-2: 87% util, 71°C");
        assert!(!s.chat_sending);
    }

    #[test]
    fn chat_input_handles_unicode_and_long_text() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_focused = true;
        // Multi-byte / emoji chars push as single chars, no panic.
        for c in "héllo 🚀 café ∑".chars() {
            apply_action(&mut s, KeyAction::ChatInput(c));
        }
        assert_eq!(s.chat_input, "héllo 🚀 café ∑");
        // Backspace removes the trailing multi-byte char correctly.
        apply_action(&mut s, KeyAction::ChatBackspace);
        assert_eq!(s.chat_input, "héllo 🚀 café ");
        // Very long input is accepted.
        for _ in 0..5000 {
            apply_action(&mut s, KeyAction::ChatInput('x'));
        }
        assert!(s.chat_input.len() > 5000);
        // Submitting unicode pushes one user turn, no panic.
        s.chat_input = "什么是 GPU-2?".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert_eq!(s.chat[0].content, "什么是 GPU-2?");
    }

    #[test]
    fn chat_scroll_clamps_at_zero_and_pages() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s, KeyAction::ChatScroll(-100));
        assert_eq!(s.chat_scroll, 0, "scroll clamps at top");
        apply_action(&mut s, KeyAction::ChatScroll(7));
        assert_eq!(s.chat_scroll, 7);
        // PageUp/PageDown map to ChatScroll on the Chat tab when accepted.
        let accepted = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Accepted,
            offer_pending: false,
        };
        assert_eq!(
            handle_key(
                press(KeyCode::PageDown),
                ActiveTab::Chat,
                &Modal::None,
                accepted
            ),
            KeyAction::ChatScroll(CHAT_SCROLL_STEP)
        );
        assert_eq!(
            handle_key(
                press(KeyCode::PageUp),
                ActiveTab::Chat,
                &Modal::None,
                accepted
            ),
            KeyAction::ChatScroll(-CHAT_SCROLL_STEP)
        );
    }

    #[tokio::test]
    async fn chat_error_path_appends_error_turn_no_panic() {
        let agent = crate::agent::MockAgentClient::failing();
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "hi".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        let snapshot = s.state_snapshot();
        let err = crate::agent::AgentClient::complete(&agent, &s.chat, snapshot)
            .await
            .unwrap_err();
        s.on_chat_error(err.to_string());
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
        assert!(!s.chat_sending);
    }
}
