// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

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

// Submodules holding cohesive pieces of `AppState` + free fns split out of this
// file to keep the core reducer + event loop focused (a file→dir module move:
// `crate::app::*` paths are unchanged).
mod chat;
mod slash;
mod summary;

use chat::{build_chat_agent, detect_local_chat, detect_managed_chat, persist_chat_endpoint};
use summary::{parse_plan_result, summarize_json_value, summarize_slash_tool};

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
    /// Used by the local/OpenAI backends.
    pub chat_api_key: Option<String>,
    /// Anthropic API key, sourced by the bin (env-first then OS secure store —
    /// NEVER argv) and carried in-process via this seam. `None` when absent;
    /// the Anthropic backend then surfaces an actionable error on switch.
    pub anthropic_api_key: Option<String>,
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
    /// Bin-injected tool-executor seam; None for demo/replay/mock — dash behaves
    /// as today. Stored here (Phase 2 plumbing); Phase 3 will use it.
    pub tool_executor: Option<crate::tool_exec::SharedRocmToolExecutor>,
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
    // 5-tab IA. Home is the default; ROCm and Serving are the two domain tabs
    // (Actions list + inline Details); Observe folds the host/instance/bench
    // telemetry; Chat is the assistant. The former single Action tab is gone —
    // its guided verbs are split across ROCm + Serving.
    #[default]
    Home,
    Rocm,
    Serving,
    Observe,
    Chat,
}

impl ActiveTab {
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Home => Self::Rocm,
            Self::Rocm => Self::Serving,
            Self::Serving => Self::Observe,
            Self::Observe => Self::Chat,
            Self::Chat => Self::Home,
        }
    }
    #[must_use]
    pub const fn prev(self) -> Self {
        match self {
            Self::Home => Self::Chat,
            Self::Rocm => Self::Home,
            Self::Serving => Self::Rocm,
            Self::Observe => Self::Serving,
            Self::Chat => Self::Observe,
        }
    }
    pub const fn from_digit(d: char) -> Option<Self> {
        match d {
            '1' => Some(Self::Home),
            '2' => Some(Self::Rocm),
            '3' => Some(Self::Serving),
            '4' => Some(Self::Observe),
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

/// Which chat LLM backend is active. The dash can switch live via `/provider`
/// (Phase 8); every backend calls the SAME ROCm tools through the seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ChatProvider {
    /// The auto-detected local OpenAI-compatible endpoint (or the no-key ChatGPT
    /// OAuth default). This is the launch default and reuses the inline build.
    #[default]
    Local,
    /// OpenAI's hosted Chat Completions API (`OPENAI_API_KEY`).
    Openai,
    /// Anthropic's Claude API (`ANTHROPIC_API_KEY`).
    Anthropic,
}

impl ChatProvider {
    /// Parse the `/provider <name>` argument (case-insensitive). `None` for an
    /// unrecognized name so the handler can hint instead of switching silently.
    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "local" => Some(Self::Local),
            "openai" => Some(Self::Openai),
            "anthropic" => Some(Self::Anthropic),
            _ => None,
        }
    }

    /// The lowercase label used in turns and hints.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

/// Actionable empty-state shown when a chat is submitted with no agent built
/// (no detected endpoint and no provider key). Surfaced as an error turn — never
/// an error dump or a panic — and names the two concrete recovery actions.
pub(crate) const NO_CHAT_BACKEND_MSG: &str = "no chat backend is configured. Press d to detect a local engine, or use \
     /provider openai|anthropic with the matching API key set.";

/// Result of routing a chat-input line through the slash-command handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlashOutcome {
    /// The line was a slash command and was handled in-reducer (state mutated,
    /// or a slash-tool request raised). It must NOT be sent to the LLM.
    Handled,
    /// The line is not a slash command — fall through to normal agent dispatch.
    NotCommand,
}

/// A pending read-only slash command that needs the bin executor (no overlay).
/// `submit_chat` sets it; the event loop drains it once, off the async thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SlashToolRequest {
    /// Tool name to execute across the seam (e.g. `rocm_command`).
    pub name: String,
    /// JSON args for the tool (e.g. `{"args":["model"]}`).
    pub args: serde_json::Value,
    /// Human label for the chat turn header (e.g. `model`).
    pub label: String,
}

/// The structured next action from a natural-language plan (Phase 7).
///
/// Plain data mirrored from the bin's `freeform_plan_next_action_with_context`
/// so the reducer can decide whether to hand a complete mutating action to the
/// approval modal. A placeholder action (`has_placeholders`) stays plan-only.
/// `pub` (not `pub(crate)`) because it is a payload of the `pub` [`ClientMsg`]
/// enum (mirrors [`crate::tool_exec::ApprovalIntent`]); the reducer entrypoints
/// that consume it stay crate-private.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedAction {
    /// The rocm CLI argv to run (e.g. `["install","sdk","--prefix","/x"]`).
    pub args: Vec<String>,
    /// Whether the planned action mutates local ROCm state (needs approval).
    pub approval_required: bool,
    /// Whether any arg is still a `<placeholder>` (the plan is incomplete).
    pub has_placeholders: bool,
    /// Whether a planner provider produced this plan. Provider-assisted plans
    /// stay review-only (never auto-forwarded to execution), mirroring the
    /// bin's `validate_freeform_execution_action` guard.
    pub provider_assisted: bool,
}

/// A surfaced mutating-tool approval awaiting the operator's decision (Phase 4).
/// Reusable for any [`crate::tool_exec::ApprovalIntent`] (the same modal serves
/// later phases: update/uninstall, permissions, plan). The modal owns keyboard
/// focus while `Some`; on Approve the `(name, arguments)` are replayed through
/// `execute_approved`; on Deny/Cancel nothing runs.
#[derive(Debug, Clone)]
pub(crate) struct PendingApproval {
    pub req: crate::ui::approval::ApprovalRequest,
    pub choice: crate::ui::approval::ApprovalChoice,
    /// Tool name to re-execute on Approve (the validator already accepted it).
    pub name: String,
    /// JSON args for the approved re-execution.
    pub arguments: serde_json::Value,
}

/// Modal overlays. Only one is shown at a time, on top of the active tab body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Modal {
    #[default]
    None,
    Help,
    Detail,
    ThemePicker,
    /// btop-style Esc main menu (Options / Help / Quit).
    Menu,
    /// "Go to…" command palette (tab/destination switch).
    Palette,
    /// Tabbed Options panel (General / CPU / GPU / Engines).
    Options,
    /// Global 2-column keyboard reference (distinct from the contextual `?`).
    GlobalHelp,
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
    /// Cursor into the Esc main-menu rows (Options / Help / Quit).
    pub menu_sel: usize,
    /// Cursor into the command-palette destination rows.
    pub palette_sel: usize,
    /// Active tab index in the Options panel (General / CPU / GPU / Engines).
    pub options_tab: usize,
    /// Cursor into the Action tab's guided-verb list.
    pub action_sel: usize,
    /// Whether the Action tab's focus is on the verb list or the detail pane.
    /// `→`/Enter moves focus into the detail box; `←` returns to the list.
    pub action_focus: ActionFocus,
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
    /// Doctor overlay (Phase 3 Wave 2). `None` = closed.
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
    /// Bin-injected tool-executor seam. Set from `ResolvedArgs` in the event
    /// loop; `None` for demo/replay/mock and by default.
    pub tool_executor: Option<crate::tool_exec::SharedRocmToolExecutor>,
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
            action_sel: 0,
            action_focus: ActionFocus::List,
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
            model_recipes: Vec::new(),
            runtimes: Vec::new(),
            automations: Vec::new(),
            tool_executor: None,
            should_quit: false,
            slash_tool: None,
            plan_request: None,
            approval: None,
            active_provider: ChatProvider::default(),
            provider_switch: None,
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
        self.approval = None;
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
            choice: crate::ui::approval::ApprovalChoice::Approve,
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
            // P1: ROCm + Serving both share the ported Action verb list as a
            // one-phase placeholder; P2 splits them into per-tab modules.
            ActiveTab::Rocm | ActiveTab::Serving => crate::ui::tabs::action::VERB_COUNT,
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
            ActiveTab::Rocm | ActiveTab::Serving => self.action_sel,
            _ => 0,
        }
    }

    const fn set_selection(&mut self, tab: ActiveTab, idx: usize) {
        match tab {
            ActiveTab::Observe => self.instance_sel = idx,
            ActiveTab::Rocm | ActiveTab::Serving => self.action_sel = idx,
            _ => {}
        }
    }

    /// Advance `gpu_scroll` so the selected GPU stays within the visible window.
    /// Derives the visible row count from the last rendered body area; with no
    /// prior draw it is a no-op (the renderer self-corrects on the next frame).
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

    // Best-effort terminal restoration: never let teardown failures override the
    // session result. If the controlling terminal already went away (e.g. the
    // PTY closed on quit), these writes can fail with a broken pipe — that must
    // not turn a clean exit into a non-zero one.
    let _ = disable_raw_mode();
    let _ = execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    );
    let _ = terminal.show_cursor();
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
    // Tool-executor seam (Phase 2 plumbing), injected by the bin; None for
    // demo/replay/mock. Phase 3 will use it.
    state.tool_executor = args.tool_executor.clone();
    state.replay = replay_controller.map(ReplayState::new);

    // Resolve the chat backend. `--chat-mock` short-circuits detection with a
    // deterministic offline MockAgentClient (no live LLM, no network); otherwise
    // we auto-detect the endpoint (the std-TCP probe runs once on a blocking
    // thread before the first frame) and build the Rig backend.
    let mut agent: Option<std::sync::Arc<dyn crate::agent::AgentClient>> = if args.chat_mock {
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
        // An endpoint we launched ourselves (managed-services registry) takes
        // priority over the well-known default port — this is how a tool-launched
        // engine on a non-default port (e.g. vLLM on :11435) is found. It does
        // NOT override an explicitly configured `chat_url`/env URL, so config
        // precedence is preserved (we only consult the registry when neither is
        // set, i.e. where the well-known default would otherwise be probed).
        let managed = if args.chat_url.is_none() && args.chat_env_url.is_none() {
            detect_managed_chat(state.tool_executor.clone()).await
        } else {
            None
        };
        let probe_target = args
            .chat_url
            .clone()
            .or_else(|| args.chat_env_url.clone())
            .unwrap_or_else(|| crate::llm::DEFAULT_CHAT_BASE_URL.to_string());
        // A managed endpoint is already readiness-verified; otherwise TCP-probe.
        let probe_ok = if managed.is_some() {
            true
        } else {
            tokio::task::spawn_blocking(move || {
                crate::llm::probe_endpoint(&probe_target, crate::llm::PROBE_TIMEOUT)
            })
            .await
            .unwrap_or(false)
        };
        let llm = managed.or_else(|| {
            crate::llm::resolve_llm_config(
                args.chat_url.as_deref(),
                args.chat_model.as_deref(),
                None,
                None,
                args.chat_api_key.as_deref(),
                args.chat_env_url.as_deref(),
                args.chat_auth_header.as_deref(),
                probe_ok,
            )
        });
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
            crate::agent::ChatGptAgentClient::new(
                args.chat_model.clone(),
                move |url, code| {
                    let _ = oauth_tx.send(ClientMsg::ChatReply {
                        text: format!(
                            "To enable chat, sign in to ChatGPT: open {url} and enter the code {code}"
                        ),
                    });
                },
                state.tool_executor.clone(),
                Some(chat_tx.clone()),
            )
            .ok()
            .map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn crate::agent::AgentClient>)
        } else {
            // A build failure leaves `agent` None; a submit surfaces an error turn.
            match &state.chat_llm {
                Some(cfg) => crate::agent::RigAgentClient::new(
                    cfg.clone(),
                    state.tool_executor.clone(),
                    Some(chat_tx.clone()),
                )
                .ok()
                .map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn crate::agent::AgentClient>),
                None => None,
            }
        }
    };

    // Snapshot the auto-detected local backend so `/provider local` can restore
    // it after a switch to a remote provider. Without this, switching to OpenAI
    // and back to local would leave `agent` pointing at the OpenAI backend
    // (silent wrong-backend bug) — `build_chat_agent(Local)` returns None by
    // design (Local is the inline-built backend), so the caller must restore the
    // saved clone here. `Option<Arc<…>>` clone is a cheap Arc refcount bump.
    let local_agent = agent.clone();

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
                    Some(ClientMsg::SlashToolReply { text }) => state.on_slash_tool_reply(text),
                    Some(ClientMsg::ChatError { message }) => state.on_chat_error(message),
                    Some(ClientMsg::ChatDetectResult { offer }) => state.set_detect_result(offer),
                    // A mutating tool (or slash command) surfaced an approval —
                    // open the modal; nothing executes until the operator approves.
                    Some(ClientMsg::ChatApprovalRequired { intent }) => state.open_approval(intent),
                    // An approved action finished: append the result turn and
                    // fire exactly one automatic follow-up agent turn.
                    Some(ClientMsg::ChatApprovalResult { text }) => state.on_approval_result(text),
                    // A `/plan` plan completed: render the review and (for a
                    // complete mutating action) hand it to the approval modal.
                    Some(ClientMsg::PlanReady { text, action }) => {
                        state.on_plan_ready(text, action);
                    }
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
                    // The approval modal, when open, owns ALL keys with the
                    // highest priority (above every operational overlay and the
                    // general handler) so the operator's decision can't be
                    // pre-empted. On Approve: replay the approved action off the
                    // event loop (spawn_blocking) and post ChatApprovalResult.
                    // On Deny/Cancel: a declined turn, no execution.
                    Some(Ok(CtEvent::Key(k))) if state.approval.is_some() => {
                        use crate::ui::approval::ApprovalVerdict;
                        match state.on_approval_key(k.code) {
                            Some(ApprovalVerdict::Approve) => {
                                if let Some((name, args)) = state.take_approval() {
                                    match state.tool_executor.clone() {
                                        Some(executor) => {
                                            let reply_tx = chat_tx.clone();
                                            tokio::task::spawn_blocking(move || {
                                                let text = run_approved(&executor, &name, &args);
                                                let _ = reply_tx
                                                    .send(ClientMsg::ChatApprovalResult { text });
                                            });
                                        }
                                        None => state.on_approval_result(
                                            "ROCm tools unavailable in this mode".to_string(),
                                        ),
                                    }
                                }
                            }
                            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => {
                                state.on_approval_declined();
                            }
                            None => { /* cursor moved or key ignored — modal stays open */ }
                        }
                    }
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
                    // The doctor overlay, when open, owns all keys (read-only
                    // `rocm doctor` job through the job-bridge).
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
                    // A terminal event-source error means the controlling
                    // terminal went away (e.g. the PTY/stdin closed) — the
                    // session is over, so quit cleanly rather than propagating a
                    // fatal error. Propagating it made `rocm chat` exit non-zero
                    // when its terminal closed before the first key was read
                    // (e.g. the acceptance PTY smoke under the embedded-daemon
                    // start delay); the legacy blocking reader treated this as
                    // end-of-session too. Mirrors the `None => break` EOF arm.
                    Some(Err(e)) => {
                        tracing::debug!(error = %e, "terminal event stream ended; quitting");
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }
        }

        // A `/quit` (or `/exit`) slash command sets `should_quit` from inside
        // the reducer; honor it here (mirrors the `KeyAction::Quit` break).
        if state.should_quit {
            break;
        }

        // Drain a pending executor-backed read-only slash command (`/model`,
        // `/daemon`). Off-thread (spawn_blocking) so the seam's synchronous
        // execute() never blocks the async event loop; the concise summary
        // returns via ClientMsg::SlashToolReply — its own message variant, so
        // the slash-tool path never disturbs the agent's `chat_sending` flag.
        if let Some(req) = state.slash_tool.take() {
            match state.tool_executor.clone() {
                Some(executor) => {
                    let reply_tx = chat_tx.clone();
                    tokio::task::spawn_blocking(move || {
                        // One path for read-only AND mutating slash commands:
                        // `Result`/`Error` → a concise reply turn; an
                        // `ApprovalRequired` (mutating) → open the approval modal
                        // via ChatApprovalRequired (nothing executes yet).
                        let msg = match executor.execute(&req.name, &req.args) {
                            crate::tool_exec::RocmToolOutcome::ApprovalRequired(intent) => {
                                ClientMsg::ChatApprovalRequired { intent }
                            }
                            outcome => ClientMsg::SlashToolReply {
                                text: summarize_slash_tool(&req.label, &outcome),
                            },
                        };
                        let _ = reply_tx.send(msg);
                    });
                }
                None => {
                    state.on_slash_tool_reply("ROCm tools unavailable in this mode".to_string());
                }
            }
        }

        // Drain a pending `/plan` natural-language plan. Off-thread
        // (spawn_blocking) so the read-only `natural_language_plan` tool's
        // synchronous execute() never blocks the async loop. The rendered plan +
        // structured next action return via ClientMsg::PlanReady; the tool only
        // PLANS — no mutation happens here. A complete mutating action is handed
        // to the approval modal by `on_plan_ready`.
        if let Some(req) = state.plan_request.take() {
            match state.tool_executor.clone() {
                Some(executor) => {
                    let reply_tx = chat_tx.clone();
                    tokio::task::spawn_blocking(move || {
                        let args = serde_json::json!({ "request": req });
                        let msg = match executor.execute("natural_language_plan", &args) {
                            crate::tool_exec::RocmToolOutcome::Result(v) => {
                                match parse_plan_result(&v) {
                                    Some((text, action)) => ClientMsg::PlanReady { text, action },
                                    None => ClientMsg::SlashToolReply {
                                        text: "/plan: planner returned no usable plan".to_string(),
                                    },
                                }
                            }
                            crate::tool_exec::RocmToolOutcome::Error(e) => {
                                ClientMsg::SlashToolReply {
                                    text: format!("/plan failed: {e}"),
                                }
                            }
                            crate::tool_exec::RocmToolOutcome::ApprovalRequired(_) => {
                                ClientMsg::SlashToolReply {
                                    text:
                                        "/plan: planning is read-only and should not need approval"
                                            .to_string(),
                                }
                            }
                        };
                        let _ = reply_tx.send(msg);
                    });
                }
                None => {
                    state.on_slash_tool_reply("ROCm tools unavailable in this mode".to_string());
                }
            }
        }

        // Drain a `/provider` switch (Phase 8). Rebuild the live `agent` for the
        // newly-selected backend. `Local` reuses whatever the inline launch path
        // built (it owns the auto-detect probe). `Openai`/`Anthropic` are built
        // from `ResolvedArgs` keys (in-process seam, never argv). A build failure
        // (e.g. missing key) leaves `agent` unchanged and surfaces an actionable
        // error turn. Construction only — no network until the next submit.
        if let Some(ProviderSwitch { previous, target }) = state.provider_switch.take() {
            match target {
                ChatProvider::Local => {
                    // Restore the auto-detected local backend saved before the
                    // event loop. `build_chat_agent(Local)` returns None by
                    // design, so the restore must happen here — otherwise a prior
                    // `/provider openai` would leave requests routed to OpenAI.
                    agent = local_agent.clone();
                    state
                        .chat
                        .push(ChatTurn::agent("switched to local".to_string()));
                }
                ChatProvider::Openai | ChatProvider::Anthropic => {
                    if let Some(new_agent) =
                        build_chat_agent(target, args, state.tool_executor.clone(), chat_tx.clone())
                    {
                        agent = Some(new_agent);
                        state
                            .chat
                            .push(ChatTurn::agent(format!("switched to {}", target.label())));
                    } else {
                        // Revert the optimistic `active_provider` set by the slash
                        // handler back to the provider active BEFORE the switch
                        // attempt — not unconditionally Local — so the displayed
                        // provider stays honest (e.g. a failed openai→anthropic
                        // switch stays on openai). `agent` is never reassigned on a
                        // failed build, so it already matches `previous`; the two
                        // stay consistent (no stale-remote routing under a wrong
                        // label).
                        state.active_provider = previous;
                        let hint = if target == ChatProvider::Anthropic {
                            "anthropic requires ANTHROPIC_API_KEY in env or secure store"
                        } else {
                            "openai requires OPENAI_API_KEY in the environment"
                        };
                        state.chat.push(ChatTurn::error(format!(
                            "could not switch to {}: {hint}",
                            target.label()
                        )));
                    }
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
                None => state.on_chat_error(NO_CHAT_BACKEND_MSG.to_string()),
            }
        }

        // Run the local-engine probe + `/v1/models` query on the detect edge,
        // off the reducer. Raised once by `request_detect`; result returns via
        // `ClientMsg::ChatDetectResult`.
        if state.chat_detect_dispatch {
            state.chat_detect_dispatch = false;
            let reply_tx = chat_tx.clone();
            let executor = state.tool_executor.clone();
            tokio::spawn(async move {
                let offer = detect_local_chat(executor).await;
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

/// Run an approved mutating action across the seam and render a concise summary
/// (never a raw JSON dump). Sync + executor-generic so the approve path is
/// unit-testable without tokio; the event loop calls it inside spawn_blocking.
fn run_approved(
    executor: &crate::tool_exec::SharedRocmToolExecutor,
    name: &str,
    args: &serde_json::Value,
) -> String {
    use crate::tool_exec::RocmToolOutcome;
    match executor.execute_approved(name, args) {
        RocmToolOutcome::Result(v) => {
            let body = summarize_json_value(&v);
            if body.is_empty() {
                format!("Approved · {name}: done")
            } else {
                format!("Approved · {name}:\n{body}")
            }
        }
        RocmToolOutcome::Error(e) => format!("Approved · {name} failed: {e}"),
        // A mutating tool's approved replay should not re-request approval; if it
        // somehow does, surface it plainly rather than silently looping.
        RocmToolOutcome::ApprovalRequired(_) => {
            format!("Approved · {name}: unexpected second approval request (not run)")
        }
    }
}

/// Apply a `KeyAction` to mutable state. Returns `true` when the action
/// requests application exit (Quit).
/// Wrap a list cursor by `delta`, cycling within `0..len`. `len == 0` → 0.
const fn wrap_cursor(cur: usize, delta: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let n = len.cast_signed();
    (cur.cast_signed() + delta).rem_euclid(n) as usize
}

fn apply_action(state: &mut AppState, action: KeyAction) -> bool {
    match action {
        KeyAction::Quit => return true,
        KeyAction::SwitchTab(t) => {
            state.active_tab = t;
            state.modal = Modal::None;
            // A fresh tab always starts with focus on its list, never stranded
            // in the Action detail pane from a previous visit.
            state.action_focus = ActionFocus::List;
        }
        KeyAction::Move(d) => {
            if state.modal == Modal::ThemePicker {
                state.theme_picker_move(d);
            } else {
                // Changing the verb selection snaps focus back to the list so
                // the detail pane re-previews the newly selected operation.
                if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                    state.action_focus = ActionFocus::List;
                }
                state.move_selection(d);
            }
        }
        KeyAction::ActionFocusDetail => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                state.action_focus = ActionFocus::Detail;
            }
        }
        KeyAction::ActionFocusList => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                state.action_focus = ActionFocus::List;
            }
        }
        KeyAction::ActionActivate => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                match state.action_focus {
                    // From the list, Enter steps INTO the detail pane.
                    ActionFocus::List => state.action_focus = ActionFocus::Detail,
                    // From the detail pane, Enter opens the operation's manager.
                    ActionFocus::Detail => {
                        let verb = crate::ui::tabs::action::verb_action(state.action_sel);
                        return apply_action(state, verb);
                    }
                }
            }
        }
        KeyAction::ActionEscape => {
            // Esc backs out one level: detail → list, then list → main menu.
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving)
                && state.action_focus == ActionFocus::Detail
            {
                state.action_focus = ActionFocus::List;
            } else {
                return apply_action(state, KeyAction::OpenMenu);
            }
        }
        KeyAction::ActionSelect(i) => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                state.action_sel = i.min(crate::ui::tabs::action::VERB_COUNT.saturating_sub(1));
                state.action_focus = ActionFocus::List;
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
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                // Action rows open the matching manager via the existing seam;
                // there is no detail modal on the ROCm/Serving tabs.
                let verb = crate::ui::tabs::action::verb_action(state.action_sel);
                return apply_action(state, verb);
            }
            if state.selection_len() > 0 {
                state.modal = Modal::Detail;
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
        KeyAction::OpenMenu => {
            state.modal = Modal::Menu;
            state.menu_sel = 0;
        }
        KeyAction::OpenPalette => {
            state.modal = Modal::Palette;
            state.palette_sel = 0;
        }
        KeyAction::MenuMove(d) => match state.modal {
            Modal::Menu => {
                state.menu_sel = wrap_cursor(state.menu_sel, d, crate::ui::modal::MENU_ITEMS);
            }
            Modal::Palette => {
                state.palette_sel =
                    wrap_cursor(state.palette_sel, d, crate::ui::modal::PALETTE_DESTS.len());
            }
            _ => {}
        },
        KeyAction::OptionsTab(d) => {
            if state.modal == Modal::Options {
                state.options_tab =
                    wrap_cursor(state.options_tab, d, crate::ui::modal::OPTIONS_TABS.len());
            }
        }
        KeyAction::MenuActivate => match state.modal {
            Modal::Menu => match state.menu_sel {
                0 => {
                    state.modal = Modal::Options;
                    state.options_tab = 0;
                }
                1 => state.modal = Modal::GlobalHelp,
                _ => return true, // Quit
            },
            Modal::Palette => {
                if let Some((_, tab)) = crate::ui::modal::PALETTE_DESTS.get(state.palette_sel) {
                    state.active_tab = *tab;
                }
                state.modal = Modal::None;
            }
            _ => {}
        },
        // ponytail: P3 folds Bench into Observe; the per-tab Bench detail modal
        // (the only scrollable detail) is no longer reachable, so modal scroll
        // is a no-op until/unless a scrollable Observe detail is wired.
        KeyAction::ScrollModal(_) => {}
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
        // Footer legend: a click on a key chip acts exactly like the key press.
        if let Some(chip) = footer_chip_hit(&state.last_footer_chips, me.column, me.row) {
            return chip;
        }
        if state.modal == Modal::None
            && let Some(area) = state.last_body_area
        {
            // ponytail: Observe folds the instances table into a stacked region;
            // body-click hit-testing best-efforts the instances rows. Keyboard
            // selection is the primary path.
            let action = match state.active_tab {
                ActiveTab::Observe => ui::tabs::instances::hit_test(area, me.column, me.row, state),
                ActiveTab::Rocm | ActiveTab::Serving => {
                    ui::tabs::action::hit_test(area, me.column, me.row)
                }
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

/// Resolve a pointer `(col, row)` against the recorded footer-legend chips.
/// Returns the chip's action when the pointer lands inside a chip span.
fn footer_chip_hit(chips: &[FooterChip], col: u16, row: u16) -> Option<KeyAction> {
    chips
        .iter()
        .find(|c| row == c.y && col >= c.x0 && col < c.x1)
        .map(|c| c.action)
}

/// Where the Action tab's keyboard focus currently sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActionFocus {
    /// Browsing the verb list (left column).
    #[default]
    List,
    /// Inside the detail pane (right column), ready to start the operation.
    Detail,
}

/// A clickable footer-legend chip: an absolute screen span on the footer row
/// plus the action a left-click should dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FooterChip {
    pub x0: u16,
    /// End-exclusive.
    pub x1: u16,
    pub y: u16,
    pub action: KeyAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    Nothing,
    Quit,
    SwitchTab(ActiveTab),
    /// Action tab: move focus into the detail pane (`→`).
    ActionFocusDetail,
    /// Action tab: move focus back to the verb list (`←`).
    ActionFocusList,
    /// Action tab: activate the current focus — from the list, focus the detail
    /// pane; from the detail pane, open the operation's manager.
    ActionActivate,
    /// Action tab: Esc — step out of the detail pane back to the list, or, when
    /// already on the list, fall through to the main menu.
    ActionEscape,
    /// Action tab: select the verb at this index and park focus on the list
    /// (from a mouse click on a verb row).
    ActionSelect(usize),
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
    /// Open the btop-style Esc main menu (P4).
    OpenMenu,
    /// Open the "Go to…" command palette (P4).
    OpenPalette,
    /// Move the cursor within the active overlay (Menu / Palette) by N rows.
    MenuMove(isize),
    /// Cycle the Options panel's tab by N (left/right).
    OptionsTab(isize),
    /// Activate the highlighted row in the active overlay (Menu / Palette).
    MenuActivate,
    /// Open the services-manager overlay (Phase 3 Wave 1).
    OpenServices,
    /// Open the serve-wizard overlay (Phase 3 Wave 1).
    OpenServeWizard,
    /// Open the engine-manager overlay (Phase 3 Wave 1).
    OpenEngineManager,
    /// Open the doctor overlay (Phase 3 Wave 2).
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
    // Global help overlay (opened from the Esc menu): close-only.
    if *modal == Modal::GlobalHelp {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('?') => KeyAction::CloseModal,
            _ => KeyAction::Nothing,
        };
    }
    // Esc main menu: ↑↓ cycle Options/Help/Quit, Enter activates, Esc closes.
    if *modal == Modal::Menu {
        return match k.code {
            KeyCode::Esc => KeyAction::CloseModal,
            KeyCode::Char('j') | KeyCode::Down => KeyAction::MenuMove(1),
            KeyCode::Char('k') | KeyCode::Up => KeyAction::MenuMove(-1),
            KeyCode::Enter => KeyAction::MenuActivate,
            _ => KeyAction::Nothing,
        };
    }
    // Command palette: ↑↓ choose destination, Enter goes, Esc closes.
    if *modal == Modal::Palette {
        return match k.code {
            KeyCode::Esc => KeyAction::CloseModal,
            KeyCode::Char('j') | KeyCode::Down => KeyAction::MenuMove(1),
            KeyCode::Char('k') | KeyCode::Up => KeyAction::MenuMove(-1),
            KeyCode::Enter => KeyAction::MenuActivate,
            _ => KeyAction::Nothing,
        };
    }
    // Options panel: ←→ switch settings tab, Esc closes.
    if *modal == Modal::Options {
        return match k.code {
            KeyCode::Esc => KeyAction::CloseModal,
            KeyCode::Char('h') | KeyCode::Left | KeyCode::BackTab => KeyAction::OptionsTab(-1),
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Tab => KeyAction::OptionsTab(1),
            _ => KeyAction::Nothing,
        };
    }
    match k.code {
        KeyCode::Char('q') => KeyAction::Quit,
        // Esc opens the main menu when idle, except on Chat (where Esc keeps its
        // existing chat meaning) — managers/approval are routed upstream.
        // On ROCm/Serving, Esc first steps out of the detail pane (resolved
        // against focus in `apply_action`); elsewhere it opens the main menu.
        KeyCode::Esc if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::ActionEscape
        }
        KeyCode::Esc if current != ActiveTab::Chat => KeyAction::OpenMenu,
        KeyCode::Esc => KeyAction::Nothing,
        KeyCode::Char(':') => KeyAction::OpenPalette,
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
        // The guided-action letter hotkeys live on Observe (the telemetry
        // surface) and Action (the verb list). Both route through the existing
        // execution seam — no second approval path.
        // Services manager: open where servers live.
        KeyCode::Char('s') if current == ActiveTab::Observe => KeyAction::OpenServices,
        // Serve wizard: launch a model.
        KeyCode::Char('w')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenServeWizard
        }
        // Engine manager: use/install/reinstall serving engines.
        KeyCode::Char('e')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenEngineManager
        }
        // Doctor: read-only environment check.
        KeyCode::Char('d')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenExamine
        }
        // Update: check/preview/apply ROCm package updates.
        KeyCode::Char('u')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenUpdate
        }
        // Install: ROCm SDK (TheRock) install / dry-run.
        KeyCode::Char('i')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenInstall
        }
        // Logs: browse recent ROCm CLI logs.
        KeyCode::Char('l')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenLogs
        }
        // Runtimes: list/activate/adopt/import ROCm runtimes.
        KeyCode::Char('r')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenRuntimes
        }
        // Onboarding: first-run setup wizard (install / adopt).
        KeyCode::Char('n')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenOnboarding
        }
        // Automations: list/enable/disable background checks.
        KeyCode::Char('a')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenAutomations
        }
        // Command runner: run any ROCm CLI subcommand (gated).
        KeyCode::Char('c')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenCommand
        }
        // Config & providers.
        KeyCode::Char('p')
            if matches!(
                current,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ) =>
        {
            KeyAction::OpenConfig
        }
        // ROCm/Serving tabs: arrow keys drive the focus-into-detail interaction;
        // Enter is focus-aware (list → focus detail, detail → open the manager).
        KeyCode::Right if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::ActionFocusDetail
        }
        KeyCode::Left if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::ActionFocusList
        }
        KeyCode::Enter if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::ActionActivate
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
        || (*modal == Modal::None
            && matches!(
                tab,
                ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
            ))
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
        assert_eq!(hk(KeyCode::Char('q'), ActiveTab::Home), KeyAction::Quit);
        // P4: Esc opens the main menu (it never quits); Chat keeps its own Esc.
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Observe), KeyAction::OpenMenu);
    }

    #[test]
    fn tab_cycles_forward_and_wraps() {
        // 5-tab IA: Home → ROCm → Serving → Observe → Chat → Home.
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Rocm)
        );
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Serving),
            KeyAction::SwitchTab(ActiveTab::Observe)
        );
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Observe),
            KeyAction::SwitchTab(ActiveTab::Chat)
        );
        // Chat wraps back to Home.
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Chat),
            KeyAction::SwitchTab(ActiveTab::Home)
        );
    }

    #[test]
    fn action_tab_arrows_and_enter_drive_focus() {
        // → steps into the detail pane, ← steps back, Enter is focus-aware.
        assert_eq!(
            hk(KeyCode::Right, ActiveTab::Rocm),
            KeyAction::ActionFocusDetail
        );
        assert_eq!(
            hk(KeyCode::Left, ActiveTab::Rocm),
            KeyAction::ActionFocusList
        );
        assert_eq!(
            hk(KeyCode::Enter, ActiveTab::Rocm),
            KeyAction::ActionActivate
        );
        // Arrows are inert on other tabs (no focus model there).
        assert_eq!(hk(KeyCode::Right, ActiveTab::Observe), KeyAction::Nothing);
        // Enter elsewhere keeps its detail-modal meaning.
        assert_eq!(
            hk(KeyCode::Enter, ActiveTab::Observe),
            KeyAction::OpenDetail
        );
    }

    #[test]
    fn action_activate_is_two_step_list_then_open() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.action_sel = 0; // Serve a model → OpenServeWizard
        assert_eq!(s.action_focus, ActionFocus::List);
        // First activate steps into the detail pane; no overlay yet.
        apply_action(&mut s, KeyAction::ActionActivate);
        assert_eq!(s.action_focus, ActionFocus::Detail);
        assert!(s.serve_wizard.is_none(), "must not open before stepping in");
        // Second activate opens the operation's manager.
        apply_action(&mut s, KeyAction::ActionActivate);
        assert!(
            s.serve_wizard.is_some(),
            "detail-focus Enter opens the manager"
        );
    }

    #[test]
    fn action_focus_resets_on_move_and_tab_switch() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.action_focus = ActionFocus::Detail;
        apply_action(&mut s, KeyAction::Move(1));
        assert_eq!(s.action_focus, ActionFocus::List, "Move snaps back to list");
        s.action_focus = ActionFocus::Detail;
        apply_action(&mut s, KeyAction::SwitchTab(ActiveTab::Home));
        assert_eq!(s.action_focus, ActionFocus::List, "tab switch resets focus");
    }

    #[test]
    fn action_esc_backs_out_of_detail_then_opens_menu() {
        // Esc on Action is intercepted (not the global OpenMenu) so it can back
        // out of the detail pane first.
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Rocm), KeyAction::ActionEscape);
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.action_focus = ActionFocus::Detail;
        apply_action(&mut s, KeyAction::ActionEscape);
        assert_eq!(s.action_focus, ActionFocus::List, "first Esc → list");
        assert_eq!(s.modal, Modal::None, "first Esc does not open the menu");
        apply_action(&mut s, KeyAction::ActionEscape);
        assert_eq!(s.modal, Modal::Menu, "second Esc opens the menu");
    }

    #[test]
    fn action_select_sets_verb_and_parks_on_list() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.action_focus = ActionFocus::Detail;
        apply_action(&mut s, KeyAction::ActionSelect(2));
        assert_eq!(s.action_sel, 2);
        assert_eq!(s.action_focus, ActionFocus::List);
        // Out-of-range clamps rather than panicking.
        apply_action(&mut s, KeyAction::ActionSelect(999));
        assert!(s.action_sel < crate::ui::tabs::action::VERB_COUNT);
    }

    #[test]
    fn footer_chip_hit_maps_click_to_action() {
        let chips = vec![
            FooterChip {
                x0: 0,
                x1: 5,
                y: 49,
                action: KeyAction::Quit,
            },
            FooterChip {
                x0: 6,
                x1: 9,
                y: 49,
                action: KeyAction::ToggleHelp,
            },
        ];
        // Inside the first chip.
        assert_eq!(footer_chip_hit(&chips, 2, 49), Some(KeyAction::Quit));
        // End-exclusive: column 5 is past the first chip, before the second.
        assert_eq!(footer_chip_hit(&chips, 5, 49), None);
        assert_eq!(footer_chip_hit(&chips, 7, 49), Some(KeyAction::ToggleHelp));
        // Wrong row never matches.
        assert_eq!(footer_chip_hit(&chips, 2, 48), None);
    }

    #[test]
    fn back_tab_and_shift_tab_both_cycle_backward() {
        // prev(Observe) = Serving in the 5-tab IA.
        assert_eq!(
            hk(KeyCode::BackTab, ActiveTab::Observe),
            KeyAction::SwitchTab(ActiveTab::Serving)
        );
        // Home's previous tab is Chat (the last tab).
        assert_eq!(
            hk(KeyCode::BackTab, ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Chat)
        );
        let shift_tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT);
        assert_eq!(
            handle_key(
                shift_tab,
                ActiveTab::Rocm,
                &Modal::None,
                ChatKeyCtx::default()
            ),
            KeyAction::SwitchTab(ActiveTab::Home)
        );
    }

    #[test]
    fn number_keys_jump_to_tab() {
        // 5-tab: '1'→Home, '2'→ROCm, '3'→Serving, '4'→Observe, '5'→Chat.
        assert_eq!(
            hk(KeyCode::Char('1'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Home)
        );
        assert_eq!(
            hk(KeyCode::Char('2'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Rocm)
        );
        assert_eq!(
            hk(KeyCode::Char('3'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Serving)
        );
        assert_eq!(
            hk(KeyCode::Char('4'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Observe)
        );
        // `5` reaches the Chat tab (digit guard widened to '1'..='5').
        assert_eq!(
            hk(KeyCode::Char('5'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Chat)
        );
        assert_eq!(hk(KeyCode::Char('6'), ActiveTab::Home), KeyAction::Nothing);
    }

    #[test]
    fn from_digit_maps_five_to_chat() {
        // 5-tab digit map; Chat is now '5', '6'/'0' are out of range.
        assert_eq!(ActiveTab::from_digit('1'), Some(ActiveTab::Home));
        assert_eq!(ActiveTab::from_digit('5'), Some(ActiveTab::Chat));
        assert_eq!(ActiveTab::from_digit('6'), None);
        assert_eq!(ActiveTab::from_digit('0'), None);
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
                ActiveTab::Home,
                &Modal::None,
                ChatKeyCtx::default()
            ),
            KeyAction::Nothing
        );
    }

    #[test]
    fn jk_arrows_and_g_drive_selection() {
        assert_eq!(
            hk(KeyCode::Char('j'), ActiveTab::Observe),
            KeyAction::Move(1)
        );
        assert_eq!(
            hk(KeyCode::Char('k'), ActiveTab::Observe),
            KeyAction::Move(-1)
        );
        assert_eq!(hk(KeyCode::Down, ActiveTab::Rocm), KeyAction::Move(1));
        assert_eq!(hk(KeyCode::Up, ActiveTab::Rocm), KeyAction::Move(-1));
        assert_eq!(
            hk(KeyCode::Char('g'), ActiveTab::Observe),
            KeyAction::SelectFirst
        );
        assert_eq!(
            hk(KeyCode::Char('G'), ActiveTab::Observe),
            KeyAction::SelectLast
        );
        assert_eq!(
            hk(KeyCode::Enter, ActiveTab::Observe),
            KeyAction::OpenDetail
        );
    }

    #[test]
    fn operational_open_keys_are_tab_scoped() {
        // `s` opens services only on Observe; Nothing elsewhere.
        assert_eq!(
            hk(KeyCode::Char('s'), ActiveTab::Observe),
            KeyAction::OpenServices
        );
        assert_eq!(hk(KeyCode::Char('s'), ActiveTab::Home), KeyAction::Nothing);
        // `w` / `e` open from Observe + Action; Nothing on other tabs.
        assert_eq!(
            hk(KeyCode::Char('w'), ActiveTab::Observe),
            KeyAction::OpenServeWizard
        );
        assert_eq!(
            hk(KeyCode::Char('e'), ActiveTab::Rocm),
            KeyAction::OpenEngineManager
        );
        assert_eq!(hk(KeyCode::Char('w'), ActiveTab::Home), KeyAction::Nothing);
        // Doctor / update open from Observe + Action.
        assert_eq!(
            hk(KeyCode::Char('d'), ActiveTab::Observe),
            KeyAction::OpenExamine
        );
        assert_eq!(
            hk(KeyCode::Char('u'), ActiveTab::Rocm),
            KeyAction::OpenUpdate
        );
        // Install / logs open from Observe + Action.
        assert_eq!(
            hk(KeyCode::Char('i'), ActiveTab::Observe),
            KeyAction::OpenInstall
        );
        assert_eq!(hk(KeyCode::Char('l'), ActiveTab::Rocm), KeyAction::OpenLogs);
        // On the Chat tab none of these open an overlay (the operational keys
        // are guarded to Observe/Action). `i` is the one that means
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
    fn esc_opens_menu_when_idle_but_not_on_chat() {
        // Idle (non-Chat) tabs: Esc opens the btop main menu.
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Home), KeyAction::OpenMenu);
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Observe), KeyAction::OpenMenu);
        // Chat keeps its existing Esc meaning (no menu).
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Chat), KeyAction::Nothing);
        // While an overlay modal owns the screen, Esc closes it (not OpenMenu).
        assert_eq!(
            handle_key(
                press(KeyCode::Esc),
                ActiveTab::Home,
                &Modal::Menu,
                ChatKeyCtx::default()
            ),
            KeyAction::CloseModal
        );
        assert_eq!(
            handle_key(
                press(KeyCode::Esc),
                ActiveTab::Home,
                &Modal::Options,
                ChatKeyCtx::default()
            ),
            KeyAction::CloseModal
        );
    }

    #[test]
    fn colon_opens_command_palette() {
        assert_eq!(
            hk(KeyCode::Char(':'), ActiveTab::Home),
            KeyAction::OpenPalette
        );
    }

    #[test]
    fn menu_navigation_and_activation() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s, KeyAction::OpenMenu);
        assert_eq!(s.modal, Modal::Menu);
        // ↓ from Options(0) → Help(1); activate opens the global help.
        apply_action(&mut s, KeyAction::MenuMove(1));
        assert_eq!(s.menu_sel, 1);
        apply_action(&mut s, KeyAction::MenuActivate);
        assert_eq!(s.modal, Modal::GlobalHelp);
        // Menu → Options activation opens the Options panel.
        apply_action(&mut s, KeyAction::OpenMenu);
        apply_action(&mut s, KeyAction::MenuActivate); // sel 0 = Options
        assert_eq!(s.modal, Modal::Options);
        // Options tab cycles and wraps.
        apply_action(&mut s, KeyAction::OptionsTab(-1));
        assert_eq!(s.options_tab, crate::ui::modal::OPTIONS_TABS.len() - 1);
    }

    #[test]
    fn palette_activation_switches_tab() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s, KeyAction::OpenPalette);
        apply_action(&mut s, KeyAction::MenuMove(3)); // Home→ROCm→Serving→Observe
        apply_action(&mut s, KeyAction::MenuActivate);
        assert_eq!(s.active_tab, ActiveTab::Observe);
        assert_eq!(s.modal, Modal::None);
    }

    #[test]
    fn question_mark_toggles_help() {
        assert_eq!(
            hk(KeyCode::Char('?'), ActiveTab::Home),
            KeyAction::ToggleHelp
        );
    }

    #[test]
    fn t_opens_theme_picker() {
        assert_eq!(
            hk(KeyCode::Char('t'), ActiveTab::Home),
            KeyAction::OpenThemePicker
        );
    }

    #[test]
    fn theme_picker_absorbs_navigation_keys() {
        let with_picker = |c| {
            handle_key(
                press(c),
                ActiveTab::Home,
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
        // 5-tab layout: Home 0..10, ROCm 11..21, Serving 22..35, Observe 36..49,
        // Chat 50..60.
        let bar = Rect::new(0, 0, 80, 1);
        assert_eq!(tab_bar_hit(bar, 5, 0), Some(ActiveTab::Home));
        assert_eq!(tab_bar_hit(bar, 15, 0), Some(ActiveTab::Rocm));
        assert_eq!(tab_bar_hit(bar, 28, 0), Some(ActiveTab::Serving));
        assert_eq!(tab_bar_hit(bar, 42, 0), Some(ActiveTab::Observe));
        assert_eq!(tab_bar_hit(bar, 55, 0), Some(ActiveTab::Chat));
        // Separator gap between Home (ends 10 excl.) and ROCm (starts 11).
        assert_eq!(tab_bar_hit(bar, 10, 0), None);
        // Wrong row.
        assert_eq!(tab_bar_hit(bar, 5, 2), None);
    }

    #[test]
    fn tab_bar_hit_skips_chips_that_overflow_a_narrow_bar() {
        // Bar can only fit the first two chips (ROCm ends at 21).
        let bar = Rect::new(0, 0, 25, 1);
        assert_eq!(tab_bar_hit(bar, 5, 0), Some(ActiveTab::Home));
        assert_eq!(tab_bar_hit(bar, 15, 0), Some(ActiveTab::Rocm));
        // Serving chip would be at 22..35 — overflows the 25-wide bar → None.
        assert_eq!(tab_bar_hit(bar, 28, 0), None);
    }

    #[test]
    fn tab_bar_hit_honors_x_offset() {
        // Bar offset 10 columns to the right: Home chip now spans 10..19.
        let bar = Rect::new(10, 0, 80, 1);
        assert_eq!(tab_bar_hit(bar, 15, 0), Some(ActiveTab::Home));
        // Absolute x=5 is left of the offset bar.
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
            handle_mouse(scroll_down, &Modal::None, ActiveTab::Home),
            KeyAction::Nothing
        );
        // No modal, Observe → Move (drives the instances selection)
        assert_eq!(
            handle_mouse(scroll_down, &Modal::None, ActiveTab::Observe),
            KeyAction::Move(3)
        );
        // Detail modal → ScrollModal
        assert_eq!(
            handle_mouse(scroll_down, &Modal::Detail, ActiveTab::Observe),
            KeyAction::ScrollModal(3)
        );
        // ThemePicker → Move (drives picker cursor)
        assert_eq!(
            handle_mouse(scroll_down, &Modal::ThemePicker, ActiveTab::Home),
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
                ActiveTab::Observe,
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
                ActiveTab::Observe,
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
            hk(KeyCode::Char('['), ActiveTab::Home),
            KeyAction::ReplayJump(-10)
        );
        assert_eq!(
            hk(KeyCode::Char(']'), ActiveTab::Home),
            KeyAction::ReplayJump(10)
        );
        assert_eq!(
            hk(KeyCode::Char('{'), ActiveTab::Home),
            KeyAction::ReplayJump(-60)
        );
        assert_eq!(
            hk(KeyCode::Char('}'), ActiveTab::Home),
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
            KeyAction::SwitchTab(ActiveTab::Home)
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
            KeyAction::SwitchTab(ActiveTab::Rocm)
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

    // --- Slash-command dispatch (Phase 3 nav/session + read-only) ---

    fn st() -> AppState {
        AppState::new("t".into(), "default-dark".into())
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
    fn slash_gpu_switches_to_hardware() {
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
            chat_url: None,
            chat_model: None,
            chat_auth_header: None,
            chat_env_url: None,
            chat_api_key: None,
            anthropic_api_key: key.map(str::to_string),
            chat_auto_consent: false,
            chat_mock: false,
            model_recipes: Vec::new(),
            runtimes: Vec::new(),
            automations: Vec::new(),
            tool_executor: None,
        }
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
        // The Local arm's restore line (mirrors app.rs): the factory cannot help.
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
        // Enter on the default (Approve) choice yields an Approve verdict.
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
        // Tab toggles the cursor to Deny without producing a verdict.
        assert_eq!(s.on_approval_key(crossterm::event::KeyCode::Tab), None);
        assert_eq!(
            s.approval.as_ref().unwrap().choice,
            crate::ui::approval::ApprovalChoice::Deny
        );
        // Enter now confirms Deny.
        assert_eq!(
            s.on_approval_key(crossterm::event::KeyCode::Enter),
            Some(crate::ui::approval::ApprovalVerdict::Deny)
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
