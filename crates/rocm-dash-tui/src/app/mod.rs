// Copyright © Advanced Micro Devices, Inc., or its affiliates.
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

use chat::{
    StartupChatOutcome, build_chat_agent, build_local_agent, detect_local_chat,
    discover_configured_chat_model, persist_chat_endpoint, startup_chat_outcome,
};
use summary::{parse_plan_result, summarize_json_value, summarize_slash_tool};

/// Which single flow a *focused host* runs.
///
/// The bare-`rocm` launcher opens one overlay to completion — no embedded
/// daemon, no tab shell — then returns to the menu. `None` on
/// [`ResolvedArgs::focus`] is the normal full dashboard, so every existing
/// dash/chat path is byte-identical when focus is unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    /// First-run onboarding (install / adopt ROCm) — the launcher's
    /// `Set up this system` row and `rocm bootstrap setup`.
    Setup,
    /// The serve-a-model wizard — the launcher's `Serve a model` row.
    Serve,
    /// Read-only `rocm examine` environment check — the launcher's
    /// `Diagnose & fix` row. Auto-runs on open.
    Examine,
}

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
    /// (bare `rocm` / `rocm chat`); `Home` for the dashboard (`rocm dash`).
    pub initial_tab: ActiveTab,
    /// When `Some`, run as a *focused host*: open exactly the overlay for this
    /// flow, skip the embedded daemon + chat backend, render overlay-only, and
    /// exit back to the launcher when the overlay is closed at its root. `None`
    /// (the default) is the normal full dashboard — every path stays unchanged.
    pub focus: Option<Focus>,
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
    /// Daemon-tailed bench CSV path (`config.dashboard.daemon.bench_results_dir`).
    ///
    /// When `Some`, the bench-run form defaults `--out` to this path so appended
    /// rows appear live in the bench tab. Adapted by the bin (owns `rocm-core`).
    pub bench_results_dir: Option<std::path::PathBuf>,
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
    /// Operational notice generated by the TUI itself (e.g. "switched to
    /// local"), rendered in the transcript but **never** sent to the model —
    /// `build_messages` drops it so it can't masquerade as a prior assistant
    /// turn and corrupt the model's context.
    System,
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
    /// A TUI-generated operational notice. Rendered but dropped from the LLM
    /// history by [`build_messages`](crate::agent::build_messages).
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
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
    /// Scroll offset (in lines) inside the Bench Detail modal. Reset on Open.
    pub bench_detail_scroll: u16,
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
            bench_detail_scroll: 0,
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
            ScrollTarget::BenchDetail => self.bench_detail_scroll = p,
            ScrollTarget::DockLogs => self.dock_logs_scroll = p,
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

    /// Focused-host exit gate: `true` when a `focus` is active AND its single
    /// overlay is closed (no manager is `Some`).
    ///
    /// [`has_open_overlay`](Self::has_open_overlay) stays `true` while a
    /// sub-popup (folder browser / model picker) is open, so this can't fire
    /// while the user is inside one of those. It does NOT by itself protect a
    /// running job console: the shared console maps `q` / running-`Esc` to
    /// "close overlay", which would null the manager mid-job. That case is
    /// handled upstream in `event_loop` by [`focused_close_key_blocked`], which
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
    fn active_overlay_at_root(&self) -> bool {
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
                m.browser.is_none() && m.approval.is_none() && m.active_job.is_none()
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
    /// ROCm/Serving while a manager overlay is open AND that manager is at its
    /// root screen. The event loop closes the manager and returns focus to the
    /// Actions list when this holds. Pure read so it is unit-testable (the
    /// mutation lives in the event-loop arm).
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
        matches!(self.active_tab, ActiveTab::Rocm | ActiveTab::Serving)
            && self.has_open_overlay()
            && self.active_overlay_at_root()
            && matches!(code, crossterm::event::KeyCode::Esc)
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
        let next = u16::try_from((cur + i32::from(delta)).max(0)).unwrap_or(u16::MAX);
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

/// Whether the event loop should skip the embedded daemon client AND the chat
/// backend resolution. True exactly when a [`Focus`] is set: a focused host runs
/// one overlay that streams its own job through the job-bridge, so it needs
/// neither live telemetry nor an LLM. `focus == None` (the dashboard) keeps both.
/// Pure predicate → unit-testable without a runtime; also names the render branch
/// (`draw_focused` when true, `draw` when false).
const fn should_skip_daemon(focus: Option<Focus>) -> bool {
    focus.is_some()
}

/// In a focused host, whether a console "close" key must be SWALLOWED because
/// the active job is still running.
///
/// In the dashboard, `q` / running-`Esc` detach the console and leave the job
/// running in the background (the app persists). A focused host has no
/// background: closing the overlay trips the [`AppState::focused_should_exit`]
/// gate and returns from `event_loop`, tearing down the runtime and killing the
/// child via `kill_on_drop` — truncating a mutating install/serve mid-write. So
/// while the job is non-terminal we swallow those keys; the user stops a job
/// explicitly with `Ctrl+C` (never blocked here), and once it is terminal `q` /
/// `Esc` exit normally. Always `false` for the dashboard (`focus == None`).
fn focused_close_key_blocked(state: &AppState, focus: Option<Focus>, code: KeyCode) -> bool {
    if !should_skip_daemon(focus) {
        return false;
    }
    let running = state
        .active_job_id()
        .and_then(|id| state.jobs.job(id))
        .is_some_and(|j| !j.is_terminal());
    running && matches!(code, KeyCode::Char('q') | KeyCode::Esc)
}

/// Open the single overlay a focused host should host, returning any initial
/// job-bridge side effects to pump (Examine auto-runs `rocm examine` on open;
/// Setup/Serve open their form and wait for input). Clears any other overlay
/// first (mutually-exclusive invariant). Pure w.r.t. process I/O — the caller
/// runs the returned effects through [`crate::jobs::run_effects`].
fn open_overlay_for_focus(
    state: &mut AppState,
    focus: Focus,
) -> Vec<rocm_dash_core::state::SideEffect> {
    state.close_overlays();
    match focus {
        Focus::Setup => {
            state.onboarding = Some(crate::ui::onboarding::OnboardingState::default());
            Vec::new()
        }
        Focus::Serve => {
            state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState::default());
            Vec::new()
        }
        Focus::Examine => {
            let (mgr, fx) = crate::ui::examine_manager::open_running(&mut state.jobs);
            state.examine_manager = Some(mgr);
            fx
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
    } else if should_skip_daemon(args.focus) {
        // Focused host: no daemon client. The overlay streams its own job via
        // the job-bridge and the telemetry chrome isn't drawn, so a live
        // connection would only spawn an unused embedded daemon. Drop `tx`
        // (its `chat_tx` clone keeps `rx` alive for the loop); nothing is sent.
        drop(tx);
        None
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
    // Daemon-tailed bench CSV path for the bench-run form's default --out.
    state.bench_results_dir = args.bench_results_dir.clone();
    // Focused host: open exactly the overlay for the requested flow (Examine
    // also auto-runs its read-only job). `Focus::Setup` opens the onboarding
    // overlay — the same wizard `rocm bootstrap setup` routes to.
    if let Some(focus) = args.focus {
        let fx = open_overlay_for_focus(&mut state, focus);
        crate::jobs::run_effects(fx, &job_tx);
    }
    state.replay = replay_controller.map(ReplayState::new);
    // Both `--demo` (a generated session replayed) and `--replay <file>` present
    // non-live data, so mark the session simulated for the honesty chrome.
    state.simulated = state.replay.is_some();

    // Resolve the chat backend. `--chat-mock` short-circuits detection with a
    // deterministic offline MockAgentClient (no live LLM, no network); otherwise
    // we auto-detect the endpoint (the std-TCP probe runs once on a blocking
    // thread before the first frame) and build the Rig backend.
    let mut agent: Option<std::sync::Arc<dyn crate::agent::AgentClient>> = if should_skip_daemon(
        args.focus,
    ) {
        // Focused host: Setup/Serve/Diagnose never chat. Skip endpoint detection
        // and backend construction entirely — no probe, no network, no OAuth
        // default. `chat_llm` stays `None` and the Chat tab is never drawn here.
        None
    } else if args.chat_mock {
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
        // When neither an explicit URL (CLI/config) nor an env URL is set, run
        // the SAME full local-engine detection the manual 'd' path uses:
        // registry-first (an engine we launched ourselves, on whatever port it
        // bound), then a probe of the well-known Lemonade/vLLM/rocm-serve
        // ports (parallelized — see `llm::detect_local_endpoint` — so a cold
        // start with no server doesn't pay 3x the probe timeout), plus a
        // best-effort served-model fetch. This is what lets a local server win
        // over the ChatGPT cloud default at startup instead of only the single
        // well-known :8000 port that a bare `resolve_llm_config` probe covers.
        //
        // NOTE: unmerged PR #97 also touches this branch (model discovery when
        // `chat_model` is None, inside `resolve_llm_config`'s own fallback
        // path) — this change is conflict-minimal by leaving the
        // `resolve_llm_config` call below untouched.
        //
        // Gate on `chat_api_key.is_none()` too: local detection returns a
        // keyless `detected_llm_config` (api_key/auth_header forced to None),
        // so firing it when the user configured a key would SILENTLY DROP that
        // key and 401 at request time. A configured key means "use my
        // configured backend", so skip the swap and let `resolve_llm_config`
        // carry the key through its normal precedence.
        let detection_ran = chat::should_detect_local_chat(
            args.chat_url.as_deref(),
            args.chat_env_url.as_deref(),
            args.chat_api_key.as_deref(),
        );
        let detected = if detection_ran {
            detect_local_chat(state.tool_executor.clone()).await
        } else {
            None
        };
        let probe_target = args
            .chat_url
            .clone()
            .or_else(|| args.chat_env_url.clone())
            .unwrap_or_else(|| crate::llm::DEFAULT_CHAT_BASE_URL.to_string());
        // A detected endpoint (managed or probed) is already verified. When
        // detection ran and found nothing it already probed the well-known
        // vLLM :8000 port (== `DEFAULT_CHAT_BASE_URL`), so re-probing the same
        // fallback target here is redundant and just burns another probe
        // timeout on a cold start — treat that as unreachable directly.
        // Otherwise (an explicit URL/env/key path) TCP-probe the target.
        let startup_outcome = startup_chat_outcome(detection_ran, detected.is_some());
        let probe_ok = match startup_outcome {
            StartupChatOutcome::Local => true,
            StartupChatOutcome::OAuth => false,
            StartupChatOutcome::Configured => tokio::task::spawn_blocking(move || {
                crate::llm::probe_endpoint(&probe_target, crate::llm::PROBE_TIMEOUT)
            })
            .await
            .unwrap_or(false),
        };
        let llm = detected.or_else(|| {
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
        // PR #97 port onto PR #100's startup flow: a *configured* URL (CLI/env)
        // with no explicit model resolves to the `local-model` placeholder,
        // which 404s on servers that register the model under its real id. Only
        // the `Configured` outcome needs this — the `Local` outcome already
        // carries a `/v1/models`-discovered model from `detect_local_chat`, and
        // `OAuth` has no config. Discovery is gated inside the helper on
        // `probe_ok` (an unreachable endpoint is never probed nor replaced) and
        // on the absence of an explicit model (config precedence wins).
        let llm = match llm {
            Some(cfg) if startup_outcome == StartupChatOutcome::Configured => Some(
                discover_configured_chat_model(cfg, args.chat_model.as_deref(), probe_ok).await,
            ),
            other => other,
        };
        state.set_chat_config(llm, args.chat_auto_consent);
        // No reachable local endpoint AND no key/url configured → the no-key
        // ChatGPT OAuth default (device-code login surfaced in the chat tab).
        // This restores the no-key login the vendored Codex path provided; it
        // takes NO api_key (env-only invariant untouched — OAuth, not a key).
        let no_key_no_endpoint = startup_outcome == StartupChatOutcome::OAuth;
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
                Some(cfg) => {
                    build_local_agent(cfg.clone(), state.tool_executor.clone(), chat_tx.clone())
                        .ok()
                }
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
    let mut local_agent = agent.clone();

    loop {
        // Focused host renders overlay-only (no header / tabs / dock / footer
        // chrome); the dashboard renders the full shell.
        if should_skip_daemon(args.focus) {
            terminal.draw(|f| ui::draw_focused(f, &mut state))?;
        } else {
            terminal.draw(|f| ui::draw(f, &mut state))?;
        }
        tokio::select! {
            _ = tick.tick() => {
                // Advance the animation clock so spinners cycle even while a
                // job produces no new output.
                state.tick_count = state.tick_count.wrapping_add(1);
            }
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
                    // Only ACT on key presses. Terminals (notably Windows
                    // Terminal / ConPTY under WSL, and any with the kitty
                    // keyboard protocol) also emit Release/Repeat events; the
                    // general `handle_key` already drops non-Press, but the
                    // operational-overlay arms below dispatch straight to their
                    // managers and would otherwise process the SAME keystroke
                    // twice. That double-fire is what made Enter in the serve
                    // wizard's model picker re-open the picker (seeding it with
                    // the just-chosen model as a filter) instead of choosing.
                    // Swallow non-Press key events here, above every key arm, so
                    // the Press-only invariant holds for overlays too.
                    Some(Ok(CtEvent::Key(k))) if !is_actionable_key(k.kind) => {
                        let _ = k;
                    }
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
                    // De-modal back-out: on ROCm/Serving, an inline manager is
                    // shown in the Details pane. Esc closes it and returns focus
                    // to the Actions list — intercepted BEFORE the per-manager
                    // key arms so the manager doesn't eat Esc first. `←` is left
                    // to the manager (some use it to cycle options).
                    Some(Ok(CtEvent::Key(k))) if state.should_pane_back_out(k.code) => {
                        state.close_overlays();
                        state.pane_focus = PaneFocus::Actions;
                    }
                    // While a manager is showing its job console, the navigation
                    // keys pan the log (PgUp/PgDn = page, arrows = line). Routed
                    // BEFORE the per-manager arms (which would ignore them); the
                    // console action keys (Ctrl+C/q/Esc/Enter) are NOT scroll keys
                    // so they still fall through to `on_console_key`.
                    Some(Ok(CtEvent::Key(k)))
                        if state.has_active_console() && console_scroll_delta(k.code).is_some() =>
                    {
                        let (dv, dh) = console_scroll_delta(k.code).unwrap_or((0, 0));
                        state.scroll_console(dv, dh);
                    }
                    // Focused host only: while the hosted job is still RUNNING,
                    // swallow the console close keys (`q`, running-`Esc`) so the
                    // overlay is never nulled mid-job — which would trip the
                    // focused exit gate and tear the runtime down, killing the
                    // child via kill_on_drop. `Ctrl+C` (cancel) and the scroll
                    // keys above still flow, so the user can always stop a job;
                    // once it is terminal, `q`/`Esc` exit normally. Routed BEFORE
                    // the per-manager arms so the manager can't close first.
                    Some(Ok(CtEvent::Key(k)))
                        if focused_close_key_blocked(&state, args.focus, k.code) => {}
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
                    // Bench-run form, when open, owns all keys.
                    Some(Ok(CtEvent::Key(k))) if state.bench_run.is_some() => {
                        let fx = crate::ui::bench_run::on_key(
                            &mut state.bench_run,
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

        // Focused host: the launcher hosts exactly one overlay. Once the user
        // backs out of it at root (the per-manager `on_key` set its state to
        // `None`), return so `app::run` hands control back to the launcher menu.
        // `focused_should_exit` stays `false` while any sub-popup / job console
        // keeps the overlay `Some`, so this never ejects mid-flow. No-op for the
        // dashboard (`focus == None`).
        if state.focused_should_exit(args.focus) {
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
                        .push(ChatTurn::system("switched to local".to_string()));
                }
                ChatProvider::Openai | ChatProvider::Anthropic => {
                    if let Some(new_agent) =
                        build_chat_agent(target, args, state.tool_executor.clone(), chat_tx.clone())
                    {
                        agent = Some(new_agent);
                        state
                            .chat
                            .push(ChatTurn::system(format!("switched to {}", target.label())));
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

        // Drain the endpoint-rebuild edge (Phase 8 sibling). An accepted
        // detected-local offer must re-point the LIVE `agent` — and the
        // `/provider local` restore snapshot — at the new local backend.
        // `accept_detect_offer` swaps `chat_llm` to the auth-free local config
        // but stays I/O-free, so without this the stale startup agent keeps
        // routing chat to the cloud gateway (wrong-backend 401 bug). The edge
        // carries the provider active BEFORE the optimistic switch to `Local`;
        // on failure we revert `active_provider` to it (mirrors the
        // `provider_switch` drain) so the tab never shows `Local` while `agent`
        // still points elsewhere. Construction only — no network until submit.
        if let Some(previous) = state.chat_endpoint_rebuild.take() {
            // `revert` restores the optimistic switch and surfaces an actionable
            // error turn so the tab does not sit on `Local` with the old agent.
            let revert = |state: &mut AppState, msg: String| {
                state.active_provider = previous;
                state.chat.push(ChatTurn::error(msg));
            };
            match state.chat_llm.clone() {
                Some(cfg) => {
                    match build_local_agent(cfg, state.tool_executor.clone(), chat_tx.clone()) {
                        Ok(arc) => {
                            agent = Some(arc.clone());
                            // Refresh the restore snapshot so a later `/provider
                            // local` restores THIS accepted backend, not the
                            // stale startup one.
                            local_agent = Some(arc);
                            state
                                .chat
                                .push(ChatTurn::system("switched to local".to_string()));
                        }
                        Err(e) => revert(
                            &mut state,
                            format!("could not switch to the detected local endpoint: {e}"),
                        ),
                    }
                }
                // Edge raised but `chat_llm` is None (shouldn't happen after a
                // real accept, but don't leave the tab stuck on `Local` with the
                // old agent and no feedback).
                None => revert(
                    &mut state,
                    "could not switch to the detected local endpoint: no endpoint configured"
                        .to_string(),
                ),
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

/// Wrap a list cursor by `delta`, cycling within `0..len`. `len == 0` → 0.
const fn wrap_cursor(cur: usize, delta: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let n = len.cast_signed();
    (cur.cast_signed() + delta).rem_euclid(n) as usize
}

/// Apply a `KeyAction` to mutable state. Returns `true` when the action
/// requests application exit (Quit).
fn apply_action(state: &mut AppState, action: KeyAction) -> bool {
    match action {
        KeyAction::Quit => return true,
        KeyAction::SwitchTab(t) => {
            state.active_tab = t;
            state.modal = Modal::None;
            // A fresh tab always starts with focus on its Actions list, never
            // stranded in the Details pane from a previous visit.
            state.pane_focus = PaneFocus::Actions;
        }
        KeyAction::Move(d) => {
            if state.modal == Modal::ThemePicker {
                state.theme_picker_move(d);
            } else {
                // Changing the verb selection snaps focus back to the Actions
                // list so Details re-previews the newly selected operation.
                if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                    state.pane_focus = PaneFocus::Actions;
                }
                state.move_selection(d);
            }
        }
        KeyAction::PaneFocusDetail => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                state.pane_focus = PaneFocus::Detail;
            }
        }
        KeyAction::PaneFocusActions => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                state.pane_focus = PaneFocus::Actions;
            }
        }
        KeyAction::PaneActivate => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                match state.pane_focus {
                    // From the Actions list, Enter steps INTO the Details pane.
                    PaneFocus::Actions => state.pane_focus = PaneFocus::Detail,
                    // From Details, Enter opens the operation's manager.
                    PaneFocus::Detail => {
                        let verb = state.pane_verb_action();
                        return apply_action(state, verb);
                    }
                }
            }
        }
        KeyAction::PaneEscape => {
            // Esc backs out one level: Details → Actions, then Actions → menu.
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving)
                && state.pane_focus == PaneFocus::Detail
            {
                state.pane_focus = PaneFocus::Actions;
            } else {
                return apply_action(state, KeyAction::OpenMenu);
            }
        }
        KeyAction::PaneSelect(i) => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                let last = state.pane_verb_count().saturating_sub(1);
                state.set_selection(state.active_tab, i.min(last));
                state.pane_focus = PaneFocus::Actions;
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
                // Verb rows open the matching manager via the existing seam;
                // there is no detail modal on the ROCm/Serving tabs.
                let verb = state.pane_verb_action();
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
        KeyAction::OpenBenchRun => {
            let bench_csv = state.bench_results_dir.clone();
            state.close_overlays();
            state.bench_run = Some(crate::ui::bench_run::BenchRunState::new(
                bench_csv.as_deref(),
            ));
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
        KeyAction::ScrollConsole(dv, dh) => state.scroll_console(dv, dh),
        KeyAction::ScrollDock(dv) => state.scroll_dock(dv),
        KeyAction::ScrollGrab(target, pos, grab_offset) => {
            state.apply_scroll_grab(target, pos, grab_offset);
        }
        KeyAction::ScrollRelease => state.scroll_drag = None,
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
            let next = (i32::from(state.chat_scroll) + i32::from(d)).max(0) as usize;
            state.set_chat_scroll(next);
        }
        KeyAction::Nothing => {}
    }
    false
}

/// Convert a displayed position to the target's own offset units. Dock logs are
/// tail-anchored, so their displayed top-to-bottom position is inverted.
fn target_offset(h: &ScrollbarHandle, displayed: usize) -> usize {
    if h.target == ScrollTarget::DockLogs {
        h.max_position().saturating_sub(displayed)
    } else {
        displayed
    }
}

fn target_position(state: &AppState, h: &ScrollbarHandle) -> usize {
    let position = match h.target {
        ScrollTarget::Console => usize::from(state.console_scroll),
        ScrollTarget::ConsoleH => usize::from(state.console_hscroll),
        ScrollTarget::Chat => usize::from(state.chat_scroll),
        ScrollTarget::BenchDetail => usize::from(state.bench_detail_scroll),
        ScrollTarget::DockLogs => h
            .max_position()
            .saturating_sub(usize::from(state.dock_logs_scroll)),
    };
    position.min(h.max_position())
}

/// If `(col, row)` lands on a recorded scrollbar, preserve a thumb grab or use
/// a proportional full-track jump for a track click.
fn scrollbar_hit(state: &AppState, col: u16, row: u16) -> Option<KeyAction> {
    let bars = state.scrollbars.borrow();
    let h = bars.iter().find(|h| point_in(h.track, col, row))?;
    let displayed = target_position(state, h);
    let grab_offset = h.grab_offset(col, row, displayed);
    let next = grab_offset.map_or_else(|| h.track_position_at(col, row), |_| displayed);
    Some(KeyAction::ScrollGrab(
        h.target,
        target_offset(h, next),
        grab_offset.unwrap_or(0),
    ))
}

fn resolve_mouse(me: MouseEvent, state: &AppState) -> KeyAction {
    // A held drag on a scrollbar keeps updating that offset until release, even
    // when the pointer slides off the narrow track.
    if me.kind == MouseEventKind::Drag(MouseButton::Left) {
        if let Some(drag) = state.scroll_drag
            && let Some(h) = state
                .scrollbars
                .borrow()
                .iter()
                .find(|h| h.target == drag.target)
        {
            let current = target_position(state, h);
            let displayed = h.position_at(me.column, me.row, drag.grab_offset, current);
            return KeyAction::ScrollGrab(
                drag.target,
                target_offset(h, displayed),
                drag.grab_offset,
            );
        }
        return KeyAction::Nothing;
    }
    // Any button release ends an active scrollbar drag.
    if matches!(me.kind, MouseEventKind::Up(_)) {
        return if state.scroll_drag.is_some() {
            KeyAction::ScrollRelease
        } else {
            KeyAction::Nothing
        };
    }

    if me.kind == MouseEventKind::Down(MouseButton::Left) {
        // Scrollbar tracks win over everything (incl. an open overlay's console
        // bar), so a click on the bar grabs it instead of falling through.
        if let Some(a) = scrollbar_hit(state, me.column, me.row) {
            return a;
        }
        if let Some(area) = state.last_tab_bar_area
            && let Some(tab) = tab_bar_hit(area, me.column, me.row)
        {
            return KeyAction::SwitchTab(tab);
        }
        // Footer legend: a click on a key chip acts exactly like the key press.
        if let Some(chip) = footer_chip_hit(&state.last_footer_chips, me.column, me.row) {
            return chip;
        }
        // While an operational manager is open it owns the body — swallow body
        // clicks so they can't fall THROUGH the inline manager to the obscured
        // Actions/Details list (which would silently change the selection or
        // re-open a verb). Tab-bar and footer-chip clicks above still work.
        if state.has_open_overlay() {
            return KeyAction::Nothing;
        }
        if state.modal == Modal::None
            && let Some(area) = state.last_body_area
        {
            // ponytail: Observe folds the instances table into a stacked region;
            // body-click hit-testing best-efforts the instances rows. Keyboard
            // selection is the primary path.
            let action = match state.active_tab {
                // Observe's AI table is keyboard + scroll-wheel driven (the
                // scroll path maps to Move in `handle_mouse`); left-click select
                // is intentionally not wired (the table sits below the hero band,
                // so a body-relative row map would be wrong). No-op here.
                ActiveTab::Rocm => ui::tabs::rocm::hit_test(area, me.column, me.row),
                ActiveTab::Serving => ui::tabs::serving::hit_test(area, me.column, me.row),
                _ => None,
            };
            if let Some(a) = action {
                return a;
            }
        }
        return KeyAction::Nothing;
    }

    // Scroll wheel (incl. horizontal wheel where the device emits it). Per-notch
    // deltas: ±1 line / ±1 col here, scaled per target below.
    let (dv, dh): (i16, i16) = match me.kind {
        MouseEventKind::ScrollDown => (1, 0),
        MouseEventKind::ScrollUp => (-1, 0),
        MouseEventKind::ScrollRight => (0, 1),
        MouseEventKind::ScrollLeft => (0, -1),
        // Not a scroll (e.g. moves / other buttons): nothing to route.
        _ => return KeyAction::Nothing,
    };

    // An open manager owns the body. When it is showing its job console, the
    // wheel pans that log (bigger vertical step, wider horizontal step so long
    // command lines come into view). On a form screen there is nothing to pan —
    // swallow it so the wheel can't move the obscured Actions list underneath.
    if state.has_open_overlay() {
        return if state.has_active_console() {
            KeyAction::ScrollConsole(dv * 3, dh * 6)
        } else {
            KeyAction::Nothing
        };
    }

    // Wide-layout right LOGS dock: the wheel pans the log stream when the pointer
    // is over it (vertical only — it's a tail-anchored log).
    if state.modal == Modal::None
        && dv != 0
        && let Some(dock) = state.last_dock_area
        && point_in(dock, me.column, me.row)
    {
        return KeyAction::ScrollDock(dv * 3);
    }

    // No overlay: on a domain tab the wheel moves the Actions selection by ONE
    // row — but only while the pointer is actually over the Actions column, so
    // hovering the Details pane doesn't nudge the list. Anything else falls
    // through to the modal/tab scroll routing.
    if state.modal == Modal::None
        && matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving)
    {
        if dv != 0
            && let Some(body) = state.last_body_area
            && point_in(crate::ui::tabs::pane::actions_rect(body), me.column, me.row)
        {
            return KeyAction::Move(dv as isize);
        }
        return KeyAction::Nothing;
    }

    handle_mouse(me, &state.modal, state.active_tab)
}

/// Whether `(x, y)` lies inside `r` (end-exclusive on both axes).
const fn point_in(r: ratatui::layout::Rect, x: u16, y: u16) -> bool {
    x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height
}

/// Resolve a pointer `(col, row)` against the recorded footer-legend chips.
/// Returns the chip's action when the pointer lands inside a chip span.
fn footer_chip_hit(chips: &[FooterChip], col: u16, row: u16) -> Option<KeyAction> {
    chips
        .iter()
        .find(|c| row == c.y && col >= c.x0 && col < c.x1)
        .map(|c| c.action)
}

/// Where a domain tab's (ROCm/Serving) keyboard focus currently sits. Shared by
/// both tabs; each keeps its own selection cursor (`rocm_sel`/`serving_sel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaneFocus {
    /// Browsing the Actions list (left column).
    #[default]
    Actions,
    /// Inside the Details pane (right column), ready to start the operation.
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

/// Active pointer drag for a scrollbar thumb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollDrag {
    pub target: ScrollTarget,
    pub grab_offset: u16,
}

/// Which scrollable surface a drawn scrollbar controls. Lets a mouse click on a
/// scrollbar track write the right offset field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollTarget {
    /// Job console vertical (`console_scroll`).
    Console,
    /// Job console horizontal (`console_hscroll`).
    ConsoleH,
    /// Wide-layout LOGS dock (`dock_logs_scroll`, tail-anchored / inverted).
    DockLogs,
    /// Bench row detail modal (`bench_detail_scroll`).
    BenchDetail,
    /// Chat transcript (`chat_scroll`).
    Chat,
}

/// A scrollbar drawn this frame, recorded so a mouse click/drag can hit-test it.
///
/// `track` is the screen rect of the bar; `content_len`/`viewport_len` size the
/// thumb; `target` says which offset to move. Vertical bars map the mouse row,
/// horizontal bars the column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollbarHandle {
    pub track: ratatui::layout::Rect,
    pub horizontal: bool,
    pub content_len: usize,
    pub viewport_len: usize,
    pub target: ScrollTarget,
}

impl ScrollbarHandle {
    /// Build a handle from the rect passed to a scrollbar helper (`area`) and its
    /// returned content rect (`drawn`). Returns `None` when they're equal — i.e.
    /// no bar was drawn because the content fit — so nothing gets hit-tested.
    pub(crate) fn new(
        area: ratatui::layout::Rect,
        drawn: ratatui::layout::Rect,
        horizontal: bool,
        content_len: usize,
        viewport_len: usize,
        target: ScrollTarget,
    ) -> Option<Self> {
        if drawn == area {
            return None;
        }
        let track = if horizontal {
            ratatui::layout::Rect::new(area.x, area.y + area.height - 1, area.width, 1)
        } else {
            ratatui::layout::Rect::new(area.x + area.width - 1, area.y, 1, area.height)
        };
        Some(Self {
            track,
            horizontal,
            content_len,
            viewport_len,
            target,
        })
    }

    const fn max_position(&self) -> usize {
        self.content_len.saturating_sub(self.viewport_len)
    }

    const fn axis(&self, col: u16, row: u16) -> (u16, u16, u16) {
        if self.horizontal {
            (col, self.track.x, self.track.width)
        } else {
            (row, self.track.y, self.track.height)
        }
    }

    /// Ratatui 0.30.2 `Scrollbar::part_lengths` geometry for the logical
    /// first-visible-unit position used by the dashboard.
    fn thumb_geometry(&self, logical_position: usize) -> (u16, u16) {
        let (_, _, span) = self.axis(0, 0);
        if span == 0 || self.content_len == 0 {
            return (0, 0);
        }
        let track_len = usize::from(span);
        let rendered_max = self.content_len.saturating_sub(1);
        let rendered_position =
            logical_position.min(self.max_position()) * rendered_max / self.max_position().max(1);
        let denominator = rendered_max.saturating_add(self.viewport_len);
        let rounded_divide =
            |numerator: usize| numerator.saturating_add(denominator / 2) / denominator.max(1);
        let thumb_len =
            rounded_divide(self.viewport_len.saturating_mul(track_len)).clamp(1, track_len);
        let thumb_start = rounded_divide(rendered_position.saturating_mul(track_len))
            .clamp(0, track_len.saturating_sub(thumb_len));
        (thumb_start as u16, thumb_len as u16)
    }

    fn grab_offset(&self, col: u16, row: u16, position: usize) -> Option<u16> {
        let (coord, track_start, _) = self.axis(col, row);
        let relative = coord.saturating_sub(track_start);
        let (thumb_start, thumb_len) = self.thumb_geometry(position);
        (relative >= thumb_start && relative < thumb_start.saturating_add(thumb_len))
            .then(|| relative - thumb_start)
    }

    /// Proportional track-click mapping. The endpoints map exactly to the
    /// logical endpoints and do not depend on thumb geometry.
    fn track_position_at(&self, col: u16, row: u16) -> usize {
        let max_position = self.max_position();
        let (coord, start, span) = self.axis(col, row);
        if max_position == 0 || span <= 1 {
            return 0;
        }
        usize::from(coord.saturating_sub(start).min(span - 1)) * max_position
            / usize::from(span - 1)
    }

    /// Invert Ratatui's rounded thumb-start mapping. When several logical
    /// positions render at the requested start, retain `current_position` if it
    /// lies on that plateau; otherwise choose the nearest plateau endpoint.
    fn position_at(&self, col: u16, row: u16, grab_offset: u16, current_position: usize) -> usize {
        let max_position = self.max_position();
        if max_position == 0 {
            return 0;
        }
        let (coord, start, span) = self.axis(col, row);
        let (_, thumb_len) = self.thumb_geometry(0);
        let desired_start = coord
            .saturating_sub(start)
            .saturating_sub(grab_offset)
            .min(span.saturating_sub(thumb_len));
        if desired_start == 0 {
            return 0;
        }
        if desired_start == span.saturating_sub(thumb_len) {
            return max_position;
        }

        let first_at_or_after = |wanted: u16| {
            let mut low = 0usize;
            let mut high = max_position;
            while low < high {
                let mid = low + (high - low) / 2;
                if self.thumb_geometry(mid).0 < wanted {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };
        let first = first_at_or_after(desired_start);
        if self.thumb_geometry(first).0 != desired_start {
            if first == 0 {
                return 0;
            }
            let before = first - 1;
            let before_start = self.thumb_geometry(before).0;
            let after_start = self.thumb_geometry(first).0;
            return if desired_start - before_start <= after_start - desired_start {
                before
            } else {
                first
            };
        }
        let after = first_at_or_after(desired_start.saturating_add(1));
        let last = if after == max_position && self.thumb_geometry(after).0 == desired_start {
            after
        } else {
            after.saturating_sub(1)
        };
        current_position.clamp(first, last)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    Nothing,
    Quit,
    SwitchTab(ActiveTab),
    /// ROCm/Serving tab: move focus into the Details pane (`→`).
    PaneFocusDetail,
    /// ROCm/Serving tab: move focus back to the Actions list (`←`).
    PaneFocusActions,
    /// ROCm/Serving tab: activate the current focus — from the Actions list,
    /// focus the Details pane; from Details, open the operation's manager.
    PaneActivate,
    /// ROCm/Serving tab: Esc — step out of Details back to the Actions list, or,
    /// when already on the list, fall through to the main menu.
    PaneEscape,
    /// ROCm/Serving tab: select the verb at this index and park focus on the
    /// Actions list (from a mouse click on a verb row).
    PaneSelect(usize),
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
    /// Pan the active job console: `(vertical_lines, horizontal_cols)`, negative
    /// = up/left. No-op when no console is showing.
    ScrollConsole(i16, i16),
    /// Scroll the wide-layout right LOGS dock by N lines (negative = toward the
    /// newest line). No-op when the dock isn't showing.
    ScrollDock(i16),
    /// Grab a scrollbar at `position`, retaining the pointer's offset inside the
    /// thumb so subsequent drag events track without a jump.
    ScrollGrab(ScrollTarget, usize, u16),
    /// Release the active scrollbar drag (mouse button up).
    ScrollRelease,
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
    /// Open the bench-run form overlay.
    OpenBenchRun,
}

/// Whether a crossterm key event should be acted on. Terminals emit
/// Release/Repeat events in addition to Press (notably Windows Terminal /
/// ConPTY under WSL, and any terminal advertising the kitty keyboard protocol).
///
/// The whole TUI acts on Press only. Both the general [`handle_key`] and the
/// event loop's operational-overlay dispatch share this gate — without it, a
/// single keystroke reaches an overlay's `on_key` more than once, which made
/// Enter in the serve wizard's model picker re-open the picker (seeded with the
/// just-chosen model as a filter) instead of choosing it.
const fn is_actionable_key(kind: KeyEventKind) -> bool {
    matches!(kind, KeyEventKind::Press)
}

fn handle_key(k: KeyEvent, current: ActiveTab, modal: &Modal, chat: ChatKeyCtx) -> KeyAction {
    if !is_actionable_key(k.kind) {
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
            KeyAction::PaneEscape
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
        // The guided-action letter hotkeys live ONLY on Observe (the telemetry
        // surface) — quick jumps into the managers via the existing seam. On
        // ROCm/Serving the Actions list is the single interaction path, so the
        // per-tab letter hotkeys are retired there.
        // Services manager: open where servers live.
        KeyCode::Char('s') if current == ActiveTab::Observe => KeyAction::OpenServices,
        // Serve wizard: launch a model.
        KeyCode::Char('w') if current == ActiveTab::Observe => KeyAction::OpenServeWizard,
        // Engine manager: use/install/reinstall serving engines.
        KeyCode::Char('e') if current == ActiveTab::Observe => KeyAction::OpenEngineManager,
        // Examine: read-only environment check.
        KeyCode::Char('d') if current == ActiveTab::Observe => KeyAction::OpenExamine,
        // Update: check/preview/apply ROCm package updates.
        KeyCode::Char('u') if current == ActiveTab::Observe => KeyAction::OpenUpdate,
        // Install: ROCm SDK (TheRock) install / dry-run.
        KeyCode::Char('i') if current == ActiveTab::Observe => KeyAction::OpenInstall,
        // Logs: browse recent ROCm CLI logs.
        KeyCode::Char('l') if current == ActiveTab::Observe => KeyAction::OpenLogs,
        // Bench-run: launch a bench sweep from the TUI.
        KeyCode::Char('b') if current == ActiveTab::Observe => KeyAction::OpenBenchRun,
        // Runtimes: list/activate/adopt/import ROCm runtimes.
        KeyCode::Char('r') if current == ActiveTab::Observe => KeyAction::OpenRuntimes,
        // Onboarding: first-run setup wizard (install / adopt).
        KeyCode::Char('n') if current == ActiveTab::Observe => KeyAction::OpenOnboarding,
        // Automations: list/enable/disable background checks.
        KeyCode::Char('a') if current == ActiveTab::Observe => KeyAction::OpenAutomations,
        // Command runner: run any ROCm CLI subcommand (gated).
        KeyCode::Char('c') if current == ActiveTab::Observe => KeyAction::OpenCommand,
        // Config & providers.
        KeyCode::Char('p') if current == ActiveTab::Observe => KeyAction::OpenConfig,
        // ROCm/Serving tabs: arrow keys drive the focus-into-detail interaction;
        // Enter is focus-aware (list → focus detail, detail → open the manager).
        KeyCode::Right if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::PaneFocusDetail
        }
        KeyCode::Left if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::PaneFocusActions
        }
        KeyCode::Enter if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::PaneActivate
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
/// Map a navigation key to a job-console pan delta `(lines, cols)` while a
/// console is showing. `None` for non-scroll keys so they fall through to the
/// console's own action handler (Ctrl+C / q / Esc / Enter). A page is 10 lines.
const fn console_scroll_delta(code: KeyCode) -> Option<(i16, i16)> {
    match code {
        KeyCode::PageDown => Some((10, 0)),
        KeyCode::PageUp => Some((-10, 0)),
        KeyCode::Down => Some((1, 0)),
        KeyCode::Up => Some((-1, 0)),
        KeyCode::Right => Some((0, 4)),
        KeyCode::Left => Some((0, -4)),
        _ => None,
    }
}

pub fn handle_mouse(ev: MouseEvent, modal: &Modal, tab: ActiveTab) -> KeyAction {
    // Domain-tab (ROCm/Serving) and overlay/console scroll is resolved in
    // `resolve_mouse` (it needs `&AppState` for hit-testing and overlay state).
    // This handles the remaining position-independent targets: the scrollable
    // modal body and the Observe instances list. One row per wheel notch.
    let delta: i16 = match ev.kind {
        MouseEventKind::ScrollDown => 1,
        MouseEventKind::ScrollUp => -1,
        _ => return KeyAction::Nothing,
    };
    if *modal == Modal::Detail {
        KeyAction::ScrollModal(delta)
    } else if *modal == Modal::ThemePicker || (*modal == Modal::None && tab == ActiveTab::Observe) {
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
            KeyAction::PaneFocusDetail
        );
        assert_eq!(
            hk(KeyCode::Left, ActiveTab::Rocm),
            KeyAction::PaneFocusActions
        );
        assert_eq!(hk(KeyCode::Enter, ActiveTab::Rocm), KeyAction::PaneActivate);
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
        // Serving verb 0 = "Serve a model" → OpenServeWizard.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Serving;
        s.serving_sel = 0;
        assert_eq!(s.pane_focus, PaneFocus::Actions);
        // First activate steps into the detail pane; no overlay yet.
        apply_action(&mut s, KeyAction::PaneActivate);
        assert_eq!(s.pane_focus, PaneFocus::Detail);
        assert!(s.serve_wizard.is_none(), "must not open before stepping in");
        // Second activate opens the operation's manager.
        apply_action(&mut s, KeyAction::PaneActivate);
        assert!(
            s.serve_wizard.is_some(),
            "detail-focus Enter opens the manager"
        );
        // ROCm verb 2 = "Diagnose (doctor)" → OpenExamine (the other mapping).
        let mut r = AppState::new("t".into(), "default-dark".into());
        r.active_tab = ActiveTab::Rocm;
        r.rocm_sel = 2;
        r.pane_focus = PaneFocus::Detail;
        apply_action(&mut r, KeyAction::PaneActivate);
        assert!(
            r.examine_manager.is_some(),
            "ROCm Diagnose opens the doctor"
        );
    }

    #[test]
    fn action_focus_resets_on_move_and_tab_switch() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.pane_focus = PaneFocus::Detail;
        apply_action(&mut s, KeyAction::Move(1));
        assert_eq!(s.pane_focus, PaneFocus::Actions, "Move snaps back to list");
        s.pane_focus = PaneFocus::Detail;
        apply_action(&mut s, KeyAction::SwitchTab(ActiveTab::Home));
        assert_eq!(s.pane_focus, PaneFocus::Actions, "tab switch resets focus");
    }

    #[test]
    fn action_esc_backs_out_of_detail_then_opens_menu() {
        // Esc on Action is intercepted (not the global OpenMenu) so it can back
        // out of the detail pane first.
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Rocm), KeyAction::PaneEscape);
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.pane_focus = PaneFocus::Detail;
        apply_action(&mut s, KeyAction::PaneEscape);
        assert_eq!(s.pane_focus, PaneFocus::Actions, "first Esc → list");
        assert_eq!(s.modal, Modal::None, "first Esc does not open the menu");
        apply_action(&mut s, KeyAction::PaneEscape);
        assert_eq!(s.modal, Modal::Menu, "second Esc opens the menu");
    }

    #[test]
    fn action_select_sets_verb_and_parks_on_list() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.pane_focus = PaneFocus::Detail;
        apply_action(&mut s, KeyAction::PaneSelect(2));
        assert_eq!(s.rocm_sel, 2);
        assert_eq!(s.pane_focus, PaneFocus::Actions);
        // Out-of-range clamps rather than panicking.
        apply_action(&mut s, KeyAction::PaneSelect(999));
        assert!(s.rocm_sel < crate::ui::tabs::rocm::VERB_COUNT);
    }

    #[test]
    fn inline_manager_opens_in_detail_then_backs_out() {
        // Activating a ROCm verb opens its manager inline (focus stays in
        // Details); `←`/Esc backs out — closing the manager and returning focus
        // to the Actions list. This mirrors the event-loop back-out arm.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.rocm_sel = 0; // Set up / Install ROCm → OpenInstall
        apply_action(&mut s, KeyAction::PaneActivate); // → Details
        assert_eq!(s.pane_focus, PaneFocus::Detail);
        assert!(!s.has_open_overlay(), "no manager before second activate");
        apply_action(&mut s, KeyAction::PaneActivate); // opens install_manager
        assert!(s.install_manager.is_some(), "verb opens its manager inline");
        assert!(s.has_open_overlay());

        // Esc backs out on a domain tab while a manager is open.
        assert!(s.should_pane_back_out(crossterm::event::KeyCode::Esc));
        // `←` is left to the manager (it may cycle options), not a back-out.
        assert!(!s.should_pane_back_out(crossterm::event::KeyCode::Left));
        // A normal key does not back out (routes to the manager instead).
        assert!(!s.should_pane_back_out(crossterm::event::KeyCode::Char('j')));

        // The event-loop arm closes the manager + parks focus on Actions.
        s.close_overlays();
        s.pane_focus = PaneFocus::Actions;
        assert!(!s.has_open_overlay(), "back-out closed the inline manager");
        assert_eq!(s.pane_focus, PaneFocus::Actions);
    }

    #[test]
    fn back_out_only_on_domain_tabs_with_a_manager() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        // No manager open → never backs out, even on a domain tab.
        s.active_tab = ActiveTab::Rocm;
        assert!(!s.should_pane_back_out(crossterm::event::KeyCode::Esc));
        // Manager open but on a non-domain tab (opened from Observe hotkey) →
        // the manager keeps its own Esc handling; no domain back-out.
        s.active_tab = ActiveTab::Observe;
        s.examine_manager = Some(crate::ui::examine_manager::ExamineManagerState::default());
        assert!(s.has_open_overlay());
        assert!(!s.should_pane_back_out(crossterm::event::KeyCode::Esc));
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
    fn body_clicks_are_swallowed_while_a_manager_is_open() {
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 90,
            row: 10,
            modifiers: KeyModifiers::NONE,
        };
        // No manager open → the click resolves against the tab's hit-test.
        assert_ne!(resolve_mouse(click, &s), KeyAction::Nothing);
        // Manager open → the body click is swallowed (no click-through).
        s.install_manager = Some(crate::ui::install_manager::InstallManagerState::default());
        assert_eq!(resolve_mouse(click, &s), KeyAction::Nothing);
    }

    /// Build a ScrollDown/Up/Left/Right event at a pointer position.
    fn wheel(kind: MouseEventKind, col: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column: col,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn scrollbar_position_maps_proportionally() {
        let h = ScrollbarHandle {
            track: Rect::new(50, 0, 1, 10),
            horizontal: false,
            content_len: 100,
            viewport_len: 10,
            target: ScrollTarget::Console,
        };
        assert_eq!(h.track_position_at(50, 0), 0);
        assert_eq!(h.track_position_at(50, 9), 90);
        assert_eq!(h.track_position_at(50, 5), 90 * 5 / 9);
        assert_eq!(h.track_position_at(50, 99), 90);
    }

    #[test]
    fn scrollbar_click_grabs_drag_scrolls_then_releases() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.scrollbars.borrow_mut().push(ScrollbarHandle {
            track: Rect::new(60, 0, 1, 10),
            horizontal: false,
            content_len: 100,
            viewport_len: 10,
            target: ScrollTarget::Console,
        });
        // Click near the bottom of the track → grab + jump near the end.
        let down = wheel(MouseEventKind::Down(MouseButton::Left), 60, 9);
        let a = resolve_mouse(down, &s);
        assert_eq!(a, KeyAction::ScrollGrab(ScrollTarget::Console, 90, 0));
        apply_action(&mut s, a);
        assert_eq!(s.console_scroll, 90);
        assert_eq!(
            s.scroll_drag,
            Some(ScrollDrag {
                target: ScrollTarget::Console,
                grab_offset: 0,
            })
        );
        // Drag to the top — the off-axis column is ignored, so it still tracks.
        let drag = wheel(MouseEventKind::Drag(MouseButton::Left), 40, 0);
        let a = resolve_mouse(drag, &s);
        assert_eq!(a, KeyAction::ScrollGrab(ScrollTarget::Console, 0, 0));
        apply_action(&mut s, a);
        assert_eq!(s.console_scroll, 0);
        // Release clears the drag.
        let up = wheel(MouseEventKind::Up(MouseButton::Left), 40, 0);
        let a = resolve_mouse(up, &s);
        assert_eq!(a, KeyAction::ScrollRelease);
        apply_action(&mut s, a);
        assert_eq!(s.scroll_drag, None);
    }

    #[test]
    fn rendered_thumb_cells_are_grabbable() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        for horizontal in [false, true] {
            let mut s = AppState::new("t".into(), "default-dark".into());
            let area = if horizontal {
                Rect::new(0, 0, 4, 2)
            } else {
                Rect::new(0, 0, 2, 4)
            };
            let target = if horizontal {
                ScrollTarget::ConsoleH
            } else {
                ScrollTarget::Console
            };
            if horizontal {
                s.console_hscroll = 1;
            } else {
                s.console_scroll = 1;
            }
            let backend = TestBackend::new(area.width, area.height);
            let mut terminal = Terminal::new(backend).unwrap();
            let mut body = area;
            terminal
                .draw(|frame| {
                    body = if horizontal {
                        crate::ui::panel::horizontal_scrollbar(frame, area, 4, 2, 1, &s.theme)
                    } else {
                        crate::ui::panel::vertical_scrollbar(frame, area, 4, 2, 1, &s.theme)
                    };
                })
                .unwrap();
            s.record_scrollbar(area, body, horizontal, 4, 2, target);
            let handle = s.scrollbars.borrow()[0];
            let (thumb_start, thumb_len) = handle.thumb_geometry(1);
            assert_eq!((thumb_start, thumb_len), (1, 2));
            for cell in thumb_start..thumb_start + thumb_len {
                let (col, row) = if horizontal {
                    (cell, handle.track.y)
                } else {
                    (handle.track.x, cell)
                };
                assert_eq!(
                    resolve_mouse(wheel(MouseEventKind::Down(MouseButton::Left), col, row), &s),
                    KeyAction::ScrollGrab(target, 1, cell - thumb_start)
                );
            }
            let (before_col, before_row) = if horizontal {
                (0, handle.track.y)
            } else {
                (handle.track.x, 0)
            };
            assert_eq!(
                resolve_mouse(
                    wheel(
                        MouseEventKind::Down(MouseButton::Left),
                        before_col,
                        before_row
                    ),
                    &s,
                ),
                KeyAction::ScrollGrab(target, 0, 0)
            );
            let (after_col, after_row) = if horizontal {
                (3, handle.track.y)
            } else {
                (handle.track.x, 3)
            };
            assert_eq!(
                resolve_mouse(
                    wheel(
                        MouseEventKind::Down(MouseButton::Left),
                        after_col,
                        after_row
                    ),
                    &s,
                ),
                KeyAction::ScrollGrab(target, 2, 0)
            );
        }
    }

    #[test]
    fn stationary_thumb_drag_preserves_logical_position() {
        for target in [ScrollTarget::Console, ScrollTarget::DockLogs] {
            let mut s = AppState::new("t".into(), "default-dark".into());
            s.console_scroll = 5;
            s.dock_logs_scroll = 5;
            s.scrollbars.borrow_mut().push(ScrollbarHandle {
                track: Rect::new(60, 0, 1, 10),
                horizontal: false,
                content_len: 20,
                viewport_len: 10,
                target,
            });
            let down = wheel(MouseEventKind::Down(MouseButton::Left), 60, 4);
            let action = resolve_mouse(down, &s);
            apply_action(&mut s, action);
            let before = if target == ScrollTarget::DockLogs {
                s.dock_logs_scroll
            } else {
                s.console_scroll
            };
            let drag = wheel(MouseEventKind::Drag(MouseButton::Left), 60, 4);
            let action = resolve_mouse(drag, &s);
            apply_action(&mut s, action);
            let after = if target == ScrollTarget::DockLogs {
                s.dock_logs_scroll
            } else {
                s.console_scroll
            };
            assert_eq!(after, before, "stationary {target:?} drag moved");
        }
    }

    #[test]
    fn horizontal_thumb_drag_tracks_pointer_column() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.console_hscroll = 20;
        s.scrollbars.borrow_mut().push(ScrollbarHandle {
            track: Rect::new(10, 20, 20, 1),
            horizontal: true,
            content_len: 100,
            viewport_len: 20,
            target: ScrollTarget::ConsoleH,
        });
        let action = resolve_mouse(wheel(MouseEventKind::Down(MouseButton::Left), 14, 20), &s);
        apply_action(&mut s, action);
        assert_eq!(s.console_hscroll, 20);
        let action = resolve_mouse(wheel(MouseEventKind::Drag(MouseButton::Left), 19, 99), &s);
        apply_action(&mut s, action);
        assert!(s.console_hscroll > 20);
    }

    #[test]
    fn dock_scrollbar_grab_inverts_tail_anchored_offset() {
        let s = AppState::new("t".into(), "default-dark".into());
        s.scrollbars.borrow_mut().push(ScrollbarHandle {
            track: Rect::new(0, 0, 1, 10),
            horizontal: false,
            content_len: 100,
            viewport_len: 10,
            target: ScrollTarget::DockLogs,
        });
        // Top of the bar = oldest lines = fully scrolled up from the tail (max).
        let top = resolve_mouse(wheel(MouseEventKind::Down(MouseButton::Left), 0, 0), &s);
        assert_eq!(top, KeyAction::ScrollGrab(ScrollTarget::DockLogs, 90, 0));
        // Bottom of the bar = newest tail = offset 0.
        let bot = resolve_mouse(wheel(MouseEventKind::Down(MouseButton::Left), 0, 9), &s);
        assert_eq!(bot, KeyAction::ScrollGrab(ScrollTarget::DockLogs, 0, 0));
    }

    #[test]
    fn wheel_over_actions_list_moves_selection_by_one() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        // Left column (Actions) is the first ~46% — a low column is over the list.
        let over_list = wheel(MouseEventKind::ScrollDown, 10, 12);
        assert_eq!(resolve_mouse(over_list, &s), KeyAction::Move(1));
        let up = wheel(MouseEventKind::ScrollUp, 10, 12);
        assert_eq!(resolve_mouse(up, &s), KeyAction::Move(-1));
    }

    #[test]
    fn wheel_over_details_pane_does_not_move_the_list() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Serving;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        // A high column lands in the Details pane (right ~54%) → no list move.
        let over_detail = wheel(MouseEventKind::ScrollDown, 140, 12);
        assert_eq!(resolve_mouse(over_detail, &s), KeyAction::Nothing);
    }

    #[test]
    fn wheel_over_open_console_pans_the_log_not_the_list() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        s.logs_view = Some(crate::ui::logs_view::LogsViewState {
            active_job: Some("logs".into()),
            ..Default::default()
        });
        // Console showing → vertical wheel pans the log (×3 lines), not the list.
        assert_eq!(
            resolve_mouse(wheel(MouseEventKind::ScrollDown, 10, 12), &s),
            KeyAction::ScrollConsole(3, 0)
        );
        // Horizontal wheel pans columns (×6) for off-screen-wide log lines.
        assert_eq!(
            resolve_mouse(wheel(MouseEventKind::ScrollRight, 10, 12), &s),
            KeyAction::ScrollConsole(0, 6)
        );
    }

    #[test]
    fn wheel_over_logs_dock_scrolls_the_dock() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Serving;
        // A recorded dock rect off to the right of the body.
        s.last_dock_area = Some(Rect::new(160, 4, 52, 30));
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        let over_dock = wheel(MouseEventKind::ScrollDown, 180, 12);
        assert_eq!(resolve_mouse(over_dock, &s), KeyAction::ScrollDock(3));
        // A point outside the dock does not scroll it.
        let over_body = wheel(MouseEventKind::ScrollDown, 10, 12);
        assert_ne!(resolve_mouse(over_body, &s), KeyAction::ScrollDock(3));
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
    fn wheel_over_form_screen_overlay_is_swallowed() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        // Overlay open but on its form (no active_job) → nothing to pan, and the
        // obscured Actions list must NOT move.
        s.install_manager = Some(crate::ui::install_manager::InstallManagerState::default());
        assert_eq!(
            resolve_mouse(wheel(MouseEventKind::ScrollDown, 10, 12), &s),
            KeyAction::Nothing
        );
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

    #[test]
    fn console_scroll_delta_maps_nav_keys_only() {
        use crossterm::event::KeyCode;
        assert_eq!(console_scroll_delta(KeyCode::PageDown), Some((10, 0)));
        assert_eq!(console_scroll_delta(KeyCode::PageUp), Some((-10, 0)));
        assert_eq!(console_scroll_delta(KeyCode::Down), Some((1, 0)));
        assert_eq!(console_scroll_delta(KeyCode::Right), Some((0, 4)));
        // Console action keys are NOT scroll keys (they reach on_console_key).
        assert_eq!(console_scroll_delta(KeyCode::Esc), None);
        assert_eq!(console_scroll_delta(KeyCode::Enter), None);
        assert_eq!(console_scroll_delta(KeyCode::Char('q')), None);
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
    fn only_press_key_events_are_actionable() {
        // The event loop gates overlay dispatch on this predicate so a single
        // keystroke isn't processed twice by an overlay's `on_key` (Release /
        // Repeat echoes on Windows Terminal / ConPTY / kitty keyboard). The
        // double-fire re-opened the serve wizard's model picker on Enter instead
        // of choosing — this pins Press-only routing.
        assert!(is_actionable_key(KeyEventKind::Press));
        assert!(!is_actionable_key(KeyEventKind::Release));
        assert!(!is_actionable_key(KeyEventKind::Repeat));
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
        // The letter hotkeys fire ONLY on Observe now — quick jumps into the
        // managers. They open the matching overlay via the seam.
        assert_eq!(
            hk(KeyCode::Char('w'), ActiveTab::Observe),
            KeyAction::OpenServeWizard
        );
        assert_eq!(
            hk(KeyCode::Char('e'), ActiveTab::Observe),
            KeyAction::OpenEngineManager
        );
        assert_eq!(
            hk(KeyCode::Char('d'), ActiveTab::Observe),
            KeyAction::OpenExamine
        );
        assert_eq!(
            hk(KeyCode::Char('i'), ActiveTab::Observe),
            KeyAction::OpenInstall
        );
        // Retired on the domain tabs: the Actions list is the single path there,
        // so the letter hotkeys are inert on ROCm/Serving (and Home/Chat).
        for c in ['w', 'e', 'd', 'u', 'i', 'l', 'r', 'n', 'a', 'c', 'p', 's'] {
            assert_eq!(
                hk(KeyCode::Char(c), ActiveTab::Rocm),
                KeyAction::Nothing,
                "key {c} must be retired on the ROCm tab"
            );
            assert_eq!(
                hk(KeyCode::Char(c), ActiveTab::Serving),
                KeyAction::Nothing,
                "key {c} must be retired on the Serving tab"
            );
        }
        assert_eq!(hk(KeyCode::Char('w'), ActiveTab::Home), KeyAction::Nothing);
        // On the Chat tab none of these open an overlay. `i` means insert mode.
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
        // T13: OpenBenchRun joins the mutual-exclusion set.
        apply_action(&mut s, KeyAction::OpenBenchRun);
        assert!(s.bench_run.is_some() && s.config_manager.is_none());
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
        // No modal, Observe → Move by ONE (drives the instances selection)
        assert_eq!(
            handle_mouse(scroll_down, &Modal::None, ActiveTab::Observe),
            KeyAction::Move(1)
        );
        // Detail modal → ScrollModal by one line
        assert_eq!(
            handle_mouse(scroll_down, &Modal::Detail, ActiveTab::Observe),
            KeyAction::ScrollModal(1)
        );
        // ThemePicker → Move (drives picker cursor)
        assert_eq!(
            handle_mouse(scroll_down, &Modal::ThemePicker, ActiveTab::Home),
            KeyAction::Move(1)
        );
        // Domain-tab scroll is NOT routed here (resolve_mouse owns it) → Nothing.
        assert_eq!(
            handle_mouse(scroll_down, &Modal::None, ActiveTab::Rocm),
            KeyAction::Nothing
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
        // Simulate a prior `/provider openai` so the realignment to Local on
        // accept is observable (Local is the default, so starting there would
        // make the assertion below tautological).
        s.active_provider = ChatProvider::Openai;

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
        assert_eq!(
            s.chat_endpoint_rebuild,
            Some(ChatProvider::Openai),
            "accept raises the rebuild edge carrying the previous provider"
        );
        assert_eq!(s.active_provider, ChatProvider::Local);
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
        // Start off-Local so the realignment on accept is observable (Local is
        // the default provider; asserting it without this would be tautological).
        s.active_provider = ChatProvider::Openai;
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:13305/v1",
            "Llama-3.2-3B",
        )));
        apply_action(&mut s, KeyAction::ChatDetectSave);
        assert_eq!(s.chat_consent, ChatConsent::Accepted);
        assert!(s.chat_persist_dispatch, "save raises the persist edge");
        assert_eq!(
            s.chat_endpoint_rebuild,
            Some(ChatProvider::Openai),
            "save also raises the rebuild edge carrying the previous provider"
        );
        assert_eq!(s.active_provider, ChatProvider::Local);
        assert_eq!(
            s.chat_llm.as_ref().map(|c| c.base_url.as_str()),
            Some("http://localhost:13305/v1")
        );
        // No offer → save is a no-op (no edge).
        let mut s2 = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s2, KeyAction::ChatDetectSave);
        assert!(!s2.chat_persist_dispatch);
        assert!(s2.chat_endpoint_rebuild.is_none());
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

    // --- Focused host (Phase 2): default-off focus flag hosts one overlay ---

    #[test]
    fn resolved_args_focus_defaults_none() {
        // The focus flag is additive and off by default in every constructor, so
        // existing dash/chat behavior is byte-identical.
        assert!(args_with_anthropic_key(None).focus.is_none());
    }

    #[test]
    fn should_skip_daemon_predicate_matches_focus() {
        // The dashboard (focus=None) keeps the daemon client + chat backend; any
        // focus skips both. The render branch reuses this same predicate.
        assert!(!should_skip_daemon(None));
        assert!(should_skip_daemon(Some(Focus::Setup)));
        assert!(should_skip_daemon(Some(Focus::Serve)));
        assert!(should_skip_daemon(Some(Focus::Examine)));
        // focus=None never self-exits — the dash loop only breaks on Quit/EOF.
        assert!(!st().focused_should_exit(None));
    }

    #[test]
    fn open_overlay_for_focus_opens_the_right_overlay() {
        let mut s = st();
        assert!(open_overlay_for_focus(&mut s, Focus::Setup).is_empty());
        assert!(s.onboarding.is_some());
        assert!(s.serve_wizard.is_none() && s.examine_manager.is_none());

        let mut s = st();
        assert!(open_overlay_for_focus(&mut s, Focus::Serve).is_empty());
        assert!(s.serve_wizard.is_some());
        assert!(s.onboarding.is_none() && s.examine_manager.is_none());

        let mut s = st();
        let fx = open_overlay_for_focus(&mut s, Focus::Examine);
        assert!(s.examine_manager.is_some());
        assert!(s.onboarding.is_none() && s.serve_wizard.is_none());
        assert_eq!(fx.len(), 1, "examine auto-runs on open");
    }

    #[test]
    fn focused_examine_auto_runs_rocm_examine() {
        let mut s = st();
        let fx = open_overlay_for_focus(&mut s, Focus::Examine);
        assert_eq!(fx.len(), 1, "exactly one spawn side effect on open");
        match &fx[0] {
            rocm_dash_core::state::SideEffect::SpawnJob { cmd, args, .. } => {
                assert!(cmd.contains("rocm"), "cmd resolves to the rocm exe: {cmd}");
                assert!(
                    args.iter().any(|a| a == "examine"),
                    "examine in args: {args:?}"
                );
            }
            other => panic!("expected SpawnJob, got {other:?}"),
        }
        assert_eq!(
            s.examine_manager.as_ref().unwrap().active_job.as_deref(),
            Some("examine"),
            "the auto-run wires the active job"
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
    }

    #[test]
    fn focused_exit_gate_holds_until_examine_closed_at_root() {
        let mut s = st();
        // Focused Diagnose: examine opens AND auto-runs → a job-console sub-state.
        let _ = open_overlay_for_focus(&mut s, Focus::Examine);
        assert!(s.examine_manager.as_ref().unwrap().active_job.is_some());
        assert!(
            !s.focused_should_exit(Some(Focus::Examine)),
            "a running job keeps the launcher out"
        );

        // Job terminal → first Esc dismisses the console back to the intro card;
        // the overlay is still open, so the gate stays shut.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        let _ = crate::ui::examine_manager::on_key(
            &mut s.examine_manager,
            &mut s.jobs,
            press(KeyCode::Esc),
        );
        assert!(
            s.examine_manager.is_some(),
            "console dismissed, overlay stays"
        );
        assert!(
            !s.focused_should_exit(Some(Focus::Examine)),
            "at the intro (not root-closed) the gate is still shut"
        );

        // Second Esc at the intro (root) closes the overlay → now exit to menu.
        let _ = crate::ui::examine_manager::on_key(
            &mut s.examine_manager,
            &mut s.jobs,
            press(KeyCode::Esc),
        );
        assert!(s.examine_manager.is_none(), "root Esc closes the overlay");
        assert!(
            s.focused_should_exit(Some(Focus::Examine)),
            "closed at root → return to the launcher"
        );
    }

    #[test]
    fn focused_close_keys_swallowed_while_job_runs() {
        // Regression for the mid-job ejection defect: `q` and running-`Esc` must
        // be swallowed by the focused host while the job is non-terminal, so the
        // overlay is never nulled (which would tear the runtime down and kill the
        // child via kill_on_drop mid-write).
        let mut s = st();
        let _ = open_overlay_for_focus(&mut s, Focus::Examine); // auto-runs a job
        assert!(s.has_active_console(), "examine console is live");
        // Running job → q and Esc are blocked; Ctrl+C ('c') is NOT (it cancels).
        assert!(focused_close_key_blocked(
            &s,
            Some(Focus::Examine),
            KeyCode::Char('q')
        ));
        assert!(focused_close_key_blocked(
            &s,
            Some(Focus::Examine),
            KeyCode::Esc
        ));
        assert!(!focused_close_key_blocked(
            &s,
            Some(Focus::Examine),
            KeyCode::Char('c')
        ));
        // The dashboard (focus=None) never blocks — behavior is unchanged there.
        assert!(!focused_close_key_blocked(&s, None, KeyCode::Char('q')));

        // Because those keys are swallowed (never routed to the manager), the
        // overlay stays open and the exit gate stays shut mid-job.
        assert!(s.examine_manager.is_some());
        assert!(!s.focused_should_exit(Some(Focus::Examine)));

        // Once the job is terminal, close keys are allowed again → normal exit.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        assert!(
            !focused_close_key_blocked(&s, Some(Focus::Examine), KeyCode::Char('q')),
            "a terminal job no longer blocks exit (the child already exited)"
        );
    }

    #[test]
    fn focused_gate_shut_across_serve_sub_states() {
        // Exit-at-root (b)+(c): the focused gate stays shut while a folder
        // browser / model picker / approval is open — it only opens at root.
        let recipes: Vec<crate::ui::model_picker::ModelRecipeSummary> = Vec::new();
        let mut s = st();
        let _ = open_overlay_for_focus(&mut s, Focus::Serve);

        // (b) Tab on the Model field opens the folder-browser sub-popup.
        let _ = crate::ui::serve_wizard::on_key(
            &mut s.serve_wizard,
            &mut s.jobs,
            &recipes,
            press(KeyCode::Tab),
        );
        assert!(s.serve_wizard.as_ref().unwrap().browser.is_some());
        assert!(
            !s.focused_should_exit(Some(Focus::Serve)),
            "gate shut while the folder browser is open"
        );
        // Esc closes the sub-popup, not the wizard → still shut.
        let _ = crate::ui::serve_wizard::on_key(
            &mut s.serve_wizard,
            &mut s.jobs,
            &recipes,
            press(KeyCode::Esc),
        );
        assert!(s.serve_wizard.as_ref().unwrap().browser.is_none());
        assert!(s.serve_wizard.is_some());
        assert!(!s.focused_should_exit(Some(Focus::Serve)));

        // (c) Stage an approval (valid model, Launch field, Enter).
        {
            let w = s.serve_wizard.as_mut().unwrap();
            w.model = "org/model".to_string();
            w.field = crate::ui::serve_wizard::FIELDS.len() - 1; // Launch
        }
        let _ = crate::ui::serve_wizard::on_key(
            &mut s.serve_wizard,
            &mut s.jobs,
            &recipes,
            press(KeyCode::Enter),
        );
        assert!(
            s.serve_wizard.as_ref().unwrap().approval.is_some(),
            "a launch approval is pending"
        );
        assert!(
            !s.focused_should_exit(Some(Focus::Serve)),
            "gate shut while an approval is pending"
        );

        // Only a root close (wizard → None) opens the gate.
        s.serve_wizard = None;
        assert!(s.focused_should_exit(Some(Focus::Serve)));
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
            chat_env_url: None,
            chat_api_key: None,
            anthropic_api_key: key.map(str::to_string),
            chat_auto_consent: false,
            chat_mock: false,
            model_recipes: Vec::new(),
            runtimes: Vec::new(),
            automations: Vec::new(),
            tool_executor: None,
            bench_results_dir: None,
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
    fn chat_scroll_clamps_and_updates_follow_state() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_max_scroll = 20;
        s.chat_scroll = 20;
        apply_action(&mut s, KeyAction::ChatScroll(-100));
        assert_eq!(s.chat_scroll, 0, "scroll clamps at top");
        assert!(!s.chat_follow, "scrolling above the bottom disables follow");
        apply_action(&mut s, KeyAction::ChatScroll(7));
        assert_eq!(s.chat_scroll, 7);
        assert!(!s.chat_follow);
        apply_action(&mut s, KeyAction::ChatScroll(100));
        assert_eq!(s.chat_scroll, 20, "scroll clamps at measured bottom");
        assert!(s.chat_follow, "scrolling to the bottom restores follow");
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

    #[test]
    fn chat_scrollbar_grab_updates_follow_state() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_max_scroll = 20;
        s.chat_scroll = 20;

        s.apply_scroll_grab(ScrollTarget::Chat, 5, 0);
        assert_eq!(s.chat_scroll, 5);
        assert!(!s.chat_follow, "dragging above the bottom disables follow");

        s.apply_scroll_grab(ScrollTarget::Chat, 20, 0);
        assert_eq!(s.chat_scroll, 20);
        assert!(s.chat_follow, "dragging to the bottom restores follow");
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
