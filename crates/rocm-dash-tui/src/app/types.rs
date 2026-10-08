// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared type and enum definitions for the dashboard reducer: `Focus`,
//! `ResolvedArgs`, connection/tab/chat/replay state, `Modal`, `UpdateStatus`,
//! and the slash/plan/approval payload types. No `AppState` access — split
//! out of `app/mod.rs` to keep the core reducer focused.

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
    /// Sampling temperature for chat requests, CLI-flag value merged over
    /// config. `None` leaves the endpoint default untouched.
    pub chat_temperature: Option<f32>,
    /// Nucleus-sampling `top_p` for chat requests, CLI merged over config.
    pub chat_top_p: Option<f32>,
    /// Max generated tokens for chat requests, CLI merged over config.
    pub chat_max_tokens: Option<u32>,
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
    /// System prompt for the chat assistant: the ROCm tool-use prompt plus this
    /// machine's detected facts (OS, WSL, AMD GPU, available engines). Composed
    /// by the bin (`apps/rocm`, which has `rocm-core`) so this crate needs no
    /// `rocm-core` dep. `None` for demo/replay/`--chat-mock`, which have no bin
    /// seam and keep the agent's built-in default preamble.
    pub chat_system_prompt: Option<String>,
    /// Bin-injected tool-executor seam; None for demo/replay/mock — dash behaves
    /// as today. Stored here (Phase 2 plumbing); Phase 3 will use it.
    pub tool_executor: Option<crate::tool_exec::SharedRocmToolExecutor>,
    /// Daemon-tailed bench CSV path (`config.dashboard.daemon.bench_results_dir`).
    ///
    /// When `Some`, the bench-run form defaults `--out` to this path so appended
    /// rows appear live in the bench tab. Adapted by the bin (owns `rocm-core`).
    pub bench_results_dir: Option<std::path::PathBuf>,
    /// Managed-service records that are no longer running, counted from the
    /// registry by the bin at launch (the same seam `model_recipes` / `runtimes`
    /// / `automations` use - a snapshot, not a live feed). The services overlay
    /// only ever renders the live instances the daemon surfaces, so without this
    /// a host whose servers had all failed showed an empty overlay and no sign
    /// that any record existed. 0 when there are none.
    pub services_past_attempts: usize,
    /// Whether any managed-service record on disk was already `ready`/`running`
    /// at launch — the same statuses `Instance::status.is_serving()` treats as
    /// "actively serving" — read the same way `services_past_attempts` is:
    /// synchronously, from the same registry, before the TUI starts.
    /// `AppState::instances` starts empty and is only populated once the
    /// daemon's first instance snapshot lands, so
    /// without this a `q` pressed in that startup window (or at any point the
    /// daemon connection never succeeds) saw no live instance and exited with
    /// no confirm prompt even though a model actually was being served (issue
    /// #145). `AppState::has_live_instance` ORs this in only until that first
    /// snapshot arrives, so it never outlives its own staleness.
    pub startup_has_live_service: bool,
}

impl ResolvedArgs {
    /// The optional sampling controls (temperature/top_p/max_tokens) resolved
    /// for chat, bundled for the agent builders. CLI-over-config merge already
    /// happened in the bin, so these are the final values.
    pub(crate) const fn inference_params(&self) -> crate::agent::InferenceParams {
        crate::agent::InferenceParams {
            temperature: self.chat_temperature,
            top_p: self.chat_top_p,
            max_tokens: self.chat_max_tokens,
        }
    }
}

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
/// `pub` (not `pub(crate)`) because it is a payload of the `pub`
/// [`crate::client::ClientMsg`] enum (mirrors [`crate::tool_exec::ApprovalIntent`]);
/// the reducer entrypoints that consume it stay crate-private.
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

/// Reduction of a completed `rocm update --json` check, for the Home tab's
/// Updates tile. Distinct from `update_manager`'s interactive job state — this
/// tracks the periodic background check only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateStatus {
    /// No check has completed yet (startup, or `state.simulated`).
    Unknown,
    NoManagedRuntimes,
    UpToDate,
    UpdateAvailable {
        latest_version: String,
    },
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_digit_maps_five_to_chat() {
        // 5-tab digit map; Chat is now '5', '6'/'0' are out of range.
        assert_eq!(ActiveTab::from_digit('1'), Some(ActiveTab::Home));
        assert_eq!(ActiveTab::from_digit('5'), Some(ActiveTab::Chat));
        assert_eq!(ActiveTab::from_digit('6'), None);
        assert_eq!(ActiveTab::from_digit('0'), None);
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
}
