// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Rig `Tool` wrappers over the cached telemetry snapshot, plus the
//! read-only and mutating ROCm machine-inspection tools and their registration
//! onto a Rig `AgentBuilder`.
//!
//! Split out of `agent.rs` to keep the `AgentClient` seam focused. The shared
//! seam types (`StateSnapshot`, `AgentError`, `AgentClient`) stay in
//! `agent/mod.rs`; the JSON summarization these tools call lives in
//! `agent::snapshot`.
//!
//! This is also where a read-only tool result is shaped for the model:
//! [`model_facing_tool_result`] drops the command output the bin's MCP
//! envelope carries twice and bounds what is left to
//! [`TOOL_RESULT_CHAR_BUDGET`], so one large result cannot exhaust a locally
//! served model's context. See `docs/llm-tool-use.md`.

use std::sync::{Arc, Mutex};

use serde::Deserialize;
use serde_json::{Value, json};

use rig::completion::ToolDefinition;
use rig::tool::Tool;

use crate::client::ClientMsg;
use crate::tool_exec::{RocmToolOutcome, SharedRocmToolExecutor};

use tokio::sync::mpsc::UnboundedSender;

use super::StateSnapshot;
use super::snapshot::{
    bench_summary_json, gpu_status_json, list_instances_json, tokens_per_watt_json,
};

// ---------------------------------------------------------------------------
// Rig Tool wrappers for the telemetry tools. Each holds an `Arc<StateSnapshot>`
// (read-only) and a shared `fired` log so the reply can cite which tools ran.
// `call` only reads the snapshot — no mutation, no network, no file I/O.
// ---------------------------------------------------------------------------

/// Shared "which tools fired" log, threaded into every tool struct below and
/// into the three real `AgentClient` backends via `agent::clients`.
pub(super) type FiredLog = Arc<Mutex<Vec<String>>>;

fn record(fired: &FiredLog, name: &str) {
    if let Ok(mut g) = fired.lock() {
        g.push(name.to_string());
    }
}

/// Error type for all tools. Tools are read-only and effectively infallible,
/// but the trait requires an error type.
#[derive(Debug, thiserror::Error)]
#[error("tool error: {0}")]
pub struct ToolError(String);

/// Empty argument payload for tools that take no parameters.
#[derive(Debug, Deserialize, Default)]
pub struct NoArgs {}

pub struct GpuStatusTool {
    pub snap: Arc<StateSnapshot>,
    pub fired: FiredLog,
}

#[derive(Debug, Deserialize, Default)]
pub struct GpuStatusArgs {
    #[serde(default)]
    pub gpu_index: Option<usize>,
}

impl Tool for GpuStatusTool {
    const NAME: &'static str = "gpu_status";
    type Error = ToolError;
    type Args = GpuStatusArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Per-GPU utilization %, temperature °C, power W and VRAM MB \
                          from the latest telemetry snapshot. Optional gpu_index \
                          selects a single GPU."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "gpu_index": {
                        "type": "integer",
                        "description": "Zero-based GPU index; omit for all GPUs."
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        record(&self.fired, Self::NAME);
        Ok(gpu_status_json(&self.snap, args.gpu_index))
    }
}

pub struct ListInstancesTool {
    pub snap: Arc<StateSnapshot>,
    pub fired: FiredLog,
}

impl Tool for ListInstancesTool {
    const NAME: &'static str = "list_instances";
    type Error = ToolError;
    type Args = NoArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List discovered serving instances: name, model, status, \
                          KV-cache usage %, running/waiting request counts, \
                          the whole-node `node` figure (see tokens_per_watt), \
                          generation tok/s, TTFT/TPOT (each with a fresh/held \
                          flag; held means the last value, not a live reading), \
                          and attributed_power_w (this instance's share of its \
                          GPUs' power, not whole-board power)."
                .to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        record(&self.fired, Self::NAME);
        Ok(list_instances_json(&self.snap))
    }
}

pub struct BenchSummaryTool {
    pub snap: Arc<StateSnapshot>,
    pub fired: FiredLog,
}

impl Tool for BenchSummaryTool {
    const NAME: &'static str = "bench_summary";
    type Error = ToolError;
    type Args = NoArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Pass^N / Pass@N benchmark rollup grouped by cell/model/\
                          engine/tp/dtype/concurrency over the cached bench rows."
                .to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        record(&self.fired, Self::NAME);
        Ok(bench_summary_json(&self.snap))
    }
}

pub struct TokensPerWattTool {
    pub snap: Arc<StateSnapshot>,
    pub fired: FiredLog,
}

impl Tool for TokensPerWattTool {
    const NAME: &'static str = "tokens_per_watt";
    type Error = ToolError;
    type Args = NoArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Per-instance efficiency: generation tokens/sec divided by \
                          the instance's attributed share of its GPUs' power (W). \
                          GPUs shared by several instances are split between them. \
                          Also returns `node`: whole-node tokens/sec over total \
                          board power (whole-board, so it is not the sum of the \
                          per-instance values), with freshness `fresh`, `held` \
                          (carried from a prior scrape, expires within about \
                          30 s) or null when unknown. `node` is null, meaning \
                          no figure and not zero, when there is no traffic or \
                          no power telemetry."
                .to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        record(&self.fired, Self::NAME);
        Ok(tokens_per_watt_json(&self.snap))
    }
}

/// All telemetry tool names, for definition/uniqueness checks and docs.
pub const TELEMETRY_TOOL_NAMES: [&str; 4] = [
    GpuStatusTool::NAME,
    ListInstancesTool::NAME,
    BenchSummaryTool::NAME,
    TokensPerWattTool::NAME,
];

// ---------------------------------------------------------------------------
// Read-only ROCm machine-inspection tools (group B). These forward the model's
// tool-call intent across the rocm-core-free [`crate::tool_exec`] seam to the
// bin-supplied executor (live dash only). They are READ-ONLY in the approval
// sense: none of them performs a mutating action (an install, start, stop,
// uninstall, configuration change, or anything else that needs operator
// approval). An intent the bin classifies as mutating comes back as
// `ApprovalRequired` and [`run_rocm_read_tool`] turns it into an error, and the
// mutating tools are group D below, which only surface an approval request
// rather than executing. That is a narrower claim than "no side effects":
// reading can still refresh local bookkeeping and caches. `services`,
// `service_logs`, `port_status`, `bridge_snapshot` and the `rocm_command`
// service views reload the managed-service records, which rewrites a record
// whose state changed (and drops the endpoint key of a service whose requested
// stop has finished). `doctor`/`examine` can re-register the configured setup
// runtime in the runtime registry when its entry is missing. `update_check` and
// `install_sdk_dry_run` cache package-index metadata under the cache folder.
// Neither prepares a Python or saves a Python path: on a host with no usable
// Python on PATH they report the Python a real update or install would prepare
// (`update_check` runs `rocm update`, whose text report then carries a
// `python_note:` line saying so), and only a real apply or install, which these
// tools never run, prepares it. `install_sdk_dry_run` fetches no `uv` either: a
// ROCm 10 preview runs a `uv` that is already installed to resolve the package
// versions and, with none, prints the lines that depend on them as not resolved,
// so it creates no `tools` folder in the data folder. (`update_check` on a ROCm
// 10 or newer runtime still resolves with `uv` and can fetch the managed `uv`
// into the data folder, though never a Python.) The boundary type
// (`SharedRocmToolExecutor`) is plain data — importing it does NOT pull
// `rocm-core` into this crate; the `agent` module stays the sole `rig` namer.
// ---------------------------------------------------------------------------

/// How many characters of one tool result may reach the model.
///
/// A tool result is not free: it is sent back as a message and counts against
/// the model's context on that request and on every later turn of the same
/// conversation. A locally served model is typically configured with a 32k
/// context (llama.cpp's default `n_ctx`), of which the system prompt and the
/// tool schemas already take a few thousand tokens, so the remaining room has
/// to cover several turns. 16 KiB is roughly 4.5k tokens — one large result
/// still fits alongside the prompt and a handful of turns, where an uncapped
/// one (`rocm_command` with `["model"]` measured 204,470 characters, about
/// 58k tokens) cannot fit in such a context at all and the provider rejects
/// the whole request.
pub(crate) const TOOL_RESULT_CHAR_BUDGET: usize = 16 * 1024;

/// Shape one executor result for the model: drop the command output the
/// envelope carries twice, then bound what is left to `budget`.
///
/// Both halves exist because the result is JSON the dash hands to the model
/// verbatim, unlike the CLI chat path (`apps/rocm/src/chat/turn.rs`), which
/// reads only the envelope's text.
fn model_facing_tool_result(value: Value, budget: usize) -> Value {
    cap_tool_result(drop_duplicated_command_output(value), budget)
}

/// Drop `structuredContent.stdout` / `.stderr` when the envelope's own text
/// already carries them.
///
/// The bin's `internal_mcp_tool_result_from_command` puts a command's captured
/// output in `content[0].text` *and* again in `structuredContent.stdout` (and
/// `.stderr`). The MCP envelope keeps both — `rocm mcp-call` and the approval
/// replay path read the structured fields — but the model only needs one copy,
/// and charging it twice for the same bytes is what makes a big read-only
/// result twice as expensive as it has to be. Only the duplicated keys go; a
/// tool whose data lives *only* in `structuredContent` (`engines`,
/// `bridge_snapshot`, `automations`, `gpu_snapshot`) is untouched.
fn drop_duplicated_command_output(mut value: Value) -> Value {
    let text = tool_result_content_text(&value);
    if text.is_empty() {
        return value;
    }
    let Some(structured) = value
        .get_mut("structuredContent")
        .and_then(Value::as_object_mut)
    else {
        return value;
    };
    for key in ["stdout", "stderr"] {
        let duplicated = structured
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|s| {
                let trimmed = s.trim();
                trimmed.is_empty() || text.contains(trimmed)
            });
        if duplicated {
            structured.remove(key);
        }
    }
    value
}

/// The text items of an MCP tool-result envelope, joined. Empty when the value
/// is not such an envelope (a bare `{"error": …}`, say).
fn tool_result_content_text(value: &Value) -> String {
    value
        .get("content")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// Everything the envelope carries, rendered as plain text.
///
/// Used only when a result has to be truncated, so the model reads prose
/// rather than a JSON document cut off mid-escape. Nothing is dropped
/// silently: the text items come first, then whatever `structuredContent`
/// still holds after [`drop_duplicated_command_output`].
fn tool_result_plain_text(value: &Value) -> String {
    let mut parts = Vec::new();
    let text = tool_result_content_text(value);
    if !text.is_empty() {
        parts.push(text);
    }
    match value.get("structuredContent") {
        Some(structured) if !structured.is_null() => {
            parts.push(
                serde_json::to_string_pretty(structured).unwrap_or_else(|_| structured.to_string()),
            );
        }
        _ => {}
    }
    if parts.is_empty() {
        serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
    } else {
        parts.join("\n")
    }
}

/// The line appended to a truncated result.
///
/// Explicit about being a partial result and about how much is missing, so
/// neither the model nor the user reading the model's answer can take a cut
/// list for a complete one.
fn truncation_marker(shown_chars: usize, total_chars: usize) -> String {
    format!(
        "\n[output truncated: {shown_chars} of {total_chars} characters shown, \
         {omitted} omitted. This is NOT the whole result — say so when you use it, \
         and run the matching `rocm` command in a terminal for the full output.]",
        omitted = total_chars.saturating_sub(shown_chars)
    )
}

/// Bound a tool result to `budget` characters of serialized JSON.
///
/// A result that already fits is returned untouched, so the common case keeps
/// its exact shape. One that does not is replaced by the plain-text rendering
/// cut on a line boundary plus [`truncation_marker`]; `budget` is measured
/// against the *serialized* value, which is what actually reaches the model,
/// so JSON escaping cannot push it back over.
fn cap_tool_result(value: Value, budget: usize) -> Value {
    if value.to_string().len() <= budget {
        return value;
    }
    // The truncated form is a fresh object, so whatever the rest of the chat
    // path reads off the envelope has to be carried over deliberately.
    // `isError` is the success/failure signal (`turn.rs` reads it), and a
    // command that failed must not read as a success merely because its
    // output was too long to keep.
    let is_error = value.get("isError").and_then(Value::as_bool);
    let full = tool_result_plain_text(&value);
    let total_chars = full.chars().count();
    // Candidate cuts: the start, then the end of every line, so a cut never
    // lands inside a line (and never inside a UTF-8 character).
    let mut cuts = vec![0usize];
    cuts.extend(full.match_indices('\n').map(|(index, _)| index + 1));
    if !full.is_empty() && !full.ends_with('\n') {
        cuts.push(full.len());
    }
    // Largest cut that still fits. `best` is only ever set to a cut that was
    // measured as fitting, so the returned value is within budget even where
    // the length is not perfectly monotonic in the cut (the marker's digit
    // count shifts by one across a power of ten).
    let (mut low, mut high, mut best) = (0usize, cuts.len() - 1, 0usize);
    while low <= high {
        let mid = low.midpoint(high);
        if truncated_tool_result(&full, cuts[mid], total_chars, is_error)
            .to_string()
            .len()
            <= budget
        {
            best = mid;
            low = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            high = mid - 1;
        }
    }
    truncated_tool_result(&full, cuts[best], total_chars, is_error)
}

/// The truncated form of `full` kept up to the byte offset `cut`. `is_error`
/// is the envelope's `isError`, carried over when it had one so truncating a
/// failed command's output does not turn it into a success.
fn truncated_tool_result(
    full: &str,
    cut: usize,
    total_chars: usize,
    is_error: Option<bool>,
) -> Value {
    let kept = &full[..cut];
    let shown_chars = kept.chars().count();
    let mut truncated = json!({
        "truncated": true,
        "tool_output": format!("{kept}{}", truncation_marker(shown_chars, total_chars)),
    });
    if let Some(is_error) = is_error {
        truncated["isError"] = json!(is_error);
    }
    truncated
}

/// Shared body for every read-only ROCm tool: forward the intent to the injected
/// executor (None ⇒ a clear "not available in this mode" message; ApprovalRequired
/// ⇒ a "needs approval" error and nothing runs — `rocm_command` lands here for
/// any argv the bin classifies as mutating, since a read tool has no approval
/// channel).
///
/// `execute()` is synchronous and can block for seconds or longer: the bin's
/// read tools run host-probe subprocesses, make HTTP readiness probes of
/// running managed services, and can fetch a package index (the SDK install
/// preview) or call a configured model endpoint (a provider-assisted plan). So
/// it runs on tokio's blocking pool, as the slash-tool path does, rather than on
/// the async worker driving the agent — where it would stall every other task
/// queued on that worker for as long as the executor takes. If the agent's
/// request timeout fires first, the blocking call still runs to completion,
/// including its incidental writes (refreshed service records, the package
/// index cache), and its result is dropped.
async fn run_rocm_read_tool(
    exec: Option<&SharedRocmToolExecutor>,
    name: &'static str,
    args: Value,
) -> Value {
    let Some(exec) = exec.cloned() else {
        return json!({ "error": "ROCm tools are unavailable in this mode (demo/replay/mock)." });
    };
    match tokio::task::spawn_blocking(move || exec.execute(name, &args)).await {
        Ok(RocmToolOutcome::Result(v)) => model_facing_tool_result(v, TOOL_RESULT_CHAR_BUDGET),
        Ok(RocmToolOutcome::Error(s)) => json!({ "error": s }),
        Ok(RocmToolOutcome::ApprovalRequired(_)) => {
            json!({ "error": "this action requires approval (interactive chat only)" })
        }
        // A panic (or cancellation at runtime shutdown): log the details rather
        // than hand a panic payload to the model, and so to a possibly remote
        // provider; the model only needs to know the tool failed.
        Err(e) => {
            tracing::warn!(tool = name, error = %e, "ROCm read tool task failed");
            json!({ "error": format!("ROCm tool `{name}` failed to run") })
        }
    }
}

/// Declare one zero-cost read-only ROCm tool type. Rig requires a `const NAME`
/// per type, so a macro generates one struct per tool to keep them DRY: each
/// holds the optional executor + the shared `fired` log and routes `call()`
/// through [`run_rocm_read_tool`]. Args are accepted as raw JSON (the bin
/// validates them), so every tool shares `type Args = serde_json::Value`.
macro_rules! rocm_read_tool {
    ($ty:ident, $name:literal, $desc:literal, $params:tt) => {
        pub struct $ty {
            pub executor: Option<SharedRocmToolExecutor>,
            pub fired: FiredLog,
        }
        impl Tool for $ty {
            const NAME: &'static str = $name;
            type Error = ToolError;
            type Args = serde_json::Value;
            type Output = Value;
            async fn definition(&self, _p: String) -> ToolDefinition {
                ToolDefinition {
                    name: $name.to_string(),
                    description: $desc.to_string(),
                    parameters: json!($params),
                }
            }
            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                record(&self.fired, $name);
                Ok(run_rocm_read_tool(self.executor.as_ref(), $name, args).await)
            }
        }
    };
}

rocm_read_tool!(
    DoctorRocmTool,
    "doctor",
    "Run the rocm-dash environment doctor: detected AMD GPU/driver, active ROCm \
     runtime status, and readiness checks. Read-only.",
    { "type": "object", "properties": {} }
);
// The same machine inspection under the name the assistant prompt uses. The
// bin has exposed it as `examine` since before the dash existed and accepts
// both names (`validate_chat_tool_call`), but the dash schema advertised only
// `doctor` — so the shared prompt's "use examine … before answering" named a
// tool the model could not see here. Registering the alias makes the one prompt
// valid against both catalogs.
rocm_read_tool!(
    ExamineRocmTool,
    "examine",
    "Alias of `doctor`: the same read-only environment check (detected AMD \
     GPU/driver, active ROCm runtime status, readiness). Read-only.",
    { "type": "object", "properties": {} }
);
rocm_read_tool!(
    EnginesRocmTool,
    "engines",
    "List the available inference engines (e.g. Lemonade, vLLM, ComfyUI) and \
     their install/availability status. Read-only.",
    { "type": "object", "properties": {} }
);
rocm_read_tool!(
    ServicesRocmTool,
    "services",
    "List managed services / serving instances and their current status. \
     Read-only.",
    { "type": "object", "properties": {} }
);
rocm_read_tool!(
    ServiceLogsRocmTool,
    "service_logs",
    "Fetch recent log lines for a managed service by id. Read-only.",
    {
        "type": "object",
        "properties": {
            "service_id": { "type": "string", "description": "Service identifier whose logs to read." }
        },
        "required": ["service_id"]
    }
);
rocm_read_tool!(
    BridgeSnapshotRocmTool,
    "bridge_snapshot",
    "Return the current job-bridge state snapshot (background jobs and their \
     status). Read-only.",
    { "type": "object", "properties": {} }
);
rocm_read_tool!(
    GpuSnapshotRocmTool,
    "gpu_snapshot",
    "Return a point-in-time hardware snapshot of the AMD GPUs as seen by the \
     machine (distinct from the live telemetry gpu_status). Read-only.",
    { "type": "object", "properties": {} }
);
rocm_read_tool!(
    AutomationsRocmTool,
    "automations",
    "List the configured background automations / scheduled checks and their \
     last status. Read-only.",
    { "type": "object", "properties": {} }
);
rocm_read_tool!(
    PathExistsRocmTool,
    "path_exists",
    "Check whether a filesystem path exists on the machine. Read-only.",
    {
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Absolute or relative path to test." }
        },
        "required": ["path"]
    }
);
rocm_read_tool!(
    PortStatusRocmTool,
    "port_status",
    "Check whether a local TCP port is open / in use. Read-only.",
    {
        "type": "object",
        "properties": {
            "port": { "type": "integer", "description": "TCP port number to probe." }
        },
        "required": ["port"]
    }
);
rocm_read_tool!(
    UpdateCheckRocmTool,
    "update_check",
    "Check for an available rocm-cli update (current vs latest version). \
     Read-only — does not install anything.",
    { "type": "object", "properties": {} }
);
rocm_read_tool!(
    InstallSdkDryRunRocmTool,
    "install_sdk_dry_run",
    "Show what installing the TheRock ROCm SDK WOULD do (the resolved wheels / \
     steps) WITHOUT installing anything. Read-only dry run.",
    {
        "type": "object",
        "properties": {
            "channel": { "type": "string", "description": "Release channel, e.g. 'release'." },
            "format": { "type": "string", "description": "Artifact format, e.g. 'wheel'." },
            "prefix": { "type": "string", "description": "Optional install prefix to evaluate." },
            "version": { "type": "string", "description": "Optional explicit version selector." },
            "build_date": { "type": "string", "description": "Optional build-date selector." }
        }
    }
);
rocm_read_tool!(
    RocmCommandRocmTool,
    "rocm_command",
    // The examples are deliberately the cheap commands. A small local model
    // copies the example in a tool description literally, and the example used
    // to be `["model"]` — the single largest read-only result there is, which
    // on a 32k-context model used to end the turn with a provider
    // context-overflow error instead of an answer.
    "Run a READ-ONLY rocm CLI subcommand and return its output. Allowed: \
     config show, runtimes (list), logs, daemon status, model/models. Pass the \
     argv as `args`, e.g. [\"config\",\"show\"] or [\"runtimes\",\"list\"]. \
     [\"model\"] lists the recommended models and [\"model\",\"--verbose\"] \
     adds the per-recipe diagnostics, which are long. Mutating subcommands are \
     rejected.",
    {
        "type": "object",
        "properties": {
            "args": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Argv for the rocm subcommand, e.g. [\"config\",\"show\"]."
            }
        },
        "required": ["args"]
    }
);
rocm_read_tool!(
    NaturalLanguagePlanRocmTool,
    "natural_language_plan",
    "Turn a natural-language ROCm request into a reviewed, structured plan \
     (does not execute).",
    {
        "type": "object",
        "properties": {
            "request": { "type": "string", "description": "The natural-language ROCm request to plan." }
        },
        "required": ["request"]
    }
);

/// All read-only ROCm tool names (mirrors [`TELEMETRY_TOOL_NAMES`]).
///
/// Used for uniqueness/registration checks and the parity map. Mutating tools
/// are intentionally absent. `natural_language_plan` is read-only: it plans but
/// never executes (Phase 7).
pub const ROCM_READ_TOOL_NAMES: [&str; 14] = [
    DoctorRocmTool::NAME,
    ExamineRocmTool::NAME,
    EnginesRocmTool::NAME,
    ServicesRocmTool::NAME,
    ServiceLogsRocmTool::NAME,
    BridgeSnapshotRocmTool::NAME,
    GpuSnapshotRocmTool::NAME,
    AutomationsRocmTool::NAME,
    PathExistsRocmTool::NAME,
    PortStatusRocmTool::NAME,
    UpdateCheckRocmTool::NAME,
    InstallSdkDryRunRocmTool::NAME,
    RocmCommandRocmTool::NAME,
    NaturalLanguagePlanRocmTool::NAME,
];

// ---------------------------------------------------------------------------
// Mutating ROCm tools (group D, Phase 4). These do NOT execute inside the rig
// tool loop. `execute()` returns `ApprovalRequired(intent)` (a descriptor that
// the bin's validators already accepted); the tool posts the intent to the app
// via `approval_tx` (a `ClientMsg::ChatApprovalRequired`) and returns a
// "surfaced for approval" note to the model. The actual action runs only after
// the operator approves the modal, via `execute_approved` off the event loop.
// The `agent` module stays the sole `rig` namer; the seam types are plain data.
// ---------------------------------------------------------------------------

/// Declare one mutating ROCm tool type. Mirrors [`rocm_read_tool!`] but its
/// `call()` surfaces the approval intent rather than executing: on
/// `ApprovalRequired` it forwards the descriptor over `approval_tx` and returns
/// a terse "surfaced" note (no execution, no retry); on `Error` it returns the
/// validator error; a `Result` (shouldn't happen for a mutating tool, handled
/// defensively) is passed through. Args are raw JSON (the bin validates them).
macro_rules! rocm_mutating_tool {
    ($ty:ident, $name:literal, $desc:literal, $params:tt) => {
        pub struct $ty {
            pub executor: Option<SharedRocmToolExecutor>,
            pub approval_tx: Option<UnboundedSender<ClientMsg>>,
            pub fired: FiredLog,
        }
        impl Tool for $ty {
            const NAME: &'static str = $name;
            type Error = ToolError;
            type Args = serde_json::Value;
            type Output = Value;
            async fn definition(&self, _p: String) -> ToolDefinition {
                ToolDefinition {
                    name: $name.to_string(),
                    description: $desc.to_string(),
                    parameters: json!($params),
                }
            }
            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                record(&self.fired, $name);
                match self.executor.as_ref() {
                    None => Ok(json!({ "error": "ROCm tools unavailable in this mode." })),
                    Some(e) => match e.execute($name, &args) {
                        RocmToolOutcome::ApprovalRequired(intent) => {
                            if let Some(tx) = self.approval_tx.as_ref() {
                                let _ = tx.send(ClientMsg::ChatApprovalRequired { intent });
                            }
                            Ok(json!({
                                "status": "surfaced_for_approval",
                                "note": "This action needs operator approval; it has been \
                                         surfaced to the operator. Do not retry.",
                            }))
                        }
                        RocmToolOutcome::Error(s) => Ok(json!({ "error": s })),
                        RocmToolOutcome::Result(v) => Ok(v),
                    },
                }
            }
        }
    };
}

rocm_mutating_tool!(
    InstallSdkRocmTool,
    "install_sdk",
    "Install the TheRock ROCm SDK. MUTATING — this is surfaced for operator \
     approval before anything runs; it does NOT install immediately. The user \
     must supply an install `prefix` (folder); ask for it first.",
    {
        "type": "object",
        "properties": {
            "channel": { "type": "string", "description": "Release channel: 'release' or 'nightly'." },
            "format": { "type": "string", "description": "Artifact format: 'wheel' or 'tarball'." },
            "prefix": { "type": "string", "description": "Install folder (required; never a system path)." },
            "version": { "type": "string", "description": "Optional explicit wheel version selector." }
        }
    }
);
rocm_mutating_tool!(
    InstallEngineRocmTool,
    "install_engine",
    "Install an inference engine (e.g. lemonade, vllm, comfyui). MUTATING — \
     surfaced for operator approval before anything runs.",
    {
        "type": "object",
        "properties": {
            "engine": { "type": "string", "description": "Engine to install, e.g. 'vllm'." },
            "runtime_id": { "type": "string", "description": "Optional ROCm runtime id to target." },
            "python_version": { "type": "string", "description": "Optional Python version for the engine env." },
            "reinstall": { "type": "boolean", "description": "Reinstall even if already present." }
        },
        "required": ["engine"]
    }
);
rocm_mutating_tool!(
    LaunchServerRocmTool,
    "launch_server",
    "Start a local managed model server. MUTATING — surfaced for operator \
     approval before anything runs. Host is loopback-only (no public bind); CPU \
     execution is rejected (ROCm GPU required).",
    {
        "type": "object",
        "properties": {
            "model": { "type": "string", "description": "Model id/name to serve." },
            "engine": { "type": "string", "description": "Optional engine, e.g. 'vllm'." },
            "host": { "type": "string", "description": "Loopback host only (e.g. 127.0.0.1)." },
            "port": { "type": "integer", "description": "TCP port. Required unless host is exactly 127.0.0.1 (localhost/::1 still need one)." },
            "device": { "type": "string", "description": "GPU device selector (CPU is rejected)." }
        },
        "required": ["model"]
    }
);
rocm_mutating_tool!(
    StopServerRocmTool,
    "stop_server",
    "Stop a running local managed model server by service id. MUTATING — \
     surfaced for operator approval before anything runs.",
    {
        "type": "object",
        "properties": {
            "service_id": { "type": "string", "description": "Managed service identifier to stop." }
        },
        "required": ["service_id"]
    }
);
rocm_mutating_tool!(
    WatcherEnableRocmTool,
    "watcher_enable",
    "Enable a background automation watcher, optionally choosing how far it may \
     act (mode: observe | propose | contained). MUTATING — surfaced for operator \
     approval before anything runs.",
    {
        "type": "object",
        "properties": {
            "watcher": { "type": "string", "description": "Watcher id to enable." },
            "mode": {
                "type": "string",
                "enum": ["observe", "propose", "contained"],
                "description": "Optional autonomy mode: observe (log only), propose (suggest), or contained (act within guardrails)."
            }
        },
        "required": ["watcher"]
    }
);
rocm_mutating_tool!(
    WatcherDisableRocmTool,
    "watcher_disable",
    "Disable a background automation watcher by id. MUTATING — surfaced for \
     operator approval before anything runs.",
    {
        "type": "object",
        "properties": {
            "watcher": { "type": "string", "description": "Watcher id to disable." }
        },
        "required": ["watcher"]
    }
);

/// All mutating ROCm tool names (mirrors [`ROCM_READ_TOOL_NAMES`]).
///
/// Used for uniqueness/registration checks and the parity map. Phase 4 ships
/// the install/engine/serve/services mutating set; Phase 6 adds the automations
/// (watcher) toggles. `proposal_action` (reviews approve/reject) is NOT a rig
/// mutating tool — it routes via the slash seam only — so it is intentionally
/// absent here. update/comfyui/uninstall/setup use the `rocm_command` tool.
pub const ROCM_MUTATING_TOOL_NAMES: [&str; 6] = [
    InstallSdkRocmTool::NAME,
    InstallEngineRocmTool::NAME,
    LaunchServerRocmTool::NAME,
    StopServerRocmTool::NAME,
    WatcherEnableRocmTool::NAME,
    WatcherDisableRocmTool::NAME,
];

/// Register every mutating ROCm tool on a Rig `AgentBuilder`, cloning the
/// optional executor + approval channel + the shared `fired` log into each.
/// Generic over the builder's model + preamble so all three backends (Rig,
/// ChatGPT, Anthropic) reuse one registration site (DRY). Called after
/// [`register_rocm_read_tools`].
pub(super) fn register_rocm_mutating_tools<M, P>(
    builder: rig::agent::AgentBuilder<M, P, rig::agent::WithBuilderTools>,
    executor: Option<&SharedRocmToolExecutor>,
    approval_tx: Option<&UnboundedSender<ClientMsg>>,
    fired: &FiredLog,
) -> rig::agent::AgentBuilder<M, P, rig::agent::WithBuilderTools>
where
    M: rig::completion::CompletionModel,
    P: rig::agent::PromptHook<M>,
{
    builder
        .tool(InstallSdkRocmTool {
            executor: executor.cloned(),
            approval_tx: approval_tx.cloned(),
            fired: fired.clone(),
        })
        .tool(InstallEngineRocmTool {
            executor: executor.cloned(),
            approval_tx: approval_tx.cloned(),
            fired: fired.clone(),
        })
        .tool(LaunchServerRocmTool {
            executor: executor.cloned(),
            approval_tx: approval_tx.cloned(),
            fired: fired.clone(),
        })
        .tool(StopServerRocmTool {
            executor: executor.cloned(),
            approval_tx: approval_tx.cloned(),
            fired: fired.clone(),
        })
        .tool(WatcherEnableRocmTool {
            executor: executor.cloned(),
            approval_tx: approval_tx.cloned(),
            fired: fired.clone(),
        })
        .tool(WatcherDisableRocmTool {
            executor: executor.cloned(),
            approval_tx: approval_tx.cloned(),
            fired: fired.clone(),
        })
}

/// Register the telemetry tools (GpuStatus, ListInstances, BenchSummary,
/// TokensPerWatt — all snapshot-backed) on a fresh Rig `AgentBuilder`, cloning
/// the snapshot + shared `fired` log into each. Kept generic over the builder's
/// completion model + preamble so all three backends (RigAgentClient,
/// ChatGptAgentClient, AnthropicAgentClient) reuse one registration site (DRY —
/// the tool list lives in exactly one place, mirroring
/// [`register_rocm_read_tools`]). Takes the base builder (`NoToolConfig`) and
/// returns it in `WithBuilderTools` because the first `.tool()` transitions the
/// type-state; the ROCm read/mutating registrations chain after it.
pub(super) fn register_telemetry_tools<M, P>(
    builder: rig::agent::AgentBuilder<M, P>,
    snap: &Arc<StateSnapshot>,
    fired: &FiredLog,
) -> rig::agent::AgentBuilder<M, P, rig::agent::WithBuilderTools>
where
    M: rig::completion::CompletionModel,
    P: rig::agent::PromptHook<M>,
{
    builder
        .tool(GpuStatusTool {
            snap: snap.clone(),
            fired: fired.clone(),
        })
        .tool(ListInstancesTool {
            snap: snap.clone(),
            fired: fired.clone(),
        })
        .tool(BenchSummaryTool {
            snap: snap.clone(),
            fired: fired.clone(),
        })
        .tool(TokensPerWattTool {
            snap: snap.clone(),
            fired: fired.clone(),
        })
}

/// Register every read-only ROCm tool on a Rig `AgentBuilder`, cloning the
/// optional executor + the shared `fired` log into each. Kept generic over the
/// builder's completion model + preamble so all three backends (the
/// OpenAI-compatible, ChatGPT, and Anthropic paths) reuse one registration
/// site (DRY — the tool list lives in exactly one place). The builder is
/// already in the `WithBuilderTools` state because the telemetry tools were
/// registered first.
pub(super) fn register_rocm_read_tools<M, P>(
    builder: rig::agent::AgentBuilder<M, P, rig::agent::WithBuilderTools>,
    executor: Option<&SharedRocmToolExecutor>,
    fired: &FiredLog,
) -> rig::agent::AgentBuilder<M, P, rig::agent::WithBuilderTools>
where
    M: rig::completion::CompletionModel,
    P: rig::agent::PromptHook<M>,
{
    builder
        .tool(DoctorRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(ExamineRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(EnginesRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(ServicesRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(ServiceLogsRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(BridgeSnapshotRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(GpuSnapshotRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(AutomationsRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(PathExistsRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(PortStatusRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(UpdateCheckRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(InstallSdkDryRunRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(RocmCommandRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
        .tool(NaturalLanguagePlanRocmTool {
            executor: executor.cloned(),
            fired: fired.clone(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::fixture_snapshot;

    #[test]
    fn telemetry_tool_names_are_unique_and_non_empty() {
        for n in TELEMETRY_TOOL_NAMES {
            assert!(!n.is_empty(), "telemetry tool name must be non-empty");
        }
        let mut sorted = TELEMETRY_TOOL_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            TELEMETRY_TOOL_NAMES.len(),
            "telemetry tool names must be unique"
        );
    }

    #[tokio::test]
    async fn gpu_status_tool_call_returns_typed_output() {
        let tool = GpuStatusTool {
            snap: Arc::new(fixture_snapshot()),
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        // ToolDefinition is valid: name matches, parameters is an object.
        let def = tool.definition(String::new()).await;
        assert_eq!(def.name, "gpu_status");
        assert!(def.parameters.is_object());
        // call() returns the expected GPU output and records that it fired.
        let out = tool
            .call(GpuStatusArgs { gpu_index: Some(2) })
            .await
            .expect("tool call ok");
        assert_eq!(out["gpu"]["temperature_c"], 71.0);
        assert_eq!(tool.fired.lock().unwrap().as_slice(), ["gpu_status"]);
    }

    #[tokio::test]
    async fn all_tools_expose_valid_definitions() {
        let snap = Arc::new(fixture_snapshot());
        let fired: FiredLog = Arc::new(Mutex::new(Vec::new()));
        let g = GpuStatusTool {
            snap: snap.clone(),
            fired: fired.clone(),
        }
        .definition(String::new())
        .await;
        let l = ListInstancesTool {
            snap: snap.clone(),
            fired: fired.clone(),
        }
        .definition(String::new())
        .await;
        let b = BenchSummaryTool {
            snap: snap.clone(),
            fired: fired.clone(),
        }
        .definition(String::new())
        .await;
        let t = TokensPerWattTool {
            snap: snap.clone(),
            fired: fired.clone(),
        }
        .definition(String::new())
        .await;
        for def in [g, l, b, t] {
            assert!(!def.name.is_empty());
            assert!(def.parameters.is_object());
        }
    }

    #[test]
    fn retired_skill_registry_tools_are_not_advertised() {
        // `list_skills` / `skill_plan` described install steps (a GitHub
        // releases download plus a `tar` extraction) that are not how the product
        // installs an engine, and no executor ever applied their plans. They
        // must not come back in any tool set the model is offered.
        for gone in ["list_skills", "skill_plan"] {
            assert!(!TELEMETRY_TOOL_NAMES.contains(&gone), "{gone} is retired");
            assert!(!ROCM_READ_TOOL_NAMES.contains(&gone), "{gone} is retired");
            assert!(
                !ROCM_MUTATING_TOOL_NAMES.contains(&gone),
                "{gone} is retired"
            );
        }
    }

    /// Minimal executor that echoes a fixed JSON value for any tool call.
    #[derive(Debug)]
    struct FakeExec(serde_json::Value);
    impl crate::tool_exec::RocmToolExecutor for FakeExec {
        fn execute(&self, _name: &str, _args: &Value) -> RocmToolOutcome {
            RocmToolOutcome::Result(self.0.clone())
        }
        fn execute_approved(&self, _name: &str, _args: &Value) -> RocmToolOutcome {
            RocmToolOutcome::Result(self.0.clone())
        }
    }

    #[tokio::test]
    async fn read_only_tool_round_trips_to_json() {
        // chat → tool → executor → JSON: the tool forwards across the seam and
        // returns the executor's payload verbatim.
        let exec: SharedRocmToolExecutor = Arc::new(FakeExec(json!({ "ok": true })));
        let tool = DoctorRocmTool {
            executor: Some(exec),
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let out = tool.call(json!({})).await.expect("tool call ok");
        assert_eq!(out, json!({ "ok": true }));
        // The fired log records that the tool ran.
        assert_eq!(
            tool.fired.lock().unwrap().as_slice(),
            &["doctor".to_string()]
        );
    }

    /// A command-backed MCP envelope of roughly `text_chars` characters,
    /// shaped exactly as the bin's `internal_mcp_tool_result_from_command`
    /// builds one: the command output in `content[0].text` and again in
    /// `structuredContent.stdout`.
    fn command_envelope(text_chars: usize) -> (String, serde_json::Value) {
        let line = "Qwen/Qwen3.5-4B  Q4_K_M  fits this GPU\n";
        let stdout = line.repeat(text_chars.div_ceil(line.len()));
        let envelope = json!({
            "content": [{ "type": "text", "text": format!("Ran `rocm` command.\n\n{stdout}") }],
            "structuredContent": {
                "argv": ["rocm", "model"],
                "exit_status": 0,
                "stdout": stdout,
                "stderr": "",
            },
            "isError": false,
        });
        (stdout, envelope)
    }

    /// The `[output truncated: A of B characters shown, C omitted` figures.
    fn truncation_figures(text: &str) -> (usize, usize, usize) {
        let marker = text
            .split_once("[output truncated: ")
            .unwrap_or_else(|| panic!("no truncation marker in:\n{text}"))
            .1;
        let numbers = marker
            .split(|c: char| !c.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .take(3)
            .map(|part| part.parse::<usize>().expect("marker figure"))
            .collect::<Vec<_>>();
        (numbers[0], numbers[1], numbers[2])
    }

    #[tokio::test]
    async fn an_oversized_read_only_tool_result_is_capped_before_the_model_sees_it() {
        // A ~200 KB result is what `rocm_command` with `["model"]` used to
        // return (204,470 characters measured on a developer host). Sent whole
        // to a locally served model it exceeds the entire context and the
        // provider refuses the request, so the user gets an HTTP 400 instead
        // of an answer.
        let (_, envelope) = command_envelope(200 * 1024);
        assert!(
            envelope.to_string().len() > 200 * 1024,
            "the fixture must be bigger than the budget or it proves nothing"
        );
        let tool = RocmCommandRocmTool {
            executor: Some(Arc::new(FakeExec(envelope)) as SharedRocmToolExecutor),
            fired: Arc::new(Mutex::new(Vec::new())),
        };

        let out = tool.call(json!({ "args": ["model"] })).await.expect("ok");

        // What the model is handed is the serialized value, so that is what
        // the budget is measured against.
        let rendered = out.to_string();
        assert!(
            rendered.len() <= TOOL_RESULT_CHAR_BUDGET,
            "a tool result of {} characters reached the model; the budget is {TOOL_RESULT_CHAR_BUDGET}",
            rendered.len()
        );
        // It says it is partial, rather than reading as a complete answer.
        assert_eq!(out["truncated"], json!(true), "{out}");
        let text = out["tool_output"].as_str().expect("tool_output text");
        assert!(
            text.contains("[output truncated:") && text.trim_end().ends_with(']'),
            "the marker must close the result so the model cannot miss it: {text}"
        );
        // …and says how much is missing, consistently.
        let (shown, total, omitted) = truncation_figures(text);
        assert_eq!(shown + omitted, total, "the marker's figures must add up");
        assert!(
            omitted > total / 2,
            "a 200 KB result cut to {TOOL_RESULT_CHAR_BUDGET} characters must report most of it omitted, got {omitted} of {total}"
        );
        // The part that is shown is real output, cut on a line boundary.
        assert!(text.contains("Qwen/Qwen3.5-4B"), "{text}");
        assert!(
            text.split_once("\n[output truncated:")
                .is_some_and(|(kept, _)| kept.is_empty() || kept.ends_with('\n')),
            "the cut must land on a line boundary: {text}"
        );
    }

    #[tokio::test]
    async fn truncating_a_failed_command_keeps_it_a_failure() {
        // The capped form is a fresh object, so `isError` survives only if it
        // is carried over. A command that failed with a lot of output would
        // otherwise reach the model looking like a success, and the chat path
        // reads that field to decide exactly that.
        let (_, mut envelope) = command_envelope(200 * 1024);
        envelope["isError"] = json!(true);
        envelope["structuredContent"]["exit_status"] = json!(1);
        let tool = RocmCommandRocmTool {
            executor: Some(Arc::new(FakeExec(envelope)) as SharedRocmToolExecutor),
            fired: Arc::new(Mutex::new(Vec::new())),
        };

        let out = tool.call(json!({ "args": ["model"] })).await.expect("ok");

        assert_eq!(
            out["truncated"],
            json!(true),
            "the fixture must truncate: {out}"
        );
        assert_eq!(
            out["isError"],
            json!(true),
            "a failed command must still read as failed after truncation: {out}"
        );
    }

    #[tokio::test]
    async fn truncating_a_successful_command_keeps_it_a_success() {
        // The mirror of the case above: carrying `isError` over must not
        // invent a failure for output that was merely long.
        let (_, envelope) = command_envelope(200 * 1024);
        let tool = RocmCommandRocmTool {
            executor: Some(Arc::new(FakeExec(envelope)) as SharedRocmToolExecutor),
            fired: Arc::new(Mutex::new(Vec::new())),
        };

        let out = tool.call(json!({ "args": ["model"] })).await.expect("ok");

        assert_eq!(
            out["truncated"],
            json!(true),
            "the fixture must truncate: {out}"
        );
        assert_eq!(out["isError"], json!(false), "{out}");
    }

    #[tokio::test]
    async fn a_command_backed_result_reaches_the_model_once_not_twice() {
        // The bin's envelope carries the command output twice — once as the
        // text a reader sees, once as `structuredContent.stdout` — because
        // `rocm mcp-call` consumers read the structured field. The dash hands
        // the whole value to the model, so without this the model pays for
        // every byte twice.
        let (stdout, envelope) = command_envelope(2 * 1024);
        let before = envelope.to_string().len();
        let tool = RocmCommandRocmTool {
            executor: Some(Arc::new(FakeExec(envelope)) as SharedRocmToolExecutor),
            fired: Arc::new(Mutex::new(Vec::new())),
        };

        let out = tool
            .call(json!({ "args": ["config", "show"] }))
            .await
            .expect("ok");

        // One copy survives, and it is the readable one.
        let text = out["content"][0]["text"].as_str().expect("content text");
        assert!(text.contains(stdout.trim()), "{out}");
        assert!(
            out["structuredContent"].get("stdout").is_none(),
            "the duplicated copy must not reach the model: {out}"
        );
        assert!(
            out["structuredContent"].get("stderr").is_none(),
            "an empty stderr is pure overhead for the model: {out}"
        );
        // The fields that are not duplicates stay, so the model can still see
        // what ran and whether it worked.
        assert_eq!(out["structuredContent"]["exit_status"], json!(0), "{out}");
        assert_eq!(out["structuredContent"]["argv"][1], json!("model"), "{out}");
        assert!(
            out.to_string().len() < before / 2 + 512,
            "dropping the duplicate should roughly halve the result: {} vs {before}",
            out.to_string().len()
        );
    }

    #[tokio::test]
    async fn a_result_within_the_budget_keeps_its_exact_shape() {
        // The cap must not reshape ordinary results: a tool whose data lives
        // only in `structuredContent` (engines, bridge_snapshot, automations)
        // would lose it if every result were flattened to text.
        let value = json!({
            "content": [{ "type": "text", "text": "Found 3 engine entries." }],
            "structuredContent": { "engines": [{ "id": "lemonade", "available": true }] },
            "isError": false,
        });
        let tool = EnginesRocmTool {
            executor: Some(Arc::new(FakeExec(value.clone())) as SharedRocmToolExecutor),
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let out = tool.call(json!({})).await.expect("ok");
        assert_eq!(
            out, value,
            "an in-budget result must pass through unchanged"
        );
    }

    /// Recording executor for the mutating-tool surfacing test: `execute`
    /// returns `ApprovalRequired`; `execute_approved` records a call (which must
    /// NOT happen during the rig tool loop).
    #[derive(Debug)]
    struct RecordingMutatingExec {
        approved: Arc<Mutex<Vec<String>>>,
    }
    impl crate::tool_exec::RocmToolExecutor for RecordingMutatingExec {
        fn execute(&self, name: &str, args: &Value) -> RocmToolOutcome {
            RocmToolOutcome::ApprovalRequired(crate::tool_exec::ApprovalIntent {
                title: "T".to_string(),
                body: vec!["cmd".to_string()],
                name: name.to_string(),
                arguments: args.clone(),
            })
        }
        fn execute_approved(&self, name: &str, _args: &Value) -> RocmToolOutcome {
            self.approved.lock().unwrap().push(name.to_string());
            RocmToolOutcome::Result(json!({ "ok": true }))
        }
    }

    #[tokio::test]
    async fn mutating_tool_surfaces_approval_not_execution() {
        // (f) a mutating rig tool's call() posts ChatApprovalRequired over the
        // approval channel and returns a "surfaced" note — it must NOT execute
        // (execute_approved is never called from the rig loop).
        let approved = Arc::new(Mutex::new(Vec::<String>::new()));
        let exec: SharedRocmToolExecutor = Arc::new(RecordingMutatingExec {
            approved: approved.clone(),
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ClientMsg>();
        let tool = InstallSdkRocmTool {
            executor: Some(exec),
            approval_tx: Some(tx),
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let out = tool
            .call(json!({ "channel": "release", "format": "wheel", "prefix": "/tmp/x" }))
            .await
            .expect("tool call ok");
        assert_eq!(out["status"], "surfaced_for_approval");
        // The intent was posted to the app (modal would open).
        match rx.try_recv().expect("approval intent posted") {
            ClientMsg::ChatApprovalRequired { intent } => {
                assert_eq!(intent.name, "install_sdk");
                assert_eq!(intent.arguments["channel"], "release");
            }
            other => panic!("expected ChatApprovalRequired, got {other:?}"),
        }
        // Crucially, nothing executed in the rig loop.
        assert!(
            approved.lock().unwrap().is_empty(),
            "mutating tool must not execute in the rig loop"
        );
    }

    #[tokio::test]
    async fn read_only_rocm_command_reports_approval_without_executing() {
        // `rocm_command` is registered as a read-only tool, so it has no
        // approval channel: when the executor classifies the argv as mutating,
        // the tool must hand the model an error and run nothing.
        let approved = Arc::new(Mutex::new(Vec::<String>::new()));
        let exec: SharedRocmToolExecutor = Arc::new(RecordingMutatingExec {
            approved: approved.clone(),
        });
        let tool = RocmCommandRocmTool {
            executor: Some(exec),
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let out = tool
            .call(json!({ "args": ["serve", "qwen", "--managed"] }))
            .await
            .expect("tool call ok");
        assert!(
            out.get("error")
                .and_then(Value::as_str)
                .is_some_and(|e| e.contains("requires approval")),
            "an approval request must reach the model as an error, got {out}"
        );
        assert!(
            approved.lock().unwrap().is_empty(),
            "a read-only tool must never execute a mutating command"
        );
    }

    #[tokio::test]
    async fn mutating_tool_none_executor_is_graceful() {
        let tool = LaunchServerRocmTool {
            executor: None,
            approval_tx: None,
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let out = tool.call(json!({ "model": "m" })).await.expect("ok");
        assert!(out.get("error").and_then(Value::as_str).is_some());
    }

    #[test]
    fn mutating_tool_names_are_complete_and_disjoint() {
        // The registry is the source of truth: every entry is non-empty and the
        // Phase 4 mutating set is present in full.
        for expected in [
            "install_sdk",
            "install_engine",
            "launch_server",
            "stop_server",
            "watcher_enable",
            "watcher_disable",
        ] {
            assert!(
                ROCM_MUTATING_TOOL_NAMES.contains(&expected),
                "missing mutating tool: {expected}"
            );
        }
        // Disjoint from read-only + telemetry tool names (iterates the registry itself).
        for n in ROCM_MUTATING_TOOL_NAMES {
            assert!(
                !TELEMETRY_TOOL_NAMES.contains(&n),
                "collision with telemetry tool: {n}"
            );
            // install_sdk_dry_run (read-only) must not clash with install_sdk.
            assert!(
                !ROCM_READ_TOOL_NAMES.contains(&n),
                "collision with read tool: {n}"
            );
        }
    }

    /// Executor whose `execute()` can only finish if the async runtime keeps
    /// making progress while it runs: it spawns a task onto the current runtime
    /// and blocks until that task has run. On an async worker thread the task
    /// is stuck behind the blocked worker, so `execute()` gives up after
    /// [`STALL_TIMEOUT`] and reports the stall instead of hanging the test.
    #[derive(Debug)]
    struct NeedsRuntimeProgressExec;
    const STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    impl crate::tool_exec::RocmToolExecutor for NeedsRuntimeProgressExec {
        fn execute(&self, _name: &str, _args: &Value) -> RocmToolOutcome {
            let (tx, rx) = std::sync::mpsc::channel();
            tokio::runtime::Handle::current().spawn(async move {
                let _ = tx.send(());
            });
            match rx.recv_timeout(STALL_TIMEOUT) {
                Ok(()) => RocmToolOutcome::Result(json!({ "ok": true })),
                Err(_) => RocmToolOutcome::Error("async runtime stalled".to_string()),
            }
        }
        fn execute_approved(&self, name: &str, args: &Value) -> RocmToolOutcome {
            self.execute(name, args)
        }
    }

    #[test]
    fn read_only_tool_call_does_not_block_the_async_runtime() {
        // The bin's executor blocks (see `run_rocm_read_tool`). The dash runs
        // the agent on a multi-thread runtime, which on a single-CPU host has
        // one worker; a `call()` that ran `execute()` inline would hold that
        // worker, and no other async task could run until the executor returned.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("build runtime");
        let tool = ExamineRocmTool {
            executor: Some(Arc::new(NeedsRuntimeProgressExec)),
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        // Spawned like the event loop spawns `agent.complete`, so the call runs
        // on the runtime's worker rather than on the `block_on` thread.
        let out = rt
            .block_on(async move { tokio::spawn(async move { tool.call(json!({})).await }).await })
            .expect("tool task joins")
            .expect("tool call ok");
        assert_eq!(
            out,
            json!({ "ok": true }),
            "the executor ran on an async worker and starved the runtime"
        );
    }

    /// Executor whose `execute()` panics, standing in for a bug in the bin.
    #[derive(Debug)]
    struct PanickingExec;
    impl crate::tool_exec::RocmToolExecutor for PanickingExec {
        fn execute(&self, _name: &str, _args: &Value) -> RocmToolOutcome {
            panic!("executor bug");
        }
        fn execute_approved(&self, name: &str, args: &Value) -> RocmToolOutcome {
            self.execute(name, args)
        }
    }

    /// In-memory `tracing` writer, so a test can read back what was logged.
    #[derive(Clone, Default)]
    struct LogCapture(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl tracing_subscriber::fmt::MakeWriter<'_> for LogCapture {
        type Writer = Self;
        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    #[tokio::test]
    async fn read_only_tool_reports_a_panicking_executor_as_a_tool_error() {
        // A panic inside `execute()` must come back to the model as a tool
        // error. Unwinding instead would end the spawned agent task without a
        // reply, leaving the chat waiting on a turn that never finishes.
        let tool = ExamineRocmTool {
            executor: Some(Arc::new(PanickingExec)),
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        // `#[tokio::test]` runs on one thread, so the thread-local subscriber
        // sees the warning logged after the blocking task is joined.
        let logs = LogCapture::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logs.clone())
            .with_ansi(false)
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        let out = tool.call(json!({})).await.expect("tool call ok");
        drop(guard);
        let logged = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        assert!(
            logged.contains("WARN")
                && logged.contains("examine")
                && logged.contains("executor bug"),
            "the panic must be logged with the tool name and payload, got: {logged}"
        );
        let error = out.get("error").and_then(Value::as_str).unwrap_or_default();
        assert!(
            error.contains("`examine` failed to run"),
            "expected a tool error naming the tool, got {out}"
        );
        assert!(
            !error.contains("executor bug"),
            "the panic payload must not be sent to the model, got {out}"
        );
    }

    #[tokio::test]
    async fn read_only_tool_none_executor_is_graceful() {
        // No seam (demo/replay/mock): a clear error object, never a panic.
        let tool = EnginesRocmTool {
            executor: None,
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let out = tool.call(json!({})).await.expect("tool call ok");
        assert!(out.get("error").and_then(Value::as_str).is_some());
    }

    /// Executor whose `execute`/`execute_approved` both return a seam-level
    /// `Error`, exercising the recoverable error path (not None, not Approval).
    #[derive(Debug)]
    struct FakeErrorExec;
    impl crate::tool_exec::RocmToolExecutor for FakeErrorExec {
        fn execute(&self, _name: &str, _args: &Value) -> RocmToolOutcome {
            RocmToolOutcome::Error("boom".to_string())
        }
        fn execute_approved(&self, _name: &str, _args: &Value) -> RocmToolOutcome {
            RocmToolOutcome::Error("boom".to_string())
        }
    }

    #[tokio::test]
    async fn read_only_tool_seam_error_is_recoverable() {
        // Edge: the injected executor returns RocmToolOutcome::Error("boom").
        // call() must return a Value carrying an `error` key (recoverable),
        // never panic — the model can read and recover from it.
        let exec: SharedRocmToolExecutor = Arc::new(FakeErrorExec);
        let tool = DoctorRocmTool {
            executor: Some(exec),
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let out = tool.call(json!({})).await.expect("tool call ok (no panic)");
        assert_eq!(out.get("error").and_then(Value::as_str), Some("boom"));
    }

    #[tokio::test]
    async fn mutating_tool_seam_error_is_recoverable() {
        // Edge: a mutating tool whose executor returns Error("boom") on the
        // validate step (e.g. bad args) returns the error as a recoverable
        // Value — no approval surfaced, no panic.
        let exec: SharedRocmToolExecutor = Arc::new(FakeErrorExec);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ClientMsg>();
        let tool = LaunchServerRocmTool {
            executor: Some(exec),
            approval_tx: Some(tx),
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let out = tool
            .call(json!({ "model": "m" }))
            .await
            .expect("tool call ok (no panic)");
        assert_eq!(out.get("error").and_then(Value::as_str), Some("boom"));
        // No approval intent is surfaced on the error path.
        assert!(rx.try_recv().is_err(), "error path surfaces no approval");
    }

    #[test]
    fn rocm_read_tool_names_are_complete_and_disjoint_from_telemetry_tools() {
        // Every expected read-only tool is registered…
        for expected in [
            "doctor",
            // The prompt's name for the same check; both must be advertised or
            // the shared assistant prompt names a tool the dash never offers.
            "examine",
            "engines",
            "services",
            "service_logs",
            "bridge_snapshot",
            "gpu_snapshot",
            "automations",
            "path_exists",
            "port_status",
            "update_check",
            "install_sdk_dry_run",
            "rocm_command",
            "natural_language_plan",
        ] {
            assert!(
                ROCM_READ_TOOL_NAMES.contains(&expected),
                "missing read tool: {expected}"
            );
        }
        // natural_language_plan is read-only (plans, never executes) — Phase 7.
        assert!(ROCM_READ_TOOL_NAMES.contains(&"natural_language_plan"));
        // The telemetry tools are still present and disjoint from the new set.
        assert!(TELEMETRY_TOOL_NAMES.contains(&"gpu_status"));
        for n in ROCM_READ_TOOL_NAMES {
            assert!(!TELEMETRY_TOOL_NAMES.contains(&n), "name collision: {n}");
        }
    }
}
