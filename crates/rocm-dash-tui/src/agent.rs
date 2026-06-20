// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! The chat agent backend, built on **Rig**, plus the read-only "Skills"
//! (Rig Tools) the agent calls over cached telemetry.
//!
//! THE ONLY FILE THAT NAMES `rig` TYPES. Everything else talks to the
//! [`AgentClient`] trait, so the Rig dependency is a single swappable seam:
//! tests and the offline demo use [`MockAgentClient`]; the live path uses
//! [`RigAgentClient`] against an OpenAI-compatible endpoint.
//!
//! Rig API verified against `rig-core = "=0.38.1"` (Context7 `/websites/rig_rs`
//! and vendored source). The local-endpoint seam is the Chat Completions API
//! (not the default Responses API), reached via `CompletionsClient`. Tool
//! calling uses `agent.prompt(text).max_turns(N).with_history(history)` so the
//! model can call read-only tools and then answer — one final reply to the UI.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use rig::completion::ToolDefinition;
use rig::tool::Tool;

use rocm_dash_core::bench_schema::BenchmarkRow;
use rocm_dash_core::metrics::{GpuMetrics, Instance, Snapshot};

use crate::app::{ChatRole, ChatTurn};
use crate::client::ClientMsg;
use crate::llm::LlmConfig;
use crate::tool_exec::{RocmToolOutcome, SharedRocmToolExecutor};

use tokio::sync::mpsc::UnboundedSender;

/// One-shot request budget. A hung backend becomes a timeout error turn, never
/// a frozen pane.
pub const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(45);

/// Max tool-calling turns the model may take before producing a final answer.
const MAX_TOOL_TURNS: usize = 5;

/// Default system preamble for the dashboard assistant.
const DEFAULT_PREAMBLE: &str = "You are the rocm-dash assistant, embedded in a terminal dashboard for AMD \
     Instinct GPU telemetry and benchmarks. Use the provided tools (gpu_status, \
     list_instances, bench_summary, tokens_per_watt) to answer questions about \
     live GPU, serving instance, and benchmark state. Prefer short, direct answers.";

/// Errors from a chat completion. Public form is string-only so no `rig` type
/// leaks past this file. Messages never include the api_key (it is a header,
/// never part of base_url or the request path).
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("no message to send")]
    Empty,
    #[error("failed to build chat client: {0}")]
    Build(String),
    #[error("chat sign-in failed: {0}")]
    Auth(String),
    #[error("request timed out after {}s", REQUEST_TIMEOUT.as_secs())]
    Timeout,
    #[error("chat request failed: {0}")]
    Request(String),
}

/// A plain, cloneable read-only view of the cached telemetry the tools read.
/// Captured at spawn time so tools never touch the pure reducer or `&AppState`.
#[derive(Debug, Clone, Default)]
pub struct StateSnapshot {
    pub latest: Option<Snapshot>,
    pub instances: Vec<Instance>,
    pub bench_rows: Vec<BenchmarkRow>,
}

/// The swappable chat backend seam.
#[async_trait]
pub trait AgentClient: Send + Sync {
    /// Complete a conversation. `history` ends with the current user turn;
    /// `snapshot` is the read-only telemetry view the tools may query.
    async fn complete(
        &self,
        history: &[ChatTurn],
        snapshot: StateSnapshot,
    ) -> Result<String, AgentError>;
}

/// Map our TUI-local turns to Rig messages, preserving role + order. `Error`
/// turns are UI-local annotations and are never sent to the model. Pure: no I/O.
pub fn build_messages(turns: &[ChatTurn]) -> Vec<rig::completion::Message> {
    turns
        .iter()
        .filter_map(|t| match t.role {
            ChatRole::User => Some(rig::completion::Message::user(t.content.clone())),
            ChatRole::Agent => Some(rig::completion::Message::assistant(t.content.clone())),
            ChatRole::Error => None,
        })
        .collect()
}

/// Append a "via: tool, tool" annotation so the operator can see which Skills
/// fired. Deduplicates, preserving first-seen order. Pure.
pub fn annotate_reply(reply: String, skills: &[String]) -> String {
    if skills.is_empty() {
        return reply;
    }
    let mut seen: Vec<String> = Vec::new();
    for s in skills {
        if !seen.contains(s) {
            seen.push(s.clone());
        }
    }
    format!("{reply}\n⚙ via: {}", seen.join(", "))
}

// ---------------------------------------------------------------------------
// Pure tool computations over the snapshot (testable without Rig / async).
// ---------------------------------------------------------------------------

fn gpus_of(snap: &StateSnapshot) -> &[GpuMetrics] {
    snap.latest.as_ref().map_or(&[], |s| s.gpus.as_slice())
}

fn gpu_json(g: &GpuMetrics) -> Value {
    json!({
        "device_id": g.device_id,
        "gpu_utilization_pct": g.gpu_utilization_pct,
        "temperature_c": g.temperature_c,
        "power_w": g.power_w,
        "vram_used_mb": g.vram_used_mb,
        "vram_total_mb": g.vram_total_mb,
    })
}

/// Per-GPU util/temp/power/VRAM from the latest snapshot. `gpu_index` selects
/// one GPU; `None` returns all.
pub fn gpu_status_json(snap: &StateSnapshot, gpu_index: Option<usize>) -> Value {
    let gpus = gpus_of(snap);
    match gpu_index {
        Some(i) => match gpus.get(i) {
            Some(g) => json!({ "gpu_index": i, "gpu": gpu_json(g) }),
            None => json!({ "error": format!("no GPU at index {i}"), "gpu_count": gpus.len() }),
        },
        None => json!({ "gpus": gpus.iter().map(gpu_json).collect::<Vec<_>>() }),
    }
}

/// Discovered serving instances: name, model, status, KV-cache %, req counts.
pub fn list_instances_json(snap: &StateSnapshot) -> Value {
    let arr: Vec<Value> = snap
        .instances
        .iter()
        .map(|i| {
            json!({
                "name": i.container_name,
                "model": i.model_name,
                "status": format!("{:?}", i.status),
                "kv_cache_usage_pct": i.kv_cache_usage_pct,
                "running_reqs": i.running_reqs,
                "waiting_reqs": i.waiting_reqs,
            })
        })
        .collect();
    json!({ "instances": arr, "instance_count": arr.len() })
}

/// Per-instance tokens-per-watt: gen tok/s ÷ summed power of its GPUs. Reuses
/// the core efficiency derivation so it matches the reducer exactly.
pub fn tokens_per_watt_json(snap: &StateSnapshot) -> Value {
    let gpus = gpus_of(snap);
    let arr: Vec<Value> = snap
        .instances
        .iter()
        .map(|i| {
            let tpw = rocm_dash_core::efficiency::tokens_per_watt(i.gen_tps, &i.gpu_ids, gpus);
            json!({
                "name": i.container_name,
                "gen_tps": i.gen_tps,
                "tokens_per_watt": tpw,
            })
        })
        .collect();
    json!({ "instances": arr })
}

/// Pass^N / Pass@N rollup over the cached bench rows, reusing the core rollup.
pub fn bench_summary_json(snap: &StateSnapshot) -> Value {
    let rollups = rocm_dash_core::bench_rollup::rollup_pass_n(snap.bench_rows.iter());
    let arr: Vec<Value> = rollups
        .iter()
        .map(|r| {
            json!({
                "cell": r.cell,
                "model": r.model,
                "n_trials": r.n_trials,
                "n_passed": r.n_passed,
                "pass_n_of_n": r.pass_n_of_n,
                "pass_at_n": r.pass_at_n,
            })
        })
        .collect();
    json!({ "groups": arr, "group_count": rollups.len() })
}

// ---------------------------------------------------------------------------
// Rig Tool ("Skill") wrappers. Each holds an `Arc<StateSnapshot>` (read-only)
// and a shared `fired` log so the reply can cite which Skills ran. `call` only
// reads the snapshot — no mutation, no network, no file I/O.
// ---------------------------------------------------------------------------

type FiredLog = Arc<Mutex<Vec<String>>>;

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
                          KV-cache usage %, and running/waiting request counts."
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
                          the summed power (W) of the GPUs each instance occupies."
                .to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        record(&self.fired, Self::NAME);
        Ok(tokens_per_watt_json(&self.snap))
    }
}

/// Read-only tool exposing the rocm-dash **skills** registry (auto-config / auto-install).
///
/// The agent can list skills and fetch a skill's dry-run plan;
/// it never executes a skill (execution is `--apply`-gated in the CLI).
pub struct ListSkillsTool {
    pub fired: FiredLog,
}

impl Tool for ListSkillsTool {
    const NAME: &'static str = "list_skills";
    type Error = ToolError;
    type Args = NoArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List the available rocm-dash skills (auto-config / \
                          auto-install helpers like install-lemonade and \
                          auto-config-endpoint) the user can run."
                .to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        record(&self.fired, Self::NAME);
        let skills: Vec<Value> = crate::skills::builtin_skills()
            .iter()
            .map(|s| json!({ "name": s.name, "description": s.description }))
            .collect();
        Ok(json!({ "skills": skills, "skill_count": skills.len() }))
    }
}

/// Read-only tool returning a skill's ordered dry-run plan (no execution).
pub struct SkillPlanTool {
    pub fired: FiredLog,
}

#[derive(Debug, Deserialize, Default)]
pub struct SkillPlanArgs {
    pub name: String,
}

impl Tool for SkillPlanTool {
    const NAME: &'static str = "skill_plan";
    type Error = ToolError;
    type Args = SkillPlanArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Show the ordered dry-run step plan for a named skill \
                          WITHOUT executing it. Use after list_skills."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Skill name, e.g. install-lemonade." }
                },
                "required": ["name"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        record(&self.fired, Self::NAME);
        match crate::skills::builtin_skill(&args.name) {
            Some(m) => Ok(json!({ "name": m.name, "plan": crate::skills::build_plan(&m) })),
            None => Ok(json!({ "error": format!("unknown skill: {}", args.name) })),
        }
    }
}

/// All Skill names, for definition/uniqueness checks and docs.
pub const SKILL_NAMES: [&str; 6] = [
    GpuStatusTool::NAME,
    ListInstancesTool::NAME,
    BenchSummaryTool::NAME,
    TokensPerWattTool::NAME,
    ListSkillsTool::NAME,
    SkillPlanTool::NAME,
];

// ---------------------------------------------------------------------------
// Read-only ROCm machine-inspection tools (group B). These forward the model's
// tool-call intent across the rocm-core-free [`crate::tool_exec`] seam to the
// bin-supplied executor (live dash only). They are READ-ONLY: no mutating tool
// is registered here (mutating tools + the approval UI land in Phase 4). The
// boundary type (`SharedRocmToolExecutor`) is plain data — importing it does NOT
// pull `rocm-core` into this crate; agent.rs stays the sole `rig` namer.
// ---------------------------------------------------------------------------

/// Shared body for every read-only ROCm tool: forward the intent to the injected
/// executor (None ⇒ a clear "not available in this mode" message; ApprovalRequired
/// ⇒ a "needs approval" note — read-only tools shouldn't hit it, handled defensively).
fn run_rocm_read_tool(exec: Option<&SharedRocmToolExecutor>, name: &str, args: &Value) -> Value {
    match exec {
        None => json!({ "error": "ROCm tools are unavailable in this mode (demo/replay/mock)." }),
        Some(e) => match e.execute(name, args) {
            RocmToolOutcome::Result(v) => v,
            RocmToolOutcome::Error(s) => json!({ "error": s }),
            RocmToolOutcome::ApprovalRequired(_) => {
                json!({ "error": "this action requires approval (interactive chat only)" })
            }
        },
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
                Ok(run_rocm_read_tool(self.executor.as_ref(), $name, &args))
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
rocm_read_tool!(
    EnginesRocmTool,
    "engines",
    "List the available inference engines (e.g. vLLM, llama.cpp, ComfyUI) and \
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
    "Run a READ-ONLY rocm CLI subcommand and return its output. Allowed: \
     model/models, config show, runtimes (list), logs, daemon status. Pass the \
     argv as `args`, e.g. [\"model\"] or [\"config\",\"show\"]. Mutating \
     subcommands are rejected.",
    {
        "type": "object",
        "properties": {
            "args": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Argv for the rocm subcommand, e.g. [\"model\"]."
            }
        },
        "required": ["args"]
    }
);

/// All read-only ROCm tool names (mirrors [`SKILL_NAMES`]). Used for
/// uniqueness/registration checks and the parity map. Mutating tools and
/// `natural_language_plan` are intentionally absent (later phases).
pub const ROCM_READ_TOOL_NAMES: [&str; 12] = [
    DoctorRocmTool::NAME,
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
];

// ---------------------------------------------------------------------------
// Mutating ROCm tools (group D, Phase 4). These do NOT execute inside the rig
// tool loop. `execute()` returns `ApprovalRequired(intent)` (a descriptor that
// the bin's validators already accepted); the tool posts the intent to the app
// via `approval_tx` (a `ClientMsg::ChatApprovalRequired`) and returns a
// "surfaced for approval" note to the model. The actual action runs only after
// the operator approves the modal, via `execute_approved` off the event loop.
// agent.rs stays the sole `rig` namer; the seam types are plain data.
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
    "Install an inference engine (e.g. vllm, llama-cpp, comfyui). MUTATING — \
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
            "port": { "type": "integer", "description": "Optional TCP port." },
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

/// All mutating ROCm tool names (mirrors [`ROCM_READ_TOOL_NAMES`]).
///
/// Used for uniqueness/registration checks and the parity map. Phase 4 ships
/// exactly the install/engine/serve/services mutating set;
/// update/comfyui/uninstall/setup and automations toggles are later phases.
pub const ROCM_MUTATING_TOOL_NAMES: [&str; 4] = [
    InstallSdkRocmTool::NAME,
    InstallEngineRocmTool::NAME,
    LaunchServerRocmTool::NAME,
    StopServerRocmTool::NAME,
];

/// Register every mutating ROCm tool on a Rig `AgentBuilder`, cloning the
/// optional executor + approval channel + the shared `fired` log into each.
/// Generic over the builder's model + preamble so both client paths reuse one
/// registration site (DRY). Called after [`register_rocm_read_tools`].
fn register_rocm_mutating_tools<M, P>(
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
}

/// Live Rig-backed client for an OpenAI-compatible endpoint. The Rig client is
/// constructed once; the agent + tools are rebuilt per request from the
/// captured snapshot.
pub struct RigAgentClient {
    client: rig::providers::openai::CompletionsClient,
    model: String,
    preamble: String,
    /// Bin-injected tool executor (None for tests / no live seam).
    executor: Option<SharedRocmToolExecutor>,
    /// Channel to surface mutating-tool approval intents to the app (None for
    /// tests / no live seam). Mutating tools post here instead of executing.
    approval_tx: Option<UnboundedSender<ClientMsg>>,
}

impl RigAgentClient {
    pub fn new(
        cfg: LlmConfig,
        executor: Option<SharedRocmToolExecutor>,
        approval_tx: Option<UnboundedSender<ClientMsg>>,
    ) -> Result<Self, AgentError> {
        // Custom-auth gateway (e.g. Azure APIM `Ocp-Apim-Subscription-Key`):
        // the key goes in a custom header, NOT `Authorization: Bearer`. Rig
        // still requires an api_key, so pass a dummy Bearer the gateway ignores.
        let custom_headers = match (cfg.auth_header.as_deref(), cfg.api_key.as_deref()) {
            (Some(name), Some(key)) => Some(auth_header_map(name, key)?),
            _ => None,
        };
        // Bearer carries the real key ONLY in the standard (no custom header)
        // case; otherwise a dummy (local endpoints / custom-header gateways).
        let bearer = match (&custom_headers, cfg.api_key.as_deref()) {
            (None, Some(key)) => key.to_string(),
            _ => "sk-no-key".to_string(),
        };

        // `.api_key()` sets the builder typestate, so it must be in the chain;
        // `.base_url()` / `.http_headers()` return Self and can follow.
        let mut builder = rig::providers::openai::CompletionsClient::builder()
            .api_key(&bearer)
            .base_url(&cfg.base_url);
        if let Some(headers) = custom_headers {
            builder = builder.http_headers(headers);
        }

        let client = builder
            .build()
            .map_err(|e| AgentError::Build(e.to_string()))?;
        Ok(Self {
            client,
            model: cfg.model,
            preamble: DEFAULT_PREAMBLE.to_string(),
            executor,
            approval_tx,
        })
    }
}

/// Build a single-entry `HeaderMap` carrying the gateway's custom auth header.
/// The value is marked sensitive so the HTTP stack won't log it; errors never
/// embed the key value.
fn auth_header_map(name: &str, value: &str) -> Result<http::HeaderMap, AgentError> {
    let header_name = http::HeaderName::from_bytes(name.as_bytes())
        .map_err(|e| AgentError::Build(format!("invalid chat_auth_header name: {e}")))?;
    let mut header_value = http::HeaderValue::from_str(value)
        .map_err(|_| AgentError::Build("invalid auth header value".to_string()))?;
    header_value.set_sensitive(true);
    let mut map = http::HeaderMap::new();
    map.insert(header_name, header_value);
    Ok(map)
}

#[async_trait]
impl AgentClient for RigAgentClient {
    async fn complete(
        &self,
        history: &[ChatTurn],
        snapshot: StateSnapshot,
    ) -> Result<String, AgentError> {
        use rig::client::CompletionClient;
        use rig::completion::Prompt;
        use std::future::IntoFuture;

        let Some((last, prior)) = history.split_last() else {
            return Err(AgentError::Empty);
        };
        let snap = Arc::new(snapshot);
        let fired: FiredLog = Arc::new(Mutex::new(Vec::new()));

        let agent = self
            .client
            .agent(&self.model)
            .preamble(&self.preamble)
            .max_tokens(1024)
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
            // Skills registry tools (read-only: list + dry-run plan; never execute).
            .tool(ListSkillsTool {
                fired: fired.clone(),
            })
            .tool(SkillPlanTool {
                fired: fired.clone(),
            });
        // Read-only ROCm machine-inspection tools (forward across the seam).
        let agent = register_rocm_read_tools(agent, self.executor.as_ref(), &fired);
        // Mutating ROCm tools (surface approval; never execute in the rig loop).
        let agent = register_rocm_mutating_tools(
            agent,
            self.executor.as_ref(),
            self.approval_tx.as_ref(),
            &fired,
        )
        .build();

        let req = agent
            .prompt(last.content.clone())
            .max_turns(MAX_TOOL_TURNS)
            .with_history(build_messages(prior));

        let reply = match tokio::time::timeout(REQUEST_TIMEOUT, req.into_future()).await {
            Err(_) => return Err(AgentError::Timeout),
            Ok(Ok(reply)) => reply,
            Ok(Err(e)) => return Err(AgentError::Request(e.to_string())),
        };

        let skills = fired.lock().map(|g| g.clone()).unwrap_or_default();
        Ok(annotate_reply(reply, &skills))
    }
}

/// Register every read-only ROCm tool on a Rig `AgentBuilder`, cloning the
/// optional executor + the shared `fired` log into each. Kept generic over the
/// builder's completion model + preamble so both the OpenAI-compatible and
/// ChatGPT paths reuse one registration site (DRY — the tool list lives in
/// exactly one place). The builder is already in the `WithBuilderTools` state
/// because the telemetry/skill tools were registered first.
fn register_rocm_read_tools<M, P>(
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
}

/// No-key ChatGPT backend over Rig's native `chatgpt` OAuth provider.
///
/// This is the no-key default that restores the ChatGPT device-login the vendored Codex
/// path provided. It takes NO api_key (the env-only key invariant is untouched:
/// this path authenticates with an OAuth device-code flow, not a key). The
/// `on_device_code` callback surfaces the verification URL + user code so the
/// chat tab can show the operator how to sign in; the resulting token is
/// persisted by the provider so re-launches don't re-prompt.
pub struct ChatGptAgentClient {
    client: rig::providers::chatgpt::Client,
    model: String,
    preamble: String,
    /// Bin-injected tool executor (None for tests / no live seam).
    executor: Option<SharedRocmToolExecutor>,
    /// Channel to surface mutating-tool approval intents to the app (None for
    /// tests / no live seam).
    approval_tx: Option<UnboundedSender<ClientMsg>>,
}

impl ChatGptAgentClient {
    /// Build the OAuth client. `model` defaults to the provider's Codex model.
    /// `on_device_code(verification_uri, user_code)` is invoked during the first
    /// `authorize()` (device-code flow). No network I/O happens here — login is
    /// deferred to the first `complete()`.
    pub fn new<F>(
        model: Option<String>,
        on_device_code: F,
        executor: Option<SharedRocmToolExecutor>,
        approval_tx: Option<UnboundedSender<ClientMsg>>,
    ) -> Result<Self, AgentError>
    where
        F: Fn(String, String) + Send + Sync + 'static,
    {
        use rig::providers::chatgpt;
        // Store OAuth tokens in a user-owned directory so the plaintext
        // access/refresh tokens are not readable by other local users.
        // Refuse to fall back to a predictable temp path: a pre-existing
        // directory there could be owned by another user.
        let token_dir = std::env::var_os("HOME")
            .filter(|v| !v.is_empty())
            .or_else(|| std::env::var_os("USERPROFILE").filter(|v| !v.is_empty())) // Windows
            .map(|h| {
                std::path::PathBuf::from(h)
                    .join(".rocm")
                    .join("data")
                    .join("chatgpt-tokens")
            })
            .ok_or_else(|| {
                AgentError::Build(
                    "home directory not found ($HOME/$USERPROFILE unset); \
                     cannot safely store OAuth tokens"
                        .to_string(),
                )
            })?;
        #[cfg(unix)]
        {
            // Create with mode 0o700 up-front so the directory is never
            // world-accessible even for the brief window before set_permissions.
            // The subsequent set_permissions call tightens pre-existing dirs.
            use std::os::unix::fs::DirBuilderExt;
            use std::os::unix::fs::PermissionsExt;
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(&token_dir)
                .map_err(|e| AgentError::Build(format!("cannot create token cache dir: {e}")))?;
            if let Err(e) =
                std::fs::set_permissions(&token_dir, std::fs::Permissions::from_mode(0o700))
            {
                tracing::warn!(dir = %token_dir.display(), error = %e, "could not restrict token cache dir permissions");
            }
        }
        #[cfg(not(unix))]
        std::fs::create_dir_all(&token_dir)
            .map_err(|e| AgentError::Build(format!("cannot create token cache dir: {e}")))?;
        // The closure param is the provider's `DeviceCodePrompt` (its `auth`
        // module is private, so we let inference name it); its `verification_uri`
        // and `user_code` fields are public.
        let client = chatgpt::Client::builder()
            .oauth()
            .token_dir(token_dir)
            .on_device_code(move |p| on_device_code(p.verification_uri, p.user_code))
            .build()
            .map_err(|e| AgentError::Build(e.to_string()))?;
        Ok(Self {
            client,
            model: model.unwrap_or_else(|| chatgpt::GPT_5_3_CODEX.to_string()),
            preamble: DEFAULT_PREAMBLE.to_string(),
            executor,
            approval_tx,
        })
    }
}

#[async_trait]
impl AgentClient for ChatGptAgentClient {
    async fn complete(
        &self,
        history: &[ChatTurn],
        snapshot: StateSnapshot,
    ) -> Result<String, AgentError> {
        use rig::agent::AgentBuilder;
        use rig::completion::Prompt;
        use rig::providers::chatgpt::ResponsesCompletionModel;
        use std::future::IntoFuture;

        let Some((last, prior)) = history.split_last() else {
            return Err(AgentError::Empty);
        };

        // Device-code login on first use; the provider caches the token after.
        // A failed/declined sign-in is an Auth error (distinct from a build
        // failure), surfaced as a clear error turn — never leaks the token.
        self.client
            .authorize()
            .await
            .map_err(|e| AgentError::Auth(e.to_string()))?;

        let snap = Arc::new(snapshot);
        let fired: FiredLog = Arc::new(Mutex::new(Vec::new()));
        let model = ResponsesCompletionModel::new(self.client.clone(), self.model.clone());
        let agent = AgentBuilder::new(model)
            .preamble(&self.preamble)
            .max_tokens(1024)
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
            // Read-only Skills registry tools (list + dry-run plan; never execute).
            .tool(ListSkillsTool {
                fired: fired.clone(),
            })
            .tool(SkillPlanTool {
                fired: fired.clone(),
            });
        // Read-only ROCm machine-inspection tools (forward across the seam).
        let agent = register_rocm_read_tools(agent, self.executor.as_ref(), &fired);
        // Mutating ROCm tools (surface approval; never execute in the rig loop).
        let agent = register_rocm_mutating_tools(
            agent,
            self.executor.as_ref(),
            self.approval_tx.as_ref(),
            &fired,
        )
        .build();

        let req = agent
            .prompt(last.content.clone())
            .max_turns(MAX_TOOL_TURNS)
            .with_history(build_messages(prior));

        let reply = match tokio::time::timeout(REQUEST_TIMEOUT, req.into_future()).await {
            Err(_) => return Err(AgentError::Timeout),
            Ok(Ok(reply)) => reply,
            Ok(Err(e)) => return Err(AgentError::Request(e.to_string())),
        };

        let skills = fired.lock().map(|g| g.clone()).unwrap_or_default();
        Ok(annotate_reply(reply, &skills))
    }
}

/// Deterministic in-memory client for tests and the offline demo. Never touches
/// the network. Can emit a canned tool-calling-style answer (cites a Skill).
pub struct MockAgentClient {
    reply: String,
    fail: bool,
    cited: Vec<String>,
}

impl MockAgentClient {
    /// A mock that returns a fixed canned reply.
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
            fail: false,
            cited: Vec::new(),
        }
    }

    /// A mock that returns a canned reply annotated as if `tool_name` fired —
    /// drives the offline tool-calling demo deterministically.
    pub fn with_tool_call(reply: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
            fail: false,
            cited: vec![tool_name.into()],
        }
    }

    /// A mock whose `complete` always fails (to exercise the error path).
    pub const fn failing() -> Self {
        Self {
            reply: String::new(),
            fail: true,
            cited: Vec::new(),
        }
    }
}

#[async_trait]
impl AgentClient for MockAgentClient {
    async fn complete(
        &self,
        history: &[ChatTurn],
        _snapshot: StateSnapshot,
    ) -> Result<String, AgentError> {
        if self.fail {
            return Err(AgentError::Request("mock failure".to_string()));
        }
        if history.is_empty() {
            return Err(AgentError::Empty);
        }
        Ok(annotate_reply(self.reply.clone(), &self.cited))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_dash_core::bench_schema::{BenchmarkRow, PassFail};
    use rocm_dash_core::metrics::{GpuMetrics, Instance, Snapshot};

    fn fixture_snapshot() -> StateSnapshot {
        let snap = Snapshot {
            gpus: vec![
                GpuMetrics {
                    device_id: "gpu-0".into(),
                    gpu_utilization_pct: 12.0,
                    temperature_c: 40.0,
                    power_w: 100.0,
                    vram_used_mb: 1000,
                    vram_total_mb: 192_000,
                    ..Default::default()
                },
                GpuMetrics {
                    device_id: "gpu-1".into(),
                    gpu_utilization_pct: 55.0,
                    temperature_c: 60.0,
                    power_w: 200.0,
                    vram_used_mb: 50000,
                    vram_total_mb: 192_000,
                    ..Default::default()
                },
                GpuMetrics {
                    device_id: "gpu-2".into(),
                    gpu_utilization_pct: 87.0,
                    temperature_c: 71.0,
                    power_w: 250.0,
                    vram_used_mb: 90000,
                    vram_total_mb: 192_000,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let inst = Instance {
            container_name: "vllm-a".into(),
            model_name: "deepseek-r1".into(),
            gpu_ids: vec!["2".into()],
            kv_cache_usage_pct: Some(42.0),
            running_reqs: Some(3),
            waiting_reqs: Some(1),
            gen_tps: Some(500.0),
            ..Default::default()
        };
        let row = BenchmarkRow {
            cell: "c1".into(),
            model: Some("deepseek-r1".into()),
            pass_fail: PassFail::Pass,
            ..Default::default()
        };
        StateSnapshot {
            latest: Some(snap),
            instances: vec![inst],
            bench_rows: vec![row],
        }
    }

    #[test]
    fn build_messages_preserves_role_and_order_and_drops_errors() {
        let turns = vec![
            ChatTurn::user("first user"),
            ChatTurn::agent("first agent"),
            ChatTurn::error("local error annotation"),
            ChatTurn::user("second user"),
        ];
        let msgs = build_messages(&turns);
        assert_eq!(msgs.len(), 3);
        let dbg = format!("{msgs:?}");
        let i_first_user = dbg.find("first user").expect("first user present");
        let i_first_agent = dbg.find("first agent").expect("first agent present");
        let i_second_user = dbg.find("second user").expect("second user present");
        assert!(i_first_user < i_first_agent);
        assert!(i_first_agent < i_second_user);
        assert!(!dbg.contains("local error annotation"));
    }

    #[test]
    fn gpu_status_json_returns_known_gpu_metrics() {
        let snap = fixture_snapshot();
        let v = gpu_status_json(&snap, Some(2));
        let g = &v["gpu"];
        assert_eq!(g["device_id"], "gpu-2");
        assert_eq!(g["gpu_utilization_pct"], 87.0);
        assert_eq!(g["temperature_c"], 71.0);
        assert_eq!(g["power_w"], 250.0);
        assert_eq!(g["vram_used_mb"], 90000);
        // All-GPU form lists every GPU.
        let all = gpu_status_json(&snap, None);
        assert_eq!(all["gpus"].as_array().unwrap().len(), 3);
        // Out-of-range index → graceful error object, not a panic.
        let oob = gpu_status_json(&snap, Some(9));
        assert!(oob["error"].is_string());
    }

    #[test]
    fn list_instances_json_reports_instance_fields() {
        let v = list_instances_json(&fixture_snapshot());
        assert_eq!(v["instance_count"], 1);
        let i = &v["instances"][0];
        assert_eq!(i["name"], "vllm-a");
        assert_eq!(i["model"], "deepseek-r1");
        assert_eq!(i["kv_cache_usage_pct"], 42.0);
        assert_eq!(i["running_reqs"], 3);
    }

    #[test]
    fn tokens_per_watt_json_matches_core_efficiency() {
        // gen_tps 500 / power 250 (gpu-2) = 2.0, matching the reducer.
        let v = tokens_per_watt_json(&fixture_snapshot());
        assert_eq!(v["instances"][0]["tokens_per_watt"], 2.0);
    }

    #[test]
    fn bench_summary_json_rolls_up_groups() {
        let v = bench_summary_json(&fixture_snapshot());
        assert_eq!(v["group_count"], 1);
        let g = &v["groups"][0];
        assert_eq!(g["cell"], "c1");
        assert_eq!(g["n_trials"], 1);
        assert_eq!(g["pass_at_n"], true);
    }

    #[test]
    fn skill_names_are_unique_and_non_empty() {
        for n in SKILL_NAMES {
            assert!(!n.is_empty(), "skill name must be non-empty");
        }
        let mut sorted = SKILL_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            SKILL_NAMES.len(),
            "skill names must be unique"
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

    #[tokio::test]
    async fn skill_tools_expose_both_demo_skills() {
        // list_skills returns both demo skills (the agent can see them).
        let list = ListSkillsTool {
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let def = list.definition(String::new()).await;
        assert_eq!(def.name, "list_skills");
        let out = list.call(NoArgs::default()).await.expect("list ok");
        let names: Vec<String> = out["skills"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["name"].as_str().unwrap().to_string())
            .collect();
        assert!(names.contains(&"install-lemonade".to_string()));
        assert!(names.contains(&"auto-config-endpoint".to_string()));

        // skill_plan returns the ordered dry-run plan for a named skill.
        let plan_tool = SkillPlanTool {
            fired: Arc::new(Mutex::new(Vec::new())),
        };
        let out = plan_tool
            .call(SkillPlanArgs {
                name: "install-lemonade".to_string(),
            })
            .await
            .expect("plan ok");
        let plan = out["plan"].as_array().unwrap();
        assert!(
            plan.iter()
                .any(|l| l.as_str().unwrap().contains("lemonade-sdk"))
        );
        // Unknown skill → graceful error object, not a panic.
        let miss = plan_tool
            .call(SkillPlanArgs {
                name: "nope".to_string(),
            })
            .await
            .unwrap();
        assert!(miss["error"].is_string());
    }

    #[test]
    fn skill_tool_names_registered_in_skill_names() {
        assert!(SKILL_NAMES.contains(&"list_skills"));
        assert!(SKILL_NAMES.contains(&"skill_plan"));
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
        for expected in [
            "install_sdk",
            "install_engine",
            "launch_server",
            "stop_server",
        ] {
            assert!(
                ROCM_MUTATING_TOOL_NAMES.contains(&expected),
                "missing mutating tool: {expected}"
            );
        }
        // Disjoint from read-only + skill names.
        for n in ROCM_MUTATING_TOOL_NAMES {
            assert!(!SKILL_NAMES.contains(&n), "collision with skill: {n}");
            // install_sdk_dry_run (read-only) must not clash with install_sdk.
            assert!(
                !ROCM_READ_TOOL_NAMES.contains(&n),
                "collision with read tool: {n}"
            );
        }
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

    #[test]
    fn rocm_read_tool_names_are_complete_and_disjoint_from_skills() {
        // Every expected read-only tool is registered…
        for expected in [
            "doctor",
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
        ] {
            assert!(
                ROCM_READ_TOOL_NAMES.contains(&expected),
                "missing read tool: {expected}"
            );
        }
        // …no mutating / planning tool leaked in (later phases).
        assert!(!ROCM_READ_TOOL_NAMES.contains(&"natural_language_plan"));
        // The telemetry/skill tools are still present and disjoint from the new set.
        assert!(SKILL_NAMES.contains(&"gpu_status"));
        for n in ROCM_READ_TOOL_NAMES {
            assert!(!SKILL_NAMES.contains(&n), "name collision: {n}");
        }
    }

    #[test]
    fn annotate_reply_appends_deduped_skills() {
        let r = annotate_reply("hi".into(), &["gpu_status".into(), "gpu_status".into()]);
        assert!(r.contains("hi"));
        assert!(r.contains("via: gpu_status"));
        // No skills → unchanged.
        assert_eq!(annotate_reply("hi".into(), &[]), "hi");
    }

    #[tokio::test]
    async fn mock_tool_calling_answer_cites_skill() {
        // The offline "what's GPU-2 doing?" demo path — no live LLM.
        let agent =
            MockAgentClient::with_tool_call("GPU-2 is at 87% util, 71°C, 250 W.", "gpu_status");
        let history = vec![ChatTurn::user("what's GPU-2 doing?")];
        let reply = agent
            .complete(&history, fixture_snapshot())
            .await
            .expect("mock reply");
        assert!(reply.contains("87% util"));
        assert!(
            reply.contains("gpu_status"),
            "reply cites the Skill that fired"
        );
    }

    #[tokio::test]
    async fn mock_error_path_is_err_not_panic() {
        let agent = MockAgentClient::failing();
        let history = vec![ChatTurn::user("hi")];
        let err = agent
            .complete(&history, StateSnapshot::default())
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::Request(_)));
    }

    #[tokio::test]
    async fn mock_empty_history_is_empty_error() {
        let agent = MockAgentClient::new("x");
        let err = agent
            .complete(&[], StateSnapshot::default())
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::Empty));
    }

    /// Manual-demo verification of the live Rig path (tool-calling) against a
    /// local OpenAI-compatible endpoint. NOT run in CI (no live LLM). Run with:
    /// `cargo test -p rocm-dash-tui --lib rig_round_trip -- --ignored`
    /// after starting a local endpoint (e.g. vLLM/Ollama at 127.0.0.1:8000/v1).
    #[tokio::test]
    #[ignore = "requires a live local OpenAI-compatible endpoint"]
    async fn rig_round_trip_against_local_endpoint() {
        let cfg = LlmConfig {
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            model: "local-model".to_string(),
            api_key: None,
            auth_header: None,
        };
        let client = RigAgentClient::new(cfg, None, None).expect("build rig client");
        let history = vec![ChatTurn::user("What's GPU-2 doing? Use the tools.")];
        let reply = client
            .complete(&history, fixture_snapshot())
            .await
            .expect("live reply");
        assert!(!reply.is_empty());
    }

    /// Live round-trip against an OpenAI-compatible endpoint with a custom auth
    /// header. NOT run in CI. Configure via environment variables:
    /// - `LLM_TEST_BASE_URL` (required): base URL of the endpoint
    /// - `LLM_TEST_API_KEY` (required): API key or subscription key
    /// - `LLM_TEST_AUTH_HEADER` (optional): auth header name; defaults to `Authorization` Bearer
    /// - `LLM_TEST_MODEL` (optional): model name; defaults to `gpt-4o-mini`
    ///
    /// Run with:
    /// `cargo test -p rocm-dash-tui --lib rig_round_trip_against_custom_endpoint -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requires LLM_TEST_BASE_URL and LLM_TEST_API_KEY environment variables"]
    async fn rig_round_trip_against_custom_endpoint() {
        let base_url = std::env::var("LLM_TEST_BASE_URL").expect("set LLM_TEST_BASE_URL");
        let key = std::env::var("LLM_TEST_API_KEY").expect("set LLM_TEST_API_KEY");
        let auth_header = std::env::var("LLM_TEST_AUTH_HEADER").ok();
        let model = std::env::var("LLM_TEST_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
        let cfg = LlmConfig {
            base_url,
            model,
            api_key: Some(key),
            auth_header,
        };
        let client = RigAgentClient::new(cfg, None, None).expect("build rig client");
        let history = vec![ChatTurn::user("Reply with exactly: gateway ok")];
        let reply = client
            .complete(&history, fixture_snapshot())
            .await
            .expect("gateway reply");
        assert!(!reply.is_empty());
    }

    #[test]
    fn chatgpt_oauth_client_builds_offline_without_taking_a_key() {
        // Construction is offline (login is deferred to authorize() in
        // complete()); the device-code callback is wired but not yet invoked.
        // Crucially, the constructor signature takes NO api_key — the env-only
        // key invariant is structurally preserved on the no-key path.
        let fired = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = fired.clone();
        let client = ChatGptAgentClient::new(
            Some("gpt-5.3-codex".to_string()),
            move |url, code| {
                // Would surface in the chat tab during a real device-code login.
                sink.lock().unwrap().push(format!("{url}|{code}"));
            },
            None,
            None,
        )
        .expect("build chatgpt oauth client");
        assert_eq!(client.model, "gpt-5.3-codex");
        // No network happened, so the handler has not fired yet.
        assert!(fired.lock().unwrap().is_empty());
    }

    #[test]
    fn chatgpt_oauth_client_defaults_model_when_none() {
        let client = ChatGptAgentClient::new(None, |_url, _code| {}, None, None)
            .expect("build chatgpt oauth client");
        assert_eq!(
            client.model,
            rig::providers::chatgpt::GPT_5_3_CODEX,
            "the no-key default uses the provider's Codex model"
        );
    }

    /// Live no-key device-code round-trip against ChatGPT. NOT run in CI
    /// (interactive OAuth + network). Run with:
    /// `cargo test -p rocm-dash-tui --lib chatgpt_oauth_round_trip -- --ignored --nocapture`
    /// then complete the device login in a browser.
    #[tokio::test]
    #[ignore = "interactive ChatGPT OAuth device-code login + network"]
    async fn chatgpt_oauth_round_trip() {
        let client = ChatGptAgentClient::new(
            None,
            |url, code| {
                eprintln!("Sign in: open {url} and enter code {code}");
            },
            None,
            None,
        )
        .expect("build chatgpt oauth client");
        let history = vec![ChatTurn::user("Reply with exactly: oauth ok")];
        let reply = client
            .complete(&history, fixture_snapshot())
            .await
            .expect("oauth reply");
        eprintln!("ChatGPT OAuth reply: {reply}");
        assert!(!reply.is_empty());
    }
}
