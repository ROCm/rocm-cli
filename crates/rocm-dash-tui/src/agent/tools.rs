// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Rig `Tool` ("Skill") wrappers over the cached telemetry snapshot, plus the
//! read-only and mutating ROCm machine-inspection tools and their registration
//! onto a Rig `AgentBuilder`.
//!
//! Split out of `agent.rs` to keep the `AgentClient` seam focused. The shared
//! seam types (`StateSnapshot`, `AgentError`, `AgentClient`) stay in
//! `agent/mod.rs`; the JSON summarization these tools call lives in
//! `agent::snapshot`.

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
// Rig Tool ("Skill") wrappers. Each holds an `Arc<StateSnapshot>` (read-only)
// and a shared `fired` log so the reply can cite which Skills ran. `call` only
// reads the snapshot — no mutation, no network, no file I/O.
// ---------------------------------------------------------------------------

/// Shared "which Skills fired" log, threaded into every tool struct below and
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
// pull `rocm-core` into this crate; the `agent` module stays the sole `rig` namer.
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

/// All read-only ROCm tool names (mirrors [`SKILL_NAMES`]).
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

/// Register the telemetry + skill registry tools (GpuStatus, ListInstances,
/// BenchSummary, TokensPerWatt — snapshot-backed; ListSkills, SkillPlan —
/// read-only registry) on a fresh Rig `AgentBuilder`, cloning the snapshot +
/// shared `fired` log into each. Kept generic over the builder's completion
/// model + preamble so all three backends (RigAgentClient, ChatGptAgentClient,
/// AnthropicAgentClient) reuse one registration site (DRY — the tool list lives
/// in exactly one place, mirroring [`register_rocm_read_tools`]). Takes the base
/// builder (`NoToolConfig`) and returns it in `WithBuilderTools` because the
/// first `.tool()` transitions the type-state; the ROCm read/mutating
/// registrations chain after it. Note: ListSkillsTool/SkillPlanTool take only
/// `fired`; the other four take `snap` + `fired`.
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
        // Skills registry tools (read-only: list + dry-run plan; never execute).
        .tool(ListSkillsTool {
            fired: fired.clone(),
        })
        .tool(SkillPlanTool {
            fired: fired.clone(),
        })
}

/// Register every read-only ROCm tool on a Rig `AgentBuilder`, cloning the
/// optional executor + the shared `fired` log into each. Kept generic over the
/// builder's completion model + preamble so all three backends (the
/// OpenAI-compatible, ChatGPT, and Anthropic paths) reuse one registration
/// site (DRY — the tool list lives in exactly one place). The builder is
/// already in the `WithBuilderTools` state because the telemetry/skill tools
/// were registered first.
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
        // Disjoint from read-only + skill names (iterates the registry itself).
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
    fn rocm_read_tool_names_are_complete_and_disjoint_from_skills() {
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
        // The telemetry/skill tools are still present and disjoint from the new set.
        assert!(SKILL_NAMES.contains(&"gpu_status"));
        for n in ROCM_READ_TOOL_NAMES {
            assert!(!SKILL_NAMES.contains(&n), "name collision: {n}");
        }
    }
}
