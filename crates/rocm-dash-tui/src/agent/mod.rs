// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The chat agent backend, built on **Rig**, plus the Rig tools it calls.
//!
//! The telemetry tools only read the cached in-memory snapshot. The read-only
//! ROCm tools inspect the host and perform no action that needs operator
//! approval (see the group B comment in `tools` for the bookkeeping writes
//! reading can still trigger), and the mutating ROCm tools only surface an
//! approval request.
//!
//! THE ONLY MODULE SUBTREE THAT NAMES `rig` TYPES — only `tools` and
//! `clients` actually import `rig::*` (this file and `snapshot` do not); no
//! code outside `crate::agent` may name a `rig` type. Everything else talks
//! to the [`AgentClient`] trait, so the Rig dependency is a single swappable
//! seam: tests and the offline demo use [`MockAgentClient`]; the live path
//! uses [`RigAgentClient`] against an OpenAI-compatible endpoint.
//!
//! Rig API verified against `rig-core = "=0.38.1"` (Context7 `/websites/rig_rs`
//! and vendored source). The local-endpoint seam is the Chat Completions API
//! (not the default Responses API), reached via `CompletionsClient`. Tool
//! calling uses `agent.prompt(text).max_turns(N).with_history(history)` so the
//! model can call tools and then answer — one final reply to the UI. A
//! mutating tool never runs inside that loop: it posts its intent on the
//! approval channel for the operator and the model is told it was surfaced.
//!
//! Split into submodules mirroring the crate's `app/mod.rs` → `app/chat.rs`/
//! `slash.rs`/`summary.rs` mechanical-relocation convention: this file holds
//! the shared `AgentClient` seam; `snapshot` holds the pure JSON telemetry
//! helpers; `tools` holds the rig `Tool` wrappers and dispatch; `clients`
//! holds the four backend implementations. Like `app/mod.rs`, this file
//! re-exports the submodules' public items — this module (and its pre-split
//! `crate::agent::*` surface) has in-crate and cross-crate consumers, so the
//! existing paths must keep resolving.

use async_trait::async_trait;

use rocm_dash_core::bench_schema::BenchmarkRow;
use rocm_dash_core::metrics::{Instance, Snapshot};

use crate::app::ChatTurn;

mod clients;
mod snapshot;
mod tools;

pub use clients::{
    AnthropicAgentClient, ChatGptAgentClient, MockAgentClient, RigAgentClient, annotate_reply,
    build_messages,
};
pub use snapshot::{
    bench_summary_json, gpu_status_json, list_instances_json, tokens_per_watt_json,
};
pub use tools::{
    AutomationsRocmTool, BenchSummaryTool, BridgeSnapshotRocmTool, DoctorRocmTool, EnginesRocmTool,
    ExamineRocmTool, GpuSnapshotRocmTool, GpuStatusArgs, GpuStatusTool, InstallEngineRocmTool,
    InstallSdkDryRunRocmTool, InstallSdkRocmTool, LaunchServerRocmTool, ListInstancesTool,
    NaturalLanguagePlanRocmTool, NoArgs, PathExistsRocmTool, PortStatusRocmTool,
    ROCM_MUTATING_TOOL_NAMES, ROCM_READ_TOOL_NAMES, RocmCommandRocmTool, ServiceLogsRocmTool,
    ServicesRocmTool, StopServerRocmTool, TELEMETRY_TOOL_NAMES, TokensPerWattTool, ToolError,
    UpdateCheckRocmTool, WatcherDisableRocmTool, WatcherEnableRocmTool,
};

/// Default one-shot request budget. A hung backend becomes a timeout error
/// turn, never a frozen pane.
///
/// Sized for a locally served model on modest hardware: the first message
/// prefills the whole system prompt plus tool schemas on a cold prompt cache
/// (about 22 s for roughly 3k tokens on an integrated GPU), and the budget
/// covers every tool-calling turn of the request, not just that prefill.
pub const DEFAULT_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(2);

/// Environment variable that overrides [`DEFAULT_REQUEST_TIMEOUT`], in whole
/// seconds.
pub const REQUEST_TIMEOUT_ENV: &str = "ROCM_DASH_CHAT_TIMEOUT_SECS";

/// Longest budget an override may set, so a typo cannot park a chat turn for
/// hours.
const MAX_REQUEST_TIMEOUT_SECS: u64 = 3600;

/// The request budget in effect: [`REQUEST_TIMEOUT_ENV`] when it holds a whole
/// number of seconds in `1..=3600`, otherwise [`DEFAULT_REQUEST_TIMEOUT`].
pub fn request_timeout() -> std::time::Duration {
    parse_request_timeout(std::env::var(REQUEST_TIMEOUT_ENV).ok().as_deref())
}

fn parse_request_timeout(raw: Option<&str>) -> std::time::Duration {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|secs| (1..=MAX_REQUEST_TIMEOUT_SECS).contains(secs))
        .map_or(DEFAULT_REQUEST_TIMEOUT, std::time::Duration::from_secs)
}

/// Optional sampling controls forwarded to the chat backend.
///
/// Parity with the `rocm chat` / `rocm serve` CLI flags. `None` means "leave
/// the model / endpoint default untouched", so an unset knob never overrides a
/// server- or recipe-configured value. Applied uniformly across all three live
/// backends.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InferenceParams {
    /// Sampling temperature (already validated `>= 0.0` by the bin).
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass (already validated in `0.0..=1.0`).
    pub top_p: Option<f32>,
    /// Upper bound on generated tokens; overrides `MAX_AGENT_TOKENS` (defined
    /// in `agent::clients`) when set.
    pub max_tokens: Option<u32>,
}

/// Errors from a chat completion. Public form is string-only so no `rig` type
/// leaks past this module. Messages never include the api_key (it is a
/// header, never part of base_url or the request path).
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("no message to send")]
    Empty,
    #[error("failed to build chat client: {0}")]
    Build(String),
    #[error("chat sign-in failed: {0}")]
    Auth(String),
    #[error("request timed out after {0}s (set {REQUEST_TIMEOUT_ENV} to allow longer)")]
    Timeout(u64),
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

/// Shared test fixture: a representative [`StateSnapshot`] used across
/// `snapshot`/`tools`/`clients` test modules. Centralized here (rather than
/// tripled in each submodule) since it's ~55 lines of test-only setup with no
/// production-code role.
#[cfg(test)]
fn fixture_snapshot() -> StateSnapshot {
    use rocm_dash_core::bench_schema::PassFail;
    use rocm_dash_core::metrics::GpuMetrics;

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
        // Daemon-computed value: 500 tok/s ÷ 250 W (gpu-2) = 2.0 tok/W.
        // Set explicitly so tokens_per_watt_json reads the daemon field.
        tokens_per_watt: Some(2.0),
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

#[cfg(test)]
mod request_timeout_tests {
    use std::time::Duration;

    use super::{
        AgentError, DEFAULT_REQUEST_TIMEOUT, MAX_REQUEST_TIMEOUT_SECS, REQUEST_TIMEOUT_ENV,
        parse_request_timeout,
    };

    #[test]
    fn an_unset_override_leaves_the_default_budget() {
        assert_eq!(parse_request_timeout(None), DEFAULT_REQUEST_TIMEOUT);
    }

    // Pins the two numbers the rest of this module only compares against
    // themselves. Without these, restoring the 45 s budget this change exists
    // to replace leaves every other test green. Both values are published in
    // README.md, docs/testing.md and docs/manual-testing.md section 4d; change
    // them together or the docs drift silently.
    #[test]
    fn the_budget_and_its_upper_bound_are_the_values_the_docs_promise() {
        assert_eq!(
            DEFAULT_REQUEST_TIMEOUT.as_secs(),
            120,
            "the default budget is 2 minutes; 45 s is the cut-off this change removed"
        );
        assert_eq!(
            MAX_REQUEST_TIMEOUT_SECS, 3600,
            "an override may reach one hour"
        );
    }

    #[test]
    fn a_whole_number_of_seconds_overrides_the_default() {
        assert_eq!(parse_request_timeout(Some("300")), Duration::from_mins(5));
        assert_eq!(
            parse_request_timeout(Some(" 90 ")),
            Duration::from_secs(90),
            "surrounding whitespace from a shell is not a reason to reject it"
        );
        assert_eq!(
            parse_request_timeout(Some(&MAX_REQUEST_TIMEOUT_SECS.to_string())),
            Duration::from_secs(MAX_REQUEST_TIMEOUT_SECS)
        );
    }

    #[test]
    fn an_unusable_override_falls_back_to_the_default_instead_of_disabling_the_budget() {
        for bad in [
            "",
            "0",
            "-5",
            "abc",
            "1.5",
            "99999",
            &(MAX_REQUEST_TIMEOUT_SECS + 1).to_string(),
        ] {
            assert_eq!(
                parse_request_timeout(Some(bad)),
                DEFAULT_REQUEST_TIMEOUT,
                "{bad:?}"
            );
        }
    }

    #[test]
    fn the_timeout_message_names_the_budget_and_the_variable_that_changes_it() {
        let msg = AgentError::Timeout(120).to_string();
        assert!(msg.contains("after 120s"), "{msg}");
        // `parse_request_timeout` is what `request_timeout()` feeds
        // `REQUEST_TIMEOUT_ENV` into, and the unit tests above prove it
        // honours a valid value; this pins that the message advertises that
        // same variable. That the request path really reads it is proven by
        // `the_request_path_honours_the_chat_timeout_environment_variable`
        // in `clients.rs`.
        assert!(msg.contains(REQUEST_TIMEOUT_ENV), "{msg}");
    }
}
