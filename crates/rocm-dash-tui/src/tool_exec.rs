// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! rocm-core-free tool-call boundary (the execution seam).
//!
//! Plain-data signatures ONLY (serde_json / std / serde). The bin (`apps/rocm`,
//! which owns `rocm-core` and the tool engine) implements [`RocmToolExecutor`];
//! the dash holds it as `Option<Arc<dyn RocmToolExecutor>>` and never depends on
//! `rocm-core`. Phase 2 only stores the seam; Phase 3 will use it.

use std::sync::Arc;

/// Plain-data approval descriptor surfaced to the app event loop (Phase 4 renders it).
#[derive(Debug, Clone)]
pub struct ApprovalIntent {
    pub title: String,
    pub body: Vec<String>,
    pub args: Vec<String>,
}

/// Outcome of a tool-call intent executed by the bin across the seam.
#[must_use]
#[derive(Debug, Clone)]
pub enum RocmToolOutcome {
    Result(serde_json::Value),
    ApprovalRequired(ApprovalIntent),
    Error(String),
}

/// rocm-core-free tool-executor boundary.
///
/// The bin implements this; the dash holds it as
/// `Option<Arc<dyn RocmToolExecutor>>` (None for demo/replay/mock). The `Debug`
/// supertrait keeps `ResolvedArgs`/`AppState` deriving Debug.
///
/// NOTE: the mutating "execute approved" path is intentionally deferred to
/// Phase 4, where it will be added alongside the approval modal with proper
/// stdout/stderr capture (TUI-safe), spawn_blocking off the async loop, and an
/// approval-provenance barrier so only descriptors from `execute()`'s
/// ApprovalRequired can be run.
pub trait RocmToolExecutor: std::fmt::Debug + Send + Sync {
    /// Execute a tool-call intent: read-only → Result(json); mutating → ApprovalRequired; failure → Error.
    /// (Return value carries `#[must_use]` via the `RocmToolOutcome` enum.)
    fn execute(&self, name: &str, args: &serde_json::Value) -> RocmToolOutcome;
}

/// Arc-wrapped executor as stored in `ResolvedArgs`/`AppState`.
pub type SharedRocmToolExecutor = Arc<dyn RocmToolExecutor>;
