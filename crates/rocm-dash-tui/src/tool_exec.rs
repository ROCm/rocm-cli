// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! rocm-core-free tool-call boundary (the execution seam).
//!
//! Plain-data signatures ONLY (serde_json / std / serde). The bin (`apps/rocm`,
//! which owns `rocm-core` and the tool engine) implements [`RocmToolExecutor`];
//! the dash holds it as `Option<Arc<dyn RocmToolExecutor>>` and never depends on
//! `rocm-core`. Phase 2 stored the seam; Phase 3 used it for read-only tools;
//! Phase 4 adds the mutating "execute approved" path + the approval descriptor.

use std::sync::Arc;

/// Plain-data approval descriptor surfaced to the app event loop, which renders
/// it in the approval modal and — only on Approve — replays it via
/// [`RocmToolExecutor::execute_approved`].
///
/// `title` + `body` are the human-readable display (the rendered command and an
/// optional explanation); `name` + `arguments` are the actionable payload used
/// to re-execute the *same* call the validator already accepted. Re-executing by
/// `(name, arguments)` (not by a free-form command) keeps the safety validators
/// the single gate — `execute_approved` re-validates before running.
#[derive(Debug, Clone)]
pub struct ApprovalIntent {
    pub title: String,
    pub body: Vec<String>,
    pub name: String,
    pub arguments: serde_json::Value,
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
pub trait RocmToolExecutor: std::fmt::Debug + Send + Sync {
    /// Execute a tool-call intent: read-only → Result(json); mutating → ApprovalRequired; failure → Error.
    /// (Return value carries `#[must_use]` via the `RocmToolOutcome` enum.)
    fn execute(&self, name: &str, args: &serde_json::Value) -> RocmToolOutcome;

    /// Run an *approved* mutating action via the bin's captured-subprocess path
    /// (piped stdout/stderr → JSON; no printing to the TUI terminal, so it is
    /// TUI-safe). Called only after the operator approves the modal, with the
    /// `(name, arguments)` taken from the [`ApprovalIntent`] that `execute()`
    /// returned. It re-validates the call, so the safety validators remain the
    /// single gate and an unapproved/invalid call can never run here. This is a
    /// blocking call — invoke it off the async event loop (spawn_blocking).
    fn execute_approved(&self, name: &str, args: &serde_json::Value) -> RocmToolOutcome;
}

/// Arc-wrapped executor as stored in `ResolvedArgs`/`AppState`.
pub type SharedRocmToolExecutor = Arc<dyn RocmToolExecutor>;
