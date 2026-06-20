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
    fn execute(&self, name: &str, args: &serde_json::Value) -> RocmToolOutcome;
    /// Run a previously-approved set of CLI args in-process (the mutating "execute approved" path).
    fn execute_approved(&self, args: &[String]) -> RocmToolOutcome;
}

pub type SharedRocmToolExecutor = Arc<dyn RocmToolExecutor>;
