//! Bin-side implementation of the dash execution seam.
//!
//! The dash crates stay free of `rocm-core`; this module lives in the bin
//! (`apps/rocm`, which owns `rocm-core` and the tool engine) and implements the
//! rocm-core-free [`RocmToolExecutor`] boundary by reusing the existing bin
//! engine functions. Plain data in/out only — no dash internals leak here.

use clap::Parser;
use rocm_core::AppPaths;
use rocm_dash_tui::tool_exec::{ApprovalIntent, RocmToolExecutor, RocmToolOutcome};

use crate::Cli;
use crate::providers;

/// Concrete tool executor injected into a live dash. Carries the resolved
/// [`AppPaths`] so read-only tool calls can be served in-process.
#[derive(Debug)]
pub(crate) struct BinToolExecutor {
    paths: AppPaths,
}

impl BinToolExecutor {
    pub(crate) const fn new(paths: AppPaths) -> Self {
        Self { paths }
    }
}

impl RocmToolExecutor for BinToolExecutor {
    fn execute(&self, name: &str, args: &serde_json::Value) -> RocmToolOutcome {
        let call = providers::ChatToolCall {
            id: None,
            name: name.to_owned(),
            arguments: args.clone(),
        };
        if let Err(e) = crate::validate_chat_tool_call(&call) {
            return RocmToolOutcome::Error(e.to_string());
        }
        if crate::chat_tool_call_is_read_only(&call) {
            match crate::run_internal_mcp_call(&self.paths, name, args.clone(), false) {
                Ok(v) => RocmToolOutcome::Result(v),
                Err(e) => RocmToolOutcome::Error(e.to_string()),
            }
        } else {
            match crate::chat_tool_approval_request(&call, None) {
                Ok(req) => RocmToolOutcome::ApprovalRequired(ApprovalIntent {
                    title: req.pending_title,
                    body: {
                        let mut b = vec![req.command_title];
                        if let Some(dc) = req.display_command {
                            b.push(dc);
                        }
                        b
                    },
                    args: req.args,
                }),
                Err(e) => RocmToolOutcome::Error(e.to_string()),
            }
        }
    }

    fn execute_approved(&self, args: &[String]) -> RocmToolOutcome {
        let argv = std::iter::once("rocm".to_owned()).chain(args.iter().cloned());
        match Cli::try_parse_from(argv) {
            Ok(cli) => match crate::dispatch(cli) {
                Ok(()) => RocmToolOutcome::Result(serde_json::json!({ "status": "ok" })),
                Err(e) => RocmToolOutcome::Error(e.to_string()),
            },
            Err(e) => RocmToolOutcome::Error(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::AppPaths;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A hermetic `AppPaths` rooted under the OS temp dir so tests never touch
    /// real home. Built directly (no env mutation) so it stays unsafe-free and
    /// safe under test parallelism.
    fn temp_paths() -> AppPaths {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rocm-dash-seam-{}-{}-{n}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        }
    }

    #[test]
    fn seam_read_only_intent_returns_json() {
        let exec = BinToolExecutor::new(temp_paths());
        let outcome = exec.execute("engines", &serde_json::json!({}));
        match outcome {
            RocmToolOutcome::Result(v) => {
                assert!(v.is_object(), "engines result should be a JSON object");
                assert!(
                    v.get("structuredContent")
                        .and_then(|d| d.get("engines"))
                        .is_some_and(serde_json::Value::is_array),
                    "engines result should carry an engines array, got: {v}"
                );
            }
            other => panic!("expected Result for read-only `engines`, got {other:?}"),
        }
    }

    #[test]
    fn seam_mutating_intent_returns_approval() {
        let exec = BinToolExecutor::new(temp_paths());
        let outcome = exec.execute(
            "install_sdk",
            &serde_json::json!({
                "channel": "release",
                "format": "wheel",
                "prefix": "/tmp/rocm-seam-test-prefix",
            }),
        );
        match outcome {
            RocmToolOutcome::ApprovalRequired(intent) => {
                assert_eq!(
                    &intent.args[..2],
                    &["install".to_owned(), "sdk".to_owned()],
                    "install_sdk approval args should start with [\"install\", \"sdk\"], got: {:?}",
                    intent.args
                );
            }
            other => panic!("expected ApprovalRequired for `install_sdk`, got {other:?}"),
        }
    }
}
