// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Bin-side implementation of the dash execution seam.
//!
//! The dash crates stay free of `rocm-core`; this module lives in the bin
//! (`apps/rocm`, which owns `rocm-core` and the tool engine) and implements the
//! rocm-core-free [`RocmToolExecutor`] boundary by reusing the existing bin
//! engine functions. Plain data in/out only — no dash internals leak here.

use rocm_core::AppPaths;
use rocm_dash_tui::tool_exec::{ApprovalIntent, RocmToolExecutor, RocmToolOutcome};

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
                    name: name.to_owned(),
                    arguments: args.clone(),
                }),
                Err(e) => RocmToolOutcome::Error(e.to_string()),
            }
        }
    }

    fn execute_approved(&self, name: &str, args: &serde_json::Value) -> RocmToolOutcome {
        // Replay the approved mutating call via the captured-subprocess path
        // (`allow_mutation = true`). `run_internal_mcp_call` re-validates the
        // call first, so the safety validators stay the single gate; it runs the
        // action with piped stdout/stderr (TUI-safe) — no `dispatch`, no stdout
        // corruption.
        match crate::run_internal_mcp_call(&self.paths, name, args.clone(), true) {
            Ok(v) => RocmToolOutcome::Result(v),
            // `Error` is *not* the failed-command arm. `run_rocm_capture_for_paths`
            // captures a non-zero `rocm` exit, so a refused command still returns
            // `Ok` — an `isError: true` envelope that takes the `Result` arm above
            // and gets collapsed by `summarize_json_value`. This arm fires only
            // when the call itself fails: validation, spawn, timeout, unknown
            // tool. That split is pinned by
            // `seam_execute_approved_captures_a_failing_command_as_a_result`
            // below, which drives this function against a real `rocm`
            // subprocess that exits non-zero; the collapsing half is pinned by
            // `approved_command_failure_stays_a_collapsed_envelope` in
            // `crates/rocm-dash-tui/src/app/mod.rs`.
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
                    intent.name, "install_sdk",
                    "approval intent carries the tool name for re-execution"
                );
                assert!(
                    !intent.body.is_empty(),
                    "approval body should carry human-readable lines, got: {:?}",
                    intent.body
                );
                // The replayable payload is the same args object we passed in.
                assert_eq!(intent.arguments["channel"], "release");
            }
            other => panic!("expected ApprovalRequired for `install_sdk`, got {other:?}"),
        }
    }

    #[test]
    fn seam_execute_rejects_public_bind_before_approval() {
        // (d) an UNSAFE mutating call fails validation in execute() → Error, NOT
        // ApprovalRequired. The approval modal never opens for a rejected call.
        let exec = BinToolExecutor::new(temp_paths());
        let outcome = exec.execute(
            "launch_server",
            &serde_json::json!({ "model": "m", "host": "0.0.0.0" }),
        );
        assert!(
            matches!(outcome, RocmToolOutcome::Error(_)),
            "public-bind launch_server must be rejected, got {outcome:?}"
        );
    }

    #[test]
    fn seam_execute_rejects_cpu_device_before_approval() {
        let exec = BinToolExecutor::new(temp_paths());
        let outcome = exec.execute(
            "launch_server",
            &serde_json::json!({ "model": "m", "host": "127.0.0.1", "device": "cpu" }),
        );
        assert!(
            matches!(outcome, RocmToolOutcome::Error(_)),
            "CPU-device launch_server must be rejected, got {outcome:?}"
        );
    }

    /// Drives the *real* approved-replay chain for a `rocm` command that exits
    /// non-zero: `execute_approved` → `run_internal_mcp_call(…, true)` →
    /// `run_rocm_capture_for_paths` (a genuine subprocess) →
    /// `internal_mcp_tool_result_from_command`. None of those are stubbed here,
    /// which is the point: the arm a refused command lands in is what the
    /// ComfyUI e2e scenario's CLI-only scope rests on. Making this function map
    /// a captured non-zero exit to `RocmToolOutcome::Error` turns this red.
    ///
    /// `runtimes activate <unknown key>` is the cheapest command that gets
    /// there: `chat_rocm_command_action_from_args` classifies any non-`list`
    /// `runtimes` invocation as `Approval`, and against the hermetic
    /// [`temp_paths`] registry below (which has no manifests at all) the child
    /// refuses deterministically with no network, no GPU and no real state.
    /// The asserted stderr is that refusal verbatim, so the test cannot pass
    /// off some other non-zero exit as the captured one.
    #[test]
    fn seam_execute_approved_captures_a_failing_command_as_a_result() {
        // `run_rocm_capture_for_paths` spawns `daemon_binary_path()`, which from
        // a unit test means "the `rocm` next to the test harness". If the binary
        // has not been built it silently falls back to the harness itself, which
        // would re-enter libtest instead of running a command — so refuse to
        // proceed rather than measure the wrong process.
        // The binary is only there because `apps/rocm/tests/` exists and makes
        // cargo build it; delete those and this guard fires for that reason.
        let binary = rocm_core::daemon_binary_path().expect("resolve the rocm binary");
        assert_eq!(
            binary.file_stem().and_then(std::ffi::OsStr::to_str),
            Some("rocm"),
            "this test replays through a real `rocm` subprocess but resolved `{}`; \
             build the binary first (`cargo build -p rocm`) — `cargo test -p rocm \
             --bin rocm` on its own only builds the unit-test harness",
            binary.display()
        );

        let exec = BinToolExecutor::new(temp_paths());
        let outcome = exec.execute_approved(
            "rocm_command",
            &serde_json::json!({
                "args": ["runtimes", "activate", "no-such-runtime-key"],
            }),
        );
        let RocmToolOutcome::Result(v) = outcome else {
            panic!(
                "a captured non-zero `rocm` exit must stay a Result envelope, got {outcome:?}; \
                 the `Error` arm is for calls that never ran"
            );
        };
        assert_eq!(
            v["structuredContent"]["exit_status"],
            serde_json::json!(1),
            "the child really did refuse; without that this test proves nothing: {v}"
        );
        assert_eq!(
            v["isError"],
            serde_json::json!(true),
            "a captured non-zero exit is flagged inside the envelope, not raised: {v}"
        );
        // The refusal text is buried in the envelope rather than surfaced —
        // exactly the shape `approved_command_failure_stays_a_collapsed_envelope`
        // (`crates/rocm-dash-tui/src/app/mod.rs`) then collapses out of the chat.
        assert!(
            v["structuredContent"]["stderr"]
                .as_str()
                .is_some_and(|stderr| stderr.contains("installed runtime not found")),
            "the CLI's own refusal is carried as captured stderr: {v}"
        );
    }

    #[test]
    fn seam_execute_approved_rejects_unsafe_call_via_validator() {
        // execute_approved re-validates: a public-bind launch_server is rejected
        // (the validators stay the single gate even on the approved path).
        let exec = BinToolExecutor::new(temp_paths());
        let outcome = exec.execute_approved(
            "launch_server",
            &serde_json::json!({ "model": "m", "host": "0.0.0.0" }),
        );
        assert!(
            matches!(outcome, RocmToolOutcome::Error(_)),
            "approved-path execution must still reject an unsafe call, got {outcome:?}"
        );
    }
}
