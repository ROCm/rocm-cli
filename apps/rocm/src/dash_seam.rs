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

use crate::{chat, providers};

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
        if let Err(e) = chat::validate_chat_tool_call(&call) {
            return RocmToolOutcome::Error(e.to_string());
        }
        if chat::chat_tool_call_is_read_only(&call) {
            match chat::run_internal_mcp_call(&self.paths, name, args.clone(), false) {
                Ok(v) => RocmToolOutcome::Result(v),
                Err(e) => RocmToolOutcome::Error(e.to_string()),
            }
        } else {
            match chat::chat_tool_approval_request(&call, None) {
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
        match chat::run_internal_mcp_call(&self.paths, name, args.clone(), true) {
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
            // `crates/rocm-dash-tui/src/app/tool_approval.rs`.
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

    /// The dash registers `rocm_command` as a *read-only* tool and runs whatever
    /// `Result` `execute()` hands back, so this seam is what stops a
    /// model-issued mutating argv from running. A bare `serve` fails the safety
    /// validator; a well-formed managed `serve` is classified as mutating and
    /// only surfaces an approval request. Neither may come back as a `Result`,
    /// which is what an executed command looks like.
    #[test]
    fn seam_rejects_mutating_rocm_command() {
        let exec = BinToolExecutor::new(temp_paths());

        let outcome = exec.execute("rocm_command", &serde_json::json!({ "args": ["serve"] }));
        assert!(
            matches!(&outcome, RocmToolOutcome::Error(msg) if msg.contains("--managed")),
            "a bare `serve` must be rejected by the managed-serving rule, got {outcome:?}"
        );

        let args = serde_json::json!({ "args": ["serve", "qwen", "--managed"] });
        match exec.execute("rocm_command", &args) {
            RocmToolOutcome::ApprovalRequired(intent) => {
                assert_eq!(intent.name, "rocm_command");
                assert_eq!(intent.arguments, args);
            }
            other => panic!("a managed `serve` must only surface for approval, got {other:?}"),
        }
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
    /// The asserted argv, exit code and stderr together pin *that* refusal, so
    /// the test cannot pass off some other non-zero exit as the captured one.
    #[test]
    fn seam_execute_approved_captures_a_failing_command_as_a_result() {
        // `run_rocm_capture_for_paths` spawns `daemon_binary_path()`, which from
        // a unit test means "the `rocm` next to the test harness". If the binary
        // has not been built it silently falls back to the harness itself, which
        // would re-enter libtest instead of running a command — so skip rather
        // than measure the wrong process.
        //
        // This *skips* instead of failing on purpose. The binary is present for
        // every gate that matters (unfiltered `cargo test -p rocm` builds it,
        // as do the nextest and `--all-targets` lanes), but `docs/testing.md`
        // teaches the filtered `cargo test -p rocm --bin rocm <filter>` form,
        // which builds only the unit-test harness. Failing there would hand a
        // contributor a red test unrelated to their change.
        let binary = rocm_core::daemon_binary_path().expect("resolve the rocm binary");
        if binary.file_stem().and_then(std::ffi::OsStr::to_str) != Some("rocm") {
            eprintln!(
                "skipping `seam_execute_approved_captures_a_failing_command_as_a_result`: \
                 it replays through a real `rocm` subprocess but resolved `{}`. \
                 Build the binary first (`cargo build -p rocm`) — `cargo test -p rocm \
                 --bin rocm <filter>` on its own only builds the unit-test harness.",
                binary.display()
            );
            return;
        }

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
        // `argv[0]` is the resolved binary path, which is machine-dependent; the
        // arguments after it are what this test pins.
        assert_eq!(
            v["structuredContent"]["argv"].as_array().map(|argv| argv
                .iter()
                .skip(1)
                .filter_map(serde_json::Value::as_str)
                .collect()),
            Some(vec!["runtimes", "activate", "no-such-runtime-key"]),
            "the envelope must carry the argv actually spawned: {v}"
        );
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
        // (`crates/rocm-dash-tui/src/app/tool_approval.rs`) then collapses out of the chat.
        //
        // Anchored on the rejected selector rather than on `select_runtime_manifest`'s
        // current "installed runtime not found" wording: that message is on this
        // PR's Deferred list to be reworked into the "refuse and list keys" shape,
        // and any such refusal still names the key it would not resolve. This
        // keeps the test proving "the child really refused, and said why" without
        // pinning a sentence already scheduled for rewrite.
        assert!(
            v["structuredContent"]["stderr"]
                .as_str()
                .is_some_and(|stderr| stderr.contains("no-such-runtime-key")),
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

    /// One approval-gated tool call per mutating tool, each carrying a second,
    /// *different* payload under a nested `arguments` key. The outer object is
    /// what the model's call looks like on its face; the inner one differs in
    /// every field the approval card shows, so any path that reads one object
    /// for the card and the other for the command would be visible.
    fn nested_arguments_calls() -> Vec<(&'static str, serde_json::Value)> {
        vec![
            (
                "launch_server",
                serde_json::json!({
                    "model": "shown/model",
                    "arguments": { "model": "other/model", "port": 8123 },
                }),
            ),
            (
                "install_sdk",
                serde_json::json!({
                    "channel": "release",
                    "format": "wheel",
                    "prefix": "/tmp/rocm-seam-shown-prefix",
                    "arguments": {
                        "channel": "nightly",
                        "format": "wheel",
                        "prefix": "/tmp/rocm-seam-other-prefix",
                    },
                }),
            ),
            (
                "install_engine",
                serde_json::json!({
                    "engine": "lemonade",
                    "arguments": { "engine": "vllm", "reinstall": true },
                }),
            ),
            (
                "stop_server",
                serde_json::json!({
                    "service_id": "shown-service",
                    "arguments": { "service_id": "other-service" },
                }),
            ),
            (
                "watcher_enable",
                serde_json::json!({
                    "watcher": "therock-update",
                    "mode": "observe",
                    "arguments": { "watcher": "server-recover", "mode": "contained" },
                }),
            ),
            (
                "watcher_disable",
                serde_json::json!({
                    "watcher": "therock-update",
                    "arguments": { "watcher": "server-recover" },
                }),
            ),
        ]
    }

    /// The command line the approved replay of `arguments` would run — the
    /// same call `run_internal_mcp_call` builds, without spawning it.
    fn approved_command_line(name: &str, arguments: &serde_json::Value) -> Option<String> {
        let call = chat::internal_mcp_tool_call(name, arguments.clone()).ok()?;
        let args = chat::rocm_chat_tool_requested_args(&call)?;
        Some(crate::rocm_argv::format_structured_tool_call("rocm", &args))
    }

    /// The approval card for a mutating tool and the command its approval runs
    /// must describe the same action. For every approval-gated tool, a call
    /// carrying a nested `arguments` payload either never reaches a card, or
    /// the command run on approval is exactly the one the card shows.
    ///
    /// This is the end-to-end statement of the guarantee and the regression
    /// test for the original defect: with both halves of the fix reverted (no
    /// refusal, and the replay unwrapping the nested object) it reports a
    /// disagreement for all six tools. Either half alone keeps it green, by
    /// design — the fix is two independent layers — so each half is pinned by
    /// its own test: removing the refusal fails
    /// `seam_refuses_nested_arguments_before_approval_and_on_replay` and
    /// `validate_chat_tool_call_refuses_a_nested_arguments_field_for_every_tool`,
    /// and reintroducing the unwrap ahead of validation fails
    /// `mcp_call_refuses_tool_fields_mixed_with_a_nested_arguments_object` in
    /// `chat/mcp_bridge.rs`, whose replay of the mixed object then no longer
    /// reports the nested field.
    #[test]
    fn seam_approval_card_matches_the_command_approval_runs() {
        let exec = BinToolExecutor::new(temp_paths());
        let mut disagreements = Vec::new();
        for (name, payload) in nested_arguments_calls() {
            match exec.execute(name, &payload) {
                RocmToolOutcome::Error(_) => {}
                RocmToolOutcome::ApprovalRequired(intent) => {
                    let ran = approved_command_line(&intent.name, &intent.arguments);
                    if !ran.as_ref().is_some_and(|ran| intent.body.contains(ran)) {
                        disagreements.push(format!(
                            "`{name}`: the approval card shows {:?} but approving it runs {ran:?}",
                            intent.body
                        ));
                    }
                }
                RocmToolOutcome::Result(v) => {
                    panic!("`{name}` is approval-gated but ran without approval: {v}")
                }
            }
        }
        assert!(disagreements.is_empty(), "{}", disagreements.join("\n"));
    }

    /// The same calls with the nested payload removed are ordinary, valid
    /// requests: each must still reach an approval card, and that card must
    /// show exactly the command its approval runs. Without this the agreement
    /// test above could pass by refusing every call.
    #[test]
    fn seam_approval_card_matches_the_command_for_plain_calls() {
        let exec = BinToolExecutor::new(temp_paths());
        for (name, mut payload) in nested_arguments_calls() {
            payload
                .as_object_mut()
                .expect("fixture is an object")
                .remove("arguments");
            let RocmToolOutcome::ApprovalRequired(intent) = exec.execute(name, &payload) else {
                panic!("plain `{name}` call must reach an approval card");
            };
            assert_eq!(
                intent.arguments, payload,
                "`{name}`: the intent replays the object the card was built from"
            );
            let ran = approved_command_line(&intent.name, &intent.arguments)
                .unwrap_or_else(|| panic!("plain `{name}` call must be runnable once approved"));
            assert!(
                intent.body.contains(&ran),
                "`{name}`: the approval card shows {:?} but approving it runs {ran}",
                intent.body
            );
        }
    }

    /// A nested `arguments` payload is refused before any card is shown, and
    /// the approved replay refuses it too, without spawning anything — so an
    /// intent assembled some other way still cannot run the inner payload.
    #[test]
    fn seam_refuses_nested_arguments_before_approval_and_on_replay() {
        let exec = BinToolExecutor::new(temp_paths());
        for (name, payload) in nested_arguments_calls() {
            match exec.execute(name, &payload) {
                RocmToolOutcome::Error(e) => assert!(
                    e.contains("nested `arguments`"),
                    "`{name}`: refusal should name the nested field, got: {e}"
                ),
                other => panic!("`{name}` with a nested payload must be refused, got {other:?}"),
            }
            match exec.execute_approved(name, &payload) {
                RocmToolOutcome::Error(e) => assert!(
                    e.contains("nested `arguments`"),
                    "`{name}`: replay refusal should name the nested field, got: {e}"
                ),
                other => panic!(
                    "`{name}` replay with a nested payload must be refused before running, got {other:?}"
                ),
            }
        }
    }

    /// Read-only tools go through the same gate: the result returned to the
    /// model must describe the call the model made, not an object inside it.
    #[test]
    fn seam_refuses_nested_arguments_on_read_only_tools() {
        let exec = BinToolExecutor::new(temp_paths());
        let outcome = exec.execute(
            "rocm_command",
            &serde_json::json!({
                "args": ["engines"],
                "arguments": { "args": ["runtimes", "activate", "some-runtime"] },
            }),
        );
        assert!(
            matches!(&outcome, RocmToolOutcome::Error(e) if e.contains("nested `arguments`")),
            "read-only call with a nested payload must be refused, got {outcome:?}"
        );
    }
}
