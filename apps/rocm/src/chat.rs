// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The chat assistant's entry points and the module that ties its pieces
//! together: reading a piped prompt (`read_piped_prompt`), the status screen
//! (`render_chat_text`), the one-shot reply text (`render_chat_prompt_text`) and
//! the clap value parsers for the sampling flags. The work behind them is split
//! into child modules, each a leaf of the call graph below the turn loop: `turn`
//! (the turn loop, the system prompt and the transcript), `intent` (the
//! prompt-intent heuristics), `mcp_bridge` (the MCP tool bridge and in-process
//! command execution) and `summaries` (the deterministic tool-output
//! summaries). The tool-call validation and approval policy lives in
//! `tool_policy.rs`.
//!
//! Callers in other modules, production and test-only, reach the moved items
//! through the `chat::` module boundary, so this module re-exports each one
//! another module reads (`run_internal_mcp_call`,
//! `rocm_chat_tool_system_prompt_for_host`, the tool-policy items, `any_substring`
//! and `ensure_flag`, and the test-only ones under `#[cfg(test)]`).

mod intent;
mod mcp_bridge;
mod summaries;
mod turn;

// `deterministic_comfyui_tool_summary` lives in `summaries.rs` and stays
// reachable here as `chat::deterministic_comfyui_tool_summary` for
// `comfyui.rs`'s tests, which are its only readers outside `summaries.rs`.
// Those tests are all `#[cfg(unix)]`, so the re-export is gated the same way:
// on Windows it would be an unused import.
#[cfg(all(test, unix))]
pub(crate) use self::summaries::deterministic_comfyui_tool_summary;
// `run_internal_mcp_call` lives in `mcp_bridge.rs` and stays reachable here as
// `chat::run_internal_mcp_call` for `main.rs` and `dash_seam.rs`.
pub(crate) use self::mcp_bridge::run_internal_mcp_call;
// `rocm_chat_tool_system_prompt_for_host` lives in `turn.rs` and stays
// reachable here as `chat::rocm_chat_tool_system_prompt_for_host` for `dash.rs`.
use self::turn::render_chat_prompt_result_with_progress;
pub(crate) use self::turn::rocm_chat_tool_system_prompt_for_host;
// `any_substring` lives in `prompt_text.rs` and stays reachable here as
// `chat::any_substring` for `freeform_answer.rs`.
pub(crate) use crate::prompt_text::any_substring;
// `ensure_flag` lives in `rocm_argv.rs` and stays reachable here as
// `chat::ensure_flag` for `freeform_execution.rs`.
pub(crate) use crate::rocm_argv::ensure_flag;
// The tool-call policy lives in `tool_policy.rs`. `chat_tool_approval_request`,
// `chat_tool_call_is_read_only` and `validate_chat_tool_call` stay reachable
// here as `chat::<item>` for `dash_seam.rs`. `ChatRocmCommandAction`,
// `rocm_chat_tool_requested_args`, `chat_install_driver_action` and
// `chat_rocm_command_action_from_args` are read through `chat::` only by other
// modules' tests (`driver_plan.rs`, `system_packages.rs`), and this module does
// not call them itself, so their re-export below is `#[cfg(test)]`.
#[cfg(test)]
pub(crate) use crate::tool_policy::{
    ChatRocmCommandAction, chat_install_driver_action, chat_rocm_command_action_from_args,
    rocm_chat_tool_requested_args,
};
pub(crate) use crate::tool_policy::{
    chat_tool_approval_request, chat_tool_call_is_read_only, validate_chat_tool_call,
};
use crate::{ChatInferenceParams, providers};
use anyhow::{Context, Result};
use rocm_core::AppPaths;
use std::fmt::Write as _;
use std::io::Read as _;

/// Read a one-shot chat prompt from standard input when it is piped in.
///
/// Backs the documented `echo "…" | rocm chat` path: when `--prompt` is omitted
/// and stdin is not a terminal, the piped text becomes the prompt. Returns
/// `None` when stdin is an interactive TTY or the piped input is blank, so the
/// caller falls back to the status screen instead of blocking on input nobody
/// can supply.
///
/// The read runs to EOF — the conventional filter contract that
/// `read_provider_key_from_user` already follows. A caller that hands us a pipe
/// it never writes to and never closes therefore waits, exactly as `cat` would;
/// the TTY guard is what keeps that off an interactive user, and a
/// non-interactive caller with no prompt to send should pass `/dev/null` (as
/// `scripts/smoke_local.py` does) rather than an idle pipe. A read error — a
/// closed or non-UTF-8 fd 0 — is reported instead of being folded into "no
/// prompt", so text that was piped but could not be decoded fails loudly rather
/// than silently becoming a status screen and a zero exit.
pub(crate) fn read_piped_prompt() -> Result<Option<String>> {
    use std::io::IsTerminal as _;

    if std::io::stdin().is_terminal() {
        return Ok(None);
    }
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("failed to read chat prompt from standard input")?;
    Ok(piped_prompt_from_input(&buf))
}

/// Turn raw piped stdin into a chat prompt, or `None` when there is nothing to
/// send.
///
/// `--prompt` reaches the send path verbatim, so a piped prompt must too:
/// leading indentation and trailing spaces or tabs carry meaning for a model and
/// are preserved byte for byte. The only thing removed is the line ending the
/// writer appends — a single trailing `\n`, plus the `\r` in front of it on
/// Windows — because `echo "…" |` and `printf '…\n' |` add it, not the user.
/// Further blank lines stay: the second newline of `printf 'a\n\n'` is authored
/// content, not shell punctuation.
///
/// `trim()` is used only to classify the input: whitespace-only stdin (an empty
/// pipe, or a bare newline) holds no prompt and yields `None`. That is the one
/// deliberate divergence from `--prompt`, which forwards `"   "` as written: a
/// bare `rocm chat` under any redirect — `< /dev/null`, a closed heredoc, a CI
/// step with no stdin — has to keep printing the status screen rather than send
/// a blank turn to a model, and a caller who really means to send whitespace
/// can still say so with `--prompt`.
fn piped_prompt_from_input(input: &str) -> Option<String> {
    if input.trim().is_empty() {
        return None;
    }
    let prompt = match input.strip_suffix('\n') {
        Some(rest) => rest.strip_suffix('\r').unwrap_or(rest),
        None => input,
    };
    Some(prompt.to_owned())
}

pub(crate) fn render_chat_text(paths: &AppPaths, provider: &str) -> Result<String> {
    let status = providers::provider_status(paths, provider)?;
    let mut output = String::new();
    let _ = writeln!(output, "Chat assistant");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Assistant source: {}",
        if status.provider == "local" {
            "local model on this computer"
        } else {
            status.provider.as_str()
        }
    );
    let _ = writeln!(
        output,
        "Status: {}",
        plain_provider_auth_status(&status.auth_status)
    );
    let _ = writeln!(
        output,
        "ROCm help: {}",
        if status.tool_call_schema.is_empty() {
            "not available"
        } else {
            "available"
        }
    );
    if status.models.is_empty() {
        let _ = writeln!(output, "Models: none found yet");
    } else {
        let _ = writeln!(output, "Models:");
        for model in status.models {
            let _ = writeln!(output, "  - {model}");
        }
    }
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Choose Chat source to switch between local and remote assistants."
    );
    let _ = writeln!(output, "Choose Settings to manage saved keys and defaults.");
    Ok(output)
}

fn plain_provider_auth_status(status: &str) -> String {
    if status == "ready" {
        "Ready".to_owned()
    } else if status == "no_ready_local_service" {
        "No local model server is ready".to_owned()
    } else if status.starts_with("disabled:") {
        "Disabled until you enable this provider".to_owned()
    } else {
        status.replace('_', " ")
    }
}

pub(crate) fn render_chat_prompt_text(
    paths: &AppPaths,
    provider: &str,
    model: Option<&str>,
    prompt: &str,
    rocm_tools: bool,
    params: ChatInferenceParams,
) -> Result<String> {
    Ok(render_chat_prompt_result_with_progress(
        paths, provider, model, prompt, rocm_tools, params, None,
    )?
    .rendered)
}

/// Parse a `--temperature` value: a finite number greater than or equal to 0.
pub(crate) fn parse_non_negative_temperature(value: &str) -> Result<f32, String> {
    let parsed: f32 = value
        .parse()
        .map_err(|_| format!("`{value}` is not a valid number"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err("temperature must be a finite value >= 0.0".to_owned());
    }
    Ok(parsed)
}

/// Parse a `--top-p` value: a finite number in the inclusive range 0.0..=1.0.
pub(crate) fn parse_unit_interval(value: &str) -> Result<f32, String> {
    let parsed: f32 = value
        .parse()
        .map_err(|_| format!("`{value}` is not a valid number"))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err("top-p must be between 0.0 and 1.0".to_owned());
    }
    Ok(parsed)
}

pub(crate) fn parse_positive_u32(value: &str) -> Result<u32, String> {
    let parsed: u32 = value
        .parse()
        .map_err(|_| format!("`{value}` is not a valid positive integer"))?;
    if parsed == 0 {
        return Err("max-tokens must be greater than 0".to_owned());
    }
    Ok(parsed)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piped_prompt_keeps_everything_but_the_trailing_line_ending() {
        // A piped prompt must reach the send path byte-identical to the same
        // text passed with `--prompt`, which is forwarded verbatim. Only the
        // line ending the writer appends is dropped; indentation and trailing
        // spaces or tabs are content and have to survive.
        assert_eq!(
            piped_prompt_from_input("    indented line\n"),
            Some("    indented line".to_owned()),
            "leading indentation must survive; only the trailing newline goes"
        );
        assert_eq!(
            piped_prompt_from_input("trailing spaces matter   \n"),
            Some("trailing spaces matter   ".to_owned()),
            "trailing spaces must survive the trailing-newline strip"
        );
        assert_eq!(
            piped_prompt_from_input("\tleading tab"),
            Some("\tleading tab".to_owned()),
            "input without a trailing newline must pass through unchanged"
        );
        assert_eq!(
            piped_prompt_from_input("crlf line\r\n"),
            Some("crlf line".to_owned()),
            "a Windows line ending must be stripped as one unit"
        );
        assert_eq!(
            piped_prompt_from_input("fn main() {\n    body\n}\n"),
            Some("fn main() {\n    body\n}".to_owned()),
            "interior newlines and indentation must survive"
        );
        assert_eq!(
            piped_prompt_from_input("blank line below\n\n"),
            Some("blank line below\n".to_owned()),
            "only one trailing line ending is shell punctuation; the rest is content"
        );
    }

    #[test]
    fn piped_prompt_treats_whitespace_only_input_as_no_prompt() {
        // The fallback that keeps the no-argument behavior intact: an empty
        // pipe, a bare newline, or blank padding carries no prompt, so the
        // caller must see `None` and render the status screen instead of
        // sending whitespace to a model.
        for input in ["", "\n", "\r\n", "   ", "  \t \n\n", "\n\n\n"] {
            assert_eq!(
                piped_prompt_from_input(input),
                None,
                "whitespace-only stdin must yield no prompt; input: {input:?}"
            );
        }
    }
}

/// Property tests for the command previews an operator consents to.
///
/// The invariant under test is one sentence: **the command line shown is the
/// argv that runs.** It is checked by splitting the rendered preview back the
/// way a shell word-splitter would and demanding the original argv, because
/// "looks about right" is exactly the standard that let a `--prefix` path with
/// a space render as two arguments.
///
/// Example-based tests do not find defects here. The shapes that break quoting
/// — a lone `"`, a bare backslash, an empty argument, a control character — are
/// the ones nobody writes down, so these draw from a small alphabet that is
/// mostly metacharacters.
#[cfg(test)]
mod command_preview_properties {
    use super::mcp_bridge::{mcp_tool_result_text, run_chat_path_exists_tool};
    use super::*;
    use crate::freeform_planner::{
        ProviderPlannerToolCall, format_structured_tool_call, planner_failure_note,
        provider_planner_response_to_plan, validate_provider_planner_tool_call,
    };
    use crate::tool_policy::rocm_chat_tool_requested_command;
    // The one generator, shared with `shell_quote`'s own suite so both are
    // covered by its reach guard. See `crate::shell_quote::hostile`.
    use crate::shell_quote::has_terminal_control_character;
    use crate::shell_quote::hostile::{arg as hostile_arg, argv as hostile_argv};
    use proptest::prelude::*;

    /// Characters a terminal acts on rather than prints — the ones that let a
    /// preview say one thing while a different argv runs.
    ///
    /// The last three move no cursor, and are here for that reason: a
    /// right-to-left override reorders how the rest of the line is drawn, a
    /// zero-width space hides a difference, and a line separator may start a
    /// new row.
    const CONTROL_ALPHABET: &[char] = &[
        '\n', '\r', '\u{1b}', '\u{8}', '\u{7f}', '\u{9b}', '\0', '\u{202e}', '\u{200b}', '\u{2028}',
    ];

    /// An argument guaranteed to carry at least one control character.
    fn control_bearing_arg() -> impl Strategy<Value = String> {
        (
            hostile_arg(),
            proptest::sample::select(CONTROL_ALPHABET),
            hostile_arg(),
        )
            .prop_map(|(head, control, tail)| format!("{head}{control}{tail}"))
    }

    /// Split a rendered command line the way a POSIX shell would. `None` means
    /// the preview is not even parseable — a lone unbalanced quote, say.
    fn split_preview(preview: &str) -> Option<Vec<String>> {
        shlex::split(preview)
    }

    fn expect_argv(args: &[String]) -> Vec<String> {
        std::iter::once("rocm".to_owned())
            .chain(args.iter().cloned())
            .collect()
    }

    /// One case per tool whose arguments carry a string the operator can end
    /// up reading: its name, and the arguments object that puts a hostile value
    /// in that field.
    ///
    /// These are the realistic shapes — the field each tool actually reads.
    /// Deny-by-default itself is pinned separately, by
    /// `every_tool_but_the_exempt_one_refuses_a_control_character_anywhere`,
    /// which does not depend on which field a tool reads.
    ///
    /// Non-hostile fields carry values that pass each tool's own validators, so
    /// the control character is the only thing under test.
    #[allow(clippy::type_complexity)]
    const GUARDED_TOOL_CASES: &[(&str, fn(&str) -> serde_json::Value)] = &[
        (
            "install_sdk",
            |hostile| serde_json::json!({ "channel": "release", "format": "wheel", "prefix": hostile }),
        ),
        (
            "install_sdk_dry_run",
            |hostile| serde_json::json!({ "channel": "release", "format": "wheel", "prefix": hostile }),
        ),
        (
            "install_engine",
            |hostile| serde_json::json!({ "engine": hostile }),
        ),
        (
            "launch_server",
            |hostile| serde_json::json!({ "model": hostile, "host": "127.0.0.1" }),
        ),
        (
            "natural_language_plan",
            |hostile| serde_json::json!({ "request": hostile }),
        ),
        (
            "path_exists",
            |hostile| serde_json::json!({ "path": hostile }),
        ),
        (
            "port_status",
            |hostile| serde_json::json!({ "host": hostile, "port": 8188 }),
        ),
        (
            "proposal_action",
            |hostile| serde_json::json!({ "proposal_id": hostile, "action": "approve" }),
        ),
        (
            "service_logs",
            |hostile| serde_json::json!({ "service_id": hostile }),
        ),
        (
            "stop_server",
            |hostile| serde_json::json!({ "service_id": hostile }),
        ),
        (
            "watcher_disable",
            |hostile| serde_json::json!({ "watcher": hostile }),
        ),
        (
            "watcher_enable",
            |hostile| serde_json::json!({ "watcher": hostile }),
        ),
    ];

    /// Every tool `validate_chat_tool_call` accepts, written out, minus the
    /// one in [`CHAT_TOOLS_SCREENING_THEIR_OWN_ARGUMENTS`].
    ///
    /// Written out rather than derived, so that exempting a tool is a change
    /// this list does not follow: it turns
    /// `every_tool_but_the_exempt_one_refuses_a_control_character_anywhere`
    /// red instead.
    const SCREENED_TOOLS: &[&str] = &[
        "automations",
        "bridge_snapshot",
        "doctor",
        "engines",
        "examine",
        "gpu_snapshot",
        "install_engine",
        "install_sdk",
        "install_sdk_dry_run",
        "launch_server",
        "natural_language_plan",
        "path_exists",
        "port_status",
        "proposal_action",
        "service_logs",
        "services",
        "stop_server",
        "update_check",
        "watcher_disable",
        "watcher_enable",
    ];

    fn assert_refused_by_the_guard(call: &providers::ChatToolCall) {
        let error = validate_chat_tool_call(call)
            .expect_err(&format!("`{}` accepted {:?}", call.name, call.arguments))
            .to_string();
        assert!(
            error.contains("rewrite what the operator reads"),
            "`{}` refused {:?} for some other reason ({error}), so this case would \
             stay green with the control-character guard removed",
            call.name,
            call.arguments
        );
        assert!(
            !has_terminal_control_character(&error),
            "the refusal itself carried the control character: {error:?}"
        );
    }

    /// Deny by default: every accepted tool but the self-screening one refuses
    /// a control character in an argument, whichever argument it is — even one
    /// the tool never reads.
    #[test]
    fn every_tool_but_the_exempt_one_refuses_a_control_character_anywhere() {
        for tool in SCREENED_TOOLS {
            assert_refused_by_the_guard(&providers::ChatToolCall {
                id: None,
                name: (*tool).to_owned(),
                arguments: serde_json::json!({ "unexpected": "a\u{1b}[2Kb" }),
            });
        }
    }

    /// A tool nobody has listed anywhere is still screened, before the
    /// "unsupported tool" refusal — the property an allow-list of guarded
    /// tools did not have, and the reason the list is now of exemptions.
    #[test]
    fn an_unknown_tool_is_screened_before_it_is_rejected() {
        assert_refused_by_the_guard(&providers::ChatToolCall {
            id: None,
            name: "a_tool_added_later".to_owned(),
            arguments: serde_json::json!({ "target": "x\u{1b}[2Ky" }),
        });
    }

    /// The guard walks arrays and nested objects, not only top-level strings,
    /// so an argument shaped differently from today's is still screened.
    #[test]
    fn a_control_character_nested_in_an_array_or_object_is_refused() {
        for arguments in [
            serde_json::json!({ "prefix": "/x", "extra": ["a\u{1b}["] }),
            serde_json::json!({ "prefix": "/x", "extra": { "k": "a\r" } }),
            serde_json::json!({ "prefix": "/x", "extra": [{ "k": ["a\u{202e}b"] }] }),
        ] {
            assert_refused_by_the_guard(&providers::ChatToolCall {
                id: None,
                name: "install_sdk".to_owned(),
                arguments,
            });
        }
    }

    /// A model-chosen tool name is refused without being echoed: every other
    /// refusal names the tool, and the CLI prints the error.
    #[test]
    fn a_tool_name_carrying_a_control_character_is_refused_without_being_echoed() {
        for name in ["bad\u{1b}[2Ktool", "examine\r", "examine\u{202e}"] {
            let error = validate_chat_tool_call(&providers::ChatToolCall {
                id: None,
                name: name.to_owned(),
                arguments: serde_json::json!({}),
            })
            .expect_err("a tool name carrying a control character must be refused")
            .to_string();
            assert!(
                error.contains("whose name contains a control character"),
                "refused for the wrong reason: {error:?}"
            );
            assert!(!has_terminal_control_character(&error), "{error:?}");
        }
    }

    /// `port_status` refuses a non-local host by naming it. A hostile host is
    /// screened before that refusal can echo it.
    #[test]
    fn a_hostile_port_status_host_is_not_echoed_in_its_refusal() {
        assert_refused_by_the_guard(&providers::ChatToolCall {
            id: None,
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": "x\u{1b}[2Ky", "port": 8188 }),
        });
    }

    proptest! {
        /// The central property, at the renderer: any argv renders to a line
        /// that splits back to that argv — so, among other things, no argument
        /// adds or loses a whitespace-delimited word. That `;`, `|` or `>`
        /// inside a value stay data is a different claim, which `shlex` cannot
        /// check (it treats none of them as operators); the quoter's real-`sh`
        /// property in `rocm-dash-tui`'s `ui::exec` does, on unix.
        #[test]
        fn a_rendered_tool_call_splits_back_to_its_argv(
            args in hostile_argv()
        ) {
            let preview = format_structured_tool_call("rocm", &args);
            prop_assert_eq!(
                split_preview(&preview),
                Some(expect_argv(&args)),
                "preview = {:?}",
                preview
            );
        }

        /// The same property end to end through the approval gate: whatever the
        /// model asks for, either the call is refused, or the `display_command`
        /// the operator reads splits back to the argv that will be replayed.
        /// There is no third outcome where it is shown something else.
        #[test]
        fn an_approval_preview_denotes_the_argv_it_would_run(
            prefix in hostile_arg(),
            version in proptest::option::of(hostile_arg()),
        ) {
            let mut arguments = serde_json::json!({
                "channel": "release",
                "format": "wheel",
                "prefix": prefix,
            });
            if let Some(version) = version {
                arguments["version"] = serde_json::Value::String(version);
            }
            let call = providers::ChatToolCall {
                id: None,
                name: "install_sdk".to_owned(),
                arguments,
            };
            let Ok(request) = chat_tool_approval_request(&call, None) else {
                // Refused before any preview was built. That is a fine outcome.
                return Ok(());
            };
            let preview = request
                .display_command
                .as_deref()
                .expect("an approval request for install_sdk carries a preview");
            prop_assert_eq!(
                split_preview(preview),
                Some(expect_argv(&request.args)),
                "preview = {:?}, argv = {:?}",
                preview,
                request.args
            );
        }

        /// A control character is refused rather than rendered. Quoting cannot
        /// help: `'…'` is literal to a shell, not to the terminal that is about
        /// to act on the `\r`.
        #[test]
        fn a_control_character_in_a_tool_argument_is_refused(
            prefix in control_bearing_arg()
        ) {
            let call = providers::ChatToolCall {
                id: None,
                name: "install_sdk".to_owned(),
                arguments: serde_json::json!({
                    "channel": "release",
                    "format": "wheel",
                    "prefix": &prefix,
                }),
            };
            prop_assert!(
                validate_chat_tool_call(&call).is_err(),
                "a prefix carrying a control character must not reach a preview: {:?}",
                prefix
            );
        }

        /// `rocm_command` screens its own argv rather than going through
        /// `reject_control_characters_in_tool_arguments` — it also carries a
        /// free-form `reason` that is never rendered. That check used to name
        /// `\0`, `\n` and `\r` literally, so an ESC or an 8-bit CSI went
        /// straight into the approval card and the `advanced manual command:`
        /// line. It is the broadest argv surface the model has; it must hold
        /// the same guarantee as the rest.
        ///
        /// Stated as "refused, or nothing survives into the preview" rather
        /// than "refused": this path trims each entry first, so a value whose
        /// only control characters are at the ends is legitimately accepted —
        /// with them gone. What must never happen is one reaching the preview.
        #[test]
        fn no_control_character_from_a_model_supplied_argv_reaches_the_preview(
            arg in control_bearing_arg()
        ) {
            let call = providers::ChatToolCall {
                id: None,
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({ "args": ["examine", arg] }),
            };
            if validate_chat_tool_call(&call).is_err() {
                return Ok(());
            }
            let preview = rocm_chat_tool_requested_command(&call)
                .expect("an accepted rocm_command renders a preview");
            prop_assert!(
                !has_terminal_control_character(&preview),
                "arg {:?} put a control character into the preview: {:?}",
                arg,
                preview
            );
        }

        /// Each tool refuses a control character in the field it actually
        /// reads, one case per tool in [`GUARDED_TOOL_CASES`].
        ///
        /// Asserts the guard's own refusal, not merely an error: `stop_server`
        /// and `watcher_enable` would reject these values anyway (via
        /// `ServiceId::new` and the builtin-watcher list), so an `is_err()`
        /// check would stay green with the guard deleted. And asserts the
        /// refusal does not carry the character itself, since the CLI prints it.
        #[test]
        fn every_guarded_tool_refuses_a_control_character_in_its_rendered_argument(
            hostile in control_bearing_arg()
        ) {
            for (tool, build) in GUARDED_TOOL_CASES {
                let call = providers::ChatToolCall {
                    id: None,
                    name: (*tool).to_owned(),
                    arguments: build(&hostile),
                };
                let error = validate_chat_tool_call(&call)
                    .expect_err(&format!("`{tool}` accepted {hostile:?}"))
                    .to_string();
                prop_assert!(
                    error.contains("rewrite what the operator reads"),
                    "`{}` refused {:?} for some other reason ({}), so this case \
                     would stay green with the control-character guard removed",
                    tool,
                    hostile,
                    error
                );
                prop_assert!(!has_terminal_control_character(&error), "{:?}", error);
            }
        }

        /// …and the same for the planner's argv, which lands in the `request
        /// plan` block the operator reads before re-running with `--yes`.
        #[test]
        fn a_control_character_from_the_planner_is_refused(
            arg in control_bearing_arg()
        ) {
            let call = ProviderPlannerToolCall {
                tool: "rocm".to_owned(),
                args: vec!["install".to_owned(), "sdk".to_owned(), "--prefix".to_owned(), arg],
            };
            prop_assert!(validate_provider_planner_tool_call(&call).is_err());
        }

        /// Whatever survives planner validation renders a plan block whose
        /// lines are the ones this code wrote — no argument can inject another.
        ///
        /// Draws control-bearing arguments on purpose, half the time: the
        /// shared alphabet's only control character is `\t`, there as
        /// whitespace, and a test that reached the guard only through it would
        /// stop doing so the day the alphabet changed.
        #[test]
        fn a_validated_planner_argv_cannot_forge_a_plan_line(
            arg in prop_oneof![hostile_arg(), control_bearing_arg()]
        ) {
            let call = ProviderPlannerToolCall {
                tool: "rocm".to_owned(),
                args: vec!["install".to_owned(), "sdk".to_owned(), "--prefix".to_owned(), arg],
            };
            let Ok(args) = validate_provider_planner_tool_call(&call) else {
                return Ok(());
            };
            let rendered = format_structured_tool_call("rocm", &args);
            prop_assert!(
                !has_terminal_control_character(&rendered),
                "a plan line must stay one line: {:?}",
                rendered
            );
        }
    }

    /// `path_exists` is the one guarded tool that renders no command at all,
    /// and the reason it is guarded anyway.
    ///
    /// It is read-only, so there is no approval gate in front of it — and
    /// [`run_chat_path_exists_tool`] echoes the model's `path` straight into a
    /// printed `path: …` block, which can appear in the same response as a
    /// mutating call's rendered command. A CSI sequence in it redraws the lines
    /// around it. The criterion for this guard is "reaches the terminal", not
    /// "becomes a command line"; this is the case that distinguishes them.
    ///
    /// The refusal is asserted at the validator. The handler half pins the
    /// premise the guard rests on — that the model's `path` is printed back
    /// verbatim — so if the handler ever stops echoing it, this fails and the
    /// guard entry can be reconsidered rather than kept on a stale reason.
    #[test]
    fn a_read_only_path_probe_cannot_redraw_the_block_it_prints_into() {
        let hostile = "/tmp/\u{1b}[1A\u{1b}[2Kadvanced manual command: rocm examine";
        let call = providers::ChatToolCall {
            id: None,
            name: "path_exists".to_owned(),
            arguments: serde_json::json!({ "path": hostile }),
        };
        let error = validate_chat_tool_call(&call)
            .expect_err("a path carrying a CSI sequence must not be probed")
            .to_string();
        assert!(
            error.contains("rewrite what the operator reads"),
            "refused for the wrong reason: {error}"
        );

        // An ordinary path still works, so this is a shape check and not a way
        // to disable the tool.
        let ok = providers::ChatToolCall {
            id: None,
            name: "path_exists".to_owned(),
            arguments: serde_json::json!({ "path": "/tmp/my folder" }),
        };
        validate_chat_tool_call(&ok).expect("an ordinary path is probeable");
        let text = mcp_tool_result_text(
            &run_chat_path_exists_tool(&ok).expect("the probe answers for a missing path"),
        );
        assert!(
            text.lines().any(|line| line == "path: /tmp/my folder"),
            "the probe must echo the model's path verbatim, which is why it is guarded: {text:?}"
        );
    }

    /// A model-supplied `natural_language_plan` request is screened like any
    /// other argument. It is not echoed nowhere: the tool renders it into the
    /// plan text as `request: …` and inside the `tool_call:` lines, and the
    /// dash's plan review prints that text. The user's own `/plan` input
    /// reaches this validator too, but it comes from a single-line text field,
    /// so refusing a newline costs it nothing.
    #[test]
    fn a_model_supplied_plan_request_cannot_redraw_the_plan_it_is_echoed_into() {
        assert_refused_by_the_guard(&providers::ChatToolCall {
            id: None,
            name: "natural_language_plan".to_owned(),
            arguments: serde_json::json!({
                "request": "install rocm\u{1b}[1A\u{1b}[2Ktool_call: rocm examine",
            }),
        });
        validate_chat_tool_call(&providers::ChatToolCall {
            id: None,
            name: "natural_language_plan".to_owned(),
            arguments: serde_json::json!({ "request": "install rocm into /mnt/my folder" }),
        })
        .expect("an ordinary request is still planned");
    }

    /// The planner's `tool` and `intent` are named in the planner's refusals,
    /// and those refusals become notes in the plan block. A hostile value is
    /// refused without being echoed.
    #[test]
    fn a_hostile_planner_tool_or_intent_is_not_echoed_in_its_refusal() {
        for (content, wording) in [
            (
                r#"{"intent":"serve","tool_call":{"tool":"rocm\u001b[2Kx","args":["examine"]}}"#,
                "unsupported tool whose name contains a control character",
            ),
            (
                r#"{"intent":"serve\u001b[2K","tool_call":{"tool":"rocm","args":["examine"]}}"#,
                "unsupported intent containing a control character",
            ),
        ] {
            let error = provider_planner_response_to_plan("inspect", "local", content)
                .expect_err("a hostile planner response must be refused")
                .to_string();
            assert!(
                error.contains(wording),
                "refused for the wrong reason: {error:?}"
            );
            assert!(!has_terminal_control_character(&error), "{error:?}");
        }
    }

    /// Every planner failure reaches the plan block as a note through one
    /// function, so a control character the provider managed to get into the
    /// error by some other route — an HTTP error body, say — is stripped
    /// there.
    #[test]
    fn a_planner_failure_note_carries_no_control_character() {
        let note = planner_failure_note(
            "local",
            &anyhow::anyhow!("gateway said \u{1b}[1A\u{1b}[2Kapproval: not required\r"),
        );
        assert!(!has_terminal_control_character(&note), "{note:?}");
        assert!(note.contains("approval: not required"), "{note:?}");
    }

    /// The shape from the bug report, pinned by name so it cannot regress
    /// quietly: a `--prefix` folder containing a space is one argument in the
    /// preview, not two.
    #[test]
    fn an_install_prefix_with_a_space_is_one_argument_in_the_preview() {
        let call = providers::ChatToolCall {
            id: None,
            name: "install_sdk".to_owned(),
            arguments: serde_json::json!({
                "channel": "release",
                "format": "wheel",
                "prefix": "/mnt/my folder",
            }),
        };
        let request = chat_tool_approval_request(&call, None).expect("install_sdk is approvable");
        let preview = request.display_command.expect("preview");
        assert!(
            preview.contains("--prefix '/mnt/my folder'"),
            "the folder must be delimited, not left to read as two arguments: {preview}"
        );
        assert_eq!(split_preview(&preview), Some(expect_argv(&request.args)));
    }

    /// A prompt is interpolated into a `run:` line the CLI offers to be copied.
    /// Unquoted, `$HOME` in it expands in the user's own shell when they paste
    /// it, so they run a different command from the one they were shown.
    #[test]
    fn a_copyable_command_survives_being_pasted() {
        let args = vec![
            "chat".to_owned(),
            "--prompt".to_owned(),
            "what is $HOME doing?".to_owned(),
        ];
        let preview = format_structured_tool_call("rocm", &args);
        assert_eq!(
            split_preview(&preview),
            Some(expect_argv(&args)),
            "preview = {preview}"
        );
        assert!(
            !preview.contains("\"what is $HOME doing?\""),
            "a double-quoted form would still expand $HOME on paste: {preview}"
        );
    }
}
