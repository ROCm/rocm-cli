// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Argv helpers shared by `chat.rs`, `tool_policy.rs` and the freeform planner:
//! reading an option's value out of an argv slice (`chat_cli_arg_value`,
//! `chat_cli_arg_value_checked`, `chat_cli_has_flag`), appending a flag at
//! most once (`ensure_flag`), recognising a CPU device request in `serve`
//! arguments (`serve_args_request_cpu_device`), and rendering an argv as the
//! quoted one-line command shown in plans and prompts
//! (`format_structured_tool_call`).
//!
//! A leaf module whose only `crate::` dependency is the leaf `shell_quote.rs`:
//! `chat.rs`, `tool_policy.rs` and `freeform_planner.rs` all call into it, so
//! the planner does not have to import `chat` for these helpers. They were
//! relocated from `chat.rs` and `freeform_planner.rs`, which re-export the
//! items other modules still reach as `chat::ensure_flag` and
//! `freeform_planner::format_structured_tool_call`.

use crate::shell_quote::shell_quote;
use anyhow::{Result, bail};

pub(crate) fn ensure_flag(args: &mut Vec<String>, flag: &str) {
    if !args.iter().any(|arg| arg == flag) {
        args.push(flag.to_owned());
    }
}

pub(crate) fn chat_cli_arg_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if let Some((option, value)) = arg.split_once('=')
            && option == name
        {
            return Some(value);
        }
        if arg == name {
            return args.get(index + 1).map(String::as_str);
        }
        index += 1;
    }
    None
}

pub(crate) fn chat_cli_arg_value_checked<'a>(
    args: &'a [String],
    name: &str,
) -> Result<Option<&'a str>> {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if let Some((option, value)) = arg.split_once('=')
            && option == name
        {
            if value.trim().is_empty() {
                bail!("local assistant provided empty value for {name}");
            }
            return Ok(Some(value));
        }
        if arg == name {
            let Some(value) = args.get(index + 1).map(String::as_str) else {
                bail!("local assistant omitted the value for {name}");
            };
            if value.starts_with("--") || value.trim().is_empty() {
                bail!("local assistant omitted the value for {name}");
            }
            return Ok(Some(value));
        }
        index += 1;
    }
    Ok(None)
}

pub(crate) fn chat_cli_has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| {
        arg == name
            || arg
                .strip_prefix(name)
                .is_some_and(|rest| rest.starts_with('='))
    })
}

pub(crate) fn device_policy_is_cpu_mode(policy: &str) -> bool {
    matches!(
        policy.trim().to_ascii_lowercase().as_str(),
        "cpu" | "cpu_only"
    )
}

pub(crate) fn serve_args_request_cpu_device(args: &[String]) -> bool {
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if let Some(value) = arg.strip_prefix("--device=")
            && device_policy_is_cpu_mode(value)
        {
            return true;
        }
        if arg == "--device"
            && let Some(value) = iter.peek()
            && device_policy_is_cpu_mode(value)
        {
            return true;
        }
    }
    false
}

/// Render an argv as the command line a human is shown for it.
///
/// This is the only thing an operator sees before approving a mutating tool
/// call, and some of its output (`rocm chat --prompt …`, the `install command:`
/// line) is offered to be copied and run. So it has to denote the argv it was
/// built from: quoted with [`shell_quote`], every argument splits back to
/// itself under a shell's own word-splitting, and nothing inside a value can
/// read as a second command.
///
/// Values carrying terminal control characters never get here — they are
/// refused at validation
/// ([`has_terminal_control_character`](crate::shell_quote::has_terminal_control_character)),
/// because no amount of quoting stops a `\r` from rewriting the line the
/// operator is reading.
pub(crate) fn format_structured_tool_call(tool: &str, args: &[String]) -> String {
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(tool.to_owned());
    parts.extend(args.iter().map(|arg| shell_quote(arg)));
    parts.join(" ")
}
