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
//! A leaf module with no `crate::` dependencies: `chat.rs`, `tool_policy.rs`
//! and `freeform_planner.rs` all call into it, so the planner does not have to
//! import `chat` for these helpers. They were relocated verbatim from `chat.rs` and
//! `freeform_planner.rs`, which re-export the items other modules still reach
//! as `chat::ensure_flag` and `freeform_planner::format_structured_tool_call`.

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

pub(crate) fn format_structured_tool_call(tool: &str, args: &[String]) -> String {
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(tool.to_owned());
    parts.extend(args.iter().map(|arg| quote_tool_arg(arg)));
    parts.join(" ")
}

/// Quotes `value` for rendered plan text and LLM prompt text only -- never
/// executed directly. Deliberately kept separate from `rocm_core::quoting`
/// (ROCMAI-727); see that module's doc comment for why.
fn quote_tool_arg(value: &str) -> String {
    if value.is_empty() || value.chars().any(char::is_whitespace) {
        format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        value.to_owned()
    }
}
