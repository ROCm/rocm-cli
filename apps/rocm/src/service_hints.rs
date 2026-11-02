// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The one place every `rocm services <action> <id> --yes` hint is built.
//!
//! `main.rs`, `remote/mod.rs`, and `serve_summary.rs` each used to format this
//! string by hand, and it drifted: `serve_summary.rs`'s stop row printed the
//! hint without `--yes`, even though the command it names refuses without it
//! (#417). A caller that needs this text calls a function here instead of
//! writing another `format!`.
//!
//! The module also holds the small helpers that turn a service tool name
//! (`stop_server`, `restart_server`) into the words those hints and messages
//! are built from — the command (`service_action_command`), the progressive
//! verb (`service_action_verb`) and the past tense (`service_action_past_tense`)
//! — plus `sandbox_tool_arg_from_service_tool`, which maps the tool name to its
//! `SandboxToolArg`. They were relocated verbatim from `main.rs`, where
//! `run_approved_service_action` calls all four; `services_text.rs` reads
//! `service_action_past_tense` from here.

use anyhow::{Result, bail};

use crate::SandboxToolArg;

/// The directly-runnable `rocm services <action> <id> --yes` hint for a local
/// managed service. `action` is typically `"stop"` or `"restart"` — whatever
/// [`service_action_command`] returns for the gate in question — but
/// any `rocm services` subcommand name that takes `--yes` works, e.g.
/// `"remove"`.
pub(crate) fn service_action_hint(action: &str, service_id: &str) -> String {
    format!("rocm services {action} {service_id} --yes")
}

/// The directly-runnable `rocm services stop <id> --yes` hint.
pub(crate) fn service_stop_hint(service_id: &str) -> String {
    service_action_hint("stop", service_id)
}

/// The directly-runnable `<remote_cli> services stop <id> --yes` hint for a
/// service on a remote machine, where `remote_cli` is that machine's own
/// `rocm` binary name (not necessarily `"rocm"` — it is whatever the remote
/// session recorded). Callers that feed the result to a transport's `exec`
/// shell-quote `service_id` themselves before calling this; callers building
/// human-readable text pass it unquoted.
pub(crate) fn remote_service_stop_hint(remote_cli: &str, service_id: &str) -> String {
    format!("{remote_cli} services stop {service_id} --yes")
}

pub(crate) fn sandbox_tool_arg_from_service_tool(tool: &str) -> Result<SandboxToolArg> {
    match tool {
        "stop_server" => Ok(SandboxToolArg::StopServer),
        "restart_server" => Ok(SandboxToolArg::RestartServer),
        "list_servers" => Ok(SandboxToolArg::ListServers),
        other => bail!("unsupported service tool `{other}`"),
    }
}

pub(crate) fn service_action_command(tool: &str) -> &'static str {
    match tool {
        "restart_server" => "restart",
        "stop_server" => "stop",
        _ => "run",
    }
}

pub(crate) fn service_action_verb(tool: &str) -> &'static str {
    match tool {
        "restart_server" => "Restarting",
        "stop_server" => "Stopping",
        _ => "Changing",
    }
}

pub(crate) fn service_action_past_tense(tool: &str) -> &'static str {
    match tool {
        "restart_server" => "restarted",
        "stop_server" => "stopped",
        _ => "updated",
    }
}
