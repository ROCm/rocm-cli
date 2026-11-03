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
//! `SandboxToolArg`. They were relocated verbatim from `main.rs`;
//! `services_cmd.rs`'s `run_approved_service_action` calls all four, and
//! `services_text.rs` reads `service_action_past_tense` from here.
//!
//! The service id goes through [`shell_quote`] here, once, so every hint is a
//! line whose words are exactly the argv it names: an id that is not a single
//! inert word still pastes back as one argument. An ordinary id is inert and is
//! printed unchanged.

use anyhow::{Result, bail};

use crate::SandboxToolArg;
use crate::shell_quote::shell_quote;

/// The directly-runnable `rocm services <action> <id> --yes` hint for a local
/// managed service. `action` is typically `"stop"` or `"restart"` — whatever
/// [`service_action_command`] returns for the gate in question — but
/// any `rocm services` subcommand name that takes `--yes` works, e.g.
/// `"remove"`.
pub(crate) fn service_action_hint(action: &str, service_id: &str) -> String {
    format!("rocm services {action} {} --yes", shell_quote(service_id))
}

/// The directly-runnable `rocm services stop <id> --yes` hint.
pub(crate) fn service_stop_hint(service_id: &str) -> String {
    service_action_hint("stop", service_id)
}

/// The directly-runnable `<remote_cli> services stop <id> --yes` hint for a
/// service on a remote machine, where `remote_cli` is that machine's own
/// `rocm` binary name (not necessarily `"rocm"` — it is whatever the remote
/// session recorded). `service_id` is quoted here, so the same line is both
/// the human-readable hint and the command a transport's `exec` hands to the
/// remote shell; callers pass it unquoted. `remote_cli` is not quoted: it is
/// the command word itself, which [`shell_quote`] does not cover (see
/// [`rocm_core::posix_shell_quote`]'s note on `=`).
pub(crate) fn remote_service_stop_hint(remote_cli: &str, service_id: &str) -> String {
    format!(
        "{remote_cli} services stop {} --yes",
        shell_quote(service_id)
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shell_quote::hostile::arg as hostile_arg;
    use proptest::prelude::*;

    // An ordinary id is inert, so every hint the tests elsewhere pin
    // byte-for-byte is unchanged by the quoting.
    #[test]
    fn an_ordinary_service_id_is_printed_unchanged() {
        assert_eq!(
            service_stop_hint("vllm-qwen-8000"),
            "rocm services stop vllm-qwen-8000 --yes"
        );
        assert_eq!(
            remote_service_stop_hint("rocm", "vllm-qwen-8000"),
            "rocm services stop vllm-qwen-8000 --yes"
        );
    }

    proptest! {
        // Mutation: printing `service_id` bare in either builder fails this
        // for any id carrying whitespace or a quote.
        #[test]
        fn a_hint_denotes_the_argv_it_names(service_id in hostile_arg()) {
            for (hint, program) in [
                (service_action_hint("restart", &service_id), "rocm"),
                (remote_service_stop_hint("rocm-remote", &service_id), "rocm-remote"),
            ] {
                let action = if program == "rocm" { "restart" } else { "stop" };
                prop_assert_eq!(
                    shlex::split(&hint),
                    Some(vec![
                        program.to_owned(),
                        "services".to_owned(),
                        action.to_owned(),
                        service_id.clone(),
                        "--yes".to_owned(),
                    ]),
                    "{}",
                    hint
                );
            }
        }
    }
}
