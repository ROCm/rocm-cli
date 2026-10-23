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
//! The service id goes through [`shell_quote`] here, once, so every hint is a
//! line whose words are exactly the argv it names: an id that is not a single
//! inert word still pastes back as one argument. An ordinary id is inert and is
//! printed unchanged.

use crate::shell_quote::shell_quote;

/// The directly-runnable `rocm services <action> <id> --yes` hint for a local
/// managed service. `action` is typically `"stop"` or `"restart"` — whatever
/// [`crate::service_action_command`] returns for the gate in question — but
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
/// the command word itself, which [`shell_quote`] does not cover (see its note
/// on `=`).
pub(crate) fn remote_service_stop_hint(remote_cli: &str, service_id: &str) -> String {
    format!(
        "{remote_cli} services stop {} --yes",
        shell_quote(service_id)
    )
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
