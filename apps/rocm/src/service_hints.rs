// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The one place every `rocm services <action> <id> --yes` hint is built.
//!
//! `main.rs`, `remote/mod.rs`, and `serve_summary.rs` each used to format this
//! string by hand, and twice that drifted: a hint printed without `--yes` even
//! though the command it names refuses without it (#417, #601, #604). A
//! caller that needs this text calls a function here instead of writing
//! another `format!`.

/// The directly-runnable `rocm services <action> <id> --yes` hint for a local
/// managed service. `action` is `"stop"` or `"restart"` — whatever
/// [`crate::service_action_command`] returns for the gate in question.
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
