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

/// The directly-runnable `rocm services <action> <id> --yes` hint for a local
/// managed service. `action` is typically `"stop"` or `"restart"` — whatever
/// [`crate::service_action_command`] returns for the gate in question — but
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

/// The `  <task> endpoint: <url>` line printed beside a server's `endpoint`
/// line when the server does not serve chat: chat clients append
/// `/chat/completions` to the base URL, which such a server does not have, so
/// this names the route it does answer. `None` for chat, and for a task this
/// build cannot name a route for. Every surface that prints it calls this.
pub(crate) fn task_endpoint_line(task: rocm_core::ModelTask, endpoint_url: &str) -> Option<String> {
    (!task.is_chat() && !task.api_route().is_empty()).then(|| {
        format!(
            "  {} endpoint: {endpoint_url}{}",
            task.display_name(),
            task.api_route()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::ModelTask;

    #[test]
    fn only_a_server_that_does_not_serve_chat_gets_a_task_endpoint_line() {
        assert_eq!(
            task_endpoint_line(ModelTask::Transcription, "http://127.0.0.1:11435/v1").as_deref(),
            Some("  speech-to-text endpoint: http://127.0.0.1:11435/v1/audio/transcriptions")
        );
        assert_eq!(task_endpoint_line(ModelTask::Chat, "http://h/v1"), None);
        assert_eq!(
            task_endpoint_line(ModelTask::Unknown, "http://h/v1"),
            None,
            "no route to name"
        );
    }
}
