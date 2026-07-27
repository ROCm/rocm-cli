// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Reading back the log a managed serve wrote, for failure diagnostics.
//!
//! When `rocm serve --managed` returns `readiness: starting` and the model never
//! appears on `/v1/models`, the harness sees only a timeout: the engine's own
//! reason (an out-of-memory at engine init, a stalled weight download, a runtime
//! that failed to import) is written to the service log in a per-scenario temp
//! directory that no report artifact captures. Quoting its tail into the failure
//! is what makes such a stall diagnosable from CI output alone.

/// How many trailing lines of a stalled service's log to quote in a failure.
const DEFAULT_TAIL_LINES: usize = 40;

/// The `log_path:` a `rocm serve --managed` plan reports for the service it
/// launched, if the output carries one.
fn parse_log_path(serve_stdout: &str) -> Option<&str> {
    serve_stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("log_path:"))
        .map(str::trim)
        .filter(|path| !path.is_empty())
}

/// The last `max` lines of `text`.
fn tail_lines(text: &str, max: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(max)..].join("\n")
}

/// The tail of the log written by the managed service `serve_stdout` launched.
///
/// Never fails: every reason the log can't be quoted (no `log_path` line, the
/// file is missing or unreadable, the engine wrote nothing) becomes part of the
/// returned text, because this only ever runs while reporting another failure.
#[must_use]
pub fn service_log_tail(serve_stdout: &str) -> String {
    let Some(path) = parse_log_path(serve_stdout) else {
        return "<no log_path in serve output>".to_owned();
    };
    match std::fs::read_to_string(path) {
        Ok(log) if log.trim().is_empty() => format!("<{path} is empty>"),
        Ok(log) => tail_lines(&log, DEFAULT_TAIL_LINES),
        Err(error) => format!("<failed to read {path}: {error}>"),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_log_path, service_log_tail, tail_lines};

    const SERVE_PLAN: &str = "\
serve plan
  requested model: Qwen/Qwen3.5-0.8B
  engine: vllm
managed service launched
  service_id: vllm-qwen-qwen3-5-0-8b-1785145816856
  log_path: /tmp/rocm-e2e-abc123/data/services/vllm-qwen.log
  readiness: starting
";

    #[test]
    fn log_path_is_read_from_the_indented_plan_line() {
        assert_eq!(
            parse_log_path(SERVE_PLAN),
            Some("/tmp/rocm-e2e-abc123/data/services/vllm-qwen.log")
        );
    }

    #[test]
    fn output_without_a_log_path_yields_none() {
        assert_eq!(parse_log_path("serve plan\n  engine: vllm\n"), None);
        assert_eq!(parse_log_path("  log_path:   \n"), None);
    }

    #[test]
    fn tail_keeps_the_last_lines_and_never_panics_on_short_input() {
        let text = (1..=5)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(tail_lines(&text, 2), "4\n5");
        assert_eq!(tail_lines(&text, 99), text);
        assert_eq!(tail_lines("", 3), "");
    }

    #[test]
    fn tail_of_a_real_file_is_quoted_verbatim() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("service.log");
        std::fs::write(&path, "boot\nENGINE ERROR: out of memory\n").expect("write log");
        let stdout = format!("managed service launched\n  log_path: {}\n", path.display());
        assert_eq!(
            service_log_tail(&stdout),
            "boot\nENGINE ERROR: out of memory"
        );
    }

    #[test]
    fn every_unreadable_case_explains_itself_instead_of_failing() {
        assert_eq!(
            service_log_tail("managed service launched\n"),
            "<no log_path in serve output>"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let empty = dir.path().join("empty.log");
        std::fs::write(&empty, "  \n").expect("write log");
        assert_eq!(
            service_log_tail(&format!("  log_path: {}\n", empty.display())),
            format!("<{} is empty>", empty.display())
        );

        let missing = dir.path().join("missing.log");
        let reported = service_log_tail(&format!("  log_path: {}\n", missing.display()));
        assert!(
            reported.starts_with(&format!("<failed to read {}", missing.display())),
            "unexpected message: {reported}"
        );
    }
}
