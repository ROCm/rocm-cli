// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Say why a `rocm serve` run printed no `resolved model:` line.
//!
//! `rocm serve` prints its plan (`serve plan` / `resolved model: ...`) only after a
//! chain of refusals has passed, among them, in the order `serve_cmd.rs` runs
//! them: the OS gate that rules an engine out (vLLM on native Windows), the
//! no-usable-GPU bail, the Docker-aware engine gate, the "no active ROCm runtime"
//! bail and the engine's model-resolution call.
//! A run that stops at any of them leaves stdout without the line, and the refusal
//! goes to stderr. The short-name scenarios used to keep stdout only, so every
//! such run read as "the name was not expanded" whatever actually stopped it. The
//! lane logs show exactly that: a vLLM serve whose stdout was empty, with nothing
//! to say which refusal it was.

/// One `rocm serve` run: the engine it asked for and everything it produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeRun {
    pub engine: String,
    pub stdout: String,
    pub stderr: String,
    pub rc: i32,
}

impl ServeRun {
    /// The model named by the plan's `resolved model:` line, if the plan was
    /// printed at all.
    #[must_use]
    pub fn resolved_model(&self) -> Option<&str> {
        resolved_model(&self.stdout)
    }

    /// What to say when [`resolved_model`](Self::resolved_model) is `None`.
    ///
    /// Names the engine, the exit code and the first line of stderr, so a refusal
    /// reads as a refusal and a run that exited 0 without a plan reads as that.
    #[must_use]
    pub fn missing_plan_message(&self) -> String {
        let engine = &self.engine;
        let rc = self.rc;
        let stderr_line = self
            .stderr
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty());
        let verdict = match (rc, stderr_line) {
            (0, _) => format!(
                "`rocm serve --engine {engine}` exited 0 but printed no serve plan, so the name was never shown"
            ),
            (_, Some(line)) => format!(
                "`rocm serve --engine {engine}` stopped before it printed a serve plan (exit {rc}): {line}"
            ),
            (_, None) => format!(
                "`rocm serve --engine {engine}` stopped before it printed a serve plan (exit {rc}) and wrote nothing to stderr"
            ),
        };
        format!(
            "no 'resolved model' in output: {verdict}\n\n{}\n\n{}",
            crate::section("stdout", &self.stdout),
            crate::section("stderr", &self.stderr)
        )
    }
}

/// The model a serve plan resolved to (`resolved model: <id>`), if it printed one.
///
/// It can differ from the requested id: `rocm serve <model>` with no `--engine`
/// picks the recipe's preferred model and engine. `None` means the plan has no
/// such line, for example because the serve stopped before resolving.
#[must_use]
pub fn resolved_model(output: &str) -> Option<&str> {
    output
        .lines()
        .find_map(|l| l.trim().strip_prefix("resolved model:"))
        .map(str::trim)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attempt(engine: &str, stdout: &str, stderr: &str, rc: i32) -> ServeRun {
        ServeRun {
            engine: engine.to_owned(),
            stdout: stdout.to_owned(),
            stderr: stderr.to_owned(),
            rc,
        }
    }

    #[test]
    fn a_printed_plan_names_its_resolved_model() {
        let plan = "serve plan\n  requested model: qwen2.5\n  resolved model: Qwen/Qwen2.5-1.5B-Instruct\n  engine: lemonade\n";
        assert_eq!(
            attempt("lemonade", plan, "", 0).resolved_model(),
            Some("Qwen/Qwen2.5-1.5B-Instruct")
        );
    }

    #[test]
    fn a_refusal_is_reported_as_a_refusal_with_its_stderr_line() {
        let refused = attempt(
            "vllm",
            "",
            "\nError: no active ROCm runtime is configured; run `rocm runtimes list`\n",
            1,
        );
        assert_eq!(refused.resolved_model(), None);
        let message = refused.missing_plan_message();
        assert!(
            message.contains("`rocm serve --engine vllm` stopped before it printed a serve plan (exit 1): Error: no active ROCm runtime is configured"),
            "{message}"
        );
        assert!(
            !message.contains("exited 0"),
            "a non-zero exit must not read as a clean one: {message}"
        );
    }

    #[test]
    fn an_exit_zero_without_a_plan_is_not_called_a_refusal() {
        let message = attempt("lemonade", "", "warning: ignored", 0).missing_plan_message();
        assert!(
            message.contains("exited 0 but printed no serve plan"),
            "{message}"
        );
        assert!(!message.contains("stopped before"), "{message}");
    }

    #[test]
    fn a_silent_failure_says_stderr_was_empty() {
        let message = attempt("vllm", "", "", 1).missing_plan_message();
        assert!(
            message.contains("(exit 1) and wrote nothing to stderr"),
            "{message}"
        );
        assert!(
            message.contains("--- stdout: (empty) ---"),
            "an empty stream is marked, not left as a bare heading: {message}"
        );
    }

    #[test]
    fn blank_stderr_lines_are_skipped_and_the_reported_line_is_trimmed() {
        let indented = attempt("vllm", "", "   \n\t\n  Error: refused\n", 1).missing_plan_message();
        assert!(indented.contains("(exit 1): Error: refused"), "{indented}");
        let blank_only = attempt("vllm", "", "  \n", 1).missing_plan_message();
        assert!(
            blank_only.contains("(exit 1) and wrote nothing to stderr"),
            "whitespace is not a stderr line: {blank_only}"
        );
    }
}
