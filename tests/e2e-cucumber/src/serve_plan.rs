// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Say why a `rocm serve` run printed no `resolved model:` line.
//!
//! `rocm serve` prints its plan (`serve plan` / `resolved model: ...`) only after a
//! chain of refusals has passed: the platform and engine gate, the no-usable-GPU
//! bail, the "no active ROCm runtime" bail and the engine's model-resolution call.
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
            "no 'resolved model' in output: {verdict}\n\nstdout:\n{}\n\nstderr:\n{}",
            self.stdout, self.stderr
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
    }
}
