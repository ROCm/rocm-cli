// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The shared `[y/N]` confirmation prompt.
//!
//! Before this module existed, `fix.rs`, `apps/rocm/src/uninstall_plan.rs` and
//! `apps/rocm/src/therock.rs` each hand-rolled the same print-prompt / flush
//! stdout / read a line / normalize / match `"y"`/`"yes"` sequence. This is
//! that sequence, once.
//!
//! What stays out of this module on purpose: `fix.rs`'s `confirm()` also
//! short-circuits on `--yes` and a non-interactive shell, and prints its own
//! refusal wording ("refusing to apply") when it declines. That wrapper is
//! specific to the `fix` recipe flow and is asserted verbatim by
//! `tests/e2e-cucumber/tests/e2e/diagnose_steps.rs`, so it stays local to
//! `fix.rs`, built on top of [`read_yes_no`] here. The other two callers
//! already guard the non-interactive/`--yes` case themselves before ever
//! reaching a prompt function, so they only need the raw read-and-parse
//! primitive, not a second copy of that short-circuit.

use std::io::Write;

/// Parse a user's typed response to a `[y/N]` prompt.
pub fn is_affirmative_answer(line: &str) -> bool {
    matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
}

/// Print `prompt`, read one line from stdin, and parse it as yes/no.
///
/// `prompt` is expected to already include the trailing `[y/N]: `. Only
/// errors if the prompt can't be written or the line can't be read; a
/// declined answer is `Ok(false)`, not an error.
pub fn read_yes_no(prompt: &str) -> std::io::Result<bool> {
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(is_affirmative_answer(&line))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_affirmative_answer_accepts_only_y_and_yes() {
        for accepted in ["y", "Y", "yes", "YES", "Yes", "  y  ", "  yes\n"] {
            assert!(
                is_affirmative_answer(accepted),
                "expected {accepted:?} to be treated as a yes"
            );
        }
    }

    #[test]
    fn is_affirmative_answer_rejects_everything_else() {
        for declined in ["n", "no", "", "\n", "yep", "ye"] {
            assert!(
                !is_affirmative_answer(declined),
                "expected {declined:?} to be treated as a decline"
            );
        }
    }
}
