// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `Try:` footer a refusal ends with: the one command that clears it.
//!
//! A refusal that knows exactly what the user should run next ends with a
//! blank line and `Try: <command>`, so the advice is the last line of the
//! message and can be copied as it stands. `rocm` prints an error as
//! `{error:?}`, and with `RUST_BACKTRACE` set anyhow appends a
//! `Stack backtrace:` section after the message: the footer then ends the
//! message, not the screen. [`without_backtrace`] cuts that section off, and [`try_command`]
//! reads the footer through it.
//!
//! Every such footer is built here ([`with_try`] for a message being written,
//! [`advise`] for an error a shared guard already returned) and read back by
//! the one parser ([`try_command`]), so a test that runs the advised command
//! reads it the same way a user does.
//!
//! The footer names one command. Advice that needs two steps (a dry run, then
//! the real thing) is not a `Try:` footer and is written by hand.

use std::fmt::Display;

/// The label that opens the footer line.
pub const TRY_LABEL: &str = "Try:";

/// `message`, a blank line, and `Try: {command}` as its last line.
///
/// `command` is a single line: a footer spanning lines would not read back
/// through [`try_command`].
#[must_use]
pub fn with_try(message: impl Display, command: &str) -> String {
    debug_assert!(
        !command.contains('\n'),
        "a Try: footer names one command on one line: {command:?}"
    );
    format!("{message}\n\n{TRY_LABEL} {command}")
}

/// `error` with a `Try: {command}` footer appended, as a new error.
///
/// The footer has to be the last line, and `rocm` prints an error as
/// `Error: {error:?}`, which puts a `.context()` *before* its cause; so this
/// flattens `error` (`{error:#}`) into the message instead of wrapping it.
/// That drops the original error's type, so use it only on plain refusals,
/// never on an error a caller downcasts (one carrying an exit code).
#[must_use]
pub fn advise(error: anyhow::Error, command: &str) -> anyhow::Error {
    anyhow::anyhow!(with_try(format_args!("{error:#}"), command))
}

/// Where anyhow's `{:?}` rendering starts its backtrace section, when
/// `RUST_BACKTRACE` or `RUST_LIB_BACKTRACE` is set (CI sets it): a blank line,
/// then this label.
const BACKTRACE_SECTION_START: &str = "\n\nStack backtrace:";

/// `text` up to the backtrace section anyhow's `{:?}` rendering appends, with
/// the blank line before it; all of `text` when it has none.
///
/// Only the label after a blank line starts the section, so a message that
/// merely mentions the words keeps them.
#[must_use]
pub fn without_backtrace(text: &str) -> &str {
    text.split_once(BACKTRACE_SECTION_START)
        .map_or(text, |(message, _)| message)
}

/// The command a [`with_try`] footer names, when `text` ends with one.
///
/// Only the last non-blank line of the message counts: a footer is the end of
/// the message, and a `Try:` line anywhere else is not one. A backtrace
/// section after the message (see the module docs) is not part of it.
#[must_use]
pub fn try_command(text: &str) -> Option<&str> {
    without_backtrace(text)
        .trim_end()
        .lines()
        .next_back()?
        .strip_prefix(TRY_LABEL)?
        .strip_prefix(' ')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_try_ends_the_message_with_a_blank_line_and_the_command() {
        assert_eq!(
            with_try("uninstall requires --yes", "rocm uninstall --yes"),
            "uninstall requires --yes\n\nTry: rocm uninstall --yes"
        );
    }

    #[test]
    fn try_command_reads_back_what_with_try_wrote() {
        let text = with_try("first line\nsecond line", "rocm services stop svc --yes");
        assert_eq!(try_command(&text), Some("rocm services stop svc --yes"));
        // A trailing newline, as a captured stderr has, changes nothing.
        assert_eq!(
            try_command(&format!("Error: {text}\n")),
            Some("rocm services stop svc --yes")
        );
    }

    #[test]
    fn try_command_ignores_trailing_whitespace_after_the_footer() {
        // Blank and whitespace-only lines after the footer do not hide it:
        // the footer is the last non-blank line.
        assert_eq!(
            try_command("refused\n\nTry: rocm uninstall --yes\n\n  \n"),
            Some("rocm uninstall --yes")
        );
        // Trailing spaces on the footer line are not part of the command.
        assert_eq!(
            try_command("refused\n\nTry: rocm uninstall --yes  "),
            Some("rocm uninstall --yes")
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a Try: footer names one command on one line")]
    fn with_try_refuses_a_command_spanning_lines() {
        // A footer over two lines would not read back through `try_command`,
        // so debug builds catch the mistake where the footer is built.
        let _ = with_try(
            "refused",
            "rocm services prune --dry-run\nrocm services prune --yes",
        );
    }

    #[test]
    fn try_command_is_none_without_a_footer() {
        assert_eq!(try_command("uninstall requires --yes"), None);
        assert_eq!(try_command(""), None);
        // Not the last line: a two-step hint, or a context added afterwards.
        assert_eq!(
            try_command(
                "refused\n\nTry: rocm services prune --dry-run\nThen: rocm services prune --yes"
            ),
            None
        );
        // The label alone, or glued to the command, is not a footer.
        assert_eq!(try_command("refused\n\nTry:rocm uninstall"), None);
    }

    #[test]
    fn advise_keeps_the_whole_error_and_puts_the_footer_last() {
        let error = anyhow::anyhow!("inner cause").context("outer refusal");
        let advised = advise(error, "rocm uninstall --yes");
        // `{:#}` is the message alone; `{:?}` is what `rocm` prints. Whether
        // that carries a backtrace depends on the environment, so the strip is
        // proved with literal input in `try_command_reads_through_a_backtrace_section`.
        assert_eq!(
            format!("{advised:#}"),
            "outer refusal: inner cause\n\nTry: rocm uninstall --yes"
        );
        assert_eq!(
            try_command(&format!("{advised:?}")),
            Some("rocm uninstall --yes")
        );
    }

    #[test]
    fn try_command_reads_through_a_backtrace_section() {
        let text = with_try("refused", "rocm uninstall --yes");
        let backtrace = format!("{text}\n\nStack backtrace:\n   0: frame\n      at src/lib.rs:1\n");
        assert_eq!(try_command(&backtrace), Some("rocm uninstall --yes"));
        // No footer before the backtrace: nothing to read.
        assert_eq!(try_command("refused\n\nStack backtrace:\n   0: f"), None);
        // The words inside a message, not after a blank line, are just words.
        assert_eq!(
            try_command("see Stack backtrace: below\n\nTry: rocm uninstall"),
            Some("rocm uninstall")
        );
        // A mention that is not a section start does not hide the real
        // backtrace after the footer.
        assert_eq!(
            try_command(
                "see Stack backtrace: x\n\nTry: rocm uninstall\n\nStack backtrace:\n   0: f"
            ),
            Some("rocm uninstall")
        );
    }

    #[test]
    fn without_backtrace_drops_the_section_and_the_blank_line_before_it() {
        assert_eq!(
            without_backtrace("Error: refused\n\nStack backtrace:\n   0: f\n"),
            "Error: refused"
        );
        assert_eq!(without_backtrace("Error: refused\n"), "Error: refused\n");
        assert_eq!(
            without_backtrace("see Stack backtrace: x"),
            "see Stack backtrace: x"
        );
    }
}
