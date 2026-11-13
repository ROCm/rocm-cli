// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `Try:` footer a refusal ends with: the one command that clears it.
//!
//! A refusal that knows exactly what the user should run next ends with a
//! blank line and `Try: <command>`, so the advice is the last thing on screen
//! and can be copied as it stands. Every such footer is built here
//! ([`with_try`] for a message being written, [`advise`] for an error a
//! shared guard already returned) and read back by the one parser
//! ([`try_command`]), so a test that runs the advised command reads it the
//! same way a user does.
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

/// The command a [`with_try`] footer names, when `text` ends with one.
///
/// Only the last non-blank line counts: a footer is the end of the message,
/// and a `Try:` line anywhere else is not one.
#[must_use]
pub fn try_command(text: &str) -> Option<&str> {
    text.trim_end()
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
        let shown = format!("{advised:?}");
        assert_eq!(
            shown,
            "outer refusal: inner cause\n\nTry: rocm uninstall --yes"
        );
        assert_eq!(try_command(&shown), Some("rocm uninstall --yes"));
    }
}
