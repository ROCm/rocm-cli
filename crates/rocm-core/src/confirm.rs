// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The shared `[y/N]` confirmation prompt.
//!
//! Before this module existed, `fix.rs`, `apps/rocm/src/uninstall_plan.rs` and
//! `apps/rocm/src/therock/consent.rs` each hand-rolled the same print-prompt /
//! flush stdout / read a line / normalize / match `"y"`/`"yes"` sequence. This
//! is that sequence, split into the two fallible steps callers compose.
//!
//! [`print_prompt`] and [`read_answer`] stay two calls rather than one bundled
//! "ask and read" function because `fix/consent.rs`'s `confirm()` attempts the read
//! even when the prompt couldn't be written — its behavior before this module
//! existed — while `uninstall_plan.rs` and `therock/consent.rs` treat a failed
//! write as fatal, each with their own phase-specific message. A single
//! `Result`-returning function can't express "ignore this step's error and
//! keep going"; only a caller holding a separate `Result` per step can.
//!
//! Both take the stream to use (`&mut impl Write` / `&mut impl BufRead`)
//! rather than reaching for `stdin()`/`stdout()` themselves, so the sequence
//! — including a write failure and an EOF read — is unit-testable here
//! without a controlled terminal. Real callers pass `std::io::stdout()` and
//! `std::io::stdin().lock()`.
//!
//! [`read_answer`] also tells the user when their answer was not understood, and
//! ends the prompt line when the input ended without a newline. Both go to the
//! `out` stream the caller passes, so every caller gets them and none can forget.
//! `y`/`yes` approve; `n`/`no`/an empty line decline silently (an empty line is
//! the documented default); anything else declines with a line saying so, rather
//! than being mistaken for a deliberate "no".
//!
//! [`interactive_terminal`] answers the question callers ask before prompting at
//! all: whether this process has a terminal on both streams.
//! [`require_yes_or_interactive`] is the guard built on it for a mutating
//! command that confirms with a `[y/N]` prompt: `--yes` approves, a terminal
//! gets asked, and anything else is refused with one shared message.

use anyhow::{Result, bail};
use std::io::{BufRead, IsTerminal, Write, stdin, stdout};

/// How a typed response to a `[y/N]` prompt reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    /// `y` or `yes`, in any case.
    Yes,
    /// `n` or `no`, or an empty line: the prompt's default.
    No,
    /// Anything else. Declined like [`Answer::No`], but worth saying so.
    Unrecognized,
}

/// Classify a user's typed response to a `[y/N]` prompt.
pub fn parse_answer(line: &str) -> Answer {
    match line.trim().to_lowercase().as_str() {
        "y" | "yes" => Answer::Yes,
        "" | "n" | "no" => Answer::No,
        _ => Answer::Unrecognized,
    }
}

/// Write `prompt` to `out` and flush it, so it's visible before a blocking
/// read. `prompt` is expected to already include the trailing `[y/N]: `.
///
/// Uses a fallible write rather than `print!`, which panics on a write
/// failure instead of returning it. In practice a short prompt like these
/// only reaches an unbuffered write (and so can only fail) at `flush`, since
/// it fits in stdout's line buffer; a fallible write only matters once a
/// prompt is long enough to overflow that buffer, but costs nothing to have
/// either way.
pub fn print_prompt(out: &mut impl Write, prompt: &str) -> std::io::Result<()> {
    write!(out, "{prompt}")?;
    out.flush()
}

/// Read one line from `input` and parse it as yes/no. Only errors if the
/// line can't be read; a declined answer — including an empty read at EOF —
/// is `Ok(false)`, not an error.
///
/// Two things are written to `out` so the caller's own message lands where the
/// user can read it:
/// - when the line did not end in a newline (EOF, as from Ctrl-D, or piped input
///   with no final newline such as `printf no |`), a newline, ending the prompt
///   line: either way nothing echoed a line end, so the cursor is still on the
///   prompt line and the caller's next line would run into it;
/// - when the answer is [`Answer::Unrecognized`], a line saying it is being
///   treated as no, with the answer shown back quoted and escaped and cut to
///   [`MAX_ECHOED_ANSWER_CHARS`] characters, so a long pasted line does not fill
///   the screen.
///
/// Both writes are best effort: a failure to explain a decline must not turn
/// into an error that replaces the decline, which `fix`'s `confirm()` relies on.
pub fn read_answer(input: &mut impl BufRead, out: &mut impl Write) -> std::io::Result<bool> {
    let mut line = String::new();
    input.read_line(&mut line)?;
    if !line.ends_with('\n') {
        let _ = writeln!(out);
    }
    let answer = parse_answer(&line);
    if answer == Answer::Unrecognized {
        let _ = writeln!(
            out,
            "{} is not yes or no; treating it as no.",
            echoed_answer(line.trim())
        );
    }
    Ok(answer == Answer::Yes)
}

/// The most characters of a typed answer that the "not understood" line shows.
const MAX_ECHOED_ANSWER_CHARS: usize = 40;

/// `answer` as shown back to the user: quoted with control characters escaped, and
/// cut to [`MAX_ECHOED_ANSWER_CHARS`] characters (never in the middle of one), with
/// `...` after the closing quote when there was more.
fn echoed_answer(answer: &str) -> String {
    let mut chars = answer.chars();
    let shown: String = chars.by_ref().take(MAX_ECHOED_ANSWER_CHARS).collect();
    let more = if chars.next().is_some() { "..." } else { "" };
    format!("{shown:?}{more}")
}

/// Whether this process can hold an interactive exchange with a user.
///
/// Both streams must be a terminal: stdin so an answer can be read, stdout so
/// the question is seen. Anything that captures either — a pipe, a CI step, the
/// dashboard spawning `rocm` as a child — makes this false, and callers then
/// skip the prompt rather than block on input nobody can supply.
///
/// A property of the invocation, not of the host. `rocm examine` reports it so a
/// pasted report explains why prompts were skipped.
pub fn interactive_terminal() -> bool {
    stdin().is_terminal() && stdout().is_terminal()
}

/// Guard a mutating command that confirms with a `[y/N]` prompt.
///
/// Returns `Ok(true)` when `yes` (the command's `--yes`) already approves it, so
/// the caller proceeds without prompting. Returns `Ok(false)` when there is no
/// `--yes` but [`interactive_terminal`] holds, so the caller still has to ask.
/// Otherwise nobody can answer a prompt and the command is refused with
/// `"{action} requires --yes outside an interactive terminal"`; `action` is the
/// command as the user typed it (`"uninstall"`, `"runtimes uninstall"`).
///
/// `--yes` is checked first, so it never touches the terminal probe. Commands
/// whose refusal deliberately differs (a different exit code, or two consents
/// that are not interchangeable) do not use this.
pub fn require_yes_or_interactive(yes: bool, action: &str) -> Result<bool> {
    require_yes_or_interactive_with(yes, action, interactive_terminal)
}

/// [`require_yes_or_interactive`] with the terminal probe injected.
///
/// The refusal is then testable without depending on how the test runner's
/// streams are wired, here and in a caller that adds to the refusal. The probe
/// is only called when `yes` is false.
pub fn require_yes_or_interactive_with(
    yes: bool,
    action: &str,
    interactive: impl FnOnce() -> bool,
) -> Result<bool> {
    if yes {
        return Ok(true);
    }
    if !interactive() {
        bail!("{action} requires --yes outside an interactive terminal");
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error, Result};

    #[test]
    fn parse_answer_accepts_only_y_and_yes() {
        for accepted in ["y", "Y", "yes", "YES", "Yes", "  y  ", "  yes\n"] {
            assert_eq!(
                parse_answer(accepted),
                Answer::Yes,
                "expected {accepted:?} to be treated as a yes"
            );
        }
    }

    #[test]
    fn parse_answer_reads_n_no_and_an_empty_line_as_the_default_no() {
        for declined in ["n", "N", "no", "NO", " no ", "", "\n", "   \n"] {
            assert_eq!(
                parse_answer(declined),
                Answer::No,
                "expected {declined:?} to be the default no"
            );
        }
    }

    #[test]
    fn parse_answer_flags_everything_else_as_unrecognized() {
        for odd in ["maybe", "yep", "ye", "yes please", "nope", "1", "y n"] {
            assert_eq!(
                parse_answer(odd),
                Answer::Unrecognized,
                "expected {odd:?} to be unrecognized"
            );
        }
    }

    /// A writer that always fails, standing in for a full or closed stdout
    /// without depending on `/dev/full` or any other platform-specific setup.
    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> Result<usize> {
            Err(Error::other("write failed"))
        }
        fn flush(&mut self) -> Result<()> {
            Err(Error::other("flush failed"))
        }
    }

    #[test]
    fn print_prompt_writes_and_flushes_the_whole_prompt() {
        let mut out = Vec::new();
        print_prompt(&mut out, "Proceed? [y/N]: ").expect("write to a Vec cannot fail");
        assert_eq!(out, b"Proceed? [y/N]: ");
    }

    #[test]
    fn print_prompt_surfaces_a_write_failure_as_an_error_not_a_panic() {
        let mut out = FailingWriter;
        print_prompt(&mut out, "Proceed? [y/N]: ")
            .expect_err("a failing writer must return Err, not panic");
    }

    /// A writer whose `write` succeeds but whose `flush` always fails, so a
    /// flush-specific failure is exercised separately from a write failure:
    /// `FailingWriter` above fails at `write` and so never reaches `flush` at
    /// all, which would let a dropped `out.flush()?` pass every test here.
    struct FlushFailingWriter {
        written: Vec<u8>,
    }

    impl Write for FlushFailingWriter {
        fn write(&mut self, buf: &[u8]) -> Result<usize> {
            self.written.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> Result<()> {
            Err(Error::other("flush failed"))
        }
    }

    #[test]
    fn print_prompt_surfaces_a_flush_failure_as_an_error_not_a_panic() {
        let mut out = FlushFailingWriter {
            written: Vec::new(),
        };
        print_prompt(&mut out, "Proceed? [y/N]: ")
            .expect_err("a writer whose flush fails must return Err, not panic");
        assert_eq!(
            out.written, b"Proceed? [y/N]: ",
            "the write itself must still have gone through before flush ran"
        );
    }

    /// Run `read_answer` over `typed` and return the verdict and everything it
    /// wrote to the output stream.
    fn answer_and_output(typed: &[u8]) -> (bool, String) {
        let mut input = typed;
        let mut out = Vec::new();
        let verdict =
            read_answer(&mut input, &mut out).expect("read from a byte slice cannot fail");
        (
            verdict,
            String::from_utf8(out).expect("the helper writes UTF-8"),
        )
    }

    #[test]
    fn read_answer_approves_y_and_yes_without_writing_anything() {
        for typed in [&b"y\n"[..], b"yes\n", b"YES\n"] {
            let (verdict, written) = answer_and_output(typed);
            assert!(verdict, "{typed:?} must approve");
            assert_eq!(written, "", "an approval needs no comment");
        }
    }

    #[test]
    fn read_answer_declines_n_no_and_enter_without_writing_anything() {
        for typed in [&b"n\n"[..], b"no\n", b"\n"] {
            let (verdict, written) = answer_and_output(typed);
            assert!(!verdict, "{typed:?} must decline");
            assert_eq!(
                written, "",
                "a deliberate no (or the default) must not be commented on"
            );
        }
    }

    #[test]
    fn read_answer_declines_an_unrecognized_answer_and_says_so() {
        let (verdict, written) = answer_and_output(b"maybe\n");
        assert!(!verdict, "an unrecognized answer must decline");
        assert_eq!(written, "\"maybe\" is not yes or no; treating it as no.\n");
    }

    #[test]
    fn read_answer_at_eof_declines_and_ends_the_prompt_line() {
        let (verdict, written) = answer_and_output(b"");
        assert!(!verdict, "EOF is Ok(false), not an error");
        assert_eq!(written, "\n", "EOF leaves the cursor on the prompt line");
    }

    #[test]
    fn read_answer_for_a_partial_line_ends_the_prompt_line_then_says_so() {
        // Typed text and then EOF, with no newline: the terminal echoed the text
        // but no newline, so the line must be ended before the feedback line.
        let (verdict, written) = answer_and_output(b"maybe");
        assert!(!verdict);
        assert_eq!(
            written,
            "\n\"maybe\" is not yes or no; treating it as no.\n"
        );
    }

    #[test]
    fn read_answer_for_a_yes_without_a_newline_still_approves() {
        let (verdict, written) = answer_and_output(b"yes");
        assert!(verdict, "a yes is a yes with or without the final newline");
        assert_eq!(written, "\n", "the unfinished prompt line is still ended");
    }

    #[test]
    fn read_answer_cuts_a_long_answer_short_and_marks_the_cut() {
        let typed = format!("{}\n", "a".repeat(100));
        let (verdict, written) = answer_and_output(typed.as_bytes());
        assert!(!verdict);
        assert_eq!(
            written,
            format!(
                "\"{}\"... is not yes or no; treating it as no.\n",
                "a".repeat(MAX_ECHOED_ANSWER_CHARS)
            )
        );
    }

    #[test]
    fn read_answer_shows_an_answer_of_exactly_the_limit_whole() {
        let answer = "b".repeat(MAX_ECHOED_ANSWER_CHARS);
        let (_, written) = answer_and_output(format!("{answer}\n").as_bytes());
        assert_eq!(
            written,
            format!("\"{answer}\" is not yes or no; treating it as no.\n"),
            "an answer that fits is not marked as cut"
        );
        let (_, written) = answer_and_output(format!("{answer}c\n").as_bytes());
        assert_eq!(
            written,
            format!("\"{answer}\"... is not yes or no; treating it as no.\n"),
            "one character over the limit is cut"
        );
    }

    #[test]
    fn read_answer_cuts_between_characters_not_inside_one() {
        // 2-byte and 4-byte characters: a cut by bytes would split one and panic
        // or write invalid UTF-8; the limit counts characters.
        for ch in ['é', '🦀'] {
            let typed = format!("{}\n", ch.to_string().repeat(MAX_ECHOED_ANSWER_CHARS + 5));
            let (_, written) = answer_and_output(typed.as_bytes());
            assert_eq!(
                written,
                format!(
                    "\"{}\"... is not yes or no; treating it as no.\n",
                    ch.to_string().repeat(MAX_ECHOED_ANSWER_CHARS)
                ),
                "answer of {ch:?}"
            );
        }
    }

    #[test]
    fn read_answer_bounds_the_line_even_when_every_character_is_escaped() {
        // An escaped control character is longer than itself, so the limit is on
        // the typed characters and the line stays bounded by it.
        let typed = format!("{}\n", "\x1b".repeat(100));
        let (_, written) = answer_and_output(typed.as_bytes());
        assert_eq!(
            written,
            format!(
                "\"{}\"... is not yes or no; treating it as no.\n",
                "\\u{1b}".repeat(MAX_ECHOED_ANSWER_CHARS)
            )
        );
        assert!(
            !written.contains('\x1b'),
            "no raw escape reaches the terminal"
        );
    }

    #[test]
    fn read_answer_for_a_no_without_a_newline_declines_and_only_ends_the_prompt_line() {
        // Piped input such as `printf no |` has no final newline either. The
        // verdict is the decline `no\n` gives and nothing explains it, but the
        // prompt line is still ended: no newline was echoed after the prompt.
        let (verdict, written) = answer_and_output(b"no");
        assert!(!verdict, "a no is a no with or without the final newline");
        assert_eq!(written, "\n", "only the unfinished prompt line is ended");
    }

    #[test]
    fn read_answer_quotes_an_answer_that_holds_control_characters() {
        // The answer is echoed back, so it is escaped rather than written raw.
        let (_, written) = answer_and_output(b"\x1b[31mred\n");
        assert_eq!(
            written,
            "\"\\u{1b}[31mred\" is not yes or no; treating it as no.\n"
        );
    }

    #[test]
    fn read_answer_keeps_the_verdict_when_the_feedback_cannot_be_written() {
        // The explanation is best effort: a closed stdout must not turn a
        // decline into an error, which `fix`'s `confirm()` relies on.
        let mut input: &[u8] = b"maybe\n";
        let mut out = FailingWriter;
        let verdict = read_answer(&mut input, &mut out)
            .expect("a failed feedback write must not replace the verdict");
        assert!(!verdict);
    }

    /// A reader whose `read_line` always fails, so a real I/O failure on the
    /// input side (as opposed to EOF, which is not an error) is exercised too.
    struct FailingReader;

    impl std::io::Read for FailingReader {
        fn read(&mut self, _buf: &mut [u8]) -> Result<usize> {
            Err(Error::other("read failed"))
        }
    }

    impl BufRead for FailingReader {
        fn fill_buf(&mut self) -> Result<&[u8]> {
            Err(Error::other("read failed"))
        }
        fn consume(&mut self, _amt: usize) {}
    }

    #[test]
    fn read_answer_surfaces_a_read_failure_as_an_error() {
        let mut input = FailingReader;
        let mut out = Vec::new();
        read_answer(&mut input, &mut out).expect_err("a failing reader must return Err");
        assert!(out.is_empty(), "nothing is written for a failed read");
    }

    #[test]
    fn guard_approves_on_yes_without_probing_the_terminal() {
        let approved = require_yes_or_interactive_with(true, "uninstall", || {
            panic!("--yes must short-circuit before the terminal probe")
        })
        .expect("--yes approves");
        assert!(approved);
    }

    #[test]
    fn guard_asks_for_a_prompt_when_a_terminal_is_present_and_yes_is_absent() {
        let approved = require_yes_or_interactive_with(false, "uninstall", || true)
            .expect("an interactive terminal is allowed to prompt");
        assert!(!approved, "the caller still has to ask");
    }

    #[test]
    fn guard_refuses_without_yes_outside_a_terminal_naming_the_action() {
        let error = require_yes_or_interactive_with(false, "runtimes uninstall", || false)
            .expect_err("no --yes and no terminal must refuse");
        assert_eq!(
            error.to_string(),
            "runtimes uninstall requires --yes outside an interactive terminal"
        );
    }
}
