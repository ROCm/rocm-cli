// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The shared `[y/N]` confirmation prompt.
//!
//! Before this module existed, `fix.rs`, `apps/rocm/src/uninstall_plan.rs` and
//! `apps/rocm/src/therock.rs` each hand-rolled the same print-prompt / flush
//! stdout / read a line / normalize / match `"y"`/`"yes"` sequence. This is
//! that sequence, split into the two fallible steps callers compose.
//!
//! [`print_prompt`] and [`read_answer`] stay two calls rather than one bundled
//! "ask and read" function because `fix.rs`'s `confirm()` attempts the read
//! even when the prompt couldn't be written — its behavior before this module
//! existed — while `uninstall_plan.rs` and `therock.rs` treat a failed write
//! as fatal, each with their own phase-specific message. A single
//! `Result`-returning function can't express "ignore this step's error and
//! keep going"; only a caller holding a separate `Result` per step can.
//!
//! Both take the stream to use (`&mut impl Write` / `&mut impl BufRead`)
//! rather than reaching for `stdin()`/`stdout()` themselves, so the sequence
//! — including a write failure and an EOF read — is unit-testable here
//! without a controlled terminal. Real callers pass `std::io::stdout()` and
//! `std::io::stdin().lock()`.

use std::io::{BufRead, Write};

/// Parse a user's typed response to a `[y/N]` prompt.
pub fn is_affirmative_answer(line: &str) -> bool {
    matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
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
pub fn read_answer(input: &mut impl BufRead) -> std::io::Result<bool> {
    let mut line = String::new();
    input.read_line(&mut line)?;
    Ok(is_affirmative_answer(&line))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error, Result};

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

    #[test]
    fn read_answer_parses_a_typed_line() {
        let mut input: &[u8] = b"yes\n";
        assert!(read_answer(&mut input).expect("read from a byte slice cannot fail"));

        let mut input: &[u8] = b"n\n";
        assert!(!read_answer(&mut input).expect("read from a byte slice cannot fail"));
    }

    #[test]
    fn read_answer_at_eof_declines_without_erroring() {
        let mut input: &[u8] = b"";
        assert!(!read_answer(&mut input).expect("EOF is Ok(false), not an error"));
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
        read_answer(&mut input).expect_err("a failing reader must return Err");
    }
}
