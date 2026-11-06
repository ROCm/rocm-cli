// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The one routine that reads the last lines of a log file.
//!
//! The engine adapters, the ComfyUI log view, the services text module and the
//! `rocmd` MCP `service_logs` tool all go through [`tail_lines_decoded`]
//! (`tail_lines` is the plain-text form of it), so the seek-boundary handling,
//! the lossy decoding and the "never read the whole file" bound each have one
//! implementation. A caller picks how the bytes become text with a
//! [`LogDecode`]; it does not pick how the file is read.

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// How many bytes from the end of a log the log views (ComfyUI logs, `rocm
/// services logs`, the MCP `service_logs` tool) read at most. The engine
/// adapters keep their own budgets.
pub const LOG_VIEW_MAX_TAIL_READ: u64 = 4 * 1024 * 1024;

/// How the bytes of a log become lines of text.
///
/// Whatever the choice, text that is not valid in the chosen encoding is
/// replaced with U+FFFD rather than failing the read: a log view must show
/// what it can.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogDecode {
    /// Decode the log as UTF-16 little-endian when it carries a byte order
    /// mark, or (without one) when at least a quarter of the odd-numbered
    /// bytes in its first 512 bytes are zero (mostly-ASCII UTF-16 text), as a
    /// Windows shell redirect writes it. Otherwise the log is UTF-8.
    pub utf16_le: bool,
    /// Remove ANSI CSI escape sequences (colour and cursor codes) from each
    /// returned line.
    pub strip_ansi: bool,
}

impl LogDecode {
    /// UTF-8 only, escape sequences kept: what the engine adapters and the
    /// service logs return.
    pub const PLAIN: Self = Self {
        utf16_le: false,
        strip_ansi: false,
    };
    /// Detect UTF-16 little-endian and strip ANSI sequences: for logs a
    /// Windows shell may have redirected, shown to a person in a terminal.
    pub const CONSOLE: Self = Self {
        utf16_le: true,
        strip_ansi: true,
    };
}

/// The lines a bounded tail read returned, and what the bound left unread.
///
/// A caller that only wants the lines uses [`tail_lines_decoded`]; one that
/// tells a person what it found uses [`tail_log_decoded`], so that "the log
/// has no lines" and "the log's newest line does not fit in the read bound"
/// are not the same empty list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LogTail {
    /// The last lines of the log, oldest first.
    pub lines: Vec<String>,
    /// How many bytes at the start of the file the read did not look at
    /// because of the byte budget; zero when the whole file was read.
    pub skipped_bytes: u64,
}

impl LogTail {
    /// True when the read found no complete line even though the log has
    /// content: its newest line begins before the byte budget's window, so the
    /// window held only a fragment of it, which is dropped. An empty log, or a
    /// limit of zero, is not this.
    pub const fn newest_line_exceeds_the_read_bound(&self) -> bool {
        self.lines.is_empty() && self.skipped_bytes > 0
    }
}

/// Reads the last `limit` lines from a file as UTF-8, seeking near the end
/// instead of reading the whole file for files larger than `max_tail_read`.
///
/// Shorthand for [`tail_lines_decoded`] with [`LogDecode::PLAIN`].
pub fn tail_lines(path: &Path, limit: usize, max_tail_read: u64) -> Result<Vec<String>> {
    tail_lines_decoded(path, limit, max_tail_read, LogDecode::PLAIN)
}

/// Reads the last `limit` lines from a file, seeking near the end instead of
/// reading the whole file for files larger than `max_tail_read`.
///
/// Shared by every reader of a log tail so the seek-boundary classification
/// has one implementation: when the seek lands in the middle of a line, the
/// first line read back is a partial fragment and must be dropped; when it
/// lands exactly on a line boundary (the character before the seek position
/// is a newline), that first line is complete and must be kept.
/// `max_tail_read` stays a parameter rather than a shared constant because
/// each engine picks its own tail-read budget (the log views share
/// [`LOG_VIEW_MAX_TAIL_READ`]).
///
/// A limit of zero returns no lines without opening the file. Line endings
/// are `\n` or `\r\n`. See [`LogDecode`] for the text decoding.
pub fn tail_lines_decoded(
    path: &Path,
    limit: usize,
    max_tail_read: u64,
    decode: LogDecode,
) -> Result<Vec<String>> {
    Ok(tail_log_decoded(path, limit, max_tail_read, decode)?.lines)
}

/// [`tail_lines_decoded`], also reporting how much of the start of the file
/// the byte budget left unread (see [`LogTail`]). The lines are the same.
pub fn tail_log_decoded(
    path: &Path,
    limit: usize,
    max_tail_read: u64,
    decode: LogDecode,
) -> Result<LogTail> {
    if limit == 0 {
        return Ok(LogTail::default());
    }
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?;
    tail_lines_from(file, metadata.len(), path, limit, max_tail_read, decode)
}

/// The tail read over any seekable source of `size` bytes; `path` only names
/// the source in error messages. Split out so a test can count the bytes the
/// read touches.
fn tail_lines_from<R: Read + Seek>(
    mut source: R,
    size: u64,
    path: &Path,
    limit: usize,
    max_tail_read: u64,
    decode: LogDecode,
) -> Result<LogTail> {
    let utf16_bom = if decode.utf16_le {
        sniff_utf16_le(&mut source, size, path)?
    } else {
        None
    };
    // The width of one character unit, and so of the newline probe below: a
    // UTF-16 window must start on a unit boundary.
    let unit: u64 = if utf16_bom.is_some() { 2 } else { 1 };
    let bom_len: u64 = if utf16_bom == Some(true) { 2 } else { 0 };

    let mut first_line_is_partial = false;
    let window_start = if size > max_tail_read {
        let mut seek_pos = size - max_tail_read;
        if unit == 2 && seek_pos % 2 == 1 {
            seek_pos += 1;
        }
        if seek_pos >= size {
            return Ok(LogTail {
                lines: Vec::new(),
                skipped_bytes: size,
            });
        }
        // Probe the unit preceding `seek_pos` to classify the first line, then
        // leave the cursor at `seek_pos` for the read below.
        source
            .seek(SeekFrom::Start(seek_pos - unit))
            .with_context(|| format!("failed to seek in {}", path.display()))?;
        let mut probe = [0u8; 2];
        let probe = &mut probe[..unit as usize];
        source
            .read_exact(probe)
            .with_context(|| format!("failed to read from {}", path.display()))?;
        // The unit before the first line of a file with a byte order mark is
        // the mark itself, not the end of a previous line.
        first_line_is_partial = seek_pos != bom_len && probe != [b'\n'] && probe != [b'\n', 0];
        seek_pos
    } else {
        // The whole file is the window; skip a byte order mark the way a
        // reader of the file from its start would.
        source
            .seek(SeekFrom::Start(bom_len))
            .with_context(|| format!("failed to seek in {}", path.display()))?;
        bom_len
    };

    // Capped, not `VecDeque::with_capacity(limit)`: callers may pass
    // `usize::MAX` to mean "no limit", which would try to allocate that
    // capacity upfront. Real callers ask for small limits (tens to a few
    // hundred lines), so capping still gives them a single allocation.
    let mut lines: VecDeque<String> = VecDeque::with_capacity(limit.min(4096));
    let mut keep = |text: String, terminated: bool| {
        // The unterminated tail of a console log can be only the escape
        // sequence a program writes after its last newline (a colour reset).
        // It is not a line: counting it would push the last real line out
        // when the limit is small. A terminated line that strips to nothing
        // is a blank line the program wrote, and stays.
        if decode.strip_ansi && !terminated && strip_ansi_sequences(&text).is_empty() {
            return;
        }
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(text);
    };

    if unit == 2 {
        // The window is at most `max_tail_read` bytes (or the whole, smaller,
        // file), so it is read in one go and split on the newline unit.
        let mut window = Vec::new();
        (&mut source)
            .take(size - window_start)
            .read_to_end(&mut window)
            .with_context(|| format!("failed to read lines from {}", path.display()))?;
        let units: Vec<u16> = window
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        let mut rest = units.as_slice();
        while !rest.is_empty() {
            let (mut line, terminated) =
                if let Some(end) = rest.iter().position(|&unit| unit == 0x000A) {
                    let line = &rest[..end];
                    rest = &rest[end + 1..];
                    (line, true)
                } else {
                    let line = rest;
                    rest = &[];
                    (line, false)
                };
            if terminated && line.last() == Some(&0x000D) {
                line = &line[..line.len() - 1];
            }
            if first_line_is_partial {
                first_line_is_partial = false;
                continue;
            }
            keep(String::from_utf16_lossy(line), terminated);
        }
    } else {
        // Capped at the snapshot like the sixteen-bit window above: a log
        // whose writer is still appending must not pull the read past the
        // budget by however much it grew since `size` was taken.
        let mut reader = BufReader::new(source.take(size - window_start));
        // Read raw bytes rather than `BufRead::lines()`: when the seek above
        // lands inside a multi-byte UTF-8 character, the leading fragment is
        // invalid UTF-8 on its own. `lines()` would surface that as an `Err`
        // and abort the whole read before `first_line_is_partial` gets a
        // chance to drop exactly that fragment. Decoding lossily keeps the
        // drop-the-fragment logic in charge of it instead.
        let mut raw_line = Vec::new();
        loop {
            raw_line.clear();
            let bytes_read = reader
                .read_until(b'\n', &mut raw_line)
                .with_context(|| format!("failed to read lines from {}", path.display()))?;
            if bytes_read == 0 {
                break;
            }
            let terminated = raw_line.last() == Some(&b'\n');
            if terminated {
                raw_line.pop();
                if raw_line.last() == Some(&b'\r') {
                    raw_line.pop();
                }
            }
            if first_line_is_partial {
                first_line_is_partial = false;
                continue;
            }
            keep(String::from_utf8_lossy(&raw_line).into_owned(), terminated);
        }
    }

    let lines = lines
        .into_iter()
        .map(|line| {
            if decode.strip_ansi {
                strip_ansi_sequences(&line)
            } else {
                line
            }
        })
        .collect();
    Ok(LogTail {
        lines,
        // Not `window_start`: a byte order mark skipped at the start of a
        // whole-file read is not something the budget left unread.
        skipped_bytes: if size > max_tail_read {
            window_start
        } else {
            0
        },
    })
}

/// `Some(has_byte_order_mark)` when the start of the source reads as UTF-16
/// little-endian, `None` when it reads as UTF-8. Looks only at the first 512
/// bytes, so it costs the same however large the log is.
fn sniff_utf16_le<R: Read + Seek>(source: &mut R, size: u64, path: &Path) -> Result<Option<bool>> {
    const SAMPLE: u64 = 512;
    let mut head = Vec::new();
    source
        .seek(SeekFrom::Start(0))
        .with_context(|| format!("failed to seek in {}", path.display()))?;
    source
        .take(size.min(SAMPLE))
        .read_to_end(&mut head)
        .with_context(|| format!("failed to read from {}", path.display()))?;
    if head.starts_with(&[0xff, 0xfe]) {
        return Ok(Some(true));
    }
    // Mostly-ASCII UTF-16 has a zero in every other byte.
    if size >= 8 {
        let zero_odd_bytes = head
            .iter()
            .enumerate()
            .filter(|(index, byte)| index % 2 == 1 && **byte == 0)
            .count();
        if zero_odd_bytes * 4 >= head.len() {
            return Ok(Some(false));
        }
    }
    Ok(None)
}

/// Removes ANSI CSI sequences from one line of text.
///
/// A sequence is `ESC [` up to and including a final byte in `@`..=`~`. A bare
/// `ESC` not followed by `[` is dropped alone. The log tail applies this per
/// line, so an unterminated sequence ends with its line instead of swallowing
/// the lines after it.
pub fn strip_ansi_sequences(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\x1b' {
            cleaned.push(ch);
            continue;
        }
        if chars.peek() == Some(&'[') {
            let _ = chars.next();
            for next in chars.by_ref() {
                if ('@'..='~').contains(&next) {
                    break;
                }
            }
        }
    }
    cleaned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_suffix;

    const TEST_MAX_TAIL_READ: u64 = 4 * 1024 * 1024;

    #[test]
    fn tail_lines_returns_suffix() -> Result<()> {
        let path = std::env::temp_dir().join(format!("rocm-core-tail-{}.log", unique_suffix()));
        fs::write(&path, "a\nb\nc\n")?;
        let lines = tail_lines(&path, 2, TEST_MAX_TAIL_READ)?;
        fs::remove_file(&path).ok();
        assert_eq!(lines, vec!["b".to_owned(), "c".to_owned()]);
        Ok(())
    }

    #[test]
    fn tail_lines_keeps_first_line_when_seek_lands_on_boundary() -> Result<()> {
        // Build a file where the tail window starts exactly on a line
        // boundary: a prefix ending in '\n', followed by exactly
        // TEST_MAX_TAIL_READ bytes of complete lines. The first windowed line
        // must NOT be dropped. This is the regression case for issue #597.
        let prefix = format!("{}\n", "p".repeat(63));
        let mut tail = String::from("FIRSTLINE\n");
        while tail.len() + 2 <= TEST_MAX_TAIL_READ as usize {
            tail.push_str("y\n");
        }
        while tail.len() < TEST_MAX_TAIL_READ as usize {
            tail.push('z');
        }
        assert_eq!(tail.len(), TEST_MAX_TAIL_READ as usize);

        let path =
            std::env::temp_dir().join(format!("rocm-core-tail-boundary-{}.log", unique_suffix()));
        fs::write(&path, format!("{prefix}{tail}"))?;
        let lines = tail_lines(&path, usize::MAX, TEST_MAX_TAIL_READ)?;
        fs::remove_file(&path).ok();

        assert_eq!(
            lines.first().map(String::as_str),
            Some("FIRSTLINE"),
            "complete first line must be preserved when the seek lands on a newline boundary"
        );
        assert!(
            !lines.iter().any(|line| line.contains('p')),
            "bytes before the tail window must not appear"
        );
        Ok(())
    }

    #[test]
    fn tail_lines_drops_partial_first_line_when_seek_lands_midline() -> Result<()> {
        // The window starts in the middle of a line, so the leading fragment
        // is partial and must be dropped.
        let prefix = "p".repeat(64);
        let mut tail = String::from("PARTIALFRAGMENT");
        tail.push('\n');
        tail.push_str("SECONDLINE\n");
        while tail.len() < TEST_MAX_TAIL_READ as usize {
            tail.push_str("y\n");
        }
        // Trim back to exactly TEST_MAX_TAIL_READ bytes so the window starts
        // mid-line.
        tail.truncate(TEST_MAX_TAIL_READ as usize);

        let path =
            std::env::temp_dir().join(format!("rocm-core-tail-midline-{}.log", unique_suffix()));
        fs::write(&path, format!("{prefix}{tail}"))?;
        let lines = tail_lines(&path, usize::MAX, TEST_MAX_TAIL_READ)?;
        fs::remove_file(&path).ok();

        assert_eq!(
            lines.first().map(String::as_str),
            Some("SECONDLINE"),
            "partial leading fragment must be dropped when the seek lands mid-line"
        );
        Ok(())
    }

    #[test]
    fn tail_lines_drops_fragment_when_seek_lands_inside_a_multibyte_utf8_char() -> Result<()> {
        // '█' (U+2588) encodes as the 3 bytes E2 96 88. Pick max_tail_read so
        // the seek lands one byte into that character: the probe byte (the
        // preceding byte) is the character's own first byte, so it is
        // correctly classified as partial, but the bytes read back from the
        // seek position start mid-character and are not valid UTF-8 on their
        // own.
        let first_line = format!("{}\n", "p".repeat(10));
        let second_line = "\u{2588}REST\n";
        let third_line = "THIRD\n";
        let content = format!("{first_line}{second_line}{third_line}");

        let seek_pos = first_line.len() as u64 + 1;
        let max_tail_read = content.len() as u64 - seek_pos;

        let path =
            std::env::temp_dir().join(format!("rocm-core-tail-utf8-{}.log", unique_suffix()));
        fs::write(&path, &content)?;
        let lines = tail_lines(&path, usize::MAX, max_tail_read);
        fs::remove_file(&path).ok();

        assert_eq!(
            lines?,
            vec!["THIRD".to_owned()],
            "a seek landing inside a multi-byte UTF-8 character must drop the \
             partial leading fragment instead of erroring out the whole read"
        );
        Ok(())
    }

    #[test]
    fn tail_lines_strips_crlf_and_lossily_decodes_invalid_utf8_in_a_kept_line() -> Result<()> {
        // A small budget so the window starts exactly at this file's start
        // (no seek, no partial-fragment classification) and exercises only
        // the per-line finishing steps: CRLF stripping and lossy decoding of
        // a kept (not dropped) line. "\xff" is never valid UTF-8 on its own,
        // so it must come back as the replacement character rather than
        // erroring the whole read.
        let mut content = b"first\r\n".to_vec();
        content.extend_from_slice(b"bad\xffbyte\r\n");
        let max_tail_read = content.len() as u64;

        let path =
            std::env::temp_dir().join(format!("rocm-core-tail-crlf-{}.log", unique_suffix()));
        fs::write(&path, &content)?;
        let lines = tail_lines(&path, usize::MAX, max_tail_read)?;
        fs::remove_file(&path).ok();

        assert_eq!(
            lines,
            vec!["first".to_owned(), "bad\u{FFFD}byte".to_owned()],
            "CRLF must be stripped and an invalid byte in a kept line must decode \
             lossily instead of failing the read"
        );
        Ok(())
    }

    #[test]
    fn tail_lines_on_a_large_file_returns_only_whole_lines_from_the_window() -> Result<()> {
        // 1 MiB of 1 KiB lines under a 64 KiB budget, with a marked first and
        // last line. The first line lies far outside the window, so a read
        // that returned it was not bounded; the window starts mid-line (the
        // budget is not a multiple of the line length), so a leading fragment
        // that was kept would be shorter than a whole line.
        let mut content = String::from("HEAD-MARKER\n");
        let filler = format!("{}\n", "x".repeat(1023));
        for _ in 0..1024 {
            content.push_str(&filler);
        }
        content.push_str("TAIL-MARKER\n");
        let path =
            std::env::temp_dir().join(format!("rocm-core-tail-large-{}.log", unique_suffix()));
        fs::write(&path, &content)?;

        let lines = tail_lines(&path, usize::MAX, 64 * 1024 + 7)?;
        fs::remove_file(&path).ok();

        assert!(!lines.iter().any(|line| line == "HEAD-MARKER"));
        assert_eq!(lines.last().map(String::as_str), Some("TAIL-MARKER"));
        assert!(lines.len() <= 65, "got {} lines", lines.len());
        for line in &lines[..lines.len() - 1] {
            assert_eq!(line.len(), 1023, "a partial leading fragment was kept");
        }
        Ok(())
    }

    #[test]
    fn tail_lines_missing_file_is_an_open_error_naming_the_path() {
        let path = Path::new("/does/not/exist/rocm-core-tail-missing.log");

        let error = tail_lines(path, 5, TEST_MAX_TAIL_READ).expect_err("missing file");

        assert_eq!(
            error.to_string(),
            format!("failed to open {}", path.display())
        );
    }

    #[test]
    fn tail_lines_returns_every_line_when_fewer_than_the_limit() -> Result<()> {
        let path = std::env::temp_dir().join(format!("rocm-core-tail-few-{}.log", unique_suffix()));
        fs::write(&path, "a\nb\n")?;
        let lines = tail_lines(&path, 50, TEST_MAX_TAIL_READ)?;
        fs::remove_file(&path).ok();
        assert_eq!(lines, vec!["a".to_owned(), "b".to_owned()]);
        Ok(())
    }

    #[test]
    fn tail_lines_keeps_a_final_line_without_a_trailing_newline() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("rocm-core-tail-no-nl-{}.log", unique_suffix()));
        fs::write(&path, "a\nb\nc")?;
        let lines = tail_lines(&path, 2, TEST_MAX_TAIL_READ)?;
        fs::remove_file(&path).ok();
        assert_eq!(lines, vec!["b".to_owned(), "c".to_owned()]);
        Ok(())
    }

    #[test]
    fn tail_lines_limit_zero_returns_empty_without_opening_file() -> Result<()> {
        let path = Path::new("/does/not/exist/rocm-core-tail-limit-zero.log");
        let lines = tail_lines(path, 0, TEST_MAX_TAIL_READ)?;
        assert!(lines.is_empty());
        Ok(())
    }

    // --- bounded reads -------------------------------------------------

    /// Counts the bytes its inner reader hands out, so a test can show how
    /// much of a source a tail read touched.
    struct CountingReader<R> {
        inner: R,
        bytes_read: std::rc::Rc<std::cell::Cell<u64>>,
    }

    impl<R: Read> Read for CountingReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let read = self.inner.read(buf)?;
            self.bytes_read.set(self.bytes_read.get() + read as u64);
            Ok(read)
        }
    }

    impl<R: Seek> Seek for CountingReader<R> {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(position)
        }
    }

    /// Runs the tail read over `content` and returns the lines with the number
    /// of bytes the read pulled from the source.
    fn read_counting(
        content: Vec<u8>,
        limit: usize,
        max_tail_read: u64,
        decode: LogDecode,
    ) -> Result<(Vec<String>, u64)> {
        let size = content.len() as u64;
        let bytes_read = std::rc::Rc::new(std::cell::Cell::new(0));
        let source = CountingReader {
            inner: std::io::Cursor::new(content),
            bytes_read: bytes_read.clone(),
        };
        let lines = tail_lines_from(
            source,
            size,
            Path::new("counting-source"),
            limit,
            max_tail_read,
            decode,
        )?
        .lines;
        Ok((lines, bytes_read.get()))
    }

    fn utf16_le_bytes(text: &str, bom: bool) -> Vec<u8> {
        let mut bytes = if bom { vec![0xff, 0xfe] } else { Vec::new() };
        for unit in text.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn a_tail_read_touches_no_more_than_its_budget_of_a_large_file() -> Result<()> {
        // 8 MiB under a 64 KiB budget: a whole-file read would pull 8 MiB.
        let filler = format!("{}\n", "x".repeat(1023));
        let content = filler.repeat(8192).into_bytes();
        let budget = 64 * 1024;

        let (lines, bytes_read) = read_counting(content, usize::MAX, budget, LogDecode::PLAIN)?;

        assert!(!lines.is_empty());
        // The window plus the one byte probed before it.
        assert!(bytes_read <= budget + 1, "read {bytes_read} bytes");
        Ok(())
    }

    #[test]
    fn a_sixteen_bit_tail_read_touches_no_more_than_its_budget_of_a_large_file() -> Result<()> {
        let filler = format!("\u{1b}[33m{}\u{1b}[0m\r\n", "y".repeat(500));
        let content = utf16_le_bytes(&filler.repeat(8000), true);
        assert!(content.len() > 8_000_000);
        let budget = 64 * 1024 + 1;

        let (lines, bytes_read) = read_counting(content, usize::MAX, budget, LogDecode::CONSOLE)?;

        assert!(!lines.is_empty());
        assert!(
            lines.iter().all(|line| line == &"y".repeat(500)),
            "decoded from a unit boundary, ANSI stripped, no fragment"
        );
        // The window, the unit probed before it, and the 512-byte sniff of the
        // start of the file.
        assert!(bytes_read <= budget + 2 + 512, "read {bytes_read} bytes");
        Ok(())
    }

    /// For every budget from nothing to the whole file, the lines returned are
    /// a suffix of the file's complete lines: never a fragment, never garbage.
    fn assert_every_budget_returns_a_suffix_of_whole_lines(
        content: &[u8],
        expected: &[String],
        decode: LogDecode,
    ) -> Result<()> {
        // A byte order mark is not part of the first line.
        let bom_len = if decode.utf16_le && content.starts_with(&[0xff, 0xfe]) {
            2
        } else {
            0
        };
        for budget in 0..=content.len() as u64 + 2 {
            let (lines, _) = read_counting(content.to_vec(), usize::MAX, budget, decode)?;
            assert!(
                lines.len() <= expected.len() && lines == expected[expected.len() - lines.len()..],
                "budget {budget}: {lines:?} is not a suffix of {expected:?}"
            );
            if budget + bom_len >= content.len() as u64 {
                assert_eq!(lines, expected, "budget {budget} covers the whole file");
            }
        }
        Ok(())
    }

    #[test]
    fn every_budget_returns_whole_utf8_lines() -> Result<()> {
        let text = "alpha\nbravo\r\ncharlie\n\ndelta\u{2588}\nlast";
        let expected = ["alpha", "bravo", "charlie", "", "delta\u{2588}", "last"]
            .map(str::to_owned)
            .to_vec();
        assert_every_budget_returns_a_suffix_of_whole_lines(
            text.as_bytes(),
            &expected,
            LogDecode::PLAIN,
        )?;
        assert_every_budget_returns_a_suffix_of_whole_lines(
            text.as_bytes(),
            &expected,
            LogDecode::CONSOLE,
        )
    }

    #[test]
    fn every_budget_returns_whole_sixteen_bit_lines_including_odd_ones() -> Result<()> {
        // An odd budget puts the seek position on the second byte of a unit,
        // which must be rounded to a unit boundary rather than decoded from.
        let text = "alpha\nbravo\r\ncharlie\n\ndelta\u{2588}\u{1f600}\nlast";
        let expected = [
            "alpha",
            "bravo",
            "charlie",
            "",
            "delta\u{2588}\u{1f600}",
            "last",
        ]
        .map(str::to_owned)
        .to_vec();
        for bom in [true, false] {
            assert_every_budget_returns_a_suffix_of_whole_lines(
                &utf16_le_bytes(text, bom),
                &expected,
                LogDecode::CONSOLE,
            )?;
        }
        Ok(())
    }

    // --- decoding ------------------------------------------------------

    fn read_bytes(
        name: &str,
        bytes: &[u8],
        limit: usize,
        decode: LogDecode,
    ) -> Result<Vec<String>> {
        let path =
            std::env::temp_dir().join(format!("rocm-core-tail-{name}-{}.log", unique_suffix()));
        fs::write(&path, bytes)?;
        let lines = tail_lines_decoded(&path, limit, TEST_MAX_TAIL_READ, decode);
        fs::remove_file(&path).ok();
        lines
    }

    #[test]
    fn console_decode_reads_a_sixteen_bit_log_with_a_byte_order_mark() -> Result<()> {
        let bytes = utf16_le_bytes("\u{1b}[32m[INFO]\u{1b}[0m first\r\nsecond\r\n", true);

        let lines = read_bytes("bom", &bytes, 10, LogDecode::CONSOLE)?;

        assert_eq!(lines, vec!["[INFO] first", "second"]);
        Ok(())
    }

    #[test]
    fn console_decode_detects_a_sixteen_bit_log_without_a_byte_order_mark() -> Result<()> {
        let bytes = utf16_le_bytes("first\nsecond\nthird", false);

        let lines = read_bytes("nobom", &bytes, 2, LogDecode::CONSOLE)?;

        assert_eq!(lines, vec!["second", "third"]);
        Ok(())
    }

    #[test]
    fn console_decode_replaces_a_lone_surrogate_and_ignores_an_odd_trailing_byte() -> Result<()> {
        let mut bytes = utf16_le_bytes("ok\n", true);
        bytes.extend_from_slice(&[0x00, 0xd8, b'x', 0x00, b'\n', 0x00, 0x41]);

        let lines = read_bytes("surrogate", &bytes, 10, LogDecode::CONSOLE)?;

        assert_eq!(lines, vec!["ok", "\u{FFFD}x"]);
        Ok(())
    }

    #[test]
    fn console_decode_leaves_a_short_or_ascii_log_as_utf8() -> Result<()> {
        // Under 8 bytes there is too little to call it sixteen-bit; plain
        // ASCII has no zero bytes to suggest it.
        let short = read_bytes("short", b"a\0b\0c\0", 5, LogDecode::CONSOLE)?;
        let ascii = read_bytes("ascii", b"plain ascii text\n", 5, LogDecode::CONSOLE)?;

        assert_eq!(short, vec!["a\0b\0c\0"]);
        assert_eq!(ascii, vec!["plain ascii text"]);
        Ok(())
    }

    #[test]
    fn console_decode_strips_ansi_per_line_so_an_unterminated_sequence_ends_with_its_line()
    -> Result<()> {
        let bytes = b"\x1b[31mred\x1b[0m\ncut off \x1b[1;3\nnext line\n";

        let lines = read_bytes("ansi", bytes, 10, LogDecode::CONSOLE)?;

        assert_eq!(lines, vec!["red", "cut off ", "next line"]);
        Ok(())
    }

    #[test]
    fn plain_decode_neither_detects_sixteen_bit_text_nor_strips_ansi() -> Result<()> {
        let ansi = read_bytes("plain-ansi", b"\x1b[32mgreen\x1b[0m\n", 5, LogDecode::PLAIN)?;
        let wide = read_bytes(
            "plain-wide",
            &utf16_le_bytes("hi\n", true),
            5,
            LogDecode::PLAIN,
        )?;

        assert_eq!(ansi, vec!["\u{1b}[32mgreen\u{1b}[0m"]);
        assert_eq!(wide[0], "\u{FFFD}\u{FFFD}h\0i\0");
        Ok(())
    }

    #[test]
    fn strip_ansi_sequences_removes_csi_sequences_and_a_bare_escape() {
        assert_eq!(
            strip_ansi_sequences("\u{1b}[1;31mred\u{1b}[0m \u{1b}[2Kdone \u{1b}x"),
            "red done x"
        );
        assert_eq!(strip_ansi_sequences("no escapes"), "no escapes");
    }

    #[test]
    fn a_sixteen_bit_limit_zero_returns_empty_without_opening_the_file() -> Result<()> {
        let path = Path::new("/does/not/exist/rocm-core-tail-limit-zero-wide.log");
        assert!(tail_lines_decoded(path, 0, TEST_MAX_TAIL_READ, LogDecode::CONSOLE)?.is_empty());
        Ok(())
    }

    // --- a source that grows after its size was read -------------------

    /// A source whose underlying data grows by `appended` on the first read
    /// after the size was taken, as a log does when its writer is still busy
    /// between the `stat` and the read.
    struct GrowsOnFirstRead {
        cursor: std::io::Cursor<Vec<u8>>,
        appended: Option<Vec<u8>>,
        bytes_read: std::rc::Rc<std::cell::Cell<u64>>,
    }

    impl GrowsOnFirstRead {
        fn new(content: Vec<u8>, appended: Vec<u8>) -> (Self, std::rc::Rc<std::cell::Cell<u64>>) {
            let bytes_read = std::rc::Rc::new(std::cell::Cell::new(0));
            let source = Self {
                cursor: std::io::Cursor::new(content),
                appended: Some(appended),
                bytes_read: bytes_read.clone(),
            };
            (source, bytes_read)
        }
    }

    impl Read for GrowsOnFirstRead {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if let Some(appended) = self.appended.take() {
                self.cursor.get_mut().extend_from_slice(&appended);
            }
            let read = self.cursor.read(buf)?;
            self.bytes_read.set(self.bytes_read.get() + read as u64);
            Ok(read)
        }
    }

    impl Seek for GrowsOnFirstRead {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.cursor.seek(position)
        }
    }

    #[test]
    fn a_source_that_grows_after_its_size_was_read_is_not_read_past_that_size() -> Result<()> {
        // 64 KiB of 16-byte lines, then 64 KiB more appended after the size
        // snapshot. With a budget of a quarter of the snapshot the window
        // starts mid-file; with a budget above the snapshot there is no seek.
        // Either way the tail must end where the snapshot ended and the read
        // must touch no more than the window (plus the one probe byte).
        let settled = "settled-line-xx\n".repeat(4096).into_bytes();
        let appended = "APPENDED-LINE-x\n".repeat(4096).into_bytes();
        let size = settled.len() as u64;
        for (label, budget) in [
            ("window inside the file", size / 4),
            ("whole file", size + 1),
        ] {
            let (source, bytes_read) = GrowsOnFirstRead::new(settled.clone(), appended.clone());

            let lines = tail_lines_from(
                source,
                size,
                Path::new("growing-source"),
                usize::MAX,
                budget,
                LogDecode::PLAIN,
            )?
            .lines;

            assert!(
                lines.iter().all(|line| line == "settled-line-xx"),
                "{label}: read past the size snapshot into {:?}",
                lines.iter().find(|line| line.starts_with("APPENDED"))
            );
            assert_eq!(lines.last().map(String::as_str), Some("settled-line-xx"));
            assert!(
                bytes_read.get() <= budget.min(size) + 1,
                "{label}: read {} bytes against a budget of {budget}",
                bytes_read.get()
            );
        }
        Ok(())
    }

    #[test]
    fn a_growing_sixteen_bit_source_is_not_read_past_its_size_either() -> Result<()> {
        let settled = utf16_le_bytes(&"settled-line-xx\n".repeat(1024), true);
        let appended = utf16_le_bytes(&"APPENDED-LINE-x\n".repeat(1024), false);
        let size = settled.len() as u64;
        let (source, _) = GrowsOnFirstRead::new(settled, appended);

        let lines = tail_lines_from(
            source,
            size,
            Path::new("growing-source"),
            usize::MAX,
            size / 4,
            LogDecode::CONSOLE,
        )?
        .lines;

        assert!(!lines.is_empty());
        assert!(lines.iter().all(|line| line == "settled-line-xx"));
        Ok(())
    }

    // --- a final escape-only fragment ----------------------------------

    /// What the ComfyUI log view returned before the tail routine replaced its
    /// private reader: decode the whole file, strip ANSI sequences from the
    /// whole text, then take the last `limit` of `str::lines`.
    fn comfyui_reader_before_the_shared_routine(bytes: &[u8], limit: usize) -> Vec<String> {
        let looks_utf16 = bytes.len() >= 8 && {
            let sample = &bytes[..bytes.len().min(512)];
            let zero_odd = sample
                .iter()
                .enumerate()
                .filter(|(index, byte)| index % 2 == 1 && **byte == 0)
                .count();
            zero_odd * 4 >= sample.len()
        };
        let text = if bytes.starts_with(&[0xff, 0xfe]) || looks_utf16 {
            let start = if bytes.starts_with(&[0xff, 0xfe]) {
                2
            } else {
                0
            };
            let units: Vec<u16> = bytes[start..]
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            String::from_utf16_lossy(&units)
        } else {
            String::from_utf8_lossy(bytes).into_owned()
        };
        let text = strip_ansi_sequences(&text);
        let mut lines: Vec<String> = text.lines().rev().take(limit).map(str::to_owned).collect();
        lines.reverse();
        lines
    }

    /// The same text as UTF-8 and as UTF-16 with and without a byte order mark.
    fn console_encodings(text: &str) -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("utf-8", text.as_bytes().to_vec()),
            ("utf-16 with bom", utf16_le_bytes(text, true)),
            ("utf-16 without bom", utf16_le_bytes(text, false)),
        ]
    }

    #[test]
    fn a_colour_reset_after_the_final_newline_does_not_displace_the_last_line() -> Result<()> {
        for (label, bytes) in console_encodings("\u{1b}[31mfailed\n\u{1b}[0m") {
            let lines = read_bytes("reset", &bytes, 1, LogDecode::CONSOLE)?;
            assert_eq!(lines, vec!["failed"], "{label}");
        }
        Ok(())
    }

    #[test]
    fn a_final_escape_only_fragment_is_not_a_line_but_blank_lines_are() -> Result<()> {
        // (text, limit, expected): the unterminated escape-only tail never
        // counts; a terminated blank line, even one made only of an escape
        // sequence, is a real line and keeps its place.
        let cases: [(&str, usize, &[&str]); 5] = [
            ("one\ntwo\n\u{1b}[0m", 2, &["one", "two"]),
            ("one\ntwo\n\u{1b}[1;3", 2, &["one", "two"]),
            ("one\ntwo\n\n", 2, &["two", ""]),
            ("one\ntwo\n\u{1b}[0m\n", 2, &["two", ""]),
            ("one\ntwo\n\u{1b}[0mtext", 2, &["two", "text"]),
        ];
        for (text, limit, expected) in cases {
            for (label, bytes) in console_encodings(text) {
                let lines = read_bytes("fragment", &bytes, limit, LogDecode::CONSOLE)?;
                assert_eq!(lines, expected, "{label}: {text:?} limit {limit}");
            }
        }
        Ok(())
    }

    #[test]
    fn plain_decode_keeps_a_final_escape_fragment_because_it_strips_nothing() -> Result<()> {
        let lines = read_bytes("plain-reset", b"a\n\x1b[0m", 2, LogDecode::PLAIN)?;

        assert_eq!(lines, vec!["a", "\u{1b}[0m"]);
        Ok(())
    }

    #[test]
    fn console_decode_matches_the_comfyui_reader_it_replaced_on_small_logs() -> Result<()> {
        // Inputs on which the old reader and the per-line strip agree: no
        // escape sequence runs across a newline. The old reader is the
        // oracle, so a difference here is a change in what a user sees.
        let texts = [
            "plain\nlines\n",
            "\u{1b}[31mfailed\n\u{1b}[0m",
            "\u{1b}[31mfailed\u{1b}[0m\n\u{1b}[0m",
            "a\n\u{1b}[0m\n\u{1b}[0m",
            "a\r\nb\r\n\u{1b}[0m",
            "a\n\n\n",
            "a\n\u{1b}[2K",
            "\u{1b}[0m",
            "",
        ];
        for text in texts {
            for (label, bytes) in console_encodings(text) {
                for limit in 1..=4 {
                    let expected = comfyui_reader_before_the_shared_routine(&bytes, limit);
                    let actual = read_bytes("oracle", &bytes, limit, LogDecode::CONSOLE)?;
                    assert_eq!(actual, expected, "{label}: {text:?} limit {limit}");
                }
            }
        }
        Ok(())
    }

    // --- an empty tail: no lines, or no line that fits the bound --------

    fn read_log_bytes(
        name: &str,
        bytes: &[u8],
        limit: usize,
        max_tail_read: u64,
        decode: LogDecode,
    ) -> Result<LogTail> {
        let path =
            std::env::temp_dir().join(format!("rocm-core-tail-{name}-{}.log", unique_suffix()));
        fs::write(&path, bytes)?;
        let tail = tail_log_decoded(&path, limit, max_tail_read, decode);
        fs::remove_file(&path).ok();
        tail
    }

    #[test]
    fn an_empty_tail_says_whether_the_log_is_empty_or_its_newest_line_is_too_long() -> Result<()> {
        // The state matrix of an empty `lines`: (content, limit, whether the
        // newest line exceeds the bound, bytes the budget left unread).
        let long_line = "x".repeat(300);
        let long_with_newline = format!("{long_line}\n");
        let long_utf16 = utf16_le_bytes(&long_line, true);
        let cases: Vec<(&str, Vec<u8>, usize, bool, u64)> = vec![
            ("empty file", Vec::new(), 5, false, 0),
            (
                "limit zero on a long line",
                long_line.clone().into_bytes(),
                0,
                false,
                0,
            ),
            (
                "long line without a newline",
                long_line.into_bytes(),
                5,
                true,
                300 - 64,
            ),
            (
                "long line with a newline",
                long_with_newline.clone().into_bytes(),
                5,
                true,
                301 - 64,
            ),
            ("long sixteen-bit line", long_utf16, 5, true, 602 - 64),
            (
                "long line that ends a larger log",
                format!("first\n{long_with_newline}").into_bytes(),
                5,
                true,
                6 + 301 - 64,
            ),
            ("short log", b"a\nb\n".to_vec(), 5, false, 0),
        ];
        for (label, content, limit, exceeds, skipped) in cases {
            let tail = read_log_bytes("empty-tail", &content, limit, 64, LogDecode::CONSOLE)?;
            assert_eq!(
                tail.newest_line_exceeds_the_read_bound(),
                exceeds,
                "{label}: {tail:?}"
            );
            assert_eq!(tail.skipped_bytes, skipped, "{label}: {tail:?}");
            // The flag is only ever set for an empty list, and the empty list
            // only ever comes from the cases that name it.
            if exceeds || content.is_empty() || limit == 0 {
                assert!(tail.lines.is_empty(), "{label}: {tail:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn a_tail_that_found_lines_never_reports_an_exceeded_bound() -> Result<()> {
        let content = "line\n".repeat(100).into_bytes();

        let tail = read_log_bytes("found", &content, 5, 64, LogDecode::PLAIN)?;

        assert_eq!(tail.lines.len(), 5);
        assert!(tail.skipped_bytes > 0, "the budget did leave bytes unread");
        assert!(!tail.newest_line_exceeds_the_read_bound());
        Ok(())
    }

    #[test]
    fn a_byte_order_mark_is_not_counted_as_bytes_the_budget_skipped() -> Result<()> {
        let content = utf16_le_bytes("a\nb\n", true);

        let tail = read_log_bytes("bom-skipped", &content, 5, 1024, LogDecode::CONSOLE)?;

        assert_eq!(tail.lines, vec!["a", "b"]);
        assert_eq!(tail.skipped_bytes, 0);
        Ok(())
    }
}
