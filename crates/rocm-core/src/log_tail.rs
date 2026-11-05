// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Reads the last `limit` lines from a file, seeking near the end instead of
/// reading the whole file for files larger than `max_tail_read`.
///
/// Shared by the engine adapters so the seek-boundary classification has one
/// implementation: when the seek lands in the middle of a line, the first
/// line read back is a partial fragment and must be dropped; when it lands
/// exactly on a line boundary (the byte before the seek position is a
/// newline), that first line is complete and must be kept. `max_tail_read`
/// stays a parameter rather than a shared constant because each engine picks
/// its own tail-read budget.
pub fn tail_lines(path: &Path, limit: usize, max_tail_read: u64) -> Result<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?;
    let file_size = metadata.len();

    let mut first_line_is_partial = false;
    if file_size > max_tail_read {
        let seek_pos = file_size - max_tail_read;
        // Probe the byte preceding `seek_pos` to classify the first line, then
        // leave the cursor at `seek_pos` for the buffered read below.
        file.seek(SeekFrom::Start(seek_pos - 1))
            .with_context(|| format!("failed to seek in {}", path.display()))?;
        let mut probe = [0u8; 1];
        file.read_exact(&mut probe)
            .with_context(|| format!("failed to read from {}", path.display()))?;
        first_line_is_partial = probe[0] != b'\n';
    }

    let mut reader = BufReader::new(file);
    // Capped, not `VecDeque::with_capacity(limit)`: callers may pass
    // `usize::MAX` to mean "no limit", which would try to allocate that
    // capacity upfront. Real callers ask for small limits (tens to a few
    // hundred lines), so capping still gives them a single allocation.
    let mut lines: VecDeque<String> = VecDeque::with_capacity(limit.min(4096));
    // Read raw bytes rather than `BufRead::lines()`: when the seek above lands
    // inside a multi-byte UTF-8 character, the leading fragment is invalid
    // UTF-8 on its own. `lines()` would surface that as an `Err` and abort the
    // whole read before `first_line_is_partial` gets a chance to drop exactly
    // that fragment. Decoding lossily keeps the drop-the-fragment logic in
    // charge of it instead.
    let mut raw_line = Vec::new();
    loop {
        raw_line.clear();
        let bytes_read = reader
            .read_until(b'\n', &mut raw_line)
            .with_context(|| format!("failed to read lines from {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        if raw_line.last() == Some(&b'\n') {
            raw_line.pop();
            if raw_line.last() == Some(&b'\r') {
                raw_line.pop();
            }
        }
        if first_line_is_partial {
            first_line_is_partial = false;
            continue;
        }
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(String::from_utf8_lossy(&raw_line).into_owned());
    }
    Ok(lines.into_iter().collect())
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
}
