// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm diagnose --symptom-file <PATH>`: reading the symptom text from a file,
//! or from standard input when the path is `-`, instead of from a shell
//! argument.
//!
//! A pasted error is hostile to every shell's argument parser: bash runs a
//! `$(...)` inside double quotes, PowerShell does the same and also treats a
//! leading `-` and a stray quote its own way, and cmd.exe has no single-quote at
//! all. Reading the text as data sidesteps all three, so the only rule this
//! module adds is how the bytes become the string `--symptom` would have
//! delivered:
//!
//! - one trailing line break (`\n` or `\r\n`) is removed. A heredoc, `echo`,
//!   an editor's final newline and PowerShell's pipe to a native command each
//!   add exactly one, and none of them is part of the paste. Only one: any
//!   other whitespace is the user's own text and is passed through unchanged;
//! - a UTF-8 byte-order mark is dropped (PowerShell adds one when
//!   `$OutputEncoding` is UTF-8 with a BOM, and `Out-File -Encoding utf8` writes
//!   one), and text with a UTF-16 byte-order mark, which Windows PowerShell's
//!   `>` writes, is decoded. On Windows `--symptom` arrives as UTF-16 and is
//!   converted the same way, so the matcher sees the same string;
//! - anything else must be UTF-8. Input that is not, an unreadable path, and
//!   input past [`MAX_SYMPTOM_BYTES`] are refused with a [`SymptomFileError`],
//!   which the dispatcher turns into a usage error (exit code 2).
//!
//! When the path is `-` and standard input is a terminal, nothing has been
//! piped in, so the user is told on stderr how to paste the text and end it
//! ([`TERMINAL_PASTE_HINT`]); otherwise the command would wait silently.

use std::fmt;
use std::io::{self, IsTerminal, Read, Write as _};
use std::path::Path;

/// The `--symptom-file` value that means standard input.
pub(crate) const STDIN_PATH: &str = "-";

/// The most a symptom may be. A whole engine log fits; a device or an endless
/// pipe passed by mistake does not get read into memory without bound.
pub(crate) const MAX_SYMPTOM_BYTES: u64 = 16 * 1024 * 1024;

/// Printed on stderr when `--symptom-file -` finds standard input is a
/// terminal: the user is expected to paste now, and must be told how to finish.
/// Ctrl-D at the start of a line ends terminal input on Linux; on Windows the
/// console's end-of-input is Ctrl-Z followed by Enter.
pub(crate) const TERMINAL_PASTE_HINT: &str = "Paste the error text, then end it with Ctrl-D at the \
start of a line (Ctrl-Z then Enter on Windows).";

/// Why a `--symptom-file` could not be turned into symptom text.
#[derive(Debug)]
pub(crate) enum SymptomFileError {
    /// The file could not be opened or read.
    Unreadable { source: String, error: io::Error },
    /// The bytes are not UTF-8 (or UTF-16 with a byte-order mark).
    NotText { source: String, detail: String },
    /// More than [`MAX_SYMPTOM_BYTES`].
    TooLarge { source: String },
}

impl fmt::Display for SymptomFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable { source, error } => {
                write!(f, "could not read the symptom from {source}: {error}")
            }
            Self::NotText { source, detail } => write!(
                f,
                "the symptom in {source} is not text: {detail}. Save it as UTF-8 text and try again"
            ),
            Self::TooLarge { source } => write!(
                f,
                "the symptom in {source} is larger than {} MiB; pass only the lines around the failure",
                MAX_SYMPTOM_BYTES / (1024 * 1024)
            ),
        }
    }
}

impl std::error::Error for SymptomFileError {}

/// `error` as the usage error clap would have raised for a bad
/// `--symptom-file` value: printed with `rocm diagnose`'s usage line, exit
/// code 2. The file is read after parsing, so clap cannot raise it itself.
pub(crate) fn usage_error(error: &SymptomFileError) -> clap::Error {
    let mut command = <crate::Cli as clap::CommandFactory>::command();
    // Building sets each subcommand's full `rocm diagnose` name for the usage line.
    command.build();
    command
        .find_subcommand_mut("diagnose")
        .expect("the diagnose subcommand is defined")
        .error(
            clap::error::ErrorKind::ValueValidation,
            format!("--symptom-file: {error}"),
        )
}

/// The symptom text `rocm diagnose` matches on: `--symptom` as given, or the
/// contents of `--symptom-file` (clap makes the two mutually exclusive).
pub(crate) fn resolve(
    symptom: Option<String>,
    symptom_file: Option<&Path>,
) -> Result<Option<String>, SymptomFileError> {
    let Some(path) = symptom_file else {
        return Ok(symptom);
    };
    if path.as_os_str() == STDIN_PATH {
        let stdin = io::stdin();
        if stdin.is_terminal() {
            // A closed stderr must not turn the hint into a panic.
            let _ = writeln!(io::stderr(), "{TERMINAL_PASTE_HINT}");
        }
        return read_symptom(stdin.lock(), "standard input").map(Some);
    }
    let source = path.display().to_string();
    let file = std::fs::File::open(path).map_err(|error| SymptomFileError::Unreadable {
        source: source.clone(),
        error,
    })?;
    read_symptom(file, &source).map(Some)
}

/// Read at most [`MAX_SYMPTOM_BYTES`] from `reader` and decode it as described
/// in the module docs. `source` names the input in an error.
fn read_symptom(reader: impl Read, source: &str) -> Result<String, SymptomFileError> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_SYMPTOM_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| SymptomFileError::Unreadable {
            source: source.to_owned(),
            error,
        })?;
    if bytes.len() as u64 > MAX_SYMPTOM_BYTES {
        return Err(SymptomFileError::TooLarge {
            source: source.to_owned(),
        });
    }
    let text = decode(bytes).map_err(|detail| SymptomFileError::NotText {
        source: source.to_owned(),
        detail,
    })?;
    Ok(trim_one_trailing_line_break(text))
}

/// UTF-8 (a leading byte-order mark dropped), or UTF-16 announced by its
/// byte-order mark. The error says where decoding failed. Takes the buffer by
/// value so UTF-8 becomes the string in place, without a second buffer.
fn decode(mut bytes: Vec<u8>) -> Result<String, String> {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        bytes.drain(..3);
        return utf8(bytes);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return utf16(rest, u16::from_le_bytes);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return utf16(rest, u16::from_be_bytes);
    }
    utf8(bytes)
}

fn utf8(bytes: Vec<u8>) -> Result<String, String> {
    String::from_utf8(bytes).map_err(|error| {
        format!(
            "it is not valid UTF-8 (invalid byte at offset {})",
            error.utf8_error().valid_up_to()
        )
    })
}

fn utf16(bytes: &[u8], unit: fn([u8; 2]) -> u16) -> Result<String, String> {
    let chunks = bytes.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(
            "it starts with a UTF-16 byte-order mark but has an odd number of bytes".into(),
        );
    }
    let units: Vec<u16> = chunks.map(|pair| unit([pair[0], pair[1]])).collect();
    String::from_utf16(&units)
        .map_err(|_| "it starts with a UTF-16 byte-order mark but is not valid UTF-16".into())
}

/// Remove one trailing `\n` or `\r\n`, and nothing else.
fn trim_one_trailing_line_break(mut text: String) -> String {
    if text.ends_with("\r\n") {
        text.truncate(text.len() - 2);
    } else if text.ends_with('\n') {
        text.pop();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A paste every shell would mangle: double and single quotes, a command
    /// substitution, a leading minus, a backtick and a percent sign.
    const HOSTILE_PASTE: &str =
        "-x \"HSA_STATUS_ERROR_INVALID_ISA\" it's $(touch pwned) `id` %PATH% & echo";

    fn read(bytes: &[u8]) -> Result<String, SymptomFileError> {
        read_symptom(bytes, "test input")
    }

    #[test]
    fn a_paste_reaches_the_matcher_unchanged() {
        assert_eq!(read(HOSTILE_PASTE.as_bytes()).unwrap(), HOSTILE_PASTE);
    }

    #[test]
    fn exactly_one_trailing_line_break_is_removed() {
        assert_eq!(read(b"oom\n").unwrap(), "oom");
        assert_eq!(read(b"oom\r\n").unwrap(), "oom");
        assert_eq!(read(b"oom\n\n").unwrap(), "oom\n");
        assert_eq!(read(b"oom\r\n\r\n").unwrap(), "oom\r\n");
        // A bare `\r` is a progress-bar repaint, not a line ending a shell added.
        assert_eq!(read(b"oom\r").unwrap(), "oom\r");
        assert_eq!(read(b"  oom  ").unwrap(), "  oom  ");
        assert_eq!(read(b"").unwrap(), "");
        assert_eq!(read(b"a\nb\n").unwrap(), "a\nb");
    }

    #[test]
    fn a_utf8_byte_order_mark_is_dropped() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(HOSTILE_PASTE.as_bytes());
        bytes.extend_from_slice(b"\r\n");
        assert_eq!(read(&bytes).unwrap(), HOSTILE_PASTE);
    }

    #[test]
    fn utf16_with_a_byte_order_mark_is_decoded() {
        let text = format!("{HOSTILE_PASTE} \u{00e9}\u{1F600}\r\n");
        let mut le = vec![0xFF, 0xFE];
        let mut be = vec![0xFE, 0xFF];
        for unit in text.encode_utf16() {
            le.extend_from_slice(&unit.to_le_bytes());
            be.extend_from_slice(&unit.to_be_bytes());
        }
        let expected = format!("{HOSTILE_PASTE} \u{00e9}\u{1F600}");
        assert_eq!(read(&le).unwrap(), expected);
        assert_eq!(read(&be).unwrap(), expected);
    }

    #[test]
    fn input_that_is_not_text_is_refused_with_where_it_failed() {
        let error = read(b"HIP error \xff\xfe at").unwrap_err();
        assert!(
            matches!(error, SymptomFileError::NotText { .. }),
            "{error:?}"
        );
        let message = error.to_string();
        assert!(message.contains("test input"), "{message}");
        assert!(message.contains("offset 10"), "{message}");
        assert!(message.contains("UTF-8"), "{message}");

        // A UTF-16 BOM with an odd tail, and one with a lone surrogate.
        assert!(matches!(
            read(&[0xFF, 0xFE, 0x41]),
            Err(SymptomFileError::NotText { .. })
        ));
        assert!(matches!(
            read(&[0xFF, 0xFE, 0x00, 0xD8]),
            Err(SymptomFileError::NotText { .. })
        ));
    }

    #[test]
    fn input_past_the_limit_is_refused_without_reading_it_all() {
        let endless = io::repeat(b'a');
        let error = read_symptom(endless, "an endless pipe").unwrap_err();
        assert!(
            matches!(error, SymptomFileError::TooLarge { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains("16 MiB"), "{error}");

        let at_limit = vec![b'a'; MAX_SYMPTOM_BYTES as usize];
        assert_eq!(read(&at_limit).unwrap().len(), at_limit.len());
    }

    #[test]
    fn a_missing_file_is_unreadable_and_names_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no-such-error.txt");
        let error = resolve(None, Some(&missing)).unwrap_err();
        assert!(
            matches!(error, SymptomFileError::Unreadable { .. }),
            "{error:?}"
        );
        assert!(
            error.to_string().contains(&missing.display().to_string()),
            "{error}"
        );
    }

    #[test]
    fn a_file_is_read_and_symptom_alone_passes_through() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("error.txt");
        std::fs::write(&path, format!("{HOSTILE_PASTE}\n")).unwrap();
        assert_eq!(
            resolve(None, Some(&path)).unwrap().as_deref(),
            Some(HOSTILE_PASTE)
        );
        assert_eq!(
            resolve(Some("as typed\n".into()), None).unwrap().as_deref(),
            Some("as typed\n"),
            "--symptom is never trimmed: only a file's trailing line break is"
        );
        assert_eq!(resolve(None, None).unwrap(), None);
    }

    #[test]
    fn a_bad_file_is_a_usage_error_naming_the_command() {
        let error = usage_error(&SymptomFileError::TooLarge {
            source: "standard input".into(),
        });
        assert_eq!(error.exit_code(), 2);
        let rendered = error.render().to_string();
        assert!(
            rendered.contains("--symptom-file: the symptom in standard input is larger"),
            "{rendered}"
        );
        assert!(rendered.contains("Usage: rocm diagnose"), "{rendered}");
    }

    #[test]
    fn a_directory_is_unreadable_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let error = resolve(None, Some(dir.path())).unwrap_err();
        assert!(
            matches!(error, SymptomFileError::Unreadable { .. }),
            "{error:?}"
        );
    }
}
