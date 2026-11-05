// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared formatter for the "headline, then indented `key: value` details"
//! convention already used ad hoc across the CLI (SDK install success,
//! service stop/restart, runtime activation). Every action a user approves
//! should produce an equally clear reaction; this gives commands a single
//! place to build that reaction instead of hand-rolling `println!`/`writeln!`
//! sequences.
//!
//! It also holds [`JsonOrText`] and [`print_json_or_text`], the one way a
//! command with a `--json` flag prints a single value as either JSON or text.

use std::fmt::Write as _;

use anyhow::{Context, Result};
use serde::Serialize;

/// A headline plus indented `key: value` details, matching the convention:
/// headline with no indent, details indented two spaces.
pub(crate) struct ActionReport {
    headline: String,
    details: Vec<(String, String)>,
}

impl ActionReport {
    pub(crate) fn new(headline: impl Into<String>) -> Self {
        Self {
            headline: headline.into(),
            details: Vec::new(),
        }
    }

    pub(crate) fn detail(mut self, key: impl Into<String>, value: impl std::fmt::Display) -> Self {
        self.details.push((key.into(), value.to_string()));
        self
    }

    pub(crate) fn render(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "{}", self.headline);
        for (key, value) in &self.details {
            let _ = writeln!(output, "  {key}: {value}");
        }
        output
    }
}

/// Appends one row of a marked listing: `{indent}{marker} {body}\n`, where an
/// unmarked row (`marker: None`) gets a single space in the marker column so
/// every row's body stays aligned under the marked ones.
///
/// Used by the listings that flag a row with a glyph and explain it in a
/// `legend:` line (the default engine in `rocm engines`/`rocm examine`, the
/// active/rollback runtime in `rocm runtimes list`). The caller owns the glyph
/// constants, the legend and the shape of `body`; this owns only the shared
/// marker-column layout, so a new listing cannot quietly pick a different one.
/// A marker must be a single column wide: the blank used for unmarked rows is
/// one space, so a wider marker would push its own row out of line.
pub(crate) fn write_marked_row(
    output: &mut String,
    indent: &str,
    marker: Option<&str>,
    body: impl std::fmt::Display,
) {
    let _ = writeln!(output, "{indent}{} {body}", marker.unwrap_or(" "));
}

/// A value a command prints either as pretty JSON (`--json`) or as text for a
/// person.
///
/// `Serialize` is a supertrait so the JSON and text forms cannot be implemented
/// for different types. `render_text` returns the complete text form, trailing
/// newline included; it is written as-is.
///
/// New commands with a `--json` flag implement this and call
/// [`print_json_or_text`] rather than branching on the flag and calling
/// `serde_json::to_string_pretty` themselves. It fits a command whose output is
/// one value with one text rendering. When the text needs a parameter (a row
/// limit, a note about how the value was gathered), put it in a small borrowing
/// view struct that serializes as the value and implements this trait. Output
/// that streams, has a different shape in each mode, or interleaves other work
/// (side effects, audit events) with the printing stays hand-written.
pub(crate) trait JsonOrText: Serialize {
    fn render_text(&self) -> String;
}

/// Writes `value` to `writer` as pretty JSON followed by one newline when
/// `json` is set, otherwise as `value.render_text()`.
///
/// The whole output is built before the first byte is written, so a value that
/// fails to serialize leaves `writer` untouched instead of half a document.
/// Text mode does not render the JSON and JSON mode does not render the text.
fn write_json_or_text(
    writer: &mut impl std::io::Write,
    json: bool,
    value: &impl JsonOrText,
) -> Result<()> {
    let rendered = if json {
        let mut document = serde_json::to_string_pretty(value)
            .context("failed to serialize the command output as JSON")?;
        document.push('\n');
        document
    } else {
        value.render_text()
    };
    writer
        .write_all(rendered.as_bytes())
        .context("failed to write the command output")
}

/// Prints `value` to stdout as pretty JSON (`json`) or as text; see
/// [`JsonOrText`]. Diagnostics and progress belong on stderr, not here, so the
/// JSON a caller pipes stays clean.
pub(crate) fn print_json_or_text(json: bool, value: &impl JsonOrText) -> Result<()> {
    write_json_or_text(&mut std::io::stdout().lock(), json, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marked_row_puts_the_marker_before_the_body() {
        let mut output = String::new();
        write_marked_row(&mut output, "", Some("*"), "lemonade   note");
        write_marked_row(&mut output, "  ", Some("-"), format_args!("rt-{}", 1));
        assert_eq!(output, "* lemonade   note\n  - rt-1\n");
    }

    #[test]
    fn unmarked_row_keeps_the_marker_column_blank() {
        let mut output = String::new();
        write_marked_row(&mut output, "", None, "vllm       note");
        write_marked_row(&mut output, "  ", None, "aim");
        assert_eq!(output, "  vllm       note\n    aim\n");
    }

    #[test]
    fn renders_headline_then_indented_details() {
        let report = ActionReport::new("driver install completed")
            .detail("reboot_required", true)
            .detail("state", "/tmp/state.json");

        assert_eq!(
            report.render(),
            "driver install completed\n  reboot_required: true\n  state: /tmp/state.json\n"
        );
    }

    #[derive(Serialize)]
    struct Sample {
        name: &'static str,
        count: u32,
    }

    impl JsonOrText for Sample {
        fn render_text(&self) -> String {
            format!("{} x{}\n", self.name, self.count)
        }
    }

    fn written(json: bool, value: &impl JsonOrText) -> String {
        let mut output = Vec::new();
        write_json_or_text(&mut output, json, value).expect("writing to a Vec succeeds");
        String::from_utf8(output).expect("output is UTF-8")
    }

    #[test]
    fn json_mode_writes_pretty_json_and_one_trailing_newline() {
        let sample = Sample {
            name: "gpu",
            count: 2,
        };
        assert_eq!(written(true, &sample), "{\n  \"name\": \"gpu\",\n  \"count\": 2\n}\n");
    }

    #[test]
    fn text_mode_writes_the_rendering_verbatim() {
        let sample = Sample {
            name: "gpu",
            count: 2,
        };
        assert_eq!(written(false, &sample), "gpu x2\n");
    }

    #[test]
    fn text_mode_adds_no_newline_of_its_own() {
        struct NoNewline;
        impl Serialize for NoNewline {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_unit()
            }
        }
        impl JsonOrText for NoNewline {
            fn render_text(&self) -> String {
                "no newline".to_owned()
            }
        }
        assert_eq!(written(false, &NoNewline), "no newline");
    }

    /// Serializes to an error, and panics if asked for its text, so one value
    /// proves both that a serialization failure writes nothing and that JSON
    /// mode never renders the text form.
    struct Unserializable;

    impl Serialize for Unserializable {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("deliberately unserializable"))
        }
    }

    impl JsonOrText for Unserializable {
        fn render_text(&self) -> String {
            panic!("JSON mode must not render the text form");
        }
    }

    #[test]
    fn serialization_failure_is_an_error_and_writes_nothing() {
        let mut output = Vec::new();
        let error = write_json_or_text(&mut output, true, &Unserializable)
            .expect_err("an unserializable value must fail");
        assert!(output.is_empty(), "nothing is written on failure");
        let chain = format!("{error:#}");
        assert!(
            chain.contains("failed to serialize") && chain.contains("deliberately unserializable"),
            "unexpected error: {chain}"
        );
    }

    struct BrokenPipe;

    impl std::io::Write for BrokenPipe {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn write_failure_is_an_error_not_a_panic() {
        let sample = Sample {
            name: "gpu",
            count: 2,
        };
        for json in [true, false] {
            let error = write_json_or_text(&mut BrokenPipe, json, &sample)
                .expect_err("a failed write must surface");
            assert!(
                format!("{error:#}").contains("failed to write the command output"),
                "unexpected error: {error:#}"
            );
        }
    }
}
