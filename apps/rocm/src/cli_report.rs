// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared formatter for the "headline, then indented `key: value` details"
//! convention already used ad hoc across the CLI (SDK install success,
//! service stop/restart, runtime activation). Every action a user approves
//! should produce an equally clear reaction; this gives commands a single
//! place to build that reaction instead of hand-rolling `println!`/`writeln!`
//! sequences.

use std::fmt::Write as _;

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
}
