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

#[cfg(test)]
mod tests {
    use super::*;

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
