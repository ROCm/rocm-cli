// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `<unknown>` placeholder printed in place of a value that came back
//! empty.
//!
//! Rendered reports — `rocm install driver`'s plan, the chat tool summaries,
//! the Windows GPU diagnostics — print a field even when its source had
//! nothing for it, so a reader sees `<unknown>` rather than a dangling
//! `os_id: `. Keeping the rule here stops each renderer from growing its own
//! copy that quietly disagrees with the others.
//!
//! The two helpers differ only in whether surrounding whitespace counts as
//! empty. Pick the one the caller already had: switching a caller between
//! them changes what it prints for padded or whitespace-only input.

/// `value` unchanged, or `<unknown>` when it is the empty string.
///
/// Whitespace is kept as-is: `" "` is returned as `" "`, not `<unknown>`.
/// Use [`trimmed_or_unknown`] when padding should be dropped first.
#[must_use]
pub const fn empty_as_unknown(value: &str) -> &str {
    if value.is_empty() { "<unknown>" } else { value }
}

/// `value` with surrounding whitespace trimmed, or `<unknown>` when nothing
/// is left.
///
/// For raw probe output, where a field that is only padding means the probe
/// had nothing for it.
#[must_use]
pub fn trimmed_or_unknown(value: &str) -> &str {
    empty_as_unknown(value.trim())
}

#[cfg(test)]
mod tests {
    use super::{empty_as_unknown, trimmed_or_unknown};

    #[test]
    fn empty_as_unknown_replaces_only_the_empty_string() {
        assert_eq!(empty_as_unknown(""), "<unknown>");
        assert_eq!(empty_as_unknown("ubuntu"), "ubuntu");
        // Whitespace is a value here, not an absence: this helper must not
        // trim, or every caller that relied on it would change its output.
        assert_eq!(empty_as_unknown(" "), " ");
        assert_eq!(empty_as_unknown("\t\n"), "\t\n");
        assert_eq!(empty_as_unknown("  jammy "), "  jammy ");
    }

    #[test]
    fn trimmed_or_unknown_trims_and_treats_blank_as_empty() {
        assert_eq!(trimmed_or_unknown(""), "<unknown>");
        assert_eq!(trimmed_or_unknown(" "), "<unknown>");
        assert_eq!(trimmed_or_unknown("\t\r\n"), "<unknown>");
        assert_eq!(
            trimmed_or_unknown("  Radeon RX 7900 XTX\r\n"),
            "Radeon RX 7900 XTX"
        );
        assert_eq!(trimmed_or_unknown("Radeon"), "Radeon");
    }
}
