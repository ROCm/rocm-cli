// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shell quoting for the scripts a scenario plants as stand-in commands.
//!
//! Shared by every stand-in script writer (`stand_in_stat`, `stand_in_uname`),
//! so the quoting they depend on is written and tested once.

/// `value` as one single-quoted shell word: each embedded single quote closes the
/// quotes, is written escaped and reopens them. (`rocm_core::quoting` has the same
/// quoting, but `rocm-core` is only a dev-dependency of this library.)
pub(crate) fn single_quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::single_quoted;

    #[test]
    fn a_plain_value_is_wrapped_in_single_quotes() {
        assert_eq!(single_quoted("a b"), "'a b'");
    }

    #[test]
    fn an_embedded_single_quote_closes_escapes_and_reopens() {
        assert_eq!(single_quoted("it's"), r"'it'\''s'");
    }

    #[test]
    fn shell_metacharacters_stay_inside_the_quotes() {
        assert_eq!(single_quoted("$HOME `id`; x"), "'$HOME `id`; x'");
    }
}
