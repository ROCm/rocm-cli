// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Maintainable fixture generation for the PowerShell installer's pinned keys.
//!
//! Lifecycle tests need controlled current/next trust-root slots without changing
//! the real root installer. Rather than matching a multi-line PowerShell
//! here-string with a broad regular expression, this module parses only the two
//! exact top-level assignments it owns. Each assignment may be an empty string or
//! a double-quoted here-string; any other shape, duplicate, or missing assignment
//! fails explicitly so an installer refactor cannot silently produce a bad fixture.

use std::fmt::Write as _;

const CURRENT_KEY: &str = "PinnedReleasePublicKeyCurrent";
const NEXT_KEY: &str = "PinnedReleasePublicKeyNext";

/// Return an installer fixture with controlled pinned current and next public
/// keys. Empty values are emitted as `""`; non-empty values use a PowerShell
/// double-quoted here-string.
pub fn with_pinned_release_keys(
    installer: &str,
    current_key: &str,
    next_key: &str,
) -> Result<String, String> {
    let installer = replace_assignment(installer, CURRENT_KEY, current_key)?;
    replace_assignment(&installer, NEXT_KEY, next_key)
}

fn replace_assignment(source: &str, variable: &str, value: &str) -> Result<String, String> {
    let lines: Vec<&str> = source.lines().collect();
    let assignment = format!("${variable} =");
    let matches: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| line.trim_start().starts_with(&assignment).then_some(index))
        .collect();
    let [start] = matches.as_slice() else {
        return Err(format!(
            "expected exactly one ${variable} assignment, found {}",
            matches.len()
        ));
    };

    let trimmed = lines[*start].trim();
    let rhs = trimmed
        .strip_prefix(&assignment)
        .expect("matched assignment prefix")
        .trim();
    let end = match rhs {
        "\"\"" => *start,
        "@\"" => lines
            .iter()
            .enumerate()
            .skip(*start + 1)
            .find_map(|(index, line)| (line.trim() == "\"@").then_some(index))
            .ok_or_else(|| format!("unterminated ${variable} here-string"))?,
        other => {
            return Err(format!(
                "unsupported ${variable} assignment value {other:?}; expected \"\" or @\""
            ));
        }
    };

    let mut output = String::new();
    for line in &lines[..*start] {
        writeln!(output, "{line}").expect("writing to String cannot fail");
    }
    write_assignment(&mut output, variable, value);
    for line in &lines[end + 1..] {
        writeln!(output, "{line}").expect("writing to String cannot fail");
    }
    Ok(output)
}

fn write_assignment(output: &mut String, variable: &str, value: &str) {
    if value.trim().is_empty() {
        writeln!(output, "${variable} = \"\"").expect("writing to String cannot fail");
    } else {
        writeln!(output, "${variable} = @\"").expect("writing to String cannot fail");
        writeln!(output, "{}", value.trim()).expect("writing to String cannot fail");
        writeln!(output, "\"@").expect("writing to String cannot fail");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INSTALLER: &str = r#"before
$PinnedReleasePublicKeyCurrent = @"
-----BEGIN PUBLIC KEY-----
original
-----END PUBLIC KEY-----
"@
between
$PinnedReleasePublicKeyNext = ""
after
"#;

    #[test]
    fn replaces_here_string_and_empty_slots_without_touching_other_content() {
        let result = with_pinned_release_keys(INSTALLER, "malformed", "valid").unwrap();
        assert_eq!(
            result,
            "before\n\
$PinnedReleasePublicKeyCurrent = @\"\n\
malformed\n\
\"@\n\
between\n\
$PinnedReleasePublicKeyNext = @\"\n\
valid\n\
\"@\n\
after\n"
        );
    }

    #[test]
    fn transforms_the_real_installer_shape() {
        let installer = include_str!("../../../install.ps1");
        let result = with_pinned_release_keys(installer, "malformed", "valid").unwrap();
        assert_eq!(
            result.matches("$PinnedReleasePublicKeyCurrent =").count(),
            1
        );
        assert_eq!(result.matches("$PinnedReleasePublicKeyNext =").count(), 1);
        assert!(result.contains("$PinnedReleasePublicKeyCurrent = @\"\nmalformed\n\"@"));
        assert!(result.contains("$PinnedReleasePublicKeyNext = @\"\nvalid\n\"@"));
        assert!(result.contains("function Test-HasPinnedReleaseKey"));
    }

    #[test]
    fn emits_empty_slots_as_empty_strings() {
        let result = with_pinned_release_keys(INSTALLER, "", "").unwrap();
        assert!(result.contains("$PinnedReleasePublicKeyCurrent = \"\"\n"));
        assert!(result.contains("$PinnedReleasePublicKeyNext = \"\"\n"));
        assert!(!result.contains("original"));
    }

    #[test]
    fn rejects_missing_duplicate_and_unsupported_assignments() {
        let missing = with_pinned_release_keys("nothing\n", "a", "b").unwrap_err();
        assert!(missing.contains("found 0"));

        let duplicate = format!("{INSTALLER}$PinnedReleasePublicKeyCurrent = \"\"\n");
        let duplicate = with_pinned_release_keys(&duplicate, "a", "b").unwrap_err();
        assert!(duplicate.contains("found 2"));

        let unsupported = INSTALLER.replace(
            "$PinnedReleasePublicKeyNext = \"\"",
            "$PinnedReleasePublicKeyNext = 'value'",
        );
        let unsupported = with_pinned_release_keys(&unsupported, "a", "b").unwrap_err();
        assert!(unsupported.contains("unsupported"));
    }

    #[test]
    fn rejects_unterminated_here_string() {
        let unterminated = INSTALLER.replace("\"@\nbetween", "between");
        let error = with_pinned_release_keys(&unterminated, "a", "b").unwrap_err();
        assert!(error.contains("unterminated"));
    }
}
