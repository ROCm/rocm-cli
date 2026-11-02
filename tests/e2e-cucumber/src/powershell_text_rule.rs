// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The text rule for the lines `rocm fix` and `rocm diagnose` print for Windows.
//!
//! This is a copy of `fails_in_windows_powershell` in `crates/rocm-core/src/fix.rs`
//! (test-only there; this crate does not depend on rocm-core, and
//! `cargo xtask check-crate-edges` is not to be bypassed). Two copies drift, so
//! both are tested against the one table in
//! `fixtures/powershell_text_rule_cases.txt`: the test below reads it here, and a
//! rocm-core test reads the same file. A case added there that one rule does not
//! handle fails that side's test.
//!
//! The rule is a text check over known-bad constructs. It is not a parser and
//! does not prove a line runs.

/// Why `line` would break in Windows PowerShell 5.1 or cmd, or `None` when the
/// rule finds nothing wrong with it.
#[must_use]
pub fn refusal(line: &str) -> Option<&'static str> {
    if line.contains("&&") || line.contains("||") {
        return Some("`&&` or `||`, a syntax error in Windows PowerShell 5.1");
    }
    let lowered = line.to_ascii_lowercase();
    let command = command_argument(&lowered);
    let starts_bare_where = |text: &str| {
        text.split(['|', ';', '&', '(', '\u{7b}']).any(|part| {
            let mut words = part.split_whitespace().map(|w| w.trim_matches('"'));
            let mut first = words.next();
            if first == Some("cmd") {
                first = words.find(|w| !w.starts_with('/'));
            }
            first == Some("where")
        })
    };
    if starts_bare_where(&lowered)
        || command.is_some_and(|(_, _, body)| starts_bare_where(body.trim_start()))
    {
        return Some("a bare `where`, which is `Where-Object` in PowerShell");
    }
    let outer = command.map_or(lowered.as_str(), |(at, _, _)| &lowered[..at]);
    if outer
        .split_whitespace()
        .any(|word| word == "\"\"" || word == "''")
    {
        return Some("an empty-string argument, which Windows PowerShell 5.1 drops");
    }
    if let Some((_, quote, body)) = command {
        if quote == '\'' {
            return Some("a single-quoted `-Command` argument, which cmd does not take");
        }
        if body.contains("\\\"") {
            return Some("a backslash-escaped quote inside a `-Command` body");
        }
        if body.contains('$') {
            return Some("a `$` inside a double-quoted `-Command` body");
        }
        if body.contains("get-item") && !body.contains("-erroraction stop") {
            return Some("a `Get-Item` without `-ErrorAction Stop`");
        }
    }
    let pieces: Vec<&str> = line.split('%').collect();
    let names_a_cmd_variable = pieces
        .get(1..pieces.len().saturating_sub(1))
        .unwrap_or_default()
        .iter()
        .any(|name| {
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        });
    names_a_cmd_variable.then_some("a `%NAME%` reference, which PowerShell does not expand")
}

/// Where a `powershell ... -Command "..."` argument starts in `lowered`: the
/// offset of the flag, the quote that opens the argument, and the text after
/// it. PowerShell takes any prefix of `-Command` down to `-c`.
fn command_argument(lowered: &str) -> Option<(usize, char, &str)> {
    lowered.match_indices('-').find_map(|(at, _)| {
        let head = &lowered[..at];
        if head.chars().next_back().is_some_and(|c| !c.is_whitespace())
            || !(head.contains("powershell") || head.contains("pwsh"))
        {
            return None;
        }
        let rest = &lowered[at..];
        let flag = rest.split_whitespace().next()?;
        if flag.len() < 2 || !"-command".starts_with(flag) {
            return None;
        }
        let after = rest[flag.len()..].trim_start();
        let quote = after.chars().next().filter(|c| matches!(c, '"' | '\''))?;
        Some((at, quote, &after[1..]))
    })
}

#[cfg(test)]
mod tests {
    use super::refusal;

    /// The cases both copies of the rule are held to.
    const CASES: &str = include_str!("../fixtures/powershell_text_rule_cases.txt");

    #[test]
    fn the_rule_agrees_with_the_shared_case_table() {
        let mut flagged = 0;
        let mut passed = 0;
        for (number, raw) in CASES.lines().enumerate() {
            let raw = raw.trim_end();
            if raw.is_empty() || raw.starts_with('#') {
                continue;
            }
            if let Some(line) = raw.strip_prefix("flag ") {
                assert!(refusal(line).is_some(), "line {}: `{line}`", number + 1);
                flagged += 1;
            } else if let Some(line) = raw.strip_prefix("pass ") {
                assert_eq!(refusal(line), None, "line {}: `{line}`", number + 1);
                passed += 1;
            } else {
                panic!("line {}: neither `flag` nor `pass`: {raw}", number + 1);
            }
        }
        // The table was read, not skipped.
        assert!(
            flagged >= 20 && passed >= 15,
            "{flagged} flag, {passed} pass"
        );
    }
}
