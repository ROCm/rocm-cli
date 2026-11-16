// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Lexical helpers shared by the source-scan contract tests.
//!
//! The contracts in this crate read Rust sources as text. Each of them has to
//! look at code only: a brace, a call or a macro name inside a string literal
//! or a comment must not count. [`strip_literals_and_comments`] blanks those
//! out while keeping line structure, so every contract that needs the question
//! answered gets the same answer.

/// Replace the CONTENTS of string literals, char literals and comments with
/// spaces, preserving line structure and everything outside them.
///
/// A scan that counts braces or looks for call text is wrong if a literal can
/// contribute either. Two real files in this repo break the naive version, as
/// `env_mutation_contract.rs` found:
///
/// * `engines/lemonade/src/lib.rs` contains
///   `ensure_cached_archive("test://archive", ..., |_, destination| {`.
///   Truncating the line at `//` throws away the closure's opening brace,
///   so the enclosing test module reads as closed hundreds of lines early
///   and every mutation after it becomes invisible.
/// * `tests/e2e-cucumber/src/mock_server.rs` contains format strings such
///   as `vllm:num_requests_running{{model=...}}`. Those braces inflate the
///   depth, which drags PRODUCTION code into the scanned region and makes
///   the guard fail a build for a mutation it should not be watching.
///
/// Both directions are silent, which is the one failure mode a permanent
/// guard must not have, so literals are tokenized rather than approximated.
pub(crate) fn strip_literals_and_comments(text: &str) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        LineComment,
        BlockComment(usize),
        Str,
        RawStr(usize),
        Char,
    }

    let mut out = String::with_capacity(text.len());
    let mut state = State::Code;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match state {
            State::Code => {
                // A raw string opener: `r"`, `r#"`, `br##"` and so on. Checked
                // before the plain-string case so the hashes are counted.
                if (c == 'r' || c == 'b')
                    && let Some((hashes, consumed)) = raw_string_opener(&chars, i)
                {
                    out.extend(std::iter::repeat_n(' ', consumed));
                    state = State::RawStr(hashes);
                    i += consumed;
                    continue;
                }
                if c == '/' && next == Some('/') {
                    state = State::LineComment;
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                if c == '/' && next == Some('*') {
                    state = State::BlockComment(1);
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                if c == '"' {
                    state = State::Str;
                    out.push(' ');
                    i += 1;
                    continue;
                }
                // A lifetime (`'a`) is not a char literal. A char literal is
                // `'x'` or `'\n'`, so require a closing quote nearby.
                if c == '\'' && is_char_literal(&chars, i) {
                    state = State::Char;
                    out.push(' ');
                    i += 1;
                    continue;
                }
                out.push(c);
                i += 1;
            }
            State::LineComment => {
                if c == '\n' {
                    state = State::Code;
                    out.push('\n');
                } else {
                    out.push(' ');
                }
                i += 1;
            }
            State::BlockComment(depth) => {
                if c == '*' && next == Some('/') {
                    state = if depth == 1 {
                        State::Code
                    } else {
                        State::BlockComment(depth - 1)
                    };
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                if c == '/' && next == Some('*') {
                    state = State::BlockComment(depth + 1);
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                out.push(if c == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            State::Str => {
                if c == '\\' {
                    // The escaped character cannot close the literal, so
                    // it is consumed here. When it is a NEWLINE -- a line
                    // continuation, which several strings in this tree use,
                    // including this file's own assertion message -- the
                    // newline is not part of the string's value but it is
                    // part of the file's line structure. Replacing it with
                    // a space merges two source lines and shifts the line
                    // number of every offense reported after it, which is
                    // the `file:line` the assertion promises.
                    out.push(' ');
                    out.push(if next == Some('\n') { '\n' } else { ' ' });
                    i += 2;
                    continue;
                }
                if c == '"' {
                    state = State::Code;
                }
                out.push(if c == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            State::RawStr(hashes) => {
                if c == '"' && closing_hashes(&chars, i + 1) >= hashes {
                    state = State::Code;
                    out.extend(std::iter::repeat_n(' ', hashes + 1));
                    i += hashes + 1;
                    continue;
                }
                out.push(if c == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            State::Char => {
                if c == '\\' {
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                if c == '\'' {
                    state = State::Code;
                }
                out.push(' ');
                i += 1;
            }
        }
    }
    out
}

/// `Some((hash_count, chars_consumed))` when a raw-string literal opens at
/// `i` (`r"`, `r#"`, `br##"`, ...).
fn raw_string_opener(chars: &[char], i: usize) -> Option<(usize, usize)> {
    let mut j = i;
    // Accepting the byte-string `b` here is cosmetic, and deliberately has
    // no fixture: removing it only means the caller declines at the `b` and
    // matches one character later at the `r`, which opens the same literal
    // and strips the same span. The single character of difference is the
    // `b` itself, emitted as code rather than as a space. In
    // `env_mutation_contract.rs`, four things read
    // the stripped text — the brace counter, the names in `MUTATIONS`,
    // [`serializer_column`] and [`has_test_attribute`] — and a `b` whose only
    // neighbour on the right is the stripped literal's run of spaces cannot
    // complete a brace, a `set_var(`, a `_TEST_LOCK`/`ScopedTestEnv`, or a
    // `#[`. Equivalent on every path that reads it, so a test for it would
    // assert nothing.
    if chars.get(j) == Some(&'b') {
        j += 1;
    }
    if chars.get(j) != Some(&'r') {
        return None;
    }
    j += 1;
    let hash_start = j;
    while chars.get(j) == Some(&'#') {
        j += 1;
    }
    if chars.get(j) == Some(&'"') {
        Some((j - hash_start, j - i + 1))
    } else {
        None
    }
}

fn closing_hashes(chars: &[char], mut i: usize) -> usize {
    let start = i;
    while chars.get(i) == Some(&'#') {
        i += 1;
    }
    i - start
}

/// Distinguish `'x'` from a lifetime such as `'a` in `&'a str`.
fn is_char_literal(chars: &[char], i: usize) -> bool {
    match chars.get(i + 1) {
        // `'\n'`, `'\''`, `'\\'` -- always a literal.
        Some('\\') => true,
        Some(_) => chars.get(i + 2) == Some(&'\''),
        // A `'` as the final character of the file. Deliberately untested:
        // the two answers are indistinguishable, because the caller's loop
        // ends on the next step either way and the one character of output
        // that differs (`'` versus a space) is neither a brace nor part of
        // a mutating call's name. Flipping this arm is an equivalent
        // mutant, so a fixture for it would assert nothing.
        None => false,
    }
}
