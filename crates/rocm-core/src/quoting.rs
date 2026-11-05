// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Argument-quoting helpers whose output must read back as exactly the value given.
//!
//! The output is either embedded into a command string a shell or interpreter
//! actually runs, or shown as a command a person is asked to approve or paste.
//!
//! [`posix_shell_quote`] and [`powershell_quote`]/`ps_single_quote_escape`
//! are two different escaping domains -- POSIX `sh`/`bash` single-quoting and
//! PowerShell single-quoting do not share a convention -- that happen to live
//! in the same module because both have the same kind of stakes:
//! `posix_shell_quote`'s output is sent over SSH to a remote shell
//! (`apps/rocm/src/remote/mod.rs`'s `remote_serve_command` and friends), and
//! `powershell_quote`'s output is written into a `.ps1` script executed via
//! `powershell.exe -File` (`apps/rocm/src/comfyui/process.rs`'s Windows
//! runtime launch) or interpolated into a `-Command` string
//! (`crate::fix::path_export_windows`, `crate::diagnose`'s matching
//! printed copy). A mistake in either one is a real command-injection or
//! script-corruption bug, which is why both carry direct unit coverage here
//! (ROCMAI-727 consolidated them out of per-call-site near-duplicates).
//!
//! [`posix_shell_quote`] is also what `apps/rocm`'s `shell_quote` uses for
//! the command lines the CLI *shows*: the chat approval card, the free-form
//! request plan, the install dry-run's `command:` line and the `rocm …` hints
//! it tells a user to run. Those are offered to be copied, and an approval is
//! given for what they say, so a value that reads back as a different argv is
//! the same defect whether a shell or a person runs it.
//! `crates/rocm-dash-tui/src/ui/exec.rs`'s `quote_display_arg` is a copy for
//! the dash's previews, because `rocm-dash-tui` may not depend on this crate;
//! two tests in `apps/rocm/src/shell_quote.rs` hold the two to the
//! same output.
//!
//! What is deliberately **not** here: `apps/rocm/src/comfyui.rs`'s
//! `quote_log_arg` (log text, Rust's own `{value:?}` Debug-format escaping,
//! with the real unquoted args going to `Command::args` alongside it), which
//! uses a different convention; unifying it would change logged output.

/// Escapes a value for embedding inside a PowerShell single-quoted string
/// literal.
///
/// PowerShell single-quoted strings are literal, and the language's own
/// convention for including a literal `'` in one is to double it (`''`) --
/// an unescaped `'` instead ends the literal early, so whatever follows on
/// the line is parsed as separate PowerShell tokens.
///
/// This matters for any value of filesystem/environment origin embedded into
/// a generated PowerShell command -- for example
/// `crate::fix::path_export_windows`'s write script, built from the real
/// existing User PATH read back from the host, and `crate::diagnose`'s
/// matching printed copy of that same command. A path containing an
/// apostrophe (`C:\Users\O'Brien\bin` is an ordinary Windows path, not a
/// contrived edge case) would otherwise split the generated `-Command`
/// script mid-statement -- this is the ROCMAI-449 bug, present on both
/// surfaces until the write and the printed copy shared this one escape.
///
/// See also [`powershell_quote`], which wraps this escape in the surrounding
/// `'...'` delimiters for the common case of quoting a whole argument rather
/// than splicing an escaped fragment into an existing literal.
pub(crate) fn ps_single_quote_escape(value: &str) -> String {
    value.replace('\'', "''")
}

/// Quote `value` as a single PowerShell argument: wraps it in a
/// single-quoted string literal (`'...'`), escaping any embedded `'` first
/// via `ps_single_quote_escape`.
///
/// Used wherever a value is written into a generated `.ps1` script or
/// `-Command` string as one argument, rather than spliced into an
/// already-open literal (that case calls `ps_single_quote_escape`
/// directly).
pub fn powershell_quote(value: &str) -> String {
    format!("'{}'", ps_single_quote_escape(value))
}

/// Quote `value` for safe embedding in a POSIX shell (`sh`/`bash`) command
/// line, as a single word.
///
/// Values made entirely of characters inert to shell parsing are returned
/// unquoted for readability; anything else is wrapped in `'...'`, with each
/// embedded `'` closed, escaped, and reopened (`'\''`) -- the standard POSIX
/// trick for including a literal single quote inside a single-quoted string,
/// since POSIX shells have no escape sequence that works *inside* one.
///
/// Its output is embedded into a command string sent over SSH to a remote
/// shell (`apps/rocm/src/remote/mod.rs`), and it renders the command lines the
/// CLI shows for approval or pasting, so a hostile value (`$(id)`, `` `id` ``,
/// `; rm -rf ~`) must come back as inert, byte-identical shell text. See the
/// `hostile_values_survive_a_real_shell_as_one_literal_argument` test below,
/// which execs a real `sh -c` and round-trips such inputs rather than only
/// asserting on the escaped string's shape.
///
/// `=` is left bare, which is safe only *after* the command word: POSIX gives
/// it meaning only in a word ahead of the command name, where `A=b` is an
/// assignment rather than an argument. Every caller quotes arguments after a
/// fixed program word, never argv[0]. Nothing enforces that, so a caller that
/// puts a quoted value in command position must quote `=` as well. zsh is the
/// other exception — with its default `EQUALS` option a word *starting* with
/// `=` expands to a command path — and is outside the POSIX contract stated
/// here.
pub fn posix_shell_quote(value: &str) -> String {
    let inert = |character: char| {
        character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | '/' | ':' | '=')
    };
    if !value.is_empty() && value.chars().all(inert) {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ps_single_quote_escape` must double every embedded `'`, per
    /// PowerShell's own single-quoted-string escaping convention.
    #[test]
    fn ps_single_quote_escape_doubles_every_embedded_quote() {
        assert_eq!(ps_single_quote_escape("plain"), "plain");
        assert_eq!(ps_single_quote_escape("O'Brien"), "O''Brien");
        assert_eq!(
            ps_single_quote_escape("'leading and trailing'"),
            "''leading and trailing''"
        );
        assert_eq!(ps_single_quote_escape("a'b'c"), "a''b''c");
    }

    #[test]
    fn powershell_quote_wraps_the_escaped_value_in_single_quotes() {
        assert_eq!(powershell_quote("plain"), "'plain'");
        assert_eq!(powershell_quote("O'Brien"), "'O''Brien'");
        assert_eq!(powershell_quote(""), "''");
        assert_eq!(powershell_quote("a b"), "'a b'");
    }

    #[test]
    fn shell_quoting_leaves_ordinary_values_alone_and_wraps_the_rest() {
        for inert in ["qwen2.5-7b-instruct", "vllm", "/models/a.gguf", "auto"] {
            assert_eq!(posix_shell_quote(inert), inert);
        }
        assert_eq!(posix_shell_quote("a b"), "'a b'");
        assert_eq!(posix_shell_quote(""), "''");
        assert_eq!(posix_shell_quote("it's"), r"'it'\''s'");
    }

    #[cfg(unix)]
    #[test]
    fn hostile_values_survive_a_real_shell_as_one_literal_argument() {
        // The property that matters is not the shape of the quoting but what a
        // shell does with it. Ask one: each value must come back byte-identical,
        // proving it was neither expanded nor split nor able to start a second
        // command.
        for value in [
            "x'; rm -rf ~; echo '",
            "$(id)",
            "`id`",
            "a b",
            "it's",
            "*",
            "--not-a-flag",
            "qwen2.5-7b-instruct",
        ] {
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!("printf %s {}", posix_shell_quote(value)))
                .output()
                .expect("sh should run");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                value,
                "shell mangled {value:?}"
            );
        }
    }
}
