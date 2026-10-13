// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! One POSIX shell quoter for every command line this binary prints or runs.
//!
//! They look unrelated and are not:
//!
//! * [`crate::remote`] builds a command line that a shell on *another* machine
//!   will parse, so an unquoted value there is a remote command injection.
//! * [`crate::format_structured_tool_call`] builds the command line a human is
//!   shown — in the chat approval card, in `rocm <request>`'s request plan, and
//!   in the `run:` lines the CLI offers to be copied. An unquoted value there
//!   means the line the operator reads is not the argv that runs, and the
//!   copyable ones mean it is not what runs when they paste it either.
//! * [`crate::therock::quote_display_arg`] builds the `command: uv … && uv …`
//!   line the install dry-run prints. Its install root comes from a `--prefix`
//!   the local assistant can choose, and the line is offered to be run.
//! * [`crate::service_hints`] and the runtime-activation report in
//!   [`crate::runtime_services`] build the `rocm services … <id> --yes`,
//!   `rocm services logs <id>` and `rocm runtimes activate <key> …` lines a
//!   user is told to run next. A service id is not limited to inert
//!   characters, so a bare one could split into several words on paste.
//!
//! All of them want the same guarantee — *this value is one word and nothing
//! else* — so they share one implementation. Keeping them apart is how two of
//! them ended up with quoters that did not escape `$`, backticks or `;`, and
//! left a lone `"` unquoted entirely.
//!
//! That implementation is `rocm-dash-tui`'s `ui::exec::quote_display_arg`,
//! which also renders the dash's approval previews, and its tests — the
//! round-trip properties and the real-`sh` read-back — live beside it.
//! `apps/rocm` already depends on `rocm-dash-tui`, so delegating needs no new
//! crate edge; the opposite direction would (`rocm-dash-tui` may not depend on
//! `rocm-core`, see `xtask/src/crate_edges.rs`). This module is the CLI's name
//! for it, plus the control-character predicate the chat guards share.
//!
//! Scope: the surfaces above. Older `rocm …` hints elsewhere in the CLI that
//! interpolate a value without it are not covered by this module's guarantee.
//!
//! Single quotes, not double: inside `'...'` POSIX gives no expansion at all,
//! so the only character needing care is `'` itself. A double-quoted form has
//! to keep escaping `$`, `` ` `` and `\` correctly forever, and the one that
//! used to live in `main.rs` escaped the last two and not the first.
//!
//! Platform note: this is POSIX quoting on every platform. The previous
//! double-quoted form happened to paste cleanly into `cmd.exe`; `'…'` does not.
//! It is correct in any POSIX shell (bash, sh, WSL). PowerShell also reads
//! `'…'` literally, but escapes an embedded `'` as `''` rather than `'\''`, so a
//! value containing a single quote is the one case that does not paste cleanly
//! there. On Windows it is still unambiguous to a *reader* — which is what an
//! approval gate needs — where the old form was ambiguous everywhere.

/// Quote `value` so a POSIX shell reads it back as exactly one word equal to
/// `value`.
///
/// Values that cannot mean anything but themselves are returned bare, so an
/// ordinary command line stays readable: `rocm serve qwen2.5-7b --managed`,
/// not `rocm 'serve' 'qwen2.5-7b' '--managed'`. `=` is among them, which is
/// safe only after the program word — see the implementation's comment before
/// putting a quoted value in command position.
pub(crate) fn shell_quote(value: &str) -> String {
    rocm_dash_tui::ui::exec::quote_display_arg(value)
}

/// Whether `value` carries a character a terminal acts on rather than prints.
///
/// Quoting cannot help here, and that is the point. `'…'` makes a byte literal
/// to the *shell*, but the operator is not a shell: a `\r`, a backspace or a
/// CSI sequence inside the quotes still moves their cursor or erases the line
/// when the preview is printed, so the command they read can be chosen by
/// whoever supplied the argument. A value in this shape is refused upstream
/// rather than rendered.
///
/// The predicate is [`rocm_core::terminal::is_control_or_format`] — C0, DEL
/// and C1 (so ESC and the 8-bit CSI alike), plus the Unicode format characters:
/// a bidi override such as U+202E reorders how the rest of the line is drawn,
/// and a zero-width character hides a difference between what is shown and
/// what runs. U+2028/U+2029 are added because a terminal may start a new row
/// at either, which would let a value forge a line of its own.
pub(crate) fn has_terminal_control_character(value: &str) -> bool {
    value.chars().any(|character| {
        rocm_core::terminal::is_control_or_format(character)
            || matches!(character, '\u{2028}' | '\u{2029}')
    })
}

/// The argument generator the CLI's preview property tests share.
///
/// One copy in this crate, deliberately: a second generator here would not be
/// covered by [`tests::generator_reaches_the_shapes_that_break_quoting`], and
/// could be tuned into harmlessness with nothing to catch it. `rocm-dash-tui`
/// keeps its own for the quoter's tests, because a `#[cfg(test)]` item is not
/// visible across crates; the two alphabets are the same by choice.
#[cfg(test)]
pub(crate) mod hostile {
    use proptest::prelude::*;

    /// Deliberately tiny and deliberately hostile.
    ///
    /// A uniform `.*` generator is close to useless here: the shapes that break
    /// a quoter are a lone quote, a bare backslash, an empty string and the
    /// metacharacters, and uniform text produces them at a rate that never
    /// finds anything. Drawing from this alphabet puts a quote in ~34% of
    /// generated argvs and some metacharacter in ~82% (measured by
    /// `generator_reaches_the_shapes_that_break_quoting`, which prints the
    /// rates under `--nocapture`).
    ///
    /// `\t` is the one control character, and it is here as whitespace — a
    /// word separator to a shell — not as a control character. Control
    /// characters are refused upstream rather than quoted, so tests of that
    /// guard draw from `control_bearing_arg` in `main.rs` and must not count on
    /// this alphabet happening to contain one.
    ///
    /// `[ ] { } % , @ +` are here although most shells give them little or no
    /// meaning in an argument: `[`/`]` glob and `{`/`}` brace-expand in bash,
    /// so a renderer that left them bare would otherwise be caught only by a
    /// single example.
    const ALPHABET: &[char] = &[
        ' ', '\t', '"', '\'', '\\', '$', '`', ';', '|', '&', '*', '?', '(', ')', '<', '>', '#',
        '=', '-', '~', '!', '[', ']', '{', '}', '%', ',', '@', '+', 'a', '1', '/', 'é',
    ];

    pub(crate) fn arg() -> impl Strategy<Value = String> {
        prop_oneof![
            8 => proptest::collection::vec(proptest::sample::select(ALPHABET), 0..=5)
                .prop_map(|chars| chars.into_iter().collect::<String>()),
            // A handful of real argv shapes so ordinary usage is covered too —
            // including the path-with-a-space from the bug report.
            1 => proptest::sample::select(&[
                "install", "sdk", "--prefix", "/mnt/my folder",
                "C:\\Program Files\\ROCm", "qwen2.5-7b-instruct", "",
                "package[extra]",
            ][..]).prop_map(str::to_owned),
        ]
    }

    pub(crate) fn argv() -> impl Strategy<Value = Vec<String>> {
        proptest::collection::vec(arg(), 1..=4)
    }
}

/// The shell the real-`sh` tests ask, by absolute path.
///
/// Not `"sh"`: that is looked up on `PATH` at spawn time, and other tests in
/// this binary narrow the process-global `PATH` to a stub directory while they
/// run (`render_update_json` in `therock.rs`, for one). Under `cargo test`,
/// which runs every test in one process, a spawn landing in that window fails
/// with "No such file or directory" — a failure about the test harness, not
/// about quoting. `/bin/sh` is where POSIX systems put it.
#[cfg(all(test, unix))]
pub(crate) const TEST_SH: &str = "/bin/sh";

#[cfg(test)]
mod tests {
    use super::hostile::argv as hostile_argv;
    use super::*;

    // The quoter's own properties — the shlex round-trip and a real `sh`
    // reading the line back — are tested where it is implemented,
    // `rocm-dash-tui`'s `ui::exec`. What is tested here is this crate's: that
    // the generator the CLI's renderer tests draw from still reaches the
    // shapes that break quoting, and the control-character predicate.

    /// The CLI's entry point is the shared quoter, not a look-alike.
    #[test]
    fn the_cli_quotes_exactly_as_the_shared_quoter_does() {
        for value in [
            "qwen2.5-7b",
            "/mnt/my folder",
            "it's",
            "",
            "$HOME",
            "C:\\ROCm",
        ] {
            assert_eq!(
                shell_quote(value),
                rocm_dash_tui::ui::exec::quote_display_arg(value),
                "{value:?}"
            );
        }
    }

    #[test]
    fn generator_reaches_the_shapes_that_break_quoting() {
        // A generator that never produces the dangerous shapes passes against a
        // broken quoter — that has happened here before. Measure it, and fail
        // if the alphabet is ever tuned into harmlessness.
        use proptest::strategy::{Strategy, ValueTree};
        use proptest::test_runner::TestRunner;

        let mut runner = TestRunner::deterministic();
        let strategy = hostile_argv();
        let (mut quote, mut meta, mut space, mut empty, mut backslash) = (0, 0, 0, 0, 0);
        let total = 5_000;
        for _ in 0..total {
            let argv = strategy
                .new_tree(&mut runner)
                .expect("generator produces a value")
                .current();
            let joined = argv.concat();
            quote += i32::from(joined.contains('\'') || joined.contains('"'));
            meta += i32::from(joined.contains(|c| "$`;|&*?()<>".contains(c)));
            space += i32::from(joined.contains(' ') || joined.contains('\t'));
            empty += i32::from(argv.iter().any(String::is_empty));
            backslash += i32::from(joined.contains('\\'));
        }
        // Floors, not exact rates: the point is that each dangerous shape is
        // common, not that the generator is frozen.
        for (label, count, floor) in [
            ("quote", quote, total / 5),
            ("metacharacter", meta, total / 2),
            ("whitespace", space, total / 5),
            ("empty argument", empty, total / 5),
            ("backslash", backslash, total / 10),
        ] {
            // Printed so `--nocapture` reports the generator's actual reach, not
            // only that it cleared the bar.
            println!(
                "reach: {label:<16} {count:>5}/{total}  ({:.1}%)",
                f64::from(count) * 100.0 / f64::from(total)
            );
            assert!(
                count >= floor,
                "generator produced {label} in only {count}/{total} cases; \
                 below {floor} it stops exercising the quoting it is meant to test"
            );
        }
    }

    #[test]
    fn control_characters_are_recognised_whatever_their_width() {
        for value in [
            "a\rb",
            "a\nb",
            "a\u{1b}[2Kb",
            "a\u{8}b",
            "a\u{7f}b",
            "a\u{9b}b",
            // Format and line-separator characters: no cursor movement, but a
            // right-to-left override reorders everything drawn after it, a
            // zero-width space hides a difference, and a line separator may
            // start a new row.
            "a\u{202e}b",
            "a\u{2066}b",
            "a\u{200b}b",
            "a\u{feff}b",
            "a\u{2028}b",
            "a\u{2029}b",
        ] {
            assert!(
                has_terminal_control_character(value),
                "{value:?} should be refused before it reaches a preview"
            );
        }
        for value in ["/mnt/my folder", "qwen2.5-7b", "x'; rm -rf ~; echo '", "é"] {
            assert!(!has_terminal_control_character(value), "{value:?}");
        }
    }
}
