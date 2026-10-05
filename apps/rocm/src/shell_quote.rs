// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! One POSIX shell quoter for this binary's three command lines.
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
//!
//! All three want the same guarantee — *this value is one word and nothing
//! else* — so they share one implementation and one set of tests. Keeping them
//! apart is how two of them ended up with quoters that did not escape `$`,
//! backticks or `;`, and left a lone `"` unquoted entirely.
//!
//! The fourth is in another crate and stays there: `rocm-dash-tui`'s
//! `ui::exec::quote_display_arg`. `crates/rocm-dash-tui` must not depend on
//! `rocm-core` — an invariant the workspace enforces, not a preference — and
//! `apps/rocm` is the crate that owns `rocm-core`, so nothing here can be
//! shared with it without inverting that. Four lines of POSIX quoting with its
//! own copy of these property tests is the smaller price.
//!
//! Single quotes, not double: inside `'...'` POSIX gives no expansion at all,
//! so the only character needing care is `'` itself. A double-quoted form has
//! to keep escaping `$`, `` ` `` and `\` correctly forever, and the one that
//! used to live in `main.rs` escaped the last two and not the first.
//!
//! Platform note: this is POSIX quoting on every platform. The previous
//! double-quoted form happened to paste cleanly into `cmd.exe`; `'…'` does not.
//! It is correct in bash, WSL and PowerShell, and on Windows it is still
//! unambiguous to a *reader* — which is what an approval gate needs — where the
//! old form was ambiguous everywhere.

/// Quote `value` so a POSIX shell reads it back as exactly one word equal to
/// `value`.
///
/// Values that cannot mean anything but themselves are returned bare, so an
/// ordinary command line stays readable: `rocm serve qwen2.5-7b --managed`,
/// not `rocm 'serve' 'qwen2.5-7b' '--managed'`.
pub(crate) fn shell_quote(value: &str) -> String {
    let inert = |character: char| {
        character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | '/' | ':' | '=')
    };
    if !value.is_empty() && value.chars().all(inert) {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', r"'\''"))
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
/// `char::is_control` covers C0, DEL and C1 (so ESC and the 8-bit CSI alike).
pub(crate) fn has_terminal_control_character(value: &str) -> bool {
    value.chars().any(char::is_control)
}

/// The argument generator the preview property tests share.
///
/// One copy, deliberately: a second generator somewhere else would not be
/// covered by [`tests::generator_reaches_the_shapes_that_break_quoting`], and
/// could be tuned into harmlessness with nothing to catch it.
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
    /// No control characters: those are refused upstream rather than quoted, so
    /// they belong to that guard's own generator, not to this one.
    const ALPHABET: &[char] = &[
        ' ', '\t', '"', '\'', '\\', '$', '`', ';', '|', '&', '*', '?', '(', ')', '<', '>', '#',
        '=', '-', '~', '!', 'a', '1', '/', 'é',
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
            ][..]).prop_map(str::to_owned),
        ]
    }

    pub(crate) fn argv() -> impl Strategy<Value = Vec<String>> {
        proptest::collection::vec(arg(), 1..=4)
    }
}

#[cfg(test)]
mod tests {
    use super::hostile::{arg as hostile_arg, argv as hostile_argv};
    use super::*;
    use proptest::prelude::*;

    /// Render an argv the way every display site does, then split it back the
    /// way a shell would. `shlex` is the oracle: it is a POSIX word-splitter,
    /// which is precisely the reader the rendered line is written for.
    fn render_and_split(argv: &[String]) -> Option<Vec<String>> {
        let rendered = argv
            .iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" ");
        shlex::split(&rendered)
    }

    proptest! {
        /// The property everything else rests on: what is rendered denotes the
        /// argv that runs. Not "looks similar" — splits back to it exactly.
        #[test]
        fn a_rendered_argv_splits_back_to_the_same_argv(argv in hostile_argv()) {
            let split = render_and_split(&argv);
            prop_assert_eq!(split, Some(argv));
        }

        /// Quoting is total: no argument content can make the renderer produce
        /// something a shell cannot parse at all.
        #[test]
        fn every_argument_renders_to_something_parseable(arg in hostile_arg()) {
            prop_assert!(shlex::split(&shell_quote(&arg)).is_some(), "arg={:?}", arg);
        }

        /// Quoting an already-quoted token does not corrupt it: the second pass
        /// yields a word whose value is the first pass's output, verbatim.
        #[test]
        fn quoting_an_already_quoted_value_still_round_trips(arg in hostile_arg()) {
            let once = shell_quote(&arg);
            let twice = shell_quote(&once);
            prop_assert_eq!(shlex::split(&twice), Some(vec![once]));
        }

        /// An argument can never introduce a second command or a redirect: after
        /// rendering, the whole line is one word per argument, so a `;`, `|` or
        /// `>` inside a value is data, never syntax.
        #[test]
        fn an_argument_can_never_add_a_word(argv in hostile_argv()) {
            let split = render_and_split(&argv).expect("rendered line must parse");
            prop_assert_eq!(split.len(), argv.len());
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

    /// A readability check, not the regression guard — the old quoter also
    /// round-tripped this one (it wrapped on whitespace), so this cannot fail on
    /// the reported bug. What pins that is
    /// `an_install_prefix_with_a_space_is_one_argument_in_the_preview` in
    /// `main.rs`, which asserts the rendered form and does fail on a revert.
    #[test]
    fn a_path_with_a_space_stays_one_argument() {
        let argv = [
            "install".to_owned(),
            "sdk".to_owned(),
            "--prefix".to_owned(),
            "/mnt/my folder".to_owned(),
        ];
        assert_eq!(render_and_split(&argv), Some(argv.to_vec()));
    }

    #[test]
    fn ordinary_values_are_left_alone_and_the_rest_are_wrapped() {
        for inert in ["qwen2.5-7b-instruct", "vllm", "/models/a.gguf", "auto"] {
            assert_eq!(shell_quote(inert), inert);
        }
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
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
            "/mnt/my folder",
        ] {
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!("printf %s {}", shell_quote(value)))
                .output()
                .expect("sh should run");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                value,
                "shell mangled {value:?}"
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
