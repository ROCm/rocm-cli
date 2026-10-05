// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared helpers for launching `rocm` sub-commands from operational screens (Phase 3 Wave 1).
//!
//! Every screen that routes a mutating action through the
//! approval gate + job-bridge resolves the binary the same way, so the logic
//! lives here once instead of being re-implemented per screen.

/// The `rocm` binary to invoke: this process's own path (so an in-tree dev
/// build calls itself), or the bare name `rocm` (PATH lookup) when
/// `current_exe()` is unavailable — never a silent no-op.
pub fn resolve_exe() -> String {
    std::env::current_exe()
        .ok()
        .map_or_else(|| "rocm".to_string(), |p| p.to_string_lossy().into_owned())
}

/// Short, human-readable basename of a resolved command, for approval previews.
pub fn exe_label(cmd: &str) -> &str {
    cmd.rsplit(['/', '\\']).next().unwrap_or(cmd)
}

/// Quote one argument so a POSIX shell reads it back as exactly one word.
///
/// An approval preview is the only thing the operator sees before approving,
/// so it has to denote the argv the job is spawned with — not merely mark
/// where a value with a space ends. A double-quoted form does not get there:
/// inside `"..."` a shell still expands `$` and `` ` ``, so `$HOME` or
/// `` `id` `` in a value would read as something other than itself, and a
/// value ending in `\` escapes its own closing quote.
///
/// Single quotes do: inside `'...'` POSIX performs no expansion at all, so the
/// only character needing care is `'` itself, closed and reopened as `'\''`.
///
/// Values that cannot mean anything but themselves are returned bare, so an
/// ordinary preview stays readable — `rocm serve qwen2.5-7b --managed`, not
/// `rocm 'serve' 'qwen2.5-7b' '--managed'`.
///
/// Mirrors the CLI-side quoter in `apps/rocm/src/shell_quote.rs`. Deliberately
/// a second copy: `xtask`'s crate-edge contract forbids
/// `rocm-dash-tui -> rocm-core` (`xtask/src/crate_edges.rs`), and `apps/rocm`
/// is the crate that owns `rocm-core`, so there is nowhere both can reach
/// without breaking that. Nothing ties the two copies to each other: each is
/// held to the same contract by its own tests, including a real `sh` reading
/// the rendered line back, so either one *breaking* fails its own suite. They
/// may still differ harmlessly in which values they leave bare.
pub fn quote_display_arg(value: &str) -> String {
    let inert = |character: char| {
        character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | '/' | ':' | '=')
    };
    if !value.is_empty() && value.chars().all(inert) {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', r"'\''"))
}

/// Join `args` into a single display string, quoting any argument that needs
/// it so an approval preview or job-console title unambiguously shows where
/// each argument begins and ends.
pub fn display_args(args: &[String]) -> String {
    args.iter()
        .map(|a| quote_display_arg(a))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn exe_label_strips_unix_and_windows_paths() {
        assert_eq!(exe_label("/usr/local/bin/rocm"), "rocm");
        assert_eq!(exe_label("C:\\tools\\rocm.exe"), "rocm.exe");
        assert_eq!(exe_label("rocm"), "rocm");
    }

    #[test]
    fn resolve_exe_is_never_empty() {
        assert!(!resolve_exe().is_empty());
    }

    #[test]
    fn display_args_passes_through_plain_values() {
        let args = vec!["--channel".to_string(), "release".to_string()];
        assert_eq!(display_args(&args), "--channel release");
    }

    #[test]
    fn display_args_quotes_values_with_spaces() {
        let args = vec!["--prefix".to_string(), "/mnt/my folder".to_string()];
        assert_eq!(display_args(&args), "--prefix '/mnt/my folder'");
    }

    #[test]
    fn display_args_escapes_embedded_quotes() {
        let args = vec!["say \"hi\"".to_string(), "it's".to_string()];
        assert_eq!(display_args(&args), r#"'say "hi"' 'it'\''s'"#);
    }

    #[test]
    fn display_args_quotes_embedded_quote_without_whitespace() {
        let args = vec!["say\"hi".to_string(), "there buddy".to_string()];
        assert_eq!(
            display_args(&args),
            r#"'say"hi' 'there buddy'"#,
            "a bare embedded quote must be quoted even with no whitespace in the value"
        );
    }

    #[test]
    fn display_args_quotes_empty_value() {
        let args = vec!["--tag".to_string(), String::new()];
        assert_eq!(display_args(&args), "--tag ''");
    }

    #[test]
    fn display_args_quotes_shell_metacharacters() {
        let args = vec!["a&b".to_string(), "c;d".to_string(), "e|f".to_string()];
        assert_eq!(display_args(&args), "'a&b' 'c;d' 'e|f'");
    }

    /// Expansion characters are quoted too, not only word separators: inside
    /// a double-quoted form `$HOME` and `` `id` `` would still expand, so the
    /// preview would not read as the value the job receives.
    #[test]
    fn display_args_keeps_expansion_characters_literal() {
        let args = vec!["$HOME".to_string(), "`id`".to_string(), "dir\\".to_string()];
        assert_eq!(display_args(&args), r"'$HOME' '`id`' 'dir\'");
    }

    /// Deliberately tiny and mostly metacharacters.
    ///
    /// The shapes that break a quoter — a lone quote, a bare backslash, an
    /// empty argument — are the ones nobody writes an example for, and uniform
    /// random text produces them too rarely to find anything. Reach is measured
    /// by `the_generator_reaches_the_shapes_that_break_quoting`.
    const HOSTILE_ALPHABET: &[char] = &[
        ' ', '\t', '"', '\'', '\\', '$', '`', ';', '|', '&', '*', '?', '(', ')', '<', '>', '#',
        '=', '-', '~', '!', 'a', '1', '/', 'é',
    ];

    fn hostile_arg() -> impl Strategy<Value = String> {
        prop_oneof![
            8 => proptest::collection::vec(proptest::sample::select(HOSTILE_ALPHABET), 0..=5)
                .prop_map(|chars| chars.into_iter().collect::<String>()),
            1 => proptest::sample::select(&[
                "install", "sdk", "--prefix", "/mnt/my folder",
                "C:\\Program Files\\ROCm", "qwen2.5-7b-instruct", "",
            ][..]).prop_map(str::to_owned),
        ]
    }

    fn hostile_argv() -> impl Strategy<Value = Vec<String>> {
        proptest::collection::vec(hostile_arg(), 1..=4)
    }

    proptest! {
        /// The property an approval gate rests on: the line the operator reads
        /// denotes the argv the job is spawned with. Checked by splitting the
        /// rendered line the way a shell would and demanding that argv back —
        /// `shlex` is a POSIX word-splitter, which is exactly the reader this
        /// line is written for.
        #[test]
        fn a_preview_splits_back_to_the_argv_the_job_will_run(argv in hostile_argv()) {
            let preview = format!("{} {}", exe_label("/usr/local/bin/rocm"), display_args(&argv));
            let expected: Vec<String> = std::iter::once("rocm".to_owned())
                .chain(argv.into_iter())
                .collect();
            let split = shlex::split(&preview);
            prop_assert_eq!(split, Some(expected), "preview = {:?}", preview);
        }

        /// No argument can add a word to the preview, so a `;`, `|` or `>`
        /// inside a value reads as data rather than as a second command.
        #[test]
        fn an_argument_can_never_add_a_word(argv in hostile_argv()) {
            let preview = display_args(&argv);
            let split = shlex::split(&preview).expect("preview must parse");
            prop_assert_eq!(split.len(), argv.len(), "preview = {:?}", preview);
        }

        /// Quoting an already-quoted token does not corrupt it.
        #[test]
        fn quoting_an_already_quoted_value_still_round_trips(arg in hostile_arg()) {
            let once = quote_display_arg(&arg);
            let twice = quote_display_arg(&once);
            prop_assert_eq!(shlex::split(&twice), Some(vec![once]));
        }
    }

    /// Hand `line` to a real `sh` as the operands of `printf` and return the
    /// words it received.
    ///
    /// `shlex` only splits words and removes quotes; it performs no expansion,
    /// so it hands back `$HOME`, `~`, `?` and `` `id` `` unchanged and calls a
    /// line that leaves them bare a clean round-trip. Only a shell can say
    /// whether a value would be expanded on paste. The shell runs in a
    /// directory holding files `a` and `1`, so an unquoted `?` or `*` really
    /// globs, and with `HOME` pointed at a sentinel, so an unquoted `~` really
    /// changes.
    #[cfg(unix)]
    fn words_a_real_shell_reads(line: &str) -> Vec<String> {
        static GLOB_BAIT: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        let dir = GLOB_BAIT.get_or_init(|| {
            let dir = std::env::temp_dir()
                .join(format!("rocm-dash-quote-glob-bait-{}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("create glob-bait dir");
            for name in ["a", "1"] {
                std::fs::write(dir.join(name), b"").expect("create glob-bait file");
            }
            dir
        });
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("printf '%s\\0' {line}"))
            .current_dir(dir)
            .env("HOME", "/nonexistent/rocm-dash-quote-home")
            .output()
            .expect("sh should run");
        let mut words: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .split('\0')
            .map(str::to_owned)
            .collect();
        // `printf` terminates every word, so the final piece is always empty.
        words.pop();
        words
    }

    #[cfg(unix)]
    proptest! {
        /// The preview round-trip with a real shell as the reader rather than a
        /// word-splitter: no argument is split, expanded, globbed or executed.
        /// This is the check that fails if a character with meaning to a shell
        /// (`$`, `~`, `?`, `*`, a backtick) is ever treated as inert.
        #[test]
        fn a_real_shell_reads_the_preview_back_as_the_argv(argv in hostile_argv()) {
            let preview = format!("{} {}", exe_label("/usr/local/bin/rocm"), display_args(&argv));
            let expected: Vec<String> = std::iter::once("rocm".to_owned())
                .chain(argv.into_iter())
                .collect();
            prop_assert_eq!(
                words_a_real_shell_reads(&preview),
                expected,
                "preview = {:?}",
                preview
            );
        }
    }

    #[test]
    fn the_generator_reaches_the_shapes_that_break_quoting() {
        // A generator that never produces the dangerous shapes passes against a
        // broken quoter. Measure it, and fail if the alphabet is ever tuned
        // into harmlessness.
        use proptest::strategy::ValueTree;
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
        for (label, count, floor) in [
            ("quote", quote, total / 5),
            ("metacharacter", meta, total / 2),
            ("whitespace", space, total / 5),
            ("empty argument", empty, total / 5),
            ("backslash", backslash, total / 10),
        ] {
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
    fn a_windows_install_folder_keeps_its_separators() {
        let argv = [
            "install".to_owned(),
            "sdk".to_owned(),
            "--prefix".to_owned(),
            "C:\\Program Files\\ROCm".to_owned(),
        ];
        let preview = format!("{} {}", exe_label("C:\\tools\\rocm.exe"), display_args(&argv));
        assert_eq!(
            shlex::split(&preview),
            Some(vec![
                "rocm.exe".to_owned(),
                "install".to_owned(),
                "sdk".to_owned(),
                "--prefix".to_owned(),
                "C:\\Program Files\\ROCm".to_owned(),
            ])
        );
    }
}
