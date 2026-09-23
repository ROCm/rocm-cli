// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static contract test: a test that mutates the process environment must
//! serialize itself.
//!
//! Regression guard for EAI-8397. `std::env::set_var`/`remove_var` change state
//! shared by every thread in the process, so two tests touching the same key
//! race — one reads the other's value and fails an assertion that has nothing
//! to do with what it is testing.
//!
//! This is invisible on most of CI. `ci.yml`'s `Test (affected crates)` job
//! runs `cargo nextest` (a process per test, so the mutation cannot escape),
//! but `windows-build-and-test` runs `cargo test --workspace --all-targets` —
//! all tests as threads in ONE process. So the Windows lane, a required check,
//! was the only place the race could fire, and it read as "your change broke
//! Windows" on branches that never touched the crate. Nothing about the bug is
//! Windows-specific; it is a property of the runner.
//!
//! "the Linux lanes use nextest" would be too strong: `coverage`
//! (`cargo llvm-cov`, no `--nextest`) and `e2e` (`cargo test -p e2e-cucumber
//! --lib`) are threaded single-process harnesses on `ubuntu-latest` too. They
//! happen not to reach the crates that raced, which is luck rather than
//! design — so this guard covers the whole tree rather than one crate.
//!
//! The repo already had two ways of handling this — `ScopedTestEnv` in
//! `apps/rocm/src/main.rs` and `PROCESS_ENV_TEST_LOCK` in
//! `apps/rocm/src/therock.rs`, both of which take a process-wide lock — and the
//! `fix.rs` tests that flaked were simply the ones that skipped the discipline.
//! This guard makes the omission a build failure rather than an intermittent
//! red on a required check.
//!
//! Best is to not touch the environment at all: pass the value in through a
//! test seam, as `discover_rocm_installs_on_host_in` and
//! `newest_rocm_install_dir_in` do. Where a test must exercise the real
//! env-reading path, holding one of the shared locks above is the accepted
//! alternative and satisfies this guard.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    fn repo_root() -> PathBuf {
        // CARGO_MANIFEST_DIR is the xtask/ crate dir; its parent is the repo
        // root (same idiom as workflow_contract::repo_root).
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Every `.rs` file in the repository.
    ///
    /// A whole-tree walk rather than a list of source directories, so a crate
    /// added under a new top-level directory is covered without anyone
    /// remembering to extend this. Two kinds of directory are skipped:
    /// `target/`, which holds generated and vendored code we do not own, and
    /// dot-directories, which include the `.claude/worktrees` checkouts of
    /// other branches — scanning either would make the guard's verdict depend
    /// on state outside the commit under test.
    fn workspace_rust_sources() -> Vec<PathBuf> {
        let mut found = Vec::new();
        collect_rust_files(&repo_root(), &mut found);
        assert!(
            !found.is_empty(),
            "found no Rust sources to scan -- the walk is broken, not the tree"
        );
        found.sort();
        found
    }

    fn collect_rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skip = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_none_or(|name| name == "target" || name.starts_with('.'));
                if skip {
                    continue;
                }
                collect_rust_files(&path, found);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }

    /// The mutating calls that need serializing.
    const MUTATIONS: [&str; 2] = ["env::set_var", "env::remove_var"];

    /// Named helpers that serialize env mutation for their whole scope.
    ///
    /// Anything ending `_TEST_LOCK` counts too, and is matched by suffix rather
    /// than listed — see [`serializes`]. A fixed list of lock NAMES went stale
    /// within days of this guard being written: `main` added
    /// `UPDATE_CHECK_ENV_TEST_LOCK` and the guard then flagged correctly
    /// disciplined code and told its author to rename the lock.
    const NAMED_SERIALIZERS: [&str; 2] = ["ScopedTestEnv", "ScopedEnvVar"];

    /// Whether `text` takes one of the process-wide serializers.
    ///
    /// The suffix rule recognises the DISCIPLINE rather than a specific lock, so
    /// a new `*_TEST_LOCK` is covered the day it is declared.
    fn serializes(text: &str) -> bool {
        NAMED_SERIALIZERS.iter().any(|name| text.contains(name))
            || text
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .any(|word| word.ends_with("_TEST_LOCK"))
    }

    /// Replace the CONTENTS of string literals, char literals and comments with
    /// spaces, preserving line structure and everything outside them.
    ///
    /// The scan below counts braces and looks for call text, and both are
    /// wrong if a literal can contribute either. Two real files in this repo
    /// break the naive version:
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
    fn strip_literals_and_comments(text: &str) -> String {
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
                        out.push_str("  ");
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
            None => false,
        }
    }

    /// A `#[test]` function that mutates the environment.
    #[derive(Debug, PartialEq, Eq)]
    struct Offense {
        line: usize,
        call: String,
    }

    /// Mutations inside a `#[test]` function that does not serialize itself.
    ///
    /// Scoped to the test FUNCTION, not the file. The file-level rule this
    /// replaces asked only whether a marker appeared anywhere in the text, so a
    /// comment reading "maybe migrate this to ScopedTestEnv one day" exempted
    /// every test in the file — and `apps/rocm/src/main.rs` (379 tests) and
    /// `apps/rocm/src/therock.rs` (73) were both wholly exempt for that reason,
    /// which is precisely where the next unguarded test is most likely to land.
    ///
    /// Known limit: a `#[test]` that delegates its mutation to an unguarded
    /// helper is not caught, because the helper's body is a different scope.
    /// Mutations outside `#[test]` functions are deliberately not flagged —
    /// that is production code, and test-support types such as `ScopedTestEnv`
    /// whose whole job is to perform the mutation on a test's behalf.
    fn env_mutations_in_unserialized_tests(text: &str) -> Vec<Offense> {
        let stripped = strip_literals_and_comments(text);
        let mut offenses = Vec::new();
        let mut depth: usize = 0;
        let mut pending_test_attr = false;
        // (depth at which the test fn body opened, body text so far, hits so far)
        let mut current: Option<(usize, String, Vec<Offense>)> = None;

        for (index, line) in stripped.lines().enumerate() {
            let code = line.trim();

            if current.is_none() && code.contains("#[test]") {
                pending_test_attr = true;
            }

            if let Some((_, body, hits)) = current.as_mut() {
                body.push_str(line);
                body.push('\n');
                if let Some(found) = MUTATIONS.iter().find(|needle| code.contains(*needle)) {
                    hits.push(Offense {
                        line: index + 1,
                        call: (*found).to_owned(),
                    });
                }
            }

            let opens = code.matches('{').count();
            let closes = code.matches('}').count();

            if pending_test_attr && opens > 0 {
                pending_test_attr = false;
                current = Some((depth, String::new(), Vec::new()));
            } else if pending_test_attr && code.ends_with(';') {
                // The attribute gated a braceless item; disarm so the next brace
                // anywhere in the file is not mistaken for a test body.
                pending_test_attr = false;
            }

            // Saturating purely so a file this scan misreads cannot panic the
            // build. With literals tokenized away the count is balanced on any
            // file that compiles, so the saturation is unreachable in practice
            // — it is a backstop, not part of the logic.
            depth = (depth + opens).saturating_sub(closes);

            if let Some((open_depth, body, hits)) = current.as_ref()
                && depth <= *open_depth
            {
                if !serializes(body) {
                    offenses.extend(hits.iter().map(|hit| Offense {
                        line: hit.line,
                        call: hit.call.clone(),
                    }));
                }
                current = None;
            }
        }

        // A test body still open when the file ends means the brace bookkeeping
        // lost track of something. Report what it collected instead of dropping
        // it on the floor: over-reporting is visible and gets fixed, whereas a
        // silently discarded hit is the failure mode this guard exists to
        // prevent.
        if let Some((_, body, hits)) = current
            && !serializes(&body)
        {
            offenses.extend(hits);
        }

        offenses
    }

    #[test]
    fn a_test_mutating_the_environment_serializes_itself() {
        let root = repo_root();
        let mut offenders: Vec<String> = Vec::new();

        for path in workspace_rust_sources() {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            let relative = path.strip_prefix(&root).unwrap_or(&path).display();
            for offense in env_mutations_in_unserialized_tests(&text) {
                offenders.push(format!("{relative}:{}: {}", offense.line, offense.call));
            }
        }

        assert!(
            offenders.is_empty(),
            "a test that mutates the process environment must serialize itself \
             -- env is shared by every thread, so an unguarded mutation races \
             any test reading the same key and fails only under a threaded \
             runner (the Windows lane, a required check). Prefer passing the \
             value in through a seam (see `newest_rocm_install_dir_in`); \
             otherwise take `ScopedTestEnv` or any `*_TEST_LOCK` in the test \
             body. Offenders:\n{}",
            offenders.join("\n")
        );
    }

    /// A mutating call, assembled at runtime.
    ///
    /// Spelling `env::set_var` literally in a fixture would make this file's own
    /// test code match the scan it defines, so the fixtures build the needle
    /// instead of containing it. The `MUTATIONS` const above still spells them
    /// literally; that is fine now the exemption is per-test-function rather
    /// than per-file, because the const is not inside a `#[test]` body.
    fn mutation_call(kind: &str) -> String {
        format!("unsafe {{ std::env::{kind}(\"KEY\", \"value\") }}")
    }

    fn unguarded_test(body: &str) -> String {
        format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n        {body}\n    }}\n}}\n"
        )
    }

    #[test]
    fn the_scanner_sees_a_mutation_inside_a_test() {
        let call = mutation_call("set_var");
        let source = format!(
            "fn production() {{\n    {call}\n}}\n\
             #[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n        {call}\n    }}\n}}\n"
        );
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "expected exactly the in-test hit: {hits:?}");
        assert_eq!(hits[0].line, 8, "should flag the line inside the test");
    }

    /// Every entry in [`MUTATIONS`] is actually detected.
    ///
    /// The kinds are spelled out here rather than read from the const on
    /// purpose: a fixture that iterated `MUTATIONS` would shrink along with it,
    /// so deleting an entry would leave the suite green — which is exactly the
    /// hole this closes. `remove_var` previously appeared in one fixture that
    /// asserts `is_empty()`, so nothing anywhere proved an unguarded
    /// `remove_var` inside a `#[test]` was flagged at all.
    ///
    /// The length assertion is what stops the next entry arriving untested: a
    /// third call added to the const fails here until it gains a case.
    #[test]
    fn every_mutating_call_is_flagged_inside_an_unguarded_test() {
        let kinds = ["set_var", "remove_var"];
        assert_eq!(
            MUTATIONS.len(),
            kinds.len(),
            "a call was added to MUTATIONS without a case here"
        );
        for kind in kinds {
            let source = unguarded_test(&mutation_call(kind));
            assert_eq!(
                env_mutations_in_unserialized_tests(&source).len(),
                1,
                "an unguarded std::env::{kind} inside a #[test] must be flagged"
            );
        }
    }

    #[test]
    fn the_scanner_ignores_production_mutations() {
        // Production code owns the process and may legitimately set a variable;
        // the hazard is specific to tests sharing one process under `cargo test`.
        let source = format!("fn production() {{\n    {}\n}}\n", mutation_call("set_var"));
        assert!(
            env_mutations_in_unserialized_tests(&source).is_empty(),
            "a mutation outside test code is not an offense"
        );
    }

    #[test]
    fn the_scanner_stops_flagging_after_the_test_closes() {
        let source = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n        let _ = 1;\n    }}\n}}\n\
             fn later_production() {{\n    {}\n}}\n",
            mutation_call("remove_var")
        );
        assert!(
            env_mutations_in_unserialized_tests(&source).is_empty(),
            "the test ended; the later mutation is production code"
        );
    }

    #[test]
    fn the_scanner_ignores_a_cfg_test_attribute_on_a_braceless_item() {
        // `#[cfg(test)] use ...;` gates an import, not a block. Treating it as
        // the opening of a test block would make every brace after it -- the
        // whole rest of the file -- read as test code.
        let source = format!(
            "#[cfg(test)]\nuse foo::Bar;\n\nfn production() {{\n    {}\n}}\n",
            mutation_call("set_var")
        );
        assert!(
            env_mutations_in_unserialized_tests(&source).is_empty(),
            "the attribute gated an import; nothing below it is test code"
        );
    }

    #[test]
    fn a_serialized_test_is_exempt_but_an_unguarded_one_is_not() {
        let unguarded = unguarded_test(&mutation_call("set_var"));
        assert_eq!(
            env_mutations_in_unserialized_tests(&unguarded).len(),
            1,
            "nothing in this test takes a process-wide lock"
        );

        for marker in [
            "let _env = ScopedTestEnv::new();",
            "let _env = ScopedEnvVar::new(\"K\");",
            "let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();",
            // The suffix rule: a lock this guard has never heard of.
            "let _guard = SOME_BRAND_NEW_TEST_LOCK.lock().unwrap();",
        ] {
            let guarded =
                unguarded_test(&format!("{marker}\n        {}", mutation_call("set_var")));
            assert!(
                env_mutations_in_unserialized_tests(&guarded).is_empty(),
                "taking {marker} should satisfy the guard"
            );
        }
    }

    /// The exemption must not be file-wide.
    ///
    /// Under the previous rule this whole file was exempt because SOME test in
    /// it took a lock -- or merely because a comment named one. That is how
    /// `apps/rocm/src/main.rs` (379 tests) and `apps/rocm/src/therock.rs` (73)
    /// ended up with no line-level enforcement at all.
    #[test]
    fn one_serialized_test_does_not_exempt_its_neighbour() {
        let call = mutation_call("set_var");
        let source = format!(
            "#[cfg(test)]\nmod tests {{\n\
             \x20   #[test]\n    fn guarded() {{\n        let _env = ScopedTestEnv::new();\n        {call}\n    }}\n\
             \x20   #[test]\n    fn unguarded() {{\n        {call}\n    }}\n}}\n"
        );
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(
            hits.len(),
            1,
            "only the unguarded test is an offense: {hits:?}"
        );
        assert_eq!(hits[0].line, 10);
    }

    #[test]
    fn a_comment_naming_a_serializer_does_not_exempt_anything() {
        let source = unguarded_test(&format!(
            "// TODO: maybe migrate this to ScopedTestEnv one day\n        {}",
            mutation_call("set_var")
        ));
        assert_eq!(
            env_mutations_in_unserialized_tests(&source).len(),
            1,
            "a comment is not a lock"
        );
    }

    /// A `}}` inside a string literal used to close the enclosing test early,
    /// making every later mutation invisible. Real instance:
    /// `tests/e2e-cucumber/src/mock_server.rs` builds Prometheus format strings
    /// containing `{{...}}`.
    #[test]
    fn a_brace_inside_a_string_literal_does_not_close_the_test() {
        let source = unguarded_test(&format!(
            "let _s = \"}}}}\";\n        {}",
            mutation_call("set_var")
        ));
        assert_eq!(
            env_mutations_in_unserialized_tests(&source).len(),
            1,
            "a brace inside a literal must not end the test block"
        );
    }

    /// A `//` inside a string literal used to truncate the line, discarding any
    /// brace after it. Real instance: `engines/lemonade/src/lib.rs` passes
    /// `"test://archive"` on a line that also opens a closure.
    #[test]
    fn a_double_slash_inside_a_string_literal_does_not_truncate_the_line() {
        let call = mutation_call("set_var");
        let source = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n\
             \x20       helper(\"test://archive\", |_| {{\n            let _ = 1;\n        }});\n\
             \x20       {call}\n    }}\n}}\n"
        );
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(
            hits.len(),
            1,
            "the closure's brace was inside a literal-bearing line: {hits:?}"
        );
    }

    /// The mirror failure: unbalanced `{{` in a literal inflates the depth and
    /// keeps the scan inside a test long after it ended, so PRODUCTION code
    /// downstream gets flagged and the build fails for nothing.
    ///
    /// Two braces rather than one on purpose. One is swallowed by the enclosing
    /// `mod tests` close, which would leave this passing for a reason unrelated
    /// to the defect; two keep the scan open past the end of the file, where the
    /// EOF flush turns the leak into a reported offense — so this fails against
    /// a scanner that does not tokenize literals. `{{model=...}}` in a
    /// Prometheus format string is the real shape of it.
    #[test]
    fn an_opening_brace_in_a_literal_does_not_drag_in_production_code() {
        let source = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n        let _s = \"{{{{\";\n    }}\n}}\n\
             fn later_production() {{\n    {}\n}}\n",
            mutation_call("set_var")
        );
        assert!(
            env_mutations_in_unserialized_tests(&source).is_empty(),
            "an opening brace inside a literal must not extend the test block"
        );
    }

    #[test]
    fn literals_and_comments_are_stripped_but_line_numbers_survive() {
        let source =
            "let a = \"}}}}\"; // }}\nlet b = r#\"raw } {\"#;\n/* block } */\nlet c = 1;\n";
        let stripped = strip_literals_and_comments(source);
        assert_eq!(
            stripped.lines().count(),
            source.lines().count(),
            "line structure must survive: {stripped:?}"
        );
        assert!(
            !stripped.contains('}') && !stripped.contains('{'),
            "every brace here is inside a literal or comment: {stripped:?}"
        );
        assert!(
            stripped.contains("let c = 1;"),
            "code must survive: {stripped:?}"
        );
    }

    #[test]
    fn a_lifetime_is_not_mistaken_for_a_char_literal() {
        // `'a` opens no literal; treating it as one would swallow the rest of
        // the line, including any brace or call on it.
        let stripped = strip_literals_and_comments("fn f<'a>(x: &'a str) -> &'a str { x }\n");
        assert!(
            stripped.contains('{') && stripped.contains('}'),
            "{stripped:?}"
        );
    }
}
