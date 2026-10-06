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
    ///
    /// Matched unqualified, because the qualifier is a matter of how the file
    /// happens to import: `std::env::set_var(..)`, `env::set_var(..)`, a bare
    /// `set_var(..)` after `use std::env::set_var`, and `e::set_var(..)` after
    /// `use std::env as e` are all the same call and all hazardous. Keying on
    /// `env::` caught only the first two.
    ///
    /// The trailing `(` is what keeps that widening from over-reporting: it
    /// pins the match to a call rather than a prefix, so `env::set_variable(..)`
    /// — an unrelated function whose name merely starts the same way — is not
    /// an offense. Both directions have fixtures. That holds for the entries
    /// that END in `(`; the one bare type name below is matched by a different
    /// rule, described with it and implemented in [`mutation_column`].
    /// `RestoredEnvVar::set(` is here because the guard matches TEXT, and a
    /// mutation spelled as a method on a restoring wrapper contains neither
    /// direct call. The two tests that use it are the sanctioned exception this
    /// change documents, and before this entry existed they sat outside the
    /// guard entirely — deleting their lock acquisition left the scan green,
    /// which is the opposite of what their comments told the next contributor.
    /// Listing the helper keeps the escape hatch narrow: it is still the lock
    /// that is required, the wrapper just stops hiding the requirement.
    ///
    /// `UnsetKeyOnExit` (`crates/rocm-core/src/test_env.rs`) is here for the
    /// same reason, from the other side: it is a unit struct whose `Drop`
    /// removes a variable, so the mutation happens in an impl the scan never
    /// enters and the test body only ever NAMES the type. It carries no `(`
    /// because constructing a unit struct has none, so a plain substring match
    /// would also flag `UnsetKeyOnExitHelper`, `MyUnsetKeyOnExit` and
    /// `Option<UnsetKeyOnExit>`. A bare name therefore counts as any whole
    /// identifier EXCEPT in a type position: right after `<`, right after a
    /// type-ascription `:` (not `::`), or right before `>`, blanks between
    /// ignored. That flags every way a value of it is built — `let _g =
    /// UnsetKeyOnExit;`, `drop(UnsetKeyOnExit)`, `Some(UnsetKeyOnExit)`,
    /// `vec![UnsetKeyOnExit]`, a tuple of them, `g = Some(UnsetKeyOnExit);` —
    /// and rejects the longer and prefixed names, `Option<..>`/`Vec<..>`
    /// arguments and `_x: UnsetKeyOnExit` signatures, none of which holds
    /// anything. It is a deny-list of type positions because a miss here is an
    /// unguarded mutation, whereas an over-report (say, a `-> UnsetKeyOnExit`
    /// return type inside a test body) is visible and rewordable. The
    /// definition and `impl Drop for .. {` also match, but sit outside any test
    /// body, where the scan does not look. Both directions are pinned by
    /// [`a_bare_type_entry_matches_a_whole_name_outside_type_positions`] and
    /// [`holding_a_mutating_guard_still_needs_the_lock`].
    const MUTATIONS: [&str; 4] = [
        "set_var(",
        "remove_var(",
        "RestoredEnvVar::set(",
        "UnsetKeyOnExit",
    ];

    /// Named helpers that serialize env mutation for their whole scope.
    ///
    /// Membership is "holds a lock for its whole scope", not "wraps an env
    /// mutation". `ScopedEnvVar` (`apps/rocm/src/main.rs`) was listed here and
    /// only ever saved and restored — its one caller,
    /// `with_scoped_builtin_engine_env`, takes `BUILTIN_ENGINE_ENV_LOCK`
    /// itself. The type is visible to the largest test module in the tree, in
    /// the same file, so any test that merely named it was granted the full
    /// exemption while holding nothing: the exact failure this guard exists to
    /// prevent, written into the guard.
    ///
    /// Anything ending `_TEST_LOCK` counts too, and is matched by suffix rather
    /// than listed — see [`lock_name_column`]. A fixed list of lock NAMES went stale
    /// within days of this guard being written: `main` added
    /// `UPDATE_CHECK_ENV_TEST_LOCK` and the guard then flagged correctly
    /// disciplined code and told its author to rename the lock.
    const NAMED_SERIALIZERS: [&str; 1] = ["ScopedTestEnv"];

    /// Where `line` takes a serializer, as a column, or `None` if it does not.
    ///
    /// Applied per LINE by the caller, which is what lets it answer "was the
    /// lock taken before this mutation?" rather than merely "somewhere in this
    /// body". Every lock in this tree names itself on the line that acquires it
    /// (`let _guard = SOME_TEST_LOCK` ...), so a per-line match loses nothing.
    ///
    /// A column rather than a bool because that per-line answer is not enough
    /// on its own: the caller also has to order the acquisition against a
    /// mutation on the SAME line. Comparing by line number alone read
    /// `set_var(..); let _g = LOCK.lock();` as serialized, which is the
    /// retroactive-lock bug the per-line rule was introduced to stop, written
    /// on one line instead of two.
    ///
    /// A `*_TEST_LOCK` has to be ACQUIRED, not merely named: `let _x = &LOCK;`
    /// mentions one and holds nothing. `rest` carries the remainder of the
    /// statement for that check, because rustfmt splits the real shape —
    ///
    /// ```text
    /// let _guard = PROCESS_ENV_TEST_LOCK
    ///     .lock()
    ///     .unwrap_or_else(std::sync::PoisonError::into_inner);
    /// ```
    ///
    /// — across three lines, so demanding `.lock(` on the naming line would
    /// reject every correctly disciplined test in `crates/rocm-core`.
    ///
    /// `NAMED_SERIALIZERS` are types, not mutexes, so they carry no `.lock(`
    /// and are matched on the name alone.
    ///
    /// The `.min()` over that list is equivalent to taking its first element
    /// while the list holds one entry, so no fixture can distinguish them —
    /// stated rather than tested. It earns its keep the moment a second name is
    /// added, which is why it is written positionally now; the equivalent
    /// choice over [`MUTATIONS`], which does hold several, is pinned by
    /// [`the_reported_mutation_is_the_leftmost_one_on_the_line`].
    fn serializer_column(line: &str, rest: &str) -> Option<usize> {
        if let Some(at) = NAMED_SERIALIZERS.iter().filter_map(|n| line.find(n)).min() {
            return Some(at);
        }
        let at = lock_name_column(line)?;
        (line[at..].contains(".lock(") || rest.contains(".lock(")).then_some(at)
    }

    /// The column a `*_TEST_LOCK` static is named at.
    ///
    /// The suffix rule recognises the DISCIPLINE rather than a specific lock,
    /// so a new `*_TEST_LOCK` is covered the day it is declared. It is a suffix
    /// and not a substring on purpose: `NOT_A_TEST_LOCK_HELPER` names something
    /// else.
    fn lock_name_column(line: &str) -> Option<usize> {
        const SUFFIX: &str = "_TEST_LOCK";
        line.match_indices(SUFFIX).find_map(|(at, _)| {
            let ends_word = line[at + SUFFIX.len()..]
                .chars()
                .next()
                .is_none_or(|c| !is_identifier_char(c));
            let start = line[..at]
                .rfind(|c: char| !is_identifier_char(c))
                .map_or(0, |i| i + 1);
            ends_word.then_some(start)
        })
    }

    fn is_identifier_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    /// The leftmost [`MUTATIONS`] entry on `line`, with its column.
    ///
    /// An entry ending in `(` is a call and is matched as text. A bare type
    /// name is matched by [`names_a_value`], for the reasons given on
    /// [`MUTATIONS`].
    fn mutation_column(line: &str) -> Option<(usize, &'static str)> {
        MUTATIONS
            .iter()
            .filter_map(|needle| {
                let at = if needle.ends_with('(') {
                    line.find(needle)
                } else {
                    line.match_indices(needle)
                        .map(|(at, _)| at)
                        .find(|&at| names_a_value(line, at, needle.len()))
                };
                at.map(|at| (at, *needle))
            })
            .min()
    }

    /// Whether the `len`-byte identifier at `line[at..]` can stand for a value.
    ///
    /// It must be a whole identifier, and it must not sit in one of the three
    /// TYPE positions: right after `<` (`Option<T>`), right after a type
    /// ascription `:` that is not half of a `::` path separator
    /// (`fn f(_x: T)`, `let g: T`), or right before `>` (`Vec<T>`,
    /// `HashMap<K, T>`). Blanks between the name and that neighbour are
    /// skipped, so rustfmt-hostile spellings like `Option< T >` and `_x : T`
    /// are type positions too. Everything else counts — a deny-list of type
    /// positions rather than an allow-list of value positions, because a value
    /// of a unit struct can be built in more places than any allow-list will
    /// name: `drop(T)`, `Some(T)`, `vec![T]`, `(T, T)`, `mem::forget(T)`,
    /// `g = Some(T);`. A missed construction is an unguarded mutation; an
    /// over-report is a type mention the author can see and reword.
    fn names_a_value(line: &str, at: usize, len: usize) -> bool {
        let before = line[..at].trim_end();
        let after = line[at + len..].trim_start();
        let whole = !line[..at].ends_with(is_identifier_char)
            && !line[at + len..].starts_with(is_identifier_char);
        let ascribed = before.ends_with(':') && !before.ends_with("::");
        whole && !before.ends_with('<') && !ascribed && !after.starts_with('>')
    }

    /// The rest of the statement beginning at `lines[index]`.
    ///
    /// Empty when that line already ends the statement. Bounded, because a line
    /// with no `;` after it at all — the last line of a truncated file — would
    /// otherwise drag the whole remainder of the body in; rustfmt's widest
    /// split of a lock acquisition is three lines.
    fn statement_continuation(lines: &[&str], index: usize) -> String {
        if lines.get(index).is_some_and(|line| line.contains(';')) {
            return String::new();
        }
        let mut out = String::new();
        for line in lines.iter().skip(index + 1).take(4) {
            out.push_str(line);
            if line.contains(';') {
                break;
            }
        }
        out
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
        // `b` itself, emitted as code rather than as a space. Four things read
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

    /// Whether `code` carries a test attribute.
    ///
    /// The rule is "the attribute path's last segment is `test`", which covers
    /// `#[test]`, `#[tokio::test]` and `#[tokio::test(flavor = "...")]` alike.
    /// Arming only on the literal `#[test]` left the 87 `#[tokio::test]`
    /// functions in this tree invisible to the scan — none of them mutates the
    /// environment today, but nothing was watching if one started.
    ///
    /// `#[cfg(test)]` and `#[cfg_attr(test, ...)]` stop at the `(`, so their
    /// path is `cfg`/`cfg_attr` and neither is mistaken for a test.
    fn has_test_attribute(code: &str) -> bool {
        code.match_indices("#[").any(|(start, _)| {
            let path: String = code[start + 2..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == ':')
                .collect();
            path.rsplit("::").next() == Some("test")
        })
    }

    /// A test function that mutates the environment.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Offense {
        line: usize,
        /// Column the call starts at, used only to order it against a
        /// serializer taken on the same line. Never reported: the offender is
        /// located by `file:line`, and a column would go stale under rustfmt.
        column: usize,
        call: String,
    }

    /// A test body being scanned.
    struct OpenTest {
        /// The brace depth the body opened at; it closes on the way back down.
        open_depth: usize,
        /// Where the first serializer was taken, as `(line, column)`, if any. A
        /// mutation is exempt only from that POINT onward — see
        /// [`OpenTest::unserialized_hits`]. The column is what keeps a lock
        /// written after a mutation on the same line from covering it.
        serialized_at: Option<(usize, usize)>,
        hits: Vec<Offense>,
    }

    impl OpenTest {
        /// The hits this body does not serialize.
        ///
        /// A lock is an RAII guard: it protects from where it is taken until the
        /// end of the scope, and nothing before. Matching the whole accumulated
        /// body — which is what this used to do — accepted a lock written AFTER
        /// the mutation it was supposed to protect, which is exactly the bug
        /// this guard exists to catch, spelled slightly differently.
        fn unserialized_hits(&self) -> impl Iterator<Item = Offense> + '_ {
            self.hits
                .iter()
                .filter(move |hit| {
                    self.serialized_at
                        .is_none_or(|at| (hit.line, hit.column) < at)
                })
                .cloned()
        }
    }

    /// Mutations inside a test function that does not serialize itself.
    ///
    /// Scoped to the test FUNCTION, not the file. The file-level rule this
    /// replaces asked only whether a marker appeared anywhere in the text, so a
    /// comment reading "maybe migrate this to ScopedTestEnv one day" exempted
    /// every test in the file — and the two largest test modules in the tree,
    /// `apps/rocm/src/main.rs` and `apps/rocm/src/therock.rs`, were both wholly
    /// exempt for that reason, which is precisely where the next unguarded test
    /// is most likely to land. (Deliberately no test counts here: they were
    /// wrong within weeks of being written, and the point does not need them.)
    ///
    /// Known limits:
    ///
    /// * A test that delegates its mutation to an unguarded helper is not
    ///   caught, because the helper's body is a different scope.
    /// * Attributes are recognised by path (see [`has_test_attribute`]), so a
    ///   harness whose attribute does not end in `test` — `#[test_case(..)]`,
    ///   `#[rstest]` — would not arm the scan. Neither is used in this tree.
    /// * For the same reason `#[cfg_attr(unix, test)]` does not arm it either:
    ///   the path stops at the `(`, so it reads as `cfg_attr`. Recognising it
    ///   means parsing the argument list and skipping the FIRST argument, which
    ///   is the condition — `#[cfg_attr(test, derive(Debug))]` also contains a
    ///   bare `test` and must keep NOT arming the scan, or the rest of the file
    ///   reads as one test body. That is a second attribute grammar inside a
    ///   text scanner, for a spelling this tree does not use and whose
    ///   idiomatic form — `#[cfg(unix)]` above `#[test]` — this scan does
    ///   catch. Stated rather than implemented, deliberately.
    /// * The scan checks that A lock is held, not that it is THE lock every
    ///   other mutator of the same key takes. Two tests replacing one key under
    ///   two different mutexes both pass and still race each other. Closing
    ///   that needs the key each call names, and the keys are string literals
    ///   this scan has deliberately stripped before it starts — recovering them
    ///   would mean re-parsing, and they are often constants or variables
    ///   anyway. The rejection message states the requirement instead.
    ///
    /// Mutations outside test functions are deliberately not flagged — that is
    /// production code, and test-support types such as `ScopedTestEnv` whose
    /// whole job is to perform the mutation on a test's behalf.
    fn env_mutations_in_unserialized_tests(text: &str) -> Vec<Offense> {
        scan(text).offenses
    }

    /// What one pass over a file saw.
    ///
    /// `bodies` exists so the whole-tree assertion can tell "no offenders" from
    /// "nothing was scanned" — see
    /// `the_whole_tree_scan_actually_enters_test_bodies`.
    struct Scan {
        offenses: Vec<Offense>,
        bodies: usize,
    }

    fn scan(text: &str) -> Scan {
        let stripped = strip_literals_and_comments(text);
        let lines: Vec<&str> = stripped.lines().collect();
        let mut offenses = Vec::new();
        let mut bodies = 0usize;
        let mut depth: usize = 0;
        let mut pending_test_attr = false;
        let mut current: Option<OpenTest> = None;

        for index in 0..lines.len() {
            let line = lines[index];
            let code = line.trim();

            if current.is_none() && has_test_attribute(code) {
                pending_test_attr = true;
            }

            let opens = code.matches('{').count();
            let closes = code.matches('}').count();

            // Opened BEFORE this line is scanned, so a mutation sharing the
            // line with the body's opening brace is seen. The shape is a
            // one-line `fn t() { unsafe { set_var(..) } }`; rustfmt normally
            // splits it, so only a `#[rustfmt::skip]` test reaches it -- and
            // that one used to slip through in silence.
            if pending_test_attr && opens > 0 {
                pending_test_attr = false;
                bodies += 1;
                current = Some(OpenTest {
                    open_depth: depth,
                    serialized_at: None,
                    hits: Vec::new(),
                });
            } else if pending_test_attr && code.ends_with(';') {
                // The attribute gated a braceless item; disarm so the next brace
                // anywhere in the file is not mistaken for a test body.
                pending_test_attr = false;
            }

            if let Some(open) = current.as_mut() {
                if open.serialized_at.is_none()
                    && let Some(column) =
                        serializer_column(code, &statement_continuation(&lines, index))
                {
                    open.serialized_at = Some((index + 1, column));
                }
                if let Some((column, found)) = mutation_column(code) {
                    open.hits.push(Offense {
                        line: index + 1,
                        column,
                        // Without the `(` the needle carries; the offender is
                        // named by the call, not by the match pattern.
                        call: found.trim_end_matches('(').to_owned(),
                    });
                }
            }

            // Saturating purely so a file this scan misreads cannot panic the
            // build. With literals tokenized away the count is balanced on any
            // file that compiles, so the saturation is unreachable on this
            // tree — it is a backstop, not part of the logic. It is still
            // load-bearing and still observable: a plain `-` panics with
            // `attempt to subtract with overflow` on a leading `}`, where this
            // returns no offenses. Pinned by
            // `an_unbalanced_close_returns_nothing_rather_than_panicking`.
            depth = (depth + opens).saturating_sub(closes);

            if let Some(open) = current.as_ref()
                && depth <= open.open_depth
            {
                offenses.extend(open.unserialized_hits());
                current = None;
            }
        }

        // A test body still open when the file ends means the brace bookkeeping
        // lost track of something. Report what it collected instead of dropping
        // it on the floor: over-reporting is visible and gets fixed, whereas a
        // silently discarded hit is the failure mode this guard exists to
        // prevent.
        if let Some(open) = current.as_ref() {
            offenses.extend(open.unserialized_hits());
        }

        Scan { offenses, bodies }
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
             body. The lock must be taken BEFORE the mutation -- it protects \
             from where it is acquired, not retroactively -- and it must be \
             the SAME lock every test mutating that key takes, which this scan \
             cannot check for you: two tests replacing one key under two \
             different mutexes both satisfy it and still race. Offenders:\n{}",
            offenders.join("\n")
        );
    }

    /// The whole-tree assertion above is `offenders.is_empty()`, which also
    /// holds when the scan entered no test bodies at all.
    ///
    /// `workspace_rust_sources` pins that files were found; nothing pinned that
    /// test code inside them was reached. An arming regression — the attribute
    /// rule narrowing, the brace bookkeeping closing bodies early — would turn
    /// the guard off and read as a green gate, which is the one failure mode
    /// this module is written against.
    ///
    /// The floor is deliberately far below the real count (3218 at the time of
    /// writing) and is not a census: it has to survive tests being added and
    /// removed without anyone remembering this number, while still failing
    /// loudly if arming collapses.
    #[test]
    fn the_whole_tree_scan_actually_enters_test_bodies() {
        const FLOOR: usize = 500;

        let bodies: usize = workspace_rust_sources()
            .iter()
            .map(|path| {
                let text = std::fs::read_to_string(path)
                    .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
                scan(&text).bodies
            })
            .sum();

        assert!(
            bodies >= FLOOR,
            "the scan entered {bodies} test bodies across the tree, under the \
             {FLOOR} floor -- the guard is not looking at test code any more, \
             so its green result means nothing"
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

    /// The same call reached without the `std::env::` qualifier.
    ///
    /// `use std::env::set_var` leaves `qualifier` empty; `use std::env as e`
    /// makes it `e::`. Assembled at runtime for the same reason as
    /// [`mutation_call`].
    fn imported_mutation_call(qualifier: &str, kind: &str) -> String {
        format!("unsafe {{ {qualifier}{kind}(\"KEY\", \"value\") }}")
    }

    /// A mutation spelled as a method on a restoring wrapper.
    ///
    /// Assembled rather than written out for the same reason as
    /// [`mutation_call`]: a fixture containing the literal text would make this
    /// file match its own scan.
    fn delegated_mutation_call(path: &str) -> String {
        format!("let _restore = crate::test_env::{path}(\"KEY\", &value);")
    }

    /// A guard whose `Drop` mutates, held by a test that only names the type.
    ///
    /// Assembled rather than written out for the same reason as
    /// [`mutation_call`].
    fn held_mutating_guard(name: &str) -> String {
        format!("let _held = {name};")
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
        let delegated = ["RestoredEnvVar::set"];
        let held = ["UnsetKeyOnExit"];
        assert_eq!(
            MUTATIONS.len(),
            kinds.len() + delegated.len() + held.len(),
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
        for path in delegated {
            let source = unguarded_test(&delegated_mutation_call(path));
            assert_eq!(
                env_mutations_in_unserialized_tests(&source).len(),
                1,
                "an unguarded {path} inside a #[test] must be flagged"
            );
        }
        for name in held {
            let source = unguarded_test(&held_mutating_guard(name));
            assert_eq!(
                env_mutations_in_unserialized_tests(&source).len(),
                1,
                "a test holding an unguarded {name} must be flagged"
            );
        }
    }

    /// A restoring wrapper does not exempt the test that drives it.
    ///
    /// The wrapper restores; it does not serialize. Taking the key's lock first
    /// is still required, and because the call text carries neither direct
    /// mutation the guard only enforces that while the helper is named in
    /// [`MUTATIONS`]. Both directions are pinned: unguarded is an offense,
    /// guarded is not.
    #[test]
    fn a_mutation_delegated_to_a_restoring_wrapper_still_needs_the_lock() {
        let call = delegated_mutation_call("RestoredEnvVar::set");

        let hits = env_mutations_in_unserialized_tests(&unguarded_test(&call));
        assert_eq!(hits.len(), 1, "the wrapper does not serialize: {hits:?}");
        assert_eq!(
            hits[0].call, "RestoredEnvVar::set",
            "the offender is named by the call"
        );

        let guarded = unguarded_test(&format!(
            "let _guard = SOME_TEST_LOCK.lock().unwrap();\n        {call}"
        ));
        assert!(
            env_mutations_in_unserialized_tests(&guarded).is_empty(),
            "the lock above is what makes this shape acceptable"
        );
    }

    /// A guard whose `Drop` mutates does not exempt the test that holds it.
    ///
    /// The mirror of
    /// [`a_mutation_delegated_to_a_restoring_wrapper_still_needs_the_lock`]
    /// for the bare type entry: the test body only names the type, so the
    /// guard enforces the lock only while that name is in [`MUTATIONS`].
    /// Unguarded is an offense, guarded is not.
    #[test]
    fn holding_a_mutating_guard_still_needs_the_lock() {
        let held = held_mutating_guard("UnsetKeyOnExit");

        let hits = env_mutations_in_unserialized_tests(&unguarded_test(&held));
        assert_eq!(hits.len(), 1, "holding the guard mutates on drop: {hits:?}");
        assert_eq!(
            hits[0].call, "UnsetKeyOnExit",
            "the offender is named by the type"
        );

        let guarded = unguarded_test(&format!(
            "let _guard = SOME_TEST_LOCK.lock().unwrap();\n        {held}"
        ));
        assert!(
            env_mutations_in_unserialized_tests(&guarded).is_empty(),
            "the lock above is what makes this shape acceptable"
        );
    }

    /// A bare type entry is a whole identifier outside a type position, not a
    /// substring and not an allow-list of construction shapes.
    ///
    /// The flagged half are all ways to build a value of the unit struct, and
    /// each one was missed by the earlier rule that required the next
    /// character to be `;`, `(`, `{` or `::`. The reassignment is the shape
    /// that rule hid on BOTH of its lines: the declaration is a type mention,
    /// and the construction is followed by `)`.
    ///
    /// In the other half, each shape defeats one part of the rule on its own:
    /// the longer name fails the boundary after the match, the prefixed one
    /// fails the boundary before it, and the rest are whole identifiers in a
    /// type position — after `<`, before `>`, after an ascription `:` — with
    /// blanks around them for the spellings rustfmt would normalise away. The
    /// first and last generic arguments of a multi-argument type sit next to
    /// only ONE angle bracket, which is what makes each of the `<` and `>`
    /// carve-outs necessary on its own: `Vec<..>` alone is caught by either.
    #[test]
    fn a_bare_type_entry_matches_a_whole_name_outside_type_positions() {
        for line in [
            "drop(UnsetKeyOnExit);",
            "let _s = Some(UnsetKeyOnExit);",
            "let _v = vec![UnsetKeyOnExit];",
            "let _t = (UnsetKeyOnExit, UnsetKeyOnExit);",
            "std::mem::forget(UnsetKeyOnExit);",
            "let _p = test_env::UnsetKeyOnExit;",
            "let mut g: Option<UnsetKeyOnExit> = None;\n        g = Some(UnsetKeyOnExit);",
        ] {
            let hits = env_mutations_in_unserialized_tests(&unguarded_test(line));
            assert_eq!(hits.len(), 1, "`{line}` builds the guard: {hits:?}");
            assert_eq!(hits[0].call, "UnsetKeyOnExit");
        }
        for line in [
            "let _h = UnsetKeyOnExitHelper;",
            "let _m = MyUnsetKeyOnExit;",
            "let _o: Option<UnsetKeyOnExit> = None;",
            "let _o: Option< UnsetKeyOnExit > = None;",
            "let _v: Vec<UnsetKeyOnExit> = Vec::new();",
            "let _r: Result<UnsetKeyOnExit, ()> = Err(());",
            "let _m: BTreeMap<u8, UnsetKeyOnExit> = BTreeMap::new();",
            "fn f(_x: UnsetKeyOnExit) {}",
            "fn f(_x : UnsetKeyOnExit) {}",
        ] {
            let hits = env_mutations_in_unserialized_tests(&unguarded_test(line));
            assert!(hits.is_empty(), "`{line}` holds no such guard: {hits:?}");
        }
    }

    /// The qualifier is a matter of how the file imports, not of what the call
    /// does. Keying the match on `env::` made a test that had written
    /// `use std::env::set_var` invisible to the scan while doing exactly the
    /// thing the guard exists to stop.
    #[test]
    fn a_mutation_reached_through_an_import_is_flagged() {
        for qualifier in ["", "e::"] {
            for kind in ["set_var", "remove_var"] {
                let source = unguarded_test(&imported_mutation_call(qualifier, kind));
                let hits = env_mutations_in_unserialized_tests(&source);
                assert_eq!(
                    hits.len(),
                    1,
                    "an unguarded {qualifier}{kind} inside a #[test] must be flagged: {hits:?}"
                );
                assert_eq!(
                    hits[0].call, kind,
                    "the offender is named by the call, not by the match pattern"
                );
            }
        }
    }

    /// The other side of widening the match: it is a call, not a prefix.
    ///
    /// Before the trailing `(`, any name merely STARTING `env::set_var` was
    /// reported — a build failure naming a function that never touches the
    /// environment, which is the same credibility problem as a miss.
    #[test]
    fn a_call_merely_starting_like_a_mutation_is_not_flagged() {
        let source = unguarded_test(&format!("let _ = env::{}able(\"KEY\");", "set_var"));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert!(
            hits.is_empty(),
            "set_variable is a different function: {hits:?}"
        );
    }

    /// An async test is a test.
    ///
    /// The scan used to arm on the literal `#[test]`, which left every
    /// `#[tokio::test]` in the tree outside the guard. They run in the same
    /// process under the same threaded harness, so the hazard is identical.
    #[test]
    fn the_scanner_sees_a_mutation_inside_an_async_test() {
        for attribute in [
            "#[tokio::test]",
            "#[tokio::test(flavor = \"multi_thread\")]",
        ] {
            let source = format!(
                "#[cfg(test)]\nmod tests {{\n    {attribute}\n    async fn t() {{\n        {}\n    }}\n}}\n",
                mutation_call("set_var")
            );
            let hits = env_mutations_in_unserialized_tests(&source);
            assert_eq!(hits.len(), 1, "{attribute} should arm the scan: {hits:?}");
        }
    }

    /// The line after a test attribute need not open a block.
    ///
    /// This is a text scan, not a parser, so it cannot assume it does. Without
    /// the disarm the attribute stays armed and the next brace anywhere in the
    /// file — here, production code — is taken for the test body.
    #[test]
    fn a_test_attribute_on_a_braceless_item_does_not_open_a_block() {
        let source = format!(
            "#[test]\nfn declared_elsewhere();\n\nfn production() {{\n    {}\n}}\n",
            mutation_call("set_var")
        );
        assert!(
            env_mutations_in_unserialized_tests(&source).is_empty(),
            "no test body was opened; the mutation below is production code"
        );
    }

    /// `#[cfg(test)]` names `test` but is not a test attribute. Arming on it
    /// would make the whole rest of the file read as one test body.
    #[test]
    fn a_cfg_test_gate_is_not_a_test_attribute() {
        for attribute in ["#[cfg(test)]", "#[cfg_attr(test, derive(Debug))]"] {
            assert!(
                !has_test_attribute(attribute),
                "{attribute} must not arm the scan"
            );
        }
        assert!(has_test_attribute("#[test]"));
        assert!(has_test_attribute("#[tokio::test]"));
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
    /// `apps/rocm/src/main.rs` and `apps/rocm/src/therock.rs`, the two largest
    /// test modules in the tree, ended up with no line-level enforcement at all.
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

    /// A raw string's closer is the quote AND its hashes, and all of it has to
    /// be consumed.
    ///
    /// Stopping at the quote returns to the code stream sitting on the hashes,
    /// which then leak out as code. A `#` emitted where the source had a
    /// literal is how text turns into an attribute: leaked immediately before a
    /// `[`, it spells the `#[` that [`has_test_attribute`] arms on.
    #[test]
    fn a_raw_string_close_consumes_its_hashes() {
        let source = "let a = r#\"x\"#;\nlet b = r##\"y\"##;\n";
        let stripped = strip_literals_and_comments(source);
        assert!(
            !stripped.contains('#'),
            "the closing hashes belong to the literal: {stripped:?}"
        );
    }

    /// A `\` at end of line continues a string literal onto the next one. The
    /// newline is not part of the string's value, but it is part of the file's
    /// line structure, and losing it renumbers everything below.
    #[test]
    fn a_line_continuation_in_a_literal_does_not_renumber_the_lines() {
        let source = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n\
             \x20       let _s = \"continued \\\n             here\";\n\
             \x20       {}\n    }}\n}}\n",
            mutation_call("set_var")
        );
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(
            hits[0].line, 7,
            "the continuation spans lines 5-6, so the mutation is on line 7"
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

    /// A raw string ends at its hashes, not at the first `"` inside it.
    ///
    /// The suite's only raw-string fixture used to be `r#"raw } {"#`, which
    /// holds no quote — so the ordinary-string state consumed exactly the same
    /// span and the raw-string branch could be deleted with all 18 tests still
    /// green. A fixture that looks like coverage and provides none is the worst
    /// case for a permanent gate, because the build stays green either way.
    ///
    /// Each literal here carries a quote, which is what makes the two states
    /// diverge: read as an ordinary string, the text after that quote becomes
    /// code and its `{` inflates the depth, which leaves the mutation below
    /// inside a literal and invisible.
    #[test]
    fn a_raw_string_holding_a_quote_does_not_leak_its_braces() {
        // A hashless `r"..."` cannot hold a quote at all -- that is what the
        // hashes are for -- so it gets its own fixture below.
        for literal in ["r#\"a \" b {\"#", "r##\"a \"# b {\"##", "br#\"a \" b {\"#"] {
            let source = unguarded_test(&format!(
                "let _s = {literal};\n        {}",
                mutation_call("set_var")
            ));
            let hits = env_mutations_in_unserialized_tests(&source);
            assert_eq!(
                hits.len(),
                1,
                "{literal} must be stripped whole, braces included: {hits:?}"
            );
        }
    }

    /// A raw string with no hashes is closed by its very next `"`, so the
    /// opener has to consume that quote. Consuming one character less leaves
    /// the scan sitting on the quote it just opened, which closes the literal
    /// immediately and spills its contents into the code stream.
    #[test]
    fn a_hashless_raw_string_is_not_closed_by_its_own_opening_quote() {
        let source = unguarded_test(&format!(
            "let _s = r\"leaks }} here\";\n        {}",
            mutation_call("set_var")
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "r\"...\" must be stripped whole: {hits:?}");
    }

    /// A char literal is a literal. Without the state the `}` in `'}'` closes
    /// the enclosing test early and everything after it goes unwatched; with
    /// the state but no exit the rest of the file is swallowed instead. Both
    /// directions end with the mutation below unreported.
    #[test]
    fn a_char_literal_holding_a_brace_does_not_close_the_test() {
        let source = unguarded_test(&format!(
            "let _c = '}}';\n        {}",
            mutation_call("set_var")
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "'}}' is a literal, not a brace: {hits:?}");
    }

    /// `'\"'` is a char literal holding a quote. Failing to recognise it lets
    /// that quote open an ordinary string, which then runs on and swallows the
    /// mutation on the next line.
    #[test]
    fn an_escaped_quote_in_a_char_literal_does_not_open_a_string() {
        let source = unguarded_test(&format!(
            "let _c = '\\\"';\n        {}",
            mutation_call("set_var")
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "'\\\"' holds the quote: {hits:?}");
    }

    /// `'\''` is the one char literal whose escape matters: the escaped `'` is
    /// not the closing `'`. Consuming it as the close leaves a stray quote in
    /// the code stream, and here that quote is followed by `,'` — which reads
    /// as another char literal opening, so the scan re-enters the literal state
    /// one quote out of phase and the `}` after it escapes as code, closing the
    /// test before the mutation.
    ///
    /// The adjacency is what makes this discriminating, and rustfmt would put a
    /// space there. The scan runs over every `.rs` file including
    /// `#[rustfmt::skip]` ones, so the unformatted shape is in scope.
    #[test]
    fn an_escaped_quote_does_not_end_a_char_literal_early() {
        let source = unguarded_test(&format!(
            "let _pair = ('\\'','}}');\n        {}",
            mutation_call("set_var")
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "the escaped quote is content: {hits:?}");
    }

    /// Rust block comments nest, so the first `*/` does not necessarily end
    /// one. Treating it as the end hands the rest of the outer comment to the
    /// code stream — here a `}` that closes the test early.
    ///
    /// The comment spans two lines so the same fixture also pins the newline:
    /// collapsing it merges the two source lines and renumbers the mutation,
    /// which is the `file:line` the assertion message promises.
    #[test]
    fn a_nested_block_comment_runs_to_its_outer_close() {
        let source = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n\
             \x20       /* outer /* inner */\n\
             \x20          }} still outer */\n\
             \x20       {}\n    }}\n}}\n",
            mutation_call("set_var")
        );
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "the whole comment is a comment: {hits:?}");
        assert_eq!(
            hits[0].line, 7,
            "the comment spans lines 5-6, so the mutation is on line 7"
        );
    }

    /// `\"` inside a string is content, not the close. Ending the string there
    /// spills the rest of the literal into the code stream — here a `}` that
    /// closes the test before the mutation is reached.
    #[test]
    fn an_escaped_quote_in_a_string_does_not_close_it() {
        let source = unguarded_test(&format!(
            "let _s = \"esc \\\" }}\";\n        {}",
            mutation_call("set_var")
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "the escaped quote is content: {hits:?}");
    }

    /// A literal may span lines without a continuation backslash. Every
    /// literal state has to put the newline back for the same reason the
    /// continuation case does: the reported line number is the whole point.
    #[test]
    fn a_literal_spanning_two_lines_does_not_renumber_them() {
        for (opener, closer) in [("\"", "\""), ("r#\"", "\"#")] {
            let source = format!(
                "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n\
                 \x20       let _s = {opener}multi\nline{closer};\n\
                 \x20       {}\n    }}\n}}\n",
                mutation_call("set_var")
            );
            let hits = env_mutations_in_unserialized_tests(&source);
            assert_eq!(hits.len(), 1, "{opener}: {hits:?}");
            assert_eq!(
                hits[0].line, 7,
                "{opener} spans lines 5-6, so the mutation is on line 7"
            );
        }
    }

    /// A file that ends with a test body still open had its brace bookkeeping
    /// defeated by something. Reporting what was collected is the safe
    /// direction — over-reporting is visible and gets fixed, a dropped hit is
    /// the silent failure this guard exists to prevent — but the exemption
    /// still applies, or a correctly locked test would fail the build.
    #[test]
    fn an_unclosed_test_body_still_reports_what_it_collected() {
        let call = mutation_call("set_var");
        let unclosed =
            format!("#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n        {call}\n");
        assert_eq!(
            env_mutations_in_unserialized_tests(&unclosed).len(),
            1,
            "a hit collected before the file ran out must not be dropped"
        );

        let unclosed_but_locked = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n\
             \x20       let _guard = SOME_TEST_LOCK.lock().unwrap();\n        {call}\n"
        );
        assert!(
            env_mutations_in_unserialized_tests(&unclosed_but_locked).is_empty(),
            "the exemption applies at the end of the file too"
        );
    }

    /// The documented limit, pinned so it cannot drift silently.
    ///
    /// `#[rstest]` and `#[test_case(..)]` both contain `test`; neither is one.
    /// Relaxing the last-segment rule to a substring would arm the scan on
    /// them, and since neither harness is used in this tree nothing else would
    /// notice.
    #[test]
    fn an_attribute_merely_containing_test_does_not_arm_the_scan() {
        for attribute in ["#[rstest]", "#[test_case(1)]", "#[tests]"] {
            assert!(
                !has_test_attribute(attribute),
                "{attribute}'s path does not end in `test`"
            );
        }
    }

    /// A `#[test]` met while a body is already open belongs to something
    /// nested. Re-arming on it starts a fresh body and throws away everything
    /// the enclosing test had collected, mutation included.
    #[test]
    fn a_nested_test_attribute_does_not_restart_the_enclosing_body() {
        let source = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn outer() {{\n        {}\n\
             \x20       mod inner {{\n            #[test]\n            fn t() {{}}\n\
             \x20       }}\n    }}\n}}\n",
            mutation_call("set_var")
        );
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "the outer test's hit survives: {hits:?}");
        assert_eq!(hits[0].line, 5);
    }

    /// A lock protects from where it is taken, not retroactively.
    ///
    /// Matching the whole accumulated body accepted a lock written after the
    /// mutation it was supposed to cover — the same race the guard exists to
    /// stop, spelled so that the guard agreed with it.
    #[test]
    fn a_lock_taken_after_the_mutation_does_not_serialize_it() {
        let call = mutation_call("set_var");
        let lock = "let _guard = SOME_TEST_LOCK.lock().unwrap();";

        let late = unguarded_test(&format!("{call}\n        {lock}"));
        let hits = env_mutations_in_unserialized_tests(&late);
        assert_eq!(hits.len(), 1, "the mutation ran unlocked: {hits:?}");

        let early = unguarded_test(&format!("{lock}\n        {call}"));
        assert!(
            env_mutations_in_unserialized_tests(&early).is_empty(),
            "taking the lock first is the discipline this guard asks for"
        );

        // Same line, lock first in source order: still covered.
        let same_line = unguarded_test(&format!("{lock} {call}"));
        assert!(
            env_mutations_in_unserialized_tests(&same_line).is_empty(),
            "a lock earlier on the same line precedes the mutation"
        );

        // ...and the same line the other way round is NOT. Ordering by line
        // number alone accepted this, which is the same retroactive lock the
        // per-line rule rejects when it is written on the next line down.
        let same_line_late = unguarded_test(&format!("{call} {lock}"));
        let hits = env_mutations_in_unserialized_tests(&same_line_late);
        assert_eq!(
            hits.len(),
            1,
            "the mutation ran before the lock on that line: {hits:?}"
        );
    }

    /// A lock has to be ACQUIRED, not merely named. `let _x = &LOCK;` mentions
    /// one and holds nothing, and matching the name alone handed that the
    /// exemption.
    ///
    /// The acquisition is looked for across the statement rather than the line,
    /// because rustfmt splits the real shape in this repo over three lines —
    /// which is why "require `.lock(` on the naming line" is not the fix: it
    /// would reject every disciplined test in `crates/rocm-core`. Both shapes
    /// are asserted here so neither can regress into the other.
    #[test]
    fn a_lock_that_is_named_but_not_taken_does_not_serialize_anything() {
        let call = mutation_call("set_var");

        let named_only = unguarded_test(&format!("let _x = &SOME_TEST_LOCK;\n        {call}"));
        let hits = env_mutations_in_unserialized_tests(&named_only);
        assert_eq!(hits.len(), 1, "naming a lock is not holding it: {hits:?}");

        for acquisition in [
            "let _guard = SOME_TEST_LOCK.lock().unwrap();",
            // rustfmt's split, as every `*_TEST_LOCK` site in `rocm-core` is
            // actually written.
            "let _guard = SOME_TEST_LOCK\n            .lock()\n            .unwrap_or_else(std::sync::PoisonError::into_inner);",
        ] {
            let source = unguarded_test(&format!("{acquisition}\n        {call}"));
            assert!(
                env_mutations_in_unserialized_tests(&source).is_empty(),
                "this lock is taken, however rustfmt broke the lines up"
            );
        }
    }

    /// The serialization point is the FIRST lock, not the last one seen.
    ///
    /// Letting a later acquisition overwrite it turns the exemption into a
    /// false accusation: a test that locks, mutates, then takes a second lock
    /// gets reported for a mutation that was covered the whole time. A guard
    /// that cries wolf on disciplined code is how a build check stops being
    /// believed.
    #[test]
    fn a_later_lock_does_not_move_the_serialization_point() {
        let call = mutation_call("set_var");
        let lock = "let _guard = SOME_TEST_LOCK.lock().unwrap();";

        let source = unguarded_test(&format!("{lock}\n        {call}\n        {lock}"));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert!(
            hits.is_empty(),
            "the first lock already covered this mutation: {hits:?}"
        );
    }

    /// A statement that ends on its own line pulls in no continuation.
    ///
    /// [`statement_continuation`] returns early on a line already carrying its
    /// `;`. Without that early return it would read the lines BELOW as part of
    /// the statement, so a line that merely NAMES a lock would borrow the
    /// `.lock(` from an unrelated acquisition further down and hand out the
    /// exemption retroactively — the mutation here sits between the two.
    ///
    /// The loop's `break` on the terminator is pinned separately, by
    /// [`a_statement_ending_inside_the_window_stops_the_walk_there`]: the early
    /// return means a line carrying its own `;` never reaches the loop at all.
    #[test]
    fn a_self_terminating_line_does_not_borrow_a_later_acquisition() {
        let call = mutation_call("set_var");
        let source = unguarded_test(&format!(
            "let _x = &SOME_TEST_LOCK;\n        {call}\n        \
             let _real = OTHER_TEST_LOCK.lock().unwrap();"
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(
            hits.len(),
            1,
            "the naming line ended its own statement; the lock below it covers nothing \
             above it: {hits:?}"
        );
    }

    /// The continuation window is bounded, and the bound is load-bearing in
    /// both directions.
    ///
    /// Too narrow and rustfmt's real three-line split of a lock acquisition
    /// stops being recognised, which rejects every disciplined test in
    /// `rocm-core` — that direction is pinned by
    /// [`a_lock_that_is_named_but_not_taken_does_not_serialize_anything`]. Too
    /// wide and the walk keeps running past the end of the statement into
    /// whatever follows, so an acquisition that is nowhere near the naming line
    /// still exempts it. This fixture splits one wider than the window.
    #[test]
    fn an_acquisition_split_wider_than_the_window_is_not_credited() {
        let call = mutation_call("set_var");
        let padding = "\n            // filler\n".repeat(5);
        let source = unguarded_test(&format!(
            "let _guard = SOME_TEST_LOCK{padding}            .lock()\n            .unwrap();\n        {call}"
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(
            hits.len(),
            1,
            "the acquisition is further from its name than the window reaches: {hits:?}"
        );
    }

    /// A statement that ends partway through the window stops the walk there.
    ///
    /// The naming line carries no `;`, so it does reach the loop — and the loop
    /// must stop at the line that terminates the statement. Without the
    /// `break` it keeps reading to the window's edge and picks up an unrelated
    /// acquisition below the mutation, exempting a mutation that ran first.
    ///
    /// The `.clone()` continuation is what makes this distinct from
    /// [`a_self_terminating_line_does_not_borrow_a_later_acquisition`], which
    /// never enters the loop.
    #[test]
    fn a_statement_ending_inside_the_window_stops_the_walk_there() {
        let call = mutation_call("set_var");
        let source = unguarded_test(&format!(
            "let _x = &SOME_TEST_LOCK\n            .clone();\n        {call}\n        \
             let _real = OTHER_TEST_LOCK.lock().unwrap();"
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(
            hits.len(),
            1,
            "the statement ended at `.clone();`; nothing below it belongs to it: {hits:?}"
        );
    }

    /// The offense is the LEFTMOST mutation on the line, not the first one that
    /// happens to be listed in [`MUTATIONS`].
    ///
    /// The reported column is what orders the mutation against a lock taken on
    /// the same line, so picking by list order instead of position is a silent
    /// false negative: here the mutation that ran BEFORE the lock is the one
    /// later in the list, and taking the list's first reports a column to the
    /// right of the acquisition, which reads as serialized.
    #[test]
    fn the_reported_mutation_is_the_leftmost_one_on_the_line() {
        let source = unguarded_test(&format!(
            "{} let _g = SOME_TEST_LOCK.lock().unwrap(); {}",
            mutation_call("remove_var"),
            mutation_call("set_var")
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(
            hits.len(),
            1,
            "the mutation left of the lock is unserialized: {hits:?}"
        );
        assert_eq!(
            hits[0].call, "remove_var",
            "the leftmost call is the one that ran before the lock"
        );
    }

    /// `_TEST_LOCK` has to END the identifier.
    ///
    /// Dropping the word-boundary check hands the exemption to anything merely
    /// CARRYING the suffix — `SOME_TEST_LOCK_HELPER` is a different item, and a
    /// test naming one holds no lock at all. A false negative, and a silent
    /// one: the guard reports nothing and the race stays.
    #[test]
    fn an_identifier_merely_carrying_the_lock_suffix_is_not_a_lock() {
        let call = mutation_call("set_var");
        let source = unguarded_test(&format!(
            "let _h = SOME_TEST_LOCK_HELPER.lock().unwrap();\n        {call}"
        ));
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(
            hits.len(),
            1,
            "`SOME_TEST_LOCK_HELPER` is not a `*_TEST_LOCK`: {hits:?}"
        );
    }

    /// The depth counter saturates, and that is observable behaviour rather
    /// than a formality: on a file this scan misreads, a plain `-` panics with
    /// `attempt to subtract with overflow` and takes the build down with a
    /// message about the guard rather than about the offending test. Saturating
    /// reports nothing instead, which is the right answer for a close with
    /// nothing open.
    ///
    /// This was previously called an equivalent mutant on the grounds that
    /// swapping the operator left the suite green. It is not: the suite simply
    /// had no input in this shape.
    #[test]
    fn an_unbalanced_close_returns_nothing_rather_than_panicking() {
        assert!(
            env_mutations_in_unserialized_tests("}\n}\n").is_empty(),
            "a close with nothing open is nothing to report"
        );
    }

    /// The suffix rule is a suffix. `NOT_A_TEST_LOCK_HELPER` contains
    /// `_TEST_LOCK` and is not a lock, so relaxing the match to a substring
    /// would hand out the exemption to whatever happens to be named that way.
    #[test]
    fn a_name_merely_containing_test_lock_does_not_exempt_anything() {
        let source = unguarded_test(&format!(
            "let _x = NOT_A_TEST_LOCK_HELPER.get();\n        {}",
            mutation_call("set_var")
        ));
        assert_eq!(
            env_mutations_in_unserialized_tests(&source).len(),
            1,
            "nothing here takes a lock"
        );
    }

    /// A mutation sharing the line that opens the test body.
    ///
    /// The body used to be opened after the line was scanned, so this one line
    /// was never looked at. rustfmt splits the shape, which kept it out of
    /// sight — but a `#[rustfmt::skip]` test would have gone through in
    /// silence, and a guard that can silently stop catching things is the
    /// failure mode this file is written against.
    #[test]
    fn a_mutation_sharing_the_opening_brace_line_is_flagged() {
        let source = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{ {} }}\n}}\n",
            mutation_call("set_var")
        );
        let hits = env_mutations_in_unserialized_tests(&source);
        assert_eq!(hits.len(), 1, "the whole test is on one line: {hits:?}");
        assert_eq!(hits[0].line, 4);
    }
}
