// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static contract test: every saved proptest failure file must sit where
//! proptest will read it.
//!
//! proptest finds a test's saved failure cases (`cc <hash> # shrinks to ...`
//! lines) by the SOURCE FILE the `proptest!` block is written in. With the
//! default `FileFailurePersistence::SourceParallel("proptest-regressions")`,
//! a block in `<crate>/src/<path>.rs` reads and writes
//! `<crate>/proptest-regressions/<path>.txt`. Move the test to another file
//! and its seed file is no longer read: no warning, no failure, and the
//! recorded counterexample is silently never replayed again. That happened
//! twice during module splits in `apps/rocm` -- once for real
//! (`proptest-regressions/main.txt`, whose test had moved to another file) and
//! once caught in review before it merged.
//!
//! This guard walks the whole tree, finds every file under a
//! `proptest-regressions/` directory, and maps it back to the source file
//! proptest would have derived it from. A seed file is an orphan when that
//! source file does not exist, or exists but holds no `proptest!` block. A
//! block is the name, the `!` and an opening delimiter as separate tokens, so
//! `proptest ! { .. }` and `proptest /* note */ ! { .. }` count; a mention in a
//! comment or a string literal, a longer name (`my_proptest!`) and a bang that
//! opens nothing (`proptest != x`) do not. The failure names the file, says
//! which source file it expected, and says what to do about it.
//!
//! What it deliberately does not check: that the seed still reproduces
//! anything (a fixed bug leaves a harmless, still-replayed seed), or which
//! `proptest!` block in a file owns which line (proptest does not record
//! that; all blocks in a file share one seed file). It also assumes the
//! default persistence. A test configured with `failure_persistence: None`
//! never writes a seed file, and one with a different `FileFailurePersistence`
//! would put its seeds outside `proptest-regressions/`, where this does not
//! look. `cfg(test)` only, like the other source-scan contracts in this
//! crate: `cargo test -p xtask` runs it, and the merge queue (which runs the
//! whole workspace) is where a change confined to another crate meets it.

#[cfg(test)]
mod tests {
    use crate::rust_source::strip_literals_and_comments;
    use std::path::{Path, PathBuf};

    /// The directory name proptest's default persistence uses.
    const REGRESSIONS_DIR: &str = "proptest-regressions";

    /// The name of the macro that marks a file as holding a property test,
    /// which `proptest!` spells with a `!` and a delimiter after it.
    const PROPTEST_MACRO: &str = "proptest";

    fn repo_root() -> PathBuf {
        // CARGO_MANIFEST_DIR is the xtask/ crate dir; its parent is the repo
        // root (same idiom as workflow_contract::repo_root).
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Every file under `root`, as root-relative `/` paths, sorted.
    ///
    /// A whole-tree walk rather than a list of crate directories, so a crate
    /// added under a new top-level directory is covered without anyone
    /// remembering to extend this. `target/` and dot-directories are skipped:
    /// the latter include the `.claude/worktrees` checkouts of other
    /// branches, and scanning those would make the verdict depend on state
    /// outside the commit under test.
    fn workspace_files(root: &Path) -> Vec<String> {
        let mut found = Vec::new();
        collect_files(root, root, &mut found);
        found.sort();
        found
    }

    fn collect_files(root: &Path, dir: &Path, found: &mut Vec<String>) {
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
                if !skip {
                    collect_files(root, &path, found);
                }
            } else {
                let relative = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                found.push(relative);
            }
        }
    }

    /// The files that live under a `proptest-regressions/` directory.
    fn regression_files(all_files: &[String]) -> Vec<String> {
        all_files
            .iter()
            .filter(|file| file.split('/').any(|part| part == REGRESSIONS_DIR))
            .cloned()
            .collect()
    }

    /// The source file proptest derives `seed` from, or `None` when `seed` is
    /// not shaped like one of its seed files (`<crate dir>/proptest-regressions/<path>.txt`).
    ///
    /// The crate directory is everything before the `proptest-regressions`
    /// component, which is empty for a crate at the repository root. The
    /// inverse of proptest's own rule: strip the directory, swap `.txt` for
    /// `.rs`, and put the result under the crate's `src/`.
    fn expected_source(seed: &str) -> Option<String> {
        let parts: Vec<&str> = seed.split('/').collect();
        let at = parts.iter().position(|part| *part == REGRESSIONS_DIR)?;
        let (crate_dir, below) = (&parts[..at], &parts[at + 1..]);
        let (last, dirs) = below.split_last()?;
        let stem = last.strip_suffix(".txt").filter(|stem| !stem.is_empty())?;
        let mut path: Vec<String> = crate_dir.iter().map(|part| (*part).to_string()).collect();
        path.push("src".to_string());
        path.extend(dirs.iter().map(|part| (*part).to_string()));
        path.push(format!("{stem}.rs"));
        Some(path.join("/"))
    }

    /// True when `source` has a `proptest!` block in code.
    ///
    /// A `proptest!` named in a comment or a string literal must not keep an
    /// orphaned seed alive: a note about where a test moved, whether a `//`
    /// line, a trailing `// ...` after code, or a `/* ... */` block (nested
    /// ones included), and a test disabled by wrapping it in a block comment, are
    /// not property tests. So the text is tokenized first, with the same
    /// [`strip_literals_and_comments`] the other source-scan contracts use,
    /// which blanks the contents of comments, string literals, raw strings and
    /// char literals, and only what is left is searched.
    ///
    /// What is searched for is the name, the `!` and an opening delimiter as
    /// separate tokens, not the text `proptest!`: Rust allows whitespace and
    /// comments between them (`proptest ! { .. }`, `proptest /* note */ ! { .. }`),
    /// and proptest reads the seed file of such a block all the same. The
    /// stripper leaves whitespace alone and blanks a comment to spaces, so
    /// skipping whitespace after the name, and again after the `!`, covers
    /// both. The name must not be glued to a longer identifier (`my_proptest!`
    /// is another macro), and a `!` that does not open a delimiter
    /// (`proptest != x`, a variable compared with something) is not an
    /// invocation.
    ///
    /// It still does not parse Rust: a `proptest!` inside another macro's
    /// arguments counts, and so does one in a block that is compiled out by a
    /// `cfg`. Nothing in this tree does either.
    fn has_proptest_block(source: &str) -> bool {
        let code = strip_literals_and_comments(source);
        code.match_indices(PROPTEST_MACRO).any(|(at, name)| {
            let glued_to_a_longer_name = code[..at]
                .chars()
                .next_back()
                .is_some_and(|before| before.is_alphanumeric() || before == '_');
            if glued_to_a_longer_name {
                return false;
            }
            code[at + name.len()..]
                .trim_start()
                .strip_prefix('!')
                .is_some_and(|after_bang| after_bang.trim_start().starts_with(['{', '(', '[']))
        })
    }

    /// Why a seed file is not read by any test.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Orphan {
        /// Not `<path>.txt` under a `proptest-regressions` directory, so
        /// proptest would never produce or read it.
        NotASeedFile { seed: String },
        /// The source file proptest would read this seed for does not exist.
        SourceMissing { seed: String, expected: String },
        /// The source file exists but no longer holds a `proptest!` block.
        NoPropertyTest { seed: String, expected: String },
    }

    /// The seed files nothing reads.
    ///
    /// `read_source` returns a file's text by root-relative path, or `None`
    /// when it does not exist; taking it as a parameter keeps the decision
    /// pure so fixtures can stand in for the tree.
    fn orphaned_seeds(
        seeds: &[String],
        read_source: impl Fn(&str) -> Option<String>,
    ) -> Vec<Orphan> {
        let mut orphans = Vec::new();
        for seed in seeds {
            let Some(expected) = expected_source(seed) else {
                orphans.push(Orphan::NotASeedFile { seed: seed.clone() });
                continue;
            };
            match read_source(&expected) {
                None => orphans.push(Orphan::SourceMissing {
                    seed: seed.clone(),
                    expected,
                }),
                Some(text) if !has_proptest_block(&text) => orphans.push(Orphan::NoPropertyTest {
                    seed: seed.clone(),
                    expected,
                }),
                Some(_) => {}
            }
        }
        orphans
    }

    /// The failure text: the file, what proptest expected, the likely cause,
    /// and the fix. The cause and fix are the same for every kind, so they are
    /// written once.
    fn describe(orphans: &[Orphan]) -> String {
        let lines: Vec<String> = orphans
            .iter()
            .map(|orphan| match orphan {
                Orphan::NotASeedFile { seed } => format!(
                    "  {seed}\n      not shaped like a proptest seed file \
                     (<crate>/{REGRESSIONS_DIR}/<path>.txt), so proptest never reads it"
                ),
                Orphan::SourceMissing { seed, expected } => format!(
                    "  {seed}\n      proptest would read it for {expected}, which does not exist"
                ),
                Orphan::NoPropertyTest { seed, expected } => format!(
                    "  {seed}\n      proptest would read it for {expected}, which has no \
                     `{PROPTEST_MACRO}` block"
                ),
            })
            .collect();
        format!(
            "{} proptest seed file(s) that no test reads. proptest names a seed file after the \
             source file of the test (<crate>/src/<path>.rs reads \
             <crate>/{REGRESSIONS_DIR}/<path>.txt), so the likely cause is that the property \
             test moved to another file, or the file was renamed: the saved counterexample is \
             then never replayed again, with no warning. Fix: put its `cc` lines in \
             <crate>/{REGRESSIONS_DIR}/<path of the file the test is in now, relative to src/, \
             .txt>: `git mv` the file when nothing is there yet, and append the lines to the \
             existing file when something is (a destination that already has seeds of its own \
             keeps them). Then check it is read by appending a deliberately unparsable line and \
             running the test with `--nocapture` (proptest prints `unparsable line, \
             ignoring`); remove the line again. If the test is gone, delete the seed file.\n{}",
            orphans.len(),
            lines.join("\n")
        )
    }

    /// Orphans among the real files under `root`.
    fn orphans_under(root: &Path) -> Vec<Orphan> {
        let seeds = regression_files(&workspace_files(root));
        orphaned_seeds(&seeds, |path| std::fs::read_to_string(root.join(path)).ok())
    }

    /// The contract: every seed file is read by a test that still exists.
    #[test]
    fn every_proptest_regression_file_belongs_to_a_property_test_that_reads_it() {
        let orphans = orphans_under(&repo_root());
        assert!(orphans.is_empty(), "{}", describe(&orphans));
    }

    /// The walk reaches the crates' sources, so an empty seed list means
    /// "nothing to check" and not "the scan looked in the wrong place".
    #[test]
    fn the_scan_sees_the_workspace_it_is_guarding() {
        let files = workspace_files(&repo_root());
        assert!(
            files.iter().any(|file| file == "apps/rocm/src/main.rs"),
            "the walk did not reach apps/rocm/src/main.rs -- the walk is broken, not the tree"
        );
    }

    fn fixture(files: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let files: Vec<(String, String)> = files
            .iter()
            .map(|(path, text)| ((*path).to_string(), (*text).to_string()))
            .collect();
        move |path| {
            files
                .iter()
                .find(|(candidate, _)| candidate == path)
                .map(|(_, text)| text.clone())
        }
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn a_seed_maps_to_the_source_file_proptest_derives_it_from() {
        assert_eq!(
            expected_source("apps/rocm/proptest-regressions/model_catalog_text.txt").as_deref(),
            Some("apps/rocm/src/model_catalog_text.rs")
        );
        assert_eq!(
            expected_source("apps/rocm/proptest-regressions/providers/local_http/proptests.txt")
                .as_deref(),
            Some("apps/rocm/src/providers/local_http/proptests.rs")
        );
        assert_eq!(
            expected_source("crates/rocm-dash-tui/proptest-regressions/ui/format.txt").as_deref(),
            Some("crates/rocm-dash-tui/src/ui/format.rs")
        );
        // A crate at the repository root has an empty crate directory.
        assert_eq!(
            expected_source("proptest-regressions/a.txt").as_deref(),
            Some("src/a.rs")
        );
    }

    #[test]
    fn a_file_that_is_not_a_txt_under_a_regressions_directory_has_no_source() {
        assert_eq!(expected_source("apps/rocm/proptest-regressions/a.rs"), None);
        assert_eq!(expected_source("apps/rocm/proptest-regressions/.txt"), None);
        assert_eq!(expected_source("apps/rocm/proptest-regressions/"), None);
        assert_eq!(expected_source("apps/rocm/src/a.txt"), None);
    }

    #[test]
    fn a_seed_whose_test_moved_to_another_file_is_an_orphan() {
        // The shape of the real failure: the seed is still named for the file
        // the test used to be in, and that file no longer has the test.
        let read = fixture(&[
            ("apps/rocm/src/main.rs", "fn main() {}\n"),
            (
                "apps/rocm/src/examine_report_text.rs",
                "proptest! { fn format_bytes_renders_a_size_in_its_own_unit() {} }\n",
            ),
        ]);
        let orphans = orphaned_seeds(&strings(&["apps/rocm/proptest-regressions/main.txt"]), read);
        assert_eq!(
            orphans,
            vec![Orphan::NoPropertyTest {
                seed: "apps/rocm/proptest-regressions/main.txt".to_string(),
                expected: "apps/rocm/src/main.rs".to_string(),
            }]
        );
    }

    #[test]
    fn a_seed_whose_source_file_is_gone_is_an_orphan() {
        let read = fixture(&[("apps/rocm/src/other.rs", "proptest! {}\n")]);
        let orphans = orphaned_seeds(&strings(&["apps/rocm/proptest-regressions/gone.txt"]), read);
        assert_eq!(
            orphans,
            vec![Orphan::SourceMissing {
                seed: "apps/rocm/proptest-regressions/gone.txt".to_string(),
                expected: "apps/rocm/src/gone.rs".to_string(),
            }]
        );
    }

    #[test]
    fn a_stray_file_in_a_regressions_directory_is_an_orphan() {
        let read = fixture(&[]);
        let orphans = orphaned_seeds(&strings(&["apps/rocm/proptest-regressions/notes.md"]), read);
        assert_eq!(
            orphans,
            vec![Orphan::NotASeedFile {
                seed: "apps/rocm/proptest-regressions/notes.md".to_string(),
            }]
        );
    }

    #[test]
    fn a_seed_beside_its_property_test_is_not_an_orphan() {
        let read = fixture(&[
            (
                "apps/rocm/src/model_catalog_text.rs",
                "mod tests {\n    proptest! {\n        #[test] fn p() {}\n    }\n}\n",
            ),
            (
                "crates/rocm-core/src/runtime.rs",
                "    ::proptest::proptest! { }\n",
            ),
        ]);
        let seeds = strings(&[
            "apps/rocm/proptest-regressions/model_catalog_text.txt",
            "crates/rocm-core/proptest-regressions/runtime.txt",
        ]);
        assert_eq!(orphaned_seeds(&seeds, read), Vec::new());
    }

    #[test]
    fn a_comment_naming_proptest_does_not_keep_a_seed_alive() {
        // A note about where the test moved to is not a property test.
        assert!(!has_proptest_block(
            "// the proptest! that was here moved to examine_report_text.rs\n/// proptest! docs\n"
        ));
        assert!(has_proptest_block("    proptest! {\n"));
        let read = fixture(&[(
            "apps/rocm/src/main.rs",
            "// the proptest! moved to examine_report_text.rs\nfn main() {}\n",
        )]);
        let orphans = orphaned_seeds(&strings(&["apps/rocm/proptest-regressions/main.txt"]), read);
        assert_eq!(orphans.len(), 1);
    }

    /// The seed of `apps/rocm/src/main.rs`, whose property test moved away and
    /// left only `source` behind: it is an orphan whenever `source` has no
    /// `proptest!` in code.
    fn orphans_when_main_holds(source: &str) -> Vec<Orphan> {
        let files = [("apps/rocm/src/main.rs", source)];
        let seeds = strings(&["apps/rocm/proptest-regressions/main.txt"]);
        orphaned_seeds(&seeds, fixture(&files))
    }

    #[test]
    fn a_proptest_named_outside_code_does_not_keep_a_seed_alive() {
        // Everything a note about a moved test, or a disabled one, can look like
        // without being a property test. Each used to be accepted: only a line
        // that STARTED with `//` was excluded.
        let expected = vec![Orphan::NoPropertyTest {
            seed: "apps/rocm/proptest-regressions/main.txt".to_string(),
            expected: "apps/rocm/src/main.rs".to_string(),
        }];
        let mut accepted = Vec::new();
        for (what, source) in [
            (
                "a trailing line comment",
                "fn main() {} // proptest! moved elsewhere\n",
            ),
            (
                "a block comment on one line",
                "fn main() {} /* proptest! moved elsewhere */\n",
            ),
            (
                "a block comment around the whole property block",
                "/*\nproptest! {\n    #[test]\n    fn p() {}\n}\n*/\nfn main() {}\n",
            ),
            (
                "a nested block comment",
                "/* outer /* inner */ proptest! { fn p() {} } still outer */\nfn main() {}\n",
            ),
            (
                "a block comment that opens after code",
                "fn main() {} /* proptest! {\n    fn p() {}\n} */\n",
            ),
            (
                "a string literal",
                "const NOTE: &str = \"the proptest! moved elsewhere\";\n",
            ),
            (
                "a raw string literal",
                "const NOTE: &str = r#\"the \"proptest!\" moved\"#;\n",
            ),
        ] {
            if orphans_when_main_holds(source) != expected {
                accepted.push(what);
            }
        }
        assert!(
            accepted.is_empty(),
            "the guard counted a proptest! outside code as a property test: {accepted:?}"
        );
    }

    #[test]
    fn text_that_only_resembles_the_macro_does_not_keep_a_seed_alive() {
        // The other side of reading `proptest`, `!` and a delimiter as separate
        // tokens: the looser match must not accept what is not an invocation.
        let expected = vec![Orphan::NoPropertyTest {
            seed: "apps/rocm/proptest-regressions/main.txt".to_string(),
            expected: "apps/rocm/src/main.rs".to_string(),
        }];
        let mut accepted = Vec::new();
        for (what, source) in [
            ("a longer macro name", "my_proptest! { }\n"),
            ("a longer macro name, spaced", "my_proptest ! { }\n"),
            ("a name that continues", "proptest_cases! { }\n"),
            ("the name alone", "use proptest::prelude::*;\n"),
            ("a function of that name", "fn proptest() {}\n"),
            (
                "a comparison",
                "fn f(proptest: u8) -> bool { proptest != 3 }\n",
            ),
            ("a bang with no delimiter", "const X: u8 = proptest ! 3;\n"),
            ("a sibling macro", "proptest::prop_compose! { }\n"),
            ("a trailing name", "proptest\n"),
        ] {
            if orphans_when_main_holds(source) != expected {
                accepted.push(what);
            }
        }
        assert!(
            accepted.is_empty(),
            "the guard counted text that is not a proptest! invocation: {accepted:?}"
        );
    }

    #[test]
    fn a_property_block_next_to_comments_and_literals_still_counts() {
        // The other direction: stripping must not hide a real block. Each is
        // a source the guard has to accept, so the stricter scan cannot turn a
        // seed that something reads into a false orphan.
        let mut hidden = Vec::new();
        for (what, source) in [
            ("after a block comment", "/* note */ proptest! { }\n"),
            ("after a line comment", "// note\nproptest! { }\n"),
            (
                "after a nested block comment",
                "/* a /* b */ c */ proptest! { }\n",
            ),
            (
                "after a string literal",
                "const A: &str = \"x\"; proptest! { }\n",
            ),
            (
                "after a lifetime",
                "fn f<'a>(_: &'a str) {} proptest! { }\n",
            ),
            (
                "after a char literal",
                "const C: char = '\"'; proptest! { }\n",
            ),
            (
                "with a comment inside the block",
                "proptest! {\n    // a note\n    #[test] fn p() {}\n}\n",
            ),
            ("through its path", "::proptest::proptest! { }\n"),
            ("with a space before the bang", "proptest ! { }\n"),
            ("with a tab before the bang", "proptest\t! { }\n"),
            ("with the bang on the next line", "proptest\n    ! {\n}\n"),
            (
                "with a block comment before the bang",
                "proptest /* note */ ! { }\n",
            ),
            (
                "with a nested block comment before the bang",
                "proptest/* a /* b */ c */! { }\n",
            ),
            (
                "with a line comment before the bang",
                "proptest // note\n    ! { }\n",
            ),
            (
                "with a space before the bang, through its path",
                "::proptest::proptest ! { }\n",
            ),
            ("with spaces around the bang", "proptest  !  {\n}\n"),
            ("with parentheses as its delimiter", "proptest! (\n);\n"),
            ("with brackets as its delimiter", "proptest![];\n"),
            (
                "with a comment between the bang and its delimiter",
                "proptest! /* note */ { }\n",
            ),
        ] {
            if !orphans_when_main_holds(source).is_empty() {
                hidden.push(what);
            }
        }
        assert!(
            hidden.is_empty(),
            "the guard did not see a real proptest! block: {hidden:?}"
        );
    }

    #[test]
    fn the_failure_names_the_file_the_likely_cause_and_the_fix() {
        let text = describe(&[Orphan::NoPropertyTest {
            seed: "apps/rocm/proptest-regressions/main.txt".to_string(),
            expected: "apps/rocm/src/main.rs".to_string(),
        }]);
        assert!(
            text.contains("apps/rocm/proptest-regressions/main.txt"),
            "{text}"
        );
        assert!(text.contains("apps/rocm/src/main.rs"), "{text}");
        assert!(text.contains("moved to another file"), "{text}");
        assert!(text.contains("git mv"), "{text}");
        // A destination that already has seeds keeps them: the advice must not
        // be an unconditional `git mv`, which would replace or refuse there.
        assert!(text.contains("append the lines"), "{text}");
        assert!(text.contains("unparsable line, ignoring"), "{text}");
    }

    /// The walk and the mapping together, on a tree on disk: a seed beside its
    /// test passes, and the same tree with the test moved out from under it
    /// fails. Also covers a nested seed path and a skipped dot-directory.
    #[test]
    fn a_tree_on_disk_is_orphaned_exactly_when_the_test_moves() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let write = |relative: &str, text: &str| {
            let path = tmp.path().join(relative);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
            std::fs::write(path, text).expect("write");
        };
        write("apps/demo/src/main.rs", "fn main() {}\n");
        write("apps/demo/src/sizes.rs", "proptest! { }\n");
        write("apps/demo/src/nested/deep.rs", "proptest! { }\n");
        write(
            "apps/demo/proptest-regressions/sizes.txt",
            "cc 00 # shrinks to n = 1\n",
        );
        write(
            "apps/demo/proptest-regressions/nested/deep.txt",
            "cc 00 # shrinks to n = 1\n",
        );
        // Another checkout's worktree: skipped, so its orphan does not count.
        write(
            ".claude/worktrees/x/apps/demo/proptest-regressions/stale.txt",
            "cc 00\n",
        );
        assert_eq!(orphans_under(tmp.path()), Vec::new());

        // The test moves out of sizes.rs; the seed stays behind.
        write("apps/demo/src/sizes.rs", "pub fn size() {}\n");
        write("apps/demo/src/report.rs", "proptest! { }\n");
        assert_eq!(
            orphans_under(tmp.path()),
            vec![Orphan::NoPropertyTest {
                seed: "apps/demo/proptest-regressions/sizes.txt".to_string(),
                expected: "apps/demo/src/sizes.rs".to_string(),
            }]
        );

        // Re-homing the seed to the file the test is in now clears it.
        std::fs::rename(
            tmp.path().join("apps/demo/proptest-regressions/sizes.txt"),
            tmp.path().join("apps/demo/proptest-regressions/report.txt"),
        )
        .expect("git mv stand-in");
        assert_eq!(orphans_under(tmp.path()), Vec::new());
    }
}
