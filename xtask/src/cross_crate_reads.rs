// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Files a crate's tests read from outside that crate's own directory, and the
//! widening of the directory-based crate selection by them.
//!
//! [`affected::select`] attributes a changed file to the
//! crate whose directory holds it and treats Markdown and `docs/` as unable to
//! affect any crate. A test that reads a file from another directory (a doc, a
//! skill, a fixture of another crate, every Rust source in the repo) is invisible
//! to that rule, so an edit to the file alone would never run the test.
//! [`widen`] adds the reader crates of every changed file in
//! [`CROSS_CRATE_READS`], and their dependents, to a selection.
//! [`affected::select_with_readers`] applies it, and both CI's
//! `cargo xtask affected` and the pre-push hook ([`crate::prepush`]) select
//! through that one function, so they share every rule: given the same
//! changed files, they select the same crates. Their changed files can
//! differ: CI diffs the pull request against its base, while the hook diffs
//! the pushed range and runs only for a push that changes a Rust file.

use std::collections::BTreeSet;

use crate::affected::{self, Graph, Selection};

/// Files a crate's tests (or `include_str!`) read from outside that crate's own
/// directory, with the crates that read them. Without these entries a change
/// that edits only such a file would never run the reader's tests in CI's
/// `Test (affected crates)` job, nor, for every reader but `e2e-cucumber`, in
/// the pre-push hook's `test` step (which never tests that crate).
///
/// A pattern ending in `/` is a directory prefix, one starting with `*` is a
/// file-name suffix, anything else is an exact path. Files that no crate owns
/// and that are not Markdown (`install.sh`, `scripts/`, `.github/`, ...) need no
/// entry: the selection already sends them to the whole workspace, as it does
/// the docs in [`affected::XTASK_READ_DOCS`]. The guard
/// `every_cross_crate_read_selects_its_reader` scans the workspace's code for
/// `include_str!`/`include_bytes!`, manifest-relative reads and
/// `repo_root().join("...")` calls with a path written out at the call that
/// leave their crate, and fails when one is not covered here or by CI's `rust`
/// path category, or names a path Windows resolves differently (a component
/// ending in `.` or a space). Comments and string literals are ignored, so an
/// example call in a doc comment or a test fixture string is not a read. A
/// path held in a const, or built at run time, is not seen,
/// so such a read is guarded only by what pins the table entry or the CI glob
/// by hand. Reads made by the e2e harness (the `e2e` test target, which
/// only `cargo xtask e2e` runs) are not listed.
///
/// - `*.rs`: xtask's contract tests scan every Rust source file in the repo
///   (`ascii_output_contract`, `env_mutation_contract`) and the pinned-key test
///   reads `apps/rocm/src/therock/fetch.rs`.
/// - `README.md`, `docs/`, `skills/`: `rocm`'s `advised_commands` test checks
///   every `rocm ...` command shown there, `rocm` compiles in the assistant
///   skill, and xtask's tests read the pinned release key under `docs/keys` and
///   other docs (`docs/nightly-on-the-queue.md`, `docs/testing.md`, ...).
/// - `skills/rocm-doctor/`: the e2e crate's `skill_reference` test (a plain
///   test target) parses the skill's catalog table. CI runs it in the `test`
///   job's `cargo nextest` over this selection and in the `windows-debug`
///   job's `cargo test --workspace --all-targets`, whose steps run when the
///   `heavy` category matches (`skills/**` is in it); the pre-push hook's `test`
///   step excludes `e2e-cucumber`, so there the entry only adds the clippy
///   steps.
/// - the e2e PowerShell fixture: `rocm-core`'s `fix` tests `include_str!` it
///   although it lives in the e2e crate.
/// - `apps/rocm/Cargo.toml`: xtask's `e2e-test-hooks` contract in
///   `workflow_contract` reads `rocm`'s `e2e-test-hooks` feature list, which
///   must turn the feature on in every other crate with code behind it.
pub(crate) const CROSS_CRATE_READS: &[(&str, &[&str])] = &[
    ("*.rs", &["xtask"]),
    ("README.md", &["rocm", "xtask"]),
    ("docs/", &["rocm", "xtask"]),
    ("skills/", &["rocm", "xtask"]),
    ("skills/rocm-doctor/", &["e2e-cucumber"]),
    ("tests/e2e-cucumber/fixtures/", &["rocm-core"]),
    ("apps/rocm/Cargo.toml", &["xtask"]),
];

fn read_pattern_matches(pattern: &str, file: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix('*') {
        file.ends_with(suffix)
    } else if pattern.ends_with('/') {
        file.starts_with(pattern)
    } else {
        file == pattern
    }
}

/// `base`, the directory-based selection for `changed_files`, widened with the
/// reader crates in `reads` of every changed file and their dependents.
///
/// The whole workspace stays the whole workspace. A reader that is not a
/// workspace member means the table is stale, so the answer is the whole
/// workspace rather than a guess.
pub(crate) fn widen(
    base: Selection,
    changed_files: &[String],
    graph: &Graph,
    reads: &[(&str, &[&str])],
) -> Selection {
    if base == Selection::Workspace {
        return base;
    }
    let readers: BTreeSet<&str> = reads
        .iter()
        .filter(|(pattern, _)| {
            changed_files
                .iter()
                .any(|file| read_pattern_matches(pattern, file))
        })
        .flat_map(|(_, readers)| readers.iter().copied())
        .collect();
    if readers.is_empty() {
        return base;
    }
    let members: BTreeSet<&str> = graph
        .crate_dirs
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    if readers.iter().any(|reader| !members.contains(reader)) {
        return Selection::Workspace;
    }
    let mut crates = match base {
        Selection::Packages(packages) => packages,
        Selection::Workspace | Selection::Empty => BTreeSet::new(),
    };
    crates.extend(readers.into_iter().map(String::from));
    Selection::Packages(affected::reverse_closure(&crates, &graph.rev_deps))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    // ---- the widening rule ----

    fn reader_graph() -> Graph {
        let mut rev_deps = BTreeMap::new();
        rev_deps.insert(
            "rocm-core".to_string(),
            vec!["rocm".to_string(), "xtask".to_string()],
        );
        for name in ["rocm", "xtask", "e2e-cucumber"] {
            rev_deps.insert(name.to_string(), vec![]);
        }
        Graph {
            crate_dirs: vec![
                ("rocm-core".into(), "crates/rocm-core".into()),
                ("rocm".into(), "apps/rocm".into()),
                ("xtask".into(), "xtask".into()),
                ("e2e-cucumber".into(), "tests/e2e-cucumber".into()),
            ],
            rev_deps,
        }
    }

    fn selected(paths: &[&str]) -> Selection {
        let changed: Vec<String> = paths.iter().map(|path| (*path).to_string()).collect();
        affected::select_with_readers(&changed, &reader_graph())
    }

    fn selected_names(paths: &[&str]) -> BTreeSet<String> {
        match selected(paths) {
            Selection::Packages(names) => names,
            other => panic!("expected packages for {paths:?}, got {other:?}"),
        }
    }

    #[test]
    fn the_fixture_alone_selects_its_reader() {
        // The fixture's own crate, its reader `rocm-core`, and the crates that
        // depend on the reader (`rocm`, `xtask`): a change to what the reader
        // tests can break theirs too.
        let want: BTreeSet<String> = ["e2e-cucumber", "rocm-core", "rocm", "xtask"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            selected(&["tests/e2e-cucumber/fixtures/powershell_text_rule_cases.txt"]),
            Selection::Packages(want)
        );
    }

    #[test]
    fn docs_and_skills_select_the_crates_whose_tests_read_them() {
        for path in [
            "README.md",
            "docs/testing.md",
            "skills/rocm-cli-assistant/SKILL.md",
        ] {
            let names = selected_names(&[path]);
            assert!(names.contains("rocm"), "{path}: {names:?}");
            assert!(names.contains("xtask"), "{path}: {names:?}");
            assert!(!names.contains("rocm-core"), "{path}: {names:?}");
        }
    }

    #[test]
    fn the_doctor_skill_selects_the_e2e_test_that_parses_it() {
        assert!(selected_names(&["skills/rocm-doctor/reference.md"]).contains("e2e-cucumber"));
        assert!(!selected_names(&["skills/rocm-cli-assistant/SKILL.md"]).contains("e2e-cucumber"));
    }

    #[test]
    fn any_rust_file_selects_xtask_for_its_repo_wide_source_scans() {
        assert!(selected_names(&["apps/rocm/src/main.rs"]).contains("xtask"));
    }

    #[test]
    fn markdown_nothing_reads_still_selects_nothing() {
        assert_eq!(selected(&["CHANGELOG.md"]), Selection::Empty);
    }

    #[test]
    fn the_whole_workspace_stays_the_whole_workspace() {
        assert_eq!(
            selected(&["Cargo.lock", "apps/rocm/src/main.rs"]),
            Selection::Workspace
        );
    }

    #[test]
    fn a_reader_that_is_not_a_workspace_member_runs_everything() {
        let changed = ["docs/testing.md".to_string()];
        let table: &[(&str, &[&str])] = &[("docs/", &["a-crate-that-was-renamed"])];
        let base = affected::select(&changed, &reader_graph());
        assert_eq!(
            widen(base, &changed, &reader_graph(), table),
            Selection::Workspace
        );
    }

    #[test]
    fn read_patterns_match_by_directory_suffix_or_exact_path() {
        assert!(read_pattern_matches("docs/", "docs/keys/a.pem"));
        assert!(!read_pattern_matches("docs/", "documents/a.md"));
        assert!(read_pattern_matches("*.rs", "xtask/src/main.rs"));
        assert!(!read_pattern_matches("*.rs", "xtask/Cargo.toml"));
        assert!(read_pattern_matches("README.md", "README.md"));
        assert!(!read_pattern_matches("README.md", "docs/README.md"));
    }

    // ---- drift guard: every cross-crate read must be covered ----

    use std::path::{Path, PathBuf};

    use crate::workflow_contract::tests as ci_filters;

    /// A read that leaves the crate: a repo-relative path, or the repo root
    /// itself when the code derives it and joins the rest elsewhere.
    #[derive(Debug, PartialEq, Eq)]
    enum Read {
        Root,
        Path(String),
    }

    /// The e2e harness: the `e2e` test target, which `cargo test` and
    /// `cargo nextest` skip (`test = false`) and only `cargo xtask e2e` runs.
    /// No crate selection runs it, so its reads need no entry.
    fn is_e2e_harness(source: &str) -> bool {
        source == "tests/e2e-cucumber/tests/e2e.rs"
            || source.starts_with("tests/e2e-cucumber/tests/e2e/")
    }

    /// The text of the sole string literal in `args`, if that is all it holds.
    /// Plain (`"..."`) and raw (`r"..."`, `r#"..."#`) literals are recognised.
    fn sole_literal(args: &str) -> Option<&str> {
        let args = args.trim();
        if let Some(inner) = args.strip_prefix('"') {
            let end = inner.find('"')?;
            return inner[end + 1..].trim().is_empty().then_some(&inner[..end]);
        }
        let raw = args.strip_prefix('r')?;
        let hashes = raw.len() - raw.trim_start_matches('#').len();
        let body = raw[hashes..].strip_prefix('"')?;
        let closing = format!("\"{}", "#".repeat(hashes));
        let end = body.find(&closing)?;
        body[end + closing.len()..]
            .trim()
            .is_empty()
            .then_some(&body[..end])
    }

    /// Index of the `)` that closes the `(` at `open`, skipping string literals.
    fn matching_paren(text: &str, open: usize) -> Option<usize> {
        let bytes = text.as_bytes();
        let mut depth = 0usize;
        let mut i = open;
        while i < bytes.len() {
            match bytes[i] {
                b'"' => {
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        i += if bytes[i] == b'\\' { 2 } else { 1 };
                    }
                }
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// Lexically resolve `..` and `.`; `None` when the path leaves the root.
    /// A name Windows resolves differently is kept, so that
    /// [`target_problems`] reports it instead of the read vanishing.
    fn normalize(path: &str) -> Option<String> {
        let mut parts: Vec<&str> = Vec::new();
        for part in path.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    parts.pop()?;
                }
                other => parts.push(other),
            }
        }
        Some(parts.join("/"))
    }

    /// Why Windows would not open `path` as written, if it would not: a
    /// component ending in `.` or a space (other than `.` and `..`) has those
    /// characters dropped, so `<root>\...` opens the repo root itself. Such a
    /// read means something else on Windows than on Linux, so it is a problem.
    fn windows_unsafe_name(path: &str) -> Option<String> {
        let part = path
            .split('/')
            .find(|part| !matches!(*part, "." | "..") && part.ends_with(['.', ' ']))?;
        Some(format!(
            "names a path Windows resolves differently: {path} (component {part:?})"
        ))
    }

    /// [`windows_unsafe_name`] for every read the three scanners find in
    /// `text` (the file `source`): each scanner keeps such a name, so the guard,
    /// which passes every target to [`target_problems`], reports it.
    fn windows_unsafe_reads(source: &str, text: &str) -> Vec<String> {
        let manifest_paths = manifest_reads(text)
            .into_iter()
            .filter_map(|read| match read {
                Read::Path(path) => Some(path),
                Read::Root => None,
            });
        include_targets(source, text)
            .into_iter()
            .chain(root_join_targets(text))
            .chain(manifest_paths)
            .filter_map(|target| windows_unsafe_name(&target))
            .collect()
    }

    /// For each byte of `text`, whether it starts a character of code: one that
    /// is not inside a comment or a string or char literal. A scanner checks
    /// the first byte of what it matched, so a call shown in a doc comment
    /// (`repo_root().join("...")`) or held in a test fixture string is not
    /// taken for a read.
    fn code_mask(text: &str) -> Vec<bool> {
        let stripped = crate::rust_source::strip_literals_and_comments(text);
        // The stripper writes one character for every character it reads.
        let mut kept = stripped.chars();
        let mut mask = vec![false; text.len()];
        for (at, c) in text.char_indices() {
            mask[at] = kept.next() == Some(c) && !c.is_whitespace();
        }
        mask
    }

    /// Repo-relative targets of the `include_str!`/`include_bytes!` literals in
    /// the file `file_rel`.
    fn include_targets(file_rel: &str, text: &str) -> Vec<String> {
        let dir = file_rel.rsplit_once('/').map_or("", |(dir, _)| dir);
        let code = code_mask(text);
        let mut targets = Vec::new();
        for macro_name in ["include_str", "include_bytes"] {
            let needle = format!("{macro_name}!(");
            for (at, _) in text.match_indices(&needle) {
                if !code[at] {
                    continue;
                }
                let open = at + needle.len() - 1;
                let Some(close) = matching_paren(text, open) else {
                    continue;
                };
                if let Some(literal) = sole_literal(&text[open + 1..close]) {
                    targets.extend(normalize(&format!("{dir}/{literal}")));
                }
            }
        }
        targets
    }

    /// The reads that start at `env!("CARGO_MANIFEST_DIR")` and leave the crate
    /// directory (through `.parent()`, `.ancestors()` or a `..` segment). Scratch
    /// directories under `.rocm-work` and `target` are not reads of repo files.
    fn manifest_reads(text: &str) -> Vec<Read> {
        let needle = format!("env!(\"{}\")", "CARGO_MANIFEST_DIR");
        let bytes = text.as_bytes();
        let code = code_mask(text);
        let mut reads = Vec::new();
        for (at, _) in text.match_indices(&needle) {
            if !code[at] {
                continue;
            }
            let mut i = at + needle.len();
            let (mut escapes, mut segments) = (false, Vec::new());
            loop {
                while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b')') {
                    i += 1;
                }
                if i >= bytes.len() || bytes[i] != b'.' {
                    break;
                }
                i += 1;
                let name_start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let name = &text[name_start..i];
                if i >= bytes.len() || bytes[i] != b'(' {
                    break;
                }
                let Some(close) = matching_paren(text, i) else {
                    break;
                };
                let literal = sole_literal(&text[i + 1..close]);
                match (name, literal) {
                    ("parent" | "ancestors", _) => escapes = true,
                    ("join", Some(path)) => {
                        for part in path.split('/') {
                            match part {
                                ".." => escapes = true,
                                "" | "." => {}
                                other => segments.push(other.to_string()),
                            }
                        }
                    }
                    _ => {}
                }
                i = close + 1;
            }
            if !escapes
                || matches!(
                    segments.first().map(String::as_str),
                    Some(".rocm-work" | "target")
                )
            {
                continue;
            }
            reads.push(if segments.is_empty() {
                Read::Root
            } else {
                Read::Path(segments.join("/"))
            });
        }
        reads
    }

    /// The literal targets of `repo_root().join("...")` calls: the files a test
    /// reads through a crate's repo-root helper with the path written out at the
    /// call. A path joined later, or built at run time, is not seen.
    fn root_join_targets(text: &str) -> Vec<String> {
        let needle = "repo_root()";
        let code = code_mask(text);
        let mut targets = Vec::new();
        for (at, _) in text.match_indices(needle) {
            if !code[at] {
                continue;
            }
            let rest = text[at + needle.len()..].trim_start_matches('?');
            let Some(args) = rest.trim_start().strip_prefix(".join") else {
                continue;
            };
            let args = args.trim_start();
            let open = text.len() - args.len();
            if !args.starts_with('(') {
                continue;
            }
            let Some(close) = matching_paren(text, open) else {
                continue;
            };
            if let Some(literal) = sole_literal(&text[open + 1..close]) {
                targets.extend(normalize(literal));
            }
        }
        targets
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Collects the files under `dir`, skipping build output, vendored trees and
    /// every dot-directory (`.git`, `.rocm-work`, and nested checkouts such as
    /// `.claude/worktrees/...`, whose sources belong to another branch).
    ///
    /// Symbolic links are neither followed nor listed: a link loop would recurse
    /// until the OS refuses, and a link out of the tree would scan files that are
    /// not this workspace's. A subdirectory that cannot be read contributes
    /// nothing instead of failing the test, since a push should not be blocked by
    /// a folder the developer has no access to. `dir` itself is different: the
    /// caller asked for it, so failing to list it is an error, not an empty
    /// directory.
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        walk_entries(std::fs::read_dir(dir)?, files);
        Ok(())
    }

    fn walk_entries(entries: std::fs::ReadDir, files: &mut Vec<PathBuf>) {
        const SKIPPED: &[&str] = &["target", "node_modules", "third_party"];
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_symlink() {
                continue;
            }
            let path = entry.path();
            if kind.is_dir() {
                let name = entry.file_name();
                let name = name.to_str().unwrap_or_default();
                if !name.starts_with('.')
                    && !SKIPPED.contains(&name)
                    && let Ok(entries) = std::fs::read_dir(&path)
                {
                    walk_entries(entries, files);
                }
            } else {
                files.push(path);
            }
        }
    }

    /// Why CI's `test` job would not run for a change to a read, or `None`
    /// when it would. `files` are the files the read reaches: the file itself,
    /// or every file under a directory, each checked on its own (so a narrow
    /// glob such as `dir/*.json` is held to every file it must match). A file
    /// must be in the `rust` path category and not `inert`. A directory with no
    /// file under it is a gap too: nothing could be classified, so nothing
    /// shows the reader runs when a file is added there.
    fn ci_gap(files: &[String], rust: &[String], inert: &[String]) -> Option<String> {
        if files.is_empty() {
            return Some("it names a directory with no file to classify".to_string());
        }
        files.iter().find_map(|file| {
            if !ci_filters::filter_matches(rust, file) {
                Some(format!("{file} is not in CI's `rust` path category"))
            } else if ci_filters::filter_matches(inert, file) {
                Some(format!("{file} is `inert` in CI's path categories"))
            } else {
                None
            }
        })
    }

    #[test]
    fn ci_gap_classifies_each_file_a_read_reaches() {
        let globs = |g: &[&str]| g.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        let files = |f: &[&str]| f.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        let rust = globs(&["docs/**", "scripts/*.sh"]);
        let inert = globs(&["docs/private.md"]);
        assert_eq!(ci_gap(&files(&["docs/a.md"]), &rust, &inert), None);
        assert_eq!(
            ci_gap(&files(&["README.md"]), &rust, &inert).as_deref(),
            Some("README.md is not in CI's `rust` path category")
        );
        assert_eq!(
            ci_gap(&files(&["docs/private.md"]), &rust, &inert).as_deref(),
            Some("docs/private.md is `inert` in CI's path categories")
        );
        // A directory: every file is classified, so one file a narrow glob
        // misses is a gap even when the others match.
        assert_eq!(
            ci_gap(&files(&["scripts/a.sh", "scripts/b.sh"]), &rust, &inert),
            None
        );
        assert_eq!(
            ci_gap(&files(&["scripts/a.sh", "scripts/b.py"]), &rust, &inert).as_deref(),
            Some("scripts/b.py is not in CI's `rust` path category")
        );
        assert_eq!(
            ci_gap(&[], &rust, &inert).as_deref(),
            Some("it names a directory with no file to classify")
        );
    }

    /// The three reads the Windows job reported came from doc comments of this
    /// file (`repo_root().join("...")` as an example), and `<root>\...` opens
    /// the repo root on Windows. A call in a comment or a string is not a read,
    /// and a component Windows resolves differently is not a path.
    #[test]
    fn a_call_in_a_comment_or_a_string_is_not_a_read() {
        let call = |target: &str| format!("repo_root().join(\"{target}\")");
        let text = format!(
            "/// {}\n// {}\nlet s = \"{}\";\nlet p = {};\n",
            call("docs/a.md"),
            call("docs/b.md"),
            call("docs/c.md").replace('"', "\\\""),
            call("docs/d.md"),
        );
        assert_eq!(root_join_targets(&text), ["docs/d.md"]);
        // The same text as code is a read, so the empty answers below come
        // from the comment and not from a scanner that sees nothing.
        let include = format!("{}!(\"../../x.md\")\n", "include_str");
        assert_eq!(
            include_targets("crates/a/src/lib.rs", &include),
            ["crates/x.md"]
        );
        assert!(include_targets("crates/a/src/lib.rs", &format!("// {include}")).is_empty());
        let manifest = format!(
            "{}.parent().unwrap().join(\"README.md\")\n",
            "env!(\"CARGO_MANIFEST_DIR\")"
        );
        assert!(!manifest_reads(&manifest).is_empty());
        assert!(manifest_reads(&format!("// {manifest}")).is_empty());
    }

    #[test]
    fn normalize_keeps_names_windows_resolves_differently_for_the_guard() {
        assert_eq!(normalize("docs/a.md").as_deref(), Some("docs/a.md"));
        assert_eq!(
            normalize("./docs/../README.md").as_deref(),
            Some("README.md")
        );
        assert_eq!(normalize("../x"), None);
        assert_eq!(normalize("docs/a.").as_deref(), Some("docs/a."));
        assert_eq!(windows_unsafe_name("docs/a.md"), None);
        assert_eq!(windows_unsafe_name("./docs/../a.md"), None);
        for path in ["...", "docs/.../a.md", "docs/a.", "docs/a "] {
            assert!(windows_unsafe_name(path).is_some(), "{path}");
        }
        assert_eq!(
            windows_unsafe_name("...").as_deref(),
            Some("names a path Windows resolves differently: ... (component \"...\")")
        );
    }

    /// The three scanners route their reads through one rule: a name Windows
    /// resolves differently is reported whichever way it is written.
    #[test]
    fn every_scanner_reports_a_name_windows_resolves_differently() {
        let include = format!("{}!(\"../../x.\");\n", "include_str");
        let root_join = format!("let p = {}().join(\"docs/.../a.md\");\n", "repo_root");
        let env = format!("env!(\"{}\")", "CARGO_MANIFEST_DIR");
        let manifest =
            format!("let q = Path::new({env}).join(\"..\").join(\"..\").join(\"skills \");\n");
        let text = format!("{include}{root_join}{manifest}");
        assert_eq!(
            windows_unsafe_reads("crates/a/src/lib.rs", &text),
            [
                "names a path Windows resolves differently: crates/x. (component \"x.\")",
                "names a path Windows resolves differently: docs/.../a.md (component \"...\")",
                "names a path Windows resolves differently: skills  (component \"skills \")",
            ]
        );
        assert!(windows_unsafe_reads("crates/a/src/lib.rs", "let p = 1;\n").is_empty());
    }

    #[test]
    fn code_mask_marks_only_code_outside_comments_and_literals() {
        let text = "a // b\n\"c\" d /* \u{e9} */ e";
        let code: String = text
            .char_indices()
            .filter(|(at, _)| code_mask(text)[*at])
            .map(|(_, c)| c)
            .collect();
        assert_eq!(code, "ade");
    }

    /// Listing the directory the caller asked for fails loudly; only the
    /// subdirectories under it are skipped when unreadable.
    #[test]
    fn walk_reports_a_directory_it_cannot_list() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let mut files = Vec::new();
        assert!(walk(&tmp.path().join("missing"), &mut files).is_err());
        let file = tmp.path().join("plain.rs");
        std::fs::write(&file, "").expect("write");
        assert!(walk(&file, &mut files).is_err());
        assert!(files.is_empty());
        walk(tmp.path(), &mut files).expect("list the temp dir");
        assert_eq!(files.len(), 1);
    }

    /// What the guard needs to classify one read.
    struct ReadContext<'a> {
        root: &'a Path,
        graph: &'a Graph,
        rust: &'a [String],
        inert: &'a [String],
    }

    /// The guard's problems with one read: `source`, in crate `reader`, reads
    /// the repo-relative `target`.
    fn target_problems(
        context: &ReadContext<'_>,
        reader: &str,
        source: &str,
        target: &str,
    ) -> Vec<String> {
        // A path that means something else on Windows is reported and not
        // looked up: on Windows `<root>\...` is the repo root.
        if let Some(problem) = windows_unsafe_name(target) {
            return vec![format!("{source} ({reader}) {problem}")];
        }
        if is_e2e_harness(source)
            || affected::owning_crate(target, &context.graph.crate_dirs) == Some(reader)
        {
            return Vec::new();
        }
        let mut problems = Vec::new();
        // The `rust`/`inert` classification needs real files: a directory
        // target stands for the files under it, and a literal that names
        // nothing in the tree has no file to classify. A directory that cannot
        // be listed is reported as such, not taken for an empty one. The table
        // check below runs for every literal.
        let target_path = context.root.join(target);
        let files = if target_path.is_dir() {
            let mut under = Vec::new();
            match walk(&target_path, &mut under) {
                Ok(()) => Some(under.iter().map(|p| relative(context.root, p)).collect()),
                Err(error) => {
                    problems.push(format!(
                        "{source} ({reader}) reads {target}, a directory that cannot be \
                         listed: {error}"
                    ));
                    None
                }
            }
        } else if target_path.exists() {
            Some(vec![target.to_string()])
        } else {
            None
        };
        if let Some(files) = files
            && let Some(gap) = ci_gap(&files, context.rust, context.inert)
        {
            problems.push(format!("{source} ({reader}) reads {target}: {gap}"));
        }
        let selection = affected::select_with_readers(&[target.to_string()], context.graph);
        let covered = match selection {
            Selection::Workspace => true,
            Selection::Packages(names) => names.contains(reader),
            Selection::Empty => false,
        };
        if !covered {
            problems.push(format!("{source} ({reader}) reads {target}"));
        }
        problems
    }

    fn one_crate_graph() -> Graph {
        Graph {
            crate_dirs: vec![("xtask".to_string(), "xtask".to_string())],
            rev_deps: std::iter::once(("xtask".to_string(), Vec::new())).collect(),
        }
    }

    /// A name Windows resolves differently is one problem and is not looked
    /// up: `<root>\...` would classify the repo root on Windows.
    #[test]
    fn a_windows_unsafe_target_is_reported_once_and_not_classified() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let graph = one_crate_graph();
        let context = ReadContext {
            root: tmp.path(),
            graph: &graph,
            rust: &[],
            inert: &[],
        };
        assert_eq!(
            target_problems(&context, "xtask", "xtask/src/a.rs", "..."),
            [
                "xtask/src/a.rs (xtask) names a path Windows resolves differently: ... \
              (component \"...\")"
            ]
        );
    }

    /// A target directory the guard cannot list is reported as such, not taken
    /// for an empty one. Running as root lists anything, so there the test has
    /// nothing to show and stops.
    #[cfg(unix)]
    #[test]
    fn a_target_directory_that_cannot_be_listed_is_a_problem() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().expect("temp dir");
        let data = tmp.path().join("data");
        std::fs::create_dir(&data).expect("mkdir");
        std::fs::write(data.join("a.txt"), "").expect("write");
        std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o000))
            .expect("remove access");
        let listable = std::fs::read_dir(&data).is_ok();
        let graph = one_crate_graph();
        let context = ReadContext {
            root: tmp.path(),
            graph: &graph,
            rust: &[],
            inert: &[],
        };
        let problems = target_problems(&context, "xtask", "xtask/src/a.rs", "data");
        std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o755))
            .expect("restore access");
        if listable {
            eprintln!("skipping: this user can list a directory with no permissions");
            return;
        }
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].starts_with(
                "xtask/src/a.rs (xtask) reads data, a directory that cannot be listed: "
            ),
            "{problems:?}"
        );
    }

    fn relative(root: &Path, path: &Path) -> String {
        path.strip_prefix(root)
            .expect("under the repo root")
            .to_string_lossy()
            .replace('\\', "/")
    }

    /// A raw-string include is a cross-crate read like any other; the scanner
    /// must not skip it.
    #[test]
    fn include_targets_reads_plain_and_raw_string_literals() {
        // Built with `format!` so this file never contains a literal embed that
        // `ascii_output_contract` would count as a data file compiled into a binary.
        let call = |name: &str, arg: &str| format!("{name}!({arg})");
        let target = "../../../README.md";
        let file = "crates/reader/src/lib.rs";
        for text in [
            call("include_str", &format!("\"{target}\"")),
            call("include_str", &format!("r\"{target}\"")),
            call("include_str", &format!("r#\"{target}\"#")),
            call("include_bytes", &format!("r##\"{target}\"##")),
        ] {
            assert_eq!(
                include_targets(file, &text),
                vec!["README.md".to_string()],
                "{text}"
            );
        }
        assert!(include_targets(file, &call("include_str", "concat!(\"a\", \"b\")")).is_empty());
    }

    /// A nested checkout (`.claude/worktrees/...`) holds another branch's
    /// sources; scanning it reports reads that are not this tree's.
    #[test]
    fn walk_skips_dot_directories_and_build_output() {
        let tmp = tempfile::tempdir().expect("temp dir");
        for file in [
            "crates/a/src/lib.rs",
            ".claude/worktrees/w/crates/b/src/lib.rs",
            ".git/hooks/x.rs",
            "target/debug/y.rs",
        ] {
            let path = tmp.path().join(file);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
            std::fs::write(&path, "").expect("write");
        }
        let mut files = Vec::new();
        walk(tmp.path(), &mut files).expect("list the temp dir");
        let found: Vec<String> = files.iter().map(|p| relative(tmp.path(), p)).collect();
        assert_eq!(found, vec!["crates/a/src/lib.rs".to_string()]);
    }

    /// A link that points back up the tree would be followed forever, and one
    /// that points out of it would scan files that are not this workspace's. Neither
    /// is descended into or listed.
    #[cfg(unix)]
    #[test]
    fn walk_does_not_follow_symbolic_links() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("elsewhere.rs"), "").expect("write");
        let a = tmp.path().join("crates/a");
        std::fs::create_dir_all(a.join("src")).expect("mkdir");
        std::fs::write(a.join("src/lib.rs"), "").expect("write");
        std::os::unix::fs::symlink(tmp.path(), a.join("loop")).expect("link up the tree");
        std::os::unix::fs::symlink(outside.path(), a.join("out")).expect("link out of the tree");
        std::os::unix::fs::symlink(a.join("src/lib.rs"), a.join("alias.rs")).expect("file link");
        let mut files = Vec::new();
        walk(tmp.path(), &mut files).expect("list the temp dir");
        let found: Vec<String> = files.iter().map(|p| relative(tmp.path(), p)).collect();
        assert_eq!(found, vec!["crates/a/src/lib.rs".to_string()]);
    }

    /// A folder the developer cannot read must not fail the scan (and so block
    /// a push); the files that can be read are still found. Running as root
    /// reads everything, so this only asserts that nothing panics and that the
    /// readable file is there.
    #[cfg(unix)]
    #[test]
    fn walk_skips_an_unreadable_directory_instead_of_panicking() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().expect("temp dir");
        for file in ["crates/a/src/lib.rs", "private/hidden.rs"] {
            let path = tmp.path().join(file);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
            std::fs::write(&path, "").expect("write");
        }
        let private = tmp.path().join("private");
        std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o000))
            .expect("remove access");
        let mut files = Vec::new();
        walk(tmp.path(), &mut files).expect("list the temp dir");
        // Restore access first so the temp dir can be removed even if an assert fails.
        std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o755))
            .expect("restore access");
        let found: Vec<String> = files.iter().map(|p| relative(tmp.path(), p)).collect();
        assert!(
            found.contains(&"crates/a/src/lib.rs".to_string()),
            "{found:?}"
        );
    }

    /// Fails when a crate reads a file outside its own directory that the shared
    /// selection ([`affected::select_with_readers`], used by CI's
    /// `cargo xtask affected` and by the pre-push hook) would not send to that
    /// crate's tests, when that file is not in the `rust` path category of
    /// `.github/actions/classify-changes/action.yml` (or is `inert`), so CI's
    /// `test` job would skip, or when a crate derives the repo root without
    /// being declared as a reader. Add the path to [`CROSS_CRATE_READS`] (or
    /// declare the crate), or to the `rust` category, when this fails.
    ///
    /// What it can see is limited, for both checks. A path held in a const
    /// (`SHARED_PREFLIGHT` in `workflow_contract.rs`) is not seen at all. An
    /// `include_str!`/`include_bytes!` literal, a
    /// `env!("CARGO_MANIFEST_DIR")` chain that ends in literal `.join("...")`
    /// segments, and a `repo_root().join("...")` call with a literal path name
    /// their file, and that file is checked. A chain that only derives the repo
    /// root and is joined somewhere else (a helper such as `repo_root()` whose
    /// result is stored and joined later, or joined with a computed path) is
    /// recorded as a root derivation: the guard then requires the crate to be
    /// declared as a reader in [`CROSS_CRATE_READS`], but not that the files it
    /// goes on to read are covered. A test that reads a new kind of file (a
    /// `Cargo.toml` or a `.feature` of another crate) that way is selected only
    /// as that file's own crate, so its entry in [`CROSS_CRATE_READS`] has to be
    /// added by hand. Widening the rule would need data flow through helper
    /// functions, which a text scan cannot give exactly.
    #[test]
    fn every_cross_crate_read_selects_its_reader() {
        let root = repo_root();
        let mut files = Vec::new();
        walk(&root, &mut files).expect("list the repo root");
        // The workspace members, from the same `cargo metadata` the hook selects
        // with, so a stray `Cargo.toml` (a fixture, a nested checkout) is not
        // taken for a workspace crate.
        let crate_dirs = affected::load_graph()
            .expect("read the workspace graph")
            .crate_dirs;
        assert!(
            crate_dirs.iter().any(|(name, _)| name == "xtask"),
            "found no crates under {}",
            root.display()
        );
        let graph = Graph {
            rev_deps: crate_dirs
                .iter()
                .map(|(name, _)| (name.clone(), Vec::new()))
                .collect(),
            crate_dirs,
        };

        // CI's `changes` job decides whether the reader's tests run at all:
        // the `test` job is gated on the `rust` category, so a file a test reads
        // must be in it, and must not be `inert`.
        let filters = ci_filters::paths_filters(&ci_filters::read_classify_action());
        let rust = ci_filters::paths_filter(&filters, "rust");
        let inert = ci_filters::paths_filter(&filters, "inert");

        let mut problems = Vec::new();
        let mut root_readers = BTreeSet::new();
        let mut checked = 0usize;
        let mut root_joins = 0usize;
        let context = ReadContext {
            root: &root,
            graph: &graph,
            rust,
            inert,
        };
        let mut check = |reader: &str, source: &str, target: &str| {
            checked += 1;
            problems.extend(target_problems(&context, reader, source, target));
        };

        for path in files
            .iter()
            .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        {
            let source = relative(&root, path);
            let Some(reader) = affected::owning_crate(&source, &graph.crate_dirs) else {
                continue;
            };
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            for target in include_targets(&source, &text) {
                check(reader, &source, &target);
            }
            for target in root_join_targets(&text) {
                root_joins += 1;
                check(reader, &source, &target);
            }
            for read in manifest_reads(&text) {
                match read {
                    Read::Root if is_e2e_harness(&source) => {}
                    Read::Root => {
                        root_readers.insert(reader.to_string());
                    }
                    Read::Path(target) => check(reader, &source, &target),
                }
            }
        }

        // A scanner that finds nothing would pass vacuously. xtask's tests derive
        // the repo root, and crates include files from outside themselves.
        assert!(
            root_readers.contains("xtask"),
            "the manifest scanner found nothing"
        );
        assert!(checked > 0, "the include scanner found nothing");
        assert!(root_joins > 0, "the repo_root() scanner found nothing");

        let declared: BTreeSet<&str> = CROSS_CRATE_READS
            .iter()
            .flat_map(|(_, readers)| readers.iter().copied())
            .collect();
        for reader in root_readers {
            if !declared.contains(reader.as_str()) {
                problems.push(format!(
                    "crate {reader} derives the repo root but is not a reader in CROSS_CRATE_READS"
                ));
            }
        }
        assert!(
            problems.is_empty(),
            "reads whose change would not select the reader's tests:\n{}",
            problems.join("\n")
        );
    }

    #[test]
    fn the_scanner_finds_includes_that_leave_the_crate() {
        let include = |name: &str, path: &str| format!("let a = {name}!(\"{path}\");");
        let text = format!(
            "{}\n{}\n{}",
            include("include_str", "../../../tests/e2e-cucumber/fixtures/a.txt"),
            include("include_bytes", "../data.bin"),
            include("include_str", "../../../../../escapes.txt"),
        );
        assert_eq!(
            include_targets("crates/rocm-core/src/fix.rs", &text),
            [
                "tests/e2e-cucumber/fixtures/a.txt",
                "crates/rocm-core/data.bin"
            ]
        );
    }

    #[test]
    fn the_scanner_follows_a_manifest_relative_chain_out_of_the_crate() {
        let manifest = format!("env!(\"{}\")", "CARGO_MANIFEST_DIR");
        let chain = |tail: &str| format!("fn p() -> PathBuf {{ Path::new({manifest}){tail} }}");
        assert_eq!(
            manifest_reads(&chain(
                ".join(\"..\").join(\"..\").join(\"skills\").join(\"d\").join(\"f.md\")"
            )),
            [Read::Path("skills/d/f.md".to_string())]
        );
        assert_eq!(
            manifest_reads(&chain(".join(\"../..\").join(\"docs\")")),
            [Read::Path("docs".to_string())]
        );
        assert_eq!(
            manifest_reads(&chain(".parent().expect(\"root\")")),
            [Read::Root]
        );
        assert_eq!(
            manifest_reads(&chain(".ancestors().nth(2).expect(\"root\")")),
            [Read::Root]
        );
    }

    #[test]
    fn the_scanner_ignores_scratch_dirs_and_reads_inside_the_crate() {
        let manifest = format!("env!(\"{}\")", "CARGO_MANIFEST_DIR");
        let chain = |tail: &str| format!("fn p() -> PathBuf {{ PathBuf::from({manifest}){tail} }}");
        assert!(
            manifest_reads(&chain(
                ".join(\"..\").join(\"..\").join(\".rocm-work\").join(\"t\")"
            ))
            .is_empty()
        );
        assert!(manifest_reads(&chain(".join(\"target\").join(\"x\")")).is_empty());
        assert!(manifest_reads(&chain(".join(\"features\")")).is_empty());
    }

    #[test]
    fn the_scanner_reads_the_literal_a_repo_root_helper_joins() {
        let call = |tail: &str| format!("let p = {}(){tail};", "repo_root");
        let text = [
            call(".join(\"docs/a.md\")"),
            call("?\n            .join(r\"skills/b/SKILL.md\")"),
            call(".join(name)"),
            call(".join(\"..\")"),
            call(".parent()"),
        ]
        .join("\n");
        assert_eq!(root_join_targets(&text), ["docs/a.md", "skills/b/SKILL.md"]);
    }

    #[test]
    fn the_scanner_ignores_a_join_argument_that_is_not_a_plain_literal() {
        let manifest = format!("env!(\"{}\")", "CARGO_MANIFEST_DIR");
        let text = format!(
            "let p = PathBuf::from({manifest}).parent().unwrap().join(format!(\"x-{{}}\", id));"
        );
        assert_eq!(manifest_reads(&text), [Read::Root]);
    }
}
