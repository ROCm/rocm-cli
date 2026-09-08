// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Assert every relative documentation path cited in a Rust doc comment
//! resolves to a file in the tree.
//!
//! Doc comments accumulated citations of markdown files that had never existed
//! (a whole `../wiki/…` tree of them). Nothing pointed the reader anywhere:
//! `rustdoc` does not resolve a bare path in prose, so a dangling citation is
//! invisible to the compiler, to clippy, and to review. This check makes the
//! next one fail CI instead.
//!
//! Design notes:
//!
//! - **Doc lines only.** Only lines whose trimmed text starts with `//!` or
//!   `///` are scanned. Ordinary code mentions `.md` files legitimately — a
//!   `format!("{dist}/README.md")` naming an artifact that exists only after a
//!   build, a shell command embedded in a workflow-contract test — and those are
//!   not citations to resolve.
//! - **No backtick requirement.** Citations in this tree appear both quoted and
//!   bare (`docs/ux-guidelines.md` in `app/summary.rs` is bare), so requiring
//!   backticks would miss the majority of them.
//! - **Two resolution roots.** A candidate passes if it resolves next to the
//!   citing file (the `./` and `../` forms) *or* against the workspace root (the
//!   `docs/…` and bare `README.md`/`AGENTS.md` forms). Both are how the tree
//!   actually cites files, and accepting either keeps the check free of
//!   false positives without weakening it: a name that resolves under neither
//!   root points at nothing.
//! - **Every violation, once.** Failures are collected and reported together.
//!   Reporting only the first would cost one CI round trip per dangling path.
//!
//! Matching is hand-rolled rather than regex-based to keep xtask's dependency
//! set as-is; the pure helpers take their inputs as parameters so they are
//! unit-testable without touching process-global state.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Characters that may appear in a cited path.
///
/// Deliberately excludes the placeholder delimiters (`{`, `<`) and quoting
/// characters (backtick, parenthesis, comma), so those terminate a candidate
/// rather than becoming part of it.
const fn is_path_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '@' | '/' | '-')
}

/// Characters that mark a candidate as a placeholder rather than a real path,
/// when they sit immediately either side of it (`<name>.md`, `{dist}/x.md`).
const fn is_placeholder_delimiter(c: char) -> bool {
    matches!(c, '{' | '}' | '<' | '>')
}

/// Drop whitespace-separated tokens that contain a URL scheme separator, so an
/// external link ending in `.md` is never mistaken for a path in the tree.
fn strip_urls(text: &str) -> String {
    text.split_whitespace()
        .filter(|token| !token.contains("://"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Candidate `.md` paths in one doc-comment line's text.
fn candidates_in_line(text: &str) -> Vec<String> {
    let text = strip_urls(text);
    let chars: Vec<char> = text.chars().collect();
    let mut found = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        if !is_path_char(chars[start]) {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < chars.len() && is_path_char(chars[end]) {
            end += 1;
        }
        let run: String = chars[start..end].iter().collect();
        // Trailing sentence punctuation is a path character, so `docs/x.md.`
        // would otherwise not look like a citation.
        let candidate = run.trim_end_matches('.');
        let before = start.checked_sub(1).map(|i| chars[i]);
        let after = chars.get(end).copied();
        let placeholder = before.is_some_and(is_placeholder_delimiter)
            || after.is_some_and(is_placeholder_delimiter);
        // A bare `.md` is the tail of a split placeholder like `{name}.md`
        // rather than a path, so the stem must be non-empty; an absolute path is
        // not a repo-relative citation. The suffix match is deliberately
        // case-sensitive — every markdown file in this tree is `.md`.
        let named = candidate
            .strip_suffix(".md")
            .is_some_and(|stem| !stem.is_empty());
        if named && !candidate.starts_with('/') && !placeholder {
            found.push(candidate.to_owned());
        }
        start = end;
    }
    found
}

/// Every `.md` path cited in a doc comment in `source`, as `(line number, path)`
/// with 1-based line numbers. Non-doc lines are ignored, and a path cited twice
/// on one line is reported once.
pub fn extract_citations(source: &str) -> Vec<(usize, String)> {
    let mut citations: Vec<(usize, String)> = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        let Some(text) = trimmed
            .strip_prefix("//!")
            .or_else(|| trimmed.strip_prefix("///"))
        else {
            continue;
        };
        let line_number = index + 1;
        for candidate in candidates_in_line(text) {
            if !citations.iter().any(|(existing_line, existing)| {
                *existing_line == line_number && *existing == candidate
            }) {
                citations.push((line_number, candidate));
            }
        }
    }
    citations
}

/// Whether `candidate` names a file, resolved either next to the citing file or
/// against the workspace root.
pub fn resolve(candidate: &str, file_dir: &Path, root: &Path) -> bool {
    file_dir.join(candidate).is_file() || root.join(candidate).is_file()
}

/// All tracked `*.rs` files, excluding `third_party/`, as paths relative to `root`.
fn discover_rust_files(root: &Path) -> Result<Vec<PathBuf>> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["ls-files", "*.rs"])
        .output()
        .context("failed to run `git ls-files`")?;
    if !output.status.success() {
        bail!(
            "`git ls-files` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("third_party/"))
        .map(PathBuf::from)
        .collect())
}

/// Verify every documentation path cited in a Rust doc comment resolves.
pub fn run() -> Result<()> {
    let root = crate::paths::workspace_root()?;
    let files = discover_rust_files(&root)?;

    let mut violations = Vec::new();
    let mut checked = 0usize;
    for file in &files {
        let path = root.join(file);
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", file.display()))?;
        let file_dir = path
            .parent()
            .map_or_else(|| root.clone(), Path::to_path_buf);
        for (line, candidate) in extract_citations(&source) {
            checked += 1;
            if !resolve(&candidate, &file_dir, &root) {
                violations.push(format!(
                    "{}:{line}: doc comment cites `{candidate}`, which does not resolve to a \
                     file in the tree",
                    file.display()
                ));
            }
        }
    }

    if !violations.is_empty() {
        bail!(
            "{} dangling documentation path(s) cited in doc comments:\n{}",
            violations.len(),
            violations.join("\n")
        );
    }
    println!(
        "doc paths: {checked} citation(s) across {} file(s) resolve.",
        files.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_from_doc_lines_only() {
        let source = "//! Module doc, see docs/a.md for details.\n\
                      /// Item doc, see `docs/b.md`.\n\
                      // Ordinary comment about docs/c.md.\n\
                      let path = docs_d_md;\n";
        assert_eq!(
            extract_citations(source),
            vec![(1, "docs/a.md".to_owned()), (2, "docs/b.md".to_owned())]
        );
    }

    #[test]
    fn ignores_md_in_a_string_literal_on_a_code_line() {
        // The shape that made a naive whole-file scan unusable: real code names
        // build artifacts and shell commands that are not citations.
        let source = "let readme = format!(\"{dist}/README.md\");\n\
                      assert!(jobs.contains(\"base64 -w0 regenerated/MANIFEST.md > manifest.b64\"));\n\
                      let inputs = [\"docs/guide.md\".to_string()];\n";
        assert!(extract_citations(source).is_empty());
    }

    #[test]
    fn ignores_urls_ending_in_md() {
        let source = "//! See https://example.com/docs/remote.md for the upstream note.\n\
                      /// Mirrored at <http://example.org/a/b.md>.\n";
        assert!(extract_citations(source).is_empty());
    }

    #[test]
    fn ignores_placeholder_and_absolute_paths() {
        let source = "//! Writes `<name>.md` under the bundle, or /etc/motd.md on request.\n";
        assert!(extract_citations(source).is_empty());
    }

    #[test]
    fn strips_trailing_sentence_punctuation() {
        let source = "//! Refer to docs/ux-guidelines.md.\n";
        assert_eq!(
            extract_citations(source),
            vec![(1, "docs/ux-guidelines.md".to_owned())]
        );
    }

    #[test]
    fn reports_a_repeated_citation_on_one_line_once() {
        let source = "//! docs/a.md and again docs/a.md, plus docs/b.md.\n";
        assert_eq!(
            extract_citations(source),
            vec![(1, "docs/a.md".to_owned()), (1, "docs/b.md".to_owned())]
        );
    }

    #[test]
    fn resolves_relative_to_the_citing_file_and_to_the_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path();
        std::fs::create_dir_all(root.join("docs")).expect("docs dir");
        std::fs::create_dir_all(root.join("crates/thing/src")).expect("src dir");
        std::fs::write(root.join("README.md"), "readme").expect("readme");
        std::fs::write(root.join("docs/x.md"), "x").expect("docs file");
        std::fs::write(root.join("crates/thing/sibling.md"), "sibling").expect("sibling");
        let file_dir = root.join("crates/thing/src");

        assert!(resolve("../sibling.md", &file_dir, root));
        assert!(resolve("docs/x.md", &file_dir, root));
        assert!(resolve("README.md", &file_dir, root));
        // A directory is not a file, and neither form of a missing path resolves.
        assert!(!resolve("docs", &file_dir, root));
        assert!(!resolve("../wiki/foo.md", &file_dir, root));
        assert!(!resolve("foo.md", &file_dir, root));
    }

    #[test]
    fn flags_every_unresolvable_citation_not_just_the_first() {
        // The regression this check exists for: a `../wiki/…` tree that never
        // existed, plus a bare filename that resolves against neither root.
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path();
        std::fs::create_dir_all(root.join("crates/thing/src")).expect("src dir");
        std::fs::write(root.join("README.md"), "readme").expect("readme");
        let file_dir = root.join("crates/thing/src");

        let source = "//! See `../wiki/concepts/metric-registry.md`.\n\
                      //! And `rocm-cli-unification-working-agreements.md`.\n\
                      //! And README.md, which is fine.\n";
        let unresolved: Vec<_> = extract_citations(source)
            .into_iter()
            .filter(|(_, candidate)| !resolve(candidate, &file_dir, root))
            .collect();
        assert_eq!(
            unresolved,
            vec![
                (1, "../wiki/concepts/metric-registry.md".to_owned()),
                (2, "rocm-cli-unification-working-agreements.md".to_owned()),
            ]
        );
    }
}
