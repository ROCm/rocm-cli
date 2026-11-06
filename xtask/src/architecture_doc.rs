// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Require every production source file to appear in `docs/architecture.md`.
//!
//! `docs/architecture.md` is the contributor-facing module map and is meant to
//! be "updated in the same PR as the code it documents". Nothing enforced that
//! until now. The `lychee` link check (the `docs-links` CI job and the `lychee`
//! prek hook) runs the other way round: it confirms that every link already in
//! the doc still resolves, so a moved or deleted file is caught, but a file the
//! doc never mentions has no link to check and passes silently.
//!
//! This check starts from the source tree instead. Every tracked `.rs` file
//! under `apps/*/src`, `crates/*/src` and `engines/*/src` must be cited by a
//! relative markdown link in `docs/architecture.md` (a complete inline link,
//! `[text](path)`, with its text on one line, that renders as a link: a path
//! inside an HTML comment, a code block or a code span does not count, and
//! neither does a link missing its `[` or `)`, one with an escaped bracket, an
//! image, or the outer brackets of a link that contains another link),
//! or be listed in [`EXEMPT`], or be a test-only helper (see
//! [`is_test_helper`]). Out of scope: `build.rs` files at a crate's root and the
//! `tests/` and `examples/` trees, which sit outside `<crate>/src`.
//!
//! It reads the git index and the working tree, never a git diff, so the answer
//! does not depend on a base ref. That is the same answer in CI and in the
//! pre-commit hook, where prek stashes unstaged changes first. The pre-push hook
//! runs it too, but it reads the working tree, not the commits being pushed:
//! with a clean tree that is the same thing, and with edits that are staged or
//! unstaged it can pass where CI fails. CI is the gate; the hook is feedback.
//!
//! The doc must use inline `[text](path)` links. Reference-style links, a
//! `?query` suffix and a root-relative path render as working links but are not
//! recognised, so the failure message asks for the inline form.
//!
//! [`EXEMPT`] is the files that existed, uncited, when the check was added. It
//! follows the pattern of `ALLOWLIST` in [`crate::crate_edges`] and is checked
//! in both directions: a source file in neither the doc nor the list fails,
//! and so does a list entry that is now cited or no longer exists. The list is
//! meant to shrink as files are documented. Nothing can stop a determined edit
//! from adding to it, so [`EXEMPT_CEILING`] pins its size in a test: growing
//! the list means raising that constant too, a second, visible line in the diff
//! that a reviewer has to accept.
//!
//! What this does not check: whether an entry's prose still describes the file,
//! and functions that move between two files that are both already cited.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::paths;

/// The module map, relative to the workspace root.
const DOC: &str = "docs/architecture.md";

/// The directory `DOC` lives in. Its links are relative to this.
const DOC_DIR: &str = "docs";

/// Source roots whose `<crate>/src` trees are in scope.
const SOURCE_ROOTS: [&str; 3] = ["apps", "crates", "engines"];

/// Production files that have no entry in `docs/architecture.md` yet: everything
/// that was uncited when this check was added, mostly the dashboard crates.
/// Do not add to this list to make the check pass: document the file in the doc
/// instead. Growing it is a reviewed exception and also needs
/// [`EXEMPT_CEILING`] raised. Remove an entry when the file gets one. Keep it
/// sorted.
const EXEMPT: &[&str] = &[
    "apps/rocmd/src/main.rs",
    "crates/rocm-dash-collectors/src/amd_smi.rs",
    "crates/rocm-dash-collectors/src/bench_load.rs",
    "crates/rocm-dash-collectors/src/bench_tail.rs",
    "crates/rocm-dash-collectors/src/cgroup.rs",
    "crates/rocm-dash-collectors/src/docker.rs",
    "crates/rocm-dash-collectors/src/engine_registry.rs",
    "crates/rocm-dash-collectors/src/host.rs",
    "crates/rocm-dash-collectors/src/lemonade.rs",
    "crates/rocm-dash-collectors/src/parallel.rs",
    "crates/rocm-dash-collectors/src/proc_scan.rs",
    "crates/rocm-dash-collectors/src/sysfs.rs",
    "crates/rocm-dash-collectors/src/vllm_log.rs",
    "crates/rocm-dash-collectors/src/vllm_prom.rs",
    "crates/rocm-dash-core/src/bench_rollup.rs",
    "crates/rocm-dash-core/src/bench_schema.rs",
    "crates/rocm-dash-core/src/config.rs",
    "crates/rocm-dash-core/src/efficiency.rs",
    "crates/rocm-dash-core/src/metrics.rs",
    "crates/rocm-dash-core/src/observation.rs",
    "crates/rocm-dash-core/src/partition.rs",
    "crates/rocm-dash-core/src/persist.rs",
    "crates/rocm-dash-core/src/protocol.rs",
    "crates/rocm-dash-core/src/state.rs",
    "crates/rocm-dash-core/src/traits.rs",
    "crates/rocm-dash-core/src/vram.rs",
    "crates/rocm-dash-daemon/src/bench_ring.rs",
    "crates/rocm-dash-daemon/src/demo.rs",
    "crates/rocm-dash-daemon/src/persist.rs",
    "crates/rocm-dash-daemon/src/snapshot_ring.rs",
    "crates/rocm-dash-daemon/src/transport.rs",
    "crates/rocm-dash-tui/src/client.rs",
    "crates/rocm-dash-tui/src/jobs.rs",
    "crates/rocm-dash-tui/src/lib.rs",
    "crates/rocm-dash-tui/src/llm.rs",
    "crates/rocm-dash-tui/src/reconnect.rs",
    "crates/rocm-dash-tui/src/registry_probe.rs",
    "crates/rocm-dash-tui/src/replay.rs",
    "crates/rocm-dash-tui/src/tool_exec.rs",
    "crates/rocm-dash-tui/src/transport.rs",
    "crates/rocm-dash-tui/src/ui/automations_manager.rs",
    "crates/rocm-dash-tui/src/ui/bench.rs",
    "crates/rocm-dash-tui/src/ui/bench_run.rs",
    "crates/rocm-dash-tui/src/ui/command_screen.rs",
    "crates/rocm-dash-tui/src/ui/config_manager.rs",
    "crates/rocm-dash-tui/src/ui/core_bars.rs",
    "crates/rocm-dash-tui/src/ui/dock.rs",
    "crates/rocm-dash-tui/src/ui/engine_manager.rs",
    "crates/rocm-dash-tui/src/ui/examine_manager.rs",
    "crates/rocm-dash-tui/src/ui/folder_browser.rs",
    "crates/rocm-dash-tui/src/ui/format.rs",
    "crates/rocm-dash-tui/src/ui/gradient.rs",
    "crates/rocm-dash-tui/src/ui/install_manager.rs",
    "crates/rocm-dash-tui/src/ui/instance_list.rs",
    "crates/rocm-dash-tui/src/ui/launcher.rs",
    "crates/rocm-dash-tui/src/ui/logs_view.rs",
    "crates/rocm-dash-tui/src/ui/model_picker.rs",
    "crates/rocm-dash-tui/src/ui/onboarding.rs",
    "crates/rocm-dash-tui/src/ui/runtime_manager.rs",
    "crates/rocm-dash-tui/src/ui/serve_wizard.rs",
    "crates/rocm-dash-tui/src/ui/services_manager.rs",
    "crates/rocm-dash-tui/src/ui/sparkline.rs",
    "crates/rocm-dash-tui/src/ui/tabs/bench.rs",
    "crates/rocm-dash-tui/src/ui/tabs/chat.rs",
    "crates/rocm-dash-tui/src/ui/tabs/hardware.rs",
    "crates/rocm-dash-tui/src/ui/tabs/home.rs",
    "crates/rocm-dash-tui/src/ui/tabs/mod.rs",
    "crates/rocm-dash-tui/src/ui/tabs/observe.rs",
    "crates/rocm-dash-tui/src/ui/tabs/pane.rs",
    "crates/rocm-dash-tui/src/ui/tabs/rocm.rs",
    "crates/rocm-dash-tui/src/ui/tabs/serving.rs",
    "crates/rocm-dash-tui/src/ui/update_manager.rs",
    "crates/rocm-deps/src/lib.rs",
    "engines/aim/src/main.rs",
    "engines/lemonade/src/main.rs",
    "engines/vllm/src/main.rs",
];

/// The most entries [`EXEMPT`] may hold. Lower it when entries are removed; do
/// not raise it to make the check pass, since a file with no entry in the module
/// map should get one. Raising it is the visible, reviewed way to grow the list.
const EXEMPT_CEILING: usize = 76;

/// Whether `path` is a tracked-file path this check covers: a `.rs` file under
/// `<root>/<crate>/src/` for one of the [`SOURCE_ROOTS`].
#[allow(clippy::case_sensitive_file_extension_comparisons)] // Rust sources are always lowercase `.rs`
fn in_scope(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    parts.len() >= 4
        && SOURCE_ROOTS.contains(&parts[0])
        && parts[2] == "src"
        && path.ends_with(".rs")
}

/// Whether `path` is a test-only helper that does not need an architecture
/// entry: shared test support, or a property-test module. Matched on the file
/// stem so a new helper of the same kind never needs an [`EXEMPT`] entry.
fn is_test_helper(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = name.strip_suffix(".rs").unwrap_or(name);
    matches!(stem, "test_support" | "test_env" | "proptests")
        || stem.ends_with("_proptests")
        || stem.ends_with("_props")
        || stem.ends_with("_properties")
}

/// Resolve one markdown link target found in [`DOC`] to a repo-relative `.rs`
/// path, or `None` if it is not a link to a source file inside the repo.
#[allow(clippy::case_sensitive_file_extension_comparisons)] // Rust sources are always lowercase `.rs`
fn resolve_link(target: &str) -> Option<String> {
    let target = target.split('#').next()?;
    if target.contains("://") || !target.ends_with(".rs") {
        return None;
    }
    let mut parts: Vec<&str> = vec![DOC_DIR];
    for segment in target.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                // Climbing out of the repo root names nothing in the tree.
                parts.pop()?;
            }
            other => parts.push(other),
        }
    }
    Some(parts.join("/"))
}

/// The first line of `text` without its terminator, and the byte length of that
/// line including the terminator.
fn next_line(text: &str) -> (&str, usize) {
    match text.find('\n') {
        Some(end) => (&text[..end], end + 1),
        None => (text, text.len()),
    }
}

/// A code-fence marker at the start of `line`: its character and run length.
fn fence_marker(line: &str) -> Option<(u8, usize)> {
    let line = line.trim_start_matches([' ', '\t']);
    let first = *line.as_bytes().first()?;
    if first != b'`' && first != b'~' {
        return None;
    }
    let run = line.bytes().take_while(|&byte| byte == first).count();
    // A backtick fence's info string cannot contain a backtick: that line is an
    // inline code span, not a fence.
    if run < 3 || (first == b'`' && line[run..].contains('`')) {
        return None;
    }
    Some((first, run))
}

/// Whether `line` closes a fence opened with `open`.
fn closes_fence(line: &str, open: (u8, usize)) -> bool {
    fence_marker(line).is_some_and(|(character, run)| {
        character == open.0 && run >= open.1 && line.trim().bytes().all(|byte| byte == open.0)
    })
}

/// The byte length of the code span that opens at the start of `text` (which
/// starts with a backtick run), or `None` if no run of the same length closes
/// it. Like a CommonMark code span it cannot cross a blank line, so a stray
/// backtick in one paragraph does not pair with one in a later paragraph.
fn code_span_len(text: &str) -> Option<usize> {
    let window = &text[..text.find("\n\n").unwrap_or(text.len())];
    let open = window.bytes().take_while(|&byte| byte == b'`').count();
    let mut at = open;
    while let Some(offset) = window[at..].find('`') {
        let start = at + offset;
        let run = window[start..]
            .bytes()
            .take_while(|&byte| byte == b'`')
            .count();
        if run == open {
            return Some(start + run);
        }
        at = start + run;
    }
    None
}

/// `doc` with everything that does not render as a link removed: HTML comments,
/// fenced code blocks and inline code spans. A link inside any of them is not
/// part of the rendered module map, so it must not count as a citation. Each
/// removed span leaves a space or newline behind so link text and line
/// structure stay in place.
///
/// It scans once, left to right, so the usual precedence holds: a backtick
/// inside a comment is part of the comment, and `<!--` inside a code span is
/// code. A backslash-escaped backtick or `<` opens nothing. Where it cannot tell
/// (an unclosed backtick), it errs towards hiding a link, which can only make
/// the check fail, never pass. Indented code blocks are not recognised.
fn link_source_text(doc: &str) -> String {
    let doc = doc.replace("\r\n", "\n");
    let mut out = String::with_capacity(doc.len());
    let mut at = 0;
    while at < doc.len() {
        let rest = &doc[at..];
        let line_start = at == 0 || doc.as_bytes()[at - 1] == b'\n';
        if line_start {
            let (line, mut skipped) = next_line(rest);
            if let Some(open) = fence_marker(line) {
                while skipped < rest.len() {
                    let (inner, length) = next_line(&rest[skipped..]);
                    skipped += length;
                    if closes_fence(inner, open) {
                        break;
                    }
                }
                at += skipped;
                out.push('\n');
                continue;
            }
        }
        if escapes_next(doc.as_bytes(), at) {
            // An escaped backtick or `<` opens nothing. Keep both characters so
            // `link_destinations` still sees the escape.
            out.push_str(&rest[..2]);
            at += 2;
            continue;
        }
        if rest.starts_with("<!--") {
            at += rest.find("-->").map_or(rest.len(), |end| end + 3);
            out.push(' ');
            continue;
        }
        if rest.starts_with('`') {
            if let Some(length) = code_span_len(rest) {
                at += length;
                out.push(' ');
            } else {
                let run = rest.bytes().take_while(|&byte| byte == b'`').count();
                out.push_str(&rest[..run]);
                at += run;
            }
            continue;
        }
        let Some(character) = rest.chars().next() else {
            break;
        };
        out.push(character);
        at += character.len_utf8();
    }
    out
}

/// Whether the byte at `at` is a backslash that escapes the ASCII punctuation
/// character after it.
fn escapes_next(bytes: &[u8], at: usize) -> bool {
    bytes[at] == b'\\' && bytes.get(at + 1).is_some_and(u8::is_ascii_punctuation)
}

/// `text` with the backslash removed from every escaped punctuation character.
fn unescape(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while at < bytes.len() {
        if escapes_next(bytes, at) {
            at += 1;
        }
        // Every byte skipped above or copied here is ASCII, so `at` stays on a
        // character boundary.
        let Some(character) = text[at..].chars().next() else {
            break;
        };
        out.push(character);
        at += character.len_utf8();
    }
    out
}

/// Advance `at` past spaces, tabs and newlines.
fn skip_space(bytes: &[u8], mut at: usize) -> usize {
    while matches!(bytes.get(at), Some(b' ' | b'\t' | b'\n')) {
        at += 1;
    }
    at
}

/// Parse the rest of an inline link, `text` being everything after its `(`:
/// `destination`, an optional title, and the closing `)`. Returns the
/// destination (its `<...>` delimiters and escapes removed) and the byte length
/// consumed through the `)`, or `None` if it is not a complete inline link.
fn inline_destination(text: &str) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    let mut at = skip_space(bytes, 0);
    let destination;
    if bytes.get(at) == Some(&b'<') {
        let start = at + 1;
        at = start;
        loop {
            match *bytes.get(at)? {
                b'>' => break,
                b'<' | b'\n' => return None,
                _ if escapes_next(bytes, at) => at += 2,
                _ => at += 1,
            }
        }
        destination = unescape(&text[start..at]);
        at += 1;
    } else {
        let start = at;
        let mut depth = 0usize;
        loop {
            match bytes.get(at) {
                None => return None,
                Some(b' ' | b'\t' | b'\n') => break,
                Some(b'(') => {
                    depth += 1;
                    at += 1;
                }
                Some(b')') if depth == 0 => break,
                Some(b')') => {
                    depth -= 1;
                    at += 1;
                }
                Some(_) if escapes_next(bytes, at) => at += 2,
                Some(_) => at += 1,
            }
        }
        destination = unescape(&text[start..at]);
    }
    at = skip_space(bytes, at);
    if let Some(&opener @ (b'"' | b'\'' | b'(')) = bytes.get(at) {
        let closer = if opener == b'(' { b')' } else { opener };
        at += 1;
        loop {
            match *bytes.get(at)? {
                byte if byte == closer => break,
                _ if escapes_next(bytes, at) => at += 2,
                _ => at += 1,
            }
        }
        at = skip_space(bytes, at + 1);
    }
    (bytes.get(at) == Some(&b')')).then_some((destination, at + 1))
}

/// An unmatched `[` while scanning for links.
struct Opener {
    /// Preceded by an unescaped `!`: an image, whose destination is not a link.
    image: bool,
    /// Cleared once a link has formed inside this one: links do not nest, so the
    /// outer brackets then render as plain text.
    active: bool,
}

/// The destination of every complete inline link, `[text](destination)`, in
/// `text`. A bracket or parenthesis escaped with a backslash does not count,
/// the `[` must be on the same line as its `]`, and a link missing its `[`, its
/// `)` or its destination is not a link: the rendered page shows it as plain
/// text, so it must not count as a citation either. The same goes for an image,
/// `![alt](destination)`, and for the outer brackets of a link that contains
/// another link; an image inside a link's text is fine.
fn link_destinations(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    let mut openers: Vec<Opener> = Vec::new();
    let mut last_bang = None;
    let mut at = 0;
    while at < bytes.len() {
        if escapes_next(bytes, at) {
            at += 2;
            continue;
        }
        match bytes[at] {
            b'\n' => openers.clear(),
            b'!' => last_bang = Some(at),
            b'[' => openers.push(Opener {
                image: at > 0 && last_bang == Some(at - 1),
                active: true,
            }),
            b']' => {
                if let Some(opener) = openers.pop()
                    && opener.active
                    && bytes.get(at + 1) == Some(&b'(')
                    && let Some((destination, length)) = inline_destination(&text[at + 2..])
                {
                    if !opener.image {
                        found.push(destination);
                        for outer in openers.iter_mut().filter(|outer| !outer.image) {
                            outer.active = false;
                        }
                    }
                    at += 2 + length;
                    continue;
                }
            }
            _ => {}
        }
        at += 1;
    }
    found
}

/// Every `.rs` file [`DOC`] links to, as repo-relative paths. Only a complete
/// inline link counts, and not one inside an HTML comment, a code block or a
/// code span.
fn cited_paths(doc: &str) -> BTreeSet<String> {
    link_destinations(&link_source_text(doc))
        .iter()
        .filter_map(|destination| resolve_link(destination))
        .collect()
}

/// Check the tracked files against the doc's citations and `exempt`, returning
/// every violation found.
fn check(tracked: &[String], cited: &BTreeSet<String>, exempt: &[&str]) -> Vec<String> {
    let in_tree: BTreeSet<&str> = tracked
        .iter()
        .map(String::as_str)
        .filter(|path| in_scope(path))
        .collect();

    let mut violations = Vec::new();

    for path in &in_tree {
        if !cited.contains(*path) && !exempt.contains(path) && !is_test_helper(path) {
            violations.push(format!(
                "`{path}` is not cited in {DOC}: add an entry for it to the module map, as an \
                 inline `[text](path)` markdown link relative to docs/ (reference-style links, \
                 `?query` suffixes and root-relative paths are not recognised)"
            ));
        }
    }

    for entry in exempt {
        if !in_tree.contains(entry) {
            violations.push(format!(
                "EXEMPT entry `{entry}` is stale: no tracked source file has that path any more; \
                 remove it from EXEMPT in xtask/src/architecture_doc.rs"
            ));
        } else if cited.contains(*entry) {
            violations.push(format!(
                "EXEMPT entry `{entry}` is stale: {DOC} now cites it; remove it from EXEMPT in \
                 xtask/src/architecture_doc.rs so a later removal from the doc is caught"
            ));
        } else if is_test_helper(entry) {
            violations.push(format!(
                "EXEMPT entry `{entry}` is redundant: test-only helpers are exempt by name; \
                 remove it from EXEMPT in xtask/src/architecture_doc.rs"
            ));
        }
    }

    violations
}

/// Tracked files under the [`SOURCE_ROOTS`], as the index lists them.
fn tracked_files(root: &Path) -> Result<Vec<String>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--"])
        .args(SOURCE_ROOTS)
        .output()
        .context("failed to run `git ls-files`")?;
    if !output.status.success() {
        bail!(
            "`git ls-files` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect())
}

/// The violation, if any, of `entries` exempt files exceeding `ceiling`.
fn ceiling_violation(entries: usize, ceiling: usize) -> Option<String> {
    (entries > ceiling).then(|| {
        format!(
            "EXEMPT holds {entries} entries but EXEMPT_CEILING is {ceiling}: a file with no entry \
             in the module map should get one, not an EXEMPT line. Growing the list means raising \
             EXEMPT_CEILING in xtask/src/architecture_doc.rs too, which is a reviewed exception"
        )
    })
}

/// Run the check against the real workspace.
fn violations_in(root: &Path) -> Result<Vec<String>> {
    let doc =
        std::fs::read_to_string(root.join(DOC)).with_context(|| format!("failed to read {DOC}"))?;
    let mut violations = check(&tracked_files(root)?, &cited_paths(&doc), EXEMPT);
    violations.extend(ceiling_violation(EXEMPT.len(), EXEMPT_CEILING));
    Ok(violations)
}

/// Fail if a production source file has no entry in `docs/architecture.md`, or
/// [`EXEMPT`] has gone stale.
pub fn run() -> Result<()> {
    let root = paths::workspace_root()?;
    let violations = violations_in(&root)?;
    if !violations.is_empty() {
        bail!(
            "{DOC} is out of step with the source tree ({} problem(s)):\n{}",
            violations.len(),
            violations.join("\n")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(paths: &[&str]) -> Vec<String> {
        paths.iter().map(|path| (*path).to_owned()).collect()
    }

    fn cites(paths: &[&str]) -> BTreeSet<String> {
        paths.iter().map(|path| (*path).to_owned()).collect()
    }

    #[test]
    fn cited_file_passes() {
        let tracked = files(&["apps/rocm/src/cli.rs"]);
        assert!(check(&tracked, &cites(&["apps/rocm/src/cli.rs"]), &[]).is_empty());
    }

    #[test]
    fn new_uncited_file_fails_and_is_named() {
        let tracked = files(&["apps/rocm/src/cli.rs", "apps/rocm/src/brand_new.rs"]);
        let violations = check(&tracked, &cites(&["apps/rocm/src/cli.rs"]), &[]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("apps/rocm/src/brand_new.rs"));
        assert!(violations[0].contains("is not cited"));
    }

    #[test]
    fn a_citation_does_not_cover_the_same_name_in_another_crate() {
        let tracked = files(&["apps/rocm/src/cli.rs", "apps/rocmd/src/cli.rs"]);
        let violations = check(&tracked, &cites(&["apps/rocm/src/cli.rs"]), &[]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("apps/rocmd/src/cli.rs"));
    }

    #[test]
    fn exempt_file_passes() {
        let tracked = files(&["crates/rocm-dash-core/src/state.rs"]);
        let exempt = ["crates/rocm-dash-core/src/state.rs"];
        assert!(check(&tracked, &cites(&[]), &exempt).is_empty());
    }

    #[test]
    fn exempt_entry_for_a_missing_file_is_stale() {
        let exempt = ["crates/rocm-dash-core/src/gone.rs"];
        let violations = check(&files(&[]), &cites(&[]), &exempt);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("crates/rocm-dash-core/src/gone.rs"));
        assert!(violations[0].contains("no tracked source file"));
    }

    #[test]
    fn exempt_entry_that_is_now_cited_is_stale() {
        let tracked = files(&["crates/rocm-dash-core/src/state.rs"]);
        let exempt = ["crates/rocm-dash-core/src/state.rs"];
        let cited = cites(&["crates/rocm-dash-core/src/state.rs"]);
        let violations = check(&tracked, &cited, &exempt);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("now cites it"));
    }

    #[test]
    fn exempt_entry_for_a_test_helper_is_redundant() {
        let tracked = files(&["crates/rocm-core/src/test_support.rs"]);
        let exempt = ["crates/rocm-core/src/test_support.rs"];
        let violations = check(&tracked, &cites(&[]), &exempt);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("redundant"));
    }

    #[test]
    fn files_outside_the_source_trees_are_ignored() {
        let tracked = files(&[
            "apps/rocm/tests/broken_pipe.rs",
            "apps/rocm/build.rs",
            "crates/rocm-dash-tui/examples/gen_cast.rs",
            "xtask/src/anything.rs",
            "apps/rocm/src/data.json",
            "tests/e2e-cucumber/src/lib.rs",
        ]);
        assert!(check(&tracked, &cites(&[]), &[]).is_empty());
    }

    #[test]
    fn nested_source_files_are_in_scope() {
        let tracked = files(&["crates/rocm-dash-tui/src/ui/tabs/home.rs"]);
        let violations = check(&tracked, &cites(&[]), &[]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("ui/tabs/home.rs"));
    }

    #[test]
    fn test_helpers_need_no_entry() {
        let tracked = files(&[
            "apps/rocm/src/test_support.rs",
            "apps/rocm/src/therock/test_support.rs",
            "crates/rocm-core/src/test_env.rs",
            "crates/rocm-core/src/archive_props.rs",
            "crates/rocm-core/src/process_tree_properties.rs",
            "crates/rocm-core/src/examine_driver_proptests.rs",
            "apps/rocm/src/providers/local_http/proptests.rs",
        ]);
        assert!(check(&tracked, &cites(&[]), &[]).is_empty());
    }

    #[test]
    fn helper_matching_is_by_whole_file_stem() {
        // Names that merely contain a helper word are production files.
        for path in [
            "apps/rocm/src/test_supporter.rs",
            "apps/rocm/src/properties.rs",
            "apps/rocm/src/gpu_props_view.rs",
            "apps/rocm/src/proptests_runner.rs",
        ] {
            assert!(
                !is_test_helper(path),
                "{path} must not be treated as a helper"
            );
        }
    }

    #[test]
    fn links_resolve_relative_to_the_doc() {
        let doc = "[cli](../apps/rocm/src/cli.rs) and [x](./../crates/a/src/b.rs#L1)";
        assert_eq!(
            cited_paths(doc),
            cites(&["apps/rocm/src/cli.rs", "crates/a/src/b.rs"])
        );
    }

    #[test]
    fn only_in_repo_rust_links_count_as_citations() {
        let doc = "[md](../README.md) [url](https://example.com/a/src/x.rs) \
                   [out](../../outside/src/x.rs) [bare](apps/rocm/src/cli.rs)";
        assert_eq!(cited_paths(doc), cites(&["docs/apps/rocm/src/cli.rs"]));
    }

    #[test]
    fn a_link_with_a_title_is_still_a_citation() {
        let doc = "[cli](../apps/rocm/src/cli.rs \"the command tree\")";
        assert_eq!(cited_paths(doc), cites(&["apps/rocm/src/cli.rs"]));
    }

    const CLI: &str = "[cli](../apps/rocm/src/cli.rs)";

    fn cited_cli(doc: &str) -> bool {
        cited_paths(doc).contains("apps/rocm/src/cli.rs")
    }

    #[test]
    fn commenting_out_the_only_entry_leaves_the_file_uncited() {
        let doc = format!("<!-- {CLI} -->");
        let tracked = files(&["apps/rocm/src/cli.rs"]);
        let violations = check(&tracked, &cited_paths(&doc), &[]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("apps/rocm/src/cli.rs"));
    }

    #[test]
    fn a_link_in_an_html_comment_is_not_a_citation() {
        assert!(!cited_cli(&format!("<!-- {CLI} -->")));
        assert!(!cited_cli(&format!(
            "before\n<!--\nold entry: {CLI}\n-->\nafter"
        )));
        // An unterminated comment hides everything after it, as a renderer does.
        assert!(!cited_cli(&format!("<!-- {CLI}")));
        assert!(cited_cli(&format!("<!-- note -->\n{CLI}")));
    }

    #[test]
    fn a_link_in_a_fenced_code_block_is_not_a_citation() {
        assert!(!cited_cli(&format!("```\n{CLI}\n```")));
        assert!(!cited_cli(&format!("```md\n{CLI}\n```")));
        assert!(!cited_cli(&format!("~~~\n{CLI}\n~~~")));
        assert!(!cited_cli(&format!("  ```\n  {CLI}\n  ```")));
        // A longer closing fence closes it; a shorter or different one does not.
        assert!(!cited_cli(&format!("````\n```\n{CLI}\n```\n````")));
        assert!(!cited_cli(&format!("~~~\n```\n{CLI}\n```\n~~~")));
        // An unclosed fence runs to the end of the document.
        assert!(!cited_cli(&format!("```\n{CLI}")));
        assert!(cited_cli(&format!("```\ncode\n```\n{CLI}")));
    }

    #[test]
    fn a_link_in_an_inline_code_span_is_not_a_citation() {
        assert!(!cited_cli(&format!("see `{CLI}` for more")));
        assert!(!cited_cli(&format!("see ``a ` b {CLI}`` for more")));
        assert!(cited_cli(&format!("see `code` and {CLI}")));
    }

    #[test]
    fn a_code_span_inside_link_text_still_cites() {
        assert!(cited_cli("[`cli.rs`](../apps/rocm/src/cli.rs)"));
        assert!(cited_cli(
            "- [`cli.rs`](../apps/rocm/src/cli.rs) - the command tree"
        ));
    }

    #[test]
    fn a_stray_backtick_does_not_hide_a_later_paragraph() {
        assert!(cited_cli(&format!("a ` stray\n\n{CLI}")));
        assert!(cited_cli(&format!("a ` stray {CLI}")));
    }

    #[test]
    fn comment_and_code_markers_take_the_usual_precedence() {
        // `<!--` inside a code span is code, so the comment never opens.
        assert!(cited_cli(&format!("`<!--` {CLI} `-->`")));
        // A fence marker inside a comment does not open a fence.
        assert!(cited_cli(&format!("<!--\n```\n-->\n{CLI}")));
        // A backtick inside a comment does not open a code span.
        assert!(cited_cli(&format!("<!-- ` -->\n{CLI}\n<!-- ` -->")));
    }

    #[test]
    fn crlf_documents_are_read_the_same() {
        assert!(!cited_cli(&format!("```\r\n{CLI}\r\n```\r\n")));
        assert!(cited_cli(&format!("a ` stray\r\n\r\n{CLI}")));
    }

    #[test]
    fn an_angle_bracketed_destination_is_a_citation() {
        assert!(cited_cli("[cli](<../apps/rocm/src/cli.rs>)"));
        assert!(cited_cli("[cli](<../apps/rocm/src/cli.rs#L10>)"));
        assert!(cited_cli("[cli](  <../apps/rocm/src/cli.rs>  )"));
        assert!(cited_cli(
            "[cli](<../apps/rocm/src/cli.rs> \"the command tree\")"
        ));
        // The delimiters are not part of the path: a destination with a space is
        // read whole, and an unterminated one is not a link.
        assert_eq!(
            cited_paths("[x](<../crates/a b/src/c.rs>)"),
            cites(&["crates/a b/src/c.rs"])
        );
        assert!(!cited_cli("[cli](<../apps/rocm/src/cli.rs)"));
        assert!(!cited_cli("[cli](<../apps/rocm/src/cli.rs\n>)"));
    }

    #[test]
    fn a_link_missing_its_closing_parenthesis_is_not_a_citation() {
        assert!(!cited_cli("[cli](../apps/rocm/src/cli.rs"));
        assert!(!cited_cli("[cli](../apps/rocm/src/cli.rs and more text)"));
        assert!(!cited_cli(
            "[cli](../apps/rocm/src/cli.rs \"unterminated title)"
        ));
        assert!(!cited_cli(
            "[cli](../apps/rocm/src/cli.rs\n\nnext paragraph)"
        ));
    }

    #[test]
    fn a_link_missing_its_opening_bracket_is_not_a_citation() {
        assert!(!cited_cli("cli](../apps/rocm/src/cli.rs)"));
        // A stray `[` on an earlier line is not this link's text.
        assert!(!cited_cli("a [ stray\ncli](../apps/rocm/src/cli.rs)"));
    }

    #[test]
    fn an_escaped_bracket_or_parenthesis_is_not_a_citation() {
        assert!(!cited_cli("\\[cli](../apps/rocm/src/cli.rs)"));
        assert!(!cited_cli("[cli\\](../apps/rocm/src/cli.rs)"));
        assert!(!cited_cli("[cli]\\(../apps/rocm/src/cli.rs)"));
        // An escaped backslash does not escape the bracket after it.
        assert!(cited_cli("[cli\\\\](../apps/rocm/src/cli.rs)"));
    }

    #[test]
    fn breaking_the_only_entry_leaves_the_file_uncited() {
        let tracked = files(&["apps/rocm/src/cli.rs"]);
        for broken in [
            "[cli](../apps/rocm/src/cli.rs",
            "cli](../apps/rocm/src/cli.rs)",
            "\\[cli](../apps/rocm/src/cli.rs)",
        ] {
            let violations = check(&tracked, &cited_paths(broken), &[]);
            assert_eq!(violations.len(), 1, "{broken:?} must not cite the file");
        }
    }

    #[test]
    fn nested_brackets_and_parentheses_do_not_break_a_link() {
        assert!(cited_cli("[a [b] c](../apps/rocm/src/cli.rs)"));
        assert_eq!(
            cited_paths("[x](../crates/a/src/b(1).rs)"),
            cites(&["crates/a/src/b(1).rs"])
        );
    }

    #[test]
    fn an_image_is_not_a_citation() {
        assert!(!cited_cli("![cli](../apps/rocm/src/cli.rs)"));
        assert!(!cited_cli("see ![cli](../apps/rocm/src/cli.rs \"title\")"));
        let tracked = files(&["apps/rocm/src/cli.rs"]);
        let violations = check(
            &tracked,
            &cited_paths("![cli](../apps/rocm/src/cli.rs)"),
            &[],
        );
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn an_escaped_exclamation_mark_does_not_make_an_image() {
        assert!(cited_cli("\\![cli](../apps/rocm/src/cli.rs)"));
        assert!(cited_cli("wow! [cli](../apps/rocm/src/cli.rs)"));
    }

    #[test]
    fn an_image_inside_a_links_text_leaves_the_link_valid() {
        let doc = "[![icon](../apps/rocm/src/icon.rs)](../apps/rocm/src/cli.rs)";
        assert_eq!(cited_paths(doc), cites(&["apps/rocm/src/cli.rs"]));
    }

    #[test]
    fn a_link_containing_a_link_cites_only_the_inner_one() {
        // Links cannot nest: the page renders only the inner link, so the outer
        // destination is plain text.
        let doc = "[cli [d](../apps/rocm/src/d.rs)](../apps/rocm/src/cli.rs)";
        assert_eq!(cited_paths(doc), cites(&["apps/rocm/src/d.rs"]));
        assert!(!cited_cli(
            "[cli [details](../README.md)](../apps/rocm/src/cli.rs)"
        ));
        // Plain brackets in the text do not form a link, so the outer one stays.
        assert!(cited_cli("[cli [note]](../apps/rocm/src/cli.rs)"));
        assert!(cited_cli("[cli [note] x](../apps/rocm/src/cli.rs)"));
        // Later links on the line are unaffected.
        let doc = "[a [b](../apps/rocm/src/b.rs)](../x.rs) and [c](../apps/rocm/src/c.rs)";
        assert_eq!(
            cited_paths(doc),
            cites(&["apps/rocm/src/b.rs", "apps/rocm/src/c.rs"])
        );
    }

    #[test]
    fn an_escaped_backtick_does_not_open_a_code_span() {
        assert!(cited_cli(
            "Use \\` for a backtick; [`cli.rs`](../apps/rocm/src/cli.rs)"
        ));
        // An escaped backslash does not escape the backtick after it.
        assert!(!cited_cli("\\\\`[cli](../apps/rocm/src/cli.rs)`"));
        // An escaped `<` does not open a comment.
        assert!(cited_cli("\\<!-- [cli](../apps/rocm/src/cli.rs) -->"));
    }

    #[test]
    fn the_exempt_list_stays_within_its_ceiling() {
        assert_eq!(ceiling_violation(EXEMPT.len(), EXEMPT_CEILING), None);
        assert_eq!(ceiling_violation(77, 77), None);
        assert_eq!(ceiling_violation(70, 77), None);
        let violation = ceiling_violation(78, 77).expect("one over the ceiling fails");
        assert!(violation.contains("78 entries"));
        assert!(violation.contains("EXEMPT_CEILING is 77"));
    }

    #[test]
    fn the_link_forms_the_failure_message_rules_out_are_not_citations() {
        // The message tells authors these forms are not recognised; this holds it
        // to what the parser does.
        assert!(!cited_cli("[cli][c]\n\n[c]: ../apps/rocm/src/cli.rs"));
        assert!(!cited_cli("[cli](../apps/rocm/src/cli.rs?plain=1)"));
        assert!(!cited_cli("[cli](/apps/rocm/src/cli.rs)"));
        let message = &check(&files(&["apps/rocm/src/new.rs"]), &cites(&[]), &[])[0];
        assert!(message.contains("inline `[text](path)`"));
        assert!(message.contains("reference-style"));
        assert!(message.contains("`?query`"));
        assert!(message.contains("root-relative"));
    }

    #[test]
    fn the_failure_message_for_an_uncited_file_only_asks_for_an_entry() {
        let violations = check(&files(&["apps/rocm/src/new.rs"]), &cites(&[]), &[]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("add an entry for it to the module map"));
        assert!(!violations[0].contains("EXEMPT"));
    }

    #[test]
    fn several_links_on_a_line_are_all_found() {
        let doc = "[a](../apps/rocm/src/a.rs), [b](../apps/rocm/src/b.rs) and [c](<../apps/rocm/src/c.rs>)";
        assert_eq!(
            cited_paths(doc),
            cites(&[
                "apps/rocm/src/a.rs",
                "apps/rocm/src/b.rs",
                "apps/rocm/src/c.rs"
            ])
        );
    }

    #[test]
    fn exempt_list_is_sorted_and_unique() {
        let mut sorted = EXEMPT.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, EXEMPT, "keep EXEMPT sorted, without duplicates");
    }

    #[test]
    fn current_tree_matches_the_architecture_doc() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory");
        let violations = violations_in(root).expect("git and the doc are readable in a checkout");
        assert!(
            violations.is_empty(),
            "docs/architecture.md is out of step with the source tree:\n{}",
            violations.join("\n")
        );
    }
}
