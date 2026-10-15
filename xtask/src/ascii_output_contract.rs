// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static contract test: a string literal that can reach a redirectable stream
//! or an on-disk file must be ASCII-only.
//!
//! Regression guard for ROCMAI-543. `ΓÇö` is the UTF-8 bytes `E2 80 94` (an em
//! dash) read back as code page 437. That misdecode can only happen once raw
//! bytes leave the process — on a redirected stream, or in a file some other
//! tool opens later. A write to a real Windows console never takes that path:
//! the handle is a console, so Rust emits UTF-16 via `WriteConsoleW` and no
//! code page is consulted. So the rule is about the **stream**, not the
//! character, and the exemptions below are all "these bytes cannot reach a
//! redirectable sink".
//!
//! # Why this is a whole-tree scan and not per-site tests
//!
//! The first two rounds of the ROCMAI-543 fix covered each converted string
//! with its own assertion. That shape failed twice, both times the same way:
//! the finder pass that drove it was anchored on the macro name (`warn!`,
//! `println!`, ...), so it never saw an em dash sitting on a **continuation
//! line of a multi-line string literal**. Two such leftovers shipped inside
//! the fix itself — `apps/rocm/src/dash.rs`'s test-clock warning (file-only
//! client log) and `crates/rocm-dash-daemon/src/runner.rs`'s amd-smi warning
//! (serialized into the NDJSON session file). Neither was catchable by adding
//! more per-site assertions, because the problem was never assertion
//! strength; it was that nothing enumerated the sites.
//!
//! This guard enumerates them. It lexes each file and inspects **literals**,
//! so a literal's continuation lines are part of the literal and a dash
//! hiding on one is found like any other. It also makes the per-site tests
//! unnecessary as *revert detectors*: changing any converted string back to
//! an em dash makes that literal non-ASCII, and this test names the file,
//! line and text. New code is covered the day it lands, with nobody
//! remembering to add anything.
//!
//! # Why the exemption list cannot rot
//!
//! The objection to a scan like this is that its allowlist decays: entries
//! accumulate, nobody prunes them, and the guard quietly stops guarding. So
//! the check is **bidirectional**, exactly as [`crate::crate_edges`]'s
//! `ALLOWLIST` is. An exempt entry that matches nothing is as much a failure
//! as an unexempted literal. Convert a spinner label to ASCII, or delete the
//! file an exemption names, and this test goes red until the stale entry is
//! removed. The list can therefore only ever describe literals that are
//! really there.
//!
//! # What this does *not* cover
//!
//! Only literals in the source. Two gaps, both covered elsewhere by the
//! registry-walking tests this complements:
//!
//! * **Clap `--help` text** comes from `///` doc comments, which are comments
//!   and so are skipped here. `every_help_text_in_the_command_tree_is_ascii`
//!   (`apps/rocm/src/main.rs`) walks the whole clap command tree and asserts
//!   the *rendered* help is ASCII, which also covers `#[arg(help = ...)]`.
//! * **Text assembled at runtime** from non-literal sources.
//!   `every_message_the_diagnose_catalog_can_emit_is_ascii`
//!   (`crates/rocm-core/src/diagnose.rs`) walks the `CHECKERS` catalog and
//!   asserts the rendered report and the `--json` payload are ASCII.
//!
//! Source scan and rendered-output walk catch different mistakes; keep both.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    /// Directory and file prefixes, relative to the repo root, whose
    /// non-ASCII literals are out of scope wholesale. Paths use `/`.
    ///
    /// Each entry is a claim about where the bytes can go, not a convenience.
    const EXEMPT_PREFIXES: &[(&str, &str)] = &[
        (
            "crates/rocm-dash-tui/",
            "The TUI only runs on a real console: it takes over the alternate screen \
             and refuses to start without a TTY, so its bytes go out as UTF-16 and \
             never meet a code page. Box drawing, Braille sparklines and arrows are \
             its visual language and cannot be transliterated without replacing it.",
        ),
        (
            "crates/e2e-report/",
            "A CI report generator, not CLI output. It is reachable only from `xtask` \
             and the e2e harness (never from the shipped `rocm` binary -- see \
             `crate_edges::ALLOWLIST`), and what it writes is HTML carrying an \
             explicit `meta charset=\"utf-8\"` plus GitHub-flavoured Markdown for the \
             Actions job summary. Both are decoded as UTF-8 by declaration, not by \
             whatever code page a console happens to be on. Its pass/fail glyph \
             vocabulary is the same case as the TUI's.",
        ),
        (
            "xtask/",
            "Developer and CI tooling. Never shipped in any released binary, so it \
             cannot be what a user runs in a Windows console.",
        ),
        (
            "tests/",
            "The e2e harness: test code and assertion text, not product output.",
        ),
        (
            "apps/rocm/src/cli_progress.rs",
            "Every spinner in this module is hard-gated on `stderr().is_terminal()` \
             and renders nothing at all when that is false, so its bytes reach a \
             console or nowhere. It also paints Braille frames and a U+2026 overflow \
             marker on every repaint, so transliterating anything here in isolation \
             would change nothing observable.",
        ),
    ];

    /// Path fragments that mark test-only sources: integration-test crates,
    /// examples, benches, and the test-support/proptest modules that are
    /// compiled into a lib but only ever exercised from tests.
    const TEST_ONLY_FRAGMENTS: &[&str] = &[
        "/tests/",
        "/examples/",
        "/benches/",
        "test_support.rs",
        "proptests.rs",
        "fake_executable.rs",
    ];

    /// Data files embedded by `include_str!` whose non-ASCII content is out of
    /// scope, as `(repo-relative path, why)`.
    ///
    /// An embedded file is as much a source of output as a literal is -- the
    /// labels in `model_catalog.json` are what `rocm model` prints, which is
    /// the command ROCMAI-543 was filed against -- but it is not Rust, so the
    /// lexer above never sees it. This list is the exception to a blanket
    /// "embedded data must be ASCII".
    const EXEMPT_EMBEDDED: &[(&str, &str)] = &[
        (
            "skills/rocm-cli-assistant/SKILL.md",
            "The `©` is in the copyright line of a document embedded as the chat \
             tool's skill prompt. It is sent to a model, never printed to a stream.",
        ),
        (
            "install.sh",
            "The `©` is in a `#` shell comment of the installer `rocm remote \
             provision` uploads. The shell never echoes a comment.",
        ),
    ];

    /// Individual literals that are out of scope, as
    /// `(path suffix, exact literal source text, why)`.
    ///
    /// Keyed on the literal's text rather than a line number so the entry
    /// survives the file moving around it, and so a *reworded* message is a
    /// new decision rather than something an old entry silently keeps
    /// covering.
    const EXEMPT_LITERALS: &[(&str, &str, &str)] = &[
        (
            "apps/rocm/src/comfyui.rs",
            "Resolving and installing packages with uv…",
            SPINNER_LABEL,
        ),
        (
            "apps/rocm/src/comfyui.rs",
            "Fetching ComfyUI source archive…",
            SPINNER_LABEL,
        ),
        (
            "apps/rocm/src/main.rs",
            "Waiting for a launch already under way…",
            SPINNER_LABEL,
        ),
        (
            "apps/rocm/src/serve_cmd.rs",
            "Starting {model} on {selected_engine}…",
            SPINNER_LABEL,
        ),
        (
            "apps/rocm/src/serve_cmd.rs",
            "Running smoke test…",
            SPINNER_LABEL,
        ),
        ("apps/rocm/src/therock.rs", "Downloading {}…", SPINNER_LABEL),
        ("apps/rocm/src/therock.rs", "Extracting {}…", SPINNER_LABEL),
        ("crates/rocm-dash-core/src/state.rs", "⋯", JOB_GLYPH),
        ("crates/rocm-dash-core/src/state.rs", "✓", JOB_GLYPH),
        ("crates/rocm-dash-core/src/state.rs", "✗", JOB_GLYPH),
        ("crates/rocm-dash-core/src/state.rs", "○", JOB_GLYPH),
    ];

    const SPINNER_LABEL: &str = "A `cli_progress::Spinner`/`AnimatedSpinner` label. The spinner is hard-gated on \
         `stderr().is_terminal()` and renders nothing when that is false, so this text \
         reaches a console or nothing at all -- see the `cli_progress.rs` exemption.";

    const JOB_GLYPH: &str = "`JobStatus::glyph()`. Every caller is in `rocm-dash-tui` (the Home tab activity \
         feed and the job console banner), so these reach the alternate screen only; \
         the type lives in `rocm-dash-core` because that is the crate both sides share.";

    fn repo_root() -> PathBuf {
        // CARGO_MANIFEST_DIR is the xtask/ crate dir; its parent is the repo
        // root (same idiom as workflow_contract::repo_root).
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Every `.rs` file in the repository, as repo-root-relative `/` paths.
    ///
    /// A whole-tree walk rather than a list of source directories, so a crate
    /// added under a new top-level directory is covered without anyone
    /// remembering to extend this. `target/` and dot-directories are skipped:
    /// the latter include the `.claude/worktrees` checkouts of other
    /// branches, and scanning those would make the verdict depend on state
    /// outside the commit under test.
    fn workspace_rust_sources() -> Vec<String> {
        let root = repo_root();
        let mut found = Vec::new();
        collect_rust_files(&root, &root, &mut found);
        assert!(
            !found.is_empty(),
            "found no Rust sources to scan -- the walk is broken, not the tree"
        );
        found.sort();
        found
    }

    fn collect_rust_files(root: &Path, dir: &Path, found: &mut Vec<String>) {
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
                collect_rust_files(root, &path, found);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let relative = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                found.push(relative);
            }
        }
    }

    /// One non-ASCII literal, located.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Finding {
        /// 1-based line on which the literal *starts*. A multi-line literal is
        /// reported once, at its opening quote -- the continuation lines are
        /// part of the same literal, which is the whole point.
        line: usize,
        /// The literal's source text, between the delimiters and with escapes
        /// left as written.
        text: String,
    }

    /// A byte range of the source covered by a `#[cfg(test)]` / `#[test]` item.
    #[derive(Debug, Clone, Copy)]
    struct TestSpan {
        start: usize,
        end: usize,
    }

    /// Lex `source` far enough to tell literals from comments, and find every
    /// string/char literal holding a non-ASCII character outside test code.
    ///
    /// Hand-rolled rather than `syn`-based so this guard adds no dependency to
    /// `xtask`; the same tradeoff `env_mutation_contract` makes. It handles the
    /// forms that actually appear in this tree: `//` and nested `/* */`
    /// comments, `"..."` with escapes and `\`-newline continuations, raw
    /// strings at any hash count, `b"..."`/`br"..."`, and char literals kept
    /// apart from lifetimes.
    ///
    /// Comments are skipped deliberately: a `©` in a licence header or an em
    /// dash in prose explaining the code is not output, and flagging those
    /// would make the guard unusable. Clap help text *is* doc comments and so
    /// is not covered here -- see the module docs for the test that does.
    fn non_ascii_literals(source: &str) -> Vec<Finding> {
        let bytes = source.as_bytes();
        let mut findings = Vec::new();
        let mut spans: Vec<TestSpan> = Vec::new();

        let mut i = 0usize;
        let mut line = 1usize;
        let mut depth: i32 = 0;
        let mut pending_test = false;
        let mut open_test: Option<(usize, i32)> = None;

        while i < bytes.len() {
            // A stray non-ASCII byte outside any literal can only be part of a
            // multi-byte character the loop stepped into (a `/* © */` comment
            // body, say). Every token this scanner opens on starts with an
            // ASCII byte, so skipping to the next boundary cannot lose one --
            // and slicing `source` mid-character would panic.
            if !source.is_char_boundary(i) {
                i += 1;
                continue;
            }
            let rest = &source[i..];

            if bytes[i] == b'\n' {
                line += 1;
                i += 1;
                continue;
            }

            // Comments.
            if rest.starts_with("//") {
                let end = rest.find('\n').map_or(bytes.len(), |n| i + n);
                i = end;
                continue;
            }
            if rest.starts_with("/*") {
                let mut nesting = 1usize;
                i += 2;
                while i < bytes.len() && nesting > 0 {
                    if bytes[i] == b'\n' {
                        line += 1;
                        i += 1;
                    } else if bytes[i..].starts_with(b"/*") {
                        nesting += 1;
                        i += 2;
                    } else if bytes[i..].starts_with(b"*/") {
                        nesting -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                continue;
            }

            // Raw strings: an optional `b`, then `r`, then any number of `#`.
            if let Some((literal, start_line, next)) = raw_string_at(source, i, line) {
                record(&mut findings, start_line, literal);
                line = next.1;
                i = next.0;
                continue;
            }

            // Ordinary and byte strings.
            if let Some((literal, start_line, next)) = quoted_string_at(source, i, line) {
                record(&mut findings, start_line, &literal);
                line = next.1;
                i = next.0;
                continue;
            }

            // Char literals, kept apart from lifetimes (`'a`, `'static`): a
            // lifetime has no closing quote, so requiring one is the whole
            // discriminator.
            if bytes[i] == b'\'' {
                if let Some((literal, next)) = char_literal_at(source, i) {
                    record(&mut findings, line, literal);
                    i = next;
                    continue;
                }
                i += 1;
                continue;
            }

            // Test-item tracking, on code bytes only.
            if bytes[i] == b'#' && starts_test_attribute(rest) {
                pending_test = true;
                i += 1;
                continue;
            }
            if bytes[i] == b'{' {
                if pending_test && open_test.is_none() {
                    open_test = Some((i, depth));
                }
                depth += 1;
                i += 1;
                continue;
            }
            if bytes[i] == b'}' {
                depth -= 1;
                if let Some((start, at)) = open_test
                    && depth == at
                {
                    // A `#[cfg(test)] use a::{b, c};` brace group is not an
                    // item body; the `;` right after it gives it away. The
                    // attribute belonged to that `use` and is now spent --
                    // carrying `pending_test` forward would make the *next*
                    // brace in the file, typically a production `fn`, look
                    // like a test body and silently exempt everything in it.
                    let tail = source[i + 1..].trim_start();
                    if tail.starts_with(';') {
                        open_test = None;
                        pending_test = false;
                    } else {
                        spans.push(TestSpan { start, end: i });
                        open_test = None;
                        pending_test = false;
                    }
                }
                i += 1;
                continue;
            }

            // Same reasoning for a brace-less `#[cfg(test)] use a::b;`: the
            // statement ends, so the attribute cannot still be waiting for an
            // item body.
            if bytes[i] == b';' && pending_test && open_test.is_none() {
                pending_test = false;
                i += 1;
                continue;
            }

            i += 1;
        }

        let line_of = |offset: usize| source[..offset].matches('\n').count() + 1;
        let test_lines: Vec<(usize, usize)> = spans
            .iter()
            .map(|s| (line_of(s.start), line_of(s.end)))
            .collect();

        findings.retain(|f| !test_lines.iter().any(|&(a, b)| a <= f.line && f.line <= b));
        findings
    }

    fn record(findings: &mut Vec<Finding>, line: usize, literal: &str) {
        if !literal.is_ascii() {
            findings.push(Finding {
                line,
                text: literal.to_owned(),
            });
        }
    }

    fn starts_test_attribute(rest: &str) -> bool {
        const MARKERS: &[&str] = &[
            "#[test]",
            "#[cfg(test)]",
            "#[tokio::test",
            "#[rstest",
            "#[proptest",
        ];
        MARKERS.iter().any(|m| rest.starts_with(m))
    }

    /// `(literal, start line, (next offset, next line))` if a raw string starts
    /// at `at`.
    fn raw_string_at(
        source: &str,
        at: usize,
        line: usize,
    ) -> Option<(&str, usize, (usize, usize))> {
        let bytes = source.as_bytes();
        let mut i = at;
        if bytes.get(i) == Some(&b'b') {
            i += 1;
        }
        if bytes.get(i) != Some(&b'r') {
            return None;
        }
        i += 1;
        let hash_start = i;
        while bytes.get(i) == Some(&b'#') {
            i += 1;
        }
        let hashes = i - hash_start;
        if bytes.get(i) != Some(&b'"') {
            return None;
        }
        // `r` must begin a token: `for#` or an identifier ending in `r` is not
        // a raw-string prefix.
        if at > 0 && is_ident_byte(bytes[at - 1]) {
            return None;
        }
        i += 1;
        let body_start = i;
        let terminator = format!("\"{}", "#".repeat(hashes));
        let body_end = source[body_start..]
            .find(&terminator)
            .map_or(source.len(), |n| body_start + n);
        let literal = &source[body_start..body_end];
        let next_line = line + literal.matches('\n').count();
        let next = ((body_end + terminator.len()).min(source.len()), next_line);
        Some((literal, line, next))
    }

    /// `(literal, start line, (next offset, next line))` if an ordinary or byte
    /// string starts at `at`.
    fn quoted_string_at(
        source: &str,
        at: usize,
        line: usize,
    ) -> Option<(String, usize, (usize, usize))> {
        let bytes = source.as_bytes();
        let mut i = at;
        if bytes.get(i) == Some(&b'b') {
            i += 1;
        }
        if bytes.get(i) != Some(&b'"') {
            return None;
        }
        if at > 0 && is_ident_byte(bytes[at - 1]) {
            return None;
        }
        i += 1;
        let mut literal = String::new();
        let mut newlines = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => {
                    if bytes.get(i + 1) == Some(&b'\n') {
                        newlines += 1;
                    }
                    // Push the escaped byte as written; what matters is only
                    // whether a non-ASCII character is present, and an escape
                    // sequence never is.
                    let end = (i + 2).min(bytes.len());
                    literal.push_str(&source[i..end]);
                    i = end;
                }
                b'"' => break,
                b'\n' => {
                    newlines += 1;
                    literal.push('\n');
                    i += 1;
                }
                _ => {
                    let ch = source[i..].chars().next().unwrap_or('\0');
                    literal.push(ch);
                    i += ch.len_utf8();
                }
            }
        }
        Some((literal, line, ((i + 1).min(source.len()), line + newlines)))
    }

    /// `(literal, next offset)` if a char literal starts at `at`, else `None`
    /// (a lifetime).
    fn char_literal_at(source: &str, at: usize) -> Option<(&str, usize)> {
        let bytes = source.as_bytes();
        let body_start = at + 1;
        if bytes.get(body_start) == Some(&b'\\') {
            let close = source[body_start..].find('\'')?;
            return Some((
                &source[body_start..body_start + close],
                body_start + close + 1,
            ));
        }
        let ch = source[body_start..].chars().next()?;
        let close = body_start + ch.len_utf8();
        if bytes.get(close) != Some(&b'\'') {
            return None;
        }
        Some((&source[body_start..close], close + 1))
    }

    const fn is_ident_byte(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_'
    }

    fn exempt_prefix(path: &str) -> Option<&'static str> {
        EXEMPT_PREFIXES
            .iter()
            .find(|(prefix, _)| path.starts_with(prefix))
            .map(|(prefix, _)| *prefix)
    }

    fn is_test_only(path: &str) -> bool {
        TEST_ONLY_FRAGMENTS
            .iter()
            .any(|fragment| path.contains(fragment))
    }

    /// Every non-ASCII literal in the tree, as `(path, finding)`, with
    /// wholesale-exempt and test-only files already dropped.
    fn in_scope_findings() -> Vec<(String, Finding)> {
        let root = repo_root();
        let mut out = Vec::new();
        for path in workspace_rust_sources() {
            if exempt_prefix(&path).is_some() || is_test_only(&path) {
                continue;
            }
            let Ok(source) = std::fs::read_to_string(root.join(&path)) else {
                continue;
            };
            for finding in non_ascii_literals(&source) {
                out.push((path.clone(), finding));
            }
        }
        out
    }

    /// The contract: no unexempted non-ASCII literal anywhere a byte can reach
    /// a redirectable stream or a file.
    #[test]
    fn every_literal_that_can_reach_a_redirectable_sink_is_ascii() {
        let offenders: Vec<String> = in_scope_findings()
            .into_iter()
            .filter(|(path, finding)| {
                !EXEMPT_LITERALS
                    .iter()
                    .any(|(file, literal, _)| path.ends_with(file) && finding.text == *literal)
            })
            .map(|(path, finding)| format!("  {}:{}  {:?}", path, finding.line, finding.text))
            .collect();

        assert!(
            offenders.is_empty(),
            "non-ASCII in {} literal(s) that can reach a redirectable stream or a file \
             (ROCMAI-543: a Windows console on code page 437 reads the UTF-8 bytes of an \
             em dash, E2 80 94, as three mojibake characters). Replace with ASCII, or -- \
             if the bytes provably cannot reach such a sink -- add the site to \
             EXEMPT_LITERALS/EXEMPT_PREFIXES in xtask/src/ascii_output_contract.rs \
             together with the reason:\n{}",
            offenders.len(),
            offenders.join("\n")
        );
    }

    /// The other direction: an exemption that no longer matches anything is a
    /// failure too, so the list cannot accumulate dead entries and quietly
    /// stop guarding. Same bidirectional discipline as
    /// `crate_edges::ALLOWLIST`.
    #[test]
    fn every_exemption_still_describes_a_literal_that_is_really_there() {
        let findings = in_scope_findings();
        let stale: Vec<String> = EXEMPT_LITERALS
            .iter()
            .filter(|(file, literal, _)| {
                !findings
                    .iter()
                    .any(|(path, f)| path.ends_with(file) && f.text == *literal)
            })
            .map(|(file, literal, _)| format!("  {file}  {literal:?}"))
            .collect();

        assert!(
            stale.is_empty(),
            "{} EXEMPT_LITERALS entr(y/ies) match no literal in the tree. The string was \
             converted, reworded or deleted -- remove the entry, or this list will outlive \
             what it describes:\n{}",
            stale.len(),
            stale.join("\n")
        );
    }

    /// A wholesale prefix exemption is a large claim; require it to still be
    /// load-bearing. A prefix covering nothing means the crate was cleaned up
    /// or removed and the blanket exemption should go with it.
    #[test]
    fn every_exempt_prefix_still_covers_a_non_ascii_literal() {
        let root = repo_root();
        let sources = workspace_rust_sources();
        let stale: Vec<&str> = EXEMPT_PREFIXES
            .iter()
            .filter(|(prefix, _)| {
                !sources
                    .iter()
                    .filter(|path| path.starts_with(prefix))
                    .any(|path| {
                        std::fs::read_to_string(root.join(path))
                            .is_ok_and(|source| !non_ascii_literals(&source).is_empty())
                    })
            })
            .map(|(prefix, _)| *prefix)
            .collect();

        assert!(
            stale.is_empty(),
            "{} EXEMPT_PREFIXES entr(y/ies) cover no non-ASCII literal any more -- drop the \
             blanket exemption: {:?}",
            stale.len(),
            stale
        );
    }

    /// Every non-test `include_str!` in the tree, as
    /// `(embedding source, repo-relative path of the embedded file)`.
    ///
    /// `include_str!` paths are relative to the *including file*, so they are
    /// resolved against its directory and then normalised back to repo-root
    /// relative. Embeds of `.rs` files are skipped: those are already walked
    /// as sources in their own right.
    fn embedded_data_files() -> Vec<(String, String)> {
        let root = repo_root();
        let mut out = Vec::new();
        for path in workspace_rust_sources() {
            if is_test_only(&path) || path.starts_with("tests/") {
                continue;
            }
            let Ok(source) = std::fs::read_to_string(root.join(&path)) else {
                continue;
            };
            for (offset, _) in source.match_indices("include_str!(\"") {
                let start = offset + "include_str!(\"".len();
                let Some(len) = source[start..].find('"') else {
                    continue;
                };
                let relative = &source[start..start + len];
                if Path::new(relative)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"))
                {
                    continue;
                }
                // A `#[cfg(test)]`-only embed is test data; the three
                // `model_catalog.json` embeds include two inside the test
                // module, and dropping those just de-duplicates.
                let Some(dir) = Path::new(&path).parent() else {
                    continue;
                };
                let joined = dir.join(relative);
                let joined = joined.to_string_lossy().replace('\\', "/");
                let mut parts: Vec<&str> = Vec::new();
                for part in joined.split('/') {
                    match part {
                        "." | "" => {}
                        ".." => {
                            parts.pop();
                        }
                        other => parts.push(other),
                    }
                }
                let normalised = parts.join("/");
                if !out.iter().any(|(_, p): &(String, String)| *p == normalised) {
                    out.push((path.clone(), normalised));
                }
            }
        }
        out
    }

    /// A data file compiled into the binary is output too. `rocm model` prints
    /// the `label` fields of `model_catalog.json` verbatim, and that command is
    /// what ROCMAI-543 was reported against -- but the file is JSON, so the
    /// literal scan above cannot see it.
    #[test]
    fn every_embedded_data_file_is_ascii() {
        let root = repo_root();
        let mut offenders = Vec::new();
        for (source, embedded) in embedded_data_files() {
            if EXEMPT_EMBEDDED.iter().any(|(p, _)| *p == embedded) {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(root.join(&embedded)) else {
                continue;
            };
            for (number, line) in content.lines().enumerate() {
                if !line.is_ascii() {
                    offenders.push(format!(
                        "  {embedded}:{} (embedded by {source})  {line:?}",
                        number + 1
                    ));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "{} non-ASCII line(s) in data files compiled into a binary (ROCMAI-543). \
             These reach the same streams a literal does:\n{}",
            offenders.len(),
            offenders.join("\n")
        );
    }

    /// The embed walk must really be finding the catalog; a regex that matched
    /// nothing would make the test above vacuous.
    #[test]
    fn the_embed_walk_finds_the_model_catalog() {
        let embedded: Vec<String> = embedded_data_files()
            .into_iter()
            .map(|(_, path)| path)
            .collect();
        assert!(
            embedded.contains(&"crates/rocm-core/src/model_catalog.json".to_owned()),
            "the `rocm model` catalog must be covered, got {embedded:?}"
        );
        for (path, _) in EXEMPT_EMBEDDED {
            assert!(
                embedded.contains(&(*path).to_owned()),
                "EXEMPT_EMBEDDED entry {path} is no longer embedded anywhere -- remove it"
            );
        }
    }

    /// Every exemption carries a reason, because the reason is what a reviewer
    /// checks. An empty one would make the list a rubber stamp.
    #[test]
    fn every_exemption_records_why() {
        for (path, reason) in EXEMPT_EMBEDDED {
            assert!(
                reason.len() > 40,
                "EXEMPT_EMBEDDED entry {path:?} needs a reason, not {reason:?}"
            );
        }
        for (prefix, reason) in EXEMPT_PREFIXES {
            assert!(
                reason.len() > 40,
                "EXEMPT_PREFIXES entry {prefix:?} needs a reason explaining where its bytes \
                 can go, not {reason:?}"
            );
        }
        for (file, literal, reason) in EXEMPT_LITERALS {
            assert!(
                reason.len() > 40,
                "EXEMPT_LITERALS entry {file}/{literal:?} needs a reason, not {reason:?}"
            );
        }
    }

    // --- scanner fixtures -----------------------------------------------------
    //
    // The guard is only as good as the lexer, and the bug it exists to catch
    // was precisely a finder that could not see a particular shape. So each
    // shape that defeated an earlier pass has a fixture here.

    fn lines(source: &str) -> Vec<(usize, String)> {
        non_ascii_literals(source)
            .into_iter()
            .map(|f| (f.line, f.text))
            .collect()
    }

    /// The shape that let two leftovers ship: the dash is not on the line the
    /// macro name is on, so a call-site-anchored grep never saw it.
    #[test]
    fn an_em_dash_on_a_continuation_line_of_a_multiline_literal_is_found() {
        let source = "fn f() {\n    tracing::warn!(\n        \"first line \\\n         second \u{2014} line\"\n    );\n}\n";
        let found = lines(source);
        assert_eq!(found.len(), 1, "expected exactly one finding: {found:?}");
        assert_eq!(found[0].0, 3, "reported at the literal's opening quote");
        assert!(
            found[0].1.contains('\u{2014}'),
            "the finding carries the offending text: {found:?}"
        );
    }

    /// Prose about the code is not output. The licence header on every file in
    /// this repo contains `©`; flagging that would make the guard unusable.
    #[test]
    fn non_ascii_in_comments_is_not_a_finding() {
        assert!(lines("// Copyright \u{a9} someone\nfn f() {}\n").is_empty());
        assert!(lines("/// An em dash \u{2014} in prose.\nfn f() {}\n").is_empty());
        assert!(lines("/* \u{2014} */ /* a /* nested \u{2014} */ one */ fn f() {}\n").is_empty());
    }

    /// Raw strings are still literals, at any hash count.
    #[test]
    fn non_ascii_in_a_raw_string_is_a_finding() {
        assert_eq!(lines("fn f() { let _ = r\"a \u{2014} b\"; }\n").len(), 1);
        assert_eq!(lines("fn f() { let _ = r#\"a \u{2014} b\"#; }\n").len(), 1);
        assert_eq!(
            lines("fn f() { let _ = br##\"a \u{2014} b\"##; }\n").len(),
            1
        );
    }

    /// A `'` is usually a lifetime, and treating one as an unterminated char
    /// literal would swallow the rest of the file -- silently hiding every
    /// finding after it.
    #[test]
    fn lifetimes_are_not_mistaken_for_char_literals() {
        let source =
            "struct S<'a>(&'a str);\nfn f<'a>(x: &'a str) {\n    let _ = \"\u{2014}\";\n}\n";
        let found = lines(source);
        assert_eq!(found.len(), 1, "the string after the lifetimes: {found:?}");
        assert_eq!(found[0].0, 3);
    }

    /// A non-ASCII char literal is output too.
    #[test]
    fn a_non_ascii_char_literal_is_a_finding() {
        assert_eq!(lines("fn f() { let _ = '\u{2026}'; }\n").len(), 1);
        assert!(lines("fn f() { let _ = '\\n'; }\n").is_empty());
    }

    /// Test code is assertion text, not product output.
    #[test]
    fn literals_inside_test_items_are_skipped() {
        let source = "fn f() {}\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() {\n        assert_eq!(x, \"\u{2014}\");\n    }\n}\n";
        assert!(lines(source).is_empty(), "{:?}", lines(source));
    }

    /// ...but a `#[cfg(test)] use a::{b, c};` brace group is not an item body.
    /// Treating it as one would exempt the whole rest of the file, which is
    /// how a naive "first `#[cfg(test)]` wins" rule silently drops findings.
    #[test]
    fn a_cfg_test_use_group_does_not_exempt_the_rest_of_the_file() {
        let source = "#[cfg(test)]\nuse std::fmt::{Debug, Display};\n\nfn f() {\n    let _ = \"\u{2014}\";\n}\n";
        let found = lines(source);
        assert_eq!(
            found.len(),
            1,
            "the production literal still counts: {found:?}"
        );
        assert_eq!(found[0].0, 5);
    }

    /// The brace-less form of the same trap.
    #[test]
    fn a_brace_less_cfg_test_use_does_not_exempt_the_rest_of_the_file() {
        let source =
            "#[cfg(test)]\nuse std::fmt::Debug;\n\nfn f() {\n    let _ = \"\u{2014}\";\n}\n";
        let found = lines(source);
        assert_eq!(
            found.len(),
            1,
            "the production literal still counts: {found:?}"
        );
        assert_eq!(found[0].0, 5);
    }

    /// Escapes are not non-ASCII, however they are spelled.
    #[test]
    fn escaped_quotes_and_escape_sequences_do_not_confuse_the_scanner() {
        assert!(lines("fn f() { let _ = \"a \\\" b \\u{41}\"; }\n").is_empty());
        let found = lines("fn f() { let _ = \"a \\\" b\"; let _ = \"\u{2014}\"; }\n");
        assert_eq!(found.len(), 1, "{found:?}");
    }

    /// The scan must actually be reaching the tree it claims to cover; a walk
    /// that found nothing would make every assertion above vacuously true.
    #[test]
    fn the_scan_reaches_the_whole_workspace() {
        let sources = workspace_rust_sources();
        assert!(
            sources.len() > 200,
            "expected the whole tree, got {} files",
            sources.len()
        );
        for expected in [
            "apps/rocm/src/main.rs",
            "crates/rocm-core/src/diagnose.rs",
            "crates/rocm-dash-daemon/src/runner.rs",
        ] {
            assert!(
                sources.iter().any(|p| p == expected),
                "{expected} missing from the walk"
            );
        }
    }

    /// The two leftovers that shipped inside the ROCMAI-543 fix itself. If
    /// either file ever regrows a non-ASCII literal this names it, and these
    /// assertions say out loud which sinks made them bugs.
    #[test]
    fn the_file_log_and_session_record_sinks_are_ascii() {
        let root = repo_root();
        for (path, sink) in [
            (
                "apps/rocm/src/dash.rs",
                "tracing::warn! here goes to the file-only client log rocm-cli.log",
            ),
            (
                "crates/rocm-dash-daemon/src/runner.rs",
                "Snapshot.warnings is serialized to the NDJSON session file by SessionWriter",
            ),
        ] {
            let source = std::fs::read_to_string(root.join(path)).expect("source is readable");
            let found = non_ascii_literals(&source);
            assert!(
                found.is_empty(),
                "{path} must stay ASCII-only ({sink}): {found:?}"
            );
        }
    }
}
