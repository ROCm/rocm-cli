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
            "crates/rocm-core/src/terminal.rs",
            "A classifier for untrusted terminal output, and the one file in the tree \
             whose job is to name non-ASCII characters rather than print them. Every \
             literal here is a `Cf`/`Zl`/`Zp` code point in a `matches!` arm of \
             `is_control_or_format`/`is_control_or_line_separator`, enumerated so it can \
             be *removed* from a captured subprocess log before that log is quoted back \
             at the user. The module emits no text of its own: both public functions \
             return substrings of their input. Transliterating a code point here would \
             not change any output, it would stop the stripper recognising the thing it \
             exists to strip.",
        ),
        (
            "apps/rocm/src/cli_progress.rs",
            "Every spinner in this module is hard-gated on `stderr().is_terminal()` \
             and renders nothing at all when that is false, so its bytes reach a \
             console or nowhere. It also paints Braille frames and a U+2026 overflow \
             marker on every repaint, so transliterating anything here in isolation \
             would change nothing observable.",
        ),
        (
            "apps/rocm/src/advised_commands.rs",
            "Test code, compiled into no binary: `apps/rocm/src/main.rs` declares it \
             only as `#[cfg(test)] mod advised_commands;`, which `TEST_ONLY_FRAGMENTS` \
             (a path test) cannot see. Its non-ASCII literals are needles and fixtures \
             of the advised-command checker -- the em dash, en dash and arrow that end \
             a command in prose, the ellipsis a doc writes for an omitted argument, \
             and a Unicode model reference fed to `generate_service_id` -- so they are \
             matched against the docs, never written anywhere. Spelling them as \
             `\\u{..}` escapes would not help: this scan flags an escape that denotes \
             non-ASCII just the same.",
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
        ("apps/rocm/src/main.rs", "\\u{feff}", STRIPPED_FROM_INPUT),
        (
            "crates/rocm-core/src/crypto.rs",
            "\\u{feff}",
            STRIPPED_FROM_INPUT,
        ),
        (
            "crates/rocm-core/src/examine.rs",
            "(\\u{2122})",
            STRIPPED_FROM_INPUT,
        ),
        ("crates/rocm-dash-core/src/state.rs", "⋯", JOB_GLYPH),
        ("crates/rocm-dash-core/src/state.rs", "✓", JOB_GLYPH),
        ("crates/rocm-dash-core/src/state.rs", "✗", JOB_GLYPH),
        ("crates/rocm-dash-core/src/state.rs", "○", JOB_GLYPH),
    ];

    const SPINNER_LABEL: &str = "A `cli_progress::Spinner`/`AnimatedSpinner` label. The spinner is hard-gated on \
         `stderr().is_terminal()` and renders nothing when that is false, so this text \
         reaches a console or nothing at all -- see the `cli_progress.rs` exemption.";

    const STRIPPED_FROM_INPUT: &str = "Matched so it can be taken *out* of text arriving from outside the process -- a \
         BOM ahead of JSON (`strip_prefix`/`trim_start_matches`) or a `™` in a \
         vendor marketing name (`replace`). The character is a needle, never \
         something this code writes; the only way it reaches a stream is if the input \
         already carried it, which no transliteration here would change.";

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
    /// `xtask`; the same tradeoff `env_mutation_contract` makes. It handles
    /// every string and char form the language has: `//` and nested `/* */`
    /// comments, `"..."` with escapes and `\`-newline continuations, raw
    /// strings at any hash count, the `b`/`c`/`br`/`cr` prefixes, and char
    /// literals kept apart from lifetimes. A literal is non-ASCII either
    /// because its source text is, or -- outside a raw string, which has no
    /// escapes -- because an escape in it *denotes* a non-ASCII value; see
    /// [`escape_denotes_non_ascii`].
    ///
    /// Comments are skipped deliberately: a `©` in a licence header or an em
    /// dash in prose explaining the code is not output, and flagging those
    /// would make the guard unusable. Clap help text *is* doc comments and so
    /// is not covered here -- see the module docs for the test that does.
    ///
    /// Only `//`-style prose is treated that way, though. The attribute
    /// spelling of the same thing -- `#[doc = "..."]`, or the `note` of a
    /// `#[deprecated(note = "...")]` -- is a string literal and is scanned
    /// like any other, so semantically identical prose is judged differently
    /// depending on which syntax it is written in. Nothing in the tree hits
    /// that today; whoever first writes an em dash in a `#[doc = "..."]` and
    /// is told to exempt it should widen the comment skip here rather than
    /// add an `EXEMPT_LITERALS` entry for prose.
    fn non_ascii_literals(source: &str) -> Vec<Finding> {
        let bytes = source.as_bytes();
        let mut findings = Vec::new();
        let mut spans: Vec<TestSpan> = Vec::new();

        let mut i = 0usize;
        let mut line = 1usize;
        let mut depth: i32 = 0;
        let mut pending_test = false;
        let mut attribute_at = 0usize;
        let mut is_use_item = false;
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

            // Raw strings: an optional `b`/`c`, then `r`, then any number of
            // `#`. A backslash in a raw string is a backslash, so escapes are
            // not interpreted.
            if let Some((literal, start_line, next)) = raw_string_at(source, i, line) {
                record(&mut findings, start_line, literal, Escapes::Literal);
                line = next.1;
                i = next.0;
                continue;
            }

            // Ordinary, byte and C strings.
            if let Some((literal, start_line, next)) = quoted_string_at(source, i, line) {
                record(&mut findings, start_line, &literal, Escapes::Interpreted);
                line = next.1;
                i = next.0;
                continue;
            }

            // Char literals, kept apart from lifetimes (`'a`, `'static`): a
            // lifetime has no closing quote, so requiring one is the whole
            // discriminator.
            if bytes[i] == b'\'' {
                if let Some((literal, next)) = char_literal_at(source, i) {
                    record(&mut findings, line, literal, Escapes::Interpreted);
                    i = next;
                    continue;
                }
                i += 1;
                continue;
            }

            // Test-item tracking, on code bytes only.
            if bytes[i] == b'#' {
                if starts_test_attribute(rest) && !pending_test {
                    pending_test = true;
                    attribute_at = i;
                    is_use_item = false;
                }
                i += 1;
                continue;
            }

            // Does this test item say `use`? That is the one word that decides
            // whether the attribute belongs to an import group (no body to
            // exempt) or to an item that owns one, and reading it is what
            // distinguishes `#[cfg(test)] use a::{b, c};` from a
            // `#[cfg(test)] const X: Foo = Foo { .. };`, which trails the same
            // `;` after the same closing brace. `use_x` or `Use` are
            // identifiers in their own right, not this keyword.
            if pending_test && open_test.is_none() && is_ident_start(bytes[i]) {
                let end = ident_end(source, i);
                if &source[i..end] == "use" {
                    is_use_item = true;
                }
                i = end;
                continue;
            }

            // A bracket group between a test attribute and its body belongs to
            // that test item: the `#[cfg(test)]` attribute's own `[..]`, the
            // `[&str; 2]` of a `#[cfg(test)] static`, or the `[..]` value
            // after it. Stepping over it keeps the `cfg`/`test` inside the
            // attribute out of the keyword scan and the `;` inside an array
            // type from ending the item early.
            if bytes[i] == b'[' && pending_test && open_test.is_none() {
                let (end, newlines) = bracket_group_end(source, i);
                line += newlines;
                i = end;
                continue;
            }

            if bytes[i] == b'{' {
                if pending_test && open_test.is_none() {
                    open_test = Some((attribute_at, depth));
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
                    // item body: the attribute belonged to that `use` and is
                    // now spent. Carrying `pending_test` forward would make
                    // the *next* brace in the file, typically a production
                    // `fn`, look like a test body and silently exempt
                    // everything in it.
                    //
                    // The keyword is what tells the two apart, not the `;`
                    // after the closing brace: `#[cfg(test)] const X: Foo =
                    // Foo { .. };` trails a `;` exactly as the `use` group
                    // does, yet it *is* a test item whose literals must be
                    // skipped.
                    if !is_use_item {
                        spans.push(TestSpan { start, end: i });
                    }
                    open_test = None;
                    pending_test = false;
                    is_use_item = false;
                }
                i += 1;
                continue;
            }

            // A test item with no braces at all: `#[cfg(test)] use a::b;`,
            // `#[cfg(test)] mod tests;`, or a `#[cfg(test)] static T: [&str;
            // 2] = [..];`. The statement ends, so the attribute cannot still
            // be waiting for an item body -- and for everything but `use` the
            // statement itself was the test item.
            if bytes[i] == b';' && pending_test && open_test.is_none() {
                if !is_use_item {
                    spans.push(TestSpan {
                        start: attribute_at,
                        end: i,
                    });
                }
                pending_test = false;
                is_use_item = false;
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

    /// Whether a backslash in the recorded text denotes an escape sequence.
    /// Raw strings have no escapes; everything else does.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Escapes {
        Interpreted,
        Literal,
    }

    fn record(findings: &mut Vec<Finding>, line: usize, literal: &str, escapes: Escapes) {
        let non_ascii = !literal.is_ascii()
            || (escapes == Escapes::Interpreted && escape_denotes_non_ascii(literal));
        if non_ascii {
            findings.push(Finding {
                line,
                text: literal.to_owned(),
            });
        }
    }

    /// Whether any escape sequence in `text` *denotes* a non-ASCII value even
    /// though the escape is written in ASCII.
    ///
    /// `"\u{2014}"` is an em dash the moment it is compiled, and the UTF-8
    /// bytes it puts on a redirected stream are the same `E2 80 94` that
    /// ROCMAI-543 is about -- but every byte of the *source* is ASCII, so a
    /// plain `is_ascii()` on the literal text says nothing is wrong. The same
    /// holds for `\xNN` above `7F`, which is legal in a byte string (`b"..."`)
    /// and puts that byte straight on the wire.
    ///
    /// Escapes are walked in order rather than searched for, so an escaped
    /// backslash cannot be misread: in `"\\u{2014}"` the `\\` is consumed as
    /// one escape and the following `u{2014}` is the ASCII text it really is.
    ///
    /// Anything malformed counts as non-ASCII. This scanner is not a Rust
    /// parser and such a literal would not compile anyway; erring towards a
    /// finding keeps the failure visible instead of silent.
    fn escape_denotes_non_ascii(text: &str) -> bool {
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] != b'\\' {
                i += 1;
                continue;
            }
            match bytes.get(i + 1) {
                Some(b'u') if bytes.get(i + 2) == Some(&b'{') => {
                    // `i + 3` follows three ASCII bytes, so it is a char
                    // boundary and slicing cannot panic.
                    let Some(close) = text[i + 3..].find('}') else {
                        return true;
                    };
                    let digits = text[i + 3..i + 3 + close].replace('_', "");
                    match u32::from_str_radix(&digits, 16) {
                        Ok(value) if value < 0x80 => {}
                        _ => return true,
                    }
                    i = i + 3 + close + 1;
                }
                Some(b'x') => {
                    match text
                        .get(i + 2..i + 4)
                        .and_then(|hex| u8::from_str_radix(hex, 16).ok())
                    {
                        Some(value) if value < 0x80 => {}
                        _ => return true,
                    }
                    i += 4;
                }
                Some(_) => i += 2,
                None => i += 1,
            }
        }
        false
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

    /// `(offset just past the matching `]`, newlines crossed)` for the bracket
    /// group opening at `at`.
    ///
    /// Brackets are counted on *code* bytes only. A `]` inside a string, a
    /// char literal or a comment is text, and counting it closed the group
    /// one character early -- returning an offset in the middle of that
    /// literal. The main loop then resumed there, so the literal's real
    /// closing `"` was read as an *opening* one, every quote after it was
    /// paired off by one, and the braces of the next `\u{....}` escape were
    /// taken for code braces. That desync is not local: `open_test` and
    /// `depth` stay wrong for the rest of the file, and a genuinely
    /// non-ASCII **production** literal further down is then never recorded
    /// at all -- the exact silent miss this whole-tree scan exists to make
    /// impossible. See
    /// [`a_bracket_inside_a_literal_does_not_hide_later_production_text`].
    ///
    /// So the group is walked with the same recognisers the outer loop uses
    /// rather than re-scanned from `at` on suspicion: they already know every
    /// string form the language has, and bailing out would bring back the
    /// `[&str; 2]` semicolon this function was added to step over.
    fn bracket_group_end(source: &str, at: usize) -> (usize, usize) {
        let bytes = source.as_bytes();
        let mut i = at;
        let mut depth = 0usize;
        let mut newlines = 0usize;
        while i < bytes.len() {
            // Mid-character bytes of a non-ASCII literal or comment: nothing
            // this walk opens on starts with one, and slicing there panics.
            if !source.is_char_boundary(i) {
                i += 1;
                continue;
            }
            let rest = &source[i..];

            if rest.starts_with("//") {
                i = rest.find('\n').map_or(bytes.len(), |n| i + n);
                continue;
            }
            if rest.starts_with("/*") {
                let mut nesting = 1usize;
                i += 2;
                while i < bytes.len() && nesting > 0 {
                    if bytes[i] == b'\n' {
                        newlines += 1;
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

            // Line 0 is passed in so the returned line is the newline count.
            if let Some((_, _, next)) = raw_string_at(source, i, 0) {
                newlines += next.1;
                i = next.0;
                continue;
            }
            if let Some((_, _, next)) = quoted_string_at(source, i, 0) {
                newlines += next.1;
                i = next.0;
                continue;
            }
            if bytes[i] == b'\'' {
                if let Some((_, next)) = char_literal_at(source, i) {
                    i = next;
                    continue;
                }
                i += 1;
                continue;
            }

            match bytes[i] {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return (i + 1, newlines);
                    }
                }
                b'\n' => newlines += 1,
                _ => {}
            }
            i += 1;
        }
        (bytes.len(), newlines)
    }

    const fn is_ident_start(b: u8) -> bool {
        b.is_ascii_alphabetic() || b == b'_'
    }

    fn ident_end(source: &str, at: usize) -> usize {
        let bytes = source.as_bytes();
        let mut i = at;
        while i < bytes.len() && is_ident_byte(bytes[i]) {
            i += 1;
        }
        i
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
        if matches!(bytes.get(i), Some(&b'b' | &b'c')) {
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
        if matches!(bytes.get(i), Some(&b'b' | &b'c')) {
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
                    // Copy `\` plus the escaped byte verbatim and resume
                    // after it. Only the two bytes are consumed here even for
                    // a longer escape such as `\u{2014}`: the rest of that
                    // escape is ASCII and lands in the recorded text as
                    // written, which is what `escape_denotes_non_ascii` then
                    // reads. Consuming the `\` with its next byte is what
                    // keeps `\\` from being mistaken for the start of one.
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
            // Start looking for the closing quote *past* the escaped byte, so
            // that `'\''` closes on its third quote rather than its second.
            // The whole escape is handed back, `\u{2014}` included, because
            // that is what `escape_denotes_non_ascii` needs to see.
            let after_escape = body_start + 2;
            let close = after_escape + source.get(after_escape..)?.find('\'')?;
            return Some((&source[body_start..close], close + 1));
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
    ///
    /// Note the scoping difference from [`non_ascii_literals`]: "test-only" is
    /// decided by the *including file's path*, not by whether the
    /// `include_str!` sits in a `#[cfg(test)]` block. So a production file
    /// that embeds a fixture from inside its own test module is still walked
    /// -- `crates/rocm-dash-tui/src/skills.rs:451` does exactly that, from
    /// inside the `#[cfg(test)] mod tests` beginning at `:415`, and passes
    /// only because that fixture is pure ASCII. A fixture carrying a real
    /// non-ASCII character would be a false positive here.
    ///
    /// That is deliberate, not an oversight. Scoping by call site would mean
    /// asking the hand-rolled test-span tracker in [`non_ascii_literals`]
    /// which embeds to *drop*, so a lexer bug there would silently blind this
    /// test to a real catalog -- a false negative, and the one failure mode
    /// this guard cannot survive. Path-based scoping is independent of the
    /// lexer, and the cost of being wrong is an `EXEMPT_EMBEDDED` entry with
    /// a one-line reason. If that false positive ever actually lands, prefer
    /// the entry; only move to call-site scoping if they start accumulating.
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

        // A test span is recorded by line, so a `use` group that was wrongly
        // treated as an item body would swallow anything sharing its line.
        // Only recognising the `use` keeps that line in scope.
        let crowded = "#[cfg(test)] use a::{b, c}; fn f() { let _ = \"\u{2014}\"; }\n";
        assert_eq!(lines(crowded).len(), 1, "{:?}", lines(crowded));
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

    /// An escaped quote does not end the literal, and an escape whose value is
    /// ASCII is not a finding.
    ///
    /// Named for what it checks: an earlier version of this fixture was called
    /// "escape sequences do not confuse the scanner" while only ever passing
    /// `\"` and `\u{41}` through it -- both ASCII, so it could not have caught
    /// the `\u{2014}` hole that
    /// [`an_escape_denoting_a_non_ascii_character_is_a_finding`] covers.
    #[test]
    fn an_escaped_quote_does_not_end_the_literal_and_an_ascii_escape_is_not_a_finding() {
        assert!(lines("fn f() { let _ = \"a \\\" b \\u{41} \\x41 \\n\"; }\n").is_empty());
        let found = lines("fn f() { let _ = \"a \\\" b\"; let _ = \"\u{2014}\"; }\n");
        assert_eq!(found.len(), 1, "{found:?}");
    }

    /// The hole this fixture exists for: `"\u{2014}"` is an em dash on the
    /// wire -- the same `E2 80 94` ROCMAI-543 is about -- while every byte of
    /// its *source* is ASCII. A guard that only calls `is_ascii()` on the
    /// literal text cannot see it, so a conversion reverted in this spelling
    /// would have been invisible.
    #[test]
    fn an_escape_denoting_a_non_ascii_character_is_a_finding() {
        let found = lines("fn f() { let _ = \"a \\u{2014} b\"; }\n");
        assert_eq!(found.len(), 1, "the em dash escape: {found:?}");
        assert!(
            found[0].1.contains("\\u{2014}"),
            "the finding carries the escape as written: {found:?}"
        );
        // Every surrounding form, so the fix is not just the one shape.
        assert_eq!(lines("fn f() { let _ = '\\u{2014}'; }\n").len(), 1);
        assert_eq!(lines("fn f() { let _ = \"\\u{2_014}\"; }\n").len(), 1);
        assert_eq!(lines("fn f() { let _ = \"\\u{85}\"; }\n").len(), 1);
        // `\xNN` above 7F is legal in a byte string and is a non-ASCII byte.
        assert_eq!(
            lines("fn f() { let _ = b\"a \\xe2\\x80\\x94 b\"; }\n").len(),
            1
        );
        assert_eq!(lines("fn f() { let _ = c\"a \\u{2014} b\"; }\n").len(), 1);
    }

    /// The other direction, which is what makes decoding the escape worth the
    /// code over treating every `\u{...}` as suspect: `\u{1b}`, `\u{7}` and
    /// `\u{41}` are ASCII and appear all over this tree as terminal control
    /// bytes. Flagging them would have forced dozens of exemptions that say
    /// nothing.
    #[test]
    fn an_escape_denoting_an_ascii_character_is_not_a_finding() {
        assert!(lines("fn f() { let _ = \"\\u{1b}[0m\\u{7}\"; }\n").is_empty());
        assert!(lines("fn f() { let _ = '\\u{0}'; }\n").is_empty());
        assert!(lines("fn f() { let _ = \"\\x7f\"; }\n").is_empty());
    }

    /// A backslash is not an escape inside a raw string, so `r"\u{2014}"` is
    /// the eight ASCII characters it looks like and must not be flagged --
    /// while a *real* em dash in one still is (see
    /// [`non_ascii_in_a_raw_string_is_a_finding`]).
    #[test]
    fn an_escape_written_in_a_raw_string_is_not_interpreted() {
        assert!(lines("fn f() { let _ = r\"a \\u{2014} b\"; }\n").is_empty());
        assert!(lines("fn f() { let _ = r#\"\\xe2\\x80\\x94\"#; }\n").is_empty());
        // Nor is an escaped backslash in an ordinary string: `\\` is consumed
        // as one escape, so the `u{2014}` after it is the ASCII text it is.
        assert!(lines("fn f() { let _ = \"a \\\\u{2014} b\"; }\n").is_empty());
    }

    /// `'\''` must close on its *third* quote. Closing on the second leaves a
    /// stray `'` that the next pass reads as the opening of something else --
    /// in `['\'','\u{2014}']` it pairs that quote with the comma, making a
    /// char literal `','` whose closing quote is really the em dash literal's
    /// *opening* one, so the em dash is scanned as code and never recorded.
    #[test]
    fn an_escaped_quote_char_literal_is_consumed_whole() {
        let found = lines("fn f() { let _ = ['\\'','\u{2014}']; }\n");
        assert_eq!(found.len(), 1, "the em dash after it: {found:?}");
        assert_eq!(found[0].1, "\u{2014}", "{found:?}");
    }

    /// `#[cfg(test)] const X: Foo = Foo { .. };` closes its brace group with a
    /// `;` exactly as a `use` group does, so a rule that reads the trailing
    /// punctuation instead of the item keyword calls it production code and
    /// wrongly flags a test-only fixture.
    #[test]
    fn a_cfg_test_const_with_a_braced_initialiser_is_a_test_item() {
        let source = "#[cfg(test)]\nconst X: Foo = Foo { label: \"\u{2014}\" };\n\nfn f() {}\n";
        assert!(lines(source).is_empty(), "{:?}", lines(source));

        // Same for a visibility-qualified `static`: `pub` is not `use`.
        let qualified = "#[cfg(test)]\npub(crate) static X: Foo = Foo { label: \"\u{2014}\" };\n";
        assert!(lines(qualified).is_empty(), "{:?}", lines(qualified));
    }

    /// The same misclassification with no braces at all: a `const` whose value
    /// is just a string also ends in `;`, so the item itself has to be
    /// recorded as a test span rather than the attribute merely marked spent.
    #[test]
    fn a_brace_less_cfg_test_const_is_a_test_item() {
        let source = "#[cfg(test)]\nconst X: &str = \"\u{2014}\";\n\nfn f() {}\n";
        assert!(lines(source).is_empty(), "{:?}", lines(source));
    }

    /// ...and one whose type is an array, where the `;` of `[&str; 2]` sits
    /// inside the item rather than ending it. Reading that `;` as the end
    /// closes the test span early, and the fixture data after it is reported
    /// as production output.
    #[test]
    fn an_array_type_semicolon_does_not_end_a_test_item_early() {
        let source = "#[cfg(test)]\nstatic X: [&str; 2] = [\n    \"\u{2014}\",\n    \"b\",\n];\n";
        assert!(lines(source).is_empty(), "{:?}", lines(source));
    }

    /// ...and the inverse of that stepping-over: a `]` *inside* one of the
    /// array's string elements is text, not the group's close.
    ///
    /// This fixture asserts the **consequence**, not an offset. Counting that
    /// `]` ended the group one character early and resumed the main scan
    /// inside the element, so the element's real closing `"` was read as an
    /// *opening* quote; from there every quote in the file is paired off by
    /// one and the scan swallows whole functions as if they were one long
    /// string. The em dash on line 5 is production output in every case
    /// below, and before the fix each one reported **zero** findings -- a
    /// silent miss, which is the only failure mode a guard like this cannot
    /// survive. The test item itself (lines 1-2) must still be skipped.
    #[test]
    fn a_bracket_inside_a_literal_does_not_hide_later_production_text() {
        const PRODUCTION: &str = "\n\nfn f() {\n    let _ = \"\u{2014}\";\n}\n";
        for array in [
            // An ordinary string element holding a `]`.
            "[\"a]b\", \"c\"]",
            // ...with the element after it an escaped em dash, the shape that
            // also fed the `{` and `}` of `\\u{2014}` to the brace tracking.
            "[\"a]b\", \"\\u{2014}\"]",
            // A raw string, which has no escapes to hide behind.
            "[r#\"a]b\"#, \"c\"]",
            // A byte string: the `b` prefix is what the desynced scan latched
            // onto as a literal opener.
            "[b\"a]b\", b\"c\"]",
            // A `]` preceded by an escaped quote inside the element.
            "[\"a\\\"]b\", \"c\"]",
        ] {
            let source = format!("#[cfg(test)]\nstatic X: [&str; 2] = {array};{PRODUCTION}");
            let found = lines(&source);
            assert_eq!(
                found.len(),
                1,
                "the production literal after {array} must still be reported: {found:?}"
            );
            assert_eq!(found[0].0, 5, "line of the production literal: {found:?}");
            assert_eq!(found[0].1, "\u{2014}", "{found:?}");
        }
    }

    /// The helper itself, at the offsets the fixture above can only show
    /// indirectly. Comments are included for completeness rather than because
    /// a miss was reproducible through them: a `]` in a comment did return a
    /// mid-comment offset before the fix, but the main loop happened to
    /// resynchronise on the group's real `]` in every shape tried, so no
    /// hidden production literal could be demonstrated for that case.
    #[test]
    fn bracket_group_end_counts_brackets_on_code_bytes_only() {
        // Each group is a `[..]` followed by a `!` sentinel; the walk must
        // stop exactly on the sentinel however the inner `]`s are spelled.
        for group in [
            "[\"a]b\", \"c\"]!",
            "[r#\"a]b\"#, \"c\"]!",
            "[b\"a]b\"]!",
            "[\"a\\\"]b\"]!",
            "[']']!",
            "[/* ] */ \"a\"]!",
            "[\n    // a ] in prose\n    \"a\",\n]!",
            "[[\"a]b\"], [\"c\"]]!",
        ] {
            let (end, newlines) = bracket_group_end(group, 0);
            assert_eq!(
                end,
                group.len() - 1,
                "must stop just past the real closing bracket of {group:?}"
            );
            assert_eq!(
                newlines,
                group[..end].matches('\n').count(),
                "newlines crossed in {group:?}"
            );
        }
    }

    /// A multi-line test attribute must not throw the line counter off; the
    /// finding's line number is the only thing that locates it.
    #[test]
    fn a_multi_line_test_attribute_keeps_the_line_count_straight() {
        let source = "#[tokio::test(\n    flavor = \"multi_thread\"\n)]\nasync fn t() {}\n\nfn f() {\n    let _ = \"\u{2014}\";\n}\n";
        let found = lines(source);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].0, 7, "line of the production literal: {found:?}");
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
