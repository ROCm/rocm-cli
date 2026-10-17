// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Every `rocm …` / `rocmd …` command this repository tells a user to run must
//! exist and accept the flags it is given (AGENTS.md §3: remediation advice
//! naming a command must name one that parses).
//!
//! Example tests pin the *wording* of advice; nothing runs the advised command,
//! so a renamed flag or a removed subcommand strands users silently. This
//! module closes that gap mechanically: it walks every user-facing surface,
//! extracts each invocation it names, substitutes placeholders with values the
//! advice implies are valid, and routes the result through the same entry
//! points `run()` uses — the natural-language router, then the real clap tree.
//!
//! Surfaces scanned:
//! - the long `--help` of every visible command, rendered from the real clap
//!   tree, so doc-comment help, `long_about` and `after_help` EXAMPLES are
//!   checked at their source of truth;
//! - production Rust string literals under `apps/`, `crates/` and `engines/`:
//!   backtick spans naming `rocm …`, and literals that *start* with a command
//!   (the `rocm_core::fix` RECIPES `commands`/`verify` fields, dashboard
//!   "Runs:" lines). Whole-line `//` comments, items gated on `cfg(test)`
//!   or `cfg(all(test, …))`, and whole files declared only as test modules
//!   are skipped — they are never printed. A trailing comment or `/* … */`
//!   block on a line of code is still scanned;
//! - `README.md`, `docs/**/*.md` and `skills/**/*.md` (inline backtick spans
//!   and fenced code-block lines), and the VHS tapes under `docs/tapes/`.
//!
//! Out of scope by construction: extraction anchors on a leading `rocm ` /
//! `rocmd ` word, so commands for other tools (`apt`, `uv`, `amd-smi`,
//! `HIP_VISIBLE_DEVICES=…`) are never picked up. Contributor tooling
//! (`xtask/`, `crates/e2e-report/`, `tests/`) is not scanned: it is not shown to
//! users of the CLI.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use clap::FromArgMatches;
use clap::error::{ContextKind, ContextValue, ErrorKind};

use super::{
    Cli, cli_command, command_invocation_error, parse_freeform_invocation, should_treat_as_freeform,
};

/// What kind of text an invocation was found in. It decides whether leaving
/// out a required value is acceptable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Surface {
    /// An inline backtick span in prose, which may name a command or flag
    /// without its values ("pass `rocm serve --engine`").
    InlineProse,
    /// A line the user is meant to run as written: a fenced code line, a
    /// string literal that starts with a command (RECIPES `commands`/`verify`,
    /// a dashboard `cmd`, a tape `Type`), a labelled `next step:`/`Try:`/
    /// `apply with:` line, or a help EXAMPLES row.
    CommandLine,
}

/// One advised invocation and where it came from.
#[derive(Debug, Clone)]
pub(crate) struct Advice {
    pub source: String,
    pub raw: String,
    pub surface: Surface,
}

/// How the real entry points treat an advised argv.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Parses into a structured command (or prints help/version).
    Parses,
    /// Names a real command or flag but leaves a required value off: a
    /// *reference* to a command in prose ("pass `rocm serve --engine`"), not a
    /// full command line. Every token it names exists, so it is not a
    /// violation.
    IncompleteReference,
    /// Sent to the natural-language planner instead of clap.
    Freeform,
    /// clap rejects it: unknown subcommand or flag, invalid value, conflicting
    /// flags, too many positionals.
    Rejected(String),
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("apps/rocm sits two levels below the repo root")
        .to_path_buf()
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

fn starts_with_invocation(text: &str) -> bool {
    let text = text.trim_start();
    text.starts_with("rocm ") || text.starts_with("rocmd ")
}

/// Closed inline `` `rocm …` `` spans on one line. An unclosed trailing span is
/// ignored: callers join wrapped lines first.
fn backtick_spans(line: &str) -> Vec<String> {
    let parts: Vec<&str> = line.split('`').collect();
    let mut spans = Vec::new();
    // Parts at odd indexes are inside backticks; the last part is only closed
    // when the split produced an odd number of parts.
    let closed_inside = if parts.len() % 2 == 1 {
        parts.len()
    } else {
        parts.len() - 1
    };
    for part in parts.iter().take(closed_inside).skip(1).step_by(2) {
        let candidate = part.strip_prefix("$ ").unwrap_or(part);
        if starts_with_invocation(candidate) {
            spans.push(candidate.trim().to_owned());
        }
    }
    spans
}

/// Unquoted invocations inside one string literal's content:
/// - the literal *starts* with a command — the `commands`/`verify` fields of
///   `rocm_core::fix` RECIPES, a dashboard `cmd: "rocm update"`, a VHS
///   `Type "rocm examine"`; a leading `#` (a commented-out command shown to
///   the user) is tolerated;
/// - a labelled line — `next step: rocm …`, `stop: rocm …`, `apply with:
///   rocm …`, `Try: rocm …` — the house style for remediation output;
/// - a command run over ssh — `ssh {target} -- rocm …`.
///
/// Each `\n`-separated line of the literal is considered on its own.
fn literal_invocations(content: &str) -> Vec<String> {
    let mut found = Vec::new();
    for segment in content.split("\\n") {
        // A commented-out command (`# rocm …`) keeps no indentation of its own.
        let uncommented = segment.trim_start_matches('#');
        let start_stripped = if uncommented.len() == segment.len() {
            segment
        } else {
            uncommented.trim_start()
        };
        if let Some(command) = examples_row(start_stripped) {
            found.push(command);
            continue;
        }
        let mut search = 0;
        while let Some(offset) = segment[search..].find("rocm") {
            let at = search + offset;
            search = at + 4;
            let rest = &segment[at..];
            if !starts_with_invocation(rest) {
                continue;
            }
            let before = segment[..at].trim_end();
            if before.ends_with(':') || before.ends_with(" --") || before.ends_with(" -- '") {
                let command = rest.trim_end().trim_end_matches('\'');
                found.push(command.to_owned());
            }
        }
    }
    found
}

/// Invocations named by the string literals on one source line.
fn command_literals(line: &str) -> Vec<String> {
    let bytes = line.as_bytes();
    let mut found = Vec::new();
    let mut index = 0;
    while let Some(offset) = line[index..].find('"') {
        let start = index + offset + 1;
        let mut end = start;
        let mut escaped = false;
        while end < bytes.len() {
            match bytes[end] {
                b'\\' if !escaped => escaped = true,
                b'"' if !escaped => break,
                _ => escaped = false,
            }
            end += 1;
        }
        found.extend(literal_invocations(&line[start..end.min(line.len())]));
        if end + 1 >= line.len() {
            break;
        }
        index = end + 1;
    }
    found
}

/// Whether `attr` is a whole-line `#[cfg(...)]` that compiles its item only
/// under test: `#[cfg(test)]` or `#[cfg(all(test, ...))]`.
fn is_test_cfg(attr: &str) -> bool {
    (attr.starts_with("#[cfg(test)") || attr.starts_with("#[cfg(all(test")) && attr.ends_with(")]")
}

/// Files that `files` declare only as test modules (`#[cfg(test)] mod x;`,
/// `#[cfg(all(test, unix))] mod x;`, following a `#[path = ...]` when one is
/// given). Like a `*tests.rs` file they are test code throughout, but their
/// names say nothing about it, so the declaration is the only reliable sign.
fn test_only_module_files(files: &[PathBuf]) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let Some(dir) = file.parent() else {
            continue;
        };
        let is_mod_root = matches!(
            file.file_name().and_then(|n| n.to_str()),
            Some("main.rs" | "lib.rs" | "mod.rs")
        );
        let child_dir = if is_mod_root {
            dir.to_path_buf()
        } else {
            dir.join(file.file_stem().unwrap_or_default())
        };
        let mut attrs: Vec<&str> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("#[") {
                attrs.push(line);
                continue;
            }
            if line.starts_with("//") {
                continue;
            }
            let declared = line
                .strip_prefix("pub(crate) ")
                .or_else(|| line.strip_prefix("pub "))
                .unwrap_or(line)
                .strip_prefix("mod ")
                .and_then(|rest| rest.strip_suffix(';'));
            if let Some(name) = declared
                && attrs.iter().any(|attr| is_test_cfg(attr))
            {
                let explicit = attrs.iter().find_map(|attr| {
                    attr.strip_prefix("#[path = \"")
                        .and_then(|rest| rest.strip_suffix("\"]"))
                });
                if let Some(path) = explicit {
                    found.push(dir.join(path));
                } else {
                    found.push(child_dir.join(format!("{name}.rs")));
                    found.push(child_dir.join(name).join("mod.rs"));
                }
            }
            attrs.clear();
        }
    }
    found
        .into_iter()
        .filter_map(|path| std::fs::canonicalize(path).ok())
        .collect()
}

/// Production lines of a Rust file, with string continuations (`\` at end of
/// line) joined the way the compiler joins them. Whole-line `//` comments
/// (doc comments included) and `#[cfg(test)]` items are dropped: they are
/// never printed to a user. A trailing `// …` comment or a `/* … */` block on
/// a line of code is kept, and its backtick spans are scanned.
fn rust_production_lines(text: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut kept: Vec<(usize, String)> = Vec::new();
    let mut continuing = false;
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        if is_test_cfg(line.trim()) {
            // Skip the gated item by brace depth (or to its `;`).
            let mut depth = 0i32;
            let mut seen_open = false;
            index += 1;
            while index < lines.len() {
                for ch in lines[index].chars() {
                    match ch {
                        '{' => {
                            depth += 1;
                            seen_open = true;
                        }
                        '}' => depth -= 1,
                        _ => {}
                    }
                }
                let ends_item = !seen_open && lines[index].trim_end().ends_with(';');
                index += 1;
                if (seen_open && depth <= 0) || ends_item {
                    break;
                }
            }
            continuing = false;
            continue;
        }
        let is_comment = line.trim_start().starts_with("//");
        if continuing && !is_comment {
            let (_, joined) = kept.last_mut().expect("a continued line exists");
            joined.pop(); // the trailing `\`
            joined.push_str(line.trim_start());
        } else if !is_comment {
            kept.push((index + 1, line.trim_end().to_owned()));
        }
        continuing = !is_comment && line.trim_end().ends_with('\\');
        index += 1;
    }
    kept
}

fn extract_rust(path: &Path, rel: &str, out: &mut Vec<Advice>) {
    let text = std::fs::read_to_string(path).expect("read Rust source");
    for (line_no, line) in rust_production_lines(&text) {
        let mut seen: Vec<String> = Vec::new();
        let spans = backtick_spans(&line)
            .into_iter()
            .map(|raw| (raw, Surface::InlineProse));
        let literals = command_literals(&line)
            .into_iter()
            .map(|raw| (raw, Surface::CommandLine));
        for (raw, surface) in spans.chain(literals) {
            if seen.contains(&raw) {
                continue;
            }
            seen.push(raw.clone());
            out.push(Advice {
                source: format!("{rel}:{line_no}"),
                raw,
                surface,
            });
        }
    }
}

fn extract_markdown(path: &Path, rel: &str, out: &mut Vec<Advice>) {
    let text = std::fs::read_to_string(path).expect("read markdown");
    extract_markdown_text(&text, rel, out);
}

fn extract_markdown_text(text: &str, rel: &str, out: &mut Vec<Advice>) {
    let fenced = |start: usize, raw: String| Advice {
        source: format!("{rel}:{start}"),
        raw,
        surface: Surface::CommandLine,
    };
    let mut in_fence = false;
    // A fenced command continued with `\` (sh) or `` ` `` (PowerShell).
    let mut pending: Option<(usize, String)> = None;
    // Prose with an inline span wrapped across lines.
    let mut prose: Option<(usize, String)> = None;
    for (index, line) in text.lines().enumerate() {
        let line_no = index + 1;
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            // A continuation the fence cut off is still a command; checking it
            // as it stands beats gluing the next fence's first line onto it.
            if let Some((start, joined)) = pending.take() {
                out.push(fenced(start, joined));
            }
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            let continued = trimmed.ends_with('\\') || trimmed.ends_with(" `");
            let piece = trimmed.trim_end_matches(['\\', '`']).trim_end();
            if let Some((start, mut joined)) = pending.take() {
                joined.push(' ');
                joined.push_str(piece);
                if continued {
                    pending = Some((start, joined));
                } else {
                    out.push(fenced(start, joined));
                }
                continue;
            }
            let command = piece
                .strip_prefix("$ ")
                .or_else(|| piece.strip_prefix("PS> "))
                .unwrap_or(piece);
            if starts_with_invocation(command) {
                if continued {
                    pending = Some((line_no, command.to_owned()));
                } else {
                    out.push(fenced(line_no, command.to_owned()));
                }
            }
            continue;
        }
        let (start, joined) = match prose.take() {
            Some((start, mut joined)) => {
                joined.push(' ');
                joined.push_str(trimmed);
                (start, joined)
            }
            None => (line_no, line.to_owned()),
        };
        // An odd backtick count means a span wraps onto the next line; a blank
        // line ends the paragraph either way.
        if joined.matches('`').count() % 2 == 1 && !trimmed.is_empty() {
            prose = Some((start, joined));
            continue;
        }
        push_spans(start, &joined, rel, out);
    }
    // The file ended inside a continued command or a wrapped paragraph.
    if let Some((start, joined)) = pending {
        out.push(fenced(start, joined));
    }
    if let Some((start, joined)) = prose {
        push_spans(start, &joined, rel, out);
    }
}

fn push_spans(start: usize, prose: &str, rel: &str, out: &mut Vec<Advice>) {
    for raw in backtick_spans(prose) {
        out.push(Advice {
            source: format!("{rel}:{start}"),
            raw,
            surface: Surface::InlineProse,
        });
    }
}

fn walk(dir: &Path, extensions: &[&str], skip: &[&str], files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if skip.contains(&name) {
            continue;
        }
        if path.is_dir() {
            walk(&path, extensions, skip, files);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| extensions.contains(&ext))
        {
            files.push(path);
        }
    }
}

/// Every advised invocation on a source surface. Help text is collected
/// separately by [`help_text_advice`].
pub(crate) fn source_advice() -> Vec<Advice> {
    let root = repo_root();
    let rel = |path: &Path| {
        path.strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    };
    let mut advice = Vec::new();

    // `tests` directories and `*tests.rs` files are test code; `target` is
    // build output; `e2e-report` renders CI reports for contributors.
    let rust_skip = ["target", "tests", "e2e-report", "benches"];
    let mut rust_files = Vec::new();
    for top in ["apps", "crates", "engines"] {
        walk(&root.join(top), &["rs"], &rust_skip, &mut rust_files);
    }
    let test_only = test_only_module_files(&rust_files);
    for path in rust_files {
        if std::fs::canonicalize(&path).is_ok_and(|path| test_only.contains(&path)) {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        // This module's own fixtures are not advice.
        if name.ends_with("tests.rs") || name == "build.rs" {
            continue;
        }
        extract_rust(&path, &rel(&path), &mut advice);
    }

    let mut markdown = vec![root.join("README.md")];
    walk(&root.join("docs"), &["md"], &[], &mut markdown);
    walk(&root.join("skills"), &["md"], &[], &mut markdown);
    for path in markdown {
        extract_markdown(&path, &rel(&path), &mut advice);
    }

    let mut tapes = Vec::new();
    walk(&root.join("docs").join("tapes"), &["tape"], &[], &mut tapes);
    for path in tapes {
        let text = std::fs::read_to_string(&path).expect("read tape");
        for (index, line) in text.lines().enumerate() {
            if line.trim_start().starts_with('#') {
                continue;
            }
            for raw in command_literals(line) {
                advice.push(Advice {
                    source: format!("{}:{}", rel(&path), index + 1),
                    raw,
                    surface: Surface::CommandLine,
                });
            }
        }
    }
    advice
}

/// The command on a line that starts with one. An *indented* line is a row of
/// an EXAMPLES table (`  rocm examine      Check GPU …`): clap renders a
/// description column after a run of spaces, so the command ends at the first
/// double space. Only there: on any other line a double space is ordinary
/// whitespace inside a command, and cutting at it would hide what follows
/// (`rocm install sdk --channel release  --bogus-flag`).
///
/// The rows are found twice, by design: in the rendered `--help`, and in the
/// `after_help` string literal they are written in.
fn examples_row(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !starts_with_invocation(trimmed) {
        return None;
    }
    let indented = line.starts_with(char::is_whitespace);
    let command = match trimmed.split_once("  ") {
        Some((command, _)) if indented => command,
        _ => trimmed,
    };
    Some(command.trim_end().to_owned())
}

/// Every invocation named in the long `--help` of every visible command.
pub(crate) fn help_text_advice() -> Vec<Advice> {
    help_advice_of(cli_command())
}

/// [`help_text_advice`] for any command tree, so a fixture tree can test it.
fn help_advice_of(mut root: clap::Command) -> Vec<Advice> {
    fn visit(command: &mut clap::Command, path: &str, out: &mut Vec<Advice>) {
        let help = command.render_long_help().to_string();
        let push_spans = |start: usize, prose: &str, out: &mut Vec<Advice>| {
            for raw in backtick_spans(prose) {
                out.push(Advice {
                    source: format!("`{path} --help` line {start}"),
                    raw,
                    surface: Surface::InlineProse,
                });
            }
        };
        // Help text with an inline span broken across lines (an authored
        // `\n` inside one), joined as Markdown prose is.
        let mut prose: Option<(usize, String)> = None;
        for (index, line) in help.lines().enumerate() {
            let line_no = index + 1;
            if let Some(raw) = examples_row(line) {
                out.push(Advice {
                    source: format!("`{path} --help` line {line_no}"),
                    raw,
                    surface: Surface::CommandLine,
                });
            }
            let trimmed = line.trim();
            let (start, joined) = match prose.take() {
                Some((start, mut joined)) => {
                    joined.push(' ');
                    joined.push_str(trimmed);
                    (start, joined)
                }
                None => (line_no, line.to_owned()),
            };
            if joined.matches('`').count() % 2 == 1 && !trimmed.is_empty() {
                prose = Some((start, joined));
                continue;
            }
            push_spans(start, &joined, out);
        }
        if let Some((start, joined)) = prose {
            push_spans(start, &joined, out);
        }
        // clap's generated `help` subcommand renders the same help again; one
        // bad line would be reported once per nesting level.
        let names: Vec<String> = command
            .get_subcommands()
            .filter(|sub| !sub.is_hide_set() && sub.get_name() != "help")
            .map(|sub| sub.get_name().to_owned())
            .collect();
        for name in names {
            let sub = command
                .find_subcommand_mut(&name)
                .expect("subcommand listed above");
            visit(sub, &format!("{path} {name}"), out);
        }
    }
    root.build();
    let mut out = Vec::new();
    let path = root.get_name().to_owned();
    visit(&mut root, &path, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Normalisation: from an advised string to argv variants
// ---------------------------------------------------------------------------

/// Split a shell list into its commands at `&&`, `||`, `;` and `|`. A
/// separator inside quotes, a `<…>` placeholder, a `[…]` optional group or a
/// `{…}` template is notation, not a separator, and so is a `|` joined to a
/// word (`stop|restart`): a pipe stands alone between spaces.
fn split_shell_list(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut commands = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut closers: Vec<char> = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let ch = chars[index];
        let previous = index.checked_sub(1).map(|i| chars[i]);
        let next = chars.get(index + 1).copied();
        let at_word_start = previous.is_none_or(char::is_whitespace);
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
            current.push(ch);
            index += 1;
            continue;
        }
        let separator_len = match (ch, next) {
            _ if !closers.is_empty() => 0,
            ('&', Some('&')) | ('|', Some('|')) => 2,
            (';', _) => 1,
            ('|', _) if at_word_start && next.is_none_or(char::is_whitespace) => 1,
            _ => 0,
        };
        if separator_len > 0 {
            commands.push(std::mem::take(&mut current));
            index += separator_len;
            continue;
        }
        match ch {
            '"' | '\'' => quote = Some(ch),
            // `<` opens a placeholder only at a word start and before a
            // non-space: ` < file` is a redirection.
            '<' if at_word_start && next.is_some_and(|c| !c.is_whitespace()) => closers.push('>'),
            '[' => closers.push(']'),
            '{' => closers.push('}'),
            c if closers.last() == Some(&c) => {
                closers.pop();
            }
            _ => {}
        }
        current.push(ch);
        index += 1;
    }
    commands.push(current);
    commands
}

/// The commands an advised string names, each cut down to the command
/// itself: a shell comment or escaped newline ends the whole line; each
/// command of a shell list is kept when it runs `rocm`/`rocmd` (commands for
/// other tools are not this contract's business) and is then stripped of
/// redirections and trailing prose.
pub(crate) fn command_parts(raw: &str) -> Vec<String> {
    let mut text = raw.trim();
    for marker in ["\\n", " # "] {
        if let Some(position) = text.find(marker) {
            text = &text[..position];
        }
    }
    let mut parts = Vec::new();
    for command in split_shell_list(text) {
        let command = command.trim();
        if !(starts_with_invocation(command) || command == "rocm" || command == "rocmd") {
            continue;
        }
        let mut command = command.to_owned();
        for marker in [
            " >> ", " > ", " < ", " 2>", " (", " —", " –", " → ", " before ", " then ",
        ] {
            if let Some(position) = command.find(marker) {
                command.truncate(position);
            }
        }
        let command = command.trim().trim_end_matches([',', ':']).trim();
        if !command.is_empty() {
            parts.push(command.to_owned());
        }
    }
    parts
}

/// Shell-like word split honouring single and double quotes. A `<…>`
/// placeholder is one word even when it contains spaces
/// (`<natural language request>`).
fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut has_word = false;
    for ch in text.chars() {
        match (quote, ch) {
            (Some('>'), '>') => {
                current.push('>');
                quote = None;
            }
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => current.push(c),
            (None, '"' | '\'') => {
                quote = Some(ch);
                has_word = true;
            }
            (None, '<') if current.is_empty() || current.ends_with(['[', ':', '#']) => {
                current.push('<');
                quote = Some('>');
            }
            (None, c) if c.is_whitespace() => {
                if has_word || !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                    has_word = false;
                }
            }
            (None, c) => current.push(c),
        }
    }
    if has_word || !current.is_empty() {
        words.push(current);
    }
    words
}

/// The value a placeholder stands for. `previous` is the word before it, which
/// disambiguates bare `{}` / `<id>` placeholders. Values are ones the advice
/// implies are valid: a recognised TheRock family, a real engine, a real shell.
pub(crate) fn placeholder_value(name: &str, previous: &str) -> String {
    let key = name
        .trim_matches(['<', '>', '{', '}'])
        .trim()
        .to_ascii_lowercase();
    let by_previous = match previous {
        "fix" => Some("fix-1-arch"),
        "--distro" => Some("ubuntu"),
        "--engine" => Some("vllm"),
        "--channel" => Some("release"),
        "--format" => Some("wheel"),
        "--family" => Some(rocm_core::known_therock_families()[0]),
        "--provider" => Some("openai"),
        "--runtime" | "--runtime-id" | "activate" | "uninstall" => Some("runtime-key-1"),
        "--service" | "stop" | "restart" | "logs" | "remove" => Some("svc-1"),
        "--mode" => Some("propose"),
        "--endpoint" => Some("http://127.0.0.1:8000"),
        "--replay" => Some("/tmp/replay.json"),
        "--keep" | "--top" | "--device-index" | "--older-than-hours" => Some("1"),
        "--concurrency" => Some("1,2"),
        "--artifact-max-bytes" => Some("1048576"),
        "--local-webhook-port" | "--port" => Some("8080"),
        "--host" => Some("127.0.0.1"),
        "--version" => Some("7.0.0"),
        "--yes" => Some("start a local model"),
        "completions" => Some("bash"),
        "doctor" => Some("box-1"),
        _ => None,
    };
    if let Some(value) = by_previous {
        return value.to_owned();
    }
    let value = match key.as_str() {
        k if k.contains("family") => rocm_core::known_therock_families()[0],
        "model" | "base" | "owner/repo" => "Qwen/Qwen3-0.6B",
        "quant" => "Q4_0",
        k if k.contains("fix") => "fix-1-arch",
        k if k.contains("key") || k.contains("runtime") || k == "candidate" => "runtime-key-1",
        k if k.contains("service") || k.contains("session") || k == "id" => "svc-1",
        "engine" => "vllm",
        "shell" => "bash",
        "url" => "http://127.0.0.1:8000",
        "machine" | "target" => "box-1",
        "provider" | "name" => "openai",
        "port" => "8080",
        "version" => "7.0.0",
        "host" => "127.0.0.1",
        "mode" => "propose",
        "watcher" => "server-recover",
        "n" | "bytes" => "1",
        "file" | "missing" => "/tmp/replay.json",
        "command" | "args" => "examine",
        k if k.contains("request") || k.contains("text") || k.contains("error") => {
            "start a local model"
        }
        _ => "x1",
    };
    value.to_owned()
}

/// Whether a bare word is synopsis notation for a value (`URL`, `NAME`, `N`).
fn is_upper_placeholder(word: &str) -> bool {
    word.chars()
        .all(|c| c.is_ascii_uppercase() || c == '_' || c == '-')
        && word.chars().any(|c| c.is_ascii_uppercase())
}

fn substitute_placeholders(word: &str, previous: &str) -> String {
    if is_upper_placeholder(word)
        || word.starts_with("N,")
        || (word.starts_with('<') && word.ends_with('>'))
        || (word.starts_with('{') && word.ends_with('}'))
    {
        return placeholder_value(word, previous);
    }
    // Embedded placeholders: `<owner/repo>:<quant>`, `svc-{id}`.
    let mut out = String::new();
    let mut rest = word;
    while let Some(open) = rest.find(['<', '{']) {
        let close_char = if rest.as_bytes()[open] == b'<' {
            '>'
        } else {
            '}'
        };
        let Some(close) = rest[open..].find(close_char) else {
            break;
        };
        out.push_str(&rest[..open]);
        out.push_str(&placeholder_value(&rest[open..=open + close], previous));
        rest = &rest[open + close + 1..];
    }
    out.push_str(rest);
    out
}

/// Whether a word is a placeholder for options the advice spells out elsewhere
/// (`<case-appropriate options above>`). It stands for zero or more flags the
/// reader picks, so it contributes no words: the command around it is still
/// checked, rather than failing on an invented value for a prose placeholder.
fn is_options_placeholder(word: &str) -> bool {
    word.starts_with('<') && word.ends_with('>') && word.to_ascii_lowercase().contains("options")
}

fn is_ellipsis(word: &str) -> bool {
    word == "…" || word == "..."
}

/// Alternatives a single word stands for: `enable|disable`, or
/// `activate/rollback` in a subcommand position. Ellipsis alternatives
/// (`openai|...`) are dropped.
fn word_alternatives(word: &str, in_subcommand_position: bool) -> Vec<String> {
    let alternatives: Vec<String> = if word.contains('|') && !word.starts_with('<') {
        word.split('|').map(str::to_owned).collect()
    } else if in_subcommand_position
        && word.contains('/')
        && word
            .split('/')
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_lowercase()))
    {
        // Model ids (`Qwen/Qwen3`) carry uppercase and never sit here.
        word.split('/').map(str::to_owned).collect()
    } else {
        vec![word.to_owned()]
    };
    alternatives
        .into_iter()
        .filter(|alt| !alt.is_empty() && !is_ellipsis(alt))
        .collect()
}

/// What `rocm --yes …` stands for: the request form, spelled out the way the
/// docs do, so it reads as an advised natural-language request.
const NATURAL_LANGUAGE_REQUEST: &str = "<natural language request>";

/// A synopsis item: a required word (with alternatives) or an optional
/// `[...]` group (with `|`-separated alternative word lists).
enum Item {
    Required(Vec<String>),
    Optional(Vec<Vec<String>>),
}

fn synopsis_items(words: &[String]) -> Vec<Item> {
    let mut items = Vec::new();
    let mut group: Option<Vec<Vec<String>>> = None;
    // Bracket depth inside `group`: a nested `[...]` (`[--restart-services
    // [--yes]]`, `[--distro [NAME]]`) is optional *within* the outer group.
    let mut depth = 0_usize;
    // Where the alternative being written starts in the group's forms: a
    // value list or a nested group turns it into several forms.
    let mut branch = 0_usize;
    for (index, raw_word) in words.iter().enumerate() {
        let opening = raw_word.chars().take_while(|c| *c == '[').count();
        let closing = raw_word.chars().rev().take_while(|c| *c == ']').count();
        let word = raw_word.trim_matches(['[', ']']);
        if opening > 0 && group.is_none() {
            group = Some(vec![Vec::new()]);
            branch = 0;
        }
        if let Some(alternatives) = group.as_mut() {
            if opening > 0 && depth > 0 {
                // The words before a nested group are a form of their own:
                // `[--restart-services [--yes]]` allows `--restart-services`
                // alone as well as with `--yes`.
                let forms: Vec<Vec<String>> = alternatives[branch..]
                    .iter()
                    .filter(|form| !form.is_empty())
                    .cloned()
                    .collect();
                let count = forms.len();
                alternatives.splice(branch..branch, forms);
                branch += count;
            }
            depth += opening;
            if word == "|" {
                alternatives.push(Vec::new());
                branch = alternatives.len() - 1;
            } else if !word.is_empty() && !is_ellipsis(word) {
                // A value list (`[--provider anthropic|openai|...]`, `[stop|restart
                // <id>]`) branches every form of the current alternative, and
                // the words after it extend each branch.
                let values = word_alternatives(word, false);
                if !values.is_empty() {
                    let forms: Vec<Vec<String>> = alternatives.drain(branch..).collect();
                    for form in forms {
                        for value in &values {
                            let mut extended = form.clone();
                            extended.push(value.clone());
                            alternatives.push(extended);
                        }
                    }
                }
            }
            depth = depth.saturating_sub(closing);
            if depth == 0 {
                let alternatives = group.take().expect("open group");
                items.push(Item::Optional(
                    alternatives.into_iter().filter(|a| !a.is_empty()).collect(),
                ));
            }
            continue;
        }
        if is_ellipsis(word) {
            // `rocm --yes ...`: the ellipsis stands for the request.
            if index > 0 && words[index - 1] == "--yes" {
                items.push(Item::Required(vec![NATURAL_LANGUAGE_REQUEST.to_owned()]));
            }
            continue;
        }
        let in_subcommand_position = items.len() <= 1;
        let alternatives = word_alternatives(word, in_subcommand_position);
        if !alternatives.is_empty() {
            items.push(Item::Required(alternatives));
        }
    }
    items
}

/// One argv word, and the advice word it was filled in from: the same text,
/// or a placeholder (`<TEXT>`) that [`placeholder_value`] substituted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Word {
    pub value: String,
    pub advised: String,
}

/// The argv variants one command (an item of [`command_parts`]) stands for.
/// Synopsis notation is expanded: each `[...]` optional group is tried on its
/// own (groups may be mutually exclusive, as in `rocm dash [--demo] [--replay
/// <file>]`), each `a|b` alternative produces a variant, `…`/`...` are
/// dropped, and placeholders are substituted.
fn command_variants(command: &str) -> Vec<Vec<Word>> {
    let mut words = split_words(command);
    if words.is_empty() {
        return Vec::new();
    }
    let program = words.remove(0);
    let items = synopsis_items(&words);

    // `None` = required words only; `Some((group, alternative))` adds one
    // optional group's alternative.
    let mut selections: Vec<Option<(usize, usize)>> = vec![None];
    let mut group_count = 0;
    for item in &items {
        if let Item::Optional(alternatives) = item {
            for alternative in 0..alternatives.len() {
                selections.push(Some((group_count, alternative)));
            }
            group_count += 1;
        }
    }

    let start = Word {
        value: program.clone(),
        advised: program,
    };
    let mut variants: Vec<Vec<Word>> = Vec::new();
    for selection in selections {
        let mut partial: Vec<Vec<Word>> = vec![vec![start.clone()]];
        let mut group_index = 0;
        for item in &items {
            let choices: Vec<Vec<String>> = match item {
                Item::Required(alternatives) => {
                    alternatives.iter().map(|alt| vec![alt.clone()]).collect()
                }
                Item::Optional(alternatives) => {
                    let this = group_index;
                    group_index += 1;
                    match selection {
                        Some((group, alternative)) if group == this => {
                            vec![alternatives[alternative].clone()]
                        }
                        _ => continue,
                    }
                }
            };
            let mut next = Vec::new();
            for variant in &partial {
                for choice in &choices {
                    let mut extended = variant.clone();
                    for word in choice {
                        if is_options_placeholder(word) {
                            continue;
                        }
                        let previous = extended.last().map_or("", |w| w.value.as_str());
                        let value = substitute_placeholders(word, previous);
                        extended.push(Word {
                            value,
                            advised: word.clone(),
                        });
                    }
                    next.push(extended);
                }
            }
            partial = next;
        }
        for variant in partial {
            if !variants.contains(&variant) {
                variants.push(variant);
            }
        }
    }
    variants
}

fn values(words: &[Word]) -> Vec<String> {
    words.iter().map(|word| word.value.clone()).collect()
}

/// The argv variants an advised string stands for: those of every command it
/// names (see [`command_parts`] and [`command_variants`]).
pub(crate) fn argv_variants(raw: &str) -> Vec<Vec<String>> {
    command_parts(raw)
        .iter()
        .flat_map(|command| command_variants(command))
        .map(|words| values(&words))
        .collect()
}

// ---------------------------------------------------------------------------
// Verdict: the routing `run()` applies
// ---------------------------------------------------------------------------

fn is_empty_value_error(error: &clap::Error) -> bool {
    error.kind() == ErrorKind::InvalidValue
        && matches!(
            error.get(ContextKind::InvalidValue),
            Some(ContextValue::String(value)) if value.is_empty()
        )
}

fn classify_clap_error(error: &clap::Error) -> Verdict {
    match error.kind() {
        ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => Verdict::Parses,
        ErrorKind::MissingRequiredArgument
        | ErrorKind::MissingSubcommand
        | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => Verdict::IncompleteReference,
        _ if is_empty_value_error(error) => Verdict::IncompleteReference,
        _ => Verdict::Rejected(error.to_string()),
    }
}

/// Route `rocm <args>` exactly as `run()` does: natural-language requests go
/// to the planner (unless they look like a mistyped command), everything else
/// through `cli_command()` and `Cli`.
pub(crate) fn rocm_verdict(args: &[String]) -> Verdict {
    if args.is_empty() {
        return Verdict::Parses; // bare `rocm` opens the launcher.
    }
    let invocation = parse_freeform_invocation(args);
    if should_treat_as_freeform(&invocation) {
        return match command_invocation_error(&invocation.request_args) {
            Some(error) => Verdict::Rejected(error.to_string()),
            None => Verdict::Freeform,
        };
    }
    let argv = std::iter::once("rocm".to_owned()).chain(args.iter().cloned());
    let parsed = cli_command()
        .try_get_matches_from(argv)
        .and_then(|matches| Cli::from_arg_matches(&matches).map(|_| ()));
    match parsed {
        Ok(()) => Verdict::Parses,
        Err(error) => classify_clap_error(&error),
    }
}

/// `rocmd`'s parser, through [`rocmd::parse_args`]: the full clap validation
/// (unknown words, invalid values, conflicting flags, missing required
/// values), with nothing run.
pub(crate) fn rocmd_verdict(args: &[String]) -> Verdict {
    let argv = std::iter::once("rocmd".into())
        .chain(args.iter().map(Into::into))
        .collect();
    match rocmd::parse_args(argv) {
        Ok(()) => Verdict::Parses,
        Err(error) => classify_clap_error(&error),
    }
}

pub(crate) fn verdict(argv: &[String]) -> Verdict {
    match argv.split_first() {
        Some((program, args)) if program == "rocm" => rocm_verdict(args),
        Some((program, args)) if program == "rocmd" => rocmd_verdict(args),
        _ => Verdict::Rejected("not a rocm/rocmd invocation".to_owned()),
    }
}

// ---------------------------------------------------------------------------
// Exclusions and the contract
// ---------------------------------------------------------------------------

/// Extracted strings that start with `rocm ` / `rocmd ` but are not advice to
/// run anything, keyed by `(file, exact extracted text)`. Each carries the
/// reason. Keep it short: an entry is something the contract does not cover,
/// and an entry whose text no longer occurs fails
/// `exclusions_still_match_something`.
const NOT_INVOCATIONS: &[(&str, &str, &str)] = &[
    // Prose that happens to begin with the program name.
    (
        "crates/rocm-core/src/diagnose.rs",
        "rocm serve/rocm chat would see this result, a plain shell might not",
        "note naming the two commands a managed runtime affects, not advice",
    ),
    (
        "crates/rocm-core/src/diagnose.rs",
        "rocm serve/rocm chat prepend the active managed runtime's own directories ahead of LD_LIBRARY_PATH, so the export below may not change what they load; reinstalling or repairing that runtime so it ships its own code object manager is the fix that reaches it directly.",
        "explanation of what the two commands load, not advice",
    ),
    (
        "apps/rocm/src/dash.rs",
        "rocm bench load supports http:// endpoints only (no TLS backend compiled in)",
        "error message naming the command, not advice",
    ),
    (
        "apps/rocm/src/main.rs",
        "rocm interactive shell",
        "heading of the non-interactive launcher report",
    ),
    (
        "apps/rocm/src/chat.rs",
        "rocm tools: {}",
        "`rocm tools: enabled` status row",
    ),
    (
        "crates/rocm-core/src/model_readiness.rs",
        "rocm diagnose --model {}: {}",
        "heading of the model-readiness report, `<command>: <verdict>`, naming what was run",
    ),
    (
        "apps/rocm/src/chat.rs",
        "rocm install folder",
        "natural-language phrase the planner matches in a request",
    ),
    (
        "apps/rocm/src/chat.rs",
        "rocm installed at",
        "natural-language phrase the planner matches in a request",
    ),
    (
        "apps/rocm/src/chat.rs",
        "rocm installed",
        "natural-language phrase the planner matches in a request",
    ),
    (
        "apps/rocm/src/chat.rs",
        "rocm please",
        "natural-language phrase the planner matches in a request",
    ),
    (
        "apps/rocm/src/chat.rs",
        "rocm command requires at least one argument",
        "validation error naming the chat `rocm` tool",
    ),
    (
        "apps/rocm/src/chat.rs",
        "rocm command",
        "fallback label for an unrenderable chat tool call",
    ),
    (
        "apps/rocm/src/main.rs",
        "rocm serve requires ROCm GPU execution; CPU mode is not a fallback path in rocm-cli",
        "error message naming the command, not advice",
    ),
    (
        "apps/rocmd/src/sandbox.rs",
        "rocmd executable has no parent directory",
        "error message",
    ),
    (
        "apps/rocmd/src/service.rs",
        "rocmd automation supervisor started",
        "log line",
    ),
    (
        "apps/rocmd/src/service.rs",
        "rocmd automation supervisor stopped",
        "log line",
    ),
    (
        "crates/rocm-core/src/diagnose.rs",
        "rocm diagnose covers Linux, Windows and WSL2. This host reports '{}', which the \
         catalog has no entries for, so nothing was checked -- this is not a clean bill of \
         health. Run `rocm examine --json` and report the platform upstream.",
        "out-of-scope report prose",
    ),
    (
        "crates/rocm-core/src/diagnose.rs",
        "rocm diagnose: out of scope for this platform.",
        "report heading",
    ),
    (
        "crates/rocm-core/src/diagnose.rs",
        "rocm diagnose: no known misconfiguration matched.",
        "report heading",
    ),
    (
        "apps/rocm/src/main.rs",
        "rocm config",
        "heading of the `rocm config` report (`writeln!(output, \"rocm config\")`)",
    ),
    (
        "crates/rocm-dash-collectors/src/bench_load.rs",
        "rocm bench load (local smoke)",
        "`launcher` label recorded in a benchmark result, not advice",
    ),
    (
        "apps/rocm/src/service_hints.rs",
        "rocm services {action} {service_id} --yes",
        "the verb is a format argument; every value it takes is checked against the \
         real hint by `service_action_retry_advice_parses_for_generated_ids` (the \
         `run_approved_service_action` verbs) and `every_service_action_hint_verb_parses` \
         (each literal verb a caller passes)",
    ),
    (
        "apps/rocm/src/runtime_services.rs",
        "rocm runtimes {command} --restart-services --yes",
        "the subcommand is a format argument; the real hint for every `rocm runtimes` \
         subcommand that takes `--restart-services` is checked, and re-parsed to show it \
         clears the refusal, by `runtime_restart_retry_advice_parses_and_clears_the_refusal`",
    ),
    (
        "crates/rocm-core/src/examine.rs",
        "rocm examine supports Linux and Windows; got {}. This skill cannot help on this platform.",
        "error message naming the command, not advice",
    ),
    (
        "crates/rocm-core/src/fix.rs",
        "rocm fix can run it",
        "remediation-flag wording (`rocm fix can run it`), not a command",
    ),
    (
        "crates/rocm-core/src/host_gpu.rs",
        "rocm debug: command capture {stage} failed for {}: {detail}",
        "debug log line",
    ),
    (
        "crates/rocm-dash-tui/src/ui/command_screen.rs",
        "rocm {}",
        "echo of whatever the user typed into the command runner",
    ),
    // Deliberate negative examples: the doc asserts these are rejected.
    (
        "docs/manual-testing.md",
        "rocm services prune --any-age --older-than-hours 0",
        "the doc's expected result is that the parser rejects this combination",
    ),
];

/// Natural-language examples must reach the planner with a multi-word request:
/// that is what tells a deliberate example (`rocm "start a local model"`) apart
/// from a structured command that no longer exists. A single word routed to
/// the planner (`rocm doctor`) means the advised subcommand is not real — the
/// user gets a request plan instead of the command they were told about.
///
/// The *first* request word must be multi-word *in the advice*: a quoted
/// request, or a placeholder that names one (`<natural language request>`).
/// That is the word real routing judges ([`command_invocation_error`] treats a
/// request as prose only when its first word has whitespace), so a quoted
/// argument after a single word does not count — `rocm frobnicate "two
/// words"` names a subcommand. Nor does a value [`placeholder_value`] filled
/// in — `rocm frobnicate <TEXT>` fills `<TEXT>` with several words, but the
/// advice names a subcommand.
fn is_deliberate_natural_language(words: &[Word]) -> bool {
    let args = values(&words[1..]);
    let request = parse_freeform_invocation(&args).request_args;
    // The request is a suffix of the arguments (`--yes` is the only prefix).
    let request_words = &words[words.len() - request.len()..];
    request_words
        .first()
        .is_some_and(|word| word.advised.contains(char::is_whitespace))
}

pub(crate) struct Finding {
    pub source: String,
    pub raw: String,
    pub argv: Vec<String>,
    pub reason: String,
}

fn is_excluded(item: &Advice) -> bool {
    NOT_INVOCATIONS
        .iter()
        .any(|(file, raw, _)| *raw == item.raw && item.source.starts_with(&format!("{file}:")))
}

/// Whether advice may name a command without its required values. Only inline
/// prose may ("pass `rocm serve --engine`"), or text that marks the omission
/// itself with an ellipsis (`rocm runtimes …`). A command line meant to be run
/// as written may not: if `rocm examine` grew a required argument, every bare
/// `rocm examine` in a RECIPE, a `next step:` line or a fenced example would
/// fail for the user who ran it.
fn may_omit_required_values(item: &Advice, command: &str) -> bool {
    item.surface == Surface::InlineProse
        || split_words(command)
            .iter()
            .any(|word| word.trim_matches(['[', ']']).ends_with('…') || word.ends_with("..."))
}

pub(crate) fn findings(advice: &[Advice]) -> Vec<Finding> {
    let mut out = Vec::new();
    for item in advice {
        if is_excluded(item) {
            continue;
        }
        for (command, words) in command_parts(&item.raw).iter().flat_map(|command| {
            command_variants(command)
                .into_iter()
                .map(move |words| (command, words))
        }) {
            let argv = values(&words);
            let reason = match verdict(&argv) {
                Verdict::Parses => continue,
                Verdict::IncompleteReference if may_omit_required_values(item, command) => continue,
                Verdict::IncompleteReference => "a required argument or subcommand is missing: \
                                                 this line is meant to be run as written"
                    .to_owned(),
                Verdict::Freeform if is_deliberate_natural_language(&words) => continue,
                Verdict::Freeform => "not a subcommand: `rocm` sends it to the natural-language \
                                      planner instead of running a command"
                    .to_owned(),
                Verdict::Rejected(error) => error,
            };
            out.push(Finding {
                source: item.source.clone(),
                raw: item.raw.clone(),
                argv,
                reason,
            });
        }
    }
    out
}

pub(crate) fn render_findings(findings: &[Finding]) -> String {
    let mut report = String::new();
    for finding in findings {
        let _ = writeln!(
            report,
            "- {}\n    advised: `{}`\n    argv:    {:?}\n    error:   {}",
            finding.source,
            finding.raw,
            finding.argv,
            finding.reason.lines().next().unwrap_or_default()
        );
    }
    report
}

fn all_advice() -> Vec<Advice> {
    let mut advice = source_advice();
    advice.extend(help_text_advice());
    advice
}

/// Where an advised invocation was found, for the per-source floors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    Help,
    Rust,
    Markdown,
    Tape,
}

fn origin(item: &Advice) -> Origin {
    if item.source.starts_with('`') {
        return Origin::Help;
    }
    let file = item.source.rsplit_once(':').map_or("", |(file, _)| file);
    match Path::new(file).extension().and_then(|ext| ext.to_str()) {
        Some("rs") => Origin::Rust,
        Some("md") => Origin::Markdown,
        Some("tape") => Origin::Tape,
        _ => panic!("advice from an unknown source: {}", item.source),
    }
}

/// Guards the scanner itself, source by source: a broken extractor would make
/// `every_advised_command_parses` pass vacuously for everything it feeds. Each
/// source (help, Rust, Markdown, tapes) must still yield a named, long-lived
/// command and a floor of entries (set well below today's counts so ordinary
/// doc edits do not trip it).
#[test]
fn every_source_is_scanned() {
    let advice = all_advice();
    let count = |wanted: Origin, surface: Surface| {
        advice
            .iter()
            .filter(|item| origin(item) == wanted && item.surface == surface)
            .count()
    };
    let has = |source_prefix: &str, raw: &str, surface: Surface| {
        advice.iter().any(|item| {
            item.source.starts_with(source_prefix)
                && item.raw.starts_with(raw)
                && item.surface == surface
        })
    };

    // Help: the top-level EXAMPLES row and inline spans.
    assert!(
        has("`rocm --help`", "rocm examine", Surface::CommandLine),
        "help EXAMPLES row `rocm examine` not found: help extraction is broken"
    );
    assert!(count(Origin::Help, Surface::CommandLine) >= 20);
    assert!(count(Origin::Help, Surface::InlineProse) >= 20);
    // Rust: a dashboard `cmd` literal, labelled lines, and inline spans.
    assert!(
        has(
            "crates/rocm-dash-tui/src/ui/tabs/rocm.rs:",
            "rocm update",
            Surface::CommandLine
        ),
        "dashboard `cmd: \"rocm update\"` not found: Rust literal extraction is broken"
    );
    assert!(
        has(
            "apps/rocm/src/main.rs:",
            "rocm services prune --dry-run",
            Surface::CommandLine
        ),
        "labelled `Try: rocm services prune --dry-run` line not found: labelled-line \
         extraction is broken"
    );
    assert!(
        has(
            "apps/rocm/src/main.rs:",
            "rocm services prune --yes",
            Surface::CommandLine
        ),
        "labelled `Then: rocm services prune --yes` line (after a `\\n` in the same \
         literal) not found: per-line literal extraction is broken"
    );
    assert!(count(Origin::Rust, Surface::CommandLine) >= 90);
    assert!(count(Origin::Rust, Surface::InlineProse) >= 130);
    // Markdown: the quick-start fenced line and an inline span.
    assert!(
        has("README.md:", "rocm install sdk", Surface::CommandLine),
        "README fenced `rocm install sdk` not found: fenced-block extraction is broken"
    );
    assert!(
        has("README.md:", "rocm examine", Surface::InlineProse),
        "README inline `rocm examine` not found: inline-span extraction is broken"
    );
    assert!(count(Origin::Markdown, Surface::CommandLine) >= 100);
    assert!(count(Origin::Markdown, Surface::InlineProse) >= 120);
    assert!(
        count(Origin::Markdown, Surface::CommandLine)
            + count(Origin::Markdown, Surface::InlineProse)
            >= 300
    );
    // Tapes: the CLI demo's first command.
    assert!(
        has("docs/tapes/cli.tape:", "rocm examine", Surface::CommandLine),
        "`Type \"rocm examine\"` not found in the CLI tape: tape extraction is broken"
    );
    assert!(count(Origin::Tape, Surface::CommandLine) >= 5);
}

#[test]
fn every_advised_command_parses() {
    let advice = all_advice();
    let found = findings(&advice);
    assert!(
        found.is_empty(),
        "{} advised `rocm`/`rocmd` invocation(s) do not parse with the real CLI. \
         Fix the advice (or the CLI), or — only for text that is not advice to run \
         anything — add it to NOT_INVOCATIONS with a reason:\n{}",
        found.len(),
        render_findings(&found)
    );
}

#[test]
fn exclusions_still_match_something() {
    let advice = source_advice();
    for (file, raw, reason) in NOT_INVOCATIONS {
        assert!(
            advice
                .iter()
                .any(|item| item.raw == *raw && item.source.starts_with(&format!("{file}:"))),
            "NOT_INVOCATIONS entry ({file}, {raw:?}: {reason}) no longer matches any \
             extracted text; remove it"
        );
    }
}

#[test]
fn checker_rejects_what_users_would_hit() {
    // The verdicts the contract rests on, each against the real parser.
    let argv = |text: &str| argv_variants(text).remove(0);
    assert!(matches!(
        verdict(&argv("rocm update --frobnicate")),
        Verdict::Rejected(_)
    ));
    assert!(matches!(
        verdict(&argv("rocm install --channel release")),
        Verdict::Rejected(_)
    ));
    assert_eq!(verdict(&argv("rocm frobnicate")), Verdict::Freeform);
    let words = |text: &str| command_variants(text).remove(0);
    assert!(!is_deliberate_natural_language(&words("rocm frobnicate")));
    assert!(matches!(
        verdict(&argv("rocm instal sdk")),
        Verdict::Rejected(_)
    ));
    assert!(matches!(
        verdict(&argv("rocmd run --no-such-flag")),
        Verdict::Rejected(_)
    ));
    assert_eq!(
        verdict(&argv("rocm install sdk --family <FAMILY>")),
        Verdict::Parses
    );
    assert_eq!(
        verdict(&argv("rocm serve --engine")),
        Verdict::IncompleteReference
    );
    assert_eq!(
        verdict(&argv("rocmd run --automations-enabled")),
        Verdict::Parses
    );
    // `rocmd` gets the full validation too, not only unknown words: flags
    // declared as conflicting, and a missing required value.
    let supervise = "rocmd supervise svc --engine vllm --model-ref m \
                     --canonical-model-id c --port 8000";
    assert_eq!(
        verdict(&argv(&format!("{supervise} --runtime-id a"))),
        Verdict::Parses
    );
    assert!(matches!(
        verdict(&argv(&format!("{supervise} --runtime-id a --env-id b"))),
        Verdict::Rejected(error) if error.contains("cannot be used with")
    ));
    assert_eq!(
        verdict(&argv("rocmd supervise svc --engine vllm")),
        Verdict::IncompleteReference
    );
    assert!(is_deliberate_natural_language(&words(
        "rocm --yes \"start a local model\""
    )));
}

#[test]
fn only_prose_or_an_ellipsis_may_leave_required_values_out() {
    let advice = |raw: &str, surface| Advice {
        source: "fixture.md:1".to_owned(),
        raw: raw.to_owned(),
        surface,
    };
    // `runtimes activate` requires a runtime key.
    assert!(findings(&[advice("rocm runtimes activate", Surface::InlineProse)]).is_empty());
    assert_eq!(
        findings(&[advice("rocm runtimes activate", Surface::CommandLine)]).len(),
        1,
        "a command line missing a required value is a violation"
    );
    assert!(findings(&[advice("rocm runtimes …", Surface::CommandLine)]).is_empty());
    assert!(findings(&[advice("rocm serve --managed ...", Surface::CommandLine)]).is_empty());
}

#[test]
fn synopsis_notation_expands_to_each_documented_form() {
    assert_eq!(
        argv_variants("rocm dash [--demo] [--replay <file>]"),
        vec![
            vec!["rocm", "dash"],
            vec!["rocm", "dash", "--demo"],
            vec!["rocm", "dash", "--replay", "/tmp/replay.json"],
        ]
    );
    assert_eq!(
        argv_variants("rocm services stop|restart <id> --yes"),
        vec![
            vec!["rocm", "services", "stop", "svc-1", "--yes"],
            vec!["rocm", "services", "restart", "svc-1", "--yes"],
        ]
    );
    assert_eq!(
        argv_variants("rocm serve <model> --engine vllm   # then check"),
        vec![vec!["rocm", "serve", "Qwen/Qwen3-0.6B", "--engine", "vllm"]]
    );
    assert_eq!(
        argv_variants("rocm --yes <natural language request>"),
        vec![vec!["rocm", "--yes", "start a local model"]]
    );
    // A nested group is optional within the outer one: each prefix is a form.
    assert_eq!(
        argv_variants("rocm runtimes rollback [--restart-services [--yes]]"),
        vec![
            vec!["rocm", "runtimes", "rollback"],
            vec!["rocm", "runtimes", "rollback", "--restart-services"],
            vec![
                "rocm",
                "runtimes",
                "rollback",
                "--restart-services",
                "--yes"
            ],
        ]
    );
    // A value list first in a group branches, and later words extend each
    // branch.
    assert_eq!(
        argv_variants("rocm services [stop|restart <id>] list"),
        vec![
            vec!["rocm", "services", "list"],
            vec!["rocm", "services", "stop", "svc-1", "list"],
            vec!["rocm", "services", "restart", "svc-1", "list"],
        ]
    );
    assert_eq!(
        argv_variants("rocm x [--a|--b [--c] | --d]"),
        vec![
            vec!["rocm", "x"],
            vec!["rocm", "x", "--a"],
            vec!["rocm", "x", "--b"],
            vec!["rocm", "x", "--a", "--c"],
            vec!["rocm", "x", "--b", "--c"],
            vec!["rocm", "x", "--d"],
        ]
    );
    assert_eq!(
        argv_variants("rocm storage [report [--json]] [--top N]"),
        vec![
            vec!["rocm", "storage"],
            vec!["rocm", "storage", "report"],
            vec!["rocm", "storage", "report", "--json"],
            vec!["rocm", "storage", "--top", "1"],
        ]
    );
}

/// Side-by-side groups claim each flag works on its own. `--yes` on
/// `runtimes activate`/`rollback` requires `--restart-services`, so writing
/// them as `[--restart-services] [--yes]` advertises a command that fails;
/// nesting the dependent flag says what the parser accepts.
#[test]
fn a_flag_that_requires_another_is_advised_only_with_it() {
    let found = findings(&[fixture(
        "rocm runtimes rollback [--restart-services] [--yes]",
        Surface::CommandLine,
    )]);
    assert_eq!(found.len(), 1, "{}", render_findings(&found));
    assert_eq!(found[0].argv, ["rocm", "runtimes", "rollback", "--yes"]);
    let found = findings(&[fixture(
        "rocm runtimes activate <runtime-key> [--restart-services [--yes]]",
        Surface::CommandLine,
    )]);
    assert!(found.is_empty(), "{}", render_findings(&found));
}

/// Templated advice, checked through the real message function rather than
/// its source text: `rocm services stop|restart <id>` without `--yes` fails
/// with `Try: rocm services <verb> <id> --yes`. The id comes from
/// `generate_service_id`, which takes an arbitrary model reference, so the
/// model references cover every character class it handles differently
/// (alphanumerics kept, everything else — separators, whitespace, quotes,
/// shell metacharacters, non-ASCII, a leading `-` — mapped to `-`). An
/// exhaustive class corpus rather than random sampling: the function is a
/// per-character map, so one representative per class covers it.
#[test]
fn service_action_retry_advice_parses_for_generated_ids() {
    use super::{AppPaths, run_approved_service_action};

    // Never touched: the missing-`--yes` branch bails before any disk access.
    let unused = std::env::temp_dir().join("rocm-advised-commands-never-created");
    let paths = AppPaths {
        config_dir: unused.join("config"),
        data_dir: unused.join("data"),
        cache_dir: unused.join("cache"),
    };
    // The tools the `services stop` / `services restart` dispatch passes in.
    let actions = [("stop_server", "stop"), ("restart_server", "restart")];
    for id in generated_service_ids() {
        for (tool, verb) in actions {
            let error = run_approved_service_action(&paths, tool, &id, false)
                .expect_err("an action without --yes is refused")
                .to_string();
            let advised = error
                .lines()
                .find_map(|line| line.strip_prefix("Try: "))
                .unwrap_or_else(|| panic!("no `Try:` advice in: {error}"));
            let argv = split_words(advised);
            assert_eq!(
                verdict(&argv),
                Verdict::Parses,
                "advice {advised:?} for id {id:?} does not parse"
            );
            // It names the same action on the same service and carries the
            // `--yes` the refusal asked for, so following it clears the gate.
            assert_eq!(argv, ["rocm", "services", verb, id.as_str(), "--yes"]);
        }
    }
}

/// Service ids `generate_service_id` produces for every engine, from model
/// references covering each character class it handles differently.
fn generated_service_ids() -> Vec<String> {
    let long = "x".repeat(80);
    let model_refs = [
        "Qwen/Qwen3-0.6B",
        "unsloth/Qwen3-0.6B-GGUF:Q4_0",
        "C:\\models\\local.gguf",
        "  leading and trailing  ",
        "-starts-with-dash",
        "--looks-like-a-flag",
        "with \"quotes\" and 'apostrophes'",
        "a;b|c&&d$(e)`f`",
        "ünïcödé/模型",
        "",
        long.as_str(),
    ];
    super::SUPPORTED_ENGINES
        .iter()
        .flat_map(|engine| {
            model_refs
                .iter()
                .map(move |model_ref| rocm_core::generate_service_id(engine, model_ref))
        })
        .collect()
}

/// `service_hints::service_action_hint(action, id)` is excluded from the scan
/// because its verb is a format argument. The verbs come from its callers:
/// `run_approved_service_action` (checked through the real message above) and
/// string literals at the other call sites, which this test finds in the
/// source and checks one by one. A call with any other kind of argument fails
/// here, so a new caller cannot slip past both checks.
#[test]
fn every_service_action_hint_verb_parses() {
    use super::service_hints;

    let call = concat!("service_action_hint", "(");
    let mut files = Vec::new();
    walk(&repo_root().join("apps/rocm/src"), &["rs"], &[], &mut files);
    let mut verbs: Vec<String> = Vec::new();
    for file in &files {
        if file.ends_with("advised_commands.rs") {
            continue; // this test's own calls
        }
        let text = std::fs::read_to_string(file).expect("readable source");
        for (index, line) in text.lines().enumerate() {
            let Some(at) = line.find(call) else { continue };
            let argument = &line[at + call.len()..];
            let code = line.trim_start();
            if code.starts_with("//") || code.starts_with("pub(crate) fn ") {
                continue; // a comment, or the definition
            }
            if argument.starts_with("service_action_command(tool)") {
                continue; // `service_action_retry_advice_parses_for_generated_ids`
            }
            let Some((verb, _)) = argument
                .strip_prefix('"')
                .and_then(|rest| rest.split_once('"'))
            else {
                panic!(
                    "{}:{}: `service_action_hint` called with a verb this test \
                     cannot check; pass a string literal or extend this test: {line}",
                    file.display(),
                    index + 1
                )
            };
            let verb = verb.to_owned();
            if !verbs.contains(&verb) {
                verbs.push(verb);
            }
        }
    }
    // `service_stop_hint` is `service_action_hint("stop", …)`.
    verbs.push("stop".to_owned());
    for expected in ["restart", "remove"] {
        assert!(
            verbs.iter().any(|verb| verb == expected),
            "call site with {expected:?} not found ({verbs:?}): the scan is broken"
        );
    }
    for id in generated_service_ids() {
        let mut hints: Vec<(String, String)> = verbs
            .iter()
            .map(|verb| (verb.clone(), service_hints::service_action_hint(verb, &id)))
            .collect();
        hints.push(("stop".to_owned(), service_hints::service_stop_hint(&id)));
        for (verb, hint) in hints {
            let argv = split_words(&hint);
            assert_eq!(
                verdict(&argv),
                Verdict::Parses,
                "hint {hint:?} for id {id:?} does not parse"
            );
            assert_eq!(
                argv,
                ["rocm", "services", verb.as_str(), id.as_str(), "--yes"]
            );
        }
    }
}

/// The `rocm runtimes <command> --restart-services --yes` retry hint, checked
/// through the real refusal for every `rocm runtimes` subcommand that takes
/// `--restart-services` (found in the clap tree, so a new one is covered too).
/// The hint must parse, and parsed back it must be the same command with the
/// approval added — following it clears the refusal that printed it.
#[test]
fn runtime_restart_retry_advice_parses_and_clears_the_refusal() {
    use super::{Command, RuntimesCommand};
    use crate::runtimes_cmd::ensure_runtime_service_restart_approved;

    fn runtimes_command(argv: &[String]) -> RuntimesCommand {
        let matches = cli_command()
            .try_get_matches_from(argv)
            .unwrap_or_else(|error| panic!("{argv:?} does not parse: {error}"));
        match Cli::from_arg_matches(&matches).map(|cli| cli.command) {
            Ok(Some(Command::Runtimes {
                command: Some(command),
            })) => command,
            other => panic!("{argv:?} is not a `rocm runtimes` subcommand: {other:?}"),
        }
    }

    // Runtime keys as `rocm install sdk` writes them, a runtime id, a key in
    // other letter case (selectors match case-insensitively), and an adopted
    // runtime's free-form `--runtime-key`.
    let selectors = [
        "release-pip-gfx120x-all-7-13-0",
        "release-pip-gfx120x-all-7-12-0-devel",
        "therock-7.13.0-gfx120x",
        "RELEASE-PIP-GFX120X-ALL-7-13-0",
        "my_adopted.rocm",
    ];
    let mut root = cli_command();
    let runtimes = root
        .find_subcommand_mut("runtimes")
        .expect("`rocm runtimes` exists");
    let mut checked = Vec::new();
    for subcommand in runtimes.get_subcommands_mut() {
        if !subcommand
            .get_arguments()
            .any(|arg| arg.get_long() == Some("restart-services"))
        {
            continue;
        }
        let name = subcommand.get_name().to_owned();
        let takes_runtime = subcommand
            .get_arguments()
            .any(|arg| arg.is_positional() && arg.is_required_set());
        let values: Vec<Option<&str>> = if takes_runtime {
            selectors.iter().copied().map(Some).collect()
        } else {
            vec![None]
        };
        for value in values {
            let mut argv: Vec<String> = vec!["rocm".into(), "runtimes".into(), name.clone()];
            argv.extend(value.map(str::to_owned));
            argv.push("--restart-services".into());
            let refused = runtimes_command(&argv);
            let error = ensure_runtime_service_restart_approved(&refused)
                .expect_err("--restart-services without --yes is refused")
                .to_string();
            let advised = error
                .lines()
                .find_map(|line| line.strip_prefix("Try: "))
                .unwrap_or_else(|| panic!("no `Try:` advice in: {error}"));
            let retry = split_words(advised);
            assert_eq!(
                verdict(&retry),
                Verdict::Parses,
                "advice {advised:?} after {argv:?} does not parse"
            );
            let mut expected = argv.clone();
            expected.push("--yes".into());
            assert_eq!(
                retry, expected,
                "the advice must repeat what the user ran, with the approval added"
            );
            ensure_runtime_service_restart_approved(&runtimes_command(&retry))
                .expect("following the advice clears the refusal");
        }
        checked.push(name);
    }
    assert_eq!(
        checked,
        ["activate", "rollback"],
        "the subcommands taking --restart-services changed; check the hint covers them"
    );
}

#[test]
#[ignore = "report: prints every advised invocation and its verdict"]
fn dump_advised_invocations() {
    let advice = all_advice();
    for item in &advice {
        for argv in argv_variants(&item.raw) {
            println!(
                "{}\t{}\t{:?}\t{:?}\t{:?}",
                item.source,
                item.raw,
                argv,
                verdict(&argv),
                item.surface
            );
        }
    }
    for wanted in [Origin::Help, Origin::Rust, Origin::Markdown, Origin::Tape] {
        for surface in [Surface::CommandLine, Surface::InlineProse] {
            let count = advice
                .iter()
                .filter(|item| origin(item) == wanted && item.surface == surface)
                .count();
            println!("COUNT\t{wanted:?}\t{surface:?}\t{count}");
        }
    }
    println!("TOTAL\t{}", advice.len());
}

/// A span an author broke across help lines is joined and checked, like a
/// wrapped span in Markdown prose.
#[test]
fn a_span_wrapped_in_help_text_is_checked() {
    let tree = clap::Command::new("rocm")
        .after_long_help("Then run `rocm\nfrobnicate --now` to finish.\nAlso `rocm examine`.");
    let advice = help_advice_of(tree);
    let raws: Vec<&str> = advice.iter().map(|item| item.raw.as_str()).collect();
    assert!(
        raws.contains(&"rocm frobnicate --now") && raws.contains(&"rocm examine"),
        "{advice:?}"
    );
    let found = findings(&advice);
    assert_eq!(found.len(), 1, "{}", render_findings(&found));
    assert_eq!(found[0].argv[1], "frobnicate");
}

/// Markdown advice in each shape the extractor handles. Every shape carries a
/// removed subcommand that must be reported, so dropping the branch that
/// reads the shape fails here rather than leaving that advice unchecked.
#[test]
fn every_markdown_shape_is_extracted_and_checked() {
    let doc = "\
Run `rocm frobnicate-inline` first.

Wrapped prose: run `rocm
frobnicate-wrapped --now` to check.

```sh
$ rocm frobnicate-dollar
```

```powershell
PS> rocm frobnicate-ps
```

```sh
rocm frobnicate-continued \\
  --flag
```
";
    let mut advice = Vec::new();
    extract_markdown_text(doc, "fixture.md", &mut advice);
    let reported: Vec<String> = findings(&advice)
        .iter()
        .map(|finding| format!("{} {}", finding.source, finding.argv[1]))
        .collect();
    assert_eq!(
        reported,
        [
            "fixture.md:1 frobnicate-inline",
            "fixture.md:3 frobnicate-wrapped",
            "fixture.md:7 frobnicate-dollar",
            "fixture.md:11 frobnicate-ps",
            "fixture.md:15 frobnicate-continued",
        ],
        "{advice:?}"
    );
    let continued = advice
        .iter()
        .find(|item| item.raw.contains("frobnicate-continued"))
        .expect("continued command extracted");
    assert_eq!(continued.raw, "rocm frobnicate-continued --flag");
}

/// A `\` continuation the closing fence (or the end of the file) cuts off is
/// checked as it stands, and does not swallow the next fence's first line.
#[test]
fn a_cut_off_continuation_stays_one_command() {
    let doc = "\
```sh
rocm frobnicate-cut \\
```

```sh
rocm examine
```

```sh
rocm frobnicate-eof \\";
    let mut advice = Vec::new();
    extract_markdown_text(doc, "fixture.md", &mut advice);
    let raws: Vec<(&str, &str)> = advice
        .iter()
        .map(|item| (item.source.as_str(), item.raw.as_str()))
        .collect();
    assert_eq!(
        raws,
        [
            ("fixture.md:2", "rocm frobnicate-cut"),
            ("fixture.md:6", "rocm examine"),
            ("fixture.md:10", "rocm frobnicate-eof"),
        ]
    );
}

fn fixture(raw: &str, surface: Surface) -> Advice {
    Advice {
        source: "fixture.md:1".to_owned(),
        raw: raw.to_owned(),
        surface,
    }
}

/// Every command in a shell list is advice, not only the first: a removed
/// subcommand after `&&`, `||`, `;` or `|` strands the user just the same.
#[test]
fn every_command_in_a_shell_list_is_checked() {
    for raw in [
        "rocm update && rocm frobnicate --x",
        "rocm update || rocm frobnicate --x",
        "rocm update; rocm frobnicate --x",
        "rocm update | rocm frobnicate --x",
        "rocm update && rocmd frobnicate",
    ] {
        let found = findings(&[fixture(raw, Surface::CommandLine)]);
        assert_eq!(found.len(), 1, "{raw}: {}", render_findings(&found));
        assert!(
            found[0].argv[1] == "frobnicate",
            "{raw}: {:?}",
            found[0].argv
        );
    }
    // Commands for other tools in the list are not this contract's business.
    for raw in [
        "rocm examine --json | jq .gpus",
        "rocm update && echo done",
        "cd /tmp; rocm examine",
    ] {
        assert!(
            findings(&[fixture(raw, Surface::CommandLine)]).is_empty(),
            "{raw}"
        );
    }
    // `|` inside a word, a quoted request, a placeholder or an optional group
    // is notation, not a pipe.
    assert_eq!(
        command_parts("rocm services stop|restart <id> --yes"),
        vec!["rocm services stop|restart <id> --yes"]
    );
    assert_eq!(
        command_parts("rocm \"start a model && check it | twice\""),
        vec!["rocm \"start a model && check it | twice\""]
    );
    assert_eq!(
        command_parts("rocm serve <a | b> [--x | --y] && rocm examine"),
        vec!["rocm serve <a | b> [--x | --y]", "rocm examine"]
    );
}

/// Prose may follow a command after an em dash, an en dash or an arrow
/// (`rocm examine — check the host first`). Each marker ends the command, so
/// the description is not read as arguments; dropping any one of them from
/// `command_parts` turns that prose into words clap rejects.
#[test]
fn a_dash_or_arrow_ends_the_command_before_its_description() {
    for raw in [
        "rocm examine — check the host first",
        "rocm examine – check the host first",
        "rocm examine → then read the report",
    ] {
        assert_eq!(command_parts(raw), vec!["rocm examine"], "{raw}");
        assert!(
            findings(&[fixture(raw, Surface::CommandLine)]).is_empty(),
            "{raw}"
        );
    }
}

/// A double space inside a command is just whitespace; only a help EXAMPLES
/// row puts a description column after one.
#[test]
fn a_double_space_does_not_hide_the_rest_of_a_command() {
    let found = findings(&[fixture(
        "rocm install sdk --channel release  --bogus-flag",
        Surface::CommandLine,
    )]);
    assert_eq!(found.len(), 1, "{}", render_findings(&found));
    assert_eq!(
        examples_row("  rocm examine      Check GPU, driver and runtime state"),
        Some("rocm examine".to_owned())
    );
    assert_eq!(
        examples_row("  rocm serve <model> --engine vllm"),
        Some("rocm serve <model> --engine vllm".to_owned())
    );
    // An unindented line is not a table row: a string literal or help line
    // that starts with a command keeps everything after a double space.
    assert_eq!(
        examples_row("rocm install sdk --channel release  --bogus-flag"),
        Some("rocm install sdk --channel release  --bogus-flag".to_owned())
    );
    assert_eq!(examples_row("Usage: rocm [OPTIONS]"), None);
}

#[test]
fn an_options_placeholder_stands_for_no_arguments() {
    // `<case-appropriate options above>` names flags the reader picks from
    // the advice above it; the command around it must still be checked.
    assert_eq!(
        argv_variants("rocm serve <model> <case-appropriate options above>"),
        vec![vec![
            "rocm".to_owned(),
            "serve".to_owned(),
            "Qwen/Qwen3-0.6B".to_owned(),
        ]],
    );
    // Only a placeholder that names options is dropped.
    assert_eq!(
        argv_variants("rocm serve <model>"),
        vec![vec![
            "rocm".to_owned(),
            "serve".to_owned(),
            "Qwen/Qwen3-0.6B".to_owned(),
        ]],
    );
}

/// A natural-language request is deliberate only when the advice itself
/// spells one out. A placeholder filled with a multi-word value does not make
/// `rocm frobnicate <TEXT>` a request: `frobnicate` is a removed subcommand.
#[test]
fn only_advised_text_makes_a_request_deliberate() {
    // A removed subcommand stays a finding whatever follows it: a filled-in
    // placeholder, or a quoted argument after the single word routing judges.
    for raw in ["rocm frobnicate <TEXT>", "rocm frobnicate \"two words\""] {
        let found = findings(&[fixture(raw, Surface::CommandLine)]);
        assert_eq!(found.len(), 1, "{raw}: {}", render_findings(&found));
    }
    for raw in [
        "rocm \"start a local model\"",
        "rocm --yes \"start a local model\"",
        "rocm --yes <natural language request>",
        "rocm --yes ...",
    ] {
        let found = findings(&[fixture(raw, Surface::CommandLine)]);
        assert!(found.is_empty(), "{raw}: {}", render_findings(&found));
    }
}

/// A file declared only as a test module is test code throughout, whatever it
/// is named: its comments and fixtures are not advice to anyone running the
/// CLI. Both declaration shapes in the tree are covered — a `cfg(all(test, …))`
/// gate and a `#[path]` pointing outside the declaring file's own folder.
#[test]
fn test_only_module_files_follow_the_declaration() -> std::io::Result<()> {
    let root = tempfile::tempdir()?;
    let src = root.path().join("app").join("src");
    let shared = root.path().join("shared");
    std::fs::create_dir_all(src.join("tool"))?;
    std::fs::create_dir_all(&shared)?;
    std::fs::write(
        src.join("main.rs"),
        "#[cfg(test)]\nmod helper;\n\
         #[cfg(all(test, unix))]\n#[path = \"../../shared/props.rs\"]\nmod props;\n\
         mod tool;\n#[cfg(not(test))]\nmod real;\n",
    )?;
    std::fs::write(src.join("tool.rs"), "#[cfg(test)]\nmod nested;\n")?;
    for file in [
        src.join("helper.rs"),
        shared.join("props.rs"),
        src.join("tool").join("nested.rs"),
        src.join("real.rs"),
    ] {
        std::fs::write(file, "")?;
    }

    let mut found = test_only_module_files(&[src.join("main.rs"), src.join("tool.rs")]);
    found.sort();
    let mut expected = vec![
        std::fs::canonicalize(src.join("helper.rs"))?,
        std::fs::canonicalize(shared.join("props.rs"))?,
        std::fs::canonicalize(src.join("tool").join("nested.rs"))?,
    ];
    expected.sort();
    assert_eq!(found, expected);
    Ok(())
}

/// An item gated on `cfg(all(test, …))` inside a production file is skipped
/// like one gated on plain `cfg(test)`.
#[test]
fn an_item_gated_on_all_test_is_not_production_text() {
    let text = "fn shown() { println!(\"rocm examine\"); }\n\
                #[cfg(all(test, unix))]\n\
                fn hidden() { println!(\"rocm frobnicate\"); }\n";
    let lines: Vec<String> = rust_production_lines(text)
        .into_iter()
        .map(|(_, line)| line)
        .collect();
    assert!(
        lines.iter().any(|line| line.contains("rocm examine")),
        "{lines:?}"
    );
    assert!(
        !lines.iter().any(|line| line.contains("frobnicate")),
        "{lines:?}"
    );
}
