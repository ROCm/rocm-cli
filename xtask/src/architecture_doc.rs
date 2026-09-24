// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Guard `docs/architecture.md`'s path citations against silent drift.
//!
//! The doc tells contributors "verify current file and function boundaries
//! directly ... rather than trusting this doc's wording" — that disclaimer
//! exists because nothing previously checked that the paths it cites still
//! exist. A renamed or removed file then rots silently in the doc until a
//! reader notices. This check makes the doc self-policing instead: it
//! extracts every backtick-quoted path citation and fails with the stale
//! path named if one no longer exists in the tree.
//!
//! Same shape as [`crate::crate_edges`]: a reusable [`run`] plus
//! `#[cfg(test)]` unit tests on the pure extraction/lookup helpers, and one
//! test asserting the real doc passes today.
//!
//! ## Telling a path citation from a code snippet
//!
//! The doc is prose, not a code listing, but it still backtick-quotes
//! plenty of non-path Rust syntax alongside real paths: `` `crate::` ``,
//! `` `pub(crate) fn` ``, `` `comfyui::render_status(...)` ``,
//! `` `too_many_lines = "allow"` ``, and bare type names like
//! `` `ActionReport` ``. [`is_path_candidate`] filters those out; see its
//! doc comment for the exact rule and its known blind spot (a bare,
//! non-hyphenated word like `` `xtask` `` is indistinguishable from a plain
//! English word like `` `grep` `` and is deliberately never treated as a
//! citation, so it goes unchecked rather than risk false-flagging prose).
//!
//! ## Scoping bare filename citations to their section
//!
//! The doc cites `` `main.rs` ``/`` `lib.rs` `` bare, by design, under
//! several different `### \`<crate>\`` headings — one per subsystem still
//! pending modularization. The workspace has ~17 files literally named
//! `main.rs` or `lib.rs`, so checking a bare citation against "exists
//! anywhere in the repo" would make the check nearly a no-op for exactly
//! the citations it most needs to catch: renaming *`apps/rocmd`'s* `lib.rs`
//! would go undetected as long as some unrelated crate's `lib.rs` still
//! exists. [`extract_path_citations`] records each citation's nearest
//! preceding heading's directories, and [`citation_exists`] uses that
//! context to check "does this specific section's file still exist"
//! instead — see both functions' doc comments for the exact rule and its
//! (narrower, documented) fallback.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Path to the guarded doc, relative to the repo root.
const DOC_PATH: &str = "docs/architecture.md";

/// File extensions the doc cites by bare name (no directory), trusting
/// surrounding prose — or, since [`extract_path_citations`], its nearest
/// heading — for which subsystem the file lives in.
const BARE_FILE_EXTENSIONS: [&str; 3] = [".rs", ".md", ".toml"];

/// Extensions among [`BARE_FILE_EXTENSIONS`] whose bare citations get scoped
/// to their heading's directories in [`citation_exists`]. Only `.rs`: every
/// bare `.md`/`.toml` citation in the doc today (`` `AGENTS.md` ``,
/// `` `Cargo.toml` ``, `` `runtime-deps.toml` ``) is a singleton file that
/// lives at the repo root regardless of which subsystem's section mentions
/// it — e.g. `runtime-deps.toml` is cited inside the `crates/rocm-deps`
/// section but the doc's own prose calls it out as "workspace-root", so
/// scoping it to that crate's directory would be wrong. `.rs` bare
/// citations, by contrast, are always per-crate source files
/// (`main.rs`/`lib.rs`/`agent.rs`), which is exactly the case that needs
/// scoping (see the module doc comment).
const SCOPED_BARE_EXTENSIONS: [&str; 1] = [".rs"];

/// The remaining [`BARE_FILE_EXTENSIONS`] — `.md`/`.toml` — matched in
/// [`citation_exists`] against the repo root specifically, rather than by
/// path component anywhere in the tree: the workspace has 14 nested
/// `Cargo.toml` manifests, so an "any component" match would keep passing
/// for a stale root `` `Cargo.toml` `` citation as long as any crate's
/// manifest still existed, the same masking bug `.rs` scoping exists to
/// prevent — just for a fixed location (the root) instead of a
/// heading-derived one.
const ROOT_LEVEL_BARE_EXTENSIONS: [&str; 2] = [".md", ".toml"];

/// One path citation from the doc, paired with the directories named by its
/// nearest preceding heading (e.g. `` ### `apps/rocmd` `` → `["apps/rocmd"]`).
/// Empty when the citation isn't under a directory-naming heading (the top
/// of the doc, or a heading like `## Module map` with no path in it).
///
/// See [`citation_exists`] for why a bare filename citation needs this
/// context to be checked precisely.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Citation {
    text: String,
    section_dirs: Vec<String>,
}

/// Run `git` with the given args (relative to `root`) and return trimmed
/// stdout, failing on a non-zero exit. Same shape as
/// [`crate::verify_commits`]'s private `git` helper.
fn git(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .with_context(|| format!("failed to run `git {}`", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Every path tracked in the git tree, as `root`-relative [`PathBuf`]s.
/// Scoping to tracked files (rather than a raw filesystem walk) matches the
/// issue's "still exists in the tree" wording and skips build artifacts
/// under `target/` for free.
fn tracked_files(root: &Path) -> Result<Vec<PathBuf>> {
    Ok(git(root, &["ls-files"])?
        .lines()
        .map(PathBuf::from)
        .collect())
}

/// Whether a backtick-quoted span from the doc looks like a file or
/// directory path citation, as opposed to a Rust type name, function call,
/// or other code-syntax snippet also written in backticks throughout the
/// doc.
///
/// A span qualifies if every character is path-safe (alphanumeric, `/`,
/// `_`, `-`, `.`) AND one of:
/// - it contains a `/` — an explicit relative path
///   (`apps/rocm/src/therock.rs`) or a bare directory citation
///   (`apps/rocm`);
/// - it has no `/` but ends in a [`BARE_FILE_EXTENSIONS`] extension — the
///   doc cites many files by bare name (`main.rs`, `lib.rs`,
///   `bootstrap.rs`), trusting surrounding prose for which subsystem
///   directory they live in rather than repeating the full path;
/// - it has no `/` and no extension, but is an all-lowercase hyphenated
///   word (`rocm-dash-collectors`) — the doc's convention for citing a
///   crate directory by its Cargo package name.
///
/// A bare, non-hyphenated word (`xtask`) is deliberately NOT treated as a
/// candidate: nothing at the lexical level distinguishes a genuine bare
/// directory name from a plain English word or shell command mentioned in
/// prose (e.g. `grep`), and a false-flagged prose word would break this
/// check on the very doc it exists to validate. Missing a rare citation
/// like `xtask` is the safer failure mode.
fn is_path_candidate(span: &str) -> bool {
    if span.is_empty() || !is_path_safe(span) {
        return false;
    }
    if span.contains('/') {
        return true;
    }
    if BARE_FILE_EXTENSIONS
        .iter()
        .any(|ext| span.len() > ext.len() && span.ends_with(ext))
    {
        return true;
    }
    is_hyphenated_bare_word(span)
}

/// Whether every character in `span` is safe to appear in a path, as
/// opposed to prose punctuation: alphanumeric, `/`, `_`, `-`, or `.`. Shared
/// by [`is_path_candidate`] and [`is_directory_shaped`] so a malformed span
/// (stray punctuation from surrounding prose) is rejected the same way by
/// both, rather than one accepting what the other would reject.
fn is_path_safe(span: &str) -> bool {
    span.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '_' | '-' | '.'))
}

/// A bare (no `/`, no extension), all-lowercase, hyphenated word — the
/// doc's convention for citing a crate directory by its Cargo package name
/// (`rocm-dash-collectors`), as opposed to a Rust type name or other
/// PascalCase identifier also written in backticks (`ActionReport`).
fn is_hyphenated_bare_word(span: &str) -> bool {
    span.contains('-')
        && span
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Whether `span` is directory-shaped: a slash-path (`apps/rocmd`) or a
/// bare hyphenated crate name (`rocm-dash-collectors`) — the doc's two
/// conventions for citing a directory, shared by heading-directory
/// collection and possessive-owner narrowing in
/// [`extract_path_citations`]. A slash-path must also be path-safe (see
/// [`is_path_safe`]) — otherwise a stray bit of prose punctuation next to a
/// `/` (`` `foo/bar!`'s `main.rs` ``) would be accepted as a directory,
/// producing an unmatchable `section_dirs` entry that fails an accurate
/// citation — and must NOT end in a [`BARE_FILE_EXTENSIONS`] extension,
/// otherwise a full file path (`` `crates/rocm-core/src/diagnose.rs` ``)
/// would be accepted as if it were the directory containing it, which is
/// equally unmatchable. [`is_hyphenated_bare_word`] already excludes
/// extensions and guarantees path-safety on its own, so only the slash
/// branch needs the extra checks.
fn is_directory_shaped(span: &str) -> bool {
    let is_directory_path = span.contains('/')
        && is_path_safe(span)
        && !BARE_FILE_EXTENSIONS.iter().any(|ext| span.ends_with(ext));
    is_directory_path || is_hyphenated_bare_word(span)
}

/// Whether `line` is a markdown ATX heading (`# `..`###### `): at most 3
/// leading spaces (see [`fence_line`] for the same CommonMark indentation
/// cap — 4+ makes a line an indented code block instead, so a `#` there is
/// literal prose, not a heading marker), then 1-6 `#` characters followed
/// by a space or end of line — NOT just "starts with `#`", which would
/// also match ordinary prose that happens to open a line with a literal
/// `#` (e.g. a bare issue reference like `#1234 tracks ...`) and wrongly
/// reset the section context.
fn is_heading(line: &str) -> bool {
    let indent = line.chars().take_while(|&c| c == ' ').count();
    if indent > 3 {
        return false;
    }
    // `indent` counts only ASCII spaces, so byte-slicing at that offset
    // can't land mid-codepoint.
    let rest = &line[indent..];
    let hashes = rest.chars().take_while(|c| *c == '#').count();
    if !(1..=6).contains(&hashes) {
        return false;
    }
    // `hashes` counts only ASCII '#' characters, so byte-slicing at that
    // offset can't land mid-codepoint.
    let rest = &rest[hashes..];
    rest.is_empty() || rest.starts_with([' ', '\t'])
}

/// A line that opens or closes a fenced-code delimiter, per CommonMark's
/// actual rules — three properties, not just "three or more backticks or
/// tildes":
/// - `marker`/`run`: which character repeats and how many times — the
///   fence's identity, and (see [`extract_path_citations`]) a closing fence
///   must be at least as long as the one that opened it, so a four-backtick
///   fence can safely contain a three-backtick example as literal content;
/// - indentation up to 3 leading spaces is still a fence delimiter; 4 or
///   more makes the line an indented code block instead, where the
///   backticks/tildes are just literal prose characters, not a fence at
///   all — [`fence_line`] returns `None` for those regardless of what
///   follows;
/// - `has_info_string`: whether anything besides trailing whitespace
///   follows the marker run (e.g. `` ```rust ``). CommonMark permits this
///   on an OPENING fence (an "info string") but says a closing fence "may
///   be followed only by spaces or tabs" — so the same line text means
///   something different depending on whether a fence is already open (see
///   the caller, which only lets a match without an info string close).
#[derive(Debug, PartialEq, Eq)]
struct FenceLine {
    marker: char,
    run: usize,
    has_info_string: bool,
}

fn fence_line(line: &str) -> Option<FenceLine> {
    let indent = line.chars().take_while(|&c| c == ' ').count();
    if indent > 3 {
        return None;
    }
    // `indent` counts only ASCII spaces, so byte-slicing at that offset
    // can't land mid-codepoint.
    let rest = &line[indent..];
    ['`', '~'].into_iter().find_map(|marker| {
        let run = rest.chars().take_while(|&c| c == marker).count();
        // A run shorter than 3 isn't a valid fence delimiter at all (this
        // is also what makes a blank line, or "take(3)" over one, not
        // vacuously match — `run` here is an exact count, never assumed).
        if run < 3 {
            return None;
        }
        let has_info_string = !rest[run..].trim().is_empty();
        Some(FenceLine {
            marker,
            run,
            has_info_string,
        })
    })
}

/// The possessive owner that narrows citation `i` (a scoped-extension span
/// like `agent.rs`, or a partial slash-path like `app/mod.rs`) to one
/// specific crate, if the text immediately before it (`parts[i - 1]`, the
/// outside-span between two backtick-quoted spans) matches a connector this
/// checker recognizes — `None` if `span` isn't itself a
/// [`SCOPED_BARE_EXTENSIONS`] citation, or the connector isn't one of the
/// three below. This is the single place that recognizes
/// possessive-ownership prose; extending it to a new phrasing means adding
/// one more arm here rather than touching [`extract_path_citations`]'s main
/// loop.
///
/// Three connectors are recognized:
/// - a literal `'s` immediately after a directory-shaped owner (``
///   `rocm-dash-tui`'s `agent.rs` `` or `` `crates/rocm-dash-tui`'s
///   `agent.rs` ``, split into `["rocm-dash-tui", "'s ", "agent.rs"]`)
///   starts a new possessive clause, owned by that directory;
/// - a literal `/` immediately after another citation already narrowed by
///   `current_owner` (`` `agent.rs`/`app.rs` ``, the doc's own convention
///   for "either file" — see `docs/architecture.md`'s ``
///   `diagnose.rs`/`examine.rs` ``) continues that same clause;
/// - the word `and` immediately after another citation already narrowed by
///   `current_owner` (`` `rocm-dash-tui`'s `agent.rs` and `app/mod.rs` ``,
///   the doc's own real phrasing) also continues that same clause — the
///   possessive still applies to the second citation grammatically, even
///   though the connecting word isn't a punctuation mark.
///
/// Either way, the whole chain narrows to one owner rather than just the
/// citation immediately after `'s`. This is a heuristic, not a parse of
/// English grammar: a citation connected by `and` to an *unrelated* prior
/// citation (rather than a shared possessive) would be mis-narrowed too —
/// acceptable for the doc's own limited prose conventions, not a general
/// solution.
fn possessive_owner_for<'a>(
    parts: &[&'a str],
    i: usize,
    is_scoped_extension: bool,
    current_owner: Option<&'a str>,
) -> Option<&'a str> {
    if !is_scoped_extension {
        return None;
    }
    if i >= 2 && parts[i - 1].trim() == "'s" && is_directory_shaped(parts[i - 2]) {
        return Some(parts[i - 2]);
    }
    if matches!(parts[i - 1].trim(), "/" | "and") {
        return current_owner;
    }
    None
}

/// Split `line` into alternating outside-text (even index) and
/// inside-code-span (odd index) segments — like `line.split('`')`, but
/// implementing CommonMark's actual code-span rule: a span opens at a
/// backtick run and closes at the NEXT run of the exact same length, not at
/// the next single backtick. The result strictly alternates, starting and
/// ending with an (possibly empty) outside-text segment, exactly like
/// `str::split`'s own guarantee, so [`extract_path_citations`] can index
/// `parts[i - 1]`/`parts[i - 2]` around a code-span index `i` the same way
/// it would against a real `split('`')` call.
///
/// This matters two ways a naive single-backtick split gets wrong:
/// - a double-backtick span (`` ``apps/rocm`` ``, delimited by TWO
///   backticks on each side) must itself become a citation candidate —
///   splitting on every individual backtick instead puts its content
///   ("apps/rocm") at an EVEN index, as if it were ordinary prose, so it's
///   never checked at all and a stale citation there goes undetected;
/// - a longer span whose content contains a literal, unmatched-length
///   backtick run (e.g. a double-backtick span demonstrating Markdown
///   syntax itself, `` ``prefix`old/path`suffix`` ``, one span whose
///   content happens to include single backticks) must stay ONE span —
///   splitting on every backtick instead fractures it, false-extracting
///   the enclosed text ("old/path") as if it were its own citation, which
///   can false-fail CI on a citation that was never really there.
///
/// An unmatched backtick run (no later run of the same length on this
/// line) is not a code span at all, per CommonMark — its backticks stay
/// literal text, and scanning continues from right after it for the next
/// potential opening run.
fn split_code_spans(line: &str) -> Vec<&str> {
    let bytes = line.as_bytes();
    let mut parts = Vec::new();
    let mut text_start = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < bytes.len() && bytes[i] == b'`' {
            i += 1;
        }
        let run_len = i - run_start;
        // Look for the next run of EXACTLY this length to close the span;
        // a run of a different length along the way is literal content
        // inside the (still-open) span candidate, not a delimiter.
        let mut j = i;
        let mut closing = None;
        while j < bytes.len() {
            if bytes[j] != b'`' {
                j += 1;
                continue;
            }
            let close_start = j;
            while j < bytes.len() && bytes[j] == b'`' {
                j += 1;
            }
            if j - close_start == run_len {
                closing = Some((close_start, j));
                break;
            }
        }
        let Some((close_start, close_end)) = closing else {
            continue;
        };
        parts.push(&line[text_start..run_start]);
        parts.push(&line[i..close_start]);
        text_start = close_end;
        i = close_end;
    }
    parts.push(&line[text_start..]);
    parts
}

/// Extract every backtick-quoted path citation from the doc's markdown
/// source, paired with its section context. Fenced code blocks (using
/// either backtick or tilde fences, see [`fence_line`]) are skipped — the
/// doc is prose today with none, but a future code example shouldn't have
/// its stray backticks misparsed as inline spans.
///
/// A heading line (see [`is_heading`]) becomes the "section directories"
/// for every path-candidate span cited on it AND on later lines, until the
/// next heading resets it (to a new set, or to empty for a heading with no
/// path in it, e.g. `## Module map`) — a heading's own citations use its
/// own directories, not the previous heading's, so a future heading that
/// bare-cites a scoped file in its own title wouldn't be checked against
/// stale leftover context. All of a citation's section directories must
/// hold for it to count as existing (see [`citation_exists`]) — correct for
/// a heading naming several crates that each independently make the same
/// claim (`` Both crates' `lib.rs` are not yet modularized ``). A citation
/// that instead names ONE specific crate from a multi-crate heading
/// (`` `rocm-dash-tui`'s `agent.rs` ``) is narrowed to just that crate —
/// see the possessive-connector check below — so it isn't wrongly held to
/// every crate the heading lists.
fn extract_path_citations(markdown: &str) -> BTreeSet<Citation> {
    let mut citations = BTreeSet::new();
    let mut fence: Option<(char, usize)> = None;
    let mut section_dirs: Vec<String> = Vec::new();
    for line in markdown.lines() {
        if let Some(candidate) = fence_line(line) {
            // A closing fence must use the same marker, be at least as long
            // as the opener (CommonMark) — a four-backtick fence can safely
            // contain a three-backtick example as literal content, only a
            // run of 4+ backticks (or a `~~~` line, different marker)
            // actually closes it — AND carry no info string (`` ```rust ``
            // can open a fence but can't close one; see [`FenceLine`]).
            fence = match fence {
                Some((open_marker, open_run))
                    if candidate.marker == open_marker
                        && candidate.run >= open_run
                        && !candidate.has_info_string =>
                {
                    None
                }
                Some(open) => Some(open),
                None => Some((candidate.marker, candidate.run)),
            };
            continue;
        }
        if fence.is_some() {
            continue;
        }
        // Alternates outside-span (even index) and inside-span (odd index)
        // segments, for any number of balanced inline spans on that line —
        // see [`split_code_spans`] for why this can't just be
        // `line.split('`')`.
        let parts: Vec<&str> = split_code_spans(line);
        let mut heading_dirs = Vec::new();
        for i in (1..parts.len()).step_by(2) {
            let span = parts[i];
            // Only directory-shaped candidates (a slash-path or a bare
            // hyphenated crate name) become section directories — a bare
            // extensioned file citation on the same line (however
            // unlikely on a real heading today) isn't itself a directory
            // and must not become one.
            if is_directory_shaped(span) {
                heading_dirs.push(span.to_string());
            }
        }
        // A heading's own citations are scoped to its OWN directories, not
        // whatever the previous heading left behind.
        let effective_section_dirs = if is_heading(line) {
            &heading_dirs
        } else {
            &section_dirs
        };
        // The owner of the possessive clause currently in progress, fed
        // into and updated by `possessive_owner_for` on each iteration; see
        // its doc comment for the connector grammar this recognizes. Reset
        // to `None` whenever a span isn't a path candidate at all, breaking
        // any chain in progress.
        let mut possessive_owner: Option<&str> = None;
        for i in (1..parts.len()).step_by(2) {
            let span = parts[i];
            if !is_path_candidate(span) {
                possessive_owner = None;
                continue;
            }

            // Not restricted to a bare (no `/`) span: a partial slash-path
            // file citation (`` `app/mod.rs` ``) is just as eligible for
            // possessive narrowing as a bare one (`` `agent.rs` ``) — see
            // [`citation_exists`] for why a partial suffix match needs this
            // scope just as much as a bare one does.
            let is_scoped_extension = SCOPED_BARE_EXTENSIONS.iter().any(|ext| span.ends_with(ext));
            let owner = possessive_owner_for(&parts, i, is_scoped_extension, possessive_owner);

            citations.insert(Citation {
                text: span.to_string(),
                section_dirs: match owner {
                    Some(owner) => vec![owner.to_string()],
                    None => effective_section_dirs.clone(),
                },
            });
            possessive_owner = owner;
        }
        if is_heading(line) {
            section_dirs = heading_dirs;
        }
    }
    citations
}

/// Whether `path` lives under `dir`. `dir` may be a full relative path
/// (`apps/rocmd`, checked as a component-wise prefix) or a bare crate
/// directory name (`rocm-dash-tui`, checked as any path component) — a
/// heading can cite either shape (`` ### `crates/rocm-dash-core`,
/// `rocm-dash-collectors`, ... ``), matching the two shapes
/// [`is_path_candidate`] accepts for a directory citation.
fn path_is_under(path: &Path, dir: &str) -> bool {
    if dir.contains('/') {
        path.starts_with(Path::new(dir))
    } else {
        path.components().any(|c| c.as_os_str() == dir)
    }
}

/// Whether `citation` still exists among `tracked` files.
///
/// A slash-containing citation matches immediately if some tracked file
/// *is* that path or lives under it as a directory (`Path::starts_with`,
/// which compares whole path components, not raw strings — this keeps
/// `apps/rocm` from spuriously matching the unrelated `apps/rocmd/...`).
///
/// Otherwise it falls back to a *partial* suffix match (`Path::ends_with`)
/// — the doc cites `` `app/mod.rs` `` (disambiguating which crate's
/// `mod.rs`, without repeating the full
/// `crates/rocm-dash-tui/src/app/mod.rs`). A bare suffix like this is
/// ambiguous by itself: if the doc's own crate later moved the file away
/// while an unrelated crate happened to have an identically-suffixed file,
/// the stale citation would still "match". So a partial match in a
/// [`SCOPED_BARE_EXTENSIONS`] citation is additionally constrained to the
/// citation's own section directories (via [`path_is_under`]), the same
/// scoping a bare citation gets below — every directory must have its own
/// matching file. A slash-path citation with no section context (or a
/// non-scoped extension) falls back to an unconstrained suffix match, same
/// as before.
///
/// A bare citation (no `/`) with a [`SCOPED_BARE_EXTENSIONS`] extension,
/// cited under at least one heading with known directories, is scoped to
/// those directories via [`path_is_under`] rather than matched anywhere in
/// the repo: the doc cites `lib.rs`/`main.rs` under several different
/// `### <crate>` headings, and ~17 files share those two bare names
/// workspace-wide, so an unscoped match would stay silent if the *specific*
/// file a section is talking about were renamed away, as long as some
/// unrelated crate's same-named file still existed. Every section
/// directory must have the file — not just one — so a citation naming
/// several crates at once (`` Both crates' `lib.rs` `` under a two-crate
/// heading) is verified for each of them; a citation narrowed to one
/// specific crate (see [`extract_path_citations`]) has only that single
/// directory to satisfy, so this is never stricter than intended.
///
/// A bare citation with a [`ROOT_LEVEL_BARE_EXTENSIONS`] extension is
/// matched only at the repo root (a single-component path equal to the
/// citation) rather than by path component anywhere: `` `Cargo.toml` ``
/// must mean the root manifest, not any of the workspace's 14 nested ones,
/// which would otherwise mask the root file going stale.
///
/// A bare citation with no section context (mentioned outside a
/// directory-naming heading) or a bare crate-directory name (unique
/// repo-wide, so scoping adds nothing) falls back to matching any path
/// component anywhere in the tree.
fn citation_exists(citation: &Citation, tracked: &[PathBuf]) -> bool {
    let text = citation.text.as_str();
    if text.contains('/') {
        let citation_path = Path::new(text);
        if tracked.iter().any(|p| p.starts_with(citation_path)) {
            return true;
        }
        let is_scoped_extension = SCOPED_BARE_EXTENSIONS.iter().any(|ext| text.ends_with(ext));
        if is_scoped_extension && !citation.section_dirs.is_empty() {
            return citation.section_dirs.iter().all(|dir| {
                tracked
                    .iter()
                    .any(|p| p.ends_with(citation_path) && path_is_under(p, dir))
            });
        }
        return tracked.iter().any(|p| p.ends_with(citation_path));
    }
    let is_scoped_extension = SCOPED_BARE_EXTENSIONS.iter().any(|ext| text.ends_with(ext));
    if is_scoped_extension && !citation.section_dirs.is_empty() {
        return citation.section_dirs.iter().all(|dir| {
            tracked
                .iter()
                .any(|p| path_is_under(p, dir) && p.file_name().is_some_and(|f| f == text))
        });
    }
    if ROOT_LEVEL_BARE_EXTENSIONS
        .iter()
        .any(|ext| text.ends_with(ext))
    {
        return tracked.iter().any(|p| p == Path::new(text));
    }
    tracked
        .iter()
        .any(|p| p.components().any(|c| c.as_os_str() == text))
}

/// Fetch the doc's current path citations and fail, naming every one, if
/// any no longer exist in the tracked tree.
pub fn run() -> Result<()> {
    let root = crate::paths::workspace_root()?;
    let doc_path = root.join(DOC_PATH);
    let markdown = std::fs::read_to_string(&doc_path)
        .with_context(|| format!("reading {}", doc_path.display()))?;
    let citations = extract_path_citations(&markdown);
    let tracked = tracked_files(&root)?;

    // `citations` is already a `BTreeSet` of distinct `(text, section_dirs)`
    // pairs, so this naturally reports the same bare text once per distinct
    // section it was cited (and found stale) under, rather than collapsing
    // them and losing which section's file is actually missing.
    let stale: Vec<&Citation> = citations
        .iter()
        .filter(|citation| !citation_exists(citation, &tracked))
        .collect();

    if !stale.is_empty() {
        bail!(stale_message(&stale));
    }
    Ok(())
}

/// Build the failure message naming every stale citation — and, for a
/// scoped bare citation, which section's directory it was expected under —
/// so a contributor doesn't have to manually re-derive which of the doc's
/// several same-named mentions (e.g. `lib.rs` under both `apps/rocmd` and
/// `crates/rocm-core`) is the one that's actually stale.
fn stale_message(stale: &[&Citation]) -> String {
    format!(
        "{DOC_PATH} cites {} path(s) that no longer exist in the tree:\n{}\n\
         update the citation to the path's new location, or remove it if the \
         file/directory is gone for good",
        stale.len(),
        stale
            .iter()
            .map(|citation| format_stale_citation(citation))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

/// One line of [`stale_message`]'s report for a single stale citation.
fn format_stale_citation(citation: &Citation) -> String {
    if citation.section_dirs.is_empty() {
        format!("  `{}`", citation.text)
    } else {
        format!(
            "  `{}` (expected under `{}`)",
            citation.text,
            citation.section_dirs.join("`, `")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_paths_and_extensioned_bare_files_are_candidates() {
        for path in [
            "apps/rocm",
            "apps/rocm/src/therock.rs",
            "crates/rocm-dash-tui/src/ui/approval.rs",
            "main.rs",
            "lib.rs",
            "Cargo.toml",
            "AGENTS.md",
            "runtime-deps.toml",
        ] {
            assert!(is_path_candidate(path), "expected {path} to be a candidate");
        }
    }

    #[test]
    fn hyphenated_bare_words_are_candidates() {
        assert!(is_path_candidate("rocm-dash-collectors"));
        assert!(is_path_candidate("rocm-dash-tui"));
    }

    #[test]
    fn code_syntax_and_identifiers_are_not_candidates() {
        for span in [
            "crate::",
            "pub(crate) fn",
            "comfyui::render_status(...)",
            "too_many_lines = \"allow\"",
            "mod x;",
            "pub mod x;",
            "pub use x::{...};",
            "ActionReport",
            "AnimatedSpinner",
            "ComfyuiCommand",
            "comfyui()",
            "runtimes()",
        ] {
            assert!(
                !is_path_candidate(span),
                "did not expect {span} to be a candidate"
            );
        }
    }

    #[test]
    fn bare_non_hyphenated_word_is_not_a_candidate() {
        // The documented blind spot: `xtask` is a real directory cited bare
        // in the doc, but is lexically identical in shape to a plain word
        // like `grep` (also cited bare, also not a path) — so neither is
        // treated as a candidate, favoring missing a rare citation over
        // false-flagging prose.
        assert!(!is_path_candidate("xtask"));
        assert!(!is_path_candidate("grep"));
    }

    #[test]
    fn split_code_spans_recognizes_a_double_backtick_span() {
        // Splitting on every individual backtick would put "apps/rocm" at
        // an EVEN index (as if it were ordinary prose between two
        // single-character delimiters), never checking it as a citation at
        // all — a real double-backtick span must land at an ODD index like
        // any other candidate.
        assert_eq!(split_code_spans("``apps/rocm``"), vec!["", "apps/rocm", ""]);
    }

    #[test]
    fn split_code_spans_keeps_an_embedded_shorter_run_inside_one_span() {
        // The content of this double-backtick span contains two literal
        // single backticks around "old/path" — CommonMark keeps the whole
        // thing as ONE span (closed only by the next double-backtick run),
        // not three spans that would false-extract "old/path" as its own
        // candidate.
        assert_eq!(
            split_code_spans("``prefix`old/path`suffix``"),
            vec!["", "prefix`old/path`suffix", ""]
        );
    }

    #[test]
    fn split_code_spans_treats_an_unmatched_run_as_literal_text() {
        // No closing run of the same length anywhere on the line: per
        // CommonMark the opening backticks are not a code span at all.
        assert_eq!(
            split_code_spans("prose ``` unmatched"),
            vec!["prose ``` unmatched"]
        );
    }

    #[test]
    fn extract_path_citations_recognizes_a_double_backtick_directory_citation() {
        let markdown = "See ``apps/rocm`` for the CLI entry point.\n";
        let citations = extract_path_citations(markdown);
        assert!(citations.iter().any(|c| c.text == "apps/rocm"));
    }

    fn tracked(paths: &[&str]) -> Vec<PathBuf> {
        paths.iter().map(PathBuf::from).collect()
    }

    fn citation(text: &str, section_dirs: &[&str]) -> Citation {
        Citation {
            text: text.to_string(),
            section_dirs: section_dirs.iter().map(ToString::to_string).collect(),
        }
    }

    #[test]
    fn exact_file_citation_matches() {
        let tracked = tracked(&["apps/rocm/src/therock.rs"]);
        assert!(citation_exists(
            &citation("apps/rocm/src/therock.rs", &[]),
            &tracked
        ));
    }

    #[test]
    fn directory_citation_matches_a_file_beneath_it() {
        let tracked = tracked(&["apps/rocm/src/main.rs", "apps/rocmd/src/lib.rs"]);
        assert!(citation_exists(&citation("apps/rocm", &[]), &tracked));
    }

    #[test]
    fn directory_citation_does_not_match_a_sibling_with_a_shared_prefix() {
        // `apps/rocm` must not spuriously match `apps/rocmd` via a plain
        // string-prefix check; component-wise `starts_with` gets this right.
        let tracked = tracked(&["apps/rocmd/src/lib.rs"]);
        assert!(!citation_exists(&citation("apps/rocm", &[]), &tracked));
    }

    #[test]
    fn partial_suffix_citation_matches_the_file_it_disambiguates() {
        // Regression case: the real doc cites `app/mod.rs`, a 2-component
        // suffix of `crates/rocm-dash-tui/src/app/mod.rs`, to disambiguate
        // it from other crates' `mod.rs` files without repeating the full
        // path.
        let tracked = tracked(&["crates/rocm-dash-tui/src/app/mod.rs"]);
        assert!(citation_exists(&citation("app/mod.rs", &[]), &tracked));
    }

    #[test]
    fn scoped_partial_suffix_citation_does_not_match_an_unrelated_crates_file() {
        // The precision gap a reviewer found: an unscoped partial suffix
        // match (`app/mod.rs`) would silently pass even after the doc's own
        // crate moved the file away, as long as some UNRELATED crate
        // happened to have an identically-suffixed `app/mod.rs` — the doc's
        // claim about `rocm-dash-tui` specifically would be stale but
        // undetected. Scoped to `rocm-dash-tui`, a same-named file living
        // only under a different crate must not satisfy it.
        let tracked = tracked(&["crates/rocm-dash-collectors/src/app/mod.rs"]);
        assert!(!citation_exists(
            &citation("app/mod.rs", &["rocm-dash-tui"]),
            &tracked
        ));
    }

    #[test]
    fn scoped_partial_suffix_citation_matches_its_own_crates_file() {
        let tracked = tracked(&[
            "crates/rocm-dash-collectors/src/app/mod.rs",
            "crates/rocm-dash-tui/src/app/mod.rs",
        ]);
        assert!(citation_exists(
            &citation("app/mod.rs", &["rocm-dash-tui"]),
            &tracked
        ));
    }

    #[test]
    fn extract_path_citations_narrows_a_partial_slash_path_citation_joined_by_and() {
        // The real doc's dashboard/telemetry section reads `` `rocm-dash-tui`'s
        // `agent.rs` and `app/mod.rs` `` — naming ONE specific crate from the
        // heading's four, not asserting the claim about all of them, via a
        // bare citation AND a partial slash-path citation joined by the word
        // "and" rather than a bare `/`. Both must narrow to `rocm-dash-tui`;
        // without that, the (correct, ALL-of) multi-directory check in
        // `citation_exists` would require each to exist under every one of
        // the other three crates too, which they do not — a false failure
        // on a doc that's actually accurate.
        let markdown = "\
### `crates/rocm-dash-core`, `rocm-dash-collectors`, `rocm-dash-daemon`, `rocm-dash-tui` — dashboard/telemetry

`rocm-dash-tui`'s `agent.rs` and `app/mod.rs` are **not yet modularized**.
";
        let citations = extract_path_citations(markdown);
        let agent_citation = citations
            .iter()
            .find(|c| c.text == "agent.rs")
            .expect("expected an agent.rs citation");
        let mod_citation = citations
            .iter()
            .find(|c| c.text == "app/mod.rs")
            .expect("expected an app/mod.rs citation");
        assert_eq!(
            agent_citation.section_dirs,
            vec!["rocm-dash-tui".to_string()]
        );
        assert_eq!(mod_citation.section_dirs, vec!["rocm-dash-tui".to_string()]);
    }

    #[test]
    fn unscoped_bare_file_name_matches_regardless_of_directory() {
        // No section context (e.g. an "Examples:" mention outside a
        // `### <crate>` heading) falls back to matching anywhere.
        let tracked = tracked(&["apps/rocm/src/main.rs", "apps/rocmd/src/lib.rs"]);
        assert!(citation_exists(&citation("main.rs", &[]), &tracked));
        assert!(citation_exists(&citation("lib.rs", &[]), &tracked));
    }

    #[test]
    fn scoped_bare_citation_does_not_match_an_unrelated_same_named_file() {
        // The precision gap a reviewer found: `lib.rs` cited under the
        // `apps/rocmd` heading must NOT be satisfied merely because some
        // other crate's `lib.rs` still exists — it has to be *that
        // section's* file specifically.
        let tracked = tracked(&[
            "crates/rocm-core/src/lib.rs",
            "crates/rocm-dash-core/src/lib.rs",
        ]);
        assert!(
            !citation_exists(&citation("lib.rs", &["apps/rocmd"]), &tracked),
            "apps/rocmd's lib.rs was deleted; an unrelated crate's lib.rs must not mask that"
        );
    }

    #[test]
    fn scoped_bare_citation_matches_its_own_sections_file() {
        let tracked = tracked(&["apps/rocmd/src/lib.rs", "crates/rocm-core/src/lib.rs"]);
        assert!(citation_exists(
            &citation("lib.rs", &["apps/rocmd"]),
            &tracked
        ));
    }

    #[test]
    fn scoped_bare_citation_resolves_a_bare_section_directory_name() {
        // Section dirs can themselves be bare crate names (`rocm-dash-tui`)
        // when a heading cites them without a `crates/` prefix —
        // `path_is_under` must resolve those the same as slash-qualified
        // section dirs. Both listed dirs have the file, so the (ALL-of)
        // check passes.
        let tracked = tracked(&[
            "crates/rocm-dash-core/src/agent.rs",
            "crates/rocm-dash-tui/src/agent.rs",
        ]);
        let scoped = citation("agent.rs", &["rocm-dash-core", "rocm-dash-tui"]);
        assert!(citation_exists(&scoped, &tracked));
    }

    #[test]
    fn multi_directory_citation_requires_every_directory_to_have_the_file() {
        // The precision gap a reviewer found: `` Both crates' `lib.rs` ``
        // under the `engines/lemonade`, `engines/vllm` heading asserts the
        // file exists in EACH of those crates, not merely in one of them —
        // if engines/vllm's lib.rs is renamed away while
        // engines/lemonade's is untouched, that must be caught, not masked
        // by the surviving lemonade file.
        let tracked = tracked(&["engines/lemonade/src/lib.rs"]);
        let both_crates = citation("lib.rs", &["engines/lemonade", "engines/vllm"]);
        assert!(
            !citation_exists(&both_crates, &tracked),
            "engines/vllm's lib.rs is gone; engines/lemonade's surviving lib.rs must not mask that"
        );
    }

    #[test]
    fn root_level_md_and_toml_citations_are_not_scoped_to_their_section() {
        // Regression case: the real doc cites `` `AGENTS.md` `` inside the
        // `crates/rocm-engine-protocol` section and `` `runtime-deps.toml` ``
        // inside `crates/rocm-deps` (explicitly calling it out as
        // "workspace-root" in the same sentence) — neither file lives under
        // that section's crate directory, so `.md`/`.toml` bare citations
        // must match at the repo root regardless of which section cites
        // them, rather than being scoped like `.rs` citations are.
        let tracked = tracked(&[
            "AGENTS.md",
            "runtime-deps.toml",
            "crates/rocm-deps/build.rs",
        ]);
        assert!(citation_exists(
            &citation("AGENTS.md", &["crates/rocm-engine-protocol"]),
            &tracked
        ));
        assert!(citation_exists(
            &citation("runtime-deps.toml", &["crates/rocm-deps"]),
            &tracked
        ));
    }

    #[test]
    fn root_level_toml_citation_is_not_masked_by_a_nested_manifest() {
        // Regression case a reviewer found: matching a bare `.md`/`.toml`
        // citation by "any path component anywhere" would let a stale root
        // `Cargo.toml` citation keep passing as long as ANY of the
        // workspace's 14 nested crate manifests still existed (each one's
        // basename is also literally `Cargo.toml`) — the same masking bug
        // `.rs` heading-scoping exists to prevent, just for a fixed root
        // location instead of a heading-derived one.
        let without_root = tracked(&["crates/rocm-core/Cargo.toml"]);
        assert!(
            !citation_exists(&citation("Cargo.toml", &[]), &without_root),
            "the root Cargo.toml is gone; a nested crate's manifest must not mask that"
        );

        let with_root = tracked(&["Cargo.toml", "crates/rocm-core/Cargo.toml"]);
        assert!(citation_exists(&citation("Cargo.toml", &[]), &with_root));
    }

    #[test]
    fn bare_crate_directory_name_matches_a_path_component() {
        let tracked = tracked(&["crates/rocm-dash-collectors/src/amd_smi.rs"]);
        assert!(citation_exists(
            &citation("rocm-dash-collectors", &[]),
            &tracked
        ));
    }

    #[test]
    fn removed_path_does_not_exist() {
        let tracked = tracked(&["apps/rocm/src/main.rs"]);
        assert!(!citation_exists(
            &citation("apps/rocm/src/removed.rs", &[]),
            &tracked
        ));
        assert!(!citation_exists(&citation("removed.rs", &[]), &tracked));
    }

    #[test]
    fn extract_path_citations_matches_the_real_docs_ambiguity() {
        let markdown = "\
See `apps/rocm/src/automations.rs` and bare `main.rs`, plus `crates/rocm-dash-core`.
Dispatch stays in `main.rs` via `crate::` and `pub(crate) fn` helpers, using
`ComfyuiCommand`/`comfyui()` and `too_many_lines = \"allow\"`. Check with `grep`.
";
        let citations = extract_path_citations(markdown);
        let texts: BTreeSet<&str> = citations.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            BTreeSet::from([
                "apps/rocm/src/automations.rs",
                "main.rs",
                "crates/rocm-dash-core",
            ])
        );
    }

    #[test]
    fn extract_path_citations_skips_fenced_code_blocks() {
        let markdown = "\
Prose citing `main.rs`.

```
`this/looks/like/a/path.rs` but is inside a fence and must be ignored
```

More prose citing `lib.rs`.
";
        let citations = extract_path_citations(markdown);
        let texts: BTreeSet<&str> = citations.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, BTreeSet::from(["main.rs", "lib.rs"]));
    }

    #[test]
    fn extract_path_citations_skips_tilde_fenced_code_blocks() {
        // Regression case: CommonMark also permits `~~~` fences; only
        // recognizing backtick fences would misparse a tilde-fenced
        // example's stray backticks as inline spans.
        let markdown = "\
Prose citing `main.rs`.

~~~
`this/looks/like/a/path.rs` but is inside a tilde fence and must be ignored
~~~

More prose citing `lib.rs`.
";
        let citations = extract_path_citations(markdown);
        let texts: BTreeSet<&str> = citations.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, BTreeSet::from(["main.rs", "lib.rs"]));
    }

    #[test]
    fn a_backtick_fence_line_does_not_close_an_open_tilde_fence() {
        // A `` ``` `` line inside a still-open `~~~` fence (e.g. a shell
        // snippet demonstrating backtick-fenced Markdown) is literal fence
        // content, not a close — only a matching marker closes a fence.
        let markdown = "\
~~~
```
`this/looks/like/a/path.rs` is still inside the outer tilde fence
~~~

Prose citing `lib.rs` after the fence closes.
";
        let citations = extract_path_citations(markdown);
        let texts: BTreeSet<&str> = citations.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, BTreeSet::from(["lib.rs"]));
    }

    #[test]
    fn a_shorter_same_marker_run_does_not_close_a_longer_opening_fence() {
        // CommonMark: a closing fence must be at least as long as its
        // opener. A four-backtick block can safely contain a
        // three-backtick example (e.g. this very file's own doc comments
        // demonstrating fenced Markdown) as literal content — the inner
        // ``` line must not prematurely close the outer ```` fence and
        // expose the path-like span between them.
        let markdown = "\
````
`this/looks/like/a/path.rs` is inside the four-backtick fence
```
`still/inside/the/four/backtick/fence.rs` too — the triple-backtick line above didn't close it
````

Prose citing `lib.rs` after the fence actually closes.
";
        let citations = extract_path_citations(markdown);
        let texts: BTreeSet<&str> = citations.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, BTreeSet::from(["lib.rs"]));
    }

    fn fence(marker: char, run: usize, has_info_string: bool) -> Option<FenceLine> {
        Some(FenceLine {
            marker,
            run,
            has_info_string,
        })
    }

    #[test]
    fn fence_line_requires_at_least_three_characters() {
        // A naive `chars().take(3).all(...)` would pass vacuously on a line
        // with fewer than 3 characters (including a blank line, which has
        // 0) — `fence_line` must require 3 real matching characters, not
        // "however many happened to be there".
        for non_fence in ["", "`", "``", "~", "~~", "prose"] {
            assert_eq!(fence_line(non_fence), None, "{non_fence:?} is not a fence");
        }
        assert_eq!(fence_line("```"), fence('`', 3, false));
        assert_eq!(fence_line("~~~"), fence('~', 3, false));
    }

    #[test]
    fn fence_line_reports_the_exact_run_length() {
        assert_eq!(fence_line("````"), fence('`', 4, false));
        assert_eq!(fence_line("~~~~~"), fence('~', 5, false));
    }

    #[test]
    fn fence_line_allows_up_to_three_leading_spaces() {
        assert_eq!(fence_line("   ```"), fence('`', 3, false));
    }

    #[test]
    fn four_or_more_leading_spaces_is_an_indented_code_block_not_a_fence() {
        // CommonMark: a code fence indented 4+ spaces is an indented code
        // block instead — the backticks there are literal prose characters,
        // not a fence delimiter, so `fence_line` must not match them.
        assert_eq!(fence_line("    ```"), None);
    }

    #[test]
    fn fence_line_reports_an_info_string_after_the_marker_run() {
        assert_eq!(fence_line("```rust"), fence('`', 3, true));
        // Trailing whitespace alone is not an info string.
        assert_eq!(fence_line("```   "), fence('`', 3, false));
    }

    #[test]
    fn an_info_string_line_opens_a_fence_but_cannot_close_one() {
        // CommonMark permits an info string (`` ```rust ``) on an OPENING
        // fence but says a closing fence "may be followed only by spaces or
        // tabs" — so the exact same line text must open a fence when none
        // is open, yet fail to close one that already is.
        let markdown = "\
```rust
`this/looks/like/a/path.rs` is inside the fence, opened with an info string
```rust
`still/inside/the/fence.rs` too — the info-string line above didn't close it
```

Prose citing `lib.rs` after the fence actually closes.
";
        let citations = extract_path_citations(markdown);
        let texts: BTreeSet<&str> = citations.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, BTreeSet::from(["lib.rs"]));
    }

    #[test]
    fn an_over_indented_fence_marker_does_not_hide_later_prose() {
        // A `` ``` `` indented 4+ spaces is an indented code block, not a
        // fence — it must not be misread as opening a fence that then
        // swallows the real citations after it as "inside the block".
        //
        // NOT a `"\` continuation string here: that strips all leading
        // whitespace off the next line, which would silently erase the
        // 4-space indent this test exists to exercise.
        let markdown =
            "    ```\nProse citing `lib.rs` right after an indented (non-fence) code block.\n";
        let citations = extract_path_citations(markdown);
        let texts: BTreeSet<&str> = citations.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, BTreeSet::from(["lib.rs"]));
    }

    #[test]
    fn blank_lines_inside_prose_do_not_toggle_fence_state() {
        // Regression case for the vacuous-match bug above: a blank line
        // between two citations (ordinary paragraph breaks, not a fence)
        // must not be misparsed as opening a fence and swallowing the
        // second citation.
        let markdown = "\
Prose citing `main.rs`.

More prose citing `lib.rs`.
";
        let citations = extract_path_citations(markdown);
        let texts: BTreeSet<&str> = citations.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, BTreeSet::from(["main.rs", "lib.rs"]));
    }

    #[test]
    fn is_heading_requires_atx_space_or_end_of_line() {
        for heading in ["# Architecture", "## Module map", "###### deep", "###"] {
            assert!(is_heading(heading), "expected {heading:?} to be a heading");
        }
        for prose in [
            "#1234 tracks the follow-up.",
            "#no-space-either",
            "text with # inside",
        ] {
            assert!(
                !is_heading(prose),
                "did not expect {prose:?} to be a heading"
            );
        }
    }

    #[test]
    fn is_heading_allows_up_to_three_leading_spaces() {
        assert!(is_heading("   # Architecture"));
    }

    #[test]
    fn four_or_more_leading_spaces_is_not_a_heading() {
        // CommonMark: 4+ leading spaces makes a line an indented code
        // block, not an ATX heading — a `#` there is literal prose. Without
        // this cap, an indented example line would wrongly reset
        // `section_dirs`, letting later bare citations lose their crate
        // scope.
        assert!(!is_heading("    # Architecture"));
    }

    #[test]
    fn a_bare_issue_reference_does_not_reset_section_context() {
        // Regression case: a reviewer found that naive "starts with #"
        // heading detection would misparse a line like `#1234 tracks ...`
        // as a heading, silently resetting section scope to empty.
        let markdown = "\
### `apps/rocmd` — background daemon

#1234 tracks a related follow-up.

`lib.rs` is **not yet modularized**.
";
        let citations = extract_path_citations(markdown);
        assert!(citations.contains(&citation("lib.rs", &["apps/rocmd"])));
    }

    #[test]
    fn heading_self_citation_uses_its_own_directories_not_the_previous_headings() {
        // Regression case: a reviewer found that a heading's own citations
        // were scoped to the PREVIOUS heading's directories, since
        // `section_dirs` was only reassigned after that line's citations
        // were already recorded.
        let markdown = "\
### `apps/rocmd` — background daemon

### `crates/rocm-core` `lib.rs` — core library
";
        let citations = extract_path_citations(markdown);
        let heading_lib_citation = citations
            .iter()
            .find(|c| c.text == "lib.rs")
            .expect("expected a lib.rs citation from the second heading");
        assert_eq!(heading_lib_citation.section_dirs, vec!["crates/rocm-core"]);
    }

    #[test]
    fn bare_extension_alone_is_not_a_candidate() {
        // Regression case: a reviewer found `".rs".ends_with(".rs")` is
        // trivially true, so prose like "renamed to use the `.rs`
        // extension" would be misparsed as a citation of a file literally
        // named `.rs`.
        assert!(!is_path_candidate(".rs"));
        assert!(!is_path_candidate(".md"));
        assert!(!is_path_candidate(".toml"));
    }

    #[test]
    fn extract_path_citations_scopes_bare_citations_to_the_preceding_heading() {
        let markdown = "\
### `apps/rocmd` — background daemon

`lib.rs` is **not yet modularized**.

### `crates/rocm-core` — core library

`lib.rs` itself is **not yet modularized**.
";
        let citations = extract_path_citations(markdown);
        assert!(citations.contains(&citation("lib.rs", &["apps/rocmd"])));
        assert!(citations.contains(&citation("lib.rs", &["crates/rocm-core"])));
    }

    #[test]
    fn extract_path_citations_collects_every_heading_directory_shape() {
        // The real doc's dashboard/telemetry heading mixes one
        // slash-qualified directory with three bare crate names. A bare
        // `.rs` citation with no possessive qualifier (unlike `agent.rs`
        // just below it, see the next test) keeps the whole list.
        let markdown = "\
### `crates/rocm-dash-core`, `rocm-dash-collectors`, `rocm-dash-daemon`, `rocm-dash-tui` — dashboard/telemetry

Every listed crate's `mod.rs` is a placeholder example, not real doc prose.
";
        let citations = extract_path_citations(markdown);
        let mod_citation = citations
            .iter()
            .find(|c| c.text == "mod.rs")
            .expect("expected a mod.rs citation");
        assert_eq!(
            mod_citation.section_dirs,
            vec![
                "crates/rocm-dash-core".to_string(),
                "rocm-dash-collectors".to_string(),
                "rocm-dash-daemon".to_string(),
                "rocm-dash-tui".to_string(),
            ]
        );
    }

    #[test]
    fn extract_path_citations_narrows_a_slash_qualified_possessive_owner_too() {
        // Same shape as the bare-owner case above, but the possessive owner
        // is spelled as a slash-path (`` `crates/rocm-dash-tui`'s ``) rather
        // than a bare hyphenated crate name — a shape the heading itself
        // already mixes with bare names on this same line. This must narrow
        // exactly like the bare-owner case, not fall back to the whole
        // heading's directory list.
        let markdown = "\
### `crates/rocm-dash-core`, `rocm-dash-collectors`, `rocm-dash-daemon`, `rocm-dash-tui` — dashboard/telemetry

`crates/rocm-dash-tui`'s `agent.rs` is **not yet modularized**.
";
        let citations = extract_path_citations(markdown);
        let agent_citation = citations
            .iter()
            .find(|c| c.text == "agent.rs")
            .expect("expected an agent.rs citation");
        assert_eq!(
            agent_citation.section_dirs,
            vec!["crates/rocm-dash-tui".to_string()]
        );
    }

    #[test]
    fn extract_path_citations_narrows_every_citation_in_a_slash_chained_possessive() {
        // The real doc's module-map prose reads `` `crates/rocm-core`'s
        // `diagnose.rs`/`examine.rs` `` — TWO bare `.rs` citations chained
        // by `/` after a single possessive owner, not just one. Both must
        // narrow to that owner, not just the citation immediately after
        // `'s` (which would otherwise leave the second one scoped to the
        // whole heading's crate list — or, worse, a false CI failure if
        // that second file only exists in the possessive owner's crate).
        let markdown = "\
### `crates/rocm-dash-core`, `rocm-dash-collectors`, `rocm-dash-daemon`, `rocm-dash-tui` — dashboard/telemetry

`rocm-dash-tui`'s `agent.rs`/`app.rs` are **not yet modularized**.
";
        let citations = extract_path_citations(markdown);
        let agent_citation = citations
            .iter()
            .find(|c| c.text == "agent.rs")
            .expect("expected an agent.rs citation");
        let app_citation = citations
            .iter()
            .find(|c| c.text == "app.rs")
            .expect("expected an app.rs citation");
        assert_eq!(
            agent_citation.section_dirs,
            vec!["rocm-dash-tui".to_string()]
        );
        assert_eq!(app_citation.section_dirs, vec!["rocm-dash-tui".to_string()]);
    }

    #[test]
    fn extract_path_citations_still_scopes_a_slash_path_citation_to_its_heading() {
        // `citation_exists` never consults `section_dirs` for a
        // slash-containing citation's own existence check (it matches by
        // path component/prefix/suffix instead) — but `format_stale_citation`
        // still reads it to print a "(expected under ...)" hint if the
        // citation ever goes stale. A slash-path citation must keep that
        // heading scope, same as a bare citation, even though one of its two
        // consumers doesn't need it.
        let markdown = "\
### `apps/rocmd` — daemon

See `apps/rocmd/src/main.rs` for the entry point.
";
        let citations = extract_path_citations(markdown);
        let main_citation = citations
            .iter()
            .find(|c| c.text == "apps/rocmd/src/main.rs")
            .expect("expected an apps/rocmd/src/main.rs citation");
        assert_eq!(main_citation.section_dirs, vec!["apps/rocmd".to_string()]);
    }

    #[test]
    fn extract_path_citations_ignores_a_malformed_possessive_owner() {
        // A directory-shaped check based on "contains a slash" alone would
        // accept a slash-adjacent span with stray prose punctuation
        // (`foo/bar!`) as a possessive owner, scoping `main.rs` to a
        // directory that can never match any real tracked path — a false
        // stale-citation failure on a doc that's actually accurate. The
        // malformed owner must be rejected, falling back to the (empty,
        // this line isn't under any heading) section scope instead.
        let markdown = "`foo/bar!`'s `main.rs` is not yet modularized.\n";
        let citations = extract_path_citations(markdown);
        let main_citation = citations
            .iter()
            .find(|c| c.text == "main.rs")
            .expect("expected a main.rs citation");
        assert!(main_citation.section_dirs.is_empty());
    }

    #[test]
    fn extract_path_citations_ignores_a_full_file_path_as_a_possessive_owner() {
        // A directory-shaped check based on "contains a slash" alone,
        // without excluding file extensions, would accept a full file path
        // (`` `crates/rocm-core/src/diagnose.rs`'s `` ) as if it were the
        // directory containing it — scoping `examine.rs` to a directory
        // that can never match any real tracked path (no file's parent is
        // itself a file). The owner must be rejected the same way a
        // malformed one is, falling back to the section scope instead.
        let markdown =
            "`crates/rocm-core/src/diagnose.rs`'s `examine.rs` is not yet modularized.\n";
        let citations = extract_path_citations(markdown);
        let examine_citation = citations
            .iter()
            .find(|c| c.text == "examine.rs")
            .expect("expected an examine.rs citation");
        assert!(examine_citation.section_dirs.is_empty());
    }

    #[test]
    fn extract_path_citations_ignores_a_full_file_path_as_a_heading_directory() {
        // The same file-path-vs-directory distinction applies to heading
        // directories: a heading that backtick-quotes a full file path
        // (however unlikely today) must not be collected as if it were a
        // section directory — that would scope every unscoped bare `.rs`
        // citation under it to an unmatchable directory, false-failing
        // every one of them.
        let markdown = "\
### `crates/rocm-core/src/diagnose.rs` — an unlikely heading shape

`lib.rs` is **not yet modularized**.
";
        let citations = extract_path_citations(markdown);
        let lib_citation = citations
            .iter()
            .find(|c| c.text == "lib.rs")
            .expect("expected a lib.rs citation");
        assert!(lib_citation.section_dirs.is_empty());
    }

    #[test]
    fn run_passes_against_the_real_doc() {
        // Regression guard against the real workspace: exercises the full
        // read-doc -> extract -> tracked-files -> existence-check path, the
        // same path `cargo xtask check-architecture-doc` runs in CI.
        run().expect("docs/architecture.md's path citations should all exist");
    }

    #[test]
    fn stale_message_names_every_stale_path() {
        let unscoped = citation("apps/rocm/src/deleted.rs", &[]);
        let hyphenated = citation("old-crate-dir", &[]);
        let message = stale_message(&[&unscoped, &hyphenated]);
        assert!(message.contains("2 path(s)"));
        assert!(message.contains("`apps/rocm/src/deleted.rs`"));
        assert!(message.contains("`old-crate-dir`"));
    }

    #[test]
    fn stale_message_names_the_expected_section_for_a_scoped_citation() {
        // Regression case: a reviewer found the message collapsed `lib.rs`
        // cited (and stale) under two different sections into one
        // unhelpful line — a contributor shouldn't have to guess which
        // section's file actually went missing.
        let apps_rocmd_lib = citation("lib.rs", &["apps/rocmd"]);
        let message = stale_message(&[&apps_rocmd_lib]);
        assert!(message.contains("`lib.rs` (expected under `apps/rocmd`)"));
    }
}
