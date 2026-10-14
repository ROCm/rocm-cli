// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Nightly release-readiness summary: a mechanical GO/NO-GO verdict and
//! status table, plus the categorized commit/issue facts an LLM narrative
//! step drafts prose from.
//!
//! The verdict, gate rows, and commit/issue counts are computed entirely from
//! `git`/`gh` data and the caller-supplied job results and E2E report — never
//! by the LLM step. That step is only ever handed [`Facts`] and asked to draft
//! prose from them, so a bad or missing narrative pass can never change what
//! the table says.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Serialize;

/// One categorized commit: its squash-merge PR number (if the subject carries
/// a trailing `(#NNN)`, GitHub's convention for this repo) and the subject
/// with its Conventional Commit type prefix stripped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommitEntry {
    pub pr: Option<u64>,
    pub subject: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CommitBuckets {
    pub features: Vec<CommitEntry>,
    pub fixes: Vec<CommitEntry>,
    /// Commits using the `!` breaking-change marker (e.g. `feat!:`,
    /// `fix(scope)!:`). No commit in this repo's history uses the Conventional
    /// Commits `BREAKING CHANGE:` footer instead, so only the `!` form is
    /// detected; add footer scanning if that convention starts being used.
    pub breaking: Vec<CommitEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KnownIssue {
    pub number: u64,
    pub title: String,
    pub reactions: u64,
    /// First ~240 characters of the issue body, single-line. Title alone is
    /// often not enough to tell an end-user-blocking bug from a CI/internal
    /// one, which the narrative step is asked to filter by; this gives it
    /// the content to judge that from.
    pub body_excerpt: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Verdict {
    Go,
    NoGo,
}

#[derive(Debug, Clone, Serialize)]
pub struct Facts {
    pub since_tag: String,
    pub verdict: Verdict,
    /// Named gates that did not pass (build/windows-build job results, or
    /// "E2E" if any platform reported a failure or the report couldn't be
    /// read). Empty iff `verdict` is `Go`.
    pub failing_gates: Vec<String>,
    /// `None` when the E2E report's `**Total**` row couldn't be found or
    /// parsed — treated as a failing gate rather than silently reading as
    /// zero failures, since a missing row more often means the report is
    /// broken than that E2E had a perfect run.
    pub e2e_fail_total: Option<u64>,
    pub commits: CommitBuckets,
    pub known_issues: Vec<KnownIssue>,
}

pub fn run(
    since_tag: &str,
    e2e_report: &Path,
    nightly_result: &str,
    windows_result: &str,
    out: &Path,
    table_out: &Path,
) -> Result<()> {
    let subjects = collect_commit_subjects(since_tag)?;
    let commits = categorize(&subjects);
    let known_issues = collect_known_issues()?;

    let e2e_text = std::fs::read_to_string(e2e_report)
        .with_context(|| format!("reading e2e report {}", e2e_report.display()))?;
    let e2e_fail_total = total_e2e_failures(&e2e_text);

    let facts = assemble_facts(
        since_tag,
        commits,
        known_issues,
        e2e_fail_total,
        nightly_result,
        windows_result,
    );

    write_parented(out, &serde_json::to_string_pretty(&facts)?)?;
    write_parented(table_out, &render_table(&facts))?;
    Ok(())
}

/// Pure assembly of [`Facts`] from already-collected inputs: no I/O, so the
/// GO/NO-GO logic can be unit-tested without shelling out to `git`/`gh` or
/// touching the filesystem.
fn assemble_facts(
    since_tag: &str,
    commits: CommitBuckets,
    known_issues: Vec<KnownIssue>,
    e2e_fail_total: Option<u64>,
    nightly_result: &str,
    windows_result: &str,
) -> Facts {
    let mut failing_gates = Vec::new();
    // `skipped` is a legitimate outcome on a no-op night (e.g.
    // windows-nightly's own skip flag), not a failure to gate on.
    if !matches!(nightly_result, "success" | "skipped") {
        failing_gates.push(format!("nightly build ({nightly_result})"));
    }
    if !matches!(windows_result, "success" | "skipped") {
        failing_gates.push(format!("windows build ({windows_result})"));
    }
    match e2e_fail_total {
        None => failing_gates.push("E2E (report unreadable)".to_string()),
        Some(n) if n > 0 => failing_gates.push(format!("E2E ({n} failure(s))")),
        Some(_) => {}
    }

    let verdict = if failing_gates.is_empty() {
        Verdict::Go
    } else {
        Verdict::NoGo
    };

    Facts {
        since_tag: since_tag.to_string(),
        verdict,
        failing_gates,
        e2e_fail_total,
        commits,
        known_issues,
    }
}

fn write_parented(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating output dir {}", parent.display()))?;
    }
    std::fs::write(path, contents).with_context(|| format!("writing {}", path.display()))
}

/// Commit subjects (first line only) in `since_tag..HEAD`, oldest first.
fn collect_commit_subjects(since_tag: &str) -> Result<Vec<String>> {
    let range = format!("{since_tag}..HEAD");
    let output = Command::new("git")
        .args(["log", "--format=%s", "--reverse", &range])
        .output()
        .with_context(|| format!("failed to run `git log {range}`"))?;
    if !output.status.success() {
        bail!(
            "`git log {range}` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect())
}

/// Conventional Commit scopes that denote internal CI/dev tooling rather than
/// CLI behavior an end user would notice. A multi-scope commit (e.g.
/// `fix(core,ci): ...`) is kept if *any* of its scopes falls outside this
/// set — only commits scoped *entirely* to internal tooling are dropped.
const INTERNAL_SCOPES: &[&str] = &[
    "ci",
    "nightly",
    "github",
    "hooks",
    "xtask",
    "build",
    "deps",
    "e2e",
    "e2e-report",
    "e2e-cucumber",
    "e2e-prewarm",
];

fn is_internal_only_scope(scope: &str) -> bool {
    scope
        .split(',')
        .all(|s| INTERNAL_SCOPES.contains(&s.trim()))
}

/// Classify commit subjects by their Conventional Commit type prefix
/// (`feat`/`fix`), splitting out `!`-marked breaking changes into their own
/// bucket rather than double-counting them under feature/fix. Commits scoped
/// entirely to internal CI/dev tooling (see [`INTERNAL_SCOPES`]) are dropped:
/// this summary is for end-user-facing functionality and fixes, not the
/// workflow that generates it.
pub fn categorize(subjects: &[String]) -> CommitBuckets {
    let mut buckets = CommitBuckets::default();
    for subject in subjects {
        let Some(colon) = subject.find(':') else {
            continue;
        };
        let head = &subject[..colon];
        let mut rest = subject[colon + 1..].trim().to_string();
        let breaking = head.ends_with('!');
        let head_no_bang = head.trim_end_matches('!');
        let kind = head_no_bang.split('(').next().unwrap_or("");
        let scope = head_no_bang
            .find('(')
            .and_then(|i| head_no_bang[i + 1..].strip_suffix(')'));
        if scope.is_some_and(is_internal_only_scope) {
            continue;
        }
        let pr = extract_pr_number(subject);
        // `rest` still carries the trailing `(#NNN)` GitHub appends on
        // squash-merge, since it's after the colon too; strip it here so
        // `render_table` isn't left printing `subject (#NNN) (#NNN)`.
        if let Some(n) = pr {
            let suffix = format!(" (#{n})");
            if let Some(stripped) = rest.strip_suffix(&suffix) {
                rest = stripped.to_string();
            }
        }
        let entry = CommitEntry { pr, subject: rest };
        if breaking {
            buckets.breaking.push(entry);
        } else {
            match kind {
                "feat" => buckets.features.push(entry),
                "fix" => buckets.fixes.push(entry),
                _ => {}
            }
        }
    }
    buckets
}

/// Extract a trailing `(#NNN)` GitHub squash-merge PR reference, if present.
/// Only matches when `(#NNN)` is the very end of the subject (after
/// trimming trailing whitespace) — `(#NNN)` appearing mid-sentence (e.g.
/// `"fix: handle (#12) refs"`) is not a PR reference.
fn extract_pr_number(subject: &str) -> Option<u64> {
    let trimmed = subject.trim_end();
    if !trimmed.ends_with(')') {
        return None;
    }
    let start = trimmed.rfind("(#")?;
    trimmed[start + 2..trimmed.len() - 1].parse().ok()
}

#[derive(serde::Deserialize)]
struct RawReactionGroup {
    users: RawUsers,
}
#[derive(serde::Deserialize)]
struct RawUsers {
    #[serde(rename = "totalCount")]
    total_count: u64,
}
#[derive(serde::Deserialize)]
struct RawIssue {
    number: u64,
    title: String,
    #[serde(default)]
    body: String,
    #[serde(rename = "reactionGroups", default)]
    reaction_groups: Vec<RawReactionGroup>,
}

/// All open `bug`-labeled issues, via the preinstalled, token-authenticated
/// `gh` CLI (same pattern as `verify_commits::verified_via_github`). Deciding
/// which of these are actually relevant to end users (as opposed to
/// non-blocking or CI/internal-only) needs reading the issue, which is a
/// judgment call for the narrative step, not something this mechanical pass
/// can sort or cut down to a top-N by itself; it only ranks by reaction count
/// as a readability order for the raw list in `table.md`.
fn collect_known_issues() -> Result<Vec<KnownIssue>> {
    let output = Command::new("gh")
        .args([
            "issue",
            "list",
            "--label",
            "bug",
            "--state",
            "open",
            "--json",
            "number,title,body,reactionGroups",
            "--limit",
            "100",
        ])
        .output()
        .context(
            "failed to run `gh issue list` (is the GitHub CLI installed and authenticated?)",
        )?;
    if !output.status.success() {
        bail!(
            "`gh issue list` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let raw: Vec<RawIssue> =
        serde_json::from_slice(&output.stdout).context("parsing `gh issue list` JSON output")?;

    let mut issues: Vec<KnownIssue> = raw
        .into_iter()
        .map(|i| KnownIssue {
            number: i.number,
            title: i.title,
            reactions: i.reaction_groups.iter().map(|g| g.users.total_count).sum(),
            body_excerpt: excerpt(&i.body, 240),
        })
        .collect();
    issues.sort_by(|a, b| b.reactions.cmp(&a.reactions).then(a.number.cmp(&b.number)));
    Ok(issues)
}

/// First `max_chars` characters of `text`, collapsed to a single line.
fn excerpt(text: &str, max_chars: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        collapsed
    } else {
        let mut truncated: String = collapsed.chars().take(max_chars).collect();
        truncated.push('…');
        truncated
    }
}

/// Sum of the `Fail` column across the consolidated E2E markdown's `**Total**`
/// row (`crates/e2e-report`'s `consolidated_summary_markdown` format: `|
/// Platform | OS | Total | Pass | Fail | Skip | Xfail | Status |`). Parsing
/// the already-rendered totals row, rather than re-deriving per-platform
/// pass/fail here, keeps this in lockstep with that crate's known-bugs/xfail
/// handling instead of duplicating it.
fn total_e2e_failures(markdown: &str) -> Option<u64> {
    markdown
        .lines()
        .find(|l| l.trim_start().starts_with("| **Total**"))
        .and_then(|line| line.split('|').nth(5))
        .and_then(|cell| cell.trim().parse().ok())
}

/// Neutralize markdown syntax in untrusted text (GitHub issue titles) before
/// it's embedded in the generated wiki page. Newlines are collapsed to a
/// space since each issue renders as a single bullet line.
fn escape_markdown(text: &str) -> String {
    text.chars()
        .flat_map(|c| match c {
            '\n' | '\r' => vec![' '],
            '|' | '[' | ']' | '*' | '_' | '`' | '<' | '>' => vec!['\\', c],
            _ => vec![c],
        })
        .collect()
}

pub(crate) fn render_table(facts: &Facts) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let (verdict_text, verdict_glyph) = match facts.verdict {
        Verdict::Go => ("GO", "✅"),
        Verdict::NoGo => ("NO-GO", "❌"),
    };
    let _ = writeln!(
        out,
        "## Release Readiness — {verdict_glyph} {verdict_text}\n"
    );
    let _ = writeln!(out, "Commits considered: since `{}`\n", facts.since_tag);

    // Unreadable E2E evidence means the gate table itself can't be trusted;
    // render a short warning instead of a table carrying a bogus "0
    // failures" row. `publish-release-readiness-wiki`'s guard greps for the
    // table header below, so omitting it here also makes that guard trip.
    let Some(e2e_fail_total) = facts.e2e_fail_total else {
        out.push_str(
            "⚠️ E2E evidence was unreadable; the gate table could not be rendered. See `failing_gates` in `facts.json`.\n",
        );
        return out;
    };

    out.push_str("| Area | Evidence | Status |\n|---|---|:--:|\n");
    let _ = writeln!(
        out,
        "| Build (Linux + Windows nightly) | nightly / windows-nightly jobs | {} |",
        if facts.failing_gates.iter().any(|g| g.contains("build")) {
            "❌"
        } else {
            "✅"
        }
    );
    let _ = writeln!(
        out,
        "| E2E | {e2e_fail_total} failure(s) across all platforms | {} |",
        if e2e_fail_total > 0 { "❌" } else { "✅" }
    );
    out.push('\n');

    let section = |out: &mut String, title: &str, entries: &[CommitEntry]| {
        let _ = writeln!(out, "### {title} ({})\n", entries.len());
        if entries.is_empty() {
            out.push_str("_None._\n\n");
            return;
        }
        for e in entries {
            match e.pr {
                Some(pr) => {
                    let _ = writeln!(out, "- {} (#{pr})", e.subject);
                }
                None => {
                    let _ = writeln!(out, "- {}", e.subject);
                }
            }
        }
        out.push('\n');
    };
    section(&mut out, "New Features", &facts.commits.features);
    section(&mut out, "Bug Fixes", &facts.commits.fixes);
    section(&mut out, "Breaking Changes", &facts.commits.breaking);

    let _ = writeln!(
        out,
        "### Known Issues — raw list ({} open, `bug`-labeled)\n",
        facts.known_issues.len()
    );
    out.push_str(
        "_Unfiltered; see the narrative below for the issues judged relevant to end-user experience._\n\n",
    );
    if facts.known_issues.is_empty() {
        out.push_str("_None._\n");
    } else {
        for issue in &facts.known_issues {
            let _ = writeln!(
                out,
                "- #{} {} (👍 {})",
                issue.number,
                escape_markdown(&issue.title),
                issue.reactions
            );
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorize_buckets_feat_fix_and_breaking() {
        let subjects = vec![
            "feat: add support for vLLM multi-GPU serving (#123)".to_string(),
            "fix(scope): correct VRAM probe (#124)".to_string(),
            "fix!: drop deprecated --legacy-flag (#125)".to_string(),
            "docs: clarify GPU selection flag behavior".to_string(),
            "chore: bump rust-toolchain".to_string(),
        ];
        let got = categorize(&subjects);
        assert_eq!(
            got.features,
            vec![CommitEntry {
                pr: Some(123),
                subject: "add support for vLLM multi-GPU serving".to_string(),
            }]
        );
        assert_eq!(
            got.fixes,
            vec![CommitEntry {
                pr: Some(124),
                subject: "correct VRAM probe".to_string(),
            }]
        );
        assert_eq!(
            got.breaking,
            vec![CommitEntry {
                pr: Some(125),
                subject: "drop deprecated --legacy-flag".to_string(),
            }]
        );
    }

    #[test]
    fn categorize_drops_commits_scoped_entirely_to_internal_tooling() {
        let subjects = vec![
            "fix(nightly): ensure a newline before the GITHUB_OUTPUT delimiter".to_string(),
            "fix(ci): poll the GPU at least once before the preflight deadline (#608)".to_string(),
            "fix(core,ci): bound local-service HTTP reads; raise strix-ubuntu E2E job timeout (#348)"
                .to_string(),
            "fix: report the ComfyUI source folder consistently in status and install (#600)"
                .to_string(),
        ];
        let got = categorize(&subjects);
        assert_eq!(
            got.fixes,
            vec![
                CommitEntry {
                    pr: Some(348),
                    subject: "bound local-service HTTP reads; raise strix-ubuntu E2E job timeout"
                        .to_string(),
                },
                CommitEntry {
                    pr: Some(600),
                    subject: "report the ComfyUI source folder consistently in status and install"
                        .to_string(),
                },
            ]
        );
    }

    #[test]
    fn extract_pr_number_reads_trailing_reference() {
        assert_eq!(
            extract_pr_number("fix: correct VRAM probe (#124)"),
            Some(124)
        );
        assert_eq!(extract_pr_number("fix: correct VRAM probe"), None);
    }

    #[test]
    fn extract_pr_number_ignores_a_parenthetical_mid_sentence() {
        assert_eq!(extract_pr_number("fix: handle (#12) refs"), None);
    }

    #[test]
    fn excerpt_collapses_whitespace_and_truncates_with_ellipsis() {
        assert_eq!(excerpt("short text", 240), "short text");
        assert_eq!(
            excerpt("line one\nline two\n\nline three", 240),
            "line one line two line three"
        );
        assert_eq!(excerpt("abcdefghij", 5), "abcde…");
    }

    #[test]
    fn total_e2e_failures_reads_the_totals_row() {
        let md = "| Platform | OS | Total | Pass | Fail | Skip | Xfail | Status |\n\
                   |---|---|--:|--:|--:|--:|--:|:--|\n\
                   | Mock | Linux | 10 | 9 | 1 | 0 | n/a | FAIL |\n\
                   | **Total** | | 10 | 9 | 1 | 0 | 0 | |\n";
        assert_eq!(total_e2e_failures(md), Some(1));
    }

    #[test]
    fn total_e2e_failures_returns_none_without_a_totals_row() {
        assert_eq!(total_e2e_failures("no table here"), None);
    }

    #[test]
    fn assemble_facts_is_go_when_everything_succeeds() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::Go);
        assert!(facts.failing_gates.is_empty());
    }

    #[test]
    fn assemble_facts_is_no_go_when_e2e_evidence_is_unreadable() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            None,
            "success",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["E2E (report unreadable)"]);
    }

    #[test]
    fn assemble_facts_treats_a_skipped_windows_build_as_passing() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "skipped",
        );
        assert_eq!(facts.verdict, Verdict::Go);
        assert!(facts.failing_gates.is_empty());
    }

    #[test]
    fn assemble_facts_is_no_go_when_windows_build_fails() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "failure",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["windows build (failure)"]);
    }

    #[test]
    fn render_table_omits_the_header_when_e2e_evidence_is_unreadable() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            None,
            "success",
            "success",
        );
        let table = render_table(&facts);
        assert!(!table.contains("| Area | Evidence | Status |"));
    }

    #[test]
    fn render_table_escapes_markdown_in_known_issue_titles() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![KnownIssue {
                number: 1,
                title: "![x](evil) | injected row".to_string(),
                reactions: 0,
                body_excerpt: String::new(),
            }],
            Some(0),
            "success",
            "success",
        );
        let table = render_table(&facts);
        assert!(table.contains("| Area | Evidence | Status |"));
        assert!(table.contains(r"!\[x\](evil) \| injected row"));
        assert!(!table.contains("![x](evil) | injected row"));
    }
}
