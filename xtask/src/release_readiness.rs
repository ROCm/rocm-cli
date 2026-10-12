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
    /// "E2E" if any platform reported a failure). Empty iff `verdict` is `Go`.
    pub failing_gates: Vec<String>,
    pub e2e_fail_total: u64,
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

    let mut failing_gates = Vec::new();
    if nightly_result != "success" {
        failing_gates.push(format!("nightly build ({nightly_result})"));
    }
    if windows_result != "success" {
        failing_gates.push(format!("windows build ({windows_result})"));
    }
    if e2e_fail_total > 0 {
        failing_gates.push(format!("E2E ({e2e_fail_total} failure(s))"));
    }

    let verdict = if failing_gates.is_empty() {
        Verdict::Go
    } else {
        Verdict::NoGo
    };

    let facts = Facts {
        since_tag: since_tag.to_string(),
        verdict,
        failing_gates,
        e2e_fail_total,
        commits,
        known_issues,
    };

    write_parented(out, &serde_json::to_string_pretty(&facts)?)?;
    write_parented(table_out, &render_table(&facts))?;
    Ok(())
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

/// Classify commit subjects by their Conventional Commit type prefix
/// (`feat`/`fix`), splitting out `!`-marked breaking changes into their own
/// bucket rather than double-counting them under feature/fix.
pub fn categorize(subjects: &[String]) -> CommitBuckets {
    let mut buckets = CommitBuckets::default();
    for subject in subjects {
        let Some(colon) = subject.find(':') else {
            continue;
        };
        let head = &subject[..colon];
        let mut rest = subject[colon + 1..].trim().to_string();
        let breaking = head.ends_with('!');
        let kind = head.trim_end_matches('!').split('(').next().unwrap_or("");
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
fn extract_pr_number(subject: &str) -> Option<u64> {
    let start = subject.rfind("(#")?;
    let tail = &subject[start + 2..];
    let end = tail.find(')')?;
    tail[..end].parse().ok()
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
fn total_e2e_failures(markdown: &str) -> u64 {
    markdown
        .lines()
        .find(|l| l.trim_start().starts_with("| **Total**"))
        .and_then(|line| line.split('|').nth(5))
        .and_then(|cell| cell.trim().parse().ok())
        .unwrap_or(0)
}

fn render_table(facts: &Facts) -> String {
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
        "| E2E | {} failure(s) across all platforms | {} |",
        facts.e2e_fail_total,
        if facts.e2e_fail_total > 0 {
            "❌"
        } else {
            "✅"
        }
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
                issue.number, issue.title, issue.reactions
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
    fn extract_pr_number_reads_trailing_reference() {
        assert_eq!(
            extract_pr_number("fix: correct VRAM probe (#124)"),
            Some(124)
        );
        assert_eq!(extract_pr_number("fix: correct VRAM probe"), None);
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
        assert_eq!(total_e2e_failures(md), 1);
    }

    #[test]
    fn total_e2e_failures_defaults_to_zero_without_a_totals_row() {
        assert_eq!(total_e2e_failures("no table here"), 0);
    }
}
