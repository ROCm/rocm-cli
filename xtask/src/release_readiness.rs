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
    /// Named gates that did not pass (build/windows-build job results, the
    /// E2E failure count or unreadable report, or the `e2e-report-nightly`
    /// lane report check). Empty iff `verdict` is `Go`.
    pub failing_gates: Vec<String>,
    /// `None` when the E2E report's `**Total**` row couldn't be found or
    /// parsed — treated as a failing gate rather than silently reading as
    /// zero failures, since a missing row more often means the report is
    /// broken than that E2E had a perfect run.
    pub e2e_fail_total: Option<u64>,
    /// Whether the nightly or Windows build gate was skipped (a legitimate
    /// no-op-night outcome, not a failure) rather than actually run.
    pub build_skipped: bool,
    pub commits: CommitBuckets,
    pub known_issues: Vec<KnownIssue>,
}

pub fn run(
    since_tag: &str,
    e2e_report: &Path,
    nightly_result: &str,
    windows_result: &str,
    e2e_report_job_result: &str,
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
        e2e_report_job_result,
    );

    // Trailing newline so the file ends like a text file.
    write_parented(out, &format!("{}\n", serde_json::to_string_pretty(&facts)?))?;
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
    e2e_report_job_result: &str,
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
    // The Total row above only counts platforms whose `report.json` actually
    // got uploaded — a GPU lane that dies before uploading (preflight
    // failure, timeout, runner loss) just drops out of it, silently reading
    // as zero failures. Every lane is `continue-on-error`, so the lane job's
    // own `needs.<job>.result` reads `success` even when it failed; gating on
    // that (as an earlier version of this code did) can never fire. Gate on
    // `e2e-report-nightly`'s result instead: its "every lane that ran
    // uploaded its report" check (`nightly.yml`, `LANE_REPORTS`) already
    // fails that job when a lane is silently missing, covering every lane in
    // `LANE_REPORTS` — including ones added later — for free (a re-run can
    // pass on an earlier attempt's artifact, so it can still miss a lane). Unlike the
    // build gates, this job is `if: always()` and has no legitimate
    // no-op-night skip, so `skipped` isn't accepted here.
    if e2e_report_job_result != "success" {
        failing_gates.push(format!("E2E (lane report check: {e2e_report_job_result})"));
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
        build_skipped: nightly_result == "skipped" || windows_result == "skipped",
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
        let rest = subject[colon + 1..].trim();
        let head_no_bang = head.trim_end_matches('!');
        let kind = head_no_bang.split('(').next().unwrap_or("");
        // Conventional Commits allows `!` on any type, not just `feat`/`fix`,
        // to denote a breaking change.
        let breaking = head.ends_with('!');
        let scope = head_no_bang
            .find('(')
            .and_then(|i| head_no_bang[i + 1..].strip_suffix(')'));
        if scope.is_some_and(is_internal_only_scope) {
            continue;
        }
        // `rest` still carries the trailing `(#NNN)` GitHub appends on
        // squash-merge, since it's after the colon too; strip it here so
        // `render_table` isn't left printing `subject (#NNN) (#NNN)`.
        let (pr, rest) = split_trailing_pr_reference(rest);
        let entry = CommitEntry {
            pr,
            subject: rest.to_string(),
        };
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

/// Split off a trailing `(#NNN)` GitHub squash-merge PR reference, if
/// present, returning the PR number and the text with it removed. Only
/// matches when `(#NNN)` is the very end of the text (after trimming
/// trailing whitespace) — `(#NNN)` appearing mid-sentence (e.g. `"fix:
/// handle (#12) refs"`) is not a PR reference.
fn split_trailing_pr_reference(text: &str) -> (Option<u64>, &str) {
    let trimmed = text.trim_end();
    if !trimmed.ends_with(')') {
        return (None, text);
    }
    let Some(start) = trimmed.rfind("(#") else {
        return (None, text);
    };
    let Some(number) = trimmed[start + 2..trimmed.len() - 1].parse().ok() else {
        return (None, text);
    };
    (Some(number), trimmed[..start].trim_end())
}

#[derive(serde::Deserialize)]
struct RawReactionGroup {
    #[serde(default)]
    content: String,
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
    collect_known_issues_in(Path::new("."))
}

/// `collect_known_issues`, with the directory `gh` resolves its repo context
/// from as a seam: lets a test force the "`gh issue list` failed" path (a
/// directory with no GitHub remote) without touching the process environment
/// or `PATH`.
fn collect_known_issues_in(dir: &Path) -> Result<Vec<KnownIssue>> {
    // git exports GIT_DIR/GIT_WORK_TREE to hooks, and gh would resolve the repo
    // from those instead of `dir` (e.g. when run from a pre-push hook).
    let output = Command::new("gh")
        .current_dir(dir)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
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
            "1000",
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
    Ok(build_known_issues(raw))
}

/// Maps `gh issue list` JSON into `KnownIssue`s, ranked by 👍 reaction count
/// (ties broken by issue number). Only counts 👍 as a vote of "this affects
/// me"; other reactions (👎, 😕, etc.) shouldn't boost an issue's rank.
fn build_known_issues(raw: Vec<RawIssue>) -> Vec<KnownIssue> {
    let mut issues: Vec<KnownIssue> = raw
        .into_iter()
        .map(|i| KnownIssue {
            number: i.number,
            title: i.title,
            reactions: i
                .reaction_groups
                .iter()
                .filter(|g| g.content == "THUMBS_UP")
                .map(|g| g.users.total_count)
                .sum(),
            body_excerpt: excerpt(&i.body, 240),
        })
        .collect();
    issues.sort_by(|a, b| b.reactions.cmp(&a.reactions).then(a.number.cmp(&b.number)));
    issues
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
/// Platform | OS | Total | Pass | Fail | Skip | Xfail | Status |`), plus one
/// for every per-platform row that reports a non-`PASS` `Status` with a `Fail`
/// cell of `0`.
///
/// The second part exists because an `EMPTY` platform (no `report.json`, or
/// one that's empty/unparseable) contributes `0` to every numeric column —
/// including `Fail` — so the Total row's own Fail cell has no record that the
/// platform was ever there at all. A genuine `FAIL` status always has a
/// nonzero own `Fail` cell (already folded into the Total row), so this only
/// adds for platforms the Total row would otherwise read as having zero
/// failures.
///
/// Parsing the already-rendered rows, rather than re-deriving per-platform
/// pass/fail here, keeps this in lockstep with that crate's known-bugs/xfail
/// handling instead of duplicating it.
fn total_e2e_failures(markdown: &str) -> Option<u64> {
    // Scope strictly to the summary table's own row range — the document
    // keeps going after it with an expectation grid and a command-coverage
    // table, whose `|`-rows have a different column layout and used to get
    // misread as platform rows here (a `n/a`/`❌` cell landing on the Status
    // column's index by coincidence looked like a non-PASS platform).
    let lines: Vec<&str> = markdown.lines().collect();
    let header_idx = lines
        .iter()
        .position(|l| l.trim_start().starts_with("| Platform"))?;
    let total_idx = header_idx
        + 1
        + lines[header_idx + 1..]
            .iter()
            .position(|l| l.trim_start().starts_with("| **Total**"))?;

    let total_row_fails = lines[total_idx]
        .split('|')
        .nth(5)
        .and_then(|cell| cell.trim().parse::<u64>().ok())?;

    let silently_empty_platforms = lines[header_idx + 1..total_idx]
        .iter()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with('|') && !t.starts_with("|---")
        })
        .filter(|l| {
            let cells: Vec<&str> = l.split('|').collect();
            let status = cells.get(8).map(|c| c.trim());
            let fail = cells
                .get(5)
                .and_then(|c| c.trim().parse::<u64>().ok())
                .unwrap_or(0);
            !matches!(status, Some("PASS") | None) && fail == 0
        })
        .count() as u64;

    Some(total_row_fails + silently_empty_platforms)
}

/// Neutralize markdown syntax in untrusted text (GitHub issue titles) before
/// it's embedded in the generated wiki page. Newlines are collapsed to a
/// space since each issue renders as a single bullet line.
fn escape_markdown(text: &str) -> String {
    text.chars()
        .flat_map(|c| match c {
            '\n' | '\r' => vec![' '],
            '\\' => vec!['\\', '\\'],
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

    // Unreadable E2E evidence is itself a NO-GO signal (see
    // `assemble_facts`), not grounds to skip the table — a NO-GO page must
    // still publish so it's visible on the wiki. The table always carries
    // the header below so `publish-release-readiness-wiki`'s guard (which
    // greps for it) only trips on a truly empty/degenerate table.
    let e2e_failing = facts.failing_gates.iter().any(|g| g.starts_with("E2E"));
    let e2e_evidence = match facts.e2e_fail_total {
        Some(n) => format!("{n} failure(s) across all platforms"),
        None => "evidence unreadable".to_string(),
    };

    out.push_str("| Area | Evidence | Status |\n|---|---|:--:|\n");
    let _ = writeln!(
        out,
        "| Build (Linux + Windows nightly) | nightly / windows-nightly jobs | {} |",
        if facts.failing_gates.iter().any(|g| g.contains("build")) {
            "❌"
        } else if facts.build_skipped {
            "not run"
        } else {
            "✅"
        }
    );
    let _ = writeln!(
        out,
        "| E2E | {e2e_evidence} | {} |",
        if e2e_failing { "❌" } else { "✅" }
    );
    out.push('\n');

    // The status table above only has room for a glyph per row, which is
    // ambiguous on its own (e.g. a cancelled E2E lane renders as
    // "0 failure(s)" with a red ❌, which looks contradictory). Spell out
    // the actual gate strings `assemble_facts` computed so the reason is
    // never left implicit.
    if !facts.failing_gates.is_empty() {
        out.push_str("Failing gates:\n");
        for gate in &facts.failing_gates {
            let _ = writeln!(out, "- {}", escape_markdown(gate));
        }
        out.push('\n');
    }

    let section = |out: &mut String, title: &str, entries: &[CommitEntry]| {
        let _ = writeln!(out, "### {title} ({})\n", entries.len());
        if entries.is_empty() {
            out.push_str("_None._\n\n");
            return;
        }
        for e in entries {
            let subject = escape_markdown(&e.subject);
            match e.pr {
                Some(pr) => {
                    let _ = writeln!(out, "- {subject} (#{pr})");
                }
                None => {
                    let _ = writeln!(out, "- {subject}");
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
    fn split_trailing_pr_reference_reads_trailing_reference() {
        assert_eq!(
            split_trailing_pr_reference("fix: correct VRAM probe (#124)"),
            (Some(124), "fix: correct VRAM probe")
        );
        assert_eq!(
            split_trailing_pr_reference("fix: correct VRAM probe"),
            (None, "fix: correct VRAM probe")
        );
    }

    #[test]
    fn split_trailing_pr_reference_ignores_a_parenthetical_mid_sentence() {
        assert_eq!(
            split_trailing_pr_reference("fix: handle (#12) refs"),
            (None, "fix: handle (#12) refs")
        );
    }

    #[test]
    fn split_trailing_pr_reference_does_not_panic_on_multibyte_text_trailing_a_mid_sentence_reference()
     {
        // Without the `ends_with(')')` guard, `rfind("(#")` would still find
        // the mid-sentence `(#12)` here, and slicing up to `trimmed.len() - 1`
        // would land mid-character (the trailing text is multibyte) and
        // panic. The guard must reject this input before that slice runs.
        assert_eq!(
            split_trailing_pr_reference("fix: handle (#12) 文字"),
            (None, "fix: handle (#12) 文字")
        );
    }

    #[test]
    fn split_trailing_pr_reference_handles_multibyte_text_before_the_reference() {
        // `(#NNN)` is matched by byte offset via `rfind`/`len() - 1`; both
        // land on ASCII bytes here, so this must not panic on a non-char-
        // boundary slice even when the text before it is multibyte.
        assert_eq!(
            split_trailing_pr_reference("修复 VRAM 探测 (#124)"),
            (Some(124), "修复 VRAM 探测")
        );
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
    fn excerpt_does_not_truncate_at_exactly_the_character_limit() {
        let exactly_five = "abcde";
        assert_eq!(excerpt(exactly_five, 5), "abcde");
        assert_eq!(excerpt("abcdef", 5), "abcde…");
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
    fn total_e2e_failures_returns_none_for_an_unparseable_fail_cell() {
        let md = "| Platform | OS | Total | Pass | Fail | Skip | Xfail | Status |\n\
                   |---|---|--:|--:|--:|--:|--:|:--|\n\
                   | **Total** | | 10 | 9 | n/a | 0 | 0 | |\n";
        assert_eq!(total_e2e_failures(md), None);
    }

    #[test]
    fn total_e2e_failures_returns_none_without_a_totals_row() {
        assert_eq!(total_e2e_failures("no table here"), None);
    }

    #[test]
    fn total_e2e_failures_returns_none_instead_of_panicking_when_total_precedes_header() {
        // A `**Total**` line before the `| Platform` header (e.g. a document
        // that embeds the summary table out of order) must fail closed via
        // `None`, not panic on a reversed slice range.
        let md = "| **Total** | | 0 | 0 | 0 | 0 | 0 | |\n\
                   | Platform | OS | Total | Pass | Fail | Skip | Xfail | Status |\n\
                   |---|---|--:|--:|--:|--:|--:|:--|\n";
        assert_eq!(total_e2e_failures(md), None);
    }

    #[test]
    fn total_e2e_failures_catches_a_silently_empty_platform() {
        // A missing/unparseable report.json renders every numeric column as
        // 0, so the Total row's own Fail cell has no idea the platform was
        // ever there — only the per-row Status cell reveals it.
        let md = "| Platform | OS | Total | Pass | Fail | Skip | Xfail | Status |\n\
                   |---|---|--:|--:|--:|--:|--:|:--|\n\
                   | Windows | Windows | 0 | 0 | 0 | 0 | n/a | EMPTY |\n\
                   | **Total** | | 0 | 0 | 0 | 0 | 0 | |\n";
        assert_eq!(total_e2e_failures(md), Some(1));
    }

    #[test]
    fn total_e2e_failures_ignores_the_expectation_grid_and_command_coverage_tables() {
        // Regression: the scan used to run over every `|`-prefixed line in the
        // whole document, not just the summary table. Real consolidated
        // markdown keeps going after the summary table with an expectation
        // grid and a command-coverage table, whose rows have a different
        // column layout — a `n/a`/`❌` cell landing on index 8 by coincidence
        // used to get counted as a phantom platform failure. Build real
        // multi-platform producer output (enough platform columns for the
        // command-coverage table to reach column 8) and confirm the parser
        // reads only the summary table's own Total row.
        fn feature_json(scenario: &str) -> String {
            format!(
                r#"[{{"name":"F","uri":"f.feature","elements":[{{"name":"{scenario}","tags":[],"steps":[{{"keyword":"Given ","name":"x","result":{{"status":"passed"}}}}]}}]}}]"#
            )
        }

        let dirs: Vec<_> = (0..6)
            .map(|i| {
                let dir = tempfile::tempdir().expect("tempdir");
                let report = dir.path().join("report.json");
                std::fs::write(&report, feature_json(&format!("s{i}"))).expect("write report");
                (dir, report)
            })
            .collect();
        // A commands.jsonl sidecar on one platform only: command-coverage
        // still lists every platform as a column (populated or `n/a`), so
        // this alone is enough to grow a downstream row past column 8.
        std::fs::write(
            dirs[0].1.with_file_name("commands.jsonl"),
            r#"{"scenario":"s0","subcommand":"serve","command":"rocm serve x","engine":"vllm","engine_is_default":false}"#,
        )
        .expect("write commands.jsonl");

        let inputs = vec![
            ("e2e-report".to_string(), dirs[0].1.clone()),
            ("e2e-gpu-report".to_string(), dirs[1].1.clone()),
            ("e2e-gpu-rad3-report".to_string(), dirs[2].1.clone()),
            ("e2e-gpu-mi350p-report".to_string(), dirs[3].1.clone()),
            ("e2e-gpu-w7900-report".to_string(), dirs[4].1.clone()),
            ("e2e-gpu-strix-ubuntu-report".to_string(), dirs[5].1.clone()),
        ];
        let md =
            e2e_report::consolidated_summary_markdown(&e2e_report::load_platform_reports(&inputs));
        assert!(
            md.contains("### Command coverage"),
            "expected a command-coverage table in:\n{md}"
        );
        assert_eq!(
            total_e2e_failures(&md),
            Some(0),
            "real producer output with 0 actual failures must read as 0, not inflated by \
             downstream tables:\n{md}"
        );
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
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["E2E (report unreadable)"]);
    }

    #[test]
    fn assemble_facts_treats_a_skipped_nightly_build_as_passing() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "skipped",
            "success",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::Go);
        assert!(facts.failing_gates.is_empty());
        assert!(
            facts.build_skipped,
            "a skipped nightly build must set build_skipped"
        );
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
            "success",
        );
        assert_eq!(facts.verdict, Verdict::Go);
        assert!(facts.failing_gates.is_empty());
        assert!(
            facts.build_skipped,
            "a skipped windows build must set build_skipped"
        );
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
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["windows build (failure)"]);
    }

    #[test]
    fn assemble_facts_is_no_go_when_e2e_failures_are_nonzero() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(3),
            "success",
            "success",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["E2E (3 failure(s))"]);
    }

    #[test]
    fn assemble_facts_is_no_go_at_exactly_one_e2e_failure() {
        // Pins the gate boundary at `n > 0`: a nonzero-failures test using a
        // larger count (e.g. 3) would still pass under a mutated `n > 1`.
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(1),
            "success",
            "success",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["E2E (1 failure(s))"]);
    }

    #[test]
    fn assemble_facts_is_no_go_when_nightly_build_fails() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "failure",
            "success",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["nightly build (failure)"]);
    }

    #[test]
    fn assemble_facts_is_no_go_when_the_e2e_report_job_fails() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "success",
            "failure",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(
            facts.failing_gates,
            vec!["E2E (lane report check: failure)"]
        );
    }

    #[test]
    fn assemble_facts_treats_a_skipped_e2e_report_job_as_a_failing_gate() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "success",
            "skipped",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(
            facts.failing_gates,
            vec!["E2E (lane report check: skipped)"]
        );
    }

    #[test]
    fn render_table_renders_the_verdict_headline_for_go_and_no_go() {
        let go = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "success",
            "success",
        );
        assert!(render_table(&go).contains("## Release Readiness — ✅ GO\n"));

        let no_go = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "failure",
            "success",
            "success",
        );
        assert!(render_table(&no_go).contains("## Release Readiness — ❌ NO-GO\n"));
    }

    #[test]
    fn render_table_renders_pr_number_suffix_only_when_present() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets {
                features: vec![
                    CommitEntry {
                        pr: Some(42),
                        subject: "add widget".to_string(),
                    },
                    CommitEntry {
                        pr: None,
                        subject: "add gadget".to_string(),
                    },
                ],
                ..CommitBuckets::default()
            },
            vec![],
            Some(0),
            "success",
            "success",
            "success",
        );
        let table = render_table(&facts);
        assert!(table.contains("- add widget (#42)\n"));
        assert!(table.contains("- add gadget\n"));
        assert!(!table.contains("- add gadget (#"));
    }

    #[test]
    fn render_table_always_renders_the_header_even_when_e2e_evidence_is_unreadable() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            None,
            "success",
            "success",
            "success",
        );
        let table = render_table(&facts);
        assert!(table.contains("| Area | Evidence | Status |"));
        assert!(table.contains("| E2E | evidence unreadable | ❌ |"));
    }

    #[test]
    fn render_table_marks_the_build_row_failed_on_a_build_gate() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "failure",
            "success",
            "success",
        );
        let table = render_table(&facts);
        assert!(
            table.contains(
                "| Build (Linux + Windows nightly) | nightly / windows-nightly jobs | ❌ |"
            )
        );
    }

    #[test]
    fn render_table_marks_the_build_row_not_run_on_a_skipped_windows_build() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "skipped",
            "success",
        );
        let table = render_table(&facts);
        assert!(table.contains(
            "| Build (Linux + Windows nightly) | nightly / windows-nightly jobs | not run |"
        ));
    }

    #[test]
    fn render_table_lists_the_failing_gate_reasons_when_a_lane_is_cancelled() {
        // A cancelled E2E lane renders "0 failure(s)" with a red ❌ in the
        // status table, which looks contradictory on its own; the gate
        // string must spell out *why* it's failing.
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "success",
            "cancelled",
        );
        let table = render_table(&facts);
        assert!(table.contains("| E2E | 0 failure(s) across all platforms | ❌ |"));
        assert!(table.contains("Failing gates:"));
        assert!(table.contains("- E2E (lane report check: cancelled)"));
    }

    #[test]
    fn render_table_omits_the_failing_gates_section_on_a_go_verdict() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "success",
            "success",
        );
        let table = render_table(&facts);
        assert!(!table.contains("Failing gates:"));
    }

    #[test]
    fn render_table_marks_the_e2e_row_failed_on_nonzero_failures() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(2),
            "success",
            "success",
            "success",
        );
        let table = render_table(&facts);
        assert!(table.contains("| E2E | 2 failure(s) across all platforms | ❌ |"));
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
            "success",
        );
        let table = render_table(&facts);
        assert!(table.contains("| Area | Evidence | Status |"));
        assert!(table.contains(r"!\[x\](evil) \| injected row"));
        assert!(!table.contains("![x](evil) | injected row"));
    }

    #[test]
    fn render_table_escapes_markdown_in_commit_subjects() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets {
                features: vec![CommitEntry {
                    pr: Some(1),
                    subject: "add [evil](link) | row".to_string(),
                }],
                ..CommitBuckets::default()
            },
            vec![],
            Some(0),
            "success",
            "success",
            "success",
        );
        let table = render_table(&facts);
        assert!(table.contains(r"add \[evil\](link) \| row"));
        assert!(!table.contains("add [evil](link) | row"));
    }

    #[test]
    fn escape_markdown_doubles_a_literal_backslash() {
        assert_eq!(escape_markdown(r"a\b"), r"a\\b");
    }

    #[test]
    fn escape_markdown_collapses_newlines_and_escapes_angle_brackets() {
        assert_eq!(escape_markdown("line one\nline two"), "line one line two");
        assert_eq!(escape_markdown("<script>"), r"\<script\>");
    }

    #[test]
    fn escape_markdown_escapes_emphasis_and_code_markers() {
        assert_eq!(
            escape_markdown("*bold* _em_ `code`"),
            r"\*bold\* \_em\_ \`code\`"
        );
    }

    #[test]
    fn escape_markdown_collapses_a_bare_carriage_return() {
        assert_eq!(escape_markdown("line one\rline two"), "line one line two");
    }

    #[test]
    fn categorize_treats_a_bang_on_any_type_as_breaking() {
        let subjects = vec!["docs!: rewrite the README".to_string()];
        let got = categorize(&subjects);
        assert_eq!(
            got.breaking,
            vec![CommitEntry {
                pr: None,
                subject: "rewrite the README".to_string(),
            }]
        );
        assert!(got.features.is_empty());
        assert!(got.fixes.is_empty());
    }

    #[test]
    fn categorize_does_not_double_append_the_pr_suffix() {
        let subjects = vec!["fix: correct VRAM probe (#124)".to_string()];
        let got = categorize(&subjects);
        assert_eq!(got.fixes[0].subject, "correct VRAM probe");
        assert!(!got.fixes[0].subject.contains("(#124)"));
    }

    #[test]
    fn categorize_drops_a_breaking_commit_scoped_entirely_to_internal_tooling() {
        // The scope must be read off `head_no_bang`, not `head`: the raw
        // head still has the trailing `!`, so `head[i+1..]` after `(` ends in
        // `ci)!` rather than `ci)` and `strip_suffix(')')` would miss,
        // leaking an internal-only breaking change into the Breaking bucket.
        let subjects = vec!["fix(nightly)!: rewrite the release-readiness gate".to_string()];
        let got = categorize(&subjects);
        assert!(got.breaking.is_empty());
        assert!(got.fixes.is_empty());
    }

    #[test]
    fn categorize_trims_whitespace_around_scope_names() {
        let subjects = vec!["fix(nightly, ci): tidy up the dispatch matrix".to_string()];
        let got = categorize(&subjects);
        assert!(got.fixes.is_empty());
    }

    #[test]
    fn collect_known_issues_counts_only_thumbs_up_reactions() {
        let raw: Vec<RawIssue> = serde_json::from_str(
            r#"[
                {"number": 1, "title": "low vote", "body": "", "reactionGroups":
                    [{"content":"THUMBS_UP","users":{"totalCount":3}},{"content":"THUMBS_DOWN","users":{"totalCount":5}}]},
                {"number": 2, "title": "high vote", "body": "", "reactionGroups":
                    [{"content":"THUMBS_UP","users":{"totalCount":4}}]}
            ]"#,
        )
        .unwrap();

        let issues = build_known_issues(raw);

        assert_eq!(issues[0].number, 2, "higher 👍 count must rank first");
        assert_eq!(issues[0].reactions, 4);
        assert_eq!(issues[1].number, 1);
        assert_eq!(issues[1].reactions, 3, "👎 must not count toward the rank");
    }

    #[test]
    fn build_known_issues_breaks_a_reaction_tie_by_ascending_issue_number() {
        let raw: Vec<RawIssue> = serde_json::from_str(
            r#"[
                {"number": 9, "title": "later", "body": "", "reactionGroups":
                    [{"content":"THUMBS_UP","users":{"totalCount":2}}]},
                {"number": 3, "title": "earlier", "body": "", "reactionGroups":
                    [{"content":"THUMBS_UP","users":{"totalCount":2}}]}
            ]"#,
        )
        .unwrap();

        let issues = build_known_issues(raw);

        assert_eq!(
            issues.iter().map(|i| i.number).collect::<Vec<_>>(),
            vec![3, 9],
            "equal reaction counts must break ties by ascending issue number"
        );
    }

    #[test]
    fn collect_commit_subjects_fails_on_a_bogus_since_tag() {
        let err = collect_commit_subjects("this-tag-definitely-does-not-exist-zzz").unwrap_err();
        assert!(
            err.to_string().contains("git log"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn collect_known_issues_in_fails_outside_a_github_repo() {
        // Depends on the host: without `gh` the error is the spawn context, and
        // with `GH_REPO` set `gh` skips repo detection and calls the live API.
        if Command::new("gh").arg("--version").output().is_err()
            || std::env::var_os("GH_REPO").is_some()
        {
            eprintln!("skipping: needs `gh` installed and GH_REPO unset");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let err = collect_known_issues_in(dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("`gh issue list` failed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn assemble_facts_is_no_go_when_nightly_build_is_cancelled() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "cancelled",
            "success",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["nightly build (cancelled)"]);
    }

    #[test]
    fn assemble_facts_is_no_go_when_nightly_build_times_out() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "timed_out",
            "success",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["nightly build (timed_out)"]);
    }

    #[test]
    fn assemble_facts_is_no_go_when_windows_build_is_cancelled() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "cancelled",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["windows build (cancelled)"]);
    }

    #[test]
    fn assemble_facts_is_no_go_when_windows_build_times_out() {
        let facts = assemble_facts(
            "v1.0.0",
            CommitBuckets::default(),
            vec![],
            Some(0),
            "success",
            "timed_out",
            "success",
        );
        assert_eq!(facts.verdict, Verdict::NoGo);
        assert_eq!(facts.failing_gates, vec!["windows build (timed_out)"]);
    }
}
