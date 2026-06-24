// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Commit-trust enforcement: verify that every commit in a range is both
//! cryptographically signed and carries a DCO `Signed-off-by` trailer.
//!
//! The decision logic ([`evaluate_commits`]) is pure and operates on already
//! collected per-commit data so it can be unit-tested without git or network
//! access. The git/`gh` I/O lives at the edges ([`collect_commits`],
//! [`verified_via_github`]).

use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};

/// Default base ref used when none is supplied on the command line.
const DEFAULT_BASE: &str = "origin/main";

/// One commit's data, as gathered from git (and optionally GitHub), in the
/// structured form the pure decision logic consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitInfo {
    /// Abbreviated commit hash, for reporting.
    pub short_sha: String,
    /// First line of the commit message.
    pub subject: String,
    /// Signature status as reported by `git log --format=%G?`.
    ///
    /// `N` = no signature, `B` = bad signature; any other value
    /// (`G`/`U`/`E`/`X`/`Y`/`R`) means a signature is present.
    pub signature_status: char,
    /// Whether GitHub reports the signature as `verified` (strict mode only;
    /// `None` when strict mode is not in effect).
    pub github_verified: Option<bool>,
    /// Whether the message carries a `Signed-off-by` trailer.
    pub has_signoff: bool,
}

/// A single failed check for one commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitFailure {
    pub short_sha: String,
    pub subject: String,
    /// Human-readable reason the commit failed.
    pub reason: String,
}

/// Pure decision logic: given the collected commits and whether strict
/// (GitHub "Verified") mode is requested, return the list of failures.
///
/// An empty input slice (empty commit range) yields no failures.
pub fn evaluate_commits(commits: &[CommitInfo], require_verified: bool) -> Vec<CommitFailure> {
    let mut failures = Vec::new();
    for commit in commits {
        if !commit.has_signoff {
            failures.push(CommitFailure {
                short_sha: commit.short_sha.clone(),
                subject: commit.subject.clone(),
                reason: "missing `Signed-off-by` trailer".to_string(),
            });
        }
        if require_verified {
            // Strict mode: GitHub must report the signature as verified.
            if commit.github_verified != Some(true) {
                failures.push(CommitFailure {
                    short_sha: commit.short_sha.clone(),
                    subject: commit.subject.clone(),
                    reason: "signature is not GitHub-\"Verified\"".to_string(),
                });
            }
        } else {
            // Default/local mode: a signature must be present and not bad.
            // `N` (none) and `B` (bad) fail; everything else passes.
            match commit.signature_status {
                'N' => failures.push(CommitFailure {
                    short_sha: commit.short_sha.clone(),
                    subject: commit.subject.clone(),
                    reason: "commit is not signed".to_string(),
                }),
                'B' => failures.push(CommitFailure {
                    short_sha: commit.short_sha.clone(),
                    subject: commit.subject.clone(),
                    reason: "commit has a bad signature".to_string(),
                }),
                _ => {}
            }
        }
    }
    failures
}

/// Whether a commit message body contains a `Signed-off-by` trailer.
///
/// We accept any line that begins (ignoring leading whitespace) with the
/// case-insensitive `signed-off-by:` token. This is deliberately more lenient
/// than git's own trailer-block detection (which additionally requires the
/// trailer to sit in the final blank-line-separated paragraph): for a DCO
/// gate the looser match is sufficient and cannot miss a real trailer.
fn body_has_signoff(message: &str) -> bool {
    message.lines().any(|line| {
        line.trim_start()
            .to_ascii_lowercase()
            .starts_with("signed-off-by:")
    })
}

/// Run `git` with the given args and return trimmed stdout, failing on a
/// non-zero exit.
fn git(args: &[&str]) -> Result<String> {
    let output = Command::new("git")
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

/// Collect the commits in `<base>..HEAD` into structured [`CommitInfo`] records.
///
/// When `require_verified` is set, each commit is additionally queried against
/// the GitHub API for its `verification.verified` status.
fn collect_commits(base: &str, require_verified: bool) -> Result<Vec<CommitInfo>> {
    let range = format!("{base}..HEAD");
    // One line per commit: "<short-sha> <full-sha> <signature-status>". `%h` is
    // the abbreviated hash (for reporting), `%H` the full hash (for the API and
    // for fetching the message/subject separately to avoid delimiter ambiguity),
    // and `%G?` a single character.
    let listing = git(&["log", "--format=%h %H %G?", &range])?;

    let repo_slug = if require_verified {
        Some(github_repo_slug()?)
    } else {
        None
    };

    let mut commits = Vec::new();
    for line in listing.lines().filter(|line| !line.trim().is_empty()) {
        let mut fields = line.splitn(3, ' ');
        let (Some(short_sha), Some(full_sha), Some(status)) =
            (fields.next(), fields.next(), fields.next())
        else {
            return Err(anyhow!("unexpected `git log` line: {line:?}"));
        };
        let short_sha = short_sha.to_string();
        let signature_status = status.trim().chars().next().unwrap_or('N');

        let subject = git(&["log", "-1", "--format=%s", full_sha])?;
        let message = git(&["log", "-1", "--format=%B", full_sha])?;

        let github_verified = match &repo_slug {
            Some(slug) => Some(verified_via_github(slug, full_sha)?),
            None => None,
        };

        commits.push(CommitInfo {
            short_sha,
            subject,
            signature_status,
            github_verified,
            has_signoff: body_has_signoff(&message),
        });
    }
    Ok(commits)
}

/// Derive the `owner/repo` slug from `GITHUB_REPOSITORY` (set on GitHub-hosted
/// runners) or, failing that, from the `origin` remote URL.
fn github_repo_slug() -> Result<String> {
    if let Ok(slug) = std::env::var("GITHUB_REPOSITORY")
        && !slug.trim().is_empty()
    {
        return Ok(slug.trim().to_string());
    }
    let url = git(&["remote", "get-url", "origin"])?;
    parse_repo_slug(&url)
        .ok_or_else(|| anyhow!("could not derive owner/repo from origin URL {url:?}"))
}

/// Parse an `owner/repo` slug from a GitHub remote URL (HTTPS or SSH form).
fn parse_repo_slug(url: &str) -> Option<String> {
    let url = url.trim().trim_end_matches('/');
    let tail = url
        .rsplit_once("github.com/")
        .map(|(_, tail)| tail)
        .or_else(|| url.rsplit_once("github.com:").map(|(_, tail)| tail))?;
    let tail = tail.strip_suffix(".git").unwrap_or(tail);
    if tail.split('/').count() == 2 && !tail.contains(' ') {
        Some(tail.to_string())
    } else {
        None
    }
}

/// Query GitHub for whether a commit's signature is `verified`, via the
/// preinstalled, token-authenticated `gh` CLI.
fn verified_via_github(repo_slug: &str, full_sha: &str) -> Result<bool> {
    let endpoint = format!("/repos/{repo_slug}/commits/{full_sha}");
    let output = Command::new("gh")
        .args(["api", &endpoint, "--jq", ".commit.verification.verified"])
        .output()
        .context("failed to run `gh api` (is the GitHub CLI installed and authenticated?)")?;
    if !output.status.success() {
        bail!(
            "`gh api {endpoint}` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    match stdout.trim() {
        "true" => Ok(true),
        "false" => Ok(false),
        // `null`/empty means the field was absent — typically because the commit
        // is not (yet) visible on GitHub. Surface that distinctly rather than
        // reporting it as an unverified signature.
        other => bail!(
            "`gh api {endpoint}` returned {other:?} for `verification.verified`; \
             the commit may not be pushed to GitHub yet"
        ),
    }
}

/// Remediation guidance printed when commits fail, tailored to the mode.
fn remediation(require_verified: bool) -> String {
    let mut text = String::from(
        "\nTo fix:\n\
         - Sign-off: amend with `git commit -s --amend` (one commit) or\n  \
           `git rebase --signoff <base>` (a range), then re-push.\n\
         - Signing: enable commit signing once, e.g. SSH signing:\n      \
           git config --global gpg.format ssh\n      \
           git config --global user.signingkey ~/.ssh/id_ed25519.pub\n      \
           git config --global commit.gpgsign true\n  \
           (or GPG: set `user.signingkey` to your GPG key id and\n   \
           `git config --global commit.gpgsign true`), then re-sign with\n   \
           `git rebase --exec 'git commit --amend --no-edit -S' <base>`.\n",
    );
    if require_verified {
        text.push_str(
            "- GitHub \"Verified\": register your signing key on GitHub\n  \
             (Settings > SSH and GPG keys) and make sure your committer email is\n  \
             a verified email on your GitHub account.\n",
        );
    }
    text
}

/// Entry point for the `verify-commits` subcommand.
pub fn run(base: Option<String>, require_verified: bool) -> Result<()> {
    let base = base.unwrap_or_else(|| {
        // In CI, prefer the PR base branch when GitHub provides it.
        std::env::var("GITHUB_BASE_REF")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map_or_else(
                || DEFAULT_BASE.to_string(),
                |branch| format!("origin/{branch}"),
            )
    });

    let commits = collect_commits(&base, require_verified)?;
    if commits.is_empty() {
        println!("verify-commits: no commits in {base}..HEAD; nothing to check.");
        return Ok(());
    }

    let failures = evaluate_commits(&commits, require_verified);
    if failures.is_empty() {
        println!(
            "verify-commits: all {} commit(s) in {base}..HEAD are signed and signed-off.",
            commits.len()
        );
        return Ok(());
    }

    eprintln!("verify-commits: {} check(s) failed:\n", failures.len());
    for failure in &failures {
        eprintln!(
            "  {} {}\n    -> {}",
            failure.short_sha, failure.subject, failure.reason
        );
    }
    eprint!("{}", remediation(require_verified));
    bail!("commit signature/sign-off enforcement failed");
}

/// Entry point for the `--check-config` mode: assert that commit signing is
/// configured locally (used by the fast pre-commit hook).
pub fn check_config() -> Result<()> {
    let gpgsign = git(&["config", "--get", "commit.gpgsign"])
        .unwrap_or_default()
        .to_ascii_lowercase();
    let signingkey = git(&["config", "--get", "user.signingkey"]).unwrap_or_default();

    let signing_enabled = gpgsign == "true";
    let key_set = !signingkey.trim().is_empty();
    if signing_enabled && key_set {
        return Ok(());
    }

    eprintln!("verify-commits: commit signing is not configured.");
    if !signing_enabled {
        eprintln!("  -> `commit.gpgsign` is not set to `true`.");
    }
    if !key_set {
        eprintln!("  -> `user.signingkey` is not set.");
    }
    eprint!("{}", remediation(false));
    bail!("commit signing is not configured");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commit(status: char, verified: Option<bool>, has_signoff: bool) -> CommitInfo {
        CommitInfo {
            short_sha: "abc1234".to_string(),
            subject: "example commit".to_string(),
            signature_status: status,
            github_verified: verified,
            has_signoff,
        }
    }

    #[test]
    fn empty_range_passes() {
        assert!(evaluate_commits(&[], false).is_empty());
        assert!(evaluate_commits(&[], true).is_empty());
    }

    #[test]
    fn signed_and_signed_off_passes_local() {
        let commits = [commit('G', None, true)];
        assert!(evaluate_commits(&commits, false).is_empty());
    }

    #[test]
    fn missing_signoff_fails() {
        let commits = [commit('G', None, false)];
        let failures = evaluate_commits(&commits, false);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].reason.contains("Signed-off-by"));
    }

    #[test]
    fn no_signature_fails_local() {
        let commits = [commit('N', None, true)];
        let failures = evaluate_commits(&commits, false);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].reason.contains("not signed"));
    }

    #[test]
    fn bad_signature_fails_local() {
        let commits = [commit('B', None, true)];
        let failures = evaluate_commits(&commits, false);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].reason.contains("bad signature"));
    }

    #[test]
    fn lenient_signature_states_pass_local() {
        // Local mode only rejects `N` (none) and `B` (bad). Every other `%G?`
        // state — including expired (`X`/`Y`), revoked (`R`), and
        // can't-verify (`E`) keys — is intentionally accepted here; strict
        // mode (GitHub "Verified") is the gate that judges trust in CI.
        for status in ['G', 'U', 'E', 'X', 'Y', 'R'] {
            let commits = [commit(status, None, true)];
            assert!(
                evaluate_commits(&commits, false).is_empty(),
                "status {status:?} should pass in local mode"
            );
        }
    }

    #[test]
    fn unverified_fails_strict() {
        let commits = [commit('G', Some(false), true)];
        let failures = evaluate_commits(&commits, true);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].reason.contains("Verified"));
    }

    #[test]
    fn verified_and_signed_off_passes_strict() {
        let commits = [commit('G', Some(true), true)];
        assert!(evaluate_commits(&commits, true).is_empty());
    }

    #[test]
    fn strict_missing_github_verdict_fails() {
        // In strict mode a `None` verdict (the API was never consulted) must
        // not slip through as verified.
        let commits = [commit('G', None, true)];
        let failures = evaluate_commits(&commits, true);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].reason.contains("Verified"));
    }

    #[test]
    fn strict_ignores_local_signature_status() {
        // In strict mode the `%G?` char is irrelevant; only GitHub's verdict
        // and the sign-off matter.
        let commits = [commit('N', Some(true), true)];
        assert!(evaluate_commits(&commits, true).is_empty());
    }

    #[test]
    fn multiple_failures_reported_per_commit() {
        // Unsigned AND missing sign-off => two failures for one commit.
        let commits = [commit('N', None, false)];
        let failures = evaluate_commits(&commits, false);
        assert_eq!(failures.len(), 2);
    }

    #[test]
    fn signoff_detected_in_trailer() {
        assert!(body_has_signoff(
            "Subject\n\nBody text.\n\nSigned-off-by: Dev <dev@example.com>\n"
        ));
    }

    #[test]
    fn signoff_detection_is_case_insensitive() {
        assert!(body_has_signoff(
            "Subject\n\nsigned-off-by: Dev <dev@example.com>"
        ));
    }

    #[test]
    fn signoff_detected_with_leading_whitespace() {
        assert!(body_has_signoff(
            "Subject\n\n  Signed-off-by: Dev <dev@example.com>"
        ));
    }

    #[test]
    fn signoff_absent_is_detected() {
        assert!(!body_has_signoff("Subject\n\nNo trailer here."));
    }

    #[test]
    fn repo_slug_parsed_from_https_and_ssh() {
        assert_eq!(
            parse_repo_slug("https://github.com/ROCm/rocm-cli.git").as_deref(),
            Some("ROCm/rocm-cli")
        );
        assert_eq!(
            parse_repo_slug("git@github.com:ROCm/rocm-cli.git").as_deref(),
            Some("ROCm/rocm-cli")
        );
        assert_eq!(
            parse_repo_slug("https://github.com/ROCm/rocm-cli").as_deref(),
            Some("ROCm/rocm-cli")
        );
        assert_eq!(
            parse_repo_slug("ssh://git@github.com/ROCm/rocm-cli.git").as_deref(),
            Some("ROCm/rocm-cli")
        );
        assert_eq!(parse_repo_slug("https://example.com/not/github"), None);
    }
}
