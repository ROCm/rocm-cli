// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static contract tests for the CI workflow split.
//!
//! Regression guard for EAI-7548: the self-hosted GPU E2E lanes must live in a
//! SEPARATE workflow from `ci.yml`, with a DISTINCT concurrency group, so a job
//! queued on an offline self-hosted runner (which GitHub cannot cancel) can never
//! hold `ci.yml`'s concurrency group and stall its merge-required checks.
//!
//! There is no YAML dependency in this crate, so instead of a whole-file
//! substring match (which can false-pass — a label hidden across a multiline
//! `runs-on`, or `github.workflow` found only in a comment) these helpers
//! extract the COMPLETE value of each `runs-on` and of the top-level
//! `concurrency.group`, then assert on those extracted values.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    fn repo_root() -> PathBuf {
        // CARGO_MANIFEST_DIR is the xtask/ crate dir; its parent is the repo root
        // (same idiom as verify_pinned_keys::repo_root).
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    fn read_workflow(name: &str) -> String {
        let p = repo_root().join(".github/workflows").join(name);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
    }

    /// The self-hosted runner labels that must not appear in a `runs-on`.
    const SELF_HOSTED_LABELS: [&str; 3] = ["self-hosted", "amd-gpu", "strix-halo"];

    /// Strip a trailing `# …` comment from a YAML line (best-effort: our
    /// workflows never put a literal `#` inside a runs-on/group value).
    fn strip_comment(line: &str) -> &str {
        line.split_once(" #").map_or(line, |(v, _)| v)
    }

    fn indent_of(line: &str) -> usize {
        line.len() - line.trim_start().len()
    }

    /// Extract the COMPLETE value of every `runs-on:` in the workflow, joining any
    /// block/flow continuation lines so a label split across lines can't hide.
    /// Returns one flattened string per `runs-on` key.
    fn runs_on_values(text: &str) -> Vec<String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut out = Vec::new();
        for (i, raw) in lines.iter().enumerate() {
            let line = strip_comment(raw);
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix("runs-on:") else {
                continue;
            };
            let key_indent = indent_of(line);
            let mut value = rest.trim().to_owned();
            // Gather deeper-indented continuation lines (block list `- x`, or a
            // flow list `[…]` wrapped across lines).
            for cont in &lines[i + 1..] {
                let c = strip_comment(cont);
                if c.trim().is_empty() {
                    continue;
                }
                if indent_of(c) <= key_indent {
                    break;
                }
                value.push(' ');
                value.push_str(c.trim());
            }
            out.push(value);
        }
        out
    }

    /// Extract the top-level `concurrency.group` value, joining folded (`>-`)
    /// continuation lines. Returns the whole group expression as one string.
    fn concurrency_group(text: &str) -> String {
        let lines: Vec<&str> = text.lines().collect();
        // Find a top-level (column-0) `concurrency:` key.
        let start = lines
            .iter()
            .position(|l| *l == "concurrency:")
            .expect("workflow declares a top-level concurrency:");
        // Within that block, find the `group:` key.
        let mut group_val = String::new();
        let mut in_group = false;
        let mut group_indent = 0;
        for line in &lines[start + 1..] {
            // A new column-0 key ends the concurrency block.
            if !line.is_empty() && indent_of(line) == 0 {
                break;
            }
            let stripped = strip_comment(line);
            let trimmed = stripped.trim_start();
            if !in_group {
                if let Some(rest) = trimmed.strip_prefix("group:") {
                    in_group = true;
                    group_indent = indent_of(stripped);
                    group_val = rest.trim().to_owned();
                }
                continue;
            }
            // Collecting folded continuation lines under group:.
            if trimmed.is_empty() {
                continue;
            }
            if indent_of(stripped) <= group_indent {
                break;
            }
            group_val.push(' ');
            group_val.push_str(trimmed);
        }
        assert!(in_group, "concurrency block has no group: key");
        group_val
    }

    fn workflow_name(text: &str) -> String {
        text.lines()
            .find_map(|l| l.strip_prefix("name:"))
            .map(|n| strip_comment(n).trim().to_owned())
            .expect("workflow declares a top-level name:")
    }

    /// Extract a top-level YAML value/block, stopping at the next column-zero
    /// key or comment. This is deliberately small: workflow contract tests
    /// inspect only checked-in files with the repository's established layout.
    fn top_level_block(text: &str, key: &str) -> String {
        let marker = format!("{key}:");
        let lines: Vec<&str> = text.lines().collect();
        let start = lines
            .iter()
            .position(|line| line.starts_with(&marker))
            .unwrap_or_else(|| panic!("workflow declares a top-level {marker}"));
        let end = lines[start + 1..]
            .iter()
            .position(|line| !line.is_empty() && !line.starts_with(char::is_whitespace))
            .map_or(lines.len(), |offset| start + 1 + offset);
        let mut block = lines[start]
            .strip_prefix(&marker)
            .expect("top-level key prefix was just matched")
            .to_owned();
        for line in &lines[start + 1..end] {
            block.push('\n');
            block.push_str(line);
        }
        block
    }

    /// Extract an exact nested YAML mapping block, including its key line and
    /// all more-deeply-indented children.
    fn nested_block(text: &str, marker: &str) -> String {
        let lines: Vec<&str> = text.lines().collect();
        let start = lines
            .iter()
            .position(|line| *line == marker)
            .unwrap_or_else(|| panic!("workflow declares nested block `{marker}`"));
        let marker_indent = indent_of(marker);
        let end = lines[start + 1..]
            .iter()
            .position(|line| !line.trim().is_empty() && indent_of(line) <= marker_indent)
            .map_or(lines.len(), |offset| start + 1 + offset);
        lines[start..end].join("\n")
    }

    /// Parse the scalar entries under an exact nested `permissions:` block.
    /// Returning every entry makes equality assertions fail if any capability
    /// is added, even when the new permission is not `contents: write`.
    fn permission_mapping(text: &str, marker: &str) -> std::collections::BTreeMap<String, String> {
        let block = nested_block(text, marker);
        let mut permissions = std::collections::BTreeMap::new();
        for raw in block.lines().skip(1) {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            let (name, access) = line
                .split_once(':')
                .unwrap_or_else(|| panic!("permission entry is not `name: access`: {line}"));
            let name = name.trim();
            let access = access.trim();
            assert!(
                !name.is_empty() && !access.is_empty(),
                "permission entry must have a scalar name and access: {line}"
            );
            assert!(
                permissions
                    .insert(name.to_owned(), access.to_owned())
                    .is_none(),
                "duplicate permission entry: {name}"
            );
        }
        permissions
    }

    #[test]
    fn permission_mapping_extractor_preserves_every_declared_permission() {
        let job = "\
  example:
    permissions:
      # This comment is not a capability.
      contents: read
      pull-requests: write
    steps: []
";
        assert_eq!(
            permission_mapping(job, "    permissions:"),
            std::collections::BTreeMap::from([
                ("contents".to_owned(), "read".to_owned()),
                ("pull-requests".to_owned(), "write".to_owned()),
            ])
        );
    }

    #[test]
    fn dependabot_pull_request_workflow_is_strictly_read_only() {
        let generator = read_workflow("dependabot-manifests.yml");
        let trigger = top_level_block(&generator, "on");
        let permissions = top_level_block(&generator, "permissions");
        let jobs = top_level_block(&generator, "jobs");

        assert!(trigger.contains("pull_request:"));
        assert_eq!(permissions.trim(), "{}");
        let generate_job = nested_block(&jobs, "  generate:");
        assert_eq!(
            permission_mapping(&generate_job, "    permissions:"),
            std::collections::BTreeMap::from([("contents".to_owned(), "read".to_owned())]),
            "the untrusted generator job must have exactly contents: read"
        );
        assert!(jobs.contains("actions/upload-artifact@"));
        assert!(jobs.contains("MANIFEST.md"));
        assert!(jobs.contains("THIRD_PARTY_NOTICES.txt"));
        assert!(
            !jobs.contains("contents: write")
                && !jobs.contains("createCommitOnBranch")
                && !jobs.contains("Commit regenerated manifests"),
            "a Dependabot pull_request run receives a read-only GITHUB_TOKEN; it must only \
             generate and upload the bounded manifest artifact, never contain a write job"
        );
    }

    #[test]
    fn dependabot_manifest_commit_is_a_guarded_workflow_run_follow_up() {
        let follow_up = read_workflow("dependabot-manifests-commit.yml");
        let trigger = top_level_block(&follow_up, "on");
        let permissions = top_level_block(&follow_up, "permissions");
        let jobs = top_level_block(&follow_up, "jobs");

        assert!(trigger.contains("workflow_run:"));
        assert!(trigger.contains("Dependabot manifests"));
        assert!(trigger.contains("completed"));
        assert_eq!(permissions.trim(), "{}");
        let commit_job = nested_block(&jobs, "  commit:");
        assert_eq!(
            permission_mapping(&commit_job, "    permissions:"),
            std::collections::BTreeMap::from([
                ("actions".to_owned(), "read".to_owned()),
                ("contents".to_owned(), "write".to_owned()),
                ("pull-requests".to_owned(), "read".to_owned()),
            ]),
            "the privileged follow-up job must have exactly its minimal API permissions"
        );
        let commit_job_if = nested_block(&commit_job, "    if: >-");
        for required_job_gate in [
            "github.event.workflow_run.conclusion == 'success'",
            "github.event.workflow_run.actor.login == 'dependabot[bot]'",
            "github.event.workflow_run.event == 'pull_request'",
        ] {
            assert!(
                commit_job_if.contains(required_job_gate),
                "privileged commit.if is missing semantic clause `{required_job_gate}`"
            );
        }
        assert!(
            !commit_job_if.contains("||"),
            "privileged commit.if must not weaken its required gates with an OR clause"
        );

        for required_guard in [
            "dependabot[bot]",
            "pull_request",
            ".pull_requests | length",
            "commits/$run_sha/pulls",
            "select(.state == \"open\")",
            ".head.repo.full_name",
            "dependabot/*",
            ".head.sha",
            "!= \"$run_sha\"",
        ] {
            assert!(
                jobs.contains(required_guard),
                "follow-up workflow is missing fail-closed guard `{required_guard}`"
            );
        }

        assert!(
            jobs.contains("actions/runs/$run_id/artifacts?name=regenerated-manifests&per_page=100")
        );
        assert!(jobs.contains("select(.name == \"regenerated-manifests\""));
        assert!(jobs.contains("artifact_id=$(jq -r '.[0].id'"));
        assert!(jobs.contains("actions/artifacts/$ARTIFACT_ID/zip"));
        assert!(!jobs.contains("actions/download-artifact@"));
        assert!(jobs.contains("artifact_count\" -gt 1"));
        assert!(jobs.contains("MAX_ARTIFACT_BYTES=$((25 * 1024 * 1024))"));
        assert!(jobs.contains("artifact_size=$(jq -r '.[0].size_in_bytes // empty'"));
        assert!(jobs.contains("artifact_size > MAX_ARTIFACT_BYTES"));
        let size_guard = jobs
            .find("artifact_size=$(jq -r '.[0].size_in_bytes // empty'")
            .expect("workflow validates artifact size metadata");
        let artifact_id_export = jobs
            .find("artifact_id=$(jq -r '.[0].id'")
            .expect("workflow exports the validated artifact ID");
        let artifact_download = jobs
            .find("actions/artifacts/$ARTIFACT_ID/zip")
            .expect("workflow downloads the validated artifact");
        assert!(
            size_guard < artifact_id_export && artifact_id_export < artifact_download,
            "artifact size must be validated before its ID is exported or downloaded"
        );
        assert!(jobs.contains("artifact must contain exactly the two generated files"));
        assert!(jobs.contains("from stat import S_ISREG"));
        assert!(jobs.contains("mode != 0 and not S_ISREG(mode)"));
        assert!(jobs.contains("entry.file_size > 10 * 1024 * 1024"));
        assert!(jobs.contains("(destination / entry.filename).write_bytes(archive.read(entry))"));
        assert!(!jobs.contains("archive.extract("));
        assert!(!jobs.contains("archive.extractall("));
        assert_eq!(
            jobs.matches("regenerated/").count(),
            2,
            "artifact files may only be read by the two fixed-path base64 commands"
        );
        assert!(jobs.contains("base64 -w0 regenerated/MANIFEST.md > manifest.b64"));
        assert!(jobs.contains("base64 -w0 regenerated/THIRD_PARTY_NOTICES.txt > tpn.b64"));
        for forbidden_execution in [
            "bash regenerated/",
            "sh regenerated/",
            "python3 regenerated/",
            "source regenerated/",
            "chmod +x regenerated/",
            "./regenerated/",
            "subprocess",
            "os.system",
        ] {
            assert!(
                !jobs.contains(forbidden_execution),
                "the write-scoped follow-up must not execute artifact content via \
                 `{forbidden_execution}`"
            );
        }
        assert!(jobs.contains("expectedHeadOid"));
        assert!(jobs.contains("EXPECTED_HEAD"));
        assert!(jobs.contains("Signed-off-by: github-actions[bot]"));
        assert!(jobs.contains("{ path: \"MANIFEST.md\", contents: $manifest }"));
        assert!(jobs.contains("{ path: \"THIRD_PARTY_NOTICES.txt\", contents: $tpn }"));
        assert_eq!(
            jobs.matches("{ path:").count(),
            2,
            "createCommitOnBranch must add exactly the two generated paths"
        );
        assert!(!jobs.contains("deletions:"));
        assert!(
            !jobs.contains("actions/checkout@"),
            "the write-scoped workflow_run follow-up must never check out or execute PR code"
        );
        assert!(
            !follow_up.contains("pull_request_target"),
            "the privileged follow-up must use workflow_run, never pull_request_target"
        );

        let generator = read_workflow("dependabot-manifests.yml");
        assert!(generator.contains(
            "https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/\
trigger-a-workflow#triggering-a-workflow-from-a-workflow"
        ));
    }

    #[test]
    fn ci_yml_schedules_no_self_hosted_job() {
        let ci = read_workflow("ci.yml");
        let values = runs_on_values(&ci);
        assert!(
            !values.is_empty(),
            "expected at least one runs-on in ci.yml (extractor sanity check)"
        );
        for value in values {
            for label in SELF_HOSTED_LABELS {
                assert!(
                    !value.contains(label),
                    "ci.yml runs-on `{value}` references the self-hosted label {label:?}: \
                     the GPU E2E lanes belong in e2e-selfhosted.yml so a job queued on an \
                     offline runner can't stall ci.yml's required checks (EAI-7548)"
                );
            }
        }
    }

    #[test]
    fn self_hosted_workflow_owns_the_gpu_lanes() {
        let sh = read_workflow("e2e-selfhosted.yml");
        // Each GPU lane must be present AND actually target a self-hosted runner.
        let values = runs_on_values(&sh);
        assert!(
            values.iter().any(|v| v.contains("self-hosted")),
            "e2e-selfhosted.yml must schedule at least one `self-hosted` runner (EAI-7548)"
        );
        for job in [
            "e2e-gpu:",
            "e2e-gpu-strix-ubuntu:",
            "e2e-gpu-strix-windows:",
        ] {
            assert!(
                sh.contains(job),
                "e2e-selfhosted.yml must define the self-hosted job `{job}` (EAI-7548)"
            );
        }
    }

    #[test]
    fn workflows_use_distinct_concurrency_groups() {
        // Isolation comes from the group KEY differing per workflow. Extract the
        // actual concurrency.group value from each and assert (a) both keep
        // supersession, (b) both key on github.workflow, and (c) the two group
        // expressions and workflow names differ — so at runtime the groups are
        // distinct and an offline-runner stall in one can't hold the other.
        let ci = read_workflow("ci.yml");
        let sh = read_workflow("e2e-selfhosted.yml");
        let ci_group = concurrency_group(&ci);
        let sh_group = concurrency_group(&sh);

        for (label, group, text) in [
            ("ci.yml", &ci_group, &ci),
            ("e2e-selfhosted.yml", &sh_group, &sh),
        ] {
            assert!(
                group.contains("github.workflow"),
                "{label} concurrency.group must be namespaced by github.workflow \
                 (got `{group}`) (EAI-7548)"
            );
            assert!(
                text.contains("cancel-in-progress: true"),
                "{label} must keep cancel-in-progress: true (EAI-7548)"
            );
        }
        // The github.workflow-keyed groups resolve via the workflow `name:`; those
        // names must differ for the runtime groups to be distinct.
        assert_ne!(
            workflow_name(&ci),
            workflow_name(&sh),
            "the two workflows must have different `name:` values so their \
             github.workflow-keyed concurrency groups are distinct (EAI-7548)"
        );
    }

    // Extractor guards: prove the helpers actually parse multiline forms, so the
    // contract tests above can't silently false-pass on a shape they don't handle.
    #[test]
    fn runs_on_extractor_flattens_multiline_forms() {
        let yaml = "\
jobs:
  a:
    runs-on: ubuntu-latest
  b:
    runs-on:
      - self-hosted
      - linux
      - amd-gpu
  c:
    runs-on: [self-hosted, windows,
      strix-halo, native]
";
        let vals = runs_on_values(yaml);
        assert_eq!(vals.len(), 3);
        assert!(vals[0].contains("ubuntu-latest"));
        assert!(vals[1].contains("self-hosted") && vals[1].contains("amd-gpu"));
        // The flow list split across lines must be joined so `strix-halo` is seen.
        assert!(vals[2].contains("self-hosted") && vals[2].contains("strix-halo"));
    }

    #[test]
    fn concurrency_group_extractor_reads_folded_value() {
        let yaml = "\
name: X

concurrency:
  # a comment mentioning github.workflow that must NOT count
  group: >-
    ${{ github.workflow }}-${{ github.ref }}-${{
      github.event_name == 'workflow_dispatch' && github.run_id || 'shared' }}
  cancel-in-progress: true

permissions:
  contents: read
";
        let g = concurrency_group(yaml);
        assert!(g.contains("github.workflow"));
        assert!(g.contains("github.run_id"));
        // Must stop at the next key, not swallow permissions.
        assert!(!g.contains("permissions"));
    }
}
