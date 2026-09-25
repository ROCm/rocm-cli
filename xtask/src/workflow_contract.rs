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
    use std::collections::BTreeMap;
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
        std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
            .replace("\r\n", "\n")
    }

    /// The self-hosted runner labels that must not appear in a `runs-on`.
    const SELF_HOSTED_LABELS: [&str; 5] =
        ["self-hosted", "amd-gpu", "strix-halo", "mi300x", "r9700"];

    /// Labels that do NOT narrow a `runs-on` to one kind of hardware.
    ///
    /// `self-hosted`, `linux` and `windows` are self-evidently generic.
    /// `amd-gpu` reads specific but is not: every AMD GPU runner registers it,
    /// so a lane selecting on it alone draws from a mixed-silicon pool.
    const GENERIC_LABELS: [&str; 4] = ["self-hosted", "linux", "windows", "amd-gpu"];

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
        flattened_values(text, "runs-on")
    }

    /// Extract the COMPLETE value of every `{key}:` in `text`, joining any
    /// block/flow continuation lines so a list item split across lines can't
    /// hide. Returns one flattened string per occurrence of the key.
    fn flattened_values(text: &str, key: &str) -> Vec<String> {
        let marker = format!("{key}:");
        let lines: Vec<&str> = text.lines().collect();
        let mut out = Vec::new();
        for (i, raw) in lines.iter().enumerate() {
            let line = strip_comment(raw);
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix(&marker) else {
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

    /// Split a flattened YAML sequence into its items. Handles the flow form
    /// (`[a, b]`) and the block form, which [`flattened_values`] joins into
    /// `- a - b`, so the two spellings compare equal.
    fn flattened_list_items(value: &str) -> Vec<String> {
        let value = value.trim();
        let items: Vec<String> =
            if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
                inner
                    .split(',')
                    .map(|item| item.trim().to_owned())
                    .collect()
            } else if let Some(block) = value.strip_prefix("- ") {
                block
                    .split(" - ")
                    .map(|item| item.trim().to_owned())
                    .collect()
            } else {
                vec![value.to_owned()]
            };
        items.into_iter().filter(|item| !item.is_empty()).collect()
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

    fn multiline_run_blocks(text: &str) -> Vec<String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut blocks = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if line.trim_start() != "run: |" {
                continue;
            }
            let run_indent = indent_of(line);
            let mut block = String::new();
            for body_line in &lines[i + 1..] {
                if !body_line.trim().is_empty() && indent_of(body_line) <= run_indent {
                    break;
                }
                block.push_str(body_line.trim());
                block.push('\n');
            }
            blocks.push(block);
        }
        blocks
    }

    fn invokes_e2e(block: &str) -> bool {
        block.lines().any(|line| {
            let line = line.trim();
            line == "cargo xtask e2e" || line.starts_with("cargo xtask e2e --")
        })
    }

    fn assert_prebuilt_e2e_lanes_export_rocmd(workflow: &str, text: &str) {
        let blocks: Vec<_> = multiline_run_blocks(text)
            .into_iter()
            .filter(|block| block.contains("ROCM_CLI_BINARY") && invokes_e2e(block))
            .collect();
        assert!(
            !blocks.is_empty(),
            "{workflow} must contain prebuilt E2E run blocks"
        );

        let mut unix_lanes = 0;
        let mut powershell_lanes = 0;
        for block in blocks {
            assert!(
                block.contains("cargo build --release -p rocm -p rocmd"),
                "{workflow} prebuilt E2E lane must build rocm and rocmd together:\n{block}"
            );
            assert!(
                block.contains("ROCM_CLI_ROCMD_BINARY"),
                "{workflow} prebuilt E2E lane exports ROCM_CLI_BINARY but not \
                 ROCM_CLI_ROCMD_BINARY:\n{block}"
            );

            if block.contains("export ROCM_CLI_BINARY=") {
                unix_lanes += 1;
                assert!(
                    block.contains(
                        "export ROCM_CLI_ROCMD_BINARY=\"$CARGO_TARGET_DIR/release/rocmd\""
                    ),
                    "{workflow} Unix E2E lane must export the release rocmd path:\n{block}"
                );
            } else if block.contains("$env:ROCM_CLI_BINARY") {
                powershell_lanes += 1;
                assert!(
                    block.contains(
                        "$env:ROCM_CLI_ROCMD_BINARY = \"$targetDir\\release\\rocmd.exe\""
                    ),
                    "{workflow} PowerShell E2E lane must export the release rocmd.exe path:\n{block}"
                );
            } else {
                panic!("{workflow} prebuilt E2E lane uses an unknown shell contract:\n{block}");
            }
        }
        assert!(
            unix_lanes > 0,
            "{workflow} must cover at least one Unix E2E lane"
        );
        assert!(
            powershell_lanes > 0,
            "{workflow} must cover at least one PowerShell E2E lane"
        );
    }

    /// Every lane that pre-builds `rocm` and hands it to the suite via
    /// `ROCM_CLI_BINARY` must enable the same test-hook feature `cargo xtask e2e`
    /// enables when it builds for itself.
    ///
    /// The suite's deterministic failure seams (e.g. the scripted Lemonade
    /// backend-install failure) are `#[cfg(feature = "e2e-test-hooks")]`. A lane
    /// that omits the feature ships a binary in which those seams do not exist,
    /// so the scenarios relying on them cannot reach their premise and fail as
    /// regressions — but only on whichever lane happens to select them, which is
    /// what made this divergence so hard to read the first time. Pin it here so a
    /// new lane copying an existing block cannot silently reintroduce it.
    fn assert_prebuilt_e2e_lanes_enable_test_hooks(workflow: &str, text: &str) {
        let blocks: Vec<_> = multiline_run_blocks(text)
            .into_iter()
            .filter(|block| block.contains("ROCM_CLI_BINARY") && invokes_e2e(block))
            .collect();
        assert!(
            !blocks.is_empty(),
            "{workflow} must contain prebuilt E2E run blocks"
        );
        for block in blocks {
            assert!(
                block.contains(
                    "cargo build --release -p rocm -p rocmd --features rocm/e2e-test-hooks"
                ),
                "{workflow} prebuilt E2E lane must build with \
                 `--features rocm/e2e-test-hooks`, matching what `cargo xtask e2e` \
                 builds for itself; without it the suite's scripted failure seams \
                 are compiled out:\n{block}"
            );
        }
    }

    /// Extract one top-level job's complete YAML block by its job id.
    fn job_block<'a>(text: &'a str, job: &str) -> &'a str {
        let marker = format!("  {job}:\n");
        let start = text
            .find(&marker)
            .unwrap_or_else(|| panic!("workflow defines job `{job}`"));
        let rest = &text[start + marker.len()..];
        let end = rest
            .match_indices("\n  ")
            .find_map(|(i, _)| {
                rest[i + 1..]
                    .lines()
                    .next()
                    .is_some_and(|line| line.starts_with("  ") && !line.starts_with("    "))
                    .then_some(i)
            })
            .unwrap_or(rest.len());
        &rest[..end]
    }

    /// Every job in `text` that targets a self-hosted runner, as
    /// `(job id, flattened runs-on)` pairs in file order.
    ///
    /// This is what lets the docs guard assert against the workflow instead of
    /// against a list copied into the test source: add a lane and the derived
    /// list grows, so the documentation assertions fail until the docs follow.
    ///
    /// The job-id scan is scoped to the `jobs:` block, because top-level keys
    /// like `push:` (under `on:`) and `group:` (under `concurrency:`) sit at the
    /// same indent and would otherwise read as job ids.
    fn self_hosted_e2e_jobs(text: &str) -> Vec<(String, String)> {
        let jobs = top_level_block(text, "jobs");
        jobs.lines()
            .filter(|line| indent_of(line) == 2 && !line.trim_start().starts_with('#'))
            .filter_map(|line| line.trim().strip_suffix(':'))
            .filter_map(|job| {
                let mut values = runs_on_values(job_block(text, job));
                assert!(
                    values.len() <= 1,
                    "job `{job}` declares more than one runs-on"
                );
                // A job without a runs-on (e.g. one that only `uses:` a
                // reusable workflow) schedules nothing self-hosted.
                let runs_on = values.pop()?.trim().to_owned();
                let labels = flattened_list_items(&runs_on);
                SELF_HOSTED_LABELS
                    .iter()
                    .any(|self_hosted| labels.iter().any(|label| label == self_hosted))
                    .then_some((job.to_owned(), runs_on))
            })
            .collect()
    }

    /// Extract the direct scalar entries from a named job-level mapping such as
    /// `env:`. Nested step mappings cannot satisfy this extractor.
    fn job_mapping(block: &str, mapping: &str) -> BTreeMap<String, String> {
        let marker = format!("{mapping}:");
        let lines: Vec<&str> = block.lines().collect();
        let (start, mapping_indent) = lines
            .iter()
            .enumerate()
            .find_map(|(i, line)| {
                (line.trim() == marker && indent_of(line) == 4).then_some((i, indent_of(line)))
            })
            .unwrap_or_else(|| panic!("job defines top-level mapping `{mapping}`"));

        let mut entries = BTreeMap::new();
        for raw in &lines[start + 1..] {
            if raw.trim().is_empty() || raw.trim_start().starts_with('#') {
                continue;
            }
            let line = strip_comment(raw);
            let indent = indent_of(line);
            if indent <= mapping_indent {
                break;
            }
            if indent != mapping_indent + 2 {
                continue;
            }
            let (key, raw_value) = line
                .trim()
                .split_once(':')
                .unwrap_or_else(|| panic!("mapping entry has a scalar value: `{line}`"));
            let value = raw_value.trim();
            let value = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
                .unwrap_or(value);
            assert!(
                entries.insert(key.to_owned(), value.to_owned()).is_none(),
                "mapping `{mapping}` contains duplicate key `{key}`"
            );
        }
        entries
    }

    fn job_scalar<'a>(block: &'a str, key: &str) -> &'a str {
        let marker = format!("{key}:");
        block
            .lines()
            .find_map(|line| {
                (indent_of(line) == 4)
                    .then(|| line.trim().strip_prefix(&marker))
                    .flatten()
                    .map(str::trim)
            })
            .unwrap_or_else(|| panic!("job defines top-level scalar `{key}`"))
    }

    fn markdown_table_rows(text: &str, header: &str) -> Vec<Vec<String>> {
        let mut lines = text.lines().skip_while(|line| *line != header);
        assert_eq!(
            lines.next(),
            Some(header),
            "markdown table `{header}` exists"
        );
        let separator = lines.next().expect("markdown table has a separator row");
        assert!(
            separator.starts_with("|---"),
            "markdown table has a separator row"
        );
        lines
            .take_while(|line| line.starts_with('|'))
            .map(|line| {
                line.trim_matches('|')
                    .split('|')
                    .map(|cell| cell.trim().to_owned())
                    .collect()
            })
            .collect()
    }

    /// Every backtick-delimited span in `text`, in order.
    fn backticked_items(text: &str) -> Vec<String> {
        text.split('`')
            .enumerate()
            .filter_map(|(i, item)| (i % 2 == 1).then_some(item.to_owned()))
            .collect()
    }

    fn backticked_list_between(text: &str, prefix: &str, suffix: &str) -> Vec<String> {
        let section = text
            .split_once(prefix)
            .unwrap_or_else(|| panic!("section starts with `{prefix}`"))
            .1
            .split_once(suffix)
            .unwrap_or_else(|| panic!("section ends with `{suffix}`"))
            .0;
        backticked_items(section)
    }

    fn normalized_whitespace(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
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
    fn every_ci_job_declares_a_timeout() {
        let ci = read_workflow("ci.yml");
        let jobs = top_level_block(&ci, "jobs");
        let job_ids: Vec<&str> = jobs
            .lines()
            .filter(|line| indent_of(line) == 2 && !line.trim_start().starts_with('#'))
            .filter_map(|line| line.trim().strip_suffix(':'))
            .collect();
        assert!(!job_ids.is_empty(), "expected at least one job in ci.yml");
        for job in job_ids {
            let block = job_block(&ci, job);
            assert!(
                block
                    .lines()
                    .any(|line| indent_of(line) == 4 && line.trim().starts_with("timeout-minutes:")),
                "job `{job}` in ci.yml has no timeout-minutes -- GitHub's 360min default applies, \
                 so a hung step (an unbounded network call, an unresponsive registry) holds a \
                 runner for six hours instead of failing fast (see the convention comment at the \
                 top of the jobs: block)"
            );
        }
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
    fn self_hosted_wsl_enables_merge_queue_scenarios_only_for_merge_group() {
        let sh = read_workflow("e2e-selfhosted.yml");
        let wsl = job_block(&sh, "e2e-wsl");
        let env = job_mapping(wsl, "env");
        assert_eq!(
            env.get("E2E_MERGE_QUEUE").map(String::as_str),
            Some("${{ github.event_name == 'merge_group' && '1' || '' }}"),
            "e2e-wsl's job-level env must opt into @merge-queue scenarios for merge_group only"
        );

        let nightly = read_workflow("nightly.yml");
        let nightly_wsl = job_block(&nightly, "e2e-wsl-nightly");
        let nightly_env = job_mapping(nightly_wsl, "env");
        assert!(
            !nightly_env.contains_key("E2E_MERGE_QUEUE"),
            "the nightly WSL job cannot receive merge_group events and must not opt into @merge-queue scenarios"
        );
    }

    #[test]
    fn wsl_jobs_accept_every_canonical_wsl_detection_signal() {
        for (workflow, job) in [
            ("e2e-selfhosted.yml", "e2e-wsl"),
            ("nightly.yml", "e2e-wsl-nightly"),
        ] {
            let text = read_workflow(workflow);
            let block = job_block(&text, job);
            // Remove shell line-continuation backslashes after whitespace has
            // been normalized so the assertion describes the effective test.
            let block = normalized_whitespace(block).replace("\\ ", "");
            assert!(
                block.contains("proc_version=$(cat /proc/version 2>/dev/null || true)"),
                "{workflow} job {job} must read the canonical kernel-version signal"
            );
            let distro_signal = ["$", "{", "WSL_DISTRO_NAME+x", "}"].concat();
            let canonical_union = format!(
                "if [ ! -e /dev/dxg ] && [ -z \"{distro_signal}\" ] && ! printf '%s\\n' \"$proc_version\" | grep -qiE 'microsoft|wsl'; then"
            );
            assert!(
                block.contains(&canonical_union),
                "{workflow} job {job} must accept the canonical union of WSL signals"
            );
            assert!(
                !block.contains("/proc/sys/kernel/osrelease"),
                "{workflow} job {job} must not use the narrower osrelease-only WSL check"
            );
        }
    }

    /// The nightly WSL lane's `runs-on` must track the per-PR lane's, byte for
    /// byte: both jobs claim to run on the same DevLab Dispatch pool host
    /// (`e2e-wsl-nightly`'s header comment, docs/ci-hardware-testing.md), and
    /// nothing else pins that claim -- reverting `e2e-wsl-nightly` to its old
    /// static `[self-hosted, linux, strix-halo, wsl]` labels (its pre-migration
    /// runs-on; `native` never applied to this job, only to the two
    /// `e2e-gpu-nightly-strix*` lanes) would leave every other assertion in
    /// this file green while the doc and the comment both quietly went false
    /// again.
    #[test]
    fn nightly_wsl_lane_shares_the_per_pr_pool_labels() {
        let sh = read_workflow("e2e-selfhosted.yml");
        let nightly = read_workflow("nightly.yml");
        let per_pr = runs_on_values(job_block(&sh, "e2e-wsl"));
        let nightly_wsl = runs_on_values(job_block(&nightly, "e2e-wsl-nightly"));
        assert!(!per_pr.is_empty(), "e2e-wsl declares a runs-on");
        assert!(
            per_pr.iter().any(|value| value.contains("devlab-dispatch")),
            "e2e-wsl must actually be on the DevLab Dispatch pool, not just equal to \
             nightly's (equal-and-empty would pass the assertion below): {per_pr:?}"
        );
        assert_eq!(
            per_pr, nightly_wsl,
            "e2e-wsl-nightly's runs-on must match e2e-wsl's exactly -- both are documented as \
             the same DevLab Dispatch pool"
        );
    }

    #[test]
    fn every_nightly_strix_job_uses_the_shared_machine_tui_budget() {
        let nightly = read_workflow("nightly.yml");
        for job in [
            "e2e-gpu-nightly-strix",
            "e2e-gpu-nightly-strix-windows",
            "e2e-wsl-nightly",
        ] {
            let env = job_mapping(job_block(&nightly, job), "env");
            assert_eq!(
                env.get("E2E_TUI_TIMEOUT_SECS").map(String::as_str),
                Some("90"),
                "nightly Strix job {job} must use the shared-machine TUI wait budget"
            );
        }
    }

    /// Both self-hosted workflows, so a lane added to either is covered.
    fn self_hosted_workflows() -> [(&'static str, String); 2] {
        [
            ("e2e-selfhosted.yml", read_workflow("e2e-selfhosted.yml")),
            ("nightly.yml", read_workflow("nightly.yml")),
        ]
    }

    #[test]
    fn every_self_hosted_lane_waits_for_an_available_gpu() {
        // `e2e-gpu-nightly` shipped without one and nothing noticed: it hung to
        // the 90-minute job cap on a wedged driver instead of failing in ~90s,
        // while its own per-PR twin and every sibling failed fast. The preflight
        // is what turns "absent, wedged, or still held by a leftover serve" into
        // a named error rather than a timeout.
        //
        // Deliberately derived rather than listed, so a new lane is covered the
        // day it lands. Any lane that genuinely should not wait for a GPU needs
        // an exemption added here with the reason — which is the point: it
        // becomes a decision someone makes, not one a copy-paste makes for them.
        for (workflow, text) in self_hosted_workflows() {
            for (job, _) in self_hosted_e2e_jobs(&text) {
                assert!(
                    job_block(&text, &job).contains("- name: GPU preflight"),
                    "{workflow} job `{job}` runs on self-hosted GPU hardware but has no \
                     GPU preflight step: on a wedged or occupied GPU it hangs to the job \
                     timeout instead of failing fast with a reason"
                );
            }
        }
    }

    /// Every `GPU preflight` step block in `text`, in file order.
    fn gpu_preflight_steps(text: &str) -> Vec<String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut steps = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if !line.trim_start().starts_with("- name: GPU preflight") {
                continue;
            }
            let step_indent = indent_of(line);
            let mut step = format!("{line}\n");
            for body in &lines[i + 1..] {
                if !body.trim().is_empty() && indent_of(body) <= step_indent {
                    break;
                }
                step.push_str(body);
                step.push('\n');
            }
            steps.push(step);
        }
        steps
    }

    /// The POSIX-shell body a preflight step actually runs, dedented.
    ///
    /// Two shapes carry shell: a plain `run: |` step, and the WSL lanes, which
    /// hand a here-string to `Invoke-WslBash.ps1`. A step that is PowerShell end
    /// to end yields `None` — driving those needs a PowerShell this test cannot
    /// assume, so they are covered by the shape assertions instead.
    fn preflight_shell_script(step: &str) -> Option<String> {
        let lines: Vec<&str> = step.lines().collect();
        let run_at = lines.iter().position(|l| l.trim() == "run: |")?;
        let run_indent = indent_of(lines[run_at]);
        let body: Vec<&str> = lines[run_at + 1..]
            .iter()
            .take_while(|l| l.trim().is_empty() || indent_of(l) > run_indent)
            .copied()
            .collect();
        let body = match body
            .iter()
            .position(|l| l.trim_end().ends_with("-Script @'"))
        {
            Some(open) => {
                let close = body.iter().position(|l| l.trim() == "'@")?;
                body[open + 1..close].to_vec()
            }
            None if step.contains("shell: powershell") => return None,
            None => body,
        };
        let pad = body
            .iter()
            .filter(|l| !l.trim().is_empty())
            .map(|l| indent_of(l))
            .min()?;
        let dedented: Vec<&str> = body
            .iter()
            .map(|l| if l.len() > pad { &l[pad..] } else { l.trim() })
            .collect();
        Some(dedented.join("\n") + "\n")
    }

    /// Whether this preflight guards an APU lane (the 8 GiB floor) rather than a
    /// discrete card (16 GiB).
    fn is_apu_preflight(step: &str) -> bool {
        step.contains("GPU_PREFLIGHT_MIN_FREE_GIB:-8") || step.contains("else { 8 }")
    }

    #[test]
    fn every_apu_preflight_block_measures_the_gtt_pool() {
        // A gfx1151 APU reports its BIOS carveout as VRAM total — 512 MiB on the
        // hosts these lanes run on — while the engine allocates from GTT-backed
        // system RAM. A floor applied to VRAM there can never be cleared, so the
        // lane fails on every healthy host. Six blocks guard that hardware class
        // and one was missed when the other five were converted; this is what
        // makes the next copy-paste fail loudly rather than months later.
        let mut checked = 0;
        for (workflow, text) in self_hosted_workflows() {
            for step in gpu_preflight_steps(&text)
                .iter()
                .filter(|s| is_apu_preflight(s))
            {
                assert!(
                    step.contains("showmeminfo vram gtt"),
                    "{workflow}: an APU-class GPU preflight still queries VRAM only; on a \
                     small-carveout gfx1151 host it can never clear its own floor"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 6,
            "expected at least six APU-class preflight blocks, found {checked} — if a lane \
             was removed update this floor, otherwise the extractor has stopped matching"
        );
    }

    #[test]
    fn apu_preflight_twins_do_not_drift() {
        // The nightly lanes are copies of their per-PR twins. One was left on the
        // pre-GTT script while the other five were converted and nothing failed,
        // because every contract test here asserts step names, env keys and
        // labels — never the script body.
        let scripts = |name: &str| -> Vec<String> {
            gpu_preflight_steps(&read_workflow(name))
                .iter()
                .filter(|s| is_apu_preflight(s))
                .filter_map(|s| preflight_shell_script(s))
                .collect()
        };
        let per_pr = scripts("e2e-selfhosted.yml");
        let nightly = scripts("nightly.yml");
        assert_eq!(
            per_pr.len(),
            2,
            "expected a native and an advisory APU preflight in e2e-selfhosted.yml"
        );
        assert_eq!(per_pr.len(), nightly.len(), "APU preflight count differs");
        for (i, (a, b)) in per_pr.iter().zip(nightly.iter()).enumerate() {
            assert_eq!(
                a, b,
                "APU preflight script #{i} has drifted between e2e-selfhosted.yml and nightly.yml"
            );
        }
    }

    #[cfg(unix)]
    fn smi_fixture(vram_total: u64, vram_used: u64, gtt: Option<(u64, u64)>) -> String {
        use std::fmt::Write as _;

        let mut s = String::from("===== ROCm System Management Interface =====\n");
        let _ = writeln!(s, "GPU[0]\t\t: VRAM Total Memory (B): {vram_total}");
        let _ = writeln!(s, "GPU[0]\t\t: VRAM Total Used Memory (B): {vram_used}");
        if let Some((total, used)) = gtt {
            let _ = writeln!(s, "GPU[0]\t\t: GTT Total Memory (B): {total}");
            let _ = writeln!(s, "GPU[0]\t\t: GTT Total Used Memory (B): {used}");
        }
        s
    }

    /// A `rocm-smi` stub that answers each `--showmeminfo` shape the preflight
    /// uses: both pools at once, GTT alone, VRAM alone. With `REJECT_COMBINED`
    /// the two-pool form fails the way an older build does, which is the case
    /// that must still reach GTT through the single-pool query.
    #[cfg(unix)]
    const SMI_STUB: &str = r#"#!/bin/sh
case "$*" in
  *vram*gtt*)
    [ "${REJECT_COMBINED:-0}" = 1 ] && exit 1
    cat "$FIXTURE" ;;
  *gtt*) grep GTT "$FIXTURE" ;;
  *)     grep -v GTT "$FIXTURE" ;;
esac
"#;

    /// Run a preflight script against fixture `rocm-smi` output, returning its
    /// exit code. `rocm-smi` is stubbed on `PATH`; nothing touches a real GPU.
    #[cfg(unix)]
    fn run_preflight(script: &str, fixture: &str, reject_combined: bool) -> i32 {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("temp dir");
        let fixture_path = dir.path().join("smi.txt");
        let script_path = dir.path().join("preflight.sh");
        let stub = dir.path().join("rocm-smi");
        std::fs::write(&fixture_path, fixture).expect("write fixture");
        std::fs::write(&script_path, script).expect("write script");
        std::fs::write(&stub, SMI_STUB).expect("write rocm-smi stub");
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755))
            .expect("make the stub executable");

        let path = format!(
            "{}:{}",
            dir.path().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        std::process::Command::new("bash")
            .arg("-e")
            .arg(&script_path)
            .env("PATH", path)
            .env("FIXTURE", &fixture_path)
            .env("REJECT_COMBINED", if reject_combined { "1" } else { "0" })
            // One poll, then the script's own 5s backoff ends the loop.
            .env("GPU_PREFLIGHT_CEILING_SECS", "1")
            .output()
            .expect("running the preflight script")
            .status
            .code()
            .expect("preflight exited with a status code")
    }

    #[cfg(unix)]
    #[test]
    fn apu_preflight_gates_on_the_pool_the_engine_uses() {
        const GIB: u64 = 1024 * 1024 * 1024;
        const MIB: u64 = 1024 * 1024;

        // The contract tests above cannot tell a working gate from a broken one:
        // they all pass with the production logic reverted. This drives the real
        // script against fixture tool output instead, which is what distinguishes
        // "measures the right pool" from "passes whenever any pool looks free" —
        // the zero-total row below failed before it was written.
        if std::process::Command::new("timeout")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("skipping: no `timeout` on PATH (GNU coreutils)");
            return;
        }

        // (what the host reports, whether the two-pool query is rejected,
        //  expected exit: native lane, advisory lane)
        let cases: Vec<(&str, String, bool, i32, i32)> = vec![
            (
                "carveout with a free aperture",
                smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB))),
                false,
                0,
                0,
            ),
            // An older rocm-smi rejects `--showmeminfo vram gtt` outright. The
            // single-pool fallback carries no GTT rows, so without a dedicated
            // GTT query this healthy APU would fail as "no GTT pool" — the very
            // false low-memory report this gate exists to stop producing.
            (
                "carveout where rocm-smi rejects the two-pool query",
                smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB))),
                true,
                0,
                0,
            ),
            (
                "carveout with the aperture pinned by a leftover serve",
                smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, 61 * GIB))),
                false,
                1,
                1,
            ),
            (
                "discrete card with VRAM free",
                smi_fixture(192 * GIB, 10 * GIB, None),
                false,
                0,
                0,
            ),
            (
                "discrete card with VRAM held",
                smi_fixture(192 * GIB, 191 * GIB, None),
                false,
                1,
                1,
            ),
            // A wedged driver must not be read as "this is an APU" and waved
            // through on whatever the other pool reports.
            (
                "zero VRAM total against a healthy aperture",
                smi_fixture(0, 0, Some((62 * GIB, GIB))),
                false,
                1,
                0,
            ),
            // Nothing measurable: the native lane fails, the advisory lane warns.
            (
                "carveout with no GTT pool reported",
                smi_fixture(512 * MIB, 200 * MIB, None),
                false,
                1,
                0,
            ),
        ];

        let steps = gpu_preflight_steps(&read_workflow("e2e-selfhosted.yml"));
        let native = steps
            .iter()
            .filter(|s| is_apu_preflight(s) && !s.contains("advisory"))
            .find_map(|s| preflight_shell_script(s))
            .expect("native APU preflight shell script");
        let advisory = steps
            .iter()
            .filter(|s| s.contains("advisory"))
            .find_map(|s| preflight_shell_script(s))
            .expect("advisory APU preflight shell script");

        std::thread::scope(|scope| {
            let handles: Vec<_> = cases
                .iter()
                .map(|(label, fixture, reject, want_native, want_advisory)| {
                    let (native, advisory) = (&native, &advisory);
                    scope.spawn(move || {
                        (
                            label,
                            run_preflight(native, fixture, *reject),
                            run_preflight(advisory, fixture, *reject),
                            *want_native,
                            *want_advisory,
                        )
                    })
                })
                .collect();
            for handle in handles {
                let (label, native, advisory, want_native, want_advisory) =
                    handle.join().expect("preflight case thread");
                assert_eq!(native, want_native, "native preflight, {label}");
                assert_eq!(advisory, want_advisory, "advisory preflight, {label}");
            }
        });
    }

    #[test]
    fn every_self_hosted_lane_pins_a_hardware_label() {
        // `e2e-gpu` and `e2e-gpu-nightly` shipped as `[self-hosted, linux,
        // amd-gpu]`. The Strix Halo Linux hosts carry `amd-gpu` too, so the
        // MI300X-named lanes were scheduled onto gfx1151 whenever a Strix
        // runner won the race: red on the WSL host, and a PASS on the native
        // Ubuntu one — published as `e2e-gpu-report`, which the consolidated
        // grid labels MI300X. A green run on hardware the lane does not name is
        // worse than a red one, because nothing prompts anyone to look.
        //
        // Derived like its siblings: a new lane is covered the day it lands,
        // and only a lane whose labels are ALL generic fails, so pinning a new
        // hardware label needs no change here.
        for (workflow, text) in self_hosted_workflows() {
            for (job, runs_on) in self_hosted_e2e_jobs(&text) {
                let labels = flattened_list_items(&runs_on);
                assert!(
                    labels.iter().any(|l| !GENERIC_LABELS.contains(&l.as_str())),
                    "{workflow} job `{job}` selects its runner with generic labels only \
                     (`{runs_on}`), so it can land on any AMD GPU host. Pin a label that \
                     identifies the hardware the lane is named for (e.g. `mi300x`, \
                     `r9700`, `strix-halo`)"
                );
            }
        }
    }

    #[test]
    fn every_self_hosted_lane_allows_for_a_cold_serve() {
        // The per-PR Strix Windows lane was the only lane without this, while its
        // own nightly twin set it. A first serve on shared hardware loads the
        // model before it answers; the default budget is short enough that the
        // load reads as a product failure.
        for (workflow, text) in self_hosted_workflows() {
            for (job, _) in self_hosted_e2e_jobs(&text) {
                let env = job_mapping(job_block(&text, &job), "env");
                assert_eq!(
                    env.get("E2E_SERVE_TIMEOUT_SECS").map(String::as_str),
                    Some("300"),
                    "{workflow} job `{job}` must give a cold serve the same budget as \
                     every other self-hosted lane"
                );
            }
        }
    }

    #[test]
    fn dispatchable_wsl_nightly_run_has_the_full_nightly_job_budget() {
        let self_hosted = read_workflow("e2e-selfhosted.yml");
        let nightly = read_workflow("nightly.yml");
        let dispatch_timeout = job_scalar(job_block(&self_hosted, "e2e-wsl"), "timeout-minutes");
        let nightly_timeout = job_scalar(job_block(&nightly, "e2e-wsl-nightly"), "timeout-minutes");
        assert_eq!(
            dispatch_timeout, "120",
            "the 2400s large-model readiness budget plus the ephemeral pool's per-job WSL install/build/prewarm \
             overhead needs the established 120-minute job cap for setup and the remaining suite"
        );
        assert_eq!(
            dispatch_timeout, nightly_timeout,
            "e2e-wsl supports include_nightly, so its job timeout must cover the same 2400s scenario plus setup as e2e-wsl-nightly"
        );
    }

    /// The hardware-testing doc is a map of the self-hosted lanes, so every
    /// expectation here is DERIVED from the workflows rather than restated in
    /// the test source. A list copied into the test only proves the test and the
    /// doc agree with each other; it says nothing about the YAML they describe,
    /// and both can go stale together while the guard stays green.
    #[test]
    fn hardware_testing_docs_cover_all_self_hosted_platforms() {
        let self_hosted = read_workflow("e2e-selfhosted.yml");
        let lanes = self_hosted_e2e_jobs(&self_hosted);
        assert!(
            !lanes.is_empty(),
            "expected at least one self-hosted job in e2e-selfhosted.yml (extractor sanity check)"
        );

        let docs = std::fs::read_to_string(repo_root().join("docs/ci-hardware-testing.md"))
            .expect("read hardware testing docs");
        let documented: Vec<Vec<String>> =
            markdown_table_rows(&docs, "| Job | Workflow | Platform | Runner labels |")
                .into_iter()
                .filter(|row| {
                    row.get(1)
                        .is_some_and(|workflow| workflow == "`e2e-selfhosted.yml`")
                })
                .collect();

        let documented_jobs: Vec<String> = documented.iter().map(|row| row[0].clone()).collect();
        let declared_jobs: Vec<String> = lanes.iter().map(|(job, _)| format!("`{job}`")).collect();
        assert_eq!(
            documented_jobs, declared_jobs,
            "the hardware testing table must have one row per self-hosted job in \
             e2e-selfhosted.yml, in workflow order"
        );

        for (row, (job, runs_on)) in documented.iter().zip(&lanes) {
            assert!(
                !row[2].is_empty(),
                "the Platform cell for `{job}` describes the hardware in prose (it is not \
                 derivable from the workflow), so it must at least be non-empty"
            );
            let cell = &row[3];
            let quoted = backticked_items(cell);
            assert_eq!(
                quoted.len(),
                1,
                "the Runner labels cell for `{job}` must quote exactly one label list: `{cell}`"
            );
            assert_eq!(
                flattened_list_items(&quoted[0]),
                flattened_list_items(runs_on),
                "the documented runner labels for `{job}` must match its actual runs-on"
            );
        }

        // Derived from the same scan `every_uploaded_e2e_artifact_has_a_name_the_report_can_label`
        // asserts on, so a new lane's artifact has to reach the doc too. The
        // consolidated artifacts are report outputs, not per-platform inputs.
        let mut declared_artifacts: Vec<String> = ["ci.yml", "e2e-selfhosted.yml", "nightly.yml"]
            .into_iter()
            .flat_map(|workflow| {
                crate::e2e_report::uploaded_e2e_artifacts(
                    &repo_root().join(".github/workflows").join(workflow),
                )
            })
            .filter(|name| !name.starts_with("e2e-consolidated-report"))
            .collect();
        declared_artifacts.sort();
        declared_artifacts.dedup();
        let mut documented_artifacts = backticked_list_between(
            &docs,
            "The lane artifacts are named canonically (",
            ") in every workflow",
        );
        // Sorted, not deduplicated: a name listed twice must still fail.
        documented_artifacts.sort();
        assert_eq!(
            documented_artifacts, declared_artifacts,
            "the canonical artifact list must enumerate every uploaded report artifact exactly once"
        );

        let platform_input = nested_block(
            &top_level_block(&self_hosted, "on"),
            "      platform:", // workflow_dispatch.inputs.platform
        );
        let mut options = flattened_values(&platform_input, "options");
        assert_eq!(
            options.len(),
            1,
            "the dispatch `platform` input declares exactly one options list"
        );
        let declared_options = flattened_list_items(&options.pop().expect("length just asserted"));
        assert_eq!(
            backticked_list_between(
                &docs,
                "- `platform` (choice: ",
                ") — which self-hosted job(s) to run",
            ),
            declared_options,
            "the documented dispatch choices must match workflow_dispatch.inputs.platform.options"
        );

        // Prose enumerations of the lanes. Nothing read these before, so adding
        // a lane and updating only the table left them quietly wrong.
        let declared_ids: Vec<String> = lanes.iter().map(|(job, _)| job.clone()).collect();
        for (prefix, suffix) in [
            ("The self-hosted jobs (", ") run on AMD GPU systems"),
            ("The self-hosted jobs — ", " — all run with"),
        ] {
            assert_eq!(
                backticked_list_between(&docs, prefix, suffix),
                declared_ids,
                "the `{prefix}…{suffix}` sentence must name every self-hosted lane, in \
                 workflow order"
            );
        }

        // The README's own lane lists were the one hand-copied hole in this
        // otherwise derived net: the nightly sentence was asserted by literal
        // string match against a copy in this test source, so both could go
        // stale together while the guard stayed green — which is exactly what
        // happened when the R9700 lane landed. Derive both from the YAML.
        //
        // Whitespace is normalized first because these are prose sentences and
        // a table, wrapped for readability; the lane lists must survive a
        // reflow that does not change what the doc says.
        let readme = std::fs::read_to_string(repo_root().join("tests/e2e-cucumber/README.md"))
            .expect("read E2E README");
        let readme_flat = normalized_whitespace(&readme);

        let nightly = read_workflow("nightly.yml");
        let nightly_lanes = self_hosted_e2e_jobs(&nightly);
        assert!(
            !nightly_lanes.is_empty(),
            "expected at least one self-hosted job in nightly.yml (extractor sanity check)"
        );
        let nightly_ids: Vec<String> = nightly_lanes.iter().map(|(job, _)| job.clone()).collect();
        assert_eq!(
            backticked_list_between(&readme_flat, "as non-blocking lanes (", ") with"),
            nightly_ids,
            "the E2E README must name every nightly self-hosted lane, in workflow order"
        );

        // The README's CI job table is the per-PR view: the blocking mock job
        // from ci.yml, then one row per self-hosted lane. Only the self-hosted
        // rows are derivable, so the mock row is matched by workflow and the
        // rest are compared against e2e-selfhosted.yml.
        let readme_rows = markdown_table_rows(&readme, "| Job | Workflow | Platform | Blocking |");
        let (mock_rows, self_hosted_rows): (Vec<_>, Vec<_>) = readme_rows
            .into_iter()
            .partition(|row| row.get(1).is_some_and(|workflow| workflow == "`ci.yml`"));
        assert_eq!(
            mock_rows
                .iter()
                .map(|row| row[0].clone())
                .collect::<Vec<_>>(),
            vec!["`e2e`".to_owned()],
            "the README CI job table must carry exactly one blocking mock row"
        );
        assert_eq!(
            self_hosted_rows
                .iter()
                .map(|row| row[0].clone())
                .collect::<Vec<_>>(),
            lanes
                .iter()
                .map(|(job, _)| format!("`{job}`"))
                .collect::<Vec<_>>(),
            "the README CI job table must have one row per self-hosted job in \
             e2e-selfhosted.yml, in workflow order"
        );
        for row in &self_hosted_rows {
            assert_eq!(
                row[3], "no",
                "self-hosted lane `{}` is continue-on-error, so the README must not \
                 document it as blocking",
                row[0]
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

    #[test]
    fn every_shared_uv_cache_sits_inside_the_gc_bounded_directory() {
        // Two independent properties ride on this one path, and both fail
        // silently.
        //
        // Hardlinking: uv can only link out of its cache into a managed
        // environment when the two are reachable without crossing a mount point.
        // Otherwise it copies, exits 0, and two of the three install paths
        // discard its warning — invisible unless you compare inodes.
        //
        // Eviction: the runner pod's `uv-cache-gc` initContainer bounds exactly
        // one directory — the `uv-cache` subPath of the work PVC, which surfaces
        // in the job as <runner-root>/uv-cache. It sweeps abandoned `.tmp*`
        // extractions, then deletes `archive-v0` if FREE space on the volume is
        // under 60GiB. A free-space floor, not a size cap. A cache placed
        // elsewhere on the same volume still hardlinks, so every signal stays
        // green while nothing enforces the floor — and the volume also holds
        // `.runner`, whose credentials need a repo-Administration token to
        // re-register. The 49G/49G strand that motivated this was the separate
        // 50Gi cache PVC, since deleted, which had no floor at all.
        //
        // $RUNNER_WORKSPACE is <runner-root>/_work/<repo>, so a lane that derives
        // the cache from it directly lands one level too deep and escapes the GC.
        // Asserting the derivation rather than a literal keeps this honest if the
        // runner root ever moves.
        const DERIVATION: &str = "runner_root=\"$(dirname \"$(dirname \"$RUNNER_WORKSPACE\")\")\"";
        const EXPORT: &str = "export E2E_SHARED_UV_CACHE_DIR=\"$runner_root/uv-cache\"";

        for (workflow, text) in self_hosted_workflows() {
            let mut setters = 0;
            for block in multiline_run_blocks(&text) {
                let Some(line) = block
                    .lines()
                    .find(|line| line.starts_with("export E2E_SHARED_UV_CACHE_DIR="))
                else {
                    continue;
                };
                setters += 1;
                // Full-line equality, not `contains`: a substring match accepts
                // `$runner_root/uv-cache-old`, which is outside the bounded
                // directory and would fail exactly the way this test exists to
                // prevent.
                assert_eq!(
                    line, EXPORT,
                    "{workflow} sets the shared uv cache to `{line}`. It must be \
                     exactly `{EXPORT}` — <runner-root>/uv-cache is the only directory \
                     the runner's uv-cache-gc initContainer bounds. $RUNNER_WORKSPACE \
                     itself is one level too deep, which keeps the hardlinks but \
                     silently drops the 60GiB free-space floor"
                );
                assert!(
                    block.contains(DERIVATION),
                    "{workflow} exports E2E_SHARED_UV_CACHE_DIR from `$runner_root` \
                     without defining it in the same run block; add `{DERIVATION}`"
                );
            }
            // Without this, deleting every export is a silent pass — the loop
            // above simply finds nothing to assert on. Same guard the sibling
            // `assert_prebuilt_e2e_lanes_export_rocmd` uses.
            assert!(
                setters > 0,
                "{workflow} sets E2E_SHARED_UV_CACHE_DIR nowhere. Its GPU lanes share a \
                 pre-warmed runtime, so an unset cache sends uv to a per-job default \
                 outside the GC-bounded directory"
            );
        }
    }

    #[test]
    fn self_hosted_prebuilt_e2e_lanes_export_rocmd() {
        let workflow = read_workflow("e2e-selfhosted.yml");
        assert_prebuilt_e2e_lanes_export_rocmd("e2e-selfhosted.yml", &workflow);
    }

    #[test]
    fn nightly_prebuilt_e2e_lanes_export_rocmd() {
        let workflow = read_workflow("nightly.yml");
        assert_prebuilt_e2e_lanes_export_rocmd("nightly.yml", &workflow);
    }

    #[test]
    fn self_hosted_prebuilt_e2e_lanes_enable_test_hooks() {
        let workflow = read_workflow("e2e-selfhosted.yml");
        assert_prebuilt_e2e_lanes_enable_test_hooks("e2e-selfhosted.yml", &workflow);
    }

    #[test]
    fn nightly_prebuilt_e2e_lanes_enable_test_hooks() {
        let workflow = read_workflow("nightly.yml");
        assert_prebuilt_e2e_lanes_enable_test_hooks("nightly.yml", &workflow);
    }

    #[test]
    fn gpu_prewarm_caches_are_namespaced_by_source_layout() {
        // The shared tree survives `git clean` and every branch on the runner.
        // A branch that composes runtimes under a new source layout would
        // otherwise leave keys and manifests in that tree which code on the
        // previous layout cannot read, poisoning the lanes it never touched.
        for workflow_name in ["e2e-selfhosted.yml", "nightly.yml"] {
            let workflow = read_workflow(workflow_name);
            assert!(
                workflow.contains("e2e-prewarm-multi-arch-v2"),
                "{workflow_name} must isolate the canonical multi-arch runtime tree"
            );
            assert!(
                !workflow.lines().any(|line| {
                    line.trim_end().ends_with("e2e-prewarm\"")
                        || line.trim_end().ends_with("e2e-prewarm'")
                }),
                "{workflow_name} still uses the generation-agnostic pre-warm tree"
            );
        }
    }

    /// The Windows lane must hand its lifecycle E2E run the release binaries the
    /// Build step already produced.
    ///
    /// Without `ROCM_CLI_BINARY`, `cargo xtask e2e` builds `rocm`/`rocmd` for
    /// itself WITH `--features rocm/e2e-test-hooks`. That is a different feature
    /// resolution than the Build step's, so cargo rebuilds the entire release
    /// graph instead of reusing it — a second 3-4 minute release build on the one
    /// job that alone determines total CI wall clock.
    ///
    /// Note this is the mirror image of
    /// [`assert_prebuilt_e2e_lanes_enable_test_hooks`]: lanes running the FULL
    /// suite must build WITH the hooks, while this lifecycle-only lane must build
    /// WITHOUT them. `E2E_ONLY_LIFECYCLE` keeps it to @lifecycle scenarios, none
    /// of which use a scripted seam, and the lane packages and installs the
    /// binary through the real installer — so it must ship what a release ships.
    #[test]
    fn ci_windows_lifecycle_lane_reuses_the_binaries_it_built() {
        let ci = read_workflow("ci.yml");
        let windows = job_block(&ci, "windows-build-and-test");

        assert!(
            windows.contains("cargo build --release -p rocm -p rocmd"),
            "the Windows lane must pre-build the release binaries"
        );

        assert!(
            windows.contains("cargo xtask e2e"),
            "the Windows lane must run the lifecycle E2E suite"
        );

        // A bare `run: cargo xtask e2e` is exactly the regression this guards:
        // no run block can export the binaries, so xtask rebuilds them itself.
        let lifecycle = multiline_run_blocks(windows)
            .into_iter()
            .find(|block| invokes_e2e(block))
            .unwrap_or_else(|| "<no multiline run block invoking `cargo xtask e2e`>".to_owned());

        for var in ["ROCM_CLI_BINARY", "ROCM_CLI_ROCMD_BINARY"] {
            assert!(
                lifecycle.contains(var),
                "the Windows lifecycle lane must export {var} so `cargo xtask e2e` \
                 reuses the already-built release binaries instead of recompiling \
                 the whole release graph under a different feature set:\n{lifecycle}"
            );
        }

        // Naming the variables is not enough: pointing them at `target\debug\`
        // would satisfy the check above while defeating the reuse this test is
        // named for. Pin the whole assignment for BOTH, so neither can drift to a
        // debug or stale target dir. `assert_prebuilt_e2e_lanes_export_rocmd` pins
        // only `rocmd` for the other PowerShell prebuilt lanes; closing that half
        // of the mirror belongs with the shared helper, not here.
        for (var, exe) in [
            ("ROCM_CLI_BINARY", "rocm.exe"),
            ("ROCM_CLI_ROCMD_BINARY", "rocmd.exe"),
        ] {
            assert!(
                lifecycle.contains(&format!("$env:{var} = \"$targetDir\\release\\{exe}\"")),
                "the Windows lifecycle lane must export the RELEASE {exe} path — the \
                 binaries the Build step produced, not a debug or stale target dir:\n{lifecycle}"
            );
        }

        // The fail-fast guard is a deliberate part of this lane: without it a
        // missing binary surfaces as a `failed to run` deep in the suite. Nothing
        // else here references it, so without this assertion the whole
        // `Test-Path`/`throw` block can be deleted with every test still green.
        assert!(
            lifecycle.contains("Test-Path -LiteralPath")
                && lifecycle.contains("throw \"expected the Build step to have produced"),
            "the Windows lifecycle lane must fail fast, naming the missing binary, \
             when the Build step did not produce it:\n{lifecycle}"
        );

        assert!(
            !lifecycle.contains("e2e-test-hooks"),
            "the lifecycle-only lane packages and installs what a release ships, \
             so it must NOT carry the test hooks:\n{lifecycle}"
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
    fn self_hosted_jobs_extractor_reads_ids_from_the_jobs_block_only() {
        let yaml = "\
name: X

on:
  push:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: x

jobs:
  # A comment sits at job indent and is not a job id.
  hosted:
    runs-on: ubuntu-latest
  gpu:
    runs-on: [self-hosted, linux, amd-gpu]
    steps:
      - name: irrelevant
        run: echo hi
  strix:
    runs-on:
      - self-hosted
      - windows
      - strix-halo
  reusable:
    uses: ./.github/workflows/other.yml
";
        assert_eq!(
            self_hosted_e2e_jobs(yaml),
            vec![
                ("gpu".to_owned(), "[self-hosted, linux, amd-gpu]".to_owned()),
                (
                    "strix".to_owned(),
                    "- self-hosted - windows - strix-halo".to_owned()
                ),
            ],
            "only jobs are considered (not `push:`/`group:` at the same indent), only \
             self-hosted runs-on values are kept, and both list spellings are flattened"
        );
        assert_eq!(
            flattened_list_items("- self-hosted - windows - strix-halo"),
            flattened_list_items("[self-hosted, windows, strix-halo]"),
            "the two sequence spellings must compare equal"
        );
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

    #[test]
    fn job_mapping_extractor_ignores_nested_step_env() {
        let block = "    env:\n      TOP_LEVEL: \"expected\"\n    steps:\n      - name: nested\n        env:\n          E2E_MERGE_QUEUE: wrong\n";
        let env = job_mapping(block, "env");
        assert_eq!(env.get("TOP_LEVEL").map(String::as_str), Some("expected"));
        assert!(!env.contains_key("E2E_MERGE_QUEUE"));
    }

    #[test]
    fn job_mapping_extractor_ignores_blank_and_full_line_comments() {
        let block = "    env:\n      BEFORE: one\n\n# a YAML comment may be less indented than the mapping\n      # or aligned with its entries\n      AFTER: two\n    steps:\n";
        let env = job_mapping(block, "env");
        assert_eq!(env.get("BEFORE").map(String::as_str), Some("one"));
        assert_eq!(env.get("AFTER").map(String::as_str), Some("two"));
    }

    #[test]
    #[should_panic(expected = "mapping entry has a scalar value")]
    fn job_mapping_extractor_rejects_malformed_non_comment_rows() {
        let block = "    env:\n      VALID: one\n      MALFORMED\n    steps:\n";
        let _ = job_mapping(block, "env");
    }
}
