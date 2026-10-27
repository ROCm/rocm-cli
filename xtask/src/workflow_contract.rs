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
    /// `- a - b`, so the two spellings compare equal. Surrounding quotes are
    /// stripped from each item too, so `release/**` and `"release/**"` — both
    /// valid, semantically identical YAML — also compare equal.
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
        items
            .into_iter()
            .map(|item| strip_quotes(&item))
            .filter(|item| !item.is_empty())
            .collect()
    }

    /// Strip one layer of matching `"..."` or `'...'` quoting, if present.
    fn strip_quotes(item: &str) -> String {
        let mut chars = item.chars();
        match (chars.next(), chars.next_back()) {
            (Some('"'), Some('"')) | (Some('\''), Some('\'')) if item.len() >= 2 => {
                chars.as_str().to_owned()
            }
            _ => item.to_owned(),
        }
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

    /// The full `on.push.branches` list from a workflow's top-level `push:`
    /// trigger, as a flattened item vector (e.g. `["main", "release/**"]`).
    fn push_branches(text: &str) -> Vec<String> {
        let push_block = nested_block(text, "  push:");
        let branches = flattened_values(&push_block, "branches");
        assert_eq!(
            branches.len(),
            1,
            "expected exactly one branches: list under this workflow's push trigger"
        );
        flattened_list_items(&branches[0])
    }

    /// Reverting or mistyping the `release/**` push branch (EAI-8761) would
    /// leave every other assertion in this suite green, since none of them
    /// read `on.push.branches`. Only `e2e-selfhosted.yml` carries it — a
    /// release-branch push should get the self-hosted regression matrix ahead
    /// of cutting the `v*` tag `release.yml` builds from. `ci.yml`'s own push
    /// trigger stays `main`-only, so it does not run on the release-branch
    /// commit itself; a direct push or cherry-pick to a release branch is not
    /// covered by it, and coverage instead comes from whatever pull request
    /// produced that commit.
    #[test]
    fn self_hosted_workflow_fires_on_release_branch_push() {
        let workflow = read_workflow("e2e-selfhosted.yml");
        assert_eq!(
            push_branches(&workflow),
            vec!["main".to_string(), "release/**".to_string()],
            "e2e-selfhosted.yml must run its push-triggered jobs on `main` and any \
             `release/**` branch, ahead of cutting the `v*` tag release.yml builds from (EAI-8761)"
        );
    }

    #[test]
    fn ci_workflow_push_trigger_stays_main_only() {
        let workflow = read_workflow("ci.yml");
        assert_eq!(
            push_branches(&workflow),
            vec!["main".to_string()],
            "ci.yml's push trigger must stay main-only; it does not run on the \
             release-branch commit itself, so it must not re-run on push too"
        );
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

    /// Each nightly DevLab Dispatch lane's `runs-on` must track its per-PR
    /// counterpart's, byte for byte: both jobs in each pair claim to run on
    /// the same pool host (the nightly job's header comment,
    /// docs/ci-hardware-testing.md), and nothing else pins that claim --
    /// reverting a nightly job to its old static `native`/`wsl` labels would
    /// leave every other assertion in this file green while the doc and the
    /// comment both quietly went false again.
    #[test]
    fn nightly_devlab_dispatch_lanes_share_the_per_pr_pool_labels() {
        let sh = read_workflow("e2e-selfhosted.yml");
        let nightly = read_workflow("nightly.yml");
        for (per_pr_job, nightly_job) in [
            ("e2e-wsl", "e2e-wsl-nightly"),
            ("e2e-gpu-strix-windows", "e2e-gpu-nightly-strix-windows"),
        ] {
            let per_pr = runs_on_values(job_block(&sh, per_pr_job));
            let nightly_values = runs_on_values(job_block(&nightly, nightly_job));
            assert!(!per_pr.is_empty(), "{per_pr_job} declares a runs-on");
            assert!(
                per_pr.iter().any(|value| value.contains("devlab-dispatch")),
                "{per_pr_job} must actually be on the DevLab Dispatch pool, not just equal to \
                 {nightly_job}'s (equal-and-empty would pass the assertion below): {per_pr:?}"
            );
            assert_eq!(
                per_pr, nightly_values,
                "{nightly_job}'s runs-on must match {per_pr_job}'s exactly -- both are \
                 documented as the same DevLab Dispatch pool"
            );
        }
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
        //
        // The step is looked up by NAME among the job's own `steps:` items, not
        // found as a substring of the job: the comments around these steps talk
        // about the preflight at length, and a comment — or another step's
        // script — that happened to carry the old `- name:` text would keep a
        // substring match satisfied after the step itself was gone. Presence
        // only: nothing here pins where in the job it runs.
        for (workflow, text) in self_hosted_workflows() {
            for (job, _) in self_hosted_e2e_jobs(&text) {
                let steps = job_steps(job_block(&text, &job))
                    .unwrap_or_else(|| panic!("{workflow} job `{job}` has no `steps:` list"));
                let Some((_, step)) = steps
                    .iter()
                    .find(|(name, _)| name.starts_with("GPU preflight"))
                else {
                    let names: Vec<&str> = steps
                        .iter()
                        .map(|(name, _)| name.as_str())
                        .filter(|name| !name.is_empty())
                        .collect();
                    panic!(
                        "{workflow} job `{job}` runs on self-hosted GPU hardware but has no \
                         step named `GPU preflight …`: on a wedged or occupied GPU it hangs \
                         to the job timeout instead of failing fast with a reason. Its named \
                         steps are: {names:?}"
                    );
                };
                assert!(
                    run_block(step).is_some_and(|body| body.iter().any(|l| !l.trim().is_empty())),
                    "{workflow} job `{job}` names a GPU preflight step but it has no \
                     `run: |` script, so it waits for nothing"
                );
            }
        }
    }

    /// A job's `steps:` items as `(name, whole step)` pairs, in file order;
    /// `None` when the job has no `steps:` list. `block` is what `job_block`
    /// returns.
    ///
    /// Only the list's own items count — lines at exactly the item indent that
    /// open with `- ` — so a comment, or a `- name:` line inside some step's
    /// script, can never pass for a step. A step without a `name:` key is
    /// listed with an empty name.
    fn job_steps(block: &str) -> Option<Vec<(String, String)>> {
        let lines: Vec<&str> = block.lines().collect();
        let steps_at = lines
            .iter()
            .position(|l| strip_comment(l).trim() == "steps:")?;
        let steps_indent = indent_of(lines[steps_at]);
        let item_indent = lines[steps_at + 1..]
            .iter()
            .find(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .filter(|l| l.trim_start().starts_with("- "))
            .map(|l| indent_of(l))?;
        let mut steps = Vec::new();
        for (i, line) in lines.iter().enumerate().skip(steps_at + 1) {
            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if indent_of(line) <= steps_indent {
                break;
            }
            if indent_of(line) != item_indent || !trimmed.starts_with("- ") {
                continue;
            }
            let step = step_block(&lines, i);
            steps.push((step_name(&step), step));
        }
        Some(steps)
    }

    /// The non-empty lines of the `key: |` block scalar in `block`, trimmed.
    fn block_scalar_lines(block: &str, key: &str) -> Vec<String> {
        let marker = format!("{key}: |");
        let mut lines = block.lines().skip_while(|line| line.trim() != marker);
        let head = lines
            .next()
            .unwrap_or_else(|| panic!("block defines `{marker}`"));
        let indent = indent_of(head);
        lines
            .take_while(|line| line.trim().is_empty() || indent_of(line) > indent)
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect()
    }

    const REPORT_CHECK_STEP: &str = "- name: Check every lane that ran uploaded its report";

    /// A step block, found by its exact `- name:` line, including that line.
    fn named_step(block: &str, name_line: &str) -> String {
        let lines: Vec<&str> = block.lines().collect();
        let at = lines
            .iter()
            .position(|line| line.trim() == name_line)
            .unwrap_or_else(|| panic!("block has a step `{name_line}`"));
        let step_indent = indent_of(lines[at]);
        let mut step = format!("{}\n", lines[at]);
        for body in &lines[at + 1..] {
            if !body.trim().is_empty() && indent_of(body) <= step_indent {
                break;
            }
            step.push_str(body);
            step.push('\n');
        }
        step
    }

    /// The consolidated report job of each self-hosted workflow, with its
    /// missing-report check step.
    fn report_check_steps() -> Vec<(&'static str, String, &'static str, String)> {
        self_hosted_workflows()
            .into_iter()
            .map(|(workflow, text)| {
                let report_job = match workflow {
                    "e2e-selfhosted.yml" => "e2e-report",
                    "nightly.yml" => "e2e-report-nightly",
                    other => panic!("no consolidated report job known for {other}"),
                };
                let step = named_step(job_block(&text, report_job), REPORT_CHECK_STEP);
                (workflow, text, report_job, step)
            })
            .collect()
    }

    /// The report artifact a lane's upload step names (`name: e2e-*-report`).
    fn uploaded_report_artifact(workflow: &str, text: &str, job: &str) -> String {
        // A lane with an `sdk_version` matrix (ROCMAI-430) uploads one artifact
        // per leg, each behind its own mutually exclusive `if:` — pair every
        // artifact name with the `if:` directly above it in the same step, then
        // pick the `current` leg to stand for the whole lane, the same way a
        // channel axis below is checked through its `release` leg.
        let mut last_if: Option<&str> = None;
        let mut candidates: Vec<(&str, Option<&str>)> = Vec::new();
        for line in job_block(text, job).lines() {
            let trimmed = line.trim();
            if let Some(cond) = trimmed.strip_prefix("if: ") {
                last_if = Some(cond);
            } else if let Some(name) = trimmed.strip_prefix("name: ")
                && name.starts_with("e2e-")
                && name.ends_with("-report")
            {
                candidates.push((name, last_if));
            }
        }
        let chosen = match candidates.as_slice() {
            [(name, _)] => *name,
            _ => {
                candidates
                    .iter()
                    .find(|(_, cond)| {
                        cond.is_some_and(|c| c.contains("matrix.sdk_version == 'current'"))
                    })
                    .unwrap_or_else(|| {
                        panic!(
                            "{workflow} job `{job}` should upload exactly one `e2e-*-report` \
                             artifact, or one `current`-leg one among an sdk_version matrix, \
                             found {candidates:?}"
                        )
                    })
                    .0
            }
        };
        // The report job has no `matrix` context, so a lane with a channel axis is
        // checked through its `release` leg, which stands for the whole lane.
        chosen.replace("${{ matrix.channel }}", "release")
    }

    #[test]
    fn every_lane_that_runs_is_checked_for_its_report_artifact() {
        // A lane that dies before writing report.json uploads nothing, and the
        // consolidated grid can only render reports that arrived, so the lane
        // drops out with no row and no FAIL — and on a `continue-on-error` lane
        // nothing else is red either. The report job's missing-report check is
        // what catches that, and it only knows a lane through LANE_REPORTS.
        //
        // Derived from the lanes, like its siblings: a new lane fails here until
        // it is added to the map under the exact artifact name its upload step
        // uses — a misspelt name would flag the lane missing on every run, and
        // an absent one would leave it uncovered.
        let steps = report_check_steps();
        for (workflow, text, report_job, step) in &steps {
            let (workflow, report_job) = (*workflow, *report_job);
            let block = job_block(text, report_job);
            let lanes = self_hosted_e2e_jobs(text);
            let expected: Vec<String> = lanes
                .iter()
                .map(|(job, _)| format!("{job}={}", uploaded_report_artifact(workflow, text, job)))
                .collect();
            // Read from the step itself, so a LANE_REPORTS left elsewhere in the
            // job after the step is deleted cannot satisfy this.
            assert_eq!(
                block_scalar_lines(step, "LANE_REPORTS"),
                expected,
                "{workflow} `{report_job}`: LANE_REPORTS must map every self-hosted lane, \
                 in workflow order, to the artifact its upload step names"
            );
            for (job, _) in &lanes {
                assert!(
                    block.lines().any(|line| line.trim() == format!("- {job}")),
                    "{workflow} `{report_job}` must list `{job}` under needs, or the \
                     missing-report check cannot see whether it ran"
                );
            }
            assert!(
                block.lines().any(|line| line.trim() == "actions: read"),
                "{workflow} `{report_job}` needs `actions: read` to list the run's artifacts"
            );
            // `!cancelled()` rather than the implicit `success()`: the check has
            // to run precisely when an earlier step or a lane has failed.
            let step_lines: Vec<&str> = step.lines().map(str::trim).collect();
            for required in [
                "if: ${{ !cancelled() }}",
                "GH_TOKEN: ${{ github.token }}",
                "NEEDS: ${{ toJSON(needs) }}",
            ] {
                assert!(
                    step_lines.contains(&required),
                    "{workflow} `{report_job}`'s missing-report check must carry `{required}`"
                );
            }
        }
        // The script is copied into both workflows; a fix applied to one twin
        // must not silently miss the other.
        let bodies: Vec<String> = steps
            .iter()
            .map(|(_, _, _, step)| {
                dedent(&run_block(step).expect("the check step has a `run: |` block"))
                    .expect("the check step's script is not empty")
            })
            .collect();
        assert_eq!(
            bodies[0], bodies[1],
            "the missing-report check scripts in e2e-selfhosted.yml and nightly.yml have drifted"
        );
    }

    /// Run a report job's missing-report check script under `bash -e`, as a
    /// `run:` step does, with a stub `gh` that lists `uploaded` (or fails when
    /// `uploaded` is `None`) and a `needs` context of `results`.
    #[cfg(unix)]
    fn run_report_check(
        step: &str,
        results: &[(&str, &str)],
        uploaded: Option<&[&str]>,
    ) -> (i32, String, String) {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("temp dir");
        let script = dir.path().join("check.sh");
        let listing = dir.path().join("artifacts.txt");
        let summary = dir.path().join("summary.md");
        let stub = dir.path().join("gh");
        std::fs::write(
            &script,
            dedent(&run_block(step).expect("the check step has a `run: |` block"))
                .expect("the check step's script is not empty"),
        )
        .expect("write script");
        std::fs::write(&summary, "").expect("write summary");
        match uploaded {
            Some(names) => {
                std::fs::write(&listing, names.join("\n") + "\n").expect("write listing");
                std::fs::write(&stub, "#!/bin/sh\ncat \"$LISTING\"\n").expect("write gh stub");
            }
            None => std::fs::write(&stub, "#!/bin/sh\necho 'HTTP 502' >&2\nexit 1\n")
                .expect("write gh stub"),
        }
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755))
            .expect("make the stub executable");

        let needs: serde_json::Map<String, serde_json::Value> = results
            .iter()
            .map(|(job, result)| ((*job).to_owned(), serde_json::json!({ "result": result })))
            .collect();
        let path = format!(
            "{}:{}",
            dir.path().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let out = std::process::Command::new("bash")
            .arg("-e")
            .arg(&script)
            .env("PATH", path)
            .env("LISTING", &listing)
            .env("NEEDS", serde_json::Value::Object(needs).to_string())
            // A `|` block scalar ends in a newline, as GitHub passes it, so the
            // script's read loop sees a trailing empty line it must skip.
            .env(
                "LANE_REPORTS",
                block_scalar_lines(step, "LANE_REPORTS").join("\n") + "\n",
            )
            .env("GITHUB_REPOSITORY", "owner/repo")
            .env("GITHUB_RUN_ID", "1")
            .env("GITHUB_STEP_SUMMARY", &summary)
            .output()
            .expect("running the check script");
        let mut log = String::from_utf8_lossy(&out.stdout).into_owned();
        log.push_str(&String::from_utf8_lossy(&out.stderr));
        (
            out.status.code().expect("check exited with a status code"),
            log,
            std::fs::read_to_string(&summary).expect("read summary"),
        )
    }

    #[test]
    #[cfg(unix)]
    fn missing_report_check_flags_only_lanes_that_ran_without_a_report() {
        // The contract test above pins the check's inputs; this runs its logic,
        // so a check that never fails (or always does) cannot pass for one.
        let has_jq = std::process::Command::new("jq")
            .arg("--version")
            .output()
            .is_ok_and(|out| out.status.success());
        if !has_jq {
            // A skip reads as a pass, so on CI a missing `jq` must fail loudly
            // rather than leave the check's logic silently untested.
            assert!(
                std::env::var_os("CI").is_none(),
                "`jq` is not on PATH, so the missing-report check's logic cannot be \
                 exercised; install it on this runner"
            );
            eprintln!("skipping: no `jq` on PATH");
            return;
        }
        let (.., step) = report_check_steps()
            .into_iter()
            .find(|(workflow, ..)| *workflow == "e2e-selfhosted.yml")
            .expect("the per-PR workflow has a report check");
        let lanes: Vec<(String, String)> = block_scalar_lines(&step, "LANE_REPORTS")
            .iter()
            .map(|line| {
                let (job, artifact) = line.split_once('=').expect("`job=artifact`");
                (job.to_owned(), artifact.to_owned())
            })
            .collect();
        let jobs: Vec<&str> = lanes.iter().map(|(job, _)| job.as_str()).collect();
        let every_artifact: Vec<&str> = lanes.iter().map(|(_, a)| a.as_str()).collect();
        let (victim, victim_artifact) = (&lanes[0].0, &lanes[0].1);
        let all = |result: &'static str| -> Vec<(&str, &str)> {
            jobs.iter().map(|job| (*job, result)).collect()
        };
        let without_victim: Vec<&str> = every_artifact
            .iter()
            .copied()
            .filter(|a| a != victim_artifact)
            .collect();
        let mut with_changes = all("success");
        with_changes.push(("changes", "success"));

        let (code, log, summary) = run_report_check(&step, &with_changes, Some(&every_artifact));
        assert_eq!(
            code, 0,
            "every lane reported, so nothing is missing:\n{log}"
        );
        assert!(
            summary.is_empty(),
            "nothing missing, nothing summarised: {summary}"
        );

        let mut failed = all("success");
        failed[0].1 = "failure";
        let (code, log, summary) = run_report_check(&step, &failed, Some(&without_victim));
        assert_eq!(
            code, 1,
            "a lane that ran and uploaded nothing must fail:\n{log}"
        );
        assert!(
            log.contains(&format!("::error::lane {victim} ran")) && log.contains(victim_artifact),
            "the annotation must name the lane and its artifact:\n{log}"
        );
        assert!(
            summary.contains(victim.as_str()),
            "the summary must name the lane: {summary}"
        );

        // A continue-on-error lane may present as `success`; the missing
        // artifact alone has to be enough.
        let (code, log, _) = run_report_check(&step, &all("success"), Some(&without_victim));
        assert_eq!(
            code, 1,
            "a missing report fails whatever the needs result:\n{log}"
        );

        let mut skipped = all("success");
        skipped[0].1 = "skipped";
        let (code, log, _) = run_report_check(&step, &skipped, Some(&without_victim));
        assert_eq!(
            code, 0,
            "a skipped lane uploads nothing and is not missing:\n{log}"
        );

        // A single-platform dispatch skips most lanes. Skipping one must move
        // on to the next lane, not stop checking: skip a middle lane while the
        // last one is missing.
        assert!(
            lanes.len() >= 3,
            "these cases need a first, middle and last lane"
        );
        let (middle_artifact, (last, last_artifact)) = (&lanes[1].1, lanes.last().unwrap());
        let mut skip_middle = all("success");
        skip_middle[1].1 = "skipped";
        let after_skip: Vec<&str> = every_artifact
            .iter()
            .copied()
            .filter(|a| a != middle_artifact && a != last_artifact)
            .collect();
        let (code, log, _) = run_report_check(&step, &skip_middle, Some(&after_skip));
        assert_eq!(
            code, 1,
            "lanes after a skipped one must still be checked:\n{log}"
        );
        assert!(
            log.contains(&format!("::error::lane {last} ran")),
            "the missing lane after the skipped one must be named:\n{log}"
        );

        // Every missing lane is reported, not just the first one found.
        let two_missing: Vec<&str> = without_victim
            .iter()
            .copied()
            .filter(|a| a != last_artifact)
            .collect();
        let (code, log, summary) = run_report_check(&step, &all("success"), Some(&two_missing));
        assert_eq!(code, 1, "two lanes missing must fail:\n{log}");
        for lane in [victim, last] {
            assert!(
                log.contains(&format!("::error::lane {lane} ran"))
                    && summary.contains(lane.as_str()),
                "every missing lane must be named, including `{lane}`:\n{log}\n{summary}"
            );
        }

        // A lane absent from `needs` has no result to say it was skipped, so
        // it is checked rather than waved through.
        let (code, log, _) = run_report_check(&step, &all("success")[1..], Some(&without_victim));
        assert_eq!(
            code, 1,
            "a lane missing from `needs` must not pass unchecked:\n{log}"
        );
        assert!(
            log.contains("needs result: absent"),
            "the annotation must show the lane had no `needs` entry:\n{log}"
        );

        // `cancelled` is what a lane cut off by its timeout reports: it is
        // checked like any lane that ran, and passes only if it still uploaded.
        let mut timed_out = all("success");
        timed_out[0].1 = "cancelled";
        let (code, log, _) = run_report_check(&step, &timed_out, Some(&without_victim));
        assert_eq!(
            code, 1,
            "a timed-out lane that uploaded nothing is missing:\n{log}"
        );
        let (code, log, _) = run_report_check(&step, &timed_out, Some(&every_artifact));
        assert_eq!(
            code, 0,
            "a timed-out lane that still uploaded is not missing:\n{log}"
        );

        let mut near_miss = without_victim.clone();
        let stale = format!("{victim_artifact}-old");
        near_miss.push(&stale);
        let (code, log, _) = run_report_check(&step, &all("success"), Some(&near_miss));
        assert_eq!(
            code, 1,
            "only the exact artifact name satisfies the check:\n{log}"
        );

        let (code, log, _) = run_report_check(&step, &all("success"), None);
        assert_eq!(code, 1, "an unlistable run must not pass unchecked:\n{log}");
        assert!(
            log.contains("could not list this run's artifacts"),
            "an API failure must say so rather than fail silently:\n{log}"
        );
    }

    /// The `name:` of `step`, a block as `step_block` returns it (its first
    /// line is the step's `- ` item line); empty when the step has none.
    fn step_name(step: &str) -> String {
        step_key(step, "name")
    }

    /// The one-line value of `step`'s own `{key}:`, unquoted and without a
    /// trailing comment; empty when the step has no such key.
    ///
    /// The key is either the item's own first key (`- name: …`) or one of
    /// its sibling keys, two columns in from the dash. Keys any deeper — a
    /// `with:` input called `name`, a `name:` line inside the step's script —
    /// are not the step's.
    fn step_key(step: &str, key: &str) -> String {
        let marker = format!("{key}:");
        let item_indent = step.lines().next().map_or(0, indent_of);
        step.lines()
            .find_map(|l| {
                let key_line = if indent_of(l) == item_indent {
                    l.trim_start().strip_prefix("- ")?
                } else if indent_of(l) == item_indent + 2 {
                    l.trim_start()
                } else {
                    return None;
                };
                key_line.strip_prefix(&marker)
            })
            .map(|value| strip_quotes(strip_comment(value).trim()))
            .unwrap_or_default()
    }

    /// Every `GPU preflight` step block in `text`, in file order.
    fn gpu_preflight_steps(text: &str) -> Vec<String> {
        let lines: Vec<&str> = text.lines().collect();
        lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.trim_start().starts_with("- name: GPU preflight"))
            .map(|(i, _)| step_block(&lines, i))
            .collect()
    }

    /// The whole step whose `- name:` line is `lines[at]`: that line and every
    /// line below it until a non-blank one is indented no deeper than it.
    ///
    /// Blank lines never end a block, however they are indented — a `run: |`
    /// body can contain them — so the blank separator before the next step is
    /// carried along too. `run_block` drops it again.
    fn step_block(lines: &[&str], at: usize) -> String {
        let step_indent = indent_of(lines[at]);
        let mut step = format!("{}\n", lines[at]);
        for body in &lines[at + 1..] {
            if !body.trim().is_empty() && indent_of(body) <= step_indent {
                break;
            }
            step.push_str(body);
            step.push('\n');
        }
        step
    }

    /// The lines of a step's `run: |` block, still indented.
    ///
    /// Trailing blank lines are dropped: they are the separator `step_block`
    /// lets through, not part of the script, and keeping them would make two
    /// identical scripts compare unequal over the spacing between steps.
    fn run_block(step: &str) -> Option<Vec<&str>> {
        let lines: Vec<&str> = step.lines().collect();
        let run_at = lines.iter().position(|l| l.trim() == "run: |")?;
        let run_indent = indent_of(lines[run_at]);
        let mut body: Vec<&str> = lines[run_at + 1..]
            .iter()
            .take_while(|l| l.trim().is_empty() || indent_of(l) > run_indent)
            .copied()
            .collect();
        while body.last().is_some_and(|l| l.trim().is_empty()) {
            body.pop();
        }
        Some(body)
    }

    /// The non-comment lines of a step's `run: |` block, joined.
    ///
    /// A preflight step's body is the only place a lane can still spell its own
    /// floor (the WSL `${…:-8}` default, the PowerShell `else { 8 }`), and a
    /// substring search over the raw body matches those spellings inside a
    /// comment that merely discusses them. `#` opens a comment in both bash and
    /// PowerShell, so one filter serves every shape here.
    fn preflight_body_code(step: &str) -> String {
        run_block(step)
            .map(|body| {
                body.iter()
                    .filter(|l| !l.trim_start().starts_with('#'))
                    .copied()
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    }

    /// A step's `env:` mapping, parsed as KEY AND VALUE rather than searched
    /// for as a substring.
    ///
    /// Both halves matter and both were wrong when this was a `contains`:
    /// `GPU_PREFLIGHT_MIN_FREE_GIB: 16` is a substring of `…: 160`, so a floor
    /// ten times too high satisfied a test asserting the floor; and the search
    /// ran over the whole step, so commenting the key out — `# GPU_PREFLIGHT_
    /// MIN_FREE_GIB: 8`, which leaves the lane running at the script's 16 GiB
    /// default — satisfied it too. That default on an APU lane is verbatim the
    /// thing the assertion message says it prevents.
    ///
    /// Comment and blank lines are skipped, and only scalar `key: value`
    /// entries are returned; our workflows spell step `env:` no other way.
    fn step_env(step: &str) -> BTreeMap<String, String> {
        let lines: Vec<&str> = step.lines().collect();
        let mut env = BTreeMap::new();
        let Some(env_at) = lines.iter().position(|l| l.trim() == "env:") else {
            return env;
        };
        let env_indent = indent_of(lines[env_at]);
        for line in &lines[env_at + 1..] {
            if line.trim().is_empty() {
                continue;
            }
            if indent_of(line) <= env_indent {
                break;
            }
            let entry = line.trim_start();
            if entry.starts_with('#') {
                continue;
            }
            let Some((key, value)) = entry.split_once(':') else {
                continue;
            };
            env.insert(
                key.trim().to_string(),
                strip_quotes(strip_comment(value).trim()),
            );
        }
        env
    }

    /// The floor a step sets through its `env:` mapping, if it sets one.
    fn preflight_floor(step: &str) -> Option<String> {
        step_env(step).get("GPU_PREFLIGHT_MIN_FREE_GIB").cloned()
    }

    /// A step's top-level keys, in file order — `name`, `env`, `run` and
    /// anything a lane has grown beyond them.
    ///
    /// `continue-on-error: true` on one preflight turns a required gate into an
    /// advisory one; `if: false` removes it outright. Neither touches the `run:`
    /// body, so neither is visible to a test that compares bodies.
    fn step_keys(step: &str) -> Vec<String> {
        let lines: Vec<&str> = step.lines().collect();
        let Some(first) = lines.first() else {
            return Vec::new();
        };
        // Past the `- ` that opens the list item: the step's keys all sit at
        // that column, including the `name:` sharing the first line.
        let key_indent = indent_of(first) + 2;
        lines
            .iter()
            .enumerate()
            .filter_map(|(i, line)| {
                let entry = if i == 0 {
                    line.trim_start().strip_prefix("- ")?
                } else if indent_of(line) == key_indent && !line.trim_start().starts_with('#') {
                    line.trim_start()
                } else {
                    return None;
                };
                entry.split_once(':').map(|(key, _)| key.trim().to_string())
            })
            .collect()
    }

    /// Everything in a step EXCEPT its `run:` body: the shape the body travels
    /// in, dedented so two lanes in differently-indented files compare.
    ///
    /// The per-lane floor's VALUE is replaced rather than dropped, because 8 and
    /// 16 are the one difference between these steps that is intended — while
    /// losing the key altogether is not, and still shows up as a difference.
    fn preflight_step_shape(step: &str) -> String {
        let lines: Vec<&str> = step.lines().collect();
        let Some(first) = lines.first() else {
            return String::new();
        };
        let pad = indent_of(first);
        let run_at = lines.iter().position(|l| l.trim() == "run: |");
        let run_indent = run_at.map(|at| indent_of(lines[at]));
        let mut shape = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            // Drop the body, keep the `run: |` key that introduces it: the
            // bodies are compared to each other separately.
            if let (Some(at), Some(run_indent)) = (run_at, run_indent)
                && i > at
                && indent_of(line) > run_indent
            {
                continue;
            }
            let line = if line.len() > pad {
                &line[pad..]
            } else {
                line.trim_start()
            };
            match line.split_once(':') {
                Some((key, _)) if key.trim() == "GPU_PREFLIGHT_MIN_FREE_GIB" => {
                    shape.push(format!("{key}: <per-lane floor>"));
                }
                _ => shape.push(line.to_string()),
            }
        }
        shape.join("\n")
    }

    /// Where two line-oriented texts first differ, phrased for a failure
    /// message and naming which file each side came from; `None` when they
    /// are equal.
    fn first_line_difference(a_name: &str, a: &str, b_name: &str, b: &str) -> Option<String> {
        if a == b {
            return None;
        }
        let a_lines: Vec<&str> = a.lines().collect();
        let b_lines: Vec<&str> = b.lines().collect();
        Some(
            match a_lines.iter().zip(&b_lines).position(|(x, y)| x != y) {
                Some(i) => format!(
                    "first difference at body line {}:\n  {a_name}: {:?}\n  {b_name}: {:?}",
                    i + 1,
                    a_lines[i],
                    b_lines[i]
                ),
                None if a_lines.len() != b_lines.len() => format!(
                    "every shared line matches, but {a_name} has {} lines and {b_name} has {}",
                    a_lines.len(),
                    b_lines.len()
                ),
                None => "the bodies differ only in line endings".to_string(),
            },
        )
    }

    fn dedent(body: &[&str]) -> Option<String> {
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

    /// The POSIX-shell body a preflight step actually runs, dedented.
    ///
    /// Two shapes carry shell: a plain `run: |` step, and the WSL lanes, which
    /// hand a here-string to `Invoke-WslBash.ps1`. A step that is PowerShell end
    /// to end yields `None`; `preflight_powershell_script` carries those.
    fn preflight_shell_script(step: &str) -> Option<String> {
        let body = run_block(step)?;
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
        dedent(&body)
    }

    /// The PowerShell body a preflight step runs, for the steps that are
    /// PowerShell end to end.
    ///
    /// The WSL lanes are excluded deliberately: their PowerShell is a wrapper
    /// around a bash here-string that `preflight_shell_script` already carries.
    /// Without this the Windows-native blocks are invisible to the drift check
    /// below — reproducing, on the test side, the very gap that let one of the
    /// six production blocks sit un-converted.
    fn preflight_powershell_script(step: &str) -> Option<String> {
        if !step.contains("shell: powershell") || step.contains("-Script @'") {
            return None;
        }
        dedent(&run_block(step)?)
    }

    /// The shell options a lane's body actually runs under in CI.
    ///
    /// Driving a lane under the wrong ones tests a shell production never uses,
    /// which is how a script can pass here and abort on a runner.
    #[cfg(unix)]
    #[derive(Clone, Copy)]
    enum PreflightShell {
        /// A plain `run:` step, which GitHub runs as `bash -e {0}` — `-e` but,
        /// unlike an explicit `shell: bash`, no pipefail.
        DefaultRunStep,
        /// The WSL lanes pass their body to `Invoke-WslBash.ps1` *without*
        /// `-PipeFail`, which execs a bare `bash <file>`: no `-e` either.
        WslBareBash,
    }

    /// The shared bash preflight every native Linux lane delegates to (#446,
    /// #294), as the workflows name it.
    const SHARED_PREFLIGHT: &str = ".github/scripts/gpu-preflight.sh";

    /// That script's source.
    fn shared_preflight_script() -> String {
        let p = repo_root().join(SHARED_PREFLIGHT);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
    }

    /// Whether this preflight guards an APU lane (the 8 GiB floor) rather than a
    /// discrete card (16 GiB).
    ///
    /// Three spellings, because the floor reaches the three surviving block
    /// shapes differently: a step-level `env:` key on the lanes that delegate to
    /// the shared script, a `${…:-8}` default in the WSL advisory bodies, and a
    /// PowerShell `else { 8 }` on the Windows-native ones.
    ///
    /// None of the three is a substring search over the raw step any more. The
    /// `env:` one is a parsed lookup compared EXACTLY, and the other two run
    /// over the body with comments stripped — a step whose floor is commented
    /// out is not an APU lane that set 8, it is a lane running the script's
    /// discrete 16 GiB default on hardware that can never clear it.
    fn is_apu_preflight(step: &str) -> bool {
        if let Some(floor) = preflight_floor(step) {
            return floor == "8";
        }
        let body = preflight_body_code(step);
        body.contains("GPU_PREFLIGHT_MIN_FREE_GIB:-8") || body.contains("else { 8 }")
    }

    #[test]
    fn every_apu_preflight_block_measures_the_gtt_pool() {
        // A gfx1151 APU reports its BIOS carveout as VRAM total — 512 MiB on the
        // hosts these lanes run on — while the engine allocates from GTT-backed
        // system RAM. A floor applied to VRAM there can never be cleared, so the
        // lane fails on every healthy host. Six blocks guard that hardware class
        // and one was missed when the other five were converted; this is what
        // makes the next copy-paste fail loudly rather than months later.
        //
        // A block satisfies this either by querying both pools itself or by
        // delegating to the shared script, which is then asserted to query them.
        // Delegation is accepted rather than waived: the one-line lanes carry no
        // query to inspect, and demanding the literal there would have forced
        // the shared script's extraction to leave the text behind as a comment —
        // a string this test would then happily match in a lane that does
        // nothing.
        let mut checked = 0;
        let mut delegating = 0;
        for (workflow, text) in self_hosted_workflows() {
            for step in gpu_preflight_steps(&text)
                .iter()
                .filter(|s| is_apu_preflight(s))
            {
                if step.contains(SHARED_PREFLIGHT) {
                    delegating += 1;
                } else {
                    assert!(
                        step.contains("showmeminfo vram gtt"),
                        "{workflow}: an APU-class GPU preflight neither queries both pools \
                         nor delegates to {SHARED_PREFLIGHT}; on a small-carveout gfx1151 \
                         host it can never clear its own floor"
                    );
                }
                checked += 1;
            }
        }
        assert!(
            delegating >= 2,
            "no APU lane delegates to {SHARED_PREFLIGHT} any more ({delegating} found) — if \
             that was deliberate this test is now checking a shape nothing uses"
        );
        assert!(
            shared_preflight_script().contains("showmeminfo vram gtt"),
            "{SHARED_PREFLIGHT} no longer queries the GTT pool, so every lane delegating to \
             it fails on a small-carveout gfx1151 host"
        );
        assert!(
            checked >= 6,
            "expected at least six APU-class preflight blocks, found {checked} — if a lane \
             was removed update this floor, otherwise the extractor has stopped matching"
        );
    }

    #[test]
    fn every_native_linux_preflight_delegates_to_the_shared_script() {
        // What the deduplication (#294) actually bought, and the only thing
        // keeping the next #446 from landing eight times: the eight native Linux
        // preflights are now one script plus a floor. The drift test below can no
        // longer catch a block left behind, because a block left behind would
        // carry its own body and simply not be compared against anything.
        //
        // Derived from the lanes rather than listed, so a new one is covered the
        // day it lands. The two WSL advisory bodies are excluded by name: they
        // warn where these fail, which is a different contract, not a copy. The
        // two Windows-native blocks are PowerShell and have no bash to run.
        let mut delegating = 0;
        for (workflow, text) in self_hosted_workflows() {
            for step in gpu_preflight_steps(&text) {
                if preflight_powershell_script(&step).is_some() || step.contains("-Script @'") {
                    continue;
                }
                assert!(
                    step.contains(SHARED_PREFLIGHT),
                    "{workflow} has a native Linux GPU preflight with its own inline body \
                     instead of delegating to {SHARED_PREFLIGHT}; that is the duplication \
                     #294 removed, and a fix to the gate will now miss this lane"
                );
                // Compared as a PARSED VALUE, not found as a substring. `16` is
                // a substring of `160`, and a search over the whole step is
                // satisfied by `# GPU_PREFLIGHT_MIN_FREE_GIB: 8` — a commented
                // key, which leaves the lane on the script's 16 GiB default.
                // Both mutations passed this assertion while producing exactly
                // the state its message says cannot happen.
                let floor = preflight_floor(&step);
                assert!(
                    matches!(floor.as_deref(), Some("8" | "16")),
                    "{workflow} delegates to {SHARED_PREFLIGHT} without naming its own \
                     floor in `env:` (found {floor:?}, expected the APU's 8 or a \
                     discrete card's 16); the lane would then silently inherit the \
                     script's default rather than the value its hardware was sized for"
                );
                delegating += 1;
            }
        }
        assert_eq!(
            delegating, 8,
            "expected the eight native Linux preflight lanes to delegate to \
             {SHARED_PREFLIGHT} — if a lane was added or removed, update this count"
        );
    }

    #[test]
    fn every_native_linux_preflight_delegates_in_the_same_shape() {
        // The test above asks whether each lane MENTIONS the shared script; this
        // asks whether that is all it does. The gap between the two is exactly
        // where re-inlining drift grows back: a lane that keeps the delegation
        // and adds a line of its own above it — one more `pkill -f`, a lane-local
        // retry, a second floor — satisfies `step.contains(SHARED_PREFLIGHT)`
        // and the floor check and leaves that test green, while the eight bodies
        // are no longer one body. That is how the original six inline copies
        // diverged from their PowerShell mirrors: nothing compared them to each
        // other, only to a substring.
        //
        // So compare the bodies to EACH OTHER, the way `apu_preflight_twins_do_
        // not_drift` and `both_wsl_lanes_settle_the_clock_before_running_the_
        // suite` pin their pairs, and then pin the shape they agree on: a single
        // `bash <script>` line and nothing else. Equality alone would be
        // satisfied by eight identically re-inlined copies; the shape assertion
        // alone would not notice seven of eight.
        //
        // The per-lane floor's VALUE is deliberately not part of this
        // comparison — 8 and 16 are the one intended difference — but
        // everything else outside the body IS. Comparing only the `run:` body
        // leaves the whole rest of the step free to drift, and the drift that
        // fits there is worse than the drift that fits in the body: measured,
        // each of `continue-on-error: true`, `if: false` and a second
        // `GPU_PREFLIGHT_CEILING_SECS: 1` env key on ONE lane left the
        // body-only comparison green. The first turns a required gate into an
        // advisory one — the gate still runs, still goes red, and the job
        // carries on — which is indistinguishable from deleting it.
        let mut bodies: Vec<(String, String)> = Vec::new();
        let mut shapes: Vec<(String, String)> = Vec::new();
        for (workflow, text) in self_hosted_workflows() {
            for (i, step) in gpu_preflight_steps(&text).iter().enumerate() {
                if preflight_powershell_script(step).is_some() || step.contains("-Script @'") {
                    continue;
                }
                let name = format!("{workflow} GPU preflight #{i}");
                let body = preflight_shell_script(step)
                    .unwrap_or_else(|| panic!("{name} carries no `run: |` body"));
                // Pinned per lane as well as compared across them: eight lanes
                // that ALL grew `continue-on-error: true` would still agree
                // with each other.
                assert_eq!(
                    step_keys(step),
                    ["name", "env", "run"],
                    "{name} is no longer just a named, floor-carrying delegation: a \
                     preflight that also sets `if:` or `continue-on-error:` is a gate \
                     the job can skip or survive, which is the one failure mode a \
                     required gate has"
                );
                assert_eq!(
                    step_env(step).keys().collect::<Vec<_>>(),
                    ["GPU_PREFLIGHT_MIN_FREE_GIB"],
                    "{name} sets step environment beyond its floor; a lane-local \
                     ceiling or marker is a second gate configured where the shared \
                     script cannot see it, and the next fix to it lands eight times again"
                );
                shapes.push((name.clone(), preflight_step_shape(step)));
                bodies.push((name, body));
            }
        }
        assert_eq!(
            bodies.len(),
            8,
            "expected eight native Linux preflight bodies to compare, found {} — \
             if a lane was added or removed, update this count",
            bodies.len()
        );
        let (first_name, first) = bodies
            .first()
            .expect("at least one native preflight")
            .clone();
        for (name, body) in &bodies[1..] {
            if let Some(diff) = first_line_difference(&first_name, &first, name, body) {
                panic!(
                    "the native Linux GPU preflights no longer invoke \
                     {SHARED_PREFLIGHT} identically — a lane has grown a body of its \
                     own again, which is the duplication #294 removed and the drift \
                     #446 was caused by; {diff}"
                );
            }
        }
        assert_eq!(
            first.trim_end(),
            format!("bash {SHARED_PREFLIGHT}"),
            "the native Linux GPU preflight bodies agree with each other but are no \
             longer a lone delegation to {SHARED_PREFLIGHT}; whatever they now also \
             do is a second gate living in the workflows, where the next fix to it \
             has to land eight times again"
        );
        let (first_name, first) = shapes
            .first()
            .expect("at least one native preflight")
            .clone();
        for (name, shape) in &shapes[1..] {
            if let Some(diff) = first_line_difference(&first_name, &first, name, shape) {
                panic!(
                    "the native Linux GPU preflights agree on their bodies but no longer \
                     on the step AROUND the body — one lane runs {SHARED_PREFLIGHT} under \
                     conditions, error handling or environment the other seven do not, so \
                     the eight are one gate in name only; {diff}"
                );
            }
        }
    }

    #[test]
    fn the_shared_preflight_asserts_no_leaked_engine_process_survives() {
        // #446: the free-memory floor is a fixed 8/16 GiB against pools of ~62
        // and 192 GiB, so it only trips near exhaustion — a leftover
        // llama-server holding 10-20 GiB clears every lane. The process
        // assertion is what actually catches that, and it is one line in one
        // file: deleting it leaves every other test here green, because they all
        // drive memory readings.
        //
        // The shape is pinned here; `apu_preflight_gates_on_the_pool_the_engine
        // _uses` drives it against a reclaim stub that reports survivors, which
        // is what proves it can fail.
        //
        // MATCHED ON THE CALL, not on the flag's name anywhere in the file, and
        // this is the second time round on exactly that distinction. Filtering
        // comments was the first fix: the script's header explains the mode by
        // name, so a whole-file `contains("--assert-clear")` stayed satisfied
        // after the call was gone. Then the timeout fix added a NON-comment
        // `echo "::error::… (reclaim-gpu.sh --assert-clear) did not finish…"`,
        // which contains the flag too — and the filtered search could not fail
        // again. Prose about a call is not a call; the only thing worth
        // asserting is the invocation, so that is what is matched.
        let script = shared_preflight_script();
        let code_lines: Vec<&str> = script
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .collect();
        let calls: Vec<&str> = code_lines
            .iter()
            .map(|l| l.trim())
            .filter(|l| l.contains(r#"bash "${RECLAIM_SCRIPT}" --assert-clear"#))
            .collect();
        assert!(
            !calls.is_empty(),
            "{SHARED_PREFLIGHT} no longer RUNS `reclaim-gpu.sh --assert-clear` (it may still \
             describe it, and its own error message quotes the flag); the gate is back to a \
             fixed free-memory floor, which on these pools only trips near exhaustion (#446)"
        );
        let code: String = code_lines.join("\n");
        // The floor is a BACKSTOP, not a thing the process check replaced: it is
        // the only one of the two that can see a non-E2E tenant exhausting a
        // shared host. Losing it silently would be the opposite half of #446.
        assert!(
            code.contains("never dropped below the floor"),
            "{SHARED_PREFLIGHT} no longer applies a free-memory floor; the process \
             assertion cannot see a non-E2E tenant exhausting a shared host (#446)"
        );
        // The gate must be BOUNDED, like every rocm-smi call in this script. Its
        // scan reads /proc/<pid>/cmdline for every process, and a holder wedged
        // in uninterruptible sleep can block that read — the same condition the
        // gate fires on, so the hang risk is correlated with the detection.
        // Unbounded, the one step added to stop a job hanging to its cap is the
        // step that hangs to it. `preflight_bounds_the_leaked_process_scan`
        // drives this for real; this names the loss immediately rather than by
        // a case that takes a minute to go red.
        assert!(
            calls.iter().all(|l| l.starts_with("timeout 15 bash")),
            "{SHARED_PREFLIGHT} does not invoke `reclaim-gpu.sh --assert-clear` as a plain, \
             `timeout 15`-bounded command — it is either unbounded, in which case a stalled \
             /proc scan rides to the job's own cap instead of failing fast, or guarded by \
             something that can skip it. Found: {calls:?}"
        );
        // …and the bound must FAIL the gate. `timeout` reports expiry as 124,
        // and a scan that could not finish is not evidence of a clear host.
        // Without this, `|| true`-ing the call away would still satisfy the
        // check above while turning every hang into a silent pass.
        assert!(
            code.contains(r#""${assert_rc}" -eq 124"#),
            "{SHARED_PREFLIGHT} bounds the leaked-process scan but does not handle \
             `timeout`'s expiry code (124) distinctly; a scan that never finished must \
             fail the gate with its own reason, not pass or be reported as a survivor"
        );
        // …and so must every OTHER non-zero code. 1 is the only one that means
        // the scan ran and named somebody; 127 (no such script) and 2 (a flag
        // reclaim-gpu.sh does not dispatch — what deleting its `--assert-clear`
        // arm produces) are not survivors and must not be reported as one.
        // `preflight_distinguishes_a_scan_that_never_ran_from_a_survivor` drives
        // both for real.
        assert!(
            code.contains(r#""${assert_rc}" -eq 1"#),
            "{SHARED_PREFLIGHT} treats every non-124 failure of the leaked-process scan as \
             a surviving process; a missing script or an undispatched flag then fails the \
             lane with 'a leaked E2E engine process is still holding it (listed above)' \
             and nothing listed above"
        );
    }

    /// Both WSL2 lanes must settle the clock before anything times a scenario.
    ///
    /// cucumber measures scenario durations by subtracting `SystemTime` stamps, so
    /// a guest whose clock steps backwards mid-run kills the suite (#455 makes that
    /// survivable; this keeps the durations real). The hazard is invisible: the
    /// step's absence breaks nothing that any other test here observes, and on the
    /// nightly lane job-level `continue-on-error: true` means a regression would
    /// not even turn the run red.
    ///
    /// Steps are matched as WHOLE LINES, not found as substrings of the job: the
    /// surrounding comments discuss the clock at length, and a comment line that
    /// happened to embed the step's text would otherwise keep this satisfied after
    /// the step itself was deleted — exactly the failure this pins.
    ///
    /// The two copies' `run:` bodies are also compared byte for byte, after
    /// dedenting. They are kept in sync by hand until the lanes are deduplicated
    /// (#294), and presence and order say nothing about what each copy runs —
    /// the same drift `apu_preflight_twins_do_not_drift` once caught too late.
    /// Equality pins only that the two lanes agree, not that either body still
    /// does the work: both gutted to an identical `echo hi` would pass.
    #[test]
    fn both_wsl_lanes_settle_the_clock_before_running_the_suite() {
        const STEP: &str = "      - name: Settle the clock before anything times a scenario";
        let mut bodies = Vec::new();
        for (workflow, job, suite_step) in [
            (
                "e2e-selfhosted.yml",
                "  e2e-wsl:",
                "      - name: Run E2E tests on Strix Halo WSL2",
            ),
            (
                "nightly.yml",
                "  e2e-wsl-nightly:",
                "      - name: Run E2E tests on Strix Halo WSL2 (incl. nightly-only)",
            ),
        ] {
            let job_block = nested_block(&read_workflow(workflow), job);
            let lines: Vec<&str> = job_block.lines().collect();
            let settle = lines.iter().position(|l| *l == STEP).unwrap_or_else(|| {
                panic!(
                    "{workflow}'s `{job}` has no clock-settling step — a fresh WSL2 guest \
                     steps its clock backwards mid-run and cucumber subtracts SystemTime \
                     stamps, so the durations that lane reports become fiction"
                )
            });
            let suite = lines
                .iter()
                .position(|l| *l == suite_step)
                .unwrap_or_else(|| panic!("{workflow}'s `{job}` no longer runs `{suite_step}`"));
            assert!(
                settle < suite,
                "{workflow}'s `{job}` settles the clock AFTER starting the suite — the \
                 correction has to land while nothing is being timed"
            );
            let step = step_block(&lines, settle);
            let body = run_block(&step)
                .and_then(|block| dedent(&block))
                .unwrap_or_else(|| panic!("{workflow}'s clock-settling step has no `run: |` body"));
            bodies.push((workflow, body));
        }
        let [(first, first_body), (second, second_body)] = bodies.as_slice() else {
            unreachable!("exactly two lanes are checked above");
        };
        if let Some(difference) = first_line_difference(first, first_body, second, second_body) {
            panic!(
                "the clock-settling step has drifted between {first} and {second} — the \
                 copies are kept in sync by hand until the lanes are deduplicated (#294); \
                 {difference}"
            );
        }
    }

    #[test]
    fn apu_preflight_twins_do_not_drift() {
        // The nightly lanes are copies of their per-PR twins. One was left on the
        // pre-GTT script while the other five were converted and nothing failed,
        // because every contract test here asserts step names, env keys and
        // labels — never the script body.
        //
        // What this still buys after #446 extracted the native bodies into
        // `.github/scripts/gpu-preflight.sh`: the PowerShell pair, which is
        // genuinely two hand-maintained copies, and the WSL advisory pair, which
        // is another. The native pair now compares two identical one-line
        // delegations, so equality there is trivially true and proves nothing —
        // `every_native_linux_preflight_delegates_to_the_shared_script` is what
        // carries that case, by asserting the delegation rather than the
        // agreement.
        type Extract = fn(&str) -> Option<String>;
        let scripts = |name: &str, extract: Extract| -> Vec<String> {
            gpu_preflight_steps(&read_workflow(name))
                .iter()
                .filter(|s| is_apu_preflight(s))
                .filter_map(|s| extract(s))
                .collect()
        };
        // Both languages, or the check reproduces the bug it exists to catch:
        // the Windows blocks are PowerShell end to end, and an edit landing in
        // only one of the two files would otherwise go unnoticed.
        for (language, extract, want) in [
            ("shell", preflight_shell_script as Extract, 2),
            ("PowerShell", preflight_powershell_script as Extract, 1),
        ] {
            let per_pr = scripts("e2e-selfhosted.yml", extract);
            let nightly = scripts("nightly.yml", extract);
            assert_eq!(
                per_pr.len(),
                want,
                "expected {want} {language} APU preflight block(s) in e2e-selfhosted.yml"
            );
            assert_eq!(
                per_pr.len(),
                nightly.len(),
                "{language} APU preflight count differs"
            );
            for (i, (a, b)) in per_pr.iter().zip(nightly.iter()).enumerate() {
                if let Some(difference) =
                    first_line_difference("e2e-selfhosted.yml", a, "nightly.yml", b)
                {
                    panic!(
                        "{language} APU preflight script #{i} has drifted between \
                         e2e-selfhosted.yml and nightly.yml; {difference}"
                    );
                }
            }
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
    /// uses: both pools at once, GTT alone, VRAM alone.
    ///
    /// `REJECT_COMBINED` fails the two-pool form the way an older build does —
    /// the case that must still reach GTT through the single-pool query.
    /// `FAIL_GTT` fails the GTT-alone query, which has to stay distinguishable
    /// from a host that answers and simply has no GTT pool (hence the `|| true`
    /// on the `grep`: an absent pool is an empty answer, not a failed call).
    /// `GOOD_CALLS` degrades the tool after N calls, for the polls that have to
    /// survive a reading going away mid-wait.
    #[cfg(unix)]
    const SMI_STUB: &str = r#"#!/bin/sh
n=$(cat "$COUNTER" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$COUNTER"
if [ -n "${GOOD_CALLS:-}" ] && [ "$n" -gt "$GOOD_CALLS" ]; then exit 1; fi
case "$*" in
  *vram*gtt*)
    [ "${REJECT_COMBINED:-0}" = 1 ] && exit 1
    cat "$FIXTURE" ;;
  *gtt*)
    [ "${FAIL_GTT:-0}" = 1 ] && exit 1
    grep GTT "$FIXTURE" || true ;;
  *)
    grep -v GTT "$FIXTURE" || true ;;
esac
"#;

    /// Stands in for `scripts/reclaim-gpu.sh`, which a native lane runs with
    /// `--report-holders` when it gives up, and with `--assert-clear` when the
    /// memory floor is satisfied. Without it the call resolves against whatever
    /// directory the test runs from, fails, and neither path is exercised.
    ///
    /// `RECLAIM_ASSERT_FAILS` makes `--assert-clear` report a survivor, the way
    /// the real script does when a leaked engine process outlived the reclaim.
    /// A gate that cannot be driven into its failing state is not tested by any
    /// number of passing cases: without this the preflight's process assertion
    /// could be deleted outright with every case below still green.
    ///
    /// `RECLAIM_ASSERT_HANGS` makes the scan never return, which is the case the
    /// `timeout` bound on that call exists for: `select_leaked` reads
    /// /proc/<pid>/cmdline for every process, and a holder wedged in
    /// uninterruptible sleep can block that read — the very state this gate
    /// fires on.
    ///
    /// `exec`, not a plain `sleep`: the bound's SIGTERM must land on the sleep
    /// itself. Forked, it would be orphaned still holding the captured stdout
    /// pipe, and the harness would block reading it for the full 60s no matter
    /// when the preflight exited — timing the stub instead of the bound.
    #[cfg(unix)]
    const RECLAIM_STUB: &str = r#"#!/bin/sh
echo "reclaim-gpu.sh stub called: $*"
if [ "$1" = "--report-holders" ] && [ "${RECLAIM_HOLDERS_HANGS:-0}" = 1 ]; then
  exec sleep 60
fi
if [ "$1" = "--assert-clear" ] && [ "${RECLAIM_ASSERT_HANGS:-0}" = 1 ]; then
  exec sleep 60
fi
if [ "$1" = "--assert-clear" ] && [ "${RECLAIM_ASSERT_RC:-0}" != 0 ]; then
  exit "${RECLAIM_ASSERT_RC}"
fi
if [ "$1" = "--assert-clear" ] && [ "${RECLAIM_ASSERT_FAILS:-0}" = 1 ]; then
  echo "assert-clear: leaked E2E engine process(es) still holding the GPU:" >&2
  echo "  pid=4242 cmd=/e2e-prewarm/bin/llama-server" >&2
  exit 1
fi
"#;

    /// What a lane logs when it reached the step that names what holds the GPU.
    #[cfg(unix)]
    const REPORTED_HOLDERS: &str = "reclaim-gpu.sh stub called: --report-holders";

    /// A preflight's exit code, or a panic naming the signal that killed it:
    /// `code()` is `None` exactly when there is no code to report.
    #[cfg(unix)]
    fn exit_code(status: std::process::ExitStatus, what: &str) -> i32 {
        use std::os::unix::process::ExitStatusExt;

        status.code().unwrap_or_else(|| {
            panic!(
                "{what} exited with no status code: killed by signal {:?}",
                status.signal()
            )
        })
    }

    /// Run a preflight script against fixture `rocm-smi` output, returning its
    /// exit code and everything it logged. `rocm-smi` is stubbed on `PATH`;
    /// nothing touches a real GPU.
    ///
    /// `shell` decides how the body is invoked, so each lane is driven the way
    /// CI drives it rather than under whichever options happen to be stricter.
    #[cfg(unix)]
    fn run_preflight(
        script: &str,
        shell: PreflightShell,
        fixture: &str,
        env: &[(&str, &str)],
    ) -> (i32, String) {
        run_preflight_with_reclaim(script, shell, fixture, env, RECLAIM_STUB)
    }

    /// As `run_preflight`, but with the `scripts/reclaim-gpu.sh` the preflight
    /// finds supplied by the caller.
    ///
    /// Every case but one plants `RECLAIM_STUB`, which is what makes the gate's
    /// outcomes drivable. `preflight_runs_the_real_reclaim_assert_clear` plants
    /// the REAL script instead, because a stub cannot show that the two agree
    /// about how the mode is spelled, what it exits, or that the dispatch arm
    /// for it exists at all.
    #[cfg(unix)]
    fn run_preflight_with_reclaim(
        script: &str,
        shell: PreflightShell,
        fixture: &str,
        env: &[(&str, &str)],
        reclaim: &str,
    ) -> (i32, String) {
        use rocm_core::fake_executable::write_fake_executable;

        let dir = tempfile::tempdir().expect("temp dir");
        let fixture_path = dir.path().join("smi.txt");
        let script_path = dir.path().join("preflight.sh");
        let stub = dir.path().join("rocm-smi");
        std::fs::write(&fixture_path, fixture).expect("write fixture");
        std::fs::write(&script_path, script).expect("write script");
        write_fake_executable(&stub, SMI_STUB)
            .expect("plant the fake rocm-smi for the bash/sh preflight");
        // The lanes name `scripts/reclaim-gpu.sh` relative to the checkout, so
        // run from a directory that has one.
        std::fs::create_dir(dir.path().join("scripts")).expect("scripts dir");
        std::fs::write(dir.path().join("scripts/reclaim-gpu.sh"), reclaim)
            .expect("write reclaim-gpu.sh");

        let path = format!(
            "{}:{}",
            dir.path().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let mut cmd = std::process::Command::new("bash");
        if matches!(shell, PreflightShell::DefaultRunStep) {
            cmd.arg("-e");
        }
        let out = cmd
            .arg(&script_path)
            .current_dir(dir.path())
            .env("PATH", path)
            .env("FIXTURE", &fixture_path)
            .env("COUNTER", dir.path().join("calls"))
            // One poll, then the script's own 5s backoff ends the loop. Cases
            // that need a second poll raise this.
            .env("GPU_PREFLIGHT_CEILING_SECS", "1")
            // The APU floor, which the two advisory bodies default to and the
            // shared script does not: it defaults to the discrete 16 and the APU
            // lanes set this. Supplied here so all three are driven at the same
            // floor and a case's expectations mean one thing; a case needing the
            // discrete floor overrides it through `env`.
            .env("GPU_PREFLIGHT_MIN_FREE_GIB", "8")
            .envs(env.iter().copied())
            .output()
            .expect("running the preflight script");
        let mut log = String::from_utf8_lossy(&out.stdout).into_owned();
        log.push_str(&String::from_utf8_lossy(&out.stderr));
        (exit_code(out.status, "preflight"), log)
    }

    /// The two hard-failing shell bodies an APU lane can run: the shared script
    /// every native Linux lane delegates to, and `e2e-selfhosted.yml`'s advisory
    /// WSL body.
    ///
    /// The native side is read from the FILE rather than extracted from the
    /// workflow, because since #446 the step's own body is one line that runs
    /// it. Driving that line from a scratch tree would test the delegation and
    /// nothing else;
    /// `every_native_linux_preflight_delegates_to_the_shared_script` pins the
    /// delegation, and this pins what it delegates to.
    #[cfg(unix)]
    fn apu_lanes() -> (String, String) {
        let steps = gpu_preflight_steps(&read_workflow("e2e-selfhosted.yml"));
        let native = shared_preflight_script();
        let advisory = steps
            .iter()
            .filter(|s| s.contains("advisory"))
            .find_map(|s| preflight_shell_script(s))
            .expect("advisory APU preflight shell script");
        (native, advisory)
    }

    /// The Windows-native APU preflight body, which is PowerShell end to end.
    #[cfg(unix)]
    fn windows_apu_lane() -> String {
        gpu_preflight_steps(&read_workflow("e2e-selfhosted.yml"))
            .iter()
            .filter(|s| is_apu_preflight(s))
            .find_map(|s| preflight_powershell_script(s))
            .expect("Windows APU preflight PowerShell script")
    }

    /// Whether GNU `timeout` — which every shell lane wraps `rocm-smi` in — is
    /// available to drive these scripts at all.
    #[cfg(unix)]
    fn has_timeout() -> bool {
        std::process::Command::new("timeout")
            .arg("--version")
            .output()
            .is_ok()
    }

    #[cfg(unix)]
    fn has_pwsh() -> bool {
        std::process::Command::new("pwsh")
            .arg("--version")
            .output()
            .is_ok()
    }

    /// Run the Windows preflight body under `pwsh`, stubbing `rocm-smi` on
    /// `PATH` exactly as the shell lanes do.
    ///
    /// Unix-only because the stub is a `/bin/sh` script; on Windows CI this
    /// block is still covered by the drift and shape checks. `pwsh` ships on
    /// GitHub's Ubuntu images, so this normally runs rather than skips.
    #[cfg(unix)]
    fn run_preflight_pwsh(script: &str, fixture: &str, env: &[(&str, &str)]) -> (i32, String) {
        use rocm_core::fake_executable::write_fake_executable;

        let dir = tempfile::tempdir().expect("temp dir");
        let fixture_path = dir.path().join("smi.txt");
        let script_path = dir.path().join("preflight.ps1");
        let stub = dir.path().join("rocm-smi");
        std::fs::write(&fixture_path, fixture).expect("write fixture");
        std::fs::write(&script_path, script).expect("write script");
        write_fake_executable(&stub, SMI_STUB)
            .expect("plant the fake rocm-smi for the pwsh preflight");

        let path = format!(
            "{}:{}",
            dir.path().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let out = std::process::Command::new("pwsh")
            .args(["-NoProfile", "-File"])
            .arg(&script_path)
            .env("PATH", path)
            .env("FIXTURE", &fixture_path)
            .env("COUNTER", dir.path().join("calls"))
            .env("GPU_PREFLIGHT_CEILING_SECS", "1")
            .envs(env.iter().copied())
            .output()
            .expect("running the preflight script under pwsh");
        let mut log = String::from_utf8_lossy(&out.stdout).into_owned();
        log.push_str(&String::from_utf8_lossy(&out.stderr));
        (exit_code(out.status, "pwsh preflight"), log)
    }

    /// One host reading, and what each lane must make of it.
    #[cfg(unix)]
    struct GateCase {
        label: &'static str,
        fixture: String,
        env: Vec<(&'static str, &'static str)>,
        /// Exit code, plus phrases the log must carry, for the native lane…
        native: (i32, &'static [&'static str]),
        /// …and for the advisory one.
        advisory: (i32, &'static [&'static str]),
        /// The Windows-native block's verdict, when it differs from `native`.
        ///
        /// `None` means "the same as `native`", which is the usual case: it
        /// guards the same hardware class with the same floor. It is set only
        /// where the two blocks genuinely diverge — the process assertion #446
        /// added, which the PowerShell copy has no bash to run. Stated as an
        /// expectation rather than skipped, so the gap is pinned: if that block
        /// ever grows an equivalent, this case fails and names the reason.
        windows: Option<(i32, &'static [&'static str])>,
    }

    #[cfg(unix)]
    fn assert_gate(lane: &str, case: &str, got: (i32, String), want: (i32, &[&str])) {
        let (code, log) = got;
        assert_eq!(code, want.0, "{lane} preflight, {case} — logged:\n{log}");
        for phrase in want.1 {
            assert!(
                log.contains(phrase),
                "{lane} preflight, {case}: expected {phrase:?} in the log, got:\n{log}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn apu_preflight_gates_on_the_pool_the_engine_uses() {
        const GIB: u64 = 1024 * 1024 * 1024;
        const MIB: u64 = 1024 * 1024;

        // The shape tests above cannot tell a working gate from a broken one:
        // they all pass with the production logic reverted. This drives the real
        // script against fixture tool output instead, which is what distinguishes
        // "measures the right pool" from "passes whenever any pool looks free".
        //
        // Each case pins the reason as well as the exit code. Several branches
        // here exist ONLY to produce an honest reason: deleting the "no GTT
        // pool" bail leaves every exit code untouched and swaps the message back
        // to the false "a serve is likely still holding the GPU" this change
        // exists to stop — invisible to an exit-code-only assertion.
        //
        // All three hard-failing lanes are driven, the Windows one included: it
        // is a third copy of the same logic in another language, and the bug
        // that prompted this work was one copy left behind.
        if !has_timeout() {
            eprintln!("skipping: no `timeout` on PATH (GNU coreutils)");
            return;
        }
        let windows = has_pwsh().then(windows_apu_lane);
        if windows.is_none() {
            eprintln!("skipping the Windows lane: no `pwsh` on PATH");
        }

        let cases = vec![
            GateCase {
                label: "carveout with a free aperture",
                fixture: smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB))),
                env: vec![],
                native: (0, &["GPU ready: 61 GiB GTT free"]),
                advisory: (0, &["GPU ready: 61 GiB GTT free"]),
                windows: None,
            },
            // An older rocm-smi rejects `--showmeminfo vram gtt` outright. The
            // single-pool fallback carries no GTT rows, so without a dedicated
            // GTT query this healthy APU would fail as "no GTT pool" — the very
            // false low-memory report this gate exists to stop producing. The
            // separate-query line must appear too: the pool line printed earlier
            // necessarily said `GTT n/a/n/a`, and a log that stops there
            // contradicts the verdict.
            GateCase {
                label: "carveout where rocm-smi rejects the two-pool query",
                fixture: smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB))),
                env: vec![("REJECT_COMBINED", "1")],
                native: (
                    0,
                    &[
                        "rocm-smi GTT pool (separate query):",
                        "GPU ready: 61 GiB GTT free",
                    ],
                ),
                advisory: (
                    0,
                    &[
                        "rocm-smi GTT pool (separate query):",
                        "GPU ready: 61 GiB GTT free",
                    ],
                ),
                windows: None,
            },
            GateCase {
                label: "carveout with the aperture pinned by a leftover serve",
                fixture: smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, 61 * GIB))),
                env: vec![],
                native: (1, &["GTT never dropped below the floor"]),
                advisory: (1, &["GTT never dropped below the floor"]),
                windows: None,
            },
            GateCase {
                label: "discrete card with VRAM free",
                fixture: smi_fixture(192 * GIB, 10 * GIB, None),
                env: vec![],
                native: (0, &["GPU ready: 182 GiB VRAM free"]),
                advisory: (0, &["GPU ready: 182 GiB VRAM free"]),
                windows: None,
            },
            GateCase {
                label: "discrete card with VRAM held",
                fixture: smi_fixture(192 * GIB, 191 * GIB, None),
                env: vec![],
                native: (1, &["VRAM never dropped below the floor"]),
                advisory: (1, &["VRAM never dropped below the floor"]),
                windows: None,
            },
            // The simplification #446's extraction rests on: ONE script serves
            // both the discrete and the APU lanes because the APU body is a
            // strict superset — a discrete card's VRAM total is far above any
            // floor, so the carveout branch is never entered, and the two-pool
            // query it opens with degrades to the single-pool call the discrete
            // lanes always made. The two cases above cover a rocm-smi that
            // ACCEPTS the two-pool form on a discrete card; this covers one that
            // rejects it, which is the arm that has to fall back. Without it,
            // the degradation is only ever exercised on a carveout fixture and
            // an fallback that mishandled a full-size pool would ship green.
            GateCase {
                label: "discrete card where rocm-smi rejects the two-pool query",
                fixture: smi_fixture(192 * GIB, 10 * GIB, None),
                env: vec![("REJECT_COMBINED", "1")],
                native: (0, &["GPU ready: 182 GiB VRAM free"]),
                advisory: (0, &["GPU ready: 182 GiB VRAM free"]),
                windows: None,
            },
            // The discrete lanes' 16 GiB floor, driven across its own boundary
            // rather than only asserted as a string in the workflow YAML.
            //
            // Every other discrete fixture here clears or misses the floor by
            // well over a hundred GiB, so the NUMBER never decides anything:
            // change the lanes' `GPU_PREFLIGHT_MIN_FREE_GIB: 16` to 8, or to 64,
            // and all of them keep their verdicts. This pair is the only place
            // the value is load-bearing — at 15 GiB free the lane must fail and
            // at 8 it would pass, so a floor that quietly shrank is caught here
            // and nowhere else.
            //
            // Run at the DISCRETE floor, which `run_preflight` does not default
            // to: it supplies the APU's 8 so the three lanes are driven alike,
            // and a case wanting 16 overrides it, as here.
            GateCase {
                label: "discrete card one GiB under the 16 GiB floor",
                fixture: smi_fixture(192 * GIB, 177 * GIB, None),
                env: vec![("GPU_PREFLIGHT_MIN_FREE_GIB", "16")],
                native: (
                    1,
                    &["VRAM never dropped below the floor: only 15 GiB free (< 16 GiB)"],
                ),
                advisory: (
                    1,
                    &[
                        "waiting: 15 GiB VRAM free (< 16 GiB)",
                        "VRAM never dropped below the floor",
                    ],
                ),
                windows: None,
            },
            // The other side of the same boundary, one GiB up. Pins the
            // comparison as `>=` rather than `>`: a card holding exactly the
            // floor satisfies it. Off by that one step the lanes would burn the
            // full ceiling and fail a GPU that is, by their own stated rule,
            // ready — and no fixture above would notice, because none of them
            // land on the number.
            GateCase {
                label: "discrete card exactly at the 16 GiB floor",
                fixture: smi_fixture(192 * GIB, 176 * GIB, None),
                env: vec![("GPU_PREFLIGHT_MIN_FREE_GIB", "16")],
                native: (0, &["GPU ready: 16 GiB VRAM free (>= 16 GiB)"]),
                advisory: (0, &["GPU ready: 16 GiB VRAM free (>= 16 GiB)"]),
                windows: None,
            },
            // A wedged driver must not be read as "this is an APU" and waved
            // through on whatever the other pool reports.
            GateCase {
                label: "zero VRAM total against a healthy aperture",
                fixture: smi_fixture(0, 0, Some((62 * GIB, GIB))),
                env: vec![],
                native: (1, &["reported a VRAM total of 0"]),
                advisory: (0, &["no VRAM figures under WSL"]),
                windows: None,
            },
            // The same defect one pool over: a GTT total that parses as zero is
            // a failed reading, not an exhausted aperture. Unguarded it produced
            // `0 GiB free … a serve is likely still holding the GPU` — verbatim
            // the report this change exists to eliminate — and hard-failed the
            // one lane specified never to fail on a host it cannot read.
            GateCase {
                label: "carveout with a GTT total of zero",
                fixture: smi_fixture(512 * MIB, 200 * MIB, Some((0, 0))),
                env: vec![],
                native: (1, &["reported a GTT total of 0"]),
                advisory: (0, &["no usable GTT pool"]),
                windows: None,
            },
            // Nothing measurable: the native lane fails, the advisory lane warns.
            GateCase {
                label: "carveout with no GTT pool reported",
                fixture: smi_fixture(512 * MIB, 200 * MIB, None),
                env: vec![],
                native: (1, &["reported no GTT pool"]),
                advisory: (0, &["no usable GTT pool"]),
                windows: None,
            },
            // A query that failed is not a host without the pool. The aperture
            // is right there in the fixture; only the call for it broke, and
            // saying "no GTT pool" here would report a flake as a hardware fact.
            GateCase {
                label: "carveout where the GTT query itself fails",
                fixture: smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB))),
                env: vec![("REJECT_COMBINED", "1"), ("FAIL_GTT", "1")],
                native: (1, &["the GTT query itself failed"]),
                advisory: (0, &["no usable GTT pool"]),
                windows: None,
            },
            // #446. A healthy, nearly-empty aperture — every memory reading in
            // this case says the GPU is free, and the floor passes on the first
            // poll. What is NOT free is the card: a leftover llama-server
            // survived the reclaim step. A fixed 8 GiB floor against 62 GiB
            // cannot express that, and a 10-20 GiB serve clears it on every
            // lane; the process assertion is the only thing here that fails,
            // which is the whole point of adding it.
            //
            // This is also the only case that drives the reclaim stub into its
            // failing state. Delete the `--assert-clear` call from the shared
            // script and every other case below stays green.
            GateCase {
                label: "free aperture but a leaked engine process survived the reclaim",
                fixture: smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB))),
                env: vec![("RECLAIM_ASSERT_FAILS", "1")],
                native: (
                    1,
                    &[
                        // The floor WAS satisfied, so this failure cannot be
                        // mistaken for the memory gate firing.
                        "GPU ready: 61 GiB GTT free",
                        "a leaked E2E engine process is still holding it",
                        "pid=4242",
                    ],
                ),
                // The advisory WSL body has no process assertion either, and is
                // specified never to hard-fail; it passes on the reading.
                advisory: (0, &["GPU ready: 61 GiB GTT free"]),
                // The documented Windows gap: memory-only, so it clears a card
                // that is still held. Pinned, not hidden — see the field's doc.
                windows: Some((0, &["GPU ready: 61 GiB GTT free"])),
            },
        ];

        let (native, advisory) = apu_lanes();
        std::thread::scope(|scope| {
            let handles: Vec<_> = cases
                .iter()
                .map(|case| {
                    let (native, advisory, windows) = (&native, &advisory, &windows);
                    scope.spawn(move || {
                        (
                            case,
                            run_preflight(
                                native,
                                PreflightShell::DefaultRunStep,
                                &case.fixture,
                                &case.env,
                            ),
                            run_preflight(
                                advisory,
                                PreflightShell::WslBareBash,
                                &case.fixture,
                                &case.env,
                            ),
                            windows
                                .as_ref()
                                .map(|w| run_preflight_pwsh(w, &case.fixture, &case.env)),
                        )
                    })
                })
                .collect();
            for handle in handles {
                let (case, native, advisory, windows) =
                    handle.join().expect("preflight case thread");
                // A native lane that gives up must also say what holds the card.
                if case.native.0 == 1 {
                    assert!(
                        native.1.contains(REPORTED_HOLDERS),
                        "native preflight, {}: failed without reporting the GPU's \
                         holders, logged:\n{}",
                        case.label,
                        native.1
                    );
                }
                assert_gate("native", case.label, native, case.native);
                assert_gate("advisory", case.label, advisory, case.advisory);
                // The Windows block guards the same hardware class with the
                // same floor, so it owes the same verdicts as the native lane —
                // except where it has been stated not to.
                if let Some(windows) = windows {
                    assert_gate(
                        "Windows",
                        case.label,
                        windows,
                        case.windows.unwrap_or(case.native),
                    );
                }
            }
        });
    }

    /// A scan that never finishes must fail the gate AT THE BOUND.
    ///
    /// The shape test pins that the `timeout` and its 124 branch are present;
    /// only running it shows the bound actually fires, with the reason it is
    /// supposed to give. A timeout nobody has exercised is a timeout nobody
    /// knows works — and this one can never be exercised on live hardware from
    /// per-PR CI, since the lanes that run `--assert-clear` for real are
    /// self-hosted and nightly.
    ///
    /// The stub sleeps 60s against a 15s bound, so the elapsed time separates
    /// "the bound cut it short" from "the scan happened to return": unbounded,
    /// this case takes a full minute AND passes the gate, because the stub then
    /// exits 0. Both halves are asserted.
    ///
    /// Only the native lane is driven. The advisory WSL body and the Windows
    /// block have no process assertion to bound — a gap
    /// `apu_preflight_gates_on_the_pool_the_engine_uses` already pins.
    #[cfg(unix)]
    #[test]
    fn preflight_bounds_the_leaked_process_scan() {
        const GIB: u64 = 1024 * 1024 * 1024;
        const MIB: u64 = 1024 * 1024;

        if !has_timeout() {
            eprintln!("skipping: no `timeout` on PATH (GNU coreutils)");
            return;
        }
        // A healthy, nearly-empty aperture: the memory floor passes on the
        // first poll, so the only thing that can decide this run is the gate.
        let fixture = smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB)));
        let started = std::time::Instant::now();
        let (code, log) = run_preflight(
            &shared_preflight_script(),
            PreflightShell::DefaultRunStep,
            &fixture,
            &[("RECLAIM_ASSERT_HANGS", "1")],
        );
        let elapsed = started.elapsed();
        assert_eq!(
            code, 1,
            "a leaked-process scan that never returned must FAIL the preflight: it is not \
             evidence the host is clear. Logged:\n{log}"
        );
        assert!(
            log.contains("GPU ready: 61 GiB GTT free"),
            "the memory floor should have passed before the gate ran, so this failure \
             cannot be confused with the floor firing; logged:\n{log}"
        );
        assert!(
            log.contains("did not finish within 15s"),
            "the preflight failed on a stalled scan but did not say so — a timeout \
             reported as a survivor, or as a generic failure, sends the next reader \
             hunting for a process that was never named. Logged:\n{log}"
        );
        assert!(
            log.contains(REPORTED_HOLDERS),
            "a bounded-out scan still has to name what it could see holding the card; \
             logged:\n{log}"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(45),
            "the preflight took {elapsed:?} against a 15s bound and a stub that sleeps \
             60s: the bound did not cut the scan short"
        );
    }

    /// Whether this host has the procfs `--assert-clear` walks.
    ///
    /// macOS is a `cfg(unix)` target with no `/proc`, where the scan's glob
    /// expands to nothing and every host looks clear. A case driving the real
    /// script has to know the difference between "found no survivor" and "could
    /// not have found one".
    #[cfg(unix)]
    fn has_procfs() -> bool {
        Path::new("/proc/self/cmdline").exists()
    }

    /// The preflight must run the REAL `reclaim-gpu.sh --assert-clear`, the way
    /// it runs it in CI, in both directions.
    ///
    /// Everything else here plants a stub, and the script's own self-test calls
    /// `assert_clear` as a bash function — so between them nothing crossed the
    /// process boundary the lane actually crosses. Measured: with the
    /// `--assert-clear` arm DELETED from `reclaim-gpu.sh`'s `main`, the entire
    /// suite and the whole self-test stayed green. The script then falls through
    /// to its `unknown argument` arm and exits 2, which the preflight reported
    /// as "a leaked E2E engine process is still holding it (listed above)" with
    /// nothing listed above — a gate that fails every lane, for a reason that is
    /// not true, naming a process that does not exist.
    ///
    /// `RECLAIM_SELFTEST_SCOPE` confines the scan to a decoy under a temp dir,
    /// exactly as the script's own self-test confines its decoys, so this can
    /// run on a shared machine without seeing anybody else's processes — and,
    /// in the clear direction, cannot be satisfied by a host that merely has no
    /// engine running.
    #[cfg(unix)]
    #[test]
    fn preflight_runs_the_real_reclaim_assert_clear() {
        const GIB: u64 = 1024 * 1024 * 1024;
        const MIB: u64 = 1024 * 1024;

        if !has_timeout() {
            eprintln!("skipping: no `timeout` on PATH (GNU coreutils)");
            return;
        }
        if !has_procfs() {
            eprintln!("skipping: no /proc, so `--assert-clear` cannot see any process here");
            return;
        }
        let reclaim_path = repo_root().join("scripts/reclaim-gpu.sh");
        let reclaim = std::fs::read_to_string(&reclaim_path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", reclaim_path.display()));
        // A healthy, nearly-empty aperture: the floor passes on the first poll,
        // so the gate is the only thing that can decide either run.
        let fixture = smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB)));
        let scope = tempfile::tempdir().expect("decoy scope dir");
        let scope_arg = scope.path().to_str().expect("utf-8 temp path").to_owned();
        let env = [("RECLAIM_SELFTEST_SCOPE", scope_arg.as_str())];

        let (code, log) = run_preflight_with_reclaim(
            &shared_preflight_script(),
            PreflightShell::DefaultRunStep,
            &fixture,
            &env,
            &reclaim,
        );
        // The scan forks once per /proc entry, so a heavily loaded host can
        // genuinely exceed the 15s bound (measured: ~4s at 2400 pids). That is
        // the bound doing its job, not this contract failing; say so and stop
        // rather than report a slow machine as a broken gate.
        if log.contains("did not finish within 15s") {
            eprintln!("skipping: the real /proc scan did not finish inside the gate's 15s bound");
            return;
        }
        assert_eq!(
            code, 0,
            "the preflight ran the real `reclaim-gpu.sh --assert-clear` against a scope \
             holding nothing and still failed; logged:\n{log}"
        );
        assert!(
            log.contains("assert-clear: no leaked E2E engine processes are running"),
            "the preflight passed without the real script reporting a clear scan — the \
             two disagree about how this mode is spelled or what it prints; logged:\n{log}"
        );

        // argv[0] names an E2E root and an engine marker, which is what the
        // rule matches, and sits under the scope so nothing else can.
        //
        // A copy of bash running a loop, for the reasons `spawn_decoy` in
        // reclaim-gpu.sh gives: a copied multicall `sleep` exits at once as an
        // unknown program, and bash would exec a lone `sleep 300` in place.
        let decoy_dir = scope.path().join("e2e-prewarm/bin");
        std::fs::create_dir_all(&decoy_dir).expect("decoy dir");
        let decoy = decoy_dir.join("llama-server");
        std::fs::copy("/bin/bash", &decoy).expect("copy /bin/bash as the decoy");
        let mut child = std::process::Command::new(&decoy)
            .args(["-c", "for _ in {1..300}; do sleep 1; done"])
            // Detached, as in the script: a `sleep 1` child outlives the
            // SIGKILL below by up to a second and would hold the test's
            // inherited stdio open meanwhile.
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn the decoy engine process");
        let decoy_pid = child.id();
        let got = run_preflight_with_reclaim(
            &shared_preflight_script(),
            PreflightShell::DefaultRunStep,
            &fixture,
            &env,
            &reclaim,
        );
        // Reaped before any assertion: a panic here must not leave a 300s
        // decoy behind on a self-hosted runner, of all things.
        let _ = child.kill();
        let _ = child.wait();
        let (code, log) = got;
        if log.contains("did not finish within 15s") {
            eprintln!("skipping: the real /proc scan did not finish inside the gate's 15s bound");
            return;
        }
        assert_eq!(
            code, 1,
            "a real leaked engine process was running and the preflight passed; logged:\n{log}"
        );
        assert!(
            log.contains("a leaked E2E engine process is still holding it"),
            "the preflight failed with a survivor present but did not report it as one — \
             which is what any non-zero exit from a script that never ran this mode looks \
             like; logged:\n{log}"
        );
        assert!(
            log.contains(&format!("pid={decoy_pid} ")),
            "the preflight did not name the surviving process (pid={decoy_pid}); an \
             unnamed survivor is the report this gate exists to replace; logged:\n{log}"
        );
    }

    /// An `--assert-clear` that never ran must not be reported as one that
    /// found somebody.
    ///
    /// `timeout` passes the command's own status through, so the preflight sees
    /// 127 for a missing or unreadable `reclaim-gpu.sh` and 2 for a flag the two
    /// scripts disagree about — both of which it used to report as "a leaked
    /// E2E engine process is still holding it (listed above)", with nothing
    /// listed above. Deleting the `--assert-clear` arm from `reclaim-gpu.sh`
    /// produces exactly the 2.
    #[cfg(unix)]
    #[test]
    fn preflight_distinguishes_a_scan_that_never_ran_from_a_survivor() {
        const GIB: u64 = 1024 * 1024 * 1024;
        const MIB: u64 = 1024 * 1024;

        if !has_timeout() {
            eprintln!("skipping: no `timeout` on PATH (GNU coreutils)");
            return;
        }
        let fixture = smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB)));
        for rc in ["2", "127"] {
            let (code, log) = run_preflight(
                &shared_preflight_script(),
                PreflightShell::DefaultRunStep,
                &fixture,
                &[("RECLAIM_ASSERT_RC", rc)],
            );
            assert_eq!(
                code, 1,
                "a scan that could not be run is not evidence of a clear host, so the \
                 preflight must fail (reclaim exited {rc}); logged:\n{log}"
            );
            assert!(
                log.contains("could not be run") && log.contains(&format!("exited {rc}")),
                "reclaim exited {rc} — neither clear, nor a survivor, nor the bound — and \
                 the preflight did not say so; logged:\n{log}"
            );
            assert!(
                !log.contains("is still holding it"),
                "reclaim exited {rc} without naming anybody, and the preflight claimed a \
                 leaked process was holding the GPU and 'listed above'; nothing is listed \
                 above, so the lane fails pointing at a process that was never found. \
                 Logged:\n{log}"
            );
        }
    }

    /// One path into a holder report, and the reason the preflight still owes
    /// after that report has been bounded out.
    #[cfg(unix)]
    struct HolderCase {
        label: &'static str,
        fixture: String,
        env: Vec<(&'static str, &'static str)>,
        phrase: &'static str,
    }

    /// The holder report is a DIAGNOSTIC, and must not outlive the failure it
    /// is explaining.
    ///
    /// Both `--report-holders` calls were bounded by the same commit that
    /// bounded the gate, and the comments at both call sites argue the bound is
    /// load-bearing — but every other case here stubs reclaim to return
    /// instantly, so removing either `timeout 15` left the whole suite green.
    /// `--report-holders` walks /proc too, and it is reached precisely when
    /// something is wedged, so the hang risk is the same one the gate's bound
    /// exists for: an unbounded diagnostic spends the ceiling the fast failure
    /// just saved.
    ///
    /// Both call sites are driven, because they are separate lines reached by
    /// separate paths: the gate's failure, and the ceiling expiring.
    #[cfg(unix)]
    #[test]
    fn preflight_bounds_the_holder_report() {
        const GIB: u64 = 1024 * 1024 * 1024;
        const MIB: u64 = 1024 * 1024;

        if !has_timeout() {
            eprintln!("skipping: no `timeout` on PATH (GNU coreutils)");
            return;
        }
        // First: the gate fires with a survivor, which reports holders and then
        // exits 1. Second: the floor is never met, so the ceiling expires and
        // the report at the end of the script runs instead.
        let paths = [
            HolderCase {
                label: "after the leaked-process gate fails",
                fixture: smi_fixture(512 * MIB, 200 * MIB, Some((62 * GIB, GIB))),
                env: vec![
                    ("RECLAIM_ASSERT_FAILS", "1"),
                    ("RECLAIM_HOLDERS_HANGS", "1"),
                ],
                phrase: "a leaked E2E engine process is still holding it",
            },
            HolderCase {
                label: "after the ceiling expires",
                fixture: smi_fixture(192 * GIB, 191 * GIB, None),
                env: vec![
                    ("GPU_PREFLIGHT_MIN_FREE_GIB", "16"),
                    ("RECLAIM_HOLDERS_HANGS", "1"),
                ],
                phrase: "never dropped below the floor",
            },
        ];
        std::thread::scope(|scope| {
            let handles: Vec<_> = paths
                .iter()
                .map(|case| {
                    scope.spawn(move || {
                        let started = std::time::Instant::now();
                        let got = run_preflight(
                            &shared_preflight_script(),
                            PreflightShell::DefaultRunStep,
                            &case.fixture,
                            &case.env,
                        );
                        (case, got, started.elapsed())
                    })
                })
                .collect();
            for handle in handles {
                let (case, (code, log), elapsed) =
                    handle.join().expect("holder-report case thread");
                let label = case.label;
                assert_eq!(
                    code, 1,
                    "{label}: a hanging holder report must not change the verdict; \
                     logged:\n{log}"
                );
                assert!(
                    log.contains(case.phrase),
                    "{label}: the preflight lost its own reason while reporting holders; \
                     logged:\n{log}"
                );
                assert!(
                    elapsed < std::time::Duration::from_secs(45),
                    "{label}: the preflight took {elapsed:?} against a 15s bound and a \
                     holder report that sleeps 60s — the report is unbounded, so a wedged \
                     host pays for the diagnostic with the fast failure it was printing"
                );
            }
        });
    }

    #[cfg(unix)]
    #[test]
    fn apu_preflight_verdict_reflects_the_last_completed_reading() {
        const MIB: u64 = 1024 * 1024;

        // A poll that ends in an early `continue` — rocm-smi timing out, empty
        // fields — learned nothing, so it must not overwrite what the last
        // completed poll established. Here poll 1 reads a carveout with no GTT
        // pool and the tool then stops answering entirely.
        //
        // The advisory lane must still WARN: it is specified never to hard-fail
        // on a host it cannot measure. Re-deriving that state per poll instead
        // would let the silent second poll flip this to `exit 1` with "a serve
        // is likely still holding the GPU" — a serve this run never observed,
        // which is the report the whole change exists to stop.
        if !has_timeout() {
            eprintln!("skipping: no `timeout` on PATH (GNU coreutils)");
            return;
        }

        // Two polls: one at t=0, one after the script's own 5s backoff. The
        // first spends two calls (combined, then GTT alone); everything after
        // that fails, so the second poll reads nothing.
        //
        // The ceiling is counted in bash's whole-second `$SECONDS`, so a
        // ceiling of N buys between N-1 and N real seconds. 7 keeps the second
        // check, at just over 5s, inside the budget; 6 left it a coin toss, and
        // a skipped second poll passes this test without exercising it.
        let env = [("GOOD_CALLS", "2"), ("GPU_PREFLIGHT_CEILING_SECS", "7")];
        let fixture = smi_fixture(512 * MIB, 200 * MIB, None);
        let (native, advisory) = apu_lanes();

        assert_gate(
            "advisory",
            "reading lost after the first poll",
            run_preflight(&advisory, PreflightShell::WslBareBash, &fixture, &env),
            (0, &["no usable GTT pool"]),
        );
        // The native lane hard-fails either way, but for the reason the last
        // completed poll actually found rather than a default about a serve.
        assert_gate(
            "native",
            "reading lost after the first poll",
            run_preflight(&native, PreflightShell::DefaultRunStep, &fixture, &env),
            (1, &["reported no GTT pool"]),
        );
    }

    /// A `BASH_ENV` file that stalls the script once, for just over a second,
    /// as soon as its `deadline` is set — before anything else runs.
    ///
    /// That is the descheduling a loaded host can impose at the same spot.
    /// `$SECONDS` ticks on whole wall-clock seconds, so with a 1s ceiling the
    /// deadline is already past when the script resumes.
    #[cfg(unix)]
    const STALL_AFTER_DEADLINE: &str = "trap '[ -z \"${__stalled:-}\" ] && [ -n \"${deadline:-}\" ] \
         && { __stalled=1; sleep 1.05; } || true' DEBUG\n";

    #[cfg(unix)]
    #[test]
    fn every_shell_preflight_polls_at_least_once_whatever_the_ceiling() {
        const GIB: u64 = 1024 * 1024 * 1024;

        // A lane whose deadline expires before its first poll reports a verdict
        // no reading produced: the native lanes "rocm-smi never returned within
        // its timeout", the advisory ones "no VRAM figures under WSL". This
        // flaked the gate test above whenever the second boundary passed
        // between computing the deadline and the loop's first check. Every
        // shell copy in both workflows is driven, not just the APU twins the
        // drift test pins, because each was written with the same loop.
        if !has_timeout() {
            eprintln!("skipping: no `timeout` on PATH (GNU coreutils)");
            return;
        }
        let trap_dir = tempfile::tempdir().expect("temp dir");
        let bash_env = trap_dir.path().join("stall.sh");
        std::fs::write(&bash_env, STALL_AFTER_DEADLINE).expect("write BASH_ENV");
        let bash_env = bash_env.to_str().expect("utf-8 temp path");

        let mut lanes = Vec::new();
        let mut delegating = 0;
        for (workflow, text) in self_hosted_workflows() {
            for (i, step) in gpu_preflight_steps(&text).iter().enumerate() {
                let Some(script) = preflight_shell_script(step) else {
                    continue;
                };
                // A lane that delegates carries one line that runs the shared
                // script; the loop is in the script, so driving the line would
                // assert nothing. Counted rather than skipped, so a lane whose
                // body quietly came BACK is not silently exempted from the
                // check below by never being driven.
                if script.contains(SHARED_PREFLIGHT) {
                    delegating += 1;
                    continue;
                }
                let shell = if step.contains("-Script @'") {
                    PreflightShell::WslBareBash
                } else {
                    PreflightShell::DefaultRunStep
                };
                lanes.push((format!("{workflow} shell lane #{i}"), script, shell));
            }
        }
        assert_eq!(
            delegating, 8,
            "expected the 8 native Linux preflight lanes to delegate to \
             {SHARED_PREFLIGHT} — if a lane was added or removed, update this count"
        );
        assert_eq!(
            lanes.len(),
            2,
            "expected the 2 WSL advisory preflight bodies to be the only shell blocks \
             left inline across both workflows — if a lane was added or removed, update \
             this count"
        );
        // The shared script once, standing in for all eight delegating lanes: they
        // run the same bytes, and its loop is the one that has to poll.
        lanes.push((
            format!("shared {SHARED_PREFLIGHT}"),
            shared_preflight_script(),
            PreflightShell::DefaultRunStep,
        ));

        // A card held by a leftover serve: every lane, native or advisory,
        // discrete or APU, must take a reading and fail on it.
        let fixture = smi_fixture(192 * GIB, 191 * GIB, None);
        let env = [("BASH_ENV", bash_env), ("GPU_PREFLIGHT_CEILING_SECS", "1")];
        std::thread::scope(|scope| {
            let handles: Vec<_> = lanes
                .iter()
                .map(|(label, script, shell)| {
                    let (fixture, env) = (&fixture, &env);
                    scope.spawn(move || (label, shell, run_preflight(script, *shell, fixture, env)))
                })
                .collect();
            for handle in handles {
                let (label, shell, got) = handle.join().expect("preflight lane thread");
                // The advisory lanes `cd` to the runner's absolute checkout
                // path first, so only the native ones reach the stub here.
                if matches!(shell, PreflightShell::DefaultRunStep) {
                    assert!(
                        got.1.contains(REPORTED_HOLDERS),
                        "{label}: failed without reporting the GPU's holders, logged:\n{}",
                        got.1
                    );
                }
                assert_gate(
                    label,
                    "deadline already past at the first check",
                    got,
                    (1, &["VRAM never dropped below the floor"]),
                );
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
    fn nightly_publishes_the_consolidated_matrix_to_the_wiki() {
        let nightly = read_workflow("nightly.yml");
        let report = normalized_whitespace(job_block(&nightly, "e2e-report-nightly"));
        assert!(
            report.contains("--html-out consolidated/index.html > consolidated/support-matrix.md"),
            "the consolidated report job must redirect the markdown matrix into \
             consolidated/support-matrix.md for the wiki job"
        );
        assert!(
            report.contains("cat consolidated/support-matrix.md >> \"$GITHUB_STEP_SUMMARY\""),
            "the nightly step summary must still receive the matrix"
        );
        let publish = job_block(&nightly, "publish-e2e-wiki");
        assert_eq!(
            job_scalar(publish, "needs"),
            "e2e-report-nightly",
            "the wiki job must publish the consolidated report, so it must wait for it"
        );
        let condition = job_scalar(publish, "if");
        for required in [
            "!cancelled()",
            "needs.e2e-report-nightly.result == 'success'",
            "github.repository == 'ROCm/rocm-cli'",
            "github.ref == 'refs/heads/main'",
        ] {
            assert!(
                condition.contains(required),
                "the wiki job's `if:` must require `{required}`: {condition}"
            );
        }
        assert!(
            !condition.contains("always()"),
            "always() would publish from a cancelled run: {condition}"
        );
        let permissions = job_mapping(publish, "permissions");
        assert_eq!(
            permissions.get("contents").map(String::as_str),
            Some("write"),
            "pushing to the wiki needs contents: write on this job"
        );
        let script = normalized_whitespace(publish);
        assert!(
            script.contains("name: e2e-consolidated-report-nightly")
                && script.contains(".wiki.git"),
            "the wiki job must download the nightly consolidated artifact and push to the wiki repo"
        );
        assert!(
            script.contains("git push origin HEAD"),
            "the wiki job must push the page"
        );
        assert!(
            script.contains("if ! git clone --depth 1 \"$wiki\" wiki; then")
                && script.contains("exit 1 fi"),
            "a failed wiki clone must fail the job with the prerequisite, not be ignored"
        );
        assert!(
            script
                .contains("if ! grep -q '^| Platform | OS |' consolidated/support-matrix.md; then")
                && script.contains("leaving the wiki page unchanged"),
            "an empty consolidation must leave the last good page in place"
        );
        assert!(
            !publish.contains("git init"),
            "no `git init` fallback: it cannot create a remote wiki"
        );
    }

    /// Runs the publish job's real empty-report guard against the generator's
    /// real empty output, so rewording either side (or breaking the guard's
    /// exit) is caught by behavior rather than by pinning strings.
    #[cfg(unix)]
    #[test]
    fn wiki_publish_guard_stops_on_the_generators_empty_output() {
        let nightly = read_workflow("nightly.yml");
        let publish = job_block(&nightly, "publish-e2e-wiki");
        let lines: Vec<&str> = publish.lines().collect();
        let start = lines
            .iter()
            .position(|l| {
                l.trim_start()
                    .starts_with("if ! grep -q '^| Platform | OS |'")
            })
            .expect("publish job has the empty-report guard");
        let end = start
            + lines[start..]
                .iter()
                .position(|l| l.trim() == "fi")
                .expect("guard is closed by fi");
        let guard = lines[start..=end].join("\n");

        let run = |matrix: &str| -> String {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path();
            std::fs::create_dir_all(dir.join("consolidated")).unwrap();
            std::fs::write(dir.join("consolidated/support-matrix.md"), matrix).unwrap();
            let script = format!("set -euo pipefail\n{guard}\necho REACHED_PUBLISH\n");
            let out = std::process::Command::new("bash")
                .arg("-c")
                .arg(script)
                .current_dir(dir)
                .output()
                .expect("run bash");
            assert!(out.status.success(), "guard script failed: {out:?}");
            String::from_utf8_lossy(&out.stdout).into_owned()
        };

        let empty =
            e2e_report::consolidated_summary_markdown(&e2e_report::load_platform_reports(&[]));
        assert!(
            !run(&empty).contains("REACHED_PUBLISH"),
            "the generator's empty output must stop the publish"
        );
        assert!(
            !run("").contains("REACHED_PUBLISH"),
            "a zero-byte matrix must not be published"
        );
        assert!(
            run("## E2E consolidated report\n\n| Platform | OS |\n").contains("REACHED_PUBLISH"),
            "a matrix with platforms must reach the publish step"
        );
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
            .map(|name| crate::e2e_report::without_channel_matrix_segment(&name))
            .collect();
        declared_artifacts.sort();
        declared_artifacts.dedup();
        let mut documented_artifacts = backticked_list_between(
            &docs,
            "The lane artifacts are named canonically (",
            ") in `ci.yml` and",
        );
        // Sorted, not deduplicated: a name listed twice must still fail.
        documented_artifacts.sort();
        assert_eq!(
            documented_artifacts, declared_artifacts,
            "the canonical artifact list must enumerate every uploaded report artifact exactly once"
        );

        // ponytail: brittle literal-text match, not a YAML matrix parse — it exists only
        // to keep this hardcoded literal in sync with `consolidated::parse_descriptor`'s
        // hardcoded "release"/"nightly" suffixes. If nightly.yml's channel matrix ever
        // changes, update both it and `parse_descriptor` together.
        let nightly = std::fs::read_to_string(repo_root().join(".github/workflows/nightly.yml"))
            .expect("read nightly.yml");
        assert!(
            nightly.contains("channel: [release, nightly]"),
            "nightly.yml's channel matrix must declare exactly the channel values \
             `consolidated::parse_descriptor` strips as artifact-name suffixes; if this \
             literal ever changes, `parse_descriptor` must change with it"
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
                "self-hosted lane `{}` is not a required status check, so the README \
                 must not document it as blocking (continue-on-error does not decide \
                 this; see the header of e2e-selfhosted.yml)",
                row[0]
            );
        }
    }

    /// Extract the literal glob entries from ci.yml's pull-request `rust:`
    /// paths-filter bucket, stopping at the next sibling filter key. Scoped to
    /// an exact `rust:` line (not a prefix match) so this doesn't also pick up
    /// the unrelated `rust: ${{ … }}` job output a few lines above it.
    fn rust_paths_filter_entries(ci: &str) -> Vec<String> {
        let lines: Vec<&str> = ci.lines().collect();
        let start = lines
            .iter()
            .position(|line| line.trim() == "rust:")
            .unwrap_or_else(|| panic!("ci.yml declares a `rust:` paths-filter bucket"));
        let key_indent = indent_of(lines[start]);
        let end = lines[start + 1..]
            .iter()
            .position(|line| !line.trim().is_empty() && indent_of(line) <= key_indent)
            .map_or(lines.len(), |offset| start + 1 + offset);
        lines[start + 1..end]
            .iter()
            .filter_map(|line| strip_comment(line).trim().strip_prefix("- "))
            .map(|item| strip_quotes(item.trim()))
            .collect()
    }

    /// `forces_full_workspace` (xtask/src/affected.rs) names the doc paths
    /// that fall back to a full-workspace run because an xtask test reads
    /// them directly ([`crate::affected::XTASK_READ_DOCS`]). If one of them
    /// isn't ALSO in ci.yml's `rust` paths-filter bucket,
    /// `needs.changes.outputs.rust` stays false for a PR that touches only
    /// that doc, and the `test` job's every step — including `cargo nextest
    /// run`, which is what would have caught the drift — never runs at all.
    #[test]
    fn forces_full_workspace_docs_are_in_the_rust_filter() {
        let ci = read_workflow("ci.yml");
        let rust_filter = rust_paths_filter_entries(&ci);
        for doc in crate::affected::XTASK_READ_DOCS {
            assert!(
                rust_filter.iter().any(|entry| entry == doc),
                "`{doc}` is in `forces_full_workspace` (xtask/src/affected.rs) but \
                 missing from ci.yml's `rust` paths-filter bucket, so an edit to it \
                 alone skips the rust-gated jobs entirely instead of running the \
                 test that reads it"
            );
        }
    }

    #[test]
    fn workflows_use_distinct_concurrency_groups() {
        // Isolation comes from the group KEY differing per workflow. Extract the
        // actual concurrency.group value from each and assert (a) ci.yml keeps
        // supersession (e2e-selfhosted.yml scopes it to pull_request, see
        // `self_hosted_supersession_is_scoped_to_pull_requests`), (b) both key on
        // github.workflow, and (c) the two group expressions and workflow names differ — so at runtime the groups are
        // distinct and an offline-runner stall in one can't hold the other.
        let ci = read_workflow("ci.yml");
        let sh = read_workflow("e2e-selfhosted.yml");
        let ci_group = concurrency_group(&ci);
        let sh_group = concurrency_group(&sh);

        for (label, group) in [("ci.yml", &ci_group), ("e2e-selfhosted.yml", &sh_group)] {
            assert!(
                group.contains("github.workflow"),
                "{label} concurrency.group must be namespaced by github.workflow \
                 (got `{group}`) (EAI-7548)"
            );
        }
        // Both keep supersession; e2e-selfhosted.yml narrows it to pull_request
        // (see `self_hosted_supersession_is_scoped_to_pull_requests`).
        assert!(
            ci.contains("cancel-in-progress: true"),
            "ci.yml must keep cancel-in-progress: true"
        );
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

    /// Extract the version token right after `marker` in `line`, up to the
    /// next whitespace (trailing sentence punctuation is trimmed too, so a
    /// doc sentence ending right after the version with no space, e.g.
    /// `cargo-about@0.9.1.`, doesn't glue the period onto the token). With
    /// marker `"hawkeye@"`, `cargo install hawkeye@7.0.0 --locked` yields
    /// `7.0.0`.
    ///
    /// `strip_comments` must be true only for YAML lines: there, a pin
    /// mentioned only inside a trailing `# ...` comment must not satisfy a
    /// check meant for the real line beside it. It must be false for
    /// Markdown prose, where `strip_comment`'s `" #"` delimiter also matches
    /// an issue reference (`see #474, install hawkeye@7.0.1`) that is not a
    /// comment at all — stripping there would silently drop that line's pin
    /// from the check, which is the exact stale-duplicate case this helper
    /// exists to catch. Compared for exact equality (not `contains`) so
    /// neither a stale duplicate pin nor a version sharing a prefix
    /// (`0.9.10` satisfying a check for `0.9.1`) can false-pass.
    fn pin_token<'a>(line: &'a str, marker: &str, strip_comments: bool) -> Option<&'a str> {
        let line = if strip_comments {
            strip_comment(line)
        } else {
            line
        };
        let (_, rest) = line.split_once(marker)?;
        let token = rest.split_whitespace().next().unwrap_or(rest);
        Some(token.trim_end_matches(['.', ',', ';', ':', '!', '?', ')']))
    }

    /// Every `marker` occurrence on its own line — not just the first line
    /// that has one — must pin exactly `expected`: a stale duplicate left
    /// behind by a previous bump must not go unnoticed. (`pin_token` itself
    /// only reads the first occurrence within a single line; today's pins
    /// are each one per line, so this doesn't miss anything in practice.)
    ///
    /// Comment-stripping (see `pin_token`'s doc) is scoped by `path`'s
    /// extension: on for `.yml`/`.yaml`, off for everything else (our
    /// Markdown callers).
    fn assert_pins_match(marker: &str, expected: &str, path: &Path, text: &str) {
        let strip_comments = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("yml") || ext.eq_ignore_ascii_case("yaml"));
        let mut found = false;
        for line in text.lines() {
            let Some(token) = pin_token(line, marker, strip_comments) else {
                continue;
            };
            found = true;
            assert_eq!(
                token,
                expected,
                "{} pins `{marker}{token}`, which does not match the pinned version \
                 `{expected}`:\n{line}",
                path.display()
            );
        }
        assert!(
            found,
            "{} has no `{marker}` pin; expected `{marker}{expected}`",
            path.display()
        );
    }

    /// The hawkeye and cargo-about versions pinned across CI, CONTRIBUTING.md
    /// and (for cargo-about) MANIFEST.md must never drift from each other: a
    /// mismatch lets a contributor's local tool accept or reject something CI
    /// disagrees with, silently leaving CI as the only real gate.
    ///
    /// hawkeye's canonical version lives in `ci.yml`'s `license-headers`
    /// job's `env:` mapping, read with the extractors this file already has
    /// fixture tests for (`job_block`, `job_mapping`) rather than duplicated
    /// into a constant — scoped to that one job so a nested step `env` or an
    /// unrelated job can't leak in. A bump needs three edits: this test
    /// guards two of them (ci.yml's `HAWKEYE_VERSION` and CONTRIBUTING.md);
    /// the third, `HAWKEYE_SHA256` right beside it in ci.yml, is not guarded
    /// here — see the comment at its definition. cargo-about's canonical
    /// version is the `ABOUT_VERSION` constant in `tpn.rs`, which is what
    /// `cargo xtask tpn` itself builds against. A cargo-about bump needs
    /// seven edits across five files: `ABOUT_VERSION` itself, plus six
    /// mirror pins — the four `cargo-about@` install lines (CONTRIBUTING.md,
    /// MANIFEST.md, ci.yml, dependabot-manifests.yml) and the two
    /// `${{ runner.os }}-cargo-about-<version>` cache keys in ci.yml and
    /// dependabot-manifests.yml — and this test guards all six mirrors
    /// against the `ABOUT_VERSION` canonical: the cache keys use a `-`
    /// instead of `@`, so they need their own marker row rather than the
    /// `cargo-about@` one. A forgotten cache-key bump would otherwise
    /// cache-hit the OLD binary while `tpn --check` enforces the NEW
    /// version, failing loudly but in a confusing place.
    ///
    /// One table, one check: a new pinned tool, or a new file that mentions
    /// an existing one, costs a row here rather than another ~30-line test.
    #[test]
    fn pinned_tool_versions_match_across_docs_and_workflows() {
        let ci = read_workflow("ci.yml");
        let hawkeye_env = job_mapping(job_block(&ci, "license-headers"), "env");
        let hawkeye_installed = hawkeye_env
            .get("HAWKEYE_VERSION")
            .unwrap_or_else(|| panic!("license-headers job has no HAWKEYE_VERSION entry"));
        // ci.yml's env var is `v`-prefixed (`v7.0.0`); CONTRIBUTING.md's prose
        // is not (`hawkeye@7.0.0`).
        let hawkeye_version = hawkeye_installed
            .strip_prefix('v')
            .unwrap_or_else(|| panic!("HAWKEYE_VERSION `{hawkeye_installed}` must be `v`-prefixed"))
            .to_owned();
        let about_version = crate::tpn::ABOUT_VERSION.to_owned();

        let root = repo_root();
        let read = |rel: &str| {
            let p = root.join(rel);
            std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
        };
        // Each file below is read once here rather than once per table row
        // that names it: CONTRIBUTING.md and dependabot-manifests.yml each
        // appear in two rows, and ci.yml appears in a row on top of the read
        // already done above for HAWKEYE_VERSION.
        let contributing = read("CONTRIBUTING.md");
        let manifest = read("MANIFEST.md");
        let dependabot = read(".github/workflows/dependabot-manifests.yml");
        let file_text = |rel: &str| -> &str {
            match rel {
                "CONTRIBUTING.md" => &contributing,
                "MANIFEST.md" => &manifest,
                ".github/workflows/ci.yml" => &ci,
                ".github/workflows/dependabot-manifests.yml" => &dependabot,
                other => panic!(
                    "pinned_tool_versions_match_across_docs_and_workflows: unexpected file `{other}`"
                ),
            }
        };

        let table: [(&str, String, &[&str]); 3] = [
            ("hawkeye@", hawkeye_version, &["CONTRIBUTING.md"]),
            (
                "cargo-about@",
                about_version.clone(),
                &[
                    "CONTRIBUTING.md",
                    "MANIFEST.md",
                    ".github/workflows/ci.yml",
                    ".github/workflows/dependabot-manifests.yml",
                ],
            ),
            (
                "cargo-about-",
                about_version,
                &[
                    ".github/workflows/ci.yml",
                    ".github/workflows/dependabot-manifests.yml",
                ],
            ),
        ];

        for (marker, expected, files) in table {
            for file in files {
                assert_pins_match(marker, &expected, &root.join(file), file_text(file));
            }
        }
    }

    /// Extract one prek hook's complete YAML block by its `- id:` value, up
    /// to the next line at or above this hook's own `- id:` indent — the
    /// same indent-based termination `job_block` and `nested_block` already
    /// use, rather than two needles hardcoded to the exact text that
    /// happens to follow today (`- id:`/`- repo:`). Hooks inside one
    /// `repo:` block sit back to back with no blank line between them (see
    /// `.pre-commit-config.yaml`'s `cargo-fmt`/`cargo-clippy`/... run), so a
    /// blank-line-anchored end would silently fold a following sibling
    /// hook's lines into this one's block whenever this hook is not the
    /// last in its `repo:` block — and a needle that fails to match (e.g. a
    /// reformat, or this hook ending up last with only a trailing comment
    /// after it) would silently run the block to the end of the file
    /// instead of failing loudly. A comment line at or below the hook's own
    /// indent doesn't end the block by itself: it only does when it turns
    /// out to head the next sibling, i.e. nothing deeper follows it before
    /// the next such line — a comment with real indent-8 content still
    /// after it (`hook_block_does_not_end_at_a_comment_only_line`) is just
    /// an annotation inside this hook's own mapping, not a boundary, while
    /// one immediately followed by the next `- id:`/`- repo:` (as
    /// `.pre-commit-config.yaml` has between `license-headers` and the
    /// `cargo-fmt` repo block today) is the next block's header comment and
    /// must not be folded into this one's.
    fn hook_block<'a>(text: &'a str, id: &str) -> &'a str {
        let marker = format!("- id: {id}\n");
        let start = text
            .find(&marker)
            .unwrap_or_else(|| panic!(".pre-commit-config.yaml defines hook `{id}`"));
        let line_start = text[..start].rfind('\n').map_or(0, |i| i + 1);
        let marker_indent = indent_of(&text[line_start..start]);
        let rest = &text[start + marker.len()..];

        let mut offset = 0usize;
        let mut pending_comment: Option<usize> = None;
        for line in rest.split_inclusive('\n') {
            let content = line.strip_suffix('\n').unwrap_or(line);
            let trimmed = content.trim_start();
            if trimmed.is_empty() {
                offset += line.len();
                continue;
            }
            if indent_of(content) > marker_indent {
                // Real content deeper than the hook's own indent proves any
                // dedented comment seen so far was just an annotation inside
                // this hook's mapping, not a sibling's header.
                pending_comment = None;
                offset += line.len();
                continue;
            }
            if trimmed.starts_with('#') {
                pending_comment.get_or_insert(offset);
                offset += line.len();
                continue;
            }
            return &rest[..pending_comment.unwrap_or(offset)];
        }
        rest
    }

    /// True if `block` has a line that, after stripping any trailing `#`
    /// comment, is exactly `flag` once trimmed — so `flag` mentioned only
    /// inside a comment, or commented out with the real line changed to
    /// something else, can't satisfy the check.
    fn hook_block_sets(block: &str, flag: &str) -> bool {
        block.lines().any(|line| strip_comment(line).trim() == flag)
    }

    /// Regression guard: the `license-headers` hook once had a `types_or`
    /// filter (`types_or: [rust, python, shell]`) that let a
    /// `licenserc.toml`- or Markdown-only commit skip it entirely, while
    /// CI's hawkeye check still enforced it. `always_run: true` is what
    /// actually closes that gap: it fires the hook on every commit/push
    /// regardless of any file-type filter, and hawkeye then scans the
    /// whole repo per `licenserc.toml` (`pass_filenames: false`) however
    /// it was triggered. The filter-key checks below guard the next step:
    /// `always_run: true` already makes any of them inert, so none of
    /// these alone can reopen the gap, but leaving one in place is
    /// misleading maintenance debt that could let a later edit drop
    /// `always_run` while assuming the filter still gates correctly.
    #[test]
    fn license_headers_hook_has_no_file_type_filter() {
        // CRLF-normalize: `.pre-commit-config.yaml` isn't forced to `eol=lf` in
        // .gitattributes, so a Windows checkout can give `\r\n` line endings,
        // which `hook_block`'s `\n`-anchored marker would otherwise miss.
        let config = std::fs::read_to_string(repo_root().join(".pre-commit-config.yaml"))
            .expect("reading .pre-commit-config.yaml")
            .replace("\r\n", "\n");
        let block = hook_block(&config, "license-headers");
        assert!(
            hook_block_sets(block, "always_run: true"),
            "license-headers hook must set `always_run: true`:\n{block}"
        );
        assert!(
            hook_block_sets(block, "pass_filenames: false"),
            "license-headers hook must set `pass_filenames: false` — hawkeye scans \
             the whole repo per `licenserc.toml` regardless of trigger only because \
             of this flag; without it, pre-commit would narrow the scan to changed \
             files instead:\n{block}"
        );
        for filter_key in ["types:", "types_or:", "files:", "exclude:"] {
            assert!(
                !block.contains(filter_key),
                "license-headers hook must not have a file-type filter (`{filter_key}`) — \
                 `always_run: true` above already makes it inert, but leaving one in place \
                 is misleading maintenance debt that could let a later edit drop \
                 `always_run` while assuming the filter still gates correctly:\n{block}"
            );
        }
    }

    // Extractor guards: prove the helpers above actually parse the shapes they
    // claim to — multiline hook blocks, comment-stripping, and single-line pin
    // tokens alike — so the contract tests above can't silently false-pass on a
    // shape they don't handle.
    #[test]
    fn hook_block_stops_at_next_hook_in_the_same_repo_block() {
        // license-headers is NOT last in its `hooks:` list here, unlike in the
        // real file today — this is exactly the shape that let a sibling
        // hook's `always_run: true` leak into license-headers' block.
        let yaml = "\
  - repo: local
    hooks:
      - id: license-headers
        name: license header check (hawkeye)
        pass_filenames: false
      - id: cargo-fmt
        name: cargo fmt
        always_run: true
";
        let block = hook_block(yaml, "license-headers");
        assert!(
            !block.contains("always_run"),
            "hook_block leaked into the next hook's lines:\n{block}"
        );
    }

    #[test]
    fn hook_block_stops_at_the_next_repo_entry() {
        let yaml = "\
  - repo: local
    hooks:
      - id: license-headers
        name: license header check (hawkeye)
        pass_filenames: false
  - repo: local
    hooks:
      - id: cargo-fmt
        always_run: true
";
        let block = hook_block(yaml, "license-headers");
        assert!(
            !block.contains("always_run"),
            "hook_block leaked into the next repo block:\n{block}"
        );
    }

    #[test]
    fn hook_block_does_not_end_at_a_comment_only_line() {
        // The comment sits at the hook's own `- id:` indent — exactly the
        // indent that used to end the block — with a real filter key after
        // it. A reformat could leave a repo looking like this; the filter
        // key must still be visible to the contract test above.
        let yaml = "\
  - repo: local
    hooks:
      - id: license-headers
        name: license header check (hawkeye)
      # a comment at the hook's own indent must not end the block
        types_or: [rust]
      - id: cargo-fmt
        always_run: true
";
        let block = hook_block(yaml, "license-headers");
        assert!(
            block.contains("types_or:"),
            "hook_block ended at a comment-only line, hiding a types_or filter that \
             comes after it:\n{block}"
        );
        assert!(
            !block.contains("cargo-fmt") && !block.contains("always_run"),
            "hook_block still must not leak into the next hook's lines:\n{block}"
        );
    }

    #[test]
    fn hook_block_stops_at_a_dedented_comment_heading_the_next_repo_block() {
        // Unlike the comment in the test above, this one sits right before the
        // next `repo:` entry with nothing deeper in between — it heads that
        // entry rather than annotating `license-headers`' own mapping, exactly
        // the shape `.pre-commit-config.yaml` has between `license-headers`
        // and the `cargo-fmt` repo block today.
        let yaml = "\
  - repo: local
    hooks:
      - id: license-headers
        name: license header check (hawkeye)
        pass_filenames: false

  # Rust + PowerShell: mirror the CI checks using the local toolchain.
  - repo: local
    hooks:
      - id: cargo-fmt
        always_run: true
";
        let block = hook_block(yaml, "license-headers");
        assert!(
            !block.contains("Rust + PowerShell") && !block.contains("always_run"),
            "hook_block folded the next repo block's header comment into this hook's \
             block:\n{block}"
        );
    }

    #[test]
    fn hook_block_stops_at_the_first_of_several_header_comment_lines() {
        // A multi-line header comment, still heading the next repo block (nothing
        // deeper follows before it). The boundary must be the FIRST of these
        // lines, not the last: `pending_comment` has to keep the earliest offset
        // it saw, because every line in the run belongs to the next sibling's
        // header, not to `license-headers`' own block. Keeping the last offset
        // instead would fold every comment line but the last into this block —
        // undetectable with only one header-comment line, which is why this test
        // needs at least two.
        let yaml = "\
  - repo: local
    hooks:
      - id: license-headers
        name: license header check (hawkeye)
        pass_filenames: false

  # First header line: types_or: [rust]
  # Second header line, right before the next repo entry.
  - repo: local
    hooks:
      - id: cargo-fmt
        always_run: true
";
        let block = hook_block(yaml, "license-headers");
        assert!(
            !block.contains("First header line") && !block.contains("Second header line"),
            "hook_block folded a multi-line header comment into this hook's block:\n{block}"
        );
    }

    #[test]
    fn hook_block_sets_rejects_comment_only_and_falsified_mentions() {
        assert!(hook_block_sets(
            "        always_run: true",
            "always_run: true"
        ));
        // Trailing comment after the real value is still read.
        assert!(hook_block_sets(
            "        always_run: true  # hawkeye ignores pass_filenames",
            "always_run: true"
        ));
        // The flag mentioned only inside a comment, with the real line
        // changed to something else, must not satisfy the check.
        assert!(!hook_block_sets(
            "        # always_run: true\n        always_run: false",
            "always_run: true"
        ));
    }

    #[test]
    fn pin_token_rejects_prefixes_and_comment_only_mentions() {
        assert_eq!(
            pin_token(
                "cargo install cargo-about@0.9.1 --locked",
                "cargo-about@",
                true
            ),
            Some("0.9.1")
        );
        // A version sharing a prefix must not compare equal to the one it shares
        // a prefix with — `contains` would wrongly let this satisfy `"0.9.1"`.
        assert_eq!(
            pin_token(
                "cargo install cargo-about@0.9.10 --locked",
                "cargo-about@",
                true
            ),
            Some("0.9.10")
        );
        // A pin mentioned only after a trailing ` #` comment (the real line
        // changed to something else) must not be seen, when comment-stripping
        // is on (the YAML case).
        assert_eq!(
            pin_token(
                "cargo install cargo-about --version 0.9.0  # was cargo-about@0.9.1",
                "cargo-about@",
                true,
            ),
            None
        );
        // Prose that ends the sentence right after the version, with no
        // separating whitespace, must not glue the punctuation onto the token.
        assert_eq!(
            pin_token("pinned to cargo-about@0.9.1.", "cargo-about@", true),
            Some("0.9.1")
        );
        // With comment-stripping off (the Markdown case), an issue reference
        // like `#474` must not be mistaken for a `# ...` comment that hides the
        // real pin after it — the exact shape CONTRIBUTING.md/MANIFEST.md prose
        // can take.
        assert_eq!(
            pin_token("see #474, install hawkeye@7.0.1", "hawkeye@", false),
            Some("7.0.1")
        );
    }

    #[test]
    #[should_panic(expected = "pins `cargo-about@0.8.0`")]
    fn assert_pins_match_flags_a_stale_duplicate_pin() {
        let text = "cargo install cargo-about@0.9.1 --locked\n\
                     cargo install cargo-about@0.8.0 --locked\n";
        assert_pins_match("cargo-about@", "0.9.1", Path::new("fixture"), text);
    }

    #[test]
    #[should_panic(expected = "pins `cargo-about@0.9.10`, which does not match")]
    fn assert_pins_match_flags_a_version_sharing_a_prefix() {
        // The property `pin_token` guarantees — `0.9.10` must not satisfy a
        // check for `0.9.1` — proven end to end through the real comparison
        // path, not just through extraction.
        let text = "cargo install cargo-about@0.9.10 --locked\n";
        assert_pins_match("cargo-about@", "0.9.1", Path::new("fixture"), text);
    }

    #[test]
    #[should_panic(expected = "has no `cargo-about@` pin")]
    fn assert_pins_match_flags_a_marker_with_no_pin_at_all() {
        // A file whose pin was reworded out of the marker's shape (e.g. split
        // onto `--version` instead of `cargo-about@<version>`) must not
        // silently drop out of the drift guard — deleting the `found` check
        // this fixture pins would otherwise leave every other test green.
        let text = "cargo install cargo-about --version 0.9.1 --locked\n";
        assert_pins_match("cargo-about@", "0.9.1", Path::new("fixture"), text);
    }

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

    /// Extract the top-level `concurrency.cancel-in-progress` value.
    fn concurrency_cancel_in_progress(text: &str) -> String {
        top_level_block(text, "concurrency")
            .lines()
            .map(strip_comment)
            .find_map(|line| line.trim().strip_prefix("cancel-in-progress:"))
            .map(|value| value.trim().to_owned())
            .expect("concurrency block has a cancel-in-progress: key")
    }

    #[test]
    fn self_hosted_supersession_is_scoped_to_pull_requests() {
        // With one runner per GPU class, a superseded PR run left in the queue
        // delays every run behind it, so a new push must cancel the PR's older
        // run. The same must NOT hold for the other triggers: a push or a
        // merge_group run records a commit that landed, and a dispatch is
        // deliberate, so none may cancel another or share a group with a PR run.
        let sh = read_workflow("e2e-selfhosted.yml");
        let group = concurrency_group(&sh);
        assert!(
            group.contains("${{ github.workflow }}-${{ github.event_name }}-"),
            "the group must carry the event name so runs of different triggers never \
             share a group (got `{group}`)"
        );
        // The key is an `A && x || B && y || z` chain, so the ORDER of the arms
        // decides behaviour (`format()` is always truthy: moved first, it would
        // give every PR run a unique group and silently stop supersession).
        // Pin the whole expression, not each arm, so a reorder fails.
        // PR number: a new push to the PR supersedes its older run. Dispatch:
        // unique per run. Ref AND commit for everything else: GitHub replaces a
        // still-pending run in a shared group even with cancel-in-progress off,
        // so a ref-only key would let a second push to `main` drop the first.
        assert_eq!(
            normalized_whitespace(&group).trim_start_matches(">- "),
            "${{ github.workflow }}-${{ github.event_name }}-${{ \
             github.event_name == 'pull_request' && github.event.pull_request.number \
             || github.event_name == 'workflow_dispatch' && github.run_id \
             || format('{0}-{1}', github.ref, github.sha) }}",
            "the concurrency group must be exactly: event name, then PR number \
             (pull_request), run id (workflow_dispatch), ref and commit (all else), \
             in that order"
        );
        assert_eq!(
            concurrency_cancel_in_progress(&sh),
            "${{ github.event_name == 'pull_request' }}",
            "only pull_request runs may be cancelled by a newer run"
        );
    }

    #[test]
    fn mi300x_lane_is_skipped_in_the_merge_queue() {
        // The MI300X check is not required for merging, so a merge_group run's
        // job is never waited for and, behind a single runner, outlives the PR it
        // was queued for. The other lanes are unaffected.
        let sh = read_workflow("e2e-selfhosted.yml");
        // The sibling lanes' "keep running in the queue" check below is vacuous
        // if the workflow stops triggering on merge_group at all.
        let triggers = normalized_whitespace(
            &top_level_block(&sh, "on")
                .lines()
                .map(strip_comment)
                .collect::<Vec<_>>()
                .join("\n"),
        );
        assert!(
            triggers.contains("merge_group: branches: [main] types: [checks_requested]"),
            "e2e-selfhosted.yml must still trigger on merge_group (checks_requested on main) \
             so the lanes that stay in the merge queue actually run (got `{triggers}`)"
        );
        let gate = |job: &str| {
            let block = job_block(&sh, job);
            let start = block
                .find("    if: >-\n")
                .unwrap_or_else(|| panic!("`{job}` has a folded `if:` gate"));
            block[start..]
                .lines()
                .skip(1)
                .take_while(|line| indent_of(line) > 4)
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert!(
            normalized_whitespace(&gate("e2e-gpu")).starts_with(
                "always() && needs.changes.result == 'success' \
                 && github.event_name != 'merge_group' && ("
            ),
            "e2e-gpu (MI300X) must not run on merge_group events"
        );
        assert!(
            !job_block(&sh, "e2e-gpu").contains("E2E_MERGE_QUEUE"),
            "e2e-gpu never runs in the merge queue, so it must not opt into @merge-queue scenarios"
        );
        for job in [
            "e2e-gpu-strix-ubuntu",
            "e2e-gpu-strix-windows",
            "e2e-wsl",
            "e2e-gpu-w7900",
        ] {
            // Only the dispatch split may look at the event: any other event
            // condition could silently keep the lane out of the merge queue.
            let gate = normalized_whitespace(&gate(job));
            assert!(
                !gate.contains("merge_group") && !gate.contains("'pull_request'"),
                "`{job}` must keep running in the merge queue (gate: `{gate}`)"
            );
        }
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

    /// The E2E-owned roots declared in `scripts/reclaim-gpu.sh`.
    ///
    /// Parsed rather than duplicated: a copy here would drift the same way the
    /// PowerShell mirrors can, which is the defect this test exists to prevent.
    fn reclaim_script_roots() -> Vec<String> {
        let p = repo_root().join("scripts/reclaim-gpu.sh");
        let text = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
            .replace("\r\n", "\n");
        parse_e2e_roots(&text, &p.display().to_string())
    }

    /// The parsing half of [`reclaim_script_roots`], split out so every spelling
    /// it must cope with is a standing test rather than a mutation somebody has
    /// to remember to run by hand against the tracked script.
    ///
    /// That split is the point. The previous version of this parser could only
    /// be exercised by editing `scripts/reclaim-gpu.sh` itself, so three legal
    /// spellings of the same array reached review unnoticed — and the failure
    /// they produced blamed the PowerShell mirrors, advising a maintainer to
    /// delete a real root from Windows.
    fn parse_e2e_roots(text: &str, origin: &str) -> Vec<String> {
        // Comments come off BEFORE anything is located — before the array's
        // START as well as its closing `)`. Two distinct mis-parses, one fix:
        //
        //   - an inline comment containing a `)` — `'/tmp/rocm-e2e'  # see docs
        //     (section 3)` — closes the array at that paren, so every root below
        //     it vanishes from the parse. The test then fails in the
        //     mirror->script direction naming a root that is plainly still in
        //     the script, and a maintainer who follows that message deletes a
        //     real root from both PowerShell mirrors. Measured: that spelling
        //     made this test demand the deletion of `e2e-prewarm`, the root this
        //     whole change exists to add.
        //   - a comment ABOVE the array that merely quotes the literal text
        //     `E2E_ROOTS=(` would otherwise anchor the search inside that
        //     sentence rather than at the declaration. This file already quotes
        //     bash array syntax in prose (`E2E_ROOTS+=(` appears in a comment),
        //     so that is a spelling a future header edit can reach.
        let decommented = text
            .lines()
            .map(|l| {
                if l.trim_start().starts_with('#') {
                    ""
                } else {
                    strip_comment(l)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let body = decommented
            .split_once("E2E_ROOTS=(")
            .unwrap_or_else(|| panic!("{origin} must declare E2E_ROOTS=("))
            .1
            .split_once(')')
            .unwrap_or_else(|| panic!("{origin} has an unterminated E2E_ROOTS array"))
            .0;

        // Every non-empty line inside the array must parse into exactly one
        // root. This is the load-bearing half: without it an unrecognised
        // spelling is silently DROPPED, and a dropped root is indistinguishable
        // from mirror drift downstream — so the failure arrives as a confident
        // wrong remedy rather than as "this parser did not understand the
        // array". A checker that converts a formatting change into advice that
        // removes Windows coverage is worse than no checker on that axis.
        //
        // Deliberately NOT tokenised on whitespace: that would silently split a
        // future root containing a space into two bogus roots, which is this
        // same defect class in a new place.
        let mut roots: Vec<String> = Vec::new();
        for line in body.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Both bash quotings are accepted. A quoted root can never contain
            // its OWN delimiter, so requiring that is what rejects the whole
            // array folded onto one line — `E2E_ROOTS=('a' 'b')` parses as the
            // single bogus root `a' 'b` under a prefix/suffix strip alone, and
            // then fails script->mirror naming a root nobody wrote.
            let parsed = ['\'', '"'].into_iter().find_map(|q| {
                line.strip_prefix(q)
                    .and_then(|l| l.strip_suffix(q))
                    .filter(|inner| !inner.contains(q))
            });
            let Some(root) = parsed else {
                // `strip_comment` splits on a literal " #" with no quote
                // awareness, so a root whose VALUE contains that sequence —
                // `'/tmp/e2e root #2'`, which bash itself accepts, since `#` is
                // inert inside single quotes — arrives here already truncated
                // and unterminated. Named in the message rather than fixed:
                // this lands as a loud false failure, never a silent miss, and
                // making the shared helper quote-aware for a spelling no root
                // uses would be untested code on a path every other caller
                // depends on.
                panic!(
                    "{origin}: E2E_ROOTS line `{line}` is not a single quoted root. This \
                     test's parser did not understand it — that is NOT drift against the \
                     PowerShell mirrors, so do not remove anything from them. Restore one \
                     quoted root per line, or teach this parser the new spelling. Note the \
                     line is shown after comment-stripping, so a root whose value contains \
                     \" #\" appears truncated here (EAI-8751)"
                );
            };
            roots.push(root.to_owned());
        }
        assert!(
            !roots.is_empty(),
            "{origin} declared no E2E_ROOTS entries — the parser or the array shape changed"
        );
        roots
    }

    /// An `E2E_ROOTS` array spelled the way the script spells it today, with
    /// enough surrounding file to exercise the comment handling.
    #[cfg(test)]
    fn roots_fixture(body: &str) -> String {
        format!(
            "#!/usr/bin/env bash\n# a header comment\nset -euo pipefail\nE2E_ROOTS=(\n{body})\nENGINE_MARKERS=(\n  'llama-server'\n)\n"
        )
    }

    /// Every spelling this parser must cope with, as a test rather than as a
    /// mutation of the tracked script. Each of these was found by editing
    /// `scripts/reclaim-gpu.sh` by hand during review; none of them could fail
    /// CI afterwards, which is the gap this closes.
    #[test]
    fn e2e_roots_parser_accepts_legal_respellings_of_the_same_array() {
        let canonical = vec!["/tmp/rocm-e2e".to_owned(), "e2e-shared".to_owned()];

        // As written today.
        assert_eq!(
            parse_e2e_roots(
                &roots_fixture("  '/tmp/rocm-e2e'\n  'e2e-shared'\n"),
                "fixture"
            ),
            canonical
        );

        // An inline comment containing a `)`. This closed the array early and
        // made the test demand the deletion of a real root from Windows.
        assert_eq!(
            parse_e2e_roots(
                &roots_fixture("  '/tmp/rocm-e2e'  # see docs (section 3)\n  'e2e-shared'\n"),
                "fixture"
            ),
            canonical
        );

        // Double quotes, and a mix of both. Silently dropped the entry before.
        assert_eq!(
            parse_e2e_roots(
                &roots_fixture("  \"/tmp/rocm-e2e\"\n  'e2e-shared'\n"),
                "fixture"
            ),
            canonical
        );

        // A full-line comment inside the array, indented or at column 0, and a
        // blank line. All three are skipped rather than failing completeness.
        assert_eq!(
            parse_e2e_roots(
                &roots_fixture("  '/tmp/rocm-e2e'\n# flush left\n  # indented\n\n  'e2e-shared'\n"),
                "fixture"
            ),
            canonical
        );

        // A comment ABOVE the array quoting the declaration's own text. The
        // anchor must find the declaration, not the sentence about it.
        let text = "#!/usr/bin/env bash\n# see how E2E_ROOTS=( is declared below\nE2E_ROOTS=(\n  '/tmp/rocm-e2e'\n  'e2e-shared'\n)\n";
        assert_eq!(parse_e2e_roots(text, "fixture"), canonical);

        // A root containing a space is preserved whole, which is what rules out
        // tokenising the body on whitespace.
        assert_eq!(
            parse_e2e_roots(&roots_fixture("  '/tmp/rocm e2e'\n"), "fixture"),
            vec!["/tmp/rocm e2e".to_owned()]
        );
    }

    /// The other half: spellings the parser must REJECT, and reject as its own
    /// misunderstanding rather than as drift against the PowerShell mirrors —
    /// because the mirror-drift message tells a maintainer to delete a root
    /// from Windows, and following it on a false positive is what makes this
    /// worse than having no checker.
    #[test]
    fn e2e_roots_parser_rejects_unparsable_lines_without_blaming_the_mirrors() {
        for (label, body) in [
            (
                "array folded onto one line",
                "  '/tmp/rocm-e2e' 'e2e-shared'\n",
            ),
            ("unquoted entry", "  e2e-shared\n"),
            ("unterminated quote", "  '/tmp/rocm-e2e\n"),
            // `strip_comment` is not quote-aware, so this arrives truncated.
            // Rejected loudly, never silently dropped.
            ("value containing \" #\"", "  '/tmp/rocm e2e #2'\n"),
        ] {
            let text = roots_fixture(body);
            let err = std::panic::catch_unwind(|| parse_e2e_roots(&text, "fixture"))
                .expect_err(&format!("{label} must not parse"));
            let msg = err
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| err.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(
                msg.contains("did not understand"),
                "{label} failed with the wrong message — it must not read as mirror drift: {msg}"
            );
            assert!(
                msg.contains("do not remove anything from them"),
                "{label} must tell the reader NOT to edit the mirrors: {msg}"
            );
        }
    }

    /// An empty array is a parser/shape change, not zero roots — and must not
    /// sail through as "no roots declared, nothing to compare".
    #[test]
    fn e2e_roots_parser_rejects_an_empty_array() {
        let err = std::panic::catch_unwind(|| parse_e2e_roots(&roots_fixture(""), "fixture"))
            .expect_err("an empty E2E_ROOTS must not parse as success");
        let msg = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(msg.contains("declared no E2E_ROOTS entries"), "got: {msg}");
    }

    /// The root half of each PowerShell reclaim: the FIRST `-match '…'`
    /// alternation on the `Where-Object` line.
    ///
    /// This cannot tell the reclaim's own matcher from any other
    /// `Where-Object … -match` line. Each of these workflows has exactly one
    /// today, so every alternation returned IS a reclaim matcher; add a second,
    /// unrelated one and the caller's assertions would be applied to it too.
    ///
    /// Extracted rather than substring-matched against the whole file for the
    /// reason this module's header gives: every root ALSO appears in these
    /// workflows as an env var and in prose, so a `text.contains(root)` check
    /// passes even when the alternation itself has lost that root. Confirmed by
    /// mutation — deleting `e2e-prewarm` from the alternation left the
    /// whole-file form of this test green.
    fn powershell_reclaim_root_alternations(text: &str) -> Vec<String> {
        text.lines()
            .filter(|l| l.contains("Where-Object") && l.contains("-match"))
            .map(|l| {
                let after = l.split_once("-match").expect("filtered on -match").1;
                let body = after
                    .split_once('\'')
                    .unwrap_or_else(|| panic!("no opening quote in matcher line: {l}"))
                    .1;
                body.split_once('\'')
                    .unwrap_or_else(|| panic!("unterminated matcher literal: {l}"))
                    .0
                    .to_owned()
            })
            .collect()
    }

    /// EAI-8751: native Windows has no bash, so the two PowerShell reclaim steps
    /// restate the bash rule instead of sharing it. Nothing else in CI compares
    /// the two, so a root added to the script — as `e2e-prewarm` was, to fix a
    /// leak that held the card for 16 consecutive jobs — can silently miss the
    /// Windows lanes.
    ///
    /// Only the ROOT half is pinned. The engine half is knowingly divergent on
    /// Windows (EAI-8815), so asserting parity there would fail on a difference
    /// that is recorded rather than accidental.
    #[test]
    fn reclaim_roots_are_mirrored_in_the_powershell_reclaims() {
        let roots = reclaim_script_roots();
        for workflow in ["e2e-selfhosted.yml", "nightly.yml"] {
            let text = read_workflow(workflow);
            assert!(
                text.contains("Get-CimInstance Win32_Process"),
                "{workflow} must keep a PowerShell reclaim step (EAI-8751)"
            );
            let alternations = powershell_reclaim_root_alternations(&text);
            assert!(
                !alternations.is_empty(),
                "{workflow} has a PowerShell reclaim but no parsable `-match` alternation \
                 — the step's shape changed and this guard went blind (EAI-8751)"
            );
            // The scenario root is compared on its portable segment, and that
            // asymmetry is deliberate rather than cosmetic.
            //
            // bash anchors it to an absolute path (`/tmp/rocm-e2e`, plus a
            // TMPDIR-derived form appended at run time) because its roots are
            // matched as unanchored substrings of a whole command line: the
            // bare segment would also match an ARGUMENT naming it, and kill a
            // hand-run serve the script promises to spare. The PowerShell
            // mirrors carry the bare segment and so do have that exposure —
            // pre-existing, and not something this test can fix by failing.
            //
            // That anchoring gap is NOT what EAI-8815 tracks. That ticket is
            // scoped to the engine-marker divergence (`rocm.exe daemon`
            // unmatched; `__engine-serve-http` sitting in the root alternation
            // rather than the engine one), and closing it would leave the
            // substring exposure untouched. The anchoring gap is untracked on
            // the Windows side — said plainly here, so that closing EAI-8815
            // cannot be misread as closing this as well.
            //
            // What it still pins is the part that matters here: that every root
            // the script knows about is named in both mirrors, so a root added
            // to one side cannot silently miss the Windows lanes.
            //
            // CAVEAT: the `/tmp/` prefix below is the only absolute prefix this
            // mapping knows. A future root anchored under any other one — say
            // `/var/tmp/rocm-e2e` — would be compared to the mirrors verbatim,
            // fail against their bare segment, and need this line edited by
            // hand: the manual sync this test exists to remove, reappearing one
            // level up. Named rather than generalised, because there is exactly
            // one anchored root today and a speculative prefix list would be
            // untested code.
            let expected: Vec<&str> = roots
                .iter()
                .map(|r| r.strip_prefix("/tmp/").unwrap_or(r))
                .collect();

            // Pinned at one, which is what lets the exemption below be tracked
            // per WORKFLOW rather than per alternation. Requiring EVERY
            // alternation to name the exempted marker would fail spuriously the
            // moment an unrelated `Where-Object … -match` line appeared — the
            // case `powershell_reclaim_root_alternations` warns it cannot
            // distinguish — but relaxing it to "some alternation" would stop
            // catching a second matcher that copies every root and drops the
            // marker. While there is exactly one, the two readings coincide;
            // this assertion is what keeps that true, and fails loudly with
            // something to decide if a second one is ever added.
            assert_eq!(
                alternations.len(),
                1,
                "{workflow} now has {} PowerShell reclaim matcher alternations; the \
                 divergence-exemption check below assumes exactly one, and must be \
                 re-read per alternation before this count changes (EAI-8751)",
                alternations.len()
            );
            let mut divergence_seen = false;
            for alternation in &alternations {
                let present: Vec<&str> = alternation.split('|').collect();

                // Script -> mirror: a root added to the script must reach Windows.
                for needle in &expected {
                    assert!(
                        present.contains(needle),
                        "{workflow}'s PowerShell reclaim matcher `{alternation}` does not name \
                         the E2E root `{needle}` declared in scripts/reclaim-gpu.sh (EAI-8751)"
                    );
                }

                // Mirror -> script: and a root REMOVED from the script must not be
                // left behind here. Without this direction the guard is one-way,
                // which is how the lists drifted in the first place.
                for token in &present {
                    // Known divergence, not drift: `__engine-serve-http` is an
                    // ENGINE marker that sits in this root alternation, so on
                    // Windows it over-matches. Tracked in EAI-8815 — when that is
                    // fixed, delete this arm and the assertion below will hold.
                    if *token == "__engine-serve-http" {
                        divergence_seen = true;
                        continue;
                    }
                    assert!(
                        expected.contains(token),
                        "{workflow}'s PowerShell reclaim matcher `{alternation}` names `{token}`, \
                         which is not an E2E root in scripts/reclaim-gpu.sh — remove it here too, \
                         or add it there (EAI-8751)"
                    );
                }
            }

            // An exemption nothing asserts is an exemption that rots: once
            // EAI-8815 moves `__engine-serve-http` out of the root alternation,
            // the arm above stops firing and would sit here forever as dead
            // code exempting nothing. Fail instead, so the fix is told to
            // finish the job.
            assert!(
                divergence_seen,
                "no PowerShell reclaim matcher in {workflow} names `__engine-serve-http`, so the \
                 EAI-8815 divergence looks fixed — delete the exemption arm in this test, which \
                 is now dead code (EAI-8751)"
            );
        }
    }

    /// Every `shell: pwsh` step in `text` as `(name, step block)`.
    ///
    /// The step is found from its `shell: pwsh` line by walking up to the
    /// nearest `- ` item shallower than it, so the key order inside the step
    /// does not matter.
    fn pwsh_steps(text: &str) -> Vec<(String, String)> {
        let lines: Vec<&str> = text.lines().collect();
        let mut steps = Vec::new();
        for (at, line) in lines.iter().enumerate() {
            if strip_comment(line).trim() != "shell: pwsh" {
                continue;
            }
            let shell_indent = indent_of(line);
            let Some(start) = (0..at).rev().find(|&i| {
                indent_of(lines[i]) < shell_indent && lines[i].trim_start().starts_with("- ")
            }) else {
                continue;
            };
            let step = step_block(&lines, start);
            steps.push((step_name(&step), step));
        }
        steps
    }

    /// PowerShell words that open a statement but are not a program to run.
    const PWSH_KEYWORDS: [&str; 15] = [
        "if", "elseif", "else", "foreach", "for", "while", "do", "switch", "function", "try",
        "catch", "finally", "throw", "exit", "return",
    ];

    /// Whether a trimmed, comment-free pwsh statement runs an external
    /// program: `python …`, `cargo …`, `gh …`, `.\tool.exe`, `& $exe …`.
    ///
    /// Deliberately rough, and erring towards "native": assignments and
    /// variables (`$x = …`), keywords, braces, string literals piped
    /// somewhere (`"k=v" | Out-File …`) and cmdlets (`Verb-Noun`, which start
    /// upper-case) are not; a lower-case first word that is not a keyword is.
    fn is_native_command_line(line: &str) -> bool {
        let line = line.trim();
        if line.starts_with('&') {
            return true;
        }
        let first = line
            .split(|c: char| c.is_whitespace() || c == '(' || c == '{')
            .next()
            .unwrap_or_default();
        let Some(lead) = first.chars().next() else {
            return false;
        };
        (lead.is_ascii_lowercase() || lead == '.')
            && !PWSH_KEYWORDS.contains(&first.to_ascii_lowercase().as_str())
    }

    /// The statements of a pwsh `run: |` body, in order.
    ///
    /// A line is split at its top-level `;`s (see
    /// [`split_top_level_statements`]), so `Write-Output 'ok'; sccache
    /// --start-server` is two statements. A backtick continuation is joined
    /// onto the line it continues. A here-string (`@'`/`@"` … `'@`/`"@`)
    /// stands in as `@'…'@` for the statement it opens, and whatever follows
    /// the closing marker on its line is analysed as the rest of that
    /// statement: `'@ | Out-File x; cargo build` still yields `cargo build`.
    /// Comments and blank lines are dropped.
    fn pwsh_statements(body: &[&str]) -> Vec<String> {
        let mut statements = Vec::new();
        // The logical line built so far: a line plus any continuation and
        // here-string it carries on into.
        let mut pending = String::new();
        let mut here_string_close: Option<&str> = None;
        for raw in body {
            let line = raw.trim();
            if let Some(close) = here_string_close {
                // Only a line that starts with the opener's own closing
                // marker ends a here-string; anything else is its text.
                let Some(rest) = line.strip_prefix(close) else {
                    continue;
                };
                here_string_close = None;
                pending.push('…');
                pending.push_str(close);
                pending.push_str(rest);
            } else {
                if pending.is_empty() && (line.is_empty() || line.starts_with('#')) {
                    continue;
                }
                if !pending.is_empty() {
                    pending.push(' ');
                }
                pending.push_str(line);
            }
            if let Some(continued) = pending.strip_suffix('`') {
                pending = continued.trim_end().to_owned();
                continue;
            }
            if pending.ends_with("@'") {
                here_string_close = Some("'@");
                continue;
            }
            if pending.ends_with("@\"") {
                here_string_close = Some("\"@");
                continue;
            }
            statements.extend(split_top_level_statements(&pending));
            pending.clear();
        }
        statements.extend(split_top_level_statements(&pending));
        statements
    }

    /// One logical pwsh line split at each `;` that ends a statement: those
    /// outside quotes, outside `(…)`/`{…}` (script blocks, hashtables,
    /// subexpressions — `@{ a = 1; b = 2 }` is one statement), not escaped
    /// with a backtick, and before any `#` comment, which is dropped. Each
    /// statement comes back trimmed; empty ones are skipped.
    fn split_top_level_statements(line: &str) -> Vec<String> {
        let mut statements = Vec::new();
        let mut current = String::new();
        let mut depth = 0usize;
        let mut quote: Option<char> = None;
        // A `#` opens a comment only where a token can start.
        let mut at_token_start = true;
        let mut chars = line.chars();
        while let Some(c) = chars.next() {
            match (quote, c) {
                // Inside '…' nothing is special; `''` is two adjacent strings
                // as far as splitting is concerned, which is the same thing.
                (Some('\''), '\'') | (Some('"'), '"') => quote = None,
                (Some('"') | None, '`') => {
                    current.push(c);
                    if let Some(escaped) = chars.next() {
                        current.push(escaped);
                    }
                    at_token_start = false;
                    continue;
                }
                (Some(_), _) => {}
                (None, '\'' | '"') => quote = Some(c),
                (None, '#') if at_token_start => break,
                (None, '(' | '{') => depth += 1,
                (None, ')' | '}') => depth = depth.saturating_sub(1),
                (None, ';') if depth == 0 => {
                    statements.push(std::mem::take(&mut current));
                    at_token_start = true;
                    continue;
                }
                (None, _) => {}
            }
            current.push(c);
            at_token_start = quote.is_none() && (c.is_whitespace() || "({;|".contains(c));
        }
        statements.push(current);
        statements
            .into_iter()
            .map(|statement| statement.trim().to_owned())
            .filter(|statement| !statement.is_empty())
            .collect()
    }

    /// The native-command statements in a pwsh body whose exit status
    /// nothing reads: not the body's last statement, and not followed by a
    /// statement that inspects `$LASTEXITCODE`.
    fn unchecked_native_commands(body: &[&str]) -> Vec<String> {
        let statements = pwsh_statements(body);
        statements
            .iter()
            .enumerate()
            .filter(|(i, line)| {
                is_native_command_line(line)
                    && statements
                        .get(i + 1)
                        .is_some_and(|next| !next.contains("$LASTEXITCODE"))
            })
            .map(|(_, line)| line.clone())
            .collect()
    }

    /// [`unchecked_native_commands`] over a pwsh step's script: its `run: |`
    /// block, or its one-line `run:` value, which can chain statements with
    /// `;` just the same.
    fn unchecked_native_commands_in_step(step: &str) -> Vec<String> {
        match run_block(step) {
            Some(body) => unchecked_native_commands(&body),
            None => unchecked_native_commands(&[step_key(step, "run").as_str()]),
        }
    }

    /// pwsh steps allowed to run a native command whose status is unread,
    /// keyed by `(workflow, step name)`, each with the reason it is safe.
    const PWSH_UNCHECKED_NATIVE_ALLOWED: [(&str, &str, &str); 3] = [
        (
            "ci.yml",
            "Probe sccache cache backend",
            "the probe decides for itself and ends in `exit 0`; the one unread \
             status is a best-effort `sccache --stop-server` cleanup",
        ),
        (
            "nightly.yml",
            "Package Windows bundle",
            "a failed `cargo xtask package` leaves no zip or signature, so the \
             `Copy-Item`/`throw` on its outputs stops the step under the runner's \
             `$ErrorActionPreference = 'stop'`",
        ),
        (
            "release.yml",
            "Package Windows bundle",
            "a failed `cargo xtask package` leaves no zip or signature, so the \
             `Copy-Item`/`throw` on its outputs stops the step under the runner's \
             `$ErrorActionPreference = 'stop'`",
        ),
    ];

    /// GitHub runs a `shell: pwsh` body with `$ErrorActionPreference = 'stop'`
    /// and then `exit $LASTEXITCODE`. A failing *cmdlet* stops the step, but a
    /// failing *native command* does not: the script carries on, and only the
    /// last native command's status decides the step. So in
    ///
    /// ```text
    /// python a.py --self-test   # exits 1
    /// python b.py --self-test   # exits 0
    /// ```
    ///
    /// the step is green. Every native command that is not the body's last
    /// statement must therefore have its `$LASTEXITCODE` read on the next
    /// line, or be split into its own step (#633), or be allow-listed above
    /// with the reason its failure still surfaces.
    #[test]
    fn no_pwsh_step_lets_a_failing_native_command_pass_unseen() {
        let dir = repo_root().join(".github/workflows");
        let mut workflows: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
            .map(|entry| entry.expect("workflow dir entry").path())
            .filter(|path| {
                path.extension().is_some_and(|ext| {
                    ext.eq_ignore_ascii_case("yml") || ext.eq_ignore_ascii_case("yaml")
                })
            })
            .map(|path| {
                path.file_name()
                    .expect("a read_dir entry has a file name")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        workflows.sort();

        let mut offenders = Vec::new();
        let mut allowed_seen = Vec::new();
        for workflow in &workflows {
            let text = read_workflow(workflow);
            for (name, step) in pwsh_steps(&text) {
                let unchecked = unchecked_native_commands_in_step(&step);
                if unchecked.is_empty() {
                    continue;
                }
                if PWSH_UNCHECKED_NATIVE_ALLOWED
                    .iter()
                    .any(|(w, n, _)| w == workflow && *n == name)
                {
                    allowed_seen.push((workflow.clone(), name));
                    continue;
                }
                offenders.push(format!("{workflow} step {name:?}: {unchecked:?}"));
            }
        }
        assert!(
            offenders.is_empty(),
            "these `shell: pwsh` steps run a native command whose failure the step \
             would not report, because pwsh carries on and the step takes only the last \
             command's exit code — split each into its own step, or check \
             `$LASTEXITCODE` right after it:\n  {}",
            offenders.join("\n  ")
        );
        // An allow-list entry that no longer matches anything exempts nothing
        // and would quietly cover the next step to take that name.
        for (workflow, name, _) in PWSH_UNCHECKED_NATIVE_ALLOWED {
            assert!(
                allowed_seen.iter().any(|(w, n)| w == workflow && n == name),
                "{workflow} step {name:?} is allow-listed in \
                 PWSH_UNCHECKED_NATIVE_ALLOWED but no longer has an unchecked native \
                 command; drop the entry"
            );
        }
    }

    /// The masking shape #633 removed is still caught, and the shapes that
    /// read the status are not.
    #[test]
    fn unchecked_native_commands_finds_the_masking_shape() {
        let merged = [
            "python .\\scripts\\comfyui_therock_gpu_test.py --self-test",
            "python .\\scripts\\local_assistant_therock_gpu_test.py --self-test",
            "python .\\scripts\\vllm_therock_gpu_test.py --self-test",
        ];
        assert_eq!(unchecked_native_commands(&merged), merged[..2].to_vec());

        let checked = [
            "$targetDir = if ($env:X) { $env:X } else { \"target\" }",
            "# a comment",
            "sccache --start-server",
            "if ($LASTEXITCODE -ne 0) {",
            "  Write-Output \"::warning::no cache\"",
            "}",
            "@'",
            "cargo build   # inside a here-string, not a command",
            "'@ | Out-File -FilePath $env:GITHUB_STEP_SUMMARY",
            "cargo xtask e2e",
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
            "python .\\scripts\\release_readiness.py `",
            "  --dist dist",
        ];
        assert!(
            unchecked_native_commands(&checked).is_empty(),
            "{:?}",
            unchecked_native_commands(&checked)
        );
    }

    /// A `;` ends a statement, so a native command chained before another
    /// statement on one line is checked like one on a line of its own — but
    /// only a top-level `;`: not one in a string, a hashtable or script
    /// block, a comment, or escaped with a backtick.
    #[test]
    fn unchecked_native_commands_splits_statements_at_top_level_semicolons() {
        let chained = [
            "Write-Output 'ok'; sccache --start-server",
            "cargo xtask e2e",
        ];
        assert_eq!(
            unchecked_native_commands(&chained),
            ["sccache --start-server"]
        );
        assert_eq!(
            unchecked_native_commands(&["cargo build; cargo test"]),
            ["cargo build"]
        );
        assert!(
            unchecked_native_commands(&["cargo build; if ($LASTEXITCODE -ne 0) { exit 1 }"])
                .is_empty()
        );

        // Each of these is ONE statement; splitting it would leave a
        // lower-case fragment (`cargo`, `b`, `then`) that reads as a native
        // command and, followed by `cargo xtask e2e`, as an unchecked one.
        for line in [
            "Write-Output \"done; cargo is next\"",
            "Write-Output 'done; cargo is next'",
            "Write-Output \"it`\"s done; cargo is next\"",
            "$h = @{ a = 1; b = 2 }",
            "$v = $(Get-Date; Get-Location)",
            "Write-Output done # then; cargo",
            "Write-Output done`; then",
        ] {
            let body = [line, "cargo xtask e2e"];
            assert!(
                unchecked_native_commands(&body).is_empty(),
                "{line:?} split into {:?}",
                pwsh_statements(&body)
            );
        }

        // A one-line `run:` chains statements just the same.
        let step = "      - name: Lint\n        shell: pwsh\n        \
                    run: cargo fmt --check; cargo clippy  # both\n";
        assert_eq!(
            unchecked_native_commands_in_step(step),
            ["cargo fmt --check"]
        );
    }

    /// Only the closing marker of a here-string is skipped: the rest of its
    /// line is the tail of the statement the here-string opened, and any
    /// statement chained after it is checked.
    #[test]
    fn unchecked_native_commands_reads_on_past_a_here_string_close() {
        let body = [
            "@'",
            "## Summary",
            "'@ | Out-File -FilePath $env:GITHUB_STEP_SUMMARY; sccache --start-server",
            "cargo xtask e2e",
        ];
        assert_eq!(
            pwsh_statements(&body),
            [
                "@'…'@ | Out-File -FilePath $env:GITHUB_STEP_SUMMARY",
                "sccache --start-server",
                "cargo xtask e2e",
            ]
        );
        assert_eq!(unchecked_native_commands(&body), ["sccache --start-server"]);

        // A here-string inside the statement that reads the status leaves
        // that statement whole: `if ($LASTEXITCODE …) { throw @"…"@ }`.
        let checked = [
            "cargo build",
            "if ($LASTEXITCODE -ne 0) { throw @\"",
            "build failed",
            "\"@ }",
            "cargo xtask e2e",
        ];
        assert!(
            unchecked_native_commands(&checked).is_empty(),
            "{:?}",
            pwsh_statements(&checked)
        );

        // `"@` does not close a `@'` here-string: its text runs on to `'@`.
        let mismatched = [
            "@'",
            "\"@; cargo inside the text",
            "'@ | Out-File x",
            "cargo xtask e2e",
        ];
        assert!(
            unchecked_native_commands(&mismatched).is_empty(),
            "{:?}",
            pwsh_statements(&mismatched)
        );
    }
}
