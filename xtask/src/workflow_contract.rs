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

    fn backticked_list_between(text: &str, prefix: &str, suffix: &str) -> Vec<String> {
        let section = text
            .split_once(prefix)
            .unwrap_or_else(|| panic!("section starts with `{prefix}`"))
            .1
            .split_once(suffix)
            .unwrap_or_else(|| panic!("section ends with `{suffix}`"))
            .0;
        section
            .split('`')
            .enumerate()
            .filter_map(|(i, item)| (i % 2 == 1).then_some(item.to_owned()))
            .collect()
    }

    fn normalized_whitespace(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
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

    #[test]
    fn dispatchable_wsl_nightly_run_has_the_full_nightly_job_budget() {
        let self_hosted = read_workflow("e2e-selfhosted.yml");
        let nightly = read_workflow("nightly.yml");
        let dispatch_timeout = job_scalar(job_block(&self_hosted, "e2e-wsl"), "timeout-minutes");
        let nightly_timeout = job_scalar(job_block(&nightly, "e2e-wsl-nightly"), "timeout-minutes");
        assert_eq!(
            dispatch_timeout, "90",
            "the 2400s large-model readiness budget needs the established 90-minute job cap for setup and the remaining suite"
        );
        assert_eq!(
            dispatch_timeout, nightly_timeout,
            "e2e-wsl supports include_nightly, so its job timeout must cover the same 2400s scenario plus setup as e2e-wsl-nightly"
        );
    }

    #[test]
    fn hardware_testing_docs_cover_all_four_self_hosted_platforms() {
        let docs = std::fs::read_to_string(repo_root().join("docs/ci-hardware-testing.md"))
            .expect("read hardware testing docs");
        let rows = markdown_table_rows(&docs, "| Job | Workflow | Platform | Runner labels |");
        let self_hosted_job_platforms: Vec<(String, String)> = rows
            .into_iter()
            .filter(|row| {
                row.get(1)
                    .is_some_and(|workflow| workflow == "`e2e-selfhosted.yml`")
            })
            .map(|row| (row[0].clone(), row[2].clone()))
            .collect();
        assert_eq!(
            self_hosted_job_platforms,
            vec![
                (
                    "`e2e-gpu`".to_owned(),
                    "MI300X (AMD Instinct, bare-metal Linux)".to_owned(),
                ),
                (
                    "`e2e-gpu-strix-ubuntu`".to_owned(),
                    "Strix Halo (gfx1151) on Ubuntu".to_owned(),
                ),
                (
                    "`e2e-gpu-strix-windows`".to_owned(),
                    "Strix Halo (gfx1151) on native Windows 11".to_owned(),
                ),
                (
                    "`e2e-wsl`".to_owned(),
                    "Strix Halo (gfx1151) on Ubuntu under WSL2".to_owned(),
                ),
            ],
            "hardware testing table must document the four actual self-hosted job/platform rows"
        );

        let artifacts = backticked_list_between(
            &docs,
            "The lane artifacts are named canonically (",
            ") in every workflow",
        );
        assert_eq!(
            artifacts,
            vec![
                "e2e-report",
                "e2e-gpu-report",
                "e2e-gpu-strix-ubuntu-report",
                "e2e-gpu-strix-windows-report",
                "e2e-gpu-strix-wsl-report",
            ],
            "the canonical artifact list must enumerate every report platform exactly once"
        );

        let readme = std::fs::read_to_string(repo_root().join("tests/e2e-cucumber/README.md"))
            .expect("read E2E README");
        assert!(
            normalized_whitespace(&readme).contains(
                "The nightly workflow runs four non-blocking jobs — MI300X plus Strix Halo on Ubuntu, Windows, and WSL2 — with `E2E_INCLUDE_NIGHTLY=1`"
            ),
            "E2E README must identify all four nightly job platforms"
        );
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
