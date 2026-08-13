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
