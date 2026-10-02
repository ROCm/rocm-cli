// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Guards against the hawkeye version documented in CONTRIBUTING.md drifting
//! from the version `ci.yml`'s `license-headers` job installs.
//!
//! CONTRIBUTING.md tells contributors to `cargo install hawkeye@<version>
//! --locked` so the local pre-commit hook produces the same verdict as CI's
//! pinned prebuilt binary. Nothing else ties those two strings together, so a
//! version bump in one and not the other would silently let contributors pass
//! locally on a hawkeye build that can still fail in CI (or vice versa). This
//! checks only that the two version strings agree — it does not check the
//! SHA256 pin, the `licenserc.toml` rules, or which file types the local hook
//! scans (see `.pre-commit-config.yaml`'s `license-headers` hook for that).

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    fn repo_root() -> PathBuf {
        // CARGO_MANIFEST_DIR is the xtask/ crate dir; its parent is the repo
        // root (same idiom as verify_pinned_keys::repo_root).
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Strip a trailing `# …` comment from a YAML line (same idiom as
    /// `workflow_contract.rs`'s `strip_comment`).
    fn strip_comment(line: &str) -> &str {
        line.split_once(" #").map_or(line, |(v, _)| v)
    }

    /// Extract one top-level job's complete YAML block by its job id. A local
    /// copy of `workflow_contract.rs`'s `job_block`: that extractor is private
    /// to its own `#[cfg(test)]` mod, and this guard stays in its own file
    /// (see the module doc above) rather than reaching into it.
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

    /// The anchored value of a scalar `key: value` entry directly under a
    /// job's `env:` mapping.
    ///
    /// Scoped to the `env:` block rather than a whole-file search, so a
    /// mention elsewhere in the file (another job, a comment) can't
    /// false-pass; and the full line value is extracted for the caller to
    /// compare by equality, not matched as a substring, so a version suffix
    /// (e.g. `v7.0.0-rc1`) can't satisfy a check meant for `v7.0.0`.
    fn job_env_value(block: &str, key: &str) -> String {
        let marker = "    env:";
        let lines: Vec<&str> = block.lines().collect();
        let env_start = lines
            .iter()
            .position(|line| *line == marker)
            .unwrap_or_else(|| panic!("job has no `env:` mapping"));
        let key_marker = format!("{key}:");
        lines[env_start + 1..]
            .iter()
            .take_while(|line| line.starts_with("      "))
            .find_map(|line| strip_comment(line).trim().strip_prefix(&key_marker))
            .unwrap_or_else(|| panic!("job's `env:` has no `{key}` entry"))
            .trim()
            .to_owned()
    }

    /// The documented pin and the version CI installs must never drift: if
    /// they do, a contributor's local hook can bless a header hawkeye's CI
    /// build would reject, or vice versa.
    ///
    /// The version lives in exactly one place — `ci.yml`'s `license-headers`
    /// job — and is read from there rather than duplicated into a constant
    /// here, so a bump only ever needs the two edits this test actually
    /// guards (ci.yml and CONTRIBUTING.md).
    #[test]
    fn pin_matches_ci_workflow() {
        let root = repo_root();

        let ci = root.join(".github/workflows/ci.yml");
        // Normalized the same way as workflow_contract.rs's `read_workflow`:
        // `job_block`'s marker search is `\n`-literal, and a Windows checkout
        // with `core.autocrlf` can hand back `\r\n` even though the repo
        // standardises on LF (see .gitattributes).
        let ci_text = std::fs::read_to_string(&ci)
            .unwrap_or_else(|e| panic!("reading {}: {e}", ci.display()))
            .replace("\r\n", "\n");
        let job = job_block(&ci_text, "license-headers");
        let installed = job_env_value(job, "HAWKEYE_VERSION");
        // ci.yml's env var is `v`-prefixed (`v7.0.0`); CONTRIBUTING.md's prose
        // is not (`hawkeye@7.0.0`).
        let version = installed.strip_prefix('v').unwrap_or(&installed);

        let contributing = root.join("CONTRIBUTING.md");
        let contributing_text = std::fs::read_to_string(&contributing)
            .unwrap_or_else(|e| panic!("reading {}: {e}", contributing.display()));
        let documented = format!("cargo install hawkeye@{version} --locked");
        assert!(
            contributing_text.contains(&documented),
            "CONTRIBUTING.md must instruct `{documented}` to match ci.yml's \
             license-headers job (HAWKEYE_VERSION: {installed})"
        );
    }
}
