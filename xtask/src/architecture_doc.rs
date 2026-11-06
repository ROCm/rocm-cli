// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Require every production source file to appear in `docs/architecture.md`.
//!
//! `docs/architecture.md` is the contributor-facing module map and is meant to
//! be "updated in the same PR as the code it documents". Nothing enforced that
//! until now. The `lychee` link check (the `docs-links` CI job and the `lychee`
//! prek hook) runs the other way round: it confirms that every link already in
//! the doc still resolves, so a moved or deleted file is caught, but a file the
//! doc never mentions has no link to check and passes silently.
//!
//! This check starts from the source tree instead. Every tracked `.rs` file
//! under `apps/*/src`, `crates/*/src` and `engines/*/src` must be cited by a
//! relative markdown link in `docs/architecture.md`, or be listed in
//! [`EXEMPT`], or be a test-only helper (see [`is_test_helper`]). It reads the
//! index and the working tree, never a git diff, so it behaves the same in CI,
//! in the pre-commit hook and in the pre-push hook.
//!
//! [`EXEMPT`] is the files that existed, uncited, when the check was added. It
//! follows the pattern of `ALLOWLIST` in [`crate::crate_edges`] and is checked
//! in both directions: a source file in neither the doc nor the list fails,
//! and so does a list entry that is now cited or no longer exists. The list can
//! therefore only shrink by documenting a file, and growing it is a reviewed
//! change to this file.
//!
//! What this does not check: whether an entry's prose still describes the file,
//! and functions that move between two files that are both already cited.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::paths;

/// The module map, relative to the workspace root.
const DOC: &str = "docs/architecture.md";

/// The directory `DOC` lives in. Its links are relative to this.
const DOC_DIR: &str = "docs";

/// Source roots whose `<crate>/src` trees are in scope.
const SOURCE_ROOTS: [&str; 3] = ["apps", "crates", "engines"];

/// Production files that have no entry in `docs/architecture.md` yet: everything
/// that was uncited when this check was added, mostly the dashboard crates.
/// Do not add to this list to make the check pass: document the file in the doc
/// instead. An entry belongs here only when a file deliberately has no entry,
/// and that is a reviewed decision. Remove an entry when the file gets one.
/// Keep it sorted.
const EXEMPT: &[&str] = &[
    "apps/rocmd/src/main.rs",
    "crates/rocm-dash-collectors/src/amd_smi.rs",
    "crates/rocm-dash-collectors/src/bench_load.rs",
    "crates/rocm-dash-collectors/src/bench_tail.rs",
    "crates/rocm-dash-collectors/src/cgroup.rs",
    "crates/rocm-dash-collectors/src/docker.rs",
    "crates/rocm-dash-collectors/src/engine_registry.rs",
    "crates/rocm-dash-collectors/src/host.rs",
    "crates/rocm-dash-collectors/src/lemonade.rs",
    "crates/rocm-dash-collectors/src/parallel.rs",
    "crates/rocm-dash-collectors/src/proc_scan.rs",
    "crates/rocm-dash-collectors/src/sysfs.rs",
    "crates/rocm-dash-collectors/src/vllm_log.rs",
    "crates/rocm-dash-collectors/src/vllm_prom.rs",
    "crates/rocm-dash-core/src/bench_rollup.rs",
    "crates/rocm-dash-core/src/bench_schema.rs",
    "crates/rocm-dash-core/src/config.rs",
    "crates/rocm-dash-core/src/efficiency.rs",
    "crates/rocm-dash-core/src/metrics.rs",
    "crates/rocm-dash-core/src/observation.rs",
    "crates/rocm-dash-core/src/partition.rs",
    "crates/rocm-dash-core/src/persist.rs",
    "crates/rocm-dash-core/src/protocol.rs",
    "crates/rocm-dash-core/src/state.rs",
    "crates/rocm-dash-core/src/traits.rs",
    "crates/rocm-dash-core/src/vram.rs",
    "crates/rocm-dash-daemon/src/bench_ring.rs",
    "crates/rocm-dash-daemon/src/demo.rs",
    "crates/rocm-dash-daemon/src/persist.rs",
    "crates/rocm-dash-daemon/src/registry.rs",
    "crates/rocm-dash-daemon/src/snapshot_ring.rs",
    "crates/rocm-dash-daemon/src/transport.rs",
    "crates/rocm-dash-tui/src/client.rs",
    "crates/rocm-dash-tui/src/jobs.rs",
    "crates/rocm-dash-tui/src/lib.rs",
    "crates/rocm-dash-tui/src/llm.rs",
    "crates/rocm-dash-tui/src/reconnect.rs",
    "crates/rocm-dash-tui/src/registry_probe.rs",
    "crates/rocm-dash-tui/src/replay.rs",
    "crates/rocm-dash-tui/src/tool_exec.rs",
    "crates/rocm-dash-tui/src/transport.rs",
    "crates/rocm-dash-tui/src/ui/automations_manager.rs",
    "crates/rocm-dash-tui/src/ui/bench.rs",
    "crates/rocm-dash-tui/src/ui/bench_run.rs",
    "crates/rocm-dash-tui/src/ui/command_screen.rs",
    "crates/rocm-dash-tui/src/ui/config_manager.rs",
    "crates/rocm-dash-tui/src/ui/core_bars.rs",
    "crates/rocm-dash-tui/src/ui/dock.rs",
    "crates/rocm-dash-tui/src/ui/engine_manager.rs",
    "crates/rocm-dash-tui/src/ui/examine_manager.rs",
    "crates/rocm-dash-tui/src/ui/folder_browser.rs",
    "crates/rocm-dash-tui/src/ui/format.rs",
    "crates/rocm-dash-tui/src/ui/gradient.rs",
    "crates/rocm-dash-tui/src/ui/install_manager.rs",
    "crates/rocm-dash-tui/src/ui/instance_list.rs",
    "crates/rocm-dash-tui/src/ui/launcher.rs",
    "crates/rocm-dash-tui/src/ui/logs_view.rs",
    "crates/rocm-dash-tui/src/ui/model_picker.rs",
    "crates/rocm-dash-tui/src/ui/onboarding.rs",
    "crates/rocm-dash-tui/src/ui/runtime_manager.rs",
    "crates/rocm-dash-tui/src/ui/serve_wizard.rs",
    "crates/rocm-dash-tui/src/ui/services_manager.rs",
    "crates/rocm-dash-tui/src/ui/sparkline.rs",
    "crates/rocm-dash-tui/src/ui/tabs/bench.rs",
    "crates/rocm-dash-tui/src/ui/tabs/chat.rs",
    "crates/rocm-dash-tui/src/ui/tabs/hardware.rs",
    "crates/rocm-dash-tui/src/ui/tabs/home.rs",
    "crates/rocm-dash-tui/src/ui/tabs/mod.rs",
    "crates/rocm-dash-tui/src/ui/tabs/observe.rs",
    "crates/rocm-dash-tui/src/ui/tabs/pane.rs",
    "crates/rocm-dash-tui/src/ui/tabs/rocm.rs",
    "crates/rocm-dash-tui/src/ui/tabs/serving.rs",
    "crates/rocm-dash-tui/src/ui/update_manager.rs",
    "crates/rocm-deps/src/lib.rs",
    "crates/rocm-engine-protocol/src/lib.rs",
    "engines/aim/src/main.rs",
    "engines/lemonade/src/main.rs",
    "engines/vllm/src/main.rs",
];

/// Whether `path` is a tracked-file path this check covers: a `.rs` file under
/// `<root>/<crate>/src/` for one of the [`SOURCE_ROOTS`].
#[allow(clippy::case_sensitive_file_extension_comparisons)] // Rust sources are always lowercase `.rs`
fn in_scope(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    parts.len() >= 4
        && SOURCE_ROOTS.contains(&parts[0])
        && parts[2] == "src"
        && path.ends_with(".rs")
}

/// Whether `path` is a test-only helper that does not need an architecture
/// entry: shared test support, or a property-test module. Matched on the file
/// stem so a new helper of the same kind never needs an [`EXEMPT`] entry.
fn is_test_helper(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = name.strip_suffix(".rs").unwrap_or(name);
    matches!(stem, "test_support" | "test_env" | "proptests")
        || stem.ends_with("_proptests")
        || stem.ends_with("_props")
        || stem.ends_with("_properties")
}

/// Resolve one markdown link target found in [`DOC`] to a repo-relative `.rs`
/// path, or `None` if it is not a link to a source file inside the repo.
#[allow(clippy::case_sensitive_file_extension_comparisons)] // Rust sources are always lowercase `.rs`
fn resolve_link(target: &str) -> Option<String> {
    let target = target.split('#').next()?;
    if target.contains("://") || !target.ends_with(".rs") {
        return None;
    }
    let mut parts: Vec<&str> = vec![DOC_DIR];
    for segment in target.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                // Climbing out of the repo root names nothing in the tree.
                parts.pop()?;
            }
            other => parts.push(other),
        }
    }
    Some(parts.join("/"))
}

/// Every `.rs` file [`DOC`] links to, as repo-relative paths.
fn cited_paths(doc: &str) -> BTreeSet<String> {
    let mut cited = BTreeSet::new();
    let mut rest = doc;
    while let Some(at) = rest.find("](") {
        rest = &rest[at + 2..];
        let end = rest.find([')', ' ', '\n']).unwrap_or(rest.len());
        if let Some(path) = resolve_link(&rest[..end]) {
            cited.insert(path);
        }
    }
    cited
}

/// Check the tracked files against the doc's citations and `exempt`, returning
/// every violation found.
fn check(tracked: &[String], cited: &BTreeSet<String>, exempt: &[&str]) -> Vec<String> {
    let in_tree: BTreeSet<&str> = tracked
        .iter()
        .map(String::as_str)
        .filter(|path| in_scope(path))
        .collect();

    let mut violations = Vec::new();

    for path in &in_tree {
        if !cited.contains(*path) && !exempt.contains(path) && !is_test_helper(path) {
            violations.push(format!(
                "`{path}` is not cited in {DOC}: add an entry for it to the module map, as a \
                 relative markdown link. If it deliberately has no entry, add it to EXEMPT in \
                 xtask/src/architecture_doc.rs instead"
            ));
        }
    }

    for entry in exempt {
        if !in_tree.contains(entry) {
            violations.push(format!(
                "EXEMPT entry `{entry}` is stale: no tracked source file has that path any more; \
                 remove it from EXEMPT in xtask/src/architecture_doc.rs"
            ));
        } else if cited.contains(*entry) {
            violations.push(format!(
                "EXEMPT entry `{entry}` is stale: {DOC} now cites it; remove it from EXEMPT in \
                 xtask/src/architecture_doc.rs so a later removal from the doc is caught"
            ));
        } else if is_test_helper(entry) {
            violations.push(format!(
                "EXEMPT entry `{entry}` is redundant: test-only helpers are exempt by name; \
                 remove it from EXEMPT in xtask/src/architecture_doc.rs"
            ));
        }
    }

    violations
}

/// Tracked files under the [`SOURCE_ROOTS`], as the index lists them.
fn tracked_files(root: &Path) -> Result<Vec<String>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--"])
        .args(SOURCE_ROOTS)
        .output()
        .context("failed to run `git ls-files`")?;
    if !output.status.success() {
        bail!(
            "`git ls-files` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect())
}

/// Run the check against the real workspace.
fn violations_in(root: &Path) -> Result<Vec<String>> {
    let doc =
        std::fs::read_to_string(root.join(DOC)).with_context(|| format!("failed to read {DOC}"))?;
    Ok(check(&tracked_files(root)?, &cited_paths(&doc), EXEMPT))
}

/// Fail if a production source file has no entry in `docs/architecture.md`, or
/// [`EXEMPT`] has gone stale.
pub fn run() -> Result<()> {
    let root = paths::workspace_root()?;
    let violations = violations_in(&root)?;
    if !violations.is_empty() {
        bail!(
            "{DOC} is out of step with the source tree ({} problem(s)):\n{}",
            violations.len(),
            violations.join("\n")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(paths: &[&str]) -> Vec<String> {
        paths.iter().map(|path| (*path).to_owned()).collect()
    }

    fn cites(paths: &[&str]) -> BTreeSet<String> {
        paths.iter().map(|path| (*path).to_owned()).collect()
    }

    #[test]
    fn cited_file_passes() {
        let tracked = files(&["apps/rocm/src/cli.rs"]);
        assert!(check(&tracked, &cites(&["apps/rocm/src/cli.rs"]), &[]).is_empty());
    }

    #[test]
    fn new_uncited_file_fails_and_is_named() {
        let tracked = files(&["apps/rocm/src/cli.rs", "apps/rocm/src/brand_new.rs"]);
        let violations = check(&tracked, &cites(&["apps/rocm/src/cli.rs"]), &[]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("apps/rocm/src/brand_new.rs"));
        assert!(violations[0].contains("is not cited"));
    }

    #[test]
    fn a_citation_does_not_cover_the_same_name_in_another_crate() {
        let tracked = files(&["apps/rocm/src/cli.rs", "apps/rocmd/src/cli.rs"]);
        let violations = check(&tracked, &cites(&["apps/rocm/src/cli.rs"]), &[]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("apps/rocmd/src/cli.rs"));
    }

    #[test]
    fn exempt_file_passes() {
        let tracked = files(&["crates/rocm-dash-core/src/state.rs"]);
        let exempt = ["crates/rocm-dash-core/src/state.rs"];
        assert!(check(&tracked, &cites(&[]), &exempt).is_empty());
    }

    #[test]
    fn exempt_entry_for_a_missing_file_is_stale() {
        let exempt = ["crates/rocm-dash-core/src/gone.rs"];
        let violations = check(&files(&[]), &cites(&[]), &exempt);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("crates/rocm-dash-core/src/gone.rs"));
        assert!(violations[0].contains("no tracked source file"));
    }

    #[test]
    fn exempt_entry_that_is_now_cited_is_stale() {
        let tracked = files(&["crates/rocm-dash-core/src/state.rs"]);
        let exempt = ["crates/rocm-dash-core/src/state.rs"];
        let cited = cites(&["crates/rocm-dash-core/src/state.rs"]);
        let violations = check(&tracked, &cited, &exempt);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("now cites it"));
    }

    #[test]
    fn exempt_entry_for_a_test_helper_is_redundant() {
        let tracked = files(&["crates/rocm-core/src/test_support.rs"]);
        let exempt = ["crates/rocm-core/src/test_support.rs"];
        let violations = check(&tracked, &cites(&[]), &exempt);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("redundant"));
    }

    #[test]
    fn files_outside_the_source_trees_are_ignored() {
        let tracked = files(&[
            "apps/rocm/tests/broken_pipe.rs",
            "apps/rocm/build.rs",
            "crates/rocm-dash-tui/examples/gen_cast.rs",
            "xtask/src/anything.rs",
            "apps/rocm/src/data.json",
            "tests/e2e-cucumber/src/lib.rs",
        ]);
        assert!(check(&tracked, &cites(&[]), &[]).is_empty());
    }

    #[test]
    fn nested_source_files_are_in_scope() {
        let tracked = files(&["crates/rocm-dash-tui/src/ui/tabs/home.rs"]);
        let violations = check(&tracked, &cites(&[]), &[]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("ui/tabs/home.rs"));
    }

    #[test]
    fn test_helpers_need_no_entry() {
        let tracked = files(&[
            "apps/rocm/src/test_support.rs",
            "apps/rocm/src/therock/test_support.rs",
            "crates/rocm-core/src/test_env.rs",
            "crates/rocm-core/src/archive_props.rs",
            "crates/rocm-core/src/process_tree_properties.rs",
            "crates/rocm-core/src/examine_driver_proptests.rs",
            "apps/rocm/src/providers/local_http/proptests.rs",
        ]);
        assert!(check(&tracked, &cites(&[]), &[]).is_empty());
    }

    #[test]
    fn helper_matching_is_by_whole_file_stem() {
        // Names that merely contain a helper word are production files.
        for path in [
            "apps/rocm/src/test_supporter.rs",
            "apps/rocm/src/properties.rs",
            "apps/rocm/src/gpu_props_view.rs",
            "apps/rocm/src/proptests_runner.rs",
        ] {
            assert!(
                !is_test_helper(path),
                "{path} must not be treated as a helper"
            );
        }
    }

    #[test]
    fn links_resolve_relative_to_the_doc() {
        let doc = "[cli](../apps/rocm/src/cli.rs) and [x](./../crates/a/src/b.rs#L1)";
        assert_eq!(
            cited_paths(doc),
            cites(&["apps/rocm/src/cli.rs", "crates/a/src/b.rs"])
        );
    }

    #[test]
    fn only_in_repo_rust_links_count_as_citations() {
        let doc = "[md](../README.md) [url](https://example.com/a/src/x.rs) \
                   [out](../../outside/src/x.rs) [bare](apps/rocm/src/cli.rs)";
        assert_eq!(cited_paths(doc), cites(&["docs/apps/rocm/src/cli.rs"]));
    }

    #[test]
    fn a_link_with_a_title_is_still_a_citation() {
        let doc = "[cli](../apps/rocm/src/cli.rs \"the command tree\")";
        assert_eq!(cited_paths(doc), cites(&["apps/rocm/src/cli.rs"]));
    }

    #[test]
    fn exempt_list_is_sorted_and_unique() {
        let mut sorted = EXEMPT.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, EXEMPT, "keep EXEMPT sorted, without duplicates");
    }

    #[test]
    fn current_tree_matches_the_architecture_doc() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory");
        let violations = violations_in(root).expect("git and the doc are readable in a checkout");
        assert!(
            violations.is_empty(),
            "docs/architecture.md is out of step with the source tree:\n{}",
            violations.join("\n")
        );
    }
}
