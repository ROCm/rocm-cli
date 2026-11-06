// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Code-relation-based test selection: given a git range, print the set of
//! workspace crates affected by the change — the crates whose files changed plus
//! every crate that transitively depends on them — as `cargo` package-selection
//! flags. CI (and developers) pipe the output into `cargo test`/`cargo nextest
//! run` so a one-line change in a leaf crate no longer rebuilds and re-tests the
//! whole workspace.
//!
//! The selection is deliberately **conservative**: any changed file that cannot
//! be confidently attributed to a single crate (the lockfile, the toolchain
//! file, the workspace root manifest, CI config, a doc an xtask test reads, or
//! any unrecognized path) makes the command fall back to `--workspace`.
//! Skipping a test that should have run is the failure mode this guards
//! against, so when in doubt it runs everything.
//!
//! One rule adds a crate nothing depends on: a changed `proptest-regressions`
//! file, or a Rust file whose diff adds or removes a `proptest!` line, selects
//! `xtask`, which holds the seed guard that reads the whole tree (see
//! `PROPTEST_GUARD_CRATE`).
//!
//! As with [`crate::verify_commits`], the decision logic ([`select`],
//! [`owning_crate`], [`reverse_closure`]) is pure and unit-tested without git or
//! cargo; the I/O lives at the edges ([`load_graph`], [`changed_files`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Default base ref used when none is supplied on the command line. Matches
/// [`crate::verify_commits`] so both commands diff against the same point.
const DEFAULT_BASE: &str = "origin/main";

/// The workspace dependency graph, reduced to what selection needs: each
/// member's source directory (workspace-relative, `/`-separated) and the
/// reverse-dependency edges (crate -> crates that depend on it, directly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    /// `(crate name, source directory)` pairs, source directory relative to the
    /// workspace root using `/` separators and no trailing slash.
    pub crate_dirs: Vec<(String, String)>,
    /// `dep -> [dependents]`: for each crate, the crates that directly depend on
    /// it. A reverse walk over this yields all transitive dependents.
    pub rev_deps: BTreeMap<String, Vec<String>>,
}

/// What to hand to `cargo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// Run the whole workspace (`--workspace`): something changed that could
    /// affect any crate, or could not be attributed to one.
    Workspace,
    /// Run exactly these crates (`-p a -p b ...`), already including dependents.
    Packages(BTreeSet<String>),
    /// Nothing Rust-relevant changed; no crate needs rebuilding.
    Empty,
}

impl Selection {
    /// Render as the trailing portion of a `cargo` invocation.
    ///
    /// `--workspace`, or `-p a -p b`, or the empty string.
    pub fn to_cargo_args(&self) -> String {
        match self {
            Self::Workspace => "--workspace".to_string(),
            Self::Empty => String::new(),
            Self::Packages(pkgs) => pkgs
                .iter()
                .map(|p| format!("-p {p}"))
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

/// Find the workspace crate that owns `file`, by longest directory-prefix match.
///
/// `file` and every `crate_dirs` entry are workspace-relative, `/`-separated.
/// Crate directories are non-empty (a member at the repository root is rejected
/// upstream in [`load_graph`], because a root crate would match every file and
/// silently defeat the conservative full-workspace fallback). Longest match wins
/// so a file under a nested crate is attributed to the nested crate, not an
/// ancestor. Returns `None` when no crate directory is a prefix.
pub fn owning_crate<'a>(file: &str, crate_dirs: &'a [(String, String)]) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for (name, dir) in crate_dirs {
        // Require a real path-segment boundary so "engines/vllm" does not match
        // the sibling "engines/vllm-extra".
        let is_prefix = file == dir || file.strip_prefix(dir).is_some_and(|r| r.starts_with('/'));
        if is_prefix && best.is_none_or(|(_, len)| dir.len() > len) {
            best = Some((name.as_str(), dir.len()));
        }
    }
    best.map(|(name, _)| name)
}

/// Expand `changed` to include every crate that transitively depends on any
/// crate in it, walking the reverse-dependency edges in `rev_deps`.
pub fn reverse_closure(
    changed: &BTreeSet<String>,
    rev_deps: &BTreeMap<String, Vec<String>>,
) -> BTreeSet<String> {
    let mut result: BTreeSet<String> = changed.clone();
    let mut stack: Vec<String> = changed.iter().cloned().collect();
    while let Some(crate_name) = stack.pop() {
        if let Some(dependents) = rev_deps.get(&crate_name) {
            for dependent in dependents {
                if result.insert(dependent.clone()) {
                    stack.push(dependent.clone());
                }
            }
        }
    }
    result
}

/// Markdown files an xtask test reads directly. Each belongs in this list,
/// in ci.yml's `rust` paths-filter bucket, and in the "doc an xtask test
/// reads" sentence in `docs/testing.md` —
/// `forces_full_workspace_docs_are_in_the_rust_filter` (workflow_contract.rs)
/// checks this list against ci.yml's copy.
///
/// Without a file also being in `forces_full_workspace` below, `is_ignorable`
/// would mark it as ordinary Markdown and an edit to it alone would resolve
/// to `Selection::Empty`; without the matching ci.yml filter entry, the
/// rust-gated jobs don't run at all for that edit. Either way the test that
/// reads the file silently never runs:
/// - `MANIFEST.md`: clippy runs `xtask manifest --check` against it, and it
///   carries the cargo-about pin that
///   `pinned_tool_versions_match_across_docs_and_workflows` (xtask) reads.
/// - `CONTRIBUTING.md`: carries the hawkeye and cargo-about pins that same
///   test reads.
/// - `docs/ci-hardware-testing.md` and `tests/e2e-cucumber/README.md`: both
///   read by `hardware_testing_docs_cover_all_self_hosted_platforms` (xtask).
pub(crate) const XTASK_READ_DOCS: &[&str] = &[
    "MANIFEST.md",
    "CONTRIBUTING.md",
    "docs/ci-hardware-testing.md",
    "tests/e2e-cucumber/README.md",
];

/// True for changed files that force a full-workspace run: anything that can
/// affect resolution, the build for every crate, CI itself, or a doc an xtask
/// test reads ([`XTASK_READ_DOCS`]).
fn forces_full_workspace(file: &str) -> bool {
    file == "Cargo.lock"
        || file == "Cargo.toml" // workspace root manifest
        || XTASK_READ_DOCS.contains(&file)
        || file.starts_with("rust-toolchain")
        || file.starts_with(".github/workflows/")
}

/// The workspace member whose tests include the proptest seed guard
/// (`every_proptest_regression_file_belongs_to_a_property_test_that_reads_it`
/// in `proptest_regressions_contract.rs`). That guard reads the whole tree, so
/// a change in another crate can fail it: moving a `proptest!` block inside
/// `apps/rocm` leaves its seed behind in `apps/rocm/proptest-regressions/`, and
/// nothing in the reverse-dependency closure of `apps/rocm` runs the guard.
const PROPTEST_GUARD_CRATE: &str = "xtask";

/// True for a file under a `proptest-regressions` directory: the saved-failure
/// files the guard above checks.
fn is_proptest_seed_file(file: &str) -> bool {
    file.split('/')
        .any(|segment| segment == "proptest-regressions")
}

/// True for changed files that cannot affect any Rust crate's build or tests, so
/// they neither select a crate nor force a full run.
fn is_ignorable(file: &str) -> bool {
    let is_markdown = std::path::Path::new(file)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"));
    is_markdown || file.starts_with("docs/") || file == "LICENSE" || file.starts_with("LICENSE.")
}

/// Pure selection logic: turn the changed-file list into a [`Selection`].
///
/// `proptest_sources` is the subset of changed Rust files whose diff adds or
/// removes a `proptest!` line ([`proptest_source_files`]). It, and any changed
/// file under a `proptest-regressions` directory, adds `xtask` to the selection
/// so the seed guard runs on the pull request instead of only in the merge
/// queue. When the graph has no `xtask` member the guard cannot be selected, so
/// the answer is the whole workspace.
///
/// Order matters — a single force-full file (e.g. `Cargo.lock`) outweighs any
/// number of crate-attributable files. Files owned by a crate seed the changed
/// set; ignorable files are dropped; anything left unattributed is treated as
/// force-full, because an unrecognized path might affect anything.
pub fn select(changed_files: &[String], proptest_sources: &[String], graph: &Graph) -> Selection {
    let mut changed_crates: BTreeSet<String> = BTreeSet::new();
    let mut saw_relevant = false;

    for file in changed_files {
        if forces_full_workspace(file) {
            return Selection::Workspace;
        }
        if is_proptest_seed_file(file) || proptest_sources.contains(file) {
            if !graph
                .crate_dirs
                .iter()
                .any(|(name, _)| name == PROPTEST_GUARD_CRATE)
            {
                return Selection::Workspace;
            }
            changed_crates.insert(PROPTEST_GUARD_CRATE.to_string());
            saw_relevant = true;
        }
        if let Some(owner) = owning_crate(file, &graph.crate_dirs) {
            changed_crates.insert(owner.to_string());
            saw_relevant = true;
            continue;
        }
        if is_ignorable(file) {
            continue;
        }
        // Unattributed and not known-irrelevant: be safe, run everything.
        return Selection::Workspace;
    }

    if !saw_relevant {
        return Selection::Empty;
    }
    Selection::Packages(reverse_closure(&changed_crates, &graph.rev_deps))
}

// ---- I/O edges -------------------------------------------------------------

/// Subset of `cargo metadata` output we consume.
#[derive(Deserialize)]
struct Metadata {
    workspace_root: String,
    workspace_members: Vec<String>,
    packages: Vec<Package>,
    resolve: Resolve,
}

#[derive(Deserialize)]
struct Package {
    id: String,
    name: String,
    manifest_path: String,
}

#[derive(Deserialize)]
struct Resolve {
    nodes: Vec<Node>,
}

#[derive(Deserialize)]
struct Node {
    id: String,
    /// Resolved dependency package ids (union across normal/dev/build kinds).
    dependencies: Vec<String>,
}

/// Run `cargo metadata` and reduce it to a [`Graph`].
///
/// `--locked` mirrors [`crate::manifest`]: use the committed `Cargo.lock`
/// exactly rather than letting metadata resolve to newer registry versions.
pub fn load_graph() -> Result<Graph> {
    let stdout = crate::paths::run_cargo_metadata(&[])?;
    let metadata: Metadata =
        serde_json::from_slice(&stdout).context("failed to parse `cargo metadata` output")?;
    let graph = build_graph(&metadata);
    // A member at the repository root would have an empty source dir and so own
    // every changed file, silently defeating the conservative full-workspace
    // fallback. The workspace has no such crate; fail loud if that ever changes
    // rather than quietly mis-select.
    if let Some((name, _)) = graph.crate_dirs.iter().find(|(_, dir)| dir.is_empty()) {
        bail!(
            "workspace member `{name}` resolves to the repository root; \
             affected-crate selection does not support a root crate"
        );
    }
    Ok(graph)
}

/// Reduce `cargo metadata` to the [`Graph`]: workspace-relative source dirs and
/// reverse-dependency edges restricted to workspace members.
fn build_graph(metadata: &Metadata) -> Graph {
    let members: BTreeSet<&str> = metadata
        .workspace_members
        .iter()
        .map(String::as_str)
        .collect();
    let id_to_name: BTreeMap<&str, &str> = metadata
        .packages
        .iter()
        .map(|p| (p.id.as_str(), p.name.as_str()))
        .collect();

    let root = metadata.workspace_root.replace('\\', "/");
    let root = root.trim_end_matches('/');

    let crate_dirs: Vec<(String, String)> = metadata
        .packages
        .iter()
        .filter(|p| members.contains(p.id.as_str()))
        .map(|p| {
            // manifest_path is ".../<dir>/Cargo.toml"; the crate dir is its
            // parent, made workspace-relative with `/` separators.
            let manifest = p.manifest_path.replace('\\', "/");
            let dir = manifest.strip_suffix("/Cargo.toml").unwrap_or(&manifest);
            let rel = dir
                .strip_prefix(root)
                .map_or(dir, |r| r.trim_start_matches('/'))
                .to_string();
            (p.name.clone(), rel)
        })
        .collect();

    // Forward edges among members -> invert into reverse edges. Seed every
    // member so a leaf with no dependents still has an (empty) entry.
    let mut rev_deps: BTreeMap<String, Vec<String>> = crate_dirs
        .iter()
        .map(|(name, _)| (name.clone(), Vec::new()))
        .collect();
    for node in &metadata.resolve.nodes {
        if !members.contains(node.id.as_str()) {
            continue;
        }
        let Some(&dependent) = id_to_name.get(node.id.as_str()) else {
            continue;
        };
        for dep_id in &node.dependencies {
            if !members.contains(dep_id.as_str()) {
                continue;
            }
            if let Some(&dep_name) = id_to_name.get(dep_id.as_str()) {
                rev_deps
                    .entry(dep_name.to_string())
                    .or_default()
                    .push(dependent.to_string());
            }
        }
    }

    Graph {
        crate_dirs,
        rev_deps,
    }
}

/// List files changed in `base...HEAD` (three-dot: relative to the merge base,
/// the same "what this branch changed" diff layer one filters on).
fn changed_files(base: &str) -> Result<Vec<String>> {
    diff_names(Path::new("."), base, &[], &[])
}

/// List the Rust files changed in `base...HEAD` whose diff adds or removes a
/// line containing `proptest!` (git's `-G` pickaxe, so a block moved out of one
/// file and into another lists both). The pattern also matches a longer name
/// such as `my_proptest!`; an extra hit only adds `xtask` to the selection.
fn proptest_source_files(repo: &Path, base: &str) -> Result<Vec<String>> {
    diff_names(repo, base, &["-G", "proptest!"], &["*.rs"])
}

/// `git -C <repo> diff --name-only <options> <base>...HEAD -- <pathspec>`.
fn diff_names(repo: &Path, base: &str, options: &[&str], pathspec: &[&str]) -> Result<Vec<String>> {
    let range = format!("{base}...HEAD");
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(repo)
        .args(["diff", "--name-only"])
        .args(options)
        .arg(&range);
    if !pathspec.is_empty() {
        command.arg("--").args(pathspec);
    }
    let output = command.output().context("failed to run `git diff`")?;
    if !output.status.success() {
        bail!(
            "`git diff {range}` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

/// Entry point for `cargo xtask affected`: print the cargo package-selection
/// flags for the crates affected by `base...HEAD`.
pub fn run(base: Option<String>) -> Result<()> {
    let base = base.unwrap_or_else(|| DEFAULT_BASE.to_string());
    let graph = load_graph()?;
    let files = changed_files(&base)?;
    let proptest_sources = proptest_source_files(Path::new("."), &base)?;
    let selection = select(&files, &proptest_sources, &graph);
    println!("{}", selection.to_cargo_args());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Selection from the changed files alone: no changed file adds or removes a
    /// `proptest!` line. Shadows the glob-imported [`super::select`] so the tests
    /// that predate the proptest guard stay as they were.
    fn select(changed_files: &[String], graph: &Graph) -> Selection {
        super::select(changed_files, &[], graph)
    }

    fn names(pkgs: &[&str]) -> BTreeSet<String> {
        pkgs.iter().map(|p| (*p).to_string()).collect()
    }

    fn graph() -> Graph {
        // A small stand-in for the real workspace: core <- protocol <- engine,
        // plus an independent leaf and the top app depending on everything.
        let crate_dirs = vec![
            ("core".to_string(), "crates/core".to_string()),
            ("protocol".to_string(), "crates/protocol".to_string()),
            ("engine".to_string(), "engines/engine".to_string()),
            ("leaf".to_string(), "crates/leaf".to_string()),
            ("app".to_string(), "apps/app".to_string()),
            ("xtask".to_string(), "xtask".to_string()),
        ];
        // forward: protocol->core, engine->{core,protocol}, app->{engine,leaf}
        // reverse: core->{protocol,engine}, protocol->{engine}, engine->{app}, leaf->{app}
        let mut rev_deps: BTreeMap<String, Vec<String>> = BTreeMap::new();
        rev_deps.insert("core".into(), vec!["protocol".into(), "engine".into()]);
        rev_deps.insert("protocol".into(), vec!["engine".into()]);
        rev_deps.insert("engine".into(), vec!["app".into()]);
        rev_deps.insert("leaf".into(), vec!["app".into()]);
        rev_deps.insert("app".into(), vec![]);
        rev_deps.insert("xtask".into(), vec![]);
        Graph {
            crate_dirs,
            rev_deps,
        }
    }

    #[test]
    fn owning_crate_picks_longest_prefix() {
        let dirs = vec![
            ("a".to_string(), "crates/a".to_string()),
            ("ab".to_string(), "crates/a/b".to_string()),
        ];
        assert_eq!(owning_crate("crates/a/src/x.rs", &dirs), Some("a"));
        assert_eq!(owning_crate("crates/a/b/src/x.rs", &dirs), Some("ab"));
        assert_eq!(owning_crate("crates/other/x.rs", &dirs), None);
    }

    #[test]
    fn owning_crate_respects_segment_boundaries() {
        let dirs = vec![("vllm".to_string(), "engines/vllm".to_string())];
        // A sibling whose path merely starts with the same string must not match.
        assert_eq!(owning_crate("engines/vllm-extra/src/x.rs", &dirs), None);
        assert_eq!(owning_crate("engines/vllm/src/x.rs", &dirs), Some("vllm"));
    }

    #[test]
    fn reverse_closure_collects_transitive_dependents() {
        let g = graph();
        let changed: BTreeSet<String> = std::iter::once("core".to_string()).collect();
        let got = reverse_closure(&changed, &g.rev_deps);
        let want: BTreeSet<String> = ["core", "protocol", "engine", "app"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn select_leaf_change_picks_crate_and_dependents() {
        let g = graph();
        let sel = select(&["engines/engine/src/lib.rs".to_string()], &g);
        let want: BTreeSet<String> = ["engine", "app"].into_iter().map(String::from).collect();
        assert_eq!(sel, Selection::Packages(want));
    }

    #[test]
    fn select_core_change_fans_out() {
        let g = graph();
        let sel = select(&["crates/core/src/lib.rs".to_string()], &g);
        let Selection::Packages(pkgs) = sel else {
            panic!("expected Packages, got {sel:?}");
        };
        // Every crate transitively depends on core except the independent leaf.
        // `protocol` is a mid-graph dependent; assert it explicitly so a
        // regression that drops intermediate dependents is caught.
        assert!(
            pkgs.contains("core")
                && pkgs.contains("protocol")
                && pkgs.contains("engine")
                && pkgs.contains("app")
        );
        assert!(!pkgs.contains("leaf"));
    }

    #[test]
    fn select_empty_input_is_empty() {
        assert_eq!(select(&[], &graph()), Selection::Empty);
    }

    #[test]
    fn select_force_full_outweighs_crate_change() {
        // A crate edit alongside a force-full file must still run the whole
        // workspace — the lockfile change can affect any crate.
        let g = graph();
        let sel = select(
            &[
                "crates/core/src/lib.rs".to_string(),
                "Cargo.lock".to_string(),
            ],
            &g,
        );
        assert_eq!(sel, Selection::Workspace);
    }

    #[test]
    fn select_lockfile_forces_workspace() {
        let g = graph();
        assert_eq!(
            select(&["Cargo.lock".to_string()], &g),
            Selection::Workspace
        );
        assert_eq!(
            select(&["rust-toolchain.toml".to_string()], &g),
            Selection::Workspace
        );
        assert_eq!(
            select(&[".github/workflows/ci.yml".to_string()], &g),
            Selection::Workspace
        );
    }

    #[test]
    fn select_contributing_md_forces_workspace() {
        // CONTRIBUTING.md carries the hawkeye/cargo-about pins that
        // pinned_tool_versions_match_across_docs_and_workflows asserts against;
        // unlike ordinary Markdown (see select_docs_only_is_empty) it must
        // force a full run, not resolve to `Selection::Empty`.
        let g = graph();
        assert_eq!(
            select(&["CONTRIBUTING.md".to_string()], &g),
            Selection::Workspace
        );
    }

    #[test]
    fn select_hardware_testing_docs_forces_workspace() {
        // docs/ci-hardware-testing.md is read by
        // hardware_testing_docs_cover_all_self_hosted_platforms; unlike
        // ordinary docs/ Markdown (see select_docs_only_is_empty) it must
        // force a full run, not resolve to `Selection::Empty`.
        let g = graph();
        assert_eq!(
            select(&["docs/ci-hardware-testing.md".to_string()], &g),
            Selection::Workspace
        );
    }

    #[test]
    fn select_e2e_cucumber_readme_forces_workspace() {
        // tests/e2e-cucumber/README.md is read by the same
        // hardware_testing_docs_cover_all_self_hosted_platforms test as
        // docs/ci-hardware-testing.md above, for its lane lists; it must get
        // the same treatment, not resolve to `Selection::Empty`.
        let g = graph();
        assert_eq!(
            select(&["tests/e2e-cucumber/README.md".to_string()], &g),
            Selection::Workspace
        );
    }

    #[test]
    fn select_docs_only_is_empty() {
        let g = graph();
        assert_eq!(
            select(&["README.md".to_string(), "docs/guide.md".to_string()], &g),
            Selection::Empty
        );
    }

    #[test]
    fn select_unknown_file_forces_workspace() {
        let g = graph();
        // A path owned by no crate and not on the ignore-list is unsafe to skip.
        assert_eq!(
            select(&["some/random/file.txt".to_string()], &g),
            Selection::Workspace
        );
    }

    #[test]
    fn select_mixed_crate_and_docs_ignores_docs() {
        let g = graph();
        let sel = select(
            &[
                "engines/engine/src/lib.rs".to_string(),
                "README.md".to_string(),
            ],
            &g,
        );
        let want: BTreeSet<String> = ["engine", "app"].into_iter().map(String::from).collect();
        assert_eq!(sel, Selection::Packages(want));
    }

    #[test]
    fn a_changed_proptest_seed_file_selects_the_xtask_guard() {
        // The seed guard lives in xtask and reads every crate's seeds; `app`
        // has no edge to xtask, so without this rule a seed edit in `app`
        // selects only `app`. Both the seed's own crate and the guard run.
        let sel = select(
            &["apps/app/proptest-regressions/a/b.txt".to_string()],
            &graph(),
        );
        assert_eq!(sel, Selection::Packages(names(&["app", "xtask"])));
        assert_eq!(sel.to_cargo_args(), "-p app -p xtask");
    }

    #[test]
    fn a_changed_proptest_source_selects_the_xtask_guard() {
        // The case the guard exists for: a `proptest!` block moves between
        // files inside one crate and the seed is forgotten, so no seed file is
        // in the change at all; only the diff of the Rust files says so.
        let files = [
            "apps/app/src/old.rs".to_string(),
            "apps/app/src/new.rs".to_string(),
        ];
        assert_eq!(
            super::select(&files, &files, &graph()),
            Selection::Packages(names(&["app", "xtask"]))
        );
        // The same Rust change with no `proptest!` line in its diff does not.
        assert_eq!(
            super::select(&files, &[], &graph()),
            Selection::Packages(names(&["app"]))
        );
    }

    #[test]
    fn a_proptest_change_without_an_xtask_member_runs_the_workspace() {
        // Nothing to select the guard with, so fall back rather than skip it.
        let mut g = graph();
        g.crate_dirs.retain(|(name, _)| name != "xtask");
        g.rev_deps.remove("xtask");
        let seed = ["apps/app/proptest-regressions/x.txt".to_string()];
        assert_eq!(select(&seed, &g), Selection::Workspace);
        let src = ["apps/app/src/x.rs".to_string()];
        assert_eq!(super::select(&src, &src, &g), Selection::Workspace);
    }

    #[test]
    fn only_a_proptest_regressions_directory_counts_as_a_seed_path() {
        assert!(is_proptest_seed_file(
            "apps/rocm/proptest-regressions/main.txt"
        ));
        assert!(is_proptest_seed_file(
            "crates/c/proptest-regressions/ui/format.txt"
        ));
        // A file merely named alike, or a Rust file, is not a seed.
        assert!(!is_proptest_seed_file(
            "apps/rocm/proptest-regressions-notes.md"
        ));
        assert!(!is_proptest_seed_file(
            "apps/rocm/src/proptest-regressions.rs"
        ));
        assert!(!is_proptest_seed_file("apps/rocm/src/main.rs"));
        // Unrelated crate changes still leave xtask out.
        assert_eq!(
            select(&["apps/app/src/lib.rs".to_string()], &graph()),
            Selection::Packages(names(&["app"]))
        );
    }

    /// Run `git` in `repo` with signing, hooks and the caller's `GIT_*`
    /// environment out of the way, and return stdout.
    fn git(repo: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args([
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
            ])
            .args(args)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .output()
            .expect("run git");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    #[test]
    fn proptest_source_files_lists_rust_files_whose_diff_touches_a_proptest_block() {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path();
        git(repo, &["init", "-q", "-b", "main"]);
        std::fs::write(
            repo.join("moved_from.rs"),
            "proptest! { fn a() {} }\nfn keep() {}\n",
        )
        .unwrap();
        std::fs::write(repo.join("plain.rs"), "fn plain() {}\n").unwrap();
        std::fs::write(repo.join("notes.md"), "proptest! is mentioned\n").unwrap();
        git(repo, &["add", "."]);
        git(repo, &["commit", "-q", "-m", "base"]);
        git(repo, &["checkout", "-q", "-b", "topic"]);
        // The block moves out of one Rust file and into another. A Rust file
        // whose diff has no proptest line, and a Markdown note that does but
        // is not Rust, change too; neither may be listed.
        std::fs::write(repo.join("moved_from.rs"), "fn keep() {}\n").unwrap();
        std::fs::write(repo.join("moved_to.rs"), "proptest! { fn a() {} }\n").unwrap();
        std::fs::write(repo.join("plain.rs"), "fn plain() { let _ = 1; }\n").unwrap();
        std::fs::write(repo.join("notes.md"), "proptest! is mentioned again\n").unwrap();
        git(repo, &["add", "."]);
        git(repo, &["commit", "-q", "-m", "move"]);

        let mut got = proptest_source_files(repo, "main").expect("diff");
        got.sort();
        assert_eq!(got, ["moved_from.rs", "moved_to.rs"]);
        // The plain name-only diff, which the selector otherwise relies on,
        // sees all four files: the narrowing above is what the `-G` adds.
        let all = diff_names(repo, "main", &[], &[]).expect("diff");
        assert_eq!(all.len(), 4, "{all:?}");
    }

    #[test]
    fn cargo_args_rendering() {
        assert_eq!(Selection::Workspace.to_cargo_args(), "--workspace");
        assert_eq!(Selection::Empty.to_cargo_args(), "");
        let pkgs: BTreeSet<String> = ["a", "b"].into_iter().map(String::from).collect();
        assert_eq!(Selection::Packages(pkgs).to_cargo_args(), "-p a -p b");
    }
}
