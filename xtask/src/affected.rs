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
//! file selects `xtask`, which holds the seed guard that reads the whole tree
//! (see `PROPTEST_GUARD_CRATE`). A changed Rust file reaches that guard through
//! the `*.rs` entry of [`crate::cross_crate_reads::CROSS_CRATE_READS`], which
//! selects `xtask` for every Rust change.
//!
//! The command prints [`select_with_readers`]: that directory-based selection
//! widened with the crates whose tests read a changed file from outside their
//! own directory ([`crate::cross_crate_reads`]), such as the `rocm` and `xtask`
//! tests that read `README.md`, `docs/` and `skills/`. The pre-push hook
//! ([`crate::prepush`]) selects through the same function.
//!
//! As with [`crate::verify_commits`], the decision logic
//! ([`select`], [`select_with_readers`], [`owning_crate`],
//! [`reverse_closure`]) is pure and unit-tested without git or cargo; the I/O
//! lives at the edges ([`load_graph`], [`printed_selection`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::cross_crate_reads;

/// Default base ref used when none is supplied on the command line. Matches
/// [`crate::verify_commits`] so both commands diff against the same point.
pub(crate) const DEFAULT_BASE: &str = "origin/main";

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

/// Markdown files an xtask test reads directly. This list is the one place
/// they are named for selection: `forces_full_workspace` below reads it, so an
/// edit to one of them alone selects the whole workspace instead of being
/// dropped by `is_ignorable` as ordinary Markdown (which would resolve to
/// `Selection::Empty`). Each also belongs in the `rust` path filter of
/// `.github/actions/classify-changes/action.yml`, or the rust-gated jobs do not
/// run at all for that edit, and in the "doc an xtask test reads" sentence in
/// `docs/testing.md`. `every_cross_crate_read_is_in_the_rust_filter`
/// (workflow_contract.rs) checks this list against the filter, and
/// `each_doc_an_xtask_test_reads_selects_the_whole_workspace` against that
/// sentence and the selection. Without them the test that reads the file
/// silently never runs:
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
        || file == "clippy.toml" // configures the lints for every crate
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

/// True for changed files that no crate's build reads, so they neither select a
/// crate by directory nor force a full run. A test can still read one from
/// outside its crate; [`select_with_readers`] adds those readers back.
fn is_ignorable(file: &str) -> bool {
    let is_markdown = std::path::Path::new(file)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"));
    is_markdown || file.starts_with("docs/") || file == "LICENSE" || file.starts_with("LICENSE.")
}

/// Pure directory-based selection logic: turn the changed-file list into a
/// [`Selection`]. Callers select through [`select_with_readers`].
///
/// A changed file under a `proptest-regressions` directory adds `xtask` to the
/// selection so the seed guard runs on the pull request instead of only in the
/// merge queue. When the graph has no `xtask` member the guard cannot be
/// selected, so the answer is the whole workspace. A changed Rust file (one
/// that moved a `proptest!` block, say) is not handled here: the `*.rs` entry
/// of [`cross_crate_reads::CROSS_CRATE_READS`] adds `xtask` for it in
/// [`select_with_readers`].
///
/// Order matters — a single force-full file (e.g. `Cargo.lock`) outweighs any
/// number of crate-attributable files. Files owned by a crate seed the changed
/// set; ignorable files are dropped; anything left unattributed is treated as
/// force-full, because an unrecognized path might affect anything.
pub fn select(changed_files: &[String], graph: &Graph) -> Selection {
    let mut changed_crates: BTreeSet<String> = BTreeSet::new();
    let mut saw_relevant = false;

    for file in changed_files {
        if forces_full_workspace(file) {
            return Selection::Workspace;
        }
        if is_proptest_seed_file(file) {
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

/// The selection CI's `cargo xtask affected` and the pre-push hook use:
/// [`select`], widened with the crates whose tests read a
/// changed file from outside their own directory, and their dependents
/// ([`cross_crate_reads::widen`] over [`cross_crate_reads::CROSS_CRATE_READS`]).
///
/// [`select`] alone treats Markdown and `docs/` as unable
/// to affect any crate, so a pull request that edits only a doc a test reads
/// would test nothing.
pub fn select_with_readers(changed_files: &[String], graph: &Graph) -> Selection {
    cross_crate_reads::widen(
        select(changed_files, graph),
        changed_files,
        graph,
        cross_crate_reads::CROSS_CRATE_READS,
    )
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

/// List files changed in `base...head`.
///
/// `--no-renames` makes a moved file list both its old and its new path.
/// Without it git reports only the new path, so a file moved out of one crate
/// into another would select the destination crate and miss the one that lost
/// the file.
pub(crate) fn changed_files_between(base: &str, head: &str) -> Result<Vec<String>> {
    diff_name_only(None, base, head)
}

/// The variables git reads to pick a repository. git exports an absolute
/// `GIT_DIR` to a hook run from a linked worktree, so a child that runs
/// `git -C <other repo>` must not inherit them, or it talks to the outer repo.
const GIT_REPO_ENV: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
];

/// Clear [`GIT_REPO_ENV`] from `cmd`, for a git run against an explicit directory.
pub(crate) fn scrub_git_env(cmd: &mut Command) -> &mut Command {
    for var in GIT_REPO_ENV {
        cmd.env_remove(var);
    }
    cmd
}

/// [`changed_files_between`] in `dir` (the current directory when `None`).
fn diff_name_only(dir: Option<&std::path::Path>, base: &str, head: &str) -> Result<Vec<String>> {
    diff_names(dir, base, head)
}

/// `git diff --name-only --no-renames base...head` in `dir` (the current
/// directory when `None`).
fn diff_names(dir: Option<&Path>, base: &str, head: &str) -> Result<Vec<String>> {
    let range = format!("{base}...{head}");
    let mut git = Command::new("git");
    if let Some(dir) = dir {
        scrub_git_env(&mut git).arg("-C").arg(dir);
    }
    git.args(["diff", "--name-only", "--no-renames"])
        .arg(&range);
    let output = git.output().context("failed to run `git diff`")?;
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
    println!("{}", printed_selection(None, &base, &graph)?);
    Ok(())
}

/// The line [`run`] prints for the repository in `dir` (the current
/// directory when `None`): the cargo flags for the files changed in
/// `base...HEAD` (three-dot: relative to the merge base, what this branch
/// changed), selected through [`select_with_readers`].
fn printed_selection(dir: Option<&Path>, base: &str, graph: &Graph) -> Result<String> {
    let files = diff_name_only(dir, base, "HEAD")?;
    Ok(select_with_readers(&files, graph).to_cargo_args())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            select(&["clippy.toml".to_string()], &g),
            Selection::Workspace
        );
        assert_eq!(
            select(&[".github/workflows/ci.yml".to_string()], &g),
            Selection::Workspace
        );
    }

    /// `clippy.toml` configures the lints of every crate. As an unattributed
    /// root file it already selects the whole workspace through the fallback in
    /// [`select_with_proptest_sources`]; naming it keeps that so if root-level
    /// config files are ever made ignorable.
    #[test]
    fn clippy_config_is_a_named_full_workspace_input() {
        assert!(forces_full_workspace("clippy.toml"));
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

    /// A graph with the crates whose tests read docs and skills from outside
    /// their own directory, and a leaf that reads nothing. `rocm-core` is one of
    /// them, for `README.md` alone, and has to exist here: a declared reader the
    /// graph does not know widens the selection to the whole workspace, which
    /// would pass this test for the wrong reason.
    fn reader_graph() -> Graph {
        let crate_dirs = vec![
            ("rocm".to_string(), "apps/rocm".to_string()),
            ("rocm-core".to_string(), "crates/rocm-core".to_string()),
            ("xtask".to_string(), "xtask".to_string()),
            ("leaf".to_string(), "crates/leaf".to_string()),
        ];
        let rev_deps = crate_dirs
            .iter()
            .map(|(name, _)| (name.clone(), Vec::new()))
            .collect();
        Graph {
            crate_dirs,
            rev_deps,
        }
    }

    /// What `cargo xtask affected` prints is [`select_with_readers`]. A pull
    /// request that edits only a doc or a skill a test reads selected nothing
    /// there, so CI's `Test (affected crates)` job skipped the `rocm` and
    /// `xtask` tests that read it (`docs/nightly-on-the-queue.md` is read by
    /// `the_queue_doc_states_its_channels_and_its_remaining_gap`, and
    /// `advised_commands` reads every `README.md`, `docs/` and `skills/` file).
    #[test]
    fn a_doc_or_skill_a_test_reads_selects_its_readers_in_ci() {
        for path in [
            "docs/nightly-on-the-queue.md",
            "docs/testing.md",
            "skills/rocm-cli-assistant/SKILL.md",
        ] {
            assert_eq!(
                select_with_readers(&[path.to_string()], &reader_graph()),
                Selection::Packages(names(&["rocm", "xtask"])),
                "{path}"
            );
        }
        // The README has a third reader: `rocm-core` holds the quantization cap
        // the README prints. Asserted apart from the others so that the extra
        // crate is this file's claim about the README specifically, rather than
        // something every doc quietly drags in.
        assert_eq!(
            select_with_readers(&["README.md".to_string()], &reader_graph()),
            Selection::Packages(names(&["rocm", "rocm-core", "xtask"]))
        );
    }

    /// The widening adds to the directory-based selection and keeps its other
    /// rules: a crate's own change still selects it, and a force-full file
    /// still runs the whole workspace.
    #[test]
    fn select_with_readers_adds_readers_to_the_directory_selection() {
        let files = [
            "crates/leaf/src/lib.rs".to_string(),
            "docs/a.md".to_string(),
        ];
        assert_eq!(
            select_with_readers(&files, &reader_graph()),
            Selection::Packages(names(&["leaf", "rocm", "xtask"]))
        );
        let forced = ["docs/a.md".to_string(), "Cargo.lock".to_string()];
        assert_eq!(
            select_with_readers(&forced, &reader_graph()),
            Selection::Workspace
        );
        // Markdown no test reads still selects nothing.
        assert_eq!(
            select_with_readers(&["CHANGELOG.md".to_string()], &reader_graph()),
            Selection::Empty
        );
    }

    /// Run `git` in `dir` with a fixed identity and no signing, for the
    /// throwaway repositories below.
    fn git_in(dir: &std::path::Path, args: &[&str]) {
        let mut git = Command::new("git");
        let status = scrub_git_env(&mut git)
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .status()
            .expect("run git");
        assert!(status.success(), "git {args:?} failed in {}", dir.display());
    }

    /// A hook run from a linked worktree has `GIT_DIR` set; a git run against a
    /// throwaway repository must not inherit it or it would act on the outer one.
    #[test]
    fn scrub_git_env_clears_every_repo_selecting_variable() {
        let mut git = Command::new("git");
        git.env("GIT_DIR", "/outer/.git").env("GIT_INDEX_FILE", "x");
        scrub_git_env(&mut git);
        let envs: std::collections::BTreeMap<_, _> = git
            .get_envs()
            .map(|(k, v)| (k.to_string_lossy().into_owned(), v.is_none()))
            .collect();
        for var in GIT_REPO_ENV {
            assert_eq!(envs.get(*var), Some(&true), "{var} is not cleared");
        }
    }

    /// The loop above only checks the variables [`GIT_REPO_ENV`] lists, so it
    /// cannot notice one missing from the list. This one runs git: with the
    /// repository-selecting variables a hook exports pointing at another
    /// repository, a scrubbed `git -C <repo>` must still resolve to `<repo>`.
    #[test]
    fn scrubbed_git_ignores_a_repository_a_hook_exported() {
        let outer = tempfile::tempdir().expect("outer repo");
        let inner = tempfile::tempdir().expect("inner repo");
        git_in(outer.path(), &["init", "-q"]);
        git_in(inner.path(), &["init", "-q"]);
        let resolve = |arg: &str| {
            let mut git = Command::new("git");
            git.env("GIT_DIR", outer.path().join(".git"))
                .env("GIT_WORK_TREE", outer.path());
            scrub_git_env(&mut git);
            let output = git
                .arg("-C")
                .arg(inner.path())
                .args(["rev-parse", arg])
                .output()
                .expect("run git");
            assert!(output.status.success(), "git rev-parse {arg} failed");
            std::fs::canonicalize(String::from_utf8_lossy(&output.stdout).trim())
                .expect("git printed an existing path")
        };
        let inner_root = std::fs::canonicalize(inner.path()).expect("inner path");
        assert_eq!(resolve("--show-toplevel"), inner_root);
        assert_eq!(resolve("--absolute-git-dir"), inner_root.join(".git"));
    }

    /// A file moved from one crate to another must select both crates. With git's
    /// default rename detection only the new path is listed, the crate that lost
    /// the file is not selected, and a push that breaks it is not checked.
    #[test]
    fn changed_files_between_lists_both_paths_of_a_rename() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        git_in(root, &["init", "-q", "-b", "main"]);
        std::fs::create_dir_all(root.join("crates/a/src")).expect("mkdir a");
        std::fs::create_dir_all(root.join("crates/b/src")).expect("mkdir b");
        std::fs::write(root.join("crates/a/src/moved.rs"), "pub fn f() {}\n").expect("write");
        git_in(root, &["add", "."]);
        git_in(root, &["commit", "-q", "-m", "base"]);
        git_in(root, &["checkout", "-q", "-b", "feature"]);
        git_in(
            root,
            &["mv", "crates/a/src/moved.rs", "crates/b/src/moved.rs"],
        );
        git_in(root, &["commit", "-q", "-m", "move"]);

        let mut files = diff_name_only(Some(root), "main", "feature").expect("git diff");
        files.sort();
        assert_eq!(files, ["crates/a/src/moved.rs", "crates/b/src/moved.rs"]);
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

    /// The seed guard's only route from a changed Rust file: no rule in
    /// [`select`] adds `xtask` for one, so moving a `proptest!` block inside
    /// `app` (whose seed stays behind) runs the guard only because the `*.rs`
    /// entry of [`cross_crate_reads::CROSS_CRATE_READS`] selects `xtask`.
    #[test]
    fn a_changed_rust_file_selects_the_proptest_guard_through_the_readers() {
        let files = [
            "apps/app/src/old.rs".to_string(),
            "apps/app/src/new.rs".to_string(),
        ];
        assert_eq!(
            select(&files, &graph()),
            Selection::Packages(names(&["app"]))
        );
        assert_eq!(
            select_with_readers(&files, &graph()),
            Selection::Packages(names(&["app", "xtask"]))
        );
        // Without an `xtask` member the reader cannot be selected, so the
        // answer is the whole workspace rather than a skipped guard.
        let mut g = graph();
        g.crate_dirs.retain(|(name, _)| name != "xtask");
        g.rev_deps.remove("xtask");
        assert_eq!(select_with_readers(&files, &g), Selection::Workspace);
    }

    #[test]
    fn a_proptest_change_without_an_xtask_member_runs_the_workspace() {
        // Nothing to select the guard with, so fall back rather than skip it.
        let mut g = graph();
        g.crate_dirs.retain(|(name, _)| name != "xtask");
        g.rev_deps.remove("xtask");
        let seed = ["apps/app/proptest-regressions/x.txt".to_string()];
        assert_eq!(select(&seed, &g), Selection::Workspace);
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

    /// Every doc in [`XTASK_READ_DOCS`] is the one `docs/testing.md` names as "a
    /// doc an xtask test reads", and an edit to it alone runs the whole
    /// workspace through [`select_with_readers`], so a doc dropped from the
    /// list (and left to `is_ignorable`) fails here.
    #[test]
    fn each_doc_an_xtask_test_reads_selects_the_whole_workspace() {
        let testing = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("docs/testing.md"),
        )
        .expect("read docs/testing.md");
        let marker = "a doc an xtask test reads \u{2014}";
        let start = testing.find(marker).expect("the sentence naming the docs") + marker.len();
        let sentence = &testing[start..start + testing[start..].find(')').expect("its end")];
        let mut named: Vec<&str> = sentence.split('`').skip(1).step_by(2).collect();
        named.sort_unstable();
        let mut listed = XTASK_READ_DOCS.to_vec();
        listed.sort_unstable();
        assert_eq!(listed, named, "XTASK_READ_DOCS against docs/testing.md");
        for doc in named {
            assert_eq!(
                select_with_readers(&[doc.to_string()], &graph()),
                Selection::Workspace,
                "an edit to {doc} alone"
            );
        }
    }

    /// What `cargo xtask affected` prints for a branch that edits only a doc
    /// a test reads: CI's `Test (affected crates)` job runs those flags, so the
    /// readers must be in them, not just in [`select_with_readers`]. This
    /// drives [`printed_selection`], which builds the whole line [`run`]
    /// prints; `run` itself only loads the real graph and prints that line.
    #[test]
    fn the_printed_selection_for_a_doc_only_branch_names_its_readers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path();
        git_in(repo, &["init", "-q", "-b", "main"]);
        std::fs::create_dir_all(repo.join("docs")).unwrap();
        std::fs::write(repo.join("docs/nightly-on-the-queue.md"), "a\n").unwrap();
        git_in(repo, &["add", "."]);
        git_in(repo, &["commit", "-q", "-m", "base"]);
        git_in(repo, &["checkout", "-q", "-b", "topic"]);
        std::fs::write(repo.join("docs/nightly-on-the-queue.md"), "b\n").unwrap();
        git_in(repo, &["commit", "-q", "-am", "doc"]);

        let printed = printed_selection(Some(repo), "main", &reader_graph()).expect("selection");
        assert_eq!(printed, "-p rocm -p xtask");
    }

    #[test]
    fn cargo_args_rendering() {
        assert_eq!(Selection::Workspace.to_cargo_args(), "--workspace");
        assert_eq!(Selection::Empty.to_cargo_args(), "");
        let pkgs: BTreeSet<String> = ["a", "b"].into_iter().map(String::from).collect();
        assert_eq!(Selection::Packages(pkgs).to_cargo_args(), "-p a -p b");
    }
}
