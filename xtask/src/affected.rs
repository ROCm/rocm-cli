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
//! file, the workspace root manifest, CI config, or any unrecognized path) makes
//! the command fall back to `--workspace`. Skipping a test that should have run
//! is the failure mode this guards against, so when in doubt it runs everything.
//!
//! As with [`crate::verify_commits`], the decision logic ([`select`],
//! [`owning_crate`], [`reverse_closure`]) is pure and unit-tested without git or
//! cargo; the I/O lives at the edges ([`load_graph`], [`changed_files`]).

use std::collections::{BTreeMap, BTreeSet};
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

/// True for changed files that force a full-workspace run: anything that can
/// affect resolution, the build for every crate, or CI itself.
fn forces_full_workspace(file: &str) -> bool {
    file == "Cargo.lock"
        || file == "Cargo.toml" // workspace root manifest
        || file == "MANIFEST.md" // clippy job runs `xtask manifest --check`
        || file.starts_with("rust-toolchain")
        || file.starts_with(".github/workflows/")
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
    let output = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args(["metadata", "--format-version", "1", "--locked"])
        .output()
        .context("failed to run `cargo metadata`")?;
    if !output.status.success() {
        bail!(
            "`cargo metadata` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let metadata: Metadata = serde_json::from_slice(&output.stdout)
        .context("failed to parse `cargo metadata` output")?;
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
    let range = format!("{base}...HEAD");
    let output = Command::new("git")
        .args(["diff", "--name-only", &range])
        .output()
        .context("failed to run `git diff`")?;
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
    let selection = select(&files, &graph);
    println!("{}", selection.to_cargo_args());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> Graph {
        // A small stand-in for the real workspace: core <- protocol <- engine,
        // plus an independent leaf and the top app depending on everything.
        let crate_dirs = vec![
            ("core".to_string(), "crates/core".to_string()),
            ("protocol".to_string(), "crates/protocol".to_string()),
            ("engine".to_string(), "engines/engine".to_string()),
            ("leaf".to_string(), "crates/leaf".to_string()),
            ("app".to_string(), "apps/app".to_string()),
        ];
        // forward: protocol->core, engine->{core,protocol}, app->{engine,leaf}
        // reverse: core->{protocol,engine}, protocol->{engine}, engine->{app}, leaf->{app}
        let mut rev_deps: BTreeMap<String, Vec<String>> = BTreeMap::new();
        rev_deps.insert("core".into(), vec!["protocol".into(), "engine".into()]);
        rev_deps.insert("protocol".into(), vec!["engine".into()]);
        rev_deps.insert("engine".into(), vec!["app".into()]);
        rev_deps.insert("leaf".into(), vec!["app".into()]);
        rev_deps.insert("app".into(), vec![]);
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
    fn cargo_args_rendering() {
        assert_eq!(Selection::Workspace.to_cargo_args(), "--workspace");
        assert_eq!(Selection::Empty.to_cargo_args(), "");
        let pkgs: BTreeSet<String> = ["a", "b"].into_iter().map(String::from).collect();
        assert_eq!(Selection::Packages(pkgs).to_cargo_args(), "-p a -p b");
    }
}
