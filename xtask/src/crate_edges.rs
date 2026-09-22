// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Guard the first-party crate dependency graph against undeclared layering
//! edges.
//!
//! `rocm-cli`'s workspace has three layering invariants that are otherwise
//! enforced and documented nowhere else in the repo:
//!
//! 1. `rocmd` must never depend on `rocm` (the reverse holds: `rocm` depends
//!    on `rocmd` for the in-process `rocm daemon` foreground path).
//! 2. `rocm-dash-core` must have no first-party dependencies — it is a pure
//!    leaf crate other subsystems build on.
//! 3. `rocm-dash-tui` must not depend on `rocm-core`.
//!
//! A literal cycle check would be a no-op: Cargo's resolver already rejects
//! dependency cycles among normal/build dependencies. The value here is
//! entirely in the **allowlist**: every first-party normal/build edge must be
//! declared in [`ALLOWLIST`], so *any* new first-party coupling — not just
//! the three named invariants — requires a conscious, reviewed addition to
//! this file. This file is the only written record of these rules; keep this
//! comment and [`ALLOWLIST`] in sync with reality.
//!
//! Dev-dependencies are exempt: Cargo permits dev-dependency cycles (e.g.
//! `rocm-dash-daemon` dev-depends on `rocm-core` for a test-only contract
//! pin), so they are collected but never checked against the allowlist.

use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Every currently-allowed first-party `(source, target)` normal/build
/// dependency edge — `source` depends on `target`. Adding a first-party path
/// dependency anywhere in the workspace requires adding its edge here after
/// review; that review is the entire point of this guard.
const ALLOWLIST: &[(&str, &str)] = &[
    ("rocm", "rocm-core"),
    ("rocm", "rocm-deps"),
    ("rocm", "rocm-dash-core"),
    ("rocm", "rocm-dash-daemon"),
    ("rocm", "rocm-dash-tui"),
    ("rocm", "rocm-engine-lemonade"),
    ("rocm", "rocm-engine-protocol"),
    ("rocm", "rocmd"),
    ("rocm", "rocm-engine-vllm"),
    ("rocmd", "rocm-core"),
    ("rocmd", "rocm-engine-protocol"),
    ("rocm-engine-protocol", "rocm-core"),
    ("rocm-dash-collectors", "rocm-dash-core"),
    ("rocm-dash-daemon", "rocm-dash-core"),
    ("rocm-dash-daemon", "rocm-dash-collectors"),
    ("rocm-dash-tui", "rocm-dash-core"),
    ("rocm-dash-tui", "rocm-deps"),
    ("rocm-engine-lemonade", "rocm-core"),
    ("rocm-engine-lemonade", "rocm-deps"),
    ("rocm-engine-lemonade", "rocm-engine-protocol"),
    ("rocm-engine-vllm", "rocm-core"),
    ("rocm-engine-vllm", "rocm-engine-protocol"),
    ("xtask", "e2e-report"),
    ("xtask", "rocm-core"),
    ("e2e-cucumber", "e2e-report"),
];

/// Subset of `cargo metadata --no-deps` output we consume: the manifest-level
/// dependency list per workspace-member package. Unlike
/// `resolve.nodes[].dependencies` (used by [`crate::affected`]), which is a
/// kind-blind union used for "does anything here need rebuilding", this list
/// carries a `kind` per edge so normal/build dependencies can be told apart
/// from dev-dependencies.
#[derive(Deserialize)]
struct Metadata {
    packages: Vec<Package>,
}

#[derive(Deserialize)]
struct Package {
    name: String,
    dependencies: Vec<Dependency>,
}

#[derive(Deserialize)]
struct Dependency {
    name: String,
    /// `None` = normal, `Some("build")`, `Some("dev")`.
    kind: Option<String>,
    /// `Some(_)` for a path dependency (first-party, within this workspace);
    /// `None` for a registry/git dependency.
    path: Option<String>,
}

/// Whether a first-party edge is subject to the allowlist.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Normal or build dependency — checked against [`ALLOWLIST`].
    Enforced,
    /// Dev-dependency — Cargo permits these to cycle, so never enforced.
    Dev,
}

fn edge_kind(kind: Option<&String>) -> Kind {
    match kind.map(String::as_str) {
        Some("dev") => Kind::Dev,
        _ => Kind::Enforced,
    }
}

/// Run `cargo metadata --no-deps` and reduce it to first-party edges:
/// `(source, target, kind)` triples for every path dependency declared by a
/// workspace member. `--no-deps` restricts `packages` to workspace members
/// only, so there's no need to resolve or filter out the external dependency
/// tree.
fn load_edges() -> Result<Vec<(String, String, Kind)>> {
    let output = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args(["metadata", "--format-version", "1", "--locked", "--no-deps"])
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
    Ok(edges_from_metadata(&metadata))
}

fn edges_from_metadata(metadata: &Metadata) -> Vec<(String, String, Kind)> {
    metadata
        .packages
        .iter()
        .flat_map(|pkg| {
            pkg.dependencies
                .iter()
                .filter(|dep| dep.path.is_some())
                .map(move |dep| {
                    (
                        pkg.name.clone(),
                        dep.name.clone(),
                        edge_kind(dep.kind.as_ref()),
                    )
                })
        })
        .collect()
}

/// Whether `source -> target` matches one of the three specifically-named
/// layering invariants, as opposed to being merely an undeclared edge. Shared
/// between [`invariant_for`] (which uses it to pick a specific message) and
/// the `allowlist_never_contains_a_named_invariant_violation` test (which
/// uses it to assert [`ALLOWLIST`] itself can never smuggle one of these
/// three edges back in — otherwise a red CI run could be "fixed" by just
/// allowlisting the very edge these invariants exist to forbid).
fn violates_named_invariant(source: &str, target: &str) -> bool {
    matches!(
        (source, target),
        ("rocmd", "rocm") | ("rocm-dash-tui", "rocm-core")
    ) || source == "rocm-dash-core"
}

/// Explain which named invariant an offending edge breaks, when it matches
/// one of the three documented rules; otherwise a generic explanation.
fn invariant_for(source: &str, target: &str) -> String {
    if !violates_named_invariant(source, target) {
        return format!(
            "not in the declared allowlist; add (\"{source}\", \"{target}\") to ALLOWLIST in \
             xtask/src/crate_edges.rs after review if this edge is intentional"
        );
    }
    match (source, target) {
        ("rocmd", "rocm") => {
            "rocmd must never depend on rocm — rocm depends on rocmd, not the reverse".to_string()
        }
        ("rocm-dash-tui", "rocm-core") => "rocm-dash-tui must not depend on rocm-core".to_string(),
        _ => "rocm-dash-core must have no first-party dependencies (pure leaf crate)".to_string(),
    }
}

/// Check a set of edges against [`ALLOWLIST`], returning every violation
/// found. Dev-dependency edges are always skipped.
fn check(edges: &[(String, String, Kind)]) -> Vec<String> {
    edges
        .iter()
        .filter(|(_, _, kind)| *kind == Kind::Enforced)
        .filter(|(source, target, _)| {
            !ALLOWLIST
                .iter()
                .any(|(a, b)| *a == source.as_str() && *b == target.as_str())
        })
        .map(|(source, target, _)| {
            format!(
                "first-party dependency edge `{source} -> {target}` violates a layering rule: {}",
                invariant_for(source, target)
            )
        })
        .collect()
}

/// Fetch the current first-party crate dependency graph and fail if any
/// normal/build edge is not in [`ALLOWLIST`].
pub fn run() -> Result<()> {
    let edges = load_edges()?;
    let violations = check(&edges);
    if !violations.is_empty() {
        bail!(
            "first-party crate dependency graph has {} disallowed edge(s):\n{}",
            violations.len(),
            violations.join("\n")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(source: &str, target: &str, kind: Kind) -> (String, String, Kind) {
        (source.to_string(), target.to_string(), kind)
    }

    fn allowlisted_edges() -> Vec<(String, String, Kind)> {
        ALLOWLIST
            .iter()
            .map(|(a, b)| edge(a, b, Kind::Enforced))
            .collect()
    }

    #[test]
    fn allowlist_passes() {
        assert!(check(&allowlisted_edges()).is_empty());
    }

    #[test]
    fn rocmd_depending_on_rocm_fails() {
        let mut edges = allowlisted_edges();
        edges.push(edge("rocmd", "rocm", Kind::Enforced));
        let violations = check(&edges);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("rocmd -> rocm"));
        assert!(violations[0].contains("rocmd must never depend on rocm"));
    }

    #[test]
    fn rocm_dash_core_gaining_a_dependency_fails() {
        let mut edges = allowlisted_edges();
        edges.push(edge("rocm-dash-core", "rocm-deps", Kind::Enforced));
        let violations = check(&edges);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("must have no first-party dependencies"));
    }

    #[test]
    fn rocm_dash_tui_depending_on_rocm_core_fails() {
        let mut edges = allowlisted_edges();
        edges.push(edge("rocm-dash-tui", "rocm-core", Kind::Enforced));
        let violations = check(&edges);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("rocm-dash-tui must not depend on rocm-core"));
    }

    #[test]
    fn dev_dependency_edges_never_fail() {
        let mut edges = allowlisted_edges();
        // Not in ALLOWLIST, but a dev-dependency edge — must be exempt.
        edges.push(edge("rocm-dash-daemon", "rocm-core", Kind::Dev));
        assert!(check(&edges).is_empty());
    }

    #[test]
    fn unrelated_disallowed_edge_gets_generic_message() {
        let mut edges = allowlisted_edges();
        edges.push(edge("rocm-engine-vllm", "rocm-deps", Kind::Enforced));
        let violations = check(&edges);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("not in the declared allowlist"));
    }

    #[test]
    fn allowlist_never_contains_a_named_invariant_violation() {
        // Guards against the failure mode this file exists to prevent: a
        // contributor hitting a red CI run "fixing" it by adding the
        // disallowed edge straight into ALLOWLIST instead of removing it.
        for &(source, target) in ALLOWLIST {
            assert!(
                !violates_named_invariant(source, target),
                "ALLOWLIST contains `{source} -> {target}`, which violates a documented \
                 layering invariant — the fix is to remove the edge, not allowlist it"
            );
        }
    }

    #[test]
    fn build_dependency_edges_are_enforced_not_exempt() {
        // Exercises the `edges_from_metadata`/`edge_kind` path for a
        // `kind: "build"` first-party edge, which no real workspace member
        // currently has (both first-party-adjacent build-deps, `cc` and
        // `toml`, are third-party) — so without this fixture the `_ =>
        // Kind::Enforced` arm in `edge_kind` is otherwise untested.
        let metadata = Metadata {
            packages: vec![Package {
                name: "rocm-core".to_string(),
                dependencies: vec![Dependency {
                    name: "some-first-party-build-dep".to_string(),
                    kind: Some("build".to_string()),
                    path: Some("/workspace/some-first-party-build-dep".to_string()),
                }],
            }],
        };
        let edges = edges_from_metadata(&metadata);
        assert_eq!(
            edges,
            vec![(
                "rocm-core".to_string(),
                "some-first-party-build-dep".to_string(),
                Kind::Enforced,
            )]
        );
    }

    #[test]
    fn current_workspace_has_no_disallowed_edges() {
        // Regression guard against the real workspace: exercises the full
        // `cargo metadata` -> edge extraction -> allowlist-check path, the
        // same path `cargo xtask check-crate-edges` runs in CI, so a real
        // accidental edge or a stale allowlist is caught here too.
        let edges = load_edges().expect("cargo metadata should succeed in a workspace checkout");
        let violations = check(&edges);
        assert!(
            violations.is_empty(),
            "unexpected first-party edge(s) on the real workspace:\n{}",
            violations.join("\n")
        );
    }
}
