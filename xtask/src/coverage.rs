// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Per-crate line-coverage floors, committed to `coverage-floors.toml` and
//! enforced in CI.
//!
//! # Why per-crate, and why a committed file
//!
//! The gate this replaces was a single `--fail-under-lines 70` over four of the
//! fourteen workspace crates. That shape has two independent weaknesses:
//!
//! 1. **Scope.** Ten crates — including the largest — were covered by no floor
//!    at all, so deleting their tests could not fail any check.
//! 2. **Slack.** One number has to sit below the weakest crate in its scope, so
//!    the strong crates get no protection. Measured 2026-10-01, the four gated
//!    crates sat at 89.1% against a floor of 70%: nineteen points of headroom
//!    inside the gate's own scope.
//!
//! A single workspace-wide floor would reproduce weakness 2 across the whole
//! repo. Per-crate floors set just under each crate's measured value make the
//! gate load-bearing everywhere at once, without requiring anybody to raise
//! coverage first.
//!
//! The floors live in a committed file rather than in workflow YAML because
//! that is the pattern this repo already uses for measured-and-pinned values
//! (`expectations.toml`, `MANIFEST.md`, `THIRD_PARTY_NOTICES.txt`,
//! `runtime-deps.toml`): the value is reviewable in a diff, and `--bless`
//! regenerates it so raising a floor is a one-line change rather than an edit
//! to a workflow file.
//!
//! # What this metric does and does not measure
//!
//! The percentages are `cargo llvm-cov`'s reported line coverage, which counts
//! `#[cfg(test)]` modules as covered source. That inflates every figure, and
//! inflates hardest in the files with the most tests. It also *damps* this
//! gate: deleting a test module removes near-100%-covered lines from the
//! denominator at the same time as it removes coverage from the numerator, so
//! the reported percentage falls by less than the real loss.
//!
//! Measured against `crates/e2e-report` on 2026-10-01, deleting a 172-line test
//! module moved the crate from 89.49% to 88.22% — a real 1.27-point drop, but
//! only about a seventh of the per-file loss (`parse.rs` fell 89.22% → 70.18%).
//! The direction is reliable; the magnitude is not. That is why [`TOLERANCE`]
//! is small and the floors are per crate: a large crate dilutes a single
//! deleted module, so a workspace-wide number would hide it entirely.
//!
//! Excluding test modules from the measurement would be more honest, but there
//! is no robust way to do it on a stable toolchain for inline `#[cfg(test)]`
//! modules, and a line-counting heuristic would silently mis-attribute code the
//! day somebody writes `#[cfg(test)]` on an item rather than a module.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::paths;

/// Name of the committed floors file, relative to the workspace root.
const FLOORS_FILE: &str = "coverage-floors.toml";

/// Crates deliberately outside the gate.
///
/// `e2e-cucumber` is the end-to-end harness: its only test target is `e2e`,
/// which drives real binaries and is excluded from the unit-test run that
/// produces these numbers (see `test = false` on that target). Measuring it
/// here would report a floor for a target that never ran.
const EXCLUDED: &[&str] = &["e2e-cucumber"];

/// Slack allowed between a crate's committed floor and its measured coverage,
/// in percentage points.
///
/// Coverage is deterministic for a deterministic test suite, so this exists to
/// absorb small genuine movements (a refactor that moves lines between files,
/// a dependency bump that changes an inlined branch) rather than flakiness.
/// Kept small on purpose: the experiment documented in the module comment moved
/// a crate 1.27 points, so a tolerance much above this would start hiding real
/// losses in the larger crates.
const TOLERANCE: f64 = 0.2;

/// One crate's measured line coverage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    /// Lines with at least one executable region.
    pub lines: u64,
    /// Percentage of those lines executed at least once.
    pub percent: f64,
}

/// `cargo llvm-cov --json --summary-only` output, narrowed to what we read.
#[derive(Deserialize)]
struct LlvmCovReport {
    data: Vec<LlvmCovData>,
}

#[derive(Deserialize)]
struct LlvmCovData {
    files: Vec<LlvmCovFile>,
}

#[derive(Deserialize)]
struct LlvmCovFile {
    filename: String,
    summary: LlvmCovFileSummary,
}

#[derive(Deserialize)]
struct LlvmCovFileSummary {
    lines: LlvmCovLines,
}

#[derive(Deserialize)]
struct LlvmCovLines {
    count: u64,
    covered: u64,
}

/// `cargo metadata --no-deps` output, narrowed to the workspace members.
#[derive(Deserialize)]
struct Metadata {
    workspace_root: String,
    packages: Vec<MetadataPackage>,
}

#[derive(Deserialize)]
struct MetadataPackage {
    name: String,
    manifest_path: String,
}

/// The committed floors file.
#[derive(Deserialize)]
struct FloorsFile {
    floors: BTreeMap<String, f64>,
}

/// Map each workspace member to the directory holding its `Cargo.toml`.
///
/// Attribution is by manifest directory rather than by guessing at path
/// prefixes, so a crate that moves between `crates/` and `apps/` keeps
/// reporting under the same name.
fn member_dirs() -> Result<(PathBuf, BTreeMap<String, PathBuf>)> {
    let stdout = paths::run_cargo_metadata(&["--no-deps"])?;
    let metadata: Metadata =
        serde_json::from_slice(&stdout).context("failed to parse `cargo metadata` output")?;
    let root = PathBuf::from(&metadata.workspace_root);
    let mut dirs = BTreeMap::new();
    for package in metadata.packages {
        let manifest = PathBuf::from(&package.manifest_path);
        let dir = manifest
            .parent()
            .with_context(|| format!("manifest path has no parent: {}", manifest.display()))?
            .to_path_buf();
        dirs.insert(package.name, dir);
    }
    Ok((root, dirs))
}

/// Attribute each covered file to the workspace member whose directory is its
/// longest matching prefix, and total the lines per crate.
///
/// The longest-prefix rule matters because `cargo llvm-cov` reports files from
/// every crate in one document: a shorter prefix would let a nested member be
/// counted against its parent directory's crate.
fn aggregate(
    report: &LlvmCovReport,
    member_dirs: &BTreeMap<String, PathBuf>,
) -> BTreeMap<String, Measured> {
    let mut totals: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for data in &report.data {
        for file in &data.files {
            let path = Path::new(&file.filename);
            let owner = member_dirs
                .iter()
                .filter(|(_, dir)| path.starts_with(dir))
                .max_by_key(|(_, dir)| dir.as_os_str().len())
                .map(|(name, _)| name.clone());
            let Some(owner) = owner else { continue };
            let entry = totals.entry(owner).or_insert((0, 0));
            entry.0 += file.summary.lines.covered;
            entry.1 += file.summary.lines.count;
        }
    }
    totals
        .into_iter()
        .filter(|(_, (_, count))| *count > 0)
        .map(|(name, (covered, count))| {
            // `count > 0` is guaranteed by the filter above.
            #[allow(clippy::cast_precision_loss)]
            let percent = (covered as f64) * 100.0 / (count as f64);
            (
                name,
                Measured {
                    lines: count,
                    percent,
                },
            )
        })
        .collect()
}

/// Round down to two decimal places, so a blessed floor can never sit above the
/// value it was blessed from.
fn floor_to_hundredths(percent: f64) -> f64 {
    (percent * 100.0).floor() / 100.0
}

/// Compare measured coverage against committed floors.
///
/// Returns the human-readable failures, empty when everything holds. The check
/// is bidirectional, matching `crate_edges`: a crate with no floor fails just
/// as loudly as a crate below its floor, so adding a workspace member cannot
/// quietly land outside the gate, and a floor left behind by a deleted crate
/// cannot sit in the file pretending to guard something.
fn check_floors(
    measured: &BTreeMap<String, Measured>,
    floors: &BTreeMap<String, f64>,
) -> Vec<String> {
    let mut failures = Vec::new();

    let measured_names: BTreeSet<&str> = measured.keys().map(String::as_str).collect();
    let floor_names: BTreeSet<&str> = floors.keys().map(String::as_str).collect();

    for name in measured_names.difference(&floor_names) {
        failures.push(format!(
            "{name}: no floor in {FLOORS_FILE} — every gated crate needs one; run `cargo xtask coverage --bless`"
        ));
    }
    for name in floor_names.difference(&measured_names) {
        failures.push(format!(
            "{name}: has a floor in {FLOORS_FILE} but reported no coverage — remove the entry if the crate is gone"
        ));
    }

    for (name, value) in measured {
        let Some(floor) = floors.get(name) else {
            continue;
        };
        if value.percent < floor - TOLERANCE {
            failures.push(format!(
                "{name}: {:.2}% lines, below its floor of {floor:.2}% (tolerance {TOLERANCE}pp, {} lines measured)",
                value.percent, value.lines
            ));
        }
    }
    failures
}

/// Render the floors file from measured coverage.
fn render_floors(measured: &BTreeMap<String, Measured>) -> String {
    use std::fmt::Write as _;

    let mut out = String::from(
        "# Per-crate line-coverage floors, enforced by `cargo xtask coverage --check`.\n\
         #\n\
         # Generated by `cargo xtask coverage --bless`; do not edit by hand.\n\
         #\n\
         # Each floor is the crate's measured line coverage at the time it was blessed,\n\
         # rounded down. A crate falling more than the tolerance in `xtask/src/coverage.rs`\n\
         # below its floor fails CI. Raising a floor after adding tests is a re-bless.\n\
         #\n\
         # These are `cargo llvm-cov` line percentages, which count `#[cfg(test)]` modules\n\
         # as covered source. See the module comment in `xtask/src/coverage.rs` for what\n\
         # that inflates, and why the floors are per crate rather than one workspace number.\n\
         \n\
         [floors]\n",
    );
    for (name, value) in measured {
        // Writing to a String is infallible.
        let _ = writeln!(
            out,
            "\"{name}\" = {:.2}  # {} lines",
            floor_to_hundredths(value.percent),
            value.lines
        );
    }
    out
}

/// Run `cargo llvm-cov` over the gated crates and return its JSON report.
fn measure(root: &Path) -> Result<LlvmCovReport> {
    let report_path = paths::target_dir(root).join("xtask-coverage-summary.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut command = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    command
        .current_dir(root)
        // `--no-cfg-coverage` keeps this measurement identical to what the
        // crates compile to outside coverage, so a `cfg(coverage)` branch
        // cannot change which lines exist between a normal build and this one.
        .args(["llvm-cov", "--no-cfg-coverage", "--locked", "--workspace"])
        .args(["--summary-only", "--json", "--output-path"])
        .arg(&report_path);
    for name in EXCLUDED {
        command.args(["--exclude", name]);
    }

    let status = command.status().context(
        "failed to run `cargo llvm-cov`; install it with `cargo install cargo-llvm-cov`",
    )?;
    if !status.success() {
        bail!("`cargo llvm-cov` failed; fix the failing tests before checking coverage floors");
    }

    let raw = fs::read(&report_path)
        .with_context(|| format!("failed to read {}", report_path.display()))?;
    serde_json::from_slice(&raw).context("failed to parse `cargo llvm-cov` JSON report")
}

/// Verify (or, with `bless`, rewrite) the per-crate coverage floors.
pub fn run(bless: bool) -> Result<()> {
    let (root, member_dirs) = member_dirs()?;
    let gated: BTreeMap<String, PathBuf> = member_dirs
        .into_iter()
        .filter(|(name, _)| !EXCLUDED.contains(&name.as_str()))
        .collect();

    let report = measure(&root)?;
    let measured = aggregate(&report, &gated);
    if measured.is_empty() {
        bail!("`cargo llvm-cov` reported no coverage for any workspace crate");
    }

    let floors_path = root.join(FLOORS_FILE);

    if bless {
        fs::write(&floors_path, render_floors(&measured))
            .with_context(|| format!("failed to write {}", floors_path.display()))?;
        println!(
            "wrote {} ({} crates)",
            floors_path.display(),
            measured.len()
        );
        return Ok(());
    }

    let raw = fs::read_to_string(&floors_path).with_context(|| {
        format!(
            "failed to read {}; run `cargo xtask coverage --bless` to create it",
            floors_path.display()
        )
    })?;
    let floors: FloorsFile = toml::from_str(&raw)
        .with_context(|| format!("failed to parse {}", floors_path.display()))?;

    let failures = check_floors(&measured, &floors.floors);
    if !failures.is_empty() {
        bail!(
            "coverage floors not met ({} problem(s)):\n{}\n\n\
             If the drop is intended — a crate shrank, or tests moved to another crate — \
             re-bless with `cargo xtask coverage --bless` and explain the change in the commit.",
            failures.len(),
            failures.join("\n")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(entries: &[(&str, f64)]) -> BTreeMap<String, Measured> {
        entries
            .iter()
            .map(|(name, percent)| {
                (
                    (*name).to_string(),
                    Measured {
                        lines: 1000,
                        percent: *percent,
                    },
                )
            })
            .collect()
    }

    fn floors(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
        entries
            .iter()
            .map(|(name, floor)| ((*name).to_string(), *floor))
            .collect()
    }

    #[test]
    fn a_crate_at_its_floor_passes() {
        let failures = check_floors(&measured(&[("rocm", 77.0)]), &floors(&[("rocm", 77.0)]));
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_crate_below_its_floor_beyond_tolerance_fails_and_names_itself() {
        let failures = check_floors(&measured(&[("rocm", 76.5)]), &floors(&[("rocm", 77.0)]));
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].starts_with("rocm: 76.50% lines, below its floor"),
            "{failures:?}"
        );
    }

    #[test]
    fn a_drop_within_tolerance_passes() {
        // 0.1pp under the floor, inside the 0.2pp tolerance.
        let failures = check_floors(&measured(&[("rocm", 76.9)]), &floors(&[("rocm", 77.0)]));
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_drop_just_past_tolerance_fails() {
        // 0.25pp under the floor, outside the 0.2pp tolerance. Pins the boundary
        // so widening the tolerance cannot pass unnoticed.
        let failures = check_floors(&measured(&[("rocm", 76.75)]), &floors(&[("rocm", 77.0)]));
        assert_eq!(failures.len(), 1, "{failures:?}");
    }

    #[test]
    fn a_measured_crate_with_no_floor_fails() {
        // The case that lets a new workspace member land outside the gate.
        let failures = check_floors(
            &measured(&[("rocm", 77.0), ("newcomer", 12.0)]),
            &floors(&[("rocm", 77.0)]),
        );
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("newcomer: no floor"), "{failures:?}");
    }

    #[test]
    fn a_floor_with_no_measured_crate_fails() {
        let failures = check_floors(
            &measured(&[("rocm", 77.0)]),
            &floors(&[("rocm", 77.0), ("departed", 50.0)]),
        );
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("departed: has a floor"),
            "{failures:?}"
        );
    }

    #[test]
    fn every_failing_crate_is_reported_not_just_the_first() {
        let failures = check_floors(
            &measured(&[("rocm", 10.0), ("rocm-core", 20.0)]),
            &floors(&[("rocm", 77.0), ("rocm-core", 81.0)]),
        );
        assert_eq!(failures.len(), 2, "{failures:?}");
    }

    #[test]
    fn blessing_rounds_down_so_a_floor_never_exceeds_what_it_was_blessed_from() {
        // 77.999% must not bless to 78.00%, which the next run would fail.
        // Compared with a tolerance far below the 0.01 the function rounds to,
        // so this still distinguishes 77.99 from 78.00.
        let near = |actual: f64, expected: f64| (actual - expected).abs() < 1e-9;
        assert!(near(floor_to_hundredths(77.999), 77.99));
        assert!(near(floor_to_hundredths(77.0), 77.0));
        // Already on a hundredth: rounding down must not drop a step.
        assert!(near(floor_to_hundredths(63.82), 63.82));
    }

    #[test]
    fn a_blessed_file_passes_the_check_it_was_blessed_from() {
        // The round trip the whole design rests on: bless then check must be green.
        let m = measured(&[("rocm", 77.999), ("rocm-core", 81.004)]);
        let rendered = render_floors(&m);
        let parsed: FloorsFile = toml::from_str(&rendered).expect("rendered floors must parse");
        assert!(check_floors(&m, &parsed.floors).is_empty());
    }

    #[test]
    fn a_file_is_attributed_to_the_longest_matching_member_directory() {
        // A member nested inside another member's directory must not be counted
        // against the outer crate.
        let dirs: BTreeMap<String, PathBuf> = [
            ("outer".to_string(), PathBuf::from("/ws/apps")),
            ("inner".to_string(), PathBuf::from("/ws/apps/nested")),
        ]
        .into_iter()
        .collect();
        let report = LlvmCovReport {
            data: vec![LlvmCovData {
                files: vec![LlvmCovFile {
                    filename: "/ws/apps/nested/src/lib.rs".to_string(),
                    summary: LlvmCovFileSummary {
                        lines: LlvmCovLines {
                            count: 10,
                            covered: 5,
                        },
                    },
                }],
            }],
        };
        let totals = aggregate(&report, &dirs);
        assert_eq!(totals.len(), 1);
        assert_eq!(
            totals["inner"],
            Measured {
                lines: 10,
                percent: 50.0
            }
        );
    }

    #[test]
    fn files_outside_every_member_directory_are_ignored() {
        // Registry sources land in the report too; they are not ours to gate.
        let dirs: BTreeMap<String, PathBuf> =
            BTreeMap::from([("rocm".to_string(), PathBuf::from("/ws/apps/rocm"))]);
        let report = LlvmCovReport {
            data: vec![LlvmCovData {
                files: vec![LlvmCovFile {
                    filename: "/home/user/.cargo/registry/src/foo-1.0/src/lib.rs".to_string(),
                    summary: LlvmCovFileSummary {
                        lines: LlvmCovLines {
                            count: 10,
                            covered: 0,
                        },
                    },
                }],
            }],
        };
        assert!(aggregate(&report, &dirs).is_empty());
    }
}
