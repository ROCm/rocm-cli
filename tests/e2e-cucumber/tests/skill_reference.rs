// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Drift guard for the shape of `skills/rocm-doctor/reference.md`'s catalog.
//!
//! The `rocm_doctor_skill.feature` scenarios compare that table against the
//! real `rocm fix` listing, cell for cell, so a cell the table spells its own
//! way is a contract failure. Those scenarios need a built `rocm` binary and
//! only run under `cargo xtask e2e`; this file runs in the ordinary `cargo
//! test` set, so the malformed cell is caught without one.
//!
//! Narrow on purpose: the CLI is authoritative for *which* fixes exist and what
//! they are scoped to, and the feature file already checks that. All this
//! checks is that the document uses the vocabulary the CLI prints, so the
//! comparison there can stay a literal one.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The platform atoms `rocm fix` prints, which it builds by joining a recipe's
/// `applies_on` with `/` (see `FixRecipe` in `crates/rocm-core/src/fix.rs`).
/// A catalog cell is any non-empty `/`-joined subset of these.
const PLATFORM_ATOMS: &[&str] = &["linux", "windows", "wsl"];

fn reference_md_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("skills")
        .join("rocm-doctor")
        .join("reference.md")
}

/// `(fix-id, os-scope cell)` for every catalog row, taken the same way
/// `skill_steps::parse_reference_catalog` takes them: a leading `|`, at least
/// five cells, and a backticked `fix-*` id first.
fn catalog_rows(md: &str) -> Vec<(String, String)> {
    let mut rows = Vec::new();
    for line in md.lines() {
        let line = line.trim();
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        if cells.len() < 5 {
            continue;
        }
        let id = cells[0].trim_matches('`');
        if !id.starts_with("fix-") {
            continue;
        }
        rows.push((id.to_owned(), cells[1].to_owned()));
    }
    rows
}

fn skill_md_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("skills")
        .join("rocm-doctor")
        .join("SKILL.md")
}

/// How many catalog rows have `yes` in the Auto-fix cell (`cells[4]`, the same
/// column `catalog_rows` leaves unparsed since the os-scope check only needs
/// `cells[1]`).
fn catalog_auto_count(md: &str) -> usize {
    md.lines()
        .filter(|line| {
            let line = line.trim();
            if !line.starts_with('|') {
                return false;
            }
            let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
            cells.len() >= 5 && cells[0].trim_matches('`').starts_with("fix-") && cells[4] == "yes"
        })
        .count()
}

/// Two free-standing counts restate the table in prose rather than in cells a
/// parser already checks: `reference.md`'s "(N failure modes)" heading, and
/// `SKILL.md`'s "the other M are print-only" line. `rocm_doctor_skill.feature`
/// says plainly that these are not parsed and can go stale even while the
/// table itself stays correct -- which is exactly what happened here (the
/// catalog grew by one row and both numbers were left behind). This does not
/// reopen that scope decision; it only adds a floor cheap enough that the next
/// drift fails a `cargo test` instead of waiting for someone to count rows by
/// hand.
#[test]
fn free_standing_catalog_counts_match_the_table() {
    let reference_md = std::fs::read_to_string(reference_md_path())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", reference_md_path().display()));
    let rows = catalog_rows(&reference_md);
    assert!(
        !rows.is_empty(),
        "no catalog rows found in {}",
        reference_md_path().display()
    );
    let total = rows.len();
    let auto = catalog_auto_count(&reference_md);

    let heading = reference_md
        .lines()
        .find(|line| line.trim_start().starts_with("## Closed catalog ("))
        .unwrap_or_else(|| {
            panic!(
                "no '## Closed catalog (N failure modes)' heading found in {}",
                reference_md_path().display()
            )
        });
    let heading_count: usize = heading
        .split('(')
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("could not parse the failure-mode count out of {heading:?}"));
    assert_eq!(
        heading_count, total,
        "skills/rocm-doctor/reference.md's '(N failure modes)' heading says {heading_count}, \
         but the table has {total} rows"
    );

    let skill_md = std::fs::read_to_string(skill_md_path())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", skill_md_path().display()));
    // "The other N" and "are **print-only**" sit on adjacent source lines that
    // wrap a single sentence, so join the whole doc on whitespace first rather
    // than matching one line -- a line-scoped search would find the
    // print-only line without the number on it and fail to parse.
    let flattened = skill_md.split_whitespace().collect::<Vec<_>>().join(" ");
    let after_the_other = flattened.split("The other ").nth(1).unwrap_or_else(|| {
        panic!(
            "no 'The other N ... print-only' phrase found in {}",
            skill_md_path().display()
        )
    });
    assert!(
        after_the_other.starts_with(|c: char| c.is_ascii_digit())
            && after_the_other.contains("print-only"),
        "'The other N' in {} is not followed by a number and 'print-only' as expected: {:?}",
        skill_md_path().display(),
        after_the_other.chars().take(60).collect::<String>()
    );
    let print_only_count: usize = after_the_other
        .split_whitespace()
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| {
            panic!("could not parse the print-only count out of {after_the_other:?}")
        });
    assert_eq!(
        print_only_count,
        total - auto,
        "skills/rocm-doctor/SKILL.md says {print_only_count} fixes are print-only, but the \
         table has {total} rows of which {auto} are auto-applicable ({} expected)",
        total - auto
    );
}

#[test]
fn catalog_os_scopes_use_the_cli_spellings() {
    // A shorthand such as `both` reads as "every platform" but cannot say
    // which, so any mapping of it back onto the CLI's atoms is a guess --
    // `linux/windows` for a row that meant linux/windows/wsl drops `wsl`
    // silently, which is precisely the mismatch the skill contract exists to
    // catch. The document does not get to invent one: it spells the platforms
    // out, the comparison stays literal, and nothing has to be normalised.
    let md = std::fs::read_to_string(reference_md_path())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", reference_md_path().display()));
    let rows = catalog_rows(&md);
    assert!(
        !rows.is_empty(),
        "no catalog rows found in {}",
        reference_md_path().display()
    );

    let allowed: BTreeSet<&str> = PLATFORM_ATOMS.iter().copied().collect();
    let bad: Vec<&(String, String)> = rows
        .iter()
        .filter(|(_, scope)| {
            scope.is_empty() || scope.split('/').any(|atom| !allowed.contains(atom))
        })
        .collect();
    assert!(
        bad.is_empty(),
        "skills/rocm-doctor/reference.md scopes these fixes with words `rocm fix` never prints: \
         {bad:?}.\n\
         An OS-scope cell must be a `/`-joined list of {PLATFORM_ATOMS:?}, matching the recipe's \
         `applies_on` in crates/rocm-core/src/fix.rs. Spell the platforms out instead of \
         introducing a shorthand: the skill contract compares these cells literally, and a \
         shorthand would have to be translated -- which is how a dropped platform hides."
    );
}
