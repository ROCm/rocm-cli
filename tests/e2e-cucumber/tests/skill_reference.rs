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

/// The marker words `rocm fix`'s listing prints (`FixClass::marker`, lower-cased
/// to match the table's style), plus `yes`/`no` -- recognised only so this
/// guard can name them explicitly as the vocabulary the class rewrite retired,
/// rather than reporting them as merely unrecognised.
const MARKER_WORDS: &[&str] = &["auto", "needs-arg", "print-only", "diagnose-only"];
const RETIRED_MARKER_WORDS: &[&str] = &["yes", "no"];

fn reference_md_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("skills")
        .join("rocm-doctor")
        .join("reference.md")
}

/// `(fix-id, os-scope cell, marker cell)` for every catalog row, taken the same
/// way `skill_steps::parse_reference_catalog` takes them: a leading `|`, at
/// least five cells, and a backticked `fix-*` id first.
fn catalog_rows(md: &str) -> Vec<(String, String, String)> {
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
        rows.push((
            id.to_owned(),
            cells[1].to_owned(),
            (*cells.last().expect("row has cells")).to_owned(),
        ));
    }
    rows
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
    let bad: Vec<&(String, String, String)> = rows
        .iter()
        .filter(|(_, scope, _)| {
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

#[test]
fn catalog_markers_use_the_cli_vocabulary() {
    // `skill_steps::parse_fix_listing` and `parse_reference_catalog` drop any
    // row whose marker is unrecognised rather than panic, so a reclassified
    // entry whose table cell still says `yes`/`no` -- the vocabulary the
    // `FixClass` rewrite retired -- disappears from the comparison silently
    // instead of failing on a marker mismatch. That is exactly the trap that
    // let `fix-9-igpu-dgpu`'s `NEEDS-ARG` reclassification vanish from the
    // skill-01 scenario's set comparison instead of showing up as a mismatch
    // on skill-02. This guard runs in the ordinary `cargo test` set, with no
    // built `rocm` binary needed, so the retired word is caught here first.
    let md = std::fs::read_to_string(reference_md_path())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", reference_md_path().display()));
    let rows = catalog_rows(&md);
    assert!(
        !rows.is_empty(),
        "no catalog rows found in {}",
        reference_md_path().display()
    );

    // A marker cell is a bare word, or a word plus a per-platform exception --
    // `print-only (auto on windows)` -- for the one entry whose behaviour
    // depends on more than the host. Only the base word is checked against the
    // vocabulary; `parse_marker_cell` in skill_steps.rs owns the exception
    // syntax itself.
    let base_word = |cell: &str| -> String {
        match cell.find('(') {
            Some(i) => cell[..i].trim().to_owned(),
            None => cell.trim().to_owned(),
        }
    };

    let allowed: BTreeSet<&str> = MARKER_WORDS.iter().copied().collect();
    let retired: BTreeSet<&str> = RETIRED_MARKER_WORDS.iter().copied().collect();
    let bad: Vec<(&String, String)> = rows
        .iter()
        .map(|(id, _, marker)| (id, base_word(marker)))
        .filter(|(_, word)| !allowed.contains(word.as_str()))
        .collect();
    assert!(
        bad.is_empty(),
        "skills/rocm-doctor/reference.md marks these fixes with words `rocm fix` never prints: \
         {bad:?}.\n\
         A marker cell must be one of {MARKER_WORDS:?}, matching `FixClass::marker` in \
         crates/rocm-core/src/fix.rs (lower-cased). {} of those are `yes`/`no`, the vocabulary \
         the class rewrite retired -- replace with the CLI's own marker word.",
        bad.iter()
            .filter(|(_, word)| retired.contains(word.as_str()))
            .count()
    );
}
