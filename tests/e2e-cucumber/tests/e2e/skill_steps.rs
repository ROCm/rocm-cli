// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Contract steps for the `rocm-doctor` skill (`skills/rocm-doctor/`).
//!
//! The skill is a thin driver: the probe, the closed catalog and the fixes all
//! live in `crates/rocm-core` and ship with the binary. What the skill owns is a
//! set of literal claims about how that binary behaves. These steps drive the
//! real `rocm` through the sequence the skill prescribes and check the claims
//! still hold.
//!
//! `reference.md` is read as the EXPECTED-value fixture. That is a deliberate,
//! narrow exception to the suite's black-box rule: nothing is imported from the
//! rocm-cli codebase — a documentation artifact is read as test data, and that
//! artifact is the thing under test.

use std::collections::BTreeMap;
use std::path::PathBuf;

use cucumber::{given, then, when};

use crate::E2eWorld;

/// Same symptom the diagnose steps use: it keys off a `LINUX_AND_WINDOWS`
/// checker and so renders identically on either OS. See the rationale on
/// `diagnose_steps::KNOWN_SYMPTOM`.
const KNOWN_SYMPTOM: &str = "HSA_STATUS_ERROR_INVALID_ISA";

/// The four ids the skill names as the ones the CLI will apply itself. Spelled
/// out here so a rename in the catalog fails loudly rather than being absorbed
/// by a set comparison that only ever sees the doc and the CLI agree.
const AUTO_APPLICABLE: [&str; 4] = [
    "fix-2-unset-override",
    "fix-4-render-group",
    "fix-6-path",
    "fix-9-igpu-dgpu",
];

/// Verdicts `rocm examine` can report, as enumerated by the skill's reference.
const KNOWN_VERDICTS: [&str; 5] = ["ok", "no-amd-gpu", "wsl", "unsupported-os", "degraded"];

/// Confidence thresholds the skill's workflow reasons about ("`score >= 75` =
/// high confidence; `50-74` = likely").
const MIN_SCORE_FOR_MATCH: i64 = 50;
const HIGH_CONFIDENCE_THRESHOLD: i64 = 75;

/// One catalog row, from either side of the comparison.
#[derive(Debug, PartialEq, Eq)]
struct Remediation {
    /// Machines it applies to, normalised to the CLI's spelling
    /// (`linux`, `windows`, `linux/windows`).
    os_scope: String,
    /// Whether the CLI applies it itself, as opposed to printing a plan.
    auto: bool,
}

fn reference_md_path() -> PathBuf {
    // <repo>/tests/e2e-cucumber -> <repo>/skills/rocm-doctor/reference.md
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("skills")
        .join("rocm-doctor")
        .join("reference.md")
}

/// Read the closed-catalog table out of the skill's reference doc.
///
/// Rows look like:
/// `| `fix-1-arch` | both | <mode> | <signal> | no |`
/// Only rows whose first cell is a backticked `fix-*` id are taken, which skips
/// the header, the separator, and the exit-code table further up the file.
fn parse_reference_catalog(md: &str) -> BTreeMap<String, Remediation> {
    let mut out = BTreeMap::new();
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
        let os_scope = match cells[1] {
            "both" => "linux/windows".to_owned(),
            other => other.to_owned(),
        };
        let auto = match *cells.last().expect("row has cells") {
            "yes" => true,
            "no" => false,
            other => panic!("{id}: unexpected Auto-fix cell {other:?} in reference.md"),
        };
        out.insert(id.to_owned(), Remediation { os_scope, auto });
    }
    assert!(
        !out.is_empty(),
        "no catalog rows found in {}",
        reference_md_path().display()
    );
    out
}

/// Read the same shape out of `rocm fix`, whose rows look like:
/// `  [      AUTO] [ linux/windows] fix-2-unset-override  -- Unset ...`
fn parse_fix_listing(stdout: &str) -> BTreeMap<String, Remediation> {
    let mut out = BTreeMap::new();
    for line in stdout.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix('[') else {
            continue;
        };
        let Some((marker, rest)) = rest.split_once(']') else {
            continue;
        };
        let auto = match marker.trim() {
            "AUTO" => true,
            "PRINT-ONLY" => false,
            other => panic!("unexpected applicability marker {other:?} in `rocm fix` listing"),
        };
        let Some((os_scope, rest)) = rest.trim_start().trim_start_matches('[').split_once(']')
        else {
            continue;
        };
        let id = rest.split_whitespace().next().unwrap_or_default();
        if !id.starts_with("fix-") {
            continue;
        }
        out.insert(
            id.to_owned(),
            Remediation {
                os_scope: os_scope.trim().to_owned(),
                auto,
            },
        );
    }
    assert!(
        !out.is_empty(),
        "no fix rows parsed out of the `rocm fix` listing:\n{stdout}"
    );
    out
}

/// Both sides of a comparison, or a panic explaining which step is missing.
fn both_sides(world: &E2eWorld) -> (BTreeMap<String, Remediation>, BTreeMap<String, Remediation>) {
    let doc = world
        .skill_reference
        .as_ref()
        .expect("skill reference not loaded");
    let listing = world
        .cli_output
        .as_ref()
        .expect("no `rocm fix` listing captured");
    (parse_reference_catalog(doc), parse_fix_listing(listing))
}

/// The parsed diagnosis report, or a panic quoting what was emitted instead.
fn diagnosis(world: &E2eWorld) -> serde_json::Value {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "the skill reads the JSON, not the exit code: diagnose must always exit 0"
    );
    let output = world.cli_output.as_ref().expect("no diagnose output");
    serde_json::from_str(output).expect("diagnose --json did not emit valid JSON")
}

// ── Given ──────────────────────────────────────────────────────────

#[given("the ROCm Doctor skill as it is published")]
async fn load_skill_reference(world: &mut E2eWorld) {
    let path = reference_md_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    world.skill_reference = Some(text);
}

#[given("a user who reports a recognised ROCm failure")]
async fn user_reports_known_failure(world: &mut E2eWorld) {
    world.model_name = Some(KNOWN_SYMPTOM.to_owned());
}

#[given("an agent that picked a remediation meant for a different kind of machine")]
async fn agent_picks_wrong_os_fix(world: &mut E2eWorld) {
    // Chosen from the OTHER OS's set so the scenario needs no @requires-os tag
    // and behaves the same on every lane.
    let id = if cfg!(windows) {
        "fix-5-amdgpu-load"
    } else {
        "fix-13-hip-sdk-missing"
    };
    world.model_name = Some(id.to_owned());
}

// ── When ───────────────────────────────────────────────────────────

#[when("an agent asks the CLI which remediations it knows")]
async fn agent_lists_remediations(world: &mut E2eWorld) {
    let (stdout, _, rc) = crate::run_rocm(world, &["fix"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("an agent asks the CLI to diagnose that report for tooling")]
async fn agent_diagnoses_for_tooling(world: &mut E2eWorld) {
    let symptom = world.model_name.clone().expect("no symptom set");
    let (stdout, _, rc) = crate::run_rocm(world, &["diagnose", "--symptom", &symptom, "--json"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("an agent inspects the machine for tooling")]
async fn agent_examines_for_tooling(world: &mut E2eWorld) {
    let (stdout, _, rc) = crate::run_rocm(world, &["examine", "--json"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("the agent asks the CLI to apply that remediation")]
async fn agent_applies_remediation(world: &mut E2eWorld) {
    let fix_id = world.model_name.clone().expect("no fix id set");
    let (stdout, stderr, rc) = crate::run_rocm(world, &["fix", &fix_id]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the skill and the CLI describe the same set of remediations")]
async fn assert_same_ids(world: &mut E2eWorld) {
    let (doc, cli) = both_sides(world);
    let documented: Vec<&String> = doc.keys().collect();
    let offered: Vec<&String> = cli.keys().collect();
    assert_eq!(
        documented, offered,
        "the skill's catalog and `rocm fix` disagree.\n\
         The catalog is authoritative in crates/rocm-core — update \
         skills/rocm-doctor/reference.md (and the amd/skills copy) to match the CLI.\n\
         documented: {documented:?}\n\
         offered:    {offered:?}"
    );
}

#[then("the skill and the CLI agree on which ones the CLI applies without help")]
async fn assert_same_auto_set(world: &mut E2eWorld) {
    let (doc, cli) = both_sides(world);
    for (id, offered) in &cli {
        let documented = doc.get(id).unwrap_or_else(|| {
            panic!(
                "`rocm fix` offers {id}, which skills/rocm-doctor/reference.md does not document"
            )
        });
        assert_eq!(
            documented.auto, offered.auto,
            "{id}: reference.md says auto-applicable={}, `rocm fix` says {}",
            documented.auto, offered.auto
        );
    }
    // Pin the set itself: both files name these four ids in prose, so a rename
    // that happened to be applied to reference.md and the CLI together would
    // still leave SKILL.md's workflow text stale.
    let auto: Vec<&String> = cli
        .iter()
        .filter(|(_, r)| r.auto)
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        auto, AUTO_APPLICABLE,
        "the set of fixes the CLI applies itself changed; the skill names these four in prose"
    );
}

#[then("the skill and the CLI agree on which machines each remediation is for")]
async fn assert_same_os_scope(world: &mut E2eWorld) {
    let (doc, cli) = both_sides(world);
    for (id, offered) in &cli {
        let documented = doc
            .get(id)
            .unwrap_or_else(|| panic!("`rocm fix` offers {id}, undocumented in reference.md"));
        assert_eq!(
            documented.os_scope, offered.os_scope,
            "{id}: reference.md scopes it to {:?}, `rocm fix` to {:?}",
            documented.os_scope, offered.os_scope
        );
    }
}

#[then("the diagnosis carries the confidence thresholds the skill reasons about")]
async fn assert_thresholds(world: &mut E2eWorld) {
    let report = diagnosis(world);
    for (field, expected) in [
        ("min_score_for_match", MIN_SCORE_FOR_MATCH),
        ("high_confidence_threshold", HIGH_CONFIDENCE_THRESHOLD),
    ] {
        let actual = report
            .get(field)
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_else(|| panic!("diagnose JSON has no numeric '{field}'"));
        assert_eq!(
            actual, expected,
            "{field} changed; the skill quotes {expected} in its workflow text"
        );
    }
}

#[then("every cause it offers carries a title, a confidence, its evidence, and a plan")]
async fn assert_match_shape(world: &mut E2eWorld) {
    let report = diagnosis(world);
    let matched = report
        .get("matched")
        .and_then(serde_json::Value::as_array)
        .expect("diagnose JSON has no 'matched' array");
    // Deliberately not asserting `matched` is non-empty: on a host the catalog
    // rules out of scope (WSL2) an empty list is the correct answer, and
    // scenario 5 covers that branch. What must hold is that anything offered is
    // fully readable by an agent following the skill.
    for cause in matched {
        for field in ["id", "title", "score", "evidence", "fix"] {
            assert!(
                cause.get(field).is_some(),
                "a matched cause is missing '{field}', which the skill reads:\n{cause:#}"
            );
        }
        let fix = &cause["fix"];
        for field in [
            "fix_id",
            "summary",
            "commands",
            "verify",
            "notes",
            "needs_sudo",
            "needs_reboot",
            "needs_relogin",
            "auto_applicable",
        ] {
            assert!(
                fix.get(field).is_some(),
                "a fix plan is missing '{field}', which the skill reads:\n{fix:#}"
            );
        }
    }
}

#[then("the inspection succeeds whatever it finds")]
async fn assert_examine_exits_zero(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "the skill reads `status`, not the exit code: examine must always exit 0"
    );
}

#[then("its verdict is one the skill accounts for")]
async fn assert_known_verdict(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no examine output");
    let report: serde_json::Value =
        serde_json::from_str(output).expect("examine --json did not emit valid JSON");
    let status = report
        .get("status")
        .and_then(serde_json::Value::as_str)
        .expect("examine JSON has no 'status'");
    assert!(
        KNOWN_VERDICTS.contains(&status),
        "examine reported {status:?}, which the skill does not account for \
         (it documents {KNOWN_VERDICTS:?})"
    );
}

#[then("the CLI declines because it does not apply here")]
async fn assert_not_applicable(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(3),
        "a fix scoped to another OS must exit 3 (not applicable on this host)"
    );
}

#[then("no managed state is written")]
async fn assert_no_managed_state(world: &mut E2eWorld) {
    // Same definition of "changed nothing" the fix-preview scenario uses:
    // incidental dirs (logging init) don't count, managed state does.
    let root = world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path();
    for managed in [
        root.join("data").join("runtimes"),
        root.join("data").join("services"),
        root.join("config"),
    ] {
        let touched = managed.read_dir().is_ok_and(|mut d| d.next().is_some());
        assert!(
            !touched,
            "a declined fix wrote managed state at {}",
            managed.display()
        );
    }
}
