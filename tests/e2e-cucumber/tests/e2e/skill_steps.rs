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

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use cucumber::{given, then, when};

use crate::E2eWorld;

/// Same symptom the diagnose steps use: it keys off a `LINUX_AND_WINDOWS`
/// checker and so renders identically on either OS. See the rationale on
/// `diagnose_steps::KNOWN_SYMPTOM`.
const KNOWN_SYMPTOM: &str = "HSA_STATUS_ERROR_INVALID_ISA";

/// Prose with no catalog keyword in it, so nothing scores *from the symptom*.
///
/// It does not follow that the report goes unexplained: several checkers score
/// from host state alone, so a machine with a real problem still matches. The
/// route is populated either way, which is what the scenario using this checks.
const UNMATCHED_SYMPTOM: &str = "the office printer keeps jamming on page three";

/// The reference states the auto-applicable set twice: once as `yes` cells in
/// the catalog table, and once in prose above it. Both are parsed, so a rename
/// applied to the table and the CLI together still fails if the prose was
/// missed — and neither side is a constant restated in this file.
const AUTO_APPLICABLE_PROSE: &str = "auto-applicable";

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
        // An unrecognised cell drops the row rather than aborting the parse, so
        // a reworded table surfaces as the scenario's own set diff — naming the
        // ids that went missing — instead of a panic from inside the reader.
        let auto = match *cells.last().expect("row has cells") {
            "yes" => true,
            "no" => false,
            _ => continue,
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

/// The fix-ids the reference's prose names as auto-applicable.
///
/// The sentence spans two lines and is delimited by em dashes:
/// `Only four fixes are auto-applicable — `fix-2-…`, `fix-4-…` — and the rest…`
/// Bounding on the dashes keeps the `rocm fix fix-2-unset-override` example
/// later in the same paragraph out of the set.
fn documented_auto_prose(md: &str) -> BTreeSet<String> {
    let joined = md.replace('\n', " ");
    let Some((_, after)) = joined.split_once(AUTO_APPLICABLE_PROSE) else {
        panic!(
            "reference.md no longer states which fixes are {AUTO_APPLICABLE_PROSE} in prose; \
             the catalog table alone cannot catch a rename that missed the prose"
        )
    };
    let (_, inside) = after
        .split_once('—')
        .expect("the auto-applicable sentence no longer opens with an em dash");
    let (list, _) = inside
        .split_once('—')
        .expect("the auto-applicable sentence no longer closes with an em dash");
    let ids: BTreeSet<String> = backticked(list)
        .into_iter()
        .filter(|span| span.starts_with("fix-"))
        .map(str::to_owned)
        .collect();
    assert!(
        !ids.is_empty(),
        "no fix-ids parsed out of reference.md's auto-applicable sentence: {list:?}"
    );
    ids
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
        // As above: a renamed marker drops the row, and the scenario reports it
        // as an id `rocm fix` no longer offers rather than as a parser panic.
        let auto = match marker.trim() {
            "AUTO" => true,
            "PRINT-ONLY" => false,
            _ => continue,
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

// ── Reading the rest of the reference as expected values ───────────
//
// The catalog table above is not the only claim the skill makes. It also names
// the fields a diagnosis carries, the confidence thresholds an agent reasons
// about, the verdicts `examine` can return, and where to send a report nothing
// matched. Those are parsed out of the document too, rather than restated as
// constants here: a constant only ever proves this file and the CLI agree,
// which is not the drift this feature exists to catch. Parsing also means a
// claim ADDED upstream starts being checked without touching this file.
//
// Every parser asserts it found something. A heading or bullet reworded
// upstream must fail loudly — a silent empty parse turns the assertions it
// feeds into no-ops, which is the worst outcome available here.

/// The body of one `##`/`###` section, selected by how its heading starts.
/// Ends at the next heading of any depth.
fn section<'a>(md: &'a str, heading_starts_with: &str) -> Vec<&'a str> {
    let mut body = Vec::new();
    let mut inside = false;
    for line in md.lines() {
        if let Some(heading) = line.strip_prefix('#') {
            if inside {
                break;
            }
            inside = heading
                .trim_start_matches('#')
                .trim_start()
                .starts_with(heading_starts_with);
            continue;
        }
        if inside {
            body.push(line);
        }
    }
    assert!(
        !body.is_empty(),
        "reference.md has no section whose heading starts {heading_starts_with:?}"
    );
    body
}

/// Every backtick-delimited span on a line, in order.
fn backticked(line: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('`') else { break };
        spans.push(&after[..close]);
        rest = &after[close + 1..];
    }
    spans
}

/// Whether a backticked span is a bare JSON field name, as opposed to the other
/// things the reference puts in backticks (`{ id, ... }` shapes, `>= 75`,
/// `--json`).
fn is_field_name(span: &str) -> bool {
    let name = span.strip_suffix("[]").unwrap_or(span);
    !name.is_empty()
        && name.starts_with(|c: char| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Top-level fields of `diagnose --json` that the reference tells an agent to
/// read.
fn documented_diagnose_fields(md: &str) -> BTreeSet<String> {
    let fields: BTreeSet<String> = section(md, "`rocm diagnose")
        .iter()
        .filter(|line| line.trim_start().starts_with("- "))
        .flat_map(|line| backticked(line))
        .filter(|span| is_field_name(span))
        .map(|span| span.strip_suffix("[]").unwrap_or(span).to_owned())
        .collect();
    assert!(
        !fields.is_empty(),
        "no diagnose fields parsed out of reference.md"
    );
    fields
}

/// The per-cause shape the reference spells out: ``matched[]`` — ranked
/// `{ id, title, score, evidence[], fix }`.
fn documented_cause_fields(md: &str) -> BTreeSet<String> {
    let fields: BTreeSet<String> = section(md, "`rocm diagnose")
        .iter()
        .flat_map(|line| backticked(line))
        .find(|span| span.trim_start().starts_with('{'))
        .unwrap_or_else(|| panic!("reference.md no longer spells out the shape of a matched cause"))
        .trim_matches(|c| c == '{' || c == '}')
        .split(',')
        .map(|field| {
            let field = field.trim();
            field.strip_suffix("[]").unwrap_or(field).to_owned()
        })
        .filter(|field| !field.is_empty())
        .collect();
    assert!(
        !fields.is_empty(),
        "no per-cause fields parsed out of reference.md"
    );
    fields
}

/// Thresholds the reference states inline, as ``name` (50)`.
fn documented_thresholds(md: &str) -> BTreeMap<String, i64> {
    let mut out = BTreeMap::new();
    for line in section(md, "`rocm diagnose") {
        let mut rest = line;
        while let Some(open) = rest.find('`') {
            let after = &rest[open + 1..];
            let Some(close) = after.find('`') else { break };
            let (name, tail) = (&after[..close], &after[close + 1..]);
            rest = tail;
            if !is_field_name(name) {
                continue;
            }
            let Some(open_paren) = tail.trim_start().strip_prefix('(') else {
                continue;
            };
            let Some((value, _)) = open_paren.split_once(')') else {
                continue;
            };
            if let Ok(parsed) = value.trim().parse::<i64>() {
                out.insert(name.to_owned(), parsed);
            }
        }
    }
    assert!(
        !out.is_empty(),
        "no confidence thresholds parsed out of reference.md"
    );
    out
}

/// Verdicts the reference enumerates for `examine --json`'s `status`.
fn documented_verdicts(md: &str) -> BTreeSet<String> {
    let line = section(md, "`rocm examine")
        .into_iter()
        .find(|line| line.contains("`status`"))
        .unwrap_or_else(|| panic!("reference.md no longer enumerates the `status` verdicts"));
    let verdicts: BTreeSet<String> = backticked(line)
        .into_iter()
        .filter(|span| *span != "status")
        .map(str::to_owned)
        .collect();
    assert!(
        !verdicts.is_empty(),
        "no examine verdicts parsed out of reference.md"
    );
    verdicts
}

/// The subset of Framework routing the reference attributes to the **CLI**.
///
/// The section names two different things: trackers the *skill* hands over from
/// its own table (Lemonade, Ollama, LM Studio — the host probe cannot detect
/// them), and the targets `route_when_no_match` actually returns. Checking the
/// CLI's route against the union of both lets a documented-but-dead CLI target
/// pass silently, which is how the removed lemonade / ollama / lm-studio arms
/// stayed in the document. So the CLI is held to its own list.
fn documented_cli_routes(md: &str) -> BTreeMap<String, Option<String>> {
    let mut out = BTreeMap::new();
    let mut inside = false;
    for line in section(md, "Framework routing") {
        let trimmed = line.trim();
        if trimmed.contains("`route_when_no_match`") && trimmed.ends_with(':') {
            inside = true;
            continue;
        }
        if !inside || trimmed.is_empty() {
            continue;
        }
        let Some(bullet) = trimmed.strip_prefix("- ") else {
            break;
        };
        let (Some(label), url) = (route_label(bullet), route_url(bullet)) else {
            continue;
        };
        out.insert(label, url);
    }
    assert!(
        !out.is_empty(),
        "reference.md no longer lists the targets `route_when_no_match` returns; \
         without that list the CLI's route can only be checked against the skill's \
         own hand-off table, which is not the same claim"
    );
    out
}

/// The bolded target name on a routing bullet, or the one after the arrow.
fn route_label(bullet: &str) -> Option<String> {
    bullet
        .split("**")
        .nth(1)
        .map(normalise_route_name)
        .filter(|label| !label.is_empty())
}

/// The tracker URL on a routing bullet, where it gives one.
fn route_url(bullet: &str) -> Option<String> {
    bullet.find("https://").map(|at| {
        bullet[at..]
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_owned()
    })
}

/// Lowercase, alphanumerics only: `LM Studio`, `lm-studio` and `lm.studio` all
/// collapse to the same key.
fn normalise_route_name(name: &str) -> String {
    name.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// The reference text a step loaded, or a panic naming the missing Given.
fn reference(world: &E2eWorld) -> &str {
    world
        .skill_reference
        .as_deref()
        .expect("skill reference not loaded")
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
    world.skill_symptom = Some(KNOWN_SYMPTOM.to_owned());
}

#[given("a user who reports a failure with no catalog keyword in it")]
async fn user_reports_unmatched_failure(world: &mut E2eWorld) {
    world.skill_symptom = Some(UNMATCHED_SYMPTOM.to_owned());
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
    let symptom = world.skill_symptom.clone().expect("no symptom set");
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
         skills/rocm-doctor/reference.md to match the CLI.\n\
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
    // The reference says it twice — `yes` cells above, prose below — and the
    // loop only checked the cells. A rename applied to the table and the CLI
    // together would leave the prose stale, and the prose is what an agent
    // reads before it decides whether to offer to run a fix.
    let prose = documented_auto_prose(reference(world));
    let offered: BTreeSet<String> = cli
        .iter()
        .filter(|(_, r)| r.auto)
        .map(|(id, _)| id.clone())
        .collect();
    assert_eq!(
        prose, offered,
        "reference.md's prose names {prose:?} as auto-applicable, `rocm fix` offers {offered:?}"
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

#[then("the diagnosis carries every field the skill names")]
async fn assert_documented_fields_present(world: &mut E2eWorld) {
    let documented = documented_diagnose_fields(reference(world));
    let per_cause = documented_cause_fields(reference(world));
    let report = diagnosis(world);

    let missing: Vec<&String> = documented
        .iter()
        .filter(|field| report.get(field.as_str()).is_none())
        .collect();
    assert!(
        missing.is_empty(),
        "diagnose --json is missing {missing:?}, which skills/rocm-doctor/reference.md \
         tells an agent to read.\n\
         The CLI is authoritative: if a field was renamed or dropped, the document \
         follows it.\n{report:#}"
    );

    // And the converse. Checking only `documented ⊆ emitted` cannot see a field
    // the CLI emits that the document never mentions — which is how `has_match`
    // stayed undocumented while the skill told agents to gate on `matched` being
    // empty, the one substitute `rocm-core` documents as wrong. An agent reads
    // the document, so a field absent from it does not exist as far as the skill
    // is concerned.
    let emitted = report
        .as_object()
        .expect("diagnose --json is not a JSON object");
    let undocumented: Vec<&String> = emitted
        .keys()
        .filter(|field| !documented.contains(field.as_str()))
        .collect();
    assert!(
        undocumented.is_empty(),
        "diagnose --json emits {undocumented:?}, which skills/rocm-doctor/reference.md \
         never names — so an agent following the skill will not read it.\n\
         Document the field, or stop emitting it."
    );

    let matched = report
        .get("matched")
        .and_then(serde_json::Value::as_array)
        .expect("diagnose JSON has no 'matched' array");
    // Deliberately not asserting `matched` is non-empty: on a host the catalog
    // rules out of scope (WSL2) an empty list is the correct answer, and the
    // routing scenario covers that branch. What must hold is that anything
    // offered is fully readable by an agent following the skill.
    //
    // The individual fields of a `fix` plan are NOT checked here. The reference
    // does not spell them out, so there is no documented claim to compare
    // against — that is a serialization shape, and `crates/rocm-core` owns it.
    for cause in matched {
        let absent: Vec<&String> = per_cause
            .iter()
            .filter(|field| cause.get(field.as_str()).is_none())
            .collect();
        assert!(
            absent.is_empty(),
            "a matched cause is missing {absent:?}, which reference.md documents \
             a cause as carrying:\n{cause:#}"
        );
    }
}

#[then("its confidence thresholds are the ones the skill reasons about")]
async fn assert_thresholds(world: &mut E2eWorld) {
    let documented = documented_thresholds(reference(world));
    let report = diagnosis(world);
    for (field, expected) in &documented {
        let actual = report
            .get(field.as_str())
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_else(|| panic!("diagnose JSON has no numeric '{field}'"));
        assert_eq!(
            actual, *expected,
            "{field} is {actual}, but reference.md states {expected} — and the skill's \
             workflow text reasons about that number when it grades a cause"
        );
    }
}

#[then("the CLI routes the report to a tracker the skill documents")]
async fn assert_route_is_documented(world: &mut E2eWorld) {
    let documented = documented_cli_routes(reference(world));
    let report = diagnosis(world);

    // Deliberately NOT asserting `has_match == false` here, even though the
    // symptom carries no catalog keyword. `diagnose` scores several checkers
    // from host state alone: the GitHub-hosted Linux runner ships
    // /etc/modprobe.d/blacklist-radeon-instinct.conf with amdgpu unloaded, which
    // is `fix-5-amdgpu-load` at score 90 whatever symptom is passed. No symptom
    // can hold that premise still, so asserting it here only encodes the
    // runner's state.
    //
    // The premise is pinned where the host can be held still instead --
    // `diagnose::tests::sub_threshold_causes_leave_has_match_false_and_route_upstream`
    // builds an Examination whose causes are all sub-threshold and asserts both
    // that `has_match` is false and that the route names somewhere to go.
    //
    // What is left here is the half only this feature can check: that whatever
    // target the CLI hands back is one the published document names.
    let route = report
        .get("route_when_no_match")
        .expect("diagnose JSON has no 'route_when_no_match'");
    let target = route
        .get("target")
        .and_then(serde_json::Value::as_str)
        .expect("the route carries no target");
    let url = route
        .get("url")
        .and_then(serde_json::Value::as_str)
        .expect("the route carries no url");

    assert!(
        !url.trim().is_empty(),
        "the skill's rule is to route upstream when nothing matched, so the route \
         must name somewhere to go:\n{route:#}"
    );
    let known = documented
        .get(&normalise_route_name(target))
        .unwrap_or_else(|| {
            panic!(
                "the CLI routes to {target:?}, which reference.md's Framework routing \
                 section does not name. Documented: {:?}",
                documented.keys().collect::<Vec<_>>()
            )
        });
    // Only where the reference actually writes the tracker down. It names
    // PyTorch and llama.cpp as in scope without giving their URLs, so for those
    // there is nothing to compare and the target check above is the whole test.
    if let Some(expected) = known {
        assert!(
            url.starts_with(expected.as_str()),
            "reference.md sends a {target} report to {expected}, the CLI to {url:?}"
        );
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

#[then("its verdict is one the skill documents")]
async fn assert_known_verdict(world: &mut E2eWorld) {
    let documented = documented_verdicts(reference(world));
    let output = world.cli_output.as_ref().expect("no examine output");
    let report: serde_json::Value =
        serde_json::from_str(output).expect("examine --json did not emit valid JSON");
    let status = report
        .get("status")
        .and_then(serde_json::Value::as_str)
        .expect("examine JSON has no 'status'");
    assert!(
        documented.contains(status),
        "examine reported {status:?}, which the skill does not account for. \
         reference.md enumerates {documented:?}, and an agent that meets a verdict \
         outside that list has no instruction for what to do next."
    );
}
