// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};

use crate::E2eWorld;

/// A symptom string that scores a catalog match on both Linux and Windows. It
/// keys off `check_1_arch_not_in_wheel` (a `LINUX_AND_WINDOWS` checker), which
/// scores 50 on the `HSA_STATUS_ERROR_INVALID_ISA` keyword regardless of host
/// state — the covered-arch penalty only applies when a framework arch list is
/// present, so with none installed the match always renders. (The earlier
/// "/dev/kfd" symptom keyed only off the Linux-only render-group checker and so
/// produced no match on Windows.) The specific fix-id is environment-dependent,
/// so scenarios assert the shape of a match, not the id.
const KNOWN_SYMPTOM: &str = "HSA_STATUS_ERROR_INVALID_ISA";

/// A print-only recipe (no runner, applies on linux+windows) whose `--dry-run`
/// is deterministic across environments — used for the preview scenario. Other
/// recipes gate on host state (e.g. `$USER`) and return non-zero even for a
/// dry-run, which would make the assertion host-dependent.
const PREVIEW_FIX_ID: &str = "fix-1-arch";

/// A recipe that would really change the machine, used to prove the CLI asks
/// first. Of the four AUTO recipes this is the only one that reaches the
/// confirmation gate on a host with nothing installed: `fix-2-unset-override`
/// never calls it on Linux, `fix-4-render-group` exits early once the user is
/// already in the groups, and `fix-6-path` exits early with "no ROCm install
/// found". This one needs only `--device-index`, which the scenario supplies.
const MUTATING_FIX_ID: &str = "fix-9-igpu-dgpu";

/// Every fix-id in the closed catalog, in the order `rocm fix` lists them.
///
/// A duplicate of the catalog, on purpose: a test that derived this list from
/// the same source it checks could not notice a change to it.
///
/// The catalog is a closed list that external tooling reads and reproduces —
/// the ids, their OS scope, and which four are auto-applicable are all part of
/// the CLI's published contract, not private detail. Changing the catalog
/// changes that contract, and this is the assertion that says so out loud.
/// When it fires, update this list along with whatever documents the catalog;
/// do not relax it.
const CATALOG_FIX_IDS: &[&str] = &[
    "fix-1-arch",
    "fix-2-unset-override",
    "fix-3-rocm-kernel",
    "fix-4-render-group",
    "fix-5-amdgpu-load",
    "fix-6-path",
    "fix-7-stale-repos",
    "fix-8-wheel-rocm",
    "fix-9-igpu-dgpu",
    "fix-10-container",
    "fix-11-iommu",
    "fix-12-installer",
    "fix-13-hip-sdk-missing",
    "fix-14-adrenalin-too-old",
    "fix-15-msvc-redist",
];

/// The fixes the CLI carries out itself. Every other entry only prints a plan.
/// Pinned exactly: a mode quietly promoted to AUTO would begin changing
/// machines that callers had been told it only ever advised on.
const AUTO_APPLICABLE_FIX_IDS: &[&str] = &[
    "fix-2-unset-override",
    "fix-4-render-group",
    "fix-6-path",
    "fix-9-igpu-dgpu",
];

/// A catalog entry that cannot apply on the host running the suite, whichever
/// host that is. Both are print-only, so the run stops at the OS gate without
/// reaching any recipe that could touch the machine.
const fn fix_id_for_the_other_os() -> &'static str {
    if cfg!(windows) {
        "fix-5-amdgpu-load" // linux-only
    } else {
        "fix-13-hip-sdk-missing" // windows-only
    }
}

/// Contents planted in the scenario's own shell rc file. The assertion is that
/// this survives the run byte for byte.
const PLANTED_RC: &str = "# planted by the e2e suite; the fix must not touch this\n";

/// The home directory handed to the fix under test, inside the scenario's
/// isolated root. `fix-9` appends to a shell rc file under `$HOME`, and the
/// piped harness otherwise lets the CLI inherit the runner's real one — so
/// without this the scenario would read (and a regression could edit) the
/// dotfiles of whoever is running the suite.
fn fix_home(world: &E2eWorld) -> std::path::PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join("fix-home")
}

/// The rc file `fix-9` resolves to under [`fix_home`]. `shell_rc_file` picks
/// `.zshrc` when `$SHELL` names zsh, so the scenario pins `$SHELL` to bash and
/// this stays `.bashrc` regardless of the runner's login shell.
fn fix_rc_file(world: &E2eWorld) -> std::path::PathBuf {
    fix_home(world).join(".bashrc")
}

// ── Given ──────────────────────────────────────────────────────────

#[given("a user who hit a known ROCm failure")]
async fn user_hit_known_failure(world: &mut E2eWorld) {
    world.model_name = Some(KNOWN_SYMPTOM.to_string());
}

#[given("a user who hit a failure the CLI does not recognise")]
async fn user_hit_unknown_failure(world: &mut E2eWorld) {
    world.model_name = Some("xyzzy totally unrelated gibberish".to_string());
}

#[given("a user who has chosen a known fix")]
async fn user_chose_known_fix(world: &mut E2eWorld) {
    world.model_name = Some(PREVIEW_FIX_ID.to_string());
}

#[given("a user who names a fix the CLI does not offer")]
async fn user_named_unknown_fix(world: &mut E2eWorld) {
    world.model_name = Some("fix-does-not-exist".to_string());
}

#[given("a user who has chosen a fix that would change the machine")]
async fn user_chose_mutating_fix(world: &mut E2eWorld) {
    let rc_file = fix_rc_file(world);
    std::fs::create_dir_all(fix_home(world)).expect("failed to create the scenario's home dir");
    std::fs::write(&rc_file, PLANTED_RC).expect("failed to plant the shell rc file");
    world.model_name = Some(MUTATING_FIX_ID.to_string());
}

#[given("a user who has chosen a fix meant for a different operating system")]
async fn user_chose_fix_for_another_os(world: &mut E2eWorld) {
    world.model_name = Some(fix_id_for_the_other_os().to_string());
}

#[given("a user who refers to a cause by its position in the diagnosis")]
async fn user_named_diagnosis_position(world: &mut E2eWorld) {
    // Quoted deliberately: unquoted, the shell treats `#1` as a comment and the
    // CLI never sees it. The product behaviour under test is what happens when
    // the argument does arrive.
    world.model_name = Some("#1".to_string());
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user asks the CLI to diagnose that symptom")]
async fn user_diagnoses(world: &mut E2eWorld) {
    let symptom = world.model_name.clone().expect("no symptom set");
    let (stdout, _, rc) = crate::run_rocm(world, &["diagnose", "--symptom", &symptom]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("the user asks the CLI to diagnose that symptom in machine-readable form")]
async fn user_diagnoses_json(world: &mut E2eWorld) {
    let symptom = world.model_name.clone().expect("no symptom set");
    let (stdout, _, rc) = crate::run_rocm(world, &["diagnose", "--symptom", &symptom, "--json"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("the user asks the CLI which fixes it offers")]
async fn user_lists_fixes(world: &mut E2eWorld) {
    let (stdout, _, rc) = crate::run_rocm(world, &["fix"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("the user previews that fix without applying it")]
async fn user_previews_fix(world: &mut E2eWorld) {
    let fix_id = world.model_name.clone().expect("no fix id set");
    let (stdout, _, rc) = crate::run_rocm(world, &["fix", &fix_id, "--dry-run"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("the user asks the CLI to apply that fix")]
async fn user_applies_fix(world: &mut E2eWorld) {
    let fix_id = world.model_name.clone().expect("no fix id set");
    let (stdout, stderr, rc) = crate::run_rocm(world, &["fix", &fix_id]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user asks the CLI to apply it without agreeing to the change")]
async fn user_applies_fix_without_agreeing(world: &mut E2eWorld) {
    let fix_id = world.model_name.clone().expect("no fix id set");
    let home = fix_home(world).display().to_string();
    // No `--yes`, and the harness pipes stdin, so this is the non-interactive
    // case the gate exists for. `--device-index` is what carries `fix-9` past
    // its own "tell me which GPU" branch and up to the gate.
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &["fix", &fix_id, "--device-index", "1"],
        &[("HOME", home.as_str()), ("SHELL", "/bin/bash")],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the CLI reports a likely cause with a suggested fix")]
async fn assert_reports_cause_and_fix(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "diagnose should exit 0 (it is a query)"
    );
    let output = world.cli_output.as_ref().expect("no diagnose output");
    // A match renders as a scored `#1 [TIER score=NN/100] <title>` header with
    // an `id:` line and a `plan:` line. Assert the shape, not a specific fix-id
    // (the top match is environment-dependent).
    assert!(
        output.contains("score=") && output.contains("id:"),
        "expected a scored match with an id:\n{output}"
    );
    assert!(
        output.contains("plan:"),
        "expected a suggested fix plan:\n{output}"
    );
}

#[then("every reported cause comes with a command that applies it")]
async fn assert_every_cause_has_a_command(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no diagnose output");
    // A plan alone is not actionable: the report used to name a `rocm fix`
    // command only when some match cleared the confidence threshold, so a report
    // of low-confidence causes left the user with nothing to run.
    let causes = output.lines().filter(|l| l.contains("score=")).count();
    let commands = output
        .lines()
        .filter(|l| l.trim().starts_with("apply with: rocm fix "))
        .count();
    assert!(causes > 0, "no scored causes to check:\n{output}");
    assert_eq!(
        commands, causes,
        "each of the {causes} causes needs its own apply command:\n{output}"
    );
}

#[then("the listing explains what those indicators mean")]
async fn assert_markers_explained(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no fix list output");
    // The markers were printed with no legend, so a reader could not tell
    // whether PRINT-ONLY meant "advisory" or "not implemented yet".
    assert!(
        output.contains("AUTO =") && output.contains("PRINT-ONLY ="),
        "expected the listing to explain its AUTO/PRINT-ONLY markers:\n{output}"
    );
}

#[then("the CLI always points to somewhere the problem can be reported")]
async fn assert_offers_escalation(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "diagnose should exit 0 (it is a query)"
    );
    let output = world.cli_output.as_ref().expect("no diagnose output");
    let report: serde_json::Value =
        serde_json::from_str(output).expect("diagnose --json did not emit valid JSON");
    // Whatever the symptom, and whatever the host's own state, the report always
    // carries an upstream escalation route so the user is never left with a dead
    // end. We deliberately do NOT assert anything about match count or
    // confidence: `diagnose` probes the REAL environment, and a black-box CI host
    // may have genuine faults (blacklisted amdgpu, user not in render group) that
    // legitimately score high for any symptom. The route is the invariant.
    let url = report
        .get("route_when_no_match")
        .and_then(|r| r.get("url"))
        .and_then(serde_json::Value::as_str)
        .expect("diagnose JSON has no escalation route url");
    assert!(
        url.starts_with("http"),
        "expected an escalation URL, got: {url:?}"
    );
}

#[then("the result is machine-readable and identifies the matched cause")]
async fn assert_json_identifies_match(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "diagnose should exit 0 (it is a query)"
    );
    let output = world.cli_output.as_ref().expect("no diagnose output");
    let report: serde_json::Value =
        serde_json::from_str(output).expect("diagnose --json did not emit valid JSON");
    // NOT `matched` being non-empty: that list also carries entries scoring too
    // low to act on, so its size does not answer "was a cause established?".
    // `has_match` is the field that does, and it is what a caller must read.
    assert_eq!(
        report.get("has_match").and_then(serde_json::Value::as_bool),
        Some(true),
        "a known symptom must be reported as an established cause:\n{output}"
    );
    let top = report
        .get("matched")
        .and_then(|m| m.as_array())
        .and_then(|m| m.first())
        .expect("a report with a match must name it");
    // A cause the caller cannot act on is not actionable: it needs an id to
    // refer to and a fix-id to hand to `rocm fix`.
    assert!(
        top.get("id").and_then(serde_json::Value::as_str).is_some(),
        "the matched cause must carry an id:\n{output}"
    );
    assert!(
        top.pointer("/fix/fix_id")
            .and_then(serde_json::Value::as_str)
            .is_some(),
        "the matched cause must name the fix that applies it:\n{output}"
    );
}

/// Hold the report to its own arithmetic: `has_match` is true exactly when some
/// cause cleared the threshold the report itself declares. Returns the ids that
/// cleared it.
///
/// This is the assertion that survives on any host. Pinning a specific verdict
/// would make the scenario a test of the runner's health — a CI box with a
/// genuine fault of its own (a blacklisted amdgpu, a user outside the render
/// group) produces real causes whatever symptom was passed. Self-consistency
/// holds regardless, and it is exactly the property that was broken: the
/// verdict used to be unavailable, so callers inferred it from the list length.
fn assert_verdict_follows_scores<'a>(report: &'a serde_json::Value, output: &str) -> Vec<&'a str> {
    let has_match = report
        .get("has_match")
        .and_then(serde_json::Value::as_bool)
        .expect("diagnose JSON must state whether a cause was established");
    // Read the threshold from the document rather than restating 50 here: the
    // report publishes it so callers do not have to hardcode it, and a test
    // that hardcodes it is not exercising that.
    let threshold = report
        .get("min_score_for_match")
        .and_then(serde_json::Value::as_i64)
        .expect("diagnose JSON must publish its match threshold");
    let cleared: Vec<&str> = report
        .get("matched")
        .and_then(|m| m.as_array())
        .expect("diagnose JSON has no 'matched' array")
        .iter()
        .filter(|d| {
            d.get("score")
                .and_then(serde_json::Value::as_i64)
                .is_some_and(|s| s >= threshold)
        })
        .filter_map(|d| d.get("id").and_then(serde_json::Value::as_str))
        .collect();
    assert_eq!(
        has_match,
        !cleared.is_empty(),
        "the verdict must follow the scores (threshold {threshold}); \
         cleared={cleared:?}\n{output}"
    );
    cleared
}

fn parsed_diagnosis(world: &E2eWorld) -> (serde_json::Value, String) {
    let output = world
        .cli_output
        .as_ref()
        .expect("no diagnose output")
        .clone();
    let report = serde_json::from_str(&output).expect("diagnose --json did not emit valid JSON");
    (report, output)
}

#[then("the result states that no cause was established")]
async fn assert_json_states_no_match(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "diagnose should exit 0 (it is a query)"
    );
    let (report, output) = parsed_diagnosis(world);
    let cleared = assert_verdict_follows_scores(&report, &output);
    if !cleared.is_empty() {
        // This host has a real fault of its own, so the premise is gone. The
        // consistency check above still ran, which is what this scenario is for.
        return;
    }
    assert!(
        !report
            .get("has_match")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        "an unrecognised symptom on a host with no real fault must not report \
         an established cause:\n{output}"
    );
}

#[then("the result says whether this platform is covered")]
async fn assert_json_states_platform_scope(world: &mut E2eWorld) {
    let (report, output) = parsed_diagnosis(world);
    // NOT "the key is present": `out_of_scope` is an Option with no
    // skip_serializing_if, so serde emits it either way and its mere presence
    // proves nothing. Cross-check the verdict against the one the host report
    // gives for the same machine — the same trick `examine-both-forms-agree-on-gpu`
    // uses, and the only version of this assertion that can fail on a covered
    // host. The two are computed by different code paths off the same probe, so
    // this is a cross-check rather than a tautology.
    let (examine, _, rc) = crate::run_rocm(world, &["examine", "--json"]);
    assert_eq!(rc, 0, "examine should exit 0 (it is an inspector)");
    let host: serde_json::Value =
        serde_json::from_str(&examine).expect("examine --json did not emit valid JSON");
    let host_says_uncovered = host
        .get("status")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|status| status == "wsl");
    let diagnosis_says_uncovered = report.get("out_of_scope").is_some_and(|v| !v.is_null());
    assert_eq!(
        diagnosis_says_uncovered, host_says_uncovered,
        "the diagnosis and the host report disagree about whether this platform \
         is covered (diagnosis={diagnosis_says_uncovered}, host={host_says_uncovered})\
         \n{output}\n{examine}"
    );
}

#[then("a platform that is not covered is given no diagnosis")]
async fn assert_uncovered_platform_gets_no_diagnosis(world: &mut E2eWorld) {
    let (report, output) = parsed_diagnosis(world);
    let Some(reason) = report
        .get("out_of_scope")
        .and_then(serde_json::Value::as_str)
    else {
        return; // covered platform — held to its own half by the next step
    };
    assert!(
        !reason.trim().is_empty(),
        "an out-of-scope verdict must say why:\n{output}"
    );
    // The whole point of routing out is to avoid emitting bare-metal findings
    // that cannot apply. A verdict with findings attached would be worse than
    // no verdict: the caller would act on them.
    let matched = report
        .get("matched")
        .and_then(|m| m.as_array())
        .expect("diagnose JSON has no 'matched' array");
    assert!(
        matched.is_empty(),
        "an out-of-scope platform must be given no findings:\n{output}"
    );
    assert_eq!(
        report.get("has_match").and_then(serde_json::Value::as_bool),
        Some(false),
        "an out-of-scope platform cannot have an established cause:\n{output}"
    );
}

#[then("a platform that is covered gets a verdict that follows the evidence")]
async fn assert_covered_platform_verdict_is_consistent(world: &mut E2eWorld) {
    let (report, output) = parsed_diagnosis(world);
    if report.get("out_of_scope").is_some_and(|v| !v.is_null()) {
        return; // uncovered platform — held to its own half by the previous step
    }
    // The covered branch used to return without asserting anything, which left
    // this scenario proving nothing at all on every lane CI actually runs (there
    // is no WSL2 runner). This is the half that holds everywhere.
    assert_verdict_follows_scores(&report, &output);
}

#[then("the CLI declines because the fix does not apply to this machine")]
async fn assert_inapplicable_fix_declined(world: &mut E2eWorld) {
    // 3 is its own outcome: not a usage error (2), not a failed attempt (4),
    // not a refusal by the user (5). A caller that cannot tell them apart
    // reports a broken machine when the truth is "wrong operating system".
    assert_eq!(
        world.cli_rc,
        Some(3),
        "a fix that does not apply here should exit 3, distinct from 2/4/5"
    );
    let output = world.cli_output.as_ref().expect("no fix output");
    assert!(
        output.contains("This fix only applies on:"),
        "the refusal must say which platforms the fix is for:\n{output}"
    );
}

#[then("every fix the catalog documents is listed")]
async fn assert_catalog_complete(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no fix list output");
    let missing: Vec<_> = CATALOG_FIX_IDS
        .iter()
        .filter(|id| !output.contains(**id))
        .collect();
    assert!(
        missing.is_empty(),
        "the listing is missing {missing:?}. If the catalog gained or lost a \
         failure mode, update CATALOG_FIX_IDS here and whatever else documents \
         the catalog — do not loosen this assertion:\n{output}"
    );
}

#[then("only the fixes the CLI can carry out itself are marked as such")]
async fn assert_auto_set_is_exact(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no fix list output");
    let marked_auto: Vec<&str> = output
        .lines()
        .filter(|line| line.contains("[      AUTO]"))
        .filter_map(|line| CATALOG_FIX_IDS.iter().copied().find(|id| line.contains(id)))
        .collect();
    // Exact, not "at least": a mode quietly promoted to AUTO would start
    // changing machines that callers were told it only ever advised.
    assert_eq!(
        marked_auto, AUTO_APPLICABLE_FIX_IDS,
        "the set of fixes the CLI applies itself has changed:\n{output}"
    );
}

#[then("the CLI lists the fixes it can apply")]
async fn assert_lists_fixes(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "fix listing should exit 0");
    let output = world.cli_output.as_ref().expect("no fix list output");
    assert!(
        output.contains("Available fix-ids"),
        "expected the fix-id listing header:\n{output}"
    );
    assert!(
        output.contains("fix-"),
        "expected at least one fix-id row:\n{output}"
    );
}

#[then("each fix indicates whether the CLI can apply it automatically")]
async fn assert_fix_auto_flag(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no fix list output");
    // Every row is tagged AUTO (the CLI can run it) or PRINT-ONLY (advisory).
    assert!(
        output.contains("AUTO") || output.contains("PRINT-ONLY"),
        "expected AUTO/PRINT-ONLY applicability markers:\n{output}"
    );
}

#[then("the CLI describes what the fix would change")]
async fn assert_describes_change(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "a dry-run of a print-only fix should exit 0"
    );
    let output = world.cli_output.as_ref().expect("no fix preview output");
    assert!(
        output.contains("Fix:") && output.contains(PREVIEW_FIX_ID),
        "expected a plan describing {PREVIEW_FIX_ID}:\n{output}"
    );
}

#[then("nothing on the machine is changed")]
async fn assert_no_mutation(world: &mut E2eWorld) {
    // A dry-run must not write MANAGED STATE. It may still create incidental
    // dirs (e.g. `data/logs/` from logging init), which are not a mutation of
    // anything the user cares about — so assert on the managed-state artifacts
    // specifically: installed runtimes, registered services, and saved config.
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
            "dry-run wrote managed state at {}",
            managed.display()
        );
    }
}

#[then("the CLI refuses and explains that the fix is not recognised")]
async fn assert_unknown_fix_refused(world: &mut E2eWorld) {
    // Unknown fix-id is a usage error, not a query: it must exit non-zero (2).
    assert_eq!(
        world.cli_rc,
        Some(2),
        "unknown fix-id should exit 2 (unknown id)"
    );
    let combined = format!(
        "{}{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    assert!(
        combined.contains("Unknown fix-id"),
        "expected an 'Unknown fix-id' message:\n{combined}"
    );
}

#[then("the CLI refuses and explains that a position is not a fix-id")]
async fn assert_position_argument_corrected(world: &mut E2eWorld) {
    // Same exit code as any unknown id — this is clearer wording on an existing
    // refusal, not a new outcome a script could come to depend on.
    assert_eq!(
        world.cli_rc,
        Some(2),
        "a position argument should exit 2 like any unknown id"
    );
    let combined = format!(
        "{}{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    assert!(
        combined.contains("position"),
        "the refusal must say the argument was read as a position:\n{combined}"
    );
    // And it must point at what to use instead, or the correction is useless.
    assert!(
        combined.contains("id:"),
        "the refusal must name the identifier to use instead:\n{combined}"
    );
}

#[then("the CLI refuses and explains that it needs agreement")]
async fn assert_refuses_without_agreement(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no fix output");
    // The refusal has to say *why* and how to proceed. A bare non-zero exit
    // reads as a broken fix rather than a deliberate stop.
    assert!(
        output.contains("--yes"),
        "the refusal must name what to pass to proceed:\n{output}"
    );
    assert!(
        output.contains("refusing to apply"),
        "the refusal must say it did not apply the fix:\n{output}"
    );
    // Distinct from the unknown-id refusal (2), so a script can tell "you did
    // not agree" apart from "no such fix".
    assert_eq!(
        world.cli_rc,
        Some(5),
        "declining to apply is its own outcome, not an error:\n{output}"
    );
}

// ── Diagnosis and fix as two views of one remedy ───────────────────

/// The remedy both commands describe for a device-permission problem: the CLI
/// can apply it, and both `diagnose` and `fix` have their own copy of what it
/// does — which is exactly why the two can drift apart.
const DEVICE_PERMISSION_FIX_ID: &str = "fix-4-render-group";

/// A symptom that makes the device-permission cause score on any host, so the
/// comparison never depends on what happens to be wrong with the runner.
const DEVICE_PERMISSION_SYMPTOM: &str = "unable to open /dev/kfd";

/// The device whose owning group the remedy is about.
const GPU_DEVICE: &str = "/dev/kfd";

/// The field the machine-readable inspection describes [`GPU_DEVICE`] under.
const GPU_DEVICE_FIELD: &str = "kfd";

/// The group the fix recipe is hard-coded to add the user to.
const DEFAULT_DEVICE_GROUP: &str = "render";

/// The line a diagnosis prints for how to check the remedy worked.
const DIAGNOSE_VERIFY_PREFIX: &str = "verify after fix:";

/// The line a fix preview prints for the same thing.
const FIX_VERIFY_PREFIX: &str = "Verify:";

/// The value after `label` on the first line that carries it.
fn labelled(text: &str, label: &str) -> Option<String> {
    text.lines()
        .filter_map(|line| line.trim().strip_prefix(label))
        .map(|value| value.trim().to_owned())
        .next()
}

/// The part of a diagnosis that belongs to one cause.
///
/// A diagnosis reports several causes, each with its own plan and its own way to
/// verify it, and which one comes first depends on the machine. Narrowing to the
/// named cause first is what keeps the comparison about the remedy under test
/// rather than about whichever cause happened to rank highest here.
fn section_for(report: &str, fix_id: &str) -> Option<String> {
    let start = report.lines().position(|line| line.contains(fix_id))?;
    // Causes are separated by a blank line, so the next one ends this section.
    let body: Vec<&str> = report
        .lines()
        .skip(start)
        .take_while(|line| !line.trim().is_empty())
        .collect();
    Some(body.join("\n"))
}

/// The `usermod` invocation a report proposes, wherever it prints it. Both
/// commands render it differently — one as a shell line under the plan, the
/// other as the command it is about to run — so match on the invocation itself.
fn proposed_group_command(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| line.contains("usermod") && line.contains("-G"))
        .map(|line| {
            // Strip whichever prefix the renderer used ("$ ", "Will run: ") so the
            // two are compared as the same kind of thing.
            line.trim_start_matches("$ ")
                .trim_start_matches("Will run: ")
                .to_owned()
        })
}

/// The group names in a `-G a,b` argument of a proposed command.
fn groups_in_command(command: &str) -> Vec<String> {
    command
        .split_whitespace()
        .skip_while(|token| *token != "-G")
        .nth(1)
        .unwrap_or_default()
        .trim_matches('"')
        .split(',')
        .filter(|name| !name.trim().is_empty())
        .map(|name| name.trim().to_owned())
        .collect()
}

/// Run the diagnosis and the matching fix preview, and stash both.
fn compare_diagnosis_with_fix(world: &mut E2eWorld, path_override: Option<&str>) {
    let args = ["diagnose", "--symptom", DEVICE_PERMISSION_SYMPTOM];
    let (diagnosis, _, _) = match path_override {
        Some(path) => crate::run_rocm_with_env(world, &args, &[("PATH", path)]),
        None => crate::run_rocm(world, &args),
    };
    // No `--yes`, so nothing is applied; the preview is the user-facing account
    // of what the fix would do.
    let (preview, preview_err, _) =
        crate::run_rocm(world, &["fix", DEVICE_PERMISSION_FIX_ID, "--dry-run"]);
    world.cli_output = Some(diagnosis);
    world.cli_stderr = Some(format!("{preview}{preview_err}"));
}

#[given("a user who hit a device-permission failure")]
async fn user_hit_device_permission_failure(world: &mut E2eWorld) {
    world.model_name = Some(DEVICE_PERMISSION_SYMPTOM.to_string());
}

#[when("the user compares the diagnosis with the matching fix preview")]
async fn user_compares_diagnosis_with_fix(world: &mut E2eWorld) {
    let path_override = world.path_override.clone();
    compare_diagnosis_with_fix(world, path_override.as_deref());
}

#[then("both give the same way to verify that the fix worked")]
async fn assert_verification_agrees(world: &mut E2eWorld) {
    let diagnosis = world.cli_output.as_ref().expect("no diagnosis output");
    let preview = world.cli_stderr.as_ref().expect("no fix preview output");
    let cause = section_for(diagnosis, DEVICE_PERMISSION_FIX_ID).unwrap_or_else(|| {
        panic!(
            "the symptom did not produce the device-permission cause, so there is nothing to \
             compare:\n{diagnosis}"
        )
    });
    let from_diagnosis = labelled(&cause, DIAGNOSE_VERIFY_PREFIX)
        .unwrap_or_else(|| panic!("the diagnosis gave no way to verify the fix:\n{cause}"));
    let from_preview = labelled(preview, FIX_VERIFY_PREFIX)
        .unwrap_or_else(|| panic!("the fix preview gave no way to verify itself:\n{preview}"));
    // A user told two different things to run cannot know which one settles it.
    assert_eq!(
        from_diagnosis, from_preview,
        "the diagnosis and the fix preview give different ways to verify the same remedy"
    );
}

#[then("both give the same command for applying the remedy")]
async fn assert_remedy_command_agrees(world: &mut E2eWorld) {
    let diagnosis = world.cli_output.as_ref().expect("no diagnosis output");
    let preview = world.cli_stderr.as_ref().expect("no fix preview output");
    let cause = section_for(diagnosis, DEVICE_PERMISSION_FIX_ID).unwrap_or_else(|| {
        panic!(
            "the symptom did not produce the device-permission cause, so there is nothing to \
             compare:\n{diagnosis}"
        )
    });
    let from_diagnosis = proposed_group_command(&cause)
        .unwrap_or_else(|| panic!("the diagnosis proposed no group command:\n{cause}"));
    let from_preview = proposed_group_command(preview)
        .unwrap_or_else(|| panic!("the fix preview proposed no group command:\n{preview}"));
    // Compare the groups rather than the whole line: the two renderings name the
    // user differently (one leaves `$USER` for the shell, the other resolves it),
    // and that difference is presentation, not disagreement about the remedy.
    assert_eq!(
        groups_in_command(&from_diagnosis),
        groups_in_command(&from_preview),
        "the diagnosis and the fix preview would add the user to different groups:\n  \
         diagnosis: {from_diagnosis}\n  preview:   {from_preview}"
    );
}

// ── Device group states induced for the machine under test ─────────

#[given("the GPU device belongs to a group the machine cannot name")]
async fn given_unnameable_device_group(world: &mut E2eWorld) {
    // What `stat` itself prints for an owning id with no entry in the group
    // database — the state a container inherits when the device's group exists
    // on the host but not inside.
    world.path_override = Some(crate::stat_shim_path(
        world,
        GPU_DEVICE,
        "crw-rw----",
        "root",
        "UNKNOWN",
    ));
}

#[given("the GPU device belongs to a recognised non-default group")]
async fn given_non_default_device_group(world: &mut E2eWorld) {
    // Any real group on this machine other than the one the fix recipe assumes.
    // `video` is excluded too: the remedy adds it alongside, so using it would
    // make the two sides agree by coincidence.
    let group = crate::machine_group_names()
        .into_iter()
        .find(|name| name != DEFAULT_DEVICE_GROUP && name != "video")
        .expect("this machine names no group other than the default");
    world.path_override = Some(crate::stat_shim_path(
        world,
        GPU_DEVICE,
        "crw-rw----",
        "root",
        &group,
    ));
}

#[given("the user can already read and write the GPU device")]
async fn given_usable_device(world: &mut E2eWorld) {
    // Two halves of one situation: the device is open to everyone, AND the user
    // is in none of the groups conventionally used to reach it. Both are
    // substituted rather than hoped for — the GPU runner's user IS in those
    // groups, which makes the finding score zero there for a reason that has
    // nothing to do with the access the scenario is about (measured: it passed
    // on MI300X until the group answer was substituted too).
    crate::stat_shim_path(world, GPU_DEVICE, "crw-rw-rw-", "root", "root");
    let path = crate::id_shim_path(world, &["users"]);
    // Prove the premise through the CLI's own eyes before relying on it: if the
    // inspection does not agree the device is usable, the scenario would be
    // measuring something else.
    let (report, _, _) =
        crate::run_rocm_with_env(world, &["examine", "--json"], &[("PATH", path.as_str())]);
    let json: serde_json::Value = serde_json::from_str(&report)
        .unwrap_or_else(|e| panic!("`examine --json` did not emit valid JSON ({e}):\n{report}"));
    // Read the GPU device the scenario shimmed, by name. The report describes
    // several devices — the single `kfd` and each entry of `render_devices` —
    // and they all carry these same field names, so a search for the first one
    // anywhere in the document would be answered by whichever device happened to
    // come first rather than by the one under test.
    let device = json.get(GPU_DEVICE_FIELD).unwrap_or_else(|| {
        panic!("`examine --json` describes no `{GPU_DEVICE_FIELD}` device:\n{report}")
    });
    for field in ["user_can_read", "user_can_write"] {
        let stated = device.get(field).and_then(serde_json::Value::as_bool);
        assert_eq!(
            stated,
            Some(true),
            "the inspection does not report {GPU_DEVICE} as {field}, so the premise this scenario \
             needs is absent:\n{report}"
        );
    }
    world.path_override = Some(path);
}

#[when("the user asks the CLI to diagnose a device-permission failure")]
async fn user_diagnoses_device_permission(world: &mut E2eWorld) {
    let path = world
        .path_override
        .clone()
        .expect("no shimmed PATH prepared");
    let (stdout, _, rc) = crate::run_rocm_with_env(
        world,
        &["diagnose", "--symptom", DEVICE_PERMISSION_SYMPTOM],
        &[("PATH", path.as_str())],
    );
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("the user asks the CLI to diagnose the machine")]
async fn user_diagnoses_machine(world: &mut E2eWorld) {
    let path = world
        .path_override
        .clone()
        .expect("no shimmed PATH prepared");
    // No symptom: the question is what this machine looks like, not what a
    // particular error message suggests.
    let (stdout, _, rc) =
        crate::run_rocm_with_env(world, &["diagnose", "--json"], &[("PATH", path.as_str())]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[then("every group named in the remedy exists on the machine")]
async fn assert_named_groups_exist(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no diagnose output");
    let cause = section_for(output, DEVICE_PERMISSION_FIX_ID).unwrap_or_else(|| {
        panic!("the symptom did not produce the device-permission cause:\n{output}")
    });
    let command = proposed_group_command(&cause)
        .unwrap_or_else(|| panic!("the diagnosis proposed no group command:\n{cause}"));
    let known = crate::machine_group_names();
    let unknown: Vec<String> = groups_in_command(&command)
        .into_iter()
        .filter(|group| !known.iter().any(|name| name == group))
        .collect();
    // Handing the user a command naming a group that does not exist gives them
    // something that cannot work and no way to see why.
    assert!(
        unknown.is_empty(),
        "the remedy tells the user to join {unknown:?}, which this machine does not \
         recognise:\n  {command}"
    );
}

#[then("adding the user to a device group is not the leading remedy")]
async fn assert_no_device_group_remedy(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no diagnose output");
    let report: serde_json::Value =
        serde_json::from_str(output).expect("diagnose --json did not emit valid JSON");
    let matched = report
        .get("matched")
        .and_then(|m| m.as_array())
        .expect("diagnose JSON has no 'matched' array");
    let offered = matched.iter().find(|entry| {
        entry.get("id").and_then(serde_json::Value::as_str) == Some(DEVICE_PERMISSION_FIX_ID)
    });
    let score = offered
        .and_then(|entry| entry.get("score"))
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    // Belonging to a conventional group is one way to reach the device, not the
    // point of doing so. With access already demonstrated, repairing permissions
    // is not a cause of anything, and offering it sends the user to change their
    // group membership for no reason.
    assert_eq!(
        score, 0,
        "the device is readable and writable, yet the diagnosis still offers \
         '{DEVICE_PERMISSION_FIX_ID}' as a cause (score {score})"
    );
}

#[then("the file the fix would have changed is untouched")]
async fn assert_rc_file_untouched(world: &mut E2eWorld) {
    let rc_file = fix_rc_file(world);
    let output = world.cli_output.as_ref().expect("no fix output");
    // The fix names its target before asking. If it ever stops naming this file
    // the scenario would be reading back a file the CLI never intended to edit,
    // and would pass without proving anything.
    assert!(
        output.contains(&rc_file.display().to_string()),
        "the fix must name {} as what it would change:\n{output}",
        rc_file.display()
    );
    let after = std::fs::read_to_string(&rc_file).expect("the planted rc file should still exist");
    assert_eq!(
        after,
        PLANTED_RC,
        "declining the fix must leave {} byte-for-byte unchanged",
        rc_file.display()
    );
}
