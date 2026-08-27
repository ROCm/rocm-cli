// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};

use crate::E2eWorld;

fn field_value<'a>(output: &'a str, field: &str) -> Option<&'a str> {
    output.lines().find_map(|line| {
        let (name, value) = line.trim().split_once(':')?;
        (name == field).then(|| value.trim())
    })
}

#[given("a machine with an AMD GPU")]
async fn setup_gpu_machine(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["examine"]);
    assert!(
        stdout.contains("AMD GPU detected") || stdout.contains("detected_gfx_target"),
        "no AMD GPU detected on this machine:\n{stdout}"
    );
}

#[given("a machine with a ROCm install that was not set up by the CLI")]
async fn setup_unmanaged_rocm(world: &mut E2eWorld) {
    world.plant_unmanaged_rocm();
}

#[given("the CLI is running in WSL")]
async fn setup_wsl_host(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["examine"]);
    assert!(
        field_value(&stdout, "wsl").is_some_and(|value| value.eq_ignore_ascii_case("true")),
        "CLI did not detect WSL:\n{stdout}"
    );
}

#[when("the user asks for the version")]
async fn user_asks_version(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["version"]);
    world.cli_output = Some(stdout);
}

#[when("the user lists available engines")]
async fn user_lists_engines(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["engines", "list"]);
    world.cli_output = Some(stdout);
}

#[when("the user inspects the system")]
async fn user_inspects_system(world: &mut E2eWorld) {
    // The exit code is recorded because `examine`'s contract is partly about it:
    // it reports whether the inspection ran, not whether it liked what it found.
    let (stdout, _, rc) = crate::run_rocm(world, &["examine"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("the user asks for help")]
async fn user_asks_help(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["help"]);
    world.cli_output = Some(stdout);
}

#[when("the user previews the driver install plan")]
async fn user_previews_driver_install_plan(world: &mut E2eWorld) {
    // `--dry-run` renders the plan and returns before touching the system, so
    // this is safe to run on any Linux host including the no-GPU mock lane.
    let (stdout, _, rc) = crate::run_rocm(world, &["install", "driver", "--dry-run"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[then("a version string is returned")]
async fn assert_version_returned(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.trim().starts_with("rocm "),
        "expected version string starting with 'rocm ': {output}"
    );
}

#[then("the plan's repo version is a concrete version, not a shell placeholder")]
async fn assert_driver_plan_repo_version_resolved(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    let repo_version = field_value(output, "repo_version")
        .unwrap_or_else(|| panic!("no repo_version line in driver install plan:\n{output}"));
    assert!(
        !repo_version.contains("${"),
        "repo_version still shows an unresolved shell placeholder: {repo_version:?}\n{output}"
    );
    assert!(
        repo_version
            .chars()
            .next()
            .is_some_and(|first| first.is_ascii_digit()),
        "repo_version is not a concrete version string: {repo_version:?}\n{output}"
    );
}

#[then("the subcommands are listed in alphabetical order")]
async fn assert_subcommands_alphabetical(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no help output");
    // Parse the leading token of each line in the `Commands:` block (the
    // subcommand name), stopping at the blank line before `Options:`. Exclude
    // the clap-appended `help` subcommand, which is conventionally listed last.
    let mut names: Vec<String> = Vec::new();
    let mut in_commands = false;
    for line in output.lines() {
        if line.trim_start().starts_with("Commands:") {
            in_commands = true;
            continue;
        }
        if in_commands {
            if line.trim().is_empty() {
                break;
            }
            if let Some(name) = line.split_whitespace().next()
                && name != "help"
            {
                names.push(name.to_string());
            }
        }
    }
    assert!(
        names.len() > 1,
        "could not parse subcommands from help output:\n{output}"
    );
    let mut sorted = names.clone();
    sorted.sort();
    assert!(
        names == sorted,
        "subcommands are not in alphabetical order.\nactual: {names:?}\nsorted: {sorted:?}"
    );
}

#[then("the inspection names the engine this host serves on by default")]
async fn assert_host_default_engine_reported(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    let Some(reported) = output
        .lines()
        .find_map(|line| line.trim().strip_prefix("default_engine:"))
        .map(str::trim)
    else {
        panic!("no default_engine line in examine output:\n{output}");
    };

    // Independently derived by the harness from the GPU family + OS (see
    // `capability::effective_serve_engine`), NOT read back out of `examine` — so
    // a product that reports a constant fails here rather than agreeing with
    // itself.
    let expected = &e2e_cucumber::capability::host_capability().effective_serve_engine;
    assert_eq!(
        reported, expected,
        "examine reports '{reported}' as the default engine, but this host serves on \
         '{expected}':\n{output}"
    );

    // The same value must appear in the engine inventory block, which is what the
    // `*` primary marker follows — the two used to be able to disagree.
    assert!(
        output
            .lines()
            .any(|line| line.trim() == format!("effective_default_engine: {expected}")),
        "engine_inventory did not report '{expected}' as the effective default:\n{output}"
    );
}

#[then("all supported engines are listed")]
async fn assert_all_engines_listed(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    for engine in ["lemonade", "vllm"] {
        assert!(
            output.contains(engine),
            "engine '{engine}' not found in:\n{output}"
        );
    }
}

#[then("the inspection reports Linux as the operating system")]
async fn assert_linux_host(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert_eq!(
        field_value(output, "os"),
        Some("linux"),
        "expected Linux in examine output:\n{output}"
    );
}

#[then("the inspection reports that the host is WSL")]
async fn assert_wsl_host(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert_eq!(
        field_value(output, "wsl"),
        Some("true"),
        "expected WSL in examine output:\n{output}"
    );
}

#[then("the inspection reports which GPU is installed")]
async fn assert_gpu_detected(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("detected_gfx_target:"),
        "no GPU target in examine output:\n{output}"
    );
    let gfx = output
        .lines()
        .find(|l| l.contains("detected_gfx_target:"))
        .and_then(|l| l.split(':').nth(1))
        .map_or("", str::trim);
    assert!(
        gfx.starts_with("gfx"),
        "GPU target does not start with 'gfx': {gfx}"
    );
}

#[then("the inspection reports that the driver is available")]
async fn assert_driver_available(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("amdgpu") || output.contains("driver_status"),
        "driver status not found in examine output:\n{output}"
    );
}

#[then("the inspection reports the install as pre-existing")]
async fn assert_rocm_unmanaged(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("detected_unmanaged") || output.contains("legacy"),
        "expected unmanaged ROCm status:\n{output}"
    );
}

#[then("the inspection names that install's version")]
async fn assert_reports_legacy_version(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    // `plant_unmanaged_rocm` writes `.info/version` containing this. The resolver
    // establishes the version already; the report used to name a path but never
    // a version, so a machine with ROCm installed could not tell you which.
    assert!(
        output.contains("legacy_rocm_version: 6.0.0"),
        "the pre-existing install's version must be reported:\n{output}"
    );
}

#[then("the inspection does not claim nothing is installed")]
async fn assert_does_not_claim_empty(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    // The summary line counted only CLI-managed runtimes, so a machine with ROCm
    // already installed was greeted with a bare "No ROCm installs saved yet".
    let claims_empty = output
        .lines()
        .any(|line| line.trim() == "No ROCm installs saved yet");
    assert!(
        !claims_empty,
        "an install was detected, so the summary must not say there is none:\n{output}"
    );
}

#[then("the inspection suggests setting up a CLI-managed install")]
async fn assert_suggests_managed_runtime(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("rocm install sdk"),
        "expected guidance to install sdk:\n{output}"
    );
}

// ── Machine-readable inspection ────────────────────────────────────

/// The verdict field `examine --json` carries, and the one the harness's own
/// capability probe and the ROCm Doctor skill both read. Asserting the field is
/// present and non-empty — rather than pinning a value — keeps this a contract
/// test: the set of verdicts is host-dependent and grows over time.
const VERDICT_FIELD: &str = "status";

fn parsed_json(world: &E2eWorld) -> serde_json::Value {
    let output = world.cli_output.as_ref().expect("no command was run");
    serde_json::from_str(output)
        .unwrap_or_else(|e| panic!("`examine --json` did not emit valid JSON ({e}):\n{output}"))
}

#[when("the user inspects the system both for reading and for scripting")]
async fn user_inspects_both_ways(world: &mut E2eWorld) {
    let (human, _, _) = crate::run_rocm(world, &["examine"]);
    let (json, _, rc) = crate::run_rocm(world, &["examine", "--json"]);
    // Both are needed by the comparison step; the human form goes in the stderr
    // slot rather than adding a World field for one scenario.
    world.cli_stderr = Some(human);
    world.cli_output = Some(json);
    world.cli_rc = Some(rc);
}

#[when("the user inspects the system without probing frameworks")]
async fn user_inspects_skipping_frameworks(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["examine", "--framework", "skip", "--json"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Facts the human report states that a tool has at least as much right to.
/// Each entry is the label the text form prints, paired with the field names the
/// machine-readable form could reasonably carry it under — it is free to name
/// things its own way, so the assertion is that *some* field carries the fact,
/// not that the two schemas match key for key.
///
/// Deliberately not the full list of eleven: these are the ones a caller cannot
/// work around. Which GPU was found, which engine this host will serve on,
/// whether an existing ROCm install was detected, and what is provisioned.
const FACTS_A_TOOL_ALSO_NEEDS: &[(&str, &[&str])] = &[
    (
        "detected_gfx_target",
        &["detected_gfx_target", "gfx_target"],
    ),
    ("effective_default_engine", &["effective_default_engine"]),
    ("legacy_rocm_status", &["legacy_rocm_status", "legacy_rocm"]),
    (
        "managed_runtimes",
        &["managed_runtimes", "managed_runtime_count"],
    ),
    ("config_dir", &["config_dir"]),
];

/// Every field name appearing anywhere in the document, at any depth.
///
/// The assertion is that the fact is *reachable*, not that it sits at the root:
/// where the machine-readable form chooses to put something is its business, and
/// pinning a path here would turn a presentation choice into a test failure.
fn field_names(value: &serde_json::Value, into: &mut std::collections::HashSet<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                into.insert(key.clone());
                field_names(child, into);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                field_names(item, into);
            }
        }
        _ => {}
    }
}

/// Whether the human report printed a `<label>:` line with a real value.
/// `<unknown>`, `<none>` and `<unset>` are the text form's own placeholders for
/// "nothing to say", and a fact it does not state cannot be one it withholds.
fn human_states(human: &str, label: &str) -> Option<String> {
    human
        .lines()
        .filter_map(|line| line.trim().strip_prefix(&format!("{label}:")))
        .map(str::trim)
        .find(|value| !value.is_empty() && !value.starts_with('<'))
        .map(str::to_owned)
}

#[then("the machine-readable form states everything the readable one does")]
async fn assert_json_states_what_human_does(world: &mut E2eWorld) {
    let human = world
        .cli_stderr
        .as_ref()
        .expect("the human report was not captured");
    let value = parsed_json(world);
    let mut present = std::collections::HashSet::new();
    field_names(&value, &mut present);
    let mut withheld = Vec::new();
    for (label, json_fields) in FACTS_A_TOOL_ALSO_NEEDS {
        let Some(stated) = human_states(human, label) else {
            continue;
        };
        if !json_fields.iter().any(|field| present.contains(*field)) {
            withheld.push(format!("  {label} (the human report says {stated:?})"));
        }
    }
    assert!(
        withheld.is_empty(),
        "the machine-readable form withholds what the readable one states:\n{}\n\n\
         A caller reading `--json` cannot learn these without scraping text.",
        withheld.join("\n")
    );
}

#[then("both reports agree on whether this machine has an AMD GPU")]
async fn assert_forms_agree_on_gpu(world: &mut E2eWorld) {
    let human = world
        .cli_stderr
        .as_ref()
        .expect("the human report was not captured");
    let json = parsed_json(world);
    // The human form names the target it found; the machine-readable form
    // carries a boolean. Two renderings of one question — and on a real MI300X
    // they have been observed to answer it differently.
    let human_found_gpu =
        human_states(human, "detected_gfx_target").is_some_and(|t| t.starts_with("gfx"));
    let json_found_gpu = json
        .get("has_amd_gpu")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    assert_eq!(
        json_found_gpu, human_found_gpu,
        "the two forms disagree about whether this machine has an AMD GPU \
         (json has_amd_gpu={json_found_gpu}, human detected a gfx target={human_found_gpu})"
    );
}

#[then("both reports agree on whether this platform is in scope")]
async fn assert_forms_agree_on_platform(world: &mut E2eWorld) {
    let human = world
        .cli_stderr
        .as_ref()
        .expect("the human report was not captured");
    let json = parsed_json(world);
    // The two forms read *different* WSL predicates: the human report goes
    // through the install/driver summary, `--json` through the probe's own. They
    // are supposed to agree, and the harness's capability probe assumes they do
    // — it decides `is_wsl` for the whole expectation matrix by reading the text
    // form. A disagreement would silently resolve every is_wsl-keyed expectation
    // against the wrong host, which is why this is worth pinning.
    let json_says_wsl = json
        .get(VERDICT_FIELD)
        .and_then(serde_json::Value::as_str)
        .is_some_and(|status| status == "wsl");
    let human_says_wsl = human
        .lines()
        .filter_map(|line| line.trim().strip_prefix("wsl:"))
        .any(|value| matches!(value.trim(), "true" | "yes" | "1"));
    assert_eq!(
        json_says_wsl, human_says_wsl,
        "the two forms disagree about WSL (json={json_says_wsl}, human={human_says_wsl});\
         \nhuman report:\n{human}\njson:\n{json:#}"
    );
}

#[then("the inspection completes successfully")]
async fn assert_inspection_succeeded(world: &mut E2eWorld) {
    // The documented contract: the outcome says whether `examine` managed to
    // look, not whether it liked what it found. On the mock lane there is no GPU
    // to find, and that is a finding rather than a failure.
    assert_eq!(
        world.cli_rc,
        Some(0),
        "examine reports what it found; finding nothing is not a failure"
    );
}

#[then("it states a verdict for this machine")]
async fn assert_states_a_verdict(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    // Guards the pairing: exiting 0 while saying nothing would satisfy the step
    // above on its own. The two forms state the verdict differently — the
    // machine-readable one in a `status` field, the human one as the setup-check
    // summary that opens the report — so accept whichever this scenario ran.
    let stated = match serde_json::from_str::<serde_json::Value>(output) {
        Ok(value) => value
            .get(VERDICT_FIELD)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| !v.trim().is_empty()),
        Err(_) => {
            output.contains("ROCm setup check")
                && output
                    .lines()
                    .any(|line| line.trim().starts_with("driver_status:"))
        }
    };
    assert!(
        stated,
        "the inspection must state a verdict for this machine:\n{output}"
    );
}

#[then("the inspection reports that it skipped the frameworks")]
async fn assert_frameworks_skipped(world: &mut E2eWorld) {
    let combined = format!(
        "{}{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    assert_eq!(
        world.cli_rc,
        Some(0),
        "asking to skip the framework probe should be accepted:\n{combined}"
    );
    let framework = parsed_json(world)
        .get("framework")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    assert_eq!(
        framework, "skipped",
        "the report must say the frameworks were skipped, not silently probe them anyway"
    );
}

#[then("it still states a verdict for this machine")]
async fn assert_still_states_a_verdict(world: &mut E2eWorld) {
    // Skipping the frameworks must narrow the probe, not hollow out the report.
    assert_states_a_verdict(world).await;
}
