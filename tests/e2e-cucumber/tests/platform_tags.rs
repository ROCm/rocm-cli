// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Drift guard: a scenario gated on a host capability also says which OS it
//! is for, or is reviewed as OS-neutral.
//!
//! A capability tag (`@requires-gpu`, `@requires-engine:<e>`, …) asks what the
//! host *has*, never what it *is*: the Windows GPU lane has a GPU, so it runs
//! every `@requires-gpu` scenario. A scenario whose premise exists on one OS
//! only — `/sys/class/kfd`, `lspci`, a Win32 call — then reaches a host that
//! cannot hold it, and a step that returns early when its premise is missing
//! passes there without asserting anything. `examine-17` did exactly that on
//! the Windows lane, reported green.
//!
//! So every capability-tagged scenario must either carry a tag that decides
//! its OS (`@requires-os:<os>` or `@requires-wsl`) or be listed in
//! [`ALLOWED_ANY_OS`] with the reason its assertions hold on every OS a lane
//! runs it on. docs/testing.md lists everything this fails on.
//!
//! What it reads is what the harness's `filter_run` reads under its default
//! discovery: every `*.feature` file (extension in any case) under `features/`,
//! recursively, with Scenario Outlines expanded by cucumber's own
//! `expand_examples` so an `Examples:` block's tags join its scenarios' tags,
//! and each scenario's tags parsed by the same [`ScenarioDecl::from_tags`]. A
//! feature file run through cucumber's `--input` instead is not read here.
//!
//! This runs in the ordinary `cargo test` set (unlike the `e2e` target, which
//! needs a real `rocm` binary), so a missing tag is caught without a suite run.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use cucumber::feature::Ext as _;
use cucumber::gherkin::{self, GherkinEnv};
use e2e_cucumber::expectation::ScenarioDecl;

/// For the remote scenarios, which share one reason.
const REMOTE_OVER_SSH: &str = "drives the CLI's SSH client against a Linux container and asserts \
     the client's output, its session list and the container's own state. The client's \
     Unix-only parts (SSH connection multiplexing, the credential file's 0600 mode) are \
     asserted by no step. Only lanes that set E2E_INCLUDE_DOCKER run it, which today are the \
     GitHub-hosted Ubuntu lanes, so a native Windows client is never exercised here";

/// Capability-tagged scenarios reviewed as meaningful on every OS a lane runs
/// them on, by `@id`, with the reason. An entry is a claim about the steps: if
/// one starts asserting something only one OS has, the entry is wrong and the
/// scenario needs `@requires-os:<os>` instead.
const ALLOWED_ANY_OS: &[(&str, &str)] = &[
    // bench.feature
    (
        "bench-load-real-serve",
        "benchmarks whichever engine the host serves with, through the endpoint the CLI prints",
    ),
    // diagnose.feature
    (
        "diagnose-fix-needing-an-argument-says-so",
        "`fix-9-igpu-dgpu` needs `--device-index` on both `linux` and `windows`, and the step \
         asserts exit code 0, the `Flags:` line and no claimed change, the same on both. \
         @requires-bare-metal keeps it off WSL2, which the CLI reports as `wsl`, a platform \
         the entry has no row for: there it is declined as not applicable, the contract \
         diagnose-fix-inapplicable-here-is-declined-not-attempted covers with another entry",
    ),
    // examine.feature
    (
        "examine-detects-gpu-and-driver",
        "asserts that `examine` names a gfx target, which it prints on every OS. The driver \
         step passes when the output contains `amdgpu` or `driver_status` anywhere, and \
         `examine` prints a `driver_status:` line on every host whatever its value, so it \
         asserts no driver state on any OS; the KFD target cross-check returns early off \
         Linux, so on Windows it checks nothing",
    ),
    (
        "examine-framework-names-the-interpreter-that-answered",
        "the managed runtime's interpreter is reported on every OS a runtime installs on",
    ),
    // model_serving.feature — the default engine, whichever the host has
    (
        "serve-large-model-inference",
        "serves through the host's default engine; inference is checked over HTTP",
    ),
    (
        "serve-default-engine-working-endpoint",
        "serves through the host's default engine; the endpoint is checked over HTTP",
    ),
    (
        "serve-default-engine-inference",
        "serves through the host's default engine; inference is checked over HTTP",
    ),
    (
        "serve-small-assistant-short-name",
        "serves through the host's default engine; the step asserts the exact model ref the \
         CLI resolved for that engine, and inference is checked over HTTP",
    ),
    (
        "serve-readiness-contract",
        "the readiness contract is the CLI's own, the same on every OS and engine",
    ),
    (
        "serve-explicit-port-reserved",
        "the managed-port reservation is checked from service records, the same on every OS",
    ),
    (
        "serve-explicit-port-in-use-by-untracked-process",
        "the port is held by a loopback listener the step opens, which works on every OS",
    ),
    (
        "serve-auto-port-skips-occupied",
        "the port is held by a loopback listener the step opens, which works on every OS",
    ),
    (
        "serve-absent-gpu-index-rejected",
        "the --gpu index check runs before any engine starts, on every OS; the step accepts \
         each lane's refusal wording",
    ),
    (
        "serve-anthropic-endpoint-advertised",
        "the Anthropic-compatible proxy is the CLI's own, in front of any engine on any OS",
    ),
    (
        "serve-anthropic-messages-inference",
        "the Anthropic-compatible proxy is the CLI's own, in front of any engine on any OS",
    ),
    (
        "serve-anthropic-endpoint-readiness",
        "the Anthropic-compatible proxy is the CLI's own, in front of any engine on any OS",
    ),
    (
        "serve-anthropic-messages-tool-use",
        "the Anthropic-compatible proxy is the CLI's own, in front of any engine on any OS",
    ),
    (
        "serve-anthropic-messages-streaming",
        "the Anthropic-compatible proxy is the CLI's own, in front of any engine on any OS",
    ),
    // model_serving.feature — Lemonade, which serves on Linux and Windows
    (
        "serve-lemonade-inference",
        "Lemonade serves on Linux and Windows; inference is checked over HTTP",
    ),
    (
        "serve-hf-checkpoint-inference",
        "Lemonade serves on Linux and Windows; inference is checked over HTTP",
    ),
    (
        "serve-missing-gguf-variant-refused",
        "serve may install Lemonade first, but the variant check is OS-neutral code (no cfg \
         branch) that refuses before any server starts; the steps assert the same on every \
         OS, none conditionally: a non-zero exit with the `owner/repo:<variant>` advice that \
         pins the refusal, neither `Q4_K_M` nor `Q4_0` named, and no record under the \
         isolated data dir's `services/`",
    ),
    (
        "serve-stop-leaves-no-background-helper",
        "Lemonade serves on Linux and Windows; the step reads serve's output, the helper's \
         autostart claim and runtime state under the scenario's own data directory, and \
         `rocm daemon --status`, which the CLI keeps the same way on both",
    ),
    // No-GPU refusal
    (
        "serve-no-gpu-fails-fast",
        "the no-GPU refusal is OS-neutral code that runs before any engine is prepared",
    ),
    (
        "serve-lemonade-preparation-recovery",
        "the retry and recovery message are OS-neutral; the failure is injected before any \
         download",
    ),
    (
        "serve-unrelated-live-service-still-fails-fast",
        "the no-GPU refusal is OS-neutral code that runs before any engine is prepared",
    ),
    // remote.feature
    ("remote-doctor-refuses-to-install", REMOTE_OVER_SSH),
    ("remote-serve-publishes-endpoint", REMOTE_OVER_SSH),
    ("remote-status-reports-both-halves", REMOTE_OVER_SSH),
    ("remote-attach-restores-endpoint", REMOTE_OVER_SSH),
    ("remote-stop-clears-both", REMOTE_OVER_SSH),
    ("remote-doctor-reports-remote-health", REMOTE_OVER_SSH),
    (
        "remote-serve-refuses-reusing-already-serving-session",
        REMOTE_OVER_SSH,
    ),
    // runtime_setup.feature
    (
        "runtime-install-sdk-active",
        "the default SDK install and its registration run on Linux and Windows alike",
    ),
    (
        "runtime-path-not-nested",
        "reads `active_runtime_root` from `examine`, which every OS prints",
    ),
    (
        "runtime-update-reports-freshness",
        "the update check compares the active runtime with the channel index on every OS",
    ),
    (
        "runtime-resolve-canonical-release",
        "a dry-run that resolves the release index for the detected target, on every OS",
    ),
    (
        "runtime-resolve-earlier-release",
        "a dry-run that resolves the pinned earlier release from the earlier stream's \
         aggregate index, which publishes for Linux and Windows alike",
    ),
    (
        "runtime-install-sdk-overwrite-requires-yes",
        "the consent refusal is OS-neutral code that runs before any download",
    ),
    (
        "runtime-install-sdk-overwrite-with-yes",
        "the default SDK reinstall runs on Linux and Windows alike",
    ),
    (
        "runtime-lemonade-backend-alignment-opt-out",
        "Lemonade and the default SDK both install on Linux and Windows",
    ),
    // therock_next_generation.feature
    (
        "therock-next-07-live-install-auto-detects-arch",
        "installs with --format wheel, which Windows supports (only the tarball format is \
         Linux-only); the GPU probe runs through the runtime's torch on any OS",
    ),
];

/// One parsed scenario: the feature file it lives in, its name, and its tags as
/// the harness reads them.
struct Parsed {
    file: String,
    name: String,
    decl: ScenarioDecl,
}

fn features_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("features")
}

/// Every `*.feature` file under `dir`, at any depth, matching the extension in
/// any case — the files cucumber's default discovery (`*.feature`,
/// case-insensitive) hands the harness.
fn feature_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let file_type = entry.file_type().expect("file type");
        if file_type.is_dir() {
            feature_files(&path, out);
        } else if file_type.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("feature"))
        {
            out.push(path);
        }
    }
}

/// What [`parse_dir`] read: every scenario the harness would filter, and
/// every tag found where the harness never looks.
struct Suite {
    scenarios: Vec<Parsed>,
    /// Tags on a `Feature:` or `Rule:` line. The harness passes only
    /// `scenario.tags` to `ScenarioDecl::from_tags`, so these do nothing.
    misplaced_tags: Vec<String>,
}

/// The real suite. Panics where [`parse_dir`] fails, since every check here
/// would otherwise pass over nothing.
fn parsed_suite() -> Suite {
    parse_dir(&features_dir()).unwrap_or_else(|e| panic!("{e}"))
}

fn parsed_scenarios() -> Vec<Parsed> {
    parsed_suite().scenarios
}

/// Every scenario under `dir` the harness would filter, including those under
/// a `Rule:` and each expanded row of a Scenario Outline, with each scenario's
/// own tags plus its `Examples:` tags once expanded. Fails where there is no
/// feature file or no scenario, either of which would leave every check here
/// passing over nothing.
fn parse_dir(dir: &Path) -> Result<Suite, String> {
    let mut paths = Vec::new();
    feature_files(dir, &mut paths);
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "no .feature files found under {} — every check here would pass vacuously",
            dir.display(),
        ));
    }
    let mut suite = Suite {
        scenarios: Vec::new(),
        misplaced_tags: Vec::new(),
    };
    for path in paths {
        let feature = gherkin::Feature::parse_path(&path, GherkinEnv::default())
            .map_err(|e| format!("{}: {e}", path.display()))?
            .expand_examples()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let file = path
            .strip_prefix(dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        let headers = std::iter::once((&feature.keyword, &feature.name, &feature.tags))
            .chain(feature.rules.iter().map(|r| (&r.keyword, &r.name, &r.tags)));
        for (keyword, name, tags) in headers {
            for tag in tags {
                suite
                    .misplaced_tags
                    .push(format!("{file}: {keyword} {name:?} carries @{tag}"));
            }
        }
        let scenarios = feature
            .scenarios
            .iter()
            .chain(feature.rules.iter().flat_map(|r| r.scenarios.iter()));
        for scenario in scenarios {
            suite.scenarios.push(Parsed {
                file: file.clone(),
                name: scenario.name.clone(),
                decl: ScenarioDecl::from_tags(&scenario.tags),
            });
        }
    }
    if suite.scenarios.is_empty() {
        return Err(format!(
            "no scenarios parsed from {} — every check here would pass vacuously",
            dir.display(),
        ));
    }
    Ok(suite)
}

/// What a scenario's tags say about where it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placement {
    /// Gated on something the host has: every `@requires-*` tag except the
    /// two that decide the OS.
    capability_gated: bool,
    /// The tags pin the OS: `@requires-os:<os>` or `@requires-wsl`.
    /// `@requires-bare-metal` does not: it skips WSL2 only, so native Windows
    /// still runs it — and `@requires-os:linux` still admits WSL2, which
    /// reports itself as Linux.
    platform_decided: bool,
}

/// Classify a scenario's declaration. Destructured without `..`, so a field
/// added to `ScenarioDecl` fails to compile here until it is put on one side or
/// the other. `@nightly`, `@lifecycle`, `@merge-queue` and `@serve-timeout`
/// choose when a scenario runs, not where, so they are neither.
const fn placement(d: &ScenarioDecl) -> Placement {
    let ScenarioDecl {
        id: _,
        requires_gpu,
        requires_multi_gpu,
        requires_gfx_target,
        requires_no_gpu,
        requires_bare_metal,
        requires_wsl,
        requires_oom_fault_injection,
        requires_case_sensitive_fs,
        requires_dir_permissions,
        requires_engine,
        requires_os,
        serve_timeout_secs: _,
        requires_docker,
        nightly: _,
        lifecycle: _,
        merge_queue: _,
    } = d;
    Placement {
        capability_gated: *requires_gpu
            || *requires_multi_gpu
            || *requires_gfx_target
            || *requires_no_gpu
            || *requires_bare_metal
            || requires_engine.is_some()
            || *requires_docker
            || *requires_oom_fault_injection
            || *requires_case_sensitive_fs
            || *requires_dir_permissions,
        platform_decided: requires_os.is_some() || *requires_wsl,
    }
}

fn label(p: &Parsed) -> String {
    let id = p.decl.id.as_deref().unwrap_or("<no @id>");
    format!("@id:{id} ({}: {:?})", p.file, p.name)
}

/// Capability-tagged scenarios that neither decide their OS nor are listed in
/// `allowed`, each with what to do about it.
fn undecided_scenarios(scenarios: &[Parsed], allowed: &[(&str, &str)]) -> Vec<String> {
    let allowed: BTreeSet<&str> = allowed.iter().map(|(id, _)| *id).collect();
    scenarios
        .iter()
        .filter(|p| {
            let placement = placement(&p.decl);
            placement.capability_gated && !placement.platform_decided
        })
        .filter_map(|p| match p.decl.id.as_deref() {
            None => Some(format!(
                "{}: add an @id:<feature-key>-<slug> tag and then an OS tag, or an \
                 ALLOWED_ANY_OS entry under that id",
                label(p)
            )),
            Some(id) if !allowed.contains(id) => Some(format!(
                "{}: add an OS tag, or add the id to ALLOWED_ANY_OS with a reason",
                label(p)
            )),
            Some(_) => None,
        })
        .collect()
}

/// `@id`s that more than one scenario carries. The harness fails a run on the
/// first one; here it would also let one scenario's allowlist entry vouch for
/// another.
fn duplicate_ids(scenarios: &[Parsed]) -> Vec<String> {
    let mut by_id: BTreeMap<&str, Vec<&Parsed>> = BTreeMap::new();
    for p in scenarios {
        if let Some(id) = p.decl.id.as_deref() {
            by_id.entry(id).or_default().push(p);
        }
    }
    by_id
        .into_values()
        .filter(|ps| ps.len() > 1)
        .map(|ps| ps.into_iter().map(label).collect::<Vec<_>>().join(" and "))
        .collect()
}

/// Entries in `allowed` that are duplicated, unexplained, name no scenario, or
/// name one that needs no entry.
fn allowlist_problems(scenarios: &[Parsed], allowed: &[(&str, &str)]) -> Vec<String> {
    let by_id: BTreeMap<&str, &Parsed> = scenarios
        .iter()
        .filter_map(|p| p.decl.id.as_deref().map(|id| (id, p)))
        .collect();
    let mut seen = BTreeSet::new();
    let mut problems = Vec::new();
    for (id, reason) in allowed {
        if !seen.insert(*id) {
            problems.push(format!("{id}: listed more than once"));
        }
        if reason.trim().is_empty() {
            problems.push(format!("{id}: has no reason"));
        }
        match by_id.get(id) {
            None => problems.push(format!("{id}: no scenario has this @id — drop the entry")),
            Some(p) if !placement(&p.decl).capability_gated => problems.push(format!(
                "{}: carries no capability tag, so it needs no entry — drop it",
                label(p)
            )),
            Some(p) if placement(&p.decl).platform_decided => problems.push(format!(
                "{}: its tags already decide its OS, which contradicts an \"any OS\" entry — \
                 drop the entry or the OS tag",
                label(p)
            )),
            Some(_) => {}
        }
    }
    problems
}

#[test]
fn capability_tagged_scenarios_decide_their_os_or_are_reviewed() {
    let violations = undecided_scenarios(&parsed_scenarios(), ALLOWED_ANY_OS);
    assert!(
        violations.is_empty(),
        "{} capability-tagged scenario(s) neither decide their OS nor are reviewed as \
         OS-neutral. A capability tag is not a platform tag: the Windows GPU lane satisfies \
         `@requires-gpu`, so a step whose premise is OS-specific runs there and can pass \
         without asserting anything (AGENTS.md §3). The OS tags are @requires-os:<os> and \
         @requires-wsl; @requires-os:linux still admits WSL2, so add @requires-bare-metal \
         too when WSL2 lacks the interface. An ALLOWED_ANY_OS entry, in \
         tests/e2e-cucumber/tests/platform_tags.rs, states why the steps hold on every OS:\n  {}",
        violations.len(),
        violations.join("\n  "),
    );
}

#[test]
fn scenario_ids_are_unique() {
    let duplicates = duplicate_ids(&parsed_scenarios());
    assert!(
        duplicates.is_empty(),
        "scenario @ids must be unique — the harness fails the run on a repeat:\n  {}",
        duplicates.join("\n  "),
    );
}

#[test]
fn feature_and_rule_lines_carry_no_tags() {
    let misplaced = parsed_suite().misplaced_tags;
    assert!(
        misplaced.is_empty(),
        "the harness reads only each scenario's own tags, so a tag on a Feature: or \
         Rule: line does nothing — put it on each scenario instead:\n  {}",
        misplaced.join("\n  "),
    );
}

#[test]
fn allowlist_entries_are_live_and_needed() {
    let problems = allowlist_problems(&parsed_scenarios(), ALLOWED_ANY_OS);
    assert!(
        problems.is_empty(),
        "ALLOWED_ANY_OS in tests/e2e-cucumber/tests/platform_tags.rs has stale or \
         contradictory entries (AGENTS.md §3):\n  {}",
        problems.join("\n  "),
    );
}

/// The guard is only as good as the parse under it: this pins the scenario
/// that motivated it, so a parse that silently drops tags cannot leave every
/// other check passing over nothing.
#[test]
fn the_kfd_topology_scenario_is_pinned_to_bare_metal_linux() {
    const ID: &str = "examine-lists-every-gpu-with-its-address-and-target";
    let scenarios = parsed_scenarios();
    let examine_17 = scenarios
        .iter()
        .find(|p| p.decl.id.as_deref() == Some(ID))
        .unwrap_or_else(|| panic!("examine.feature no longer has @id:{ID}"));
    assert_eq!(
        examine_17.decl.requires_os.as_deref(),
        Some("linux"),
        "{}: reads /sys/class/kfd, which only Linux has",
        label(examine_17),
    );
    assert!(
        examine_17.decl.requires_bare_metal,
        "{}: WSL2 has no KFD topology either, so it needs @requires-bare-metal",
        label(examine_17),
    );
}

// The checks above run over the real suite, which holds no violation, so they
// cannot show that a broken check would notice one. These feed each check a
// fixture holding exactly one.

fn fixture(tags: &[&str]) -> Parsed {
    Parsed {
        file: "fixture.feature".to_owned(),
        name: tags.join(" "),
        decl: ScenarioDecl::from_tags(tags),
    }
}

#[test]
fn every_tag_is_classified_as_a_capability_an_os_decision_or_neither() {
    // (tag, capability_gated, platform_decided). Every tag `from_tags` reads.
    let table: &[(&str, bool, bool)] = &[
        ("@requires-gpu", true, false),
        ("@requires-multi-gpu", true, false),
        ("@requires-gfx-target", true, false),
        ("@requires-no-gpu", true, false),
        ("@requires-bare-metal", true, false),
        ("@requires-engine:aim", true, false),
        ("@requires-docker", true, false),
        ("@requires-oom-fault-injection", true, false),
        ("@requires-case-sensitive-fs", true, false),
        ("@requires-dir-permissions", true, false),
        ("@requires-os:linux", false, true),
        ("@requires-wsl", false, true),
        ("@id:fixture-01", false, false),
        ("@serve-timeout:60", false, false),
        ("@nightly", false, false),
        ("@lifecycle", false, false),
        ("@merge-queue", false, false),
    ];
    for &(tag, capability_gated, platform_decided) in table {
        assert_eq!(
            placement(&ScenarioDecl::from_tags(&[tag])),
            Placement {
                capability_gated,
                platform_decided
            },
            "{tag}",
        );
    }
}

#[test]
fn an_undecided_scenario_is_reported_and_an_id_less_one_is_told_to_add_an_id() {
    let scenarios = [
        fixture(&["@id:listed", "@requires-gpu"]),
        fixture(&["@id:decided", "@requires-gpu", "@requires-os:linux"]),
        fixture(&["@id:ungated"]),
        fixture(&["@id:unlisted", "@requires-gpu"]),
        fixture(&["@requires-gpu"]),
    ];
    let violations = undecided_scenarios(&scenarios, &[("listed", "a reason")]);
    assert_eq!(violations.len(), 2, "{violations:#?}");
    assert!(
        violations[0].starts_with("@id:unlisted ") && violations[0].contains("ALLOWED_ANY_OS"),
        "{violations:#?}"
    );
    assert!(
        violations[1].starts_with("@id:<no @id> ") && violations[1].contains("add an @id"),
        "{violations:#?}"
    );
}

#[test]
fn a_repeated_id_is_reported() {
    let scenarios = [
        fixture(&["@id:twice", "@requires-gpu"]),
        fixture(&["@id:once"]),
        fixture(&["@id:twice"]),
    ];
    let duplicates = duplicate_ids(&scenarios);
    assert_eq!(duplicates.len(), 1, "{duplicates:#?}");
    assert!(duplicates[0].starts_with("@id:twice "), "{duplicates:#?}");
}

#[test]
fn each_kind_of_bad_allowlist_entry_is_reported() {
    let scenarios = [
        fixture(&["@id:needed", "@requires-gpu"]),
        fixture(&["@id:ungated"]),
        fixture(&["@id:decided", "@requires-gpu", "@requires-wsl"]),
    ];
    assert_eq!(
        allowlist_problems(&scenarios, &[("needed", "a reason")]),
        Vec::<String>::new(),
        "a needed entry with a reason is fine",
    );
    let cases: &[(&[(&str, &str)], &str)] = &[
        (
            &[("needed", "a reason"), ("needed", "a reason")],
            "needed: listed more than once",
        ),
        (&[("needed", " ")], "needed: has no reason"),
        (&[("gone", "a reason")], "gone: no scenario has this @id"),
        (&[("ungated", "a reason")], "carries no capability tag"),
        (&[("decided", "a reason")], "already decide its OS"),
    ];
    for &(allowed, expected) in cases {
        let problems = allowlist_problems(&scenarios, allowed);
        assert_eq!(problems.len(), 1, "{allowed:?}: {problems:#?}");
        assert!(
            problems[0].contains(expected),
            "{allowed:?}: expected {expected:?}, got {problems:#?}"
        );
    }
}

/// A fresh, empty directory under the test target's own scratch space.
fn scratch_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("fixture dir");
    dir
}

#[test]
fn discovery_reads_what_the_harness_reads() {
    // A nested file with an upper-case extension, an outline whose Examples
    // block carries the OS tag, and a scenario under a Rule: each is something
    // the harness's default discovery and outline expansion pick up.
    let dir = scratch_dir("platform_tags_discovery");
    std::fs::create_dir_all(dir.join("nested")).expect("fixture dir");
    std::fs::write(
        dir.join("nested").join("Outline.FEATURE"),
        "Feature: fixture\n\
         \n\
         \x20 @id:fixture-outline @requires-gpu\n\
         \x20 Scenario Outline: fixture-01 - row <n>\n\
         \x20   Given a step\n\
         \n\
         \x20   @requires-os:linux\n\
         \x20   Examples:\n\
         \x20     | n |\n\
         \x20     | 1 |\n\
         \n\
         \x20 Rule: a rule\n\
         \n\
         \x20   @id:fixture-under-rule @requires-gpu\n\
         \x20   Scenario: fixture-02 - under a rule\n\
         \x20     Given a step\n",
    )
    .expect("fixture file");
    std::fs::write(dir.join("ignored.txt"), "not a feature").expect("fixture file");
    let scenarios = parse_dir(&dir).expect("fixture parses").scenarios;
    assert_eq!(scenarios.len(), 2, "one expanded row and one rule scenario");
    let row = &scenarios[0];
    assert_eq!(row.decl.id.as_deref(), Some("fixture-outline"));
    assert_eq!(
        row.decl.requires_os.as_deref(),
        Some("linux"),
        "the Examples tag must reach the row"
    );
    assert!(Path::new(&row.file).ends_with("nested/Outline.FEATURE"));
    let under_rule = &scenarios[1];
    assert_eq!(under_rule.decl.id.as_deref(), Some("fixture-under-rule"));
    assert!(
        placement(&under_rule.decl).capability_gated,
        "a scenario under a Rule is checked like any other"
    );
}

#[test]
fn a_directory_with_no_feature_file_or_no_scenario_is_refused() {
    let empty = scratch_dir("platform_tags_empty");
    std::fs::write(empty.join("notes.txt"), "not a feature").expect("fixture file");
    let err = parse_dir(&empty)
        .err()
        .expect("no feature file must be refused");
    assert!(err.contains("no .feature files"), "{err}");

    let bare = scratch_dir("platform_tags_no_scenarios");
    std::fs::write(bare.join("bare.feature"), "Feature: nothing in it\n").expect("fixture file");
    let err = parse_dir(&bare).err().expect("no scenario must be refused");
    assert!(err.contains("no scenarios parsed"), "{err}");
}

#[test]
fn a_tag_on_a_feature_or_rule_line_is_reported() {
    let dir = scratch_dir("platform_tags_misplaced");
    std::fs::write(
        dir.join("misplaced.feature"),
        "@requires-gpu @nightly\n\
         Feature: tagged at the top\n\
         \n\
         \x20 @id:fixture-rule\n\
         \x20 Rule: tagged rule\n\
         \n\
         \x20   @id:fixture-01\n\
         \x20   Scenario: fixture-01 - a scenario\n\
         \x20     Given a step\n",
    )
    .expect("fixture file");
    let misplaced = parse_dir(&dir).expect("fixture parses").misplaced_tags;
    assert_eq!(
        misplaced,
        [
            "misplaced.feature: Feature \"tagged at the top\" carries @requires-gpu",
            "misplaced.feature: Feature \"tagged at the top\" carries @nightly",
            "misplaced.feature: Rule \"tagged rule\" carries @id:fixture-rule",
        ],
    );
}
