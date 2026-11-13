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
//! runs it on. The tags are read by [`ScenarioDecl::from_tags`] from each
//! scenario's own tags, exactly as the harness's `filter_run` does, so this
//! guard cannot disagree with the harness about what a scenario declares.
//!
//! This runs in the ordinary `cargo test` set (unlike the `e2e` target, which
//! needs a real `rocm` binary), so a missing tag is caught without a suite run.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use cucumber::gherkin::{self, GherkinEnv};
use e2e_cucumber::expectation::ScenarioDecl;

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
        "the needs-arg fix is catalogued for bare-metal Linux and Windows alike; \
         @requires-bare-metal only keeps it off WSL2, whose refusal has its own scenario",
    ),
    // examine.feature
    (
        "examine-detects-gpu-and-driver",
        "the GPU and driver assertions read `examine` output on every OS; its KFD target \
         cross-check returns early off bare-metal Linux, so on Windows that one step checks nothing",
    ),
    (
        "examine-can-skip-framework-probing",
        "skipping the framework probe is OS-neutral; @requires-bare-metal only keeps it off \
         WSL2, where the probe stops before the framework step",
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
        "serve-stop-leaves-no-background-helper",
        "Lemonade serves on Linux and Windows; the helper check reads the CLI's service records",
    ),
    // vLLM: the engine gate decides the OS
    (
        "serve-vllm-inference",
        "OS decided by the vLLM engine gate, which refuses native Windows; the native \
         Windows refusal is covered by serve-33/34/35",
    ),
    (
        "serve-vllm-default-on-instinct",
        "OS decided by the vLLM engine gate, which refuses native Windows; the native \
         Windows refusal is covered by serve-33/34/35",
    ),
    (
        "runtime-sdk-reinstall-keeps-engine-consistent",
        "OS decided by the vLLM engine gate, which refuses native Windows; the native \
         Windows refusal is covered by serve-33/34/35",
    ),
    (
        "runtime-torch-alignment-opt-out",
        "OS decided by the vLLM engine gate, which refuses native Windows; the native \
         Windows refusal is covered by serve-33/34/35",
    ),
    (
        "runtime-install-sdk-serves-without-toolchain",
        "OS decided by the vLLM engine gate, which refuses native Windows; the native \
         Windows refusal is covered by serve-33/34/35",
    ),
    (
        "runtime-lifecycle-activate-restart-services-succeeds",
        "OS decided by the vLLM engine gate, which refuses native Windows; the native \
         Windows refusal is covered by serve-33/34/35",
    ),
    (
        "therock-next-09-live-install-reports-vllm-rocm10x-discovery-pins",
        "OS decided by the vLLM engine gate, which refuses native Windows; the native \
         Windows refusal is covered by serve-33/34/35",
    ),
    // AIM: engine availability (Docker plus a verified GPU family) decides the host
    (
        "serve-aim-inference",
        "where AIM runs is decided by its engine gate (Docker and a verified GPU family), \
         not by the OS",
    ),
    (
        "serve-aim-lifecycle-parity",
        "where AIM runs is decided by its engine gate (Docker and a verified GPU family), \
         not by the OS",
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
    // remote.feature — the second machine is a container; the steps assert only what
    // the remote reports, not the client's Unix-only socket multiplexing or file modes
    (
        "remote-doctor-refuses-to-install",
        "asserts what the container reports over SSH, which is the same from any client OS",
    ),
    (
        "remote-serve-publishes-endpoint",
        "asserts what the container reports over SSH, which is the same from any client OS",
    ),
    (
        "remote-status-reports-both-halves",
        "asserts what the container reports over SSH, which is the same from any client OS",
    ),
    (
        "remote-attach-restores-endpoint",
        "asserts what the container reports over SSH, which is the same from any client OS",
    ),
    (
        "remote-stop-clears-both",
        "asserts what the container reports over SSH, which is the same from any client OS",
    ),
    (
        "remote-doctor-reports-remote-health",
        "asserts what the container reports over SSH, which is the same from any client OS",
    ),
    (
        "remote-serve-refuses-reusing-already-serving-session",
        "asserts what the container reports over SSH, which is the same from any client OS",
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

/// Every scenario in every `.feature` file, including those under a `Rule:`,
/// since the harness filters those too. Only each scenario's own tags are read:
/// the harness passes `scenario.tags` to `ScenarioDecl::from_tags` and ignores
/// feature- and rule-level tags, so a tag placed there gates nothing.
fn parsed_scenarios() -> Vec<Parsed> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(features_dir())
        .expect("features dir")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "feature"))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no .feature files found under {} — every check here would pass vacuously",
        features_dir().display(),
    );
    let mut out = Vec::new();
    for path in paths {
        let feature = gherkin::Feature::parse_path(&path, GherkinEnv::default())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let file = path
            .file_name()
            .expect("file name")
            .to_string_lossy()
            .into_owned();
        let scenarios = feature
            .scenarios
            .iter()
            .chain(feature.rules.iter().flat_map(|r| r.scenarios.iter()));
        for scenario in scenarios {
            out.push(Parsed {
                file: file.clone(),
                name: scenario.name.clone(),
                decl: ScenarioDecl::from_tags(&scenario.tags),
            });
        }
    }
    assert!(
        !out.is_empty(),
        "no scenarios parsed from {} — every check here would pass vacuously",
        features_dir().display(),
    );
    out
}

/// Whether the scenario is gated on something the host has — every
/// `@requires-*` tag `ScenarioDecl` knows, except the two that decide the OS.
/// `@nightly`, `@lifecycle`, `@merge-queue` and `@serve-timeout` choose when a
/// scenario runs, not where, so they are not capabilities.
const fn is_capability_gated(d: &ScenarioDecl) -> bool {
    d.requires_gpu
        || d.requires_multi_gpu
        || d.requires_gfx_target
        || d.requires_no_gpu
        || d.requires_bare_metal
        || d.requires_engine.is_some()
        || d.requires_docker
        || d.requires_oom_fault_injection
        || d.requires_case_sensitive_fs
        || d.requires_dir_permissions
}

/// Whether the scenario's tags pin the OS it runs on. `@requires-bare-metal`
/// does not: it skips WSL2 only, so native Windows still runs it.
const fn is_platform_decided(d: &ScenarioDecl) -> bool {
    d.requires_os.is_some() || d.requires_wsl
}

fn label(p: &Parsed) -> String {
    let id = p.decl.id.as_deref().unwrap_or("<no @id>");
    format!("@id:{id} ({}: {:?})", p.file, p.name)
}

#[test]
fn capability_tagged_scenarios_decide_their_os_or_are_reviewed() {
    let allowed: BTreeMap<&str, &str> = ALLOWED_ANY_OS.iter().copied().collect();
    let is_allowed = |p: &&Parsed| {
        p.decl
            .id
            .as_deref()
            .is_some_and(|id| allowed.contains_key(id))
    };
    let violations: Vec<String> = parsed_scenarios()
        .iter()
        .filter(|p| is_capability_gated(&p.decl) && !is_platform_decided(&p.decl))
        .filter(|p| !is_allowed(p))
        .map(label)
        .collect();
    assert!(
        violations.is_empty(),
        "{} capability-tagged scenario(s) neither decide their OS nor are reviewed as \
         OS-neutral. A capability tag is not a platform tag: the Windows GPU lane satisfies \
         `@requires-gpu`, so a step whose premise is OS-specific runs there and can pass \
         without asserting anything (AGENTS.md §3). For each, add @requires-os:<os> (plus \
         @requires-bare-metal if WSL2 lacks the interface), or add the id with a reason to \
         ALLOWED_ANY_OS in tests/e2e-cucumber/tests/platform_tags.rs:\n  {}",
        violations.len(),
        violations.join("\n  "),
    );
}

#[test]
fn allowlist_entries_are_live_and_needed() {
    let scenarios = parsed_scenarios();
    let by_id: BTreeMap<&str, &Parsed> = scenarios
        .iter()
        .filter_map(|p| p.decl.id.as_deref().map(|id| (id, p)))
        .collect();
    let mut seen = BTreeSet::new();
    let mut problems = Vec::new();
    for (id, reason) in ALLOWED_ANY_OS {
        if !seen.insert(*id) {
            problems.push(format!("{id}: listed more than once"));
        }
        if reason.trim().is_empty() {
            problems.push(format!("{id}: has no reason"));
        }
        match by_id.get(id) {
            None => problems.push(format!("{id}: no scenario has this @id — drop the entry")),
            Some(p) if !is_capability_gated(&p.decl) => problems.push(format!(
                "{}: carries no capability tag, so it needs no entry — drop it",
                label(p)
            )),
            Some(p) if is_platform_decided(&p.decl) => problems.push(format!(
                "{}: its tags already decide its OS, which contradicts an \"any OS\" entry — \
                 drop the entry or the OS tag",
                label(p)
            )),
            Some(_) => {}
        }
    }
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
