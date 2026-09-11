// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};

use crate::E2eWorld;
use crate::e2e::tui_driver::{TuiSession, default_timeout};

/// A symptom string that scores a catalog match on both Linux and Windows. It
/// keys off `check_1_arch_not_in_wheel` (a `LINUX_AND_WINDOWS` checker), which
/// scores 50 on the `HSA_STATUS_ERROR_INVALID_ISA` keyword regardless of host
/// state — the covered-arch penalty only applies when a framework arch list is
/// present, so with none installed the match always renders. (The earlier
/// "/dev/kfd" symptom keyed only off the Linux-only render-group checker and so
/// produced no match on Windows.) The specific fix-id is environment-dependent,
/// so scenarios assert the shape of a match, not the id.
const KNOWN_SYMPTOM: &str = "HSA_STATUS_ERROR_INVALID_ISA";

/// The error text a vLLM engine-startup import failure leaves behind, as a user
/// would paste it. `libtorch_cuda.so` is the token that carries it: a ROCm build
/// of torch ships `libtorch_hip.so` and never that file, so it scores on its own
/// without needing the rest of the traceback.
const ENGINE_IMPORT_SYMPTOM: &str =
    "vllm engine fails to start: OSError: libtorch_cuda.so: cannot open shared object file";

/// The catalog entry [`ENGINE_IMPORT_SYMPTOM`] must reach.
const ENGINE_IMPORT_FIX_ID: &str = "fix-17-torch-dlpack";

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

/// The recipe used to prove a failed helper command is explained on stderr with
/// exit code 4. `fix-4-render-group` is the only AUTO recipe whose command-failure
/// branch this suite can force deterministically: its helper (`usermod`, run
/// directly as root or via `sudo` otherwise) is resolved off `$PATH`, so a
/// scenario-controlled `$PATH` (see `command_fails_bin_dir`) can stand a fake
/// `usermod`/`sudo` in for it, unconditionally failing, without needing real
/// root or touching real group membership.
const COMMAND_FAILURE_FIX_ID: &str = "fix-4-render-group";

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
    "fix-16-vllm-oom",
    "fix-17-torch-dlpack",
    "fix-wsl-1-gpu-not-exposed",
    "fix-wsl-2-dxcore-missing",
    "fix-wsl-3-rocdxg-missing",
    "fix-wsl-4-rocdxg-not-linked",
    "fix-wsl-5-distro-too-old",
    "fix-wsl-6-host-driver-too-old",
    "fix-wsl-7-wsl1",
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

/// A WSL distribution name no host will have. Deliberately not a plausible one:
/// the scenario must fail for "this machine does not exist", never because the
/// runner happened to have a distro by that name.
const UNREACHABLE_DISTRO: &str = "rocm-cli-e2e-no-such-distro";

/// The WSL entry whose remedy is entirely on the Windows host, so the CLI can
/// only ever explain it. Applies on WSL, which is what makes the scenario a test
/// of "explained, not attempted" rather than of the wrong-OS refusal.
const WSL_HOST_SIDE_FIX_ID: &str = "fix-wsl-6-host-driver-too-old";

/// Causes that can only exist on bare-metal Linux: they name the amdgpu module,
/// /dev/kfd, the render group, or the distro package manager, none of which
/// govern anything under WSL2.
const BARE_METAL_ONLY_FIX_IDS: &[&str] = &[
    "fix-3-rocm-kernel",
    "fix-4-render-group",
    "fix-5-amdgpu-load",
    "fix-7-stale-repos",
    "fix-10-container",
    "fix-11-iommu",
    "fix-12-installer",
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

/// The directory this scenario stands in for `$PATH`, holding the fake
/// `usermod`/`sudo` scripts. Scoped under the scenario's isolated root so it is
/// cleaned up with everything else.
fn command_fails_bin_dir(world: &E2eWorld) -> std::path::PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join("fake-bin")
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

#[given("a user whose vLLM server ran out of GPU memory")]
async fn user_hit_vllm_oom(world: &mut E2eWorld) {
    world.model_name = Some(
        "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.".to_string(),
    );
}

#[given("a user who hit the vLLM engine-startup import failure")]
async fn user_hit_engine_import_failure(world: &mut E2eWorld) {
    world.model_name = Some(ENGINE_IMPORT_SYMPTOM.to_string());
}

#[given("a user who has chosen the fix for the engine-startup import failure")]
async fn user_chose_engine_import_fix(world: &mut E2eWorld) {
    world.model_name = Some(ENGINE_IMPORT_FIX_ID.to_string());
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

#[given("a user who has approved a fix whose helper command will fail")]
async fn user_approved_fix_that_will_fail(world: &mut E2eWorld) {
    let bin_dir = command_fails_bin_dir(world);
    std::fs::create_dir_all(&bin_dir).expect("failed to create the scenario's fake PATH dir");
    // `usermod` only has to exist for the `which` probe; the command actually run
    // -- `usermod` directly if already root, `sudo usermod ...` otherwise -- goes
    // through one of these two scripts either way, and both fail unconditionally.
    for name in ["usermod", "sudo"] {
        let script = bin_dir.join(name);
        std::fs::write(&script, "#!/bin/sh\nexit 1\n")
            .unwrap_or_else(|e| panic!("failed to write fake {name}: {e}"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                .unwrap_or_else(|e| panic!("failed to chmod fake {name}: {e}"));
        }
    }
    // Restricting `$PATH` to only the fakes above also makes the recipe's
    // root-detection deterministic: it shells out to `id -u`, which is not on
    // this PATH, so the spawn itself fails and reads as "not root" regardless of
    // who runs the suite -- the same `sudo usermod` branch fails on every host,
    // CI or developer machine, root or not.
    world.command_env.push(("PATH", bin_dir.into_os_string()));
    world.command_env.push(("USER", "e2e-test-user".into()));
    world.model_name = Some(COMMAND_FAILURE_FIX_ID.to_string());
}

#[given("a user who has chosen a WSL remedy that belongs on the Windows host")]
async fn user_chose_wsl_host_remedy(world: &mut E2eWorld) {
    world.model_name = Some(WSL_HOST_SIDE_FIX_ID.to_string());
}

#[given("a user who refers to a cause by its position in the diagnosis")]
async fn user_named_diagnosis_position(world: &mut E2eWorld) {
    // Quoted deliberately: unquoted, the shell treats `#1` as a comment and the
    // CLI never sees it. The product behaviour under test is what happens when
    // the argument does arrive.
    world.model_name = Some("#1".to_string());
}

// ── When ───────────────────────────────────────────────────────────

#[given("a user who asks to diagnose a machine that does not exist")]
async fn user_named_a_missing_machine(world: &mut E2eWorld) {
    world.model_name = Some(UNREACHABLE_DISTRO.to_string());
}

#[when("the user asks the CLI to diagnose that machine")]
async fn user_diagnoses_named_machine(world: &mut E2eWorld) {
    let distro = world.model_name.clone().expect("no machine named");
    let (stdout, stderr, rc) = crate::run_rocm(world, &["diagnose", "--distro", &distro]);
    // The refusal goes to stderr; keep both so the assertions can read whichever
    // stream carried it without caring which.
    world.cli_output = Some(format!("{stdout}\n{stderr}"));
    world.cli_rc = Some(rc);
}

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

#[when("the user asks the CLI to apply the approved fix")]
async fn user_applies_approved_fix(world: &mut E2eWorld) {
    let fix_id = world.model_name.clone().expect("no fix id set");
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(world, &["fix", &fix_id, "--yes"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user is asked interactively to apply it and types no")]
async fn user_declines_fix_interactively(world: &mut E2eWorld) {
    let fix_id = world.model_name.clone().expect("no fix id set");
    let home = fix_home(world).display().to_string();
    // `run_rocm`'s piped stdin can never reach `confirm()`'s interactive
    // branch: `is_terminal()` is always false there. A real pseudo-terminal is
    // the only way to reach it, so this step (unlike every other one in this
    // file) drives the CLI through `TuiSession` instead of `run_rocm`.
    //
    // `TuiSession`'s own isolation (`pty_env`) sets HOME to a PTY-only sandbox
    // it owns and never sets SHELL, so without overriding both here the fix
    // would resolve to a different — and possibly nonexistent — rc file than
    // the one the `Given` step planted, the same way the piped sibling
    // (`user_applies_fix_without_agreeing`) overrides them via
    // `run_rocm_with_env`.
    let mut session = TuiSession::spawn_with_env(
        world,
        &["fix", &fix_id, "--device-index", "1"],
        &[("HOME", home.as_str()), ("SHELL", "/bin/bash")],
    )
    .unwrap_or_else(|e| panic!("failed to open the fix prompt: {e}"));
    session
        .wait_for_screen("[y/N]:", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the confirmation prompt never appeared: {e}"));
    session
        .send("n\r")
        .unwrap_or_else(|e| panic!("failed to type the decline: {e}"));
    let rc = session
        .wait_for_exit_code(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the CLI never exited after declining: {e}"));
    world.cli_output = Some(session.screen_text());
    world.cli_rc = Some(rc);
    world.tui = Some(session);
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

/// Locate the `fix-16-vllm-oom` diagnosis in a parsed report's `matched` array,
/// panicking with the raw output if it is absent. Shared by the two `Then`
/// steps that assert on the vLLM OOM entry.
fn find_vllm_oom<'a>(report: &'a serde_json::Value, output: &str) -> &'a serde_json::Value {
    report
        .get("matched")
        .and_then(serde_json::Value::as_array)
        .and_then(|matches| {
            matches.iter().find(|diagnosis| {
                diagnosis.get("id").and_then(serde_json::Value::as_str) == Some("fix-16-vllm-oom")
            })
        })
        .unwrap_or_else(|| panic!("expected fix-16-vllm-oom in diagnosis:\n{output}"))
}

#[then("the diagnosis identifies the vLLM startup OOM")]
async fn assert_diagnosis_identifies_vllm_oom(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "diagnose should exit 0");
    let (report, output) = parsed_diagnosis(world);
    let oom = find_vllm_oom(&report, &output);
    let high_confidence = report
        .get("high_confidence_threshold")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or_else(|| panic!("report must publish its high_confidence_threshold:\n{output}"));
    assert!(
        oom.get("score")
            .and_then(serde_json::Value::as_i64)
            .is_some_and(|score| score >= high_confidence),
        "the distinctive OOM signature should be high-confidence:\n{output}"
    );
    assert_eq!(
        oom.pointer("/fix/auto_applicable")
            .and_then(serde_json::Value::as_bool),
        Some(false),
        "the OOM remedy must remain print-only:\n{output}"
    );
}

#[then("the OOM remedy distinguishes a busy GPU from a model that does not fit")]
async fn assert_oom_remedy_is_conditional(world: &mut E2eWorld) {
    let (report, output) = parsed_diagnosis(world);
    let fix = find_vllm_oom(&report, &output)
        .get("fix")
        .unwrap_or_else(|| panic!("expected fix-16-vllm-oom remediation:\n{output}"));
    let rendered = fix.to_string();
    assert!(
        rendered.contains("shared/busy")
            && rendered.contains("does not fit")
            && rendered.contains("--gpu-memory-utilization")
            && rendered.contains("smaller or quantized"),
        "the remedy must retain both conditional branches:\n{output}"
    );
    assert!(
        !rendered.contains("--tensor-parallel-size"),
        "the remedy must not prescribe the non-existent multi-GPU sharding flag:\n{output}"
    );
    assert!(
        !fix.get("verify")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|verify| verify.contains("--gpu-memory-utilization")),
        "verification must not unconditionally prescribe the tenancy workaround:\n{output}"
    );
}

#[then("the CLI reports the engine-startup import failure as an established cause")]
async fn assert_engine_import_failure_established(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "diagnose should exit 0 (it is a query)"
    );
    let (report, output) = parsed_diagnosis(world);
    // Read the bar out of the document rather than restating 50: the report
    // publishes it so callers need not hardcode it, and a test that hardcodes it
    // is not exercising that.
    let threshold = report
        .get("min_score_for_match")
        .and_then(serde_json::Value::as_i64)
        .expect("diagnose JSON must publish its match threshold");
    let score = report
        .get("matched")
        .and_then(|m| m.as_array())
        .expect("diagnose JSON has no 'matched' array")
        .iter()
        .find(|d| d.get("id").and_then(serde_json::Value::as_str) == Some(ENGINE_IMPORT_FIX_ID))
        .and_then(|d| d.get("score"))
        .and_then(serde_json::Value::as_i64)
        .unwrap_or_else(|| {
            panic!("the catalog did not recognise the engine-startup import failure:\n{output}")
        });
    // Below the bar the entry is presented among the sub-threshold noise it was
    // added to outrank, which is the state the report was in before it existed.
    assert!(
        score >= threshold,
        "{ENGINE_IMPORT_FIX_ID} scored {score}, under the report's own threshold \
         of {threshold}, so it is not an established cause:\n{output}"
    );
}

#[then("the printed plan says which shell each step runs in")]
async fn assert_plan_says_which_shell_each_step_runs_in(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no fix preview output");
    // The rendered block is a `Commands:` header followed by one `  $ <line>`
    // per entry; `map_while` stops at the first line that is not one of those,
    // which is the `Flags:` row underneath.
    let commands: Vec<&str> = output
        .lines()
        .skip_while(|line| !line.starts_with("Commands:"))
        .skip(1)
        .map_while(|line| line.strip_prefix("  $ "))
        .collect();
    assert!(
        !commands.is_empty(),
        "the preview printed no command block at all:\n{output}"
    );
    let position = |needle: &str| {
        commands
            .iter()
            .position(|c| c.trim() == needle)
            .unwrap_or_else(|| panic!("the printed plan no longer runs `{needle}`:\n{output}"))
    };
    let open = position("rocm engines shell vllm");
    let leave = position("exit");
    let act = position("rocm engines install vllm --reinstall");
    // Ordered: the subshell is opened, then left, and only then is the engine
    // reinstalled -- that step replaces the environment the subshell stands in.
    assert!(
        open < leave && leave < act,
        "the plan must open the subshell, leave it, and only then reinstall:\n{output}"
    );
    let is_comment = |c: &&str| c.trim_start().starts_with('#');
    // And labelled, not merely ordered: with every line carrying the same `$`
    // prefix, a reader has nothing else to tell the two contexts apart.
    assert!(
        commands[open..leave]
            .iter()
            .any(|c| is_comment(c) && c.contains("INSIDE")),
        "the plan must say the probes run INSIDE the subshell:\n{output}"
    );
    assert!(
        commands[leave..act]
            .iter()
            .any(|c| is_comment(c) && c.contains("YOUR OWN shell")),
        "the plan must say the reinstall runs back in the user's own shell:\n{output}"
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
    // uses. The two are computed by different code paths off the same probe, so
    // this is a cross-check rather than a tautology.
    //
    // This used to read `status == "wsl"` as "uncovered". WSL2 has its own
    // catalog entries now, so the two questions came apart: the platforms with no
    // entries are the ones that are neither Linux, Windows, nor WSL.
    let (examine, _, rc) = crate::run_rocm(world, &["examine", "--json"]);
    assert_eq!(rc, 0, "examine should exit 0 (it is an inspector)");
    let host: serde_json::Value =
        serde_json::from_str(&examine).expect("examine --json did not emit valid JSON");
    let host_says_uncovered = host
        .get("status")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|status| status == "unsupported-os");
    let diagnosis_says_uncovered = report.get("out_of_scope").is_some_and(|v| !v.is_null());
    assert_eq!(
        diagnosis_says_uncovered, host_says_uncovered,
        "the diagnosis and the host report disagree about whether this platform \
         is covered (diagnosis={diagnosis_says_uncovered}, host={host_says_uncovered})\
         \n{output}\n{examine}"
    );
}

#[then("no reported cause is one that only exists on bare-metal Linux")]
async fn assert_no_bare_metal_cause(world: &mut E2eWorld) {
    let (report, output) = parsed_diagnosis(world);
    let matched = report
        .get("matched")
        .and_then(|m| m.as_array())
        .expect("diagnose JSON has no 'matched' array");
    for entry in matched {
        let id = entry
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        assert!(
            !BARE_METAL_ONLY_FIX_IDS.contains(&id),
            "{id} names something WSL2 does not have (amdgpu module, /dev/kfd, \
             render group), so reporting it here would send the user after a \
             fault that cannot exist on this platform:\n{output}"
        );
    }
}

#[then("the CLI refuses and explains that it could not reach that machine")]
async fn assert_unreachable_machine_refused(world: &mut E2eWorld) {
    let output = world.cli_output.clone().unwrap_or_default();
    let rc = world.cli_rc.expect("no exit code recorded");
    assert_ne!(
        rc, 0,
        "asking about an unreachable machine must fail:\n{output}"
    );
    // Not `contains("wsl")`: that matches essentially any message this code path
    // can emit, so it would pass on a refusal that never said what went wrong.
    // The refusal has to name the machine the user asked about, or say that
    // reaching another machine is not possible from here at all.
    let lowered = output.to_lowercase();
    assert!(
        lowered.contains(&UNREACHABLE_DISTRO.to_lowercase())
            || lowered.contains("wsl.exe was not found"),
        "the refusal must name the machine it could not reach, or say why no \
         machine could be reached:\n{output}"
    );
}

#[then("no diagnosis of this machine is reported")]
async fn assert_no_local_diagnosis_substituted(world: &mut E2eWorld) {
    // The failure this guards is a silent substitution: reporting on the local
    // machine when the user asked about another one. A diagnosis is recognisable
    // by its `id:` line and its `apply with:` call to action, so neither may be
    // present.
    let output = world.cli_output.clone().unwrap_or_default();
    for marker in ["id: fix-", "apply with:"] {
        assert!(
            !output.contains(marker),
            "a request about another machine must not be answered with this \
             one's diagnosis (found {marker:?}):\n{output}"
        );
    }
}

#[then("the result says this platform is covered")]
async fn assert_platform_is_covered(world: &mut E2eWorld) {
    let (report, output) = parsed_diagnosis(world);
    let out_of_scope = report.get("out_of_scope");
    assert!(
        out_of_scope.is_none_or(serde_json::Value::is_null),
        "this platform has catalog entries, so it must not be reported as \
         uncovered:\n{output}"
    );
}

#[then("the CLI explains the remedy instead of carrying it out")]
async fn assert_remedy_explained_not_applied(world: &mut E2eWorld) {
    let fix_id = world
        .model_name
        .clone()
        .expect("scenario did not choose a fix");
    let (output, _, rc) = crate::run_rocm(world, &["fix", &fix_id]);
    // 0, not the 3 a wrong-OS refusal gives: this fix does apply here. It is
    // print-only because the change belongs to the Windows host, and the two
    // outcomes must stay distinguishable to a caller.
    assert_eq!(
        rc, 0,
        "{fix_id} applies on this host and is print-only, so it must succeed \
         without acting:\n{output}"
    );
    let lowered = output.to_lowercase();
    assert!(
        lowered.contains("print-only"),
        "the CLI must say it only printed a plan:\n{output}"
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
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert!(
        stderr.contains("This fix only applies on:"),
        "the refusal must say which platforms the fix is for, on stderr:\n{stderr}"
    );
    let stdout = world.cli_output.as_deref().unwrap_or("");
    assert!(
        !stdout.contains("This fix only applies on:"),
        "the platform refusal must not also be on stdout:\n{stdout}"
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
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert!(
        stderr.contains("Unknown fix-id"),
        "expected an 'Unknown fix-id' message on stderr:\n{stderr}"
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
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert!(
        stderr.contains("position"),
        "the refusal must say the argument was read as a position, on stderr:\n{stderr}"
    );
    // And it must point at what to use instead, or the correction is useless.
    assert!(
        stderr.contains("id:"),
        "the refusal must name the identifier to use instead, on stderr:\n{stderr}"
    );
}

#[then("the CLI refuses and explains that it needs agreement")]
async fn assert_refuses_without_agreement(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    // The refusal has to say *why* and how to proceed. A bare non-zero exit
    // reads as a broken fix rather than a deliberate stop.
    assert!(
        stderr.contains("--yes"),
        "the refusal must name what to pass to proceed, on stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("refusing to apply"),
        "the refusal must say it did not apply the fix, on stderr:\n{stderr}"
    );
    let stdout = world.cli_output.as_deref().unwrap_or("");
    assert!(
        !stdout.contains("refusing to apply"),
        "the agreement refusal must not also be on stdout:\n{stdout}"
    );
    // Distinct from the unknown-id refusal (2), so a script can tell "you did
    // not agree" apart from "no such fix".
    assert_eq!(
        world.cli_rc,
        Some(5),
        "declining to apply is its own outcome, not an error:\n{stderr}"
    );
}

#[then("the CLI declines on the terminal and explains that it needs agreement")]
async fn assert_interactive_decline_reported(world: &mut E2eWorld) {
    // Same outcome as the non-interactive refusal (diagnose-08): declining is
    // its own outcome, not an error.
    assert_eq!(
        world.cli_rc,
        Some(5),
        "declining an interactive prompt is its own outcome, not an error"
    );
    let screen = world.cli_output.as_deref().unwrap_or("");
    assert!(
        screen.contains("Not confirmed; refusing to apply."),
        "the terminal must show the same decline message the non-interactive \
         path reports, on screen:\n{screen}"
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

#[then("the CLI reports the command failure on stderr with exit code 4")]
async fn assert_command_failure_reported_on_stderr(world: &mut E2eWorld) {
    // 4 is its own outcome: a command that ran and failed, distinct from 3
    // (does not apply here) and 5 (user declined).
    assert_eq!(
        world.cli_rc,
        Some(4),
        "a fix whose helper command fails should exit 4"
    );
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert!(
        stderr.contains("usermod exited") && stderr.contains("group membership NOT changed"),
        "the command-failure explanation must be on stderr:\n{stderr}"
    );
    let stdout = world.cli_output.as_deref().unwrap_or("");
    assert!(
        !stdout.contains("group membership NOT changed"),
        "the command-failure explanation must not also be on stdout:\n{stdout}"
    );
}
