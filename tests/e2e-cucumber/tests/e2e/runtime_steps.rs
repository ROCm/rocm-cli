// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};

use crate::E2eWorld;

#[given("a machine with no CLI-managed runtimes")]
async fn setup_no_runtimes(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    assert!(
        stdout.contains("installed: none") || stdout.contains("managed_runtimes: 0"),
        "expected no managed runtimes:\n{stdout}"
    );
}

#[given("a machine with a standard ROCm install")]
async fn setup_standard_rocm(_world: &mut E2eWorld) {}

#[given("a machine whose runtimes folder is a link to somewhere else")]
async fn setup_linked_runtimes_folder(world: &mut E2eWorld) {
    world
        .link_runtimes_within_scenario()
        .expect("failed to link the scenario's runtimes folder");
}

/// The folder the scenario's `data/runtimes` link points at, resolved so it can be
/// compared against a path the CLI resolved.
///
/// The verbatim prefix has to come back off. `canonicalize` returns `\\?\C:\…` on
/// Windows and the CLI records a plain path, so comparing the two raw would fail on
/// the prefix rather than on the folder — and only on the Windows lane, long after
/// this was written. The CLI strips it for the same reason (`rocm-core`'s
/// `strip_verbatim_prefix`); this crate cannot reach that helper, and a dependency
/// on `rocm-core` for six lines of string handling is the worse trade.
fn linked_runtimes_target(world: &E2eWorld) -> std::path::PathBuf {
    let real = world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .join("data")
        .join("real-runtimes");
    let resolved = real.canonicalize().unwrap_or(real);
    let text = resolved.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return std::path::PathBuf::from(format!(r"\\{rest}"));
    }
    match text.strip_prefix(r"\\?\") {
        Some(rest) => std::path::PathBuf::from(rest),
        None => resolved.clone(),
    }
}

#[when("the user previews an SDK install")]
async fn user_previews_sdk_install(world: &mut E2eWorld) {
    // `--family` because a host with no AMD GPU has no target to detect, and the
    // preview resolves the install folder before it needs one. `--dry-run` keeps
    // this to a plan: no venv, no multi-GiB download.
    let stdout = crate::run_rocm_ok(
        world,
        &["install", "sdk", "--family", "gfx110X-all", "--dry-run"],
    );
    world.cli_output = Some(stdout);
}

/// The `  target: <path>` line of the install preview.
fn planned_runtime_folder(world: &E2eWorld) -> String {
    let output = world.cli_output.as_deref().expect("no install preview");
    output
        .lines()
        .find_map(|line| line.trim().strip_prefix("target: "))
        .unwrap_or_else(|| panic!("no planned runtime folder in the preview:\n{output}"))
        .trim()
        .to_owned()
}

#[then("the planned runtime folder is inside the folder the link points at")]
async fn assert_planned_folder_is_the_real_one(world: &mut E2eWorld) {
    let planned = planned_runtime_folder(world);
    let real = linked_runtimes_target(world);
    assert!(
        std::path::Path::new(&planned).starts_with(&real),
        "the install would record {planned}, which is not inside {}",
        real.display()
    );
}

#[then("the planned runtime folder is not expressed through the link")]
async fn assert_planned_folder_avoids_the_link(world: &mut E2eWorld) {
    // The failure this pins: a folder named through the link reads as valid until
    // the link goes, and takes the environment's console-script shebangs with it.
    let planned = planned_runtime_folder(world);
    let link = world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .join("data")
        .join("runtimes");
    assert!(
        !std::path::Path::new(&planned).starts_with(&link),
        "the install would record {planned}, which names the link at {} rather than \
         the folder it points at",
        link.display()
    );
}

#[given("a managed runtime is active")]
async fn setup_active_runtime(world: &mut E2eWorld) {
    // On a no-GPU host this is a no-op: a managed TheRock SDK runtime can only be
    // installed where there's a GPU family to select wheels for (see
    // `runtime-install-active`, @requires-gpu). The only scenarios that reach this
    // step without `@requires-gpu` are the mock-lane chat scenarios, whose serve
    // is backed by MockServer and needs no runtime at all — so skip the install
    // rather than attempt a multi-GiB SDK pull that can't succeed here.
    if !e2e_cucumber::capability::host_capability().has_amd_gpu {
        return;
    }
    // This precondition only needs *a* runtime present — it does not assert a
    // clean slate — so opt into the shared runtimes tree: the first scenario to
    // hit an empty shared tree installs once, and every later scenario finds the
    // runtime already there instead of re-installing a multi-GiB TheRock SDK
    // (the per-scenario install count is what blew the GPU time cap). No-op unless
    // E2E_SHARED_RUNTIMES_DIR is set (CI on a persistent runner).
    world.use_shared_runtimes();
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    if stdout.contains("installed: none") {
        crate::run_rocm_ok(world, &["install", "sdk"]);
    }
    // Name the runtime rather than leaving the CLI to infer it: the shared tree
    // grows a second runtime whenever the channel index publishes one, and the
    // CLI refuses to auto-select from more than one (see
    // `E2eWorld::activate_shared_runtime`). Without this the step still passes —
    // a runtime IS present — and the serve that follows fails instead.
    world.activate_shared_runtime();
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    assert!(
        !stdout.contains("installed: none"),
        "no managed runtime is active:\n{stdout}"
    );
}

#[given("a managed runtime with an inference engine already installed")]
async fn setup_runtime_with_engine(world: &mut E2eWorld) {
    // Share the runtimes tree for the same reason `a managed runtime is active` does:
    // the first scenario to find it empty pays for the multi-GiB SDK pull, the rest
    // reuse it. `install sdk` auto-installs the family's preferred engine, so one
    // install satisfies both halves of this precondition.
    world.use_shared_runtimes();
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    if stdout.contains("installed: none") {
        crate::run_rocm_ok(world, &["install", "sdk"]);
    }
    // Same reason as `a managed runtime is active`: pin the runtime explicitly,
    // or the serve that follows refuses to pick one. Not for `assert_engine_ready`
    // below — `engines list` scans every registered manifest and never consults
    // the active key, which is exactly why it cannot stand in for this call.
    world.activate_shared_runtime();
    assert_engine_ready(world);
}

#[when("the user installs the SDK")]
async fn user_installs_sdk(world: &mut E2eWorld) {
    let stdout = crate::run_rocm_ok(world, &["install", "sdk"]);
    world.cli_output = Some(stdout);
}

#[when("the user installs the SDK again")]
async fn user_reinstalls_sdk(world: &mut E2eWorld) {
    user_installs_sdk(world).await;
}

#[then("the runtime can still use the GPU")]
async fn assert_runtime_can_still_use_the_gpu(world: &mut E2eWorld) {
    let output = world.cli_output.as_deref().expect("no install output");
    // The functional signal rather than a diagnostic string: the device check asks
    // the runtime's own torch how many devices it can open. A runtime left holding
    // a torch that one of the two installers cannot use reports none, and that is
    // the failure this scenario exists to catch.
    assert!(
        output.contains("device_check: usable"),
        "the reinstall left a runtime that cannot open a GPU:\n{output}"
    );
    // A genuine unmet requirement must still fail the scenario. Torch itself is
    // expected to diverge from the engine's exact pin once it is settled on the
    // SDK's build of the same release, and that is reported as a divergence.
    assert!(
        !output.contains("dependency_check: violated"),
        "the reinstall left a genuine requirement unmet:\n{output}"
    );
}

/// The engine inventory reports a usable engine runtime.
///
/// A precondition only. It deliberately has no Then counterpart: `engines list`
/// reports `runtime: ready` even while the engine's pinned dependencies are
/// violated — that false green is the very thing this feature's scenario exists
/// to catch — so asserting it afterwards would pass whether or not the fix
/// works. Teaching that surface to notice a violated pin is tracked separately;
/// until it does, the device check is the falsifiable signal, because it asks
/// the runtime how many GPUs it can actually open rather than whether it looks
/// installed.
fn assert_engine_ready(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["engines", "list"]);
    assert!(
        stdout.contains("runtime: ready"),
        "no engine runtime is ready:\n{stdout}"
    );
}

#[when("the user tries to adopt the existing install")]
async fn user_tries_adopt(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "runtimes",
            "adopt",
            "--python",
            "/usr/bin/python3",
            "--root",
            "/opt/rocm",
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("a runtime is registered")]
async fn assert_runtime_registered(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    assert!(
        !stdout.contains("installed: none"),
        "no runtime registered after install:\n{stdout}"
    );
}

#[then("the runtime is set as active")]
async fn assert_runtime_active(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    let active = stdout
        .lines()
        .find(|l| l.contains("active_runtime_key:"))
        .and_then(|l| l.split(':').nth(1))
        .map_or("", str::trim);
    assert!(
        !active.is_empty() && active != "<unset>",
        "runtime not set as active:\n{stdout}"
    );
}

#[then("the runtime includes an inference engine")]
async fn assert_runtime_has_stack(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["examine"]);
    assert!(
        stdout.contains("torch") || stdout.contains("vllm"),
        "no inference stack found in runtime:\n{stdout}"
    );
}

#[then("the managed runtime folder path is not recursively nested")]
async fn assert_runtime_path_not_nested(world: &mut E2eWorld) {
    // `rocm examine` prints `Folder: <install_root>` for the active runtime.
    let output = world.cli_output.as_ref().expect("no examine output");
    let Some(folder) = output
        .lines()
        .find_map(|l| l.trim().strip_prefix("Folder:"))
        .map(str::trim)
    else {
        panic!("no 'Folder:' line in examine output:\n{output}");
    };
    // A healthy path contains `runtimes/wheel` at most once. Re-provisioning
    // inside an existing runtime produces `runtimes/wheel/.../runtimes/wheel/`
    // (dogfooding #17). Count occurrences of the marker segment.
    let nested = folder.matches("runtimes/wheel").count() > 1
        || folder.matches("runtimes\\wheel").count() > 1;
    assert!(
        !nested,
        "managed runtime folder path is recursively nested (dogfooding #17):\n{folder}"
    );
}

#[when("the user checks for runtime updates")]
async fn user_checks_for_updates(world: &mut E2eWorld) {
    // Plain `rocm update` — the check-only form. Without `--apply` it never
    // mutates the runtime tree, so this is safe to run against the shared runtime
    // the other scenarios serve from.
    world.use_shared_runtimes();
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// The runtime key `runtimes list` reports as active, if any.
fn active_runtime_key(world: &E2eWorld) -> Option<String> {
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    let key = stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("active_runtime_key:"))?
        .trim();
    (!key.is_empty() && key != "<unset>").then(|| key.to_owned())
}

/// Freshness verdicts `runtime_update_plan` can emit, plus the degraded `error`
/// form used when the index cannot be reached. `xtask e2e-prewarm` routes on
/// exactly these, so a rename here must break this scenario rather than silently
/// turn every pre-warm into a no-op reuse.
const UPDATE_STATUSES: [&str; 4] = ["up_to_date", "update_available", "ahead_of_index", "error"];

#[then("the report states the runtime's freshness against the channel index")]
async fn assert_update_reports_freshness(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or("");
    let rc = world.cli_rc.expect("no command was run");
    assert_eq!(rc, 0, "`rocm update` failed:\n{stdout}");

    // The line `xtask e2e-prewarm` parses: `runtime <key> ... status=<verdict>`.
    // The report carries one such line per installed runtime, newest first, and
    // the shared tree holds more than one — so select the ACTIVE runtime's line
    // rather than whichever came first, or this scenario reports on a runtime the
    // run never used.
    let active = active_runtime_key(world);
    let runtime_lines = || {
        stdout
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("runtime "))
    };
    let line = match active.as_deref() {
        // Something is active: assert on ITS line or not at all. Falling back to
        // the first line here would report on a runtime the run did not use —
        // the misattribution this selection exists to remove — and it would pass
        // while doing it.
        Some(key) => {
            let found = runtime_lines().find(|line| line.split_whitespace().nth(1) == Some(key));
            assert!(
                found.is_some(),
                "runtime `{key}` is active but the update report has no `runtime {key} …` \
                 line:\n{stdout}"
            );
            found
        }
        // Nothing active: the single-runtime case this scenario was written
        // against, where the sole line is unambiguously the right one.
        None => runtime_lines().next(),
    };
    let Some(line) = line else {
        panic!("no `runtime <key> …` line in the update report:\n{stdout}");
    };
    let status = line
        .split_whitespace()
        .find_map(|field| field.strip_prefix("status="));
    let Some(status) = status else {
        panic!("update report line carries no `status=` field:\n{line}");
    };
    assert!(
        UPDATE_STATUSES.contains(&status),
        "unrecognised freshness status `{status}`; `xtask e2e-prewarm` routes on \
         {UPDATE_STATUSES:?} and would silently reuse a stale runtime:\n{line}"
    );
    // The pre-warm selects the line for its own channel, so the field it filters
    // on must be present too — except on the degraded error line, which the
    // renderer emits without one.
    if status != "error" {
        assert!(
            line.split_whitespace()
                .any(|field| field.starts_with("channel=")),
            "update report line carries no `channel=` field, so a per-channel \
             pre-warm cannot attribute it:\n{line}"
        );
    }
}

#[then("the adoption is refused")]
async fn assert_adoption_refused(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command was run");
    assert!(rc != 0, "adopt unexpectedly succeeded");
}

#[then("the error explains which install types can be adopted")]
async fn assert_adopt_error_explains(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or("");
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let combined = format!("{stdout}{stderr}").to_lowercase();
    assert!(
        combined.contains("therock")
            || combined.contains("rocm_sdk")
            || combined.contains("not supported"),
        "error does not explain TheRock requirement:\n{stdout}\n{stderr}"
    );
}
