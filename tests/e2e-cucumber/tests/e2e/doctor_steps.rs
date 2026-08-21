// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};

use crate::E2eWorld;

/// The first line of the host report. Present on every platform the CLI runs
/// on, so it is a safe marker for "this output describes a machine" without
/// pinning any hardware-dependent detail.
const HOST_REPORT_MARKER: &str = "ROCm setup check";

/// Where the captured report is written inside the scenario's isolated root.
fn saved_report_path(world: &E2eWorld) -> std::path::PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .join("captured-report.json")
}

/// Root of the scenario's isolated tree.
fn root(world: &E2eWorld) -> std::path::PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .to_path_buf()
}

/// The runtime key planted by the "missing from the records" fixture.
const PLANTED_RUNTIME_KEY: &str = "e2e-doctor-readonly";

/// Where the CLI records a runtime once it knows about it.
fn registry_entry(world: &E2eWorld) -> std::path::PathBuf {
    root(world)
        .join("data")
        .join("runtimes")
        .join("registry")
        .join(format!("{PLANTED_RUNTIME_KEY}.json"))
}

/// Whether the output carries a findings section.
///
/// The catalog renders one of three ways depending on the host: a scored match
/// (`#1 [HIGH score=90/100] ...`), no known match, or out of scope for the
/// platform. All three are a verdict; which one appears is environment-
/// dependent, so the scenarios assert that a verdict was reached at all.
///
/// Matched without the command's own name, because each command labels the
/// section with itself — `rocm doctor:` or `rocm diagnose:` — and this helper is
/// used against both.
fn has_findings(output: &str) -> bool {
    output.contains("score=")
        || output.contains(": out of scope for this platform.")
        || output.contains(": no known misconfiguration matched.")
}

// ── Given ──────────────────────────────────────────────────────────

#[given("a script written against an earlier release")]
async fn script_against_earlier_release(_world: &mut E2eWorld) {
    // Nothing to arrange: the point is that the old command names still work
    // with no migration step on the caller's side.
}

#[given("a report captured from an earlier check")]
async fn report_captured_earlier(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["doctor", "--json"]);
    assert_eq!(rc, 0, "capturing a report should exit 0:\n{stderr}");
    let path = saved_report_path(world);
    std::fs::write(&path, &stdout).expect("failed to write the captured report");
    world.model_name = Some(path.display().to_string());
}

#[given("a machine with a ROCm install missing from the CLI's records")]
async fn install_missing_from_records(world: &mut E2eWorld) {
    // An install that is real on disk (a payload file plus its own manifest) and
    // configured as the setup runtime, but absent from the CLI's registry. That
    // is exactly the state the host report used to silently repair.
    let install_root = root(world).join("setup-install");
    std::fs::create_dir_all(&install_root).expect("failed to create the install root");
    std::fs::write(install_root.join("payload.txt"), "payload\n").expect("failed to write payload");
    let manifest = serde_json::json!({
        "runtime_key": PLANTED_RUNTIME_KEY,
        "runtime_id": "e2e-doctor-readonly-id",
        "channel": "release",
        "format": "tarball",
        "family": "gfx110X",
        "family_source": "e2e",
        "version": "0.0.0-e2e",
        "install_root": install_root,
        "selected_artifact_url": "https://example.invalid/e2e.tar.gz",
        "installed_at_unix_ms": 0,
    });
    std::fs::write(
        install_root.join(".rocm-cli-runtime.json"),
        serde_json::to_vec_pretty(&manifest).expect("failed to encode the runtime manifest"),
    )
    .expect("failed to plant the runtime manifest");

    let config_dir = root(world).join("config");
    std::fs::create_dir_all(&config_dir).expect("failed to create the config dir");
    let config = serde_json::json!({ "setup": { "therock_venv": install_root } });
    std::fs::write(
        config_dir.join("config.json"),
        serde_json::to_vec_pretty(&config).expect("failed to encode the config"),
    )
    .expect("failed to plant the config");

    assert!(
        !registry_entry(world).exists(),
        "the fixture must start with the install absent from the records"
    );
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user asks the CLI to check this machine")]
async fn user_checks_machine(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["doctor"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("it runs the superseded inspection commands")]
async fn run_superseded_commands(world: &mut E2eWorld) {
    let (host, _, host_rc) = crate::run_rocm(world, &["examine"]);
    let (findings, _, findings_rc) = crate::run_rocm(world, &["diagnose"]);
    // Both halves are stashed together so the assertion can check each kept its
    // own shape — the risk being that hiding a command silently reroutes it to
    // the other one's output.
    world.cli_output = Some(format!("--- host ---\n{host}--- findings ---\n{findings}"));
    world.cli_rc = Some(host_rc.max(findings_rc));
}

#[when("the user asks the CLI what it can do")]
async fn user_lists_commands(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["--help"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user asks the CLI to check that saved report")]
async fn user_checks_saved_report(world: &mut E2eWorld) {
    let path = world.model_name.clone().expect("no saved report path");
    let (stdout, stderr, rc) = crate::run_rocm(world, &["doctor", "--from-examination", &path]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("a script runs the superseded host inspection")]
async fn run_superseded_host_inspection(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["examine"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user pipes that saved report into the CLI")]
async fn user_pipes_saved_report(world: &mut E2eWorld) {
    use std::io::Write as _;
    use std::process::Stdio;

    let path = world.model_name.clone().expect("no saved report path");
    let report = std::fs::read(&path).expect("failed to read the captured report");
    let mut cmd = std::process::Command::new(crate::rocm_binary());
    cmd.args(["doctor", "--from-examination", "-"]);
    world.isolate_cmd(&mut cmd);
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn the CLI");
    child
        .stdin
        .take()
        .expect("no stdin on the spawned CLI")
        .write_all(&report)
        .expect("failed to pipe the report in");
    let out = child.wait_with_output().expect("the CLI did not finish");
    world.cli_output = Some(String::from_utf8_lossy(&out.stdout).to_string());
    world.cli_stderr = Some(String::from_utf8_lossy(&out.stderr).to_string());
    world.cli_rc = Some(out.status.code().unwrap_or(-1));
}

#[when("the user asks for both machine-readable reports")]
async fn user_asks_for_both_json_reports(world: &mut E2eWorld) {
    let (newer, stderr, rc) = crate::run_rocm(world, &["doctor", "--json"]);
    assert_eq!(rc, 0, "the newer report should exit 0:\n{stderr}");
    let (superseded, stderr, rc) = crate::run_rocm(world, &["examine", "--json"]);
    assert_eq!(rc, 0, "the superseded report should exit 0:\n{stderr}");
    world.cli_output = Some(newer);
    world.cli_stderr = Some(superseded);
    world.cli_rc = Some(0);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the CLI reports what hardware and ROCm setup it found")]
async fn assert_reports_host_state(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "a health check should exit 0 (it reports whether it RAN, not what it found)"
    );
    let output = world.cli_output.as_ref().expect("no output");
    assert!(
        output.contains(HOST_REPORT_MARKER),
        "expected the host report:\n{output}"
    );
}

#[then("the CLI reports what it makes of them")]
async fn assert_reports_findings(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no output");
    // Reporting state without a verdict is what the two old commands did between
    // them; the whole point of one command is that both arrive together.
    assert!(
        has_findings(output),
        "expected a findings section alongside the host report:\n{output}"
    );
}

#[then("each one still reports what it always did")]
async fn assert_superseded_still_work(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "both should still exit 0");
    let output = world.cli_output.as_ref().expect("no output");
    let (host, findings) = output
        .split_once("--- findings ---")
        .expect("both halves should have been captured");
    assert!(
        host.contains(HOST_REPORT_MARKER),
        "the host inspection lost its report:\n{host}"
    );
    assert!(
        has_findings(findings),
        "the diagnosis lost its findings:\n{findings}"
    );
    // Each kept its own scope: the diagnosis must not have started printing the
    // host report too, which is what a careless alias to the merged command
    // would do.
    assert!(
        !findings.contains(HOST_REPORT_MARKER),
        "the diagnosis should not have gained the host report:\n{findings}"
    );
}

#[then("a single health check is offered")]
async fn assert_one_health_check_offered(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "--help should exit 0");
    let output = world.cli_output.as_ref().expect("no help output");
    assert!(
        output.contains("doctor"),
        "expected the health check to be listed:\n{output}"
    );
}

#[then("the superseded inspection commands are not advertised")]
async fn assert_superseded_not_advertised(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no help output");
    for superseded in ["examine", "diagnose"] {
        assert!(
            !output.contains(superseded),
            "`{superseded}` is still advertised in the command list:\n{output}"
        );
    }
}

#[then("the CLI reports what it makes of the saved report")]
async fn assert_saved_report_findings(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "checking a saved report should exit 0:\n{:?}",
        world.cli_stderr
    );
    let output = world.cli_output.as_ref().expect("no output");
    assert!(
        has_findings(output),
        "expected findings for the saved report:\n{output}"
    );
}

#[then("the CLI does not describe this machine")]
async fn assert_saved_report_omits_local_host(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no output");
    // A saved report may have come from another machine entirely. Printing this
    // machine's inventory next to someone else's findings would read as one
    // coherent picture of a single host, and it would not be one.
    assert!(
        !output.contains(HOST_REPORT_MARKER),
        "a saved report must not be mixed with this machine's inventory:\n{output}"
    );
}

#[then("the install is still missing from the records")]
async fn assert_install_still_missing(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "the health check should still succeed:\n{:?}",
        world.cli_stderr
    );
    assert!(
        !registry_entry(world).exists(),
        "the health check wrote {} — it promises to change nothing",
        registry_entry(world).display()
    );
}

#[then("the install has been added to the records")]
async fn assert_install_now_recorded(world: &mut E2eWorld) {
    // The control for the assertion above: if this fails, the fixture never put
    // the machine in a repairable state and the read-only claim went unproven.
    assert!(
        registry_entry(world).exists(),
        "the fixture is inert — the superseded inspection did not record {} either",
        registry_entry(world).display()
    );
}

#[then("the newer report carries every fact the superseded one did")]
async fn assert_json_superset(world: &mut E2eWorld) {
    let newer: serde_json::Value =
        serde_json::from_str(world.cli_output.as_ref().expect("no newer report"))
            .expect("the newer report is not valid JSON");
    let superseded: serde_json::Value =
        serde_json::from_str(world.cli_stderr.as_ref().expect("no superseded report"))
            .expect("the superseded report is not valid JSON");
    let newer = newer
        .as_object()
        .expect("the newer report is not an object");
    let superseded = superseded
        .as_object()
        .expect("the superseded report is not an object");
    // Keys only: the values describe a live machine and some of them (timings,
    // probe ordering) legitimately differ between two consecutive runs.
    let missing: Vec<&String> = superseded
        .keys()
        .filter(|key| !newer.contains_key(*key))
        .collect();
    assert!(
        missing.is_empty(),
        "the newer report dropped keys tooling already relies on: {missing:?}"
    );
}

#[then("the newer report also carries the findings")]
async fn assert_json_has_findings(world: &mut E2eWorld) {
    let newer: serde_json::Value =
        serde_json::from_str(world.cli_output.as_ref().expect("no newer report"))
            .expect("the newer report is not valid JSON");
    assert!(
        newer.get("findings").is_some(),
        "the newer report carries no findings section"
    );
}
