// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `lemonade_gufo.feature`: serving Lemonade's Gufo models
//! (ROCMAI-1241).
//!
//! The refusal and engine-list scenarios plant a Lemonade install in the layout
//! the engine writes (install record, server binary, model registry), so
//! `rocm serve` finds Lemonade current and installs nothing. What Lemonade says
//! about its Gufo backend comes from a file named by
//! `ROCM_E2E_LEMONADE_SYSTEM_INFO`, which the `rocm/e2e-test-hooks` build reads
//! in place of a running `lemond`'s `/api/v1/system-info`.

use std::path::PathBuf;

use cucumber::{given, then, when};
use serde_json::json;

use crate::E2eWorld;

/// A model Lemonade's registry assigns to the Gufo recipe.
const GUFO_MODEL: &str = "Qwen3.8-27B-Gufo";
/// A second Gufo model, so the engine list is shown to name all of them.
const OTHER_GUFO_MODEL: &str = "DeepSeek-V4-Flash-IQ2XXS-Gufo";
/// A llama.cpp model in the same registry, which the engine list must not call
/// a Gufo model.
const LLAMACPP_MODEL: &str = "Qwen3-0.6B-GGUF";
const SYSTEM_INFO_ENV: &str = "ROCM_E2E_LEMONADE_SYSTEM_INFO";

/// What Lemonade 2026.47.1 prints for `gufo:rocm` on a host without crun and
/// without `render`, one problem and one step per line.
const SETUP_PROBLEMS: &str =
    "Podman is set to run containers with runc\nThis account is not in the render group";
const SETUP_STEPS: &str = "sudo apt install crun\nsudo usermod -aG render $USER";

fn data_dir(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join("data")
}

/// Where Lemonade installs when no ROCm runtime is active.
fn lemonade_root(world: &E2eWorld) -> PathBuf {
    data_dir(world).join("engines").join("lemonade")
}

const fn lemond_name() -> &'static str {
    if cfg!(windows) {
        "lemond.exe"
    } else {
        "lemond"
    }
}

fn combined_output(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or_default(),
        world.cli_stderr.as_deref().unwrap_or_default()
    )
}

#[given("Lemonade is installed and lists Gufo models")]
async fn plant_lemonade_with_gufo_models(world: &mut E2eWorld) {
    let root = lemonade_root(world);
    let runtime_dir = root.join("runtime");
    let resources = runtime_dir.join("resources");
    std::fs::create_dir_all(&resources).expect("failed to create the Lemonade tree");
    // Never run on the refusal paths: the Gufo answer comes from the scripted
    // system-info, and the refusal comes before any server would start. Every
    // download of a model or of the Gufo image goes through one of these two, so
    // on Unix they record any run in `calls.txt` (and fail), which is what the
    // refusal's "not downloaded" claim is checked against.
    plant_recorder(&runtime_dir.join(lemond_name()));
    plant_recorder(&runtime_dir.join("lemonade"));
    let registry = json!({
        GUFO_MODEL: {
            "checkpoints": {"main": "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q4_K_XL.gguf"},
            "recipe": "gufo",
            "suggested": true,
            "size": 19.6
        },
        OTHER_GUFO_MODEL: {"recipe": "gufo", "suggested": true, "size": 92.7},
        LLAMACPP_MODEL: {"checkpoint": "unsloth/Qwen3-0.6B-GGUF:Q4_0", "recipe": "llamacpp"}
    });
    std::fs::write(
        resources.join("server_models.json"),
        serde_json::to_vec_pretty(&registry).expect("registry serializes"),
    )
    .expect("failed to write the model registry");
    // The env id `rocm serve` checks to decide the pinned Lemonade is already
    // installed, so it prepares nothing.
    let version = rocm_deps::LEMONADE_VERSION;
    let record = root.join("manifests").join("runtime.json");
    std::fs::create_dir_all(record.parent().expect("record has a parent"))
        .expect("failed to create the record dir");
    std::fs::write(
        &record,
        serde_json::to_string_pretty(&json!({
            "env_id": format!("lemonade-embeddable-{version}"),
            "version": version,
            "runtime_dir": runtime_dir,
            "lemond": runtime_dir.join(lemond_name()),
            "lemonade": runtime_dir.join("lemonade"),
            "backend_recipe": "llamacpp",
            "backend_name": "rocm",
            "installed_at_unix_ms": 0,
        }))
        .expect("record serializes"),
    )
    .expect("failed to write the install record");
}

/// A stand-in Lemonade executable: on Unix a script that appends its argv to
/// `calls.txt` beside it and fails; elsewhere an empty file, which cannot run.
fn plant_recorder(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::write(
            path,
            "#!/bin/sh\nprintf '%s\\n' \"$0 $*\" >> \"$(dirname \"$0\")/calls.txt\"\nexit 1\n",
        )
        .expect("failed to write a Lemonade stand-in");
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .expect("failed to make a Lemonade stand-in executable");
    }
    #[cfg(not(unix))]
    std::fs::write(path, b"").expect("failed to write a Lemonade stand-in");
}

/// What the planted `lemond` and `lemonade` were run with, one line per run.
fn lemonade_calls(world: &E2eWorld) -> String {
    std::fs::read_to_string(lemonade_root(world).join("runtime").join("calls.txt"))
        .unwrap_or_default()
}

/// Script Lemonade's `/api/v1/system-info` answer for the Gufo backend.
fn script_gufo_state(world: &mut E2eWorld, state: &str, message: &str, action: &str) {
    let path = data_dir(world).join("gufo-system-info.json");
    let system_info = json!({
        "recipes": {
            "llamacpp": {"backends": {"rocm": {"state": "installed"}}},
            "gufo": {"backends": {"rocm": {
                "state": state,
                "message": message,
                "action": action,
                "tier": "guest",
                "format": "container"
            }}}
        }
    });
    std::fs::create_dir_all(path.parent().expect("has a parent"))
        .expect("failed to create the data dir");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&system_info).expect("system info serializes"),
    )
    .expect("failed to write the scripted system info");
    world
        .command_env
        .push((SYSTEM_INFO_ENV, path.into_os_string()));
}

#[given("Lemonade reports that Gufo cannot run on this machine's GPU")]
async fn gufo_unsupported(world: &mut E2eWorld) {
    script_gufo_state(world, "unsupported", "Unsupported GPU", "");
}

#[given("Lemonade reports that Gufo's container setup is not complete")]
async fn gufo_setup_incomplete(world: &mut E2eWorld) {
    script_gufo_state(world, "action_required", SETUP_PROBLEMS, SETUP_STEPS);
}

#[when("the user serves a Gufo model with Lemonade")]
async fn user_serves_gufo_model(world: &mut E2eWorld) {
    serve_gufo_with(world, &[]);
}

/// A serve whose refusal is the scenario's premise, not its subject: the
/// engine list read afterwards shows what that serve recorded.
#[given("the user has tried to serve a Gufo model with Lemonade")]
async fn user_has_tried_to_serve_gufo_model(world: &mut E2eWorld) {
    serve_gufo_with(world, &[]);
    let output = combined_output(world);
    assert_ne!(
        world.cli_rc,
        Some(0),
        "the premise is a refused serve:\n{output}"
    );
    // Refused for the setup, which is what the engine list then reports, and
    // not for some other reason, such as a binary without the hook.
    assert!(
        output.contains("the container setup on this machine is not complete"),
        "the premise is a serve refused for incomplete container setup:\n{output}"
    );
}

#[when("the user serves a Gufo model with Lemonade and a sampling temperature")]
async fn user_serves_gufo_model_with_temperature(world: &mut E2eWorld) {
    serve_gufo_with(world, &["--temperature", "0.5"]);
}

fn serve_gufo_with(world: &mut E2eWorld, extra: &[&str]) {
    let mut args = vec!["serve", GUFO_MODEL, "--engine", "lemonade", "--managed"];
    args.extend_from_slice(extra);
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(world, &args);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user lists the engines")]
async fn user_lists_engines(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(world, &["engines", "list"]);
    assert_eq!(rc, 0, "`rocm engines list` failed:\n{stdout}\n{stderr}");
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("serving the Gufo model is refused before anything is started")]
async fn assert_refused_before_start(world: &mut E2eWorld) {
    let output = combined_output(world);
    assert_ne!(
        world.cli_rc,
        Some(0),
        "serving a Gufo model Lemonade cannot run must fail:\n{output}"
    );
    // A binary without the hook ignores the scripted answer and tries to start
    // the planted `lemond`; name that cause rather than a baffling mismatch
    // below.
    assert!(
        !output.contains("gufo-probe-lemond.log") && !output.contains("failed to start"),
        "the binary under test probably lacks the `rocm/e2e-test-hooks` feature, which \
         `cargo xtask e2e` enables:\n{output}"
    );
    // Nothing was launched: the engine writes a service's state file before it
    // starts the server, so an empty state dir means no service was started.
    let state_dir = lemonade_root(world).join("state");
    let started: Vec<_> = std::fs::read_dir(&state_dir)
        .map(|entries| entries.flatten().map(|entry| entry.path()).collect())
        .unwrap_or_default();
    assert!(
        started.is_empty(),
        "a Lemonade service was started despite the refusal: {started:?}\n{output}"
    );
    // Nothing was downloaded: every model and image download is a `lemonade`
    // or `lemond` run, and neither ran. (Unix only: elsewhere the stand-ins
    // cannot record, and could not have run either.)
    let calls = lemonade_calls(world);
    assert!(
        calls.is_empty(),
        "Lemonade was run despite the refusal:\n{calls}\n{output}"
    );
    // And no managed service is recorded for it.
    let (services, services_stderr, services_rc) = crate::run_rocm(world, &["services", "list"]);
    assert_eq!(
        services_rc, 0,
        "`rocm services list` failed, so it shows nothing:\n{services}\n{services_stderr}"
    );
    assert!(
        !services.contains(GUFO_MODEL),
        "a service for the refused model is listed:\n{services}"
    );
}

#[then("the user is told Gufo runs only on a Strix Halo GPU")]
async fn assert_told_strix_halo_only(world: &mut E2eWorld) {
    let output = combined_output(world);
    for expected in [
        "Gufo models run only on an AMD Strix Halo GPU (gfx1151) under Linux",
        "(Lemonade: Unsupported GPU)",
        "The model and the Gufo image were not downloaded",
        "the model is not run on any other backend or on the CPU",
    ] {
        assert!(output.contains(expected), "missing `{expected}`:\n{output}");
    }
}

#[then("the user is told Gufo does not take the sampling flags")]
async fn assert_told_flags_not_taken(world: &mut E2eWorld) {
    let output = combined_output(world);
    for expected in [
        format!(
            "`{GUFO_MODEL}` is a Gufo model, and Gufo does not take the llama-server arguments"
        ),
        "--temperature 0.5".to_owned(),
        "nothing was started".to_owned(),
    ] {
        assert!(
            output.contains(&expected),
            "missing `{expected}`:\n{output}"
        );
    }
    // Refused for the flags, before Lemonade's answer about the GPU counted.
    assert!(
        !output.contains("Gufo models run only on an AMD Strix Halo GPU"),
        "refused for the GPU, not the flags:\n{output}"
    );
}

#[then("the user is shown the setup steps Lemonade reports")]
async fn assert_shown_setup_steps(world: &mut E2eWorld) {
    let output = combined_output(world);
    let indented = |text: &str| text.lines().collect::<Vec<_>>().join("\n  ");
    for expected in [
        "The model and the Gufo image were not downloaded, Gufo was not started".to_owned(),
        "the model is not run on the CPU instead".to_owned(),
        format!("Lemonade reports:\n  {}", indented(SETUP_PROBLEMS)),
        format!("To finish the setup:\n  {}", indented(SETUP_STEPS)),
        "Then run the same `rocm serve` command again.".to_owned(),
    ] {
        assert!(
            output.contains(&expected),
            "missing `{expected}`:\n{output}"
        );
    }
}

/// The engine list's line about Gufo.
fn gufo_line(world: &E2eWorld) -> String {
    let output = world
        .cli_output
        .as_deref()
        .expect("no engine list captured");
    output
        .lines()
        .find(|line| line.contains("Lemonade Gufo backend"))
        .unwrap_or_else(|| panic!("the engine list has no Gufo line:\n{output}"))
        .to_owned()
}

#[then("the engine list says Gufo needs container setup before it can run")]
async fn assert_engine_list_says_setup_needed(world: &mut E2eWorld) {
    let line = gufo_line(world);
    assert!(
        line.contains("needs container setup before it can run"),
        "{line}"
    );
    assert!(
        line.contains(SETUP_PROBLEMS.lines().next().unwrap()),
        "{line}"
    );
}

#[then("the engine list names the Gufo models")]
async fn assert_engine_list_names_models(world: &mut E2eWorld) {
    let line = gufo_line(world);
    assert!(
        line.ends_with(&format!("Gufo models: {OTHER_GUFO_MODEL}, {GUFO_MODEL}")),
        "{line}"
    );
    assert!(!line.contains(LLAMACPP_MODEL), "{line}");
}

#[given("a Gufo model is being served on lemonade")]
async fn serve_gufo_model(world: &mut E2eWorld) {
    let stdout = crate::e2e::serving_steps::serve_and_wait(
        world,
        &["serve", GUFO_MODEL, "--engine", "lemonade", "--managed"],
        GUFO_MODEL,
        GUFO_MODEL,
    )
    .await;
    world.cli_output = Some(stdout);
}

#[then("the serve log says the model is served by Gufo")]
async fn assert_log_names_gufo(world: &mut E2eWorld) {
    let stdout = world
        .cli_output
        .as_deref()
        .expect("no serve output captured");
    // The whole log: the notice is written once, at ready, and the inference
    // request's lines after it can push it out of a tail.
    let path = e2e_cucumber::serve_log::parse_log_path(stdout)
        .unwrap_or_else(|| panic!("no log_path in the serve output:\n{stdout}"));
    // Lossily, as `serve_log` reads logs: a progress bar or a cut multibyte
    // sequence must not fail the read.
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("cannot read the serve log {path}: {error}"));
    let log = String::from_utf8_lossy(&bytes);
    assert!(
        log.contains(&format!(
            "Lemonade is serving {GUFO_MODEL} with Gufo (gufo:rocm)"
        )),
        "the serve log does not say Gufo serves the model:\n{log}"
    );
}
