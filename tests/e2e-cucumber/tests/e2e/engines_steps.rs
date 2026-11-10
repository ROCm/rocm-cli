// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm engines shell`, and for the `rocm engines list` entry of a
//! vLLM install whose compiled extensions do not import (plain piped runs).
//!
//! The shell steps run under a pseudo-terminal on purpose. The defect they guard — a shell
//! that is correctly activated but visually identical to the one it was launched
//! from — exists only in what the terminal renders, so a piped run cannot see it.

use std::path::PathBuf;
use std::time::Duration;

use cucumber::{given, then, when};

use crate::E2eWorld;
use crate::e2e::tui_driver::TuiSession;

/// Env id for the planted engine environment.
const ENV_ID: &str = "e2e-shell";

/// The engine to open a shell for. Any supported engine exercises the same path;
/// vLLM is the one the behaviour was reported against.
const ENGINE: &str = "vllm";

const SCREEN_TIMEOUT: Duration = Duration::from_secs(30);

fn planted_env_path(world: &E2eWorld) -> PathBuf {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    root.path().join("engine-env")
}

#[given("a machine with an installed engine environment")]
async fn plant_engine_env(world: &mut E2eWorld) {
    // Plant the manifest `rocm engines install` would have written, plus a stub
    // interpreter, so the scenario reaches the shell-launch path without a
    // multi-GiB engine install or a GPU. Black-box: plain JSON matching the
    // CLI's on-disk schema, not a typed import from the product crates.
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let env_path = planted_env_path(world);
    let bin = env_path.join("bin");
    std::fs::create_dir_all(&bin).expect("failed to create the planted engine env");

    let python = bin.join("python");
    rocm_core::fake_executable::write_fake_executable(
        &python,
        "#!/bin/sh\necho planted-engine-python\n",
    )
    .expect("failed to write the planted interpreter");

    let manifests = root
        .path()
        .join("data")
        .join("engines")
        .join(ENGINE)
        .join("manifests");
    std::fs::create_dir_all(&manifests).expect("failed to create the engine manifests dir");
    let manifest = serde_json::json!({
        "env_id": ENV_ID,
        "runtime_id": "therock-release:gfx94X-dcgpu",
        "python_executable": python.display().to_string(),
        "env_path": env_path.display().to_string(),
    });
    std::fs::write(
        manifests.join(format!("{ENV_ID}.json")),
        serde_json::to_vec_pretty(&manifest).expect("manifest serialises"),
    )
    .expect("failed to write the engine env manifest");
}

#[when("the user opens a shell for that engine")]
async fn open_engine_shell(world: &mut E2eWorld) {
    // `--shell` is pinned rather than inherited: the runner's own $SHELL decides
    // whether a prompt marker is possible at all, which would make this scenario
    // pass or fail for reasons that have nothing to do with the product.
    let session = TuiSession::spawn(
        world,
        &[
            "engines",
            "shell",
            ENGINE,
            "--env-id",
            ENV_ID,
            "--shell",
            "/bin/bash",
        ],
    )
    .unwrap_or_else(|e| panic!("failed to open the engine shell: {e}"));
    world.tui = Some(session);
}

#[then("the shell is visibly marked as that engine's shell")]
async fn assert_prompt_marked(world: &mut E2eWorld) {
    let session = world.tui.as_mut().expect("no engine shell session");
    let marker = format!("(rocm:{ENGINE})");
    // The marker must be on the rendered prompt line, which is what the user
    // actually sees — not merely somewhere in the informational block above it.
    // That block names the marker too ("your prompt is now prefixed
    // (rocm:vllm)") and is printed before bash starts, so waiting for the bare
    // marker returns before any prompt exists; wait for the prompt line itself.
    // `$` for a normal user's prompt, `#` for root's -- this suite runs as root
    // on the WSL2 hardware lane, so both are genuine prompt lines.
    session
        .wait_for_screen_where(
            "the marker is on a prompt line",
            |screen| {
                screen
                    .lines()
                    .filter(|line| line.contains(&marker))
                    .any(|line| line.contains('$') || line.contains('#'))
            },
            SCREEN_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| panic!("the marker never reached a prompt line: {e}"));
}

#[then("the engine environment's interpreter is the one that runs")]
async fn assert_engine_interpreter(world: &mut E2eWorld) {
    // Guards against the prompt shim breaking the activation it decorates: the
    // shim re-runs the user's startup files, which could otherwise reorder PATH.
    let expected = planted_env_path(world).join("bin").join("python");
    let session = world.tui.as_mut().expect("no engine shell session");
    session
        .send("command -v python\n")
        .unwrap_or_else(|e| panic!("failed to query the interpreter: {e}"));
    session
        .wait_for_screen(&expected.display().to_string(), SCREEN_TIMEOUT)
        .await
        .unwrap_or_else(|e| {
            panic!(
                "the engine env interpreter did not win on PATH (wanted {}): {e}",
                expected.display()
            )
        });
}

#[when("the user leaves the engine shell")]
async fn leave_engine_shell(world: &mut E2eWorld) {
    let session = world.tui.as_mut().expect("no engine shell session");
    session
        .send("exit\n")
        .unwrap_or_else(|e| panic!("failed to leave the engine shell: {e}"));
    session
        .wait_for_exit(SCREEN_TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("the engine shell did not exit cleanly: {e}"));
}

#[then("the engine shell exits successfully")]
async fn assert_engine_shell_exited(world: &mut E2eWorld) {
    // `wait_for_exit` already reaped the child and asserted a zero status, so
    // reaching here means the whole open → interact → leave round trip worked.
    assert!(
        world.tui.is_some(),
        "no engine shell session was opened for this scenario"
    );
}

/// A vLLM install whose compiled extensions cannot be imported.
///
/// The stand-in interpreter answers every probe the way a wheel built against a
/// different torch does: `vllm._C` fails with an undefined `c10` symbol. A
/// `vllm` launcher beside it is what makes the runtime resolve as installed.
/// Black-box: the CLI finds it through `ROCM_CLI_VLLM_PYTHON`, exactly as a user
/// pointing the CLI at their own vLLM environment would.
#[given("a vLLM install whose compiled extensions do not import")]
async fn plant_vllm_with_broken_extensions(world: &mut E2eWorld) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let bin = root.path().join("broken-vllm-env").join("bin");
    std::fs::create_dir_all(&bin).expect("failed to create the stand-in vLLM env");

    let python = bin.join("python");
    rocm_core::fake_executable::write_fake_executable(
        &python,
        "#!/bin/sh\n\
         echo '{\"failures\": [{\"module\": \"vllm._C\", \"error\": \"undefined symbol: c10_hip_boom\"}]}'\n",
    )
    .expect("failed to write the stand-in interpreter");
    rocm_core::fake_executable::write_fake_executable(&bin.join("vllm"), "#!/bin/sh\nexit 0\n")
        .expect("failed to write the stand-in vllm launcher");

    world
        .command_env
        .push(("ROCM_CLI_VLLM_PYTHON", python.into_os_string()));
}

#[when("the user lists the engines of that install")]
async fn list_engines_of_that_install(world: &mut E2eWorld) {
    let (stdout, _, rc) = crate::run_rocm_with_scenario_env(world, &["engines", "list"]);
    assert_eq!(rc, 0, "`rocm engines list` failed:\n{stdout}");
    world.cli_output = Some(stdout);
}

/// The `rocm engines list` block for one engine: its heading line and the
/// indented lines under it, up to the next engine. Other engines print their own
/// `runtime:` line (AIM reads `ready` wherever a Docker daemon is reachable), so
/// the assertions below must not read the whole output.
fn engine_block(output: &str, engine: &str) -> String {
    let mut block = Vec::new();
    let mut inside = false;
    for line in output.lines() {
        // An engine's heading sits at two columns (`* ` marks the default one);
        // its `adapter:`/`runtime:`/`note:` lines are indented under it.
        let heading = line.replacen("* ", "  ", 1);
        let indent = heading.len() - heading.trim_start().len();
        if !heading.trim().is_empty() && indent <= 2 {
            inside = heading.trim_start().starts_with(engine);
        }
        if inside {
            block.push(line);
        }
    }
    block.join("\n")
}

#[then("vLLM is reported as installed but not usable")]
async fn vllm_reported_not_usable(world: &mut E2eWorld) {
    let out = world
        .cli_output
        .as_deref()
        .expect("no engines output recorded");
    let vllm = engine_block(out, "vllm");
    assert!(
        vllm.contains("runtime: installed, not usable"),
        "the broken vLLM was not reported as unusable:\n{out}"
    );
    assert!(
        !vllm.contains("runtime: ready"),
        "a runtime whose extensions do not import must not read as ready:\n{out}"
    );
}

#[then("the report names the extension that failed to import")]
async fn report_names_failing_extension(world: &mut E2eWorld) {
    let out = world
        .cli_output
        .as_deref()
        .expect("no engines output recorded");
    assert!(
        engine_block(out, "vllm").contains("vllm._C"),
        "the vLLM entry does not name the failing extension:\n{out}"
    );
}
