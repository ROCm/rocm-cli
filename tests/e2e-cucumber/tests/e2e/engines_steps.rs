// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm engines shell`.
//!
//! These run under a pseudo-terminal on purpose. The defect they guard — a shell
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

/// Plant an environment tree with an interpreter and an activation script, plus a
/// manifest naming them, so a scenario reaches the shell-launch path with no
/// multi-GiB engine install and no GPU. Black-box: plain JSON matching the CLI's on-disk schema, not a
/// typed import from the product crates.
///
/// `recorded_env_path` is the env_path the shell is handed, which is NOT always
/// the environment root: the vLLM install RESPONSE reports the directory holding
/// the engine command (`command.parent()` in `engines/vllm/src/install.rs`),
/// while Lemonade reports the root. No install writes this manifest to disk; it
/// stands in for that response so the shell is reachable without an install.
/// Callers choose the shape they need.
fn plant_engine_env_recorded_as(world: &E2eWorld, recorded_env_path: &std::path::Path) {
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
    // A real environment ships the activation script next to its interpreter.
    // Planting it is what makes "the hint names a file that exists" a statement
    // about the CLI rather than about how thorough this fixture was.
    std::fs::write(bin.join("activate"), "# planted engine env activation\n")
        .expect("failed to write the planted activation script");

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
        "env_path": recorded_env_path.display().to_string(),
    });
    std::fs::write(
        manifests.join(format!("{ENV_ID}.json")),
        serde_json::to_vec_pretty(&manifest).expect("manifest serialises"),
    )
    .expect("failed to write the engine env manifest");
}

#[given("a machine with an installed engine environment")]
async fn plant_engine_env(world: &mut E2eWorld) {
    let env_path = planted_env_path(world);
    plant_engine_env_recorded_as(world, &env_path);
}

#[given("a machine with an engine environment installed the way the engine records it")]
async fn plant_engine_env_as_installed(world: &mut E2eWorld) {
    // Same tree, but env_path is what the vLLM install response reports: the
    // directory the engine command lives in, one level inside the environment.
    let bin = planted_env_path(world).join("bin");
    plant_engine_env_recorded_as(world, &bin);
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

#[then("the printed activation hint names a file that exists")]
async fn assert_activation_hint_resolves(world: &mut E2eWorld) {
    const HINT_LABEL: &str = "activate_hint:";

    let session = world.tui.as_mut().expect("no engine shell session");
    session
        .wait_for_screen(HINT_LABEL, SCREEN_TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("the engine shell printed no activation hint: {e}"));

    // Read LOGICAL lines: the hint carries a path long enough to soft-wrap at the
    // terminal width, and a truncated path would fail the assertion below for a
    // reason that has nothing to do with the product.
    let lines = session.screen_logical_lines();
    let screen = session.screen_text();
    let hint = lines
        .iter()
        .find_map(|line| line.split_once(HINT_LABEL))
        .map_or_else(
            || panic!("no activation hint on screen:\n{screen}"),
            |(_, rest)| rest.trim(),
        );
    // `runtime_python_activation_hint` prints `source <path>` on Unix.
    let path = hint.strip_prefix("source ").unwrap_or(hint).trim();
    assert!(
        !path.is_empty(),
        "the activation hint named no path:\n{screen}"
    );
    assert!(
        std::path::Path::new(path).exists(),
        "the engine shell told the user to activate {path:?}, which does not exist:\n{screen}"
    );
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
