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
    std::fs::write(&python, "#!/bin/sh\necho planted-engine-python\n")
        .expect("failed to write the planted interpreter");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755))
            .expect("failed to mark the planted interpreter executable");
    }

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
    session
        .wait_for_screen(&marker, SCREEN_TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("engine shell prompt was not marked: {e}"));

    let screen = session.screen_text();
    let on_a_prompt_line = screen
        .lines()
        .filter(|line| line.contains(&marker))
        .any(|line| line.contains('$'));
    assert!(
        on_a_prompt_line,
        "the marker never reached a prompt line:\n{screen}"
    );
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
