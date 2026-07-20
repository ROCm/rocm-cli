// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Standalone demo environment for VHS screencasts.
//!
//! Starts the same axum mock OpenAI server the cucumber e2e suite uses, plants a
//! managed-service record into an isolated config root, and prints the env vars
//! that point `rocm` at it. `cargo xtask demos` parses this output and exports
//! it before VHS runs, so `rocm chat` / `rocm services list` hit the mock — no
//! GPU, no real model, fully deterministic. Runs until signalled; xtask kills it
//! when rendering finishes.
//!
//! Usage: `rocm-demo-env --root <dir> [--model <id>]`

use std::io::Write as _;
use std::path::PathBuf;

use e2e_cucumber::mock_server::{MockServer, write_service_record};

#[tokio::main]
async fn main() {
    let mut root: Option<PathBuf> = None;
    let mut model = "Qwen/Qwen3.5-4B".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => root = args.next().map(PathBuf::from),
            "--model" => {
                if let Some(m) = args.next() {
                    model = m;
                }
            }
            other => {
                eprintln!("rocm-demo-env: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let Some(root) = root else {
        eprintln!("rocm-demo-env: --root <dir> is required");
        std::process::exit(2);
    };

    for sub in ["config", "data", "cache"] {
        std::fs::create_dir_all(root.join(sub)).expect("failed to create isolated dir");
    }

    let mock = MockServer::start(&model).await;
    let port = mock.port();
    write_service_record(&root.join("data").join("services"), &model, port);

    // Emit the env the tape sources; the CLI reads these to find the isolated
    // config and the planted service. The trailing marker line lets the caller
    // block until the server is actually ready before running commands.
    let mut out = std::io::stdout().lock();
    let cfg = root.join("config");
    let data = root.join("data");
    let cache = root.join("cache");
    writeln!(out, "export ROCM_CLI_CONFIG_DIR={}", cfg.display()).unwrap();
    writeln!(out, "export ROCM_CLI_DATA_DIR={}", data.display()).unwrap();
    writeln!(out, "export ROCM_CLI_CACHE_DIR={}", cache.display()).unwrap();
    writeln!(
        out,
        "# rocm-demo-env ready on 127.0.0.1:{port} (model {model})"
    )
    .unwrap();
    out.flush().unwrap();
    drop(out);

    // Keep the process (and the spawned server task) alive until told to stop.
    // `mock` must stay in scope: dropping it shuts the server down.
    wait_for_shutdown().await;
    mock.stop();
}

/// Block until a stop signal. On Unix, ignore SIGINT/SIGHUP — VHS/ttyd emit some
/// during terminal setup, and catching the default disposition would kill the
/// server before the demo command runs — and exit gracefully on SIGTERM (useful
/// when this binary is run standalone and stopped with `kill`). When driven by
/// `cargo xtask demos`, the parent instead hard-kills this process (SIGKILL) once
/// rendering finishes, so this handler need not run in that path. Elsewhere, fall
/// back to Ctrl-C.
#[cfg(unix)]
async fn wait_for_shutdown() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    let mut sighup = signal(SignalKind::hangup()).expect("install SIGHUP handler");
    loop {
        tokio::select! {
            _ = sigterm.recv() => break,
            _ = sigint.recv() => {}
            _ = sighup.recv() => {}
        }
    }
}

#[cfg(not(unix))]
async fn wait_for_shutdown() {
    tokio::signal::ctrl_c().await.ok();
}
