// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `rocm-engine-vllm stdio` binary, end to end: what it reads on stdin goes
//! through the shared request reader, so a request it cannot serve, or input
//! that is not a request, is answered with an error envelope on stdout and a
//! clean exit.

use std::io::Write as _;
use std::process::{Command, Stdio};

/// Run `rocm-engine-vllm stdio` with `request` on stdin, on scratch app roots,
/// and return its exit status and the error code of the envelope it printed.
fn answer(label: &str, request: &str) -> (Option<i32>, String) {
    let root = std::env::temp_dir().join(format!(
        "rocm-engine-vllm-stdio-{label}-{}",
        std::process::id()
    ));
    let mut child = Command::new(env!("CARGO_BIN_EXE_rocm-engine-vllm"))
        .arg("stdio")
        .env("ROCM_CLI_CONFIG_DIR", root.join("config"))
        .env("ROCM_CLI_DATA_DIR", root.join("data"))
        .env("ROCM_CLI_CACHE_DIR", root.join("cache"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the engine binary");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(request.as_bytes())
        .expect("write the request");
    let output = child
        .wait_with_output()
        .expect("wait for the engine binary");
    let _ = std::fs::remove_dir_all(&root);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let envelope: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|error| {
        panic!(
            "{label}: no envelope on stdout ({error}): {stdout:?}; stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    let code = envelope["error"]["code"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    (output.status.code(), code)
}

#[test]
fn the_stdio_binary_answers_requests_it_cannot_serve() {
    assert_eq!(
        answer(
            "incompatible",
            r#"{"protocol_version":"9.0.0","method":"frobnicate","payload":{}}"#
        ),
        (Some(0), "incompatible_protocol".to_owned())
    );
    assert_eq!(
        answer("garbage", "this is not a request"),
        (Some(0), "invalid_request".to_owned())
    );
}
