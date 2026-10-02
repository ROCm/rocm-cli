// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Stand-in for the `vllm` executable of a managed runtime, for scenarios on a
//! simulated machine.
//!
//! The CLI launches vLLM as `<runtime>/bin/vllm serve <model> --host H --port P
//! [flags...]` and decides it is ready once `/v1/models` lists the model and a
//! chat completion answers. This serves exactly that — the suite's own mock
//! OpenAI server, bound where the CLI asked — so a scenario can follow `rocm
//! serve` through engine selection, the managed launch and the readiness wait
//! on a machine with no GPU. Every other flag is accepted and ignored, as a
//! recipe may add any of them. It runs until the CLI stops the service.

use e2e_cucumber::mock_server::MockServer;

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("serve") {
        eprintln!("fake-vllm: only `serve <model> --host <h> --port <p>` is simulated");
        std::process::exit(2);
    }
    let Some(model) = args.next() else {
        eprintln!("fake-vllm: serve needs a model");
        std::process::exit(2);
    };
    let (mut host, mut port) = ("127.0.0.1".to_owned(), "8000".to_owned());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => host = args.next().unwrap_or(host),
            "--port" => port = args.next().unwrap_or(port),
            _ => {}
        }
    }
    let listener = match tokio::net::TcpListener::bind(format!("{host}:{port}")).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("fake-vllm: cannot bind {host}:{port}: {error}");
            std::process::exit(1);
        }
    };
    // What a real vLLM prints once up, so the service log reads the same.
    println!("INFO:     Uvicorn running on http://{host}:{port} (fake-vllm serving {model})");
    if let Err(error) = MockServer::serve_until_killed(&model, listener).await {
        eprintln!("fake-vllm: {error}");
        std::process::exit(1);
    }
}
