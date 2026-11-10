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
//!
//! What it was launched with is written to [`argv_record_path`], one argument
//! per line, so a scenario can check the flags the CLI passed rather than only
//! that something came up.

use std::path::{Path, PathBuf};

use e2e_cucumber::mock_server::MockServer;

/// Where the stand-in launched as `program` records its arguments: beside the
/// program, as `<program>.argv`. The CLI launches a managed runtime's `vllm` by
/// its absolute path, which is the planted link, so the record lands in the
/// scenario's own runtime directory. `None` for a bare program name, which has
/// no directory of its own to write into.
fn argv_record_path(program: &Path) -> Option<PathBuf> {
    program.is_absolute().then(|| {
        let mut record = program.as_os_str().to_owned();
        record.push(".argv");
        PathBuf::from(record)
    })
}

#[tokio::main]
async fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let program = argv.first().map(PathBuf::from).unwrap_or_default();
    match argv_record_path(&program) {
        Some(record) => {
            if let Err(error) = std::fs::write(&record, argv[1..].join("\n") + "\n") {
                eprintln!(
                    "fake-vllm: cannot record its arguments to {}: {error}",
                    record.display()
                );
                std::process::exit(1);
            }
        }
        None => eprintln!(
            "fake-vllm: launched as {}, not by an absolute path, so its arguments are not recorded",
            program.display()
        ),
    }
    let mut args = argv.into_iter().skip(1);
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

#[cfg(test)]
mod tests {
    use super::argv_record_path;
    use std::path::Path;

    #[test]
    fn records_beside_an_absolute_program_and_not_for_a_bare_name() {
        let root = std::env::temp_dir()
            .join("runtime")
            .join("bin")
            .join("vllm");
        assert_eq!(
            argv_record_path(&root),
            Some(root.with_file_name("vllm.argv"))
        );
        assert_eq!(argv_record_path(Path::new("vllm")), None);
    }
}
