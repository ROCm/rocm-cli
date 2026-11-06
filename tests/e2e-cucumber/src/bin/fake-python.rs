// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A stand-in for the runtime's Python interpreter, for the ComfyUI install
//! scenarios' two probes.
//!
//! The ComfyUI install shells out to the runtime's Python twice. The first call
//! is `-c <script>`, the pre-install torch-stack version probe in
//! `apps/rocm/src/comfyui/install.rs`; this binary answers it with empty JSON,
//! so the probe reads "nothing pinned". The second call is
//! `<script-path> <result-path>`, the post-install GPU check in
//! `apps/rocm/src/comfyui.rs`; this binary answers it by writing a fixed,
//! GPU-having result to the given path. One binary answers
//! both forms unconditionally, so a scenario that only reaches the first
//! probe and one that reaches both work against the same stand-in.
//!
//! A Rust binary rather than a shell script so the scenarios run on Windows
//! too — a `.bat` stand-in can't take the `-c` form at all: its argument is a
//! genuine multi-line script, and Rust's CVE-2024-24576 mitigation rejects
//! any newline-containing argument once the target resolves to a batch file.
//!
//! Only the two forms `probe_comfyui` and the torch-stack probe actually use
//! are accepted; anything else fails loudly rather than guessing, the same
//! rule `fake-tailscale` follows for an unexpected command.

const GPU_PROBE_JSON: &str = r#"{"torch_version": "2.4.0", "torch_cuda_available": true, "device_count": 1, "devices": ["Fake GPU"]}"#;

// The probe script path is always `probe_comfyui`'s own internally-generated
// `probe.py` (never user input), so a case-sensitive extension check is fine.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        // `python -c <script>`: the pre-install torch-stack version probe.
        Some("-c") if args.len() == 3 => {
            print!("{{}}");
        }
        // `python <probe.py> <result.json>`: the post-install GPU check.
        Some(script) if script.ends_with(".py") && args.len() == 3 => {
            let result_path = &args[2];
            std::fs::write(result_path, GPU_PROBE_JSON)
                .unwrap_or_else(|e| panic!("failed to write GPU probe result: {e}"));
        }
        _ => {
            eprintln!("fake-python: unexpected invocation: {args:?}");
            std::process::exit(1);
        }
    }
}
