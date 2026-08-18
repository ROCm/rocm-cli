// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

fn main() {
    // The generated clap parser for the unified CLI has a large command graph.
    // Windows reserves a much smaller main-thread stack than our Unix targets,
    // and parsing can otherwise overflow before dispatch reaches the selected
    // command. This affects the executable only; test-harness worker threads
    // already receive Rust's independently configured stack size.
    if std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() == Some("windows".as_ref())
        && std::env::var_os("CARGO_CFG_TARGET_ENV").as_deref() == Some("msvc".as_ref())
    {
        println!("cargo:rustc-link-arg-bin=rocm=/STACK:8388608");
    }
}
