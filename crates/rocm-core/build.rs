// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Build a tiny `libmpi_cxx.so.40` stub that defines the legacy OpenMPI C++
//! binding symbols PyTorch's ROCm wheels reference.
//!
//! OpenMPI 5.x removed the C++ bindings (`libmpi_cxx`), but the ROCm PyTorch
//! wheels were built against OpenMPI 4.x and their `libtorch_cpu.so` carries a
//! handful of undefined `MPI::*` symbols (e.g. `MPI::Datatype::Free()`). A bare
//! symlink to `libmpi.so` satisfies the `libmpi_cxx.so.40` soname but not those
//! symbols, so `import torch` fails with `undefined symbol: _ZN3MPI...`. The
//! symbols are never actually called in single-node serving, so a stub that
//! merely *defines* them lets the dynamic loader resolve torch. The stub is
//! compiled here (target-arch correct, no runtime toolchain needed) and embedded
//! via `include_bytes!`; `rocm_core::openmpi` materializes it at runtime.
//!
//! The source is `mpi_cxx_stub.cpp`: real C++ declarations, so the C++ compiler
//! produces the mangled names rather than this crate spelling them out. It is
//! built with `-nostdlib -fno-exceptions -fno-rtti`, which yields a
//! self-contained object with no `NEEDED` entries (no libstdc++) that loads on
//! any Linux of the same architecture.
//!
//! Without a C++ compiler the stub is left empty and a warning is printed, which
//! disables the shim at runtime. Setting [`REQUIRE_STUB_ENV`] to `1` turns that
//! into a build failure on a Linux target; the CI jobs that build shipped Linux
//! binaries set it, so a release can never silently lose the shim.

use std::env;
use std::fs;
use std::path::Path;

/// Source of the stub, relative to the crate root.
const STUB_SOURCE: &str = "mpi_cxx_stub.cpp";

/// When set to `1`, failing to build the stub on a Linux target fails the build
/// instead of embedding an empty stub.
const REQUIRE_STUB_ENV: &str = "ROCM_CLI_REQUIRE_MPI_CXX_STUB";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={STUB_SOURCE}");
    println!("cargo:rerun-if-env-changed={REQUIRE_STUB_ENV}");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    let stub_path = Path::new(&out_dir).join("libmpi_cxx_stub.so");

    // The stub is only consumed on Linux. Always emit a file so the Linux-only
    // `include_bytes!` always has something to read; an empty file disables the
    // shim gracefully.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        let _ = fs::write(&stub_path, b"");
        return;
    }

    if let Err(reason) = build_stub(&stub_path) {
        assert!(
            env::var(REQUIRE_STUB_ENV).as_deref() != Ok("1"),
            "rocm-core: OpenMPI C++ bindings (libmpi_cxx) stub not built: {reason}; \
{REQUIRE_STUB_ENV}=1 requires it. Install a C++ compiler (g++ or clang++) or set CXX."
        );
        let _ = fs::write(&stub_path, b"");
        println!(
            "cargo:warning=rocm-core: OpenMPI C++ bindings (libmpi_cxx) stub not built: {reason}; \
PyTorch may fail to import on hosts with OpenMPI 5.x. Install a C++ compiler (g++ or clang++) \
to build it."
        );
    }
}

fn build_stub(stub_path: &Path) -> Result<(), String> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").map_err(|error| error.to_string())?;
    let src_path = Path::new(&manifest_dir).join(STUB_SOURCE);
    // A stub left by an earlier build must not pass the output check below.
    let _ = fs::remove_file(stub_path);

    let compiler = cc::Build::new()
        .cpp(true)
        .try_get_compiler()
        .map_err(|error| format!("no C++ compiler: {error}"))?;

    let status = compiler
        .to_command()
        .arg("-shared")
        .arg("-fPIC")
        .arg("-nostdlib")
        .arg("-fno-exceptions")
        .arg("-fno-rtti")
        .arg("-Wl,-soname,libmpi_cxx.so.40")
        .arg("-o")
        .arg(stub_path)
        .arg(&src_path)
        .status()
        .map_err(|error| format!("spawn C++ compiler: {error}"))?;

    if !status.success() {
        return Err(format!("C++ compiler exited with {status}"));
    }
    if !stub_path.exists() {
        return Err("C++ compiler produced no output".to_owned());
    }
    Ok(())
}
