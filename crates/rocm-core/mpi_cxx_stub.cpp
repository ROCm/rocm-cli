// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Stand-in for the legacy OpenMPI C++ bindings library (`libmpi_cxx.so.40`),
// which OpenMPI 5.x no longer ships.
//
// PyTorch's ROCm wheels were built against OpenMPI 4.x: `libtorch_global_deps.so`
// lists `libmpi_cxx.so.40` as a `NEEDED` dependency and `libtorch_cpu.so`
// references the symbols defined below. Nothing calls them in single-node
// serving, so empty definitions are enough for the dynamic loader to resolve
// `import torch`. `build.rs` compiles this file into the stub that
// `rocm_core::openmpi::ensure_mpi_cxx_compat` writes out at runtime.
//
// The declarations are written as C++ so the compiler produces the mangled
// names; they mirror only the members torch references, not the real classes.
// Adding a symbol here means adding it to the expected list in the
// `embedded_mpi_cxx_stub_exports_exactly_the_symbols_torch_needs` test in
// `src/openmpi.rs`, which says how to regenerate that list.
//
// Built with `-nostdlib -fno-exceptions -fno-rtti`: nothing here may need
// libstdc++ or the C runtime, so the stub has no `NEEDED` entries and loads on
// any Linux host of the same architecture.

namespace MPI {

// `MPI::Win::Free()` — `_ZN3MPI3Win4FreeEv`.
struct Win {
    void Free();
};

// `MPI::Datatype::Free()` — `_ZN3MPI8Datatype4FreeEv`.
struct Datatype {
    void Free();
};

// `MPI::Comm::Comm()` — the complete-object (`_ZN3MPI4CommC1Ev`) and
// base-object (`_ZN3MPI4CommC2Ev`) constructors; torch references both.
struct Comm {
    Comm();
};

void Win::Free() {}
void Datatype::Free() {}
Comm::Comm() {}

}  // namespace MPI

// The C trampolines the OpenMPI 4.x C++ bindings exported for user-defined
// reduction operations; torch references them by their unmangled names.
extern "C" void ompi_mpi_cxx_op_intercept() {}
extern "C" void ompi_op_set_cxx_callback() {}
