// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! IO-heavy collector implementations. Used by `rocm-dash-daemon` and, on Windows, `rocm-dash-tui`.
//!
//! Every collector implements one of the traits in `rocm_dash_core::traits`.

pub mod amd_smi;
pub mod bench_load;
pub mod bench_tail;
pub mod cgroup;
pub mod docker;
pub mod engine_registry;
pub mod host;
pub mod lemonade;
pub mod parallel;
pub mod vllm_prom;
pub mod winperf;
