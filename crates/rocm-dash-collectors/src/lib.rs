// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! IO-heavy collector implementations. Used by `rocm-dash-daemon`.
//!
//! Every collector implements one of the traits in `rocm_dash_core::traits`.
//! Stubs return `CollectorError::Unsupported` so the daemon can start with nothing wired.

pub mod amd_smi;
pub mod bench_tail;
pub mod cgroup;
pub mod docker;
pub mod engine_registry;
pub mod host;
pub mod lemonade;
pub mod llama_slots;
pub mod parallel;
pub mod proc_scan;
pub mod sysfs;
pub mod vllm_log;
pub mod vllm_prom;
