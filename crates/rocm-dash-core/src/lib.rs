// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! rocm-dash-core
//!
//! Pure types, traits, schemas, and the reducer for rocm-dash.
//! No rendering deps. No async deps at the type boundary.
//!
//! See `../wiki/concepts/tea-reducer-pattern.md` for the architectural pattern.

pub mod bench_rollup;
pub mod bench_schema;
pub mod config;
pub mod efficiency;
pub mod metrics;
pub mod partition;
pub mod persist;
pub mod protocol;
pub mod state;
pub mod traits;
pub mod vram;

pub use bench_rollup::{PassNRollup, rollup_pass_n, row_verdict};
pub use bench_schema::BenchmarkRow;
pub use metrics::{GpuMetrics, Instance, Snapshot, SystemMetrics};
pub use partition::{ComputePartitionMode, MemoryPartitionMode};
pub use protocol::{Command, Event};
pub use state::{SideEffect, State, StateEvent};
