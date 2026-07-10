// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

pub mod mock_server;

// The report generator lives in its own lean crate (only maud + serde) so xtask
// can reuse it without pulling this harness's heavy tree. Re-export it under the
// original path so `e2e_cucumber::report::{generate, evaluate_xfail}` call sites
// keep working.
pub use e2e_report as report;
