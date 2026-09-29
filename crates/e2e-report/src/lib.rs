// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! HTML/markdown reporting for the cucumber E2E suite.
//!
//! Lives in its own lean crate (only `maud` + `serde`/`serde_json`) so both the
//! `e2e-cucumber` test harness and `xtask` can depend on it without pulling the
//! harness's heavy tree (cucumber/axum/reqwest/tokio) into `xtask`.

mod components;
mod consolidated;
mod parse;
mod single_report;

pub use consolidated::{RunMeta, consolidated_summary_markdown, generate_consolidated};
pub use parse::{XfailReport, evaluate_xfail, scenario_results_by_id};
pub use single_report::generate;
