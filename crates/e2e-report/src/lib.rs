// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! HTML/markdown reporting for the cucumber E2E suite.
//!
//! Lives in its own lean crate (only `maud` + `serde`/`serde_json`) so both the
//! `e2e-cucumber` test harness and `xtask` can depend on it without pulling the
//! harness's heavy tree (cucumber/axum/reqwest/tokio) into `xtask`.

mod command_coverage;
mod components;
mod consolidated;
mod parse;
mod single_report;

pub use consolidated::{
    PlatformReport, RunMeta, consolidated_summary_markdown_from_reports,
    generate_consolidated_from_reports, load_platform_reports,
};
// `pub use` of a deprecated item is itself a use — rustc warns here without
// the allow. Callers still get their own deprecation warning at each call
// site; this allow only silences the warning on the re-export itself.
#[allow(deprecated)]
pub use consolidated::{consolidated_summary_markdown, generate_consolidated};
pub use parse::{XfailReport, evaluate_xfail, scenario_results_by_id};
pub use single_report::generate;
