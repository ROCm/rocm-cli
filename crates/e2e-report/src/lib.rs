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

/// Environment variable signalling the `e2e-oom-fault-injection` hook is present.
///
/// `cargo xtask e2e` sets it to `1` when it built the binary under test with the
/// `rocm/e2e-oom-fault-injection` feature, so the `e2e-cucumber` harness can tell
/// whether `@requires-oom-fault-injection` scenarios can run against it.
///
/// The single source of truth for this xtask ↔ harness contract. It lives in
/// this lean crate — the one both `xtask` and `e2e-cucumber` already depend on —
/// so the producer (`xtask::e2e`) and the consumer
/// (`e2e_cucumber::capability`) reference the same literal and cannot drift: a
/// typo previously would not fail to compile or fail a test, silently turning
/// `@requires-oom-fault-injection` into a skip (green suite, zero coverage).
pub const OOM_FAULT_INJECTION_ENV: &str = "ROCM_E2E_OOM_FAULT_INJECTION";
