// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#![allow(clippy::items_after_test_module)]

mod cli;
mod common;
mod mcp;
mod persistence;
mod sandbox;
mod sandbox_tool;
mod service;
mod supervise;
#[cfg(test)]
mod test_support;
mod watchers;
mod webhook;

pub use cli::{parse_args, run_bin_cli, run_from_args};

use std::time::Duration;

const WATCHER_TICK_INTERVAL: Duration = Duration::from_secs(5);
