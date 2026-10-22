// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#![allow(clippy::items_after_test_module)]

mod api_docs;
mod cli;
mod common;
mod daemon;
mod engine_ipc;
mod host_snapshot;
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
