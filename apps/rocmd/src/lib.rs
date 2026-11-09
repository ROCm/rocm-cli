// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#![allow(clippy::items_after_test_module)]

pub mod anthropic_proxy;
mod api_docs;
mod artifact_prefetch;
mod cli;
mod common;
mod daemon;
mod daemon_host;
mod engine_ipc;
mod host_snapshot;
mod mcp;
mod mcp_argv;
mod persistence;
mod sandbox;
mod sandbox_tool;
mod service;
mod supervise;
#[cfg(test)]
mod test_support;
mod watchers;
mod webhook;

pub use api_docs::{
    ApiDocsServer, DEFAULT_API_DOCS_PORT, ServiceLoader, documented_route_paths,
    start_api_docs_server,
};
pub use cli::{parse_args, run_bin_cli, run_from_args};
