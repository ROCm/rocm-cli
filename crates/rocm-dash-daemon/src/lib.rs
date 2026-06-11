//! rocm-dash-daemon — daemon library.
//!
//! Runs on the GPU host and serves NDJSON snapshot/bench streams to `rocm`
//! TUI clients over a Unix socket. The composition-root binary (`rocm`)
//! drives this library via its `serve` subcommand.

#![allow(dead_code)]

pub mod bench_ring;
pub mod demo;
pub mod persist;
pub mod runner;
pub mod server;
pub mod snapshot_ring;
pub mod transport;

/// Daemon crate version, surfaced in the `Welcome` handshake.
pub const DAEMON_VERSION: &str = env!("CARGO_PKG_VERSION");

pub use runner::RunnerOptions;
pub use server::run;
