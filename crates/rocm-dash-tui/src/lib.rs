// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! rocm-dash TUI client library.
//!
//! The composition-root binary (`rocm`) is a thin wrapper over `app::run`;
//! the same modules are reachable as a library so examples (e.g. screenshot
//! generation) can drive the UI in-process.

pub mod agent;
pub mod app;
pub mod client;
pub mod jobs;
pub mod llm;
pub mod reconnect;
pub mod replay;
pub mod skills;
pub mod tool_exec;
pub mod transport;
pub mod ui;
