// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Spawn, attach to, and stream logs from a managed background service
//! process, plus the endpoint-key/auth checks that gate it.
//!
//! Relocated from `main.rs`, since extended with the port-allocation and
//! launch-lock transaction. This module
//! is the managed-service-spawning tail that was deliberately deferred out of
//! the Phase 6a (ROCMAI-82) split: `start_managed_service`,
//! `run_attached_service`, `spawn_managed_engine_child`, `stream_attached_logs`,
//! and the rest of what was then a single file. It was entangled with crate-root launch helpers
//! — `record_cli_audit_event`, `existing_live_managed_service_in`, and the stop
//! action of the service-record-management cluster (this module reaches only
//! stop, via `AttachOutcome::Stop`; restart/remove/prune are not called here)
//! — and this later pass (ROCMAI-91, Phase 6b) separates the spawning tail out
//! from them; see below for where they live now.
//!
//! `dispatch()`'s call sites are unaffected: `Command::Services { command } =>
//! services(command)` doesn't call anything in this cluster directly.
//! `ServicesCommand` lives in `cli.rs` with the rest of the clap command
//! tree. Since this
//! split, `services()` has moved to `services_cmd.rs`, the record removal and
//! pruning it dispatches to to `service_records.rs`, `record_cli_audit_event`
//! to `cli_audit.rs` (still reached as `crate::record_cli_audit_event` through
//! a re-import in `main.rs`), and `existing_live_managed_service_in` (which the port allocation transaction
//! here uses) to `managed_service_status.rs`. Several functions here are `pub(crate)`
//! because other modules call into this cluster across the module boundary;
//! `main.rs` only declares the module.
//!
//! # Layout
//!
//! This file only declares the submodules and re-exports the items that other
//! modules call, so callers keep reaching them as `crate::service_launch::name`.
//! Items a submodule shares with its siblings are `pub(super)` and reached through
//! `super::<submodule>`; nothing is wider than that unless a module outside this
//! one calls it.
//!
//! - `endpoint_auth`: bind-host validation and the endpoint API key checks.
//! - `launch_transaction`: the locked port allocation decision, the cleanup of a
//!   failed launch, and the reuse refusals.
//! - `spawn`: the detached engine child, its process and stdio plumbing, and
//!   `ManagedLaunchReport`.
//! - `readiness`: waiting on a service, its liveness, and the OOM fault-injection
//!   seam.
//! - `start`: the background launch path and its plain output.
//! - `helper_autostart`: starting the background automation helper.
//! - `attach`: the attached run and its log streaming.

mod attach;
mod endpoint_auth;
mod helper_autostart;
mod launch_transaction;
mod readiness;
mod spawn;
mod start;
#[cfg(test)]
mod test_support;

pub(crate) use attach::run_attached_service;
pub(crate) use endpoint_auth::{
    drop_orphaned_endpoint_key_on_already_running, ensure_public_bind_engine_supported,
    ensure_public_service_has_endpoint_key, is_loopback_host, resolve_endpoint_auth,
    validate_bind_host,
};
pub(crate) use helper_autostart::{
    ensure_background_helper_running, ensure_background_helper_running_quiet,
};
pub(crate) use launch_transaction::{managed_service_reuse_refusal, no_usable_gpu_refusal};
#[cfg(not(windows))]
pub(crate) use readiness::managed_service_stop_marker_present;
pub(crate) use readiness::{
    StopWatch, e2e_simulate_oom_launch, managed_engine_exit_failure_detail,
    managed_service_stop_was_requested,
};
#[cfg(not(windows))]
pub(crate) use spawn::{
    attach_background_stdio, detach_background_command, managed_engine_startup_failure_detail,
    managed_service_process_command,
};
// The helper is platform-independent, but outside tests nothing on Windows
// calls it through this path, so a plain re-export there would be unused. The
// `services_text` log-tail tests call it and are compiled on Windows too.
#[cfg(all(windows, test))]
pub(crate) use spawn::managed_engine_startup_failure_detail;
pub(crate) use start::{
    print_managed_launch_plain, start_managed_service, wait_begin_client_config,
};
