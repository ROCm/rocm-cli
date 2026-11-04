// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm services`: the command handler (`services`), which dispatches the
//! `list`, `logs`, `stop`, `restart`, `remove` and `prune` subcommands, and
//! `run_approved_service_action`, the approval-gated stop/restart path.
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. `ServicesCommand`, the clap enum it matches on,
//! stays in `main.rs` with the rest of the `Command`/`Cli` definition.
//! `main.rs`'s `dispatch()` calls `services` through the re-import
//! `use crate::services_cmd::services;`, and `advised_commands.rs`'s test
//! reaches `run_approved_service_action` as `super::run_approved_service_action`
//! through a test-only re-import in `main.rs`. Both are `pub(crate)` for those
//! callers.
//!
//! This module reads from `services_text.rs` (the status, logs and action-result
//! text and the sandbox tool call), `service_records.rs` (remove and prune) and
//! `service_hints.rs` (the advice text), and records its audit events through
//! `cli_audit.rs`; none of those modules calls back into this one.
//!
//! The cluster's one regression test lives in this module's own `mod tests`,
//! reaching the shared `test_paths` fixture through `crate::tests::`.

use anyhow::{Result, bail};
use rocm_core::AppPaths;

use crate::ServicesCommand;
use crate::cli_audit::record_cli_audit_event;
use crate::service_hints::{
    self, sandbox_tool_arg_from_service_tool, service_action_command, service_action_past_tense,
    service_action_verb,
};
use crate::service_records::{
    prune_managed_service_records, remove_managed_service_record, service_prune_min_age_hours,
};
use crate::services_text::{
    render_service_action_result_with_helper_note, render_service_logs_text, render_services_json,
    render_services_text, run_internal_sandbox_tool, validate_service_id,
};

pub(crate) fn services(command: Option<ServicesCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    match command.unwrap_or(ServicesCommand::List {
        all: false,
        json: false,
    }) {
        ServicesCommand::List { all, json } => {
            if json {
                print!("{}", render_services_json(&paths, all)?);
            } else {
                print!("{}", render_services_text(&paths, all)?);
            }
            Ok(())
        }
        ServicesCommand::Logs { service_id } => {
            print!("{}", render_service_logs_text(&paths, &service_id)?);
            Ok(())
        }
        ServicesCommand::Stop { service_id, yes } => {
            run_approved_service_action(&paths, "stop_server", &service_id, yes)
        }
        ServicesCommand::Restart { service_id, yes } => {
            run_approved_service_action(&paths, "restart_server", &service_id, yes)
        }
        ServicesCommand::Remove { service_id, yes } => {
            print!(
                "{}",
                remove_managed_service_record(&paths, &service_id, yes)?
            );
            record_cli_audit_event(
                &paths,
                "service",
                "remove_record",
                "info",
                format!("removed local server record {service_id}"),
                Some(&service_id),
            );
            Ok(())
        }
        ServicesCommand::Prune {
            older_than_hours,
            any_age,
            dry_run,
            yes,
        } => {
            let older_than_hours = service_prune_min_age_hours(older_than_hours, any_age);
            let outcome = prune_managed_service_records(&paths, older_than_hours, dry_run, yes)?;
            print!("{}", outcome.text);
            if !dry_run && (outcome.removed_records > 0 || outcome.removed_files > 0) {
                record_cli_audit_event(
                    &paths,
                    "service",
                    "prune_records",
                    "info",
                    format!(
                        "removed {} local server record(s) and {} leftover file(s) older than {older_than_hours}h",
                        outcome.removed_records, outcome.removed_files
                    ),
                    None,
                );
            }
            // Deliberately after the print and the audit event: a file this run
            // could not delete still has to fail the command, but not at the
            // cost of the record of what it *did* delete.
            if !outcome.failures.is_empty() {
                // Only reachable through this dispatch, so no unit test covers
                // it: `service-cleanup-06` in
                // `features/service_record_cleanup.feature` is its cover.
                bail!(
                    "{} file(s) could not be removed; see the list above",
                    outcome.failures.len()
                );
            }
            Ok(())
        }
    }
}

pub(crate) fn run_approved_service_action(
    paths: &AppPaths,
    tool: &str,
    service_id: &str,
    yes: bool,
) -> Result<()> {
    validate_service_id(service_id)?;
    if !yes {
        bail!(
            "{} local server `{service_id}` requires --yes.\n\nTry: {}",
            service_action_verb(tool),
            service_hints::service_action_hint(service_action_command(tool), service_id)
        );
    }
    let sandbox_tool = sandbox_tool_arg_from_service_tool(tool)?;
    let result = run_internal_sandbox_tool(paths, sandbox_tool, Some(service_id.to_owned()), true)?;
    print!(
        "{}",
        render_service_action_result_with_helper_note(paths, tool, &result)
    );
    record_cli_audit_event(
        paths,
        "service",
        tool,
        "info",
        format!(
            "{} managed service {service_id}",
            service_action_past_tense(tool)
        ),
        Some(service_id),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_paths;

    #[test]
    fn service_actions_require_yes_and_render_sandbox_result() {
        let (_root, paths) = test_paths("services-action-approval");
        let error = run_approved_service_action(&paths, "stop_server", "svc-qwen", false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires --yes"));
        assert!(error.contains("rocm services stop svc-qwen --yes"));

        let rendered = crate::services_text::render_service_action_result(
            "stop_server",
            &serde_json::json!({
                "output": {
                    "status": "stopped",
                    "result": {
                        "service": {
                            "service_id": "svc-qwen",
                            "status": "stopped",
                            "endpoint_url": "http://127.0.0.1:11435/v1"
                        },
                        "signaled_pids": [1234, 5678]
                    }
                }
            }),
        );

        assert!(rendered.contains("Local server stopped"));
        assert!(rendered.contains("service: svc-qwen"));
        assert!(rendered.contains("status: stopped"));
        assert!(rendered.contains("stopped processes: 2"));
    }
}
