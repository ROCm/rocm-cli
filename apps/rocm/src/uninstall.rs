// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm uninstall` command handler.
//!
//! Mechanically relocated from `main.rs` with no behavior change — the
//! `dispatch()` call site stays `uninstall(UninstallOptions { .. })` (re-imported
//! via `use crate::uninstall::uninstall;`). The `UninstallOptions`/`UninstallPlan`
//! types and the plan/render/remove helpers remain in the crate root and are
//! reached through `crate::` (root items are visible to this descendant module).

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, interactive_terminal};

use crate::{
    ManagedServiceStopReport, UninstallOptions, UninstallPlan, build_uninstall_plan,
    confirm_uninstall, plan_removes_recovery_tooling, remove_path, render_uninstall_plan,
    stop_managed_services_before_uninstall, uninstall_removal_gate,
};

pub(crate) fn uninstall(options: UninstallOptions) -> Result<()> {
    let paths = AppPaths::discover()?;
    let plan = build_uninstall_plan(&paths, &options)?;
    print!("{}", render_uninstall_plan(&plan, &options));

    if plan.actions.is_empty() || options.dry_run {
        return Ok(());
    }

    if !options.yes {
        if !interactive_terminal() {
            bail!("uninstall requires --yes outside an interactive terminal");
        }
        if !confirm_uninstall()? {
            println!("uninstall cancelled");
            return Ok(());
        }
    }

    // Only an uninstall that takes away the means of stopping a server has to
    // stop it first; a cache-only run leaves `rocm services stop` and every
    // service record in place.
    let removes_recovery_tooling = plan_removes_recovery_tooling(&plan, &paths);
    stop_managed_services_then_remove(&plan, || {
        if removes_recovery_tooling {
            stop_managed_services_before_uninstall(&paths)
        } else {
            Ok(ManagedServiceStopReport::default())
        }
    })
}

/// Stop the servers this machine manages, then remove the planned paths — in
/// that order, and only if every stop was confirmed.
///
/// Uninstall used to report success while a publicly-bound, GPU-holding endpoint
/// kept serving, then delete the tooling needed to stop it (EAI-8014). The stop
/// runs first and the gate aborts on any unconfirmed stop, so the recovery
/// tooling stays in place.
///
/// The stop pass is a parameter, not a direct call, so the ordering guarantee is
/// testable rather than merely inspectable: a test can hand in a stop that fails
/// — which no test can reliably provoke from a real process — and assert that
/// not one planned path was removed. Deleting the stop from this function makes
/// those tests fail.
fn stop_managed_services_then_remove(
    plan: &UninstallPlan,
    stop_managed_services: impl FnOnce() -> Result<ManagedServiceStopReport>,
) -> Result<()> {
    let stop_report = stop_managed_services()?;
    if let Some(line) = uninstall_removal_gate(&stop_report)? {
        println!("{line}");
    }

    for entry in &plan.actions {
        remove_path(&entry.path)
            .with_context(|| format!("failed to remove {}", entry.path.display()))?;
        println!("removed {} {}", entry.kind, entry.path.display());
    }
    println!("uninstall complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use anyhow::bail;

    use super::stop_managed_services_then_remove;
    use crate::{
        FailedManagedServiceStop, ManagedServiceStopReport, UninstallPlan, UninstallPlanEntry,
    };

    /// An isolated root and a plan whose one action removes a real file in it.
    ///
    /// The file is what makes these tests assertions about *removal* rather than
    /// about return values: it exists before the call, and its presence
    /// afterwards is the evidence that the abort happened before the removal
    /// loop, not after it.
    fn plan_removing_one_file(name: &str) -> (PathBuf, UninstallPlan) {
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-uninstall-test-{name}-{}-{}",
            std::process::id(),
            rocm_core::unix_time_millis()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("failed to create the test root");
        let doomed = root.join("rocm");
        fs::write(&doomed, b"binary").expect("failed to seed the file the plan removes");
        (
            root,
            UninstallPlan {
                actions: vec![UninstallPlanEntry {
                    kind: "binary",
                    path: doomed,
                }],
                skipped: Vec::new(),
                warnings: Vec::new(),
            },
        )
    }

    #[test]
    fn a_service_that_cannot_be_stopped_leaves_every_planned_path_in_place() {
        // The EAI-8014 guarantee itself: while a managed server may still be
        // serving, uninstall removes NOTHING — not the binaries, not the service
        // records — so `rocm services stop` is still there to recover with.
        let (root, plan) = plan_removing_one_file("stop-unconfirmed");
        let doomed = plan.actions[0].path.clone();

        let error = stop_managed_services_then_remove(&plan, || {
            Ok(ManagedServiceStopReport {
                stopped: Vec::new(),
                failed: vec![FailedManagedServiceStop {
                    service_id: "svc-stuck".to_owned(),
                    reason: "still \"ready\" after the stop attempt".to_owned(),
                }],
            })
        })
        .expect_err("an unconfirmed stop must abort uninstall");

        assert!(
            error.to_string().contains("svc-stuck"),
            "the abort names the service: {error:#}"
        );
        assert!(
            doomed.is_file(),
            "the planned path must survive an aborted uninstall"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_stop_pass_that_cannot_run_leaves_every_planned_path_in_place() {
        // Discovery failing (an unreadable services directory) is the same class
        // of danger as a stop failing: uninstall would be removing the tooling
        // while blind to what it manages.
        let (root, plan) = plan_removing_one_file("stop-undiscoverable");
        let doomed = plan.actions[0].path.clone();

        let error = stop_managed_services_then_remove(&plan, || {
            bail!("failed to read the services directory")
        })
        .expect_err("a stop pass that cannot run must abort uninstall");

        assert!(
            error.to_string().contains("services directory"),
            "the abort carries the discovery failure: {error:#}"
        );
        assert!(
            doomed.is_file(),
            "the planned path must survive an aborted uninstall"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn removal_proceeds_once_every_managed_server_is_confirmed_stopped() {
        let (root, plan) = plan_removing_one_file("stop-confirmed");
        let doomed = plan.actions[0].path.clone();

        stop_managed_services_then_remove(&plan, || {
            Ok(ManagedServiceStopReport {
                stopped: vec!["svc-stopped".to_owned()],
                failed: Vec::new(),
            })
        })
        .expect("a confirmed stop must let uninstall proceed");

        assert!(
            !doomed.exists(),
            "the planned path must be removed once nothing is left serving"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// Linux-only: relies on `process_start_ticks` (`Some` only on Linux) and on
    /// the zombie-state detection in `terminate_verified`.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_live_managed_server_is_stopped_before_the_planned_paths_are_removed() {
        // The whole ordering, driven through the real stop pass rather than an
        // injected report: a live server dies first, and only then do the files
        // go.
        let (root, plan) = plan_removing_one_file("live-server");
        let doomed = plan.actions[0].path.clone();
        let paths = rocm_core::AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        let child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("failed to spawn the managed server");
        let pid = child.id();
        let mut record = rocm_core::ManagedServiceRecord::new(
            &paths,
            "svc-live",
            "vllm",
            "m",
            "m",
            "127.0.0.1",
            9,
            "managed",
            pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(pid);
        record.supervisor_start_ticks = rocm_core::process_start_ticks(pid);
        record.status = "ready".to_owned();
        record.write().expect("failed to write the service record");

        stop_managed_services_then_remove(&plan, || {
            crate::stop_managed_services_before_uninstall(&paths)
        })
        .expect("a server that stops must not block uninstall");

        // Reap our own child so the liveness check does not observe a zombie.
        let mut child = child;
        let _ = child.wait();
        assert!(
            !rocm_core::process_is_running(pid),
            "the managed server must be stopped by uninstall"
        );
        assert!(
            !doomed.exists(),
            "the planned path must be removed once the server is stopped"
        );
        let _ = fs::remove_dir_all(root);
    }
}
