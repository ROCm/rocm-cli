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
    UninstallOptions, build_uninstall_plan, confirm_uninstall, remove_path, render_uninstall_plan,
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

    // Stop managed servers before removing the binaries and service records that
    // stop them. Uninstall used to report success while a publicly-bound,
    // GPU-holding endpoint kept serving, then delete the tooling needed to stop
    // it (EAI-8014). If any cannot be confirmed stopped, the gate aborts without
    // removing anything so the recovery tooling stays in place.
    let stop_report = stop_managed_services_before_uninstall(&paths)?;
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
