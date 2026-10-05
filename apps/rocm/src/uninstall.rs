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
//! [`apply_uninstall_plan`] lives here: it is the one step that deletes, and it
//! refuses a plan with a protected root on its own.

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, interactive_terminal};

use crate::{
    UninstallOptions, UninstallPlan, build_uninstall_plan, confirm_uninstall, remove_path,
    render_uninstall_plan,
};

pub(crate) fn uninstall(options: UninstallOptions) -> Result<()> {
    let paths = AppPaths::discover()?;
    let plan = build_uninstall_plan(&paths, &options)?;
    print!("{}", render_uninstall_plan(&plan, &options));

    // A refused root stops the whole command, before the prompt and before the
    // first removal: removing the rest and then failing would leave a partial
    // uninstall behind a folder that was never going to be removed. A dry run
    // fails too, so `rocm uninstall --dry-run && rocm uninstall --yes` cannot
    // read a refusal as a go-ahead; its review above already carries the advice.
    if let Some(error) = plan.refusal_error() {
        if options.dry_run {
            bail!("uninstall would be refused; see the review above");
        }
        bail!(error);
    }

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

    apply_uninstall_plan(&plan)?;
    println!("uninstall complete");
    Ok(())
}

/// Remove everything `plan` lists, reporting each removal as it happens.
///
/// Refuses again on its own, so no caller can reach a removal with a protected
/// root in the plan, whatever it checked first.
pub(crate) fn apply_uninstall_plan(plan: &UninstallPlan) -> Result<()> {
    if let Some(error) = plan.refusal_error() {
        bail!(error);
    }
    for entry in &plan.actions {
        remove_path(&entry.path)
            .with_context(|| format!("failed to remove {}", entry.path.display()))?;
        println!("removed {} {}", entry.kind, entry.path.display());
    }
    Ok(())
}
