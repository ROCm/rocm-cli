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
    let (paths, sources) = AppPaths::discover_with_sources()?;
    let plan = build_uninstall_plan(&paths, &sources, &options)?;
    print!("{}", render_uninstall_plan(&plan, &options));
    run_uninstall_plan(&plan, &options, interactive_terminal(), confirm_uninstall)
}

/// Everything after the review is printed: refuse, preview, confirm, remove.
///
/// Whether the terminal is interactive and how to ask are passed in, so tests
/// can prove the refusal comes before the prompt rather than only before the
/// first removal.
pub(crate) fn run_uninstall_plan(
    plan: &UninstallPlan,
    options: &UninstallOptions,
    interactive: bool,
    confirm: impl FnOnce() -> Result<bool>,
) -> Result<()> {
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
        if !interactive {
            bail!("uninstall requires --yes outside an interactive terminal");
        }
        if !confirm()? {
            println!("uninstall cancelled");
            return Ok(());
        }
    }

    apply_uninstall_plan(plan)?;
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
        // Check, then act on what was checked. The review judged where a real
        // directory resolved to when it was made; a parent swapped for a
        // symlink while the prompt was open would point the same spelling at
        // a different folder. Resolve again, refuse if it moved, and remove
        // the resolved location rather than re-walking the spelling.
        let target = match &entry.resolved {
            None => entry.path.clone(),
            Some(planned) => match rocm_core::canonicalize_for_compare(&entry.path) {
                Ok(now) if &now == planned => now,
                Ok(now) => bail!(
                    "stopped before removing the {} folder {}: it now resolves to {}, not {} as \
                     the review showed. Anything reported as removed above is gone; nothing \
                     after it was touched. Check the folder and run the uninstall again.",
                    entry.kind,
                    entry.path.display(),
                    now.display(),
                    planned.display()
                ),
                // Already gone, e.g. inside a folder removed earlier in the plan.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "failed to resolve {} before removing it",
                            entry.path.display()
                        )
                    });
                }
            },
        };
        remove_path(&target)
            .with_context(|| format!("failed to remove {}", entry.path.display()))?;
        println!("removed {} {}", entry.kind, entry.path.display());
    }
    Ok(())
}
