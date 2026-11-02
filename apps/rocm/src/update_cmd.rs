// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm update`'s own rendering and activation logic: the read-only status
//! report (`render_update_text`, `append_update_surfaces`), the predicate
//! deciding whether `update` should route into the mutating path instead of
//! that report (`update_should_preview_or_apply`), applying a runtime update
//! (`apply_runtime_update`, `append_update_activate_summary`), and the
//! update-source/installed-runtime pickers behind it
//! (`select_runtime_update_source`, `select_installed_update_runtime`).
//!
//! Mechanically relocated from `main.rs` with no behavior change — Phase 6f
//! (ROCMAI-210) of the `main.rs` modularization (see `docs/architecture.md`).
//! `dispatch()` calls directly into this module at three points in the
//! `update` command arm — `update_should_preview_or_apply(apply, dry_run)`,
//! `apply_runtime_update(...)`, and `render_update_text(&paths)` — all
//! re-imported via `use crate::update_cmd::{...};`; `dispatch()`'s own body
//! is otherwise unaffected. `chat.rs`'s deterministic `rocm update` tool
//! summary also calls `render_update_text` the same way, outside `dispatch()`.
//!
//! One item physically interleaved in this span in `main.rs` stayed there
//! instead of moving: `const ROLLBACK_RECOVERY_HINT` sits between
//! `apply_runtime_update` and `append_update_activate_summary`, but it
//! already had callers outside this cluster — `runtimes_cmd.rs`'s
//! `RuntimesCommand::Activate` handler and `runtime_install_finalize.rs`'s
//! `render_sdk_install_success` both reach it unqualified via
//! `use crate::{...}` — so it stays a shared crate-root constant, reached
//! from here the same way via `use crate::ROLLBACK_RECOVERY_HINT;`.
//!
//! `render_update_text`, `apply_runtime_update`, and
//! `update_should_preview_or_apply` are `pub(crate)` for `dispatch()`'s own
//! calls above. `select_runtime_update_source` is also `pub(crate)`: it is
//! reached from `therock_version_proptests.rs`'s property tests via
//! `crate::update_cmd::select_runtime_update_source`. `append_update_surfaces`,
//! `append_update_activate_summary`, and `select_installed_update_runtime`
//! have no callers outside this module and stay private.
//!
//! The cluster's regression tests live in this module's own `mod tests`,
//! reaching shared fixtures (`test_paths`, `test_runtime_manifest_for_update`)
//! through `crate::tests::` and `write_test_pip_runtime` through
//! `crate::test_support::`, the same way other mechanically relocated modules
//! do.

use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ModelRecipeRegistrySource, RocmCliConfig, load_model_recipe_registry};

use crate::runtimes_cmd::{
    RuntimeActivationResult, activate_runtime, current_runtime_manifest, select_runtime_manifest,
};
use crate::{
    ROLLBACK_RECOVERY_HINT, engine_inventory, runtime_services, sanitize_log_value, therock,
};

pub(crate) fn render_update_text(paths: &AppPaths) -> Result<String> {
    let mut output = therock::render_update_report(paths)?;
    append_update_surfaces(&mut output);
    Ok(output)
}

fn append_update_surfaces(output: &mut String) {
    let _ = writeln!(output, "  update_surfaces:");
    let _ = writeln!(
        output,
        "    cli: installed={} status=not_configured reason=repository-owned CLI update feed is not published yet",
        env!("CARGO_PKG_VERSION")
    );
    let engine_ids = engine_inventory()
        .iter()
        .map(|(engine, _)| *engine)
        .collect::<Vec<_>>()
        .join(",");
    let _ = writeln!(
        output,
        "    engines: status=package_managed packaged=[{engine_ids}] reason=first-party engine binaries update with the rocm-cli package; data-dir plugins are user-managed"
    );
    match load_model_recipe_registry() {
        Ok(registry) => match registry.source {
            ModelRecipeRegistrySource::BuiltIn => {
                let _ = writeln!(
                    output,
                    "    model_recipes: status=built_in count={} reason=external signed recipe index is not configured",
                    registry.recipes.len()
                );
            }
            ModelRecipeRegistrySource::SignedIndex {
                index_path,
                signature_path,
                public_key_path,
            } => {
                let _ = writeln!(
                    output,
                    "    model_recipes: status=signed_index count={} index={} signature={} public_key={} reason=loaded signed recipe index; recipe update feed is not live in this build",
                    registry.recipes.len(),
                    index_path.display(),
                    signature_path.display(),
                    public_key_path.display()
                );
            }
        },
        Err(error) => {
            let _ = writeln!(
                output,
                "    model_recipes: status=error reason={}",
                sanitize_log_value(&error.to_string())
            );
        }
    }
    let runtime_status = if output.contains("  managed runtimes: none") {
        "none_configured"
    } else {
        "checked_above"
    };
    let _ = writeln!(
        output,
        "    runtimes: status={runtime_status} reason=TheRock runtime update checks above are the only live update checks in this build"
    );
    let _ = writeln!(
        output,
        "  note: `rocm update --apply` applies runtime updates only; CLI, engine, and recipe update feeds require published metadata before they can mutate state"
    );
}

/// Whether `rocm update` should route into the runtime update path
/// (`apply_runtime_update`) instead of the read-only status report.
///
/// `--dry-run` alone must take this path too, since `apply_runtime_update`
/// only mutates anything when `dry_run` is false — a plain status report
/// would silently ignore `--dry-run` and never show what `--apply` would do.
pub(crate) const fn update_should_preview_or_apply(apply: bool, dry_run: bool) -> bool {
    apply || dry_run
}

pub(crate) fn apply_runtime_update(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    runtime_selector: Option<&str>,
    activate: bool,
    dry_run: bool,
) -> Result<String> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let source = select_runtime_update_source(&manifests, config, runtime_selector)?;
    let plan = therock::runtime_update_plan(paths, source, &manifests, None)?;
    let mut output = String::new();
    let _ = writeln!(output, "runtime update");
    let _ = writeln!(output, "  source_runtime_key: {}", source.runtime_key);
    let _ = writeln!(output, "  source_runtime_id: {}", source.runtime_id);
    let _ = writeln!(output, "  channel: {}", source.channel);
    let _ = writeln!(output, "  format: {}", source.format);
    let _ = writeln!(output, "  family: {}", source.family);
    let _ = writeln!(
        output,
        "  installed_version: {}",
        therock::runtime_version_display(&source.version)
    );
    let _ = writeln!(
        output,
        "  latest_version: {}",
        therock::runtime_version_display(&plan.latest_version)
    );
    let _ = writeln!(output, "  status: {}", plan.status);
    let _ = writeln!(output, "  target_runtime_key: {}", plan.target_runtime_key);
    let _ = writeln!(output, "  activate_after_install: {activate}");
    if !plan.update_available {
        let _ = writeln!(output, "  result: no newer runtime found");
        return Ok(output);
    }

    if dry_run {
        let _ = writeln!(output, "  mode: dry-run");
        let install_plan = therock::install_sdk_for_update(
            paths,
            &source.channel,
            &source.format,
            &source.family,
            plan.device_target.as_deref(),
            plan.source_layout_generation.as_deref(),
            source.includes_devel(),
            true,
            activate,
        )?;
        let _ = writeln!(output, "  install_plan:");
        for line in install_plan.output.lines() {
            let _ = writeln!(output, "    {line}");
        }
        return Ok(output);
    }

    // `activate` rather than a bare `true`: the update path is preapproved either
    // way (its approval comes from the runtime the user selected, not from a
    // flag, and `rocm update` has no terminal contract), but the approval line it
    // prints must not promise an activation that only `--activate` performs
    // below.
    let install_output = therock::install_sdk_for_update(
        paths,
        &source.channel,
        &source.format,
        &source.family,
        plan.device_target.as_deref(),
        plan.source_layout_generation.as_deref(),
        // Reinstall what the user originally chose rather than silently
        // adding or dropping the compiler toolchain on update.
        source.includes_devel(),
        false,
        activate,
    )?;
    let manifests_after = therock::load_runtime_manifests(paths)?;
    // By exact key, never by version: a same-version repair installs a sibling
    // that shares the source's channel, format, family AND version, so a
    // version match would just as happily return the stale runtime this update
    // was meant to replace, and then activate it.
    let installed = select_installed_update_runtime(&manifests_after, &plan.target_runtime_key)
        .with_context(|| {
            format!(
                "runtime install completed but no manifest was written for the planned runtime key `{}`",
                plan.target_runtime_key
            )
        })?;
    let _ = writeln!(output, "  installed_runtime_key: {}", installed.runtime_key);
    let _ = writeln!(
        output,
        "  installed_runtime_root: {}",
        installed.install_root.display()
    );
    if activate {
        // No `config.save` after this: `activate_runtime` already persisted the
        // same struct and nothing mutates it in between, so a second write can
        // only fail after an activation that fully succeeded — sending the user
        // to re-run a command that already worked.
        let activation = activate_runtime(paths, config, &installed.runtime_key)?;
        append_update_activate_summary(&mut output, &activation);
    } else {
        let _ = writeln!(
            output,
            "  next step: rocm runtimes activate {}",
            installed.runtime_key
        );
    }
    let _ = writeln!(output, "  install_output:");
    for line in install_output.output.lines() {
        let _ = writeln!(output, "    {line}");
    }
    Ok(output)
}

/// Appends the `--activate` branch of the `update --apply` summary. The
/// rollback hint is only shown when a previous runtime was actually recorded —
/// `rocm runtimes rollback` hard-errors otherwise (no prior runtime to return
/// to), so hinting it unconditionally would point users at a command that
/// immediately fails on their very first activation.
fn append_update_activate_summary(output: &mut String, activation: &RuntimeActivationResult) {
    let _ = writeln!(
        output,
        "  activated_runtime_key: {}",
        activation.runtime_key
    );
    let _ = writeln!(
        output,
        "  previous_runtime_key: {}",
        activation
            .previous_runtime_key
            .as_deref()
            .unwrap_or("<unset>")
    );
    let _ = write!(
        output,
        "{}",
        runtime_services::render_runtime_service_reconciliation(
            &activation.services,
            &activation.runtime_key
        )
    );
    if activation.previous_runtime_key.is_some() {
        let _ = writeln!(output, "{ROLLBACK_RECOVERY_HINT}");
    }
}

pub(crate) fn select_runtime_update_source<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    config: &RocmCliConfig,
    runtime_selector: Option<&str>,
) -> Result<&'a therock::InstalledRuntimeManifest> {
    if let Some(selector) = runtime_selector {
        return select_runtime_manifest(manifests, selector);
    }
    if let Some(active) = current_runtime_manifest(config, manifests) {
        return Ok(active);
    }
    match manifests {
        [] => bail!(
            "no managed runtimes are registered; run `rocm install sdk --channel release --format wheel` first"
        ),
        [only] => Ok(only),
        _ => bail!(
            "multiple runtimes are registered and no active runtime is configured; pass `--runtime <runtime-key>`"
        ),
    }
}

fn select_installed_update_runtime<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    target_runtime_key: &str,
) -> Option<&'a therock::InstalledRuntimeManifest> {
    manifests
        .iter()
        .find(|manifest| manifest.runtime_key == target_runtime_key)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use anyhow::Result;
    use rocm_core::RocmCliConfig;

    use super::*;
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::{test_paths, test_runtime_manifest_for_update};

    #[test]
    fn render_update_text_reports_all_update_surfaces() -> Result<()> {
        let (root, paths) = test_paths("update-surfaces");

        let rendered = render_update_text(&paths)?;
        fs::remove_dir_all(root).ok();

        assert!(rendered.contains("update_surfaces:"));
        assert!(rendered.contains("cli: installed="));
        assert!(rendered.contains("status=not_configured"));
        assert!(rendered.contains("engines: status=package_managed"));
        assert!(rendered.contains("model_recipes: status="));
        assert!(rendered.contains("runtimes: status=none_configured"));
        assert!(rendered.contains("`rocm update --apply` applies runtime updates only"));
        Ok(())
    }

    // This only pins the free predicate's truth table. The actual dispatch
    // wiring — that `rocm update --dry-run` really does reach the preview
    // path without requiring --apply — is covered by e2e scenario
    // `update-dry-run-reaches-preview-path-without-apply`
    // (tests/e2e-cucumber/features/update.feature).
    #[test]
    fn update_should_preview_or_apply_includes_dry_run() {
        assert!(
            !update_should_preview_or_apply(false, false),
            "plain `rocm update` should stay on the read-only status report"
        );
        assert!(
            update_should_preview_or_apply(false, true),
            "the predicate must say dry-run alone should preview"
        );
        assert!(update_should_preview_or_apply(true, false));
        assert!(update_should_preview_or_apply(true, true));
    }

    #[test]
    fn runtime_update_source_uses_active_runtime_and_requires_selector_when_ambiguous() -> Result<()>
    {
        let (root, paths) = test_paths("runtime-update-source");
        let first = write_test_pip_runtime(
            &paths,
            "release-pip-gfx110x-all",
            "therock-release:gfx110X-all",
            "7.13.0a20260416",
            1,
        )?;
        let second = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            2,
        )?;
        let manifests = therock::load_runtime_manifests(&paths)?;
        let active_config = RocmCliConfig {
            default_runtime_id: Some(second.runtime_id.clone()),
            active_runtime_key: Some(second.runtime_key.clone()),
            ..RocmCliConfig::default()
        };

        let selected = select_runtime_update_source(&manifests, &active_config, None)?;
        assert_eq!(selected.runtime_key, second.runtime_key);

        let explicit = select_runtime_update_source(
            &manifests,
            &RocmCliConfig::default(),
            Some(&first.runtime_key),
        )?;
        assert_eq!(explicit.runtime_key, first.runtime_key);

        let error = select_runtime_update_source(&manifests, &RocmCliConfig::default(), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("multiple runtimes"));
        assert!(error.contains("--runtime <runtime-key>"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn installed_update_runtime_is_selected_by_exact_target_key() {
        // Everything a version match would have keyed on is identical here:
        // same channel, format, family and version. Only the composition-keyed
        // runtime key tells the freshly installed repair apart from the stale
        // runtime it was installed to replace.
        let stale = test_runtime_manifest_for_update(
            "release-wheel-multi-arch-7-14-0",
            "therock-release:gfx120X-all",
            "gfx120X-all",
            "7.14.0",
        );
        let wrong_family = test_runtime_manifest_for_update(
            "release-wheel-multi-arch-7-14-0-ffffffffffffffff",
            "therock-release:gfx110X-all",
            "gfx110X-all",
            "7.14.0",
        );
        let repaired = test_runtime_manifest_for_update(
            "release-wheel-multi-arch-7-14-0-0123456789abcdef",
            "therock-release:gfx120X-all",
            "gfx120X-all",
            "7.14.0",
        );
        let manifests = vec![stale, wrong_family, repaired.clone()];

        let selected = select_installed_update_runtime(&manifests, &repaired.runtime_key)
            .expect("the side-by-side repair must be selected by its exact key");
        assert_eq!(selected.runtime_key, repaired.runtime_key);

        assert!(
            select_installed_update_runtime(&manifests, "release-wheel-multi-arch-7-15-0")
                .is_none(),
            "an install that wrote no manifest for the planned key must not resolve to a sibling"
        );
    }

    #[test]
    fn update_activate_summary_hints_rollback_only_when_a_previous_runtime_exists() {
        let with_previous = RuntimeActivationResult {
            runtime_id: "therock-release:gfx942".to_owned(),
            runtime_key: "release-wheel-gfx942".to_owned(),
            previous_runtime_key: Some("release-wheel-gfx942-old".to_owned()),
            reactivated: false,
            // No live services: this test is about the rollback hint, and an
            // empty reconciliation is what an update on a machine with nothing
            // serving produces.
            services: runtime_services::RuntimeServiceReconciliation::default(),
        };
        let mut rendered = String::new();
        append_update_activate_summary(&mut rendered, &with_previous);
        assert!(
            rendered.contains("services_on_previous_runtime: 0"),
            "the update summary reports service state the same way `runtimes \
             activate` does, including when there is none:\n{rendered}"
        );
        assert!(
            rendered.contains("next step: if this causes problems, run `rocm runtimes rollback`"),
            "a previous runtime is recorded, so rollback is a valid recovery path:\n{rendered}"
        );
        assert!(
            rendered.contains("undoes only this one activation"),
            "the hint should state the single-step limit up front, not just in --help:\n{rendered}"
        );

        let without_previous = RuntimeActivationResult {
            runtime_id: "therock-release:gfx942".to_owned(),
            runtime_key: "release-wheel-gfx942".to_owned(),
            previous_runtime_key: None,
            reactivated: false,
            services: runtime_services::RuntimeServiceReconciliation::default(),
        };
        let mut rendered = String::new();
        append_update_activate_summary(&mut rendered, &without_previous);
        assert!(
            !rendered.contains("rocm runtimes rollback"),
            "no previous runtime is recorded, so `rocm runtimes rollback` would hard-error; \
             must not hint at a command that immediately fails:\n{rendered}"
        );
    }
}
