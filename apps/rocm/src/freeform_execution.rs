// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The freeform natural-language planner's plan-*execution*/consent cluster
//! (ROCMAI-726, a Phase 6e, ROCMAI-87 follow-up).
//!
//! Mechanically relocated from `main.rs` with no behavior change — `main.rs`'s
//! `execute_freeform_next_action()` call sites stay byte-identical
//! (`prepare_freeform_execution(...)`/`render_freeform_execution_header(...)`,
//! re-imported via
//! `use crate::freeform_execution::{prepare_freeform_execution, render_freeform_execution_header};`).
//! `execute_freeform_next_action()` and `run_freeform()` stay in `main.rs`:
//! the first is the dispatch fn that hands this cluster's output to
//! `Cli::try_parse_from`/`dispatch()`, and the second is the top-level
//! `rocm <request>` entry point that decides whether to call it at all — both
//! dispatch-level, not part of deciding whether a plan's next action is safe
//! to execute.
//!
//! This is the other half of the freeform-planner cluster that
//! `freeform_planner.rs` (Phase 6e, ROCMAI-87) deliberately left behind: that
//! module builds a plan; this one reads/acts on its `FreeformPlanAction`
//! internals, grants it the consent the outer `--yes` already carries, and
//! computes planner-specific output text — a dispatch-adjacent concern
//! distinct from building the plan (see `docs/architecture.md`'s
//! `freeform_execution.rs` entry). Follows the same mechanical-relocation,
//! dispatch-adjacent pattern as `driver_install.rs`: large enough to own its
//! own private type (`FreeformExecution`) rather than leaving it in `main.rs`.
//!
//! `FreeformExecution` stays `pub(crate)`, as it already was — both `main.rs`
//! and this module's own tests read it. `prepare_freeform_execution` and
//! `render_freeform_execution_header` are widened from module-private to
//! `pub(crate)` here, since `main.rs`'s `execute_freeform_next_action` now
//! reaches them across the module boundary; `validate_freeform_execution_action`
//! and `apply_freeform_execution_consent` have no caller outside this module
//! (including its own tests) and stay module-private, unchanged from before
//! the move. `prepare_freeform_execution` takes the already-built `plan` by
//! value rather than `request`/`paths`/`config`, matching ROCMAI-725's
//! build-once fix in `main.rs`: the caller already built the plan (and
//! rendered it to the operator), and a second build here could call out to a
//! configured cloud `planner_provider` again, which is not guaranteed
//! deterministic.

use std::fmt::Write as _;

use anyhow::{Context, Result, bail};

use crate::chat;
use crate::freeform_planner;

/// The argv `execute_freeform_next_action` is about to dispatch, plus whether
/// [`apply_freeform_execution_consent`] added a flag to it.
///
/// The flag is tracked rather than re-detected from `action.args` because the
/// execution header uses it to tell the operator *why* its `tool_call:` differs
/// from the one in the request plan above. Looking for the flag in the final
/// argv would report "added here" for a plan that already carried it.
#[derive(Debug)]
pub(crate) struct FreeformExecution {
    pub action: freeform_planner::FreeformPlanAction,
    pub consent_added: bool,
}

/// Everything `execute_freeform_next_action` decides before it hands the argv to
/// clap: derive the next action from the already-built `plan`, refuse what
/// must not run unattended, and grant the consent the outer `--yes` already
/// carries.
///
/// Split out so the consent injection is reachable from a test without
/// dispatching a real install — the header render and the `dispatch` call are
/// all that is left above it. Takes the plan by value rather than rebuilding
/// it from `request`/`paths`/`config`: the caller already built it once (and
/// already rendered it to the operator), and a second build is exactly the
/// bug this signature closes — see `execute_freeform_next_action`'s doc
/// comment.
pub(crate) fn prepare_freeform_execution(
    plan: freeform_planner::StructuredRequestPlan,
) -> Result<FreeformExecution> {
    let mut action = freeform_planner::plan_next_action(plan)
        .context("natural-language plan did not produce a structured tool call")?;
    validate_freeform_execution_action(&action)?;
    let consent_added = apply_freeform_execution_consent(&mut action.args);
    Ok(FreeformExecution {
        action,
        consent_added,
    })
}

/// Grant the generated tool call the consent the operator already gave on the
/// outer command line.
///
/// Only ever reached from `run_freeform` with `approve` set, i.e. from
/// `rocm --yes <natural-language request>`. The planner builds a bare
/// `install sdk ...` argv and `execute_freeform_next_action` re-parses and
/// dispatches it **in process**, so `install()` would otherwise run with
/// `yes = false, approve_replacing_active_default = false` no matter what the
/// outer invocation said. That made this surface print `approval: granted by
/// --yes` and then, with an active default runtime, refuse with "re-run with
/// `--approve-replacing-active-default`" — a flag this surface offers no way to
/// pass — or prompt on a terminal it had just said it did not need to ask.
///
/// Not `--yes`, for the same reason every other internal caller picks the narrow
/// flag: `--yes` on `install sdk` carries a second, unrelated consent for
/// system-package installs that run `sudo`, and the outer `--yes` here means
/// "execute the plan you were just shown", not "install system packages".
///
/// Injected before the execution header renders, so the printed `tool_call:` is
/// the argv that actually runs, and skipped under `--dry-run`, which returns
/// before the gate and needs no consent — matching the dry-run-aware arms in
/// `chat_rocm_command_action_from_args`, `rocmd` and dash-tui. The plan
/// rendering path is deliberately untouched: `rocm <request>` without `--yes`
/// prints a command for a human to review, and it must not hand them a
/// pre-approved one.
///
/// That leaves `rocm --yes <request>` printing two `tool_call:` lines that
/// differ — `run_freeform` renders the plan section before calling
/// `execute_freeform_next_action`, and the flag is injected between them — and
/// that is also deliberate. The two sections report different things: "request
/// plan" is what the planner derived from the request, and "execution" is the
/// argv handed to clap, so the added consent showing up only under the
/// `execution` header is how this surface discloses that it granted it. Nothing
/// is being solicited in between; under `--yes` the operator already approved
/// on the outer command line, and under no `--yes` the execution section is
/// never reached, so neither line is an approval prompt whose subject could
/// drift from what runs. Injecting into the plan render instead would have to
/// reach `render_structured_request_plan`, which the no-`--yes` review path
/// shares, and would print a pre-approved command to a human being asked to
/// review it — the case the paragraph above rules out.
///
/// So the difference stays deliberate but stops being unexplained: this returns
/// whether it actually added the flag, and the execution section says so in
/// words. A doc comment reaches the next reader of this file; the operator
/// looking at two `tool_call:` lines that disagree is the one who needs it.
///
/// Returns `true` only when the flag was not already present, so the disclosure
/// is about a flag this function added and not one the argv arrived with.
fn apply_freeform_execution_consent(args: &mut Vec<String>) -> bool {
    let is_install_sdk = args.first().is_some_and(|arg| arg == "install")
        && args.get(1).is_some_and(|arg| arg == "sdk");
    if !is_install_sdk || args.iter().any(|arg| arg == "--dry-run") {
        return false;
    }
    let already_present = args
        .iter()
        .any(|arg| arg == "--approve-replacing-active-default");
    chat::ensure_flag(args, "--approve-replacing-active-default");
    !already_present
}

fn validate_freeform_execution_action(action: &freeform_planner::FreeformPlanAction) -> Result<()> {
    if action.has_placeholders {
        bail!(
            "cannot execute natural-language plan because the next tool call still contains placeholder values: {}",
            freeform_planner::format_structured_tool_call("rocm", &action.args)
        );
    }
    if action.provider_assisted {
        bail!(
            "provider-assisted plans must be reviewed interactively; run the displayed structured rocm command directly after reviewing it"
        );
    }
    Ok(())
}

pub(crate) fn render_freeform_execution_header(execution: &FreeformExecution) -> String {
    let action = &execution.action;
    let mut output = String::new();
    let _ = writeln!(output);
    let _ = writeln!(output, "execution");
    let _ = writeln!(
        output,
        "  approval: {}",
        if action.approval_required {
            "granted by --yes"
        } else {
            "not required; executing because --yes was supplied"
        }
    );
    let _ = writeln!(
        output,
        "  tool_call: {}",
        freeform_planner::format_structured_tool_call("rocm", &action.args)
    );
    // Printed only when the two `tool_call:` lines actually disagree, and
    // immediately under the one that runs. Without it the operator sees a
    // consent flag on the executed command that the request plan above never
    // showed, with nothing on screen saying where it came from — the natural
    // reading being that something was approved behind their back rather than
    // that their own `--yes` was carried through.
    if execution.consent_added {
        let _ = writeln!(
            output,
            "  note: --approve-replacing-active-default was added here from your --yes, so this \
             tool_call differs from the one under `request plan` above; nothing was approved \
             between them."
        );
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;
    use rocm_core::RocmCliConfig;

    use crate::tests::test_app_paths;
    use crate::{Cli, Command, InstallTarget};

    #[test]
    fn freeform_execution_validation_rejects_placeholder_tool_calls() {
        let action =
            freeform_planner::freeform_plan_next_action("serve", &RocmCliConfig::default())
                .expect("serve request should have next action");

        let error = validate_freeform_execution_action(&action)
            .unwrap_err()
            .to_string();

        assert!(action.has_placeholders);
        assert!(error.contains("placeholder values"));
        assert!(error.contains("rocm serve <model>"));
    }

    #[test]
    fn freeform_execution_validation_accepts_fully_structured_tool_call() -> Result<()> {
        let action = freeform_planner::freeform_plan_next_action(
            "serve qwen3.5 with vllm",
            &RocmCliConfig::default(),
        )
        .expect("serve request should have next action");

        validate_freeform_execution_action(&action)?;
        assert_eq!(
            freeform_planner::format_structured_tool_call("rocm", &action.args),
            "rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed"
        );
        Ok(())
    }

    #[test]
    fn freeform_execution_grants_the_narrow_consent_the_outer_yes_already_gave() {
        // `rocm --yes <request>` dispatches the generated argv **in process**, so
        // nothing carries the outer `--yes` into `install()` unless this does.
        // Without it the surface prints `approval: granted by --yes` and then
        // either refuses non-interactively, asking for a flag it offers no way to
        // pass, or prompts on a terminal it just said it would not need to ask.
        let request =
            "install the latest TheRock nightly for this GPU into D:\\ROCm\\therock_venvs";
        let config = RocmCliConfig::default();
        let paths = test_app_paths();

        // Built exactly once, the way `run_freeform` builds it: the same
        // `plan` value is checked below and then handed to
        // `prepare_freeform_execution`, never rebuilt.
        let plan = freeform_planner::build_freeform_plan_with_context(request, &paths, &config);
        let planned = freeform_planner::plan_next_action(plan.clone())
            .expect("install request should have next action");
        assert!(
            !planned
                .args
                .iter()
                .any(|arg| arg.starts_with("--yes") || arg.starts_with("--approve-")),
            "the plan itself must stay unapproved so `rocm <request>` shows a \
             reviewable command: {:?}",
            planned.args
        );

        // Through the real pre-dispatch path, not the injector in isolation:
        // `execute_freeform_next_action` is this plus the header render and
        // `dispatch`, so dropping the injection from the pipeline fails here.
        let execution =
            prepare_freeform_execution(plan).expect("install request should prepare for execution");
        let action = &execution.action;

        assert_eq!(
            freeform_planner::format_structured_tool_call("rocm", &action.args),
            "rocm install sdk --channel nightly --format wheel --prefix \
             D:\\ROCm\\therock_venvs --approve-replacing-active-default"
        );
        // The narrow flag, never `--yes`: this surface has no terminal promise to
        // make about a sudo password prompt for system packages.
        assert!(!action.args.iter().any(|arg| arg == "--yes"));
        // The header renders after injection, so the printed tool call is the
        // argv that actually runs.
        let rendered = render_freeform_execution_header(&execution);
        assert!(rendered.contains("--approve-replacing-active-default"));
        // And the operator is told why this `tool_call:` carries a consent flag
        // the `request plan` section above it did not show. The plan assertion at
        // the top of this test is what makes the two lines differ here, so the
        // disclosure and the difference are pinned by the same test.
        assert!(
            execution.consent_added,
            "the plan arrived unapproved, so the injector must report adding the flag"
        );
        assert!(
            rendered.contains("was added here from your --yes"),
            "the execution section must explain the differing tool_call: {rendered}"
        );
        // Re-parsing must reach `install()` with the consent actually set.
        let mut argv = vec!["rocm".to_owned()];
        argv.extend(execution.action.args);
        let cli = Cli::try_parse_from(argv).expect("generated argv should parse");
        match cli.command {
            Some(Command::Install {
                target:
                    InstallTarget::Sdk {
                        yes,
                        approve_replacing_active_default,
                        ..
                    },
            }) => {
                assert!(approve_replacing_active_default);
                assert!(!yes);
            }
            other => panic!("expected `install sdk`, got {other:?}"),
        }
    }

    #[test]
    fn freeform_execution_consent_is_scoped_to_mutating_sdk_installs() {
        // Dry runs return before the consent gate, and the sibling dry-run-aware
        // arms in chat, rocmd and dash-tui all withhold the flag there.
        let mut dry_run = vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--dry-run".to_owned(),
        ];
        assert!(!apply_freeform_execution_consent(&mut dry_run));
        assert_eq!(
            dry_run,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--dry-run".to_owned()
            ]
        );

        // Nothing else the planner can emit takes this flag; injecting it would
        // not even parse.
        for mut args in [
            vec!["install".to_owned(), "driver".to_owned()],
            vec!["serve".to_owned(), "qwen".to_owned()],
            vec!["comfyui".to_owned(), "install".to_owned()],
        ] {
            let before = args.clone();
            assert!(!apply_freeform_execution_consent(&mut args));
            assert_eq!(args, before);
        }

        // Idempotent: a plan that already carries the flag is not given it twice,
        // and reports that it added nothing — the execution section must not
        // claim to have added a flag the argv arrived with.
        let mut already = vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--approve-replacing-active-default".to_owned(),
        ];
        assert!(!apply_freeform_execution_consent(&mut already));
        assert_eq!(
            already
                .iter()
                .filter(|arg| *arg == "--approve-replacing-active-default")
                .count(),
            1
        );
    }

    #[test]
    fn freeform_execution_header_surfaces_explicit_approval_and_tool_call() {
        let action = freeform_planner::freeform_plan_next_action(
            "serve qwen3.5 with vllm",
            &RocmCliConfig::default(),
        )
        .expect("serve request should have next action");
        let rendered = render_freeform_execution_header(&FreeformExecution {
            action,
            consent_added: false,
        });

        assert!(rendered.contains("execution"));
        assert!(rendered.contains("approval: granted by --yes"));
        assert!(rendered.contains(
            "tool_call: rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed"
        ));
        // Nothing was injected on this path, so the two `tool_call:` lines agree
        // and the disclosure would be noise that contradicts the plan above.
        assert!(
            !rendered.contains("was added here"),
            "the note must be scoped to an argv this surface actually changed: {rendered}"
        );
    }

    #[test]
    fn freeform_execution_validation_rejects_provider_assisted_plans() -> Result<()> {
        let content = r#"{
            "intent": "serve",
            "tool_call": {
                "tool": "rocm",
                "args": ["serve", "sshleifer/tiny-gpt2", "--engine", "vllm", "--managed"]
            }
        }"#;
        let plan = freeform_planner::provider_planner_response_to_plan(
            "start a local model",
            "local",
            content,
        )?;
        let action =
            freeform_planner::plan_next_action(plan).expect("provider plan should have an action");

        let error = validate_freeform_execution_action(&action)
            .unwrap_err()
            .to_string();

        assert!(action.provider_assisted);
        assert!(error.contains("reviewed interactively"));
        Ok(())
    }
}
