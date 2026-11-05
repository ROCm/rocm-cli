// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm install driver` command handlers and reconciliation.
//!
//! Mechanically relocated from `main.rs` with no behavior change — the
//! `install()` dispatcher's call sites stayed byte-identical
//! (`install_driver(...)`/`reconcile_driver_install(...)`, re-imported via
//! `use crate::driver_install::{install_driver, reconcile_driver_install};`);
//! `install()` has since moved to `install_cmd.rs`, which imports them itself.
//! `InstallTarget`/`Cli` remain in the crate root; neither is referenced
//! from this file. Unlike `automations.rs`/`uninstall.rs`, this cluster owns
//! private types (`DriverInstallState` and friends), so those moved here too
//! rather than staying in `main.rs`. `DriverInstallResult`/`DriverInstallError`
//! are the exception: they're `pub(crate)` and read from `install_cmd.rs`'s
//! `install()` (`result.output`/`result.executed`, `error.source`/`error.executed`).
//!
//! The plan this module runs (its model, rendering, per-distro builders and the
//! WSL rocDXG plan) lives in `driver_plan.rs`, which does not depend on this
//! module; the host's os-release comes from `system_packages.rs`. This module
//! re-exports `host_install_driver_only_reports` for `tool_policy.rs` and
//! `freeform_planner.rs`.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ExamineSummary, shell_command_for_host};
use serde::{Deserialize, Serialize};

use crate::cli_report;
pub(crate) use crate::driver_plan::host_install_driver_only_reports;
use crate::driver_plan::{
    DriverCommandPhase, DriverInstallPlan, PrivilegeEscalation, build_driver_install_plan_read,
    render_driver_install_plan,
};
use crate::system_packages::real_os_release;

/// Whether `rocm install driver` goes on to run the plan's commands. Pure so
/// the gate that keeps native Windows from installing anything (its plan is
/// neither `supported` nor `mutating`) is testable on any host.
const fn should_execute(plan: &DriverInstallPlan, yes: bool, dry_run: bool) -> bool {
    yes && !dry_run && plan.supported && plan.mutating
}

/// Whether an approved (`--yes`) run asks for a plan that installs nothing and
/// should say so with a failing exit status. Native Windows is exempt: its
/// plan only reports and promises to change nothing, so approving it is a
/// successful no-op.
fn approval_of_unsupported_plan(plan: &DriverInstallPlan, yes: bool, dry_run: bool) -> bool {
    yes && !dry_run && !plan.supported && plan.policy != "windows_validate_only"
}

pub(crate) fn install_driver(
    paths: &AppPaths,
    dkms: bool,
    yes: bool,
    dry_run: bool,
) -> std::result::Result<DriverInstallResult, DriverInstallError> {
    let examine =
        ExamineSummary::gather().map_err(|source| DriverInstallError::new(source, false))?;
    let os_release = real_os_release();
    // The only place the real privilege level is read; every builder below takes
    // it as a parameter so both branches stay testable on any host.
    let plan =
        build_driver_install_plan_read(&examine, &os_release, dkms, PrivilegeEscalation::detect());
    let mut output = render_driver_install_plan(&plan, yes, dry_run);
    if approval_of_unsupported_plan(&plan, yes, dry_run) {
        return Err(DriverInstallError::new(
            anyhow::anyhow!(
                "{output}\nnothing was installed: this host has no supported driver plan ({}). {}",
                plan.policy,
                plan.reason
            ),
            false,
        ));
    }
    if !should_execute(&plan, yes, dry_run) {
        return Ok(DriverInstallResult {
            output,
            executed: false,
        });
    }

    let boot_id = current_boot_id();
    let mut state = DriverInstallState {
        approved_at_unix_ms: rocm_core::unix_time_millis(),
        executed_at_unix_ms: None,
        pre_driver: examine.driver,
        post_driver: None,
        boot_id_at_execution: boot_id,
        reboot_required: plan.reboot_required,
        reboot_observed: false,
        commands: plan.execution_commands(),
        reconciled_at_unix_ms: None,
        reconciliation: None,
    };
    write_driver_install_state(paths, &state)
        .map_err(|source| DriverInstallError::new(source, false))?;

    execute_driver_install_plan(
        &plan,
        &mut state,
        run_driver_shell_command,
        |state| write_driver_install_state(paths, state),
        || ExamineSummary::gather().map(|summary| summary.driver),
    )
    .map_err(|source| DriverInstallError::new(source, true))?;

    let report = cli_report::ActionReport::new("driver install completed")
        .detail("reboot_required", plan.reboot_required)
        .detail("state", driver_install_state_path(paths).display());
    output.push_str(&report.render());
    Ok(DriverInstallResult {
        output,
        executed: true,
    })
}

fn execute_driver_install_plan<Run, Persist, Gather>(
    plan: &DriverInstallPlan,
    state: &mut DriverInstallState,
    mut run: Run,
    mut persist: Persist,
    gather_post_driver: Gather,
) -> Result<()>
where
    Run: FnMut(&str) -> Result<()>,
    Persist: FnMut(&DriverInstallState) -> Result<()>,
    Gather: FnOnce() -> Result<rocm_core::DriverSummary>,
{
    for command in &plan.commands {
        if plan.reboot_required && command.phase == DriverCommandPhase::Verify {
            continue;
        }
        run(&command.command)
            .with_context(|| format!("driver command failed: {}", command.command))?;
    }

    state.executed_at_unix_ms = Some(rocm_core::unix_time_millis());
    state.reboot_required = plan.reboot_required;
    state.reboot_observed = driver_reboot_observed(state.boot_id_at_execution.as_deref());
    persist(state)?;

    let post_driver = gather_post_driver()?;
    state.post_driver = Some(post_driver);
    persist(state)?;
    Ok(())
}

pub(crate) fn reconcile_driver_install(paths: &AppPaths) -> Result<String> {
    let Some(mut state) = read_driver_install_state(paths)? else {
        let mut output = String::new();
        let _ = writeln!(output, "driver install reconciliation");
        let _ = writeln!(
            output,
            "  state: {}",
            driver_install_state_path(paths).display()
        );
        let _ = writeln!(output, "  approval: not required");
        let _ = writeln!(output, "  privileged_commands: <none>");
        let _ = writeln!(output, "  status: no prior driver execution state found");
        let _ = writeln!(
            output,
            "  action: run `rocm install driver --dkms` to review the native driver plan"
        );
        return Ok(output);
    };
    let examine = ExamineSummary::gather()?;
    let checks = passive_driver_checks();
    reconcile_driver_install_state(paths, &mut state, examine.driver, current_boot_id(), checks)
}

fn reconcile_driver_install_state(
    paths: &AppPaths,
    state: &mut DriverInstallState,
    driver: rocm_core::DriverSummary,
    current_boot_id: Option<String>,
    checks: Vec<DriverPassiveCheck>,
) -> Result<String> {
    let reboot_observed = state
        .boot_id_at_execution
        .as_deref()
        .zip(current_boot_id.as_deref())
        .is_some_and(|(executed, current)| executed != current);
    state.reboot_observed = reboot_observed;
    state.post_driver = Some(driver.clone());
    let at_unix_ms = rocm_core::unix_time_millis();
    state.reconciled_at_unix_ms = Some(at_unix_ms);
    let check_summary = summarize_driver_passive_checks(&checks);
    state.reconciliation = Some(DriverReconciliationState {
        at_unix_ms,
        driver,
        reboot_observed,
        check_summary,
        checks,
    });
    write_driver_install_state(paths, state)?;
    Ok(render_driver_reconciliation(paths, state))
}

fn render_driver_reconciliation(paths: &AppPaths, state: &DriverInstallState) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "driver install reconciliation");
    let _ = writeln!(
        output,
        "  state: {}",
        driver_install_state_path(paths).display()
    );
    let _ = writeln!(output, "  approval: not required");
    let _ = writeln!(output, "  privileged_commands: <none>");
    let _ = writeln!(
        output,
        "  approved_at_unix_ms: {}",
        state.approved_at_unix_ms
    );
    let _ = writeln!(
        output,
        "  executed_at_unix_ms: {}",
        state
            .executed_at_unix_ms
            .map_or_else(|| "<not executed>".to_owned(), |value| value.to_string())
    );
    let _ = writeln!(output, "  reboot_required: {}", state.reboot_required);
    let _ = writeln!(output, "  reboot_observed: {}", state.reboot_observed);
    if let Some(reconciliation) = &state.reconciliation {
        let _ = writeln!(
            output,
            "  reconciled_at_unix_ms: {}",
            reconciliation.at_unix_ms
        );
        let _ = writeln!(output, "  driver_status: {}", reconciliation.driver.status);
        let _ = writeln!(
            output,
            "  driver_detail: {}",
            reconciliation
                .driver
                .detail
                .as_deref()
                .unwrap_or("<unknown>")
        );
        let _ = writeln!(
            output,
            "  passive_check_summary: total={} present={} missing={}",
            reconciliation.check_summary.total,
            reconciliation.check_summary.present,
            reconciliation.check_summary.missing
        );
        if reconciliation.checks.is_empty() {
            let _ = writeln!(output, "  passive_checks: <none for this platform>");
        } else {
            let _ = writeln!(output, "  passive_checks:");
            for check in &reconciliation.checks {
                let _ = writeln!(
                    output,
                    "    {}: {} ({})",
                    check.name, check.status, check.detail
                );
            }
        }
        if state.reboot_required && !state.reboot_observed {
            let _ = writeln!(
                output,
                "  action: reboot is still required before post-install checks are meaningful"
            );
        } else if reconciliation
            .checks
            .iter()
            .any(|check| check.status != "present")
        {
            let _ = writeln!(
                output,
                "  action: reconciliation recorded missing passive checks; run `rocm examine` and inspect driver logs"
            );
        } else {
            let _ = writeln!(
                output,
                "  action: reconciliation complete; run `rocm examine` for the full host summary"
            );
        }
    }
    output
}

fn summarize_driver_passive_checks(checks: &[DriverPassiveCheck]) -> DriverPassiveCheckSummary {
    let total = checks.len();
    let present = checks
        .iter()
        .filter(|check| check.status == "present")
        .count();
    DriverPassiveCheckSummary {
        total,
        present,
        missing: total.saturating_sub(present),
    }
}

fn passive_driver_checks() -> Vec<DriverPassiveCheck> {
    if !rocm_core::runtime_is_linux() {
        return Vec::new();
    }
    vec![
        passive_path_check("/sys/module/amdgpu", "amdgpu kernel module path"),
        passive_path_check("/dev/kfd", "KFD device node"),
        passive_render_node_check(),
    ]
}

fn passive_path_check(path: &str, detail: &str) -> DriverPassiveCheck {
    DriverPassiveCheck {
        name: path.to_owned(),
        status: if rocm_core::host_path(path).exists() {
            "present"
        } else {
            "missing"
        }
        .to_owned(),
        detail: detail.to_owned(),
    }
}

fn passive_render_node_check() -> DriverPassiveCheck {
    let present = fs::read_dir(rocm_core::host_path("/dev/dri"))
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(std::result::Result::ok))
        .any(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("renderD"))
        });
    DriverPassiveCheck {
        name: "/dev/dri/renderD*".to_owned(),
        status: if present { "present" } else { "missing" }.to_owned(),
        detail: "DRM render node".to_owned(),
    }
}

pub(crate) struct DriverInstallResult {
    pub(crate) output: String,
    pub(crate) executed: bool,
}

pub(crate) struct DriverInstallError {
    pub(crate) source: anyhow::Error,
    pub(crate) executed: bool,
}

impl DriverInstallError {
    const fn new(source: anyhow::Error, executed: bool) -> Self {
        Self { source, executed }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DriverInstallState {
    approved_at_unix_ms: u128,
    executed_at_unix_ms: Option<u128>,
    pre_driver: rocm_core::DriverSummary,
    post_driver: Option<rocm_core::DriverSummary>,
    boot_id_at_execution: Option<String>,
    reboot_required: bool,
    reboot_observed: bool,
    commands: Vec<String>,
    #[serde(default)]
    reconciled_at_unix_ms: Option<u128>,
    #[serde(default)]
    reconciliation: Option<DriverReconciliationState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DriverReconciliationState {
    at_unix_ms: u128,
    driver: rocm_core::DriverSummary,
    reboot_observed: bool,
    #[serde(default)]
    check_summary: DriverPassiveCheckSummary,
    checks: Vec<DriverPassiveCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DriverPassiveCheckSummary {
    total: usize,
    present: usize,
    missing: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DriverPassiveCheck {
    name: String,
    status: String,
    detail: String,
}

fn run_driver_shell_command(command: &str) -> Result<()> {
    run_shell_command_with_stdin(command, Stdio::null())
}

/// Run a hardcoded shell command, wiring its stdin to `stdin`.
///
/// Most install commands run with a null stdin, but privileged commands that may
/// trigger an interactive `sudo` password prompt (such as the OpenMPI install
/// approved with `--yes`) must inherit the terminal so the user can respond.
fn run_shell_command_with_stdin(command: &str, stdin: Stdio) -> Result<()> {
    let (program, args) = shell_command_for_host(command);
    let status = ProcessCommand::new(program)
        .args(args)
        .stdin(stdin)
        .status()
        .with_context(|| format!("failed to launch `{command}`"))?;
    if !status.success() {
        bail!("`{command}` exited with {status}");
    }
    Ok(())
}

pub(crate) fn driver_install_state_path(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("driver").join("state.json")
}

fn write_driver_install_state(paths: &AppPaths, state: &DriverInstallState) -> Result<()> {
    let path = driver_install_state_path(paths);
    let parent = path.parent().context("driver state path has no parent")?;
    fs::create_dir_all(parent)?;
    fs::write(&path, serde_json::to_vec_pretty(state)?)?;
    Ok(())
}

fn read_driver_install_state(paths: &AppPaths) -> Result<Option<DriverInstallState>> {
    let path = driver_install_state_path(paths);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let state = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(state))
}

fn current_boot_id() -> Option<String> {
    fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn driver_reboot_observed(executed_boot_id: Option<&str>) -> bool {
    let Some(executed_boot_id) = executed_boot_id else {
        return false;
    };
    current_boot_id()
        .as_deref()
        .is_some_and(|current| current != executed_boot_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver_plan::tests::{os_release_found, test_examine};
    use crate::driver_plan::{
        driver_command, invalid_driver_version_override_plan, wsl_rocdxg_driver_plan,
    };
    use crate::tests::{ScopedTestEnv, test_paths};

    #[test]
    fn driver_plan_executor_runs_verify_after_execute() -> Result<()> {
        let mut plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        plan.commands = vec![
            driver_command(DriverCommandPhase::Prepare, "prepare"),
            driver_command(DriverCommandPhase::Execute, "execute"),
            driver_command(DriverCommandPhase::Verify, "verify"),
        ];
        let mut state = DriverInstallState {
            approved_at_unix_ms: 1,
            executed_at_unix_ms: None,
            pre_driver: test_examine("linux", true).driver,
            post_driver: None,
            boot_id_at_execution: Some("boot".to_owned()),
            reboot_required: plan.reboot_required,
            reboot_observed: false,
            commands: plan.execution_commands(),
            reconciled_at_unix_ms: None,
            reconciliation: None,
        };
        let mut observed = Vec::new();

        execute_driver_install_plan(
            &plan,
            &mut state,
            |command| {
                observed.push(command.to_owned());
                Ok(())
            },
            |_| Ok(()),
            || Ok(test_examine("linux", true).driver),
        )?;

        assert_eq!(observed, ["prepare", "execute", "verify"]);
        assert!(state.executed_at_unix_ms.is_some());
        Ok(())
    }

    #[test]
    fn driver_plan_executor_defers_verify_when_reboot_is_required() -> Result<()> {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found("ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n"),
            true,
            PrivilegeEscalation::Sudo,
        );
        assert!(plan.reboot_required);
        let expected = plan.execution_commands();
        let verify_commands = plan
            .commands
            .iter()
            .filter(|command| command.phase == DriverCommandPhase::Verify)
            .map(|command| command.command.clone())
            .collect::<Vec<_>>();
        let mut state = DriverInstallState {
            approved_at_unix_ms: 1,
            executed_at_unix_ms: None,
            pre_driver: test_examine("linux", false).driver,
            post_driver: None,
            boot_id_at_execution: Some("boot".to_owned()),
            reboot_required: plan.reboot_required,
            reboot_observed: false,
            commands: plan.execution_commands(),
            reconciled_at_unix_ms: None,
            reconciliation: None,
        };
        let mut observed = Vec::new();

        execute_driver_install_plan(
            &plan,
            &mut state,
            |command| {
                observed.push(command.to_owned());
                Ok(())
            },
            |_| Ok(()),
            || Ok(test_examine("linux", false).driver),
        )?;

        assert_eq!(observed, expected);
        assert!(
            verify_commands
                .iter()
                .all(|command| !observed.contains(command)),
            "reboot-gated Verify commands must be deferred: {verify_commands:?}"
        );
        assert!(state.executed_at_unix_ms.is_some());
        assert!(state.reboot_required);
        Ok(())
    }

    #[test]
    fn failed_driver_verify_does_not_mark_execution_completed() -> Result<()> {
        let (root, paths) = test_paths("driver-verify-failure-state");
        let mut plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        plan.commands = vec![
            driver_command(DriverCommandPhase::Prepare, "prepare"),
            driver_command(DriverCommandPhase::Execute, "execute"),
            driver_command(DriverCommandPhase::Verify, "verify"),
        ];
        let mut state = DriverInstallState {
            approved_at_unix_ms: 1,
            executed_at_unix_ms: None,
            pre_driver: test_examine("linux", true).driver,
            post_driver: None,
            boot_id_at_execution: Some("boot".to_owned()),
            reboot_required: plan.reboot_required,
            reboot_observed: false,
            commands: plan.execution_commands(),
            reconciled_at_unix_ms: None,
            reconciliation: None,
        };
        write_driver_install_state(&paths, &state)?;
        let mut observed = Vec::new();
        let mut gathered = false;

        let error = execute_driver_install_plan(
            &plan,
            &mut state,
            |command| {
                observed.push(command.to_owned());
                if command == "verify" {
                    bail!("verification rejected the install");
                }
                Ok(())
            },
            |state| write_driver_install_state(&paths, state),
            || {
                gathered = true;
                Ok(test_examine("linux", true).driver)
            },
        )
        .expect_err("failed verification must fail the install");
        let saved = read_driver_install_state(&paths)?.expect("state should remain readable");

        assert_eq!(observed, ["prepare", "execute", "verify"]);
        assert!(error.to_string().contains("driver command failed: verify"));
        assert!(
            !gathered,
            "post-install state must not be gathered after failure"
        );
        assert_eq!(state.executed_at_unix_ms, None);
        assert!(state.post_driver.is_none());
        assert_eq!(saved.executed_at_unix_ms, None);
        assert!(saved.post_driver.is_none());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn failed_post_driver_gather_keeps_executed_state_persisted() -> Result<()> {
        let (root, paths) = test_paths("driver-gather-failure-state");
        let mut plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        plan.commands = vec![
            driver_command(DriverCommandPhase::Prepare, "prepare"),
            driver_command(DriverCommandPhase::Execute, "execute"),
            driver_command(DriverCommandPhase::Verify, "verify"),
        ];
        let mut state = DriverInstallState {
            approved_at_unix_ms: 1,
            executed_at_unix_ms: None,
            pre_driver: test_examine("linux", true).driver,
            post_driver: None,
            boot_id_at_execution: Some("boot".to_owned()),
            reboot_required: plan.reboot_required,
            reboot_observed: false,
            commands: plan.execution_commands(),
            reconciled_at_unix_ms: None,
            reconciliation: None,
        };
        write_driver_install_state(&paths, &state)?;

        let error = execute_driver_install_plan(
            &plan,
            &mut state,
            |_| Ok(()),
            |state| write_driver_install_state(&paths, state),
            || bail!("post-driver gather failed"),
        )
        .expect_err("a post-driver gather failure must still fail the install");
        let saved = read_driver_install_state(&paths)?.expect("state should remain readable");

        assert!(error.to_string().contains("post-driver gather failed"));
        assert!(saved.executed_at_unix_ms.is_some());
        assert!(saved.post_driver.is_none());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn driver_reconcile_without_state_gives_non_privileged_guidance() -> Result<()> {
        let (root, paths) = test_paths("driver-reconcile-empty");

        let rendered = reconcile_driver_install(&paths)?;

        assert!(rendered.contains("driver install reconciliation"));
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("privileged_commands: <none>"));
        assert!(rendered.contains("no prior driver execution state found"));
        assert!(rendered.contains("rocm install driver --dkms"));
        assert!(!driver_install_state_path(&paths).exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn driver_reconcile_updates_state_after_reboot() -> Result<()> {
        let (root, paths) = test_paths("driver-reconcile-state");
        let pre_driver = rocm_core::DriverSummary {
            policy: "linux_official_amd_dkms_wrapper".to_owned(),
            status: "not_detected".to_owned(),
            detail: None,
            windows: None,
        };
        let current_driver = rocm_core::DriverSummary {
            policy: "linux_official_amd_dkms_wrapper".to_owned(),
            status: "amdgpu_available".to_owned(),
            detail: Some("/dev/kfd is present".to_owned()),
            windows: None,
        };
        let mut state = DriverInstallState {
            approved_at_unix_ms: 1,
            executed_at_unix_ms: Some(2),
            pre_driver,
            post_driver: None,
            boot_id_at_execution: Some("old-boot".to_owned()),
            reboot_required: true,
            reboot_observed: false,
            commands: vec!["sudo apt-get install -y amdgpu-dkms".to_owned()],
            reconciled_at_unix_ms: None,
            reconciliation: None,
        };
        let checks = vec![
            DriverPassiveCheck {
                name: "/dev/kfd".to_owned(),
                status: "present".to_owned(),
                detail: "KFD device node".to_owned(),
            },
            DriverPassiveCheck {
                name: "/dev/dri/renderD*".to_owned(),
                status: "missing".to_owned(),
                detail: "DRM render node".to_owned(),
            },
        ];

        let rendered = reconcile_driver_install_state(
            &paths,
            &mut state,
            current_driver,
            Some("new-boot".to_owned()),
            checks,
        )?;
        let saved = read_driver_install_state(&paths)?.expect("state should be saved");

        assert!(rendered.contains("reboot_observed: true"));
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("privileged_commands: <none>"));
        assert!(rendered.contains("driver_status: amdgpu_available"));
        assert!(rendered.contains("passive_check_summary: total=2 present=1 missing=1"));
        assert!(rendered.contains("/dev/dri/renderD*: missing"));
        assert!(rendered.contains("missing passive checks"));
        assert!(saved.reboot_observed);
        assert!(saved.reconciled_at_unix_ms.is_some());
        assert_eq!(
            saved
                .reconciliation
                .as_ref()
                .map(|value| value.driver.status.as_str()),
            Some("amdgpu_available")
        );
        let reconciliation = saved.reconciliation.as_ref().expect("reconciliation saved");
        assert_eq!(reconciliation.check_summary.total, 2);
        assert_eq!(reconciliation.check_summary.present, 1);
        assert_eq!(reconciliation.check_summary.missing, 1);
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn driver_reconcile_preserves_explicit_reboot_policy() -> Result<()> {
        for reboot_required in [false, true] {
            let (root, paths) = test_paths(if reboot_required {
                "driver-reconcile-reboot-true"
            } else {
                "driver-reconcile-reboot-false"
            });
            let driver = rocm_core::DriverSummary {
                policy: "driver-policy".to_owned(),
                status: "available".to_owned(),
                detail: None,
                windows: None,
            };
            let mut state = DriverInstallState {
                approved_at_unix_ms: 1,
                executed_at_unix_ms: Some(2),
                pre_driver: driver.clone(),
                post_driver: None,
                boot_id_at_execution: Some("same-boot".to_owned()),
                reboot_required,
                reboot_observed: false,
                commands: vec!["execute".to_owned()],
                reconciled_at_unix_ms: None,
                reconciliation: None,
            };

            reconcile_driver_install_state(
                &paths,
                &mut state,
                driver,
                Some("same-boot".to_owned()),
                Vec::new(),
            )?;
            let saved = read_driver_install_state(&paths)?.expect("state should be saved");

            assert_eq!(state.reboot_required, reboot_required);
            assert_eq!(saved.reboot_required, reboot_required);
            let _ = fs::remove_dir_all(root);
        }
        Ok(())
    }

    #[test]
    fn driver_passive_check_summary_counts_non_present_as_missing() {
        let summary = summarize_driver_passive_checks(&[
            DriverPassiveCheck {
                name: "/dev/kfd".to_owned(),
                status: "present".to_owned(),
                detail: "KFD".to_owned(),
            },
            DriverPassiveCheck {
                name: "/dev/dri/renderD*".to_owned(),
                status: "missing".to_owned(),
                detail: "render".to_owned(),
            },
            DriverPassiveCheck {
                name: "dkms".to_owned(),
                status: "error".to_owned(),
                detail: "dkms status failed".to_owned(),
            },
        ]);

        assert_eq!(summary.total, 3);
        assert_eq!(summary.present, 1);
        assert_eq!(summary.missing, 2);
    }

    #[test]
    fn approving_an_unsupported_plan_is_an_error_but_not_a_dry_run_or_windows() {
        let mut plan = unsupported_test_plan();
        assert!(approval_of_unsupported_plan(&plan, true, false));
        assert!(!approval_of_unsupported_plan(&plan, false, false));
        assert!(!approval_of_unsupported_plan(&plan, true, true));
        plan.policy = "windows_validate_only".to_owned();
        assert!(!approval_of_unsupported_plan(&plan, true, false));
        plan.policy = "x".to_owned();
        plan.supported = true;
        assert!(!approval_of_unsupported_plan(&plan, true, false));
    }

    fn unsupported_test_plan() -> DriverInstallPlan {
        invalid_driver_version_override_plan("x".to_owned())
    }

    /// The gate behind "installs nothing on Windows": approval does not make a
    /// Windows plan execute, while the same flags do execute a supported,
    /// mutating plan, so the assertion cannot pass by always returning false.
    #[test]
    fn should_execute_never_runs_a_windows_plan() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let windows = build_driver_install_plan_read(
            &test_examine("windows", false),
            &os_release_found(""),
            true,
            PrivilegeEscalation::Sudo,
        );
        assert!(!should_execute(&windows, true, false));
        assert!(!should_execute(&windows, false, false));

        let mut mutating = windows;
        mutating.supported = true;
        mutating.mutating = true;
        assert!(should_execute(&mutating, true, false));
        assert!(!should_execute(&mutating, false, false));
        assert!(!should_execute(&mutating, true, true));
    }
}
