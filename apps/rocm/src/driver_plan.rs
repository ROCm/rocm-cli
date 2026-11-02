// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `rocm install driver` plan: the model (`DriverInstallPlan`, its commands
//! and their phases, how a command reaches root), the text the plan is printed
//! as, the helpers every plan builder shares, and the builders themselves, which
//! turn the host's examine summary and os-release into a per-distro plan (apt,
//! dnf, zypper, or the WSL rocDXG plan in the child module `rocdxg`).
//!
//! Relocated from `driver_install.rs` with no behavior change; the moved
//! struct fields are `pub(crate)` because `driver_install.rs`, which runs the
//! plan and records its state, reads and builds them.
//!
//! Production code in this file uses only `std` and `rocm_core` (the child
//! module also reads `truthy_env` from `therock.rs`), and nothing in this module
//! or its tests depends on `driver_install.rs`.

use std::fmt::Write as _;

use rocm_core::ExamineSummary;
use rocm_core::placeholder::empty_as_unknown;

mod rocdxg;

pub(crate) use rocdxg::wsl_rocdxg_driver_plan;

#[derive(Debug, Clone)]
pub(crate) struct DriverInstallPlan {
    pub(crate) supported: bool,
    pub(crate) mutating: bool,
    pub(crate) policy: String,
    pub(crate) os_id: String,
    pub(crate) version_id: String,
    pub(crate) codename: String,
    pub(crate) repo_version_expr: String,
    pub(crate) reason: String,
    pub(crate) preflight_checks: Vec<String>,
    pub(crate) commands: Vec<DriverPlanCommand>,
    pub(crate) checks: Vec<String>,
    /// Whether the host must reboot before the verification steps mean anything.
    ///
    /// True for the kernel-module paths: an amdgpu DKMS build is not live until
    /// the machine comes back up. False on WSL2, where nothing kernel-side
    /// changes — ROCDXG is a userspace library and `ldconfig` publishes it
    /// immediately, so telling the user to reboot would be wrong.
    pub(crate) reboot_required: bool,
}

impl DriverInstallPlan {
    pub(crate) fn execution_commands(&self) -> Vec<String> {
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
pub(crate) enum DriverCommandPhase {
    Prepare,
    Execute,
    Verify,
}

#[derive(Debug, Clone)]
pub(crate) struct DriverPlanCommand {
    pub(crate) phase: DriverCommandPhase,
    pub(crate) command: String,
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
pub(crate) enum PrivilegeEscalation {
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
    pub(crate) fn detect() -> Self {
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

/// Whether a string is safe to interpolate into the driver-install plan's
/// shell commands: these values (the `ROCM_CLI_AMDGPU_DRIVER_VERSION`
/// override, and `VERSION_ID`/`VERSION_CODENAME`/`UBUNTU_CODENAME` read from
/// `/etc/os-release`) end up both inside single-quoted `printf` arguments and,
/// for `VERSION_ID` in the SUSEConnect commands, directly unquoted on the
/// command line -- so only letters, digits, and `. _ -` are accepted, the
/// same charset a real amdgpu/distro version or codename is made of.
fn is_safe_shell_token(v: &str) -> bool {
    !v.is_empty()
        && v.chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
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

pub(crate) fn driver_command(phase: DriverCommandPhase, command: &str) -> DriverPlanCommand {
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
/// runtime shell — which, for the single-quoted WSL ROCDXG download command, would
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

pub(crate) fn render_driver_install_plan(
    plan: &DriverInstallPlan,
    yes: bool,
    dry_run: bool,
) -> String {
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
    let _ = writeln!(output, "  repo_version: {}", plan.repo_version_expr);
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

/// Whether `rocm install driver` only reports on this OS. True on native
/// Windows, where the plan is `windows_validate_only` and runs nothing; the
/// chat and freeform planners derive their wording and flags from this too.
pub(crate) fn install_driver_only_reports_on(os: &str) -> bool {
    os == "windows"
}

/// [`install_driver_only_reports_on`] for the OS this binary runs on.
pub(crate) fn host_install_driver_only_reports() -> bool {
    install_driver_only_reports_on(std::env::consts::OS)
}

/// The plan for a release file that cannot be read as a distro at all — a
/// malformed line ([`build_driver_install_plan_at`]'s own early return) or a
/// file that exists but could not be read
/// ([`build_driver_install_plan_read`]'s `Unreadable` branch). Both name
/// `reason`, never a distro, and differ only in it: shared here, including
/// `repo_version_expr`, so a new early branch added to one is not silently
/// absent from the other.
fn unreadable_os_release_plan(reason: String, repo_version_expr: String) -> DriverInstallPlan {
    DriverInstallPlan {
        supported: false,
        mutating: false,
        policy: "unreadable_os_release".to_owned(),
        os_id: String::new(),
        version_id: String::new(),
        codename: String::new(),
        repo_version_expr,
        reason,
        preflight_checks: Vec::new(),
        commands: Vec::new(),
        checks: vec!["rocm examine".to_owned()],
        reboot_required: false,
    }
}

const RETIRED_DRIVER_VERSION_VARS: [&str; 2] =
    ["ROCM_CLI_AMDGPU_VERSION", "ROCM_CLI_AMDGPU_PACKAGE_RELEASE"];

/// The refusal plan `resolve_repo_version_expr` returns for any
/// `ROCM_CLI_AMDGPU_DRIVER_VERSION` value that cannot be safely interpolated
/// into a shell command -- whether it failed [`is_safe_shell_token`] or could
/// not even be decoded as UTF-8.
pub(crate) fn invalid_driver_version_override_plan(value_debug: String) -> DriverInstallPlan {
    DriverInstallPlan {
        supported: false,
        mutating: false,
        policy: "invalid_driver_version_override".to_owned(),
        os_id: String::new(),
        version_id: String::new(),
        codename: String::new(),
        repo_version_expr: value_debug,
        reason: "ROCM_CLI_AMDGPU_DRIVER_VERSION is not a well-formed amdgpu driver release. It is interpolated into privileged shell commands, so only letters, digits, and `. _ -` are accepted.".to_owned(),
        preflight_checks: Vec::new(),
        commands: Vec::new(),
        checks: vec!["rocm examine".to_owned()],
        reboot_required: false,
    }
}

/// Resolves and validates `ROCM_CLI_AMDGPU_DRIVER_VERSION`, shared between
/// [`build_driver_install_plan_at`] (reached whenever the os-release text
/// itself was readable) and [`build_driver_install_plan_read`]'s `Unreadable`
/// branch (reached before `_at` is even called), so both report the same
/// `invalid_driver_version_override` plan on a malformed override rather than
/// one of them silently falling back to `latest`.
fn resolve_repo_version_expr() -> Result<String, Box<DriverInstallPlan>> {
    // The pre-`latest` pin variables are no longer read; refuse rather than
    // install a floating `latest` for a user who still exports one.
    for retired in RETIRED_DRIVER_VERSION_VARS {
        if std::env::var_os(retired).is_some_and(|v| !v.is_empty()) {
            let mut plan = invalid_driver_version_override_plan(format!("{retired} is set"));
            plan.policy = "retired_driver_version_override".to_owned();
            plan.reason = format!(
                "{retired} is no longer read; unset it, and set ROCM_CLI_AMDGPU_DRIVER_VERSION to pin an amdgpu driver release."
            );
            return Err(Box::new(plan));
        }
    }
    // Resolved here rather than left as a shell `${VAR:-default}` expression:
    // these commands are built with `printf '%s\n' '...'` (single-quoted) so the
    // written repo file has a working baseurl regardless of shell. The value is
    // interpolated straight into that single-quoted command, so it's restricted
    // to a safe URL path segment here rather than trusted verbatim: a value
    // containing `'` would otherwise close the quoting and inject shell. An
    // invalid override is refused rather than silently falling back to
    // `latest`, the same way `wsl_rocdxg_driver_plan` refuses a malformed
    // `ROCM_CLI_ROCDXG_VERSION` -- a user who set this expecting a specific
    // release should see why it did not take effect, not an unexplained
    // `latest`.
    match std::env::var("ROCM_CLI_AMDGPU_DRIVER_VERSION") {
        // Set-but-empty is treated the same as unset, matching every other
        // optional override of the driver plan (`resolve_rocdxg_verification`,
        // `resolve_shell_default_template`): a blank env var is how an env
        // file clears an override, not a request to refuse the plan.
        Err(std::env::VarError::NotPresent) => Ok("latest".to_owned()),
        Ok(v) if v.is_empty() => Ok("latest".to_owned()),
        Ok(v) if is_safe_shell_token(&v) => Ok(v),
        Ok(v) => Err(Box::new(invalid_driver_version_override_plan(format!(
            "{v:?}"
        )))),
        // A non-UTF-8 value can never pass `is_safe_shell_token` anyway, but
        // routing it through the same refusal as a malformed value (instead
        // of folding it into the NotPresent/empty "unset" case) keeps the
        // same "refuse, don't silently fall back to latest" guarantee for a
        // user who set this to something `env::var` cannot even decode.
        Err(std::env::VarError::NotUnicode(v)) => {
            // Debug, not `v.display()`, is deliberate here: this is the one
            // value worth escaping rather than rendering lossily, since the
            // whole reason it reached this arm is that it isn't valid UTF-8.
            #[allow(clippy::unnecessary_debug_formatting)]
            let value_debug = format!("{v:?}");
            Err(Box::new(invalid_driver_version_override_plan(value_debug)))
        }
    }
}

/// [`build_driver_install_plan_read`], naming `os_release_path` instead of always
/// `/etc/os-release` in an unreadable-line message.
fn build_driver_install_plan_at(
    examine: &ExamineSummary,
    os_release_path: &str,
    os_release_text: &str,
    dkms: bool,
    escalation: PrivilegeEscalation,
) -> DriverInstallPlan {
    if install_driver_only_reports_on(&examine.os) {
        return DriverInstallPlan {
            supported: false,
            mutating: false,
            policy: "windows_validate_only".to_owned(),
            os_id: "windows".to_owned(),
            version_id: String::new(),
            codename: String::new(),
            // Not interpolated into any command on this no-op plan (there are
            // none), so an override is not validated before this early return.
            // Always "latest" even if `ROCM_CLI_AMDGPU_DRIVER_VERSION` names
            // something else or is malformed: unlike the Linux arms below,
            // this value has not gone through `resolve_repo_version_expr()`,
            // so do not read the printed `repo_version:` line here (or in
            // `--dry-run` output) as confirmation that an override was seen or
            // accepted.
            repo_version_expr: "latest".to_owned(),
            // Uses the same inventory and adapter choice as `rocm examine`
            // and `rocm diagnose`. Separate probes can still fail independently.
            reason: rocm_core::windows_driver::install_driver_summary(
                examine.driver.windows.as_ref(),
            ),
            preflight_checks: Vec::new(),
            commands: Vec::new(),
            checks: vec!["rocm examine".to_owned(), "rocm diagnose".to_owned()],
            // The plan installs nothing, so nothing asks for a reboot.
            reboot_required: false,
        };
    }
    if examine.wsl.as_ref().is_some_and(|wsl| wsl.is_wsl) {
        return wsl_rocdxg_driver_plan(escalation);
    }

    let repo_version_expr = match resolve_repo_version_expr() {
        Ok(v) => v,
        Err(plan) => return *plan,
    };

    // An unreadable file is reported as itself, naming the line, rather than
    // falling through to "this distro is not supported" — which would send the
    // user looking for a support matrix when the fix is one line of a file.
    let os_release = match rocm_core::os_release::parse(os_release_text) {
        Ok(fields) => fields,
        Err(unreadable) => {
            return unreadable_os_release_plan(
                format!(
                    "{os_release_path} {unreadable}; no driver commands were planned \
                     because the distro cannot be read. To fix it, {}.",
                    unreadable.fix()
                ),
                repo_version_expr,
            );
        }
    };

    let field = |key: &str| os_release.get(key).cloned();
    // Not interpolated into any shell command today (only matched against and
    // stored on the plan), but it goes through the same allow-list as
    // `version_id`/`codename`/the env override anyway: centralizing the
    // sanitization here means a future `os_id`-based repo path can't
    // reintroduce the shell-injection class those fields close. An invalid
    // value is treated as absent, falling through to the "unsupported" plan
    // the same way the other fields do.
    // Lowercased, as `rocm-core`'s `dkms_driver_plan_is_supported` does, so the
    // two copies of the support matrix agree on `ID=Ubuntu`.
    let os_id = field("ID")
        .filter(|v| is_safe_shell_token(v))
        .unwrap_or_default()
        .to_ascii_lowercase();
    // `version_id` is interpolated unsanitized both into the same
    // single-quoted `printf '...' | tee` commands as `repo_version_expr`
    // above and, on SLES, directly unquoted on the SUSEConnect command line --
    // so it gets the same char allow-list as that override. A value that
    // fails it is treated as absent (empty), which cannot match any
    // `(os_id, version_id)` arm below and falls through to the safe,
    // non-mutating "unsupported" plan, the same way a hostile codename falls
    // through to the static mapping rather than being used verbatim.
    let version_id = field("VERSION_ID")
        .filter(|v| is_safe_shell_token(v))
        .unwrap_or_default();
    // `codename` is interpolated unsanitized into the same single-quoted
    // `printf '...' | tee` command as `repo_version_expr` above, so an
    // os-release-sourced value gets the same char allow-list: a crafted
    // VERSION_CODENAME/UBUNTU_CODENAME (compromised or custom base image)
    // could otherwise break out of the quoting and inject shell. An invalid
    // value is treated as absent, falling through to the trusted static
    // `codename_for_version` mapping rather than being used verbatim.
    let codename = field("VERSION_CODENAME")
        .or_else(|| field("UBUNTU_CODENAME"))
        .filter(|v| is_safe_shell_token(v))
        .or_else(|| codename_for_version(&os_id, &version_id).map(str::to_owned))
        .unwrap_or_default();
    let id_like = field("ID_LIKE").unwrap_or_default();

    match (os_id.as_str(), version_id.as_str()) {
        ("ubuntu", "22.04" | "24.04" | "26.04") => apt_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version_expr,
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
                repo_version_expr,
                dkms,
                false,
                escalation,
            );
            // Debian deliberately reuses AMD's Ubuntu-suite repository: AMD's
            // documented Debian install maps Debian 12 -> jammy and 13 -> noble
            // and serves them from the .../ubuntu amdgpu tree. Surface that in
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
            repo_version_expr,
            dkms,
            DnfDriverDistro::Rhel,
            escalation,
        ),
        ("ol", "10.1" | "9.7" | "8.10") => dnf_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version_expr,
            dkms,
            DnfDriverDistro::Oracle,
            escalation,
        ),
        ("rocky", "9.4" | "9.6" | "9.7") => dnf_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version_expr,
            dkms,
            DnfDriverDistro::Rocky,
            escalation,
        ),
        ("sles" | "sle", "15.7") => {
            sles_driver_plan(os_id, version_id, codename, repo_version_expr, dkms, escalation)
        }
        _ => driver_plan_via_id_like(
            &os_id,
            &version_id,
            &id_like,
            &codename,
            &repo_version_expr,
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
            repo_version_expr,
            reason: "Linux DKMS driver install is currently planned only for AMD-documented Ubuntu, Debian, RHEL, Oracle Linux, SLES, and Rocky versions; no commands were guessed for this distro.".to_owned(),
            preflight_checks: Vec::new(),
            commands: Vec::new(),
            checks: vec!["rocm examine".to_owned()],
            // Kernel module: not live until the machine comes back up.
            reboot_required: true,
        }),
    }
}

/// [`build_driver_install_plan_at`] for `read`, naming the file actually read
/// in an unreadable-line message, and reporting a file that exists but could
/// not be read as itself rather than as an unknown distribution.
///
/// Without the latter, a host whose `/etc/os-release` is merely unreadable
/// (permission denied, a directory in its place) would plan nothing and say
/// "no commands were guessed for this distro" — the same message an actually
/// unsupported distribution gets — sending the user looking for a support
/// gap instead of a filesystem problem. The Windows and WSL branches never
/// reach the os-release parse at all, so they are unaffected either way and
/// are left to the normal call below.
pub(crate) fn build_driver_install_plan_read(
    examine: &ExamineSummary,
    read: &rocm_core::hardware_root::OsRelease,
    dkms: bool,
    escalation: PrivilegeEscalation,
) -> DriverInstallPlan {
    let reaches_os_release_parse =
        examine.os != "windows" && !examine.wsl.as_ref().is_some_and(|wsl| wsl.is_wsl);
    if reaches_os_release_parse
        && let rocm_core::hardware_root::OsRelease::Unreadable { path, error } = read
    {
        let repo_version_expr = match resolve_repo_version_expr() {
            Ok(v) => v,
            Err(plan) => return *plan,
        };
        return unreadable_os_release_plan(
            format!(
                "{}; no driver commands were planned because the distro cannot be read.",
                rocm_core::hardware_root::unreadable_prefix(path, error)
            ),
            repo_version_expr,
        );
    }
    build_driver_install_plan_at(examine, read.path(), read.text(), dkms, escalation)
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
    repo_version_expr: &str,
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
    if mentions("ubuntu") && matches!(version_id, "22.04" | "24.04" | "26.04") {
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
            repo_version_expr.to_owned(),
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
            repo_version_expr.to_owned(),
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
            repo_version_expr.to_owned(),
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
    repo_version_expr: String,
    dkms: bool,
    include_linux_modules_extra: bool,
    escalation: PrivilegeEscalation,
) -> DriverInstallPlan {
    // Empty when already root, so no command depends on a `sudo` binary that a
    // container or minimal image very likely does not have.
    let sudo = escalation.prefix();
    let mut commands = Vec::new();
    if dkms {
        // AMD's page says to remove an AMDGPU driver from a previous
        // installation first. Removal is destructive, so the plan only
        // detects and warns; the user decides.
        commands.push(driver_command(
            DriverCommandPhase::Prepare,
            &format!(
                "if dpkg-query -W -f='${{Status}}' amdgpu-dkms 2>/dev/null | grep -q 'install ok installed'; then echo 'warning: amdgpu-dkms from a previous installation is installed and was not removed. AMD advises removing it first (`{sudo}apt-get autoremove -y amdgpu-dkms`); continuing anyway. If the install fails, remove it and re-run.'; fi"
            ),
        ));
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
                    "curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | {sudo}gpg --batch --yes --dearmor -o /etc/apt/keyrings/rocm.gpg"
                ),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!(
                    "printf '%s\\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/{repo_version_expr}/ubuntu {codename} main' | {sudo}tee /etc/apt/sources.list.d/amdgpu.list >/dev/null"
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
        repo_version_expr,
        reason: if dkms {
            "Plan uses AMD's package-manager DKMS flow and requires explicit approval before execution. If amdgpu-dkms from a previous installation is present, the first command only warns; the install continues and nothing is removed automatically."
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
    repo_version_expr: String,
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
            &format!("{sudo}rpm --import https://repo.radeon.com/rocm/rocm.gpg.key"),
        ));
        commands.push(driver_command(
            DriverCommandPhase::Prepare,
            &format!(
                "printf '%s\\n' '[amdgpu]' 'name=amdgpu' 'baseurl={}' 'enabled=1' 'priority=50' 'gpgcheck=1' 'gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key' | {sudo}tee /etc/yum.repos.d/amdgpu.repo >/dev/null",
                dnf_repo_baseurl(&repo_version_expr, &version_id, distro)
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
        repo_version_expr,
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
    repo_version_expr: String,
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
                &format!("{sudo}zypper --non-interactive refresh"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}zypper --non-interactive install -y kernel-default-devel"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("{sudo}rpm --import https://repo.radeon.com/rocm/rocm.gpg.key"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!(
                    "printf '%s\\n' '[amdgpu]' 'name=amdgpu' 'baseurl={}' 'enabled=1' 'autorefresh=0' 'type=rpm-md' 'gpgcheck=1' 'gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key' | {sudo}tee /etc/zypp/repos.d/amdgpu.repo >/dev/null",
                    sles_repo_baseurl(&repo_version_expr, &version_id)
                ),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                // `--gpg-auto-import-keys` is required here: `rpm --import` above
                // only seeds the RPM keyring, but zypper tracks trusted
                // *repository* signing keys separately and otherwise prompts to
                // trust/reject the new repo's key on this refresh -- a prompt
                // that would hang this unattended flow (no stdin attached).
                &format!("{sudo}zypper --non-interactive --gpg-auto-import-keys refresh"),
            ),
            driver_command(
                DriverCommandPhase::Execute,
                &format!("{sudo}zypper --non-interactive install -y amdgpu-dkms"),
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
        repo_version_expr,
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

fn dnf_repo_baseurl(repo_version_expr: &str, version_id: &str, distro: DnfDriverDistro) -> String {
    let repo_family = match distro {
        DnfDriverDistro::Rhel => "rhel",
        DnfDriverDistro::Oracle | DnfDriverDistro::Rocky | DnfDriverDistro::Generic => "el",
    };
    let repo_version = dnf_repo_version_path(version_id);
    format!(
        "https://repo.radeon.com/amdgpu/{repo_version_expr}/{repo_family}/{repo_version}/main/x86_64"
    )
}

fn sles_repo_baseurl(repo_version_expr: &str, version_id: &str) -> String {
    format!("https://repo.radeon.com/amdgpu/{repo_version_expr}/sle/{version_id}/main/x86_64")
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

fn codename_for_version(os_id: &str, version_id: &str) -> Option<&'static str> {
    match (os_id, version_id) {
        ("ubuntu", "22.04") => Some("jammy"),
        ("ubuntu", "24.04") => Some("noble"),
        ("ubuntu", "26.04") => Some("resolute"),
        ("debian", "12") => Some("jammy"),
        ("debian", "13") => Some("noble"),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(unix)]
    use std::fs;
    use std::path::PathBuf;
    #[cfg(unix)]
    use std::process::Command as ProcessCommand;

    use super::rocdxg::scoped_rocdxg_env;
    use super::*;
    use crate::tests::ScopedTestEnv;

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
    fn is_safe_shell_token_rejects_quote_and_semicolon_payloads() {
        // The hostile-codename/version-id/os-id tests above only ever send
        // `$(id)` through this filter, which proves the metacharacter
        // denylist works for *that* payload -- but a widened character class
        // that happened to still reject `$`, `(`, `)` while now accepting
        // `'`/`;` would pass every one of them undetected. `'` breaks out of
        // this value's enclosing single-quoted shell string outright, and
        // `;` chains a second command once out, so both must be checked
        // directly against the function itself, not only indirectly.
        assert!(!is_safe_shell_token("6.2.1'"));
        assert!(!is_safe_shell_token("6.2.1;rm"));
        assert!(!is_safe_shell_token("';rm -rf /;'"));
        // `/` cannot break out of the quotes, but it would let the override
        // rewrite the repo URL path.
        assert!(!is_safe_shell_token("6.2.1/../x"));
        assert!(is_safe_shell_token("6.2.1"));
    }

    pub(crate) fn test_examine(os: &str, wsl: bool) -> ExamineSummary {
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
                windows: None,
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

    /// `text` as if read from `/etc/os-release`, for tests that only care
    /// about the content and not the path — `rocm_core::hardware_root::OsRelease::found`
    /// is `pub(crate)` inside `rocm-core`, so it isn't visible from this crate's
    /// tests; this is the `apps/rocm`-side equivalent, used in place of
    /// repeating the `OsRelease::Found { .. }` literal at every call site.
    pub(crate) fn os_release_found(text: &str) -> rocm_core::hardware_root::OsRelease {
        rocm_core::hardware_root::OsRelease::Found {
            path: "/etc/os-release",
            text: text.to_owned(),
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
        build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            escalation,
        )
        .commands
        .into_iter()
        .map(|command| command.command)
        .collect()
    }

    /// Rewrite every `KEY=value` / `KEY="value"` line of an os-release fixture
    /// in single quotes, which `os-release(5)` permits just as it does double.
    fn single_quoted(os_release: &str) -> String {
        os_release
            .lines()
            .map(|line| match line.split_once('=') {
                Some((key, value)) => format!("{key}='{}'", value.trim_matches('"')),
                None => line.to_owned(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// How a distro quotes its `/etc/os-release` must not change the plan.
    ///
    /// `os-release(5)` allows single or double quotes, so every supported
    /// distro, rewritten in single quotes, must plan exactly as it does in
    /// double quotes.
    #[test]
    fn a_single_quoted_os_release_builds_the_same_plan() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        for (label, os_release) in dkms_planning_os_releases() {
            let plan = |text: &str| {
                build_driver_install_plan_read(
                    &test_examine("linux", false),
                    &os_release_found(text),
                    true,
                    PrivilegeEscalation::Sudo,
                )
            };
            let double = plan(os_release);
            let single = plan(&single_quoted(os_release));
            assert!(
                single.supported,
                "{label}: unsupported when single-quoted: {}\n{}",
                single.reason,
                single_quoted(os_release)
            );
            assert_eq!(
                (single.os_id.as_str(), single.version_id.as_str()),
                (double.os_id.as_str(), double.version_id.as_str()),
                "{label}: quote style changed the distro read"
            );
            assert_eq!(
                single.execution_commands(),
                double.execution_commands(),
                "{label}: quote style changed the planned commands"
            );
        }
    }

    /// An unreadable os-release is refused as itself: the reason names the
    /// offending line, and it is asserted together with what it claims — no
    /// distro, no commands, nothing that mutates — and against an "unsupported
    /// distro" reason, which would blame the distro instead.
    #[test]
    fn an_unreadable_os_release_names_its_line_and_plans_nothing() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(
                "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\nunset ID\n",
            ),
            true,
            PrivilegeEscalation::Sudo,
        );
        assert_eq!(
            plan.reason,
            "/etc/os-release line 4 is not a plain assignment: \"unset ID\"; no driver \
             commands were planned because the distro cannot be read. To fix it, remove \
             that line or rewrite it as NAME=value."
        );
        assert!(!plan.supported);
        assert!(plan.commands.is_empty(), "{:?}", plan.commands);
        assert_eq!(plan.policy, "unreadable_os_release");
        // Nothing about the distro is claimed — not the `ubuntu`, `24.04` or
        // `noble` the file assigns before the command — and the plan changes
        // nothing, so it neither mutates nor needs a reboot. The one check it
        // offers is the command that names the line.
        assert_eq!(
            (
                plan.os_id.as_str(),
                plan.version_id.as_str(),
                plan.codename.as_str()
            ),
            ("", "", "")
        );
        assert!(!plan.mutating);
        assert!(!plan.reboot_required);
        assert!(
            plan.preflight_checks.is_empty(),
            "{:?}",
            plan.preflight_checks
        );
        assert_eq!(plan.checks, vec!["rocm examine".to_owned()]);

        // The rendered plan says so on one line, and does not also claim the
        // distro is unsupported.
        let rendered = render_driver_install_plan(&plan, false, true);
        assert!(
            rendered.contains("reason: /etc/os-release line 4 is not a plain assignment"),
            "{rendered}"
        );
        assert!(!rendered.contains("AMD-documented"), "{rendered}");
        assert!(
            rendered.contains("execution_commands: <none>"),
            "{rendered}"
        );
    }

    /// `build_driver_install_plan_read` is the one seam `install_driver` hands
    /// its `real_os_release()` result to. Both are now one-line delegations
    /// to `hardware_root::read_os_release`, so reverting either back to a raw
    /// `/etc/os-release` read is a type error here, not a silent regression.
    /// This still pins the behaviour at the seam itself: an `OsRelease` that
    /// says the fallback path was read plans exactly what a primary-path read
    /// with the same content would.
    #[test]
    fn a_fallback_read_plans_exactly_like_a_primary_read() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let examine = test_examine("linux", false);
        let fallback = rocm_core::hardware_root::OsRelease::Found {
            path: "/usr/lib/os-release",
            text: "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n".to_owned(),
        };
        let primary = rocm_core::hardware_root::OsRelease::Found {
            path: "/etc/os-release",
            text: "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n".to_owned(),
        };

        let from_fallback =
            build_driver_install_plan_read(&examine, &fallback, true, PrivilegeEscalation::Sudo);
        let from_primary =
            build_driver_install_plan_read(&examine, &primary, true, PrivilegeEscalation::Sudo);
        assert_eq!(from_fallback.os_id, "ubuntu");
        assert_eq!(
            (
                from_fallback.os_id.clone(),
                from_fallback.version_id.clone(),
                from_fallback.codename.clone(),
                from_fallback.reason.clone(),
                from_fallback.execution_commands(),
            ),
            (
                from_primary.os_id.clone(),
                from_primary.version_id.clone(),
                from_primary.codename.clone(),
                from_primary.reason.clone(),
                from_primary.execution_commands(),
            ),
        );
    }

    /// A file that exists but could not be read plans nothing and names
    /// itself, distinctly from both an unreadable *line* (which still comes
    /// from a file that was read, see
    /// [`an_unreadable_os_release_names_its_line_and_plans_nothing`]) and from
    /// an unsupported distribution — which would send the user looking for a
    /// support gap instead of a filesystem problem.
    #[test]
    fn an_unreadable_file_plans_nothing_and_names_itself() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let read = rocm_core::hardware_root::OsRelease::Unreadable {
            path: "/usr/lib/os-release",
            error: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        };
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &read,
            true,
            PrivilegeEscalation::Sudo,
        );
        assert_eq!(plan.policy, "unreadable_os_release");
        assert!(!plan.supported);
        assert!(plan.commands.is_empty(), "{:?}", plan.commands);
        assert_eq!(
            plan.reason,
            "/usr/lib/os-release could not be read: denied; no driver commands were planned \
             because the distro cannot be read."
        );
    }

    /// The unreadable-os-release branch resolves `repo_version_expr` via the
    /// same [`resolve_repo_version_expr`] call the normal path uses, before
    /// it ever builds an `unreadable_os_release_plan`. A malformed override
    /// must still be refused here, not silently dropped because the distro
    /// itself could not be identified.
    #[test]
    fn an_unreadable_os_release_still_refuses_a_malformed_version_override() {
        let mut env = ScopedTestEnv::with_amd_overrides_cleared();
        env.set(
            "ROCM_CLI_AMDGPU_DRIVER_VERSION",
            "'; touch /tmp/pwned; echo '",
        );
        let read = rocm_core::hardware_root::OsRelease::Unreadable {
            path: "/usr/lib/os-release",
            error: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        };
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &read,
            true,
            PrivilegeEscalation::Sudo,
        );
        assert!(!plan.supported);
        assert_eq!(plan.policy, "invalid_driver_version_override");
        assert!(
            !plan.mutating,
            "a refused override must not claim to mutate"
        );
        assert!(plan.reason.contains("ROCM_CLI_AMDGPU_DRIVER_VERSION"));
        assert!(plan.commands.is_empty());
    }

    /// The same malformed line as
    /// [`an_unreadable_os_release_names_its_line_and_plans_nothing`], but read
    /// through the fallback path: the reason must name `/usr/lib/os-release`,
    /// not `/etc/os-release`. That test only exercises
    /// [`os_release_found`]'s hardcoded `/etc/os-release` path, so a
    /// regression that hardcoded the path back into
    /// `build_driver_install_plan_read` would still pass it.
    #[test]
    fn a_fallback_reads_malformed_line_names_the_fallback_path() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let read = rocm_core::hardware_root::OsRelease::Found {
            path: "/usr/lib/os-release",
            text: "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\nunset ID\n".to_owned(),
        };
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &read,
            true,
            PrivilegeEscalation::Sudo,
        );
        assert_eq!(
            plan.reason,
            "/usr/lib/os-release line 4 is not a plain assignment: \"unset ID\"; no driver \
             commands were planned because the distro cannot be read. To fix it, remove \
             that line or rewrite it as NAME=value."
        );
        assert!(!plan.supported);
        assert!(plan.commands.is_empty(), "{:?}", plan.commands);
        assert_eq!(plan.policy, "unreadable_os_release");
    }

    /// On WSL, `build_driver_install_plan_read`'s `Unreadable` branch must
    /// not fire at all: WSL has its own guest-side os-release probe (see
    /// `examine/wsl.rs`'s `WSL_REMOTE_PROBE`), so the local Rust-side read this
    /// function receives is never authoritative for a WSL host, and the plan
    /// must stay the ordinary `wsl_rocdxg` one — not `unreadable_os_release` —
    /// regardless of what it says. Without the `reaches_os_release_parse`
    /// guard this would plan nothing on a WSL host whose local read happens
    /// to come back `Unreadable`, even though the guest-side probe may read
    /// the distro just fine.
    #[test]
    fn a_wsl_host_ignores_an_unreadable_local_os_release_read() {
        let _env = scoped_rocdxg_env();
        let read = rocm_core::hardware_root::OsRelease::Unreadable {
            path: "/usr/lib/os-release",
            error: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        };
        let plan = build_driver_install_plan_read(
            &test_examine("linux", true),
            &read,
            true,
            PrivilegeEscalation::Sudo,
        );
        assert_eq!(plan.policy, "wsl_rocdxg");
        assert!(plan.supported);
        assert!(plan.mutating);
    }

    /// The Windows half of the same `reaches_os_release_parse` guard: a
    /// native Windows host's driver install is `windows_validate_only`
    /// regardless of what the local read says, since the Linux driver
    /// install this plan guards is never attempted there. Without this half
    /// of the guard, a Windows host whose `real_os_release()` happens to come
    /// back `Unreadable` would wrongly take the Linux-only
    /// `unreadable_os_release` policy instead.
    #[test]
    fn a_windows_host_ignores_an_unreadable_local_os_release_read() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let read = rocm_core::hardware_root::OsRelease::Unreadable {
            path: "/usr/lib/os-release",
            error: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        };
        let plan = build_driver_install_plan_read(
            &test_examine("windows", false),
            &read,
            true,
            PrivilegeEscalation::Sudo,
        );
        assert_eq!(plan.policy, "windows_validate_only");
        assert!(!plan.supported);
        assert!(!plan.mutating);
    }

    /// With a duplicated key, the plan is built from the last assignment — the
    /// one a shell keeps, and the one `rocm examine` reports, since both read
    /// through `rocm_core::os_release`.
    #[test]
    fn a_duplicated_os_release_key_is_planned_from_its_last_assignment() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let text = "ID=debian\nVERSION_ID=\"12\"\nID=ubuntu\nVERSION_ID=\"24.04\"\n\
                    VERSION_CODENAME=noble\n";
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(text),
            true,
            PrivilegeEscalation::Sudo,
        );
        assert_eq!(
            (plan.os_id.as_str(), plan.version_id.as_str()),
            ("ubuntu", "24.04"),
            "planned for the first assignment instead of the last"
        );
        assert!(plan.supported, "{}", plan.reason);
    }

    #[test]
    fn driver_plan_as_root_never_emits_sudo() {
        // The defect: every command was prefixed `sudo` unconditionally, so on a
        // root host without the binary the first one died with `sudo: not found`
        // before any driver work. This asserts the ABSENCE of `sudo` across every
        // distro rather than checking known commands one by one — a templating
        // site missed on some distro fails here instead of shipping.
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
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
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        for (label, os_release) in dkms_planning_os_releases() {
            let plan = build_driver_install_plan_read(
                &test_examine("linux", false),
                &os_release_found(os_release),
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
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let ubuntu = "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n";
        let commands = plan_commands(ubuntu, PrivilegeEscalation::AlreadyRoot);
        assert!(
            commands
                .iter()
                .any(|command| command.contains("| tee /etc/apt/sources.list.d/amdgpu.list")),
            "the apt-source pipeline must still tee, unprefixed: {commands:?}"
        );
        assert!(
            commands.iter().any(|command| command
                .contains("| gpg --batch --yes --dearmor -o /etc/apt/keyrings/rocm.gpg")),
            "the keyring pipeline must still call gpg, unprefixed: {commands:?}"
        );
    }

    #[test]
    fn driver_plan_records_the_commands_it_will_actually_run() {
        // `execution_commands()` is what lands in state.json. It must agree with
        // the escalation the plan was built under, or the recorded history
        // describes commands that never ran.
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let ubuntu = "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n";
        let as_root = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(ubuntu),
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
        let off_root = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(ubuntu),
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
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        for (label, os_release) in dkms_planning_os_releases() {
            let as_root = build_driver_install_plan_read(
                &test_examine("linux", false),
                &os_release_found(os_release),
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
            let off_root = build_driver_install_plan_read(
                &test_examine("linux", false),
                &os_release_found(os_release),
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
    fn driver_plan_dry_run_repo_version_line_is_resolved() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // Regression for the dry-run output leaking the wrong field (or, under
        // the old shell-template scheme this replaced, a raw unresolved
        // placeholder) on the `repo_version:` line instead of the version the
        // plan actually resolved.
        let os_release = r#"
ID=rhel
VERSION_ID="9.7"
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(rendered.contains("  repo_version: latest"), "{rendered}");
    }

    #[test]
    fn driver_plan_ubuntu_2404_uses_official_dkms_commands() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
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
                .any(|command| command.contains("repo.radeon.com/amdgpu"))
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
    fn driver_plan_ubuntu_2604_plans_the_dkms_flow_with_the_resolute_suite() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ubuntu
VERSION_ID="26.04"
VERSION_CODENAME=resolute
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
        assert_eq!(plan.codename, "resolute");
        assert!(rendered.contains("https://repo.radeon.com/amdgpu/latest/ubuntu resolute main"));
        assert!(rendered.contains("Execute: sudo apt-get install -y amdgpu-dkms"));
        assert!(plan.reboot_required);
    }

    #[test]
    fn driver_plan_ubuntu_2604_root_drops_sudo() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ubuntu
VERSION_ID="26.04"
VERSION_CODENAME=resolute
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::AlreadyRoot,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(rendered.contains("Execute: apt-get install -y amdgpu-dkms"));
        assert!(plan.commands.iter().all(|c| !c.command.contains("sudo")));
    }

    #[test]
    fn driver_plan_ubuntu_2604_falls_back_to_resolute_codename() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // No VERSION_CODENAME/UBUNTU_CODENAME in os-release: codename_for_version
        // must supply "resolute" on its own.
        let os_release = r#"
ID=ubuntu
VERSION_ID="26.04"
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert_eq!(plan.codename, "resolute");
        assert!(plan.supported);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
    }

    #[test]
    fn driver_plan_apt_warns_about_a_previous_amdgpu_dkms_without_removing_it() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let first = &plan
            .commands
            .first()
            .expect("apt plan has commands")
            .command;

        // Detection runs before anything else, and only echoes: removal stays
        // the user's call. Behavior is proven end to end by running the
        // command against a stubbed dpkg-query in
        // `previous_driver_warning_fires_only_when_amdgpu_dkms_is_installed`.
        assert!(first.contains("dpkg-query"));
        assert!(first.contains("sudo apt-get autoremove -y amdgpu-dkms"));
        assert!(
            !plan
                .commands
                .iter()
                .skip(1)
                .any(|c| c.command.contains("autoremove"))
        );
        assert!(plan.reason.contains("only warns"));
        assert!(first.contains("continuing anyway"));
        assert!(
            !plan
                .preflight_checks
                .iter()
                .any(|check| check.contains("amdgpu-dkms"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn previous_driver_warning_fires_only_when_amdgpu_dkms_is_installed() {
        use std::os::unix::fs::PermissionsExt;
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = "ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n";
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::AlreadyRoot,
        );
        let command = &plan.commands[0].command;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let run = |status: &str| {
            let stub = root.join("dpkg-query");
            fs::write(&stub, format!("#!/bin/sh\nprintf '%s' '{status}'\n")).unwrap();
            fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).unwrap();
            let out = ProcessCommand::new("sh")
                .args(["-c", command])
                .env("PATH", format!("{}:/usr/bin:/bin", root.display()))
                .output()
                .unwrap();
            assert!(out.status.success());
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        assert!(run("install ok installed").contains("warning: amdgpu-dkms"));
        assert!(run("").is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn driver_plan_rejects_a_non_utf8_version_override() {
        // `std::env::var` reports a non-UTF-8 value as `Err(VarError::NotUnicode)`,
        // which `resolve_repo_version_expr` must not fold into the
        // `NotPresent`/empty "unset" case -- that would silently use "latest"
        // for a user who explicitly set this to something.
        use std::os::unix::ffi::OsStrExt;
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let invalid = std::ffi::OsStr::from_bytes(&[0x66, 0xFF, 0x66]);
        #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
        unsafe {
            std::env::set_var("ROCM_CLI_AMDGPU_DRIVER_VERSION", invalid);
        }
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert!(!plan.supported);
        assert_eq!(plan.policy, "invalid_driver_version_override");
        assert!(
            !plan.mutating,
            "a refused override must not claim to mutate"
        );
        assert!(plan.reason.contains("ROCM_CLI_AMDGPU_DRIVER_VERSION"));
        assert!(plan.commands.is_empty());
    }

    #[test]
    fn driver_plan_rejects_shell_metacharacters_in_version_override() {
        let mut env = ScopedTestEnv::with_amd_overrides_cleared();
        env.set(
            "ROCM_CLI_AMDGPU_DRIVER_VERSION",
            "'; touch /tmp/pwned; echo '",
        );
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert!(!plan.supported);
        assert_eq!(plan.policy, "invalid_driver_version_override");
        assert!(
            !plan.mutating,
            "a refused override must not claim to mutate"
        );
        assert!(plan.reason.contains("ROCM_CLI_AMDGPU_DRIVER_VERSION"));
        // Already proves no command can contain "pwned": a vacuous check over
        // commands would be redundant with this.
        assert!(plan.commands.is_empty());
    }

    #[test]
    fn driver_plan_rejects_a_space_free_shell_injection_payload() {
        // The metacharacter test above uses a payload full of spaces, which
        // `is_safe_shell_token` already rejects on its own -- so it proves
        // nothing about whether `$`, backticks, or `()` are actually in the
        // denylist. Use a payload with no spaces at all so the rejection can
        // only come from the metacharacters themselves.
        let mut env = ScopedTestEnv::with_amd_overrides_cleared();
        env.set("ROCM_CLI_AMDGPU_DRIVER_VERSION", "$(id)");
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert!(!plan.supported);
        assert_eq!(plan.policy, "invalid_driver_version_override");
        assert!(
            !plan.mutating,
            "a refused override must not claim to mutate"
        );
        assert!(plan.commands.is_empty());
    }

    #[test]
    fn driver_plan_treats_empty_version_override_as_unset() {
        let mut env = ScopedTestEnv::with_amd_overrides_cleared();
        env.set("ROCM_CLI_AMDGPU_DRIVER_VERSION", "");
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert_ne!(plan.policy, "invalid_driver_version_override");
        assert_eq!(plan.repo_version_expr, "latest");
    }

    #[test]
    fn driver_plan_accepts_a_valid_version_override_and_uses_it_in_the_url() {
        let mut env = ScopedTestEnv::with_amd_overrides_cleared();
        env.set("ROCM_CLI_AMDGPU_DRIVER_VERSION", "30.10");
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert_eq!(plan.repo_version_expr, "30.10");
        assert!(rendered.contains("amdgpu/30.10/"));
    }

    #[test]
    fn driver_plan_falls_back_on_a_hostile_codename() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME='; touch /tmp/pwned; echo '
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert_eq!(plan.codename, "noble");
        assert!(
            plan.commands
                .iter()
                .all(|command| !command.command.contains("pwned"))
        );
    }

    #[test]
    fn driver_plan_falls_back_on_a_malformed_version_id() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04'; touch /tmp/pwned; echo '"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert_eq!(plan.version_id, "");
        assert!(!plan.supported);
        assert!(
            plan.commands
                .iter()
                .all(|command| !command.command.contains("pwned"))
        );
    }

    #[test]
    fn driver_plan_falls_back_on_a_space_free_hostile_codename() {
        // The payload above is full of spaces, which `is_safe_shell_token`
        // already rejects on its own -- it proves nothing about whether the
        // metacharacters themselves are denylisted. Use a space-free payload
        // so rejection can only come from the metacharacters.
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // Single-quoted so the os-release parser (which refuses an unquoted
        // `$`/`(`/`)` outright, making the whole file unreadable rather than
        // just this field) accepts the line and hands `$(id)` to
        // `is_safe_shell_token` to reject on its own merits.
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME='$(id)'
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert_eq!(plan.codename, "noble");
        assert!(
            plan.commands
                .iter()
                .all(|command| !command.command.contains("$(id)"))
        );
    }

    #[test]
    fn driver_plan_falls_back_on_a_space_free_malformed_version_id() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // Single-quoted for the same reason as the codename test above: a
        // bare/double-quoted `$(id)` makes the whole os-release file
        // unreadable, which would pass this test's assertions (empty
        // version_id, unsupported) without ever reaching `is_safe_shell_token`.
        let os_release = r"
ID=ubuntu
VERSION_ID='$(id)'
VERSION_CODENAME=noble
";
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert_eq!(plan.version_id, "");
        assert!(!plan.supported);
        assert!(
            plan.commands
                .iter()
                .all(|command| !command.command.contains("$(id)"))
        );
    }

    #[test]
    fn driver_plan_falls_back_on_a_space_free_hostile_os_id() {
        // Closes the gap where `os_id`'s `is_safe_shell_token` filter (the
        // `field("ID").filter(...)` call above `codename_for_version`) had no
        // dedicated test at all: deleting that filter left every other test
        // green, since none of them feed a hostile `ID` value.
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID='$(id)'
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );

        assert_eq!(plan.os_id, "");
        assert_eq!(plan.policy, "unsupported_linux_dkms_plan");
        assert!(!plan.supported);
        assert!(
            plan.commands
                .iter()
                .all(|command| !command.command.contains("$(id)"))
        );
    }

    #[test]
    fn driver_plan_default_linux_preflight_has_no_execution_commands() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
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
    fn driver_plan_debian_12_omits_linux_modules_extra() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=debian
VERSION_ID="12"
VERSION_CODENAME=bookworm
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
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
        assert!(rendered.contains("sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key"));
        assert!(
            rendered.contains("baseurl=https://repo.radeon.com/amdgpu/latest/rhel/9.7/main/x86_64")
        );
        assert!(rendered.contains("gpgcheck=1"));
        assert!(rendered.contains("/etc/yum.repos.d/amdgpu.repo"));
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(plan.supported);
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("kernel-uek-devel-$(uname -r)"));
        // `rpm --import` is pushed once after the distro-specific match, not
        // inside the RHEL/Generic arm -- Oracle's flow must get it too.
        assert!(rendered.contains("sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key"));
        assert!(
            rendered.contains("baseurl=https://repo.radeon.com/amdgpu/latest/el/10/main/x86_64")
        );
        assert!(rendered.contains("dry run only"));
    }

    #[test]
    fn driver_plan_rocky_97_uses_el_dnf_flow() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=rocky
VERSION_ID="9.7"
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(
            rendered
                .contains("sudo dnf install -y kernel-headers kernel-devel kernel-devel-matched")
        );
        // Rocky is `Generic`, not `Rhel`, in `DnfDriverDistro` -- confirms the
        // `rpm --import` push is unconditional across the dnf match arms, not
        // only wired for the `Rhel`/`Oracle` arms the other two tests cover.
        assert!(rendered.contains("sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key"));
        assert!(
            rendered.contains("baseurl=https://repo.radeon.com/amdgpu/latest/el/9.7/main/x86_64")
        );
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert!(rendered.contains("sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key"));
        assert!(
            rendered.contains("baseurl=https://repo.radeon.com/amdgpu/latest/el/9.4/main/x86_64")
        );
        assert!(rendered.contains("Execute: sudo dnf install -y amdgpu-dkms"));
    }

    #[test]
    fn driver_plan_rocky_8_and_10_remain_unsupported() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        // AMD documents Rocky Linux 9 only; keep the driver matrix scoped to 9.x.
        for version in ["8.10", "10.0"] {
            let os_release = format!("\nID=rocky\nVERSION_ID=\"{version}\"\n");
            let plan = build_driver_install_plan_read(
                &test_examine("linux", false),
                &os_release_found(&os_release),
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
        // Ubuntu-suite amdgpu tree (Debian 12 -> jammy). Lock that in and
        // ensure the plan explains the mapping is intentional.
        let os_release = r#"
ID=debian
VERSION_ID="12"
VERSION_CODENAME=bookworm
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(plan.supported);
        assert_eq!(plan.codename, "jammy");
        assert!(rendered.contains("https://repo.radeon.com/amdgpu/latest/ubuntu jammy main"));
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(rendered.contains("`zypper` package manager is available"));
        assert!(rendered.contains("SUSEConnect"));
        assert!(rendered.contains("sle-module-desktop-applications/15.7/x86_64"));
        assert!(rendered.contains("sudo zypper --non-interactive install -y kernel-default-devel"));
        assert!(rendered.contains("sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key"));
        assert!(
            rendered.contains("baseurl=https://repo.radeon.com/amdgpu/latest/sle/15.7/main/x86_64")
        );
        assert!(rendered.contains("gpgcheck=1"));
        assert!(rendered.contains("/etc/zypp/repos.d/amdgpu.repo"));
        assert!(
            rendered.contains("sudo zypper --non-interactive --gpg-auto-import-keys refresh"),
            "the post-repo-add refresh must auto-trust the new repo's signing key, \
             or zypper would prompt (and hang this unattended flow): {rendered}"
        );
        assert!(rendered.contains("Execute: sudo zypper --non-interactive install -y amdgpu-dkms"));
        assert!(rendered.contains("approval: required"));
    }

    #[test]
    fn sles_repo_baseurl_uses_the_given_repo_version_expr_not_a_hardcoded_latest() {
        // Mutation-proofs the parameter wiring itself: a mutation that
        // hardcoded `latest` into this function's body left every existing
        // SLES test green because their default env override also resolves
        // to `latest`, so none of them could tell "wired correctly" apart
        // from "hardcoded to the same value the default happens to be".
        assert_eq!(
            sles_repo_baseurl("6.2.1", "15.7"),
            "https://repo.radeon.com/amdgpu/6.2.1/sle/15.7/main/x86_64"
        );
        assert_eq!(
            sles_repo_baseurl("latest", "15.7"),
            "https://repo.radeon.com/amdgpu/latest/sle/15.7/main/x86_64"
        );
    }

    #[test]
    fn driver_plan_dnf_overrides_the_repo_version_via_the_driver_version_env() {
        let mut env = ScopedTestEnv::with_amd_overrides_cleared();
        env.set("ROCM_CLI_AMDGPU_DRIVER_VERSION", "30.10");
        let os_release = "ID=rhel\nVERSION_ID=\"9.7\"\n";
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(
            rendered.contains("amdgpu/30.10/rhel/9.7/"),
            "an overridden ROCM_CLI_AMDGPU_DRIVER_VERSION must reach the dnf repo baseurl, \
             not a hardcoded `latest`: {rendered}"
        );
    }

    #[test]
    fn driver_plan_id_like_derivatives_honour_the_driver_version_env() {
        // (os-release, expected repo path). Each derivative arm of
        // `driver_plan_via_id_like` must thread the override, not `latest`.
        let cases = [
            (
                "ID=almalinux\nID_LIKE=\"rhel centos fedora\"\nVERSION_ID=\"9.6\"\n",
                "amdgpu/30.10/el/9.6",
            ),
            (
                "ID=pop\nID_LIKE=\"ubuntu debian\"\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n",
                "amdgpu/30.10/ubuntu",
            ),
            (
                "ID=devuan\nID_LIKE=debian\nVERSION_ID=\"12\"\n",
                "amdgpu/30.10/ubuntu",
            ),
        ];
        for (os_release, want) in cases {
            let mut env = ScopedTestEnv::with_amd_overrides_cleared();
            env.set("ROCM_CLI_AMDGPU_DRIVER_VERSION", "30.10");
            let plan = build_driver_install_plan_read(
                &test_examine("linux", false),
                &os_release_found(os_release),
                true,
                PrivilegeEscalation::Sudo,
            );
            let rendered = render_driver_install_plan(&plan, false, false);
            assert!(plan.supported, "{os_release}");
            assert!(rendered.contains(want), "{os_release}: {rendered}");
            assert!(
                !rendered.contains("amdgpu/latest"),
                "{os_release}: {rendered}"
            );
        }
    }

    #[test]
    fn driver_plan_refuses_a_retired_version_pin_variable() {
        for retired in RETIRED_DRIVER_VERSION_VARS {
            let mut env = ScopedTestEnv::with_amd_overrides_cleared();
            env.set(retired, "7.2.4");
            let plan = build_driver_install_plan_read(
                &test_examine("linux", false),
                &os_release_found("ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n"),
                true,
                PrivilegeEscalation::Sudo,
            );

            assert!(!plan.supported && !plan.mutating, "{retired}");
            assert_eq!(plan.policy, "retired_driver_version_override");
            assert!(plan.reason.contains("ROCM_CLI_AMDGPU_DRIVER_VERSION"));
        }
    }

    #[test]
    fn driver_plan_support_matches_rocm_core_diagnose() {
        // `rocm-core` cannot call the plan builder, so diagnose keeps its own
        // copy of the support matrix (`dkms_driver_plan_is_supported`); this
        // fails if the two disagree on any host.
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let hosts = [
            ("ubuntu", "22.04", "debian"),
            ("ubuntu", "24.04", "debian"),
            ("Ubuntu", "24.04", "debian"),
            ("ubuntu", "26.04", "debian"),
            ("debian", "12", ""),
            ("debian", "13", ""),
            ("rhel", "9.7", "fedora"),
            ("rhel", "10.0", "fedora"),
            ("ol", "9.7", "fedora"),
            ("ol", "9.7", "rhel fedora"),
            ("ol", "9.5", "rhel fedora"),
            ("rocky", "9.6", "rhel centos fedora"),
            ("almalinux", "9.6", "rhel centos fedora"),
            ("sles", "15.7", "suse"),
            ("sle", "15.7", "suse"),
            ("pop", "24.04", "ubuntu debian"),
            ("linuxmint", "22", "ubuntu debian"),
            ("fedora", "41", ""),
            ("opensuse-leap", "15.7", "suse opensuse"),
        ];
        for (id, version, like) in hosts {
            let os_release = format!(
                "ID={id}\nVERSION_ID=\"{version}\"\nID_LIKE=\"{like}\"\nVERSION_CODENAME=noble\n"
            );
            let plan = build_driver_install_plan_read(
                &test_examine("linux", false),
                &os_release_found(&os_release),
                true,
                PrivilegeEscalation::Sudo,
            );
            let host = rocm_core::examine::Examination {
                distro_id: id.to_owned(),
                distro_version: version.to_owned(),
                distro_id_like: like.to_owned(),
                ..rocm_core::examine::Examination::default()
            };
            assert_eq!(
                plan.supported,
                rocm_core::diagnose::dkms_driver_plan_is_supported(&host),
                "{id} {version} (ID_LIKE={like}): plan policy {}",
                plan.policy
            );
        }
    }

    #[test]
    fn driver_plan_sles_overrides_the_repo_version_via_the_driver_version_env() {
        let mut env = ScopedTestEnv::with_amd_overrides_cleared();
        env.set("ROCM_CLI_AMDGPU_DRIVER_VERSION", "6.2.1");
        let os_release = r#"
ID=sles
VERSION_ID="15.7"
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(
            rendered.contains("baseurl=https://repo.radeon.com/amdgpu/6.2.1/sle/15.7/main/x86_64"),
            "an overridden ROCM_CLI_AMDGPU_DRIVER_VERSION must reach the SLES repo baseurl, \
             not a hardcoded `latest`: {rendered}"
        );
    }

    #[test]
    fn driver_plan_unsupported_linux_is_non_mutating() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let os_release = r#"
ID=fedora
VERSION_ID="41"
"#;
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
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

    /// AGENTS.md section 3: the planners' wording is asserted together with what
    /// the command does on that platform. On Windows the command is
    /// `windows_validate_only` (no execution), so the plan must be read-only,
    /// carry no `--yes` and not call itself disruptive; elsewhere the command
    /// mutates, so the plan keeps approval, `--yes` and the warning.
    #[test]
    fn planners_describe_install_driver_as_the_command_behaves_on_each_platform() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        for os in ["windows", "linux"] {
            let only_reports = install_driver_only_reports_on(os);
            let plan = build_driver_install_plan_at(
                &test_examine(os, false),
                "/etc/os-release",
                "",
                true,
                PrivilegeEscalation::Sudo,
            );
            // The planners and the command agree on the platform decision.
            assert_eq!(only_reports, plan.policy == "windows_validate_only", "{os}");
            if only_reports {
                assert!(!plan.mutating, "{os}");
            }

            let freeform = crate::freeform_planner::plan_driver_request(
                "install the driver",
                "install the driver",
                only_reports,
            );
            let action = freeform.actions.last().expect("a driver action");
            let chat = crate::chat::chat_install_driver_action(
                vec!["install".to_owned(), "driver".to_owned()],
                only_reports,
            );
            if only_reports {
                assert!(!action.args.iter().any(|a| a == "--yes"), "{action:?}");
                assert!(!action.reason.contains("disruptive"), "{action:?}");
                assert!(!freeform.approval.starts_with("required"));
                match chat {
                    crate::chat::ChatRocmCommandAction::ReadOnly(args) => {
                        assert!(!args.iter().any(|a| a == "--yes"));
                    }
                    crate::chat::ChatRocmCommandAction::Approval { .. } => {
                        panic!("chat must not ask approval for a report-only command")
                    }
                }
            } else {
                assert!(action.args.iter().any(|a| a == "--yes"));
                assert!(action.reason.contains("disruptive"));
                assert_eq!(freeform.approval, "required before driver changes");
                match chat {
                    crate::chat::ChatRocmCommandAction::Approval { args, .. } => {
                        assert!(args.iter().any(|a| a == "--yes"));
                    }
                    crate::chat::ChatRocmCommandAction::ReadOnly(_) => {
                        panic!("chat must ask approval on a mutating platform")
                    }
                }
            }
        }
    }

    /// The chat entry point passes this host's decision to the helper, so a
    /// hard-coded value fails on one of the two platforms the suite runs on.
    #[test]
    fn the_chat_entry_point_follows_the_hosts_install_driver_decision() {
        let action = crate::chat::chat_rocm_command_action_from_args(vec![
            "install".to_owned(),
            "driver".to_owned(),
        ])
        .expect("chat action");
        let reports = std::env::consts::OS == "windows";
        assert_eq!(host_install_driver_only_reports(), reports);
        match action {
            crate::chat::ChatRocmCommandAction::ReadOnly(args) => {
                assert!(reports, "read-only only where the command only reports");
                assert!(!args.iter().any(|a| a == "--yes"));
            }
            crate::chat::ChatRocmCommandAction::Approval { args, .. } => {
                assert!(!reports, "approval only where the command installs");
                assert!(args.iter().any(|a| a == "--yes"));
            }
        }
    }

    #[test]
    fn windows_install_driver_is_validate_only() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let plan = build_driver_install_plan_read(
            &test_examine("windows", false),
            &os_release_found(""),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert_eq!(plan.policy, "windows_validate_only");
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("execution_commands: <none>"));
        // The plan installs nothing, so it asks for no reboot and labels its
        // checks accordingly.
        assert!(!plan.reboot_required);
        assert!(rendered.contains("post_install_checks:"));
        assert!(!rendered.contains("post_reboot"));
        // No driver facts in this fixture: the text says so and points at the
        // commands that exist. Both commands take no flags.
        assert!(rendered.contains("could not be read; run `rocm examine`"));
        assert!(rendered.contains("rocm diagnose"));
        assert!(plan.commands.is_empty());
    }

    fn windows_examine_with_driver(version: &str, date: &str) -> ExamineSummary {
        let mut examine = test_examine("windows", false);
        examine.driver = rocm_core::DriverSummary {
            policy: "windows_validate_only".to_owned(),
            status: "amd_display_driver_detected".to_owned(),
            detail: Some(format!("AMD Radeon(TM) Graphics driver {version}")),
            windows: rocm_core::windows_driver::WindowsDriverInfo::from_probe(version, date),
        };
        examine
    }

    /// The text and the behaviour, asserted together: the plan reports the
    /// driver version, date, verdict and fix id, and the same plan runs
    /// nothing, even when approval is given.
    #[test]
    fn windows_install_driver_reports_an_old_driver_and_installs_nothing() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let examine = windows_examine_with_driver("30.0.13014.8", "2021-08-04");
        let plan = build_driver_install_plan_at(
            &examine,
            "/etc/os-release",
            "",
            true,
            PrivilegeEscalation::Sudo,
        );

        // Behaviour: nothing to execute, however the command is invoked.
        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert!(plan.commands.is_empty());
        assert!(plan.execution_commands().is_empty());
        for (yes, dry_run) in [(false, false), (true, false), (true, true)] {
            let rendered = render_driver_install_plan(&plan, yes, dry_run);
            assert!(
                rendered.contains("execution_commands: <none>"),
                "{rendered}"
            );
            assert!(
                rendered.contains("action: no driver commands will be executed"),
                "{rendered}"
            );
            assert!(!rendered.contains("rerun with --yes"), "{rendered}");

            // Text: what it reports.
            assert!(rendered.contains("version 30.0.13014.8"), "{rendered}");
            assert!(rendered.contains("dated 2021-08-04"), "{rendered}");
            assert!(rendered.contains("outdated: yes"), "{rendered}");
            assert!(
                rendered.contains(rocm_core::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID),
                "{rendered}"
            );
            // The promise that nothing is installed.
            assert!(
                rendered.contains("no driver is installed or changed"),
                "{rendered}"
            );
        }
    }

    #[test]
    fn windows_install_driver_does_not_call_a_current_driver_outdated() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let examine = windows_examine_with_driver("32.0.23033.1002", "2026-05-02");
        let plan = build_driver_install_plan_at(
            &examine,
            "/etc/os-release",
            "",
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, true, false);
        assert!(rendered.contains("version 32.0.23033.1002"), "{rendered}");
        assert!(rendered.contains("outdated: no"), "{rendered}");
        // The `no` is only about the date, so the plan says what the date
        // cannot judge; the state behind it is a report-only plan.
        assert!(
            rendered.contains(rocm_core::windows_driver::FLOOR_NOTE),
            "{rendered}"
        );
        assert!(
            !rendered.contains(rocm_core::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID),
            "{rendered}"
        );
        assert!(plan.commands.is_empty() && !plan.mutating);
    }

    #[test]
    fn windows_install_driver_says_unknown_when_the_date_is_unreadable() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let examine = windows_examine_with_driver("30.0.13014.8", "");
        let plan = build_driver_install_plan_at(
            &examine,
            "/etc/os-release",
            "",
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);
        assert!(rendered.contains("outdated: unknown"), "{rendered}");
        assert!(
            !rendered.contains(rocm_core::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID),
            "{rendered}"
        );
    }

    /// One probe result, read by both commands. The install text and the
    /// diagnose result are built from the same `WindowsDriverInfo`, the value
    /// `rocm examine` reads from the inventory, so a text that names the fix id
    /// is checked against the command it names. Reverting the `reason` wiring in
    /// `build_driver_install_plan` makes the install half fail.
    fn install_text_and_diagnose_ids(
        info: Option<&rocm_core::windows_driver::WindowsDriverInfo>,
    ) -> (String, Vec<String>) {
        let mut examine = test_examine("windows", false);
        examine.driver.windows = info.cloned();
        let plan = build_driver_install_plan_at(
            &examine,
            "/etc/os-release",
            "",
            true,
            PrivilegeEscalation::Sudo,
        );
        let text = render_driver_install_plan(&plan, false, false);

        let mut exam = rocm_core::Examination {
            os_family: "windows".to_owned(),
            ..rocm_core::Examination::default()
        };
        exam.set_windows_driver(info);
        let ids = rocm_core::run_diagnose(&exam, "")
            .matched
            .into_iter()
            .map(|d| d.id)
            .collect();
        (text, ids)
    }

    #[test]
    fn the_fix_id_the_install_text_names_is_the_one_diagnose_raises() {
        let _env = ScopedTestEnv::with_amd_overrides_cleared();
        let id = rocm_core::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID;
        let probe =
            |version, date| rocm_core::windows_driver::WindowsDriverInfo::from_probe(version, date);

        // Old driver: the text names the id, and the command it names raises it.
        let (text, ids) =
            install_text_and_diagnose_ids(probe("30.0.13014.8", "2021-08-04").as_ref());
        assert!(text.contains(id), "{text}");
        assert!(ids.iter().any(|raised| raised == id), "{ids:?}");

        // Current driver and unreadable date: the text does not name it, and
        // diagnose does not raise it.
        for info in [
            probe("32.0.23033.1002", "2026-05-02"),
            probe("30.0.13014.8", ""),
            None,
        ] {
            let (text, ids) = install_text_and_diagnose_ids(info.as_ref());
            assert!(!text.contains(id), "{text}");
            assert!(ids.iter().all(|raised| raised != id), "{ids:?}");
        }
    }

    #[test]
    fn wsl_install_driver_installs_rocdxg_without_dkms() {
        // `dkms: true` is passed deliberately: WSL2 has no kernel module to
        // build, so the flag must not pull in the bare-metal path.
        //
        // The WSL branch reads the three ROCDXG vars — an exported
        // `ROCM_CLI_ROCDXG_VERSION` would steer this plan into a refusal and
        // fail the `plan.supported` assertion below. `ROCM_CLI_AMDGPU_DRIVER_VERSION`
        // is never consulted on this path (the WSL dispatch happens before that
        // var is read), but this reader still takes the combined guard so it
        // isn't the one test that has to remember which half it needs.
        let _env = scoped_rocdxg_env();
        let plan = build_driver_install_plan_read(
            &test_examine("linux", true),
            &os_release_found(""),
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

        let bare_metal = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found("ID=ubuntu\nVERSION_ID=\"24.04\"\nVERSION_CODENAME=noble\n"),
            true,
            PrivilegeEscalation::Sudo,
        );
        assert!(bare_metal.reboot_required);
        assert!(render_driver_install_plan(&bare_metal, false, false).contains("post_reboot"));
    }

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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
        // Ubuntu-family derivatives ship the Ubuntu kernel, so linux-modules-extra applies.
        assert!(rendered.contains("linux-modules-extra-$(uname -r)"));
        assert!(rendered.contains("https://repo.radeon.com/amdgpu/latest/ubuntu jammy main"));
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        // Debian-family maps to the Ubuntu jammy repo and omits linux-modules-extra.
        assert!(rendered.contains("https://repo.radeon.com/amdgpu/latest/ubuntu jammy main"));
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
        // EL rebuilds use the vendor-neutral el/ repo path, not rhel/.
        assert!(
            rendered.contains("baseurl=https://repo.radeon.com/amdgpu/latest/el/9.6/main/x86_64")
        );
        assert!(!rendered.contains("/rhel/9.6/"));
        // el9 uses the version-aware standard-kernel prepare commands.
        assert!(rendered.contains("kernel-devel-matched-$(uname -r)"));
        assert!(rendered.contains("sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key"));
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        // EL 8 is served from the major-version path (el/8), matching AMD docs.
        assert!(
            rendered.contains("baseurl=https://repo.radeon.com/amdgpu/latest/el/8/main/x86_64")
        );
        // el8 has no kernel-devel-matched package.
        assert!(!rendered.contains("kernel-devel-matched"));
        assert!(rendered.contains("kernel-devel-$(uname -r)"));
        assert!(rendered.contains("sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key"));
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
            true,
            PrivilegeEscalation::Sudo,
        );
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(
            rendered.contains("baseurl=https://repo.radeon.com/amdgpu/latest/rhel/9.7/main/x86_64")
        );
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
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
        let plan = build_driver_install_plan_read(
            &test_examine("linux", false),
            &os_release_found(os_release),
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
