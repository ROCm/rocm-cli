// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `rocm install driver` plan model and its rendering: the plan, its
//! commands and their phases, how a command reaches root, the text the plan is
//! printed as, and the helpers every plan builder shares.
//!
//! Relocated from `driver_install.rs` with no behavior change; the moved
//! struct fields are `pub(crate)` because the plan builders, which stay in
//! `driver_install.rs` for now, construct the plan.
//!
//! A leaf: its production code uses only `std` and `rocm_core`.

use std::fmt::Write as _;

use rocm_core::placeholder::empty_as_unknown;

mod rocdxg;

#[cfg(test)]
pub(crate) use rocdxg::scoped_rocdxg_env;
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
    pub(crate) const fn prefix(self) -> &'static str {
        match self {
            Self::Sudo => "sudo ",
            Self::AlreadyRoot => "",
        }
    }

    /// Whether a plan built under this escalation depends on `sudo` being
    /// installed. Drives the preflight list, so it does not claim a
    /// precondition the plan is not relying on.
    pub(crate) const fn needs_sudo_binary(self) -> bool {
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
pub(crate) fn is_safe_shell_token(v: &str) -> bool {
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
pub(crate) fn driver_root_preflight_checks(escalation: PrivilegeEscalation) -> Vec<String> {
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
pub(crate) fn resolve_shell_default_template(expr: &str) -> String {
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

#[cfg(test)]
mod tests {
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
}
