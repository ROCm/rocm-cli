// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Deciding whether a remote machine can host a model, before we ask it to.
//!
//! Four things have to be true: the machine has a GPU stack (ROCm), it has the
//! ROCm CLI to drive it, it has Tailscale so the endpoint can be published, and
//! we know what kind of machine it is so anything we install matches.
//!
//! Every failure here is refused up front rather than discovered halfway
//! through. Starting a model server and *then* finding the endpoint cannot be
//! published leaves a process running on someone's GPU with no way for them to
//! reach it and no record that it exists.

use anyhow::{Result, bail};

use super::transport::Transport;

/// What a remote machine looks like right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RemoteReadiness {
    /// How to invoke the CLI, when it is present.
    pub(crate) cli: Option<String>,
    pub(crate) cli_version: Option<String>,
    pub(crate) rocm_present: bool,
    pub(crate) tailscale_present: bool,
    /// Operating system and CPU architecture, as the machine reports them.
    /// Needed before anything is installed onto it.
    pub(crate) platform: Option<RemotePlatform>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RemotePlatform {
    pub(crate) os: String,
    pub(crate) arch: String,
}

/// Where a provisioned CLI lands, matching the documented manual install path.
pub(crate) const REMOTE_CLI_PATH: &str = "$HOME/.local/bin/rocm";

/// Inspect the remote. Nothing here is an error: absence is a finding, and
/// [`ensure_ready`] is where it becomes a refusal.
pub(crate) fn probe(transport: &dyn Transport) -> Result<RemoteReadiness> {
    // Look on PATH first, then where we would have installed it. A CLI put
    // there by an earlier run is often not on a non-interactive shell's PATH,
    // and missing it would mean provisioning over a perfectly good install.
    let mut cli = None;
    let mut cli_version = None;
    for candidate in ["rocm", REMOTE_CLI_PATH] {
        let outcome = transport.exec(&format!("{candidate} --version"))?;
        if outcome.success {
            cli = Some(candidate.to_owned());
            cli_version = outcome
                .stdout
                .lines()
                .next()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned);
            break;
        }
    }

    // Detect ROCm independently of our own CLI: the machine may have a working
    // GPU stack and no CLI at all, which is a provisioning job, not a refusal.
    let rocm = transport.exec(
        "command -v rocminfo >/dev/null 2>&1 || command -v amd-smi >/dev/null 2>&1 || test -d /opt/rocm",
    )?;
    let tailscale = transport.exec("command -v tailscale >/dev/null 2>&1")?;

    let platform = transport.exec("uname -s && uname -m")?;
    let platform = if platform.success {
        let mut lines = platform.stdout.lines().map(str::trim);
        match (lines.next(), lines.next()) {
            (Some(os), Some(arch)) if !os.is_empty() && !arch.is_empty() => Some(RemotePlatform {
                os: os.to_ascii_lowercase(),
                arch: arch.to_ascii_lowercase(),
            }),
            _ => None,
        }
    } else {
        None
    };

    Ok(RemoteReadiness {
        cli,
        cli_version,
        rocm_present: rocm.success,
        tailscale_present: tailscale.success,
        platform,
    })
}

/// Confirm the remote can host a model, or explain what is missing.
///
/// Returns how to invoke the CLI there.
pub(crate) fn ensure_ready(
    transport: &dyn Transport,
    target: &str,
    channel: &str,
) -> Result<String> {
    ensure_ready_with(transport, target, channel, false)
}

/// As [`ensure_ready`], but able to install ROCm when explicitly asked.
///
/// Installing is opt-in and never implied. It can run for minutes, may need a
/// reboot, and is happening on a machine nobody is looking at — so the default
/// stays "tell the user what is missing" rather than "fix it while they wait".
pub(crate) fn ensure_ready_with(
    transport: &dyn Transport,
    target: &str,
    channel: &str,
    install_rocm: bool,
) -> Result<String> {
    let readiness = probe(transport)?;

    if !readiness.rocm_present {
        if !install_rocm {
            bail!(
                "no ROCm installation was found on {target}.\n\
                 Install it there first:\n  \
                 ssh {target} -- rocm bootstrap setup\n\
                 Or pass --install-rocm to have this command do it."
            );
        }
        // Installing ROCm provisions the CLI first, since the CLI is what runs
        // the install. Return the invocation it settled on rather than falling
        // through to the readiness snapshot below, which was taken *before* any
        // of that and still says there is no CLI — reading it here provisioned
        // the machine a second time, and a hiccup during that redundant install
        // failed the whole command after ROCm was already in place.
        return install_rocm_on(transport, target, channel, &readiness);
    }

    if !readiness.tailscale_present {
        // Checked before the model starts, not after. The endpoint is published
        // by the remote's own Tailscale; without it a started model would be
        // unreachable and untracked.
        bail!(
            "{target} has ROCm but no Tailscale, so it cannot publish an endpoint.\n\
             `rocm remote` reaches a model over the tailnet, not through this machine.\n\
             Install Tailscale there and run `tailscale up`, then try again."
        );
    }

    match readiness.cli {
        Some(cli) => {
            match readiness.cli_version.as_deref() {
                Some(version) => println!("  remote CLI: {version}"),
                None => println!("  remote CLI: present"),
            }
            Ok(cli)
        }
        // A missing CLI is a job, not a refusal: unlike ROCm it is one small
        // artifact, and the machine can usually fetch its own.
        None => {
            super::provision::install_cli(transport, target, readiness.platform.as_ref(), channel)
                .map(|(cli, _)| cli)
        }
    }
}

/// Install ROCm on the remote, once the CLI is there to do it with.
///
/// The CLI has to come first: it is what runs the install, and it is the
/// smaller, safer artifact of the two.
fn install_rocm_on(
    transport: &dyn Transport,
    target: &str,
    channel: &str,
    readiness: &RemoteReadiness,
) -> Result<String> {
    let remote_cli = match &readiness.cli {
        Some(cli) => cli.clone(),
        None => {
            super::provision::install_cli(transport, target, readiness.platform.as_ref(), channel)?
                .0
        }
    };

    // Look before installing. The catalog knows the states that need a person,
    // and the wizard that walks a person through them cannot run over a
    // connection with nobody watching.
    let (_, report) = super::doctor::examine_remote(transport, &remote_cli, None)?;
    let passwordless_sudo = super::install::has_passwordless_sudo(transport)?;
    if let Some(refusal) = super::install::assess(&report, passwordless_sudo) {
        bail!(
            "{}",
            super::install::describe_refusal(&refusal, target, &report, 5)
        );
    }

    super::install::install(transport, target, &remote_cli)?;
    Ok(remote_cli)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::transport::{ScriptedStep, ScriptedTransport};

    /// A machine with everything present.
    fn ready_steps() -> Vec<ScriptedStep> {
        vec![
            ScriptedStep::ok("rocm --version", "rocm 1.2.3"),
            ScriptedStep::ok("command -v rocminfo", ""),
            ScriptedStep::ok("command -v tailscale", ""),
            ScriptedStep::ok("uname -s", "Linux\nx86_64\n"),
        ]
    }

    #[test]
    fn a_ready_machine_reports_its_cli_rocm_tailscale_and_platform() {
        let transport = ScriptedTransport::new(ready_steps());
        let readiness = probe(&transport).expect("probe");

        assert_eq!(readiness.cli.as_deref(), Some("rocm"));
        assert_eq!(readiness.cli_version.as_deref(), Some("rocm 1.2.3"));
        assert!(readiness.rocm_present);
        assert!(readiness.tailscale_present);
        assert_eq!(
            readiness.platform,
            Some(RemotePlatform {
                os: "linux".to_owned(),
                arch: "x86_64".to_owned()
            })
        );
    }

    #[test]
    fn a_cli_outside_path_is_found_where_we_would_have_installed_it() {
        // A non-interactive shell often has no ~/.local/bin on PATH. Missing
        // this would reinstall over a perfectly good CLI on every run.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::fails("rocm --version", 127, "not found"),
            ScriptedStep::ok("/.local/bin/rocm --version", "rocm 1.2.3"),
            ScriptedStep::ok("command -v rocminfo", ""),
            ScriptedStep::ok("command -v tailscale", ""),
            ScriptedStep::ok("uname -s", "Linux\nx86_64\n"),
        ]);
        let readiness = probe(&transport).expect("probe");
        assert_eq!(readiness.cli.as_deref(), Some(REMOTE_CLI_PATH));
    }

    #[test]
    fn a_machine_without_rocm_is_refused_with_the_command_that_fixes_it() {
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("rocm --version", "rocm 1.2.3"),
            ScriptedStep::fails("command -v rocminfo", 1, ""),
            ScriptedStep::ok("command -v tailscale", ""),
            ScriptedStep::ok("uname -s", "Linux\nx86_64\n"),
        ]);
        let error = ensure_ready(&transport, "gpu-box", "release")
            .unwrap_err()
            .to_string();
        assert!(error.contains("no ROCm installation"), "{error}");
        assert!(error.contains("ssh gpu-box --"), "{error}");
    }

    #[test]
    fn a_machine_without_tailscale_is_refused_before_a_model_is_started() {
        // Discovering this after starting the server leaves a process on
        // someone's GPU that nobody can reach and nothing records.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("rocm --version", "rocm 1.2.3"),
            ScriptedStep::ok("command -v rocminfo", ""),
            ScriptedStep::fails("command -v tailscale", 1, ""),
            ScriptedStep::ok("uname -s", "Linux\nx86_64\n"),
        ]);
        let error = ensure_ready(&transport, "gpu-box", "release")
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot publish an endpoint"), "{error}");
    }

    #[test]
    fn a_machine_without_the_cli_has_one_installed_rather_than_being_refused() {
        // Unlike ROCm, the CLI is one small artifact the machine can usually
        // fetch itself, so its absence is a job rather than a dead end.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::fails("rocm --version", 127, "not found"),
            ScriptedStep::ok("command -v rocminfo", ""),
            ScriptedStep::ok("command -v tailscale", ""),
            ScriptedStep::ok("uname -s", "Linux\nx86_64\n"),
            ScriptedStep::ok("install.sh | sh", ""),
            ScriptedStep::ok("/.local/bin/rocm --version", "rocm 1.2.3"),
        ]);
        assert_eq!(
            ensure_ready(&transport, "gpu-box", "release").expect("provisioned"),
            REMOTE_CLI_PATH
        );
    }

    #[test]
    fn a_ready_machine_returns_how_to_invoke_its_cli() {
        let transport = ScriptedTransport::new(ready_steps());
        assert_eq!(
            ensure_ready(&transport, "gpu-box", "release").expect("ready"),
            "rocm"
        );
    }
}
