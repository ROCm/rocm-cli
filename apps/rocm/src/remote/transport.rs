// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Transport abstraction for reaching a remote GPU host.
//!
//! `rocm remote` needs three things from a remote host: run a command and read
//! its output, copy a file over, and open a local port-forward to a service the
//! remote is listening on. The [`Transport`] trait captures exactly those three
//! capabilities so the orchestration in `remote::serve` stays agnostic to *how*
//! we reach the host. v1 ships a single implementation, [`SshTransport`], that
//! shells out to the system `ssh`/`scp`; a future Tailscale-native transport can
//! slot in behind the same trait without touching the callers.
//!
//! Why shell out to `ssh` instead of a Rust SSH crate: it reuses the user's
//! existing keys/agent/`~/.ssh/config` and connection multiplexing for free,
//! keeps this code synchronous (the surrounding command handlers are sync), and
//! makes the port-forward a plain detached child process tracked by PID — the
//! same detach + liveness machinery the managed-service supervisor already uses.

use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

/// Captured result of running a command on the remote host. Distinguishes a
/// clean-but-non-zero exit (e.g. `rocm --version` when the CLI is absent) from a
/// transport failure (couldn't reach the host at all) — the bootstrap probes
/// depend on that distinction.
#[derive(Debug, Clone)]
pub struct RemoteOutcome {
    pub success: bool,
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// Ways to reach a remote host. See the module docs for the design rationale.
pub trait Transport {
    /// Run `command` on the remote host and capture its outcome. Returns `Err`
    /// only when the command could not be launched / the host is unreachable; a
    /// non-zero remote exit is reported via [`RemoteOutcome::success`].
    fn exec(&self, command: &str) -> Result<RemoteOutcome>;

    /// Run `command` and return its stdout, failing if it exits non-zero. A
    /// convenience over [`exec`](Transport::exec) for commands expected to
    /// succeed.
    fn run(&self, command: &str) -> Result<String> {
        let outcome = self.exec(command)?;
        if !outcome.success {
            bail!(
                "remote command failed (exit {}): {}\n  command: {command}",
                outcome
                    .code
                    .map_or_else(|| "signal".to_owned(), |c| c.to_string()),
                outcome.stderr.trim(),
            );
        }
        Ok(outcome.stdout)
    }

    /// Copy a local file to `remote_path` on the host.
    fn push_file(&self, local_path: &std::path::Path, remote_path: &str) -> Result<()>;

    /// Open a **detached** local port-forward: `127.0.0.1:local_port` on this
    /// machine is forwarded to `remote_host:remote_port` as seen from the remote
    /// side (typically `127.0.0.1:<service port>`). The forward runs in its own
    /// detached process that survives this command exiting; the returned PID is
    /// recorded so the session can later be checked for liveness and torn down.
    /// Killing the PID closes the forward.
    fn open_detached_forward(
        &self,
        local_port: u16,
        remote_host: &str,
        remote_port: u16,
    ) -> Result<u32>;
}

/// SSH-backed transport that shells out to the system `ssh`/`scp`.
#[derive(Debug, Clone)]
pub struct SshTransport {
    /// SSH destination as accepted by `ssh` (e.g. `user@host`, or a
    /// `~/.ssh/config` host alias).
    destination: String,
    /// Optional explicit port (`ssh -p`); `None` uses the ssh default / config.
    port: Option<u16>,
}

impl SshTransport {
    pub fn new(destination: impl Into<String>, port: Option<u16>) -> Self {
        Self {
            destination: destination.into(),
            port,
        }
    }

    /// Common `ssh` options applied to every invocation: fail fast on connect,
    /// stay non-interactive (no password prompts hanging a pipeline), and reuse
    /// a multiplexed master connection so repeated `run` calls are cheap.
    fn base_ssh_args(&self) -> Vec<String> {
        let mut args = vec![
            "-o".to_owned(),
            "BatchMode=yes".to_owned(),
            "-o".to_owned(),
            "ConnectTimeout=10".to_owned(),
            "-o".to_owned(),
            "ControlMaster=auto".to_owned(),
            "-o".to_owned(),
            "ControlPersist=60s".to_owned(),
        ];
        if let Some(port) = self.port {
            args.push("-p".to_owned());
            args.push(port.to_string());
        }
        args
    }

    /// Argument vector for running `command` on the remote (excludes the `ssh`
    /// program name). Split out for unit testing.
    fn exec_argv(&self, command: &str) -> Vec<String> {
        let mut args = self.base_ssh_args();
        args.push(self.destination.clone());
        args.push("--".to_owned());
        args.push(command.to_owned());
        args
    }

    /// Argument vector for the detached port-forward child (excludes the `ssh`
    /// program name). `-N` = no remote command, `-T` = no pty.
    ///
    /// Unlike the exec path this uses **`ControlMaster=no`** — a dedicated
    /// connection, not a shared/persisted master — so that killing this process
    /// deterministically closes the forward (a persisted master could otherwise
    /// outlive the PID). `ExitOnForwardFailure=yes` makes a failed bind exit
    /// instead of silently staying up with no forward, and `ServerAliveInterval`
    /// lets a dropped link terminate the process rather than hang. Split out for
    /// unit testing.
    fn forward_argv(&self, local_port: u16, remote_host: &str, remote_port: u16) -> Vec<String> {
        let mut args = vec![
            "-o".to_owned(),
            "BatchMode=yes".to_owned(),
            "-o".to_owned(),
            "ConnectTimeout=10".to_owned(),
            "-o".to_owned(),
            "ControlMaster=no".to_owned(),
            "-o".to_owned(),
            "ExitOnForwardFailure=yes".to_owned(),
            "-o".to_owned(),
            "ServerAliveInterval=15".to_owned(),
        ];
        if let Some(port) = self.port {
            args.push("-p".to_owned());
            args.push(port.to_string());
        }
        args.push("-N".to_owned());
        args.push("-T".to_owned());
        args.push("-L".to_owned());
        args.push(format!(
            "127.0.0.1:{local_port}:{remote_host}:{remote_port}"
        ));
        args.push(self.destination.clone());
        args
    }

    /// Argument vector for `scp` to copy a local file to the remote (excludes the
    /// `scp` program name). Split out for unit testing.
    fn scp_argv(&self, local_path: &str, remote_path: &str) -> Vec<String> {
        let mut args = vec![
            "-o".to_owned(),
            "BatchMode=yes".to_owned(),
            "-o".to_owned(),
            "ConnectTimeout=10".to_owned(),
        ];
        // scp takes the port with a capital -P, unlike ssh's lowercase -p.
        if let Some(port) = self.port {
            args.push("-P".to_owned());
            args.push(port.to_string());
        }
        args.push(local_path.to_owned());
        args.push(format!("{}:{remote_path}", self.destination));
        args
    }
}

impl Transport for SshTransport {
    fn exec(&self, command: &str) -> Result<RemoteOutcome> {
        let output = Command::new("ssh")
            .args(self.exec_argv(command))
            .stdin(Stdio::null())
            .output()
            .with_context(|| {
                format!(
                    "failed to launch ssh to run a command on {}",
                    self.destination
                )
            })?;
        Ok(RemoteOutcome {
            success: output.status.success(),
            code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    fn push_file(&self, local_path: &std::path::Path, remote_path: &str) -> Result<()> {
        let local = local_path.to_string_lossy();
        let output = Command::new("scp")
            .args(self.scp_argv(&local, remote_path))
            .stdin(Stdio::null())
            .output()
            .with_context(|| format!("failed to launch scp to {}", self.destination))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "failed to copy {} to {}:{remote_path}: {}",
                local,
                self.destination,
                stderr.trim(),
            );
        }
        Ok(())
    }

    fn open_detached_forward(
        &self,
        local_port: u16,
        remote_host: &str,
        remote_port: u16,
    ) -> Result<u32> {
        let argv = self.forward_argv(local_port, remote_host, remote_port);
        spawn_detached_ssh(&argv)
            .with_context(|| format!("failed to launch ssh port-forward to {}", self.destination))
    }
}

/// Spawn `ssh <argv>` as a detached process (its own session/process group) so
/// it survives the parent command exiting, and return its PID. Mirrors the
/// managed-service supervisor's platform-split detach.
#[cfg(not(windows))]
fn spawn_detached_ssh(argv: &[String]) -> Result<u32> {
    let mut command = Command::new("ssh");
    command
        .args(argv)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    rocm_core::detach_command_session(&mut command);
    let child = command.spawn().context("failed to spawn detached ssh")?;
    // Intentionally do not hold the Child handle: on Unix, dropping it does not
    // kill the process, and the setsid detach keeps it running past our exit.
    Ok(child.id())
}

#[cfg(windows)]
fn spawn_detached_ssh(argv: &[String]) -> Result<u32> {
    let program = std::path::Path::new("ssh");
    rocm_core::spawn_detached_no_inherit(program, argv, &[]).context("failed to spawn detached ssh")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_argv_includes_batchmode_and_destination() {
        let t = SshTransport::new("user@gpubox", None);
        let argv = t.exec_argv("rocm --version");
        assert!(argv.windows(2).any(|w| w == ["-o", "BatchMode=yes"]));
        assert_eq!(argv.last().unwrap(), "rocm --version");
        // The command follows a `--` guard so remote args aren't parsed by ssh.
        let dashdash = argv.iter().position(|a| a == "--").unwrap();
        assert_eq!(argv[dashdash - 1], "user@gpubox");
    }

    #[test]
    fn exec_argv_threads_explicit_port_lowercase() {
        let t = SshTransport::new("user@gpubox", Some(2222));
        let argv = t.exec_argv("echo hi");
        assert!(argv.windows(2).any(|w| w == ["-p", "2222"]));
    }

    #[test]
    fn forward_argv_maps_loopback_local_to_remote() {
        let t = SshTransport::new("gpubox", None);
        let argv = t.forward_argv(8001, "127.0.0.1", 11435);
        assert!(argv.contains(&"-N".to_owned()));
        assert!(
            argv.windows(2)
                .any(|w| { w[0] == "-L" && w[1] == "127.0.0.1:8001:127.0.0.1:11435" })
        );
        assert_eq!(argv.last().unwrap(), "gpubox");
    }

    #[test]
    fn forward_argv_uses_dedicated_connection_for_deterministic_teardown() {
        // ControlMaster=no ensures killing the forward's PID actually closes it
        // (a shared/persisted master could outlive the process).
        let t = SshTransport::new("gpubox", None);
        let argv = t.forward_argv(8001, "127.0.0.1", 11435);
        assert!(argv.windows(2).any(|w| w == ["-o", "ControlMaster=no"]));
        assert!(
            argv.windows(2)
                .any(|w| w == ["-o", "ExitOnForwardFailure=yes"])
        );
        assert!(!argv.iter().any(|a| a.starts_with("ControlPersist")));
    }

    #[test]
    fn scp_argv_uses_uppercase_port_and_remote_colon_path() {
        let t = SshTransport::new("user@gpubox", Some(2222));
        let argv = t.scp_argv("/tmp/rocm.tar.gz", "/tmp/rocm.tar.gz");
        assert!(argv.windows(2).any(|w| w == ["-P", "2222"]));
        assert_eq!(argv.last().unwrap(), "user@gpubox:/tmp/rocm.tar.gz");
    }
}
