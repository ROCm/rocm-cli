//! Ensuring a remote host is ready to serve: probe for the `rocm` CLI and ROCm,
//! and push the CLI if it is missing.
//!
//! Scope decisions (see `plans/rocm-remote-serve.md`):
//! - **CLI missing → auto-push.** The whole `rocm` runtime, including the managed
//!   daemon, is a single self-contained binary (`rocm daemon` is the supervisor;
//!   there is no separate `rocmd` to install), so making the remote ready is just
//!   copying one file. Because the release repo is currently private (no public
//!   `curl` URL), v1 pushes the *local* `rocm` binary over `scp`. This works when
//!   the local machine is itself linux-amd64 (the demo case); other hosts get a
//!   clear error pointing at the manual install.
//! - **ROCm missing → detect and fail.** We never attempt to install the
//!   multi-GB ROCm SDK remotely; we report it clearly and stop.

use anyhow::{Context, Result, bail};

use super::transport::Transport;

/// Where a freshly-pushed CLI lands on the remote. Chosen to match the
/// documented manual install location (`~/.local/bin`).
const REMOTE_CLI_PATH: &str = "~/.local/bin/rocm";

/// What a probe found on the remote host.
#[derive(Debug, Clone)]
pub struct RemoteReadiness {
    pub cli_present: bool,
    pub cli_version: Option<String>,
    pub rocm_present: bool,
}

/// Probe the remote for the `rocm` CLI and a ROCm installation. Neither absence
/// is an error here — that judgement belongs to [`ensure_ready`].
pub fn probe(transport: &dyn Transport) -> Result<RemoteReadiness> {
    let cli = transport.exec("rocm --version")?;
    let (cli_present, cli_version) = if cli.success {
        let version = cli.stdout.lines().next().map(|l| l.trim().to_owned());
        (true, version.filter(|v| !v.is_empty()))
    } else {
        (false, None)
    };

    // ROCm detection independent of our CLI: any of the usual markers is enough.
    let rocm = transport.exec(
        "command -v rocminfo >/dev/null 2>&1 || command -v amd-smi >/dev/null 2>&1 || test -d /opt/rocm",
    )?;

    Ok(RemoteReadiness {
        cli_present,
        cli_version,
        rocm_present: rocm.success,
    })
}

/// How to invoke `rocm` on the remote once we know it is present — either the
/// CLI already on `PATH`, or the explicit path we just installed to.
#[derive(Debug, Clone)]
pub struct RemoteCli {
    invocation: String,
}

impl RemoteCli {
    /// The command prefix to use when building remote `rocm …` invocations.
    pub fn invocation(&self) -> &str {
        &self.invocation
    }
}

/// Make the remote ready to serve, pushing the CLI if needed. Returns how to
/// invoke `rocm` on the remote. Fails (without attempting any install) if ROCm
/// is absent.
pub fn ensure_ready(transport: &dyn Transport) -> Result<RemoteCli> {
    let readiness = probe(transport)?;

    if !readiness.rocm_present {
        bail!(
            "ROCm was not detected on the remote host (no rocminfo/amd-smi and no /opt/rocm).\n\
             Install ROCm on the remote first, then re-run. `rocm remote serve` does not \
             install the ROCm SDK for you."
        );
    }

    if readiness.cli_present {
        match readiness.cli_version.as_deref() {
            Some(version) => println!("  remote rocm: {version}"),
            None => println!("  remote rocm: present"),
        }
        return Ok(RemoteCli {
            invocation: "rocm".to_owned(),
        });
    }

    // CLI absent: push the local binary.
    push_local_cli(transport)?;
    Ok(RemoteCli {
        invocation: REMOTE_CLI_PATH.to_owned(),
    })
}

/// Copy the locally-running `rocm` binary to the remote and verify it runs.
fn push_local_cli(transport: &dyn Transport) -> Result<()> {
    let local_exe = std::env::current_exe()
        .context("failed to resolve the local rocm executable to push to the remote")?;

    transport
        .run("mkdir -p \"$HOME/.local/bin\"")
        .context("failed to create ~/.local/bin on the remote")?;
    transport.push_file(&local_exe, REMOTE_CLI_PATH).context(
        "failed to copy the local rocm binary to the remote. If your local machine is not \
             linux-amd64, install rocm on the remote manually and re-run.",
    )?;
    transport
        .run("chmod +x \"$HOME/.local/bin/rocm\"")
        .context("failed to mark the pushed rocm binary executable")?;

    // Verify the pushed binary actually runs on the remote (catches arch/libc
    // mismatch early, before we try to serve with it).
    let verify = transport.exec("\"$HOME/.local/bin/rocm\" --version")?;
    if !verify.success {
        bail!(
            "pushed rocm binary does not run on the remote (exit {}): {}\n\
             This usually means an architecture or libc mismatch. Install rocm on the remote \
             manually and re-run.",
            verify
                .code
                .map_or_else(|| "signal".to_owned(), |c| c.to_string()),
            verify.stderr.trim(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::transport::RemoteOutcome;
    use std::cell::RefCell;

    /// A scripted transport: each `exec` pops the next queued outcome and records
    /// the command it was asked to run. `run` reuses the trait default over
    /// `exec`, so queued outcomes must account for `run` calls too.
    struct ScriptedTransport {
        outcomes: RefCell<Vec<RemoteOutcome>>,
        commands: RefCell<Vec<String>>,
        pushed: RefCell<Vec<String>>,
    }

    impl ScriptedTransport {
        fn new(outcomes: Vec<RemoteOutcome>) -> Self {
            Self {
                outcomes: RefCell::new(outcomes),
                commands: RefCell::new(Vec::new()),
                pushed: RefCell::new(Vec::new()),
            }
        }
    }

    fn ok(stdout: &str) -> RemoteOutcome {
        RemoteOutcome {
            success: true,
            code: Some(0),
            stdout: stdout.to_owned(),
            stderr: String::new(),
        }
    }

    fn fail() -> RemoteOutcome {
        RemoteOutcome {
            success: false,
            code: Some(1),
            stdout: String::new(),
            stderr: "not found".to_owned(),
        }
    }

    impl Transport for ScriptedTransport {
        fn exec(&self, command: &str) -> Result<RemoteOutcome> {
            self.commands.borrow_mut().push(command.to_owned());
            Ok(self
                .outcomes
                .borrow_mut()
                .drain(..1)
                .next()
                .unwrap_or_else(fail))
        }
        fn push_file(&self, _local: &std::path::Path, remote_path: &str) -> Result<()> {
            self.pushed.borrow_mut().push(remote_path.to_owned());
            Ok(())
        }
        fn forward(
            &self,
            _local_port: u16,
            _remote_host: &str,
            _remote_port: u16,
        ) -> Result<super::super::transport::ForwardGuard> {
            unreachable!("bootstrap tests do not forward")
        }
    }

    #[test]
    fn probe_reports_present_cli_and_rocm() {
        let t = ScriptedTransport::new(vec![ok("rocm 0.3.0"), ok("")]);
        let r = probe(&t).unwrap();
        assert!(r.cli_present);
        assert_eq!(r.cli_version.as_deref(), Some("rocm 0.3.0"));
        assert!(r.rocm_present);
    }

    #[test]
    fn ensure_ready_bails_when_rocm_absent() {
        // cli --version fails, rocm detection fails.
        let t = ScriptedTransport::new(vec![fail(), fail()]);
        let err = ensure_ready(&t).unwrap_err().to_string();
        assert!(err.contains("ROCm was not detected"), "got: {err}");
    }

    #[test]
    fn ensure_ready_returns_path_invocation_after_push() {
        // probe: cli absent (fail), rocm present (ok);
        // push_local_cli: mkdir ok, chmod ok, verify ok.
        let t = ScriptedTransport::new(vec![fail(), ok(""), ok(""), ok(""), ok("rocm 0.3.0")]);
        let cli = ensure_ready(&t).unwrap();
        assert_eq!(cli.invocation(), REMOTE_CLI_PATH);
        assert!(t.pushed.borrow().iter().any(|p| p == REMOTE_CLI_PATH));
    }

    #[test]
    fn ensure_ready_uses_path_rocm_when_present() {
        let t = ScriptedTransport::new(vec![ok("rocm 0.3.0"), ok("")]);
        let cli = ensure_ready(&t).unwrap();
        assert_eq!(cli.invocation(), "rocm");
        assert!(t.pushed.borrow().is_empty());
    }
}
