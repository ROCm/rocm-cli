// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The control channel to a remote GPU host.
//!
//! `rocm remote` needs exactly two things from a remote machine: run a command
//! and read its output, and copy a file over. [`Transport`] captures those and
//! nothing else, so the orchestration above it stays agnostic to how the host is
//! reached. [`SshTransport`] is the one real implementation.
//!
//! Why shell out to the system `ssh`/`scp` instead of a Rust SSH crate: the
//! user's existing keys, agent, `~/.ssh/config`, `ProxyJump` hosts, and host
//! aliases all keep working with no extra configuration, and there is no new
//! dependency surface for something as security-sensitive as an SSH client. It
//! also keeps this code synchronous, matching the command handlers around it.
//!
//! Note what is deliberately *absent*: there is no port-forwarding method. An
//! earlier prototype opened a detached `ssh -L` tunnel and tracked its PID; this
//! design instead has the remote publish its own service onto the tailnet, so
//! the data path never runs through the control channel and no local process
//! outlives the command.

// The control channel lands before its first caller: discovery talks only to the
// local Tailscale daemon, so nothing reaches a remote until `rocm remote serve`
// exists. Until then only the tests below exercise this. Remove this attribute
// in the change that adds the serve path.
#![cfg_attr(not(test), allow(dead_code))]

#[cfg(test)]
use std::cell::RefCell;
use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

/// Captured result of running a command on the remote host.
///
/// The `success` flag distinguishes a clean-but-non-zero remote exit — say
/// `rocm --version` on a host that has no CLI — from a transport failure where
/// the host could not be reached at all. Every readiness probe depends on that
/// distinction: "answered, and said no" and "never answered" call for different
/// errors, and collapsing them produces the classic misleading
/// "ROCm is not installed" on a host that is merely offline.
#[derive(Debug, Clone)]
pub(crate) struct RemoteOutcome {
    pub(crate) success: bool,
    pub(crate) code: Option<i32>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

impl RemoteOutcome {
    /// How the command exited, for error messages: an exit status, or `signal`
    /// when it was killed and carries no code.
    fn exit_label(&self) -> String {
        self.code
            .map_or_else(|| "signal".to_owned(), |code| code.to_string())
    }
}

/// Ways to reach a remote host. See the module docs for the design rationale.
pub(crate) trait Transport {
    /// Run `command` on the remote host, optionally feeding it `stdin`, and
    /// capture the outcome.
    ///
    /// Returns `Err` only when the command could not be launched or the host is
    /// unreachable; a non-zero remote exit is reported through
    /// [`RemoteOutcome::success`].
    ///
    /// `stdin` exists so a secret never has to travel in a command line. Both
    /// the local `ssh` invocation and the remote shell expose their arguments in
    /// the process table, so an API key interpolated into `command` would be
    /// readable by any other user on either machine; piped in, it is not.
    fn exec_with_stdin(&self, command: &str, stdin: Option<&str>) -> Result<RemoteOutcome>;

    /// Run `command` on the remote host and capture its outcome.
    fn exec(&self, command: &str) -> Result<RemoteOutcome> {
        self.exec_with_stdin(command, None)
    }

    /// Run `command` and return its stdout, failing if it exits non-zero. A
    /// convenience over [`exec`](Transport::exec) for commands expected to
    /// succeed.
    fn run(&self, command: &str) -> Result<String> {
        let outcome = self.exec(command)?;
        if !outcome.success {
            bail!(
                "remote command failed (exit {}): {}\n  command: {command}",
                outcome.exit_label(),
                outcome.stderr.trim(),
            );
        }
        Ok(outcome.stdout)
    }

    /// Copy a local file to `remote_path` on the host.
    fn push_file(&self, local_path: &Path, remote_path: &str) -> Result<()>;
}

/// Options that let repeated commands share one connection.
///
/// All three are needed or none are. OpenSSH defaults `ControlPath` to `none`,
/// and with no socket path it ignores `ControlMaster` entirely — so emitting the
/// master and persist options alone is a claim the client does not honour, and
/// every status poll silently pays a fresh handshake. When no private socket
/// directory can be established we emit nothing rather than options that look
/// like multiplexing and are not.
///
/// `%C` is OpenSSH's hash of the connection's identity, so one socket per
/// destination without building a filename out of user-supplied host strings.
fn multiplex_args(control_path: Option<&str>) -> Vec<String> {
    let Some(control_path) = control_path else {
        return Vec::new();
    };
    vec![
        "-o".to_owned(),
        "ControlMaster=auto".to_owned(),
        "-o".to_owned(),
        format!("ControlPath={control_path}"),
        // Long enough that a burst of polls reuses one connection, short enough
        // that nothing lingers after a command finishes.
        "-o".to_owned(),
        "ControlPersist=60s".to_owned(),
    ]
}

/// A private directory to keep control sockets in, or `None` if we cannot get
/// one — in which case multiplexing is skipped rather than half-configured.
#[cfg(unix)]
fn control_socket_path() -> Option<String> {
    use std::os::unix::fs::PermissionsExt as _;
    use std::sync::OnceLock;

    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        let directory = rocm_core::AppPaths::discover().ok()?.data_dir.join("ssh");
        std::fs::create_dir_all(&directory).ok()?;
        // Owner-only: a control socket is an authenticated channel to the remote,
        // so anything that can reach it can run commands there as this user.
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).ok()?;

        let candidate = directory.join("cm-%C").to_string_lossy().into_owned();
        // Unix socket paths are capped near 104 bytes; `%C` expands to a 40-char
        // hash. Past the cap ssh fails outright, so a deep home directory should
        // cost multiplexing, not every remote command.
        (candidate.len() + 40 < 100).then_some(candidate)
    })
    .clone()
}

/// Windows has no connection multiplexing to configure.
///
/// Win32-OpenSSH does not implement `ControlMaster`; setting it there produces
/// warnings at best. Returning `None` keeps the argument builder honest instead
/// of emitting options the platform ignores.
#[cfg(not(unix))]
fn control_socket_path() -> Option<String> {
    None
}

/// Environment variable naming an alternative SSH configuration file.
///
/// `ssh` resolves `~/.ssh/config` from the account database rather than from
/// `HOME`, so there is otherwise no way to point this at a different one —
/// awkward for anyone keeping a per-project or per-tenant ssh config, and the
/// reason the end-to-end harness could not drive the real binary at all.
const SSH_CONFIG_ENV: &str = "ROCM_REMOTE_SSH_CONFIG";

/// `-F <path>` when an alternative config was named, nothing otherwise.
fn config_args() -> Vec<String> {
    std::env::var(SSH_CONFIG_ENV)
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map_or_else(Vec::new, |path| vec!["-F".to_owned(), path])
}

/// Reject a destination that `ssh` would read as an option rather than a host.
///
/// The destination has to precede the `--` guard — that is where `ssh` expects
/// it — so unlike the remote command it is not protected by it. A value
/// beginning with `-` is therefore parsed as a local option, and
/// `-oProxyCommand=…` would run a command on *this* machine. Hostnames cannot
/// begin with a hyphen anyway (RFC 1123), so nothing legitimate is lost by
/// refusing outright rather than trying to escape it.
fn validate_destination(destination: &str) -> Result<()> {
    if destination.trim().is_empty() {
        bail!("a remote machine must be named");
    }
    // Check the host half too: `user@-h` puts the hyphen where ssh still sees it.
    let host = destination.rsplit('@').next().unwrap_or(destination);
    if destination.starts_with('-') || host.starts_with('-') {
        bail!(
            "`{destination}` is not a usable machine name: a name starting with `-` would be              read as an option by ssh rather than as a host"
        );
    }
    Ok(())
}

/// SSH-backed control channel that shells out to the system `ssh`/`scp`.
#[derive(Debug, Clone)]
pub(crate) struct SshTransport {
    /// SSH destination as accepted by `ssh` (e.g. `user@host`, or a
    /// `~/.ssh/config` host alias).
    destination: String,
    /// Optional explicit port; `None` uses the ssh default or whatever
    /// `~/.ssh/config` specifies for this destination.
    port: Option<u16>,
}

impl SshTransport {
    pub(crate) fn new(destination: impl Into<String>, port: Option<u16>) -> Result<Self> {
        let destination = destination.into();
        validate_destination(&destination)?;
        Ok(Self {
            destination,
            port: port.filter(|port| *port != 0),
        })
    }

    /// Options applied to every invocation.
    ///
    /// `BatchMode=yes` is the load-bearing one: without it a host that wants a
    /// password or key passphrase blocks forever behind a prompt nobody is
    /// watching, which for a status poll or a scripted run means a hang rather
    /// than an error. `ConnectTimeout` bounds an unreachable host the same way.
    fn base_ssh_args(&self, control_path: Option<&str>) -> Vec<String> {
        let mut args = config_args();
        args.extend([
            "-o".to_owned(),
            "BatchMode=yes".to_owned(),
            "-o".to_owned(),
            "ConnectTimeout=10".to_owned(),
        ]);
        args.extend(multiplex_args(control_path));
        if let Some(port) = self.port {
            args.push("-p".to_owned());
            args.push(port.to_string());
        }
        args
    }

    /// Argument vector for running `command` on the remote, excluding the `ssh`
    /// program name. Split out so argument construction is unit-testable without
    /// a network or an SSH server.
    fn exec_argv(&self, command: &str) -> Vec<String> {
        self.exec_argv_with(command, control_socket_path().as_deref())
    }

    /// The argument builder proper, with the socket location passed in.
    ///
    /// Separated so tests exercise both the multiplexing and no-multiplexing
    /// forms without reaching for a real socket directory — discovering one
    /// creates a directory under the user's home, which a unit test has no
    /// business doing.
    fn exec_argv_with(&self, command: &str, control_path: Option<&str>) -> Vec<String> {
        let mut args = self.base_ssh_args(control_path);
        args.push(self.destination.clone());
        // `--` stops ssh from parsing anything in the remote command as its own
        // option, so a model name or flag that happens to start with `-` reaches
        // the remote intact instead of being swallowed locally.
        args.push("--".to_owned());
        args.push(command.to_owned());
        args
    }

    /// Argument vector for `scp`, excluding the `scp` program name.
    fn scp_argv(&self, local_path: &str, remote_path: &str) -> Vec<String> {
        let mut args = config_args();
        args.extend([
            "-o".to_owned(),
            "BatchMode=yes".to_owned(),
            "-o".to_owned(),
            "ConnectTimeout=10".to_owned(),
        ]);
        // scp spells the port with a capital -P, unlike ssh's lowercase -p. A
        // perennial bug when argument-building code is copy-pasted between the
        // two, so it has its own test.
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
    fn exec_with_stdin(&self, command: &str, stdin: Option<&str>) -> Result<RemoteOutcome> {
        let mut child = Command::new("ssh")
            .args(self.exec_argv(command))
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to launch ssh to run a command on {}",
                    self.destination
                )
            })?;

        if let Some(payload) = stdin {
            let mut handle = child
                .stdin
                .take()
                .context("ssh stdin was not available to write to")?;
            handle
                .write_all(payload.as_bytes())
                .with_context(|| format!("failed to send input to {}", self.destination))?;
            // Dropping closes the pipe, which is what tells the remote reader the
            // input has ended. Without it a remote `read` waits forever.
            drop(handle);
        }

        let output = child.wait_with_output().with_context(|| {
            format!(
                "failed to read the result of a command on {}",
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

    fn push_file(&self, local_path: &Path, remote_path: &str) -> Result<()> {
        let local = local_path.to_string_lossy();
        let output = Command::new("scp")
            .args(self.scp_argv(&local, remote_path))
            .stdin(Stdio::null())
            .output()
            .with_context(|| format!("failed to launch scp to {}", self.destination))?;
        if !output.status.success() {
            bail!(
                "failed to copy {local} to {}:{remote_path}: {}",
                self.destination,
                String::from_utf8_lossy(&output.stderr).trim(),
            );
        }
        Ok(())
    }
}

/// One programmed reply in a [`ScriptedTransport`].
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct ScriptedStep {
    /// Substring the remote command must contain for this step to apply.
    ///
    /// A substring rather than an exact string on purpose: a test that pins
    /// whole command lines re-breaks every time an unrelated flag is added,
    /// which trains people to update expectations without reading them.
    matches: String,
    outcome: RemoteOutcome,
}

#[cfg(test)]
impl ScriptedStep {
    /// A step whose command succeeds, returning `stdout`.
    pub(crate) fn ok(matches: &str, stdout: &str) -> Self {
        Self {
            matches: matches.to_owned(),
            outcome: RemoteOutcome {
                success: true,
                code: Some(0),
                stdout: stdout.to_owned(),
                stderr: String::new(),
            },
        }
    }

    /// A step whose command runs but exits non-zero — the remote answering "no",
    /// as distinct from being unreachable.
    pub(crate) fn fails(matches: &str, code: i32, stderr: &str) -> Self {
        Self {
            matches: matches.to_owned(),
            outcome: RemoteOutcome {
                success: false,
                code: Some(code),
                stdout: String::new(),
                stderr: stderr.to_owned(),
            },
        }
    }
}

/// What a [`ScriptedTransport`] was asked to do, in order.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TransportCall {
    Exec {
        command: String,
        stdin: Option<String>,
    },
    PushFile {
        local_path: String,
        remote_path: String,
    },
}

/// A [`Transport`] stand-in that replays programmed outcomes and records every
/// call, so bootstrap and session flows can be tested without a network, an SSH
/// server, or a tailnet.
///
/// Unmatched commands are a hard error rather than a benign default. A silently
/// tolerant double lets a flow test keep passing after the flow stops issuing a
/// command it is supposed to issue, which is exactly the regression these tests
/// exist to catch.
#[cfg(test)]
pub(crate) struct ScriptedTransport {
    steps: Vec<ScriptedStep>,
    calls: RefCell<Vec<TransportCall>>,
    push_result: RefCell<Result<(), String>>,
}

#[cfg(test)]
impl ScriptedTransport {
    pub(crate) fn new(steps: Vec<ScriptedStep>) -> Self {
        Self {
            steps,
            calls: RefCell::new(Vec::new()),
            push_result: RefCell::new(Ok(())),
        }
    }

    /// Make the next and all subsequent file copies fail, for testing the
    /// provisioning fallback paths.
    pub(crate) fn failing_push(self, message: &str) -> Self {
        *self.push_result.borrow_mut() = Err(message.to_owned());
        self
    }

    /// Every call made so far, in order.
    pub(crate) fn calls(&self) -> Vec<TransportCall> {
        self.calls.borrow().clone()
    }
}

#[cfg(test)]
impl Transport for ScriptedTransport {
    fn exec_with_stdin(&self, command: &str, stdin: Option<&str>) -> Result<RemoteOutcome> {
        self.calls.borrow_mut().push(TransportCall::Exec {
            command: command.to_owned(),
            stdin: stdin.map(ToOwned::to_owned),
        });
        self.steps
            .iter()
            .filter(|step| command.contains(&step.matches))
            // Most specific wins. One remote command is often a suffix of
            // another — `rocm --version` and `$HOME/.local/bin/rocm --version`
            // differ only by a prefix — so first-match would answer the second
            // with the first's reply and silently test the wrong branch.
            .max_by_key(|step| step.matches.len())
            .map(|step| step.outcome.clone())
            .with_context(|| {
                format!(
                    "the scripted transport has no reply for: {command}\n  programmed: {:?}",
                    self.steps
                        .iter()
                        .map(|step| step.matches.as_str())
                        .collect::<Vec<_>>()
                )
            })
    }

    fn push_file(&self, local_path: &Path, remote_path: &str) -> Result<()> {
        self.calls.borrow_mut().push(TransportCall::PushFile {
            local_path: local_path.to_string_lossy().into_owned(),
            remote_path: remote_path.to_owned(),
        });
        match &*self.push_result.borrow() {
            Ok(()) => Ok(()),
            Err(message) => bail!("{message}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_argv_never_prompts_and_guards_the_remote_command() {
        let transport = SshTransport::new("user@gpubox", None).unwrap();
        let argv = transport.exec_argv_with("rocm --version", None);

        // Without BatchMode a host wanting a passphrase hangs instead of failing.
        assert!(argv.windows(2).any(|pair| pair == ["-o", "BatchMode=yes"]));
        assert_eq!(argv.last().unwrap(), "rocm --version");
        // The remote command sits behind `--`, immediately after the destination.
        let guard = argv.iter().position(|arg| arg == "--").unwrap();
        assert_eq!(argv[guard - 1], "user@gpubox");
    }

    #[test]
    fn a_name_ssh_would_read_as_an_option_is_refused() {
        // The destination sits before the `--` guard, so unlike the remote
        // command it is not shielded by it. `-oProxyCommand=…` as a "host" runs
        // a command on this machine.
        for hostile in ["-oProxyCommand=touch /tmp/pwned", "--fake", "user@-oX", ""] {
            let error = SshTransport::new(hostile, None)
                .expect_err(&format!("{hostile:?} should be refused"))
                .to_string();
            assert!(!error.is_empty());
        }

        // Ordinary destinations still work, including a user prefix and an alias.
        for fine in [
            "gpu-box",
            "user@gpu-box",
            "gpu-box.example-tailnet.ts.net",
            "100.88.14.21",
        ] {
            SshTransport::new(fine, None).unwrap_or_else(|error| panic!("{fine}: {error}"));
        }
    }

    #[test]
    fn connection_reuse_is_either_fully_configured_or_not_claimed() {
        // OpenSSH ignores ControlMaster when ControlPath is unset, so the two
        // options without the third are multiplexing that silently never
        // happens — every status poll paying a fresh handshake while the code
        // claims otherwise.
        let configured = multiplex_args(Some("/tmp/rocm/cm-%C"));
        assert!(
            configured
                .windows(2)
                .any(|pair| pair == ["-o", "ControlMaster=auto"]),
            "{configured:?}"
        );
        assert!(
            configured
                .iter()
                .any(|arg| arg == "ControlPath=/tmp/rocm/cm-%C"),
            "{configured:?}"
        );

        // With nowhere private to put the socket, emit nothing at all.
        assert!(multiplex_args(None).is_empty());
    }

    #[test]
    fn no_multiplexing_option_ever_appears_without_a_socket_path() {
        // The invariant, stated over the real argument vector rather than the
        // helper: a reader scanning for ControlMaster should never find it
        // orphaned, in either configuration.
        let transport = SshTransport::new("gpubox", None).unwrap();
        for control_path in [None, Some("/tmp/rocm-ssh/cm-%C")] {
            let argv = transport.exec_argv_with("true", control_path);
            let has_master = argv.iter().any(|arg| arg.starts_with("ControlMaster"));
            let has_path = argv.iter().any(|arg| arg.starts_with("ControlPath"));
            assert_eq!(
                has_master, has_path,
                "ControlMaster and ControlPath must appear together: {argv:?}"
            );
            assert_eq!(has_master, control_path.is_some(), "{argv:?}");
        }
    }

    #[test]
    fn exec_argv_threads_an_explicit_port_lowercase() {
        let transport = SshTransport::new("user@gpubox", Some(2222)).unwrap();
        let argv = transport.exec_argv_with("echo hi", None);
        assert!(argv.windows(2).any(|pair| pair == ["-p", "2222"]));
    }

    #[test]
    fn exec_argv_omits_an_unset_port_so_ssh_config_decides() {
        // Passing no port must leave the choice to ~/.ssh/config rather than
        // hardcoding 22, or a configured non-standard port is silently ignored.
        let transport = SshTransport::new("gpubox", None).unwrap();
        let argv = transport.exec_argv_with("echo hi", None);
        assert!(!argv.iter().any(|arg| arg == "-p"));
    }

    #[test]
    fn scp_argv_uses_uppercase_port_and_a_remote_colon_path() {
        let transport = SshTransport::new("user@gpubox", Some(2222)).unwrap();
        let argv = transport.scp_argv("/tmp/rocm", "/tmp/rocm");
        // Capital -P: scp's port flag differs from ssh's, and getting it wrong
        // silently copies to the default port instead.
        assert!(argv.windows(2).any(|pair| pair == ["-P", "2222"]));
        assert_eq!(argv.last().unwrap(), "user@gpubox:/tmp/rocm");
    }

    #[test]
    fn scripted_transport_separates_a_refusal_from_being_unreachable() {
        // A remote that answers "no CLI here" (exit 127) is not the same as a
        // remote we could not reach; the first is an outcome, the second an Err.
        let transport = ScriptedTransport::new(vec![ScriptedStep::fails(
            "rocm --version",
            127,
            "command not found",
        )]);

        let outcome = transport.exec("rocm --version").expect("host answered");
        assert!(!outcome.success);
        assert_eq!(outcome.code, Some(127));

        let unreachable = transport.exec("tailscale status");
        assert!(
            unreachable.is_err(),
            "an unscripted command must fail loudly, not return a benign default"
        );
    }

    #[test]
    fn run_surfaces_the_failing_command_and_its_stderr() {
        let transport = ScriptedTransport::new(vec![ScriptedStep::fails(
            "rocm serve",
            1,
            "no GPU available",
        )]);

        let error = transport
            .run("rocm serve tiny-model")
            .unwrap_err()
            .to_string();
        assert!(error.contains("no GPU available"), "{error}");
        assert!(error.contains("rocm serve tiny-model"), "{error}");
    }

    #[test]
    fn scripted_transport_records_stdin_so_secrets_stay_out_of_argv() {
        // The recording is what lets a later test assert an API key was piped in
        // rather than interpolated into the command line.
        let transport = ScriptedTransport::new(vec![ScriptedStep::ok("read -r", "")]);
        transport
            .exec_with_stdin("read -r KEY; exec rocm serve", Some("s3cret"))
            .expect("scripted");

        assert_eq!(
            transport.calls(),
            vec![TransportCall::Exec {
                command: "read -r KEY; exec rocm serve".to_owned(),
                stdin: Some("s3cret".to_owned()),
            }]
        );
    }

    #[test]
    fn scripted_transport_reports_a_failing_file_copy() {
        let transport = ScriptedTransport::new(vec![]).failing_push("no space left on device");

        let error = transport
            .push_file(Path::new("/tmp/rocm"), "~/.local/bin/rocm")
            .unwrap_err()
            .to_string();
        assert!(error.contains("no space left on device"), "{error}");
        assert_eq!(
            transport.calls(),
            vec![TransportCall::PushFile {
                local_path: "/tmp/rocm".to_owned(),
                remote_path: "~/.local/bin/rocm".to_owned(),
            }]
        );
    }
}
