// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Publishing a remote machine's loopback service onto the tailnet.
//!
//! This is the data path, and it is declared *by the remote*, not held open by
//! us. `rocm serve` binds `127.0.0.1` on the GPU machine as it always has; the
//! machine then tells its own Tailscale daemon to accept tailnet connections on
//! a port and forward them to that loopback address. Nothing on this end stays
//! running, which is why an endpoint survives the command that created it and
//! is reachable from the user's other machines rather than only this one.
//!
//! The cost of that, and the reason [`super`] insists on a credential: a
//! publish is visible to the whole tailnet, scoped only by its ACLs. Unlike a
//! point-to-point tunnel it also *outlives a reboot*, because it is
//! configuration rather than a process. A publish left behind is a GPU endpoint
//! nobody is tracking, so withdrawal is treated as a first-class operation that
//! reports failure loudly instead of being assumed to have worked.
//!
//! **Unverified against a live tailnet.** The command shapes below follow
//! Tailscale's documented surface, and the parsing follows the `ServeConfig`
//! struct definition, but neither has been run against a real daemon here.
//! Confirm both before relying on this.

use anyhow::{Result, bail};
use serde::Deserialize;
use std::collections::BTreeMap;

use super::transport::Transport;

/// Loopback address a published port forwards to. The model server binds here
/// and nowhere else; the publish is the only thing that widens its reach.
pub(crate) const LOOPBACK: &str = "127.0.0.1";

/// `tailscale serve status --json`, as much of it as we read.
///
/// Only the TCP forwards matter: this design never asks Tailscale to terminate
/// TLS or serve HTTP on our behalf, because the model server already speaks the
/// protocol the caller wants and putting a proxy in between would only add a
/// place for the two to disagree.
#[derive(Debug, Default, Deserialize)]
struct RawServeConfig {
    /// Keyed by port. Go renders integer map keys as strings, so these arrive
    /// as `"8000"` rather than `8000`.
    #[serde(rename = "TCP", default)]
    tcp: BTreeMap<String, RawTcpHandler>,
    /// Per-session configuration, used when a serve was started in the
    /// foreground. We always publish in the background, so anything here
    /// belongs to someone else — but a forward is a forward, and missing one
    /// would report a live endpoint as absent.
    #[serde(rename = "Foreground", default)]
    foreground: BTreeMap<String, RawForegroundConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct RawForegroundConfig {
    #[serde(rename = "TCP", default)]
    tcp: BTreeMap<String, RawTcpHandler>,
}

#[derive(Debug, Default, Deserialize, Clone)]
struct RawTcpHandler {
    /// Destination, as `host:port`. Absent for an HTTPS/HTTP handler, which is
    /// not something we create.
    #[serde(rename = "TCPForward", default)]
    tcp_forward: Option<String>,
}

/// What the remote's Tailscale says about one port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PublishState {
    /// The port forwards to the loopback address we expect.
    Published,
    /// The port is not forwarded at all.
    Absent,
    /// The port forwards somewhere else. Not ours to withdraw, and a warning
    /// that two things are competing for it.
    Foreign { forwards_to: String },
    /// The remote answered with something we could not read.
    ///
    /// Deliberately not folded into `Absent`. "I looked and there is no
    /// forward" and "I could not tell" differ exactly where it matters: the
    /// first confirms a withdrawal, the second must not, or a malformed reply
    /// becomes a report that an endpoint is gone while it is still published.
    Unreadable,
}

/// Ask the remote which of its ports are forwarded.
pub(crate) fn publish_state(
    transport: &dyn Transport,
    tailnet_port: u16,
    remote_port: u16,
) -> Result<PublishState> {
    let outcome = transport.exec("tailscale serve status --json")?;
    if !outcome.success {
        bail!(
            "could not read the remote's tailnet publishing state (exit {}): {}",
            outcome
                .code
                .map_or_else(|| "signal".to_owned(), |code| code.to_string()),
            outcome.stderr.trim()
        );
    }
    Ok(classify(&outcome.stdout, tailnet_port, remote_port))
}

/// Decide what a serve-status document says about one port. Pure, so the whole
/// classification is testable against fixtures.
fn classify(status_json: &str, tailnet_port: u16, remote_port: u16) -> PublishState {
    let trimmed = status_json.trim();
    // A machine publishing nothing prints an empty document; some versions
    // print literal `null` for an unset config. Neither is an error.
    if trimmed.is_empty() || trimmed == "null" {
        return PublishState::Absent;
    }
    let Ok(config) = serde_json::from_str::<RawServeConfig>(trimmed) else {
        return PublishState::Unreadable;
    };

    let key = tailnet_port.to_string();
    let handler = config.tcp.get(&key).or_else(|| {
        config
            .foreground
            .values()
            .find_map(|session| session.tcp.get(&key))
    });

    let Some(forward) = handler.and_then(|handler| handler.tcp_forward.clone()) else {
        return PublishState::Absent;
    };

    let expected = forward_target(remote_port);
    if forward == expected {
        PublishState::Published
    } else {
        PublishState::Foreign {
            forwards_to: forward,
        }
    }
}

/// Where a published port should point: the model server's loopback bind.
fn forward_target(remote_port: u16) -> String {
    format!("{LOOPBACK}:{remote_port}")
}

/// Command that declares the forward on the remote.
///
/// `--bg` is what makes it outlive the SSH command that issued it; without it
/// the publish would die with our connection and the endpoint would vanish the
/// moment `serve` returned.
fn publish_command(tailnet_port: u16, remote_port: u16) -> String {
    format!(
        "tailscale serve --bg --tcp={tailnet_port} tcp://{}",
        forward_target(remote_port)
    )
}

/// Command that removes the forward.
fn withdraw_command(tailnet_port: u16) -> String {
    format!("tailscale serve --tcp={tailnet_port} off")
}

/// Claim the port, declare the forward, then confirm the remote agrees.
///
/// Ownership is established *before* writing, not after. `tailscale serve`
/// overwrites whatever holds a port without complaint, so checking afterwards
/// is too late — by then the other forward is already gone and the state we
/// read back is our own, which reads as success. A second session reusing a
/// port would silently take the first one's endpoint away.
///
/// The confirmation afterwards is still needed, and is not ceremony:
/// `tailscale serve` can exit zero while the tailnet's policy declines to
/// publish, and trusting the exit code hands the user a URL that never answers.
pub(crate) fn publish(
    transport: &dyn Transport,
    tailnet_port: u16,
    remote_port: u16,
) -> Result<()> {
    match publish_state(transport, tailnet_port, remote_port)? {
        // Free, or already pointing where we want it. Re-declaring our own is
        // harmless and keeps `attach` idempotent.
        PublishState::Absent | PublishState::Published => {}
        PublishState::Foreign { forwards_to } => bail!(
            "port {tailnet_port} on the remote already forwards to {forwards_to}.\n\
             Refusing to take it over — publishing here would silently break whatever \
             is using it. Choose another port with `--tailnet-port`."
        ),
        PublishState::Unreadable => bail!(
            "port {tailnet_port} could not be checked before publishing, so there is no way \
             to tell whether something else is already using it.\n\
             Check it by hand: tailscale serve status"
        ),
    }

    let outcome = transport.exec(&publish_command(tailnet_port, remote_port))?;
    if !outcome.success {
        bail!(
            "the remote refused to publish port {tailnet_port} on the tailnet: {}\n\
             This is usually the tailnet's own policy. Check that the machine is allowed \
             to serve, then try again.",
            outcome.stderr.trim()
        );
    }

    match publish_state(transport, tailnet_port, remote_port)? {
        PublishState::Published => Ok(()),
        PublishState::Absent => bail!(
            "the remote accepted the publish for port {tailnet_port} but does not report it \
             as active, so the endpoint would not answer"
        ),
        // Something took the port between our check and our write.
        PublishState::Foreign { forwards_to } => bail!(
            "port {tailnet_port} on the remote now forwards to {forwards_to} rather than to \
             this model server; something else claimed it. Choose another port with \
             `--tailnet-port`."
        ),
        PublishState::Unreadable => bail!(
            "the remote accepted the publish for port {tailnet_port} but its reply could \
             not be read, so there is no way to confirm the endpoint answers.\n\
             Check it with `rocm remote status`."
        ),
    }
}

/// Remove the forward, and confirm it is gone.
///
/// Returns an error when withdrawal cannot be confirmed. Callers must not treat
/// that as cosmetic: because a publish is configuration rather than a process,
/// an unwithdrawn one survives reboots and keeps a GPU endpoint on the tailnet
/// with nothing tracking it.
pub(crate) fn withdraw(
    transport: &dyn Transport,
    tailnet_port: u16,
    remote_port: u16,
) -> Result<()> {
    // Establish it is ours before turning it off. `tailscale serve … off` takes a
    // port, not a forward, so it would happily tear down whatever is on that
    // port — including something another tool or another person put there after
    // our session was recorded.
    match publish_state(transport, tailnet_port, remote_port)? {
        PublishState::Published => {}
        // Already gone. Nothing to do, and nothing to complain about: teardown
        // has to be safe to retry after a partial one.
        PublishState::Absent => return Ok(()),
        PublishState::Foreign { forwards_to } => bail!(
            "port {tailnet_port} on the remote now forwards to {forwards_to}, not to this \
             session's model server.\n\
             Refusing to turn it off — it belongs to something else."
        ),
        PublishState::Unreadable => bail!(
            "port {tailnet_port} could not be checked before withdrawing it, so there is no \
             way to tell whether it is still this session's endpoint.\n\
             Check it by hand: tailscale serve status"
        ),
    }

    let outcome = transport.exec(&withdraw_command(tailnet_port))?;
    if !outcome.success {
        bail!(
            "failed to withdraw port {tailnet_port} on the remote: {}",
            outcome.stderr.trim()
        );
    }
    match publish_state(transport, tailnet_port, remote_port)? {
        PublishState::Absent | PublishState::Foreign { .. } => Ok(()),
        PublishState::Published => {
            bail!("the remote still reports port {tailnet_port} as published after withdrawing it")
        }
        // An unreadable reply is not a withdrawal. Accepting it here would be the
        // exact failure this function exists to prevent: reporting an endpoint
        // gone while it is still published, on a machine nobody is watching.
        PublishState::Unreadable => bail!(
            "port {tailnet_port} was asked to stop publishing, but the remote's reply could \
             not be read, so it cannot be confirmed withdrawn.\n\
             Check it by hand: tailscale serve status"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::transport::{ScriptedStep, ScriptedTransport};

    const PUBLISHED: &str = r#"{
      "TCP": { "8000": { "TCPForward": "127.0.0.1:11434" } }
    }"#;

    #[test]
    fn a_matching_forward_is_recognised_as_ours() {
        assert_eq!(classify(PUBLISHED, 8000, 11434), PublishState::Published);
    }

    #[test]
    fn an_empty_or_null_config_means_nothing_is_published() {
        // A machine that has never published prints an empty or null document.
        // Treating that as a parse failure would turn a normal state into noise.
        for document in ["", "   ", "null", "{}"] {
            assert_eq!(
                classify(document, 8000, 11434),
                PublishState::Absent,
                "{document:?}"
            );
        }
    }

    #[test]
    fn a_port_pointing_elsewhere_is_not_treated_as_ours() {
        // Withdrawing this would tear down whatever else is using the port.
        let other = r#"{"TCP": {"8000": {"TCPForward": "127.0.0.1:9999"}}}"#;
        assert_eq!(
            classify(other, 8000, 11434),
            PublishState::Foreign {
                forwards_to: "127.0.0.1:9999".to_owned()
            }
        );
    }

    #[test]
    fn an_https_handler_on_the_port_is_not_a_forward() {
        // A handler with no TCPForward terminates TLS and serves web content;
        // it is not a passthrough to our model server.
        let https = r#"{"TCP": {"8000": {"HTTPS": true}}}"#;
        assert_eq!(classify(https, 8000, 11434), PublishState::Absent);
    }

    #[test]
    fn a_foreground_publish_is_still_a_publish() {
        // We always publish in the background, but a forward someone else
        // started in the foreground still answers on that port. Missing it would
        // report a live endpoint as absent.
        let foreground = r#"{
          "Foreground": { "sess-1": { "TCP": { "8000": { "TCPForward": "127.0.0.1:11434" } } } }
        }"#;
        assert_eq!(classify(foreground, 8000, 11434), PublishState::Published);
    }

    #[test]
    fn ports_are_matched_as_string_keys_not_numbers() {
        // Go renders integer map keys as strings; looking for a numeric key
        // would silently never match and report every endpoint as absent.
        assert_eq!(classify(PUBLISHED, 8001, 11434), PublishState::Absent);
    }

    #[test]
    fn publishing_runs_in_the_background_and_targets_loopback() {
        // Without --bg the forward dies with the SSH command that made it, and
        // the endpoint vanishes the moment serve returns.
        let command = publish_command(8000, 11434);
        assert!(command.contains("--bg"), "{command}");
        assert!(command.contains("--tcp=8000"), "{command}");
        assert!(command.contains("tcp://127.0.0.1:11434"), "{command}");
    }

    #[test]
    fn a_publish_the_remote_does_not_confirm_is_an_error() {
        // `tailscale serve` can exit zero while tailnet policy declines to
        // publish. Trusting the exit code hands out a URL that never answers.
        // Free before, still nothing after: the daemon accepted and did nothing.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve --bg", ""),
            ScriptedStep::ok("tailscale serve status --json", "{}"),
        ]);
        let error = publish(&transport, 8000, 11434).unwrap_err().to_string();
        assert!(error.contains("would not answer"), "{error}");
    }

    #[test]
    fn a_confirmed_publish_succeeds() {
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve --bg", ""),
            ScriptedStep::ok("tailscale serve status --json", PUBLISHED),
        ]);
        publish(&transport, 8000, 11434).expect("publish confirmed");
    }

    #[test]
    fn a_port_someone_else_is_using_is_not_taken_over() {
        // `tailscale serve` overwrites a port without complaint, so a check
        // after the write is too late: the other forward is already gone and
        // what we read back is our own. A second session on the same port would
        // silently take the first one's endpoint away.
        let occupied = ScriptedTransport::new(vec![ScriptedStep::ok(
            "tailscale serve status --json",
            r#"{"TCP": {"8000": {"TCPForward": "127.0.0.1:9999"}}}"#,
        )]);

        let error = publish(&occupied, 8000, 11434).unwrap_err().to_string();
        assert!(error.contains("Refusing to take it over"), "{error}");
        assert!(
            !occupied.calls().iter().any(|call| matches!(
                call,
                crate::remote::transport::TransportCall::Exec { command, .. }
                    if command.contains("--bg")
            )),
            "nothing should have been written: {:?}",
            occupied.calls()
        );
    }

    #[test]
    fn re_publishing_our_own_forward_is_allowed() {
        // `attach` re-declares an endpoint that is already ours; that has to
        // stay idempotent rather than tripping the ownership guard.
        let ours = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve status --json", PUBLISHED),
            ScriptedStep::ok("tailscale serve --bg", ""),
        ]);
        publish(&ours, 8000, 11434).expect("re-publishing our own forward");
    }

    #[test]
    fn a_refused_publish_points_at_tailnet_policy() {
        // The port is free; the daemon refuses the write itself.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve status --json", "{}"),
            ScriptedStep::fails("tailscale serve --bg", 1, "serve not allowed"),
        ]);
        let error = publish(&transport, 8000, 11434).unwrap_err().to_string();
        assert!(error.contains("policy"), "{error}");
    }

    #[test]
    fn a_port_that_now_belongs_to_something_else_is_not_turned_off() {
        // `serve … off` takes a port, not a forward, so without this check a
        // teardown tears down whatever happens to hold the port — possibly
        // another tool's, or another person's, put there after our session was
        // recorded.
        let hijacked = ScriptedTransport::new(vec![ScriptedStep::ok(
            "tailscale serve status --json",
            r#"{"TCP": {"8000": {"TCPForward": "127.0.0.1:9999"}}}"#,
        )]);
        let error = withdraw(&hijacked, 8000, 11434).unwrap_err().to_string();
        assert!(error.contains("belongs to something else"), "{error}");
        // And nothing was turned off.
        assert!(
            !hijacked
                .calls()
                .iter()
                .any(|call| matches!(call, crate::remote::transport::TransportCall::Exec { command, .. } if command.contains("off"))),
            "a foreign forward must not be touched"
        );
    }

    #[test]
    fn withdrawing_an_already_absent_endpoint_is_not_an_error() {
        // Teardown has to be safe to retry after a partial one.
        let gone = ScriptedTransport::new(vec![ScriptedStep::ok(
            "tailscale serve status --json",
            "{}",
        )]);
        withdraw(&gone, 8000, 11434).expect("already gone is success");
    }

    #[test]
    fn withdrawal_is_confirmed_not_assumed() {
        // A publish outlives a reboot, so an unwithdrawn one is a GPU endpoint
        // left on the tailnet with nothing tracking it.
        // Ownership is probed first, so a stubborn remote answers PUBLISHED both
        // before and after the `off` — which is exactly the state that must fail.
        let stubborn = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve --tcp=8000 off", ""),
            ScriptedStep::ok("tailscale serve status --json", PUBLISHED),
        ]);
        let error = withdraw(&stubborn, 8000, 11434).unwrap_err().to_string();
        assert!(error.contains("still reports"), "{error}");
    }
}
