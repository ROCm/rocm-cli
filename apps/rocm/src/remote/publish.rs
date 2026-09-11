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
//! publish is visible to the whole tailnet, scoped only by its ACLs. That
//! assumes Tailscale Funnel is not already enabled for the port — Funnel is
//! what turns a tailnet-scoped forward into one reachable from the public
//! internet, and it is not something this module ever asks for. If the remote
//! already has it allowed for the target port, [`classify`] reports it as a
//! distinct state rather than folding it into "free", and both [`publish`]
//! and [`withdraw`] refuse to act until it is turned off by hand. Unlike a
//! point-to-point tunnel a publish also *outlives a reboot*, because it is
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

/// Keys a real serve config carries. Seeing none of them in a non-empty
/// document means we were handed something else, whatever it parses as.
///
/// Not all of these are actually inspected. `TCP`, `Foreground`, `Services`,
/// and `AllowFunnel` are parsed into [`RawServeConfig`] and drive
/// [`classify`]. `Web` is listed only as document-shape evidence — it
/// confirms we are looking at a real serve config, but we never read an
/// HTTPS/HTTP handler, because this design never asks Tailscale to terminate
/// TLS on our behalf. Listing a key here without parsing it is exactly what let
/// `AllowFunnel` go unchecked for a release, so if a key stays evidence-only,
/// say so here rather than leaving a reader to assume otherwise.
const SERVE_CONFIG_KEYS: &[&str] = &["TCP", "Web", "Services", "AllowFunnel", "Foreground"];

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
    foreground: BTreeMap<String, RawNestedConfig>,
    /// Tailscale "Services" (VIP services), keyed by service name. Nests its
    /// own `TCP` map the same way `Foreground` does — a forward declared here
    /// is still a forward, and missing it has the same failure mode as
    /// missing a foreground one: a live foreign endpoint reads as absent, and
    /// the `Foreign` ownership guard never fires.
    #[serde(rename = "Services", default)]
    services: BTreeMap<String, RawNestedConfig>,
    /// Keyed `host:port`. `true` means the port is exposed to the public
    /// internet via Tailscale Funnel, not just the tailnet — something this
    /// module never asks for. We only ever check this for our own target
    /// port, so we do not track which host set it.
    #[serde(rename = "AllowFunnel", default)]
    allow_funnel: BTreeMap<String, bool>,
}

/// Shape shared by `Foreground` sessions and `Services` entries: both nest a
/// `TCP` map under their own key.
#[derive(Debug, Default, Deserialize)]
struct RawNestedConfig {
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
    /// Tailscale Funnel is allowed for this port, regardless of what (if
    /// anything) forwards to it. Funnel is what exposes a port to the public
    /// internet rather than just the tailnet, and this module never turns it
    /// on. Deliberately not folded into `Absent` or `Published`: publishing
    /// over it would complete a public exposure nobody asked this command
    /// for, and withdrawing our forward while it stays on would not close
    /// anything.
    FunnelAllowed,
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
    // Parse loosely first. `RawServeConfig` defaults every field, so a document
    // that is valid JSON but not a serve config — an error object, a newer shape
    // we do not know — would deserialize to an empty config and read as "nothing
    // published". After a withdrawal that is indistinguishable from success,
    // which is the failure this whole module exists to avoid.
    let Ok(document) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return PublishState::Unreadable;
    };
    let Some(fields) = document.as_object() else {
        return PublishState::Unreadable;
    };
    // An empty object is a real, legitimate answer: nothing is published.
    if !fields.is_empty()
        && !fields
            .keys()
            .any(|key| SERVE_CONFIG_KEYS.contains(&key.as_str()))
    {
        return PublishState::Unreadable;
    }
    let Ok(config) = serde_json::from_value::<RawServeConfig>(document) else {
        return PublishState::Unreadable;
    };

    let key = tailnet_port.to_string();

    // Checked before we ask whether anything forwards to the port at all: a
    // Funnel-enabled port is a distinct hazard independent of what (if
    // anything) currently forwards to it, and must never be read as merely
    // "free" or folded into a normal `Published`/`Absent` result.
    if funnel_allowed_for_port(&config.allow_funnel, tailnet_port) {
        return PublishState::FunnelAllowed;
    }

    let handler = config
        .tcp
        .get(&key)
        .or_else(|| {
            config
                .foreground
                .values()
                .find_map(|session| session.tcp.get(&key))
        })
        .or_else(|| config.services.values().find_map(|svc| svc.tcp.get(&key)));

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

/// True if Funnel is allowed for `port` in an `AllowFunnel` map.
///
/// `AllowFunnel` is keyed `host:port`, where the host is a tailnet DNS name we
/// do not otherwise track. We only care whether *our* port is exposed, so we
/// match on the port suffix rather than requiring an exact key.
fn funnel_allowed_for_port(allow_funnel: &BTreeMap<String, bool>, port: u16) -> bool {
    let port = port.to_string();
    allow_funnel
        .iter()
        .any(|(host_port, allowed)| *allowed && host_port.rsplit(':').next() == Some(port.as_str()))
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
        PublishState::FunnelAllowed => bail!(
            "port {tailnet_port} on the remote has Tailscale Funnel allowed, which exposes it \
             to the public internet rather than just the tailnet.\n\
             Refusing to publish over it — turn Funnel off first: \
             `tailscale funnel --tcp={tailnet_port} off`."
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
        // Funnel was turned on between our check and our write. Reported the
        // same way as the pre-check case: publishing must not be allowed to
        // complete a public-internet exposure nobody asked for.
        PublishState::FunnelAllowed => bail!(
            "port {tailnet_port} on the remote now has Tailscale Funnel allowed, exposing it to \
             the public internet rather than just the tailnet.\n\
             Turn Funnel off: `tailscale funnel --tcp={tailnet_port} off`, then try again."
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
        PublishState::FunnelAllowed => bail!(
            "port {tailnet_port} on the remote has Tailscale Funnel allowed. Withdrawing our \
             forward would not close the public-internet exposure, so this needs a human \
             decision, not a silent teardown.\n\
             Turn Funnel off first: `tailscale funnel --tcp={tailnet_port} off`."
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
        // Our forward is gone, but Funnel is still allowed for the port. The
        // port may still be reachable from the public internet, so this is
        // not the clean close the caller asked for.
        PublishState::FunnelAllowed => bail!(
            "port {tailnet_port} was turned off, but Tailscale Funnel is still allowed for it, \
             so it may still be reachable from the public internet.\n\
             Turn it off: `tailscale funnel --tcp={tailnet_port} off`."
        ),
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
    fn output_we_cannot_read_is_never_reported_as_nothing_published() {
        // The distinction the module turns on. Folded into `Absent`, a garbled
        // or unexpected reply after a withdrawal reads as "gone" and the
        // endpoint stays up with nobody tracking it.
        //
        // Two shapes matter. Not JSON at all:
        for document in ["not json", "{\"TCP\": ", "<html>an error page</html>"] {
            assert_eq!(
                classify(document, 8000, 11434),
                PublishState::Unreadable,
                "{document:?}"
            );
        }
        // And valid JSON that is not a serve config. Every field defaults, so
        // without the key check these deserialize to an empty config and look
        // like a machine publishing nothing.
        for document in [
            r#"{"error": "not logged in"}"#,
            r#"{"SomeFutureShape": {"TCP": {}}}"#,
            "[]",
            r#""a string""#,
            "42",
        ] {
            assert_eq!(
                classify(document, 8000, 11434),
                PublishState::Unreadable,
                "{document}"
            );
        }

        // But an empty object is a real answer, and so is a config carrying a
        // key we know even when our port is absent from it.
        assert_eq!(classify("{}", 8000, 11434), PublishState::Absent);
        assert_eq!(
            classify(r#"{"Web": {}}"#, 8000, 11434),
            PublishState::Absent
        );
    }

    #[test]
    fn neither_publish_nor_withdraw_accepts_an_unreadable_reply() {
        // Withdrawal is the dangerous half: accepting this reports an endpoint
        // torn down while it is still live.
        let garbled = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve --tcp=8000 off", ""),
            ScriptedStep::ok("tailscale serve status --json", r#"{"unexpected": true}"#),
        ]);
        let error = withdraw(&garbled, 8000, 11434).unwrap_err().to_string();
        assert!(error.contains("could not be checked"), "{error}");

        let publishing = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve --bg", ""),
            ScriptedStep::ok("tailscale serve status --json", r#"{"unexpected": true}"#),
        ]);
        let error = publish(&publishing, 8000, 11434).unwrap_err().to_string();
        assert!(error.contains("could not be checked"), "{error}");
    }

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
    fn a_service_forward_is_still_a_forward() {
        // Same blind spot as `Foreground`: a forward declared under a named
        // Service still answers on that port. Missing it defeats the
        // `Foreign` ownership guard for whatever it points to.
        let service = r#"{
          "Services": { "svc:my-app": { "TCP": { "8000": { "TCPForward": "127.0.0.1:9999" } } } }
        }"#;
        assert_eq!(
            classify(service, 8000, 11434),
            PublishState::Foreign {
                forwards_to: "127.0.0.1:9999".to_owned()
            }
        );
    }

    #[test]
    fn our_own_forward_under_a_service_is_recognised() {
        let service = r#"{
          "Services": { "svc:my-app": { "TCP": { "8000": { "TCPForward": "127.0.0.1:11434" } } } }
        }"#;
        assert_eq!(classify(service, 8000, 11434), PublishState::Published);
    }

    #[test]
    fn funnel_allowed_on_our_port_is_never_read_as_free() {
        // No matching TCPForward at all: without the AllowFunnel check this
        // reads as `Absent`, and `publish()` would happily complete the
        // exposure it was never asked to create.
        let funnel_only = r#"{"AllowFunnel": {"my-machine.tail1234.ts.net:8000": true}}"#;
        assert_eq!(
            classify(funnel_only, 8000, 11434),
            PublishState::FunnelAllowed
        );
    }

    #[test]
    fn funnel_allowed_overrides_a_matching_forward() {
        // Even when our own forward is in place, Funnel being allowed means
        // the port is reachable from the public internet, not just the
        // tailnet. That must not be reported as an ordinary `Published`.
        let both = r#"{
          "TCP": { "8000": { "TCPForward": "127.0.0.1:11434" } },
          "AllowFunnel": { "my-machine.tail1234.ts.net:8000": true }
        }"#;
        assert_eq!(classify(both, 8000, 11434), PublishState::FunnelAllowed);
    }

    #[test]
    fn funnel_allowed_on_another_port_does_not_affect_ours() {
        let other_port = r#"{"AllowFunnel": {"my-machine.tail1234.ts.net:9000": true}}"#;
        assert_eq!(classify(other_port, 8000, 11434), PublishState::Absent);
    }

    #[test]
    fn funnel_disabled_entry_does_not_trip_the_guard() {
        // The map can carry `false` entries for a port Funnel was allowed for
        // and then turned off. Only `true` matters.
        let disabled = r#"{"AllowFunnel": {"my-machine.tail1234.ts.net:8000": false}}"#;
        assert_eq!(classify(disabled, 8000, 11434), PublishState::Absent);
    }

    #[test]
    fn publish_refuses_a_funnel_enabled_port() {
        let transport = ScriptedTransport::new(vec![ScriptedStep::ok(
            "tailscale serve status --json",
            r#"{"AllowFunnel": {"my-machine.tail1234.ts.net:8000": true}}"#,
        )]);
        let error = publish(&transport, 8000, 11434).unwrap_err().to_string();
        assert!(error.contains("tailscale funnel --tcp=8000 off"), "{error}");
        // Nothing should have been written.
        assert!(
            !transport.calls().iter().any(|call| matches!(
                call,
                crate::remote::transport::TransportCall::Exec { command, .. }
                    if command.contains("--bg")
            )),
            "{:?}",
            transport.calls()
        );
    }

    #[test]
    fn withdraw_refuses_a_funnel_enabled_port() {
        let transport = ScriptedTransport::new(vec![ScriptedStep::ok(
            "tailscale serve status --json",
            r#"{"AllowFunnel": {"my-machine.tail1234.ts.net:8000": true}}"#,
        )]);
        let error = withdraw(&transport, 8000, 11434).unwrap_err().to_string();
        assert!(error.contains("tailscale funnel --tcp=8000 off"), "{error}");
        assert!(
            !transport.calls().iter().any(|call| matches!(
                call,
                crate::remote::transport::TransportCall::Exec { command, .. }
                    if command.contains(" off")
            )),
            "{:?}",
            transport.calls()
        );
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
