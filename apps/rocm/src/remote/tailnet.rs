// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Reading the local tailnet: which machines exist, and how to address one.
//!
//! Everything here talks to the *local* Tailscale daemon and nothing else. No
//! SSH, no session, no call to a peer — listing candidate machines must not
//! require being able to reach them, or discovery would only ever show you what
//! you already knew how to contact.
//!
//! Two consumers: `rocm remote targets`, which renders the list, and the serve
//! path, which resolves one user-supplied name to a peer. They share a resolver
//! so a name that lists is a name that serves.
//!
//! Authorization is deliberately absent. Which peers a user may reach is the
//! tailnet admin's ACL policy, enforced by Tailscale; this module reports what
//! `tailscale status` already says and passes that tool's failures through
//! unchanged rather than reinterpreting them.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::ErrorKind;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Backend state the Tailscale daemon reports when it is actually usable.
const BACKEND_RUNNING: &str = "Running";

/// What the local machine's Tailscale looks like right now.
///
/// Three outcomes rather than a `Result`, because callers treat them
/// differently: `targets` reports the first two calmly and exits successfully,
/// while serving over the tailnet cannot proceed without the third.
#[derive(Debug, Clone)]
pub(crate) enum TailnetAvailability {
    /// No `tailscale` on `PATH`.
    NotInstalled,
    /// Installed, but the daemon is not in a usable state — most often the user
    /// has not run `tailscale up` yet.
    NotRunning {
        backend_state: String,
    },
    Running(TailnetStatus),
}

/// The local view of the tailnet.
#[derive(Debug, Clone)]
pub(crate) struct TailnetStatus {
    /// This machine, when the daemon reports it.
    pub(crate) this_machine: Option<TailnetPeer>,
    /// Every other machine on the tailnet, ordered by host name so output is
    /// stable between runs rather than following a hash map's iteration order.
    pub(crate) peers: Vec<TailnetPeer>,
    /// True when Tailscale is running without a kernel network device.
    ///
    /// There is no routable local address in that mode, so `ssh` cannot dial a
    /// tailnet IP directly and has to be routed through Tailscale's own relay
    /// instead. Taken from the daemon's `TUN` flag, whose documented meaning is
    /// exactly this — worth preferring over inferring the mode from a failed
    /// connection after the fact.
    pub(crate) userspace_networking: bool,
}

/// One machine on the tailnet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TailnetPeer {
    /// Short host name. Not guaranteed unique across a tailnet.
    pub(crate) host: String,
    /// Fully-qualified MagicDNS name, trailing dot removed.
    pub(crate) dns_name: String,
    pub(crate) addresses: Vec<String>,
    pub(crate) os: String,
    pub(crate) tags: Vec<String>,
    pub(crate) online: bool,
}

impl TailnetPeer {
    /// The name to hand to `ssh` and to build an endpoint URL from.
    ///
    /// Prefers the MagicDNS name over a raw address: it survives the peer being
    /// reassigned an address, and it is what a user recognises in a URL.
    pub(crate) fn endpoint_host(&self) -> &str {
        if self.dns_name.is_empty() {
            self.addresses.first().map_or("", String::as_str)
        } else {
            &self.dns_name
        }
    }

    fn has_tag(&self, tag: &str) -> bool {
        // Accept `gpu` for `tag:gpu`: the `tag:` prefix is Tailscale's wire
        // format, and making users type it adds nothing.
        let wanted = tag.strip_prefix("tag:").unwrap_or(tag);
        self.tags
            .iter()
            .any(|owned| owned.strip_prefix("tag:").unwrap_or(owned) == wanted)
    }

    /// Whether `needle` names this peer exactly, by DNS name, host name, or
    /// address.
    fn matches_exactly(&self, needle: &str) -> bool {
        let needle = needle.trim_end_matches('.');
        self.dns_name.eq_ignore_ascii_case(needle)
            || self.host.eq_ignore_ascii_case(needle)
            || self.addresses.iter().any(|address| address == needle)
    }

    /// Whether `needle` appears in this peer's names — the loose fallback used
    /// only when nothing matched exactly.
    fn matches_loosely(&self, needle: &str) -> bool {
        let needle = needle.to_ascii_lowercase();
        self.host.to_ascii_lowercase().contains(&needle)
            || self.dns_name.to_ascii_lowercase().contains(&needle)
    }
}

/// `tailscale status --json`, as much of it as we read.
///
/// Unknown fields are ignored by default, so a newer Tailscale adding output
/// cannot break us. `backend_state` is the one field with no default: every
/// real status document has it, so its absence means we were handed something
/// that is not a status document at all, and failing there gives a far better
/// error than a confidently empty peer list.
#[derive(Debug, Deserialize)]
struct RawStatus {
    #[serde(rename = "BackendState")]
    backend_state: String,
    #[serde(rename = "TUN", default)]
    tun: bool,
    #[serde(rename = "Self")]
    this_machine: Option<RawPeer>,
    #[serde(rename = "Peer", default)]
    peers: BTreeMap<String, RawPeer>,
}

#[derive(Debug, Deserialize)]
struct RawPeer {
    #[serde(rename = "HostName", default)]
    host_name: String,
    #[serde(rename = "DNSName", default)]
    dns_name: String,
    #[serde(rename = "OS", default)]
    os: String,
    #[serde(rename = "TailscaleIPs", default)]
    tailscale_ips: Vec<String>,
    #[serde(rename = "Tags", default)]
    tags: Vec<String>,
    #[serde(rename = "Online", default)]
    online: bool,
}

impl RawPeer {
    fn into_peer(self) -> TailnetPeer {
        TailnetPeer {
            host: self.host_name,
            // MagicDNS names arrive as absolute FQDNs with a trailing dot.
            // Leaving it on produces `http://box.tail.ts.net.:8000` in printed
            // URLs, which is technically valid and looks like a typo.
            dns_name: self.dns_name.trim_end_matches('.').to_owned(),
            addresses: self.tailscale_ips,
            os: self.os,
            tags: self.tags,
            online: self.online,
        }
    }
}

/// Ask the local Tailscale daemon what it can see.
pub(crate) fn local_status() -> Result<TailnetAvailability> {
    let output = match Command::new("tailscale")
        .args(["status", "--json"])
        .stdin(Stdio::null())
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return Ok(TailnetAvailability::NotInstalled);
        }
        Err(error) => {
            return Err(error).context("failed to run `tailscale status --json`");
        }
    };

    if !output.status.success() {
        // Pass Tailscale's own words through. It knows why it is unhappy —
        // logged out, daemon not running, permission denied — and paraphrasing
        // that into our own vocabulary only loses detail.
        bail!(
            "`tailscale status --json` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let status = parse_status(&String::from_utf8_lossy(&output.stdout))?;
    if status_is_running(&status.0) {
        Ok(TailnetAvailability::Running(status.1))
    } else {
        Ok(TailnetAvailability::NotRunning {
            backend_state: status.0,
        })
    }
}

fn status_is_running(backend_state: &str) -> bool {
    backend_state == BACKEND_RUNNING
}

/// Parse `tailscale status --json`, returning the backend state alongside the
/// tailnet view. Pure, so the whole surface is testable against fixtures.
fn parse_status(json: &str) -> Result<(String, TailnetStatus)> {
    let raw: RawStatus = serde_json::from_str(json).context(
        "failed to parse `tailscale status --json`; the output was not a status document",
    )?;

    let mut peers = raw
        .peers
        .into_values()
        .map(RawPeer::into_peer)
        .collect::<Vec<_>>();
    // A map gives no useful order, so impose one. Sorting by the name a user
    // would type keeps repeated runs comparable.
    peers.sort_by(|left, right| {
        left.host
            .to_ascii_lowercase()
            .cmp(&right.host.to_ascii_lowercase())
            .then_with(|| left.dns_name.cmp(&right.dns_name))
    });

    Ok((
        raw.backend_state,
        TailnetStatus {
            this_machine: raw.this_machine.map(RawPeer::into_peer),
            peers,
            userspace_networking: !raw.tun,
        },
    ))
}

/// Find the peer a user meant by `needle`.
///
/// `Ok(None)` means no peer matched — a fact, not a failure, so the caller can
/// decide whether that is fatal. An ambiguous name *is* an error: silently
/// picking one of several GPU boxes is the kind of guess that gets a model
/// served on someone else's machine.
// Resolution has no caller until a command takes a single target — `serve` and
// `doctor` are the first. It ships with discovery because it is the same
// question ("which machine did you mean") and shares the peer list. Remove this
// attribute in the change that adds the serve path.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_peer<'a>(
    status: &'a TailnetStatus,
    needle: &str,
) -> Result<Option<&'a TailnetPeer>> {
    // An SSH destination may carry a user prefix; the tailnet knows nothing
    // about that, so compare on the host part.
    let needle = needle.rsplit('@').next().unwrap_or(needle).trim();
    if needle.is_empty() {
        return Ok(None);
    }

    if let Some(peer) = status
        .peers
        .iter()
        .find(|peer| peer.matches_exactly(needle))
    {
        return Ok(Some(peer));
    }

    let loose = status
        .peers
        .iter()
        .filter(|peer| peer.matches_loosely(needle))
        .collect::<Vec<_>>();
    match loose.as_slice() {
        [] => Ok(None),
        [only] => Ok(Some(only)),
        several => bail!(
            "`{needle}` matches more than one machine on the tailnet: {}\n\
             Name one of them exactly.",
            several
                .iter()
                .map(|peer| peer.host.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Render the candidate machines.
///
/// Purely informational, and says so: being listed here means the tailnet can
/// see the machine, not that it has a GPU, ROCm, or the CLI. Each entry points
/// at the command that actually answers that question.
pub(crate) fn render_targets(status: &TailnetStatus, tag: Option<&str>) -> String {
    let peers = status
        .peers
        .iter()
        .filter(|peer| tag.is_none_or(|tag| peer.has_tag(tag)))
        .collect::<Vec<_>>();

    let mut output = String::new();
    let _ = writeln!(output, "Remote Targets");
    let _ = writeln!(output);
    // Name the machine we are looking *from*. A user on more than one tailnet
    // otherwise has no way to tell which one this list describes.
    if let Some(this_machine) = &status.this_machine {
        let _ = writeln!(output, "This machine: {}", this_machine.host);
    }
    let online = peers.iter().filter(|peer| peer.online).count();
    let _ = writeln!(
        output,
        "Status: {online} online, {} offline",
        peers.len() - online
    );
    let _ = writeln!(output);

    if peers.is_empty() {
        let _ = match tag {
            Some(tag) => writeln!(output, "No tailnet machines are tagged `{tag}`."),
            None => writeln!(output, "No other machines are on this tailnet."),
        };
        return output;
    }

    for peer in peers {
        let _ = writeln!(output, "- {}", peer.host);
        let _ = writeln!(output, "  address: {}", peer.endpoint_host());
        if !peer.addresses.is_empty() {
            let _ = writeln!(output, "  ip: {}", peer.addresses.join(", "));
        }
        if !peer.os.is_empty() {
            let _ = writeln!(output, "  os: {}", peer.os);
        }
        if !peer.tags.is_empty() {
            let _ = writeln!(output, "  tags: {}", peer.tags.join(", "));
        }
        let _ = writeln!(
            output,
            "  online: {}",
            if peer.online { "yes" } else { "no" }
        );
        let _ = writeln!(output, "  check: rocm remote doctor {}", peer.host);
    }

    let _ = writeln!(output);
    if status.userspace_networking {
        // Worth saying out loud: in this mode traffic cannot take a direct path
        // and is relayed instead, which caps throughput. Someone about to serve
        // a model should know that before they blame the GPU.
        let _ = writeln!(
            output,
            "Note: Tailscale is running here without a network device, so traffic to these"
        );
        let _ = writeln!(
            output,
            "machines is relayed rather than sent directly, which limits throughput."
        );
        let _ = writeln!(output);
    }
    let _ = writeln!(
        output,
        "Listed machines are reachable on the tailnet. That does not mean they have a GPU,"
    );
    let _ = writeln!(
        output,
        "ROCm, or the ROCm CLI — run `rocm remote doctor <machine>` to find out."
    );
    if tag.is_none() {
        let _ = writeln!(
            output,
            "Narrow this list with `--tag <tag>` once your tailnet tags its GPU machines."
        );
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped after a real `tailscale status --json`: absolute MagicDNS names,
    /// a peer map keyed by node public key, tags only on some machines, and one
    /// offline host.
    const FIXTURE: &str = r#"{
      "Version": "1.999.0-t01b2c3",
      "TUN": true,
      "BackendState": "Running",
      "MagicDNSSuffix": "example-tailnet.ts.net",
      "Self": {
        "ID": "nSELF",
        "HostName": "laptop",
        "DNSName": "laptop.example-tailnet.ts.net.",
        "OS": "linux",
        "TailscaleIPs": ["100.88.0.9"],
        "Online": true
      },
      "Peer": {
        "nodekey:bbb": {
          "ID": "nBBB",
          "HostName": "gpu-box-2",
          "DNSName": "gpu-box-2.example-tailnet.ts.net.",
          "OS": "linux",
          "TailscaleIPs": ["100.88.14.37"],
          "Tags": ["tag:gpu"],
          "Online": false,
          "LastSeen": "2026-08-30T11:02:41Z"
        },
        "nodekey:aaa": {
          "ID": "nAAA",
          "HostName": "gpu-box-1",
          "DNSName": "gpu-box-1.example-tailnet.ts.net.",
          "OS": "linux",
          "TailscaleIPs": ["100.88.14.21", "fd7a:115c:a1e0::3"],
          "Tags": ["tag:gpu", "tag:prod"],
          "Online": true
        },
        "nodekey:ccc": {
          "ID": "nCCC",
          "HostName": "phone",
          "DNSName": "phone.example-tailnet.ts.net.",
          "OS": "iOS",
          "TailscaleIPs": ["100.88.51.6"],
          "Online": true
        }
      }
    }"#;

    fn fixture() -> TailnetStatus {
        parse_status(FIXTURE).expect("fixture parses").1
    }

    #[test]
    fn parsing_orders_peers_and_strips_the_magicdns_trailing_dot() {
        let status = fixture();

        // A peer map has no inherent order; unsorted output would shuffle
        // between runs and make the listing unreadable.
        assert_eq!(
            status
                .peers
                .iter()
                .map(|peer| peer.host.as_str())
                .collect::<Vec<_>>(),
            vec!["gpu-box-1", "gpu-box-2", "phone"]
        );
        // Left on, the trailing dot shows up in printed URLs looking like a typo.
        assert_eq!(status.peers[0].dns_name, "gpu-box-1.example-tailnet.ts.net");
        assert_eq!(status.this_machine.unwrap().host, "laptop");
    }

    #[test]
    fn parsing_tolerates_a_newer_tailscale_but_rejects_a_non_status_document() {
        // Tailscale adds output over time; new keys must not break us.
        let with_extras = FIXTURE.replace(
            "\"BackendState\": \"Running\",",
            "\"BackendState\": \"Running\", \"SomethingAddedLater\": {\"a\": 1},",
        );
        assert!(parse_status(&with_extras).is_ok());

        // But a document with no backend state is not a status document, and
        // treating it as one would report an empty tailnet with full confidence.
        let not_a_status = r#"{"Peer": {}}"#;
        let error = parse_status(not_a_status).unwrap_err().to_string();
        assert!(error.contains("not a status document"), "{error}");
    }

    #[test]
    fn userspace_networking_is_read_from_the_daemon_not_guessed() {
        // With a kernel device, ssh can dial a tailnet address directly.
        assert!(!fixture().userspace_networking);

        // Without one there is no routable address, and the connection has to be
        // relayed. Tailscale reports this directly, so there is no need to infer
        // it from a connection that already failed.
        let userspace = FIXTURE.replace("\"TUN\": true", "\"TUN\": false");
        assert!(parse_status(&userspace).unwrap().1.userspace_networking);
    }

    #[test]
    fn a_daemon_that_is_not_up_is_reported_as_such_not_as_an_empty_tailnet() {
        let logged_out = FIXTURE.replace(
            "\"BackendState\": \"Running\"",
            "\"BackendState\": \"NeedsLogin\"",
        );
        let (state, _) = parse_status(&logged_out).expect("still parses");
        assert!(!status_is_running(&state));
        assert_eq!(state, "NeedsLogin");
    }

    #[test]
    fn a_peer_resolves_by_host_name_dns_name_or_address() {
        let status = fixture();
        for needle in [
            "gpu-box-1",
            "GPU-BOX-1",
            "gpu-box-1.example-tailnet.ts.net",
            "gpu-box-1.example-tailnet.ts.net.",
            "100.88.14.21",
            "user@gpu-box-1",
        ] {
            let peer = resolve_peer(&status, needle)
                .unwrap_or_else(|error| panic!("{needle}: {error}"))
                .unwrap_or_else(|| panic!("{needle} should resolve"));
            assert_eq!(peer.host, "gpu-box-1", "resolving {needle}");
        }
    }

    #[test]
    fn an_ambiguous_name_is_refused_and_names_the_candidates() {
        // Quietly picking one of two GPU boxes would serve a model on a machine
        // the user did not choose.
        let status = fixture();
        let error = resolve_peer(&status, "gpu-box").unwrap_err().to_string();
        assert!(error.contains("gpu-box-1"), "{error}");
        assert!(error.contains("gpu-box-2"), "{error}");
    }

    #[test]
    fn an_unknown_name_is_absent_rather_than_an_error() {
        // Not finding a peer is a fact the caller judges, not a failure here.
        assert!(
            resolve_peer(&fixture(), "not-on-this-tailnet")
                .expect("no match is not an error")
                .is_none()
        );
    }

    #[test]
    fn an_exact_name_wins_over_a_longer_one_that_contains_it() {
        let status = parse_status(&FIXTURE.replace(
            "\"HostName\": \"phone\"",
            "\"HostName\": \"gpu-box-1-spare\"",
        ))
        .unwrap()
        .1;
        let peer = resolve_peer(&status, "gpu-box-1")
            .expect("exact match must not be ambiguous")
            .expect("resolves");
        assert_eq!(peer.host, "gpu-box-1");
    }

    #[test]
    fn targets_can_be_narrowed_to_tagged_machines() {
        let status = fixture();

        let all = render_targets(&status, None);
        assert!(all.contains("- gpu-box-1"));
        assert!(all.contains("- phone"));

        // The `tag:` prefix is Tailscale's wire format; users should not have to
        // type it, so both spellings filter the same way.
        for spelling in ["gpu", "tag:gpu"] {
            let tagged = render_targets(&status, Some(spelling));
            assert!(tagged.contains("- gpu-box-1"), "{spelling}");
            assert!(!tagged.contains("- phone"), "{spelling}");
            assert!(tagged.contains("Status: 1 online, 1 offline"), "{spelling}");
        }
    }

    #[test]
    fn targets_states_that_listing_is_not_readiness() {
        // The whole risk of this command is being read as "these machines can
        // serve". It has to say plainly that it means no such thing.
        let rendered = render_targets(&fixture(), None);
        assert!(
            rendered.contains("does not mean they have a GPU"),
            "{rendered}"
        );
        assert!(
            rendered.contains("rocm remote doctor gpu-box-1"),
            "{rendered}"
        );
        assert!(
            rendered.contains("online: no"),
            "an offline peer is shown, not hidden"
        );
    }

    #[test]
    fn targets_names_the_machine_the_list_is_seen_from() {
        // Someone on more than one tailnet cannot otherwise tell which one they
        // are looking at.
        assert!(render_targets(&fixture(), None).contains("This machine: laptop"));
    }

    #[test]
    fn targets_warns_when_traffic_to_peers_will_be_relayed() {
        assert!(!render_targets(&fixture(), None).contains("relayed"));

        let userspace = parse_status(&FIXTURE.replace("\"TUN\": true", "\"TUN\": false"))
            .unwrap()
            .1;
        let rendered = render_targets(&userspace, None);
        assert!(
            rendered.contains("relayed"),
            "a throughput cap should be stated before someone blames the GPU: {rendered}"
        );
    }

    #[test]
    fn an_empty_tag_filter_explains_itself_rather_than_printing_nothing() {
        let rendered = render_targets(&fixture(), Some("nonexistent"));
        assert!(rendered.contains("No tailnet machines are tagged `nonexistent`."));
    }
}
