// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Central serve-port policy for the user-facing `rocm serve` verb.
//!
//! `--port` is an explicit-or-automatic *request*, not a concrete default. This
//! module turns that request into one concrete, leased port before any manifest,
//! log, endpoint key, or child process exists:
//!
//! - **Automatic** (`--port` omitted) is accepted only on the canonical default
//!   loopback host `127.0.0.1`, and scans [`AUTO_PORT_FIRST`]..=[`AUTO_PORT_LAST`]
//!   for a candidate that is neither reserved by a live managed service nor
//!   occupied on the OS.
//! - **Explicit on canonical loopback** goes through the same reservation and
//!   OS preflight, so a collision fails immediately with a clear message instead
//!   of surfacing as an engine bind error minutes into startup.
//! - **Explicit on a custom host** (`localhost`, `::1`, `0.0.0.0`, any other
//!   IPv4/IPv6 address) is passed through untouched: bind success or failure
//!   stays engine-owned, because an IPv4-loopback probe proves nothing about
//!   those addresses. Automatic is refused there rather than guessing.
//!
//! Everything downstream of this module — engine requests, service records,
//! readiness, supervision, recovery — keeps its concrete `u16` contract.

use anyhow::{Result, bail};
use rocm_core::{DEFAULT_LOCAL_HOST, DEFAULT_LOCAL_PORT, LoopbackPortLease};
use std::io;

/// First automatic candidate. Also the legacy well-known local endpoint, so an
/// otherwise-idle machine keeps serving on the port users already know.
pub(crate) const AUTO_PORT_FIRST: u16 = DEFAULT_LOCAL_PORT;
/// Number of candidates *after* the first one. The scanned range is inclusive on
/// both ends: `11435..=11535`, 101 candidates.
pub(crate) const AUTO_PORT_SPAN: u16 = 100;
/// Last automatic candidate.
pub(crate) const AUTO_PORT_LAST: u16 = AUTO_PORT_FIRST + AUTO_PORT_SPAN;

/// The one phrase used wherever a port must be described *before* it is
/// resolved. The CLI plan output and the Dash approval card share it verbatim so
/// neither promises a concrete port it has not yet acquired.
pub(crate) const AUTOMATIC_PORT_DISCLOSURE: &str = "Port: automatic; endpoint shown after launch";

/// What the user asked for on the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PortRequest {
    /// `--port` omitted: choose an available local port.
    Auto,
    /// `--port N`: use exactly this port.
    Explicit(u16),
}

impl PortRequest {
    #[must_use]
    pub(crate) const fn from_flag(port: Option<u16>) -> Self {
        match port {
            Some(port) => Self::Explicit(port),
            None => Self::Auto,
        }
    }

    #[must_use]
    pub(crate) const fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }

    /// The concrete port, when the user named one.
    #[must_use]
    pub(crate) const fn explicit(self) -> Option<u16> {
        match self {
            Self::Auto => None,
            Self::Explicit(port) => Some(port),
        }
    }
}

/// A port a live managed service still owns. Carried with its service id so a
/// collision names the thing actually holding the port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReservedPort {
    pub(crate) port: u16,
    pub(crate) service_id: String,
    pub(crate) status: String,
}

/// clap value parser for every user-facing `--port`.
///
/// Rejects `0` explicitly: the OS reads it as "any ephemeral port", which would
/// silently produce an endpoint nobody asked for.
///
/// # Errors
/// Returns a user-facing message for anything outside 1–65535.
pub(crate) fn parse_serve_port(value: &str) -> Result<u16, String> {
    match value.trim().parse::<u16>() {
        Ok(0) | Err(_) => Err(format!(
            "port must be a number between 1 and 65535, got '{value}'"
        )),
        Ok(port) => Ok(port),
    }
}

/// Reject an automatic request on a host this policy cannot preflight, before
/// any engine resolution or side effect happens.
///
/// # Errors
/// Fails when `--port` was omitted for anything but the canonical loopback host.
pub(crate) fn validate_port_request(host: &str, request: PortRequest) -> Result<()> {
    if request.is_auto() && !rocm_core::is_canonical_loopback_host(host) {
        bail!(
            "`rocm serve --host {host}` needs an explicit `--port <1-65535>`. Automatic port \
             selection is supported only on the default host {DEFAULT_LOCAL_HOST}, where ROCm \
             can prove a port is free before launching; on any other address the engine owns \
             the bind."
        );
    }
    Ok(())
}

/// Production bind probe: take a real `127.0.0.1:<port>` lease.
///
/// # Errors
/// Propagates the OS bind error verbatim so the caller can distinguish
/// "occupied" from "not permitted"/"not available".
pub(crate) fn loopback_bind_probe(port: u16) -> io::Result<LoopbackPortLease> {
    rocm_core::lease_loopback_port(port)
}

/// Turn a [`PortRequest`] into one concrete, leased port.
///
/// Callers must hold the shared managed-service allocation lock across this call
/// *and* the record publication that follows it, so a concurrent launch cannot
/// pick the same candidate between the probe and the spawn.
///
/// `probe` is injected so tests can drive deterministic OS outcomes; production
/// passes [`loopback_bind_probe`].
///
/// # Errors
/// Fails when an explicit port is reserved or occupied, when the automatic range
/// is exhausted, or when the OS reports anything other than "address in use" —
/// a permission or address-availability error must never be papered over by
/// quietly choosing a different port.
pub(crate) fn resolve_serve_port(
    host: &str,
    request: PortRequest,
    reserved: &[ReservedPort],
    probe: &mut dyn FnMut(u16) -> io::Result<LoopbackPortLease>,
) -> Result<LoopbackPortLease> {
    if !rocm_core::is_canonical_loopback_host(host) {
        // Custom host: an explicit port passes through untouched so bind success
        // or failure stays engine-owned. Automatic is refused here as well as in
        // `serve()`, because this is the shared choke point every caller reaches
        // — and it is refused by returning an error, never by panicking.
        let Some(port) = request.explicit() else {
            validate_port_request(host, request)?;
            unreachable!("custom-host automatic requests are rejected above");
        };
        return Ok(LoopbackPortLease::engine_owned(port));
    }

    match request {
        PortRequest::Explicit(port) => {
            if let Some(owner) = reserved.iter().find(|entry| entry.port == port) {
                bail!(
                    "{DEFAULT_LOCAL_HOST}:{port} is already reserved by managed service {} \
                     ({}). Stop it with `rocm services stop {} --yes`, or pass a different \
                     `--port`.",
                    owner.service_id,
                    owner.status,
                    owner.service_id
                );
            }
            probe(port).map_err(|error| explicit_bind_error(port, &error))
        }
        PortRequest::Auto => {
            for port in AUTO_PORT_FIRST..=AUTO_PORT_LAST {
                if reserved.iter().any(|entry| entry.port == port) {
                    continue;
                }
                match probe(port) {
                    Ok(lease) => return Ok(lease),
                    // Occupied is the only outcome that advances the scan.
                    Err(error) if error.kind() == io::ErrorKind::AddrInUse => {}
                    Err(error) => bail!(
                        "failed to reserve {DEFAULT_LOCAL_HOST}:{port} while choosing an \
                         automatic port: {error}"
                    ),
                }
            }
            bail!(
                "no free TCP port for {DEFAULT_LOCAL_HOST} in {AUTO_PORT_FIRST}–{AUTO_PORT_LAST}; \
                 every candidate is reserved by a live managed service or already in use. Stop an \
                 unused server with `rocm services stop <service-id> --yes`, or pass an explicit \
                 `--port`."
            );
        }
    }
}

fn explicit_bind_error(port: u16, error: &io::Error) -> anyhow::Error {
    if error.kind() == io::ErrorKind::AddrInUse {
        return anyhow::anyhow!(
            "{DEFAULT_LOCAL_HOST}:{port} is already in use by another process. Choose a free \
             `--port`, stop the process holding it, or omit `--port` to let ROCm pick an \
             available port starting at {AUTO_PORT_FIRST}."
        );
    }
    anyhow::anyhow!("failed to reserve {DEFAULT_LOCAL_HOST}:{port}: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn reserved(port: u16, service_id: &str, status: &str) -> ReservedPort {
        ReservedPort {
            port,
            service_id: service_id.to_owned(),
            status: status.to_owned(),
        }
    }

    /// A probe driven by a table of scripted per-port failures. Any port absent
    /// from the table binds successfully.
    fn scripted(
        outcomes: Vec<(u16, io::ErrorKind)>,
    ) -> impl FnMut(u16) -> io::Result<LoopbackPortLease> {
        move |port| match outcomes.iter().find(|(candidate, _)| *candidate == port) {
            Some((_, kind)) => Err(io::Error::new(*kind, "scripted")),
            None => Ok(LoopbackPortLease::engine_owned(port)),
        }
    }

    #[test]
    fn port_request_distinguishes_omitted_from_explicit() {
        assert_eq!(PortRequest::from_flag(None), PortRequest::Auto);
        assert!(PortRequest::from_flag(None).is_auto());
        assert_eq!(PortRequest::from_flag(None).explicit(), None);
        assert_eq!(
            PortRequest::from_flag(Some(8000)),
            PortRequest::Explicit(8000)
        );
        assert!(!PortRequest::from_flag(Some(8000)).is_auto());
        assert_eq!(PortRequest::from_flag(Some(8000)).explicit(), Some(8000));
    }

    #[test]
    fn port_parser_rejects_zero_and_out_of_range_values() {
        assert_eq!(parse_serve_port("8000"), Ok(8000));
        assert_eq!(parse_serve_port(" 11435 "), Ok(11_435));
        assert_eq!(parse_serve_port("65535"), Ok(65_535));
        for bad in ["0", "65536", "-1", "eleven", ""] {
            let error = parse_serve_port(bad).expect_err("must be rejected");
            assert!(
                error.contains("between 1 and 65535"),
                "unexpected message for {bad:?}: {error}"
            );
        }
    }

    #[test]
    fn automatic_range_is_exactly_the_documented_window() {
        assert_eq!(AUTO_PORT_FIRST, 11_435);
        assert_eq!(AUTO_PORT_LAST, 11_535);
        assert_eq!((AUTO_PORT_FIRST..=AUTO_PORT_LAST).count(), 101);
    }

    #[test]
    fn automatic_is_refused_on_every_custom_host() {
        validate_port_request(DEFAULT_LOCAL_HOST, PortRequest::Auto).expect("canonical host is ok");
        for host in ["localhost", "::1", "0.0.0.0", "127.0.0.2", "10.0.0.5"] {
            let error = validate_port_request(host, PortRequest::Auto)
                .expect_err("automatic must be refused on a custom host");
            let message = format!("{error:#}");
            assert!(message.contains("explicit `--port"), "{message}");
            assert!(message.contains(host), "{message}");
            // An explicit port is always accepted there.
            validate_port_request(host, PortRequest::Explicit(8000))
                .expect("explicit port is valid on a custom host");
        }
    }

    #[test]
    fn custom_host_explicit_port_bypasses_the_loopback_preflight() {
        let mut probe = |_port: u16| -> io::Result<LoopbackPortLease> {
            panic!("custom hosts must not be preflighted against IPv4 loopback")
        };
        // Even a port a live loopback service reserves is passed straight
        // through: the engine owns bind on that address.
        let lease = resolve_serve_port(
            "0.0.0.0",
            PortRequest::Explicit(11_435),
            &[reserved(11_435, "svc-a", "running")],
            &mut probe,
        )
        .expect("custom host explicit port is engine-owned");
        assert_eq!(lease.port(), 11_435);
        assert!(!lease.is_held());
    }

    #[test]
    fn resolving_auto_on_a_custom_host_errors_instead_of_panicking() {
        // The choke point must refuse Auto on its own, even when a caller skipped
        // the earlier `validate_port_request` gate — as a Result, never a panic.
        let mut probe = |_port: u16| -> io::Result<LoopbackPortLease> {
            panic!("a custom host must never be preflighted")
        };
        for host in ["localhost", "::1", "0.0.0.0", "10.0.0.5"] {
            let error = resolve_serve_port(host, PortRequest::Auto, &[], &mut probe)
                .expect_err("automatic selection is not supported on a custom host");
            let message = format!("{error:#}");
            assert!(message.contains("explicit `--port"), "{message}");
            assert!(message.contains(host), "{message}");
        }
    }

    #[test]
    fn explicit_loopback_port_outside_the_auto_range_is_retained() {
        let mut probe = scripted(vec![]);
        let lease = resolve_serve_port(
            DEFAULT_LOCAL_HOST,
            PortRequest::Explicit(8000),
            &[],
            &mut probe,
        )
        .expect("a free explicit port is kept");
        assert_eq!(lease.port(), 8000);
    }

    #[test]
    fn explicit_loopback_port_that_is_occupied_is_rejected() {
        let mut probe = scripted(vec![(8000, io::ErrorKind::AddrInUse)]);
        let error = resolve_serve_port(
            DEFAULT_LOCAL_HOST,
            PortRequest::Explicit(8000),
            &[],
            &mut probe,
        )
        .expect_err("an occupied explicit port must fail");
        let message = format!("{error:#}");
        assert!(message.contains("127.0.0.1:8000"), "{message}");
        assert!(message.contains("already in use"), "{message}");
    }

    #[test]
    fn explicit_loopback_port_reserved_by_a_live_service_names_that_service() {
        let mut probe = |_port: u16| -> io::Result<LoopbackPortLease> {
            panic!("a reserved port must be rejected before probing")
        };
        let error = resolve_serve_port(
            DEFAULT_LOCAL_HOST,
            PortRequest::Explicit(11_435),
            &[reserved(11_435, "vllm-qwen-1", "starting")],
            &mut probe,
        )
        .expect_err("a reserved explicit port must fail");
        let message = format!("{error:#}");
        assert!(message.contains("vllm-qwen-1"), "{message}");
        assert!(message.contains("starting"), "{message}");
    }

    #[test]
    fn automatic_takes_the_first_candidate_when_it_is_free() {
        let mut probe = scripted(vec![]);
        let lease = resolve_serve_port(DEFAULT_LOCAL_HOST, PortRequest::Auto, &[], &mut probe)
            .expect("first candidate is free");
        assert_eq!(lease.port(), AUTO_PORT_FIRST);
    }

    #[test]
    fn automatic_skips_occupied_candidates_until_one_binds() {
        let mut probe = scripted(vec![
            (11_435, io::ErrorKind::AddrInUse),
            (11_436, io::ErrorKind::AddrInUse),
            (11_437, io::ErrorKind::AddrInUse),
        ]);
        let lease = resolve_serve_port(DEFAULT_LOCAL_HOST, PortRequest::Auto, &[], &mut probe)
            .expect("scan advances past occupied candidates");
        assert_eq!(lease.port(), 11_438);
    }

    #[test]
    fn automatic_skips_ports_reserved_by_live_records_without_probing_them() {
        let mut probed: Vec<u16> = Vec::new();
        let lease = {
            let mut probe = |port: u16| -> io::Result<LoopbackPortLease> {
                probed.push(port);
                Ok(LoopbackPortLease::engine_owned(port))
            };
            resolve_serve_port(
                DEFAULT_LOCAL_HOST,
                PortRequest::Auto,
                &[
                    reserved(11_435, "svc-ready", "ready"),
                    reserved(11_436, "svc-recovering", "recovering"),
                ],
                &mut probe,
            )
            .expect("scan skips reserved candidates")
        };
        assert_eq!(lease.port(), 11_437);
        assert_eq!(probed, vec![11_437]);
    }

    #[test]
    fn automatic_stops_immediately_on_a_non_collision_bind_error() {
        for kind in [
            io::ErrorKind::PermissionDenied,
            io::ErrorKind::AddrNotAvailable,
        ] {
            let mut probe = scripted(vec![(11_435, kind)]);
            let error = resolve_serve_port(DEFAULT_LOCAL_HOST, PortRequest::Auto, &[], &mut probe)
                .expect_err("only AddrInUse may advance the scan");
            let message = format!("{error:#}");
            assert!(message.contains("127.0.0.1:11435"), "{message}");
        }
    }

    #[test]
    fn explicit_non_collision_bind_error_reports_host_port_and_os_detail() {
        let mut probe = |_port: u16| -> io::Result<LoopbackPortLease> {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "permission denied",
            ))
        };
        let error = resolve_serve_port(
            DEFAULT_LOCAL_HOST,
            PortRequest::Explicit(80),
            &[],
            &mut probe,
        )
        .expect_err("a privileged port must fail");
        let message = format!("{error:#}");
        assert!(message.contains("127.0.0.1:80"), "{message}");
        assert!(message.contains("permission denied"), "{message}");
    }

    #[test]
    fn automatic_exhaustion_reports_the_exact_scanned_range() {
        let mut probe = |_port: u16| -> io::Result<LoopbackPortLease> {
            Err(io::Error::new(io::ErrorKind::AddrInUse, "occupied"))
        };
        let error = resolve_serve_port(DEFAULT_LOCAL_HOST, PortRequest::Auto, &[], &mut probe)
            .expect_err("an exhausted range must fail");
        let message = format!("{error:#}");
        assert!(message.contains("11435–11535"), "{message}");
    }

    #[test]
    fn automatic_selects_against_real_loopback_listeners() {
        // Real sockets, not a script: hold the first two candidates with live
        // listeners and prove the production probe walks past them. If either is
        // already owned by something outside this test the scenario cannot be set
        // up deterministically, so skip rather than assert on a moving target.
        let (Ok(first), Ok(second)) = (
            TcpListener::bind(("127.0.0.1", AUTO_PORT_FIRST)),
            TcpListener::bind(("127.0.0.1", AUTO_PORT_FIRST + 1)),
        ) else {
            return;
        };
        let mut probe = loopback_bind_probe;
        let lease = resolve_serve_port(DEFAULT_LOCAL_HOST, PortRequest::Auto, &[], &mut probe)
            .expect("a candidate above the held ones is free");
        assert!(
            lease.port() >= AUTO_PORT_FIRST + 2,
            "selected {} while {AUTO_PORT_FIRST} and {} were held",
            lease.port(),
            AUTO_PORT_FIRST + 1
        );
        assert!(lease.is_held(), "the production probe must hold the socket");
        // Guards outlive every assertion above.
        drop((first, second));
    }
}
