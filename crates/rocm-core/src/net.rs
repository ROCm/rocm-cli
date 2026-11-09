// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Networking primitives.
//!
//! Default local host/port, URL/endpoint formatting, and the loopback
//! HTTP/TCP helpers the managed-service readiness probes in
//! `endpoint_readiness.rs` are built on. Streaming downloads live in
//! `download.rs`.

use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::net::{IpAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

pub const DEFAULT_LOCAL_PORT: u16 = 11_435;
pub const DEFAULT_LOCAL_HOST: &str = "127.0.0.1";

pub fn format_host_for_url(host: &str) -> String {
    let trimmed = host.trim();
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return trimmed.to_owned();
    }
    match trimmed.parse::<IpAddr>() {
        Ok(IpAddr::V6(_)) => format!("[{trimmed}]"),
        _ => trimmed.to_owned(),
    }
}

pub fn format_host_port(host: &str, port: u16) -> String {
    format!("{}:{port}", format_host_for_url(host))
}

pub fn format_http_base_url(host: &str, port: u16) -> String {
    format!("http://{}", format_host_port(host, port))
}

pub fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
    let without_scheme = endpoint_url.trim().strip_prefix("http://")?;
    let authority = without_scheme.split('/').next()?.trim();
    if authority.is_empty() {
        return None;
    }
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']')?;
        let host = rest[..end].to_owned();
        let port = rest[end + 1..].strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    Some((host.to_owned(), port.parse().ok()?))
}

pub fn http_get_text(endpoint_url: &str, path: &str, timeout: Duration) -> Result<String> {
    http_get_text_with_auth(endpoint_url, path, None, timeout)
}

/// As [`http_get_text`], but authenticated.
///
/// Sends `Authorization: Bearer <key>` when `endpoint_api_key` is `Some`. Used to
/// probe endpoints that `rocm serve` has protected with an API key; `None`
/// preserves the unauthenticated behavior for loopback endpoints.
pub fn http_get_text_with_auth(
    endpoint_url: &str,
    path: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    let deadline = Instant::now() + timeout;
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    let mut stream = connect_tcp_stream(&host, port, timeout)?;
    let host_header = format_host_port(&host, port);
    let auth_header = match endpoint_api_key {
        Some(key) => format!("Authorization: Bearer {key}\r\n"),
        None => String::new(),
    };
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host_header}\r\nAccept: application/json\r\n{auth_header}Connection: close\r\n\r\n"
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .with_context(|| format!("failed to write HTTP GET {path}"))?;
    let response = read_http_response_bounded(&mut stream, deadline)
        .with_context(|| format!("failed to read HTTP GET {path}"))?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .context("HTTP response was missing a body")?;
    let status_line = headers.lines().next().unwrap_or_default();
    if !status_line.contains(" 200 ") {
        bail!("HTTP endpoint returned {status_line}");
    }
    Ok(body.to_owned())
}

/// POST a JSON body and return the response status line plus body.
///
/// The POST sibling of [`http_get_text_with_auth`]. Unlike the GET helper this
/// does not treat a non-200 as an error: callers that probe an endpoint need to
/// tell "the server answered, with a refusal" apart from "the server never
/// answered", and only the former proves the request path is alive.
pub fn http_post_json_with_auth(
    endpoint_url: &str,
    path: &str,
    body: &serde_json::Value,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<HttpResponseParts> {
    let deadline = Instant::now() + timeout;
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    let mut stream = connect_tcp_stream(&host, port, timeout)?;
    let host_header = format_host_port(&host, port);
    let auth_header = match endpoint_api_key {
        Some(key) => format!("Authorization: Bearer {key}\r\n"),
        None => String::new(),
    };
    let payload = serde_json::to_string(body).context("failed to serialize HTTP JSON body")?;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\n{auth_header}Connection: close\r\n\r\n{payload}",
        payload.len()
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .with_context(|| format!("failed to write HTTP POST {path}"))?;
    let response = read_http_response_bounded(&mut stream, deadline)
        .with_context(|| format!("failed to read HTTP POST {path}"))?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .context("HTTP response was missing a body")?;
    let status_line = headers.lines().next().unwrap_or_default();
    let status = http_status_code(status_line)
        .with_context(|| format!("unparsable HTTP status line `{status_line}`"))?;
    Ok(HttpResponseParts {
        status,
        body: body.to_owned(),
    })
}

/// The parts of an HTTP response a probe needs: the status code and the body.
#[derive(Debug, Clone)]
pub struct HttpResponseParts {
    pub status: u16,
    pub body: String,
}

fn http_status_code(status_line: &str) -> Option<u16> {
    status_line.split_whitespace().nth(1)?.parse().ok()
}

pub fn connect_tcp_stream(host: &str, port: u16, timeout: Duration) -> Result<TcpStream> {
    let addrs: Vec<_> = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve {host}:{port}"))?
        .collect();
    if addrs.is_empty() {
        bail!("no socket addresses resolved for {host}:{port}");
    }
    // Bound the connect as well as the reads: a probe against an engine that is
    // pinned solid must fail within the caller's timeout, not sit in the OS
    // default SYN retry window. Try every resolved address so `localhost`
    // resolving to `::1` first still reaches an IPv4-only listener.
    let mut last_err = None;
    let mut connected = None;
    for addr in &addrs {
        match TcpStream::connect_timeout(addr, timeout) {
            Ok(s) => {
                connected = Some(s);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let Some(stream) = connected else {
        let e = last_err.expect("addrs non-empty");
        return Err(anyhow::Error::new(e).context(format!("failed to connect to {host}:{port}")));
    };
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    Ok(stream)
}

pub fn write_all_tcp_stream(stream: &mut TcpStream, bytes: &[u8]) -> Result<()> {
    stream
        .write_all(bytes)
        .context("failed to write to TCP stream")
}

/// Read one HTTP response, bounded by a wall-clock deadline.
///
/// Two problems with reading to end-of-stream instead. A response is only
/// complete at EOF if the peer actually closes: `Connection: close` asks for
/// that, but nothing obliges a server or an intervening proxy to honor it, so a
/// service that writes a perfectly good response and holds the socket open would
/// stall until the read timeout and have its answer thrown away. And a socket
/// read timeout bounds each `read` call, not the sequence of them, so a
/// slow-drip responder could stretch the total wait to an arbitrary multiple of
/// what the caller asked for. This returns as soon as the response is complete by
/// its own framing, and never runs past `deadline` in total.
pub fn read_http_response_bounded(stream: &mut TcpStream, deadline: Instant) -> Result<String> {
    let mut response = Vec::new();
    let mut chunk = [0_u8; 4096];
    while !http_response_is_complete(&response) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("timed out reading HTTP response");
        }
        stream.set_read_timeout(Some(remaining)).ok();
        match stream.read(&mut chunk) {
            // Peer closed: whatever arrived is the whole response.
            Ok(0) => break,
            Ok(read) => response.extend_from_slice(&chunk[..read]),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                bail!("timed out reading HTTP response");
            }
            // A signal delivered to this thread aborts the read with `EINTR`.
            // `SA_RESTART` does not save us: Linux never restarts a socket read
            // that has a receive timeout set, and this loop sets one on every
            // pass (see signal(7), "Interruption of system calls"). Any handler
            // in the process is enough — `crossterm`'s `SIGWINCH` hook is linked
            // into the CLI. Nothing is wrong with the connection, so read again;
            // `deadline` still bounds the total wait.
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error).context("failed to read TCP stream"),
        }
    }
    Ok(String::from_utf8_lossy(&response).into_owned())
}

/// Whether the bytes so far are a complete HTTP response by their own framing.
///
/// `false` for a response that declares neither a length nor chunked encoding —
/// those are delimited by the connection closing, so the caller must keep reading
/// until EOF.
fn http_response_is_complete(response: &[u8]) -> bool {
    // Headers are ASCII by the HTTP spec, so it is safe to lossy-decode just
    // that slice to parse them. The body length check below stays on raw
    // bytes: lossy-decoding a body that ends mid multi-byte UTF-8 sequence
    // replaces the truncated tail with a 3-byte U+FFFD, which can inflate a
    // partial body's *decoded* length past the declared Content-Length and
    // report completeness one read early.
    let Some(header_end) = response.windows(4).position(|w| w == b"\r\n\r\n") else {
        return false;
    };
    let headers = String::from_utf8_lossy(&response[..header_end]);
    let body = &response[header_end + 4..];
    let header_value = |name: &str| {
        headers.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().to_owned())
        })
    };
    if let Some(length) =
        header_value("Content-Length").and_then(|value| value.parse::<usize>().ok())
    {
        return body.len() >= length;
    }
    if header_value("Transfer-Encoding")
        .is_some_and(|value| value.to_ascii_lowercase().contains("chunked"))
    {
        return body.ends_with(b"0\r\n\r\n");
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_tcp_stream_reaches_ipv4_listener_via_localhost() {
        // `localhost` may resolve to ::1 first; every address must be tried.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        connect_tcp_stream("localhost", port, Duration::from_secs(2)).unwrap();
    }

    #[test]
    fn http_response_is_complete_does_not_miscount_a_split_multibyte_char() {
        // A body ending in a multi-byte UTF-8 character can arrive one byte
        // short of the declared Content-Length. Lossy-decoding the whole
        // buffer to check completeness turns that dangling partial sequence
        // into a 3-byte U+FFFD replacement, inflating the decoded length past
        // the declared one and reporting completeness a read early.
        let body = "hi \u{2603}"; // snowman is a 3-byte UTF-8 character
        let full = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes();
        let truncated = &full[..full.len() - 1];

        assert!(!http_response_is_complete(truncated));
        assert!(http_response_is_complete(&full));
    }
    #[test]
    fn http_host_formatting_brackets_ipv6_literals() {
        assert_eq!(format_host_port("127.0.0.1", 11435), "127.0.0.1:11435");
        assert_eq!(
            format_http_base_url("localhost", 11435),
            "http://localhost:11435"
        );
        assert_eq!(format_host_port("::1", 11435), "[::1]:11435");
        assert_eq!(format_http_base_url("::1", 11435), "http://[::1]:11435");
        assert_eq!(format_host_port("[::1]", 11435), "[::1]:11435");
    }
}
