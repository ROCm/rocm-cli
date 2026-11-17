// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Loopback credential proxy for one agent instance (PROTOTYPE).
//!
//! The agent is pointed at `http://127.0.0.1:<port>` through its base-URL
//! variable and holds only a random placeholder key. A request is forwarded
//! only if it presents exactly that placeholder; the proxy then replaces it
//! with the real key and sends the request over TLS to the single upstream the
//! agent kind is bound to. Any other key, or none, is refused and audited, so
//! another local process (or another agent) cannot spend this agent's key.
//!
//! Prototype limits: HTTP/1.1 with `Content-Length` bodies only, one request
//! per connection (`Connection: close`), no token accounting yet.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use super::registry::{self, AgentRecord};

const MAX_HEAD_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

pub(crate) struct ProxyConfig {
    pub upstream: String,
    /// Credential headers added on the host side (e.g. a gateway
    /// subscription key); any copy the agent sends is dropped.
    pub inject_headers: Vec<(String, String)>,
    pub placeholder: String,
    /// The credential to send upstream in the header the placeholder arrived
    /// in. `None` for an upstream that needs none (a local server without an
    /// endpoint key): the agent's placeholder is still required, and dropped.
    pub real_key: Option<String>,
    pub audit_path: PathBuf,
    pub record: Arc<Mutex<AgentRecord>>,
    pub record_path: PathBuf,
}

pub(crate) fn start(config: ProxyConfig) -> Result<SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0").context("bind credential proxy")?;
    let addr = listener.local_addr()?;
    let config = Arc::new(config);
    let agent = ureq::AgentBuilder::new().redirects(0).build();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let config = config.clone();
            let agent = agent.clone();
            std::thread::spawn(move || {
                let _ = handle(stream, &config, &agent);
            });
        }
    });
    Ok(addr)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Auth {
    /// The placeholder arrived in this header (`x-api-key` or `authorization`).
    Placeholder(&'static str),
    Missing,
    Mismatch,
}

pub(crate) fn authorize(headers: &[(String, String)], placeholder: &str) -> Auth {
    let mut seen = false;
    for (name, value) in headers {
        let value = value.trim();
        if name.eq_ignore_ascii_case("x-api-key") {
            seen = true;
            if value == placeholder {
                return Auth::Placeholder("x-api-key");
            }
        } else if name.eq_ignore_ascii_case("authorization") {
            seen = true;
            if value.strip_prefix("Bearer ").map(str::trim) == Some(placeholder) {
                return Auth::Placeholder("authorization");
            }
        }
    }
    if seen { Auth::Mismatch } else { Auth::Missing }
}

/// Headers to send upstream: hop-by-hop, framing and credential headers are
/// dropped and the real key is set in the header the placeholder came in.
pub(crate) fn upstream_headers(
    headers: &[(String, String)],
    key_header: &'static str,
    real_key: Option<&str>,
    inject: &[(String, String)],
) -> Vec<(String, String)> {
    const DROP: &[&str] = &[
        "host",
        "connection",
        "keep-alive",
        "proxy-connection",
        "transfer-encoding",
        "content-length",
        "accept-encoding",
        "x-api-key",
        "authorization",
    ];
    let mut out: Vec<(String, String)> = headers
        .iter()
        .filter(|(n, _)| !DROP.iter().any(|d| n.eq_ignore_ascii_case(d)))
        .filter(|(n, _)| !inject.iter().any(|(i, _)| n.eq_ignore_ascii_case(i)))
        .cloned()
        .collect();
    if let Some(real_key) = real_key {
        let value = if key_header == "authorization" {
            format!("Bearer {real_key}")
        } else {
            real_key.to_owned()
        };
        out.push((key_header.to_owned(), value));
    }
    out.extend(inject.iter().cloned());
    out
}

/// Parses `Name: Value` lines (Claude Code's `ANTHROPIC_CUSTOM_HEADERS` format).
pub(crate) fn parse_custom_headers(text: &str) -> Result<Vec<(String, String)>> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (n, v) = l
                .split_once(':')
                .context("custom header line without ':'")?;
            Ok((n.trim().to_owned(), v.trim().to_owned()))
        })
        .collect()
}

struct Request {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> Result<Request> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut head_bytes = 0usize;
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().context("empty request")?.to_owned();
    let target = parts.next().context("no request target")?.to_owned();
    let mut headers = Vec::new();
    loop {
        line.clear();
        head_bytes += reader.read_line(&mut line)?;
        anyhow::ensure!(head_bytes <= MAX_HEAD_BYTES, "request head too large");
        let l = line.trim_end_matches(['\r', '\n']);
        if l.is_empty() {
            break;
        }
        if let Some((n, v)) = l.split_once(':') {
            headers.push((n.trim().to_owned(), v.trim().to_owned()));
        }
    }
    anyhow::ensure!(
        !headers
            .iter()
            .any(|(n, _)| n.eq_ignore_ascii_case("transfer-encoding")),
        "chunked request bodies are not supported by the prototype"
    );
    let len: usize = headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case("content-length"))
        .map(|(_, v)| v.parse())
        .transpose()?
        .unwrap_or(0);
    anyhow::ensure!(len <= MAX_BODY_BYTES, "request body too large");
    let mut body = vec![0; len];
    reader.read_exact(&mut body)?;
    Ok(Request {
        method,
        target,
        headers,
        body,
    })
}

fn respond_plain(stream: &mut TcpStream, status: u16, reason: &str, body: &str) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )?;
    Ok(())
}

fn handle(mut stream: TcpStream, config: &ProxyConfig, agent: &ureq::Agent) -> Result<()> {
    let req = match read_request(&stream) {
        Ok(r) => r,
        Err(e) => return respond_plain(&mut stream, 400, "Bad Request", &format!("{e}\n")),
    };
    // Only origin-form targets: the agent cannot pick another upstream.
    if !req.target.starts_with('/') {
        audit(config, &req, 400, "absolute-form target refused");
        return respond_plain(&mut stream, 400, "Bad Request", "origin-form only\n");
    }
    let key_header = match authorize(&req.headers, &config.placeholder) {
        Auth::Placeholder(h) => h,
        other => {
            audit(config, &req, 403, &format!("{other:?}"));
            bump(config, false);
            return respond_plain(&mut stream, 403, "Forbidden", "unknown credential\n");
        }
    };
    let url = format!("{}{}", config.upstream, req.target);
    let mut up = agent.request(&req.method, &url);
    for (n, v) in upstream_headers(
        &req.headers,
        key_header,
        config.real_key.as_deref(),
        &config.inject_headers,
    ) {
        up = up.set(&n, &v);
    }
    let resp = match up.send_bytes(&req.body) {
        Ok(r) | Err(ureq::Error::Status(_, r)) => r,
        Err(e) => {
            audit(config, &req, 502, &e.to_string());
            return respond_plain(&mut stream, 502, "Bad Gateway", "upstream unreachable\n");
        }
    };
    let status = resp.status();
    audit(config, &req, status, "forwarded");
    bump(config, true);
    write!(stream, "HTTP/1.1 {status} {}\r\n", resp.status_text())?;
    for name in resp.headers_names() {
        if ["transfer-encoding", "content-length", "connection"]
            .iter()
            .any(|d| name.eq_ignore_ascii_case(d))
        {
            continue;
        }
        for value in resp.all(&name) {
            write!(stream, "{name}: {value}\r\n")?;
        }
    }
    write!(stream, "connection: close\r\n\r\n")?;
    // Stream (SSE) through as it arrives; close delimits the body.
    std::io::copy(&mut resp.into_reader(), &mut stream)?;
    Ok(())
}

fn bump(config: &ProxyConfig, forwarded: bool) {
    let mut rec = config.record.lock().unwrap();
    if forwarded {
        rec.requests += 1;
    } else {
        rec.rejected += 1;
    }
    let _ = registry::save(&config.record_path, &rec);
}

fn audit(config: &ProxyConfig, req: &Request, status: u16, outcome: &str) {
    let line = serde_json::json!({
        "ts": registry::now_unix(),
        "method": req.method,
        "path": req.target,
        "status": status,
        "outcome": outcome,
    });
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.audit_path)
    {
        let _ = writeln!(f, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(n, v)| ((*n).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn only_the_exact_placeholder_is_authorized() {
        assert_eq!(
            authorize(&h(&[("X-Api-Key", "ph")]), "ph"),
            Auth::Placeholder("x-api-key")
        );
        assert_eq!(
            authorize(&h(&[("Authorization", "Bearer ph")]), "ph"),
            Auth::Placeholder("authorization")
        );
        assert_eq!(
            authorize(&h(&[("x-api-key", "sk-real")]), "ph"),
            Auth::Mismatch
        );
        assert_eq!(authorize(&h(&[("x-api-key", "ph2")]), "ph"), Auth::Mismatch);
        assert_eq!(authorize(&h(&[("accept", "*/*")]), "ph"), Auth::Missing);
    }

    #[test]
    fn upstream_gets_real_key_and_no_placeholder() {
        let out = upstream_headers(
            &h(&[
                ("Host", "127.0.0.1:1"),
                ("x-api-key", "ph"),
                ("anthropic-version", "2023-06-01"),
                ("Content-Length", "2"),
            ]),
            "x-api-key",
            Some("sk-real"),
            &[],
        );
        assert_eq!(
            out,
            h(&[
                ("anthropic-version", "2023-06-01"),
                ("x-api-key", "sk-real")
            ])
        );
    }

    #[test]
    fn gateway_header_is_injected_and_agent_copy_dropped() {
        let inject = parse_custom_headers("Gateway-Key: real-sub\n").unwrap();
        let out = upstream_headers(
            &h(&[("x-api-key", "ph"), ("gateway-key", "agent-guess")]),
            "x-api-key",
            Some("sk-real"),
            &inject,
        );
        assert_eq!(
            out,
            h(&[("x-api-key", "sk-real"), ("Gateway-Key", "real-sub")])
        );
    }

    #[test]
    fn without_a_real_key_the_placeholder_is_dropped_and_no_credential_is_sent() {
        let out = upstream_headers(
            &h(&[("x-api-key", "ph"), ("anthropic-version", "2023-06-01")]),
            "x-api-key",
            None,
            &[],
        );
        assert_eq!(out, h(&[("anthropic-version", "2023-06-01")]));
    }
}
