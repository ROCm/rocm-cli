// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The local provider's hand-rolled HTTP/1.1 transport: a blocking JSON POST
//! ([`post_json_to_local_endpoint`]) and a streamed POST that feeds an SSE
//! emitter in whichever body framing the engine answers with
//! ([`stream_json_from_local_endpoint`]: chunked, `Content-Length`, or
//! close-delimited), plus the `http://` endpoint parser both rely on.
//!
//! The streamed path decodes into the OpenAI-format emitter, so this module
//! reads `OpenAiSseEmitter`; that is the only edge back into `providers`
//! besides its shared types.

use super::wire::OpenAiSseEmitter;
use super::{ProviderStreamEvent, SseBytesEmitter};
use anyhow::{Context, Result, bail};
use rocm_core::{
    connect_tcp_stream, format_host_port, read_http_response_bounded, write_all_tcp_stream,
};
use std::io::{BufRead, BufReader, Read};
use std::time::Duration;

pub(super) fn post_json_to_local_endpoint(
    endpoint_url: &str,
    path: &str,
    body: &serde_json::Value,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<serde_json::Value> {
    let body =
        post_json_to_local_endpoint_body(endpoint_url, path, body, endpoint_api_key, timeout)?;
    serde_json::from_str(body.trim()).context("failed to parse local provider chat JSON")
}

/// `Authorization: Bearer` request-header line for a protected local endpoint, or
/// an empty string for an unauthenticated (loopback) one.
fn local_bearer_header(endpoint_api_key: Option<&str>) -> String {
    match endpoint_api_key {
        Some(key) => format!("Authorization: Bearer {key}\r\n"),
        None => String::new(),
    }
}

fn post_json_to_local_endpoint_body(
    endpoint_url: &str,
    path: &str,
    body: &serde_json::Value,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported local endpoint URL `{endpoint_url}`"))?;
    let body = serde_json::to_string(body).context("failed to serialize chat request")?;
    let deadline = std::time::Instant::now() + timeout;
    let mut stream = connect_tcp_stream(&host, port, timeout)?;
    let host_header = format_host_port(&host, port);
    let auth_header = local_bearer_header(endpoint_api_key);
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{auth_header}Connection: close\r\n\r\n{}",
        body.len(),
        body
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .context("failed to write local provider chat request")?;

    let response = read_http_response_bounded(&mut stream, deadline)
        .context("failed to read local provider chat response")?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .context("local provider response was missing HTTP body")?;
    let status_line = headers.lines().next().unwrap_or_default();
    if !status_line.contains(" 200 ") {
        bail!("local provider returned {status_line}");
    }
    Ok(body.to_owned())
}

pub(super) fn stream_json_from_local_endpoint(
    endpoint_url: &str,
    path: &str,
    body: &serde_json::Value,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
    on_event: &mut dyn FnMut(ProviderStreamEvent) -> Result<()>,
) -> Result<()> {
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported local endpoint URL `{endpoint_url}`"))?;
    let body = serde_json::to_string(body).context("failed to serialize chat request")?;
    let mut stream = connect_tcp_stream(&host, port, timeout)?;
    let host_header = format_host_port(&host, port);
    let auth_header = local_bearer_header(endpoint_api_key);
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{auth_header}Connection: close\r\n\r\n{}",
        body.len(),
        body
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .context("failed to write local provider chat request")?;

    let mut reader = BufReader::new(stream);
    let mut headers = String::new();
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .context("failed to read local provider response headers")?;
        if read == 0 {
            bail!("local provider response ended before headers completed");
        }
        headers.push_str(&line);
        if line == "\r\n" || line == "\n" {
            break;
        }
    }
    let status_line = headers.lines().next().unwrap_or_default();
    if !status_line.contains(" 200 ") {
        bail!("local provider returned {status_line}");
    }

    let mut emitter = OpenAiSseEmitter::new(on_event);
    read_sse_body(&mut reader, &headers, &mut emitter)?;
    emitter.finish()
}

/// Read a streamed response body in whichever framing `headers` declare:
/// chunked, `Content-Length`, or delimited by the connection closing.
fn read_sse_body<R: BufRead>(
    reader: &mut R,
    headers: &str,
    emitter: &mut OpenAiSseEmitter<'_>,
) -> Result<()> {
    let chunked = header_value(headers, "Transfer-Encoding")
        .is_some_and(|value| value.to_ascii_lowercase().contains("chunked"));
    let content_length = header_value(headers, "Content-Length")
        .and_then(|value| value.trim().parse::<usize>().ok());
    if chunked {
        read_chunked_sse_body(reader, emitter)
    } else if let Some(length) = content_length {
        read_fixed_sse_body(reader, length, emitter).map_err(body_truncated_at_eof)
    } else {
        // For this framing the connection closing *is* the end of the body, so
        // end-of-stream is success here and the transport cannot tell a
        // finished answer from one cut off mid-way. That asymmetry with the
        // two framings above is deliberate: the only signal left is whether
        // `data: [DONE]` arrived, and `OpenAiSseEmitter::finish` does not
        // require it for any framing today. Making it required is a protocol
        // policy for every engine, not a transport fix.
        read_close_delimited_sse_body(reader, emitter)
    }
}

fn header_value(headers: &str, name: &str) -> Option<String> {
    headers.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name)
            .then(|| value.trim().to_owned())
    })
}

fn read_fixed_sse_body<R: Read>(
    reader: &mut R,
    mut remaining: usize,
    emitter: &mut OpenAiSseEmitter<'_>,
) -> Result<()> {
    let mut buffer = [0_u8; 1024];
    while remaining > 0 {
        let read_len = remaining.min(buffer.len());
        reader
            .read_exact(&mut buffer[..read_len])
            .context("failed to read local provider SSE body")?;
        emitter.push_bytes(&buffer[..read_len])?;
        remaining -= read_len;
    }
    Ok(())
}

fn read_close_delimited_sse_body<R: Read>(
    reader: &mut R,
    emitter: &mut OpenAiSseEmitter<'_>,
) -> Result<()> {
    let mut buffer = [0_u8; 1024];
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => return Ok(()),
            Ok(read) => read,
            // A signal delivered to this thread aborts the socket read with
            // `EINTR`, and Linux never restarts a read that has a receive
            // timeout set (see signal(7)), which this socket has. Nothing is
            // wrong with the connection, so read again. (`read_line` and
            // `read_exact`, used by the other body framings, retry internally;
            // a bare `read` does not.)
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error).context("failed to read local provider SSE body"),
        };
        emitter.push_bytes(&buffer[..read])?;
    }
}

pub(super) const LOCAL_STREAM_TRUNCATED: &str = "local provider closed the connection before the streamed response finished; \
     the response is incomplete";

/// Report an end-of-stream before a body's declared end — inside a chunk (its
/// size line, data or CRLF) or short of its `Content-Length` — the same way as
/// one between chunks: either way the engine went away mid-answer. Any other
/// error (a read timeout, a reset connection, a malformed line) passes through
/// unchanged, so it is not mislabelled as the engine closing the connection.
fn body_truncated_at_eof(error: anyhow::Error) -> anyhow::Error {
    let at_eof = error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::UnexpectedEof)
    });
    if at_eof {
        error.context(LOCAL_STREAM_TRUNCATED)
    } else {
        error
    }
}

fn read_chunked_sse_body<R: BufRead>(
    reader: &mut R,
    emitter: &mut OpenAiSseEmitter<'_>,
) -> Result<()> {
    loop {
        let mut size_line = String::new();
        let read = reader
            .read_line(&mut size_line)
            .context("failed to read local provider chunk size")?;
        // `Ok(0)` is end-of-stream, not a blank line: the engine closed the
        // connection (it crashed, was killed, or was stopped) before sending the
        // terminating zero-length chunk. Reading again would return `Ok(0)`
        // immediately, forever.
        if read == 0 {
            bail!("{LOCAL_STREAM_TRUNCATED}");
        }
        if size_line.trim().is_empty() {
            continue;
        }
        let size_text = size_line
            .trim()
            .split(';')
            .next()
            .unwrap_or_default()
            .trim();
        let size = usize::from_str_radix(size_text, 16)
            .with_context(|| format!("invalid local provider chunk size `{size_text}`"))?;
        if size == 0 {
            // The zero chunk's own CRLF (or a trailer line). Its read result is
            // deliberately ignored: every event, `[DONE]` included, has been
            // delivered by now, so an engine that closes or stalls here has
            // still sent a complete answer, and failing would discard it.
            let mut trailer = String::new();
            let _ = reader.read_line(&mut trailer);
            break;
        }
        read_fixed_sse_body(reader, size, emitter).map_err(body_truncated_at_eof)?;
        let mut crlf = [0_u8; 2];
        reader
            .read_exact(&mut crlf)
            .context("failed to read local provider chunk terminator")
            .map_err(body_truncated_at_eof)?;
        if crlf != *b"\r\n" {
            bail!("invalid local provider chunk terminator");
        }
    }
    Ok(())
}

fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
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

#[cfg(test)]
mod proptests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_provider_endpoint_parser_supports_ipv6_loopback() {
        assert_eq!(
            parse_http_endpoint("http://[::1]:11435/v1"),
            Some(("::1".to_owned(), 11435))
        );
    }
}
