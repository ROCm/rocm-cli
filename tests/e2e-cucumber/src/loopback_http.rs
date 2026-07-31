// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Minimal, dependency-free HTTP/1.x request-head parsing for the install
//! lifecycle's loopback download server (see
//! `tests/e2e/lifecycle_steps.rs`'s `http::LoopbackServer`).
//!
//! A single fixed-size `read()` call is not a valid way to receive an HTTP
//! request: a client, or the OS network stack in between, is free to deliver
//! it across any number of TCP segments, and nothing in the protocol
//! guarantees the whole request line — let alone the full header block —
//! lands in one `read()`. Treating one `read()` call as "the request" quietly
//! misparses a split request as `GET /`: reproduced locally by forcing a
//! request to arrive in two writes (`"GET "` then the rest), which made the
//! loopback server 404 a legitimate `rocm-cli-windows-amd64.zip` download.
//! This module reads until the full header block is observed and is
//! unit-tested against exactly that kind of fragmented delivery.

use std::io::{self, Read};

/// Header block terminator per RFC 9112 §2.1.
const HEADER_TERMINATOR: &[u8] = b"\r\n\r\n";

/// Cap on how much of the request head is buffered before giving up. The
/// loopback server only ever serves a handful of trusted, same-process test
/// clients, so this guards against reading forever on a malformed or endless
/// stream — it is not a real size limit any legitimate request approaches.
const MAX_REQUEST_HEAD_BYTES: usize = 64 * 1024;

/// Read from `reader` until the full HTTP request head (headers terminated by
/// a blank line) has been received, EOF is hit, or `MAX_REQUEST_HEAD_BYTES` is
/// exceeded — whichever comes first.
///
/// Loops across as many `read()` calls as needed, unlike a single fixed-size
/// `read()`, which can return an arbitrary prefix of the request if the
/// client or OS delivers it across more than one segment.
pub fn read_request_head<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        if let Some(end) = find_subslice(&buf, HEADER_TERMINATOR) {
            buf.truncate(end + HEADER_TERMINATOR.len());
            return Ok(buf);
        }
        if buf.len() >= MAX_REQUEST_HEAD_BYTES {
            return Ok(buf);
        }
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            // EOF before the header terminator arrived: return whatever the
            // client sent, best-effort, rather than blocking forever.
            return Ok(buf);
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// Extract the request-target (the path component) from a request head's
/// first line (`METHOD <path> HTTP/x.y`).
///
/// Falls back to `/` if the head is empty or malformed, matching this
/// server's historical behavior for any request it can't parse (which then
/// 404s against the served root).
pub fn parse_request_path(head: &[u8]) -> String {
    let text = String::from_utf8_lossy(head);
    text.lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
        .to_string()
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// A `Read` impl that yields the given chunks one `read()` call at a
    /// time (splitting a chunk further if it doesn't fit the caller's
    /// buffer), so tests can force a request to arrive split across multiple
    /// reads the way a real client/OS occasionally does.
    struct ChunkedReader {
        chunks: VecDeque<Vec<u8>>,
    }

    impl ChunkedReader {
        fn new(chunks: &[&[u8]]) -> Self {
            Self {
                chunks: chunks.iter().map(|c| c.to_vec()).collect(),
            }
        }
    }

    impl Read for ChunkedReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let Some(mut chunk) = self.chunks.pop_front() else {
                return Ok(0);
            };
            let n = chunk.len().min(buf.len());
            buf[..n].copy_from_slice(&chunk[..n]);
            if n < chunk.len() {
                self.chunks.push_front(chunk.split_off(n));
            }
            Ok(n)
        }
    }

    #[test]
    fn reads_a_request_delivered_in_a_single_read() {
        let request = b"GET /rocm-cli-windows-amd64.zip HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
        let mut reader = ChunkedReader::new(&[request]);
        let head = read_request_head(&mut reader).unwrap();
        assert_eq!(parse_request_path(&head), "/rocm-cli-windows-amd64.zip");
    }

    #[test]
    fn reads_a_request_split_mid_request_line_across_reads() {
        // Regression test: a single fixed-size read() call previously treated
        // whatever arrived first ("GET ") as the whole request, which parsed
        // to path "/" and 404'd a legitimate download -- reproduced against
        // the real LoopbackServer over a loopback TCP socket before this fix.
        let mut reader = ChunkedReader::new(&[
            b"GET ",
            b"/rocm-cli-windows-amd64.zip HTTP/1.1\r\n",
            b"Host: 127.0.0.1\r\n\r\n",
        ]);
        let head = read_request_head(&mut reader).unwrap();
        assert_eq!(parse_request_path(&head), "/rocm-cli-windows-amd64.zip");
    }

    #[test]
    fn reads_a_request_split_byte_by_byte() {
        let request: &[u8] = b"GET /x.zip HTTP/1.1\r\nHost: h\r\n\r\n";
        let chunks: Vec<&[u8]> = request.iter().map(std::slice::from_ref).collect();
        let mut reader = ChunkedReader::new(&chunks);
        let head = read_request_head(&mut reader).unwrap();
        assert_eq!(parse_request_path(&head), "/x.zip");
    }

    #[test]
    fn stops_at_eof_with_no_terminator_rather_than_hanging() {
        let mut reader = ChunkedReader::new(&[b"GET /partial"]);
        let head = read_request_head(&mut reader).unwrap();
        // No headers terminator ever arrived (the client hung up early); the
        // reader still returns whatever it got instead of blocking forever.
        assert_eq!(parse_request_path(&head), "/partial");
    }

    #[test]
    fn empty_request_falls_back_to_root() {
        let head: Vec<u8> = Vec::new();
        assert_eq!(parse_request_path(&head), "/");
    }

    #[test]
    fn oversized_request_head_stops_at_the_cap_instead_of_growing_unbounded() {
        let huge = vec![b'a'; MAX_REQUEST_HEAD_BYTES + 4096];
        let mut reader = ChunkedReader::new(&[&huge]);
        let head = read_request_head(&mut reader).unwrap();
        assert!(head.len() <= MAX_REQUEST_HEAD_BYTES);
    }
}
