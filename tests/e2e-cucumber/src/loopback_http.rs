// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Loopback HTTP file server for the install lifecycle E2E scenario.
//!
//! Dependency-free, including its own HTTP/1.x request-head parsing. The
//! scenario points the real installer at this server to exercise its native
//! HTTP download path.
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

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Header block terminator per RFC 9112 §2.1.
const HEADER_TERMINATOR: &[u8] = b"\r\n\r\n";

/// How long a single connection may take to deliver its request head before
/// the server gives up on it. Generous for a same-host client, but bounded so
/// a stuck peer cannot wedge the single-threaded accept loop.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Pause between retries when a socket reports "would block" — see
/// [`read_request_head`] for why a socket can be non-blocking here at all.
const WOULD_BLOCK_BACKOFF: Duration = Duration::from_millis(5);

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
///
/// `WouldBlock` is treated as "nothing has arrived yet, try again" rather than
/// as a failure. A blocking socket never reports it, but an accepted socket is
/// not guaranteed to be blocking: on Windows, `accept()` hands back a socket
/// that inherits the listening socket's non-blocking mode, so a server that
/// makes its listener non-blocking silently gets non-blocking connections too.
/// On such a socket, a request that has not landed in the receive buffer at
/// the instant of the first `read()` returns `WouldBlock` immediately, which
/// used to abort the connection with no response at all — the client saw only
/// a transport error. Retrying (bounded by `deadline`) makes the outcome
/// depend on the request arriving, not on when it arrives.
pub fn read_request_head<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {
    read_request_head_until(reader, Instant::now() + REQUEST_TIMEOUT)
}

/// [`read_request_head`], with an explicit deadline for the `WouldBlock`
/// retry loop so tests need not wait out [`REQUEST_TIMEOUT`].
pub fn read_request_head_until<R: Read>(reader: &mut R, deadline: Instant) -> io::Result<Vec<u8>> {
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
        let n = match reader.read(&mut chunk) {
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(e);
                }
                std::thread::sleep(WOULD_BLOCK_BACKOFF);
                continue;
            }
            Err(e) => return Err(e),
        };
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

/// A minimal loopback HTTP file server serving one directory, for the Windows
/// native-HTTP install scenario. Shuts down on drop.
pub struct LoopbackServer {
    port: u16,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl LoopbackServer {
    pub fn start(root: &Path) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind loopback listener");
        // Non-blocking accept lets the loop notice the stop flag promptly.
        listener
            .set_nonblocking(true)
            .expect("failed to set non-blocking");
        let port = listener.local_addr().expect("no local addr").port();
        let stop = Arc::new(AtomicBool::new(false));
        let root = root.to_path_buf();
        let stop_thread = Arc::clone(&stop);
        let handle = std::thread::spawn(move || serve(&listener, &root, &stop_thread));
        Self {
            port,
            stop,
            handle: Some(handle),
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for LoopbackServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn serve(listener: &TcpListener, root: &Path, stop: &Arc<AtomicBool>) {
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = handle_conn(stream, root);
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break,
        }
    }
}

/// Serve one connection: read its request head, then write the file (or a
/// 404) back.
///
/// The accepted socket is forced into blocking mode with explicit timeouts
/// first. `accept()` on Windows returns a socket that inherits the listening
/// socket's non-blocking mode, so without this the connection would be served
/// on a non-blocking socket and fail outright whenever the request bytes had
/// not already arrived — an inherently timing-dependent failure, and the
/// reason the second of two back-to-back downloads was the one that usually
/// broke.
pub fn handle_conn(mut stream: TcpStream, root: &Path) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(REQUEST_TIMEOUT))?;
    stream.set_write_timeout(Some(REQUEST_TIMEOUT))?;
    // Read until the full request head arrives, not just whatever a single
    // read() call happened to return: a client (or the OS network stack) is
    // free to deliver the request across more than one TCP segment, and a
    // one-shot read previously misparsed a split request as `GET /`, 404-ing
    // a legitimate download.
    let head = read_request_head(&mut stream)?;
    let path = parse_request_path(&head);
    let relative = path.trim_start_matches('/');
    let file = safe_join(root, relative);
    match file.and_then(|f| std::fs::read(&f).ok()) {
        Some(bytes) => {
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            );
            stream.write_all(header.as_bytes())?;
            stream.write_all(&bytes)?;
        }
        None => {
            stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )?;
        }
    }
    stream.flush()
}

/// Join a request path under `root`, rejecting any traversal so a malformed
/// request cannot read outside the served directory.
fn safe_join(root: &Path, relative: &str) -> Option<PathBuf> {
    if relative.is_empty() {
        return None;
    }
    let candidate = root.join(relative);
    let root = root.canonicalize().ok()?;
    let candidate = candidate.canonicalize().ok()?;
    candidate.starts_with(&root).then_some(candidate)
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

    /// A `Read` that reports `WouldBlock` a few times before delivering the
    /// request, the way a non-blocking socket behaves when the bytes have not
    /// landed in the receive buffer yet.
    struct WouldBlockThenReader {
        remaining_would_blocks: usize,
        inner: ChunkedReader,
    }

    impl Read for WouldBlockThenReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.remaining_would_blocks > 0 {
                self.remaining_would_blocks -= 1;
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "not ready"));
            }
            self.inner.read(buf)
        }
    }

    #[test]
    fn retries_past_would_block_instead_of_failing_the_request() {
        // Regression test for the Windows install-lifecycle flake: an accepted
        // socket inherits the listener's non-blocking mode on Windows, so the
        // first read() can report WouldBlock purely because the request has
        // not arrived yet. Aborting there produced a bare transport error at
        // the client with no HTTP response.
        let mut reader = WouldBlockThenReader {
            remaining_would_blocks: 3,
            inner: ChunkedReader::new(&[
                b"GET /rocm-cli-windows-amd64.zip.sha256 HTTP/1.1\r\n\r\n",
            ]),
        };
        let head = read_request_head(&mut reader).unwrap();
        assert_eq!(
            parse_request_path(&head),
            "/rocm-cli-windows-amd64.zip.sha256"
        );
    }

    #[test]
    fn gives_up_on_a_socket_that_never_becomes_ready() {
        let mut reader = WouldBlockThenReader {
            remaining_would_blocks: usize::MAX,
            inner: ChunkedReader::new(&[]),
        };
        let err = read_request_head_until(&mut reader, Instant::now()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
    }

    /// End-to-end over a real loopback socket: two back-to-back requests, the
    /// second issued immediately after the first completes, which is the exact
    /// shape of the installer's archive-then-`.sha256` download pair.
    #[test]
    fn serves_back_to_back_requests_on_a_non_blocking_accepted_socket() {
        use std::io::BufRead;

        let dir = std::env::temp_dir().join(format!("loopback-http-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("artifact.zip"), b"archive-bytes").unwrap();
        std::fs::write(dir.join("artifact.zip.sha256"), b"deadbeef").unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let root = dir.clone();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (stream, _) = listener.accept().unwrap();
                // Emulate the Windows behaviour of accept() handing back a
                // socket in the listener's non-blocking mode. handle_conn must
                // cope with this; before the fix it failed with WouldBlock and
                // wrote no response at all.
                stream.set_nonblocking(true).unwrap();
                handle_conn(stream, &root).unwrap();
            }
        });

        for (path, expected) in [
            ("/artifact.zip", "archive-bytes"),
            ("/artifact.zip.sha256", "deadbeef"),
        ] {
            let mut conn = TcpStream::connect(("127.0.0.1", port)).unwrap();
            write!(conn, "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").unwrap();
            let mut reader = io::BufReader::new(conn);
            let mut status = String::new();
            reader.read_line(&mut status).unwrap();
            assert!(status.starts_with("HTTP/1.1 200 OK"), "{path}: {status}");
            let mut body = Vec::new();
            reader.read_to_end(&mut body).unwrap();
            assert!(
                body.ends_with(expected.as_bytes()),
                "{path}: unexpected body"
            );
        }

        server.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
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
