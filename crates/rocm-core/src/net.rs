// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Networking primitives.
//!
//! Default local host/port, URL/endpoint formatting, streaming downloads
//! with resume and retry, and the loopback HTTP/TCP helpers the
//! managed-service readiness probes in `endpoint_readiness.rs` are built on.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::net::{IpAddr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crate::disk_space;

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

/// Attempts a download makes before giving up. The first attempt is not a
/// retry, so this is one initial try plus two retries.
pub const DOWNLOAD_MAX_ATTEMPTS: u32 = 3;

/// Chunk size for the streaming copy. Fixed, so peak memory is independent of
/// the artifact size — the whole point of streaming a multi-gigabyte tarball.
const DOWNLOAD_CHUNK_BYTES: usize = 64 * 1024;

/// Exponential backoff between download attempts.
///
/// A copy of the shape used by the dashboard's reconnect loop rather than a
/// shared dependency: `rocm-core` sits below the dash crates, so it cannot
/// import theirs.
#[derive(Debug, Clone, Copy)]
pub struct Backoff {
    current: Duration,
    max: Duration,
    factor: u32,
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new(Duration::from_millis(500), Duration::from_secs(8), 2)
    }
}

impl Backoff {
    pub const fn new(initial: Duration, max: Duration, factor: u32) -> Self {
        Self {
            current: initial,
            max,
            factor,
        }
    }

    /// The delay to wait before the next attempt, then grow toward `max`.
    pub fn next_delay(&mut self) -> Duration {
        let delay = self.current;
        self.current = self.current.saturating_mul(self.factor).min(self.max);
        delay
    }
}

/// A download of a single artifact to a single path.
#[derive(Debug, Clone, Copy)]
pub struct DownloadRequest<'a> {
    pub url: &'a str,
    pub destination: &'a Path,
    pub timeout: Duration,
    pub headers: &'a [(&'a str, &'a str)],
    /// Refuse a body larger than this, so an unexpected response cannot fill
    /// the disk. `None` accepts whatever the server sends.
    pub max_bytes: Option<u64>,
    /// Size the caller already knows from a manifest. Checked in addition to
    /// the server's own `Content-Length`.
    pub expected_len: Option<u64>,
    pub expected_sha256: Option<&'a str>,
}

impl<'a> DownloadRequest<'a> {
    pub const fn new(url: &'a str, destination: &'a Path, timeout: Duration) -> Self {
        Self {
            url,
            destination,
            timeout,
            headers: &[],
            max_bytes: None,
            expected_len: None,
            expected_sha256: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DownloadOutcome {
    pub bytes_written: u64,
    pub sha256: String,
}

/// Stream `request.url` to `request.destination`, retrying transient failures
/// and resuming where the server allows it.
///
/// Bytes go through a fixed [`DOWNLOAD_CHUNK_BYTES`] buffer into a sibling
/// `.part` file and are hashed on the way past, so peak memory does not scale
/// with the artifact — a multi-gigabyte SDK tarball costs the same as a small
/// one. The `.part` file is a sibling of the destination so the final rename
/// stays within one filesystem and is atomic: a caller that finds the
/// destination present knows it holds a complete download, and an interrupted
/// run never leaves a truncated file that a later run would treat as cached.
///
/// # Integrity
///
/// With `expected_sha256` the content is authenticated. Without it the only
/// integrity signal is the byte count, cross-checked against `Content-Length`
/// and `expected_len`. That catches truncation and interrupted transfers, which
/// is what this guards against, but `Content-Length` is unauthenticated and a
/// matching length proves nothing about the bytes — do not read a successful
/// return as "the artifact is genuine" unless a digest was supplied.
pub fn download_file_streaming(request: &DownloadRequest<'_>) -> Result<DownloadOutcome> {
    download_file_streaming_with_progress(request, &mut |_written, _total| {})
}

/// As [`download_file_streaming`], but reports progress via `on_progress`.
///
/// `on_progress` is called with the cumulative bytes written and, when
/// known, the total size — once before the transfer starts and once after
/// every chunk is written to disk. The byte count is monotonically
/// non-decreasing across the whole call, including across retries: an
/// attempt that restarts from scratch (the server ignored `Range`, or
/// resumed at the wrong offset and had its partial file discarded) counts
/// its own bytes from 0 internally, but the byte count `on_progress` sees
/// never drops below the highest value already reported by an earlier
/// attempt. The total is not clamped the same way and is passed through as
/// reported by the current attempt, so it can go from `None` to `Some` (or
/// back) mid-transfer if a retry's response differs on `Content-Length`.
/// Note also that for the whole duration of a from-scratch restart, the
/// byte count holds flat at the prior high-water mark until the new attempt
/// catches back up — a caller driving a static progress line should pair
/// this with an animated indicator (as the `rocm` CLI's spinner does) so a
/// long restart doesn't look hung. Callers that don't need progress should
/// use [`download_file_streaming`] instead.
pub fn download_file_streaming_with_progress(
    request: &DownloadRequest<'_>,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<DownloadOutcome> {
    if let Some(parent) = request.destination.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let partial_path = partial_download_path(request.destination);
    // Resume is scoped to retries within this call. A `.part` file found at
    // entry is debris from an earlier run — process ids get reused, so its
    // contents cannot be attributed to this URL and appending to it would
    // silently produce a corrupt artifact.
    let _ = fs::remove_file(&partial_path);
    let mut backoff = Backoff::default();
    let mut attempt = 1;
    // `download_attempt` reports whatever it has on disk for *this* attempt,
    // which resets to 0 on a from-scratch restart even though earlier
    // attempts already progressed further. Clamp to a high-water mark here
    // so every caller — not just ones that happen to add their own UI-side
    // clamp — sees a byte count that never goes backwards.
    let mut high_water = 0_u64;
    let mut monotonic_progress = move |written: u64, total: Option<u64>| {
        high_water = high_water.max(written);
        on_progress(high_water, total);
    };
    let outcome = loop {
        match download_attempt(request, &partial_path, &mut monotonic_progress) {
            Ok(outcome) => break outcome,
            Err(error) => {
                let retryable = error.retryable && attempt < DOWNLOAD_MAX_ATTEMPTS;
                if !retryable {
                    let _ = fs::remove_file(&partial_path);
                    return Err(error.error);
                }
                thread::sleep(backoff.next_delay());
                attempt += 1;
            }
        }
    };
    fs::rename(&partial_path, request.destination).map_err(|error| {
        let _ = fs::remove_file(&partial_path);
        anyhow::Error::new(error).context(format!(
            "failed to move the completed download into {}",
            request.destination.display()
        ))
    })?;
    Ok(outcome)
}

/// Where the in-progress bytes live. Keyed by process id so two processes
/// downloading the same destination cannot append into each other's file.
fn partial_download_path(destination: &Path) -> PathBuf {
    let mut name = destination.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".part-{}", std::process::id()));
    destination.with_file_name(name)
}

/// A failed attempt, and whether trying again could plausibly help.
struct DownloadAttemptError {
    error: anyhow::Error,
    retryable: bool,
}

const fn transient(error: anyhow::Error) -> DownloadAttemptError {
    DownloadAttemptError {
        error,
        retryable: true,
    }
}

const fn permanent(error: anyhow::Error) -> DownloadAttemptError {
    DownloadAttemptError {
        error,
        retryable: false,
    }
}

/// Whether a status is worth another attempt. Server-side and rate-limit
/// responses can succeed later; the rest of `4xx` means the request itself is
/// wrong, and repeating it just wastes the user's time.
const fn status_is_retryable(status: u16) -> bool {
    status == 408 || status == 429 || status >= 500
}

/// A single attempt at the transfer. `written` — and so what this reports
/// through `on_progress` — reflects only what this attempt itself has put on
/// disk: a confirmed `206` continuation starts counting from the resumed
/// offset, but a restart (the server ignored `Range`, or resumed at the
/// wrong offset and had its partial file discarded) truncates the file and
/// starts counting from 0 again, even if a previous attempt already reported
/// further along. That's fine — [`download_file_streaming_with_progress`]
/// wraps `on_progress` with a high-water mark so callers never observe the
/// drop; this function does not need to care.
fn download_attempt(
    request: &DownloadRequest<'_>,
    partial_path: &Path,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<DownloadOutcome, DownloadAttemptError> {
    // Resume from whatever a previous attempt already wrote. A missing file is
    // simply a fresh start.
    let resume_from = fs::metadata(partial_path).map_or(0, |meta| meta.len());
    let mut call = ureq::get(request.url).timeout(request.timeout);
    for (name, value) in request.headers {
        call = call.set(name, value);
    }
    if resume_from > 0 {
        call = call.set("Range", &format!("bytes={resume_from}-"));
    }
    let response = match call.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(status, _)) => {
            let error = anyhow::anyhow!("HTTP {status} while downloading {url}", url = request.url);
            return Err(if status_is_retryable(status) {
                transient(error)
            } else {
                permanent(error)
            });
        }
        // Transport failures are exactly the interruptions worth retrying.
        Err(error) => {
            return Err(transient(
                anyhow::Error::new(error).context(format!("failed to download {}", request.url)),
            ));
        }
    };
    let status = response.status();
    if status != 200 && status != 206 {
        return Err(permanent(anyhow::anyhow!(
            "HTTP {status} while downloading {url}",
            url = request.url
        )));
    }
    // A `206` that does not confirm continuing from `resume_from` cannot be
    // treated as a fresh full download either: its `Content-Length` is the
    // length of whatever slice the server chose to send, not the whole
    // artifact, so accepting it as complete would silently rename a truncated
    // body into place. Discard the partial file and restart clean with a
    // plain `GET` instead of reinterpreting the mismatched slice.
    if status == 206 && resume_from > 0 && content_range_start(&response) != Some(resume_from) {
        let _ = fs::remove_file(partial_path);
        return Err(transient(anyhow::anyhow!(
            "{url} resumed from an unexpected byte offset; restarting the download",
            url = request.url
        )));
    }
    // Only append when the server confirmed it is continuing from exactly where
    // we stopped. A server that ignores `Range` answers 200 with the whole body,
    // which restarts cleanly too.
    let resuming = status == 206 && resume_from > 0;
    let remaining_len = header_u64(&response, "Content-Length");
    let total_len =
        remaining_len.map(|len| len.saturating_add(if resuming { resume_from } else { 0 }));
    // Fall back to the caller-supplied expected length when the server omits
    // `Content-Length`, so progress reporting doesn't lose a total that's
    // already known and already used for the preflight checks below.
    let reported_total = total_len.or(request.expected_len);
    if let Some(total) = reported_total {
        if let Some(max_bytes) = request.max_bytes
            && total > max_bytes
        {
            return Err(permanent(anyhow::anyhow!(
                "{url} is {total} bytes, over the approved limit of {max_bytes}",
                url = request.url
            )));
        }
        // Free-space preflight, before a single byte is read: an exact size
        // means we can refuse upfront instead of failing partway through. Only
        // the bytes still to come need room; a resumed prefix already has it.
        let still_needed = total.saturating_sub(if resuming { resume_from } else { 0 });
        if let Err(error) = disk_space::ensure_space_for(
            &format!("download {}", request.url),
            request.destination,
            disk_space::with_margin(still_needed),
        ) {
            return Err(permanent(error));
        }
    }

    let mut hasher = Sha256::new();
    let mut written = 0_u64;
    let mut file = if resuming {
        // Re-hash the prefix so the digest covers the whole artifact, not just
        // the bytes this attempt happened to fetch.
        let mut existing =
            fs::File::open(partial_path).map_err(|error| permanent(anyhow::Error::new(error)))?;
        written = std::io::copy(&mut existing, &mut hasher)
            .map_err(|error| permanent(anyhow::Error::new(error)))?;
        fs::OpenOptions::new()
            .append(true)
            .open(partial_path)
            .map_err(|error| permanent(anyhow::Error::new(error)))?
    } else {
        fs::File::create(partial_path).map_err(|error| {
            permanent(
                anyhow::Error::new(error)
                    .context(format!("failed to create {}", partial_path.display())),
            )
        })?
    };

    on_progress(written, reported_total);

    let mut reader = response.into_reader();
    let mut buffer = vec![0_u8; DOWNLOAD_CHUNK_BYTES];
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            // A mid-body failure keeps the partial file: the next attempt
            // resumes from it. A server that drops the connection early and one
            // that closes cleanly after a short body are the same problem to
            // the user, so report the shortfall either way rather than a bare
            // transport error.
            Err(error) => {
                let reason = reported_total.map_or_else(
                    || format!("failed while downloading {}", request.url),
                    |expected| {
                        format!(
                            "incomplete download of {}: expected {expected} bytes, got {written}",
                            request.url
                        )
                    },
                );
                return Err(transient(anyhow::Error::new(error).context(reason)));
            }
        };
        written = written.saturating_add(read as u64);
        if let Some(max_bytes) = request.max_bytes
            && written > max_bytes
        {
            return Err(permanent(anyhow::anyhow!(
                "{url} exceeded the approved limit of {max_bytes} bytes",
                url = request.url
            )));
        }
        hasher.update(&buffer[..read]);
        if let Err(error) = file.write_all(&buffer[..read]) {
            return Err(permanent(disk_space::map_write_error(error, partial_path)));
        }
        on_progress(written, reported_total);
    }
    if let Err(error) = file.sync_all() {
        return Err(permanent(disk_space::map_write_error(error, partial_path)));
    }
    drop(file);

    // Two independent length contracts, checked separately because they fail
    // for different reasons and deserve different handling.
    //
    // The server's own `Content-Length`: a shortfall means the transfer was cut
    // short, so the bytes on disk are good as far as they go and the next
    // attempt resumes from them.
    if let Some(expected) = total_len
        && written != expected
    {
        return Err(transient(anyhow::anyhow!(
            "incomplete download of {url}: expected {expected} bytes, got {written}",
            url = request.url
        )));
    }
    // A size the caller knew in advance: the server delivered a complete body
    // that is not the artifact the manifest describes. Refetching returns the
    // same wrong thing, so do not spend the remaining attempts on it.
    if let Some(expected) = request.expected_len
        && written != expected
    {
        return Err(permanent(anyhow::anyhow!(
            "{url} is {written} bytes, but {expected} were expected",
            url = request.url
        )));
    }
    let sha256 = format!("{:x}", hasher.finalize());
    if let Some(expected) = request.expected_sha256
        && !sha256.eq_ignore_ascii_case(expected)
    {
        // The bytes are wrong, not merely incomplete; resuming would append to
        // a corrupt prefix, so discard it and fail.
        let _ = fs::remove_file(partial_path);
        return Err(permanent(anyhow::anyhow!(
            "SHA-256 mismatch for {url}: expected {expected}, got {sha256}",
            url = request.url
        )));
    }
    Ok(DownloadOutcome {
        bytes_written: written,
        sha256,
    })
}

fn header_u64(response: &ureq::Response, name: &str) -> Option<u64> {
    response
        .header(name)
        .and_then(|value| value.trim().parse::<u64>().ok())
}

/// First byte offset from a `Content-Range: bytes <start>-<end>/<total>` header.
fn content_range_start(response: &ureq::Response) -> Option<u64> {
    let value = response.header("Content-Range")?;
    let range = value.trim().strip_prefix("bytes")?.trim_start();
    let start = range.split('-').next()?.trim();
    start.parse::<u64>().ok()
}

pub fn download_file_to_path(url: &str, destination: &Path, timeout: Duration) -> Result<()> {
    download_file_streaming(&DownloadRequest::new(url, destination, timeout))?;
    Ok(())
}

/// As [`download_file_to_path`], but reports progress via `on_progress`. See
/// [`download_file_streaming_with_progress`] for callback semantics.
pub fn download_file_to_path_with_progress(
    url: &str,
    destination: &Path,
    timeout: Duration,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<()> {
    download_file_streaming_with_progress(
        &DownloadRequest::new(url, destination, timeout),
        on_progress,
    )?;
    Ok(())
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

    use crate::test_support::workspace_test_artifact_dir;
    use std::io::{Read, Write};

    /// How a scripted download server answers one request.
    #[derive(Clone, Copy)]
    enum DownloadReply {
        /// The whole body with a matching `Content-Length`.
        Complete,
        /// Declare the full length, send only `sent` bytes, then close —
        /// an interrupted transfer.
        Truncated {
            sent: usize,
        },
        /// Honour `Range` and serve the remainder as `206`.
        Resume,
        /// Answer `206` but from `start` regardless of what `Range` asked for,
        /// as a non-compliant server or a broken caching proxy might.
        ResumeAtWrongOffset {
            start: usize,
        },
        /// Ignore `Range` and answer `200` with the whole body, as a server
        /// without range support does.
        IgnoreRange,
        Status(&'static str),
    }
    impl DownloadReply {
        fn client_may_disconnect(self) -> bool {
            matches!(
                self,
                Self::ResumeAtWrongOffset { .. } | Self::Truncated { sent: 0 }
            )
        }
    }
    fn allow_expected_peer_disconnect(result: std::io::Result<()>) -> std::io::Result<()> {
        match result {
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
                ) =>
            {
                Ok(())
            }
            result => result,
        }
    }
    /// Serve `body` over loopback, answering each request per `replies`.
    /// Returns the port and a handle yielding the raw requests received.
    fn spawn_download_server(
        body: Vec<u8>,
        replies: Vec<DownloadReply>,
    ) -> Result<(u16, std::thread::JoinHandle<Result<Vec<String>>>)> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let handle = std::thread::spawn(move || -> Result<Vec<String>> {
            let mut requests = Vec::new();
            for reply in replies {
                let (mut stream, _) = listener.accept()?;
                // An accepted socket inherits the listener's mode on Windows,
                // where a non-blocking read fails with `WSAEWOULDBLOCK` instead
                // of waiting. Be explicit rather than rely on the platform.
                stream.set_nonblocking(false)?;
                stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                let mut request_bytes = Vec::new();
                let mut buffer = [0_u8; 512];
                while let Ok(read) = stream.read(&mut buffer) {
                    if read == 0 {
                        break;
                    }
                    request_bytes.extend_from_slice(&buffer[..read]);
                    if request_bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&request_bytes).into_owned();
                let range_start = request
                    .lines()
                    .find_map(|line| line.strip_prefix("Range: bytes="))
                    .and_then(|value| value.trim().split('-').next()?.parse::<usize>().ok())
                    .unwrap_or(0);
                requests.push(request);
                let total = body.len();
                let client_may_disconnect = reply.client_may_disconnect();
                match reply {
                    DownloadReply::Complete | DownloadReply::IgnoreRange => {
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
                        )?;
                        stream.write_all(&body)?;
                    }
                    DownloadReply::Truncated { sent } => {
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
                        )?;
                        stream.write_all(&body[..sent.min(total)])?;
                    }
                    DownloadReply::Resume => {
                        let start = range_start.min(total);
                        write!(
                            stream,
                            "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {start}-{}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            total.saturating_sub(1),
                            total - start
                        )?;
                        stream.write_all(&body[start..])?;
                    }
                    DownloadReply::ResumeAtWrongOffset { start } => {
                        let start = start.min(total);
                        write!(
                            stream,
                            "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {start}-{}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            total.saturating_sub(1),
                            total - start
                        )?;
                        allow_expected_peer_disconnect(stream.write_all(&body[start..]))?;
                    }
                    DownloadReply::Status(status_line) => {
                        write!(
                            stream,
                            "{status_line}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )?;
                    }
                }
                if client_may_disconnect {
                    allow_expected_peer_disconnect(stream.flush())?;
                } else {
                    stream.flush()?;
                }
            }
            Ok(requests)
        });
        Ok((port, handle))
    }
    /// A body large enough to span many `DOWNLOAD_CHUNK_BYTES` reads, so the
    /// streaming loop is genuinely exercised rather than fitting in one chunk.
    fn download_body() -> Vec<u8> {
        (0..DOWNLOAD_CHUNK_BYTES * 3 + 1234)
            .map(|index| (index % 251) as u8)
            .collect()
    }
    fn download_scratch(tag: &str) -> PathBuf {
        let dir = workspace_test_artifact_dir().join(format!(
            "download-{tag}-{}",
            crate::test_support::unique_suffix()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("failed to create scratch dir");
        dir
    }

    #[test]
    fn download_scratch_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| download_scratch("same-label"));
    }

    #[test]
    fn download_fixture_tolerates_only_expected_peer_disconnects() {
        assert!(DownloadReply::ResumeAtWrongOffset { start: 1 }.client_may_disconnect());
        assert!(DownloadReply::Truncated { sent: 0 }.client_may_disconnect());
        assert!(!DownloadReply::Truncated { sent: 1 }.client_may_disconnect());
        assert!(!DownloadReply::Complete.client_may_disconnect());

        for kind in [
            std::io::ErrorKind::BrokenPipe,
            std::io::ErrorKind::ConnectionReset,
        ] {
            allow_expected_peer_disconnect(Err(std::io::Error::from(kind)))
                .expect("an intentionally rejected response may close its socket");
        }

        let error =
            allow_expected_peer_disconnect(Err(std::io::Error::from(std::io::ErrorKind::TimedOut)))
                .expect_err("unrelated fixture I/O failures must remain visible");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    }
    #[test]
    fn download_streams_a_large_body_and_reports_its_digest() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(body.clone(), vec![DownloadReply::Complete])?;
        let dir = download_scratch("complete");
        let destination = dir.join("artifact.bin");

        let outcome = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(10),
        ))?;

        let written = fs::read(&destination)?;
        let expected_digest = format!("{:x}", Sha256::digest(&body));
        server.join().expect("server thread")?;
        let leftovers: Vec<_> = fs::read_dir(&dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.file_name()))
            .filter(|name| name.to_string_lossy().contains(".part"))
            .collect();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(written, body, "the saved file must match the served bytes");
        assert_eq!(outcome.bytes_written, body.len() as u64);
        assert_eq!(outcome.sha256, expected_digest);
        assert!(leftovers.is_empty(), "the .part file must not survive");
        Ok(())
    }
    #[test]
    fn download_with_progress_reports_cumulative_bytes_across_chunks() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(body.clone(), vec![DownloadReply::Complete])?;
        let dir = download_scratch("progress");
        let destination = dir.join("artifact.bin");

        let mut calls: Vec<(u64, Option<u64>)> = Vec::new();
        let outcome = download_file_streaming_with_progress(
            &DownloadRequest::new(
                &format!("http://127.0.0.1:{port}/artifact.bin"),
                &destination,
                Duration::from_secs(10),
            ),
            &mut |written, total| calls.push((written, total)),
        )?;

        server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        let total = Some(body.len() as u64);
        assert_eq!(
            calls.first(),
            Some(&(0, total)),
            "the first call must fire before any bytes are read, already knowing the total: {calls:?}"
        );
        assert_eq!(
            calls.last(),
            Some(&(body.len() as u64, total)),
            "the last call must report the complete byte count: {calls:?}"
        );
        assert!(
            calls.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "cumulative bytes must never go backwards: {calls:?}"
        );
        assert!(
            calls
                .iter()
                .all(|&(_, reported_total)| reported_total == total),
            "the total must stay constant across an attempt: {calls:?}"
        );
        assert_eq!(outcome.bytes_written, body.len() as u64);
        Ok(())
    }
    #[test]
    // This guards the resume path re-seeding `written` from the partial file
    // already on disk (see `written = std::io::copy(...)` above), not the
    // high-water-mark clamp itself — it would pass unchanged with the clamp
    // removed entirely. `download_with_progress_stays_monotonic_after_a_discarded_restart`
    // below is the one that actually exercises the clamp.
    fn download_with_progress_reports_the_resumed_offset_before_reading_more() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 5000 },
                DownloadReply::Resume,
            ],
        )?;
        let dir = download_scratch("progress-resume");
        let destination = dir.join("artifact.bin");

        let mut calls: Vec<(u64, Option<u64>)> = Vec::new();
        let outcome = download_file_streaming_with_progress(
            &DownloadRequest::new(
                &format!("http://127.0.0.1:{port}/artifact.bin"),
                &destination,
                Duration::from_secs(10),
            ),
            &mut |written, total| calls.push((written, total)),
        )?;

        server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        let total = Some(body.len() as u64);
        assert!(
            calls.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "byte counts must never regress across a retried attempt, e.g. reset to 0: {calls:?}"
        );
        assert!(
            calls.iter().any(|&(written, _)| written == 5000),
            "the resumed attempt must report the byte count already on disk before reading more: {calls:?}"
        );
        assert!(
            calls
                .iter()
                .all(|&(_, reported_total)| reported_total == total),
            "the total size must stay stable across the retry: {calls:?}"
        );
        assert_eq!(
            calls.last(),
            Some(&(body.len() as u64, total)),
            "the last call must report the complete byte count: {calls:?}"
        );
        assert_eq!(outcome.bytes_written, body.len() as u64);
        Ok(())
    }
    #[test]
    fn download_with_progress_stays_monotonic_after_a_discarded_restart() -> Result<()> {
        // The second reply resumes at the wrong offset, so its partial file is
        // discarded and the third attempt restarts from scratch — internally
        // reporting 0 bytes written again even though the first attempt had
        // already reached 5000. The caller must never see that drop.
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 5000 },
                DownloadReply::ResumeAtWrongOffset { start: 8000 },
                DownloadReply::Complete,
            ],
        )?;
        let dir = download_scratch("progress-wrong-offset");
        let destination = dir.join("artifact.bin");

        let mut calls: Vec<(u64, Option<u64>)> = Vec::new();
        let outcome = download_file_streaming_with_progress(
            &DownloadRequest::new(
                &format!("http://127.0.0.1:{port}/artifact.bin"),
                &destination,
                Duration::from_secs(10),
            ),
            &mut |written, total| calls.push((written, total)),
        )?;

        let requests = server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        let total = Some(body.len() as u64);
        assert_eq!(
            requests.len(),
            3,
            "the wrong-offset reply must be discarded and retried, not accepted"
        );
        assert!(
            calls.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "cumulative bytes must never go backwards, even across a discarded \
             partial file and a from-scratch restart: {calls:?}"
        );
        assert!(
            calls.iter().any(|&(written, _)| written == 5000),
            "the truncated first attempt's progress must not be lost once the \
             restart reports 0 internally: {calls:?}"
        );
        assert_eq!(
            calls.last(),
            Some(&(body.len() as u64, total)),
            "the last call must report the complete byte count: {calls:?}"
        );
        assert_eq!(outcome.bytes_written, body.len() as u64);
        Ok(())
    }
    #[test]
    fn download_interrupted_beyond_recovery_leaves_no_destination_file() -> Result<()> {
        // Every attempt ends early, so the download never completes and the
        // user is left with the truncation as the reported reason.
        let body = download_body();
        let (port, server) = spawn_download_server(
            body,
            vec![DownloadReply::Truncated { sent: 4096 }; DOWNLOAD_MAX_ATTEMPTS as usize],
        )?;
        let dir = download_scratch("interrupted");
        let destination = dir.join("artifact.bin");

        let error = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(5),
        ))
        .expect_err("a download that never completes must fail");

        let requests = server.join().expect("server thread")?;
        let entries: Vec<_> = fs::read_dir(&dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.file_name()))
            .collect();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(
            requests.len(),
            DOWNLOAD_MAX_ATTEMPTS as usize,
            "a truncated transfer is transient, so every attempt should be spent"
        );
        assert!(
            !destination.exists(),
            "a partial download must never appear at the destination, where a \
             later run would treat it as a complete cached artifact"
        );
        assert!(
            entries.is_empty(),
            "the .part file must be cleaned up, found {entries:?}"
        );
        assert!(
            error.to_string().contains("incomplete download"),
            "the user should be told the transfer was short: {error}"
        );
        Ok(())
    }
    #[test]
    fn download_resumes_from_where_the_transfer_stopped() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 5000 },
                DownloadReply::Resume,
            ],
        )?;
        let dir = download_scratch("resume");
        let destination = dir.join("artifact.bin");

        let outcome = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(10),
        ))?;

        let written = fs::read(&destination)?;
        let requests = server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert_eq!(
            written, body,
            "a resumed download must reconstruct the artifact exactly"
        );
        assert_eq!(
            outcome.sha256,
            format!("{:x}", Sha256::digest(&body)),
            "the digest must cover the resumed prefix too, not just the second attempt"
        );
        assert_eq!(requests.len(), 2, "one retry, not a full restart");
        assert!(
            requests[1].contains("Range: bytes=5000-"),
            "the retry must ask to continue from byte 5000: {}",
            requests[1]
        );
        Ok(())
    }
    #[test]
    fn download_restarts_cleanly_when_resume_lands_at_the_wrong_offset() -> Result<()> {
        // The second reply answers `206` from byte 8000, not the byte 5000 we
        // asked to resume from. Accepting that slice's own `Content-Length` as
        // the whole artifact would silently rename a corrupt file into place;
        // instead the third attempt must be a plain `GET` that refetches the
        // whole thing.
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 5000 },
                DownloadReply::ResumeAtWrongOffset { start: 8000 },
                DownloadReply::Complete,
            ],
        )?;
        let dir = download_scratch("wrong-offset");
        let destination = dir.join("artifact.bin");

        let outcome = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(10),
        ))?;

        let written = fs::read(&destination)?;
        let requests = server.join().expect("server thread")?;
        let leftovers: Vec<_> = fs::read_dir(&dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.file_name()))
            .filter(|name| name.to_string_lossy().contains(".part"))
            .collect();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(
            written, body,
            "a mismatched-offset resume must not be accepted as a complete artifact"
        );
        assert_eq!(
            outcome.sha256,
            format!("{:x}", Sha256::digest(&body)),
            "the digest must cover the whole artifact from the clean restart"
        );
        assert_eq!(
            requests.len(),
            3,
            "the wrong-offset reply must be discarded and retried, not accepted"
        );
        assert!(
            !requests[2].contains("Range:"),
            "the restart after a wrong-offset resume must be a plain GET: {}",
            requests[2]
        );
        assert!(leftovers.is_empty(), "the .part file must not survive");
        Ok(())
    }
    #[test]
    fn download_restarts_cleanly_when_the_server_ignores_range() -> Result<()> {
        // A server without range support answers 200 with the whole body.
        // Appending that onto the bytes already written would double the file.
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 3000 },
                DownloadReply::IgnoreRange,
            ],
        )?;
        let dir = download_scratch("ignore-range");
        let destination = dir.join("artifact.bin");

        download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(10),
        ))?;

        let written = fs::read(&destination)?;
        server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert_eq!(written, body, "the restart must not append onto the prefix");
        Ok(())
    }
    #[test]
    fn download_rejects_a_digest_mismatch_without_retrying() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(body, vec![DownloadReply::Complete])?;
        let dir = download_scratch("digest");
        let destination = dir.join("artifact.bin");
        let url = format!("http://127.0.0.1:{port}/artifact.bin");
        let wrong_digest = "a".repeat(64);
        let mut request = DownloadRequest::new(&url, &destination, Duration::from_secs(10));
        request.expected_sha256 = Some(&wrong_digest);

        let error = download_file_streaming(&request).expect_err("wrong digest must fail");

        let requests = server.join().expect("server thread")?;
        let leftovers = fs::read_dir(&dir)?.count();
        fs::remove_dir_all(&dir).ok();

        assert!(error.to_string().contains("SHA-256 mismatch"), "{error}");
        assert!(!destination.exists());
        assert_eq!(
            requests.len(),
            1,
            "corrupt bytes are not a transient failure; retrying would re-fetch the same thing"
        );
        assert_eq!(leftovers, 0, "the corrupt prefix must be discarded");
        Ok(())
    }
    #[test]
    fn download_does_not_retry_a_client_error() -> Result<()> {
        let body = download_body();
        let (port, server) =
            spawn_download_server(body, vec![DownloadReply::Status("HTTP/1.1 404 Not Found")])?;
        let dir = download_scratch("not-found");
        let destination = dir.join("artifact.bin");

        let error = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(5),
        ))
        .expect_err("404 must fail");

        let requests = server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert!(error.to_string().contains("404"), "{error}");
        assert_eq!(requests.len(), 1, "a 404 will not become a 200 on retry");
        Ok(())
    }
    #[test]
    fn download_refuses_a_body_over_the_approved_limit() -> Result<()> {
        let body = download_body();
        let total = body.len() as u64;
        let (port, server) =
            spawn_download_server(body, vec![DownloadReply::Truncated { sent: 0 }])?;
        let dir = download_scratch("max-bytes");
        let destination = dir.join("artifact.bin");
        let url = format!("http://127.0.0.1:{port}/artifact.bin");
        let mut request = DownloadRequest::new(&url, &destination, Duration::from_secs(5));
        request.max_bytes = Some(total - 1);

        let error = download_file_streaming(&request).expect_err("over-limit must fail");

        server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert!(error.to_string().contains("approved limit"), "{error}");
        assert!(!destination.exists());
        Ok(())
    }
    #[test]
    fn download_enforces_a_caller_declared_size_the_server_agrees_with() -> Result<()> {
        // The server sends a complete, self-consistent body — it just is not
        // the artifact the manifest described. A digest would catch this too,
        // but the size contract is independent and must hold on its own.
        let body = download_body();
        let (port, server) = spawn_download_server(body.clone(), vec![DownloadReply::Complete])?;
        let dir = download_scratch("expected-len");
        let destination = dir.join("artifact.bin");
        let url = format!("http://127.0.0.1:{port}/artifact.bin");
        let mut request = DownloadRequest::new(&url, &destination, Duration::from_secs(10));
        request.expected_len = Some(body.len() as u64 + 1);

        let error = download_file_streaming(&request).expect_err("wrong size must fail");

        let requests = server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert!(error.to_string().contains("were expected"), "{error}");
        assert!(!destination.exists());
        assert_eq!(
            requests.len(),
            1,
            "a manifest disagreement will not resolve itself on retry"
        );
        Ok(())
    }
    #[test]
    fn download_backoff_grows_then_settles_at_its_ceiling() {
        let mut backoff = Backoff::new(Duration::from_millis(100), Duration::from_millis(800), 2);
        let delays: Vec<u128> = (0..5).map(|_| backoff.next_delay().as_millis()).collect();
        assert_eq!(delays, vec![100, 200, 400, 800, 800]);
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
