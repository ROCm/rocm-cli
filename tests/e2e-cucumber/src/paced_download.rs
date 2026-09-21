// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Loopback HTTP server that serves one named file in delayed chunks.
//!
//! [`crate::loopback_http::LoopbackServer`] answers every request from
//! `ServeDir` as fast as the OS can read the file, which never gives a PTY
//! test harness a chance to observe an intermediate download-progress frame
//! from `cli_progress::AnimatedSpinner` — the whole transfer completes within
//! a single poll of the emulated screen. This server keeps `ServeDir` as the
//! fallback for every other path, but answers one specific file itself, with
//! an accurate `Content-Length` header and the body written as fixed-size
//! chunks separated by a fixed delay. `rocm-core`'s download client reads
//! `Content-Length` to compute the progress percentage and reads the body in
//! an ordinary streaming loop, so pacing here needs nothing special on the
//! client side — it behaves exactly as if a slow network served the file.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures::stream;
use tower_http::services::ServeDir;

use crate::http_server::{self, ServerHandle};

/// A loopback HTTP server that serves one named file in paced chunks.
///
/// Falls back to serving `root` normally (via `ServeDir`) for every other
/// path. Shuts down on drop, like [`crate::loopback_http::LoopbackServer`].
#[derive(Debug)]
pub struct PacedDownloadServer {
    server: ServerHandle,
}

impl PacedDownloadServer {
    /// Bind an ephemeral loopback port and serve `root` (via `ServeDir`)
    /// until dropped, except for `GET /<paced_file>`, which streams
    /// `contents` in `chunk_size`-byte pieces with `delay` between each.
    ///
    /// Blocks until the port is bound, matching `LoopbackServer::start`, so
    /// [`Self::base_url`] is immediately usable.
    pub fn start(
        root: &Path,
        paced_file: &str,
        contents: Vec<u8>,
        chunk_size: usize,
        delay: Duration,
    ) -> Self {
        let contents = Arc::new(contents);
        let route = format!("/{paced_file}");
        let app = Router::new()
            .route(
                &route,
                get(move || std::future::ready(paced_response(contents, chunk_size, delay))),
            )
            .fallback_service(ServeDir::new(root));
        Self {
            server: http_server::spawn_on_own_thread(app),
        }
    }

    /// The served root, without a trailing slash — see
    /// [`crate::loopback_http::LoopbackServer::base_url`].
    pub fn base_url(&self) -> String {
        self.server.base_url()
    }
}

/// High-entropy filler bytes for a paced-fixture payload.
///
/// A naive multiplicative-hash sequence looked pseudo-random but gzip still
/// compressed it by over 99%, collapsing a paced transfer into a single
/// unpaced chunk. A fixed seed keeps the fixture (and therefore the archive's
/// compressed size) deterministic across runs. Shared by both the TheRock
/// tarball and ComfyUI source-archive fixtures, which each need enough
/// incompressible bytes to stream in more than one paced chunk.
pub fn xorshift_payload(len: usize) -> Vec<u8> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect()
}

/// Builds a real gzip tarball and returns its bytes.
///
/// Packages the single top-level directory `build_dir.join(dir_name)` into
/// `build_dir.join(archive_name)`. Shared by the TheRock tarball and ComfyUI
/// source-archive fixtures, which each need a genuine archive for their
/// installer's real `tar` extraction to unpack once the paced download
/// completes.
pub fn build_gzip_tarball(build_dir: &Path, archive_name: &str, dir_name: &str) -> Vec<u8> {
    let archive_path = build_dir.join(archive_name);
    let status = std::process::Command::new("tar")
        .arg("-czf")
        .arg(&archive_path)
        .arg("-C")
        .arg(build_dir)
        .arg(dir_name)
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => panic!("tar exited with {status} while building {archive_name}"),
        Err(error) => panic!("tar is required to build {archive_name}: {error}"),
    }
    std::fs::read(&archive_path)
        .unwrap_or_else(|error| panic!("failed to read built archive {archive_name}: {error}"))
}

/// Stream `contents` as an HTTP response with an explicit `Content-Length`,
/// in `chunk_size`-byte pieces, sleeping `delay` before every chunk after the
/// first.
fn paced_response(contents: Arc<Vec<u8>>, chunk_size: usize, delay: Duration) -> Response {
    debug_assert!(chunk_size > 0, "chunk_size must be at least 1 byte");
    let total_len = contents.len();
    let chunk_size = chunk_size.max(1);
    let body = Body::from_stream(stream::unfold(0_usize, move |offset| {
        let contents = Arc::clone(&contents);
        async move {
            if offset >= contents.len() {
                return None;
            }
            if offset > 0 {
                tokio::time::sleep(delay).await;
            }
            let end = (offset + chunk_size).min(contents.len());
            let chunk = Bytes::copy_from_slice(&contents[offset..end]);
            Some((Ok::<_, std::io::Error>(chunk), end))
        }
    }));
    ([(header::CONTENT_LENGTH, total_len.to_string())], body).into_response()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// GET `path` from `server`, resolved against its root URL.
    async fn get(server: &PacedDownloadServer, path: &str) -> reqwest::Response {
        let url = server
            .server
            .url()
            .join(path)
            .unwrap_or_else(|e| panic!("{path} is not a valid relative URL: {e}"));
        reqwest::get(url)
            .await
            .unwrap_or_else(|e| panic!("request for {path} failed: {e}"))
    }

    #[tokio::test]
    async fn serves_the_paced_file_byte_for_byte() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let contents: Vec<u8> = (0..10_000).map(|i| (i % 251) as u8).collect();
        let server = PacedDownloadServer::start(
            dir.path(),
            "archive.tar.gz",
            contents.clone(),
            2_000,
            Duration::from_millis(1),
        );

        let response = get(&server, "archive.tar.gz").await;
        assert!(response.status().is_success());
        assert_eq!(
            response
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok()),
            Some("10000"),
            "Content-Length must report the exact total so the client can compute a percentage"
        );
        assert_eq!(
            response.bytes().await.expect("no body").as_ref(),
            &contents[..]
        );
    }

    #[tokio::test]
    async fn pacing_delays_the_response_by_roughly_one_delay_per_chunk_boundary() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        // 3 chunks of 10 bytes: 2 chunk boundaries after the first, so the
        // full transfer should take at least 2 delays.
        let contents = vec![0_u8; 30];
        let delay = Duration::from_millis(50);
        let server =
            PacedDownloadServer::start(dir.path(), "paced.bin", contents, 10, delay);

        let started = Instant::now();
        let response = get(&server, "paced.bin").await;
        let _ = response.bytes().await.expect("no body");
        let elapsed = started.elapsed();

        assert!(
            elapsed >= delay * 2,
            "expected the paced response to take at least {:?}, took {elapsed:?} — \
             pacing did not actually delay the chunks",
            delay * 2
        );
    }

    #[tokio::test]
    async fn falls_back_to_serving_other_files_from_root() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(dir.path().join("index.html"), b"<html></html>")
            .expect("failed to write fallback file");
        let server =
            PacedDownloadServer::start(dir.path(), "archive.tar.gz", vec![1, 2, 3], 1, Duration::ZERO);

        let response = get(&server, "index.html").await;
        assert!(response.status().is_success());
        assert_eq!(
            response.text().await.expect("no body"),
            "<html></html>"
        );
    }

    #[tokio::test]
    async fn missing_file_is_a_404() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let server =
            PacedDownloadServer::start(dir.path(), "archive.tar.gz", vec![1, 2, 3], 1, Duration::ZERO);

        let response = get(&server, "absent.zip").await;
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    }
}
