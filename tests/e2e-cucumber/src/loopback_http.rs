// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Loopback HTTP file server for the install lifecycle E2E scenario.
//!
//! The scenario points the real installer at this server to exercise its
//! native HTTP download path, so the server's only job is to serve one
//! directory correctly and unremarkably.
//!
//! It is `tower_http`'s `ServeDir` on the suite's shared server plumbing
//! ([`crate::http_server`]), rather than a hand-rolled `TcpListener` loop. A
//! hand-rolled server has to reimplement HTTP/1.x framing, socket-mode
//! handling and path-traversal defence. The first two were real defects in the
//! server this replaced: a request split across TCP segments parsed as `GET /`,
//! and `accept()` on Windows returns a socket that inherits the listener's
//! non-blocking mode, so a read failed outright whenever the request bytes had
//! not landed yet. Traversal was never one — the removed `safe_join` handled
//! it — but it is a third thing a test server should not be maintaining.
//!
//! One deliberate tradeoff: `ServeDir` rejects `..`, drive prefixes and root
//! components but does not canonicalise, so unlike `safe_join` it follows
//! symlinks out of the served root. The root is a test-created `TempDir` with
//! no attacker-controlled symlinks, so this does not matter here.

use std::path::Path;

use axum::Router;
use tower_http::services::ServeDir;

use crate::http_server::{self, ServerHandle};

/// A minimal loopback HTTP file server serving one directory, for the Windows
/// native-HTTP install scenario. Shuts down on drop.
#[derive(Debug)]
pub struct LoopbackServer {
    server: ServerHandle,
}

impl LoopbackServer {
    /// Bind an ephemeral loopback port and serve `root` until dropped.
    ///
    /// Blocks until the port is bound, so [`Self::base_url`] is immediately
    /// usable and a client can never race the bind. The server runs on its own
    /// thread because the scenario step that starts it then blocks on the
    /// installer subprocess.
    pub fn start(root: &Path) -> Self {
        let app = Router::new().fallback_service(ServeDir::new(root));
        Self {
            server: http_server::spawn_on_own_thread(app),
        }
    }

    /// The served root, without a trailing slash: the installer appends
    /// `/<file>` to `ROCM_CLI_DOWNLOAD_BASE` itself.
    pub fn base_url(&self) -> String {
        self.server.base_url()
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use super::*;

    /// GET `path` from `server`, resolved against its root URL.
    async fn get(server: &LoopbackServer, path: &str) -> reqwest::Response {
        let url = server
            .server
            .url()
            .join(path)
            .unwrap_or_else(|e| panic!("{path} is not a valid relative URL: {e}"));
        reqwest::get(url)
            .await
            .unwrap_or_else(|e| panic!("request for {path} failed: {e}"))
    }

    /// Send `GET <target>` verbatim over loopback and return the whole
    /// response as text, with no client library normalising the target.
    async fn raw_get(server: &LoopbackServer, target: &str) -> String {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let mut socket =
            tokio::net::TcpStream::connect((Ipv4Addr::LOCALHOST, server.server.port()))
                .await
                .expect("failed to connect to test server");
        let request =
            format!("GET {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        socket
            .write_all(request.as_bytes())
            .await
            .expect("failed to send raw request");

        let mut response = Vec::new();
        socket
            .read_to_end(&mut response)
            .await
            .expect("failed to read raw response");
        String::from_utf8_lossy(&response).into_owned()
    }

    /// Serve a directory of the given `(name, contents)` files.
    fn serve(files: &[(&str, &[u8])]) -> (tempfile::TempDir, LoopbackServer) {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        for (name, contents) in files {
            std::fs::write(dir.path().join(name), contents).expect("failed to write served file");
        }
        let server = LoopbackServer::start(dir.path());
        (dir, server)
    }

    #[tokio::test]
    async fn serves_a_bundle_byte_for_byte() {
        // A zip-sized payload, so the response is not a single small write.
        let bundle: Vec<u8> = (0..256 * 1024).map(|i| (i % 251) as u8).collect();
        let (_dir, server) = serve(&[("rocm-cli-windows-amd64.zip", &bundle)]);

        let response = get(&server, "rocm-cli-windows-amd64.zip").await;
        assert!(response.status().is_success());
        assert_eq!(
            response.bytes().await.expect("no body").as_ref(),
            &bundle[..]
        );
    }

    #[tokio::test]
    async fn serves_back_to_back_downloads() {
        // The sequence the install scenario drives: the bundle, then its
        // signature/checksum sidecars. This checks that sequence is served,
        // not that the Windows flake is gone — it injects no timing pressure,
        // and the flake needed a request whose bytes had not yet landed. What
        // rules that class out is the move to async I/O, which never surfaces
        // `WouldBlock` to application code at all.
        let (_dir, server) = serve(&[
            ("bundle.zip", b"bundle".as_slice()),
            ("bundle.zip.sig", b"signature".as_slice()),
            ("bundle.zip.sha256", b"checksum".as_slice()),
        ]);

        for (name, expected) in [
            ("bundle.zip", "bundle"),
            ("bundle.zip.sig", "signature"),
            ("bundle.zip.sha256", "checksum"),
            ("bundle.zip", "bundle"),
        ] {
            let response = get(&server, name).await;
            assert!(response.status().is_success(), "{name} was not served");
            assert_eq!(response.text().await.expect("no body"), expected);
        }
    }

    #[tokio::test]
    async fn missing_file_is_a_404() {
        let (_dir, server) = serve(&[("bundle.zip", b"bundle".as_slice())]);

        let response = get(&server, "absent.zip").await;
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_raw_request_for_a_served_file_is_served() {
        // Control for the traversal test below: proves a hand-written request
        // reaches the server and is answered, so the 404s there are refusals
        // rather than a malformed request the server never understood.
        let (_dir, server) = serve(&[("bundle.zip", b"bundle".as_slice())]);

        let response = raw_get(&server, "/bundle.zip").await;
        assert!(
            response.starts_with("HTTP/1.1 200 ") && response.ends_with("bundle"),
            "raw request was not served: {response}"
        );
    }

    #[tokio::test]
    async fn traversal_outside_the_served_directory_is_refused() {
        let outside = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(outside.path().join("secret.txt"), b"secret")
            .expect("failed to write secret");
        let served = outside.path().join("served");
        std::fs::create_dir(&served).expect("failed to create served dir");
        let server = LoopbackServer::start(&served);

        // Sent over a raw socket rather than through a client: any URL type
        // resolves dot segments per RFC 3986 before the request goes out, so a
        // client could only ever ask for `/secret.txt` — which 404s for the
        // mundane reason that it is not in the served directory, whatever the
        // server's traversal defence. Only an un-normalised target reaches it.
        for target in [
            "/../secret.txt",
            "/..%2Fsecret.txt",
            "/%2e%2e/secret.txt",
            "/..\\secret.txt",
            "/served/../secret.txt",
        ] {
            let response = raw_get(&server, target).await;
            assert!(
                response.starts_with("HTTP/1.1 404 "),
                "{target} was not refused: {response}"
            );
            assert!(
                !response.contains("secret"),
                "{target} served a file outside the root: {response}"
            );
        }
    }
}
