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
//! hand-rolled server has to reimplement HTTP/1.x framing (a request may
//! arrive split across any number of TCP segments), socket-mode handling
//! (`accept()` on Windows returns a socket that inherits the listener's
//! non-blocking mode, so reads fail outright whenever the request bytes have
//! not landed yet), and path-traversal defence. Each of those was a real
//! defect here, and each is an HTTP library's job — hence this one.

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

    pub fn base_url(&self) -> String {
        self.server.base_url()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let response = reqwest::get(format!("{}/rocm-cli-windows-amd64.zip", server.base_url()))
            .await
            .expect("request failed");
        assert!(response.status().is_success());
        assert_eq!(
            response.bytes().await.expect("no body").as_ref(),
            &bundle[..]
        );
    }

    #[tokio::test]
    async fn serves_back_to_back_downloads() {
        // Regression test: the install scenario fetches the bundle and then
        // its signature/checksum sidecars. The hand-rolled server this
        // replaced served the first request and then failed a later one
        // whenever its bytes had not already arrived — the timing-dependent
        // Windows flake.
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
            let response = reqwest::get(format!("{}/{name}", server.base_url()))
                .await
                .unwrap_or_else(|e| panic!("request for {name} failed: {e}"));
            assert!(response.status().is_success(), "{name} was not served");
            assert_eq!(response.text().await.expect("no body"), expected);
        }
    }

    #[tokio::test]
    async fn missing_file_is_a_404() {
        let (_dir, server) = serve(&[("bundle.zip", b"bundle".as_slice())]);

        let response = reqwest::get(format!("{}/absent.zip", server.base_url()))
            .await
            .expect("request failed");
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn traversal_outside_the_served_directory_is_refused() {
        let outside = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(outside.path().join("secret.txt"), b"secret")
            .expect("failed to write secret");
        let served = outside.path().join("served");
        std::fs::create_dir(&served).expect("failed to create served dir");
        let server = LoopbackServer::start(&served);

        let response = reqwest::get(format!("{}/../secret.txt", server.base_url()))
            .await
            .expect("request failed");
        assert_ne!(
            response.text().await.unwrap_or_default(),
            "secret",
            "server served a file outside its root"
        );
    }
}
