// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Loopback HTTP file server for the install lifecycle E2E scenario.
//!
//! The scenario points the real installer at this server to exercise its
//! native HTTP download path, so the server's only job is to serve one
//! directory correctly and unremarkably.
//!
//! It is built from `axum` + `tower_http`'s `ServeDir` — the same stack as
//! [`crate::mock_server`] — rather than a hand-rolled `TcpListener` loop. A
//! hand-rolled server has to reimplement HTTP/1.x framing (a request may
//! arrive split across any number of TCP segments), socket-mode handling
//! (`accept()` on Windows returns a socket that inherits the listener's
//! non-blocking mode, so reads fail outright whenever the request bytes have
//! not landed yet), and path-traversal defence. Each of those was a real
//! defect here, and each is an HTTP library's job — hence this one.
//!
//! The server owns a dedicated runtime on its own thread, so it keeps
//! answering while the calling step blocks on the installer subprocess.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::JoinHandle;

use axum::Router;
use tokio::sync::oneshot;
use tower_http::services::ServeDir;

/// A minimal loopback HTTP file server serving one directory, for the Windows
/// native-HTTP install scenario. Shuts down on drop.
pub struct LoopbackServer {
    port: u16,
    shutdown: Option<oneshot::Sender<()>>,
    handle: Option<JoinHandle<()>>,
}

impl LoopbackServer {
    /// Bind an ephemeral loopback port and serve `root` until dropped.
    ///
    /// Blocks until the port is bound, so [`Self::base_url`] is immediately
    /// usable and a client can never race the bind.
    pub fn start(root: &Path) -> Self {
        let root: PathBuf = root.to_path_buf();
        let (port_tx, port_rx) = mpsc::channel();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        // A dedicated single-thread runtime, not the caller's: the step that
        // starts this server then blocks its own thread on the installer
        // subprocess, and the server must keep serving throughout.
        let handle = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build loopback server runtime");
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("failed to bind loopback listener");
                let port = listener
                    .local_addr()
                    .expect("no local addr for loopback listener")
                    .port();
                // A receiver dropped before this point means the starter gave
                // up; nothing left to serve, so just fall through and exit.
                if port_tx.send(port).is_err() {
                    return;
                }
                let app = Router::new().fallback_service(ServeDir::new(&root));
                axum::serve(listener, app)
                    .with_graceful_shutdown(async {
                        shutdown_rx.await.ok();
                    })
                    .await
                    .ok();
            });
        });

        let port = port_rx.recv().expect("loopback server failed to bind");
        Self {
            port,
            shutdown: Some(shutdown_tx),
            handle: Some(handle),
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for LoopbackServer {
    fn drop(&mut self) {
        // Dropping the sender also triggers the graceful shutdown; sending is
        // just the explicit form of it.
        if let Some(shutdown) = self.shutdown.take() {
            shutdown.send(()).ok();
        }
        if let Some(handle) = self.handle.take() {
            handle.join().ok();
        }
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
