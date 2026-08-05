// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared plumbing for the suite's `axum` test servers: bind an ephemeral
//! loopback port, serve a [`Router`] on it, and shut down on drop.
//!
//! Both test servers — [`crate::mock_server`]'s inference endpoint and
//! [`crate::loopback_http`]'s install download server — differ only in their
//! routes, so the bind/serve/shutdown lifecycle lives here once rather than
//! being re-derived (and re-diverged) per server.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::mpsc;
use std::thread::JoinHandle;

use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use url::Url;

/// A running server: where it is reachable, plus the shutdown signal and (when
/// the server owns a thread) the thread to join. Shuts the server down on drop.
pub struct ServerHandle {
    url: Url,
    port: u16,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl ServerHandle {
    pub const fn port(&self) -> u16 {
        self.port
    }

    /// The server's root URL. Derive paths from it with [`Url::join`] rather
    /// than string concatenation, which has to get separators and escaping
    /// right by hand.
    pub const fn url(&self) -> &Url {
        &self.url
    }

    /// The root URL as a string with no trailing slash, for handing to a
    /// process that appends its own `/<path>` — notably the installer, whose
    /// `ROCM_CLI_DOWNLOAD_BASE` is used as `<base>/<file>`.
    pub fn base_url(&self) -> String {
        self.url.as_str().trim_end_matches('/').to_string()
    }
}

impl std::fmt::Debug for ServerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerHandle")
            .field("url", &self.url.as_str())
            .finish_non_exhaustive()
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        // Dropping the sender alone would also trigger the graceful shutdown;
        // sending is the explicit form, and is required before joining a
        // server that owns its own thread.
        if let Some(shutdown) = self.shutdown.take() {
            shutdown.send(()).ok();
        }
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }
}

/// Serve `app` on an ephemeral loopback port as a task on the *caller's*
/// runtime.
///
/// For servers started from async code that then yields normally (`.await`s)
/// while the server is in use.
pub async fn spawn(app: Router) -> ServerHandle {
    let listener = bind().await;
    let addr = local_addr(&listener);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    tokio::spawn(serve(listener, app, shutdown_rx));
    ServerHandle {
        url: root_url(addr),
        port: addr.port(),
        shutdown: Some(shutdown_tx),
        thread: None,
    }
}

/// Serve `app` on an ephemeral loopback port using a runtime on a *dedicated
/// thread*, and block until it is bound.
///
/// For servers whose caller then blocks its own thread — e.g. on an installer
/// subprocess — which would otherwise stall a server sharing that runtime.
pub fn spawn_on_own_thread(app: Router) -> ServerHandle {
    let (addr_tx, addr_rx) = mpsc::channel();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let thread = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build test server runtime");
        runtime.block_on(async move {
            let listener = bind().await;
            // A dropped receiver means the starter gave up before the bind
            // completed; there is nothing left to serve.
            if addr_tx.send(local_addr(&listener)).is_err() {
                return;
            }
            serve(listener, app, shutdown_rx).await;
        });
    });

    let addr = addr_rx.recv().expect("test server failed to bind");
    ServerHandle {
        url: root_url(addr),
        port: addr.port(),
        shutdown: Some(shutdown_tx),
        thread: Some(thread),
    }
}

/// The `http://<ip>:<port>/` root for a bound address, assembled through
/// `Url`'s setters so host and port are never formatted into a string by hand.
fn root_url(addr: SocketAddr) -> Url {
    let mut url = Url::parse("http://host").expect("static base URL is valid");
    url.set_ip_host(addr.ip())
        .expect("a socket address is a valid URL host");
    url.set_port(Some(addr.port()))
        .expect("an http URL accepts a port");
    url
}

/// The `http://127.0.0.1:<port>/` root, for callers that know a port but hold
/// no [`ServerHandle`] — e.g. building a service record for a planted mock.
pub fn loopback_url(port: u16) -> Url {
    root_url(SocketAddr::from((Ipv4Addr::LOCALHOST, port)))
}

async fn bind() -> TcpListener {
    TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind test server listener")
}

fn local_addr(listener: &TcpListener) -> SocketAddr {
    listener
        .local_addr()
        .expect("no local addr for test server listener")
}

async fn serve(listener: TcpListener, app: Router, shutdown: oneshot::Receiver<()>) {
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            shutdown.await.ok();
        })
        .await
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_url_is_the_server_root() {
        assert_eq!(loopback_url(8080).as_str(), "http://127.0.0.1:8080/");
    }

    #[test]
    fn joining_a_path_yields_one_separator() {
        let url = loopback_url(8080).join("v1").expect("valid relative path");
        assert_eq!(url.as_str(), "http://127.0.0.1:8080/v1");
    }

    #[test]
    fn base_url_string_has_no_trailing_slash() {
        // The installer appends `/<file>` to this, so a trailing slash here
        // would send it after a doubled separator.
        let handle = ServerHandle {
            url: loopback_url(8080),
            port: 8080,
            shutdown: None,
            thread: None,
        };
        assert_eq!(handle.base_url(), "http://127.0.0.1:8080");
    }
}
