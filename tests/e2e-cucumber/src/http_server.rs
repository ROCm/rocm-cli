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

use std::net::SocketAddr;
use std::sync::mpsc;
use std::thread::JoinHandle;

use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// A running server: its bound address, plus the shutdown signal and (when the
/// server owns a thread) the thread to join. Shuts the server down on drop.
pub struct ServerHandle {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl ServerHandle {
    pub const fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

impl std::fmt::Debug for ServerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerHandle")
            .field("addr", &self.addr)
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
        addr,
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
        addr,
        shutdown: Some(shutdown_tx),
        thread: Some(thread),
    }
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
