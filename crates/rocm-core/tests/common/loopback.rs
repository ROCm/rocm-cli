// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A loopback HTTP server for the integration tests that drive real network
//! code: it answers every request with one fixed `200 OK` until it is dropped.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

/// Serves one fixed response on an ephemeral loopback port, to any number of
/// requests, until dropped.
///
/// Dropping it never hangs: it wakes the accept loop with a connection of its
/// own before joining, so a client that never connected cannot leave the
/// server thread blocked in `accept`.
pub struct LoopbackServer {
    port: u16,
    stopping: Arc<AtomicBool>,
    served: Arc<AtomicUsize>,
    handle: Option<thread::JoinHandle<()>>,
}

impl LoopbackServer {
    pub fn start(content_type: Option<&str>, body: Vec<u8>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let stopping = Arc::new(AtomicBool::new(false));
        let served = Arc::new(AtomicUsize::new(0));
        let content_type = content_type
            .map(|value| format!("Content-Type: {value}\r\n"))
            .unwrap_or_default();
        let handle = thread::spawn({
            let stopping = Arc::clone(&stopping);
            let served = Arc::clone(&served);
            move || {
                for stream in listener.incoming() {
                    if stopping.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut stream) = stream else { continue };
                    // A client that connects and never sends cannot hold the
                    // accept loop, and with it `Drop`, forever.
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                    read_request_head(&mut stream);
                    // Counted before the response is written, so a client that
                    // has its answer always sees its request counted.
                    served.fetch_add(1, Ordering::SeqCst);
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\n{content_type}Content-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(&body);
                    let _ = stream.flush();
                }
            }
        });
        Self {
            port,
            stopping,
            served,
            handle: Some(handle),
        }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Requests answered so far. A test that expects a verdict about the
    /// response checks this is non-zero, so a request that never reached the
    /// server cannot pass as one the code under test rejected.
    pub fn requests_served(&self) -> usize {
        self.served.load(Ordering::SeqCst)
    }
}

impl Drop for LoopbackServer {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::SeqCst);
        // Wake the accept loop so it sees the flag. If even that connection
        // fails, leave the thread detached rather than block on it.
        let woken = TcpStream::connect(("127.0.0.1", self.port)).is_ok();
        if let Some(handle) = self.handle.take()
            && woken
        {
            let _ = handle.join();
        }
    }
}

/// Read the request line and headers, so the response is not written into a
/// request still being sent.
fn read_request_head(stream: &mut TcpStream) {
    let mut head = Vec::new();
    let mut chunk = [0_u8; 1024];
    while !head.windows(4).any(|window| window == b"\r\n\r\n") && head.len() < 64 * 1024 {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(read) => head.extend_from_slice(&chunk[..read]),
        }
    }
}
