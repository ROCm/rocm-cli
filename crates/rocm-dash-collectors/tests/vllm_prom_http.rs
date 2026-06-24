// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! End-to-end test for VllmPrometheusCollector: spins up a tiny HTTP server
//! on a random local port, serves a canned `/metrics` payload, asserts that
//! `fetch_async` returns the expected parsed sample.

use std::time::Duration;

use rocm_dash_collectors::vllm_prom::VllmPrometheusCollector;
use rocm_dash_core::traits::DiscoveredService;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const PAYLOAD: &str = "\
# HELP vllm:num_requests_running running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model=\"x\"} 17
# HELP vllm:num_requests_waiting waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model=\"x\"} 4
# HELP vllm:gpu_cache_usage_perc kv.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model=\"x\"} 0.873
";

#[tokio::test]
async fn fetch_async_against_local_mock_server() {
    // Bind on an OS-assigned port so we don't collide with anything.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    // Serve exactly one request, then exit.
    let server = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 1024];
        let _ = sock.read(&mut buf).await;
        let body = PAYLOAD;
        let resp = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/plain; version=0.0.4\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        let _ = sock.write_all(resp.as_bytes()).await;
        let _ = sock.shutdown().await;
    });

    let collector = VllmPrometheusCollector::new("127.0.0.1", Duration::from_secs(2));
    let svc = DiscoveredService {
        container_id: "test".into(),
        port: Some(port),
        ..Default::default()
    };
    let sample = collector.fetch_async(&svc).await.expect("scrape ok");
    assert_eq!(sample.running_reqs, Some(17));
    assert_eq!(sample.waiting_reqs, Some(4));
    let kv = sample.kv_cache_usage_pct.expect("kv");
    assert!((kv - 87.3).abs() < 0.1, "kv was {kv}");

    server.await.unwrap();
}

#[tokio::test]
async fn fetch_async_propagates_non_200() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    let _server = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 1024];
        let _ = sock.read(&mut buf).await;
        let resp = "HTTP/1.1 503 Service Unavailable\r\n\
                    Content-Length: 0\r\n\
                    Connection: close\r\n\r\n";
        let _ = sock.write_all(resp.as_bytes()).await;
        let _ = sock.shutdown().await;
    });

    let collector = VllmPrometheusCollector::new("127.0.0.1", Duration::from_secs(2));
    let svc = DiscoveredService {
        container_id: "test".into(),
        port: Some(port),
        ..Default::default()
    };
    let r = collector.fetch_async(&svc).await;
    assert!(r.is_err(), "expected non-200 to be Err, got {r:?}");
}
