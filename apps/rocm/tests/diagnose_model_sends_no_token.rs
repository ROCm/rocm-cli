// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm diagnose --model owner/repo:tag` reads a GGUF header from the hub and
//! says it sends no token. That is a promise that something does not happen, so
//! it is checked on the wire: with every Hugging Face token variable set in the
//! command's environment, the listing, the `HEAD` and the range requests that
//! reach a stand-in hub carry no `Authorization` and no `Cookie` header.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

/// A small but complete GGUF header: the shape keys an estimate reads and one
/// tensor record, so the read ends in a sized estimate. 2 layers of 4 KV heads,
/// key and value width 64 each: 1024 cache elements, 2048 bytes at f16, per
/// token of context.
fn gguf_header() -> Vec<u8> {
    fn string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
    fn uint(out: &mut Vec<u8>, key: &str, value: u32) {
        string(out, key);
        out.extend_from_slice(&4_u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }
    let mut metadata = Vec::new();
    string(&mut metadata, "general.architecture");
    metadata.extend_from_slice(&8_u32.to_le_bytes());
    string(&mut metadata, "llama");
    for (key, value) in [
        ("llama.block_count", 2),
        ("llama.embedding_length", 256),
        ("llama.feed_forward_length", 512),
        ("llama.attention.head_count", 4),
        ("llama.attention.head_count_kv", 4),
        ("llama.attention.key_length", 64),
        ("llama.attention.value_length", 64),
    ] {
        uint(&mut metadata, key, value);
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes()); // tensors
    bytes.extend_from_slice(&8_u64.to_le_bytes()); // metadata entries
    bytes.extend(metadata);
    string(&mut bytes, "blk.0.attn_q.weight");
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&256_u64.to_le_bytes());
    bytes.extend_from_slice(&256_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes()); // f32
    bytes.extend_from_slice(&0_u64.to_le_bytes()); // offset
    bytes
}

/// The KV cache the header above implies at the default 4096 tokens, f16.
const EXPECTED_KV_CACHE_BYTES: u64 = 2 * 4 * (64 + 64) * 2 * 4096;

/// What the file is said to weigh: larger than the header, so the reader asks
/// for a range and never the whole file.
const FILE_SIZE: u64 = 1 << 20;

fn answer(mut stream: TcpStream, requests: &Mutex<Vec<String>>, header: &[u8]) {
    let mut request = Vec::new();
    let mut byte = [0_u8; 1];
    while !request.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).unwrap_or(0) == 0 {
            return;
        }
        request.push(byte[0]);
    }
    let text = String::from_utf8_lossy(&request).into_owned();
    requests.lock().unwrap().push(text.clone());
    let mut words = text.split_whitespace();
    let method = words.next().unwrap_or("");
    let path = words.next().unwrap_or("");
    let range = text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("range")
            .then(|| value.trim().strip_prefix("bytes=").map(str::to_owned))
            .flatten()
    });
    if path.starts_with("/api/models/o/r") {
        // No size in the listing, so the file's size is asked for by `HEAD`.
        let body = r#"{"siblings":[{"rfilename":"m-Q4_K_M.gguf"}]}"#;
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            body.len()
        );
    } else if method == "HEAD" {
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {FILE_SIZE}\r\nConnection: close\r\n\r\n"
        );
    } else if let Some((start, end)) = range.as_deref().and_then(|r| r.split_once('-')) {
        let start: u64 = start.parse().unwrap_or(0);
        let end: u64 = end.parse().unwrap_or(0).min(FILE_SIZE - 1);
        let body: Vec<u8> = (start..=end)
            .map(|at| {
                usize::try_from(at)
                    .ok()
                    .and_then(|at| header.get(at))
                    .copied()
                    .unwrap_or(0)
            })
            .collect();
        let _ = write!(
            stream,
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{FILE_SIZE}\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(&body);
    } else {
        let _ = write!(
            stream,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
    }
}

#[test]
fn a_hub_header_read_sends_no_token_even_when_one_is_set() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let hub = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    thread::spawn(move || {
        let header = gguf_header();
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            answer(stream, &seen, &header);
        }
    });

    let home = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rocm"))
        .args(["diagnose", "--model", "o/r:Q4_K_M", "--json"])
        .env("HF_ENDPOINT", &hub)
        .env("HF_TOKEN", "hf_secret_token")
        .env("HUGGING_FACE_HUB_TOKEN", "hf_secret_token")
        .env("ROCM_CLI_HUGGINGFACE_TOKEN", "hf_secret_token")
        .env("ROCM_CLI_CONFIG_DIR", home.path().join("config"))
        .env("ROCM_CLI_DATA_DIR", home.path().join("data"))
        .env("ROCM_CLI_CACHE_DIR", home.path().join("cache"))
        .stdin(Stdio::null())
        .output()
        .expect("run rocm diagnose --model");

    let requests = requests.lock().unwrap().clone();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let first_line = |request: &String| request.lines().next().unwrap_or("").to_owned();
    let lines: Vec<String> = requests.iter().map(first_line).collect();
    // Each kind of request reached the hub, so the check below is not vacuous.
    for (kind, found) in [
        (
            "the listing",
            lines.iter().any(|l| l.starts_with("GET /api/models/o/r")),
        ),
        (
            "the HEAD",
            lines.iter().any(|l| l.starts_with("HEAD /o/r/resolve/")),
        ),
        (
            "a range request",
            requests.iter().any(|request| {
                request.starts_with("GET /o/r/resolve/")
                    && request.to_ascii_lowercase().contains("\r\nrange: bytes=")
            }),
        ),
    ] {
        assert!(
            found,
            "{kind} never reached the hub: {lines:?}\nstdout: {stdout}\nstderr: {stderr}"
        );
    }
    for request in &requests {
        let lower = request.to_ascii_lowercase();
        assert!(
            !lower.contains("\r\nauthorization:") && !lower.contains("\r\ncookie:"),
            "a credential went to the hub:\n{request}"
        );
        assert!(!request.contains("hf_secret_token"), "{request}");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("\r\nuser-agent: rocm-cli"),
            "{request}"
        );
    }

    // And the read is the estimate's: the answer is sized from that header.
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("{error}\nstdout: {stdout}\nstderr: {stderr}"));
    let estimate = &report["model"]["estimate"];
    assert_eq!(
        estimate["source"].as_str(),
        Some("o/r: m-Q4_K_M.gguf"),
        "{report:#}"
    );
    assert_eq!(
        estimate["architecture"].as_str(),
        Some("llama"),
        "{report:#}"
    );
    assert_eq!(estimate["layers"].as_u64(), Some(2), "{report:#}");
    assert_eq!(
        estimate["kv_cache_bytes"].as_u64(),
        Some(EXPECTED_KV_CACHE_BYTES),
        "{report:#}"
    );
    assert!(estimate["total_bytes"].as_u64().is_some(), "{report:#}");
}
