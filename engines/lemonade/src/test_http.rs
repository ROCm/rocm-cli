// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A test-owned loopback HTTP fixture shared by the engine's tests.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// One request the fixture received.
#[derive(Debug, Clone)]
pub(crate) struct Recorded {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) body: String,
}

/// What the fixture does with a request: the JSON body to answer `200` with, and
/// whether that was the last request it should serve.
pub(crate) struct Reply {
    pub(crate) body: String,
    pub(crate) last: bool,
}

/// Serve loopback HTTP until `respond` marks a reply as the last one or `deadline`
/// passes. Returns the port and a handle yielding every request received.
pub(crate) fn spawn_fixture(
    deadline: Duration,
    respond: impl Fn(&Recorded) -> Reply + Send + 'static,
) -> (u16, JoinHandle<Vec<Recorded>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();
    let handle = std::thread::spawn(move || {
        let mut seen = Vec::new();
        let stop_at = Instant::now() + deadline;
        while Instant::now() < stop_at {
            let Ok((mut stream, _)) = listener.accept() else {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            };
            // An accepted socket can inherit the listener's non-blocking mode
            // (Windows does), which would make the first read fail with WouldBlock.
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let Some(recorded) = read_request(&mut stream) else {
                continue;
            };
            let reply = respond(&recorded);
            seen.push(recorded);
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.body.len(),
                reply.body
            );
            if reply.last {
                break;
            }
        }
        seen
    });
    (port, handle)
}

fn read_request(stream: &mut impl Read) -> Option<Recorded> {
    let mut bytes = Vec::new();
    let mut buffer = [0; 1024];
    let (header_end, content_len) = loop {
        let count = stream.read(&mut buffer).ok().filter(|count| *count > 0)?;
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let len = headers
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            break (end + 4, len);
        }
    };
    while bytes.len() < header_end + content_len {
        let count = stream.read(&mut buffer).ok().filter(|count| *count > 0)?;
        bytes.extend_from_slice(&buffer[..count]);
    }
    let head = String::from_utf8_lossy(&bytes[..header_end]).into_owned();
    let mut request_line = head.lines().next()?.split_whitespace();
    Some(Recorded {
        method: request_line.next()?.to_owned(),
        path: request_line.next()?.to_owned(),
        body: String::from_utf8_lossy(&bytes[header_end..header_end + content_len]).into_owned(),
    })
}

/// Fake `lemond`/`lemonade` executables for the engine's tests. The fake CLI appends
/// one line per invocation to `calls.txt` beside it (`cat` it to see the exact argv),
/// and answers `pull` by dropping a GGUF into the hub cache under `cache/`.
#[cfg(target_os = "linux")]
pub(crate) mod fake_runtime {
    use crate::install::LemonadeInstallManifest;
    use std::path::Path;

    pub(crate) fn write_executable(path: &Path, script: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        rocm_core::fake_executable::write_fake_executable(path, script).unwrap();
    }

    /// A manifest whose `lemond` idles until killed and whose `lemonade` records its
    /// argv. `pull_exit` is the exit status of `lemonade pull`; on success the pull
    /// also creates `pulled_gguf` (relative to `dir/cache/huggingface/hub`), if given.
    pub(crate) fn manifest(
        dir: &Path,
        pull_exit: i32,
        pulled_gguf: Option<&str>,
    ) -> LemonadeInstallManifest {
        write_executable(&dir.join("lemond"), "#!/bin/sh\nexec sleep 30\n");
        let create = pulled_gguf.map_or_else(String::new, |relative| {
            let target = dir.join("cache/huggingface/hub").join(relative);
            format!(
                "mkdir -p '{}' && : > '{}'\n",
                target.parent().unwrap().display(),
                target.display()
            )
        });
        write_executable(
            &dir.join("lemonade"),
            &format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$(dirname \"$0\")/calls.txt\"\nfor arg in \"$@\"; do\n  if [ \"$arg\" = pull ]; then\n    {create}    exit {pull_exit}\n  fi\ndone\nexit 0\n"
            ),
        );
        LemonadeInstallManifest {
            env_id: "test".to_owned(),
            version: "test".to_owned(),
            runtime_dir: dir.join("runtime"),
            lemond: dir.join("lemond"),
            lemonade: dir.join("lemonade"),
            backend_recipe: crate::LLAMACPP_RECIPE.to_owned(),
            backend_name: "vulkan".to_owned(),
            installed_at_unix_ms: 0,
        }
    }

    /// The `lemonade` invocations so far, one argv per line.
    pub(crate) fn calls(dir: &Path) -> Vec<String> {
        std::fs::read_to_string(dir.join("calls.txt"))
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }
}
