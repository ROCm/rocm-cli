// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The API docs link printed for a running model server.
//!
//! The background daemon serves the docs (`rocmd`'s `api_docs` module). A link
//! is only printed once the docs server has answered for that very server, so
//! the CLI never points at a page that is not there: docs turned off, a taken
//! port, or a daemon that has not come up yet all print no link.

use rocm_core::{AppPaths, RocmCliConfig};
use std::thread;
use std::time::{Duration, Instant};

/// How long one probe of the docs server may take. It is on loopback, so a live
/// server answers in milliseconds and a missing one refuses the connection at once.
const PROBE_TIMEOUT: Duration = Duration::from_millis(500);

/// How long `rocm serve` waits for a daemon that is still starting. A cold
/// daemon has to parse its config and bind before it answers.
const STARTING_HELPER_WAIT: Duration = Duration::from_secs(5);

const RETRY_INTERVAL: Duration = Duration::from_millis(200);

/// How long `rocm serve` should wait for the docs: only a helper that is still
/// starting is worth waiting for. One that was already running answers at once
/// or never will — it may predate API docs, or its port may be taken — and
/// waiting on it would stall every serve.
pub(crate) const fn serve_docs_wait(helper_starting: bool) -> Duration {
    if helper_starting {
        STARTING_HELPER_WAIT
    } else {
        Duration::ZERO
    }
}

/// The docs link for `service_id`, if the docs server answers for it within
/// `wait` (one attempt when `wait` is zero). `on_wait` runs between attempts,
/// so a caller can keep its progress indicator moving.
pub(crate) fn service_docs_link(
    paths: &AppPaths,
    service_id: &str,
    wait: Duration,
    on_wait: &mut dyn FnMut(),
) -> Option<String> {
    let docs = RocmCliConfig::load(paths).ok()?.api_docs;
    docs.served_port()?;
    let base_url = docs.base_url();
    let deadline = Instant::now() + wait;
    loop {
        if docs_answer_for(&base_url, service_id) {
            return Some(docs.service_docs_url(service_id));
        }
        if Instant::now() + RETRY_INTERVAL > deadline {
            return None;
        }
        on_wait();
        thread::sleep(RETRY_INTERVAL);
    }
}

/// Whether the server on the docs port answers with the description of this
/// very service. Checking what it answers, not just that it answers, keeps a
/// link from naming some other program that happens to hold the port.
fn docs_answer_for(base_url: &str, service_id: &str) -> bool {
    rocm_core::http_get_text(
        base_url,
        &format!("/services/{service_id}/openapi.json"),
        PROBE_TIMEOUT,
    )
    .ok()
    .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
    .is_some_and(|openapi| {
        openapi["servers"][0]["url"].as_str() == Some(&format!("/services/{service_id}/api"))
    })
}

/// [`service_docs_link`] for each of `service_ids`, without waiting. Asks the
/// docs server once whether it is up, so listing many servers while it is down
/// costs one refused connection rather than one per server.
pub(crate) fn service_docs_links<'a>(
    paths: &AppPaths,
    service_ids: impl IntoIterator<Item = &'a str>,
) -> Vec<(String, String)> {
    let Ok(config) = RocmCliConfig::load(paths) else {
        return Vec::new();
    };
    if config.api_docs.served_port().is_none()
        || rocm_core::http_get_text(&config.api_docs.base_url(), "/health", PROBE_TIMEOUT).is_err()
    {
        return Vec::new();
    }
    service_ids
        .into_iter()
        .filter_map(|service_id| {
            service_docs_link(paths, service_id, Duration::ZERO, &mut || {})
                .map(|link| (service_id.to_owned(), link))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_paths;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A stand-in docs server answering every request with `status` and `body`
    /// until the returned flag is set.
    fn stand_in_docs_server(status: &'static str, body: String) -> (u16, Arc<AtomicBool>) {
        stand_in_docs_server_after(0, status, body)
    }

    /// As [`stand_in_docs_server`], but the first `not_yet` requests are
    /// answered 503, as by a docs server still coming up.
    fn stand_in_docs_server_after(
        not_yet: usize,
        status: &'static str,
        body: String,
    ) -> (u16, Arc<AtomicBool>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        thread::spawn(move || {
            let mut answered = 0;
            while !stop_flag.load(Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                };
                stream.set_nonblocking(false).ok();
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut request = [0_u8; 1024];
                let _ = stream.read(&mut request);
                let (status, body) = if answered < not_yet {
                    ("503 Service Unavailable", "{}")
                } else {
                    (status, body.as_str())
                };
                answered += 1;
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (port, stop)
    }

    fn closed_port() -> u16 {
        TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    fn paths_with_docs(name: &str, enabled: bool, port: u16) -> (std::path::PathBuf, AppPaths) {
        let (root, paths) = test_paths(name);
        paths.ensure().unwrap();
        let mut config = RocmCliConfig::default();
        config.api_docs.enabled = enabled;
        config.api_docs.port = port;
        config.save(&paths).unwrap();
        (root, paths)
    }

    fn openapi_for(service_id: &str) -> String {
        serde_json::json!({ "servers": [{ "url": format!("/services/{service_id}/api") }] })
            .to_string()
    }

    #[test]
    fn a_link_is_printed_when_the_docs_describe_this_server() {
        let (port, stop) = stand_in_docs_server("200 OK", openapi_for("svc-1"));
        let (root, paths) = paths_with_docs("docs-link-answers", true, port);
        assert_eq!(
            service_docs_link(&paths, "svc-1", Duration::ZERO, &mut || {}),
            Some(format!("http://127.0.0.1:{port}/services/svc-1/docs/"))
        );
        assert_eq!(
            service_docs_links(&paths, ["svc-1"]),
            vec![(
                "svc-1".to_owned(),
                format!("http://127.0.0.1:{port}/services/svc-1/docs/")
            )]
        );
        stop.store(true, Ordering::Relaxed);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn no_link_when_the_docs_do_not_know_this_server() {
        let (port, stop) = stand_in_docs_server("404 Not Found", "{}".to_owned());
        let (root, paths) = paths_with_docs("docs-link-404", true, port);
        assert_eq!(
            service_docs_link(&paths, "svc-1", Duration::ZERO, &mut || {}),
            None
        );
        assert!(service_docs_links(&paths, ["svc-1"]).is_empty());
        stop.store(true, Ordering::Relaxed);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn no_link_when_another_program_holds_the_docs_port() {
        let (port, stop) = stand_in_docs_server("200 OK", openapi_for("some-other-server"));
        let (root, paths) = paths_with_docs("docs-link-squatter", true, port);
        assert_eq!(
            service_docs_link(&paths, "svc-1", Duration::ZERO, &mut || {}),
            None
        );
        stop.store(true, Ordering::Relaxed);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn no_link_when_nothing_listens_on_the_docs_port() {
        let (root, paths) = paths_with_docs("docs-link-closed", true, closed_port());
        let started = Instant::now();
        assert_eq!(
            service_docs_link(&paths, "svc-1", Duration::ZERO, &mut || {}),
            None
        );
        assert!(service_docs_links(&paths, ["svc-1", "svc-2"]).is_empty());
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "a docs server that is not there must not stall the CLI"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn no_link_when_docs_are_turned_off_even_if_the_port_answers() {
        let (port, stop) = stand_in_docs_server("200 OK", openapi_for("svc-1"));
        let (root, paths) = paths_with_docs("docs-link-disabled", false, port);
        assert_eq!(
            service_docs_link(&paths, "svc-1", Duration::ZERO, &mut || {}),
            None
        );
        assert!(service_docs_links(&paths, ["svc-1"]).is_empty());
        stop.store(true, Ordering::Relaxed);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn no_link_when_the_docs_port_is_zero() {
        let (root, paths) = paths_with_docs("docs-link-port-zero", true, 0);
        assert_eq!(
            service_docs_link(&paths, "svc-1", Duration::ZERO, &mut || {}),
            None
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn a_link_waits_for_a_starting_helper_but_not_for_a_running_one() {
        assert_eq!(serve_docs_wait(false), Duration::ZERO);
        assert!(serve_docs_wait(true) >= Duration::from_secs(1));

        // A helper that comes up partway through the wait still gets its link,
        // and the caller's progress indicator is ticked while it waits.
        let (port, stop) = stand_in_docs_server_after(2, "200 OK", openapi_for("svc-1"));
        let (root, paths) = paths_with_docs("docs-link-late", true, port);
        let mut ticks = 0;
        assert_eq!(
            service_docs_link(&paths, "svc-1", serve_docs_wait(true), &mut || ticks += 1),
            Some(format!("http://127.0.0.1:{port}/services/svc-1/docs/"))
        );
        assert_eq!(ticks, 2, "one tick per retry");
        stop.store(true, Ordering::Relaxed);
        std::fs::remove_dir_all(root).ok();
    }
}
