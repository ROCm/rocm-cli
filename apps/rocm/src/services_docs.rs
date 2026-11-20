// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm services docs [<id>]`: browsable API docs for the running model
//! servers, served on loopback by this process until Ctrl+C.
//!
//! The docs server itself is `rocmd`'s [`rocmd::start_api_docs_server`], run
//! in-process. It runs only while this command does: nothing is left behind
//! when it stops, and the model servers are not touched. It is given the
//! loader `rocm services list` uses, so the docs list the same servers as
//! running that the list does, and a server that stops or crashes while the
//! docs are open drops off them.
//!
//! [`docs_hint`] is the line `rocm serve` and `rocm services list` print for a
//! running server, naming this command; the `engine docs` line beside it is
//! [`ManagedServiceRecord::engine_docs_url`], the docs an engine serves itself
//! (vLLM's `/docs`).

use std::io::Write as _;

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ManagedServiceRecord};

use crate::managed_service_status::load_managed_services;
use crate::services_text::validate_service_id;

/// The command that serves `service_id`'s API docs, as printed after a
/// running server's endpoint. The same string is what scenario
/// `service-docs-01` runs, so the hint and the command cannot drift apart.
pub(crate) fn docs_hint(service_id: &str) -> String {
    format!(
        "rocm services docs {}",
        crate::shell_quote::shell_quote(service_id)
    )
}

/// The rows `rocm serve` prints under a server's endpoint: `docs`, naming
/// this command, and `engine docs` when the engine serves docs of its own.
/// The record is read for its engine and address; one that cannot be read
/// still gets the `docs` row, since the command reports that itself.
pub(crate) fn launch_docs_rows(paths: &AppPaths, service_id: &str) -> Vec<(&'static str, String)> {
    let mut rows = vec![("docs", docs_hint(service_id))];
    if let Some(engine_docs) = ManagedServiceRecord::read_manifest(paths, service_id)
        .ok()
        .as_ref()
        .and_then(ManagedServiceRecord::engine_docs_url)
    {
        rows.push(("engine docs", engine_docs));
    }
    rows
}

/// `rocm services docs [<id>]`.
pub(crate) fn run(paths: &AppPaths, service_id: Option<&str>, port: Option<u16>) -> Result<()> {
    if let Some(service_id) = service_id {
        validate_service_id(service_id)?;
    }
    let records = load_managed_services(paths)?;
    let targets = docs_targets(&records, service_id)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build the runtime that waits for Ctrl+C")?;
    // Listen for Ctrl+C before printing the link: the handler is installed
    // when the future is first polled, and a Ctrl+C before that would kill
    // the command without its closing line. One poll here installs it.
    let mut interrupted = std::pin::pin!(tokio::signal::ctrl_c());
    let early = runtime.block_on(async {
        tokio::select! {
            biased;
            result = &mut interrupted => Some(result),
            () = std::future::ready(()) => None,
        }
    });
    // Defensive: Ctrl+C has to land within this first poll, which no test
    // can aim at. A Ctrl+C after it, during startup, is caught by the handler
    // and ends the wait below once the links are printed.
    if let Some(result) = early {
        // Nothing has been served or printed, so there is nothing to report.
        result.context("failed to wait for Ctrl+C")?;
        return Ok(());
    }
    let (server, fallback_note) =
        start_docs(paths, port, rocmd::DEFAULT_API_DOCS_PORT, |paths, port| {
            rocmd::start_api_docs_server(paths, port, services_loader())
        })?;

    // On stderr, so stdout carries only the command's own output (the links
    // and how to stop), which a caller reads from a pipe.
    if let Some(note) = fallback_note {
        eprintln!("{note}");
    }
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "API docs for running model servers: {}/",
        server.base_url
    )?;
    for record in &targets {
        writeln!(
            out,
            "  {} ({}): {}",
            record.service_id,
            record.served_model_name(),
            server.service_docs_url(&record.service_id)
        )?;
    }
    writeln!(
        out,
        "Press Ctrl+C to stop serving the docs; the model servers keep running."
    )?;
    // A caller reading the link from a pipe must see it now, not at exit.
    out.flush().context("failed to write the API docs link")?;
    drop(out);

    let waited = runtime
        .block_on(&mut interrupted)
        .context("failed to wait for Ctrl+C");
    server.stop();
    waited?;
    println!("API docs stopped.");
    Ok(())
}

/// The loader the docs server reads the running servers with: the one
/// `rocm services list` uses, which checks each server's process and
/// endpoint.
fn services_loader() -> rocmd::ServiceLoader {
    std::sync::Arc::new(|paths: &AppPaths| load_managed_services(paths))
}

/// The running servers whose docs links are printed: the one named, or every
/// running server. Refuses when there is nothing to document, so the command
/// never serves an empty page silently.
fn docs_targets<'a>(
    records: &'a [ManagedServiceRecord],
    service_id: Option<&str>,
) -> Result<Vec<&'a ManagedServiceRecord>> {
    let live: Vec<&ManagedServiceRecord> = records.iter().filter(|r| r.is_live()).collect();
    match service_id {
        Some(service_id) => {
            if let Some(record) = live.iter().find(|r| r.service_id == service_id) {
                return Ok(vec![record]);
            }
            match records.iter().find(|r| r.service_id == service_id) {
                // Stopped servers are listed only with `--all`; the restart
                // command is the one that list prints for this server.
                Some(record) => bail!(
                    "Server `{service_id}` is not running (status: {}), so it has no API docs. \
                     Start it again with `{}`; `rocm services list --all` shows it.",
                    record.status,
                    crate::service_hints::service_action_hint("restart", service_id)
                ),
                // The not-found error every service-id command shares.
                None => Err(rocm_core::ManagedServiceNotFound {
                    service_id: service_id.to_owned(),
                }
                .into()),
            }
        }
        None if live.is_empty() => bail!(
            "No model server is running, so there are no API docs to serve. \
             Start one with `rocm serve <model>`."
        ),
        None => Ok(live),
    }
}

/// Start the docs server on the requested port, or on `default_port` when
/// none was requested, falling back to a free port when that one is taken.
/// An explicitly requested port is never swapped for another. Returns the
/// server and, after a fallback, the line saying so.
fn start_docs<S>(
    paths: &AppPaths,
    requested: Option<u16>,
    default_port: u16,
    start: impl Fn(AppPaths, u16) -> Result<S>,
) -> Result<(S, Option<String>)> {
    if let Some(port) = requested {
        return Ok((start(paths.clone(), port)?, None));
    }
    match start(paths.clone(), default_port) {
        Ok(server) => Ok((server, None)),
        // Only a port that is in use is worked around; any other failure
        // would recur on the free port and is reported as it is.
        Err(error) if !is_addr_in_use(&error) => Err(error),
        Err(error) => {
            let server = start(paths.clone(), 0)?;
            Ok((
                server,
                Some(format!(
                    "note: port {default_port} is not available ({error:#}); serving on a free port"
                )),
            ))
        }
    }
}

fn is_addr_in_use(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::AddrInUse)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn record(paths: &AppPaths, id: &str, status: &str) -> ManagedServiceRecord {
        let mut record = ManagedServiceRecord::new(
            paths,
            id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            8000,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.status = status.to_owned();
        record
    }

    #[test]
    fn docs_targets_cover_every_state_a_server_can_be_in() {
        let (root, paths) = crate::tests::test_paths("services-docs-targets");
        let records = vec![
            record(&paths, "svc-live", "ready"),
            record(&paths, "svc-starting", "starting"),
            record(&paths, "svc-stopped", "stopped"),
        ];
        let ids = |targets: Vec<&ManagedServiceRecord>| -> Vec<String> {
            targets.iter().map(|r| r.service_id.clone()).collect()
        };

        assert_eq!(
            ids(docs_targets(&records, None).unwrap()),
            ["svc-live", "svc-starting"],
            "every running server, and only those"
        );
        assert_eq!(
            ids(docs_targets(&records, Some("svc-live")).unwrap()),
            ["svc-live"]
        );
        let stopped = docs_targets(&records, Some("svc-stopped")).unwrap_err();
        assert_eq!(
            stopped.to_string(),
            "Server `svc-stopped` is not running (status: stopped), so it has no API docs. \
             Start it again with `rocm services restart svc-stopped --yes`; \
             `rocm services list --all` shows it."
        );
        let unknown = docs_targets(&records, Some("nope")).unwrap_err();
        assert_eq!(
            unknown.to_string(),
            "No local server named `nope`. See `rocm services list --all`."
        );
        let none_running = docs_targets(&records[2..], None).unwrap_err();
        assert!(
            none_running
                .to_string()
                .starts_with("No model server is running"),
            "{none_running}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_taken_default_port_falls_back_to_a_free_one_but_a_requested_port_does_not() {
        let (root, paths) = crate::tests::test_paths("services-docs-ports");
        let tried = RefCell::new(Vec::new());
        let taken_default = |_: AppPaths, port: u16| -> Result<u16> {
            tried.borrow_mut().push(port);
            if port == 11_430 {
                return Err(anyhow::Error::from(std::io::Error::from(
                    std::io::ErrorKind::AddrInUse,
                ))
                .context("failed to bind API docs server to 127.0.0.1:11430"));
            }
            Ok(port)
        };

        let (bound, note) = start_docs(&paths, None, 11_430, taken_default).unwrap();
        assert_eq!(bound, 0, "fell back to a free port");
        assert_eq!(*tried.borrow(), [11_430, 0]);
        assert_eq!(
            note.as_deref(),
            Some(
                "note: port 11430 is not available (failed to bind API docs server to \
                 127.0.0.1:11430: address in use); serving on a free port"
            )
        );

        tried.borrow_mut().clear();
        let error = start_docs(&paths, Some(11_430), 11_430, taken_default).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to bind API docs server to 127.0.0.1:11430"
        );
        assert_eq!(
            *tried.borrow(),
            [11_430],
            "a requested port is never swapped"
        );

        tried.borrow_mut().clear();
        let (bound, note) = start_docs(&paths, None, 11_431, taken_default).unwrap();
        assert_eq!(
            (bound, note),
            (11_431, None),
            "a free default port is used as is"
        );

        // A failure that is not the port being taken would recur on any
        // port, so it is reported rather than retried elsewhere.
        tried.borrow_mut().clear();
        let broken = |_: AppPaths, port: u16| -> Result<u16> {
            tried.borrow_mut().push(port);
            bail!("failed to build the API docs runtime")
        };
        let error = start_docs(&paths, None, 11_430, broken).unwrap_err();
        assert_eq!(error.to_string(), "failed to build the API docs runtime");
        assert_eq!(*tried.borrow(), [11_430], "no fallback for other errors");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_launched_server_gets_engine_docs_only_when_its_engine_serves_them() {
        let (root, paths) = crate::tests::test_paths("services-docs-launch-rows");
        record(&paths, "svc-vllm", "ready").write().unwrap();
        let mut lemonade = record(&paths, "svc-lemonade", "ready");
        lemonade.engine = "lemonade".to_owned();
        lemonade.write().unwrap();

        assert_eq!(
            launch_docs_rows(&paths, "svc-vllm"),
            [
                ("docs", "rocm services docs svc-vllm".to_owned()),
                ("engine docs", "http://127.0.0.1:8000/docs".to_owned()),
            ]
        );
        assert_eq!(
            launch_docs_rows(&paths, "svc-lemonade"),
            [("docs", "rocm services docs svc-lemonade".to_owned())]
        );
        assert_eq!(
            launch_docs_rows(&paths, "svc-missing"),
            [("docs", "rocm services docs svc-missing".to_owned())],
            "an unreadable record still names the command, which reports it"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// `rocmd` cannot call the engines; `rocm` links them, so the check that
    /// the docs describe the routes each engine reports lives here. Each
    /// engine is asked for its `Endpoint` answer the way the CLI asks it,
    /// against a state file like the one its launch writes.
    #[test]
    fn documented_routes_are_the_routes_each_engine_reports() {
        let (root, paths) = crate::tests::test_paths("services-docs-engine-routes");
        paths.ensure().unwrap();
        let service_id = "svc-routes";
        let mut reported = Vec::new();
        for engine in crate::cli::SUPPORTED_ENGINES {
            let state_dir = paths.engine_state_dir(engine);
            std::fs::create_dir_all(&state_dir).unwrap();
            std::fs::write(
                state_dir.join(format!("{service_id}.json")),
                r#"{"status":"running","host":"127.0.0.1","port":9,"endpoint_url":"http://127.0.0.1:9/v1"}"#,
            )
            .unwrap();
            let endpoint: rocm_engine_protocol::EndpointResponse =
                crate::engine_ipc::engine_request(
                    Some(&paths),
                    engine,
                    rocm_engine_protocol::EngineMethod::Endpoint,
                    &rocm_engine_protocol::EndpointRequest {
                        service_id: service_id.to_owned(),
                    },
                )
                .unwrap_or_else(|error| panic!("{engine}: {error:#}"));
            reported.push((engine, endpoint.supported_routes));
        }
        let _ = std::fs::remove_dir_all(root);

        for (engine, routes) in &reported {
            assert_eq!(&rocmd::documented_route_paths(engine), routes, "{engine}");
        }
        // Lemonade's health route differs from both others, so Lemonade
        // reporting vLLM's or AIM's routes, or either of them reporting
        // Lemonade's, cannot pass for its own. vLLM and AIM share `/health`,
        // so their route lists are the same and a mix-up between the two
        // changes nothing the docs show.
        let routes_of = |name: &str| {
            &reported
                .iter()
                .find(|(engine, _)| *engine == name)
                .unwrap_or_else(|| panic!("{name} was not asked"))
                .1
        };
        assert_ne!(routes_of("lemonade"), routes_of("vllm"));
        assert_ne!(routes_of("lemonade"), routes_of("aim"));
        assert_eq!(routes_of("vllm"), routes_of("aim"));
    }

    /// A server whose process died, though its record still says "ready",
    /// is not running to the docs, as it is not to `rocm services list`.
    #[test]
    fn a_server_whose_process_died_has_no_docs() {
        let (root, paths) = crate::tests::test_paths("services-docs-dead-pid");
        paths.ensure().unwrap();
        std::fs::create_dir_all(paths.services_dir()).unwrap();
        let mut exited = if cfg!(windows) {
            std::process::Command::new("cmd")
                .args(["/C", "exit 0"])
                .spawn()
                .unwrap()
        } else {
            std::process::Command::new("true").spawn().unwrap()
        };
        exited.wait().unwrap();
        // Nothing listens on the recorded port either.
        let dead_port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut crashed = record(&paths, "svc-crashed", "ready");
        crashed.port = dead_port;
        crashed.supervisor_pid = exited.id();
        crashed.write().unwrap();
        assert!(crashed.is_live(), "premise: the record still says it runs");

        let server = rocmd::start_api_docs_server(paths, 0, services_loader()).unwrap();
        // Not proxied: the docs are on loopback.
        let agent = ureq::AgentBuilder::new().try_proxy_from_env(false).build();
        let index = agent
            .get(&format!("{}/", server.base_url))
            .call()
            .unwrap()
            .into_string()
            .unwrap();
        let (status, page) = match agent.get(&server.service_docs_url("svc-crashed")).call() {
            Ok(response) => (response.status(), response.into_string().unwrap()),
            Err(ureq::Error::Status(status, response)) => (status, response.into_string().unwrap()),
            Err(error) => panic!("docs request failed: {error}"),
        };
        server.stop();
        let _ = std::fs::remove_dir_all(root);

        assert!(!index.contains("svc-crashed"), "{index}");
        assert!(index.contains("No model server is running"), "{index}");
        assert_eq!(status, 404);
        assert!(page.contains("is not running"), "{page}");
    }

    #[test]
    fn the_docs_hint_names_this_command_with_the_server_id() {
        // The hint text only; that the named command serves the docs is
        // scenario service-docs-01, which runs the line `rocm services list`
        // printed.
        assert_eq!(docs_hint("svc-1"), "rocm services docs svc-1");
    }
}
