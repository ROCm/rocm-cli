// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `service_docs.feature`, and the serve canaries' last step: the
//! API docs `rocm services docs` serves for the running model servers.
//!
//! The command is the real `rocm`, run in the scenario's isolated tree, on a
//! loopback port of its own (`--port`, from [`free_port`]) unless the scenario
//! is about the default port, so parallel scenarios never share docs and
//! nothing is written to `config.json`. It runs in the foreground until
//! stopped, like a user's terminal; the scenario reads the lines it prints and
//! [`DocsCommand`]'s `Drop` stops it.

use std::io::{BufRead as _, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use cucumber::{given, then, when};
use serde_json::Value;

use e2e_cucumber::mock_server::ServiceRecordOptions;
use e2e_cucumber::polled_deadline::{PolledDeadline, RetryTiming};

use crate::E2eWorld;
use crate::e2e::serving_steps::LIVE_REUSE_MODEL;

/// How long `rocm services docs` gets to print its links.
const DOCS_START_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a stopped server may still show in the docs: their one-second
/// reuse of the server list, with room for a slow machine.
const STOP_SHOWS_WITHIN: Duration = Duration::from_secs(3);

/// How long `rocm services docs` gets to exit after Ctrl+C.
const DOCS_STOP_TIMEOUT: Duration = Duration::from_secs(15);

/// How often the waits on `rocm services docs` re-check its output or exit.
/// They sleep between checks rather than block, so the other scenarios the
/// harness polls on the same task keep running meanwhile.
const DOCS_POLL: Duration = Duration::from_millis(100);

/// A wait on `rocm services docs` of at most `timeout` of polled time.
fn docs_wait(timeout: Duration) -> PolledDeadline {
    PolledDeadline::start(RetryTiming {
        timeout,
        resend_interval: DOCS_POLL,
        poll_interval: DOCS_POLL,
    })
}

/// The port `rocm services docs` serves on when given none:
/// `rocmd::DEFAULT_API_DOCS_PORT`, which this crate does not link.
const DEFAULT_DOCS_PORT: u16 = 11_430;

/// A running `rocm services docs`, killed and reaped when dropped.
#[derive(Debug)]
pub struct DocsCommand {
    child: Child,
    /// The port it serves on, read from the link it printed.
    pub port: u16,
    /// The port it was asked to serve on with `--port`, if any.
    requested_port: Option<u16>,
    /// Every line it has printed so far, stdout and stderr.
    printed: Vec<String>,
    /// Every line it has printed on stdout and on stderr, kept apart for the
    /// steps that check which stream a line went to.
    by_stream: [Arc<Mutex<Vec<String>>>; 2],
    /// The lines it prints from here on; disconnected once it exits.
    lines: mpsc::Receiver<String>,
}

impl Drop for DocsCommand {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .and_then(|listener| listener.local_addr())
        .map(|addr| addr.port())
        .expect("failed to reserve a loopback port")
}

fn port_of(url: &str) -> u16 {
    url.strip_prefix("http://")
        .and_then(|rest| rest.split('/').next())
        .and_then(|authority| authority.rsplit_once(':'))
        .and_then(|(_, port)| port.parse().ok())
        .unwrap_or_else(|| panic!("no port in {url}"))
}

fn something_listens_on(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_secs(2),
    )
    .is_ok()
}

/// The `rocm …` command a `docs` line names, without the leading `rocm`: the
/// plain `docs: rocm services docs <id>` of `rocm services list` and
/// `rocm serve`, or the summary table's `docs   rocm services docs <id>` row.
fn docs_command_in(output: &str) -> Option<Vec<String>> {
    output.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("docs")?;
        let command = rest.trim_start_matches(':').trim();
        let mut words = command.split_whitespace();
        (words.next() == Some("rocm")).then(|| words.map(str::to_owned).collect())
    })
}

async fn get(url: &str) -> reqwest::Response {
    reqwest::get(url)
        .await
        .unwrap_or_else(|error| panic!("GET {url} failed: {error}"))
}

/// Run `rocm <args>`, with `--port <port>` when one is given, in the
/// background and wait for it to print `service_id`'s docs link. Keeps the
/// command in the world and returns the link.
async fn start_docs_command(
    world: &mut E2eWorld,
    args: &[String],
    service_id: &str,
    port: Option<u16>,
) -> String {
    let mut command = Command::new(crate::rocm_binary());
    command.args(args);
    if let Some(port) = port {
        command.args(["--port", &port.to_string()]);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    world.isolate_cmd(&mut command);
    let mut child = command.spawn().expect("failed to start rocm services docs");
    // Both streams end when the command exits, which disconnects the channel:
    // that is how an early exit is noticed without a separate wait.
    let (lines_tx, lines) = mpsc::channel();
    let by_stream: [Arc<Mutex<Vec<String>>>; 2] = Default::default();
    for (stream, kept) in [
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .zip(by_stream.iter().map(Arc::clone))
    .filter_map(|(stream, kept)| Some((stream?, kept)))
    {
        let lines_tx = lines_tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                kept.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(line.clone());
                if lines_tx.send(line).is_err() {
                    break;
                }
            }
        });
    }
    // Only the reader threads hold senders now.
    drop(lines_tx);

    let wanted = format!("/services/{service_id}/docs/");
    let mut deadline = docs_wait(DOCS_START_TIMEOUT);
    let mut printed = Vec::new();
    let link = loop {
        match lines.try_recv() {
            Ok(line) => {
                let link = line
                    .split_whitespace()
                    .find(|word| word.starts_with("http://") && word.ends_with(&wanted))
                    .map(str::to_owned);
                printed.push(line);
                if let Some(link) = link {
                    break link;
                }
            }
            Err(mpsc::TryRecvError::Empty) => {
                if let Some(expiry) = deadline.expired() {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!(
                        "`rocm {}` printed no {wanted} within {expiry}:\n{}",
                        args.join(" "),
                        printed.join("\n")
                    );
                }
                let budget = deadline.budget(DOCS_POLL);
                deadline.run(tokio::time::sleep(budget)).await;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                let _ = child.wait();
                panic!(
                    "`rocm {}` ended without printing {wanted}:\n{}",
                    args.join(" "),
                    printed.join("\n")
                )
            }
        }
    };
    world.docs_command = Some(DocsCommand {
        child,
        port: port_of(&link),
        requested_port: port,
        printed,
        by_stream,
        lines,
    });
    link
}

/// The OpenAPI description a docs page loads, found the way the page finds it:
/// from the Swagger UI initializer served next to the page.
async fn openapi_behind(docs_link: &str) -> (String, Value) {
    let initializer = get(&format!("{docs_link}swagger-initializer.js"))
        .await
        .text()
        .await
        .expect("docs initializer body");
    let start = initializer
        .find("/services/")
        .unwrap_or_else(|| panic!("the docs page names no description:\n{initializer}"));
    let path: String = initializer[start..]
        .chars()
        .take_while(|character| *character != '"')
        .collect();
    let origin = docs_link
        .split("/services/")
        .next()
        .expect("docs link has an origin");
    let openapi = get(&format!("{origin}{path}"))
        .await
        .json::<Value>()
        .await
        .expect("the docs description is JSON");
    (origin.to_owned(), openapi)
}

fn assert_chat_routes_listed(openapi: &Value) {
    for route in ["/v1/models", "/v1/chat/completions", "/v1/completions"] {
        assert!(
            openapi["paths"].get(route).is_some(),
            "the docs do not list {route}:\n{openapi:#}"
        );
    }
}

// ── Given ──────────────────────────────────────────────────────────

#[given("the user is serving the API docs for the running server")]
async fn user_serves_docs(world: &mut E2eWorld) {
    let args: Vec<String> = ["services", "docs", "e2e-mock"].map(str::to_owned).to_vec();
    let link = start_docs_command(world, &args, "e2e-mock", Some(free_port())).await;
    let page = get(&link).await;
    assert!(page.status().is_success(), "premise: the docs open");
    world.docs_link = Some(link);
}

#[given("the model is registered with the CLI as a stopped server")]
async fn register_stopped_server(world: &mut E2eWorld) {
    world.register_mock_service_with(ServiceRecordOptions {
        status: "stopped",
        ..ServiceRecordOptions::default()
    });
}

#[given("another program is listening on the port the API docs are asked to serve on")]
async fn another_program_holds_given_port(world: &mut E2eWorld) {
    let listener =
        std::net::TcpListener::bind(("127.0.0.1", 0)).expect("failed to hold a loopback port");
    let port = listener.local_addr().expect("held port").port();
    world.held_docs_port = Some((port, Some(listener)));
}

#[given("another program is listening on the default API docs port")]
async fn another_program_holds_default_port(world: &mut E2eWorld) {
    let held = match std::net::TcpListener::bind(("127.0.0.1", DEFAULT_DOCS_PORT)) {
        Ok(listener) => Some(listener),
        // Already held by another program on this machine: taken either way.
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => None,
        Err(error) => panic!("failed to hold port {DEFAULT_DOCS_PORT}: {error}"),
    };
    world.held_docs_port = Some((DEFAULT_DOCS_PORT, held));
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user runs the API docs command the services list names for the running server")]
async fn user_runs_listed_docs_command(world: &mut E2eWorld) {
    let listing = crate::run_rocm_ok(world, &["services", "list"]);
    let args = docs_command_in(&listing)
        .unwrap_or_else(|| panic!("the services list names no API docs command:\n{listing}"));
    let link = start_docs_command(world, &args, "e2e-mock", Some(free_port())).await;
    world.docs_link = Some(link);
}

#[when("the user tries the chat completions route from the server's API docs")]
async fn user_tries_chat_route(world: &mut E2eWorld) {
    let link = world.docs_link.clone().expect("no docs link");
    let (origin, openapi) = openapi_behind(&link).await;
    // What "Try it out" sends: the route's example body, to the server the
    // description names, from the docs page's own origin.
    let server = openapi["servers"][0]["url"]
        .as_str()
        .expect("the docs name a server to send requests to");
    let body = openapi["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"]
        ["application/json"]["example"]
        .clone();
    let response = reqwest::Client::new()
        .post(format!("{origin}{server}/v1/chat/completions"))
        .header(reqwest::header::ORIGIN, origin.as_str())
        .json(&body)
        .send()
        .await
        .expect("try-it request failed to send");
    let status = response.status();
    let answer = response.text().await.unwrap_or_default();
    world.cli_output = Some(format!("{status}\n{answer}"));
}

#[when("the user stops the managed model")]
async fn user_stops_managed_model(world: &mut E2eWorld) {
    crate::run_rocm_ok(world, &["services", "stop", "e2e-mock", "--yes"]);
}

/// Run a `rocm services docs` that is expected to be refused, on the port the
/// scenario holds.
fn run_refused_docs_command(world: &mut E2eWorld, args: &[&str]) {
    let (port, _) = world
        .held_docs_port
        .as_ref()
        .expect("premise: a program holds the port");
    let port_arg = port.to_string();
    let mut args = args.to_vec();
    args.extend(["--port", &port_arg]);
    let (stdout, stderr, rc) = crate::run_rocm(world, &args);
    world.cli_output = Some(format!("rc={rc}\n{stdout}{stderr}"));
    world.cli_rc = Some(rc);
}

#[when("the user asks for the API docs of a server that does not exist")]
async fn user_asks_docs_for_unknown_server(world: &mut E2eWorld) {
    run_refused_docs_command(world, &["services", "docs", "no-such-server"]);
}

#[when("the user asks for the API docs with no model server running")]
async fn user_asks_docs_without_servers(world: &mut E2eWorld) {
    run_refused_docs_command(world, &["services", "docs"]);
}

/// What a terminal sends a foreground process on Ctrl+C.
#[cfg(unix)]
fn send_ctrl_c(child: &Child) {
    let pid = libc::pid_t::try_from(child.id()).expect("pid fits pid_t");
    // SAFETY: `kill` only sends a signal to the child this scenario started
    // and has not reaped yet.
    #[allow(unsafe_code)]
    let sent = unsafe { libc::kill(pid, libc::SIGINT) };
    assert_eq!(sent, 0, "failed to send SIGINT to rocm services docs");
}

/// The scenario using this is `@requires-os:linux`.
#[cfg(not(unix))]
fn send_ctrl_c(_: &Child) {
    panic!("Ctrl+C is sent as SIGINT, which only Unix has");
}

#[when("the user asks for the API docs of the stopped server")]
async fn user_asks_docs_for_stopped_server(world: &mut E2eWorld) {
    // Held for the call, as in the other refusal scenarios: a command that
    // served instead of refusing fails on the port rather than serving
    // until the lane times out.
    let held = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("failed to hold a port");
    let port = held.local_addr().expect("held port").port().to_string();
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["services", "docs", "e2e-mock", "--port", &port]);
    drop(held);
    world.cli_output = Some(format!("{stdout}{stderr}"));
    world.cli_rc = Some(rc);
}

/// `rocm serve` of the model the planted live record serves, which reuses it.
/// The synthetic `--env-id` satisfies resolution without installing a runtime,
/// as in scenario serve-28.
fn serve_live_model_again(world: &mut E2eWorld, extra: &[&str]) {
    let mut args = vec![
        "serve",
        LIVE_REUSE_MODEL,
        "--engine",
        "vllm",
        "--env-id",
        "e2e-reuse",
    ];
    args.extend(extra);
    let (stdout, stderr, rc) = crate::run_rocm(world, &args);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user serves that same model again with plain output")]
async fn user_serves_live_model_plain(world: &mut E2eWorld) {
    serve_live_model_again(world, &[]);
}

#[when("the user serves that same model again attached to its output")]
async fn user_serves_live_model_attached(world: &mut E2eWorld) {
    serve_live_model_again(world, &["--verbose"]);
}

#[when("the user presses Ctrl+C in the API docs command")]
async fn user_presses_ctrl_c(world: &mut E2eWorld) {
    let docs = world.docs_command.as_mut().expect("no docs command");
    send_ctrl_c(&docs.child);
    let mut deadline = docs_wait(DOCS_STOP_TIMEOUT);
    let status = loop {
        if let Some(status) = docs.child.try_wait().expect("wait for rocm services docs") {
            break status;
        }
        if let Some(expiry) = deadline.expired() {
            panic!("rocm services docs did not exit within {expiry} of Ctrl+C");
        }
        let budget = deadline.budget(DOCS_POLL);
        deadline.run(tokio::time::sleep(budget)).await;
    };
    // The reader threads finish, and the channel disconnects, once both
    // streams close; the process has exited, so that is moments away.
    let mut deadline = docs_wait(DOCS_STOP_TIMEOUT);
    loop {
        match docs.lines.try_recv() {
            Ok(line) => docs.printed.push(line),
            Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {
                if let Some(expiry) = deadline.expired() {
                    panic!("rocm services docs left its output open {expiry} after exiting");
                }
                let budget = deadline.budget(DOCS_POLL);
                deadline.run(tokio::time::sleep(budget)).await;
            }
        }
    }
    world.cli_rc = status.code();
    world.cli_output = Some(docs.printed.join("\n"));
}

#[when("the user runs the API docs command for the running server without choosing a port")]
async fn user_runs_docs_on_default_port(world: &mut E2eWorld) {
    let args: Vec<String> = ["services", "docs", "e2e-mock"].map(str::to_owned).to_vec();
    let link = start_docs_command(world, &args, "e2e-mock", None).await;
    world.docs_link = Some(link);
}

#[when("the user opens the engine docs the services list names for the running server")]
async fn user_opens_engine_docs(world: &mut E2eWorld) {
    let listing = crate::run_rocm_ok(world, &["services", "list"]);
    let url = listing
        .lines()
        .find_map(|line| line.trim().strip_prefix("engine docs:"))
        .map_or_else(
            || panic!("the services list names no engine docs:\n{listing}"),
            |url| url.trim().to_owned(),
        );
    let page = get(&url).await;
    let status = page.status();
    let body = page.text().await.unwrap_or_default();
    world.docs_link = Some(url);
    world.cli_output = Some(format!("{status}\n{body}"));
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the command prints a link to the server's API docs")]
async fn command_prints_docs_link(world: &mut E2eWorld) {
    let link = world.docs_link.as_deref().expect("no docs link");
    let docs = world.docs_command.as_ref().expect("no docs command");
    let port = docs
        .requested_port
        .expect("premise: the scenario chose the port");
    assert_eq!(
        link,
        format!("http://127.0.0.1:{port}/services/e2e-mock/docs/"),
        "the link names a different server, or not the port it was asked for"
    );
}

#[then("the link opens API docs listing the chat routes with the server's model")]
async fn link_opens_chat_docs(world: &mut E2eWorld) {
    let link = world.docs_link.clone().expect("no docs link");
    let page = get(&link).await;
    assert!(page.status().is_success(), "the docs page did not open");
    assert!(
        page.text().await.unwrap_or_default().contains("swagger-ui"),
        "the link did not open the API docs page"
    );
    let (_, openapi) = openapi_behind(&link).await;
    assert_chat_routes_listed(&openapi);
    let model = world.model_name.as_deref().expect("no model name");
    assert_eq!(
        openapi["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"]["application/json"]
            ["example"]["model"],
        model,
        "the docs' example does not use the server's model"
    );
}

#[then("the model server answers the request")]
async fn model_server_answers(world: &mut E2eWorld) {
    let answer = world.cli_output.as_deref().expect("no try-it answer");
    let (status, body) = answer.split_once('\n').unwrap_or((answer, ""));
    assert!(
        status.starts_with("200"),
        "try-it was not answered: {answer}"
    );
    let body: Value = serde_json::from_str(body)
        .unwrap_or_else(|_| panic!("the answer is not the server's JSON: {answer}"));
    assert!(
        body["choices"][0]["message"]["content"].is_string(),
        "the answer is not a chat completion: {body:#}"
    );
}

#[then("the server's API docs say it is not running")]
async fn docs_say_not_running(world: &mut E2eWorld) {
    let link = world.docs_link.clone().expect("no docs link");
    // The docs reuse one read of the running servers for up to a second
    // (`SERVICE_LIST_TTL` in rocmd's `api_docs.rs`), so the stop shows within
    // that and not necessarily at once.
    let deadline = Instant::now() + STOP_SHOWS_WITHIN;
    loop {
        let page = get(&link).await;
        let status = page.status();
        let text = page.text().await.unwrap_or_default();
        if status == reqwest::StatusCode::NOT_FOUND {
            assert!(text.contains("is not running"), "{text}");
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the stopped server's docs still answer {status} after {STOP_SHOWS_WITHIN:?}"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[then("the services list still shows the stopped server, without an API docs command")]
async fn stopped_server_listed_without_docs_command(world: &mut E2eWorld) {
    // `--all`: the default list hides stopped servers, so it could not show a
    // docs line for one whatever the CLI did.
    let listing = crate::run_rocm_ok(world, &["services", "list", "--all"]);
    assert!(
        listing.contains("- e2e-mock") && listing.contains("status: stopped"),
        "premise: the stopped server is still listed:\n{listing}"
    );
    assert_eq!(docs_command_in(&listing), None, "{listing}");
}

#[then("the API docs command is refused, naming the services list")]
async fn docs_refused_for_unknown_server(world: &mut E2eWorld) {
    let output = world.cli_output.as_deref().expect("no output");
    assert_ne!(world.cli_rc, Some(0), "{output}");
    assert!(
        output.contains("No local server named `no-such-server`. See `rocm services list --all`."),
        "{output}"
    );
    // The command it names lists the running server whose docs do exist.
    let listing = crate::run_rocm_ok(world, &["services", "list", "--all"]);
    assert!(listing.contains("- e2e-mock"), "{listing}");
}

#[then("the API docs command is refused, saying no model server is running")]
async fn docs_refused_without_servers(world: &mut E2eWorld) {
    let output = world.cli_output.as_deref().expect("no output");
    assert_ne!(world.cli_rc, Some(0), "{output}");
    assert!(
        output.contains(
            "No model server is running, so there are no API docs to serve. \
             Start one with `rocm serve <model>`."
        ),
        "{output}"
    );
}

#[then("the refusal comes before the command tried to serve on its port")]
async fn refused_before_serving(world: &mut E2eWorld) {
    // A command that bound its port before refusing would have failed on the
    // port the scenario holds, and said so, instead of printing the refusal.
    let output = world.cli_output.as_deref().expect("no output");
    for sign_of_serving in ["failed to bind", "API docs for running model servers"] {
        assert!(!output.contains(sign_of_serving), "{output}");
    }
}

#[then("the API docs command says the docs stopped")]
async fn docs_command_says_stopped(world: &mut E2eWorld) {
    let output = world.cli_output.as_deref().expect("no output");
    assert_eq!(world.cli_rc, Some(0), "{output}");
    assert!(
        output.contains("Press Ctrl+C to stop serving the docs; the model servers keep running."),
        "{output}"
    );
    assert!(output.ends_with("API docs stopped."), "{output}");
}

#[then("nothing serves API docs on its port any more")]
async fn nothing_serves_docs_any_more(world: &mut E2eWorld) {
    // The scenario opened the docs on this port before Ctrl+C. The docs run
    // inside the command's own process, so this holds once it has exited
    // (the step before checks its exit code); what it rules out is anything
    // the command left behind still listening on the port.
    let port = world.docs_command.as_ref().expect("no docs command").port;
    assert!(
        !something_listens_on(port),
        "something still answers on docs port {port} after Ctrl+C"
    );
}

#[then("the model server still answers")]
async fn model_server_still_answers(world: &mut E2eWorld) {
    let port = world.mock.as_ref().expect("no model server").port();
    let models = get(&format!("http://127.0.0.1:{port}/v1/models")).await;
    assert!(models.status().is_success(), "{}", models.status());
    // And the CLI still lists it as running, with its docs command.
    let listing = crate::run_rocm_ok(world, &["services", "list"]);
    assert!(docs_command_in(&listing).is_some(), "{listing}");
}

#[then("the command says the default port is taken and that it serves on a free port")]
async fn says_default_port_taken(world: &mut E2eWorld) {
    let docs = world.docs_command.as_ref().expect("no docs command");
    let printed = docs.printed.join("\n");
    assert!(
        printed.contains(&format!("note: port {DEFAULT_DOCS_PORT} is not available"))
            && printed.contains("serving on a free port"),
        "{printed}"
    );
    assert_ne!(docs.port, DEFAULT_DOCS_PORT, "{printed}");
    // The note goes to stderr, so a caller reading the links from stdout
    // through a pipe does not get it among them.
    let [stdout, stderr] = docs.by_stream.each_ref().map(|kept| {
        kept.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    });
    assert!(
        stderr.iter().any(|line| line.starts_with("note: port ")),
        "the note is not on stderr:\n{printed}"
    );
    assert!(
        !stdout.iter().any(|line| line.contains("note: port ")),
        "the note is on stdout:\n{printed}"
    );
}

#[then("the link it prints opens the server's API docs on that free port")]
async fn link_opens_on_free_port(world: &mut E2eWorld) {
    let link = world.docs_link.clone().expect("no docs link");
    assert_ne!(port_of(&link), DEFAULT_DOCS_PORT, "{link}");
    let page = get(&link).await;
    assert!(page.status().is_success(), "{link}: {}", page.status());
    assert!(
        page.text().await.unwrap_or_default().contains("swagger-ui"),
        "{link} is not the API docs page"
    );
}

#[then("the model server's own API docs page opens")]
async fn engine_docs_page_opens(world: &mut E2eWorld) {
    let url = world.docs_link.as_deref().expect("no engine docs link");
    let port = world.mock.as_ref().expect("no model server").port();
    assert_eq!(url, format!("http://127.0.0.1:{port}/docs"));
    let answer = world.cli_output.as_deref().expect("no answer");
    assert!(answer.starts_with("200"), "{answer}");
    assert!(
        answer.contains(e2e_cucumber::mock_server::MOCK_ENGINE_DOCS_TITLE),
        "{answer}"
    );
}

/// The commands in backticks in `text`, in order, without the leading `rocm`.
fn quoted_rocm_commands(text: &str) -> Vec<Vec<String>> {
    text.split('`')
        .skip(1)
        .step_by(2)
        .filter_map(|quoted| {
            let mut words = quoted.split_whitespace();
            (words.next() == Some("rocm")).then(|| words.map(str::to_owned).collect())
        })
        .collect()
}

#[then("the API docs command is refused, naming the command that restarts the server")]
async fn docs_refused_naming_restart(world: &mut E2eWorld) {
    let output = world.cli_output.as_deref().expect("no output");
    assert_ne!(world.cli_rc, Some(0), "{output}");
    assert!(
        output.contains(
            "Server `e2e-mock` is not running (status: stopped), so it has no API docs. \
             Start it again with `rocm services restart e2e-mock --yes`; \
             `rocm services list --all` shows it."
        ),
        "{output}"
    );
    assert!(
        !output.contains("API docs for running model servers"),
        "nothing was served: {output}"
    );
}

#[then("the services list the refusal names shows the server stopped, with that restart command")]
async fn followed_list_shows_stopped_server(world: &mut E2eWorld) {
    let output = world.cli_output.clone().expect("no output");
    let commands = quoted_rocm_commands(&output);
    let [restart, list] = commands.as_slice() else {
        panic!("the refusal names a restart and a list command: {output}");
    };
    let list: Vec<&str> = list.iter().map(String::as_str).collect();
    let listing = crate::run_rocm_ok(world, &list);
    assert!(
        listing.contains("- e2e-mock") && listing.contains("status: stopped"),
        "the list the refusal names shows the stopped server:\n{listing}"
    );
    // The list's own way back for the server is the command the refusal named.
    assert!(
        listing.contains(&format!("restart: rocm {}", restart.join(" "))),
        "{listing}"
    );
}

#[then("the output names the API docs command and the engine's own docs for the running server")]
async fn serve_output_names_docs(world: &mut E2eWorld) {
    let output = world.cli_output.clone().expect("no serve output");
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    assert_eq!(world.cli_rc, Some(0), "{output}\n{stderr}");
    assert_eq!(
        docs_command_in(&output),
        Some(["services", "docs", "e2e-mock"].map(str::to_owned).to_vec()),
        "{output}"
    );
    let record: Value = serde_json::from_slice(
        &std::fs::read(
            world
                .isolated_root
                .as_ref()
                .expect("no isolated root")
                .path()
                .join("data/services/e2e-mock.json"),
        )
        .expect("read the running server's record"),
    )
    .expect("parse the record");
    let port = record["port"].as_u64().expect("record port");
    assert!(
        output
            .lines()
            .any(|line| line.trim() == format!("engine docs: http://127.0.0.1:{port}/docs")),
        "{output}"
    );
}

#[then("the API docs command it names opens that server's API docs")]
async fn named_docs_command_opens(world: &mut E2eWorld) {
    let output = world.cli_output.clone().expect("no serve output");
    let args =
        docs_command_in(&output).unwrap_or_else(|| panic!("no API docs command in:\n{output}"));
    let link = start_docs_command(world, &args, "e2e-mock", Some(free_port())).await;
    let page = get(&link).await;
    assert!(page.status().is_success(), "{link}: {}", page.status());
    assert!(
        page.text().await.unwrap_or_default().contains("swagger-ui"),
        "{link} is not the API docs page"
    );
}

#[then("the API docs command printed when it started opens the server's API docs")]
async fn serve_printed_docs_command_opens(world: &mut E2eWorld) {
    let stdout = world.serve_stdout.clone().expect("no serve output");
    let args = docs_command_in(&stdout)
        .unwrap_or_else(|| panic!("`rocm serve` printed no API docs command:\n{stdout}"));
    let service_id = args
        .last()
        .cloned()
        .unwrap_or_else(|| panic!("the docs command names no server: {args:?}"));
    let link = start_docs_command(world, &args, &service_id, Some(free_port())).await;
    let (origin, openapi) = openapi_behind(&link).await;
    assert_chat_routes_listed(&openapi);
    // Every route the docs document for a GET, health included, answers when
    // tried through the docs: a docs page listing another engine's health
    // route would fail here.
    let server = openapi["servers"][0]["url"]
        .as_str()
        .expect("the docs name a server to send requests to")
        .to_owned();
    let get_routes: Vec<String> = openapi["paths"]
        .as_object()
        .expect("paths object")
        .iter()
        .filter(|(_, item)| item.get("get").is_some())
        .map(|(path, _)| path.clone())
        .collect();
    assert!(get_routes.len() >= 2, "health and models: {get_routes:?}");
    for route in get_routes {
        let answer = get(&format!("{origin}{server}{route}")).await;
        assert!(
            answer.status().is_success(),
            "{route} through the docs did not reach the engine: {}",
            answer.status()
        );
    }
}
