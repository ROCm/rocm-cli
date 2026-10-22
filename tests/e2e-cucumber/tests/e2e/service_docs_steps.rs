// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `service_docs.feature`: the API docs the background helper serves
//! for every running model server.
//!
//! The helper is the real `rocm daemon`, started in the scenario's isolated
//! tree on a port of its own (set in that tree's `config.json`), so parallel
//! scenarios and a helper already running on the host never share docs.

use std::time::{Duration, Instant};

use cucumber::{given, then, when};
use serde_json::Value;

use crate::E2eWorld;

/// How long a cold `rocm daemon` gets to start answering.
const HELPER_START_TIMEOUT: Duration = Duration::from_secs(30);

fn free_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .and_then(|listener| listener.local_addr())
        .map(|addr| addr.port())
        .expect("failed to reserve a loopback port")
}

/// The docs link `rocm services list` printed for the one listed server.
fn docs_link_in(listing: &str) -> Option<String> {
    listing
        .lines()
        .find_map(|line| line.trim().strip_prefix("docs: "))
        .map(str::to_owned)
}

async fn get(url: &str) -> reqwest::Response {
    reqwest::get(url)
        .await
        .unwrap_or_else(|error| panic!("GET {url} failed: {error}"))
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

/// Point this scenario's API docs at a free port of their own, in the isolated
/// `config.json`, and return it. A helper `rocm serve` leaves behind in another
/// scenario keeps its own port, so it can never answer for this one. Calling it
/// again keeps the port already chosen: a serve retried within one scenario
/// reuses the helper the first attempt started, which listens there.
pub(crate) fn give_api_docs_a_port_of_their_own(world: &E2eWorld) -> u16 {
    let paths = world.app_paths();
    std::fs::create_dir_all(&paths.config_dir).expect("failed to create config dir");
    let config_path = paths.config_path();
    let mut config: Value = std::fs::read(&config_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    if let Some(port) = config["api_docs"]["port"]
        .as_u64()
        .and_then(|port| u16::try_from(port).ok())
    {
        return port;
    }
    let port = free_port();
    config["api_docs"] = serde_json::json!({ "enabled": true, "port": port });
    std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap())
        .expect("failed to write config.json");
    port
}

/// The docs link in `rocm serve`'s output: the deployment summary's `docs` row
/// or the plain output's `docs:` line.
fn docs_link_in_serve_output(stdout: &str) -> Option<String> {
    stdout.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("docs")?;
        let link = rest.trim_start_matches(':').trim();
        link.starts_with("http://").then(|| link.to_owned())
    })
}

// ── Given ──────────────────────────────────────────────────────────

#[given("the background helper is not running")]
async fn background_helper_not_running(world: &mut E2eWorld) {
    // Nothing listens on this scenario's docs port: the state a host is in
    // when no helper is up, which a helper running on the host itself cannot
    // disturb.
    give_api_docs_a_port_of_their_own(world);
}

#[given("the background helper is running")]
async fn background_helper_running(world: &mut E2eWorld) {
    let port = give_api_docs_a_port_of_their_own(world);
    let paths = world.app_paths();

    // The helper's own output names why docs did not come up (`api_docs: off
    // (...)`), so it is kept for the failure message rather than discarded.
    let log_path = paths.data_dir.join("background-helper.log");
    std::fs::create_dir_all(&paths.data_dir).expect("failed to create data dir");
    let log = std::fs::File::create(&log_path).expect("failed to create helper log");
    let mut command = std::process::Command::new(crate::rocm_binary());
    command
        .arg("daemon")
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone().expect("failed to share helper log"))
        .stderr(log);
    world.isolate_cmd(&mut command);
    world.background_helper = Some(command.spawn().expect("failed to start rocm daemon"));

    let health = format!("http://127.0.0.1:{port}/health");
    let deadline = Instant::now() + HELPER_START_TIMEOUT;
    while !reqwest::get(&health)
        .await
        .is_ok_and(|response| response.status().is_success())
    {
        let exited = world
            .background_helper
            .as_mut()
            .and_then(|helper| helper.try_wait().ok().flatten());
        let output = || std::fs::read_to_string(&log_path).unwrap_or_default();
        assert!(
            exited.is_none(),
            "the background helper exited ({exited:?}) before its API docs answered:\n{}",
            output()
        );
        assert!(
            Instant::now() < deadline,
            "the background helper's API docs did not answer on port {port} within \
             {HELPER_START_TIMEOUT:?}; its output:\n{}",
            output()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[given("the user has opened the server's API docs")]
async fn user_opened_docs(world: &mut E2eWorld) {
    let listing = crate::run_rocm_ok(world, &["services", "list"]);
    let link = docs_link_in(&listing)
        .unwrap_or_else(|| panic!("premise: a running server has a docs link:\n{listing}"));
    let page = get(&link).await;
    assert!(page.status().is_success(), "premise: the docs open");
    world.docs_link = Some(link);
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user tries the chat completions route from the server's API docs")]
async fn user_tries_chat_route(world: &mut E2eWorld) {
    let listing = crate::run_rocm_ok(world, &["services", "list"]);
    let link = docs_link_in(&listing)
        .unwrap_or_else(|| panic!("no API docs link in the services list:\n{listing}"));
    let (origin, openapi) = openapi_behind(&link).await;
    // What "Try it out" sends: the route's example body, to the server the
    // description names.
    let server = openapi["servers"][0]["url"]
        .as_str()
        .expect("the docs name a server to send requests to");
    let body = openapi["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"]
        ["application/json"]["example"]
        .clone();
    let response = reqwest::Client::new()
        .post(format!("{origin}{server}/v1/chat/completions"))
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

// ── Then ───────────────────────────────────────────────────────────

#[then("the running server has an API docs link")]
async fn running_server_has_docs_link(world: &mut E2eWorld) {
    let listing = world.cli_output.as_deref().expect("no services listing");
    let link = docs_link_in(listing)
        .unwrap_or_else(|| panic!("no API docs link in the services list:\n{listing}"));
    assert!(
        link.ends_with("/services/e2e-mock/docs/"),
        "the link names a different server: {link}"
    );
    world.docs_link = Some(link);
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
    for route in ["/v1/models", "/v1/chat/completions", "/v1/completions"] {
        assert!(
            openapi["paths"].get(route).is_some(),
            "the docs do not list {route}:\n{openapi:#}"
        );
    }
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
    let page = get(&link).await;
    assert_eq!(page.status(), reqwest::StatusCode::NOT_FOUND);
    let text = page.text().await.unwrap_or_default();
    assert!(text.contains("is not running"), "{text}");
}

#[then("the services list still shows the stopped server, without an API docs link")]
async fn stopped_server_listed_without_docs_link(world: &mut E2eWorld) {
    // `--all`: the default list hides stopped servers, so it could not show a
    // docs line for one whatever the CLI did.
    let listing = crate::run_rocm_ok(world, &["services", "list", "--all"]);
    assert!(
        listing.contains("- e2e-mock") && listing.contains("status: stopped"),
        "premise: the stopped server is still listed:\n{listing}"
    );
    assert_eq!(docs_link_in(&listing), None, "{listing}");
}

#[then("the running server has no API docs link")]
async fn running_server_has_no_docs_link(world: &mut E2eWorld) {
    let listing = world.cli_output.as_deref().expect("no services listing");
    assert!(
        listing.contains("- e2e-mock"),
        "premise: the running server is listed:\n{listing}"
    );
    assert_eq!(docs_link_in(listing), None, "{listing}");
}

#[then("the API docs link printed when it started opens the server's API docs")]
async fn serve_printed_docs_link_opens(world: &mut E2eWorld) {
    let stdout = world.serve_stdout.as_deref().expect("no serve output");
    let link = docs_link_in_serve_output(stdout)
        .unwrap_or_else(|| panic!("`rocm serve` printed no API docs link:\n{stdout}"));
    let (origin, openapi) = openapi_behind(&link).await;
    for route in ["/v1/models", "/v1/chat/completions", "/v1/completions"] {
        assert!(
            openapi["paths"].get(route).is_some(),
            "the docs do not list {route}:\n{openapi:#}"
        );
    }
    // The docs' "Try it out" reaches the real engine: list its models through
    // the docs page.
    let server = openapi["servers"][0]["url"]
        .as_str()
        .expect("the docs name a server to send requests to");
    let models = get(&format!("{origin}{server}/v1/models")).await;
    assert!(
        models.status().is_success(),
        "try-it through the docs did not reach the engine: {}",
        models.status()
    );
}
