// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the public-bind endpoint authentication contract (EAI-7409).
//!
//! `rocm serve --host 0.0.0.0 --allow-public-bind` issues an endpoint API key,
//! prints it exactly once, and hands the serving engine a key file to enforce.
//! These steps drive a real managed serve and then talk to the endpoint directly
//! over HTTP with and without credentials, because the ENGINE is what enforces the
//! key — the CLI only issues it. That is also why these steps are GPU-lane only:
//! there is no plan-only serve mode that would surface enforcement without a real
//! engine and model. Contracts read from the CLI's own output (EAI-8072).

use std::time::{Duration, Instant};

use cucumber::{then, when};

use crate::E2eWorld;
use crate::e2e::serving_steps::{ensure_serve_port_free, host_serve_target, serve_timeout_for};

/// The port every serve scenario shares (see `serving_steps::SERVE_PORT`). A public
/// bind listens on all interfaces, so the test still reaches it over loopback.
const ENDPOINT_URL: &str = "http://127.0.0.1:11435/v1";

/// Pull the issued endpoint key out of the launch output. `serve` prints it as
/// `  api key: <key>` exactly once, alongside the note and curl example.
fn issued_api_key(output: &str) -> Option<String> {
    output
        .lines()
        .find_map(|line| line.trim().strip_prefix("api key:"))
        .map(|key| key.trim().to_owned())
        .filter(|key| !key.is_empty())
}

/// GET `<endpoint>/models`, optionally presenting a bearer token. `None` means the
/// request never reached a listener (connection refused / timed out), which is
/// distinct from "answered with a status" — the readiness wait tolerates that while
/// the engine is still binding, but an assertion must not.
async fn try_models_status(bearer: Option<&str>) -> Option<reqwest::StatusCode> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("failed to build HTTP client");
    let mut request = client.get(format!("{ENDPOINT_URL}/models"));
    if let Some(token) = bearer {
        request = request.bearer_auth(token);
    }
    request.send().await.ok().map(|response| response.status())
}

/// As [`try_models_status`], but a transport failure is fatal. Used by the
/// assertions, which only run once the endpoint has already answered — so at that
/// point a refused connection is a genuine failure, not a startup race.
async fn get_models_status(bearer: Option<&str>) -> reqwest::StatusCode {
    try_models_status(bearer)
        .await
        .unwrap_or_else(|| panic!("GET {ENDPOINT_URL}/models did not reach the served endpoint"))
}

/// Wait until the authenticated endpoint answers, so the assertions run against a
/// loaded model rather than a still-starting one. Polls WITH the key, because an
/// unauthenticated probe is exactly what the endpoint is meant to reject.
///
/// A managed serve returns as soon as the supervisor is up, so for the first while
/// the engine has not bound the port yet and the connection is simply refused.
/// That is expected here and must be retried — panicking on the first refusal is
/// what made this scenario fail before it ever reached its assertions.
async fn wait_for_authenticated_endpoint(key: &str, timeout_secs: u64) {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        if try_models_status(Some(key))
            .await
            .is_some_and(|status| status.is_success())
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "public endpoint {ENDPOINT_URL} did not answer an authenticated request within {timeout_secs}s"
        );
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user serves a model on a public interface with public binding allowed")]
async fn serve_public_bind(world: &mut E2eWorld) {
    let (model, engine, _) = host_serve_target();
    ensure_serve_port_free().await;
    // No `--api-key`: the CLI must GENERATE one, which is the behaviour under test.
    let stdout = crate::run_rocm_ok(
        world,
        &[
            "serve",
            model,
            "--engine",
            engine,
            "--managed",
            "--host",
            "0.0.0.0",
            "--allow-public-bind",
        ],
    );
    let key = issued_api_key(&stdout)
        .unwrap_or_else(|| panic!("serve did not print an endpoint api key:\n{stdout}"));
    wait_for_authenticated_endpoint(&key, serve_timeout_for(world)).await;
    world.endpoint = Some(ENDPOINT_URL.to_owned());
    world.endpoint_api_key = Some(key);
    world.model_name = Some(model.to_owned());
    world.cli_output = Some(stdout);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the CLI shows the endpoint key once and how to send it")]
async fn shows_key_once(world: &mut E2eWorld) {
    let output = world.cli_output.as_deref().expect("no serve output");
    let key = world
        .endpoint_api_key
        .as_deref()
        .expect("no endpoint api key captured");
    // Shown exactly once: a key echoed on several lines would widen the window in
    // which it can be captured from a log.
    let shown = output
        .lines()
        .filter(|line| line.trim().starts_with("api key:"))
        .count();
    assert_eq!(
        shown, 1,
        "expected the key on exactly one line, got {shown}:\n{output}"
    );
    assert!(
        output.contains("shown only now"),
        "expected the show-once note, got:\n{output}"
    );
    assert!(
        output.contains("Authorization: Bearer"),
        "expected the CLI to say how to send the key, got:\n{output}"
    );
    assert!(
        key.len() >= 16,
        "expected a strong generated key, got one of {} chars",
        key.len()
    );
}

#[then("a request without the key is refused as unauthorized")]
async fn unauthenticated_refused(_world: &mut E2eWorld) {
    let status = get_models_status(None).await;
    assert_eq!(
        status,
        reqwest::StatusCode::UNAUTHORIZED,
        "an unauthenticated request to a public endpoint must be refused, got {status}"
    );
}

#[then("a request with the wrong key is refused as unauthorized")]
async fn wrong_key_refused(_world: &mut E2eWorld) {
    let status = get_models_status(Some("not-the-issued-key")).await;
    assert_eq!(
        status,
        reqwest::StatusCode::UNAUTHORIZED,
        "a request with the wrong key must be refused, got {status}"
    );
}

#[then("a request carrying the issued key is accepted")]
async fn issued_key_accepted(world: &mut E2eWorld) {
    let key = world
        .endpoint_api_key
        .clone()
        .expect("no endpoint api key captured");
    let status = get_models_status(Some(&key)).await;
    assert!(
        status.is_success(),
        "the issued key must be accepted, got {status}"
    );
}
