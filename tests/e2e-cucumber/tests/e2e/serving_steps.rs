// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::process::Command;
use std::time::{Duration, Instant};

use cucumber::{given, then, when};

use crate::E2eWorld;
use e2e_cucumber::mock_server::MockServer;

fn run_rocm(world: &E2eWorld, args: &[&str]) -> (String, String, i32) {
    let binary = crate::rocm_binary();
    let mut cmd = Command::new(&binary);
    cmd.args(args);
    world.isolate_cmd(&mut cmd);
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run {binary}: {e}"));
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
        output.status.code().unwrap_or(-1),
    )
}

async fn wait_for_endpoint(url: &str, timeout_secs: u64) {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    while Instant::now() < deadline {
        if reqwest::get(url)
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            return;
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
    panic!("endpoint {url} not ready after {timeout_secs}s");
}

// ── Given ──────────────────────────────────────────────────────────

#[given("a model is being served on the default port")]
async fn setup_mock_default_port(world: &mut E2eWorld) {
    let mock = MockServer::start("TestModel/E2E-1B").await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some("TestModel/E2E-1B".to_string());
    world.mock = Some(mock);
}

#[given("a model is being served on a non-default port")]
async fn setup_mock_custom_port(world: &mut E2eWorld) {
    let mock = MockServer::start("TestModel/E2E-1B").await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some("TestModel/E2E-1B".to_string());
    world.mock = Some(mock);
}

#[given("a model is being served on GPU")]
async fn setup_gpu_model(world: &mut E2eWorld) {
    let (stdout, _, rc) = run_rocm(
        world,
        &["serve", "qwen2.5", "--engine", "vllm", "--managed"],
    );
    assert!(rc == 0, "rocm serve failed:\n{stdout}");
    world.endpoint = Some("http://127.0.0.1:11435/v1".to_string());
    world.model_name = Some("Qwen/Qwen2.5-1.5B-Instruct".to_string());
    wait_for_endpoint("http://127.0.0.1:11435/v1/models", 120).await;
}

#[given("a model is served in the background")]
async fn setup_background_model(world: &mut E2eWorld) {
    setup_gpu_model(world).await;
}

#[given("the served model has been detected")]
async fn setup_model_detected(world: &mut E2eWorld) {
    let (stdout, _, _) = run_rocm(world, &["services", "list"]);
    let model = world.model_name.as_deref().unwrap_or("");
    assert!(
        stdout.contains(model),
        "model {model} not found in services:\n{stdout}"
    );
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user serves a model using its short name")]
async fn user_serves_short_name(world: &mut E2eWorld) {
    let (stdout, _, _) = run_rocm(world, &["serve", "qwen2.5", "--engine", "vllm"]);
    world.cli_output = Some(stdout);
}

#[when("the user serves the same short name with different engines")]
async fn user_serves_multiple_engines(world: &mut E2eWorld) {
    let mut outputs = Vec::new();
    for engine in ["lemonade", "vllm"] {
        let (stdout, _, _) = run_rocm(world, &["serve", "qwen2.5", "--engine", engine]);
        outputs.push(stdout);
    }
    world.cli_outputs = Some(outputs);
}

#[when("the user lists running services")]
async fn user_lists_services(world: &mut E2eWorld) {
    let (stdout, _, _) = run_rocm(world, &["services", "list"]);
    world.cli_output = Some(stdout);
}

#[when("the user serves a model without specifying an engine")]
async fn user_serves_default_engine(world: &mut E2eWorld) {
    let (stdout, _, rc) = run_rocm(world, &["serve", "qwen2.5", "--managed"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
    world.endpoint = Some("http://127.0.0.1:11435/v1".to_string());
    world.model_name = Some("Qwen/Qwen2.5-1.5B-Instruct".to_string());
    if rc == 0 {
        wait_for_endpoint("http://127.0.0.1:11435/v1/models", 120).await;
    }
}

#[when("the user sends a chat completion request")]
async fn user_sends_completion(world: &mut E2eWorld) {
    crate::send_chat(world).await;
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the output shows the full model name")]
async fn assert_full_model_name(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no CLI output");
    let resolved = output
        .lines()
        .find(|l| l.contains("resolved model:"))
        .and_then(|l| l.split(':').nth(1))
        .map_or_else(
            || panic!("no 'resolved model' in output:\n{output}"),
            str::trim,
        );
    assert!(
        resolved.contains('/'),
        "expected a fully qualified model name (org/model), got '{resolved}'\n\nfull output:\n{output}"
    );
}

#[then("all engines expand to the same full model name")]
async fn assert_consistent_expansion(world: &mut E2eWorld) {
    let outputs = world.cli_outputs.as_ref().expect("no CLI outputs");
    assert!(
        outputs.len() >= 2,
        "need at least 2 serve outputs to compare"
    );
    let mut resolved: Vec<String> = Vec::new();
    for output in outputs {
        let model = output
            .lines()
            .find(|l| l.contains("resolved model:"))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        resolved.push(model);
    }
    let first = &resolved[0];
    for (i, r) in resolved.iter().enumerate().skip(1) {
        assert_eq!(
            r, first,
            "inconsistent model name expansion across engines: {resolved:?} (index {i})"
        );
    }
}

#[then("the service appears with the correct model name and connection details")]
async fn assert_service_in_list(world: &mut E2eWorld) {
    let (stdout, _, _) = run_rocm(world, &["services", "list"]);
    let model = world.model_name.as_deref().unwrap_or("");
    assert!(
        stdout.to_lowercase().contains(&model.to_lowercase()),
        "model name not in services list:\n{stdout}"
    );
    assert!(
        stdout.contains("127.0.0.1"),
        "endpoint not in services list:\n{stdout}"
    );
}

#[then("the connection details match the actual server port")]
async fn assert_endpoint_port(world: &mut E2eWorld) {
    let mock = world.mock.as_ref().expect("no mock server running");
    let port = mock.port();
    let (stdout, _, _) = run_rocm(world, &["services", "list"]);
    assert!(
        stdout.contains(&port.to_string()),
        "port {port} not found in services list:\n{stdout}"
    );
}

#[then("an engine is selected automatically")]
async fn assert_engine_auto_selected(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no serve output");
    assert!(
        output.contains("engine:"),
        "no engine in serve output:\n{output}"
    );
}

#[then("the model is reachable")]
async fn assert_model_reachable(world: &mut E2eWorld) {
    let endpoint = world.endpoint.as_ref().expect("no endpoint configured");
    let url = format!("{endpoint}/models");
    let resp: serde_json::Value = reqwest::get(&url)
        .await
        .unwrap_or_else(|e| panic!("GET {url} failed: {e}"))
        .json()
        .await
        .unwrap_or_else(|e| panic!("GET {url} returned non-JSON: {e}"));
    let data = resp["data"].as_array();
    assert!(
        data.is_some_and(|d| !d.is_empty()),
        "no models at {url}: {resp}"
    );
}

#[then("the model responds to inference requests")]
async fn assert_endpoint_responds(world: &mut E2eWorld) {
    crate::send_chat(world).await;
    let resp = world.chat_response.as_ref().expect("no chat response");
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "empty reply in chat response: {resp}");
}

#[then("the response contains a model reply")]
async fn assert_response_has_reply(world: &mut E2eWorld) {
    let resp = world.chat_response.as_ref().expect("no chat response");
    let choices = resp["choices"].as_array();
    assert!(
        choices.is_some_and(|c| !c.is_empty()),
        "no choices in chat response: {resp}"
    );
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "empty reply in chat response: {resp}");
}

#[then("the response identifies the correct model")]
async fn assert_response_model_correct(world: &mut E2eWorld) {
    let resp = world.chat_response.as_ref().expect("no chat response");
    let resp_model = resp["model"].as_str().unwrap_or("");
    let expected = world.model_name.as_deref().unwrap_or("");
    assert!(
        resp_model.contains(expected),
        "response model '{resp_model}' does not match '{expected}'"
    );
}
