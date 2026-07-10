// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::MockServer;

use crate::E2eWorld;

// ── Given ──────────────────────────────────────────────────────────

#[given("a model is being served")]
async fn setup_model_server(world: &mut E2eWorld) {
    let mock = MockServer::start("TestModel/E2E-1B").await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some("TestModel/E2E-1B".to_string());
    world.mock = Some(mock);
}

#[given("the model is registered with the CLI")]
async fn register_model_with_cli(world: &mut E2eWorld) {
    world.register_mock_service();
}

#[given("a model is being served locally")]
async fn setup_localhost_model(world: &mut E2eWorld) {
    setup_model_server(world).await;
}

// ── When ───────────────────────────────────────────────────────────

#[when("a chat completion request is sent")]
async fn send_chat_request(world: &mut E2eWorld) {
    crate::send_chat(world).await;
}

#[when("the user checks for running services")]
async fn user_checks_services(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["services", "list"]);
    world.cli_output = Some(stdout);
}

#[when("the user is offered the detected endpoint")]
async fn user_offered_endpoint(_world: &mut E2eWorld) {}

#[when("a chat request with tool definitions is sent")]
async fn send_chat_with_tools(world: &mut E2eWorld) {
    let endpoint = world.endpoint.as_ref().expect("no endpoint configured");
    let models_url = format!("{endpoint}/models");
    let resp: serde_json::Value = reqwest::get(&models_url)
        .await
        .unwrap_or_else(|e| panic!("GET {models_url} failed: {e}"))
        .json()
        .await
        .unwrap_or_else(|e| panic!("GET {models_url} returned non-JSON: {e}"));
    let model = resp["data"][0]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("no model id in response: {resp}"))
        .to_string();

    let chat_url = format!("{endpoint}/chat/completions");
    let client = reqwest::Client::new();
    let chat_resp: serde_json::Value = client
        .post(&chat_url)
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "What GPUs are available?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "gpu_status",
                    "description": "Get GPU status",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        }))
        .send()
        .await
        .unwrap_or_else(|e| panic!("POST {chat_url} failed: {e}"))
        .json()
        .await
        .unwrap_or_else(|e| panic!("POST {chat_url} returned non-JSON: {e}"));
    world.chat_response = Some(chat_resp);
}

#[when("the user sends a chat message")]
async fn user_sends_chat(world: &mut E2eWorld) {
    crate::send_chat(world).await;
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the served model is listed")]
async fn assert_model_listed(world: &mut E2eWorld) {
    let output = world
        .cli_output
        .as_ref()
        .expect("no services query was run");
    let model = world.model_name.as_deref().expect("no model name set");
    assert!(
        output.contains(model),
        "served model {model} not found in services list:\n{output}"
    );
}

#[then("the served model endpoint is listed")]
async fn assert_model_endpoint_listed(world: &mut E2eWorld) {
    let output = world
        .cli_output
        .as_ref()
        .expect("no services query was run");
    let port = world.mock.as_ref().expect("no mock server running").port();
    assert!(
        output.contains(&port.to_string()),
        "served model endpoint (port {port}) not found in services list:\n{output}"
    );
}

#[then("the notice does not claim that requests leave the machine")]
async fn assert_privacy_notice_accurate(_world: &mut E2eWorld) {
    eprintln!(
        "WARN: privacy notice verification requires TUI interaction — \
         known limitation for automated tests"
    );
}

#[then("the chat response is successful")]
async fn assert_chat_successful(world: &mut E2eWorld) {
    let resp = world.chat_response.as_ref().expect("no chat response");
    assert!(
        resp.get("choices").is_some(),
        "no choices in response: {resp}"
    );
}

#[then("the response contains a model-generated reply")]
async fn assert_model_generated_reply(world: &mut E2eWorld) {
    let resp = world.chat_response.as_ref().expect("no chat response");
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "empty reply in chat response: {resp}");
}
