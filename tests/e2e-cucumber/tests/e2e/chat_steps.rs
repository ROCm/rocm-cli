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

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            crate::inference_timeout_for(world),
        ))
        .build()
        .expect("failed to build HTTP client");

    let models_url = format!("{endpoint}/models");
    let resp: serde_json::Value = client
        .get(&models_url)
        .send()
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

#[when("the user sends a one-shot chat prompt through the CLI")]
async fn user_sends_oneshot_chat(world: &mut E2eWorld) {
    // Drive the real `rocm chat` command (one-shot `--prompt`) so the command
    // surface records it as covered. The local provider resolves the planted
    // managed-service record and talks to the mock server. Passing the served
    // model id avoids depending on any default-model resolution.
    let model = world.model_name.clone().expect("no model name set");
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "chat",
            "--provider",
            "local",
            "--model",
            &model,
            "--prompt",
            "Hello",
        ],
    );
    assert!(rc == 0, "rocm chat failed (rc={rc}):\n{stdout}\n{stderr}");
    world.cli_output = Some(stdout);
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
    // The privacy notice is only shown in the interactive dash/TUI, which a
    // black-box CLI test can't drive — so this behaviour genuinely cannot be
    // verified here (EAI-7222). Rather than pass silently (a green no-op that
    // tests nothing), fail: the scenario is marked xfail in expectations.toml
    // with EAI-7222, so this failure is the *expected* outcome and the gap stays
    // visible until the notice is exposed on a non-TUI surface.
    panic!(
        "privacy notice is TUI-only and cannot be verified black-box (EAI-7222); \
         scenario is tracked as xfail"
    );
}

#[then("the chat response is successful")]
async fn assert_chat_successful(world: &mut E2eWorld) {
    let resp = world.chat_response.as_ref().expect("no chat response");
    assert!(
        e2e_cucumber::chat_response_is_successful(resp),
        "no non-empty choices array in response: {resp}"
    );
}

#[then("the CLI prints the assistant's reply")]
async fn assert_cli_prints_reply(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no chat CLI output");
    // The mock server replies "This is a mock response for testing."; the CLI's
    // one-shot renderer prints the assistant content. Assert the reply text
    // surfaced, so this proves the whole `rocm chat` path (arg parse → local
    // provider → endpoint → rendered output), not merely a zero exit code.
    assert!(
        output.contains("mock response"),
        "chat CLI output does not contain the assistant reply:\n{output}"
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
