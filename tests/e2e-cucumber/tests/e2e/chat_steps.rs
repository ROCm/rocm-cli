// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::time::Duration;

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

#[when("a chat request with tool definitions is sent")]
async fn send_chat_with_tools(world: &mut E2eWorld) {
    // Same discover-then-POST path as a plain chat (including its cold-start
    // retry and transport diagnostics), with a tool definition attached.
    let tools = serde_json::json!([{
        "type": "function",
        "function": {
            "name": "gpu_status",
            "description": "Get GPU status",
            "parameters": {"type": "object", "properties": {}}
        }
    }]);
    let response =
        crate::request_chat_completion(world, "What GPUs are available?", Some(tools)).await;
    world.chat_response = Some(response);
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

#[when("the user pipes a chat prompt to the CLI")]
async fn user_pipes_chat_prompt(world: &mut E2eWorld) {
    // Keep --model explicit: this scenario pins the stdin contract, not the
    // default-model discovery defect covered separately by EAI-8009.
    let model = world.model_name.clone().expect("no model name set");
    let (stdout, stderr, rc) = crate::run_rocm_with_stdin(
        world,
        &["chat", "--provider", "local", "--model", &model],
        "Hello from standard input\n",
        &[],
    );
    assert_eq!(
        rc, 0,
        "piped rocm chat failed (rc={rc}):\n{stdout}\n{stderr}"
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
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

#[then("the chat response is successful")]
async fn assert_chat_successful(world: &mut E2eWorld) {
    let resp = world.chat_response.as_ref().expect("no chat response");
    assert!(
        e2e_cucumber::chat_response_is_successful(resp),
        "no non-empty choices array in response: {resp}"
    );
}

#[then("the CLI sends the piped prompt to the model")]
async fn assert_piped_prompt_sent(world: &mut E2eWorld) {
    // This is the load-bearing assertion. Today the command still exits 0 and
    // prints an assistant status summary, but the mock receives no completion
    // at all. Waiting for the actual request prevents any future incidental
    // output from making the scenario pass for the wrong reason.
    let request = world
        .mock
        .as_ref()
        .expect("no mock server running")
        .wait_for_chat_request(Duration::from_secs(2))
        .await
        .unwrap_or_else(|e| panic!("the piped prompt never reached the model: {e}"));
    // Find the user-role message rather than assuming it is index 0: a correct
    // fix is free to prepend a system prompt, and pinning `messages[0]` would
    // keep this xfailed against exactly that valid behaviour. Same shape as the
    // managed-chat assertion in dash_steps.rs.
    let messages = request
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("chat request had no messages array:\n{request}"));
    let user_content = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(serde_json::Value::as_str) == Some("user"))
        .and_then(|m| m.get("content"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| panic!("no user message in chat request:\n{request}"));
    assert_eq!(
        user_content, "Hello from standard input",
        "the model received the wrong prompt: {request}"
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
