// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::MockServer;

use crate::E2eWorld;

/// The exact bytes `user_pipes_oneshot_chat` writes to the CLI's stdin. The
/// leading indentation and the trailing space are content — a model reads them,
/// the same as it would from `--prompt` — while the final newline is what
/// `echo "…" |` appends, so only that may be stripped.
const PIPED_PROMPT_STDIN: &str = "    Hello, indented and trailing-spaced \n";

/// `PIPED_PROMPT_STDIN` minus the one line ending the writer appended: what the
/// CLI must send byte for byte, so a piped prompt equals the same text passed
/// with `--prompt`. Kept next to the input so the two cannot drift apart.
const PIPED_PROMPT_SENT: &str = "    Hello, indented and trailing-spaced ";

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

#[when("the user pipes a one-shot chat prompt through the CLI")]
async fn user_pipes_oneshot_chat(world: &mut E2eWorld) {
    // Drive `rocm chat` with the prompt on stdin and no `--prompt`, matching the
    // documented `echo "…" | rocm chat` path. `run_rocm_with_stdin` pipes stdin,
    // so the child sees a non-terminal stdin and must read the prompt from it.
    // Passing the served model id avoids depending on default-model resolution.
    // The prompt carries indentation, a trailing space, and the newline `echo`
    // adds, so the `Then` step can prove which of those the CLI forwards.
    let model = world.model_name.clone().expect("no model name set");
    let (stdout, stderr, rc) = crate::run_rocm_with_stdin(
        world,
        &["chat", "--provider", "local", "--model", &model],
        PIPED_PROMPT_STDIN,
        &[],
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

#[then("the model receives the piped prompt with its whitespace intact")]
async fn piped_chat_request_kept_whitespace(world: &mut E2eWorld) {
    // The canned reply never varies with the prompt, so printing it proves only
    // that stdin reached the send path, not that it arrived unaltered — trimming
    // the piped text would still pass. Assert on the request the mock recorded:
    // the model must see the piped prompt byte for byte, minus the single line
    // ending the writer appended. The CLI process has already exited by now and
    // the mock records the body before it answers, so the snapshot is final and
    // needs no polling.
    let body = world
        .mock
        .as_ref()
        .expect("no mock server running")
        .last_chat_request()
        .unwrap_or_else(|| panic!("the mock never received a chat request"));
    let messages = body
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("chat request had no messages array:\n{body}"));
    let last_user_content = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(serde_json::Value::as_str) == Some("user"))
        .and_then(|m| m.get("content"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| panic!("no user message found in chat request:\n{body}"));
    assert_eq!(
        last_user_content, PIPED_PROMPT_SENT,
        "the piped prompt reached the model altered; full request:\n{body}"
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

// ── Direct tool calls through `rocm mcp-call` ──────────────────────

/// A directory that exists wherever the suite runs: this crate's own.
const EXISTING_DIR: &str = env!("CARGO_MANIFEST_DIR");

fn missing_path() -> String {
    std::path::Path::new(EXISTING_DIR)
        .join("e2e-no-such-path")
        .display()
        .to_string()
}

fn run_path_exists(world: &mut E2eWorld, arguments: &serde_json::Value) {
    let arguments = arguments.to_string();
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["mcp-call", "path_exists", "--arguments-json", &arguments],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("a tool-using client hands the CLI a whole tool-call request to check a path that exists")]
async fn mcp_call_with_a_params_envelope(world: &mut E2eWorld) {
    // An MCP `tools/call` request's params, as a client received them: the
    // tool's name, its arguments, and the protocol's optional `_meta`.
    run_path_exists(
        world,
        &serde_json::json!({
            "name": "path_exists",
            "arguments": { "path": EXISTING_DIR },
            "_meta": { "progressToken": 1 },
        }),
    );
}

#[when(
    "a tool-using client asks the CLI to check a path with a second set of arguments nested inside the first"
)]
async fn mcp_call_with_nested_arguments(world: &mut E2eWorld) {
    // The outer path exists and the nested one does not, so a result for
    // either would show which set ran.
    run_path_exists(
        world,
        &serde_json::json!({
            "path": EXISTING_DIR,
            "arguments": { "path": missing_path() },
        }),
    );
}

#[then("the tool reports that path as an existing directory")]
async fn path_reported_as_existing_directory(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    let stderr = world.cli_stderr.as_deref().unwrap_or_default();
    assert_eq!(
        world.cli_rc,
        Some(0),
        "rocm mcp-call failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    let value: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("rocm mcp-call printed no JSON tool result ({e}):\n{stdout}"));
    let text = value["content"][0]["text"].as_str().unwrap_or_default();
    // The tool's own report, for the path inside the request's arguments.
    assert!(
        text.contains(&format!("path: {EXISTING_DIR}\n"))
            && text.contains("exists: true")
            && text.contains("kind: directory"),
        "the tool did not report the requested directory:\n{text}"
    );
}

#[then("the CLI refuses the request for its nested arguments")]
async fn mcp_call_refused_for_nested_arguments(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.as_deref().unwrap_or_default();
    assert_ne!(
        world.cli_rc,
        Some(0),
        "the request was not refused:\n{stderr}"
    );
    assert!(
        stderr.contains("must not contain a nested `arguments` field"),
        "the refusal does not name the nested arguments field:\n{stderr}"
    );
}

#[then("no tool result is printed")]
async fn no_tool_result_printed(world: &mut E2eWorld) {
    // The tool prints its result on stdout only when it ran; neither path's
    // report may appear.
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    assert!(
        stdout.trim().is_empty(),
        "a tool result was printed for a refused request:\n{stdout}"
    );
}
