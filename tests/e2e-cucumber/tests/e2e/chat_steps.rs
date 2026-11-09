// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::{
    MockServer, ServiceRecordOptions, chat_text_reply, chat_tool_call_reply,
};

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

// ── Model output reaching the terminal (chat-18, chat-19) ──────────

/// The model id the CLI's built-in assistant is served under; see the same
/// constant in `planner_steps.rs` for why the harness repeats it.
const BUILTIN_ASSISTANT_MODEL_ID: &str = "Qwen3-4B-Instruct-2507-GGUF";

/// A question none of the CLI's own shortcuts answers, so the assistant is
/// asked (the wording chat-11 uses, for the same reason).
const OPEN_QUESTION: &str = "I would like some help figuring out what to do with my GPU";

/// The text a hostile reply conceals with SGR 8. On a terminal that honours it,
/// the line reads as if this were not there.
const CONCEALED_REPLY_TEXT: &str = " --port 9";

/// A reply that hides [`CONCEALED_REPLY_TEXT`], retitles the window (OSC 0),
/// writes the clipboard (OSC 52) and repeats the title in the 8-bit form: the
/// shapes a model can use to make the terminal show something other than what
/// was sent.
fn hostile_reply() -> String {
    format!(
        "Run rocm serve qwen\u{1b}[8m{CONCEALED_REPLY_TEXT}\u{1b}[28m --managed\n\
         \u{1b}]0;rocm\u{7}\u{1b}]52;c;aGVsbG8=\u{7}\u{9d}0;rocm\u{9c}Done."
    )
}

/// Start a scripted mock as the ready built-in assistant: it answers each chat
/// completion with the next entry of `script`, and an unscripted one with an
/// HTTP 500, so a CLI that asked more often than expected fails loudly.
async fn start_scripted_assistant(world: &mut E2eWorld, script: Vec<serde_json::Value>) {
    let mock = MockServer::start_with_chat_script(BUILTIN_ASSISTANT_MODEL_ID, script).await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some(BUILTIN_ASSISTANT_MODEL_ID.to_string());
    world.mock = Some(mock);
    world.register_mock_service_with(ServiceRecordOptions {
        // Without the latch the CLI's readiness check sends its own completion
        // probe first, which would consume the first scripted reply.
        inference_verified: true,
        ..ServiceRecordOptions::default()
    });
}

#[given("a local chat assistant whose reply hides part of itself with terminal escape sequences")]
async fn assistant_reply_hides_part_of_itself(world: &mut E2eWorld) {
    start_scripted_assistant(
        world,
        vec![chat_text_reply(
            BUILTIN_ASSISTANT_MODEL_ID,
            &hostile_reply(),
        )],
    )
    .await;
}

#[given(
    "a local chat assistant that asks to install ROCm into a folder whose name holds an ampersand"
)]
async fn assistant_asks_for_an_ampersand_folder(world: &mut E2eWorld) {
    let prefix = crate::e2e::therock_steps::ampersand_folder(world)
        .to_string_lossy()
        .into_owned();
    start_scripted_assistant(
        world,
        vec![chat_tool_call_reply(
            BUILTIN_ASSISTANT_MODEL_ID,
            "install_sdk",
            &serde_json::json!({
                "channel": "release",
                "format": "wheel",
                "prefix": prefix,
            }),
        )],
    )
    .await;
}

#[when("the user asks the chat assistant with ROCm tools a question in one-shot chat")]
async fn user_asks_the_assistant_a_question(world: &mut E2eWorld) {
    // The exit code is checked by the `Then` steps: chat-19 expects a refusal.
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "chat",
            "--tools",
            "--provider",
            "local",
            "--prompt",
            OPEN_QUESTION,
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// The characters a terminal acts on, in whatever the CLI wrote: everything
/// `char::is_control` covers apart from newline and tab.
fn terminal_controls(text: &str) -> Vec<char> {
    text.chars()
        .filter(|character| character.is_control() && !matches!(character, '\n' | '\t'))
        .collect()
}

#[then("the CLI output carries no terminal escape sequence")]
async fn cli_output_has_no_escape(world: &mut E2eWorld) {
    // The `When` step has already required a zero exit.
    let output = world.cli_output.as_deref().expect("no chat CLI output");
    // Proves the reply really went through this path: the scenario would
    // otherwise pass with the mock never consulted.
    assert!(
        output.contains("Run rocm serve qwen") && output.contains("Done."),
        "the CLI did not print the scripted reply:\n{output}"
    );
    let offending = terminal_controls(output);
    assert!(
        offending.is_empty(),
        "the CLI wrote control characters {offending:?} to the terminal:\n{output:?}"
    );
    // The payloads go with their sequences rather than being printed as text.
    assert!(
        !output.contains("aGVsbG8=") && !output.contains("0;rocm"),
        "an escape sequence's payload reached the output:\n{output:?}"
    );
}

#[then("the hidden part of the reply is shown in plain text")]
async fn hidden_reply_text_is_shown(world: &mut E2eWorld) {
    let output = world.cli_output.as_deref().expect("no chat CLI output");
    // Shown, not dropped along with its escape sequence: the user must see what
    // the model actually sent, where it sent it.
    assert!(
        output.contains(&format!(
            "Run rocm serve qwen{CONCEALED_REPLY_TEXT} --managed"
        )),
        "the concealed text must be printed where it was, in plain sight:\n{output:?}"
    );
}

#[then("the CLI refuses the assistant's folder and tells the user to supply it themselves")]
async fn cli_refuses_the_assistants_folder(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().expect("no command was run");
    let stderr = world.cli_stderr.as_deref().unwrap_or_default();
    let rc = world.cli_rc.expect("no command was run");
    assert_ne!(rc, 0, "the request must be refused:\n{stdout}\n{stderr}");
    let requests = world
        .mock
        .as_ref()
        .expect("no mock server running")
        .chat_requests();
    assert_eq!(
        requests.len(),
        1,
        "the assistant must have been asked once, and the refusal must come from its reply"
    );
    // The message, asserted with the state checked in the next step: it names
    // the character and the command the user can run with the value.
    assert!(
        stderr.contains("'&'")
            && stderr.contains(
                "If you meant that value, supply it yourself by running `rocm install sdk` with it"
            ),
        "the refusal must name the character and say how to supply the value:\n{stderr}"
    );
    assert!(
        terminal_controls(stderr).is_empty(),
        "the refusal reached the terminal with control characters:\n{stderr:?}"
    );
}

#[then("no command line holding that folder is offered to paste, and nothing is installed")]
async fn nothing_offered_and_nothing_installed(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().expect("no command was run");
    assert!(
        !stdout.contains("advanced manual command")
            && !stdout.contains(crate::e2e::therock_steps::AMPERSAND_FOLDER_NAME),
        "no line naming the assistant's folder may be offered:\n{stdout}"
    );
    let folder = crate::e2e::therock_steps::ampersand_folder(world);
    assert!(
        !folder.exists(),
        "nothing may be installed into {}",
        folder.display()
    );
}
