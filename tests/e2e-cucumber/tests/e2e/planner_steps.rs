// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the planning-assistant scenarios in `chat.feature`: a request the
//! CLI cannot plan on its own is handed to a configured local assistant, and
//! must be handed over exactly once, with the plan the assistant gave being the
//! plan the user is shown.
//!
//! The mock answers from a script (`MockServer::start_with_chat_script`), one
//! reply per completion request, so each call gets a *different* answer. A
//! request planned twice therefore shows up twice — in the request count and in
//! which plan is printed — rather than hiding behind two identical replies.
//!
//! What this cannot show black-box: that the plan shown is also the action that
//! would run. `--yes` refuses a provider-assisted plan before dispatching it,
//! and the chat tool only returns a fixed confirmation string, so neither
//! surface ever executes or reports the plan's command. That half of the
//! guarantee stays covered by the unit tests in `apps/rocm/src/providers.rs`
//! (`freeform_plan_dedup_prevents_provider_assisted_divergence` and
//! `freeform_execution_reuses_the_rendered_plan_without_rebuilding`).

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::{MockServer, ServiceRecordOptions, chat_text_reply};
use serde_json::{Value, json};

use crate::E2eWorld;

/// The model id the CLI's built-in assistant is served under.
///
/// The harness repeats the value because it is black-box and cannot import the
/// product's constant (`LEMONADE_ASSISTANT_MODEL_ID` in
/// `apps/rocm/src/providers.rs`, also exposed as `BUILTIN_ASSISTANT_MODEL_ID`).
/// It has to match: a planner call with no explicit model only accepts a ready
/// service recorded under this id, so the planted service must carry it or the
/// CLI answers "no ready managed service" instead of calling the mock.
const BUILTIN_ASSISTANT_MODEL_ID: &str = "Qwen3-4B-Instruct-2507-GGUF";

/// A request the deterministic planner cannot settle: no model is named, so its
/// plan is a `rocm serve <model>` placeholder at medium confidence and the CLI
/// asks the configured planning assistant instead. (A request it *can* settle,
/// such as "start a local model", never reaches the assistant and — worse under
/// `--yes` — dispatches a real `rocm serve`.)
const VAGUE_REQUEST: &str = "serve a model";

/// Opening words of the prompt the CLI sends the planning assistant, which is
/// how a planning request is told apart from the chat assistant's own
/// completions. The text is the CLI's own; this is a plain substring match.
const PLANNER_PROMPT_MARKER: &str = "resolving an ambiguous rocm-cli request";

/// The two plans the assistant gives, in the order it gives them. Different on
/// purpose: a second planning call would print the second.
const FIRST_PLAN_COMMAND: &str = "rocm examine";
const SECOND_PLAN_COMMAND: &str = "rocm engines list";

fn plan_reply(args: &[&str]) -> Value {
    let plan = json!({
        "intent": "inspect",
        "tool_call": {"tool": "rocm", "args": args}
    });
    chat_text_reply(BUILTIN_ASSISTANT_MODEL_ID, &plan.to_string())
}

fn examine_plan_reply() -> Value {
    plan_reply(&["examine"])
}

fn engines_list_plan_reply() -> Value {
    plan_reply(&["engines", "list"])
}

/// Start the scripted mock, plant it as the ready built-in assistant, and make
/// it the planner provider through the real `rocm config` command.
async fn start_planning_assistant(world: &mut E2eWorld, script: Vec<Value>) {
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
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "set-planner-provider", "local"]);
    assert!(
        rc == 0,
        "could not configure the planner provider (rc={rc}):\n{stdout}\n{stderr}"
    );
}

fn mock(world: &E2eWorld) -> &MockServer {
    world.mock.as_ref().expect("no planning assistant running")
}

/// The text of every message in a completion request, in order.
fn message_texts(request: &Value) -> Vec<&str> {
    request["messages"]
        .as_array()
        .map(|messages| {
            messages
                .iter()
                .filter_map(|message| message["content"].as_str())
                .collect()
        })
        .unwrap_or_default()
}

/// Whether `request` is a planning request: a *user* turn carrying the planner
/// prompt. Only user turns count, so a system prompt that happened to mention
/// the phrase could not pass for one.
fn is_planning_request(request: &Value) -> bool {
    request["messages"].as_array().is_some_and(|messages| {
        messages.iter().any(|message| {
            message["role"] == "user"
                && message["content"]
                    .as_str()
                    .is_some_and(|text| text.contains(PLANNER_PROMPT_MARKER))
        })
    })
}

/// A one-line summary of each request the assistant received, for failure
/// messages: which were planning requests and what the others asked.
fn describe_requests(requests: &[Value]) -> String {
    requests
        .iter()
        .enumerate()
        .map(|(index, request)| {
            let kind = if is_planning_request(request) {
                "planning request"
            } else {
                "other request"
            };
            let texts = message_texts(request);
            let last = texts.last().copied().unwrap_or("");
            let preview: String = last.chars().take(80).collect();
            format!("  #{}: {kind} (last message: {preview:?})", index + 1)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn cli_output(world: &E2eWorld) -> &str {
    world.cli_output.as_deref().expect("no command was run")
}

// ── Given ──────────────────────────────────────────────────────────

#[given("a local planning assistant that suggests a different plan each time it is asked")]
async fn planning_assistant_with_distinct_plans(world: &mut E2eWorld) {
    // Only the planning request reaches it; a third would be an error, so asking
    // more often than expected fails rather than recycling a plan.
    start_planning_assistant(world, vec![examine_plan_reply(), engines_list_plan_reply()]).await;
}

#[given(
    "a local assistant that hands the request to the planner, which suggests a different plan each time it is asked"
)]
async fn assistant_that_asks_the_planner(world: &mut E2eWorld) {
    // Three completions are expected, in this order: the chat assistant asking
    // for the request to be planned, the planner answering, and the chat
    // assistant's closing answer. A fourth would be a second planning call.
    let ask_for_a_plan = e2e_cucumber::mock_server::chat_tool_call_reply(
        BUILTIN_ASSISTANT_MODEL_ID,
        "natural_language_plan",
        &json!({"request": VAGUE_REQUEST}),
    );
    start_planning_assistant(
        world,
        vec![
            ask_for_a_plan,
            examine_plan_reply(),
            chat_text_reply(
                BUILTIN_ASSISTANT_MODEL_ID,
                "I looked at what the plan would do. Nothing was changed.",
            ),
        ],
    )
    .await;
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user approves a vague natural-language request with --yes")]
async fn user_approves_vague_request(world: &mut E2eWorld) {
    // Not `run_rocm_ok`: the request is expected to be refused (non-zero).
    let (stdout, stderr, rc) = crate::run_rocm(world, &["--yes", VAGUE_REQUEST]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user asks the assistant in one-shot chat for help with a vague request")]
async fn user_asks_chat_assistant_for_help(world: &mut E2eWorld) {
    // The wording avoids every phrase the CLI answers without consulting the
    // assistant (anything naming `serve`, for one), so the assistant is asked.
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "chat",
            "--tools",
            "--provider",
            "local",
            "--prompt",
            "I would like some help figuring out what to do with my GPU",
        ],
    );
    // The exit code is checked by the `Then` step that reads the chat output,
    // after the request-count steps: a chat that asks for more completions than
    // were scripted fails with an opaque HTTP 500, and the counts say why.
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the planning assistant was asked exactly once")]
async fn planning_assistant_asked_once(world: &mut E2eWorld) {
    let requests = mock(world).chat_requests();
    let planning = requests
        .iter()
        .filter(|request| is_planning_request(request))
        .count();
    assert_eq!(
        planning,
        1,
        "the request must be planned exactly once, but the assistant received {} completions \
         ({planning} of them planning requests):\n{}",
        requests.len(),
        describe_requests(&requests)
    );
}

#[then("the chat assistant was asked once before the plan and once after it")]
async fn chat_assistant_asked_around_the_plan(world: &mut E2eWorld) {
    let requests = mock(world).chat_requests();
    let kinds: Vec<bool> = requests.iter().map(is_planning_request).collect();
    assert_eq!(
        kinds,
        [false, true, false],
        "expected the chat assistant, then the planner, then the chat assistant again \
         (false = chat, true = planning), got {} completions:\n{}",
        requests.len(),
        describe_requests(&requests)
    );
}

#[then("the plan shown is the one the assistant gave")]
async fn plan_shown_is_the_assistants(world: &mut E2eWorld) {
    let out = cli_output(world);
    // Both halves matter: the planner line proves the plan came from the
    // assistant at all (the deterministic plan prints `hybrid-parser-v1` alone),
    // and the whole `next_tool_call` line proves *which* of the assistant's
    // answers it is. A substring match would also pass on the deterministic
    // plan, which proposes `rocm serve <model>`, and on the second plan, which
    // is a different command.
    assert!(
        out.contains("planner: hybrid-parser-v1 + provider:local"),
        "the plan was not produced with the planning assistant:\n{out}"
    );
    let next: Vec<&str> = out
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("next_tool_call:"))
        .collect();
    assert_eq!(
        next,
        [format!("next_tool_call: {FIRST_PLAN_COMMAND}")],
        "the plan shown must be the assistant's first answer (`{FIRST_PLAN_COMMAND}`, not the \
         `{SECOND_PLAN_COMMAND}` a second planning call would give):\n{out}"
    );
}

#[then("the request is refused rather than run without review")]
async fn request_refused_without_running(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command was run");
    let out = cli_output(world);
    let stderr = world.cli_stderr.as_deref().unwrap_or_default();
    // The behavior: a non-zero exit and no execution section. The message is
    // asserted alongside them and certifies nothing on its own — it is the
    // refusal's wording, and the two checks before it are what show that
    // nothing ran.
    assert_ne!(
        rc, 0,
        "a provider-assisted plan must not succeed under --yes"
    );
    assert!(
        !out.contains("\nexecution\n"),
        "nothing may be executed for a provider-assisted plan:\n{out}"
    );
    assert!(
        stderr.contains("reviewed interactively"),
        "the refusal must say the plan is to be reviewed interactively, got stderr:\n{stderr}"
    );
}

#[then("the chat reports that the request was planned")]
async fn chat_reports_the_plan(world: &mut E2eWorld) {
    let out = cli_output(world);
    let rc = world.cli_rc.expect("no command was run");
    let stderr = world.cli_stderr.as_deref().unwrap_or_default();
    assert!(rc == 0, "rocm chat failed (rc={rc}):\n{out}\n{stderr}");
    // The chat's own label for the planning check it ran and the status it
    // finished with; asserted next to the request counts above, which are what
    // show the plan was made once.
    assert!(
        out.contains("Planned ROCm request: done"),
        "the chat did not report running the planning check:\n{out}"
    );
}
