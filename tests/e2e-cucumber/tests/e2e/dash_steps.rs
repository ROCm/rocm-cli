// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the interactive dashboard/chat TUI, driven black-box through a
//! pseudo-terminal (see `tui_driver`). These are the only steps that exercise
//! the real crossterm event loop end to end — launch, key input, rendered
//! screen, and clean exit — which the piped-`Command` steps structurally cannot.

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::{MetricsMode, MockServer, ServiceRecordOptions};
use std::time::{Duration, Instant};

use crate::E2eWorld;
use crate::e2e::tui_driver::{TuiSession, default_timeout};

/// The exact prompt `send_managed_model_message` types, and the string the
/// corresponding `Then` step (`managed_chat_request_carried_prompt`) asserts
/// the mock actually received — so the two can never silently drift apart.
const MANAGED_MODEL_PROMPT: &str = "hello from the terminal";

/// Borrow the scenario's active TUI session, or fail clearly if none was opened.
const fn session(world: &mut E2eWorld) -> &mut TuiSession {
    world
        .tui
        .as_mut()
        .expect("no interactive TUI session is open for this scenario")
}

// ── Given ──────────────────────────────────────────────────────────

#[given("interactive chat uses an offline assistant")]
async fn chat_offline(world: &mut E2eWorld) {
    // The offline assistant is the CLI's own `--chat-mock` backend: consent is
    // pre-accepted and a fixed reply is returned, so the journey needs no live
    // model or network. The launch step reads this flag.
    world.chat_use_mock = true;
}

async fn setup_managed_model(
    world: &mut E2eWorld,
    options: ServiceRecordOptions,
    with_metrics: bool,
) {
    let model = "TestModel/E2E-1B";
    let mock = if with_metrics {
        MockServer::start_with_metrics(model).await
    } else {
        MockServer::start(model).await
    };
    world.endpoint = Some(mock.base_url());
    world.model_name = Some(model.to_string());
    world.mock = Some(mock);
    world.register_mock_service_with(options);
}

#[given("a managed model is still loading")]
async fn managed_model_loading(world: &mut E2eWorld) {
    setup_managed_model(
        world,
        ServiceRecordOptions {
            status: "starting",
            startup_phase: Some("loading"),
            ..ServiceRecordOptions::default()
        },
        false,
    )
    .await;
}

#[given("a managed model exposes serving metrics")]
async fn managed_model_with_metrics(world: &mut E2eWorld) {
    setup_managed_model(world, ServiceRecordOptions::default(), true).await;
}

#[given("a running managed model is available locally")]
async fn running_managed_model(world: &mut E2eWorld) {
    setup_managed_model(world, ServiceRecordOptions::default(), false).await;
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user opens the dashboard with demo data")]
async fn open_dashboard_demo(world: &mut E2eWorld) {
    // `--demo` replays a deterministic synthetic session, so the dashboard
    // populates with no GPU and no daemon — a stable, mock-tier target.
    let session = TuiSession::spawn(world, &["dash", "--demo"])
        .unwrap_or_else(|e| panic!("failed to open the dashboard: {e}"));
    world.tui = Some(session);
}

#[when("the user opens interactive chat")]
async fn open_chat(world: &mut E2eWorld) {
    let args: &[&str] = if world.chat_use_mock {
        &["chat", "--chat-mock"]
    } else {
        &["chat"]
    };
    let session = TuiSession::spawn(world, args)
        .unwrap_or_else(|e| panic!("failed to open interactive chat: {e}"));
    world.tui = Some(session);
}

#[when("the user opens the dashboard")]
async fn open_dashboard(world: &mut E2eWorld) {
    let tui = TuiSession::spawn(world, &["dash"])
        .unwrap_or_else(|e| panic!("failed to open the dashboard: {e}"));
    world.tui = Some(tui);
}

#[when("the user opens the ROCm view")]
async fn open_rocm_view(world: &mut E2eWorld) {
    // Dashboard tabs are currently ordered Home, ROCm, Serving, Observe; these
    // numeric shortcuts intentionally exercise that user-visible ordering.
    session(world)
        .send("2")
        .unwrap_or_else(|e| panic!("failed to switch to the ROCm tab: {e}"));
}

#[when("the user opens the Observe view")]
async fn open_observe_view(world: &mut E2eWorld) {
    let tui = session(world);
    tui.use_detail_size()
        .unwrap_or_else(|e| panic!("failed to enlarge the dashboard: {e}"));
    // Every scenario using this step launches the dashboard and comes straight
    // here, with no assertion in between to prove the TUI is reading input yet
    // (unlike the ROCm journey, which asserts the home view first). A key
    // written that early can be swallowed before the event loop exists, so
    // repeat it until the Observe tab is actually selected — the `●` marks the
    // active chip. The step then fails only if the dashboard never gets there,
    // not if it was slow to start.
    tui.send_until("4", "● Observe", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("failed to switch to the Observe tab: {e}"));
}

#[when("the user opens dashboard help")]
async fn open_dashboard_help(world: &mut E2eWorld) {
    session(world)
        .send("?")
        .unwrap_or_else(|e| panic!("failed to open dashboard help: {e}"));
}

#[when("the user closes dashboard help")]
async fn close_dashboard_help(world: &mut E2eWorld) {
    session(world)
        .send("?")
        .unwrap_or_else(|e| panic!("failed to close dashboard help: {e}"));
}

#[when("the user opens the command palette")]
async fn open_command_palette(world: &mut E2eWorld) {
    session(world)
        .send(":")
        .unwrap_or_else(|e| panic!("failed to open the command palette: {e}"));
}

#[when("the user chooses Serving")]
async fn choose_serving(world: &mut E2eWorld) {
    let tui = session(world);
    // The palette initially selects Home; Serving is the third destination, so
    // two downward moves intentionally assert the current destination ordering.
    tui.send("jj")
        .unwrap_or_else(|e| panic!("failed to select Serving: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to open Serving: {e}"));
}

#[when("the user accepts the local endpoint")]
async fn accept_local_endpoint(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("Your request only leaves this machine", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("local endpoint consent did not appear: {e}"));
    tui.send("y")
        .unwrap_or_else(|e| panic!("failed to accept local endpoint: {e}"));
}

#[when("the user sends a message to the managed model")]
async fn send_managed_model_message(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("No messages yet.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat surface never became ready: {e}"));
    // No `i` here: accepting local-endpoint consent (`accept_chat_consent`) already
    // focused the input, so an extra `i` would be typed as a literal character
    // instead of a focus gesture (contrast the offline path in
    // `send_gpu_message`, whose `--chat-mock` consent does not focus the input).
    tui.send(MANAGED_MODEL_PROMPT)
        .unwrap_or_else(|e| panic!("failed to type the chat message: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to submit the chat message: {e}"));
}

#[when("the user sends a message about GPU health")]
async fn send_gpu_message(world: &mut E2eWorld) {
    let tui = session(world);
    // Wait for the accepted, empty chat surface before typing so the input is
    // ready to receive focus.
    tui.wait_for_screen("No messages yet.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat surface never became ready: {e}"));
    // `i` focuses the input; then the message, then Enter to submit.
    tui.send("i")
        .unwrap_or_else(|e| panic!("failed to focus the chat input: {e}"));
    tui.send("how is gpu-2 doing")
        .unwrap_or_else(|e| panic!("failed to type the chat message: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to submit the chat message: {e}"));
}

async fn quit_tui(world: &mut E2eWorld, surface: &str) {
    session(world)
        .quit_and_wait(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("{surface} did not exit cleanly: {e}"));
}

#[when("the user quits the dashboard")]
async fn quit_dashboard(world: &mut E2eWorld) {
    quit_tui(world, "the dashboard").await;
}

#[when("the user quits interactive chat")]
async fn quit_interactive_chat(world: &mut E2eWorld) {
    quit_tui(world, "interactive chat").await;
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the dashboard home view is displayed")]
async fn home_view_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    // The Home tab's summary cards (Running / Health / Updates) are drawn at any
    // size; the wider "GPU UTILIZATION" hero is not, so assert on the cards.
    tui.wait_for_screen("Updates", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the home view did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("Running") && screen.contains("Health"),
        "home summary cards missing:\n{screen}"
    );
}

#[then("ROCm setup actions are displayed")]
async fn rocm_actions_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Set up / Install ROCm", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the ROCm setup actions did not appear: {e}"));
}

#[then("the assistant's GPU status response is displayed")]
async fn gpu_response_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("GPU-2 is running hot", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the assistant's response did not appear: {e}"));
}

#[then("the managed model's response is displayed")]
async fn managed_model_response_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("mock response for testing", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the managed model's response did not appear: {e}"));
}

#[then("the mock received the typed prompt")]
async fn managed_chat_request_carried_prompt(world: &mut E2eWorld) {
    // The canned reply above is fixed regardless of what was sent, so it alone
    // can't prove the TUI actually submitted `MANAGED_MODEL_PROMPT` (a stray
    // keystroke corrupting the prompt would still show that reply). Assert on
    // the request the mock actually received instead. `wait_for_chat_request`
    // polls rather than reading a single snapshot: the response already
    // rendering on screen only proves the reply arrived, not that the mock's
    // handler finished recording the request into shared state first.
    let body = world
        .mock
        .as_ref()
        .expect("no mock server running")
        .wait_for_chat_request(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the mock never received a chat request: {e}"));
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
        last_user_content, MANAGED_MODEL_PROMPT,
        "mock did not receive the exact typed prompt; full request:\n{body}"
    );
}

#[then("the managed model is shown as loading rather than ready")]
async fn managed_model_shown_loading(world: &mut E2eWorld) {
    let model = world.model_name.clone().expect("no model name set");
    let tui = session(world);
    tui.wait_for_screen(&model, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the managed model did not appear: {e}"));
    // The compact Observe table intentionally omits lifecycle status; opening
    // the selected instance exposes the status a user uses to diagnose startup.
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to open instance details: {e}"));
    tui.wait_for_screen("LOADING", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the loading state did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(screen.contains(&model), "managed model missing:\n{screen}");
    assert!(
        !screen
            .lines()
            .any(|line| line.contains(&model) && line.contains("READY")),
        "loading model was presented as ready:\n{screen}"
    );
}

#[then("live serving metrics are displayed for the managed model")]
async fn managed_model_metrics_displayed(world: &mut E2eWorld) {
    let model = world.model_name.clone().expect("no model name set");
    let tui = session(world);
    tui.wait_for_screen(&model, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the managed model did not appear: {e}"));
    tui.wait_for_screen("50ms", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("TTFT metrics did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(screen.contains("20ms"), "TPOT metrics missing:\n{screen}");
    let row = screen
        .lines()
        .find(|line| line.contains(&model))
        .unwrap_or_default();
    assert!(
        row.contains("50ms") && row.contains("20ms") && row.contains("1/0") && row.contains("25%"),
        "managed serving metrics were incomplete:\n{screen}"
    );
}

#[then("navigation and next-step guidance are displayed")]
async fn navigation_guidance_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("toggle this help", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("dashboard help did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("next / previous tab") && screen.contains("Home tab"),
        "navigation or contextual guidance missing:\n{screen}"
    );
}

#[then("dashboard destinations are displayed")]
async fn dashboard_destinations_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("Go to", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("command palette did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("Home") && screen.contains("Serving") && screen.contains("Observe"),
        "command-palette destinations missing:\n{screen}"
    );
}

#[then("Serving actions are displayed")]
async fn serving_actions_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Serving actions", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("Serving actions did not appear: {e}"));
}

#[then("the managed model is displayed")]
async fn managed_model_displayed(world: &mut E2eWorld) {
    let model = world
        .model_name
        .as_deref()
        .expect("no model name set")
        .to_string();
    session(world)
        .wait_for_screen(&model, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("managed model did not appear: {e}"));
}

fn assert_tui_opened(world: &E2eWorld) {
    // `quit_and_wait` (in the "quits" step) already reaped the process and
    // asserted a zero exit; reaching here means the whole launch→interact→quit
    // round trip through the real terminal succeeded.
    assert!(
        world.tui.is_some(),
        "no TUI session was opened for this scenario"
    );
}

#[then("the dashboard exits successfully")]
async fn dashboard_exited(world: &mut E2eWorld) {
    assert_tui_opened(world);
}

#[then("interactive chat exits successfully")]
async fn interactive_chat_exited(world: &mut E2eWorld) {
    assert_tui_opened(world);
}

// ── Then: chat privacy consent gate (real endpoint) ──

#[then("the local endpoint is shown for confirmation")]
async fn local_endpoint_shown(world: &mut E2eWorld) {
    // The consent gate echoes the detected endpoint; its port is the mock's
    // OS-assigned one, proving the CLI discovered the planted managed service
    // (not a hard-coded default) before offering it.
    let port = world
        .mock
        .as_ref()
        .expect("no mock server running")
        .port()
        .to_string();
    session(world)
        .wait_for_screen(&port, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the detected endpoint was not shown: {e}"));
}

#[then("the privacy notice is shown before any message is sent")]
async fn privacy_notice_shown(world: &mut E2eWorld) {
    // The gate is reached before any message can be submitted, so seeing this
    // line proves the notice precedes the first request.
    session(world)
        .wait_for_screen(
            "Your request only leaves this machine after you accept.",
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| panic!("the privacy notice was not shown: {e}"));
}

// ── EAI-7960: scripted metrics / validity-window regression ────────────────

/// Start the mock in Growing mode so the daemon builds a positive gen_tps
/// baseline before the scenario injects the Failure transition.
#[given("a managed model exposes scripted serving metrics")]
async fn managed_model_scripted_metrics(world: &mut E2eWorld) {
    let model = "TestModel/E2E-1B";
    let mock = MockServer::start_with_scripted_metrics(model).await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some(model.to_string());
    world.mock = Some(mock);
    world.register_mock_service_with(ServiceRecordOptions::default());
}

/// The Observe tab's node-throughput hero shows the "tok/s" unit whenever
/// `gen_tps` is `Some(_)`. Wait for it to confirm a positive baseline was
/// established through at least two successful Growing-mode scrapes.
#[then("positive generation throughput is displayed for the managed model")]
async fn positive_gen_tps_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("tok/s", default_timeout())
        .await
        .unwrap_or_else(|e| {
            panic!("positive gen_tps (\"tok/s\") never appeared after Growing-mode scrapes: {e}")
        });
}

/// Switch the scripted mock to Failure mode, then poll the mock's own failure
/// counter until the daemon delivers at least one 503 — confirming the failure
/// scrape actually landed before the assertion checks the TUI. This avoids a
/// fixed wall-time sleep while remaining deterministic.
#[when("the metrics endpoint fails transiently")]
async fn metrics_endpoint_fails(world: &mut E2eWorld) {
    let mock = world.mock.as_ref().expect("no mock server running");
    mock.set_metrics_mode(MetricsMode::Failure);

    // Poll until the daemon delivers at least one 503 to the mock endpoint.
    // The production instance_tick is 2 s, so this converges in 2–3 s.
    let budget = default_timeout();
    let deadline = Instant::now() + budget;
    loop {
        if mock.metrics_failure_count() >= 1 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "scripted failure was never served within {budget:?}; \
             check instance_tick and scrape cadence"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Allow one TUI render cycle (50 ms >> 20 ms poll) so the failure
    // snapshot is painted before the assertion reads the screen.
    tokio::time::sleep(Duration::from_millis(50)).await;
}

/// EAI-7960 principal regression assertion (must be RED with current code).
///
/// Contract: the Observe tab must still show "tok/s" immediately after the
/// first failed scrape — the held value must persist for the validity window
/// `clamp(3 × instance_tick, 6 s, 30 s)` before clearing.
///
/// **Current behaviour:** `runner.rs` lines 464-476 clear `gen_tps` on the
/// very tick that the `/metrics` fetch fails — no holding logic exists. The
/// TUI therefore renders "—" the moment the failure propagates, and this
/// assertion **FAILS**, confirming EAI-7960 is reproduced at the PTY seam.
#[then("generation throughput remains visible within the validity window")]
async fn gen_tps_held_after_failure(world: &mut E2eWorld) {
    let screen = session(world).screen_text();
    assert!(
        screen.contains("tok/s"),
        "EAI-7960 REGRESSION: gen throughput (\"tok/s\") was cleared immediately \
         after the first failed scrape instead of being held for the validity \
         window (clamp(3 × instance_tick, 6 s, 30 s)).\n\
         Root cause: runner.rs clears gen_tps on the same tick as the failure; \
         no held-value / validity-window logic exists yet.\n\
         This assertion must FAIL (RED) until the fix is applied.\n\n\
         Last screen:\n{screen}"
    );
}

// ── EAI-7960: expiry boundary helpers ───────────────────────────────────────

/// Production validity window: clamp(3 × instance_tick, 6 s, 30 s).
///
/// With the daemon's 2 s `instance_tick` the lower bound clamp(6 s, 6 s) = 6 s
/// is always reached. An additional buffer of two instance-ticks (4 s) ensures
/// the runner has had enough cycles to propagate the expiry to the TUI.
const VALIDITY_WINDOW: Duration = Duration::from_secs(6);
const VALIDITY_WINDOW_BUFFER: Duration = Duration::from_secs(5); // 2 × instance_tick + render

/// Sleep for the full observation validity window so the caller can then assert
/// that the held gen_tps has expired. Designed to follow
/// "When the metrics endpoint fails transiently" — at that step's exit at least
/// one 503 has been served, meaning the validity clock has started.
#[when("the validity window has elapsed")]
async fn validity_window_elapsed(_world: &mut E2eWorld) {
    // Sleep the full window + buffer so the daemon has had enough cycles
    // after expiry to deliver the snapshot change to the TUI.
    tokio::time::sleep(VALIDITY_WINDOW + VALIDITY_WINDOW_BUFFER).await;
}

/// Assert that gen_tps is no longer rendered on screen (BOUNDARY 2 of the
/// EAI-7960 expiry contract). After the validity window the daemon must clear
/// the held value and the TUI must show "—" in place of the "tok/s" unit.
///
/// With current code this step is unreachable because BOUNDARY 1 (the "remains
/// visible" assertion) fails first. This step becomes GREEN once the hold/expiry
/// logic is implemented.
#[then("generation throughput is no longer displayed")]
async fn gen_tps_no_longer_displayed(world: &mut E2eWorld) {
    let screen = session(world).screen_text();
    assert!(
        !screen.contains("tok/s"),
        "EAI-7960 BOUNDARY-2: gen_tps (\"tok/s\") is still visible after the \
         validity window clamp(3 × instance_tick, 6 s, 30 s) elapsed.\n\
         Expected the daemon to have cleared the held value and the TUI to \
         show the unavailable placeholder.\n\n\
         Last screen:\n{screen}"
    );
}
