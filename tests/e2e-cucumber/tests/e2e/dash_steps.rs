// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the interactive dashboard/chat TUI, driven black-box through a
//! pseudo-terminal (see `tui_driver`). These are the only steps that exercise
//! the real crossterm event loop end to end — launch, key input, rendered
//! screen, and clean exit — which the piped-`Command` steps structurally cannot.

use cucumber::{given, then, when};

use crate::E2eWorld;
use crate::e2e::tui_driver::{DEFAULT_TIMEOUT, TuiSession};

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

#[when("the user opens the ROCm view")]
async fn open_rocm_view(world: &mut E2eWorld) {
    session(world)
        .send("2")
        .unwrap_or_else(|e| panic!("failed to switch to the ROCm tab: {e}"));
}

#[when("the user sends a message about GPU health")]
async fn send_gpu_message(world: &mut E2eWorld) {
    let tui = session(world);
    // Wait for the accepted, empty chat surface before typing so the input is
    // ready to receive focus.
    tui.wait_for_screen("No messages yet.", DEFAULT_TIMEOUT)
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

#[when("the user quits the dashboard")]
async fn quit_dashboard(world: &mut E2eWorld) {
    session(world)
        .quit_and_wait(DEFAULT_TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("the dashboard did not exit cleanly: {e}"));
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the dashboard home view is displayed")]
async fn home_view_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    // The Home tab's summary cards (Running / Health / Updates) are drawn at any
    // size; the wider "GPU UTILIZATION" hero is not, so assert on the cards.
    tui.wait_for_screen("Updates", DEFAULT_TIMEOUT)
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
        .wait_for_screen("Set up / Install ROCm", DEFAULT_TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("the ROCm setup actions did not appear: {e}"));
}

#[then("the assistant's GPU status response is displayed")]
async fn gpu_response_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("GPU-2 is running hot", DEFAULT_TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("the assistant's response did not appear: {e}"));
}

#[then("the dashboard exits successfully")]
async fn dashboard_exited(world: &mut E2eWorld) {
    // `quit_and_wait` (in the "quits" step) already reaped the process and
    // asserted a zero exit; reaching here means the whole launch→interact→quit
    // round trip through the real terminal succeeded.
    assert!(
        world.tui.is_some(),
        "no TUI session was opened for this scenario"
    );
}

// ── Then: chat privacy consent gate (real endpoint, EAI-7222) ──────

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
        .wait_for_screen(&port, DEFAULT_TIMEOUT)
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
            DEFAULT_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| panic!("the privacy notice was not shown: {e}"));
}
