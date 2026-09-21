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
use crate::e2e::tui_driver::{TermSignal, TuiSession, default_timeout};
/// The exact prompt `send_managed_model_message` types, and the string the
/// corresponding `Then` step (`managed_chat_request_carried_prompt`) asserts
/// the mock actually received — so the two can never silently drift apart.
const MANAGED_MODEL_PROMPT: &str = "hello from the terminal";
/// File the daemon's test-only logical clock reads every cycle (see
/// `rocm_dash_daemon::runner`'s `TestClockDirective` for the grammar).
const DASH_CLOCK_OFFSET_FILE: &str = "dash-clock-offset-secs";

/// The Observe instances table's TTFT cell while the scripted mock is serving:
/// its histogram pins time-to-first-token at exactly 50 ms
/// (`ttft_sum_s = ticks × 0.050` over `ttft_count = ticks`), and the cell is
/// rendered `"{v:.0}ms"`. A *failed* scrape clears `ttft_ms`/`tpot_ms`
/// (`runner.rs`), so this cell changing is the screen's own proof that the
/// frame on display was assembled after the failure — the only frame the
/// held-throughput assertion is about.
const SCRIPTED_TTFT_CELL: &str = "50ms";

/// Zero-based index of the TTFT cell within an Observe instances row, counting
/// from the model id: `MODEL TOK/S TOK/W TTFT TPOT POWER QUEUE KV%`
/// (`instances.rs`).
///
/// The cell is read by position on the scripted instance's own row rather than
/// matched as a substring of the whole screen. A bare substring cannot tell a
/// cleared cell from a surviving one: any future ms-suffixed value that merely
/// *contains* the scripted one (`150ms`, `250ms`), or a second row whose TTFT
/// is also 50 ms, would keep the marker on screen and time this step out for a
/// reason that has nothing to do with the scrape it synchronises on.
const TTFT_COLUMN: usize = 3;

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

#[when("the user replays a recording that does not exist")]
async fn replay_missing_recording(world: &mut E2eWorld) {
    // Drive this under a real pseudo-terminal, not a pipe. The point of the fix is
    // fail-fast *before the terminal takeover*, and that property is unobservable
    // through a pipe: a piped `dash` can't enter the alt-screen either way (and
    // the pre-fix binary already exits non-zero through a pipe when raw-mode
    // fails, so a piped run detects nothing). Under a PTY the pre-fix binary
    // enters the alt-screen (`ESC[?1049h`) and hangs — that is the regression this
    // scenario pins.
    let missing = std::env::temp_dir().join(format!(
        "rocm-cli-e2e-no-such-recording-eai-8366-{}.ndjson",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&missing);
    let path = missing.to_string_lossy().to_string();
    let session = TuiSession::spawn(world, &["dash", "--replay", &path])
        .unwrap_or_else(|e| panic!("failed to spawn dash under a pty: {e}"));
    world.tui = Some(session);
}

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

#[when("the user opens the launcher")]
async fn open_launcher(world: &mut E2eWorld) {
    // Bare `rocm` (no subcommand) opens the launcher front door under an
    // interactive terminal — the PTY slave satisfies `interactive_terminal()`.
    let tui = TuiSession::spawn(world, &[])
        .unwrap_or_else(|e| panic!("failed to open the launcher: {e}"));
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

#[when("the user opens the Chat view")]
async fn open_chat_view(world: &mut E2eWorld) {
    // Same resend-until-it-takes rationale as `open_observe_view`: nothing
    // before this step proves the event loop is reading input yet.
    session(world)
        .send_until("5", "● Chat", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("failed to switch to the Chat tab: {e}"));
}

#[when("the user opens instance detail")]
async fn open_instance_detail(world: &mut E2eWorld) {
    // `Enter` on the Observe tab opens the selected instance's detail popup
    // (`KeyAction::OpenDetail`); the demo session always seeds at least one
    // instance, so the default selection (index 0) is always present. Enter
    // toggles `Modal::Detail` open/closed, so it is NOT safe to resend via
    // `send_until` (its own doc comment restricts that to idempotent keys) —
    // a resend while the popup is already open would immediately close it.
    // Plain `send` + `wait_for_screen` instead.
    let tui = session(world);
    // The `● Observe` marker asserted by `open_observe_view` only proves the
    // tab switch rendered — the demo replay's `InstanceDiscovered` events
    // land afterward. Sending Enter before they do finds an empty instance
    // list (`selection_len()` == 0), so `OpenDetail` is silently ignored.
    // Wait for the populated table before the (non-retryable) Enter.
    tui.wait_for_screen("Instances · AI metrics", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("instance list did not populate: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to send Enter: {e}"));
    tui.wait_for_screen("Instance · ", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("instance detail did not open: {e}"));
}

#[when("the user shrinks the terminal until the detail body overflows")]
async fn shrink_until_detail_overflows(world: &mut E2eWorld) {
    // The demo fixtures' `launch_args`/`env_vars` are too few to overflow the
    // detail popup at either the default or the enlarged (`use_detail_size`)
    // geometry — this is the only terminal size small enough to force it.
    let tui = session(world);
    tui.use_overflow_size()
        .unwrap_or_else(|e| panic!("failed to shrink the dashboard: {e}"));
    // The resize is synchronous in the emulator but the app only learns of it
    // asynchronously via SIGWINCH, so this step does not itself prove a
    // redraw at the new geometry happened — `"Instance · "` was already on
    // screen before the resize (see `open_instance_detail`), so waiting on it
    // here is satisfied immediately regardless of whether the app redrew.
    // The `Then` step's own `wait_for_screen` on the scroll hint is what
    // actually gates on the post-resize render.
}

#[when("the user opens the services manager")]
async fn open_services_manager(world: &mut E2eWorld) {
    // Bound to `s` only on the Observe tab (`OpenServices`) — a manager opened
    // from a non-domain tab, which is exactly the case
    // `should_pane_back_out`'s doc comment calls out as needing Esc to close it.
    session(world)
        .send("s")
        .unwrap_or_else(|e| panic!("failed to open the services manager: {e}"));
}

#[when("the user presses Escape")]
async fn press_escape(world: &mut E2eWorld) {
    session(world)
        .send("\x1b")
        .unwrap_or_else(|e| panic!("failed to send Escape: {e}"));
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

#[when("the user opens the theme picker")]
async fn open_theme_picker(world: &mut E2eWorld) {
    let tui = session(world);
    // `t` toggles `Modal::ThemePicker` open/closed, so it is NOT safe to resend
    // via `send_until` (its own doc comment restricts that to idempotent
    // keys) — a resend after the picker is already open would immediately
    // close it. Nothing before this step proves the event loop is reading
    // input yet, so wait for the Home tab's readiness marker before the
    // (non-retryable) `t`, same rationale as `open_instance_detail`.
    tui.wait_for_screen("Updates", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("dashboard home view did not become ready: {e}"));
    tui.send("t")
        .unwrap_or_else(|e| panic!("failed to open the theme picker: {e}"));
    tui.wait_for_screen(
        "Theme — j/k select, Enter apply, Esc cancel",
        default_timeout(),
    )
    .await
    .unwrap_or_else(|e| panic!("theme picker did not open: {e}"));
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

#[when("the user sends a message that triggers a tool approval")]
async fn send_approval_trigger_message(world: &mut E2eWorld) {
    let tui = session(world);
    // Wait for the accepted, empty chat surface before typing so the input is
    // ready to receive focus.
    tui.wait_for_screen("No messages yet.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat surface never became ready: {e}"));
    // `i` focuses the input; then the message, then Enter to submit. The
    // phrase must match `MockAgentClient`'s trigger ("install the sdk") without
    // colliding with `send_gpu_message`'s "how is gpu-2 doing".
    tui.send("i")
        .unwrap_or_else(|e| panic!("failed to focus the chat input: {e}"));
    tui.send("please install the sdk")
        .unwrap_or_else(|e| panic!("failed to type the chat message: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to submit the chat message: {e}"));
}

#[when("the user confirms the approval prompt without moving the cursor")]
async fn confirm_approval_without_moving(world: &mut E2eWorld) {
    session(world)
        .send("\r")
        .unwrap_or_else(|e| panic!("failed to press Enter on the approval prompt: {e}"));
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

#[when("the user quits the launcher")]
async fn quit_launcher(world: &mut E2eWorld) {
    quit_tui(world, "the launcher").await;
}

/// Deliver a termination signal to the TUI under test and wait for it to exit,
/// stashing the observed exit code for the `Then` steps. Shared by the
/// SIGTERM/SIGINT `When` steps so the two cannot drift.
///
/// Deliberately not named for the dashboard: the process under test is the
/// launcher in the hub round-trip scenario, and the signal handling being
/// asserted is process-wide, not dashboard-specific.
async fn signal_tui(world: &mut E2eWorld, signal: TermSignal) {
    session(world)
        .deliver_signal_and_wait(signal, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the process under test did not exit after {signal:?}: {e}"));
}

#[when("the user opens the dashboard from the launcher")]
async fn open_dashboard_from_launcher(world: &mut E2eWorld) {
    let tui = session(world);
    // Sync on the launcher front door before sending a key, so `d` is not
    // swallowed before the launcher's synchronous event loop is reading input.
    tui.wait_for_screen("Set up this system", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the launcher front door never appeared: {e}"));
    // `d` escalates straight into the full dashboard (LauncherChoice::OpenDashboard),
    // which builds and then, on quit, drops its own Tokio runtime.
    tui.send("d")
        .unwrap_or_else(|e| panic!("failed to open the dashboard from the launcher: {e}"));
    tui.wait_for_screen("Updates", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the dashboard did not open from the launcher: {e}"));
}

#[when("the user quits back to the launcher")]
async fn quit_back_to_launcher(world: &mut E2eWorld) {
    let tui = session(world);
    // `q` quits the dashboard; the hub loop drops the session runtime and
    // redraws the launcher front door — the exact "back at the menu after a
    // session" state where a per-session signal watcher would have gone deaf.
    tui.send("q")
        .unwrap_or_else(|e| panic!("failed to quit the dashboard: {e}"));
    tui.wait_for_screen("Set up this system", default_timeout())
        .await
        .unwrap_or_else(|e| {
            panic!("the launcher front door did not return after the session: {e}")
        });
}

#[when("the launcher receives a SIGTERM")]
async fn launcher_receives_sigterm(world: &mut E2eWorld) {
    signal_tui(world, TermSignal::Term).await;
}

#[when("the dashboard receives a SIGTERM")]
async fn dashboard_receives_sigterm(world: &mut E2eWorld) {
    signal_tui(world, TermSignal::Term).await;
}

#[when("the dashboard receives a SIGINT")]
async fn dashboard_receives_sigint(world: &mut E2eWorld) {
    signal_tui(world, TermSignal::Int).await;
}

/// Type a literal Ctrl-C at the running TUI and wait for it to exit. Shared by
/// the dashboard and launcher wordings, which press the same key at the two
/// separate key loops the process runs.
async fn press_ctrl_c(world: &mut E2eWorld, subject: &str) {
    session(world)
        .press_ctrl_c_and_wait(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("{subject} did not exit after Ctrl-C: {e}"));
}

#[when("the user presses Ctrl-C in the dashboard")]
async fn dashboard_ctrl_c(world: &mut E2eWorld) {
    press_ctrl_c(world, "the dashboard").await;
}

#[when("the user presses Ctrl-C in the launcher")]
async fn launcher_ctrl_c(world: &mut E2eWorld) {
    press_ctrl_c(world, "the launcher").await;
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the dashboard is refused before taking over the terminal")]
async fn dashboard_refused_before_takeover(world: &mut E2eWorld) {
    // Fail-fast contract: the child must exit non-zero *promptly*. Under a PTY the
    // pre-fix binary takes over the terminal and hangs, so `wait_for_refusal`
    // times out there — the timeout IS the regression, not a flake.
    let tui = session(world);
    tui.wait_for_refusal(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    // And it never entered the alt-screen: the refusal happened before the
    // dashboard could take over the terminal.
    assert!(
        !tui.in_alternate_screen(),
        "dash entered the alt-screen before refusing a missing replay file:\n{}",
        tui.screen_text(),
    );
}

#[then("the user is told the replay file was not found")]
async fn told_replay_file_not_found(world: &mut E2eWorld) {
    // Read the screen from the PTY session itself rather than stashing it in
    // `world.cli_output`, which carries piped stdout for the non-PTY steps.
    // Draining first lets the reader thread commit the final buffered frame, so
    // this does not race the PTY being drained after the child exits.
    let screen = session(world).drain_final_screen().await;
    assert!(
        screen.to_lowercase().contains("replay file not found"),
        "expected a clear 'replay file not found' error on screen, got:\n{screen}"
    );
}

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

#[then("a tool approval prompt is displayed")]
async fn approval_prompt_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("Review: Install TheRock ROCm SDK?", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the approval prompt did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("Approve (y)") && screen.contains("Deny (n)"),
        "approval prompt is missing its Approve/Deny buttons:\n{screen}"
    );
}

#[then("the tool call is shown as declined")]
async fn tool_call_shown_declined(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_until_gone("Review: Install TheRock ROCm SDK?", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the approval prompt is still open after Enter: {e}"));
    tui.wait_for_screen("Action declined.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the declined-tool-call message did not appear: {e}"));
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

/// The recorded chat request's message contents, in order.
///
/// The grounding steps look across every role rather than only `system`: what
/// matters is that the model was told, not which envelope carried it (the
/// built-in local provider folds system text into the user turn).
///
/// Waits for the request carrying `MANAGED_MODEL_PROMPT` specifically. Accepting
/// any chat request instead picks up the local-endpoint detection probe, which
/// is sent before the user types and carries no system prompt at all — the
/// grounding then looks absent when it was simply asserted against the wrong
/// request. Unlike `chat-03`, these steps have no screen wait ahead of them to
/// order the two.
async fn recorded_chat_messages(world: &mut E2eWorld) -> Vec<String> {
    let body = world
        .mock
        .as_ref()
        .expect("no mock server running")
        .wait_for_chat_request_where(default_timeout(), |body| {
            body.get("messages")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|messages| {
                    messages.iter().any(|m| {
                        m.get("role").and_then(serde_json::Value::as_str) == Some("user")
                            && message_text(m.get("content").unwrap_or(&serde_json::Value::Null))
                                .contains(MANAGED_MODEL_PROMPT)
                    })
                })
        })
        .await
        .unwrap_or_else(|e| panic!("the mock never received the user's chat turn: {e}"));
    body.get("messages")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("chat request had no messages array:\n{body}"))
        .iter()
        .filter_map(|m| m.get("content"))
        .map(message_text)
        .collect()
}

/// The text of one OpenAI-format message. `content` is a bare string on the
/// turns the TUI builds, but an array of typed parts on the system message the
/// chat client emits — read both, or the grounding looks absent when it is
/// simply wrapped.
fn message_text(content: &serde_json::Value) -> String {
    content.as_str().map_or_else(
        || {
            content
                .as_array()
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(serde_json::Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default()
        },
        str::to_owned,
    )
}

/// The single line of the sent prompt that opens with `label`, or a failure
/// naming what was actually sent. Asserting on the request — never on the
/// canned reply — is the point: the mock answers identically whatever it is
/// told, so only the request can show the assistant was grounded.
fn sent_fact_line(messages: &[String], label: &str) -> String {
    messages
        .iter()
        .flat_map(|m| m.lines())
        .map(str::trim)
        .find(|line| line.starts_with(label))
        .unwrap_or_else(|| {
            panic!(
                "the assistant was never told `{label}`; the request carried:\n{}",
                messages.join("\n---\n")
            )
        })
        .to_owned()
}

#[then("the assistant is told which operating system this machine runs")]
async fn assistant_told_the_operating_system(world: &mut E2eWorld) {
    let messages = recorded_chat_messages(world).await;
    let line = sent_fact_line(&messages, "- Operating system:");
    let host = e2e_cucumber::capability::host_capability();
    let expected = if host.os_family.eq_ignore_ascii_case("windows") {
        "Windows"
    } else {
        "Linux"
    };
    assert!(
        line.contains(expected),
        "this machine runs {}, but the assistant was told: {line}",
        host.os_family
    );
    // WSL is the case the old prompt got wrong — it told WSL users vLLM was
    // unavailable — so a WSL host must be named as one, not flattened to Linux.
    assert_eq!(
        line.contains("WSL"),
        host.is_wsl,
        "WSL must be stated exactly when this machine is WSL (is_wsl={}): {line}",
        host.is_wsl
    );
}

#[then("the assistant is told which GPU this machine has")]
async fn assistant_told_the_gpu(world: &mut E2eWorld) {
    let messages = recorded_chat_messages(world).await;
    let line = sent_fact_line(&messages, "- AMD GPU:");
    let host = e2e_cucumber::capability::host_capability();
    match host.gfx_target.as_deref() {
        // A host with a real GPU must see that GPU named, not a placeholder.
        Some(target) => assert!(
            line.contains(target),
            "this machine's GPU is {target}, but the assistant was told: {line}"
        ),
        // A host without one must be told so explicitly, rather than left to
        // fill the silence from pretraining.
        None => assert!(
            line.contains("no AMD GPU detected"),
            "no GPU is detectable here, so the assistant must be told that: {line}"
        ),
    }
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
        screen.contains("next / previous tab")
            && screen.contains("Home tab")
            && screen.contains("jump ±60s"),
        "navigation or contextual guidance missing:\n{screen}"
    );
}

#[then("the services manager is displayed")]
async fn services_manager_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Services — managed inference servers", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the services manager did not appear: {e}"));
}

#[then("the services manager is closed")]
async fn services_manager_closed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_until_gone("Services — managed inference servers", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the services manager is still open after Escape: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("● Observe"),
        "Escape left the Observe tab entirely, not just the manager:\n{screen}"
    );
    // Belt-and-suspenders: `wait_until_gone` above is the primary regression
    // check (the manager itself closed). This additionally guards against
    // Esc falling through to open the main menu instead — "Options"/"Quit"
    // are unique to `Modal::Menu`.
    assert!(
        !screen.contains("Options") && !screen.contains("Quit"),
        "the main menu is open on top of the closed manager:\n{screen}"
    );
}

#[then("instance details are displayed")]
async fn instance_details_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Instance · ", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("instance details did not appear: {e}"));
}

#[then("the instance detail footer shows the scroll hint")]
async fn detail_footer_shows_scroll_hint(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("↑/↓ scroll", default_timeout())
        .await
        .unwrap_or_else(|e| {
            panic!("footer did not show the scroll hint once the detail body overflowed: {e}")
        });
}

#[then("the backdrop behind the popup is dimmed")]
async fn backdrop_is_dimmed(world: &mut E2eWorld) {
    let tui = session(world);
    assert!(
        tui.corner_backdrop_is_dimmed(),
        "the screen behind the popup was not dimmed:\n{}",
        tui.screen_text()
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

#[then("the dashboard menu is displayed")]
async fn dashboard_menu_is_displayed(world: &mut E2eWorld) {
    // "Quit" is used here as a marker for `Modal::Menu`'s three items
    // (Options/Help/Quit). Unlike "Options", "Quit" appears nowhere else in
    // the TUI's rendered chrome, so it unambiguously identifies the menu.
    session(world)
        .wait_for_screen("Quit", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("dashboard menu did not appear: {e}"));
}

#[then("the dashboard menu is closed")]
async fn dashboard_menu_is_closed(world: &mut E2eWorld) {
    // A bare Escape send is not guaranteed to have been acted on yet by the
    // time the next step runs — confirm `Modal::Menu` actually closed before
    // quitting, the same way `services_manager_closed` does. Without this,
    // an unlanded close leaves the menu open and swallows the subsequent
    // quit keystroke (`Modal::Menu` has no `q` arm), hanging until the
    // quit step's timeout.
    session(world)
        .wait_until_gone("Options", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the dashboard menu is still open after Escape: {e}"));
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

/// Assert the exit code stashed by the terminating `When` step. Shared by the
/// dashboard and launcher wordings — the assertion is identical, only the
/// process under test differs, and `subject` keeps the failure message honest
/// about which one it was. `gesture` names how the exit was requested, so a
/// failure says whether the signal path or the keystroke path is broken.
fn assert_exited_with(world: &mut E2eWorld, subject: &str, gesture: &str, expected: i32) {
    let observed = session(world)
        .observed_exit_code()
        .expect("no exit code was recorded; terminate the session first");
    assert_eq!(
        observed, expected,
        "{subject} exited with {observed} after {gesture}, expected {expected}"
    );
}

#[then(expr = "the dashboard exits from the signal with code {int}")]
async fn dashboard_exited_from_signal(world: &mut E2eWorld, expected: i32) {
    assert_exited_with(world, "the dashboard", "the signal", expected);
}

// The launcher hub is a different process shape from a dashboard session (it
// outlives each session's runtime), so scenarios that signal the hub say so
// rather than borrowing the dashboard's wording.
#[then(expr = "the launcher exits from the signal with code {int}")]
async fn launcher_exited_from_signal(world: &mut E2eWorld, expected: i32) {
    assert_exited_with(world, "the launcher", "the signal", expected);
}

// Separate wording from the signal steps on purpose: a typed Ctrl-C never
// becomes a signal while the terminal is in raw mode, so a scenario that says
// "from the signal" here would assert the wrong thing about how the exit
// happened, even though the code it lands on is the same 130.
#[then(expr = "the dashboard exits from the keystroke with code {int}")]
async fn dashboard_exited_from_keystroke(world: &mut E2eWorld, expected: i32) {
    assert_exited_with(world, "the dashboard", "the keystroke", expected);
}

#[then(expr = "the launcher exits from the keystroke with code {int}")]
async fn launcher_exited_from_keystroke(world: &mut E2eWorld, expected: i32) {
    assert_exited_with(world, "the launcher", "the keystroke", expected);
}

#[then("the launcher front door is displayed")]
async fn launcher_front_door_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    // "Set up this system" is the front door's first menu entry, drawn at any
    // size. Waiting (rather than reading the screen once) synchronises on the
    // first paint after a launch or after a session hands control back.
    tui.wait_for_screen("Set up this system", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the launcher front door was not displayed: {e}"));
}

#[then("the terminal is restored to the normal screen")]
async fn terminal_restored(world: &mut E2eWorld) {
    // The dashboard's signal handler must leave the alternate screen and show
    // the cursor before exiting; otherwise the shell is left in the broken
    // raw/alt-screen state that needs a `reset`.
    session(world)
        .expect_terminal_restored()
        .unwrap_or_else(|e| panic!("{e}"));
}

#[then("the launcher shows the model serving")]
async fn launcher_shows_serving(world: &mut E2eWorld) {
    let model = world
        .model_name
        .as_deref()
        .expect("no model name set")
        .to_string();
    let tui = session(world);
    // The front door's status strip renders "Serving <model>" for a live
    // registry instance; wait on the model name to synchronise with the first
    // paint before inspecting the whole screen.
    tui.wait_for_screen(&model, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the launcher never showed the serving model: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("Serving"),
        "launcher did not show the model as serving:\n{screen}"
    );
    assert!(
        !screen.contains("Idle — nothing serving"),
        "launcher still reported idle despite a live registry instance:\n{screen}"
    );
}

#[then("the launcher exits successfully")]
async fn launcher_exited(world: &mut E2eWorld) {
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

#[given("dashboard observation time is deterministic")]
async fn dashboard_observation_time_is_deterministic(world: &mut E2eWorld) {
    let path = dash_clock_path(world);
    write_dash_clock(&path, "0");
    world.command_env.push((
        "ROCM_CLI_DASH_TEST_CLOCK_OFFSET_PATH",
        path.into_os_string(),
    ));
}

/// Path of this scenario's test-clock file, inside its isolated root.
fn dash_clock_path(world: &E2eWorld) -> std::path::PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .join(DASH_CLOCK_OFFSET_FILE)
}

/// Publish a clock directive atomically (write a sibling temp file, then
/// rename). The daemon re-reads this file every cycle, so a plain truncating
/// write can be observed mid-update as an empty file; rename makes each
/// directive visible all-at-once instead.
fn write_dash_clock(path: &std::path::Path, directive: &str) {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, directive).expect("failed to stage the dashboard test clock");
    std::fs::rename(&tmp, path).expect("failed to publish the dashboard test clock");
}

/// The TTFT cell `model`'s row is currently rendering, or `None` while that row
/// is not on screen at all.
///
/// Cells are whitespace-separated and a model id carries no spaces, so counting
/// fields from the id yields one field per column — including the `—`
/// placeholder a cleared cell renders, which keeps the columns aligned.
fn scripted_ttft_cell<'a>(screen: &'a str, model: &str) -> Option<&'a str> {
    screen
        .lines()
        .find(|line| line.contains(model))?
        .split_whitespace()
        .skip_while(|field| *field != model)
        .nth(TTFT_COLUMN)
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

/// Stop the daemon's logical observation clock where it stands.
///
/// Free-running, that clock advances one `gpu_tick` per daemon cycle and the
/// cycles are paced by a wall-clock interval — so it tracks wall time, and a
/// scenario descheduled between the failure below and its assertion spends
/// validity budget it never meant to. That is not hypothetical: on the
/// 64-concurrent-scenario mock lane this step's successor was reached four
/// failed scrapes (8 logical seconds) late, past the 6 s window, and the
/// scenario reported a regression the daemon had not committed.
///
/// Held, the clock cannot be moved by anything except this scenario rewriting
/// the file, so the assertions below hold at any later moment, and only the
/// explicit advance in `validity_window_elapsed` crosses the boundary.
///
/// Held *before* the failure, deliberately: the daemon adopts the directive
/// within one cycle of the write, independently of how the harness is
/// scheduled, so the last successful observation is at most
/// `instance_tick + gpu_tick` (3 s) older than the frozen instant — inside the
/// 6 s window with margin, and it stays there.
#[when("dashboard observation time is held")]
async fn dashboard_observation_time_is_held(world: &mut E2eWorld) {
    write_dash_clock(&dash_clock_path(world), "hold");
}

/// Switch the scripted mock to Failure mode and wait for the failure to reach
/// the screen: first the mock's own counter proves the daemon was served a 503,
/// then the cleared TTFT cell proves the frame on display is one the daemon
/// assembled after that scrape. Both waits are synchronizations on observed
/// events, not fixed sleeps, and — the clock being held — neither can consume
/// the validity window they precede.
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

    let model = world
        .model_name
        .clone()
        .expect("the scripted-metrics Given records the model this row belongs to");
    session(world)
        .wait_for_screen_where(
            &format!("the scripted instance's TTFT cell leaves {SCRIPTED_TTFT_CELL:?}"),
            |screen| {
                scripted_ttft_cell(screen, &model).is_some_and(|cell| cell != SCRIPTED_TTFT_CELL)
            },
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| {
            panic!(
                "the failed scrape never reached the screen, so no frame here is known \
                 to postdate the failure: {e}"
            )
        });
}

/// EAI-7960 principal regression assertion.
///
/// The frame under assertion is provably post-failure (the TTFT cell it used to
/// show is gone) and the logical clock is held, so the only way "tok/s" can be
/// missing here is the regression itself: the daemon clearing a held rate on a
/// failed scrape instead of keeping it for the validity window.
#[then("generation throughput remains visible within the validity window")]
async fn gen_tps_held_after_failure(world: &mut E2eWorld) {
    let screen = session(world).screen_text();
    assert!(
        screen.contains("tok/s"),
        "EAI-7960 REGRESSION: gen throughput (\"tok/s\") was cleared immediately \
         after the first failed scrape instead of being held for the validity \
         window (clamp(3 × instance_tick, 6 s, 30 s)).\n\n\
         Last screen:\n{screen}"
    );
}

// ── EAI-7960: expiry boundary helpers ───────────────────────────────────────

/// Step the held clock 7 s past where it was held — one second beyond the 6 s
/// window, from an observation at most 3 s older than the hold point, so the
/// held value is unambiguously expired and stays expired. Nothing else moves
/// this clock, so the assertion below is about the daemon's arithmetic alone.
#[when("the validity window has elapsed")]
async fn validity_window_elapsed(world: &mut E2eWorld) {
    write_dash_clock(&dash_clock_path(world), "hold 7");
}

/// Assert that gen_tps is no longer rendered after the scenario steps the held
/// clock past the validity boundary.
///
/// The expired state is published every cycle and, the clock being held, it is
/// permanent — so waiting for it to reach the screen cannot mask a daemon that
/// kept the value: that daemon simply never clears it and this times out.
#[then("generation throughput is no longer displayed")]
async fn gen_tps_no_longer_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen_where(
            "generation throughput leaves the screen",
            |screen| !screen.contains("tok/s"),
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| {
            panic!(
                "EAI-7960 BOUNDARY-2: gen_tps (\"tok/s\") is still visible after the \
                 validity window clamp(3 × instance_tick, 6 s, 30 s) elapsed. Expected \
                 the daemon to have cleared the held value and the TUI to show the \
                 unavailable placeholder: {e}"
            )
        });
}
