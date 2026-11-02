// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The tick loop that drives the dashboard: `run` (set up, run, tear down) and
//! `event_loop`, plus the frame gate and registry-probe seeding they use. Split
//! out of `app/mod.rs` to keep the core reducer focused; terminal restoration,
//! the shutdown latch and the termination-signal watcher live in
//! `app/terminal.rs`, the focused-host policy in `app/focus.rs` and the
//! Updates-tile check in `app/update_check.rs`.

use std::io;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crossterm::event::{EnableMouseCapture, Event as CtEvent, EventStream, KeyEvent};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use tokio::sync::mpsc;
use tokio::time::interval;

use crate::client::{self, ClientMsg};
use crate::ui;

use super::actions::{
    apply_action, console_scroll_delta, handle_key, is_actionable_key, run_approved,
};
use super::chat::{
    self, StartupChatOutcome, build_chat_agent, build_local_agent, detect_local_chat,
    discover_configured_chat_model, persist_chat_endpoint, startup_chat_outcome,
};
use super::focus::{apply_startup_focus, focused_close_key_blocked, should_skip_daemon};
use super::scrollbar::resolve_mouse;
use super::summary::{parse_plan_result, summarize_slash_tool};
use super::terminal::{
    SHUTTING_DOWN, exit_on_ctrl_c, is_ctrl_c, lock_terminal_writer, restore_after_session,
    restore_terminal, shutdown_claimed_on, spawn_termination_watcher,
};
use super::types::{
    ChatKeyCtx, ChatProvider, ChatTurn, ConnState, Focus, NO_CHAT_BACKEND_MSG, PaneFocus,
    ReplayState, ResolvedArgs,
};
use super::update_check::refresh_update_status;
use super::{AppState, ProviderSwitch};

type Tui = Terminal<CrosstermBackend<io::Stdout>>;

/// Hand off `ResolvedArgs`'s live registry-read seam and launcher-hosted flag
/// onto `AppState` (issue #145). This crate's own `event_loop` calls this
/// directly (rather than inlining the two assignments) so the
/// field-assignment logic itself has a synchronous unit test
/// (`seed_registry_probe_and_hosting_wires_both_fields_onto_state`) instead of
/// needing to spin up the full async terminal loop to reach it.
///
/// That unit test calls this function directly, so it does **not** cover the
/// call site in `event_loop` dropping the hand-off entirely, or passing the
/// wrong field — `run`/`event_loop` are only reachable through the full async
/// terminal loop, which only the `@requires-os:linux` e2e lane
/// (`tests/e2e-cucumber/features/dash.feature`, any scenario that opens a real
/// dashboard or chat session with a managed model present) actually drives.
fn seed_registry_probe_and_hosting(
    state: &mut AppState,
    managed_registry_probe: Option<crate::registry_probe::SharedManagedRegistryProbe>,
    launcher_hosted: bool,
) {
    state.managed_registry_probe = managed_registry_probe;
    state.launcher_hosted = launcher_hosted;
}

pub async fn run(args: ResolvedArgs) -> color_eyre::Result<()> {
    // Install the termination-signal watcher BEFORE switching the terminal into
    // raw/alternate-screen mode. A signal that arrives during startup must find
    // the listeners already registered; otherwise it takes the default
    // disposition and kills the process while the terminal is still in raw mode
    // — the exact broken-terminal state this guards against. Registration
    // failure is propagated here (via `?`), before any terminal state is
    // mutated, so we never enter raw mode without a working restore path. See
    // `spawn_termination_watcher` for the exit-code and no-unwind semantics.
    let signal_task = spawn_termination_watcher()?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = event_loop(&mut terminal, &args).await;

    // The session ended on its own — the signal watcher is no longer needed and
    // must not linger to fire (and re-restore/exit) after a clean return.
    // What `abort()` buys is cancelling a watcher still parked on its `.await`:
    // `spawn_termination_watcher`'s body runs straight from `await_termination`
    // into `process::exit` with no await point in between, so once the signal
    // has resolved there is nowhere for the cancellation to land. A signal
    // arriving in the instant before this call therefore still ends the process
    // with `128 + signo` instead of returning `res`. Narrow window, accepted:
    // the process is exiting either way, and the terminal is restored on both
    // paths.
    signal_task.abort();

    restore_after_session(&SHUTTING_DOWN, restore_terminal);
    res
}

/// Whether a key event must end the dashboard session: a typed Ctrl-C, unless a
/// console for a *still-running* job is displayed.
///
/// Split out of the event loop's match guard so the precedence is testable
/// without a terminal. The console exception is the whole reason this is not
/// simply [`is_ctrl_c`]: while a job is running, Ctrl+C already means "cancel
/// this running job" (`ui::job_console::on_console_key`), which is the documented
/// way to stop a focused install/serve without truncating it — killing the
/// process instead would be a regression.
///
/// That justification stops the moment the job reaches a terminal state, and the
/// exception has to stop with it. No manager clears its `active_job` on
/// completion — it is cleared only when the user dismisses the console with
/// Esc/Enter — so a finished console stays on screen indefinitely. Exempting it
/// unconditionally made Ctrl-C fall through to `on_console_key`, which emits
/// `CancelJob`, which the reducer ignores on a terminal job: the keystroke was a
/// **silent no-op**, leaving the user in raw mode on the alternate screen. Gating
/// on the job being non-terminal keeps "cancel the job" winning only while there
/// is a job left to cancel.
fn ctrl_c_should_exit(state: &AppState, k: KeyEvent) -> bool {
    let job = state.active_job_id().and_then(|id| state.jobs.job(id));
    // The uncodified invariant this predicate leans on: a manager's `active_job`
    // is the id of a job it just spawned into `state.jobs`, so the lookup above
    // resolves. Thirteen manager modules set `active_job`; nothing enforces the
    // pairing at a type level. If it ever breaks, `is_none_or` below silently
    // reads "no job console is up" and Ctrl-C would quit out from under a
    // *running* job — the one case this function exists to prevent. Assert it in
    // debug builds so a manager that sets an id without a matching job trips the
    // suite rather than shipping the wrong precedence.
    debug_assert!(
        state.active_job_id().is_none() || job.is_some(),
        "active_job id {:?} is not in the jobs map; ctrl_c_should_exit would \
         treat a live job console as absent",
        state.active_job_id()
    );
    // Reads as: no job console is up, or the one that is has already finished.
    is_ctrl_c(k) && job.is_none_or(rocm_dash_core::state::JobState::is_terminal)
}
/// Draw one dashboard frame, unless a shutdown has already been claimed on
/// `latch`.
///
/// This is the render gate. A termination may be in flight on another thread
/// (the signal watcher, or a typed Ctrl-C): the terminal is being restored, so
/// the frame must not land after the restore and undo it. See the ordering note
/// on [`restore_terminal`].
///
/// The gate is the latch read **plus** [`lock_terminal_writer`], and needs both.
/// The lock is taken first and held across the whole frame, so the restore can
/// neither splice its bytes into the middle of this frame nor be overtaken by its
/// tail; the latch is then read *under* that lock, so a restore that got there
/// first suppresses this frame entirely instead of merely preceding it.
///
/// Extracted from the event loop's body — and parameterised over the backend and
/// the latch — purely so the gate is *testable*: a test can drive a
/// `TestBackend`, claim a local latch, and assert no cells were painted. Inlined
/// in the loop it was unreachable from any test, and deleting it turned nothing
/// red.
///
/// Focused host renders overlay-only (no header / tabs / dock / footer chrome);
/// the dashboard renders the full shell.
fn draw_frame_unless_shutting_down<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    state: &mut AppState,
    focus: Option<Focus>,
    latch: &AtomicBool,
) -> Result<(), <B as ratatui::backend::Backend>::Error> {
    let _writer = lock_terminal_writer();
    if shutdown_claimed_on(latch) {
        return Ok(());
    }
    if should_skip_daemon(focus) {
        terminal.draw(|f| ui::draw_focused(f, state))?;
    } else {
        terminal.draw(|f| ui::draw(f, state))?;
    }
    Ok(())
}

async fn event_loop(terminal: &mut Tui, args: &ResolvedArgs) -> color_eyre::Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
    // Job-bridge channel (Phase 3 Wave 1): the async runtime streams
    // `StateEvent`s (JobLine/JobDone/JobErr) for operational screens here.
    let (job_tx, mut job_rx) = mpsc::unbounded_channel::<rocm_dash_core::state::StateEvent>();
    // Retain a sender for chat replies BEFORE `tx` is moved into the client /
    // replay task below — the spawned agent task feeds replies back through the
    // same `rx.recv()` arm the daemon events already use (no new plumbing).
    let chat_tx = tx.clone();
    let replay_controller = if let Some(path) = args.replay.clone() {
        Some(crate::replay::spawn(path, tx))
    } else if should_skip_daemon(args.focus) {
        // Focused host: no daemon client. The overlay streams its own job via
        // the job-bridge and the telemetry chrome isn't drawn, so a live
        // connection would only spawn an unused embedded daemon. Drop `tx`
        // (its `chat_tx` clone keeps `rx` alive for the loop); nothing is sent.
        drop(tx);
        None
    } else {
        client::spawn(args.connect.clone(), tx);
        None
    };

    let mut events = EventStream::new();
    let mut tick = interval(Duration::from_millis(250));
    let connect_label = match &args.replay {
        Some(p) => format!(
            "replay:{}",
            p.file_name().and_then(|n| n.to_str()).unwrap_or("?")
        ),
        None => args.connect.clone(),
    };
    let mut state = AppState::new(connect_label, args.theme.clone());
    // Honor the chat-first vs dashboard launch choice (rocm-cli semantics).
    state.active_tab = args.initial_tab;
    // Serve-wizard recipe picker source (Phase 3 Wave 1), adapted by the bin.
    state.model_recipes = args.model_recipes.clone();
    // Runtime manager source (Phase 3 Wave 2), adapted by the bin.
    state.runtimes = args.runtimes.clone();
    // Automations manager source (Phase 3 Wave 3), adapted by the bin.
    state.automations = args.automations.clone();
    // Tool-executor seam (Phase 2 plumbing), injected by the bin; None for
    // demo/replay/mock. Phase 3 will use it.
    state.tool_executor = args.tool_executor.clone();
    // Daemon-tailed bench CSV path for the bench-run form's default --out.
    state.bench_results_dir = args.bench_results_dir.clone();
    // Managed-service records that are no longer running, counted by the bin.
    state.services_past_attempts = args.services_past_attempts;
    // Live registry-read seam and launcher-hosting flag (issue #145).
    seed_registry_probe_and_hosting(
        &mut state,
        args.managed_registry_probe.clone(),
        args.launcher_hosted,
    );
    // Focused host: open exactly the overlay for the requested flow (Examine
    // also auto-runs its read-only job). `Focus::Setup` opens the onboarding
    // overlay — the same wizard `rocm bootstrap setup` routes to.
    let fx = apply_startup_focus(&mut state, args.focus);
    crate::jobs::run_effects(fx, &job_tx);
    state.replay = replay_controller.map(ReplayState::new);
    // Both `--demo` (a generated session replayed) and `--replay <file>` present
    // non-live data, so mark the session simulated for the honesty chrome.
    state.simulated = state.replay.is_some();

    // Resolve the chat backend. `--chat-mock` short-circuits detection with a
    // deterministic offline MockAgentClient (no live LLM, no network); otherwise
    // we auto-detect the endpoint (the std-TCP probe runs once on a blocking
    // thread before the first frame) and build the Rig backend.
    let mut agent: Option<std::sync::Arc<dyn crate::agent::AgentClient>> = if should_skip_daemon(
        args.focus,
    ) {
        // Focused host: Setup/Serve/Diagnose never chat. Skip endpoint detection
        // and backend construction entirely — no probe, no network, no OAuth
        // default. `chat_llm` stays `None` and the Chat tab is never drawn here.
        None
    } else if args.chat_mock {
        state.set_chat_config(
            Some(crate::llm::LlmConfig {
                base_url: "mock://offline-demo".to_string(),
                model: "mock-agent".to_string(),
                api_key: None,
                auth_header: None,
            }),
            true,
        );
        Some(std::sync::Arc::new(
            crate::agent::MockAgentClient::with_tool_call_and_approval_trigger(
                "GPU-2 is running hot: 87% util, 71°C, drawing 250 W (90 GB/192 GB VRAM).",
                "gpu_status",
                "install the sdk",
                crate::tool_exec::ApprovalIntent {
                    title: "Install TheRock ROCm SDK?".to_string(),
                    body: vec![
                        "install_sdk --channel release --format wheel --prefix ~/rocm-sdk"
                            .to_string(),
                    ],
                    name: "install_sdk".to_string(),
                    arguments: serde_json::json!({
                        "channel": "release",
                        "format": "wheel",
                        "prefix": "~/rocm-sdk",
                    }),
                },
                chat_tx.clone(),
            ),
        ) as std::sync::Arc<dyn crate::agent::AgentClient>)
    } else {
        // An endpoint we launched ourselves (managed-services registry) takes
        // priority over the well-known default port — this is how a tool-launched
        // engine on a non-default port (e.g. vLLM on :11435) is found. It does
        // NOT override an explicitly configured `chat_url`/env URL, so config
        // precedence is preserved (we only consult the registry when neither is
        // set, i.e. where the well-known default would otherwise be probed).
        // When neither an explicit URL (CLI/config) nor an env URL is set, run
        // the SAME full local-engine detection the manual 'd' path uses:
        // registry-first (an engine we launched ourselves, on whatever port it
        // bound), then a probe of the well-known Lemonade/vLLM/rocm-serve
        // ports (parallelized — see `llm::detect_local_endpoint` — so a cold
        // start with no server doesn't pay 3x the probe timeout), plus a
        // best-effort served-model fetch. This is what lets a local server win
        // over the ChatGPT cloud default at startup instead of only the single
        // well-known :8000 port that a bare `resolve_llm_config` probe covers.
        //
        // NOTE: unmerged PR #97 also touches this branch (model discovery when
        // `chat_model` is None, inside `resolve_llm_config`'s own fallback
        // path) — this change is conflict-minimal by leaving the
        // `resolve_llm_config` call below untouched.
        //
        // Gate on `chat_api_key.is_none()` too: local detection returns a
        // keyless `detected_llm_config` (api_key/auth_header forced to None),
        // so firing it when the user configured a key would SILENTLY DROP that
        // key and 401 at request time. A configured key means "use my
        // configured backend", so skip the swap and let `resolve_llm_config`
        // carry the key through its normal precedence.
        let detection_ran = chat::should_detect_local_chat(
            args.chat_url.as_deref(),
            args.chat_env_url.as_deref(),
            args.chat_api_key.as_deref(),
        );
        let detected = if detection_ran {
            detect_local_chat(state.tool_executor.clone()).await
        } else {
            None
        };
        let probe_target = args
            .chat_url
            .clone()
            .or_else(|| args.chat_env_url.clone())
            .unwrap_or_else(|| crate::llm::DEFAULT_CHAT_BASE_URL.to_string());
        // A detected endpoint (managed or probed) is already verified. When
        // detection ran and found nothing it already probed the well-known
        // vLLM :8000 port (== `DEFAULT_CHAT_BASE_URL`), so re-probing the same
        // fallback target here is redundant and just burns another probe
        // timeout on a cold start — treat that as unreachable directly.
        // Otherwise (an explicit URL/env/key path) TCP-probe the target.
        let startup_outcome = startup_chat_outcome(detection_ran, detected.is_some());
        let probe_ok = match startup_outcome {
            StartupChatOutcome::Local => true,
            StartupChatOutcome::OAuth => false,
            StartupChatOutcome::Configured => tokio::task::spawn_blocking(move || {
                crate::llm::probe_endpoint(&probe_target, crate::llm::PROBE_TIMEOUT)
            })
            .await
            .unwrap_or(false),
        };
        let llm = detected.or_else(|| {
            crate::llm::resolve_llm_config(
                args.chat_url.as_deref(),
                args.chat_model.as_deref(),
                None,
                None,
                args.chat_api_key.as_deref(),
                args.chat_env_url.as_deref(),
                args.chat_auth_header.as_deref(),
                probe_ok,
            )
        });
        // PR #97 port onto PR #100's startup flow: a *configured* URL (CLI/env)
        // with no explicit model resolves to the `local-model` placeholder,
        // which 404s on servers that register the model under its real id. Only
        // the `Configured` outcome needs this — the `Local` outcome already
        // carries a `/v1/models`-discovered model from `detect_local_chat`, and
        // `OAuth` has no config. Discovery is gated inside the helper on
        // `probe_ok` (an unreachable endpoint is never probed nor replaced) and
        // on the absence of an explicit model (config precedence wins).
        let llm = match llm {
            Some(cfg) if startup_outcome == StartupChatOutcome::Configured => Some(
                discover_configured_chat_model(cfg, args.chat_model.as_deref(), probe_ok).await,
            ),
            other => other,
        };
        state.set_chat_config(llm, args.chat_auto_consent);
        // No reachable local endpoint AND no key/url configured → the no-key
        // ChatGPT OAuth default (device-code login surfaced in the chat tab).
        // This restores the no-key login the vendored Codex path provided; it
        // takes NO api_key (env-only invariant untouched — OAuth, not a key).
        let no_key_no_endpoint = startup_outcome == StartupChatOutcome::OAuth;
        if no_key_no_endpoint {
            let oauth_tx = chat_tx.clone();
            crate::agent::ChatGptAgentClient::new(
                args.chat_model.clone(),
                args.inference_params(),
                move |url, code| {
                    let _ = oauth_tx.send(ClientMsg::ChatReply {
                        text: format!(
                            "To enable chat, sign in to ChatGPT: open {url} and enter the code {code}"
                        ),
                    });
                },
                state.tool_executor.clone(),
                Some(chat_tx.clone()),
            )
            .ok()
            .map(|c| c.with_preamble(args.chat_system_prompt.clone()))
            .map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn crate::agent::AgentClient>)
        } else {
            // A build failure leaves `agent` None; a submit surfaces an error turn.
            match &state.chat_llm {
                Some(cfg) => build_local_agent(
                    cfg.clone(),
                    args.inference_params(),
                    state.tool_executor.clone(),
                    chat_tx.clone(),
                    args.chat_system_prompt.clone(),
                )
                .ok(),
                None => None,
            }
        }
    };

    // Snapshot the auto-detected local backend so `/provider local` can restore
    // it after a switch to a remote provider. Without this, switching to OpenAI
    // and back to local would leave `agent` pointing at the OpenAI backend
    // (silent wrong-backend bug) — `build_chat_agent(Local)` returns None by
    // design (Local is the inline-built backend), so the caller must restore the
    // saved clone here. `Option<Arc<…>>` clone is a cheap Arc refcount bump.
    let mut local_agent = agent.clone();

    loop {
        draw_frame_unless_shutting_down(terminal, &mut state, args.focus, &SHUTTING_DOWN)?;
        tokio::select! {
            _ = tick.tick() => {
                // Advance the animation clock so spinners cycle even while a
                // job produces no new output.
                state.tick_count = state.tick_count.wrapping_add(1);
                // Drive the Home tab's Updates tile off a real periodic check.
                // Never in `--demo`/`--replay` sessions: simulated data must
                // never shell out or look live (see `AppState::simulated`).
                if !state.simulated {
                    let fx = refresh_update_status(&mut state);
                    crate::jobs::run_effects(fx, &job_tx);
                }
            }
            maybe_msg = rx.recv() => {
                match maybe_msg {
                    Some(ClientMsg::Connecting) => state.conn = ConnState::Connecting,
                    Some(ClientMsg::Connected { host, daemon_version }) => {
                        state.conn = ConnState::Connected { host, version: daemon_version };
                    }
                    Some(ClientMsg::Disconnected { reason }) => {
                        state.conn = ConnState::Disconnected { reason };
                        state.latest = None;
                    }
                    Some(ClientMsg::Event(ev)) => state.apply_event(*ev),
                    Some(ClientMsg::ReplaySeek) => state.reset_for_seek(),
                    Some(ClientMsg::ReplayPosition { elapsed_s, total_s }) => {
                        if let Some(r) = state.replay.as_mut() {
                            r.elapsed_s = elapsed_s;
                            r.total_s = total_s;
                        }
                    }
                    Some(ClientMsg::ChatReply { text }) => state.on_chat_reply(text),
                    Some(ClientMsg::SlashToolReply { text }) => state.on_slash_tool_reply(text),
                    Some(ClientMsg::ChatError { message }) => state.on_chat_error(message),
                    Some(ClientMsg::ChatDetectResult { offer }) => state.set_detect_result(offer),
                    // A mutating tool (or slash command) surfaced an approval —
                    // open the modal; nothing executes until the operator approves.
                    Some(ClientMsg::ChatApprovalRequired { intent }) => state.open_approval(intent),
                    // An approved action finished: append the result turn and
                    // fire exactly one automatic follow-up agent turn.
                    Some(ClientMsg::ChatApprovalResult { text }) => state.on_approval_result(text),
                    // A `/plan` plan completed: render the review and (for a
                    // complete mutating action) hand it to the approval modal.
                    Some(ClientMsg::PlanReady { text, action }) => {
                        state.on_plan_ready(text, action);
                    }
                    None => break,
                }
            }
            // Job-bridge events feed the operational-screen job model (Wave 1).
            maybe_job = job_rx.recv() => {
                if let Some(ev) = maybe_job {
                    let fx = state.jobs.apply(ev);
                    crate::jobs::run_effects(fx, &job_tx);
                }
            }
            maybe_ev = events.next() => {
                match maybe_ev {
                    // Only ACT on key presses. Terminals (notably Windows
                    // Terminal / ConPTY under WSL, and any with the kitty
                    // keyboard protocol) also emit Release/Repeat events; the
                    // general `handle_key` already drops non-Press, but the
                    // operational-overlay arms below dispatch straight to their
                    // managers and would otherwise process the SAME keystroke
                    // twice. That double-fire is what made Enter in the serve
                    // wizard's model picker re-open the picker (seeding it with
                    // the just-chosen model as a filter) instead of choosing.
                    // Swallow non-Press key events here, above every key arm, so
                    // the Press-only invariant holds for overlays too.
                    Some(Ok(CtEvent::Key(k))) if !is_actionable_key(k.kind) => {
                        let _ = k;
                    }
                    // A typed Ctrl-C. Raw mode means this reaches the app as a
                    // key event and never as SIGINT / `CTRL_C_EVENT` (see
                    // `is_ctrl_c`), so the signal watcher cannot see it: without
                    // this arm the first gesture a user reaches for does nothing
                    // at all and leaves them in a raw-mode terminal. Handled
                    // above every overlay so no screen can trap it, and routed
                    // through the same restore-and-exit path a real SIGINT takes.
                    //
                    // The one exception is a displayed job console, where Ctrl+C
                    // already means "cancel this running job"
                    // (`ui::job_console::on_console_key`, dispatched by the
                    // overlay arms below). Exiting the process there would be a
                    // regression, so that established meaning wins.
                    Some(Ok(CtEvent::Key(k))) if ctrl_c_should_exit(&state, k) => {
                        exit_on_ctrl_c();
                    }
                    // The approval modal, when open, owns every remaining key
                    // (above every operational overlay and the general handler)
                    // so the operator's decision can't be pre-empted by a screen
                    // behind it. Only the Ctrl-C arm above outranks it.
                    //
                    // Spelling out the consequence, because it is the one that
                    // surprises: a typed Ctrl-C while an approval is pending
                    // QUITS the dashboard. It is not consumed by the modal and
                    // it is not a decline. That is deliberate — Ctrl-C is the
                    // gesture a user reaches for to get out of a program, and a
                    // modal that swallowed it would recreate the wedge this PR
                    // fixes (the pre-fix dashboard ignored Ctrl-C entirely and
                    // left the user in a raw-mode terminal), which is worse here
                    // than anywhere: the approval modal is exactly where someone
                    // wants out in a hurry. Nothing is lost by quitting — the
                    // pending action has NOT run (approval is what would run
                    // it), so declining and quitting leave identical state on
                    // disk; only the chat turn differs. The help surfaces say
                    // "quit" for Ctrl-C without exception for this modal, which
                    // is therefore accurate.
                    //
                    // To decline without quitting there are `n` (deny) and
                    // `Esc` / `q` (cancel) — `ui::approval::approval_key`, which
                    // ignores Ctrl-C, so without the arm above the gesture would
                    // be a silent no-op on this screen.
                    // On Approve: replay the approved action off the
                    // event loop (spawn_blocking) and post ChatApprovalResult.
                    // On Deny/Cancel: a declined turn, no execution.
                    Some(Ok(CtEvent::Key(k))) if state.approval_pending() => {
                        use crate::ui::approval::ApprovalVerdict;
                        match state.on_approval_key(k.code) {
                            Some(ApprovalVerdict::Approve) => {
                                if let Some((name, args)) = state.take_approval() {
                                    match state.tool_executor.clone() {
                                        Some(executor) => {
                                            let reply_tx = chat_tx.clone();
                                            tokio::task::spawn_blocking(move || {
                                                let text = run_approved(&executor, &name, &args);
                                                let _ = reply_tx
                                                    .send(ClientMsg::ChatApprovalResult { text });
                                            });
                                        }
                                        None => state.on_approval_result(
                                            "ROCm tools unavailable in this mode".to_string(),
                                        ),
                                    }
                                }
                            }
                            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => {
                                state.on_approval_declined();
                            }
                            None => { /* cursor moved or key ignored — modal stays open */ }
                        }
                    }
                    // The quit-confirm prompt (issue #145), when open, owns
                    // every remaining key the same way the approval modal
                    // above does — nothing behind it can pre-empt the
                    // decision. Only the Ctrl-C arm above outranks it, for
                    // the same reason it outranks the approval modal.
                    Some(Ok(CtEvent::Key(k))) if state.quit_confirm_pending() => {
                        // `s`: jump to the Services overlay and stop it there,
                        // rather than only resolving an Approve/Deny/Cancel
                        // verdict. Checked first so it can't be shadowed by
                        // `resolve_quit_confirm_key`'s own key handling; it
                        // already mutated `state` (prompt closed, overlay
                        // opened) when it returns `true`, so there is nothing
                        // further to resolve for this key.
                        //
                        // `Some(false)`: Deny/Cancel already closed the prompt.
                        // `None`: cursor moved or key ignored — prompt stays open.
                        if !state.quit_confirm_open_services_key(k.code)
                            && state.resolve_quit_confirm_key(k.code) == Some(true)
                        {
                            break;
                        }
                    }
                    // De-modal back-out: on any tab, when an inline manager is
                    // open at its root screen, Esc closes it and returns focus
                    // to the Actions list — intercepted BEFORE the per-manager
                    // key arms so the manager doesn't eat Esc first. `←` is left
                    // to the manager (some use it to cycle options). See
                    // `should_pane_back_out`'s doc comment for why this is no
                    // longer gated to ROCm/Serving.
                    Some(Ok(CtEvent::Key(k))) if state.should_pane_back_out(k.code) => {
                        state.close_overlays();
                        state.pane_focus = PaneFocus::Actions;
                    }
                    // While a manager is showing its job console, the navigation
                    // keys pan the log (PgUp/PgDn = page, arrows = line). Routed
                    // BEFORE the per-manager arms (which would ignore them); the
                    // console action keys (Ctrl+C/q/Esc/Enter) are NOT scroll keys
                    // so they still fall through to `on_console_key`.
                    Some(Ok(CtEvent::Key(k)))
                        if state.has_active_console() && console_scroll_delta(k.code).is_some() =>
                    {
                        let (dv, dh) = console_scroll_delta(k.code).unwrap_or((0, 0));
                        state.scroll_console(dv, dh);
                    }
                    // Focused host only: while the hosted job is still RUNNING,
                    // swallow the console close keys (`q`, running-`Esc`) so the
                    // overlay is never nulled mid-job — which would trip the
                    // focused exit gate and tear the runtime down, killing the
                    // child via kill_on_drop. `Ctrl+C` (cancel) and the scroll
                    // keys above still flow, so the user can always stop a job;
                    // once it is terminal, `q`/`Esc` exit normally. Routed BEFORE
                    // the per-manager arms so the manager can't close first.
                    Some(Ok(CtEvent::Key(k)))
                        if focused_close_key_blocked(&state, args.focus, k.code) => {}
                    // The services-manager overlay, when open, owns all keys
                    // (and may spawn lifecycle jobs through the job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.services.is_some() => {
                        let fx = crate::ui::services_manager::on_key(
                            &mut state.services,
                            &mut state.jobs,
                            &state.instances,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The serve-wizard overlay, when open, owns all keys (and may
                    // spawn a launch job through the job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.serve_wizard.is_some() => {
                        let fx = crate::ui::serve_wizard::on_key(
                            &mut state.serve_wizard,
                            &mut state.jobs,
                            &state.model_recipes,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The engine-manager overlay, when open, owns all keys (and
                    // may stream an install job through the job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.engine_manager.is_some() => {
                        let fx = crate::ui::engine_manager::on_key(
                            &mut state.engine_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The examine overlay, when open, owns all keys. It runs the
                    // read-only `rocm examine` job and the Doctor report preview
                    // through the job-bridge. A report send runs only after an
                    // approval.
                    Some(Ok(CtEvent::Key(k))) if state.examine_manager.is_some() => {
                        let fx = crate::ui::examine_manager::on_key(
                            &mut state.examine_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The update overlay, when open, owns all keys (check/preview
                    // read-only; apply gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.update_manager.is_some() => {
                        let fx = crate::ui::update_manager::on_key(
                            &mut state.update_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The install overlay, when open, owns all keys (dry-run
                    // read-only; install gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.install_manager.is_some() => {
                        let fx = crate::ui::install_manager::on_key(
                            &mut state.install_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The logs overlay, when open, owns all keys (read-only
                    // `rocm logs` through the job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.logs_view.is_some() => {
                        let fx = crate::ui::logs_view::on_key(
                            &mut state.logs_view,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The runtime manager, when open, owns all keys (refresh
                    // read-only; activate/rollback/uninstall/adopt/import gated).
                    Some(Ok(CtEvent::Key(k))) if state.runtime_manager.is_some() => {
                        let fx = crate::ui::runtime_manager::on_key(
                            &mut state.runtime_manager,
                            &state.runtimes,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The onboarding wizard, when open, owns all keys (install /
                    // adopt gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.onboarding.is_some() => {
                        let fx = crate::ui::onboarding::on_key(
                            &mut state.onboarding,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The automations manager, when open, owns all keys (refresh
                    // read-only; enable/disable gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.automations_manager.is_some() => {
                        let fx = crate::ui::automations_manager::on_key(
                            &mut state.automations_manager,
                            &state.automations,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The command runner, when open, owns all keys (every
                    // command gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.command_screen.is_some() => {
                        let fx = crate::ui::command_screen::on_key(
                            &mut state.command_screen,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // The config & provider manager, when open, owns all keys
                    // (show read-only; provider toggles gated → job-bridge).
                    Some(Ok(CtEvent::Key(k))) if state.config_manager.is_some() => {
                        let fx = crate::ui::config_manager::on_key(
                            &mut state.config_manager,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    // Bench-run form, when open, owns all keys.
                    Some(Ok(CtEvent::Key(k))) if state.bench_run.is_some() => {
                        let fx = crate::ui::bench_run::on_key(
                            &mut state.bench_run,
                            &mut state.jobs,
                            k,
                        );
                        crate::jobs::run_effects(fx, &job_tx);
                    }
                    Some(Ok(CtEvent::Key(k))) => {
                        let chat_ctx = ChatKeyCtx {
                            focused: state.chat_focused,
                            consent: state.chat_consent,
                            offer_pending: state.chat_detect_offer.is_some(),
                        };
                        let action = handle_key(k, state.active_tab, &state.modal, chat_ctx);
                        if apply_action(&mut state, action) {
                            break;
                        }
                    }
                    Some(Ok(CtEvent::Mouse(me))) => {
                        let action = resolve_mouse(me, &state);
                        if apply_action(&mut state, action) {
                            break;
                        }
                    }
                    Some(Ok(CtEvent::Resize(_, _))) => { /* repaint */ }
                    // A terminal event-source error means the controlling
                    // terminal went away (e.g. the PTY/stdin closed) — the
                    // session is over, so quit cleanly rather than propagating a
                    // fatal error. Propagating it made `rocm chat` exit non-zero
                    // when its terminal closed before the first key was read
                    // (e.g. the acceptance PTY smoke under the embedded-daemon
                    // start delay); the legacy blocking reader treated this as
                    // end-of-session too. Mirrors the `None => break` EOF arm.
                    Some(Err(e)) => {
                        tracing::debug!(error = %e, "terminal event stream ended; quitting");
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }
        }

        // A `/quit` (or `/exit`) slash command sets `should_quit` from inside
        // the reducer; honor it here (mirrors the `KeyAction::Quit` break).
        if state.should_quit {
            break;
        }

        // Focused host: the launcher hosts exactly one overlay. Once the user
        // backs out of it at root (the per-manager `on_key` set its state to
        // `None`), return so `app::run` hands control back to the launcher menu.
        // `focused_should_exit` stays `false` while any sub-popup / job console
        // keeps the overlay `Some`, so this never ejects mid-flow. No-op for the
        // dashboard (`focus == None`).
        if state.focused_should_exit(args.focus) {
            break;
        }

        // Drain a pending executor-backed read-only slash command (`/model`,
        // `/daemon`). Off-thread (spawn_blocking) so the seam's synchronous
        // execute() never blocks the async event loop; the concise summary
        // returns via ClientMsg::SlashToolReply — its own message variant, so
        // the slash-tool path never disturbs the agent's `chat_sending` flag.
        if let Some(req) = state.slash_tool.take() {
            match state.tool_executor.clone() {
                Some(executor) => {
                    let reply_tx = chat_tx.clone();
                    tokio::task::spawn_blocking(move || {
                        // One path for read-only AND mutating slash commands:
                        // `Result`/`Error` → a concise reply turn; an
                        // `ApprovalRequired` (mutating) → open the approval modal
                        // via ChatApprovalRequired (nothing executes yet).
                        let msg = match executor.execute(&req.name, &req.args) {
                            crate::tool_exec::RocmToolOutcome::ApprovalRequired(intent) => {
                                ClientMsg::ChatApprovalRequired { intent }
                            }
                            outcome => ClientMsg::SlashToolReply {
                                text: summarize_slash_tool(&req.label, &outcome),
                            },
                        };
                        let _ = reply_tx.send(msg);
                    });
                }
                None => {
                    state.on_slash_tool_reply("ROCm tools unavailable in this mode".to_string());
                }
            }
        }

        // Drain a pending `/plan` natural-language plan. Off-thread
        // (spawn_blocking) so the read-only `natural_language_plan` tool's
        // synchronous execute() never blocks the async loop. The rendered plan +
        // structured next action return via ClientMsg::PlanReady; the tool only
        // PLANS — no mutation happens here. A complete mutating action is handed
        // to the approval modal by `on_plan_ready`.
        if let Some(req) = state.plan_request.take() {
            match state.tool_executor.clone() {
                Some(executor) => {
                    let reply_tx = chat_tx.clone();
                    tokio::task::spawn_blocking(move || {
                        let args = serde_json::json!({ "request": req });
                        let msg = match executor.execute("natural_language_plan", &args) {
                            crate::tool_exec::RocmToolOutcome::Result(v) => {
                                match parse_plan_result(&v) {
                                    Some((text, action)) => ClientMsg::PlanReady { text, action },
                                    None => ClientMsg::SlashToolReply {
                                        text: "/plan: planner returned no usable plan".to_string(),
                                    },
                                }
                            }
                            crate::tool_exec::RocmToolOutcome::Error(e) => {
                                ClientMsg::SlashToolReply {
                                    text: format!("/plan failed: {e}"),
                                }
                            }
                            crate::tool_exec::RocmToolOutcome::ApprovalRequired(_) => {
                                ClientMsg::SlashToolReply {
                                    text:
                                        "/plan: planning is read-only and should not need approval"
                                            .to_string(),
                                }
                            }
                        };
                        let _ = reply_tx.send(msg);
                    });
                }
                None => {
                    state.on_slash_tool_reply("ROCm tools unavailable in this mode".to_string());
                }
            }
        }

        // Drain a `/provider` switch (Phase 8). Rebuild the live `agent` for the
        // newly-selected backend. `Local` reuses whatever the inline launch path
        // built (it owns the auto-detect probe). `Openai`/`Anthropic` are built
        // from `ResolvedArgs` keys (in-process seam, never argv). A build failure
        // (e.g. missing key) leaves `agent` unchanged and surfaces an actionable
        // error turn. Construction only — no network until the next submit.
        if let Some(ProviderSwitch { previous, target }) = state.provider_switch.take() {
            match target {
                ChatProvider::Local => {
                    // Restore the auto-detected local backend saved before the
                    // event loop. `build_chat_agent(Local)` returns None by
                    // design, so the restore must happen here — otherwise a prior
                    // `/provider openai` would leave requests routed to OpenAI.
                    agent = local_agent.clone();
                    state
                        .chat
                        .push(ChatTurn::system("switched to local".to_string()));
                }
                ChatProvider::Openai | ChatProvider::Anthropic => {
                    if let Some(new_agent) =
                        build_chat_agent(target, args, state.tool_executor.clone(), chat_tx.clone())
                    {
                        agent = Some(new_agent);
                        state
                            .chat
                            .push(ChatTurn::system(format!("switched to {}", target.label())));
                    } else {
                        // Revert the optimistic `active_provider` set by the slash
                        // handler back to the provider active BEFORE the switch
                        // attempt — not unconditionally Local — so the displayed
                        // provider stays honest (e.g. a failed openai→anthropic
                        // switch stays on openai). `agent` is never reassigned on a
                        // failed build, so it already matches `previous`; the two
                        // stay consistent (no stale-remote routing under a wrong
                        // label).
                        state.active_provider = previous;
                        let hint = if target == ChatProvider::Anthropic {
                            "anthropic requires ANTHROPIC_API_KEY in env or secure store"
                        } else {
                            "openai requires OPENAI_API_KEY in the environment"
                        };
                        state.chat.push(ChatTurn::error(format!(
                            "could not switch to {}: {hint}",
                            target.label()
                        )));
                    }
                }
            }
        }

        // Drain the endpoint-rebuild edge (Phase 8 sibling). An accepted
        // detected-local offer must re-point the LIVE `agent` — and the
        // `/provider local` restore snapshot — at the new local backend.
        // `accept_detect_offer` swaps `chat_llm` to the auth-free local config
        // but stays I/O-free, so without this the stale startup agent keeps
        // routing chat to the cloud gateway (wrong-backend 401 bug). The edge
        // carries the provider active BEFORE the optimistic switch to `Local`;
        // on failure we revert `active_provider` to it (mirrors the
        // `provider_switch` drain) so the tab never shows `Local` while `agent`
        // still points elsewhere. Construction only — no network until submit.
        if let Some(previous) = state.chat_endpoint_rebuild.take() {
            // `revert` restores the optimistic switch and surfaces an actionable
            // error turn so the tab does not sit on `Local` with the old agent.
            let revert = |state: &mut AppState, msg: String| {
                state.active_provider = previous;
                state.chat.push(ChatTurn::error(msg));
            };
            match state.chat_llm.clone() {
                Some(cfg) => {
                    match build_local_agent(
                        cfg,
                        args.inference_params(),
                        state.tool_executor.clone(),
                        chat_tx.clone(),
                        args.chat_system_prompt.clone(),
                    ) {
                        Ok(arc) => {
                            agent = Some(arc.clone());
                            // Refresh the restore snapshot so a later `/provider
                            // local` restores THIS accepted backend, not the
                            // stale startup one.
                            local_agent = Some(arc);
                            state
                                .chat
                                .push(ChatTurn::system("switched to local".to_string()));
                        }
                        Err(e) => revert(
                            &mut state,
                            format!("could not switch to the detected local endpoint: {e}"),
                        ),
                    }
                }
                // Edge raised but `chat_llm` is None (shouldn't happen after a
                // real accept, but don't leave the tab stuck on `Local` with the
                // old agent and no feedback).
                None => revert(
                    &mut state,
                    "could not switch to the detected local endpoint: no endpoint configured"
                        .to_string(),
                ),
            }
        }

        // Spawn the agent round-trip on the submit edge — keeps `apply_action`
        // I/O-free. `chat_dispatch` is raised once by `submit_chat`; consume it
        // so the in-flight request is spawned exactly once (not every tick).
        if state.chat_dispatch {
            state.chat_dispatch = false;
            match agent.clone() {
                Some(agent) => {
                    let history = state.chat.clone();
                    let snapshot = state.state_snapshot();
                    let reply_tx = chat_tx.clone();
                    tokio::spawn(async move {
                        let msg = match agent.complete(&history, snapshot).await {
                            Ok(text) => ClientMsg::ChatReply { text },
                            Err(e) => ClientMsg::ChatError {
                                message: e.to_string(),
                            },
                        };
                        let _ = reply_tx.send(msg);
                    });
                }
                None => state.on_chat_error(NO_CHAT_BACKEND_MSG.to_string()),
            }
        }

        // Run the local-engine probe + `/v1/models` query on the detect edge,
        // off the reducer. Raised once by `request_detect`; result returns via
        // `ClientMsg::ChatDetectResult`.
        if state.chat_detect_dispatch {
            state.chat_detect_dispatch = false;
            let reply_tx = chat_tx.clone();
            let executor = state.tool_executor.clone();
            tokio::spawn(async move {
                let offer = detect_local_chat(executor).await;
                let _ = reply_tx.send(ClientMsg::ChatDetectResult { offer });
            });
        }

        // Persist the accepted endpoint on the save edge (a small synchronous
        // file write; the message surfaces success/failure on the gate is not
        // shown once Accepted, so we keep it terse via tracing + chat_detect_msg).
        if state.chat_persist_dispatch {
            state.chat_persist_dispatch = false;
            if let Some(cfg) = state.chat_llm.clone() {
                match persist_chat_endpoint(&cfg.base_url, &cfg.model) {
                    Ok(path) => {
                        tracing::info!(?path, "saved chat endpoint to config");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to save chat endpoint");
                        state.chat_detect_msg = Some(format!("could not save config: {e}"));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::focus::open_overlay_for_focus;
    use super::super::terminal::claim_shutdown;
    use super::super::test_support::{press, st};
    use crossterm::event::{KeyCode, KeyModifiers};

    #[test]
    fn ctrl_c_exits_the_session_but_a_job_console_keeps_cancelling_the_job() {
        // Precedence for the event loop's Ctrl-C arm. With no console up, the
        // gesture ends the session; with one up it must fall through to
        // `job_console::on_console_key`, whose Ctrl+C cancels the running job —
        // the documented way to stop a focused install/serve without killing the
        // process mid-write.
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        let mut s = st();
        assert!(
            ctrl_c_should_exit(&s, ctrl_c),
            "Ctrl+C with no job console must end the session"
        );
        assert!(
            !ctrl_c_should_exit(&s, press(KeyCode::Char('q'))),
            "an unrelated key must not take the terminate path"
        );

        let _ = open_overlay_for_focus(&mut s, Focus::Examine); // auto-runs a job
        assert!(s.has_active_console(), "examine console is live");
        assert!(
            !ctrl_c_should_exit(&s, ctrl_c),
            "Ctrl+C over a live job console must cancel the job, not the process"
        );
    }

    /// Every cell a `TestBackend` frame painted, trimmed — `""` for a frame the
    /// render gate suppressed. Shared by the two gate tests below so they assert
    /// on the same thing.
    fn painted(term: &ratatui::Terminal<ratatui::backend::TestBackend>) -> String {
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>()
            .trim()
            .to_string()
    }

    #[test]
    fn a_claimed_shutdown_stops_the_dashboard_painting_another_frame() {
        // The render gate itself, not just the latch predicate underneath it.
        // `restore_terminal()` runs on a Tokio worker while frames are drawn on
        // the `block_on` thread, and nothing locks the terminal — so a frame that
        // *starts* after the restore would hide the cursor again and repaint a
        // stale dashboard over the restored screen. The claim happens before the
        // restore begins, so gating on it is what makes that impossible.
        //
        // Driven against a `TestBackend` and a local latch: no process-global
        // state, no real terminal, and the assertion is on painted cells rather
        // than on the predicate the gate happens to call.
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Control: with nothing claimed the gate must let the frame through,
        // otherwise the assertion below would pass on a helper that never draws.
        let mut s = st();
        let open = AtomicBool::new(false);
        let mut term = Terminal::new(TestBackend::new(120, 32)).unwrap();
        draw_frame_unless_shutting_down(&mut term, &mut s, None, &open)
            .expect("drawing to a TestBackend cannot fail");
        assert!(
            !painted(&term).is_empty(),
            "with no shutdown claimed the dashboard must paint a frame"
        );

        // The real case: a watcher (or a typed Ctrl-C) has claimed the shutdown
        // and the restore is under way.
        let mut s = st();
        let claimed = AtomicBool::new(false);
        assert!(claim_shutdown(&claimed), "the test must win its own latch");
        let mut term = Terminal::new(TestBackend::new(120, 32)).unwrap();
        draw_frame_unless_shutting_down(&mut term, &mut s, None, &claimed)
            .expect("the gate must not turn a suppressed frame into an error");
        assert_eq!(
            painted(&term),
            "",
            "once the shutdown is claimed no further frame may be painted — a \
             late frame lands after `restore_terminal()` and undoes it"
        );
    }

    #[test]
    fn a_frame_cannot_paint_while_a_teardown_owns_the_terminal() {
        // The half of the gate the latch cannot provide on its own, and the
        // defect the WSL2 E2E lane caught as `dash-sigint-restores-terminal`:
        // "alternate_screen=false, cursor_hidden=true" — the alt-screen left, but
        // the cursor still invisible.
        //
        // Reading the latch before drawing stops a frame that *starts* after the
        // claim. It cannot stop the frame already in flight when the claim lands,
        // and that frame is the problem: `Terminal::draw` ends by emitting `Hide`
        // unconditionally, so its tail undoes the restore's `Show` while the
        // alt-screen stays left (`EnterAlternateScreen` is never re-emitted) —
        // exactly the half-restored terminal the lane reported. Reproduced
        // outside the harness by signalling a real `rocm dash --demo` under a
        // pty: the restore landed *inside* a frame's bytes, with `ESC[?25l` last.
        //
        // So the gate must take `lock_terminal_writer()` FIRST and read the latch
        // under it. Modelled with the teardown's half of that lock held by another
        // thread: this thread asks to draw while the teardown owns the terminal,
        // and must paint nothing, because the claim happens-before the unlock it
        // is waiting on. Delete the lock from the gate and the frame paints
        // immediately instead, which is the bug.
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let latch = AtomicBool::new(false);
        let mut s = st();
        let mut term = Terminal::new(TestBackend::new(120, 32)).unwrap();
        let (held_tx, held_rx) = std::sync::mpsc::channel::<()>();
        let (drawing_tx, drawing_rx) = std::sync::mpsc::channel::<()>();

        std::thread::scope(|scope| {
            // `mpsc::Receiver` is `Send` but not `Sync`, so the halves the
            // teardown thread uses are moved into it; the latch is shared as a
            // plain reference (the whole point is that both threads see it).
            let latch = &latch;
            scope.spawn(move || {
                // Stands in for `restore_terminal()`. It writes nothing: the
                // claim is what this test is about, not the escape bytes (those
                // are covered by `write_restore_sequences_leaves_alt_screen_…`).
                let guard = lock_terminal_writer();
                held_tx.send(()).expect("the drawing thread is alive");
                drawing_rx.recv().expect("the drawing thread is alive");
                // Only to make the unfixed code reliably red: an ungated draw
                // paints in microseconds, so it would certainly have painted
                // within this window. The fixed path does not depend on the
                // duration — the claim below happens-before the unlock either
                // way, so the assertion holds even if this were zero.
                std::thread::sleep(std::time::Duration::from_millis(200));
                assert!(
                    claim_shutdown(latch),
                    "the teardown must win a latch nothing else can see"
                );
                drop(guard);
            });

            held_rx.recv().expect("the teardown thread is alive");
            drawing_tx.send(()).expect("the teardown thread is alive");
            draw_frame_unless_shutting_down(&mut term, &mut s, None, latch)
                .expect("the gate must not turn a suppressed frame into an error");
        });

        assert_eq!(
            painted(&term),
            "",
            "a frame asked for while a teardown owned the terminal must not paint \
             — its trailing `Hide` would land after the restore's `Show` and leave \
             the user on the normal screen with an invisible cursor"
        );
    }

    #[test]
    fn ctrl_c_exits_once_the_console_job_has_finished() {
        // The gesture the PR exists to fix, in the state that used to swallow it.
        // Nothing clears `active_job` when a job completes (only an Esc/Enter
        // dismissal does), so the console stays on screen after the job is done.
        // While the exemption keyed on "a console is displayed" rather than "a
        // job is running", Ctrl-C there fell through to `on_console_key` →
        // `CancelJob`, which the reducer drops on a terminal job: nothing
        // happened at all, and the user stayed in raw mode on the alt-screen.
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        let mut s = st();
        let _ = open_overlay_for_focus(&mut s, Focus::Examine); // auto-runs a job
        let job_id = s
            .active_job_id()
            .expect("opening Examine must start a job and show its console")
            .to_string();
        assert!(
            !ctrl_c_should_exit(&s, ctrl_c),
            "while the job is still running, Ctrl+C must cancel the job"
        );

        // The job finishes. The console is NOT dismissed — this is the review
        // state the user is left sitting in.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: job_id.clone(),
            code: 0,
        });
        assert!(
            s.jobs
                .job(&job_id)
                .is_some_and(rocm_dash_core::state::JobState::is_terminal),
            "the job must have reached a terminal state"
        );
        assert!(
            s.has_active_console(),
            "the finished console must still be displayed — that is the whole \
             point of this case"
        );
        assert!(
            ctrl_c_should_exit(&s, ctrl_c),
            "Ctrl+C over a FINISHED job console must end the session; there is no \
             job left to cancel, so exempting it makes the keystroke a silent \
             no-op and traps the user in raw mode"
        );
    }

    #[test]
    fn seed_registry_probe_and_hosting_wires_both_fields_onto_state() {
        // Covers only `seed_registry_probe_and_hosting` itself — that it
        // assigns `managed_registry_probe` and `launcher_hosted` onto the
        // right fields, not swapped or dropped. It does NOT cover whether
        // `run_async` actually calls this at its own call site with the
        // right arguments: that would need the full async `event_loop` run,
        // which only the `@requires-os:linux` e2e lane drives (any
        // `dash.feature` scenario that opens a real dashboard/chat session
        // with a managed model present exercises it for real). Each half
        // (the bin producing the probe, `AppState::has_live_instance`/
        // `ui::quit_confirm::quit_confirm_body` consuming it) is tested elsewhere; this is
        // only the field-assignment logic in between.
        use crate::registry_probe::{ManagedRegistrySnapshot, StaticManagedRegistrySnapshot};

        let mut s = st();
        assert!(s.managed_registry_probe.is_none());
        assert!(!s.launcher_hosted);

        let probe: crate::registry_probe::SharedManagedRegistryProbe =
            std::sync::Arc::new(StaticManagedRegistrySnapshot(ManagedRegistrySnapshot {
                has_live_service: true,
                managed_ids: std::iter::once("svc-a".to_string()).collect(),
            }));
        seed_registry_probe_and_hosting(&mut s, Some(probe.clone()), true);

        assert!(s.launcher_hosted);
        assert_eq!(
            s.managed_registry_probe
                .as_ref()
                .map(|p| std::sync::Arc::ptr_eq(p, &probe)),
            Some(true)
        );
    }
}
