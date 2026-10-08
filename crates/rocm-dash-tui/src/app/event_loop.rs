// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Terminal lifecycle, signal handling, and the tick loop that drives the
//! dashboard: `run`, `event_loop`, the termination-signal watcher, and the
//! startup-focus / Updates-tile tick helpers. Split out of `app/mod.rs` to
//! keep the core reducer focused.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, Event as CtEvent, EventStream, KeyCode, KeyEvent,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
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
use super::scrollbar::{PaneFocus, resolve_mouse};
use super::summary::{parse_plan_result, summarize_slash_tool};
use super::types::{
    ChatKeyCtx, ChatProvider, ChatTurn, ConnState, Focus, NO_CHAT_BACKEND_MSG, ReplayState,
    ResolvedArgs, UpdateStatus,
};
use super::{AppState, ProviderSwitch};

type Tui = Terminal<CrosstermBackend<io::Stdout>>;

/// How often the background Updates-tile check re-runs.
const UPDATE_CHECK_INTERVAL: Duration = Duration::from_hours(6);

/// Job id for the periodic background update check driven off the tick loop.
/// Deliberately distinct from `update_manager`'s interactive `"update-check"`
/// so the two never clobber each other's job slot / console output.
/// `pub(crate)` so the Home tab's activity feed (`ui::tabs::home`) can filter
/// this job out — it is the tile's own plumbing, not user activity.
pub(crate) const HOME_UPDATE_CHECK_JOB_ID: &str = "home-update-check";

/// Bound on the background update check's own per-runtime index lookups, so a
/// slow/unreachable index can't leave the job running indefinitely — the same
/// principle as the CLI's own `STARTUP_UPDATE_CHECK_TIMEOUT_SECS`. The two
/// crates can't share the constant (`apps/rocm` depends on `rocm-dash-tui`,
/// not the reverse), so this value isn't required to match it.
const HOME_UPDATE_CHECK_TIMEOUT_SECS: u64 = 5;

/// Whether the event loop should skip the embedded daemon client AND the chat
/// backend resolution. True exactly when a [`Focus`] is set: a focused host runs
/// one overlay that streams its own job through the job-bridge, so it needs
/// neither live telemetry nor an LLM. `focus == None` (the dashboard) keeps both.
/// Pure predicate → unit-testable without a runtime; also names the render branch
/// (`draw_focused` when true, `draw` when false).
const fn should_skip_daemon(focus: Option<Focus>) -> bool {
    focus.is_some()
}

/// In a focused host, whether a console "close" key must be SWALLOWED because
/// the active job is still running.
///
/// In the dashboard, `q` / running-`Esc` detach the console and leave the job
/// running in the background (the app persists). A focused host has no
/// background: closing the overlay trips the [`AppState::focused_should_exit`]
/// gate and returns from `event_loop`, tearing down the runtime and killing the
/// child via `kill_on_drop` — truncating a mutating install/serve mid-write. So
/// while the job is non-terminal we swallow those keys; the user stops a job
/// explicitly with `Ctrl+C` (never blocked here), and once it is terminal `q` /
/// `Esc` exit normally. Always `false` for the dashboard (`focus == None`).
fn focused_close_key_blocked(state: &AppState, focus: Option<Focus>, code: KeyCode) -> bool {
    if !should_skip_daemon(focus) {
        return false;
    }
    let running = state
        .active_job_id()
        .is_some_and(|id| ui::job_console::console_esc_closes(state.jobs.job(id)));
    running && matches!(code, KeyCode::Char('q') | KeyCode::Esc)
}

/// Open the single overlay a focused host should host, returning any initial
/// job-bridge side effects to pump (Examine auto-runs `rocm examine` on open;
/// Setup/Serve open their form and wait for input). Clears any other overlay
/// first (mutually-exclusive invariant). Pure w.r.t. process I/O — the caller
/// runs the returned effects through [`crate::jobs::run_effects`].
fn open_overlay_for_focus(
    state: &mut AppState,
    focus: Focus,
) -> Vec<rocm_dash_core::state::SideEffect> {
    state.close_overlays();
    match focus {
        Focus::Setup => {
            state.onboarding = Some(crate::ui::onboarding::OnboardingState::default());
            Vec::new()
        }
        Focus::Serve => {
            state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState::default());
            Vec::new()
        }
        Focus::Examine => {
            let (mgr, fx) = crate::ui::examine_manager::open_running(&mut state.jobs);
            state.examine_manager = Some(mgr);
            fx
        }
    }
}

/// Reduce a parsed `rocm update --json` document's `runtimes` array into an
/// [`UpdateStatus`]. Empty ⇒ nothing managed to check; any update-available or
/// repair-available row wins over up-to-date/error rows (the tile surfaces the
/// most actionable state — a repair is as actionable as an update); otherwise
/// `UpToDate` only if every row resolved cleanly — a mixed result (some rows
/// errored, some unrecognized) can't honestly assert freshness for the
/// runtimes that didn't resolve, so it's `Error` too.
fn reduce_update_json(document: &serde_json::Value) -> UpdateStatus {
    fn status_of(row: &serde_json::Value) -> Option<&str> {
        row.get("status").and_then(serde_json::Value::as_str)
    }
    let Some(runtimes) = document
        .get("runtimes")
        .and_then(serde_json::Value::as_array)
    else {
        return UpdateStatus::Error;
    };
    if runtimes.is_empty() {
        return UpdateStatus::NoManagedRuntimes;
    }
    if let Some(row) = runtimes.iter().find(|row| {
        matches!(
            status_of(row),
            Some("update_available" | "repair_available")
        )
    }) {
        // A missing/null `latest_version` must not silently fall through to
        // the up-to-date/error checks below — that would misreport a real,
        // actionable update as "check failed". Fall back to a placeholder
        // instead of losing the actionable status.
        let latest_version = row
            .get("latest_version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(version unknown)")
            .to_owned();
        return UpdateStatus::UpdateAvailable { latest_version };
    }
    if runtimes
        .iter()
        .all(|row| matches!(status_of(row), Some("up_to_date" | "ahead_of_index")))
    {
        return UpdateStatus::UpToDate;
    }
    UpdateStatus::Error
}

/// Spawn/consume the periodic `home-update-check` job that backs the Home
/// tab's Updates tile, and return any job-bridge side effects to pump.
///
/// Pure w.r.t. process I/O — like [`open_overlay_for_focus`], the caller runs
/// the returned effects through [`crate::jobs::run_effects`]. Called once per
/// tick from `event_loop`, skipped entirely under `state.simulated`.
fn refresh_update_status(state: &mut AppState) -> Vec<rocm_dash_core::state::SideEffect> {
    if state.update_status_pending {
        let Some(job) = state.jobs.job(HOME_UPDATE_CHECK_JOB_ID) else {
            // Re-arm the same as the terminal-job path below: without this,
            // a vanished job would leave `update_check_due_at` in the past,
            // so every subsequent tick would spawn a new check immediately.
            state.update_status_pending = false;
            state.update_check_due_at = std::time::Instant::now() + UPDATE_CHECK_INTERVAL;
            return Vec::new();
        };
        if !job.is_terminal() {
            return Vec::new();
        }
        state.update_status = match &job.status {
            rocm_dash_core::state::JobStatus::Done { code: 0 } => job
                .output
                .iter()
                .rev()
                .find_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .map_or(UpdateStatus::Error, |doc| reduce_update_json(&doc)),
            _ => UpdateStatus::Error,
        };
        state.update_status_pending = false;
        state.update_check_due_at = std::time::Instant::now() + UPDATE_CHECK_INTERVAL;
        return Vec::new();
    }

    if std::time::Instant::now() < state.update_check_due_at {
        return Vec::new();
    }

    if std::env::var_os("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK").is_some() {
        return Vec::new();
    }

    let fx = state
        .jobs
        .apply(rocm_dash_core::state::StateEvent::StartJob {
            id: HOME_UPDATE_CHECK_JOB_ID.to_owned(),
            cmd: crate::ui::exec::resolve_exe(),
            args: vec![
                "update".to_owned(),
                "--json".to_owned(),
                "--timeout-secs".to_owned(),
                HOME_UPDATE_CHECK_TIMEOUT_SECS.to_string(),
            ],
        });
    if !fx.is_empty() {
        state.update_status_pending = true;
    }
    fx
}

/// The dashboard's entire startup gate: an overlay opens only when an
/// explicit `Focus` was resolved from the command line. `event_loop` calls
/// this directly (rather than inlining the `match`) so a regression test can
/// exercise the actual gate instead of `AppState::new`, which takes no focus
/// argument and can't observe it either way.
fn apply_startup_focus(
    state: &mut AppState,
    focus: Option<Focus>,
) -> Vec<rocm_dash_core::state::SideEffect> {
    match focus {
        Some(focus) => open_overlay_for_focus(state, focus),
        None => Vec::new(),
    }
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

/// Teardown for a dash *session* that has ended — the counterpart to the
/// watcher's and [`exit_on_ctrl_c`]'s teardown, for the one restore path that
/// does **not** end the process.
///
/// Best-effort, reusing the exact teardown the signal path runs so the two
/// cannot drift, and never letting teardown failures override the session
/// result: if the controlling terminal already went away (e.g. the PTY closed on
/// quit), these writes can fail with a broken pipe — that must not turn a clean
/// exit into a non-zero one (every step inside `restore` is best-effort).
///
/// # Why this reads the latch instead of claiming it
///
/// [`SHUTTING_DOWN`] is a one-shot *process-exit* arbiter: whoever claims it
/// restores the terminal and calls `process::exit`, and nothing ever releases
/// it because there is no "after" to release into. `run` returning is the one
/// teardown with an after — bare `rocm` is a persistent hub, so `run` hands
/// control back to a live launcher menu that must keep painting, keep honouring
/// Ctrl-C, and keep being killable. Claiming the latch here wedged all three at
/// once: the render gate refused every subsequent launcher frame (a blank front
/// door), [`exit_on_ctrl_c`] parked forever, and every later `await_termination`
/// lost the claim and returned without exiting — a signal-swallowing, blank,
/// unkillable hub after the user's first flow.
///
/// So the latch is *read*: if a watcher (or a typed Ctrl-C) has already claimed
/// the exit, it owns the teardown and is microseconds from ending the process,
/// and a second restore here would be pure redundancy. Otherwise this session
/// restores the terminal and leaves the latch untouched for the windows that
/// come after it.
///
/// Single-writer is preserved by [`RESTORE_LOCK`] inside [`restore_terminal`]
/// rather than by this read, which is deliberately racy on its own: `abort()`
/// above cannot stop a watcher already resumed past its `.await` and inside its
/// own synchronous `restore_terminal(); process::exit(code)`, so a SIGTERM
/// landing in the same instant the user presses `q` can still put two threads on
/// this path. They serialise on the lock; they no longer contend for the latch.
///
/// `restore` is a parameter so a unit test can assert both halves of the
/// contract — that the teardown runs, and that it leaves the latch unclaimed —
/// without writing escape sequences to the stdout every other test shares.
pub(crate) fn restore_after_session(latch: &AtomicBool, restore: impl FnOnce()) {
    if shutdown_claimed_on(latch) {
        return;
    }
    restore();
}

/// Best-effort teardown of the terminal modes `run` set up. Disables raw mode
/// (process-global terminal state, so this can safely run from the signal
/// watcher even though the [`Terminal`] backend owns its own `stdout` clone),
/// then writes the alt-screen/mouse/cursor restore sequences to a fresh
/// `stdout`. Every step is best-effort: on a signal we are about to exit anyway,
/// and a vanished controlling terminal must not turn teardown into a panic.
///
/// # Ordering against the renderer
///
/// This writer is genuinely concurrent with the renderer: the watcher runs on a
/// Tokio worker thread while frames are drawn on the thread that owns the loop
/// (`block_on`'s thread for a dashboard session, the synchronous menu thread for
/// the launcher hub). A frame that lands *after* this function's bytes cannot
/// re-enter the alternate screen — `EnterAlternateScreen` is emitted exactly
/// once at startup and `Terminal::draw` never re-emits it — but every frame ends
/// by hiding the cursor (`Terminal::draw` emits `Hide` unconditionally when the
/// frame sets no cursor position), so a late frame undoes the show-cursor half of
/// this restore and leaves the user with an invisible cursor.
///
/// Gating the loops on [`shutdown_claimed_on`] is necessary but *not* sufficient,
/// and the gap is not cosmetic: the claim cannot stop a frame that already passed
/// the gate, and that frame's tail is written after these bytes. Measured, rather
/// than argued — SIGINT delivered to a real `rocm dash --demo` under a pty, 300
/// runs on a loaded Linux box: 6 of them ended with this restore spliced into the
/// middle of a frame (`…;48;` `ESC[?1049l … ESC[?25h` `2;19;20;22m qui…ESC[?25l`),
/// leaving the alternate screen with the cursor still hidden. That is exactly
/// what the E2E scenario `dash-sigint-restores-terminal` reported from the WSL2
/// lane, and it reproduces identically on the commits before this PR, so it is
/// the long-standing shape of the race and not a new one.
///
/// So the window is closed rather than accepted: [`TERMINAL_WRITE_LOCK`] is held
/// across one whole frame and across one whole restore, and both render loops
/// re-check the latch *while holding it*. Every interleaving then ends with these
/// bytes last — either the frame completes first and this restore follows it, or
/// this restore goes first and the gate suppresses the frame behind it.
///
/// This adds no new hang: a renderer blocked mid-frame on a full terminal blocks
/// this restore's own writes to that same terminal just as surely, so the lock
/// can only make us wait where we were already waiting.
///
/// # Ordering against another restore
///
/// The same lock covers the second hazard. Two teardowns can run at once: a
/// signal watcher resumed past its `.await` races [`run`]'s clean-quit teardown
/// (which deliberately does not claim the exit latch — see
/// [`restore_after_session`]), and two watchers on two runtimes both wake on one
/// process-global signal. Holding [`TERMINAL_WRITE_LOCK`] makes this function the
/// single writer for the duration of one teardown, so two threads can never
/// interleave `write_restore_sequences` on the same stdout.
pub(crate) fn restore_terminal() {
    // Poisoning is irrelevant here: nothing inside can panic (every step is
    // best-effort), and a teardown skipped because some *other* thread panicked
    // mid-restore is strictly worse than running it again.
    let _guard = lock_terminal_writer();
    // `disable_raw_mode` mutates the real terminal (there is no in-memory
    // equivalent), so it stays outside the testable sequence writer below.
    let _ = disable_raw_mode();
    let _ = write_restore_sequences(&mut io::stdout());
}

/// Serialises everything that writes to the process's one stdout while the TUI
/// owns it: one whole frame from either render loop, and one whole teardown in
/// [`restore_terminal`].
///
/// Distinct from [`SHUTTING_DOWN`], and deliberately so: the latch answers "is
/// the process exiting" (one-shot, never released), this answers "is someone
/// writing the terminal right now" (re-armable, held for the length of one frame
/// or one teardown). Conflating them is what made a clean session exit
/// permanently wedge the launcher hub.
///
/// The two are used *together* in the render gates — the latch is re-read under
/// this lock — because neither alone is enough: the lock without the latch would
/// merely order a late frame after the restore, and the latch without the lock
/// cannot stop a frame that has already passed the gate. See the ordering notes
/// on [`restore_terminal`].
static TERMINAL_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take exclusive ownership of the terminal for the length of one frame or one
/// teardown. Poisoning is recovered from rather than propagated: a teardown or a
/// frame skipped because some *other* thread panicked while holding the lock is
/// strictly worse than doing it anyway.
///
/// `pub(crate)` so the launcher's menu loop — the crate's other render loop —
/// can take the same lock around its own frame.
pub(crate) fn lock_terminal_writer() -> std::sync::MutexGuard<'static, ()> {
    TERMINAL_WRITE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Write the escape sequences that undo `run`'s terminal setup — leave the
/// alternate screen, disable mouse capture, show the cursor — to `out`. Split
/// from [`restore_terminal`] so a unit test can drive an in-memory sink and
/// assert the emitted bytes, rather than writing to the process's shared stdout
/// (which races every other test in the single-process `cargo test` lane).
fn write_restore_sequences<W: io::Write>(out: &mut W) -> io::Result<()> {
    execute!(
        out,
        LeaveAlternateScreen,
        DisableMouseCapture,
        crossterm::cursor::Show
    )
}

/// Register the termination-signal listeners on the current Tokio runtime and
/// spawn a detached watcher that restores the terminal and exits `128 + signo`.
///
/// Returns the task handle so a caller whose process ends with the session
/// (`rocm dash`) can `abort()` it on a clean return; the persistent launcher hub
/// instead keeps its runtime alive and drops the handle, letting the watcher
/// live for the whole process so bare `rocm` stays killable across the sessions
/// it builds and drops (Tokio never unregisters its libc handler, so a
/// per-session watcher would go deaf the moment its runtime is dropped — see
/// `dash::run_launcher`).
///
/// Registration happens here, synchronously, and is surfaced via `?`; callers
/// invoke this BEFORE entering raw/alternate-screen mode so a failure never
/// leaves the terminal switched with no restore path, and a signal arriving
/// during startup is latched by the already-installed OS handlers.
///
/// The watcher ends the process with [`std::process::exit`], which does not
/// unwind — this is deliberate. SIGTERM/SIGINT means "stop now", so we restore
/// the terminal and leave promptly rather than racing an orderly teardown
/// against an imminent default-disposition kill. Two consequences are accepted
/// as the intended behavior: (1) an in-flight focused install/serve child is
/// left to the OS rather than reaped via `kill_on_drop`, matching the
/// pre-existing default disposition and avoiding truncating a mid-write child on
/// the way out; and (2) `run_async`'s embedded-daemon socket is not unlinked
/// here, but the daemon unlinks a stale socket on its next bind, so it self-heals.
///
/// More than one watcher can be live at once — bare `rocm` escalates from the
/// hub into a dashboard session, so the hub's process-lifetime watcher and
/// `run`'s session watcher coexist — and Tokio's signal registry is
/// process-global: one `kill` notifies *every* subscriber regardless of which
/// runtime registered it. Both watchers therefore wake on the same signal. The
/// [`SHUTTING_DOWN`] latch arbitrates: only the first one through restores the
/// terminal and exits, so two threads never race unsynchronised writes to
/// stdout nor call `std::process::exit` concurrently. See [`await_termination`].
pub fn spawn_termination_watcher() -> color_eyre::Result<tokio::task::JoinHandle<()>> {
    let termination = TerminationSignals::register()?;
    Ok(tokio::spawn(async move {
        let Some(code) = await_termination(termination, &SHUTTING_DOWN).await else {
            // A sibling watcher already claimed the shutdown and is about to
            // `exit`; this one must do nothing at all.
            return;
        };
        restore_terminal();
        std::process::exit(code);
    }))
}

/// Process-global "some watcher has claimed the termination path" latch.
///
/// Deliberately process-global rather than threaded through `ResolvedArgs`: the
/// hazard is two watchers on two *runtimes*, and a path-independent latch covers
/// every call site (present and future) without each one having to know whether
/// an outer watcher already exists.
///
/// One-shot by construction: it is claimed only by paths that go on to call
/// `std::process::exit` (the watcher body and [`exit_on_ctrl_c`]), so there is
/// no "after" to release it into, and every reader — the two render gates,
/// [`await_termination`]'s arbitration, [`restore_after_session`] — may treat a
/// claimed latch as "this process is ending". A teardown that does *not* end the
/// process must therefore never claim it; see [`restore_after_session`] for the
/// three separate ways that wedged the launcher hub. Mutual exclusion between
/// concurrent restores is [`RESTORE_LOCK`]'s job, not this latch's.
pub(crate) static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

/// Claim the single-shot shutdown path on `latch`. Returns `true` exactly once —
/// for whichever caller wins the swap — and `false` for every caller after it.
///
/// `SeqCst` because correctness here is "exactly one winner across threads", not
/// ordering of surrounding data; the stronger ordering costs nothing on a path
/// that runs at most once per process.
///
/// Parameterised over the latch (rather than reading [`SHUTTING_DOWN`] directly)
/// so tests can drive a fresh latch and stay order-independent — the process
/// global cannot be reset once a test has set it.
fn claim_shutdown(latch: &AtomicBool) -> bool {
    !latch.swap(true, Ordering::SeqCst)
}

/// Whether the shutdown path has already been claimed on `latch` — by a signal
/// watcher or by a typed Ctrl-C — meaning the terminal is being restored and the
/// process is about to exit.
///
/// Both render gates ([`draw_frame_unless_shutting_down`] and the launcher's
/// `draw_menu_unless_shutting_down`) consult this before every frame so a `draw`
/// cannot land after [`restore_terminal`] has run and undo it. See the ordering
/// note on [`restore_terminal`] for why that is the chosen fix rather than a lock
/// on every terminal write.
///
/// Takes the latch explicitly rather than reading [`SHUTTING_DOWN`] directly so
/// a test can drive a fresh one and stay order-independent — the process global
/// cannot be reset once a test has set it (same reason [`claim_shutdown`] is
/// parameterised). Production callers pass [`SHUTTING_DOWN`].
pub(crate) fn shutdown_claimed_on(latch: &AtomicBool) -> bool {
    latch.load(Ordering::SeqCst)
}

/// Whether `k` is a typed Ctrl-C.
///
/// Raw mode is why this must be a key match at all. `enable_raw_mode` clears
/// `ISIG` on Unix and `ENABLE_PROCESSED_INPUT` on Windows; with those off the
/// terminal driver does NOT translate the keystroke into SIGINT (or raise
/// `CTRL_C_EVENT`) — it hands the application the byte `0x03` like any other
/// key. So while the TUI is up, [`TerminationSignals`] covers an *externally*
/// delivered `kill -INT` / `GenerateConsoleCtrlEvent` (and a Ctrl-C during the
/// startup window before raw mode is entered), but never the gesture a user
/// performs inside the dashboard. That one arrives here.
///
/// Matches lowercase `c` only, mirroring
/// [`crate::ui::job_console::on_console_key`]: terminals commonly bind
/// Ctrl+Shift+C to copy, and claiming it would break a paste workflow.
pub(crate) const fn is_ctrl_c(k: KeyEvent) -> bool {
    matches!(k.code, KeyCode::Char('c')) && k.modifiers.contains(KeyModifiers::CONTROL)
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

/// End the process from a typed Ctrl-C, taking exactly the path an externally
/// delivered SIGINT takes: claim the shutdown latch, restore the terminal, exit
/// [`EXIT_CODE_SIGINT`]. Shared by the dashboard event loop and the launcher
/// menu — the two key loops are separate, and routing both here is what stops
/// the gesture from meaning different things in the two windows.
///
/// Never returns, and (like the watcher) exits without unwinding; see
/// [`spawn_termination_watcher`] for the consequences that are accepted there
/// and apply identically here.
pub(crate) fn exit_on_ctrl_c() -> ! {
    if claim_shutdown(&SHUTTING_DOWN) {
        restore_terminal();
        std::process::exit(EXIT_CODE_SIGINT);
    }
    // Lost the race to a watcher that has already claimed the shutdown and is
    // microseconds from `exit`. Park rather than racing a second
    // `std::process::exit`; the winner ends the process. `park` is allowed to
    // wake spuriously, hence the loop.
    loop {
        std::thread::park();
    }
}

/// Park until a termination signal arrives, then arbitrate on `latch`.
///
/// Returns `Some(128 + signo)` for the single watcher that wins the latch — that
/// caller must restore the terminal and exit with the code — and `None` for any
/// other watcher woken by the same process-global signal delivery.
///
/// Split out of [`spawn_termination_watcher`]'s task body so a test can drive the
/// whole register → receive → arbitrate path in-process; the body itself ends in
/// `std::process::exit` and can never be unit-tested.
async fn await_termination(termination: TerminationSignals, latch: &AtomicBool) -> Option<i32> {
    let code = termination.recv().await;
    claim_shutdown(latch).then_some(code)
}

/// Termination-signal listeners, registered up front so a signal that arrives
/// during terminal setup is latched by the OS/Tokio rather than taking the
/// default disposition.
///
/// [`TerminationSignals::register`] installs the OS handlers and is called
/// BEFORE the terminal is switched into raw/alternate-screen mode, so a
/// registration failure is surfaced (via `?`) before any terminal state is
/// mutated. [`TerminationSignals::recv`] then parks the watcher task until a
/// signal fires, returning the conventional `128 + signo` exit code the process
/// should report (SIGINT → 130, SIGTERM → 143).
///
/// Scope, stated narrowly because it is easy to overclaim: this covers signals
/// that arrive as signals — `kill -TERM` / `kill -INT` from a supervisor or
/// another process, and anything delivered during the startup window before
/// `enable_raw_mode`. It does NOT cover a Ctrl-C typed at the running TUI: raw
/// mode clears `ISIG`, so the terminal driver never turns that keystroke into a
/// SIGINT and it arrives as a key event instead. See [`is_ctrl_c`], which both
/// key loops route to this same restore-and-exit path.
#[cfg(unix)]
struct TerminationSignals {
    sigterm: tokio::signal::unix::Signal,
    sigint: tokio::signal::unix::Signal,
}

#[cfg(unix)]
impl TerminationSignals {
    fn register() -> color_eyre::Result<Self> {
        use tokio::signal::unix::{SignalKind, signal};

        // Register both handlers before entering raw mode. `?` on each
        // propagates a setup failure instead of parking forever; and if the
        // second registration fails, the first is dropped (unregistered) as we
        // return the error, so we never leave one signal silently swallowed.
        Ok(Self {
            sigterm: signal(SignalKind::terminate())?,
            sigint: signal(SignalKind::interrupt())?,
        })
    }

    async fn recv(mut self) -> i32 {
        tokio::select! {
            _ = self.sigterm.recv() => EXIT_CODE_SIGTERM,
            _ = self.sigint.recv() => EXIT_CODE_SIGINT,
        }
    }
}

/// Windows analog: console control events, which would otherwise skip terminal
/// restoration the same way. Tokio exposes the two as separate streams, so BOTH
/// are registered — `ctrl_c()` subscribes only to `CTRL_C_EVENT` and would miss
/// a Ctrl-Break. Ctrl-Break is the harder "terminate" gesture and maps to the
/// SIGTERM code; Ctrl-C maps to the SIGINT code.
///
/// Scope, narrowly, because the two arms differ and the difference matters:
///
/// - Ctrl-Break: `ENABLE_PROCESSED_INPUT` does not affect `CTRL_BREAK_EVENT`, so
///   this arm is live for the whole session and is what actually terminates a
///   running TUI through the signal path.
/// - Ctrl-C: crossterm's raw mode clears `ENABLE_PROCESSED_INPUT`, and with that
///   flag off the console delivers a typed Ctrl+C into the input buffer as a KEY
///   EVENT and never raises `CTRL_C_EVENT`. This stream therefore cannot fire
///   while the TUI is up; it is kept for the startup window before
///   `enable_raw_mode` and for a `GenerateConsoleCtrlEvent` sent by another
///   process. The typed gesture is handled as a key instead — see [`is_ctrl_c`],
///   which routes it to the same restore path.
#[cfg(windows)]
struct TerminationSignals {
    ctrl_c: tokio::signal::windows::CtrlC,
    ctrl_break: tokio::signal::windows::CtrlBreak,
}

#[cfg(windows)]
impl TerminationSignals {
    fn register() -> color_eyre::Result<Self> {
        use tokio::signal::windows::{ctrl_break, ctrl_c};

        // `?` propagates a registration failure before raw mode is entered,
        // rather than the old `let _ = ctrl_c().await` which treated a failed
        // registration as a received Ctrl-C and exited immediately.
        Ok(Self {
            ctrl_c: ctrl_c()?,
            ctrl_break: ctrl_break()?,
        })
    }

    async fn recv(mut self) -> i32 {
        tokio::select! {
            _ = self.ctrl_c.recv() => EXIT_CODE_SIGINT,
            _ = self.ctrl_break.recv() => EXIT_CODE_SIGTERM,
        }
    }
}

/// Conventional shell exit code for a process terminated by SIGINT (128 + 2).
const EXIT_CODE_SIGINT: i32 = 130;
/// Conventional shell exit code for a process terminated by SIGTERM (128 + 15).
const EXIT_CODE_SIGTERM: i32 = 143;

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
    // Pre-launch disk read of whether anything is already serving (issue
    // #145): covers the startup window before the daemon's first instance
    // snapshot lands, and the case where the daemon connection never
    // succeeds at all. `has_live_instance` stops consulting it once a real
    // snapshot arrives.
    state.startup_has_live_service = args.startup_has_live_service;
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
                        // `Some(false)`: Deny/Cancel already closed the prompt.
                        // `None`: cursor moved or key ignored — prompt stays open.
                        if state.resolve_quit_confirm_key(k.code) == Some(true) {
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
                    // The examine overlay, when open, owns all keys (read-only
                    // `rocm examine` job through the job-bridge).
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

    fn st() -> AppState {
        AppState::new("t".into(), "default-dark".into())
    }
    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// Serialises every test that touches the process-global signal machinery.
    ///
    /// The two lanes differ: Linux CI runs `cargo nextest` (one process per
    /// test, so this lock is a no-op), while the required Windows lane runs
    /// `cargo test`, which runs the whole binary's tests as THREADS IN ONE
    /// PROCESS. There, `termination_watcher_parks_until_aborted` has a live
    /// watcher whose body ends in `std::process::exit`; if the self-`kill` test
    /// below ran concurrently, that watcher would wake on the other test's
    /// signal and take the entire test binary down with exit 143. It would also
    /// steal the signal the other test is asserting on. Holding this lock for
    /// the whole of each test — including the runtime's `block_on` — makes the
    /// two strictly sequential.
    ///
    /// The tests are written as plain `#[test]` + an explicit runtime (rather
    /// than `#[tokio::test]`) precisely so the guard is held across `block_on`
    /// without holding a `std` lock across an `.await`.
    ///
    /// One residue this lock cannot undo, recorded so it is not rediscovered as
    /// a mystery: `TerminationSignals::register` installs Tokio's libc handler
    /// for SIGINT/SIGTERM process-wide, and Tokio never unregisters it — not on
    /// drop of the `Signal`, not on drop of the runtime. So from the first of
    /// these tests onward, the rest of a single-process `cargo test` run (the
    /// required Windows lane) is deaf to those signals: they are caught and
    /// discarded instead of terminating the binary. Harmless for the suite as it
    /// stands — nothing signals the test process except the test that does so
    /// deliberately, under this lock — but any future test that expects a signal
    /// to actually kill the test binary, or a CI step that relies on cancelling
    /// it with SIGINT, must not assume the default disposition is still in place.
    static SIGNAL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Upper bound on any `await_termination` in a test. The signal it waits for
    /// is already queued before the await starts, so the real latency is
    /// microseconds; this only exists so a broken registration fails the test
    /// instead of parking the `.await` forever and burning the lane's job
    /// timeout. A hanging test is worse than a failing one — it reports nothing.
    ///
    /// Gated because its only callers are: ungated, it is dead code on Windows
    /// and `-D warnings` fails that lane.
    #[cfg(unix)]
    const SIGNAL_AWAIT_TIMEOUT: Duration = Duration::from_secs(10);

    /// A current-thread runtime with the signal driver enabled, which
    /// `TerminationSignals::register` needs.
    fn signal_test_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("building a current-thread runtime for the signal tests")
    }

    #[test]
    fn only_the_first_caller_claims_the_shutdown_latch() {
        // The guard that stops the hub's process-lifetime watcher and a
        // session's watcher from both restoring the terminal and both calling
        // `process::exit` on one signal. Driven on a local latch so the test
        // never touches (or depends on the state of) the process global.
        let latch = AtomicBool::new(false);
        assert!(claim_shutdown(&latch), "the first claim must win");
        assert!(!claim_shutdown(&latch), "a second claim must lose");
        assert!(!claim_shutdown(&latch), "and so must every later one");

        // Under contention there must still be exactly one winner: a
        // non-atomic read-then-write would let several threads through.
        let contended = AtomicBool::new(false);
        let winners = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..16 {
                scope.spawn(|| {
                    if claim_shutdown(&contended) {
                        winners.fetch_add(1, Ordering::SeqCst);
                    }
                });
            }
        });
        assert_eq!(
            winners.load(Ordering::SeqCst),
            1,
            "exactly one of 16 racing watchers may claim the shutdown"
        );
    }

    #[test]
    fn claiming_the_shutdown_latch_suspends_rendering() {
        // The narrow serialization that stops a frame landing after
        // `restore_terminal()` has run and undoing it: both render loops gate on
        // the SAME latch the shutdown path claims, and the claim happens before
        // the restore begins. Driven on a local latch so the test never touches
        // the process global.
        let latch = AtomicBool::new(false);
        assert!(
            !shutdown_claimed_on(&latch),
            "rendering must be allowed while no shutdown has been claimed"
        );
        assert!(claim_shutdown(&latch), "the first claim must win");
        assert!(
            shutdown_claimed_on(&latch),
            "claiming the shutdown must suspend rendering, or a late frame can \
             repaint over the restored terminal"
        );
    }

    #[test]
    fn a_clean_session_teardown_restores_without_claiming_the_exit_latch() {
        // `run` returning is the one restore path with an "after": bare `rocm` is
        // a persistent hub, so control goes back to a live launcher menu. Claiming
        // the one-shot exit latch here wedged that hub three ways at once — the
        // render gate refused every later frame (blank front door), a typed Ctrl-C
        // parked forever in `exit_on_ctrl_c`, and every later signal lost the
        // claim in `await_termination` and was swallowed. So this path must
        // restore the terminal and leave the latch exactly as it found it.
        let latch = AtomicBool::new(false);
        let restored = std::cell::Cell::new(false);
        restore_after_session(&latch, || restored.set(true));
        assert!(
            restored.get(),
            "a clean session must restore the terminal it put into raw mode"
        );
        assert!(
            !shutdown_claimed_on(&latch),
            "a teardown that returns to a live process must not claim the exit \
             latch — nothing ever releases it, so the hub is wedged from here on"
        );

        // The one thing it may key on the latch: when a watcher or a typed
        // Ctrl-C has already claimed the exit, that owner is microseconds from
        // `process::exit` and owns the teardown; restoring again is redundant.
        let claimed = AtomicBool::new(true);
        let restored_again = std::cell::Cell::new(false);
        restore_after_session(&claimed, || restored_again.set(true));
        assert!(
            !restored_again.get(),
            "an exiting process's teardown belongs to whoever claimed the exit"
        );
    }

    #[test]
    fn ctrl_c_is_recognised_as_a_key_because_raw_mode_suppresses_the_signal() {
        // Raw mode clears ISIG (and ENABLE_PROCESSED_INPUT on Windows), so a
        // typed Ctrl-C never becomes a signal — it arrives here as a key event.
        let ctrl = |c: char| KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL);
        assert!(is_ctrl_c(ctrl('c')), "Ctrl+C must be recognised");
        assert!(
            !is_ctrl_c(press(KeyCode::Char('c'))),
            "a bare `c` must not terminate the session"
        );
        assert!(!is_ctrl_c(ctrl('d')), "Ctrl+D is a different key");
        // Terminals commonly bind Ctrl+Shift+C to copy; claiming it would kill
        // the session on a copy.
        assert!(
            !is_ctrl_c(KeyEvent::new(
                KeyCode::Char('C'),
                KeyModifiers::CONTROL | KeyModifiers::SHIFT
            )),
            "Ctrl+Shift+C is copy, not terminate"
        );
    }

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
    fn termination_watcher_parks_until_aborted() {
        let _guard = SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        signal_test_runtime().block_on(async {
            // The real wiring `run` depends on: registration must succeed, and
            // the spawned task must stay parked on `recv()` (never resolving on
            // its own and exiting the process), until the clean-return path
            // aborts it.
            let handle = spawn_termination_watcher()
                .expect("registering the termination-signal listeners must succeed");
            // Let the task actually start and park; on a current-thread runtime
            // a freshly spawned task has not been polled yet, so without this
            // `is_finished` would be trivially false.
            tokio::task::yield_now().await;
            assert!(
                !handle.is_finished(),
                "the watcher must stay parked while no signal has arrived"
            );

            handle.abort();
            let err = handle
                .await
                .expect_err("an aborted watcher must not report completion");
            assert!(
                err.is_cancelled(),
                "the watcher must end by cancellation, not by panicking: {err:?}"
            );
        });
    }

    // Unix-only. This test sends real signals to its own process, which is safe
    // ONLY because it drives `await_termination` directly: the watcher body that
    // calls `std::process::exit` is never run here. `TerminationSignals::register`
    // installs the handlers *before* the `kill`, so the signal is caught rather
    // than taking its default (fatal) disposition. Both listeners are registered
    // before the single `kill` on purpose — that is exactly the hub-watcher +
    // session-watcher shape, and Tokio's process-global registry wakes both.
    #[cfg(unix)]
    #[test]
    fn termination_signals_yield_shell_exit_codes_and_only_one_watcher_shuts_down() {
        let _guard = SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        signal_test_runtime().block_on(async {
            // Scope, so nobody reads more into this than it proves: the
            // expectation is built from the same `EXIT_CODE_*` constants the
            // code under test returns, so this pins the SIGTERM→sigterm-code /
            // SIGINT→sigint-code *mapping* (swapping the two arms turns it red)
            // but not the literal values. Editing `EXIT_CODE_SIGINT` to 7 leaves
            // this green. The literals 130/143 are pinned by the e2e scenarios
            // `dash-17` … `dash-21` in `tests/e2e-cucumber/features/dash.feature`,
            // which assert the shell-visible exit status of a real process.
            for (signo, expected) in [
                (libc::SIGTERM, EXIT_CODE_SIGTERM),
                (libc::SIGINT, EXIT_CODE_SIGINT),
            ] {
                // A fresh latch per kind keeps the test order-independent.
                let latch = AtomicBool::new(false);
                let hub_watcher = TerminationSignals::register()
                    .expect("registering the hub listeners must succeed");
                let session_watcher = TerminationSignals::register()
                    .expect("registering the session listeners must succeed");

                // SAFETY: `raise` is an async-signal-safe libc call with no
                // arguments to get wrong, and both listeners above are already
                // installed, so the signal is delivered to Tokio's handler
                // instead of terminating the test binary.
                #[allow(unsafe_code)] // libc FFI
                let rc = unsafe { libc::raise(signo) };
                assert_eq!(rc, 0, "raise({signo}) failed");

                // Both `await_termination`s are bounded. A regression that breaks
                // *registration* of one signal kind (rather than mis-mapping its
                // exit code) leaves the `.await` parked forever, and an unbounded
                // await would burn the required lane's job timeout instead of
                // reporting a failure. The bound is generous — the signal is
                // already queued by the `raise` above, so the await resolves in
                // microseconds; anything near 10 s is a genuine hang.
                let received = tokio::time::timeout(
                    SIGNAL_AWAIT_TIMEOUT,
                    await_termination(hub_watcher, &latch),
                )
                .await
                .unwrap_or_else(|_| {
                    panic!(
                        "the first watcher never received signal {signo} within \
                         {SIGNAL_AWAIT_TIMEOUT:?} — the listener for it is not \
                         registered, so the watcher would park forever instead of \
                         restoring the terminal"
                    )
                });
                assert_eq!(
                    received,
                    Some(expected),
                    "the first watcher must receive signal {signo} and map it to \
                     the conventional 128 + signo exit code"
                );

                let stood_down = tokio::time::timeout(
                    SIGNAL_AWAIT_TIMEOUT,
                    await_termination(session_watcher, &latch),
                )
                .await
                .unwrap_or_else(|_| {
                    panic!(
                        "the second watcher never woke for signal {signo} within \
                         {SIGNAL_AWAIT_TIMEOUT:?} — Tokio's registry must wake \
                         every listener registered for a kind, not just the first"
                    )
                });
                assert_eq!(
                    stood_down, None,
                    "the second watcher woken by the same signal must stand down \
                     rather than race a concurrent restore + exit"
                );
            }
        });
    }

    // Unix-only: crossterm emits ANSI escape sequences to a generic writer on
    // Unix, so an in-memory sink captures the real bytes. On Windows crossterm
    // drives the console via the WinAPI backend instead of writing ANSI, and
    // `execute!` to a `Vec` errors with "Initial console modes not set" — there
    // is no console to configure. Production `restore_terminal()` passes a real
    // stdout handle, so the Windows path is exercised there, not by this sink.
    #[cfg(unix)]
    #[test]
    fn write_restore_sequences_leaves_alt_screen_disables_mouse_and_shows_cursor() {
        // The restore path must undo all three things `run` set up: leave the
        // alternate screen, disable mouse capture, show the cursor. Driving an
        // in-memory sink asserts the actual emitted bytes without touching the
        // process's shared terminal state — the global `disable_raw_mode()` half
        // is deliberately outside this function, so nothing here races other
        // tests in the single-process `cargo test` lane.
        let mut sink: Vec<u8> = Vec::new();
        write_restore_sequences(&mut sink).expect("writing to a Vec cannot fail");
        let emitted = String::from_utf8(sink).expect("restore sequences are ASCII escapes");
        assert!(
            emitted.contains("\x1b[?1049l"),
            "expected the leave-alt-screen sequence in {emitted:?}"
        );
        // `DisableMouseCapture` is one command but five terminal modes:
        // crossterm 0.28 expands it to SGR-encoding, urxvt-encoding, any-motion,
        // button-event and normal tracking, turned off in that order. Assert
        // the whole block rather than a single mode so dropping the command from
        // `write_restore_sequences` cannot leave this test green — a terminal
        // left reporting mouse events after `rocm dash` exits is exactly the
        // broken-terminal state this restore path exists to prevent. If a
        // crossterm bump changes the expansion, re-derive it from a sink run
        // rather than weakening the assertion.
        let disable_mouse = "\x1b[?1006l\x1b[?1015l\x1b[?1003l\x1b[?1002l\x1b[?1000l";
        assert!(
            emitted.contains(disable_mouse),
            "expected the disable-mouse-capture sequences {disable_mouse:?} in {emitted:?}"
        );
        assert!(
            emitted.contains("\x1b[?25h"),
            "expected the show-cursor sequence in {emitted:?}"
        );
    }

    #[test]
    fn startup_focus_gate_only_opens_onboarding_for_explicit_setup_focus() {
        // Regression guard: this calls `apply_startup_focus`, the same gate
        // `event_loop` uses, not just `AppState::new` — which takes no focus
        // argument and hardcodes `onboarding: None` regardless, so it cannot
        // exhibit an auto-open regression either way.
        let mut s = st();
        assert!(apply_startup_focus(&mut s, None).is_empty());
        assert!(
            s.onboarding.is_none(),
            "no --focus flag must not open onboarding"
        );

        let mut s = st();
        assert!(apply_startup_focus(&mut s, Some(Focus::Setup)).is_empty());
        assert!(
            s.onboarding.is_some(),
            "an explicit Focus::Setup must open onboarding"
        );
    }

    #[test]
    fn should_skip_daemon_predicate_matches_focus() {
        // The dashboard (focus=None) keeps the daemon client + chat backend; any
        // focus skips both. The render branch reuses this same predicate.
        assert!(!should_skip_daemon(None));
        assert!(should_skip_daemon(Some(Focus::Setup)));
        assert!(should_skip_daemon(Some(Focus::Serve)));
        assert!(should_skip_daemon(Some(Focus::Examine)));
        // focus=None never self-exits — the dash loop only breaks on Quit/EOF.
        assert!(!st().focused_should_exit(None));
    }

    #[test]
    fn open_overlay_for_focus_opens_the_right_overlay() {
        let mut s = st();
        assert!(open_overlay_for_focus(&mut s, Focus::Setup).is_empty());
        assert!(s.onboarding.is_some());
        assert!(s.serve_wizard.is_none() && s.examine_manager.is_none());

        let mut s = st();
        assert!(open_overlay_for_focus(&mut s, Focus::Serve).is_empty());
        assert!(s.serve_wizard.is_some());
        assert!(s.onboarding.is_none() && s.examine_manager.is_none());

        let mut s = st();
        let fx = open_overlay_for_focus(&mut s, Focus::Examine);
        assert!(s.examine_manager.is_some());
        assert!(s.onboarding.is_none() && s.serve_wizard.is_none());
        assert_eq!(fx.len(), 1, "examine auto-runs on open");
    }

    #[test]
    fn focused_examine_auto_runs_rocm_examine() {
        let mut s = st();
        let fx = open_overlay_for_focus(&mut s, Focus::Examine);
        assert_eq!(fx.len(), 1, "exactly one spawn side effect on open");
        match &fx[0] {
            rocm_dash_core::state::SideEffect::SpawnJob { cmd, args, .. } => {
                assert!(cmd.contains("rocm"), "cmd resolves to the rocm exe: {cmd}");
                assert!(
                    args.iter().any(|a| a == "examine"),
                    "examine in args: {args:?}"
                );
            }
            other => panic!("expected SpawnJob, got {other:?}"),
        }
        assert_eq!(
            s.examine_manager.as_ref().unwrap().active_job.as_deref(),
            Some("examine"),
            "the auto-run wires the active job"
        );
    }

    #[test]
    fn focused_exit_gate_holds_until_examine_closed_at_root() {
        let mut s = st();
        // Focused Diagnose: examine opens AND auto-runs → a job-console sub-state.
        let _ = open_overlay_for_focus(&mut s, Focus::Examine);
        assert!(s.examine_manager.as_ref().unwrap().active_job.is_some());
        assert!(
            !s.focused_should_exit(Some(Focus::Examine)),
            "a running job keeps the launcher out"
        );

        // Job terminal → first Esc dismisses the console back to the intro card;
        // the overlay is still open, so the gate stays shut.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        let _ = crate::ui::examine_manager::on_key(
            &mut s.examine_manager,
            &mut s.jobs,
            press(KeyCode::Esc),
        );
        assert!(
            s.examine_manager.is_some(),
            "console dismissed, overlay stays"
        );
        assert!(
            !s.focused_should_exit(Some(Focus::Examine)),
            "at the intro (not root-closed) the gate is still shut"
        );

        // Second Esc at the intro (root) closes the overlay → now exit to menu.
        let _ = crate::ui::examine_manager::on_key(
            &mut s.examine_manager,
            &mut s.jobs,
            press(KeyCode::Esc),
        );
        assert!(s.examine_manager.is_none(), "root Esc closes the overlay");
        assert!(
            s.focused_should_exit(Some(Focus::Examine)),
            "closed at root → return to the launcher"
        );
    }

    #[test]
    fn focused_close_keys_swallowed_while_job_runs() {
        // Regression for the mid-job ejection defect: `q` and running-`Esc` must
        // be swallowed by the focused host while the job is non-terminal, so the
        // overlay is never nulled (which would tear the runtime down and kill the
        // child via kill_on_drop mid-write).
        let mut s = st();
        let _ = open_overlay_for_focus(&mut s, Focus::Examine); // auto-runs a job
        assert!(s.has_active_console(), "examine console is live");
        // Running job → q and Esc are blocked; Ctrl+C ('c') is NOT (it cancels).
        assert!(focused_close_key_blocked(
            &s,
            Some(Focus::Examine),
            KeyCode::Char('q')
        ));
        assert!(focused_close_key_blocked(
            &s,
            Some(Focus::Examine),
            KeyCode::Esc
        ));
        assert!(!focused_close_key_blocked(
            &s,
            Some(Focus::Examine),
            KeyCode::Char('c')
        ));
        // The dashboard (focus=None) never blocks — behavior is unchanged there.
        assert!(!focused_close_key_blocked(&s, None, KeyCode::Char('q')));

        // Because those keys are swallowed (never routed to the manager), the
        // overlay stays open and the exit gate stays shut mid-job.
        assert!(s.examine_manager.is_some());
        assert!(!s.focused_should_exit(Some(Focus::Examine)));

        // Once the job is terminal, close keys are allowed again → normal exit.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: "examine".into(),
            code: 0,
        });
        assert!(
            !focused_close_key_blocked(&s, Some(Focus::Examine), KeyCode::Char('q')),
            "a terminal job no longer blocks exit (the child already exited)"
        );
    }

    #[test]
    fn focused_gate_shut_across_serve_sub_states() {
        // Exit-at-root (b)+(c): the focused gate stays shut while a folder
        // browser / model picker / approval is open — it only opens at root.
        let recipes: Vec<crate::ui::model_picker::ModelRecipeSummary> = Vec::new();
        let mut s = st();
        let _ = open_overlay_for_focus(&mut s, Focus::Serve);

        // (b) Tab on the Model field opens the folder-browser sub-popup.
        let _ = crate::ui::serve_wizard::on_key(
            &mut s.serve_wizard,
            &mut s.jobs,
            &recipes,
            press(KeyCode::Tab),
        );
        assert!(s.serve_wizard.as_ref().unwrap().browser.is_some());
        assert!(
            !s.focused_should_exit(Some(Focus::Serve)),
            "gate shut while the folder browser is open"
        );
        // Esc closes the sub-popup, not the wizard → still shut.
        let _ = crate::ui::serve_wizard::on_key(
            &mut s.serve_wizard,
            &mut s.jobs,
            &recipes,
            press(KeyCode::Esc),
        );
        assert!(s.serve_wizard.as_ref().unwrap().browser.is_none());
        assert!(s.serve_wizard.is_some());
        assert!(!s.focused_should_exit(Some(Focus::Serve)));

        // (c) Stage an approval (valid model, Launch field, Enter).
        {
            let w = s.serve_wizard.as_mut().unwrap();
            w.model = "org/model".to_string();
            w.field = crate::ui::serve_wizard::FIELDS.len() - 1; // Launch
        }
        let _ = crate::ui::serve_wizard::on_key(
            &mut s.serve_wizard,
            &mut s.jobs,
            &recipes,
            press(KeyCode::Enter),
        );
        assert!(
            s.serve_wizard.as_ref().unwrap().approval.is_some(),
            "a launch approval is pending"
        );
        assert!(
            !s.focused_should_exit(Some(Focus::Serve)),
            "gate shut while an approval is pending"
        );

        // Only a root close (wizard → None) opens the gate.
        s.serve_wizard = None;
        assert!(s.focused_should_exit(Some(Focus::Serve)));
    }

    // --- Home tab update check (background job-bridge trigger) ---

    // Serializes every test in this group against
    // `refresh_update_status_skips_spawn_when_disabled_via_env`, which toggles
    // `ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK` — process env is shared across
    // test threads, so an unguarded test can observe the var mid-toggle and
    // spuriously see `refresh_update_status` skip the spawn it expects.
    static UPDATE_CHECK_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn refresh_update_status_spawns_on_first_due_tick() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        assert!(!s.update_status_pending);
        let fx = refresh_update_status(&mut s);
        assert!(!fx.is_empty(), "a due check spawns a job");
        assert!(s.update_status_pending);
        assert!(s.jobs.job(HOME_UPDATE_CHECK_JOB_ID).is_some());
    }

    #[test]
    fn refresh_update_status_does_not_duplicate_spawn_while_running() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let fx = refresh_update_status(&mut s);
        assert!(!fx.is_empty());
        assert!(s.update_status_pending);

        // Still pending, job still running (non-terminal) → no-op, no second spawn.
        let fx2 = refresh_update_status(&mut s);
        assert!(
            fx2.is_empty(),
            "no duplicate spawn while pending and running"
        );
        assert!(s.update_status_pending);
        assert_eq!(s.update_status, UpdateStatus::Unknown);
    }

    #[test]
    fn refresh_update_status_rearms_due_at_when_pending_job_vanishes() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);
        assert!(s.update_status_pending);

        // Not reachable today (jobs are never removed), but if it ever is,
        // `update_check_due_at` must still be pushed out — otherwise every
        // subsequent tick would spawn a new check immediately.
        s.jobs.jobs.remove(HOME_UPDATE_CHECK_JOB_ID);
        let fx = refresh_update_status(&mut s);
        assert!(fx.is_empty());
        assert!(!s.update_status_pending);
        assert!(
            s.update_check_due_at > std::time::Instant::now(),
            "due_at must be re-armed, not left in the past"
        );
    }

    #[test]
    fn refresh_update_status_resolves_from_terminal_success_json() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);
        assert!(s.update_status_pending);

        s.jobs.apply(rocm_dash_core::state::StateEvent::JobLine {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            line: serde_json::json!({
                "runtimes": [{
                    "runtime_key": "rocm",
                    "channel": "stable",
                    "family": "rocm",
                    "installed_version": "7.0.0",
                    "latest_version": "7.1.0",
                    "status": "update_available",
                    "message": null,
                }]
            })
            .to_string(),
        });
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            code: 0,
        });

        let fx = refresh_update_status(&mut s);
        assert!(fx.is_empty(), "resolving a terminal job spawns nothing");
        assert!(!s.update_status_pending);
        assert_eq!(
            s.update_status,
            UpdateStatus::UpdateAvailable {
                latest_version: "7.1.0".into()
            }
        );
    }

    #[test]
    fn refresh_update_status_resolves_to_error_on_terminal_failure() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);
        assert!(s.update_status_pending);

        // Nonzero exit → Error, regardless of any output on the ring.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            code: 1,
        });
        let fx = refresh_update_status(&mut s);
        assert!(fx.is_empty());
        assert!(!s.update_status_pending);
        assert_eq!(s.update_status, UpdateStatus::Error);
    }

    #[test]
    fn refresh_update_status_resolves_to_error_on_unparsable_success_output() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);

        // Exit 0 but no valid JSON line on the ring → Error, not a silent hang.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobLine {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            line: "not json".into(),
        });
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            code: 0,
        });
        let fx = refresh_update_status(&mut s);
        assert!(fx.is_empty());
        assert!(!s.update_status_pending);
        assert_eq!(s.update_status, UpdateStatus::Error);
    }

    #[test]
    fn refresh_update_status_spawn_args_include_bounded_timeout() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);
        let job = s
            .jobs
            .job(HOME_UPDATE_CHECK_JOB_ID)
            .expect("job spawned on first due tick");
        // Pinned to a literal, not `HOME_UPDATE_CHECK_TIMEOUT_SECS`: comparing
        // the constant to itself can never catch an unintentional change to
        // its value. A literal forces a deliberate test update (and a second
        // thought) whenever the bound changes.
        assert_eq!(
            job.args.last().map(String::as_str),
            Some("5"),
            "the background check's timeout bound must stay a deliberate choice: {:?}",
            job.args
        );
        assert!(job.args.iter().any(|a| a == "--timeout-secs"));
    }

    #[test]
    fn refresh_update_status_skips_spawn_when_disabled_via_env() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: serialized by `UPDATE_CHECK_ENV_TEST_LOCK`; no other thread
        // reads/writes this var concurrently.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK", "1");
        }
        let result = std::panic::catch_unwind(|| {
            let mut s = st();
            let fx = refresh_update_status(&mut s);
            assert!(fx.is_empty(), "a disabled check must not spawn a job");
            assert!(!s.update_status_pending);
            assert!(s.jobs.job(HOME_UPDATE_CHECK_JOB_ID).is_none());
        });
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK");
        }
        result.unwrap();
    }

    #[test]
    fn reduce_update_json_all_up_to_date_or_ahead_is_up_to_date() {
        let doc = serde_json::json!({
            "runtimes": [
                {"status": "up_to_date"},
                {"status": "ahead_of_index"},
            ]
        });
        assert_eq!(reduce_update_json(&doc), UpdateStatus::UpToDate);
    }

    #[test]
    fn reduce_update_json_mixed_up_to_date_and_error_is_error_not_up_to_date() {
        // One runtime resolved cleanly, one didn't — asserting "Up to date"
        // here would be a false claim about the runtime that errored.
        let doc = serde_json::json!({
            "runtimes": [
                {"status": "up_to_date"},
                {"status": "error", "message": "boom"},
            ]
        });
        assert_eq!(reduce_update_json(&doc), UpdateStatus::Error);
    }

    #[test]
    fn reduce_update_json_unrecognized_status_is_error() {
        let doc = serde_json::json!({
            "runtimes": [{"status": "something_new"}]
        });
        assert_eq!(reduce_update_json(&doc), UpdateStatus::Error);
    }

    #[test]
    fn reduce_update_json_repair_available_is_update_available_not_error() {
        // A same-version composition repair is as actionable as a version
        // bump — the tile must not report "check failed" for it.
        let doc = serde_json::json!({
            "runtimes": [{"status": "repair_available", "latest_version": "6.4.0"}]
        });
        assert_eq!(
            reduce_update_json(&doc),
            UpdateStatus::UpdateAvailable {
                latest_version: "6.4.0".to_owned()
            }
        );
    }

    #[test]
    fn reduce_update_json_missing_latest_version_is_still_update_available() {
        // A row with an actionable status but no `latest_version` must not be
        // silently skipped in favor of the up-to-date/error checks below it —
        // that would misreport a real update as "check failed".
        let doc = serde_json::json!({
            "runtimes": [{"status": "update_available"}]
        });
        assert_eq!(
            reduce_update_json(&doc),
            UpdateStatus::UpdateAvailable {
                latest_version: "(version unknown)".to_owned()
            }
        );
    }
}
