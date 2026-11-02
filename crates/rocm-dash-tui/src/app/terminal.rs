// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Terminal restoration and shutdown arbitration.
//!
//! Leaving raw/alternate-screen mode and restoring the cursor from one
//! serialized place (`restore_terminal`, `lock_terminal_writer`); the
//! process-wide `SHUTTING_DOWN` latch that decides which of several racing
//! paths restores the terminal and exits the process; the termination-signal
//! watcher (`spawn_termination_watcher`, whose `TerminationSignals` has a
//! `cfg(unix)` and a `cfg(windows)` variant); and the Ctrl+C key path
//! (`is_ctrl_c`, `exit_on_ctrl_c`) that raw mode forces onto the key loops.
//!
//! A leaf module, split out of `app/event_loop.rs`: the loop calls into it and
//! nothing here calls back.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::{DisableMouseCapture, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{LeaveAlternateScreen, disable_raw_mode};

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
/// watcher even though the [`ratatui::Terminal`] backend owns its own `stdout`
/// clone), then writes the alt-screen/mouse/cursor restore sequences to a fresh
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
/// signal watcher resumed past its `.await` races [`run`](super::event_loop::run)'s clean-quit teardown
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
pub(super) fn write_restore_sequences<W: io::Write>(out: &mut W) -> io::Result<()> {
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
pub(super) fn claim_shutdown(latch: &AtomicBool) -> bool {
    !latch.swap(true, Ordering::SeqCst)
}

/// Whether the shutdown path has already been claimed on `latch` — by a signal
/// watcher or by a typed Ctrl-C — meaning the terminal is being restored and the
/// process is about to exit.
///
/// Both render gates
/// ([`draw_frame_unless_shutting_down`](super::event_loop::draw_frame_unless_shutting_down)
/// and the launcher's `draw_menu_unless_shutting_down`) consult this before
/// every frame so a `draw` cannot land after [`restore_terminal`] has run and
/// undo it. See the ordering
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::test_support::press;
    #[cfg(unix)]
    use std::time::Duration;

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
}
