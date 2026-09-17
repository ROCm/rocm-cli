// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Black-box driver for the interactive dash/chat TUI.
//!
//! The rest of the suite spawns `rocm` with piped stdin/stdout via
//! `std::process::Command`. That can never exercise the interactive dashboard:
//! the CLI only enters the crossterm raw-mode event loop when both stdin and
//! stdout are a real terminal (`rocm_core::interactive_terminal`), and a pipe is
//! not. So the dash was previously "untestable black-box" (e.g. the chat privacy
//! notice) and only had in-process render tests.
//!
//! This driver closes that gap the way a user's terminal does: it spawns the
//! real binary under a pseudo-terminal (`portable-pty`), feeds keystrokes to the
//! master side, and parses the emitted byte stream into an emulated screen grid
//! (`vt100`). Assertions read the *current visible screen* — the same thing a
//! user sees — never the raw output transcript, so stale/erased frames or partial
//! escape sequences can't cause false matches.
//!
//! It stays black-box: it drives the compiled binary and reads its terminal
//! output, importing nothing from the product crates.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use portable_pty::{Child, CommandBuilder, MasterPty, PtySize, native_pty_system};

use e2e_cucumber::panic_capture::panic_message;
use e2e_cucumber::reader_failure::{ReaderFailure, ReaderFailureObservation};
use e2e_cucumber::send_until::{RetryTiming, TerminalState, send_until as retry_send_until};

use crate::E2eWorld;

/// Fixed terminal geometry. Pinning the size keeps layout (and therefore the
/// text we assert on) deterministic across hosts, independent of the ambient
/// terminal.
const ROWS: u16 = 24;
const COLS: u16 = 80;
/// Taller geometry for journeys that assert rows below the dashboard's summary
/// cards (managed instances and live serving metrics).
const DETAIL_ROWS: u16 = 40;
const DETAIL_COLS: u16 = 120;

/// How often `wait_for_*` re-checks the screen/process while waiting. This is a
/// poll cadence, not a fixed readiness sleep: every wait has a deadline and
/// returns the instant its condition holds.
const POLL_INTERVAL: Duration = Duration::from_millis(20);
/// How long [`TuiSession::send_until`] waits for a key to take effect before
/// sending it again. Long enough that a busy host is not spammed with repeats,
/// short enough that several attempts fit inside a normal step timeout.
const KEY_RESEND_INTERVAL: Duration = Duration::from_millis(500);
/// Maximum time to let the PTY reader consume the child's final frame after the
/// process exits. This is bounded so a misbehaving PTY cannot stall a scenario.
const DRAIN_TIMEOUT: Duration = Duration::from_millis(250);

/// Default wall-clock budget for a single wait. Generous enough for a cold dash
/// start plus the embedded-daemon connect, while still turning a genuine hang
/// into a prompt, diagnosable failure rather than a CI-timeout.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Wall-clock budget for a single wait, overridable with `E2E_TUI_TIMEOUT_SECS`.
///
/// Mirrors `E2E_SERVE_TIMEOUT_SECS` in `serving_steps.rs`: the assertion is
/// right, the budget is what varies by host. The self-hosted Strix lanes share
/// one physical machine, so a TUI frame that renders well inside 30s on an idle
/// runner can miss it when a sibling lane is loading a model on the same box —
/// which shows up as an unrelated-looking flake, not as a real hang.
///
/// Deliberately a wait budget and not a retry: a genuine hang must still fail.
#[must_use]
pub fn default_timeout() -> Duration {
    Duration::from_secs(
        std::env::var("E2E_TUI_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|secs| *secs > 0)
            .unwrap_or(DEFAULT_TIMEOUT_SECS),
    )
}

/// A running `rocm` TUI attached to a pseudo-terminal.
pub struct TuiSession {
    child: Box<dyn Child + Send + Sync>,
    writer: Box<dyn Write + Send>,
    /// The emulated screen, updated continuously by the reader thread.
    parser: Arc<Mutex<vt100::Parser>>,
    reader_stop: Arc<AtomicBool>,
    reader: Option<JoinHandle<()>>,
    /// Published by the reader thread if `vt100::Parser::process` panics. Keeps
    /// terminal failure state after the one-time diagnostic is consumed, so a
    /// retry cannot lose the cause while the reader thread is still finishing.
    reader_failure: Arc<ReaderFailure>,
    /// Kept alive for the lifetime of the session: the reader/writer are cloned
    /// from it, and dropping it early would close the PTY.
    master: Box<dyn MasterPty + Send>,
    /// `true` once the child has been reaped, so `Drop` doesn't kill/wait twice.
    finished: bool,
    /// Guards a single coverage record per session.
    recorded: bool,
    /// Whether this is a chat session (`rocm chat`) — chat quits via the `/quit`
    /// slash command while the input is focused, whereas the dashboard quits with
    /// a bare `q` (which chat would otherwise consume as typed input).
    is_chat: bool,
    scenario: Option<String>,
    argv: Vec<String>,
}

impl std::fmt::Debug for TuiSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiSession")
            .field("argv", &self.argv)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl TuiSession {
    /// Spawn `rocm <args>` under a fresh PTY with the scenario's isolated
    /// environment. The child renders into the emulated screen immediately; use
    /// [`wait_for_screen`](Self::wait_for_screen) to synchronise before asserting.
    pub fn spawn(world: &E2eWorld, args: &[&str]) -> Result<Self, String> {
        Self::spawn_binary(world, crate::rocm_binary(), args)
    }

    /// Like [`spawn`](Self::spawn), but overlaying `extra_env` on top of the
    /// scenario's isolation environment — for a step whose `Given` planted
    /// scenario-owned state (e.g. a shell rc file) that only the piped
    /// (`run_rocm_with_env`) path would otherwise pick up, since [`pty_env`]'s
    /// `HOME`/lack of `SHELL` are the PTY's own isolation, not that state.
    pub fn spawn_with_env(
        world: &E2eWorld,
        args: &[&str],
        extra_env: &[(&str, &str)],
    ) -> Result<Self, String> {
        Self::spawn_binary_with_env(world, crate::rocm_binary(), args, extra_env)
    }

    /// Spawn a specific `rocm` binary under a fresh PTY.
    ///
    /// Most scenarios use [`spawn`](Self::spawn) and exercise the harness-built
    /// binary. Install-lifecycle scenarios use this entry point so the terminal
    /// journey executes the binary copied by the real installer instead.
    pub fn spawn_binary(
        world: &E2eWorld,
        binary: impl AsRef<std::ffi::OsStr>,
        args: &[&str],
    ) -> Result<Self, String> {
        Self::spawn_binary_with_env(world, binary, args, &[])
    }

    fn spawn_binary_with_env(
        world: &E2eWorld,
        binary: impl AsRef<std::ffi::OsStr>,
        args: &[&str],
        extra_env: &[(&str, &str)],
    ) -> Result<Self, String> {
        let pair = native_pty_system()
            .openpty(PtySize {
                rows: ROWS,
                cols: COLS,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| format!("openpty failed: {e}"))?;

        let mut cmd = CommandBuilder::new(binary);
        for arg in args {
            cmd.arg(arg);
        }
        // Inherit the parent environment first, then overlay the isolation and
        // deterministic-terminal vars — mirroring how the piped `run_rocm` path
        // (`std::process::Command`, which inherits by default) resolves PATH and
        // shared libraries, so the two spawn paths behave identically.
        for (key, value) in std::env::vars_os() {
            cmd.env(key, value);
        }
        for (key, value) in world.isolate_env().into_iter().chain(world.pty_env()) {
            cmd.env(key, value);
        }
        // Behavioural fixtures attached by Given steps apply to PTY commands too,
        // just as they do to the piped `run_rocm_with_scenario_env` path.
        for (key, value) in &world.command_env {
            cmd.env(key, value);
        }
        // Caller-supplied overrides win over the scenario's own isolation
        // (e.g. a `Given` step's HOME/SHELL for state it planted itself).
        for (key, value) in extra_env {
            cmd.env(key, value);
        }
        // Provider configuration changes product startup semantics: a host API
        // key or endpoint suppresses local managed-service detection. These PTY
        // journeys exercise deterministic local/mock chat, so do not let the
        // developer's shell or CI credential environment select a cloud backend.
        for key in [
            "ROCMDASH_CHAT_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "ANTHROPIC_API_KEY",
        ] {
            cmd.env_remove(key);
        }
        // Deterministic terminal type; the PTY ioctl size above is authoritative
        // for crossterm, with COLUMNS/LINES as a belt-and-braces fallback.
        cmd.env("TERM", "xterm-256color");
        cmd.env("COLUMNS", COLS.to_string());
        cmd.env("LINES", ROWS.to_string());

        let mut child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| format!("failed to spawn rocm under a pty: {e}"))?;
        // Drop the slave in the parent: only the child needs it. Keeping it open
        // would prevent the reader from seeing EOF when the child exits.
        drop(pair.slave);

        // `Child` does not kill the process on drop. Reap it if either remaining
        // fallible setup step fails before `TuiSession` takes ownership.
        let mut reap_orphan = |error: String| -> String {
            let _ = child.kill();
            let _ = child.wait();
            error
        };
        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| reap_orphan(format!("failed to clone pty reader: {e}")))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|e| reap_orphan(format!("failed to take pty writer: {e}")))?;

        let parser = Arc::new(Mutex::new(vt100::Parser::new(ROWS, COLS, 0)));
        let reader_stop = Arc::new(AtomicBool::new(false));
        let reader_failure = Arc::new(ReaderFailure::default());
        let reader = spawn_reader(
            reader,
            Arc::clone(&parser),
            Arc::clone(&reader_stop),
            Arc::clone(&reader_failure),
        );

        Ok(Self {
            child,
            writer,
            parser,
            reader_stop,
            reader: Some(reader),
            reader_failure,
            master: pair.master,
            finished: false,
            recorded: false,
            is_chat: args.first() == Some(&"chat"),
            scenario: world.current_scenario.clone(),
            argv: args.iter().map(|s| (*s).to_string()).collect(),
        })
    }

    /// Whether the child has exited and been reaped successfully by a wait.
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    /// The current visible screen as plain text (one row per line). `vt100` has
    /// already resolved escape sequences and styling into cells, so this is
    /// exactly what a user sees — color/attribute independent.
    pub fn screen_text(&self) -> String {
        self.screen_snapshot().0
    }

    fn screen_snapshot(&self) -> (String, (u16, u16)) {
        // Recover a poisoned lock rather than defaulting to a blank screen: the
        // parser's data is still valid even if some other thread panicked while
        // holding the lock (the reader thread never panics while holding it —
        // see `spawn_reader` — but recovering here keeps this robust regardless).
        // A silent blank default would otherwise masquerade as "nothing rendered
        // yet" and burn the full poll timeout instead of failing immediately.
        let p = self
            .parser
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (p.screen().contents(), p.screen().size())
    }

    fn framed_screen(&self) -> String {
        let (screen, (rows, cols)) = self.screen_snapshot();
        format!("--- last screen ({cols}x{rows}) ---\n{screen}\n--- end screen ---")
    }

    /// Take the reader thread's recorded panic message, if any, clearing it so
    /// it's only reported once. Checked on every poll in `wait_for_screen`/
    /// `wait_for_exit` so a reader-thread fault surfaces immediately with a
    /// direct diagnostic instead of a 30s timeout over a screen that stopped
    /// updating for an unexplained reason.
    fn take_reader_panic(&self) -> Option<String> {
        self.reader_failure.take_message()
    }

    /// Resize both the real PTY and the emulated screen. The application receives
    /// the normal terminal resize event; assertions continue to inspect exactly
    /// what a user would see at the new geometry.
    pub fn use_detail_size(&mut self) -> Result<(), String> {
        let size = PtySize {
            rows: DETAIL_ROWS,
            cols: DETAIL_COLS,
            pixel_width: 0,
            pixel_height: 0,
        };
        self.master
            .resize(size)
            .map_err(|e| format!("failed to resize pty: {e}"))?;
        // As in `screen_snapshot`, recover rather than fail on a poisoned lock —
        // resizing is still meaningful even if some earlier operation panicked
        // while holding it.
        self.parser
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .screen_mut()
            .set_size(DETAIL_ROWS, DETAIL_COLS);
        Ok(())
    }

    /// Write raw bytes to the terminal (keystrokes/text). `Enter` is `"\r"`.
    pub fn send(&mut self, bytes: &str) -> Result<(), String> {
        self.writer
            .write_all(bytes.as_bytes())
            .and_then(|()| self.writer.flush())
            .map_err(|e| format!("failed to write to pty: {e}"))
    }

    /// Retrieve a terminal failure that landed after a wait's final poll but
    /// before the retry loop decides whether another key is safe to send.
    fn terminal_state_after_wait(&mut self, marker: &str) -> TerminalState {
        let reader_finished = self
            .reader
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished);
        let reader_failure = self.reader_failure.observe();
        if let ReaderFailureObservation::Message(panic_message) = &reader_failure {
            return TerminalState::Failed(format!(
                "pty reader thread panicked while waiting for {marker:?}: {panic_message}\n{}",
                self.framed_screen()
            ));
        }
        if self.finished {
            return TerminalState::Stopped;
        }
        match self.child.try_wait() {
            Ok(Some(status)) => {
                self.finished = true;
                self.record_once(i32::try_from(status.exit_code()).unwrap_or(-1));
                TerminalState::Failed(format!(
                    "process exited ({status:?}) before {marker:?} appeared.\n{}",
                    self.framed_screen()
                ))
            }
            Ok(None) => match reader_failure {
                ReaderFailureObservation::FailedWithoutMessage => TerminalState::Stopped,
                ReaderFailureObservation::Running if reader_finished => TerminalState::Stopped,
                ReaderFailureObservation::Running => TerminalState::Running,
                ReaderFailureObservation::Message(_) => unreachable!("handled above"),
            },
            Err(error) => TerminalState::Failed(format!("failed to poll TUI child: {error}")),
        }
    }

    /// Send `bytes` until the screen shows `marker`, re-sending every
    /// [`KEY_RESEND_INTERVAL`] until the deadline.
    ///
    /// A bare [`send`](Self::send) writes into the pseudo-terminal whether or
    /// not the application is reading yet, so a keystroke typed during startup
    /// can be consumed by whatever holds the terminal at that moment and never
    /// reach the event loop. The key is then simply lost — nothing retries it,
    /// and the scenario fails much later, in an assertion about a view it never
    /// left. Re-sending until the expected view appears makes the step depend on
    /// the application having acted on the key rather than on it having been
    /// ready when the key was written.
    ///
    /// Only safe for idempotent keys (a tab jump, not a toggle): the key is
    /// always sent at least once, and further copies may still be queued in the
    /// terminal when the marker appears, so the application may act on it again
    /// after this returns.
    pub async fn send_until(
        &mut self,
        bytes: &str,
        marker: &str,
        timeout: Duration,
    ) -> Result<(), String> {
        retry_send_until(
            self,
            bytes,
            marker,
            RetryTiming {
                timeout,
                resend_interval: KEY_RESEND_INTERVAL,
            },
            Self::send,
            |session, marker, attempt| Box::pin(session.wait_for_screen(marker, attempt)),
            Self::terminal_state_after_wait,
        )
        .await
    }

    /// Poll the current screen until it contains `marker`, or fail with a
    /// deadline that includes the last screen for diagnosis. Also fails fast if
    /// the child exits before the marker appears.
    pub async fn wait_for_screen(&mut self, marker: &str, timeout: Duration) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        loop {
            if self.screen_text().contains(marker) {
                return Ok(());
            }
            if let Some(panic_message) = self.take_reader_panic() {
                return Err(format!(
                    "pty reader thread panicked while waiting for {marker:?}: {panic_message}\n{}",
                    self.framed_screen()
                ));
            }
            // If the process is gone, let the reader drain the final frame for a
            // short bounded window. A single poll is not enough when a large frame
            // is still buffered behind the process exit notification.
            if let Ok(Some(status)) = self.child.try_wait() {
                self.finished = true;
                self.record_once(i32::try_from(status.exit_code()).unwrap_or(-1));
                let drain_deadline = Instant::now() + DRAIN_TIMEOUT;
                while Instant::now() < drain_deadline {
                    if self.screen_text().contains(marker) {
                        return Ok(());
                    }
                    if let Some(panic_message) = self.take_reader_panic() {
                        return Err(format!(
                            "pty reader thread panicked while draining the final frame for {marker:?}: {panic_message}\n{}",
                            self.framed_screen()
                        ));
                    }
                    if self
                        .reader
                        .as_ref()
                        .is_some_and(std::thread::JoinHandle::is_finished)
                    {
                        break;
                    }
                    tokio::time::sleep(POLL_INTERVAL).await;
                }
                // Final check after the drain window closes: the reader may have
                // committed the last frame between the loop's screen check and the
                // `is_finished`/deadline exit, so re-read before declaring failure.
                if self.screen_text().contains(marker) {
                    return Ok(());
                }
                // A reader panic landing exactly on the drain deadline would
                // otherwise be masked by the generic "process exited" error below
                // (and then swallowed entirely if `Drop` runs during another
                // unwind). Surface it here so the real cause wins.
                if let Some(panic_message) = self.take_reader_panic() {
                    return Err(format!(
                        "pty reader thread panicked while draining the final frame for {marker:?}: {panic_message}\n{}",
                        self.framed_screen()
                    ));
                }
                return Err(format!(
                    "process exited ({status:?}) before {marker:?} appeared.\n{}",
                    self.framed_screen()
                ));
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "timed out after {timeout:?} waiting for {marker:?}.\n{}",
                    self.framed_screen()
                ));
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Drain the PTY for up to [`DRAIN_TIMEOUT`] after the child has exited, so
    /// a process-exit error below reflects the final buffered frame rather
    /// than a stale mid-drain snapshot. Shared by `wait_for_screen_gone` and
    /// `assert_screen_persists`, whose exit handling is otherwise identical;
    /// `wait_for_screen` drains differently (it keeps re-checking for `marker`
    /// while draining, since exit is not automatically a failure on that path)
    /// and so isn't a candidate for this helper.
    async fn drain_after_exit(&mut self) {
        // `reader` is only ever `None` after `Drop` has taken it, which never
        // happens before this call (both callers run from an active session).
        // Guard it anyway: without this, a missing reader would read as
        // "never finished" and burn the full timeout instead of returning
        // immediately.
        if self.reader.is_none() {
            return;
        }
        let drain_deadline = Instant::now() + DRAIN_TIMEOUT;
        while Instant::now() < drain_deadline
            && !self
                .reader
                .as_ref()
                .is_some_and(std::thread::JoinHandle::is_finished)
        {
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Shared exit handling for `wait_for_screen_gone` and
    /// `assert_screen_persists`: both treat the child exiting as a definite
    /// failure (never a legitimate "gone"/"persisted" outcome), so both need
    /// to record the exit, drain the final frame, and surface a reader panic
    /// that landed during that drain — before deciding their own wording for
    /// what the exit means for their specific check. Returns `Ok(None)`
    /// while the child is still running (caller keeps polling), `Ok(Some(status))`
    /// (pre-formatted via `{:?}`) once it has exited and the drain completed
    /// cleanly, or `Err` if the drain itself surfaced a reader panic.
    /// `wait_for_screen` isn't a candidate for this helper: it drains
    /// differently, still polling for `marker` while draining since exit
    /// there is not automatically a failure.
    async fn exited_after_drain(&mut self, marker: &str) -> Result<Option<String>, String> {
        let Ok(Some(status)) = self.child.try_wait() else {
            return Ok(None);
        };
        self.finished = true;
        self.record_once(i32::try_from(status.exit_code()).unwrap_or(-1));
        self.drain_after_exit().await;
        if let Some(panic_message) = self.take_reader_panic() {
            return Err(format!(
                "pty reader thread panicked while draining the final frame for {marker:?}: {panic_message}\n{}",
                self.framed_screen()
            ));
        }
        Ok(Some(format!("{status:?}")))
    }

    /// Poll the current screen until it no longer contains `marker`, or fail
    /// with a deadline that includes the last screen for diagnosis.
    ///
    /// The inverse of [`wait_for_screen`](Self::wait_for_screen): a step that
    /// confirms a backend state change (e.g. via a mock server counter) and
    /// then reads the screen once is racing the TUI's own render cadence,
    /// which redraws on its own schedule independent of that state change.
    /// Polling gives the render loop the full timeout budget to catch up
    /// instead of assuming a fixed delay is always enough.
    ///
    /// Liveness (panic / exit) is checked *before* the absence check, unlike
    /// `wait_for_screen`'s presence-first ordering: on the alternate-screen-exit
    /// escape, `vt100` swaps back to the primary grid, which was never written
    /// to and so reads as blank — not because the parsed screen was cleared,
    /// but with the same practical effect. A dashboard that crashes or quits
    /// mid-wait would otherwise make `marker` vanish for the wrong reason and
    /// be misreported as a successful "gone" result.
    ///
    /// Edge case: because liveness is checked before the absence check on
    /// every iteration, a process that exits in the exact poll tick the
    /// marker would otherwise be confirmed gone is reported as an exit
    /// error rather than `Ok(())`. No current caller relies on marker
    /// disappearance racing process exit; a future one that does should
    /// account for this ordering.
    pub async fn wait_for_screen_gone(
        &mut self,
        marker: &str,
        timeout: Duration,
    ) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(panic_message) = self.take_reader_panic() {
                return Err(format!(
                    "pty reader thread panicked while waiting for {marker:?} to disappear: {panic_message}\n{}",
                    self.framed_screen()
                ));
            }
            if let Some(status) = self.exited_after_drain(marker).await? {
                let while_clause = if self.screen_text().contains(marker) {
                    format!("while {marker:?} was still on screen")
                } else {
                    format!("before {marker:?} was confirmed gone")
                };
                return Err(format!(
                    "process exited ({status}) {while_clause}.\n{}",
                    self.framed_screen()
                ));
            }
            if !self.screen_text().contains(marker) {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "timed out after {timeout:?} waiting for {marker:?} to disappear.\n{}",
                    self.framed_screen()
                ));
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Poll the screen for the full `duration`, failing fast if `marker`
    /// disappears (or the process panics/exits) before the window elapses.
    ///
    /// This is the "survives an event" counterpart to `wait_for_screen`'s
    /// "eventually becomes true": `wait_for_screen` returns as soon as
    /// `marker` is present, which a stale pre-event frame can already
    /// satisfy. Use this when the assertion is that a value is *held*
    /// across an interval, not merely that it appears at some point within
    /// it.
    ///
    /// Liveness is checked before the persistence check each iteration, as
    /// in `wait_for_screen_gone`, but here the outcome is `Err` regardless
    /// of ordering — exiting before the window elapses is never a pass, so
    /// checking liveness first only changes which error message is
    /// produced, not the result.
    ///
    /// Edge case: because liveness is checked before the persistence check,
    /// a process that exits in the exact poll tick the persistence window
    /// would otherwise complete is reported as an exit error rather than
    /// `Ok(())`. No current caller relies on exit racing window
    /// completion; a future one that does should account for this
    /// ordering.
    pub async fn assert_screen_persists(
        &mut self,
        marker: &str,
        duration: Duration,
    ) -> Result<(), String> {
        let deadline = Instant::now() + duration;
        let mut observed_marker = false;
        loop {
            if let Some(panic_message) = self.take_reader_panic() {
                return Err(format!(
                    "pty reader thread panicked while asserting {marker:?} persists: {panic_message}\n{}",
                    self.framed_screen()
                ));
            }
            if let Some(status) = self.exited_after_drain(marker).await? {
                return Err(format!(
                    "process exited ({status}) while asserting {marker:?} persists.\n{}",
                    self.framed_screen()
                ));
            }
            if self.screen_text().contains(marker) {
                observed_marker = true;
            } else if observed_marker {
                return Err(format!(
                    "{marker:?} disappeared before the {duration:?} persistence window elapsed.\n{}",
                    self.framed_screen()
                ));
            } else {
                return Err(format!(
                    "{marker:?} was not present when the persistence check began.\n{}",
                    self.framed_screen()
                ));
            }
            if Instant::now() >= deadline {
                return Ok(());
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Send the quit gesture appropriate to the session and wait for a clean
    /// exit. The dashboard quits with `q`; chat quits with the `/quit` slash
    /// command (a bare `q` would be typed into the focused input instead).
    pub async fn quit_and_wait(&mut self, timeout: Duration) -> Result<(), String> {
        if self.is_chat {
            self.send("/quit\r")?;
        } else {
            self.send("q")?;
        }
        self.wait_for_exit(timeout).await
    }

    /// Poll until the child exits, asserting a successful (zero) exit code.
    pub async fn wait_for_exit(&mut self, timeout: Duration) -> Result<(), String> {
        match self.wait_for_exit_code(timeout).await? {
            0 => Ok(()),
            code => Err(format!(
                "TUI exited unsuccessfully (code {code}).\n{}",
                self.framed_screen()
            )),
        }
    }

    /// Poll until the child exits, returning its raw exit code regardless of
    /// whether it is zero. Used by journeys (e.g. a declined confirmation
    /// prompt) whose success case is a specific *nonzero* code, where
    /// [`wait_for_exit`](Self::wait_for_exit)'s built-in zero-only assertion
    /// would reject the very outcome under test.
    pub async fn wait_for_exit_code(&mut self, timeout: Duration) -> Result<i32, String> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    let code = i32::try_from(status.exit_code()).unwrap_or(-1);
                    self.finished = true;
                    self.record_once(code);
                    return Ok(code);
                }
                Ok(None) => {}
                Err(e) => return Err(format!("failed to poll TUI child: {e}")),
            }
            // A reader panic doesn't affect whether the child itself has exited,
            // but it does mean the screen in any resulting error/diagnostic is
            // stale, so surface it rather than let this poll silently continue.
            if let Some(panic_message) = self.take_reader_panic() {
                return Err(format!(
                    "pty reader thread panicked while waiting for exit: {panic_message}\n{}",
                    self.framed_screen()
                ));
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "timed out after {timeout:?} waiting for the TUI to exit.\n{}",
                    self.framed_screen()
                ));
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Record this invocation once for the command-coverage report (so `rocm
    /// dash` / `rocm chat` count as covered), tied to the scenario for the
    /// pass/fail join. Best-effort and idempotent.
    fn record_once(&mut self, rc: i32) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        let argv: Vec<&str> = self.argv.iter().map(String::as_str).collect();
        crate::record_command(self.scenario.as_deref(), &argv, rc, "");
    }
}

impl Drop for TuiSession {
    fn drop(&mut self) {
        // Kill and reap the child FIRST so the slave closes and the reader thread
        // sees EOF; only then join it, so teardown can never hang on a blocked
        // read. This is the safety net for steps that panicked or returned early
        // without an explicit quit (e.g. the consent-gate scenarios).
        if !self.finished {
            let _ = self.child.kill();
            let rc = self
                .child
                .wait()
                .ok()
                .and_then(|status| i32::try_from(status.exit_code()).ok())
                .unwrap_or(-1);
            self.finished = true;
            self.record_once(rc);
        }
        self.reader_stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.reader.take() {
            // `join` returns `Err` only if the reader thread itself panicked
            // (distinct from `reader_failure`, which we publish *before* the thread
            // exits normally after catching a `p.process` panic — so `join`
            // failing here would mean some other, uncaught panic in the reader).
            // Never re-panic here: if a scenario step already panicked and this
            // `drop` is running during that unwind, turning a teardown detail
            // into a second panic would abort the process and destroy the
            // original failure's message. Log to stderr instead, and only ever
            // panic (to fail an otherwise-green test) when nothing is unwinding.
            if let Err(payload) = handle.join() {
                let message = panic_message(&payload);
                if std::thread::panicking() {
                    eprintln!(
                        "pty reader thread also panicked during teardown (suppressed to preserve the original panic): {message}"
                    );
                } else {
                    panic!("pty reader thread panicked: {message}");
                }
            } else if let Some(message) = self.take_reader_panic() {
                // The reader caught its own panic and exited cleanly (see
                // `spawn_reader`), but no `wait_for_screen`/`wait_for_exit` call
                // ever observed and reported it — surface it now rather than
                // silently dropping the diagnostic. As with the `join` branch
                // above, never turn this into a second panic while another panic
                // is already unwinding (it would abort the process and destroy
                // the original failure's message); log to stderr in that case.
                if std::thread::panicking() {
                    eprintln!(
                        "pty reader thread panicked (suppressed to preserve the original panic): {message}"
                    );
                } else {
                    panic!("pty reader thread panicked: {message}");
                }
            }
        }
    }
}

/// Continuously drain the PTY into the shared parser until EOF or stop. Runs on a
/// dedicated OS thread because PTY reads block; the assertion side only ever
/// inspects the resulting `Screen`, never the reader.
///
/// If `vt100::Parser::process` ever panics, it's caught here (rather than left
/// to unwind the reader thread silently) and published through `reader_failure`
/// before the thread exits, so `wait_for_screen`/`wait_for_exit` can fail fast
/// with the actual cause instead of quietly polling a screen that will never
/// update again.
fn spawn_reader(
    mut reader: Box<dyn Read + Send>,
    parser: Arc<Mutex<vt100::Parser>>,
    stop: Arc<AtomicBool>,
    reader_failure: Arc<ReaderFailure>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let mut p = parser
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        p.process(&buf[..n]);
                    }));
                    drop(p);
                    if let Err(payload) = result {
                        let message = panic_message(&payload);
                        reader_failure.publish(message);
                        break;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                // Any other error (e.g. EIO once the slave closes) means the
                // session is over.
                Err(_) => break,
            }
        }
    })
}
