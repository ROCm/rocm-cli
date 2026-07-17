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
//! notice, EAI-7222) and only had in-process render tests.
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

/// Default wall-clock budget for a single wait. Generous enough for a cold dash
/// start plus the embedded-daemon connect, while still turning a genuine hang
/// into a prompt, diagnosable failure rather than a CI-timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// A running `rocm` TUI attached to a pseudo-terminal.
pub struct TuiSession {
    child: Box<dyn Child + Send + Sync>,
    writer: Box<dyn Write + Send>,
    /// The emulated screen, updated continuously by the reader thread.
    parser: Arc<Mutex<vt100::Parser>>,
    reader_stop: Arc<AtomicBool>,
    reader: Option<JoinHandle<()>>,
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
        let pair = native_pty_system()
            .openpty(PtySize {
                rows: ROWS,
                cols: COLS,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| format!("openpty failed: {e}"))?;

        let mut cmd = CommandBuilder::new(crate::rocm_binary());
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
        for (key, value) in world.isolate_env() {
            cmd.env(key, value);
        }
        // Deterministic terminal type; the PTY ioctl size above is authoritative
        // for crossterm, with COLUMNS/LINES as a belt-and-braces fallback.
        cmd.env("TERM", "xterm-256color");
        cmd.env("COLUMNS", COLS.to_string());
        cmd.env("LINES", ROWS.to_string());

        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| format!("failed to spawn rocm under a pty: {e}"))?;
        // Drop the slave in the parent: only the child needs it. Keeping it open
        // would prevent the reader from seeing EOF when the child exits.
        drop(pair.slave);

        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| format!("failed to clone pty reader: {e}"))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|e| format!("failed to take pty writer: {e}"))?;

        let parser = Arc::new(Mutex::new(vt100::Parser::new(ROWS, COLS, 0)));
        let reader_stop = Arc::new(AtomicBool::new(false));
        let reader = spawn_reader(reader, Arc::clone(&parser), Arc::clone(&reader_stop));

        Ok(Self {
            child,
            writer,
            parser,
            reader_stop,
            reader: Some(reader),
            master: pair.master,
            finished: false,
            recorded: false,
            is_chat: args.first() == Some(&"chat"),
            scenario: world.current_scenario.clone(),
            argv: args.iter().map(|s| (*s).to_string()).collect(),
        })
    }

    /// The current visible screen as plain text (one row per line). `vt100` has
    /// already resolved escape sequences and styling into cells, so this is
    /// exactly what a user sees — color/attribute independent.
    pub fn screen_text(&self) -> String {
        self.parser
            .lock()
            .map(|p| p.screen().contents())
            .unwrap_or_default()
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
        self.parser
            .lock()
            .map_err(|_| "failed to lock vt100 parser".to_string())?
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

    /// Poll the current screen until it contains `marker`, or fail with a
    /// deadline that includes the last screen for diagnosis. Also fails fast if
    /// the child exits before the marker appears.
    pub async fn wait_for_screen(&mut self, marker: &str, timeout: Duration) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        loop {
            if self.screen_text().contains(marker) {
                return Ok(());
            }
            // If the process is gone, give the reader a beat to drain any final
            // bytes, then check once more before declaring failure.
            if let Ok(Some(status)) = self.child.try_wait() {
                self.finished = true;
                tokio::time::sleep(POLL_INTERVAL).await;
                let screen = self.screen_text();
                if screen.contains(marker) {
                    return Ok(());
                }
                return Err(format!(
                    "process exited ({status:?}) before {marker:?} appeared.\n{}",
                    framed_screen(&screen)
                ));
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "timed out after {timeout:?} waiting for {marker:?}.\n{}",
                    framed_screen(&self.screen_text())
                ));
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
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    self.finished = true;
                    self.record_once(i32::try_from(status.exit_code()).unwrap_or(-1));
                    return if status.success() {
                        Ok(())
                    } else {
                        Err(format!(
                            "TUI exited unsuccessfully ({status:?}).\n{}",
                            framed_screen(&self.screen_text())
                        ))
                    };
                }
                Ok(None) => {}
                Err(e) => return Err(format!("failed to poll TUI child: {e}")),
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "timed out after {timeout:?} waiting for the TUI to exit.\n{}",
                    framed_screen(&self.screen_text())
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
            let _ = self.child.wait();
            self.finished = true;
        }
        self.reader_stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.reader.take() {
            let _ = handle.join();
        }
    }
}

/// Continuously drain the PTY into the shared parser until EOF or stop. Runs on a
/// dedicated OS thread because PTY reads block; the assertion side only ever
/// inspects the resulting `Screen`, never the reader.
fn spawn_reader(
    mut reader: Box<dyn Read + Send>,
    parser: Arc<Mutex<vt100::Parser>>,
    stop: Arc<AtomicBool>,
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
                    if let Ok(mut p) = parser.lock() {
                        p.process(&buf[..n]);
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

/// Wrap a screen dump in delimiters so failure messages are easy to read.
fn framed_screen(screen: &str) -> String {
    format!("--- last screen ({COLS}x{ROWS}) ---\n{screen}\n--- end screen ---")
}
