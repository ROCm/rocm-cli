// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Async job-bridge runtime (Phase 3 Wave 0).
//!
//! Interprets [`SideEffect::SpawnJob`]: launches the child process, streams its
//! stdout+stderr back as [`StateEvent::JobLine`], finishes with
//! [`StateEvent::JobDone`] / [`StateEvent::JobErr`], and kills the child when
//! the shared `cancel` flag flips. This is the single async primitive every
//! operational screen reuses instead of the legacy rocm-cli
//! `std::thread::spawn` + mpsc + `try_recv` pattern (56 sites).
//!
//! The reducer (`rocm-dash-core`) stays pure and `tokio`-free; all process I/O
//! lives here, on the surviving side of the boundary.
//!
//! Events are posted on an [`UnboundedSender`] (the same pattern as
//! `client.rs` / `replay.rs`). Unbounded is deliberate: the reducer is pure and
//! synchronous, so it cannot exert backpressure on the async child; a bounded
//! channel would risk deadlock or dropped output on a burst. The output is
//! bounded instead in the reducer, by `JOB_OUTPUT_RING_CAP`.

use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rocm_dash_core::state::{SideEffect, StateEvent};
use tokio::io::{AsyncBufReadExt, BufReader, Lines};
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;

/// How often the runtime wakes to re-check the cancel flag while idle.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// Drive a batch of reducer side effects, spawning a job task for each [`SideEffect::SpawnJob`].
///
/// Non-job effects are ignored here — the daemon owns
/// broadcast/persist; in the TUI the job-bridge only cares about `SpawnJob`.
pub fn run_effects(effects: Vec<SideEffect>, tx: &UnboundedSender<StateEvent>) {
    for fx in effects {
        if let SideEffect::SpawnJob {
            id,
            cmd,
            args,
            cancel,
        } = fx
        {
            spawn_job(id, cmd, args, cancel, tx.clone());
        }
    }
}

/// Spawn one background job. Returns immediately; the spawned task posts
/// [`StateEvent`]s back through `tx` until the job is terminal.
pub fn spawn_job(
    id: String,
    cmd: String,
    args: Vec<String>,
    cancel: Arc<AtomicBool>,
    tx: UnboundedSender<StateEvent>,
) {
    tokio::spawn(async move { run_job(id, cmd, args, cancel, tx).await });
}

async fn run_job(
    id: String,
    cmd: String,
    args: Vec<String>,
    cancel: Arc<AtomicBool>,
    tx: UnboundedSender<StateEvent>,
) {
    let mut child = match Command::new(&cmd)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(StateEvent::JobErr {
                id,
                message: format!("spawn failed: {e}"),
            });
            return;
        }
    };

    // `take()` moves the pipe handles out of `child` so only `child.wait()`
    // borrows it later — the readers own their streams.
    let mut out = child.stdout.take().map(|s| BufReader::new(s).lines());
    let mut err = child.stderr.take().map(|s| BufReader::new(s).lines());

    loop {
        if cancel.load(Ordering::SeqCst) {
            // Reducer already marked the job `Cancelled`; tear the child down.
            let _ = child.start_kill();
            let _ = child.wait().await;
            return;
        }

        let out_open = out.is_some();
        let err_open = err.is_some();

        // `biased`: output and completion arms are checked before the idle
        // poll, so streaming throughput is never sacrificed to the wake-up
        // timer. The poll is last — it only matters when the pipes are idle.
        tokio::select! {
            biased;

            line = next_line(&mut out), if out_open => {
                match line {
                    Some(l) => emit(&tx, &id, l),
                    None => out = None, // EOF on stdout
                }
            }

            line = next_line(&mut err), if err_open => {
                match line {
                    Some(l) => emit(&tx, &id, l),
                    None => err = None, // EOF on stderr
                }
            }

            // Only reap once both pipes have closed, so all output is drained
            // before we report completion.
            status = child.wait(), if !out_open && !err_open => {
                let code = match status {
                    Ok(s) => s.code().unwrap_or(-1),
                    Err(e) => {
                        let _ = tx.send(StateEvent::JobErr {
                            id,
                            message: format!("wait failed: {e}"),
                        });
                        return;
                    }
                };
                let _ = tx.send(StateEvent::JobDone { id, code });
                return;
            }

            // Idle wake-up so cancellation is observed within CANCEL_POLL even
            // when the child produces no output.
            () = tokio::time::sleep(CANCEL_POLL) => {}
        }
    }
}

/// Read the next line from an optional reader. `Some(line)` on data, `None` on
/// EOF or read error (treated as stream close).
async fn next_line<R>(reader: &mut Option<Lines<BufReader<R>>>) -> Option<String>
where
    R: tokio::io::AsyncRead + Unpin,
{
    match reader {
        Some(lines) => lines.next_line().await.ok().flatten(),
        None => std::future::pending().await,
    }
}

fn emit(tx: &UnboundedSender<StateEvent>, id: &str, line: String) {
    let _ = tx.send(StateEvent::JobLine {
        id: id.to_string(),
        line,
    });
}
