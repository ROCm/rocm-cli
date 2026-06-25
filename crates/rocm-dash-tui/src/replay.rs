// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Offline replay source.
//!
//! Reads a daemon-produced session file (NDJSON of `PersistedEntry`) into
//! memory upfront and emits the recorded Events through the same
//! `ClientMsg` mpsc channel used by the live client task — so the rest of
//! the TUI doesn't know or care whether it's connected to a live daemon or
//! replaying from disk.
//!
//! Pacing: replay sleeps between events by the real-time delta between
//! their `ts_us` values (scaled by `speed`), capped at `MAX_GAP_MS` so a
//! long pause in the recording doesn't stall the TUI for minutes.
//!
//! Scrubbing: holding the entries in a `Vec` lets us seek. Forward jumps
//! burst-emit any events between the current cursor and the target
//! timestamp; backward jumps send `ClientMsg::ReplaySeek` (so the TUI can
//! wipe its derived state) and then burst-emit from the start of the
//! recording up to the target. The header gets `ReplayPosition` updates so
//! it can display elapsed/total.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use rocm_dash_core::persist::PersistedEntry;
use rocm_dash_core::protocol::PROTOCOL_VERSION;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{info, warn};

use crate::client::ClientMsg;

/// Cap on inter-event sleep — keeps replay responsive even if the recording
/// has long quiet stretches.
const MAX_GAP_MS: u64 = 2_000;

/// Scrubber commands from the TUI to the replay task.
#[derive(Debug, Clone)]
pub enum ReplayControl {
    Pause,
    Resume,
    SetSpeed(f64),
    /// Move the playhead by `delta_s` seconds (negative = rewind).
    Jump {
        delta_s: i64,
    },
}

#[derive(Debug, Clone)]
pub struct ReplayController {
    tx: UnboundedSender<ReplayControl>,
}

impl ReplayController {
    pub fn pause(&self) {
        let _ = self.tx.send(ReplayControl::Pause);
    }
    pub fn resume(&self) {
        let _ = self.tx.send(ReplayControl::Resume);
    }
    pub fn set_speed(&self, speed: f64) {
        let _ = self
            .tx
            .send(ReplayControl::SetSpeed(speed.clamp(0.25, 8.0)));
    }
    pub fn jump(&self, delta_s: i64) {
        let _ = self.tx.send(ReplayControl::Jump { delta_s });
    }
}

pub const SPEED_STEPS: &[f64] = &[0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

pub fn next_speed(current: f64) -> f64 {
    for &s in SPEED_STEPS {
        if s > current + f64::EPSILON {
            return s;
        }
    }
    *SPEED_STEPS.last().unwrap()
}

pub fn prev_speed(current: f64) -> f64 {
    let mut best = SPEED_STEPS[0];
    for &s in SPEED_STEPS {
        if s + f64::EPSILON < current {
            best = s;
        } else {
            break;
        }
    }
    best
}

pub fn spawn(path: PathBuf, tx: UnboundedSender<ClientMsg>) -> ReplayController {
    let (ctl_tx, ctl_rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let _ = tx.send(ClientMsg::Connecting);
        match run(&path, &tx, ctl_rx).await {
            Ok(()) => {
                info!(path = %path.display(), "replay complete");
                let _ = tx.send(ClientMsg::Disconnected {
                    reason: "end of recording".into(),
                });
            }
            Err(e) => {
                warn!(path = %path.display(), error = %e, "replay failed");
                let _ = tx.send(ClientMsg::Disconnected {
                    reason: e.to_string(),
                });
            }
        }
    });
    ReplayController { tx: ctl_tx }
}

fn read_entries(path: &Path) -> anyhow::Result<Vec<PersistedEntry>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("opening replay file {}", path.display()))?;
    let mut out = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<PersistedEntry>(line) {
            Ok(e) => out.push(e),
            Err(e) => warn!(line = lineno + 1, error = %e, "skipping malformed replay line"),
        }
    }
    Ok(out)
}

async fn run(
    path: &Path,
    tx: &UnboundedSender<ClientMsg>,
    mut ctl: UnboundedReceiver<ReplayControl>,
) -> anyhow::Result<()> {
    let entries = read_entries(path)?;

    let _ = tx.send(ClientMsg::Connected {
        host: format!(
            "replay:{}",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("file")
        ),
        daemon_version: format!("replay/v{PROTOCOL_VERSION}"),
    });

    if entries.is_empty() {
        return Ok(());
    }
    let first_ts_us = entries.first().unwrap().ts_us;
    let last_ts_us = entries.last().unwrap().ts_us;
    let total_s = last_ts_us.saturating_sub(first_ts_us) / 1_000_000;

    let mut paused = false;
    let mut speed: f64 = 1.0;
    let mut last_emit = tokio::time::Instant::now();

    // Emit the first event immediately so the UI lights up.
    emit_at(tx, &entries[0], first_ts_us, total_s);
    let mut cursor: usize = 1;

    while cursor < entries.len() {
        let entry = &entries[cursor];
        let prev_ts = entries[cursor - 1].ts_us;
        let raw_gap = Duration::from_micros(entry.ts_us.saturating_sub(prev_ts))
            .min(Duration::from_millis(MAX_GAP_MS));
        let mut target = last_emit + scale_gap(raw_gap, speed);

        // Sleep until `target`, but respond to control messages along the way.
        loop {
            if paused {
                match ctl.recv().await {
                    Some(c) => {
                        if handle_control(
                            c,
                            &mut paused,
                            &mut speed,
                            &mut cursor,
                            &entries,
                            tx,
                            first_ts_us,
                            total_s,
                        ) {
                            // Cursor moved (Jump) — break so the outer loop resets pacing.
                            break;
                        }
                        if !paused {
                            target = tokio::time::Instant::now();
                        }
                        continue;
                    }
                    None => return Ok(()),
                }
            }
            let now = tokio::time::Instant::now();
            if now >= target {
                break;
            }
            tokio::select! {
                () = tokio::time::sleep_until(target) => break,
                maybe = ctl.recv() => match maybe {
                    Some(c) => {
                        let jumped = handle_control(c, &mut paused, &mut speed, &mut cursor,
                            &entries, tx, first_ts_us, total_s);
                        if jumped {
                            break;
                        }
                        if !paused {
                            target = tokio::time::Instant::now() + scale_gap(raw_gap, speed);
                        }
                    }
                    None => return Ok(()),
                },
            }
        }

        // Re-check cursor after possible jump (handle_control may have moved it).
        if cursor >= entries.len() {
            break;
        }
        let entry = &entries[cursor];
        emit_at(tx, entry, first_ts_us, total_s);
        last_emit = tokio::time::Instant::now();
        cursor += 1;
    }
    Ok(())
}

/// Send the event + position update for a single entry. Returns `false` if
/// the receiver is gone (caller should bail).
fn emit_at(
    tx: &UnboundedSender<ClientMsg>,
    entry: &PersistedEntry,
    first_ts_us: u64,
    total_s: u64,
) -> bool {
    let elapsed_s = entry.ts_us.saturating_sub(first_ts_us) / 1_000_000;
    if tx
        .send(ClientMsg::Event(Box::new(entry.event.clone())))
        .is_err()
    {
        return false;
    }
    let _ = tx.send(ClientMsg::ReplayPosition { elapsed_s, total_s });
    true
}

/// Apply a control message. Returns `true` when the cursor was moved by a
/// jump (so the outer loop should reset its pacing baseline).
#[allow(clippy::too_many_arguments)]
fn handle_control(
    c: ReplayControl,
    paused: &mut bool,
    speed: &mut f64,
    cursor: &mut usize,
    entries: &[PersistedEntry],
    tx: &UnboundedSender<ClientMsg>,
    first_ts_us: u64,
    total_s: u64,
) -> bool {
    match c {
        ReplayControl::Pause => {
            *paused = true;
            false
        }
        ReplayControl::Resume => {
            *paused = false;
            false
        }
        ReplayControl::SetSpeed(s) => {
            *speed = s.clamp(0.25, 8.0);
            false
        }
        ReplayControl::Jump { delta_s } => {
            do_jump(delta_s, cursor, entries, tx, first_ts_us, total_s);
            true
        }
    }
}

/// Move the playhead by `delta_s` seconds and burst-emit any events the
/// jump crosses. Backward jumps additionally send `ReplaySeek` so the TUI
/// can wipe its derived state before the burst arrives.
fn do_jump(
    delta_s: i64,
    cursor: &mut usize,
    entries: &[PersistedEntry],
    tx: &UnboundedSender<ClientMsg>,
    first_ts_us: u64,
    total_s: u64,
) {
    // Anchor: where are we now in recording time?
    let now_idx = (*cursor).min(entries.len().saturating_sub(1));
    let now_ts = entries[now_idx].ts_us;
    let last_ts = entries.last().unwrap().ts_us;

    let target_ts_us = if delta_s >= 0 {
        now_ts.saturating_add((delta_s as u64).saturating_mul(1_000_000))
    } else {
        let back = (-delta_s) as u64 * 1_000_000;
        now_ts.saturating_sub(back).max(first_ts_us)
    }
    .min(last_ts);

    if target_ts_us >= now_ts {
        // Forward: emit anything we'd otherwise sleep past, then settle on
        // the first entry with ts >= target.
        while *cursor < entries.len() && entries[*cursor].ts_us < target_ts_us {
            if !emit_at(tx, &entries[*cursor], first_ts_us, total_s) {
                return;
            }
            *cursor += 1;
        }
    } else {
        // Backward: reset the TUI, then burst-emit from the start through
        // target. Land cursor on the first entry strictly past target so
        // the outer loop's next iteration paces normally.
        let _ = tx.send(ClientMsg::ReplaySeek);
        *cursor = 0;
        while *cursor < entries.len() && entries[*cursor].ts_us <= target_ts_us {
            if !emit_at(tx, &entries[*cursor], first_ts_us, total_s) {
                return;
            }
            *cursor += 1;
        }
    }
}

fn scale_gap(raw: Duration, speed: f64) -> Duration {
    if speed <= 0.0 {
        return raw;
    }
    let scaled = raw.as_secs_f64() / speed;
    Duration::from_secs_f64(scaled.max(0.001))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_dash_core::metrics::Snapshot;
    use rocm_dash_core::protocol::Event;
    use std::io::Write;
    use tokio::sync::mpsc::unbounded_channel;

    fn tmp_file(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "rocm-dash-replay-test-{}-{}-{label}.ndjson",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    fn write_lines(path: &Path, entries: &[PersistedEntry]) {
        let mut f = std::fs::File::create(path).unwrap();
        for e in entries {
            let line = serde_json::to_string(e).unwrap();
            f.write_all(line.as_bytes()).unwrap();
            f.write_all(b"\n").unwrap();
        }
        f.flush().unwrap();
    }

    #[tokio::test]
    async fn emits_events_and_then_disconnect() {
        let path = tmp_file("flow");
        let entries = vec![
            PersistedEntry {
                ts_us: 0,
                event: Event::Snapshot(Snapshot::default()),
            },
            PersistedEntry {
                ts_us: 1_000,
                event: Event::Bye,
            },
        ];
        write_lines(&path, &entries);

        let (tx, mut rx) = unbounded_channel::<ClientMsg>();
        let _ctl = spawn(path.clone(), tx);

        assert!(matches!(rx.recv().await.unwrap(), ClientMsg::Connecting));
        assert!(matches!(
            rx.recv().await.unwrap(),
            ClientMsg::Connected { .. }
        ));
        // Each event is followed by a position update.
        match rx.recv().await.unwrap() {
            ClientMsg::Event(ev) => assert!(matches!(*ev, Event::Snapshot(_))),
            other => panic!("expected event, got {other:?}"),
        }
        assert!(matches!(
            rx.recv().await.unwrap(),
            ClientMsg::ReplayPosition { .. }
        ));
        match rx.recv().await.unwrap() {
            ClientMsg::Event(ev) => assert!(matches!(*ev, Event::Bye)),
            other => panic!("expected event, got {other:?}"),
        }
        assert!(matches!(
            rx.recv().await.unwrap(),
            ClientMsg::ReplayPosition { .. }
        ));
        assert!(matches!(
            rx.recv().await.unwrap(),
            ClientMsg::Disconnected { .. }
        ));

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn malformed_lines_are_skipped() {
        let path = tmp_file("bad");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"this is not json\n").unwrap();
        let good = PersistedEntry {
            ts_us: 0,
            event: Event::Bye,
        };
        let line = serde_json::to_string(&good).unwrap();
        f.write_all(line.as_bytes()).unwrap();
        f.write_all(b"\n").unwrap();
        drop(f);

        let (tx, mut rx) = unbounded_channel::<ClientMsg>();
        let _ctl = spawn(path.clone(), tx);

        let _ = rx.recv().await.unwrap();
        let _ = rx.recv().await.unwrap();
        match rx.recv().await.unwrap() {
            ClientMsg::Event(ev) => assert!(matches!(*ev, Event::Bye)),
            other => panic!("expected event, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn missing_file_yields_disconnect_with_error() {
        let path = tmp_file("does-not-exist");
        let (tx, mut rx) = unbounded_channel::<ClientMsg>();
        let _ctl = spawn(path, tx);
        let _ = rx.recv().await.unwrap();
        match rx.recv().await.unwrap() {
            ClientMsg::Disconnected { reason } => assert!(reason.contains("opening replay file")),
            other => panic!("expected Disconnected, got {other:?}"),
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn next_speed_walks_steps_and_caps_at_max() {
        assert_eq!(next_speed(0.25), 0.5);
        assert_eq!(next_speed(1.0), 2.0);
        assert_eq!(next_speed(8.0), 8.0);
        assert_eq!(next_speed(0.3), 0.5);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn prev_speed_walks_steps_and_floors_at_min() {
        assert_eq!(prev_speed(8.0), 4.0);
        assert_eq!(prev_speed(0.25), 0.25);
        assert_eq!(prev_speed(3.0), 2.0);
    }

    #[test]
    fn scale_gap_speeds_up_inversely() {
        let raw = Duration::from_millis(200);
        assert_eq!(scale_gap(raw, 1.0), Duration::from_millis(200));
        assert_eq!(scale_gap(raw, 2.0), Duration::from_millis(100));
        assert_eq!(scale_gap(raw, 0.5), Duration::from_millis(400));
        assert_eq!(scale_gap(raw, 0.0), raw);
    }

    #[tokio::test]
    async fn controller_pauses_and_resumes_emission() {
        let path = tmp_file("scrub");
        let entries = vec![
            PersistedEntry {
                ts_us: 0,
                event: Event::Snapshot(Snapshot::default()),
            },
            PersistedEntry {
                ts_us: 200_000,
                event: Event::Bye,
            },
        ];
        write_lines(&path, &entries);

        let (tx, mut rx) = unbounded_channel::<ClientMsg>();
        let ctl = spawn(path.clone(), tx);

        let _ = rx.recv().await; // Connecting
        let _ = rx.recv().await; // Connected
        let first = rx.recv().await.unwrap();
        assert!(matches!(first, ClientMsg::Event(ref ev) if matches!(**ev, Event::Snapshot(_))));
        let _ = rx.recv().await; // ReplayPosition

        ctl.pause();
        let res = tokio::time::timeout(Duration::from_millis(300), rx.recv()).await;
        assert!(res.is_err(), "expected timeout while paused, got {res:?}");

        ctl.resume();
        // Eventually a Bye arrives. Drain until we see it.
        let deadline = tokio::time::Instant::now() + Duration::from_millis(800);
        loop {
            let msg = tokio::time::timeout_at(deadline, rx.recv())
                .await
                .expect("timed out waiting for resume")
                .unwrap();
            if let ClientMsg::Event(ev) = &msg
                && matches!(**ev, Event::Bye)
            {
                break;
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn jump_backward_resets_and_replays() {
        // Three snapshots 1 second apart, then a Bye.
        let path = tmp_file("jump-back");
        let entries = vec![
            PersistedEntry {
                ts_us: 0,
                event: Event::Snapshot(Snapshot::default()),
            },
            PersistedEntry {
                ts_us: 1_000_000,
                event: Event::Snapshot(Snapshot::default()),
            },
            PersistedEntry {
                ts_us: 2_000_000,
                event: Event::Snapshot(Snapshot::default()),
            },
            PersistedEntry {
                ts_us: 3_000_000,
                event: Event::Bye,
            },
        ];
        write_lines(&path, &entries);

        let (tx, mut rx) = unbounded_channel::<ClientMsg>();
        let ctl = spawn(path.clone(), tx);

        // Drain Connecting + Connected + first event/position.
        let _ = rx.recv().await;
        let _ = rx.recv().await;
        let _ = rx.recv().await; // first Event
        let _ = rx.recv().await; // first ReplayPosition

        // Pause so we don't race the pacer, then jump back.
        ctl.pause();
        tokio::time::sleep(Duration::from_millis(50)).await;
        ctl.jump(-10); // way before 0 → clamps to first_ts_us
        // Expect a ReplaySeek somewhere in the next few messages.
        let mut saw_seek = false;
        for _ in 0..6 {
            match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
                Ok(Some(ClientMsg::ReplaySeek)) => {
                    saw_seek = true;
                    break;
                }
                Ok(Some(_)) => {}
                _ => break,
            }
        }
        assert!(saw_seek, "expected ReplaySeek after backward jump");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn jump_forward_skips_events_without_seek() {
        let path = tmp_file("jump-fwd");
        let entries = vec![
            PersistedEntry {
                ts_us: 0,
                event: Event::Snapshot(Snapshot::default()),
            },
            PersistedEntry {
                ts_us: 1_000_000,
                event: Event::Snapshot(Snapshot::default()),
            },
            PersistedEntry {
                ts_us: 2_000_000,
                event: Event::Bye,
            },
        ];
        write_lines(&path, &entries);

        let (tx, mut rx) = unbounded_channel::<ClientMsg>();
        let ctl = spawn(path.clone(), tx);

        let _ = rx.recv().await; // Connecting
        let _ = rx.recv().await; // Connected
        let _ = rx.recv().await; // first Event
        let _ = rx.recv().await; // first ReplayPosition

        ctl.pause();
        tokio::time::sleep(Duration::from_millis(50)).await;
        ctl.jump(10); // beyond end → emits both remaining

        // Drain remaining messages and confirm no ReplaySeek appeared.
        let mut saw_seek = false;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(400);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(ClientMsg::ReplaySeek)) => {
                    saw_seek = true;
                    break;
                }
                Ok(Some(_)) => {}
                _ => break,
            }
        }
        assert!(!saw_seek, "forward jump must not emit ReplaySeek");

        let _ = std::fs::remove_file(&path);
    }
}
