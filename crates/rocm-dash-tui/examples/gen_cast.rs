// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Render the TUI against a synthetic session NDJSON to an asciicast v2
//! recording. The marketing page replays it as a real animated terminal
//! demo — no live terminal, no external capture tools, no drift.
//!
//! Usage:
//!   cargo run --release --example gen_cast -p rocm-dash-tui -- \
//!       --input /tmp/mi355x-demo.ndjson \
//!       --output ../marketing/asciinema/rocm-dash.cast
//!
//! Each frame is the actual `ui::draw` framebuffer for a point in a
//! scripted storyboard (tab + modal walk), converted to a truecolor ANSI
//! string. Frames overwrite in place (cursor home), so the cast plays back
//! as a smooth animation rather than a scrolling log.

use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::{Color, Modifier};
use rocm_dash_core::persist::PersistedEntry;
use rocm_dash_core::protocol::Event;
use rocm_dash_tui::app::{ActiveTab, AppState, ConnState, Modal};
use rocm_dash_tui::ui;

/// Per-frame cast time, in seconds. ~0.45s gives a calm, readable cadence.
const FRAME_DELAY_S: f64 = 0.45;

/// Default-dark fallback colours for `Color::Reset`/non-RGB variants, chosen
/// to match the SVG hero so the recording is visually consistent.
const DEFAULT_FG: (u8, u8, u8) = (235, 235, 236);
const DEFAULT_BG: (u8, u8, u8) = (13, 15, 18);

#[derive(Parser)]
#[command(
    name = "gen_cast",
    about = "Render TUI storyboard to an asciicast v2 recording"
)]
struct Args {
    /// Synthetic session file (produced by `make demo`).
    #[arg(long, default_value = "/tmp/mi355x-demo.ndjson")]
    input: PathBuf,

    /// Output cast file. Parent dir created if absent. Default targets the
    /// repo-root `marketing/asciinema/` directory when run from `app/`.
    #[arg(long, default_value = "../marketing/asciinema/rocm-dash.cast")]
    output: PathBuf,

    /// Terminal grid (cols × rows) the cast is rendered at.
    #[arg(long, default_value_t = 160)]
    cols: u16,
    #[arg(long, default_value_t = 44)]
    rows: u16,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    let entries = read_entries(&args.input)?;
    eprintln!(
        "loaded {} entries from {}",
        entries.len(),
        args.input.display()
    );

    // Snapshot events drive the animated widgets (sparklines, gauges). We
    // index them so the storyboard can advance the timeline a few snapshots
    // at a time between frames.
    let total_snaps = entries
        .iter()
        .filter(|e| matches!(e.event, Event::Snapshot(_)))
        .count();
    eprintln!("  {total_snaps} snapshots in timeline");

    let mut state = AppState::new("demo-mi355x-01".into(), "default-dark".into());
    state.conn = ConnState::Connected {
        host: "demo-mi355x-01".into(),
        version: "0.1.0".into(),
    };

    // Replay cursor: `cursor` indexes into `entries`, `snaps_applied` counts
    // snapshot events consumed so the storyboard can target a timeline depth.
    let mut replay = Replay {
        entries: &entries,
        cursor: 0,
        snaps_applied: 0,
        total_snaps,
    };

    // Storyboard: (frames, snapshot-advance-per-frame, tab, modal). Counts are
    // clamped against the file length so a short demo still produces a valid
    // (if shorter) cast.
    let overview_frames = 30usize;
    let hardware_frames = 10usize;
    let instances_frames = 10usize;
    let bench_frames = 15usize;
    let help_frames = 3usize;

    // Spread the snapshot timeline across the data-driven tabs so each frame
    // shows movement. Bench rows accumulate from events, so give Bench the
    // densest advance.
    let total_advancing_frames =
        overview_frames + hardware_frames + instances_frames + bench_frames;
    let per_frame_snaps = (total_snaps / total_advancing_frames.max(1)).max(1);

    let mut frames: Vec<String> = Vec::new();

    // --- Overview tab -------------------------------------------------------
    state.active_tab = ActiveTab::Overview;
    state.modal = Modal::None;
    for _ in 0..overview_frames {
        replay.advance_by(&mut state, per_frame_snaps);
        frames.push(capture(&mut state, args.cols, args.rows)?);
    }

    // --- Hardware tab -------------------------------------------------------
    state.active_tab = ActiveTab::Hardware;
    for _ in 0..hardware_frames {
        replay.advance_by(&mut state, per_frame_snaps);
        frames.push(capture(&mut state, args.cols, args.rows)?);
    }

    // --- Instances tab ------------------------------------------------------
    state.active_tab = ActiveTab::Instances;
    for _ in 0..instances_frames {
        replay.advance_by(&mut state, per_frame_snaps);
        frames.push(capture(&mut state, args.cols, args.rows)?);
    }

    // --- Bench tab ----------------------------------------------------------
    // Drain whatever remains of the timeline here so the Pass^N/Pass@N rollup
    // (incl. the mixed `S-stress-mixtral` group) fills in over the frames.
    state.active_tab = ActiveTab::Bench;
    let remaining = total_snaps.saturating_sub(replay.snaps_applied);
    let bench_advance = (remaining / bench_frames.max(1)).max(1);
    for _ in 0..bench_frames {
        replay.advance_by(&mut state, bench_advance);
        frames.push(capture(&mut state, args.cols, args.rows)?);
    }

    // --- Help modal ---------------------------------------------------------
    state.modal = Modal::Help;
    for _ in 0..help_frames {
        frames.push(capture(&mut state, args.cols, args.rows)?);
    }
    state.modal = Modal::None;
    frames.push(capture(&mut state, args.cols, args.rows)?);

    write_cast(&args, &frames)?;

    let duration = frames.len() as f64 * FRAME_DELAY_S;
    eprintln!(
        "done — {} frames, {:.1}s cast → {}",
        frames.len(),
        duration,
        args.output.display()
    );
    Ok(())
}

/// Walks the persisted entry list, applying every event to `state` while
/// counting consumed `Snapshot` events. `advance_by` moves the cursor forward
/// until `n` more snapshots have been applied (or the timeline is exhausted),
/// so each storyboard frame shows fresh, advanced data.
struct Replay<'a> {
    entries: &'a [PersistedEntry],
    cursor: usize,
    snaps_applied: usize,
    total_snaps: usize,
}

impl Replay<'_> {
    fn advance_by(&mut self, state: &mut AppState, n: usize) {
        let target = (self.snaps_applied + n).min(self.total_snaps);
        // Apply events until we've consumed `n` more snapshots.
        while self.snaps_applied < target && self.cursor < self.entries.len() {
            let event = self.entries[self.cursor].event.clone();
            let is_snap = matches!(event, Event::Snapshot(_));
            state.apply_event(event);
            if is_snap {
                self.snaps_applied += 1;
            }
            self.cursor += 1;
        }
        // Drain trailing non-snapshot events (e.g. bench rows emitted right
        // after the last counted snapshot) so they aren't stranded.
        while self.cursor < self.entries.len()
            && !matches!(self.entries[self.cursor].event, Event::Snapshot(_))
        {
            let event = self.entries[self.cursor].event.clone();
            state.apply_event(event);
            self.cursor += 1;
        }
    }
}

/// Render the current `AppState` to a `TestBackend` buffer and convert it to a
/// truecolor ANSI frame. Mirrors `gen_screenshots::render_to_svg`.
fn capture(state: &mut AppState, cols: u16, rows: u16) -> anyhow::Result<String> {
    let backend = TestBackend::new(cols, rows);
    let mut terminal = Terminal::new(backend)?;
    terminal.draw(|f| ui::draw(f, state))?;
    let buf = terminal.backend().buffer().clone();
    Ok(buffer_to_ansi(&buf))
}

fn read_entries(path: &std::path::Path) -> anyhow::Result<Vec<PersistedEntry>> {
    let raw = fs::read_to_string(path)?;
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<PersistedEntry>(l).map_err(Into::into))
        .collect()
}

/// Write the asciicast v2 file: a JSON header line followed by one
/// `[t, "o", payload]` event line per frame. `serde_json` handles all string
/// escaping (quotes, backslashes, ESC, newlines).
fn write_cast(args: &Args, frames: &[String]) -> anyhow::Result<()> {
    let file = fs::File::create(&args.output)?;
    let mut w = BufWriter::new(file);

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    let header = serde_json::json!({
        "version": 2,
        "width": args.cols,
        "height": args.rows,
        "timestamp": timestamp,
        "title": "rocm.ai — MI355X demo",
    });
    writeln!(w, "{}", serde_json::to_string(&header)?)?;

    let mut t = 0.0f64;
    for frame in frames {
        t += FRAME_DELAY_S;
        let event = serde_json::Value::Array(vec![
            serde_json::json!(t),
            serde_json::Value::String("o".into()),
            serde_json::Value::String(frame.clone()),
        ]);
        writeln!(w, "{}", serde_json::to_string(&event)?)?;
    }
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Buffer → ANSI
// ---------------------------------------------------------------------------

/// Convert a rendered buffer to a truecolor ANSI frame. Frames are written to
/// overwrite in place: hide cursor + home, then per-row SGR runs. The first
/// frame additionally clears the screen (callers prepend the clear once; here
/// we always emit home so any frame is self-positioning).
fn buffer_to_ansi(buf: &Buffer) -> String {
    let cols = buf.area.width;
    let rows = buf.area.height;
    let mut out = String::with_capacity(rows as usize * cols as usize * 8);

    // Hide cursor, clear, home. Clearing every frame is cheap and guarantees
    // no stale cells if a later frame is shorter than an earlier one.
    out.push_str("\u{1b}[?25l\u{1b}[2J\u{1b}[H");

    for y in 0..rows {
        // Reset SGR tracking at the start of each row. Position the cursor
        // absolutely (row y+1, col 1) so frames overwrite in place — emitting
        // newlines instead would scroll the emulator and stack frames.
        let mut cur_fg: Option<(u8, u8, u8)> = None;
        let mut cur_bg: Option<(u8, u8, u8)> = None;
        let mut cur_bold = false;
        let _ = write!(out, "\u{1b}[{};1H\u{1b}[0m", y + 1);

        for x in 0..cols {
            let cell = buf.cell((x, y)).unwrap();
            let fg = resolve_color(cell.style().fg, DEFAULT_FG);
            let bg = resolve_color(cell.style().bg, DEFAULT_BG);
            let bold = cell.style().add_modifier.contains(Modifier::BOLD);

            let attrs_changed = Some(fg) != cur_fg || Some(bg) != cur_bg || bold != cur_bold;
            if attrs_changed {
                // If bold is turning off, a hard reset is the only way to drop
                // the BOLD attribute; re-emit colours afterwards.
                if cur_bold && !bold {
                    out.push_str("\u{1b}[0m");
                    cur_fg = None;
                    cur_bg = None;
                }
                if bold && !cur_bold {
                    out.push_str("\u{1b}[1m");
                }
                if Some(fg) != cur_fg {
                    let (r, g, b) = fg;
                    let _ = write!(out, "\u{1b}[38;2;{r};{g};{b}m");
                }
                if Some(bg) != cur_bg {
                    let (r, g, b) = bg;
                    let _ = write!(out, "\u{1b}[48;2;{r};{g};{b}m");
                }
                cur_fg = Some(fg);
                cur_bg = Some(bg);
                cur_bold = bold;
            }

            let sym = cell.symbol();
            if sym.is_empty() {
                out.push(' ');
            } else {
                out.push_str(sym);
            }
        }
        // Clear to end of line (no newline) so we never scroll.
        out.push_str("\u{1b}[0m\u{1b}[K");
    }

    out
}

/// Map a ratatui `Color` to a concrete RGB triple, falling back to `default`
/// for `Reset`/indexed/named variants we don't translate.
const fn resolve_color(c: Option<Color>, default: (u8, u8, u8)) -> (u8, u8, u8) {
    match c {
        Some(Color::Rgb(r, g, b)) => (r, g, b),
        Some(Color::Black) => (0, 0, 0),
        Some(Color::Red) => (204, 0, 0),
        Some(Color::Green) => (78, 154, 6),
        Some(Color::Yellow) => (196, 160, 0),
        Some(Color::Blue) => (52, 101, 164),
        Some(Color::Magenta) => (117, 80, 123),
        Some(Color::Cyan) => (6, 152, 154),
        Some(Color::Gray) => (211, 215, 207),
        Some(Color::White) => (238, 238, 236),
        Some(Color::DarkGray) => (85, 87, 83),
        Some(Color::LightRed) => (239, 41, 41),
        Some(Color::LightGreen) => (138, 226, 52),
        Some(Color::LightYellow) => (252, 233, 79),
        Some(Color::LightBlue) => (114, 159, 207),
        Some(Color::LightMagenta) => (173, 127, 168),
        Some(Color::LightCyan) => (52, 226, 226),
        _ => default,
    }
}
