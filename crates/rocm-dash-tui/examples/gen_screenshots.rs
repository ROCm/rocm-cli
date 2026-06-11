//! Render the TUI against a synthetic session NDJSON and dump each view to
//! a standalone SVG. The screenshots embed cleanly in the marketing page
//! and demo decks — no live terminal, no manual capture, no drift.
//!
//! Usage:
//!   cargo run --release --example gen_screenshots -p rocm-dash-tui -- \
//!       --input /tmp/mi355x-demo.ndjson \
//!       --output-dir marketing/screenshots
//!
//! Each output is the actual `ui::draw` framebuffer for a specific
//! AppState configuration (tab + modal + theme). Cells get run-length
//! coalesced per row so the SVG stays compact (~30-80 KB / view).

use std::fs;
use std::path::PathBuf;

use clap::Parser;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::{Color, Modifier};
use ratatui::Terminal;
use rocm_dash_core::persist::PersistedEntry;
use rocm_dash_core::protocol::Event;
use rocm_dash_tui::app::{ActiveTab, AppState, ConnState, Modal};
use rocm_dash_tui::ui;

#[derive(Parser)]
#[command(
    name = "gen_screenshots",
    about = "Render TUI views to SVG for marketing"
)]
struct Args {
    /// Synthetic session file (produced by `make demo`).
    #[arg(long, default_value = "/tmp/mi355x-demo.ndjson")]
    input: PathBuf,

    /// Output directory. Created if absent. SVGs are named per view.
    /// Default targets the repo-root `marketing/screenshots/` directory
    /// when run from the `app/` workspace.
    #[arg(long, default_value = "../marketing/screenshots")]
    output_dir: PathBuf,

    /// How many snapshots to replay before capturing. Higher = fuller
    /// sparkline history. Capped by file length.
    #[arg(long, default_value_t = 90)]
    snapshots: usize,

    /// Terminal grid (cols × rows) the screenshots are rendered at.
    #[arg(long, default_value_t = 160)]
    cols: u16,
    #[arg(long, default_value_t = 44)]
    rows: u16,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)?;

    let entries = read_entries(&args.input)?;
    eprintln!(
        "loaded {} entries from {}",
        entries.len(),
        args.input.display()
    );

    // Build a baseline state by replaying the first N snapshots' worth of
    // events. We count "snapshots applied" to control the sparkline depth.
    let _base_state = build_state("default-dark", &entries, args.snapshots);

    // Each screenshot is (filename, mutator) — the mutator fork-edits a
    // clone of base_state for that view. Theme variants get a fresh state
    // with a different theme so the picker and gradients re-render.
    type BuildFn = Box<dyn Fn() -> AppState>;
    let mut tasks: Vec<(&str, BuildFn)> = Vec::new();
    let make = |theme: &'static str| {
        let entries = entries.clone();
        let n = args.snapshots;
        move || build_state(theme, &entries, n)
    };

    // Default theme: every tab + every modal.
    let mk_dark = make("default-dark");
    tasks.push((
        "overview",
        Box::new({
            let m = mk_dark.clone();
            move || {
                let mut s = m();
                s.active_tab = ActiveTab::Overview;
                s
            }
        }),
    ));
    tasks.push((
        "hardware",
        Box::new({
            let m = mk_dark.clone();
            move || {
                let mut s = m();
                s.active_tab = ActiveTab::Hardware;
                s
            }
        }),
    ));
    tasks.push((
        "hardware-detail",
        Box::new({
            let m = mk_dark.clone();
            move || {
                let mut s = m();
                s.active_tab = ActiveTab::Hardware;
                s.gpu_sel = 2;
                s.modal = Modal::Detail;
                s
            }
        }),
    ));
    tasks.push((
        "instances",
        Box::new({
            let m = mk_dark.clone();
            move || {
                let mut s = m();
                s.active_tab = ActiveTab::Instances;
                s
            }
        }),
    ));
    tasks.push((
        "instances-detail",
        Box::new({
            let m = mk_dark.clone();
            move || {
                let mut s = m();
                s.active_tab = ActiveTab::Instances;
                s.instance_sel = 0;
                s.modal = Modal::Detail;
                s
            }
        }),
    ));
    tasks.push((
        "bench",
        Box::new({
            let m = mk_dark.clone();
            move || {
                let mut s = m();
                s.active_tab = ActiveTab::Bench;
                s
            }
        }),
    ));
    tasks.push((
        "bench-detail",
        Box::new({
            let m = mk_dark.clone();
            move || {
                let mut s = m();
                s.active_tab = ActiveTab::Bench;
                s.bench_sel = s.bench_rows.len().saturating_sub(1);
                s.modal = Modal::Detail;
                s
            }
        }),
    ));
    tasks.push((
        "help",
        Box::new({
            let m = mk_dark.clone();
            move || {
                let mut s = m();
                s.modal = Modal::Help;
                s
            }
        }),
    ));
    tasks.push((
        "theme-picker",
        Box::new({
            let m = mk_dark.clone();
            move || {
                let mut s = m();
                s.open_theme_picker();
                // Highlight a non-default theme so the preview pane shows something different.
                let names = ui::theme::theme_names();
                if let Some(pos) = names.iter().position(|n| *n == "tokyo-night") {
                    s.theme_picker_sel = pos;
                }
                s
            }
        }),
    ));

    // Theme variants of the Overview tab — proof that the gradient
    // primitives retint correctly under arbitrary palettes.
    for theme in ["dracula", "gruvbox-dark", "nord", "catppuccin-mocha"] {
        let m = make(theme);
        let label = format!("overview-{theme}");
        tasks.push((
            Box::leak(label.into_boxed_str()),
            Box::new({
                move || {
                    let mut s = m();
                    s.active_tab = ActiveTab::Overview;
                    s
                }
            }),
        ));
    }

    let mut total_bytes: usize = 0;
    for (name, build) in tasks {
        let mut state = build();
        let svg = render_to_svg(&mut state, args.cols, args.rows)?;
        let path = args.output_dir.join(format!("{name}.svg"));
        fs::write(&path, &svg)?;
        total_bytes += svg.len();
        eprintln!(
            "  wrote {} ({:.1} KB)",
            path.display(),
            svg.len() as f64 / 1024.0
        );
    }
    eprintln!("done — {:.1} KB total", total_bytes as f64 / 1024.0);
    Ok(())
}

fn read_entries(path: &std::path::Path) -> anyhow::Result<Vec<PersistedEntry>> {
    let raw = fs::read_to_string(path)?;
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<PersistedEntry>(l).map_err(Into::into))
        .collect()
}

/// Replay entries into a fresh `AppState` until `snapshots_target` snapshot
/// events have been consumed (or the file runs out).
fn build_state(theme: &str, entries: &[PersistedEntry], snapshots_target: usize) -> AppState {
    let mut state = AppState::new("demo-mi355x-01".into(), theme.into());
    state.conn = ConnState::Connected {
        host: "demo-mi355x-01".into(),
        version: "0.1.0".into(),
    };
    let mut snap_count = 0usize;
    for entry in entries {
        state.apply_event(entry.event.clone());
        if matches!(entry.event, Event::Snapshot(_)) {
            snap_count += 1;
            if snap_count >= snapshots_target {
                break;
            }
        }
    }
    state
}

fn render_to_svg(state: &mut AppState, cols: u16, rows: u16) -> anyhow::Result<String> {
    let backend = TestBackend::new(cols, rows);
    let mut terminal = Terminal::new(backend)?;
    terminal.draw(|f| ui::draw(f, state))?;
    let buf = terminal.backend().buffer().clone();
    Ok(buffer_to_svg(&buf, state.theme.bg))
}

// ---------------------------------------------------------------------------
// Buffer → SVG
// ---------------------------------------------------------------------------

/// Cell box dimensions in SVG user-space. 14 px font with a tighter width
/// ratio gives a believable terminal look (~0.6 advance × 1.2 leading).
const CELL_W: f64 = 8.4;
const CELL_H: f64 = 17.0;
const FONT_PX: f64 = 14.0;
const FONT_FAMILY: &str = "ui-monospace, 'JetBrains Mono', 'Cascadia Code', \
                          'Fira Code', Menlo, 'DejaVu Sans Mono', monospace";

fn buffer_to_svg(buf: &Buffer, default_bg: Color) -> String {
    let cols = buf.area.width;
    let rows = buf.area.height;
    let w_px = cols as f64 * CELL_W;
    let h_px = rows as f64 * CELL_H;
    let bg_hex = color_to_hex(default_bg).unwrap_or_else(|| "#13141a".into());

    let mut out = String::with_capacity(rows as usize * cols as usize * 16);
    out.push_str(&format!(
        concat!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" "#,
            r#"viewBox="0 0 {w_px} {h_px}" width="{w_px}" height="{h_px}" "#,
            r#"font-family="{ff}" font-size="{fs}" "#,
            r#"shape-rendering="crispEdges" text-rendering="geometricPrecision">"#,
        ),
        w_px = w_px,
        h_px = h_px,
        ff = FONT_FAMILY,
        fs = FONT_PX,
    ));
    out.push_str(&format!(
        r#"<rect width="100%" height="100%" fill="{bg_hex}"/>"#
    ));

    // Per-row background runs.
    for y in 0..rows {
        let mut x = 0u16;
        while x < cols {
            let cell = buf.cell((x, y)).unwrap();
            let bg = cell_bg_or_default(cell);
            if bg == default_bg || matches!(bg, Color::Reset) {
                x += 1;
                continue;
            }
            let bg_hex_run = match color_to_hex(bg) {
                Some(h) => h,
                None => {
                    x += 1;
                    continue;
                }
            };
            let start = x;
            let mut end = x + 1;
            while end < cols {
                let next = buf.cell((end, y)).unwrap();
                if cell_bg_or_default(next) == bg {
                    end += 1;
                } else {
                    break;
                }
            }
            let rx = start as f64 * CELL_W;
            let ry = y as f64 * CELL_H;
            let rw = (end - start) as f64 * CELL_W;
            out.push_str(&format!(
                r#"<rect x="{rx:.2}" y="{ry:.2}" width="{rw:.2}" height="{CELL_H}" fill="{bg_hex_run}"/>"#
            ));
            x = end;
        }
    }

    // Per-row foreground runs. Group by (fg, bold) and emit one <text> per run.
    for y in 0..rows {
        let mut x = 0u16;
        let baseline = y as f64 * CELL_H + FONT_PX * 0.85;
        while x < cols {
            let cell = buf.cell((x, y)).unwrap();
            let sym = cell.symbol();
            if sym == " " || sym.is_empty() {
                x += 1;
                continue;
            }
            let fg = cell.style().fg.and_then(color_to_hex);
            let bold = cell.style().add_modifier.contains(Modifier::BOLD);
            let start = x;
            let mut text = String::new();
            text.push_str(&xml_escape(sym));
            let mut end = x + 1;
            while end < cols {
                let next = buf.cell((end, y)).unwrap();
                let nsym = next.symbol();
                if nsym == " " || nsym.is_empty() {
                    break;
                }
                let nfg = next.style().fg.and_then(color_to_hex);
                let nbold = next.style().add_modifier.contains(Modifier::BOLD);
                if nfg != fg || nbold != bold {
                    break;
                }
                text.push_str(&xml_escape(nsym));
                end += 1;
            }
            let tx = start as f64 * CELL_W;
            let fg_hex = fg.unwrap_or_else(|| "#eaebec".into());
            let weight = if bold { "700" } else { "400" };
            // Render each glyph at its own x via `textLength` so coalesced
            // runs preserve monospace alignment regardless of font metrics.
            let span_w = (end - start) as f64 * CELL_W;
            out.push_str(&format!(
                concat!(
                    r#"<text x="{tx:.2}" y="{baseline:.2}" "#,
                    r#"fill="{fg_hex}" font-weight="{weight}" "#,
                    r#"textLength="{span_w:.2}" lengthAdjust="spacingAndGlyphs">{text}</text>"#,
                ),
                tx = tx,
                baseline = baseline,
                fg_hex = fg_hex,
                weight = weight,
                span_w = span_w,
                text = text,
            ));
            x = end;
        }
    }

    out.push_str("</svg>");
    out
}

fn cell_bg_or_default(cell: &ratatui::buffer::Cell) -> Color {
    cell.style().bg.unwrap_or(Color::Reset)
}

fn color_to_hex(c: Color) -> Option<String> {
    match c {
        Color::Rgb(r, g, b) => Some(format!("#{r:02x}{g:02x}{b:02x}")),
        Color::Reset => None,
        // Named ANSI colors — map to sensible defaults for the rare cases
        // where a widget reaches for them directly. None of the dashboard's
        // widgets do; this is purely defensive.
        Color::Black => Some("#000000".into()),
        Color::Red => Some("#cc0000".into()),
        Color::Green => Some("#4e9a06".into()),
        Color::Yellow => Some("#c4a000".into()),
        Color::Blue => Some("#3465a4".into()),
        Color::Magenta => Some("#75507b".into()),
        Color::Cyan => Some("#06989a".into()),
        Color::Gray => Some("#d3d7cf".into()),
        Color::White => Some("#eeeeec".into()),
        Color::DarkGray => Some("#555753".into()),
        Color::LightRed => Some("#ef2929".into()),
        Color::LightGreen => Some("#8ae234".into()),
        Color::LightYellow => Some("#fce94f".into()),
        Color::LightBlue => Some("#729fcf".into()),
        Color::LightMagenta => Some("#ad7fa8".into()),
        Color::LightCyan => Some("#34e2e2".into()),
        Color::Indexed(_) => None,
    }
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}
