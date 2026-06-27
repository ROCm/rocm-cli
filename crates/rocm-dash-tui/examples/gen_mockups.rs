// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! High-fidelity, runnable previews of the *proposed* UX redesign described in
//! `docs/design/`. Each "scene" paints a mock layout to a real ratatui
//! framebuffer (real box-drawing, the real `Theme` palette, and the real
//! `GradientGauge` / `BrailleSparkline` instruments) and exports it to a
//! standalone SVG — so we can review the redesign at pixel fidelity *before*
//! committing it into the live `ui::draw` path.
//!
//! This example intentionally does NOT touch product code. It is a design
//! sketchbook: the draw functions here mirror what the real tabs/landing would
//! render, but live entirely under `examples/`.
//!
//! Usage:
//!   cargo run --release --example gen_mockups -p rocm-dash-tui
//!   cargo run --release --example gen_mockups -p rocm-dash-tui -- \
//!       --output-dir docs/design/mockups
//!
//! Output: one SVG per scene plus an `index.html` gallery.

#![allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_range_loop,
    // ponytail: this is a design sketchbook (renderer for the SVG mocks), not
    // production code — pedantic/nursery style lints aren't worth churn here.
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::type_complexity,
    clippy::explicit_auto_deref,
    clippy::too_many_arguments,
    clippy::extra_unused_lifetimes,
    clippy::drain_collect,
    clippy::iter_with_drain,
    clippy::uninlined_format_args
)]

use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::PathBuf;

use clap::Parser;
use ratatui::Frame;
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};

use rocm_dash_tui::ui::gradient::GradientGauge;
use rocm_dash_tui::ui::sparkline::BrailleSparkline;
use rocm_dash_tui::ui::theme::Theme;

#[derive(Parser)]
#[command(name = "gen_mockups", about = "Render proposed-UX mocks to SVG")]
struct Args {
    /// Output directory (created if absent).
    #[arg(long, default_value = "docs/design/mockups")]
    output_dir: PathBuf,
}

type SceneFn = fn(&mut Frame, &Theme);

struct Scene {
    name: &'static str,
    title: &'static str,
    theme: &'static str,
    cols: u16,
    rows: u16,
    draw: SceneFn,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)?;

    let scenes = [
        Scene {
            name: "minimal-launcher",
            title: "Surface 2 · bare `rocm` — minimal launcher (model running)",
            theme: "default-dark",
            cols: 98,
            rows: 20,
            draw: draw_minimal_running,
        },
        Scene {
            name: "minimal-launcher-idle",
            title: "Surface 2 · bare `rocm` — idle (greyed rows teach the unlock)",
            theme: "default-dark",
            cols: 98,
            rows: 20,
            draw: draw_minimal_idle,
        },
        Scene {
            name: "dash-home",
            title: "Surface 3 · `rocm dash` — Home landing (bento tiles, outlined tabs)",
            theme: "default-dark",
            cols: 150,
            rows: 30,
            draw: draw_dash_home,
        },
        Scene {
            name: "dash-home-tokyo",
            title: "Surface 3 · Home landing — alternate theme (tokyo-night)",
            theme: "tokyo-night",
            cols: 150,
            rows: 30,
            draw: draw_dash_home,
        },
        Scene {
            name: "dash-action",
            title: "Surface 3 · Action tab — guided workflows as visible tiles",
            theme: "default-dark",
            cols: 150,
            rows: 30,
            draw: draw_dash_action,
        },
        Scene {
            name: "dash-observe",
            title: "Surface 3 · Observe tab — live instruments + item-action rows",
            theme: "default-dark",
            cols: 150,
            rows: 30,
            draw: draw_dash_observe,
        },
        Scene {
            name: "dash-chat",
            title: "Surface 3 · Chat tab — agent with visible `Plan this` chip",
            theme: "default-dark",
            cols: 150,
            rows: 30,
            draw: draw_dash_chat,
        },
        Scene {
            name: "dash-palette",
            title: "Surface 3 · Command palette (`:`) — universal launcher overlay",
            theme: "default-dark",
            cols: 150,
            rows: 30,
            draw: draw_dash_palette,
        },
        Scene {
            name: "dash-menu",
            title: "Esc menu — btop-style gradient logo + Options / Help / Quit",
            theme: "default-dark",
            cols: 150,
            rows: 30,
            draw: draw_dash_menu,
        },
        Scene {
            name: "dash-options",
            title: "Options — tabbed settings (General · CPU · GPU · Engines)",
            theme: "default-dark",
            cols: 150,
            rows: 34,
            draw: draw_dash_options,
        },
        Scene {
            name: "dash-help",
            title: "Help — two-column keyboard reference",
            theme: "default-dark",
            cols: 150,
            rows: 32,
            draw: draw_dash_help,
        },
        Scene {
            name: "dash-wide-home",
            title: "Wide (15″+/27″) · Home — triptych: GPU wall ▏ workspace ▏ assistant",
            theme: "default-dark",
            cols: 220,
            rows: 54,
            draw: draw_wide_home,
        },
        Scene {
            name: "dash-wide-observe",
            title: "Wide (15″+/27″) · Observe — 8-GPU wall + instances master-detail",
            theme: "default-dark",
            cols: 220,
            rows: 54,
            draw: draw_wide_observe,
        },
        Scene {
            name: "dash-wide-action",
            title: "Wide (15″+/27″) · Action — list + live serve wizard side by side",
            theme: "default-dark",
            cols: 220,
            rows: 54,
            draw: draw_wide_action,
        },
        Scene {
            name: "dash-wide-chat",
            title: "Wide (15″+/27″) · Chat — conversation in center + agent context rail",
            theme: "default-dark",
            cols: 220,
            rows: 54,
            draw: draw_wide_chat,
        },
        Scene {
            name: "dash-wide-observe-logs",
            title: "Wide (15″+/27″) · Observe — contextual dock swaps assistant for live logs",
            theme: "default-dark",
            cols: 220,
            rows: 54,
            draw: draw_wide_observe_logs,
        },
    ];

    let mut gallery: Vec<(String, &str)> = Vec::new();
    for sc in &scenes {
        let theme = Theme::from_name(sc.theme);
        let backend = TestBackend::new(sc.cols, sc.rows);
        let mut terminal = Terminal::new(backend)?;
        terminal.draw(|f| (sc.draw)(f, &theme))?;
        let buf = terminal.backend().buffer().clone();
        let svg = buffer_to_svg(&buf, theme.bg);
        let file = format!("{}.svg", sc.name);
        fs::write(args.output_dir.join(&file), &svg)?;
        eprintln!("  wrote {file} ({:.1} KB)", svg.len() as f64 / 1024.0);
        gallery.push((file, sc.title));
    }

    let html = build_gallery(&gallery);
    fs::write(args.output_dir.join("index.html"), &html)?;
    eprintln!(
        "  wrote index.html — open it to view all {} scenes",
        scenes.len()
    );
    Ok(())
}

// ===========================================================================
// Shared painting helpers
// ===========================================================================

fn put(f: &mut Frame, x: u16, y: u16, s: &str, style: Style) {
    if y < f.area().height {
        f.buffer_mut().set_string(x, y, s, style);
    }
}

fn hline(f: &mut Frame, x0: u16, x1: u16, y: u16, ch: &str, style: Style) {
    for x in x0..=x1 {
        put(f, x, y, ch, style);
    }
}

/// Fill a rect with a flat background color (so tile surfaces show depth in SVG).
fn fill(f: &mut Frame, area: Rect, bg: Color) {
    let blank = " ".repeat(area.width as usize);
    let style = Style::default().bg(bg);
    for y in area.y..area.y.saturating_add(area.height) {
        put(f, area.x, y, &blank, style);
    }
}

/// A bento card. Returns the inner content rect. `double` = heavier border for
/// the hero tile (visual hierarchy); `accent` colors the border (focus).
fn card(
    f: &mut Frame,
    area: Rect,
    title: &str,
    theme: &Theme,
    border: Color,
    double: bool,
) -> Rect {
    fill(f, area, theme.surface);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(if double {
            BorderType::Double
        } else {
            BorderType::Rounded
        })
        .border_style(Style::default().fg(border))
        .title(format!(" {title} "))
        .title_style(Style::default().fg(theme.fg).add_modifier(Modifier::BOLD))
        .style(Style::default().bg(theme.surface));
    let inner = block.inner(area);
    f.render_widget(block, area);
    inner
}

fn text(f: &mut Frame, area: Rect, lines: Vec<Line<'_>>, bg: Color) {
    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .style(Style::default().bg(bg));
    f.render_widget(p, area);
}

fn chip<'a>(label: &'a str, theme: &Theme) -> Span<'a> {
    Span::styled(
        format!(" {label} "),
        Style::default().bg(theme.surface_2).fg(theme.fg),
    )
}

fn footer(f: &mut Frame, y: u16, width: u16, pairs: &[(&str, &str)], theme: &Theme) {
    let mut spans = Vec::new();
    for (k, desc) in pairs {
        spans.push(chip(k, theme));
        spans.push(Span::styled(
            format!(" {desc}   "),
            Style::default().fg(theme.muted),
        ));
    }
    f.render_widget(
        Paragraph::new(Line::from(spans)),
        Rect::new(1, y, width.saturating_sub(2), 1),
    );
}

fn lerp(a: Color, b: Color, t: f64) -> Color {
    let (ar, ag, ab) = rgb_of(a);
    let (br, bg, bb) = rgb_of(b);
    let m = |x: u8, y: u8| (f64::from(x) + (f64::from(y) - f64::from(x)) * t).round() as u8;
    Color::Rgb(m(ar, br), m(ag, bg), m(ab, bb))
}

fn rgb_of(c: Color) -> (u8, u8, u8) {
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => (0x80, 0x80, 0x80),
    }
}

/// Title row + a subtle accent→cyan gradient underline (the reference's warm
/// header band, kept on-brand: no AMD red, which is reserved for danger).
fn header_band(f: &mut Frame, width: u16, left: &str, right: &str, theme: &Theme) {
    put(
        f,
        1,
        0,
        left,
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );
    let rx = width.saturating_sub(right.chars().count() as u16 + 1);
    put(f, rx, 0, right, Style::default().fg(theme.muted));
    let w = width.max(1);
    for x in 0..w {
        let t = f64::from(x) / f64::from(w);
        put(
            f,
            x,
            1,
            "▁",
            Style::default().fg(lerp(theme.accent_2, theme.accent, t)),
        );
    }
}

/// Outlined folder tabs that open into the panel below. Active tab: accent
/// outline, leading ●, bottom edge opens into the content panel. Inactive:
/// muted closed boxes resting on the panel rim. Returns the inner content rect.
fn tab_panel(f: &mut Frame, outer: Rect, labels: &[&str], active: usize, theme: &Theme) -> Rect {
    let border = Style::default().fg(theme.border);
    let acc = Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD);
    let muted = Style::default().fg(theme.muted);

    let x0 = outer.x;
    let x1 = outer.x + outer.width - 1;
    let y_top = outer.y;
    let y_lab = outer.y + 1;
    let y_line = outer.y + 2;
    let y_bot = outer.y + outer.height - 1;

    // Panel box (top line first; tabs are stamped on top of it after).
    hline(f, x0, x1, y_line, "─", border);
    put(f, x0, y_line, "┌", border);
    put(f, x1, y_line, "┐", border);
    for y in (y_line + 1)..y_bot {
        put(f, x0, y, "│", border);
        put(f, x1, y, "│", border);
    }
    hline(f, x0, x1, y_bot, "─", border);
    put(f, x0, y_bot, "└", border);
    put(f, x1, y_bot, "┘", border);

    // Tabs.
    let mut sx = x0 + 2;
    for (i, lab) in labels.iter().enumerate() {
        let is_active = i == active;
        let content = if is_active {
            format!(" ● {lab} ")
        } else {
            format!(" {} {lab} ", i + 1)
        };
        let cw = content.chars().count() as u16;
        let ex = sx + cw + 1; // right border column (inclusive)
        let style = if is_active { acc } else { muted };

        // tops
        put(f, sx, y_top, "┌", style);
        hline(f, sx + 1, ex - 1, y_top, "─", style);
        put(f, ex, y_top, "┐", style);
        // label row
        put(f, sx, y_lab, "│", style);
        put(f, sx + 1, y_lab, &content, style);
        put(f, ex, y_lab, "│", style);
        // bottoms (merge with panel line)
        if is_active {
            put(f, sx, y_line, "┘", style);
            for x in (sx + 1)..ex {
                put(f, x, y_line, " ", Style::default().bg(theme.bg));
            }
            put(f, ex, y_line, "└", style);
        } else {
            put(f, sx, y_line, "┴", style);
            put(f, ex, y_line, "┴", style);
        }
        sx = ex + 2;
    }

    Rect::new(
        x0 + 2,
        y_line + 1,
        outer.width.saturating_sub(4),
        y_bot - y_line - 1,
    )
}

fn gauge(f: &mut Frame, area: Rect, ratio: f64, label: &str, theme: &Theme) {
    let g = GradientGauge::new(ratio)
        .stops(theme.ok, theme.warn, theme.err)
        .track_bg(theme.surface_2)
        .label(label)
        .label_fg(theme.fg);
    f.render_widget(g, area);
}

/// Value-gradient sparkline (green→amber→red by magnitude). For "load" metrics
/// where high = hot: utilization, VRAM, temperature, power.
fn spark(f: &mut Frame, area: Rect, data: &[u64], theme: &Theme) {
    let s = BrailleSparkline::new(data)
        .max(100)
        .style(Style::default().fg(theme.accent))
        .gradient(theme.ok, theme.warn, theme.err);
    f.render_widget(s, area);
}

/// Flat-accent sparkline. For "throughput" metrics where high = good and we
/// don't want a red alarm tint: tokens/sec, tokens/watt, gen-TPS.
fn spark_accent(f: &mut Frame, area: Rect, data: &[u64], theme: &Theme) {
    let s = BrailleSparkline::new(data)
        .max(100)
        .style(Style::default().fg(theme.accent));
    f.render_widget(s, area);
}

/// A labeled one-row instrument: `LABEL ▁▂▃▅▆▇  val`. `accent` picks the flat
/// throughput tint; otherwise the value-gradient is used.
fn mini_spark(
    f: &mut Frame,
    area: Rect,
    label: &str,
    val: &str,
    data: &[u64],
    accent: bool,
    theme: &Theme,
) {
    put(f, area.x, area.y, label, Style::default().fg(theme.muted));
    let lw = label.chars().count() as u16 + 1;
    let vw = val.chars().count() as u16 + 1;
    let sx = area.x + lw;
    let sw = area.width.saturating_sub(lw + vw);
    if sw > 1 {
        let sa = Rect::new(sx, area.y, sw, 1);
        if accent {
            spark_accent(f, sa, data, theme);
        } else {
            spark(f, sa, data, theme);
        }
    }
    put(
        f,
        area.x + area.width - vw + 1,
        area.y,
        val,
        Style::default().fg(if accent { theme.accent } else { theme.fg }),
    );
}

// Believable synthetic instrument series — each a distinct shape so the
// sparklines read as different instruments, not copies.
fn util_history() -> Vec<u64> {
    vec![
        18, 22, 27, 33, 41, 48, 55, 60, 63, 62, 58, 61, 66, 70, 68, 64, 59, 62, 67, 71, 69, 65, 60,
        63, 68, 72, 70, 66, 61, 62,
    ]
}
fn tpw_history() -> Vec<u64> {
    // tokens/watt — climbs as the model warms up, then plateaus high.
    vec![
        30, 38, 47, 55, 63, 70, 76, 80, 83, 85, 84, 86, 88, 87, 89, 90, 88, 89, 91, 90, 92, 91,
    ]
}
fn vram_history() -> Vec<u64> {
    // VRAM — steps up as weights + KV cache load, then steady.
    vec![
        8, 9, 12, 20, 33, 44, 52, 58, 60, 61, 62, 62, 63, 63, 64, 64, 63, 64, 64, 64, 65, 64,
    ]
}
fn temp_history() -> Vec<u64> {
    // temperature — gentle rise into the mid range.
    vec![
        30, 31, 33, 35, 38, 41, 43, 45, 46, 47, 48, 49, 49, 50, 50, 51, 51, 50, 51, 51, 52, 51,
    ]
}
fn power_history() -> Vec<u64> {
    // power draw — spiky under bursty decode.
    vec![
        40, 55, 48, 62, 70, 58, 66, 78, 64, 72, 85, 69, 74, 88, 71, 79, 90, 73, 81, 86, 70, 84,
    ]
}
fn gtps_history() -> Vec<u64> {
    // generation TPS across recent runs — variable but healthy.
    vec![
        52, 60, 57, 64, 70, 66, 72, 68, 75, 71, 78, 74, 69, 76, 80, 73, 77, 82, 75, 79, 84, 78,
    ]
}
fn tps_history() -> Vec<u64> {
    vec![
        20, 28, 36, 45, 52, 58, 55, 60, 64, 61, 66, 63, 68, 65, 62, 67, 70, 66, 63, 69, 72, 68,
    ]
}
// ===========================================================================
// Surface 2 — minimal launcher
// ===========================================================================

fn draw_minimal_running(f: &mut Frame, theme: &Theme) {
    let area = f.area();
    fill(f, area, theme.bg);
    header_band(f, area.width, "rocm.ai", "v0.9 · ROCm 6.2", theme);

    // Live status strip (the ccstatusline-style preview band) — now 3 rows:
    // two text lines + an instrument row of braille sparklines.
    let strip = Rect::new(2, 3, area.width - 4, 3);
    fill(f, strip, theme.surface);
    text(
        f,
        Rect::new(strip.x, strip.y, strip.width, 2),
        vec![
            Line::from(vec![
                Span::styled("GPU ", Style::default().fg(theme.muted)),
                Span::styled("Radeon 8060S (Strix Halo)", Style::default().fg(theme.fg)),
                Span::styled("  │  ", Style::default().fg(theme.border)),
                Span::styled("Util 34%", Style::default().fg(theme.ok)),
                Span::styled("  │  ", Style::default().fg(theme.border)),
                Span::styled("VRAM 18 / 128 GB", Style::default().fg(theme.fg)),
                Span::styled("  │  ", Style::default().fg(theme.border)),
                Span::styled("51°C", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled("Serving ", Style::default().fg(theme.muted)),
                Span::styled(
                    "Qwen3-72B",
                    Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" on :8000", Style::default().fg(theme.muted)),
                Span::styled("  │  ", Style::default().fg(theme.border)),
                Span::styled("42 t/s", Style::default().fg(theme.accent)),
                Span::styled("  │  ", Style::default().fg(theme.border)),
                Span::styled("✓ healthy", Style::default().fg(theme.ok)),
            ]),
        ],
        theme.surface,
    );
    let trend_y = strip.y + 2;
    let half = strip.width / 2;
    mini_spark(
        f,
        Rect::new(strip.x, trend_y, half - 1, 1),
        "GPU",
        "62%",
        &util_history(),
        false,
        theme,
    );
    mini_spark(
        f,
        Rect::new(strip.x + half, trend_y, half - 1, 1),
        "t/s",
        "42",
        &tps_history(),
        true,
        theme,
    );

    put(
        f,
        2,
        7,
        "What would you like to do?",
        Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
    );

    // (icon, label, desc, focused, greyed, badge)
    let rows: &[(&str, &str, &str, bool, bool, Option<&str>)] = &[
        (
            "◆",
            "Serve a model",
            "run another model on your GPU",
            true,
            false,
            None,
        ),
        (
            "⚙",
            "Set up this system",
            "install / update ROCm",
            false,
            false,
            None,
        ),
        (
            "⚕",
            "Diagnose & fix",
            "check GPU, driver & ROCm",
            false,
            false,
            None,
        ),
        (
            "◷",
            "Chat",
            "talk to Qwen3-72B or an API model",
            false,
            false,
            None,
        ),
        (
            "⚡",
            "Optimize a model",
            "planned — hyperloom",
            false,
            true,
            Some("soon"),
        ),
    ];
    let mut y = 9;
    for (icon, label, desc, focused, greyed, badge) in rows {
        draw_menu_row(f, y, *icon, label, desc, *focused, *greyed, *badge, theme);
        y += 1;
    }
    y += 1;
    draw_menu_row(
        f,
        y,
        "▣",
        "Open full dashboard  →",
        "live instruments & every action",
        false,
        false,
        None,
        theme,
    );

    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("↑↓", "move"),
            ("Enter", "select"),
            ("d", "dashboard"),
            ("t", "theme"),
            ("q", "quit"),
        ],
        theme,
    );
}

fn draw_minimal_idle(f: &mut Frame, theme: &Theme) {
    let area = f.area();
    fill(f, area, theme.bg);
    header_band(f, area.width, "rocm.ai", "v0.9 · ROCm 6.2", theme);

    let strip = Rect::new(2, 3, area.width - 4, 3);
    fill(f, strip, theme.surface);
    text(
        f,
        Rect::new(strip.x, strip.y, strip.width, 2),
        vec![
            Line::from(vec![
                Span::styled("GPU ", Style::default().fg(theme.muted)),
                Span::styled("Radeon 8060S (Strix Halo)", Style::default().fg(theme.fg)),
                Span::styled("  │  ", Style::default().fg(theme.border)),
                Span::styled("Util 2%", Style::default().fg(theme.muted)),
                Span::styled("  │  ", Style::default().fg(theme.border)),
                Span::styled("VRAM 0 / 128 GB", Style::default().fg(theme.muted)),
                Span::styled("  │  ", Style::default().fg(theme.border)),
                Span::styled("44°C", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("◌ ", Style::default().fg(theme.muted)),
                Span::styled("No model running", Style::default().fg(theme.muted)),
            ]),
        ],
        theme.surface,
    );
    // Idle flatline — an honest "nothing happening" instrument, not a blank.
    let idle: Vec<u64> = vec![2, 1, 2, 3, 2, 1, 2, 2, 1, 3, 2, 1, 2, 2, 1, 2, 3, 2, 1, 2];
    mini_spark(
        f,
        Rect::new(strip.x, strip.y + 2, strip.width / 2 - 1, 1),
        "GPU",
        "2%",
        &idle,
        false,
        theme,
    );

    put(
        f,
        2,
        7,
        "What would you like to do?",
        Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
    );

    draw_menu_row(
        f,
        9,
        "◆",
        "Serve a model",
        "run a model on your GPU",
        true,
        false,
        None,
        theme,
    );
    draw_menu_row(
        f,
        10,
        "⚙",
        "Set up this system",
        "install / update ROCm",
        false,
        false,
        None,
        theme,
    );
    draw_menu_row(
        f,
        11,
        "⚕",
        "Diagnose & fix",
        "check GPU, driver & ROCm",
        false,
        false,
        None,
        theme,
    );
    draw_menu_row(
        f,
        12,
        "·",
        "Chat",
        "add a provider or serve a model",
        false,
        true,
        None,
        theme,
    );
    draw_menu_row(
        f,
        13,
        "⚡",
        "Optimize a model",
        "planned — hyperloom",
        false,
        true,
        Some("soon"),
        theme,
    );
    draw_menu_row(
        f,
        15,
        "▣",
        "Open full dashboard  →",
        "live instruments & every action",
        false,
        false,
        None,
        theme,
    );

    put(
        f,
        2,
        17,
        "✦ New to rocm? Press Enter on “Set up this system”.",
        Style::default().fg(theme.accent_2),
    );

    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("↑↓", "move"),
            ("Enter", "select"),
            ("d", "dashboard"),
            ("t", "theme"),
            ("q", "quit"),
        ],
        theme,
    );
}

fn draw_menu_row(
    f: &mut Frame,
    y: u16,
    icon: &str,
    label: &str,
    desc: &str,
    focused: bool,
    greyed: bool,
    badge: Option<&str>,
    theme: &Theme,
) {
    let (cursor, icon_c, label_c) = if greyed {
        ("  ", theme.muted, theme.muted)
    } else if focused {
        ("▸ ", theme.accent, theme.accent)
    } else {
        ("  ", theme.accent_2, theme.fg)
    };
    let label_style = if focused {
        Style::default().fg(label_c).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(label_c)
    };
    let mut spans = vec![
        Span::styled(cursor, Style::default().fg(theme.accent)),
        Span::styled(format!("{icon}  "), Style::default().fg(icon_c)),
        Span::styled(format!("{label:<22}"), label_style),
    ];
    if let Some(b) = badge {
        spans.push(Span::styled(
            format!(" {b} "),
            Style::default()
                .bg(theme.warn)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw(" "));
    }
    spans.push(Span::styled(
        if greyed && badge.is_none() {
            format!("({desc})")
        } else {
            desc.to_string()
        },
        Style::default().fg(theme.muted),
    ));
    f.render_widget(
        Paragraph::new(Line::from(spans)),
        Rect::new(3, y, f.area().width - 4, 1),
    );
}

// ===========================================================================
// Surface 3 — dash scenes
// ===========================================================================

fn dash_shell<'a>(f: &mut Frame, theme: &Theme, active: usize) -> Rect {
    let area = f.area();
    fill(f, area, theme.bg);
    // Title row.
    put(
        f,
        1,
        0,
        "rocm.ai",
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );
    let status = "connected · demo-strix-01 · rocm daemon 0.9";
    put(
        f,
        area.width - status.chars().count() as u16 - 1,
        0,
        status,
        Style::default().fg(theme.ok),
    );
    put(
        f,
        1,
        0,
        "rocm.ai",
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );
    // Palette / help hint on the tab row's right side.
    let hint = "Esc  menu    t  theme    ?  help";
    put(
        f,
        area.width - hint.chars().count() as u16 - 2,
        2,
        hint,
        Style::default().fg(theme.muted),
    );
    let outer = Rect::new(0, 1, area.width, area.height - 2);
    let _ = active;
    tab_panel(
        f,
        outer,
        &["Home", "Action", "Observe", "Chat"],
        active,
        theme,
    )
}

fn draw_dash_home(f: &mut Frame, theme: &Theme) {
    let inner = dash_shell(f, theme, 0);
    let area = f.area();

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(11),
            Constraint::Length(11),
            Constraint::Min(0),
        ])
        .margin(1)
        .split(inner);

    // --- Top band: hero + next step ---
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(rows[0]);

    let hero = card(
        f,
        top[0],
        "AMD Radeon 8060S · Strix Halo",
        theme,
        theme.border,
        true,
    );
    let hcols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .spacing(2)
        .split(hero);
    // Left: the headline utilization gauge + its trace, then tokens/watt.
    let lh = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1); 6])
        .split(hcols[0]);
    put(
        f,
        lh[0].x,
        lh[0].y,
        "GPU UTILIZATION",
        Style::default().fg(theme.muted),
    );
    gauge(f, lh[1], 0.62, "62%", theme);
    spark(f, lh[2], &util_history(), theme);
    put(
        f,
        lh[4].x,
        lh[4].y,
        "⎓ 24.1 tokens / watt",
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );
    spark_accent(f, lh[5], &tpw_history(), theme);
    // Right: a stacked instrument cluster — one braille sparkline per metric.
    let rh = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1); 6])
        .split(hcols[1]);
    put(
        f,
        rh[0].x,
        rh[0].y,
        "LIVE · last 60s",
        Style::default().fg(theme.muted),
    );
    mini_spark(f, rh[1], "VRAM ", "64%", &vram_history(), false, theme);
    mini_spark(f, rh[2], "TEMP ", "51°C", &temp_history(), false, theme);
    mini_spark(f, rh[3], "POWER", "86 W", &power_history(), false, theme);
    mini_spark(f, rh[4], "T/S  ", "42", &tps_history(), true, theme);
    put(
        f,
        rh[5].x,
        rh[5].y,
        "82 / 128 GB used",
        Style::default().fg(theme.muted),
    );

    let next = card(f, top[1], "Next step", theme, theme.accent, false);
    text(
        f,
        next,
        vec![
            Line::from(vec![
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled(
                    "Qwen3-72B is live",
                    Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                "on :8000 · 42 t/s",
                Style::default().fg(theme.muted),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("▸ ", Style::default().fg(theme.accent)),
                Span::styled(
                    "Open Chat",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("        →", Style::default().fg(theme.muted)),
            ]),
            Line::from(vec![
                Span::styled("  View in Observe", Style::default().fg(theme.fg)),
                Span::styled("  →", Style::default().fg(theme.muted)),
            ]),
        ],
        theme.surface,
    );

    // --- Middle band: three tiles ---
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ])
        .split(rows[1]);

    let running = card(f, mid[0], "Running · 1", theme, theme.border, false);
    text(
        f,
        running,
        vec![
            Line::from(vec![
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled(
                    "Qwen3-72B",
                    Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                ),
                Span::styled("   vLLM   :8000", Style::default().fg(theme.muted)),
            ]),
            Line::from(Span::styled(
                "  TP 1 · 1 GPU",
                Style::default().fg(theme.muted),
            )),
        ],
        theme.surface,
    );
    mini_spark(
        f,
        Rect::new(running.x, running.y + 2, running.width, 1),
        "t/s ",
        "42",
        &tps_history(),
        true,
        theme,
    );
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("▸ ", Style::default().fg(theme.accent)),
            Span::styled("Serve another", Style::default().fg(theme.fg)),
            Span::styled("   →", Style::default().fg(theme.muted)),
        ])),
        Rect::new(running.x, running.y + 4, running.width, 1),
    );

    let health = card(f, mid[1], "Health", theme, theme.border, false);
    text(
        f,
        health,
        vec![
            Line::from(vec![
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled("GPU", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled("Driver", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled("ROCm 6.2", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("▸ ", Style::default().fg(theme.accent)),
                Span::styled("Run doctor →", Style::default().fg(theme.fg)),
            ]),
        ],
        theme.surface,
    );

    let updates = card(f, mid[2], "Updates", theme, theme.warn, false);
    text(
        f,
        updates,
        vec![
            Line::from(vec![
                Span::styled("⇲ ", Style::default().fg(theme.warn)),
                Span::styled(
                    "ROCm 6.3 ready",
                    Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                "  approval required",
                Style::default().fg(theme.muted),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("▸ ", Style::default().fg(theme.accent)),
                Span::styled("View update →", Style::default().fg(theme.fg)),
            ]),
        ],
        theme.surface,
    );

    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("↑↓←→", "move tile"),
            ("Enter", "open"),
            ("1–4", "tabs"),
            ("w", "serve"),
            ("t", "theme"),
            ("?", "help"),
            ("q", "quit"),
        ],
        theme,
    );
}

fn draw_dash_action(f: &mut Frame, theme: &Theme) {
    let inner = dash_shell(f, theme, 1);
    let area = f.area();

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .margin(1)
        .split(inner);

    // Left: action list. (icon, label, focused, greyed, soon)
    let list = card(f, cols[0], "Actions", theme, theme.border, false);
    let actions: &[(&str, &str, bool, bool, bool)] = &[
        ("◆", "Serve a model", true, false, false),
        ("⚙", "Set up / Install ROCm", false, false, false),
        ("⌬", "Engines", false, false, false),
        ("⚕", "Diagnose & fix  (doctor)", false, false, false),
        ("⇲", "Check for updates", false, false, false),
        ("⮌", "Manage providers & keys", false, false, false),
        ("⚡", "Optimize a model", false, true, true),
        ("⊘", "Uninstall", false, true, false),
    ];
    let mut lines = Vec::new();
    for (icon, label, focused, greyed, soon) in actions {
        let (cur, c) = if *greyed {
            ("  ", theme.muted)
        } else if *focused {
            ("▸ ", theme.accent)
        } else {
            ("  ", theme.fg)
        };
        let style = if *focused {
            Style::default().fg(c).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(c)
        };
        let mut row = vec![
            Span::styled(cur, Style::default().fg(theme.accent)),
            Span::styled(
                format!("{icon}  "),
                Style::default().fg(if *greyed { theme.muted } else { theme.accent_2 }),
            ),
            Span::styled(*label, style),
        ];
        if *soon {
            row.push(Span::raw("  "));
            row.push(Span::styled(
                " soon ",
                Style::default()
                    .bg(theme.warn)
                    .fg(theme.bg)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::from(row));
        lines.push(Line::from(""));
    }
    lines.push(Line::from(Span::styled(
        "Background helpers ▾        soon = planned, not yet built",
        Style::default().fg(theme.muted),
    )));
    text(f, list, lines, theme.surface);

    // Right: detail / preview pane for the focused action.
    let detail = card(f, cols[1], "◆ Serve a model", theme, theme.accent, false);
    text(
        f,
        detail,
        vec![
            Line::from(Span::styled(
                "Bring a model up as a local OpenAI-compatible endpoint.",
                Style::default().fg(theme.fg),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Recommended for your GPU:",
                Style::default().fg(theme.muted),
            )),
            Line::from(vec![
                Span::styled(
                    "  Qwen3-72B   ",
                    Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                ),
                Span::styled("✓ fits · ~minutes", Style::default().fg(theme.ok)),
            ]),
            Line::from(vec![
                Span::styled("  Engine: ", Style::default().fg(theme.muted)),
                Span::styled(
                    "Lemonade (auto-selected for Strix Halo)",
                    Style::default().fg(theme.fg),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    "  ▸ Start serving ",
                    Style::default()
                        .fg(theme.bg)
                        .bg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Enter", Style::default().fg(theme.muted)),
            ]),
            Line::from(vec![
                Span::styled("    Advanced engine options", Style::default().fg(theme.fg)),
                Span::styled("   a", Style::default().fg(theme.muted)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Mutating actions ask before they run. Default mode: ask.",
                Style::default().fg(theme.muted),
            )),
        ],
        theme.surface,
    );

    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("↑↓", "choose"),
            ("Enter", "start"),
            ("a", "advanced"),
            ("t", "theme"),
            ("?", "help"),
            ("Esc", "Home"),
        ],
        theme,
    );
}

fn draw_dash_observe(f: &mut Frame, theme: &Theme) {
    let inner = dash_shell(f, theme, 2);
    let area = f.area();

    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(8),
            Constraint::Min(0),
        ])
        .margin(1)
        .split(inner);

    // Demo-data banner (honesty rule F151).
    let banner = body[0];
    fill(f, banner, theme.surface_2);
    text(
        f,
        banner,
        vec![
            Line::from(vec![
                Span::styled(
                    " ⚠  Demo data — not your live GPU. ",
                    Style::default().fg(theme.warn).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Live telemetry isn’t available on this machine yet;",
                    Style::default().fg(theme.fg),
                ),
            ]),
            Line::from(Span::styled(
                "    showing a recorded session so you can explore.",
                Style::default().fg(theme.muted),
            )),
        ],
        theme.surface_2,
    );

    // Instrument cluster — six labeled braille sparklines in a 3×2 grid,
    // each its own metric. The headline utilization also keeps a gauge.
    let instr = card(
        f,
        body[1],
        "GPU 0 · MI300X · gfx942 — instruments",
        theme,
        theme.border,
        false,
    );
    let il = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(instr);
    // Headline gauge row.
    let gl = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(22), Constraint::Min(0)])
        .spacing(2)
        .split(il[0]);
    put(
        f,
        gl[0].x,
        gl[0].y,
        "GPU UTIL",
        Style::default().fg(theme.muted),
    );
    gauge(
        f,
        Rect::new(gl[0].x + 9, gl[0].y, gl[0].width.saturating_sub(9), 1),
        0.62,
        "62%",
        theme,
    );
    put(
        f,
        gl[1].x,
        gl[1].y,
        "live · last 60s · gradient = magnitude (green→amber→red)",
        Style::default().fg(theme.muted),
    );
    // Two bands of three sparklines.
    let band1 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .spacing(2)
        .split(il[1]);
    mini_spark(f, band1[0], "UTIL ", "62%", &util_history(), false, theme);
    mini_spark(f, band1[1], "TEMP ", "51°C", &temp_history(), false, theme);
    mini_spark(f, band1[2], "POWER", "86 W", &power_history(), false, theme);
    let band2 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .spacing(2)
        .split(il[2]);
    mini_spark(f, band2[0], "VRAM ", "64%", &vram_history(), false, theme);
    mini_spark(f, band2[1], "⎓tok/W", "24.1", &tpw_history(), true, theme);
    mini_spark(f, band2[2], "T/S  ", "42", &tps_history(), true, theme);
    gauge(
        f,
        Rect::new(il[3].x, il[3].y, il[3].width, 1),
        0.64,
        "VRAM 82 / 128 GB",
        theme,
    );

    // Instances + bench tables.
    let tbl = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(body[2]);

    let insts = card(f, tbl[0], "Instances · 1", theme, theme.border, false);
    text(
        f,
        insts,
        vec![
            Line::from(Span::styled(
                format!("{:<12}{:<14}{:>6}{:>7}", "name", "model", "port", "t/s"),
                Style::default()
                    .fg(theme.muted)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled(
                    format!(
                        "{:<10}{:<14}{:>6}{:>7}",
                        "qwen-72b", "Qwen3-72B", "8000", "42"
                    ),
                    Style::default().fg(theme.fg),
                ),
            ]),
        ],
        theme.surface,
    );
    // Per-instance throughput trace.
    mini_spark(
        f,
        Rect::new(insts.x, insts.y + 2, insts.width, 1),
        "  gen t/s ",
        "42",
        &tps_history(),
        true,
        theme,
    );
    text(
        f,
        Rect::new(
            insts.x,
            insts.y + 4,
            insts.width,
            insts.height.saturating_sub(4),
        ),
        vec![
            Line::from(vec![
                Span::styled(
                    "▸ Stop",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Restart   Logs   Detail", Style::default().fg(theme.fg)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("attribution: ", Style::default().fg(theme.muted)),
                Span::styled("per-process", Style::default().fg(theme.ok)),
                Span::styled(
                    "   legend: ● running  ◌ stopped  ↺ rollback",
                    Style::default().fg(theme.muted),
                ),
            ]),
        ],
        theme.surface,
    );

    let bench = card(f, tbl[1], "Bench · last run", theme, theme.border, false);
    text(
        f,
        bench,
        vec![
            Line::from(Span::styled(
                format!("{:<8}{:<12}{:>7}{:>5}", "cell", "model", "gTPS", "✓/✗"),
                Style::default()
                    .fg(theme.muted)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled(
                    format!("{:<8}{:<12}{:>7}", "tg128", "Qwen3-72B", "41.8"),
                    Style::default().fg(theme.fg),
                ),
                Span::styled("   ✓", Style::default().fg(theme.ok)),
            ]),
            Line::from(vec![
                Span::styled(
                    format!("{:<8}{:<12}{:>7}", "pp512", "Qwen3-72B", "3.1k"),
                    Style::default().fg(theme.fg),
                ),
                Span::styled("   ✓", Style::default().fg(theme.ok)),
            ]),
        ],
        theme.surface,
    );
    // gen-TPS across recent runs.
    mini_spark(
        f,
        Rect::new(bench.x, bench.y + 3, bench.width, 1),
        "gTPS ",
        "trend",
        &gtps_history(),
        true,
        theme,
    );
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("▸ ", Style::default().fg(theme.accent)),
            Span::styled("Run benchmark →", Style::default().fg(theme.fg)),
        ])),
        Rect::new(bench.x, bench.y + 5, bench.width, 1),
    );

    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("↑↓", "select"),
            ("Enter", "detail"),
            ("s", "stop"),
            ("l", "logs"),
            ("F5", "refresh"),
            ("t", "theme"),
            ("?", "help"),
        ],
        theme,
    );
}

fn draw_dash_chat(f: &mut Frame, theme: &Theme) {
    let inner = dash_shell(f, theme, 3);
    let area = f.area();

    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .margin(1)
        .split(inner);

    let convo = body[0];
    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                "you   ",
                Style::default()
                    .fg(theme.accent_2)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "serve qwen and tell me when it's ready",
                Style::default().fg(theme.fg),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("●     ", Style::default().fg(theme.ok)),
            Span::styled(
                "I'll start Qwen3-72B with vLLM. This downloads weights and brings",
                Style::default().fg(theme.fg),
            ),
        ]),
        Line::from(Span::styled(
            "      up an endpoint on :8000. Approve?",
            Style::default().fg(theme.fg),
        )),
        Line::from(""),
    ];
    text(f, convo, lines.drain(..).collect(), theme.bg);

    // Proposed-tool review card overlapping the convo.
    let rc = Rect::new(convo.x + 6, convo.y + 6, 52, 4);
    let rci = card(f, rc, "proposed", theme, theme.accent, false);
    text(
        f,
        rci,
        vec![
            Line::from(Span::styled(
                "Start vLLM · Qwen3-72B · :8000",
                Style::default().fg(theme.fg),
            )),
            Line::from(vec![
                Span::styled(
                    " ▸ Approve ",
                    Style::default()
                        .fg(theme.bg)
                        .bg(theme.ok)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  Reject   Edit", Style::default().fg(theme.fg)),
            ]),
        ],
        theme.surface,
    );

    // Live serving status with a throughput trace, once the tool has run.
    put(
        f,
        convo.x,
        convo.y + 11,
        "● Qwen3-72B is live on :8000",
        Style::default().fg(theme.ok),
    );
    mini_spark(
        f,
        Rect::new(convo.x, convo.y + 12, 44, 1),
        "  t/s ",
        "42",
        &tps_history(),
        true,
        theme,
    );

    // Action-chip row.
    let chips = Line::from(vec![
        Span::styled(
            " ✦ Plan this ",
            Style::default()
                .bg(theme.accent)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        chip("◆ Serve", theme),
        Span::raw("  "),
        chip("⚕ Doctor", theme),
        Span::styled(
            "        provider: claude ",
            Style::default().fg(theme.muted),
        ),
        Span::styled("●", Style::default().fg(theme.ok)),
    ]);
    f.render_widget(
        Paragraph::new(chips),
        Rect::new(inner.x + 1, body[1].y, inner.width - 2, 1),
    );

    // Composer.
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("> ", Style::default().fg(theme.accent)),
            Span::styled("▌", Style::default().fg(theme.fg)),
        ])),
        Rect::new(inner.x + 1, body[2].y, inner.width - 2, 1),
    );

    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("Enter", "send"),
            ("/", "commands"),
            ("✦", "plan"),
            ("Shift+Tab", "approve"),
            ("t", "theme"),
            ("?", "help"),
        ],
        theme,
    );
}

fn draw_dash_palette(f: &mut Frame, theme: &Theme) {
    // Reuse Home as the dimmed backdrop, then overlay the palette card.
    draw_dash_home(f, theme);
    let area = f.area();

    let w = 72u16;
    let h = 12u16;
    let px = (area.width - w) / 2;
    let py = (area.height - h) / 2;
    let rect = Rect::new(px, py, w, h);
    fill(f, rect, theme.surface_2);
    let inner = card(f, rect, "Go to…", theme, theme.accent, false);
    // override card fill to surface_2 for the overlay
    fill(f, inner, theme.surface_2);

    let rows = vec![
        Line::from(vec![
            Span::styled(": ", Style::default().fg(theme.muted)),
            Span::styled("serv", Style::default().fg(theme.fg)),
            Span::styled("▌", Style::default().fg(theme.accent)),
        ]),
        Line::from(Span::styled(
            "─".repeat(inner.width as usize),
            Style::default().fg(theme.border),
        )),
        Line::from(vec![
            Span::styled("▸ ◆  ", Style::default().fg(theme.accent)),
            Span::styled(
                "Serve a model",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "            start a local endpoint",
                Style::default().fg(theme.muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  ⌬  ", Style::default().fg(theme.accent_2)),
            Span::styled("Engines", Style::default().fg(theme.fg)),
            Span::styled(
                "                  install / pick an engine",
                Style::default().fg(theme.muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  ⚕  ", Style::default().fg(theme.accent_2)),
            Span::styled("Services", Style::default().fg(theme.fg)),
            Span::styled(
                "                 stop or inspect running",
                Style::default().fg(theme.muted),
            ),
        ]),
        Line::from(Span::styled(
            "─".repeat(inner.width as usize),
            Style::default().fg(theme.border),
        )),
        Line::from(vec![
            Span::styled("recent:  ", Style::default().fg(theme.muted)),
            Span::styled(
                "Doctor · Serve Qwen3-72B · Theme",
                Style::default().fg(theme.fg),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "type to filter   ↑↓ move   Enter open   Esc close",
            Style::default().fg(theme.muted),
        )),
    ];
    text(f, inner, rows, theme.surface_2);
}

// ===========================================================================
// Esc main menu — btop-style gradient logo + Options / Help / Quit
// ===========================================================================

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    Rect::new(x, y, w.min(area.width), h.min(area.height))
}

/// Dim the whole frame to a grey wash so a modal reads as focused on top.
fn grey_overlay(f: &mut Frame) {
    let area = f.area();
    let wash = Color::Rgb(0x1c, 0x1e, 0x22);
    let dim = Color::Rgb(0x4c, 0x50, 0x57);
    let buf = f.buffer_mut();
    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            if let Some(c) = buf.cell_mut((x, y)) {
                c.set_style(Style::default().fg(dim).bg(wash));
            }
        }
    }
}

/// btop-style three-stop gradient (accent_2 → accent → bright cyan).
fn grad3(theme: &Theme, t: f64) -> Color {
    let light = Color::Rgb(0xc4, 0xf2, 0xff);
    if t < 0.5 {
        lerp(theme.accent_2, theme.accent, t * 2.0)
    } else {
        lerp(theme.accent, light, (t - 0.5) * 2.0)
    }
}

/// Big block "ROCm" with a horizontal gradient sweep. `cx` is the left column
/// of the 31-wide logo; it occupies 5 rows from `y`.
fn draw_logo(f: &mut Frame, cx: u16, y: u16, theme: &Theme) {
    const R: [&str; 5] = ["██████ ", "██   ██", "██████ ", "██   ██", "██   ██"];
    const O: [&str; 5] = [" █████ ", "██   ██", "██   ██", "██   ██", " █████ "];
    const C: [&str; 5] = [" ██████", "██     ", "██     ", "██     ", " ██████"];
    // lowercase "m": blank top row, sits on the baseline like R/O/C bottoms.
    const M: [&str; 5] = ["       ", "██████ ", "██ █ ██", "██ █ ██", "██ █ ██"];
    for i in 0..5 {
        let line = format!("{} {} {} {}", R[i], O[i], C[i], M[i]);
        let n = line.chars().count().max(2);
        for (j, ch) in line.chars().enumerate() {
            if ch != ' ' {
                let t = j as f64 / (n - 1) as f64;
                put(
                    f,
                    cx + j as u16,
                    y + i as u16,
                    &ch.to_string(),
                    Style::default().fg(grad3(theme, t)),
                );
            }
        }
    }
}

fn draw_dash_menu(f: &mut Frame, theme: &Theme) {
    draw_dash_home(f, theme); // backdrop
    grey_overlay(f);
    let area = f.area();
    let modal = centered(area, 60, 17);
    fill(f, modal, theme.surface);
    let inner = card(f, modal, "", theme, theme.accent, true);

    let logo_w = 31u16;
    let cx = inner.x + inner.width.saturating_sub(logo_w) / 2;
    draw_logo(f, cx, inner.y + 1, theme);

    let sub = "AMD ROCm · local AI control room";
    put(
        f,
        inner.x + inner.width.saturating_sub(sub.chars().count() as u16) / 2,
        inner.y + 7,
        sub,
        Style::default().fg(theme.muted),
    );

    let items = [("Options", true), ("Help", false), ("Quit", false)];
    let mx = inner.x + inner.width / 2 - 6;
    for (i, (label, focused)) in items.iter().enumerate() {
        let y = inner.y + 9 + i as u16;
        let (cur, c) = if *focused {
            ("▸ ", theme.accent)
        } else {
            ("  ", theme.fg)
        };
        let st = if *focused {
            Style::default().fg(c).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(c)
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(cur, Style::default().fg(theme.accent)),
                Span::styled(*label, st),
            ])),
            Rect::new(mx, y, 16, 1),
        );
    }

    let hint = "↑↓ move   Enter select   Esc close";
    put(
        f,
        inner.x + inner.width.saturating_sub(hint.chars().count() as u16) / 2,
        inner.y + inner.height - 1,
        hint,
        Style::default().fg(theme.muted),
    );
}

// ===========================================================================
// Esc menu → Options (tabbed: General / CPU / GPU / Engines)
// ===========================================================================

/// One settings row: label left, value + control right-aligned.
fn opt_row(
    f: &mut Frame,
    area: Rect,
    label: &str,
    value: &str,
    control: &str,
    focused: bool,
    theme: &Theme,
) {
    let (cur, lc) = if focused {
        ("▸ ", theme.accent)
    } else {
        ("  ", theme.fg)
    };
    let lstyle = if focused {
        Style::default().fg(lc).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(lc)
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(cur, Style::default().fg(theme.accent)),
            Span::styled(label, lstyle),
        ])),
        Rect::new(area.x, area.y, area.width, 1),
    );
    let val_w = (value.chars().count() + control.chars().count() + 2) as u16;
    let vx = area.x + area.width.saturating_sub(val_w);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(value, Style::default().fg(theme.accent)),
            Span::raw("  "),
            Span::styled(control, Style::default().fg(theme.muted)),
        ])),
        Rect::new(vx, area.y, val_w, 1),
    );
}

fn draw_dash_options(f: &mut Frame, theme: &Theme) {
    draw_dash_home(f, theme);
    grey_overlay(f);
    let area = f.area();
    let modal = centered(area, 112, 26);
    fill(f, modal, theme.surface);
    put(
        f,
        modal.x + 2,
        modal.y,
        "⚙  Options",
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );
    let tabarea = Rect::new(modal.x, modal.y + 1, modal.width, modal.height - 1);
    let inner = tab_panel(f, tabarea, &["General", "CPU", "GPU", "Engines"], 0, theme);

    let body = Rect::new(inner.x + 2, inner.y + 1, inner.width - 4, inner.height - 2);
    let rows: &[(&str, &str, &str, bool)] = &[
        ("Theme", "default-dark", "◂ ▸", true),
        ("Start screen", "Home", "◂ ▸", false),
        ("Refresh interval", "2s", "◂ ▸", false),
        ("Confirm before changes", "Ask (recommended)", "◂ ▸", false),
        ("Store telemetry on this PC only", "on", "●", false),
        ("Soft bell on long-job complete", "off", "○", false),
        ("Reduce motion (fewer sparkline redraws)", "off", "○", false),
        ("Show file locations", "", "→", false),
    ];
    for (i, (label, value, control, focused)) in rows.iter().enumerate() {
        opt_row(
            f,
            Rect::new(body.x, body.y + i as u16 * 2, body.width, 1),
            label,
            value,
            control,
            *focused,
            theme,
        );
    }
    put(
        f,
        body.x,
        inner.y + inner.height - 1,
        "↑↓ select    ←→ change    Enter toggle    Esc back",
        Style::default().fg(theme.muted),
    );
}

// ===========================================================================
// Esc menu → Help (two-column keyboard reference)
// ===========================================================================

fn help_section(theme: &Theme, title: &str, rows: &[(&str, &str)]) -> Vec<Line<'static>> {
    let mut out = vec![Line::from(Span::styled(
        title.to_string(),
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    ))];
    for (k, d) in rows {
        out.push(Line::from(vec![
            Span::styled(
                format!(" {k:<10}"),
                Style::default().bg(theme.surface_2).fg(theme.fg),
            ),
            Span::raw("  "),
            Span::styled((*d).to_string(), Style::default().fg(theme.fg)),
        ]));
    }
    out.push(Line::from(""));
    out
}

fn draw_dash_help(f: &mut Frame, theme: &Theme) {
    draw_dash_home(f, theme);
    grey_overlay(f);
    let area = f.area();
    let modal = centered(area, 104, 28);
    fill(f, modal, theme.surface);
    let inner = card(f, modal, "Keyboard shortcuts", theme, theme.accent, false);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .spacing(2)
        .split(inner);

    let mut left = Vec::new();
    left.extend(help_section(
        theme,
        "NAVIGATE",
        &[
            ("Tab", "next tab"),
            ("Shift+Tab", "previous tab"),
            ("1 – 4", "jump to a tab"),
            ("↑ ↓", "move selection"),
            ("← →", "change / cycle value"),
            ("Enter", "open / confirm"),
            ("Esc", "main menu · back"),
            ("PgUp/Dn", "scroll details"),
        ],
    ));
    left.extend(help_section(
        theme,
        "OVERLAYS",
        &[
            ("t", "theme picker"),
            ("?", "this help"),
            (":", "command palette"),
        ],
    ));
    text(f, cols[0], left, theme.surface);

    let mut right = Vec::new();
    right.extend(help_section(
        theme,
        "ACTIONS",
        &[
            ("w", "serve a model"),
            ("e", "engines"),
            ("d", "doctor"),
            ("u", "check updates"),
            ("i", "install"),
            ("l", "logs"),
            ("s", "services / stop"),
            ("F5", "refresh"),
        ],
    ));
    right.extend(help_section(
        theme,
        "CHAT",
        &[
            ("/", "slash commands"),
            ("✦", "plan this"),
            ("Shift+Tab", "approve proposal"),
        ],
    ));
    right.extend(help_section(theme, "GLOBAL", &[("q", "quit")]));
    text(f, cols[1], right, theme.surface);

    put(
        f,
        inner.x,
        inner.y + inner.height - 1,
        "↑↓ scroll    Esc back",
        Style::default().fg(theme.muted),
    );
}

// ===========================================================================
// Surface 3 — WIDE (15″+/27″) layouts: the triptych command center
// ===========================================================================

// Deterministic per-GPU variety (no RNG in workflow-safe code).
fn gpu_series(seed: usize) -> Vec<u64> {
    let base = util_history();
    let off = (seed * 5) % base.len();
    let lvl = 55 + ((seed * 13) % 45) as u64; // 55..99 peak band
    base.iter()
        .cycle()
        .skip(off)
        .take(base.len())
        .map(|&v| (v * lvl / 72).min(100))
        .collect()
}
fn gpu_util(seed: usize) -> u64 {
    45 + ((seed * 17) % 50) as u64
}
fn gpu_vram(seed: usize) -> u64 {
    34 + ((seed * 23) % 60) as u64
}

/// One compact GPU instrument card in the left rail: name, util gauge, util
/// trace, VRAM gauge. 4 content rows.
fn gpu_mini(f: &mut Frame, area: Rect, idx: usize, model: &str, theme: &Theme) {
    let u = gpu_util(idx);
    let v = gpu_vram(idx);
    put(
        f,
        area.x,
        area.y,
        &format!("● GPU{idx}  {model}"),
        Style::default().fg(theme.ok),
    );
    gauge(
        f,
        Rect::new(area.x, area.y + 1, area.width, 1),
        u as f64 / 100.0,
        &format!("{u}%"),
        theme,
    );
    spark(
        f,
        Rect::new(area.x, area.y + 2, area.width, 1),
        &gpu_series(idx),
        theme,
    );
    gauge(
        f,
        Rect::new(area.x, area.y + 3, area.width, 1),
        v as f64 / 100.0,
        &format!("VRAM {v}%"),
        theme,
    );
}

/// The persistent left rail: a wall of GPU instrument cards.
fn gpu_wall(f: &mut Frame, dock: Rect, n: usize, theme: &Theme) {
    let inner = card(f, dock, &format!("GPUs · {n}"), theme, theme.border, false);
    let per = 5u16; // 4 content rows + 1 gap
    let model = "MI300X";
    for i in 0..n {
        let y = inner.y + i as u16 * per;
        if y + 4 > inner.y + inner.height {
            put(
                f,
                inner.x,
                y,
                &format!("… +{} more", n - i),
                Style::default().fg(theme.muted),
            );
            break;
        }
        gpu_mini(f, Rect::new(inner.x, y, inner.width, 4), i, model, theme);
    }
}

/// The persistent right dock: a live assistant column.
fn assistant_dock(f: &mut Frame, dock: Rect, theme: &Theme) {
    let inner = card(f, dock, "Assistant · always on", theme, theme.accent, false);
    let h = inner.height;
    // Next-step header.
    text(
        f,
        Rect::new(inner.x, inner.y, inner.width, 2),
        vec![
            Line::from(vec![
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled(
                    "next: open Chat with Qwen3-72B",
                    Style::default().fg(theme.fg),
                ),
            ]),
            Line::from(Span::styled(
                "─".repeat(inner.width as usize),
                Style::default().fg(theme.border),
            )),
        ],
        theme.surface,
    );
    // Mini transcript.
    text(
        f,
        Rect::new(inner.x, inner.y + 2, inner.width, h.saturating_sub(5)),
        vec![
            Line::from(vec![
                Span::styled(
                    "you  ",
                    Style::default()
                        .fg(theme.accent_2)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("serve qwen", Style::default().fg(theme.fg)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("●    ", Style::default().fg(theme.ok)),
                Span::styled("Starting Qwen3-72B with", Style::default().fg(theme.fg)),
            ]),
            Line::from(Span::styled(
                "     vLLM on :8000. Approve?",
                Style::default().fg(theme.fg),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    " ▸ Approve ",
                    Style::default()
                        .fg(theme.bg)
                        .bg(theme.ok)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  Reject", Style::default().fg(theme.fg)),
            ]),
        ],
        theme.surface,
    );
    // Chips + composer pinned to the dock bottom.
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " ✦ Plan ",
                Style::default()
                    .bg(theme.accent)
                    .fg(theme.bg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            chip("◆ Serve", theme),
            Span::raw(" "),
            chip("⚕ Doctor", theme),
        ])),
        Rect::new(inner.x, inner.y + h - 2, inner.width, 1),
    );
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("> ", Style::default().fg(theme.accent)),
            Span::styled("▌", Style::default().fg(theme.fg)),
        ])),
        Rect::new(inner.x, inner.y + h - 1, inner.width, 1),
    );
}

/// Wide shell: title bar + triptych geometry. Draws the persistent rails'
/// borders are left to callers; returns (left_dock, center_inner, right_dock).
/// The outlined tabs are drawn over the center column only.
fn wide_shell(f: &mut Frame, theme: &Theme, active: usize) -> (Rect, Rect, Rect) {
    let area = f.area();
    fill(f, area, theme.bg);
    put(
        f,
        1,
        0,
        "rocm.ai",
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    );
    let status = "connected · node-mi300x-8 · rocm daemon 0.9";
    put(f, 10, 0, status, Style::default().fg(theme.ok));
    let hint = "Esc  menu    t  theme    ?  help";
    put(
        f,
        area.width - hint.chars().count() as u16 - 1,
        0,
        hint,
        Style::default().fg(theme.muted),
    );

    let lw = 40u16;
    let rw = 52u16;
    let body_y = 1u16;
    let body_h = area.height - 2; // leave footer row
    // Rails align with the center *content panel* (which starts 2 rows below the
    // tab tops), so the tab band reads as belonging to the center only.
    let rail_y = body_y + 2;
    let rail_h = body_h - 2;
    let left = Rect::new(1, rail_y, lw, rail_h);
    let right = Rect::new(area.width - rw - 1, rail_y, rw, rail_h);
    let center_x = 1 + lw + 1;
    let center_w = right.x - center_x - 1;
    let center_outer = Rect::new(center_x, body_y, center_w, body_h);
    let center_inner = tab_panel(
        f,
        center_outer,
        &["Home", "Action", "Observe", "Chat"],
        active,
        theme,
    );
    (left, center_inner, right)
}

fn draw_wide_home(f: &mut Frame, theme: &Theme) {
    let (left, center, right) = wide_shell(f, theme, 0);
    let area = f.area();
    gpu_wall(f, left, 4, theme);
    assistant_dock(f, right, theme);

    // Center: wide tokens/watt hero strip, a status bento, then an activity feed
    // that uses the vertical room a big screen gives us.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(11),
            Constraint::Min(0),
        ])
        .margin(1)
        .split(center);

    let hero = card(
        f,
        rows[0],
        "Node throughput · 8 × MI300X · ROCm 6.2",
        theme,
        theme.border,
        true,
    );
    let hl = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .spacing(2)
        .split(hero);
    text(
        f,
        hl[0],
        vec![
            Line::from(vec![
                Span::styled(
                    "⎓ 192.4 ",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("tokens / watt (node)", Style::default().fg(theme.muted)),
            ]),
            Line::from(vec![
                Span::styled(
                    "Σ 2,180 t/s",
                    Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                ),
                Span::styled("   3 models live", Style::default().fg(theme.muted)),
            ]),
        ],
        theme.surface,
    );
    mini_spark(
        f,
        Rect::new(hl[1].x, hl[1].y, hl[1].width, 1),
        "tok/W",
        "192",
        &tpw_history(),
        true,
        theme,
    );
    mini_spark(
        f,
        Rect::new(hl[1].x, hl[1].y + 1, hl[1].width, 1),
        "Σt/s ",
        "2.2k",
        &gtps_history(),
        true,
        theme,
    );
    mini_spark(
        f,
        Rect::new(hl[1].x, hl[1].y + 2, hl[1].width, 1),
        "POWER",
        "1.1kW",
        &power_history(),
        false,
        theme,
    );

    let bento = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .spacing(1)
        .split(rows[1]);

    let running = card(f, bento[0], "Running · 3", theme, theme.border, false);
    text(
        f,
        running,
        vec![
            Line::from(vec![
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled("Qwen3-72B   :8000", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled("Llama-3.3   :8001", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled("DeepSeek-R1 :8002", Style::default().fg(theme.fg)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("▸ ", Style::default().fg(theme.accent)),
                Span::styled("Serve another →", Style::default().fg(theme.fg)),
            ]),
        ],
        theme.surface,
    );
    let health = card(f, bento[1], "Health", theme, theme.border, false);
    text(
        f,
        health,
        vec![
            Line::from(vec![
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled("8 GPUs · gfx942", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled("Driver · ROCm 6.2", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled("Fabric · XGMI", Style::default().fg(theme.fg)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("▸ ", Style::default().fg(theme.accent)),
                Span::styled("Run doctor →", Style::default().fg(theme.fg)),
            ]),
        ],
        theme.surface,
    );
    let updates = card(f, bento[2], "Updates", theme, theme.warn, false);
    text(
        f,
        updates,
        vec![
            Line::from(vec![
                Span::styled("⇲ ", Style::default().fg(theme.warn)),
                Span::styled(
                    "ROCm 6.3 ready",
                    Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                "  approval required",
                Style::default().fg(theme.muted),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("▸ ", Style::default().fg(theme.accent)),
                Span::styled("View update →", Style::default().fg(theme.fg)),
            ]),
        ],
        theme.surface,
    );

    // Activity feed — fills the tall center on a big screen, with a node-load
    // trace so the panel reads as live.
    let activity = card(f, rows[2], "Activity · node", theme, theme.border, false);
    mini_spark(
        f,
        Rect::new(activity.x, activity.y, activity.width, 1),
        "node load ",
        "live",
        &util_history(),
        false,
        theme,
    );
    text(
        f,
        Rect::new(
            activity.x,
            activity.y + 2,
            activity.width,
            activity.height.saturating_sub(2),
        ),
        vec![
            Line::from(vec![
                Span::styled("12:04  ", Style::default().fg(theme.muted)),
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled(
                    "DeepSeek-R1-32B serving on :8002 (GPU 4)",
                    Style::default().fg(theme.fg),
                ),
            ]),
            Line::from(vec![
                Span::styled("12:01  ", Style::default().fg(theme.muted)),
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled(
                    "Benchmark tg128 complete · 41.8 gen t/s",
                    Style::default().fg(theme.fg),
                ),
            ]),
            Line::from(vec![
                Span::styled("11:58  ", Style::default().fg(theme.muted)),
                Span::styled("⇲ ", Style::default().fg(theme.warn)),
                Span::styled(
                    "ROCm 6.3 upgrade proposal awaiting approval",
                    Style::default().fg(theme.fg),
                ),
            ]),
            Line::from(vec![
                Span::styled("11:42  ", Style::default().fg(theme.muted)),
                Span::styled("● ", Style::default().fg(theme.ok)),
                Span::styled(
                    "Llama-3.3-70B serving on :8001 (GPU 2-3)",
                    Style::default().fg(theme.fg),
                ),
            ]),
        ],
        theme.surface,
    );

    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("Tab", "tabs"),
            ("↑↓←→", "move"),
            ("Enter", "open"),
            ("t", "theme"),
            ("?", "help"),
            ("q", "quit"),
        ],
        theme,
    );
}

fn draw_wide_observe(f: &mut Frame, theme: &Theme) {
    let (left, center, right) = wide_shell(f, theme, 2);
    let area = f.area();
    gpu_wall(f, left, 8, theme); // the wall scales to the whole node
    assistant_dock(f, right, theme);
    wide_observe_center(f, center, theme);
    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("Tab", "tabs"),
            ("↑↓", "select"),
            ("s", "stop"),
            ("l", "logs"),
            ("F5", "refresh"),
            ("t", "theme"),
            ("?", "help"),
        ],
        theme,
    );
}

/// Observe variant: the right dock is **contextual** — a live logs stream for
/// the selected service instead of the assistant. Same center, different dock.
fn draw_wide_observe_logs(f: &mut Frame, theme: &Theme) {
    let (left, center, right) = wide_shell(f, theme, 2);
    let area = f.area();
    gpu_wall(f, left, 8, theme);
    logs_dock(f, right, theme);
    wide_observe_center(f, center, theme);
    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("Tab", "tabs"),
            ("↑↓", "select"),
            ("s", "stop"),
            ("l", "logs"),
            ("F5", "refresh"),
            ("t", "theme"),
            ("?", "help"),
        ],
        theme,
    );
}

fn wide_observe_center(f: &mut Frame, center: Rect, theme: &Theme) {
    // Center: instances master (top) + selected-instance detail (bottom).
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Min(0)])
        .margin(1)
        .split(center);

    let insts = card(
        f,
        rows[0],
        "Instances · 3   (per-process attribution)",
        theme,
        theme.border,
        false,
    );
    let header = format!(
        "{:<3}{:<14}{:<14}{:>6}{:>7}{:>6}",
        "", "name", "model", "port", "t/s", "GPU"
    );
    let mut lines = vec![Line::from(Span::styled(
        header,
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    ))];
    let data = [
        ("qwen-72b", "Qwen3-72B", "8000", "42", "0-1", true),
        ("llama-33", "Llama-3.3-70B", "8001", "38", "2-3", false),
        ("deepseek", "DeepSeek-R1-32B", "8002", "57", "4", false),
    ];
    for (name, model, port, tps, gpu, sel) in data {
        let cur = if sel { "▸ " } else { "  " };
        let st = if sel {
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.fg)
        };
        lines.push(Line::from(vec![
            Span::styled(cur, Style::default().fg(theme.accent)),
            Span::styled("● ", Style::default().fg(theme.ok)),
            Span::styled(
                format!("{name:<13}{model:<14}{port:>6}{tps:>7}{gpu:>6}"),
                st,
            ),
        ]));
    }
    text(f, insts, lines, theme.surface);

    let detail = card(
        f,
        rows[1],
        "Selected · qwen-72b · Qwen3-72B",
        theme,
        theme.accent,
        false,
    );
    let dl = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .spacing(2)
        .split(detail);
    // Left of detail: live throughput + util traces (deep history).
    let dll = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1); 6])
        .split(dl[0]);
    put(
        f,
        dll[0].x,
        dll[0].y,
        "LIVE · last 60s",
        Style::default().fg(theme.muted),
    );
    mini_spark(f, dll[1], "gen t/s", "42", &tps_history(), true, theme);
    mini_spark(f, dll[2], "prmpt  ", "3.1k", &gtps_history(), true, theme);
    mini_spark(f, dll[3], "UTIL   ", "62%", &util_history(), false, theme);
    mini_spark(f, dll[4], "VRAM   ", "64%", &vram_history(), false, theme);
    mini_spark(f, dll[5], "KV$    ", "71%", &power_history(), false, theme);
    // Right of detail: config + actions, all inline (no overlay needed).
    text(
        f,
        dl[1],
        vec![
            Line::from(vec![
                Span::styled("engine  ", Style::default().fg(theme.muted)),
                Span::styled("vLLM 0.6", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("TP / GPUs ", Style::default().fg(theme.muted)),
                Span::styled("2 · GPU 0-1", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("ctx len ", Style::default().fg(theme.muted)),
                Span::styled("32,768", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("uptime  ", Style::default().fg(theme.muted)),
                Span::styled("2h 15m", Style::default().fg(theme.fg)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    "▸ Stop",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Restart   Logs   Bench", Style::default().fg(theme.fg)),
            ]),
        ],
        theme.surface,
    );
}

/// Contextual right dock: a live log stream for the selected service.
fn logs_dock(f: &mut Frame, dock: Rect, theme: &Theme) {
    let inner = card(
        f,
        dock,
        "Logs · qwen-72b (live)",
        theme,
        theme.border,
        false,
    );
    let rows: &[(&str, &str, Color)] = &[
        (
            "12:06:02",
            "GET /v1/chat/completions 200 · 412ms",
            theme.muted,
        ),
        ("12:06:01", "decode: 42.1 tok/s · batch 6", theme.fg),
        ("12:05:58", "KV cache 71% · 0 evictions", theme.fg),
        ("12:05:55", "GET /v1/models 200", theme.muted),
        ("12:05:49", "decode: 41.6 tok/s · batch 5", theme.fg),
        (
            "12:04:33",
            "WARN first scrape: throughput warming up",
            theme.warn,
        ),
        ("12:04:31", "engine ready on 127.0.0.1:8000", theme.ok),
        ("12:04:30", "KV cache allocated · 32768 ctx", theme.fg),
        ("12:04:25", "weights loaded (cached) · 72B", theme.fg),
        ("12:04:11", "starting vLLM · TP 2 · GPU 0-1", theme.muted),
    ];
    let mut lines = Vec::new();
    for (ts, msg, c) in rows {
        lines.push(Line::from(vec![
            Span::styled(format!("{ts} "), Style::default().fg(theme.muted)),
            Span::styled(*msg, Style::default().fg(*c)),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "▸ follow  ·  PgUp scroll  ·  / filter",
        Style::default().fg(theme.muted),
    )));
    text(f, inner, lines, theme.surface);
}

/// Contextual right dock for Chat: what the agent can see (live GPU/services
/// state, recent tools, active skills) — the 1-pager's MCP-grounded assistant.
fn context_rail(f: &mut Frame, dock: Rect, theme: &Theme) {
    let inner = card(
        f,
        dock,
        "Context · what the agent sees",
        theme,
        theme.border,
        false,
    );
    let mut lines = vec![
        Line::from(Span::styled(
            "RUNNING SERVICES",
            Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("● ", Style::default().fg(theme.ok)),
            Span::styled("Qwen3-72B   :8000  42 t/s", Style::default().fg(theme.fg)),
        ]),
        Line::from(vec![
            Span::styled("● ", Style::default().fg(theme.ok)),
            Span::styled("Llama-3.3   :8001  38 t/s", Style::default().fg(theme.fg)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "GPU STATE",
            Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD),
        )),
    ];
    text(
        f,
        Rect::new(inner.x, inner.y, inner.width, 5),
        lines.drain(..).collect(),
        theme.surface,
    );
    mini_spark(
        f,
        Rect::new(inner.x, inner.y + 5, inner.width, 1),
        "avg util",
        "62%",
        &util_history(),
        false,
        theme,
    );
    mini_spark(
        f,
        Rect::new(inner.x, inner.y + 6, inner.width, 1),
        "VRAM    ",
        "71%",
        &vram_history(),
        false,
        theme,
    );
    text(
        f,
        Rect::new(
            inner.x,
            inner.y + 8,
            inner.width,
            inner.height.saturating_sub(8),
        ),
        vec![
            Line::from(vec![
                Span::styled("8× MI300X · gfx942 · ", Style::default().fg(theme.fg)),
                Span::styled("51°C", Style::default().fg(theme.fg)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "RECENT TOOLS",
                Style::default()
                    .fg(theme.muted)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled("serve(qwen3-72b)", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("✓ ", Style::default().fg(theme.ok)),
                Span::styled("doctor()", Style::default().fg(theme.fg)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "ACTIVE SKILLS",
                Style::default()
                    .fg(theme.muted)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "instinct-llm-serving",
                Style::default().fg(theme.accent),
            )),
            Line::from(Span::styled("hyperloom", Style::default().fg(theme.accent))),
        ],
        theme.surface,
    );
}

/// Chat tab on a wide screen: the conversation expands to fill the center, with
/// the GPU wall still on the left and a context rail (not the assistant dock,
/// since chat *is* the center now) on the right.
fn draw_wide_chat(f: &mut Frame, theme: &Theme) {
    let (left, center, right) = wide_shell(f, theme, 3);
    let area = f.area();
    gpu_wall(f, left, 4, theme);
    context_rail(f, right, theme);

    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .margin(1)
        .split(center);

    let convo = body[0];
    text(
        f,
        convo,
        vec![
            Line::from(vec![
                Span::styled(
                    "you   ",
                    Style::default()
                        .fg(theme.accent_2)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "set up rocm and serve qwen3-72b so I can chat with it",
                    Style::default().fg(theme.fg),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("●     ", Style::default().fg(theme.ok)),
                Span::styled(
                    "Plan: 3 steps. I'll check the GPU, serve the model, then open",
                    Style::default().fg(theme.fg),
                ),
            ]),
            Line::from(Span::styled(
                "      chat. Each mutating step asks for approval first.",
                Style::default().fg(theme.fg),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("      1. ", Style::default().fg(theme.muted)),
                Span::styled(
                    "✓ check GPU & ROCm  (doctor)",
                    Style::default().fg(theme.ok),
                ),
            ]),
            Line::from(vec![
                Span::styled("      2. ", Style::default().fg(theme.muted)),
                Span::styled(
                    "● serve Qwen3-72B · vLLM · :8000  (starting…)",
                    Style::default().fg(theme.warn),
                ),
            ]),
            Line::from(vec![
                Span::styled("      3. ", Style::default().fg(theme.muted)),
                Span::styled(
                    "◌ open chat with the model",
                    Style::default().fg(theme.muted),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("●     ", Style::default().fg(theme.ok)),
                Span::styled("Live so far:", Style::default().fg(theme.fg)),
            ]),
        ],
        theme.bg,
    );
    // A live throughput trace inside the conversation.
    mini_spark(
        f,
        Rect::new(convo.x + 6, convo.y + 11, 48, 1),
        "gen t/s ",
        "42",
        &tps_history(),
        true,
        theme,
    );

    // Action-chip row, composer.
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " ✦ Plan this ",
                Style::default()
                    .bg(theme.accent)
                    .fg(theme.bg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            chip("◆ Serve", theme),
            Span::raw("  "),
            chip("⚕ Doctor", theme),
            Span::styled(
                "        provider: claude ",
                Style::default().fg(theme.muted),
            ),
            Span::styled("●", Style::default().fg(theme.ok)),
        ])),
        Rect::new(center.x + 1, body[1].y, center.width - 2, 1),
    );
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "─".repeat(center.width as usize - 2),
            Style::default().fg(theme.border),
        ))),
        Rect::new(center.x + 1, body[2].y, center.width - 2, 1),
    );
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("> ", Style::default().fg(theme.accent)),
            Span::styled("▌", Style::default().fg(theme.fg)),
        ])),
        Rect::new(center.x + 1, body[3].y, center.width - 2, 1),
    );

    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("Enter", "send"),
            ("/", "commands"),
            ("✦", "plan"),
            ("Shift+Tab", "approve"),
            ("t", "theme"),
            ("?", "help"),
        ],
        theme,
    );
}

fn draw_wide_action(f: &mut Frame, theme: &Theme) {
    let (left, center, right) = wide_shell(f, theme, 1);
    let area = f.area();
    gpu_wall(f, left, 4, theme);
    assistant_dock(f, right, theme);

    // Center: actions list AND the serve wizard, side by side (no Enter needed).
    let split = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(0)])
        .spacing(1)
        .margin(1)
        .split(center);

    let list = card(f, split[0], "Actions", theme, theme.border, false);
    // (icon, label, focused, soon)
    let actions = [
        ("◆", "Serve a model", true, false),
        ("⚙", "Set up / Install", false, false),
        ("⌬", "Engines", false, false),
        ("⚕", "Diagnose & fix", false, false),
        ("⇲", "Check updates", false, false),
        ("⮌", "Providers & keys", false, false),
        ("⚡", "Optimize", false, true),
    ];
    let mut lines = Vec::new();
    for (icon, label, sel, soon) in actions {
        let cur = if sel { "▸ " } else { "  " };
        let lc = if soon { theme.muted } else { theme.fg };
        let st = if sel {
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(lc)
        };
        let mut row = vec![
            Span::styled(cur, Style::default().fg(theme.accent)),
            Span::styled(
                format!("{icon}  "),
                Style::default().fg(if soon { theme.muted } else { theme.accent_2 }),
            ),
            Span::styled(label, st),
        ];
        if soon {
            row.push(Span::raw("  "));
            row.push(Span::styled(
                " soon ",
                Style::default()
                    .bg(theme.warn)
                    .fg(theme.bg)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::from(row));
        lines.push(Line::from(""));
    }
    lines.push(Line::from(Span::styled(
        "soon = planned, not yet built",
        Style::default().fg(theme.muted),
    )));
    text(f, list, lines, theme.surface);

    // The serve wizard, fully expanded — model list + fit + engine + advanced
    // all visible at once (what the compact layout reveals step by step).
    let wiz = card(
        f,
        split[1],
        "◆ Serve a model — guided",
        theme,
        theme.accent,
        false,
    );
    let wl = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .spacing(2)
        .split(wiz);
    let mut models = vec![Line::from(Span::styled(
        "Choose a model (fits 8 × 192 GB):",
        Style::default().fg(theme.muted),
    ))];
    for (m, note, sel) in [
        ("Qwen3-72B", "✓ fits · recommended", true),
        ("Llama-3.3-70B", "✓ fits", false),
        ("DeepSeek-R1-671B", "✓ fits · TP 8", false),
        ("Mixtral-8x22B", "✓ fits", false),
        ("Browse all models  →", "", false),
    ] {
        let cur = if sel { "▸ " } else { "  " };
        let st = if sel {
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.fg)
        };
        models.push(Line::from(vec![
            Span::styled(cur, Style::default().fg(theme.accent)),
            Span::styled(format!("{m:<22}"), st),
            Span::styled(note, Style::default().fg(theme.ok)),
        ]));
    }
    text(f, wl[0], models, theme.surface);
    text(
        f,
        wl[1],
        vec![
            Line::from(vec![
                Span::styled("Engine   ", Style::default().fg(theme.muted)),
                Span::styled("vLLM (auto)", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("Tensor-parallel ", Style::default().fg(theme.muted)),
                Span::styled("8", Style::default().fg(theme.fg)),
            ]),
            Line::from(vec![
                Span::styled("Port     ", Style::default().fg(theme.muted)),
                Span::styled("8003 (loopback)", Style::default().fg(theme.fg)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Advanced ▾  quantization · KV dtype · ctx",
                Style::default().fg(theme.muted),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    "  ▸ Start serving ",
                    Style::default()
                        .fg(theme.bg)
                        .bg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  Enter", Style::default().fg(theme.muted)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Asks before it runs. Default mode: ask.",
                Style::default().fg(theme.muted),
            )),
        ],
        theme.surface,
    );

    footer(
        f,
        area.height - 1,
        area.width,
        &[
            ("Tab", "tabs"),
            ("↑↓", "choose"),
            ("Enter", "start"),
            ("a", "advanced"),
            ("t", "theme"),
            ("?", "help"),
        ],
        theme,
    );
}

// ===========================================================================
// Buffer → SVG (self-contained; mirrors examples/gen_screenshots.rs)
// ===========================================================================

const CELL_W: f64 = 8.4;
const CELL_H: f64 = 17.0;
const FONT_PX: f64 = 14.0;
const FONT_FAMILY: &str = "ui-monospace, 'JetBrains Mono', 'Cascadia Code', \
                          'Fira Code', Menlo, 'DejaVu Sans Mono', monospace";

fn buffer_to_svg(buf: &Buffer, default_bg: Color) -> String {
    let cols = buf.area.width;
    let rows = buf.area.height;
    let w_px = f64::from(cols) * CELL_W;
    let h_px = f64::from(rows) * CELL_H;
    let bg_hex = color_to_hex(default_bg).unwrap_or_else(|| "#131416".into());

    let mut out = String::with_capacity(rows as usize * cols as usize * 16);
    let _ = write!(
        out,
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
    );
    let _ = write!(out, r#"<rect width="100%" height="100%" fill="{bg_hex}"/>"#);

    for y in 0..rows {
        let mut x = 0u16;
        while x < cols {
            let cell = buf.cell((x, y)).unwrap();
            let bg = cell.style().bg.unwrap_or(Color::Reset);
            if bg == default_bg || matches!(bg, Color::Reset) {
                x += 1;
                continue;
            }
            let Some(bg_hex_run) = color_to_hex(bg) else {
                x += 1;
                continue;
            };
            let start = x;
            let mut end = x + 1;
            while end < cols
                && buf
                    .cell((end, y))
                    .unwrap()
                    .style()
                    .bg
                    .unwrap_or(Color::Reset)
                    == bg
            {
                end += 1;
            }
            let rx = f64::from(start) * CELL_W;
            let ry = f64::from(y) * CELL_H;
            let rw = f64::from(end - start) * CELL_W;
            let _ = write!(
                out,
                r#"<rect x="{rx:.2}" y="{ry:.2}" width="{rw:.2}" height="{CELL_H}" fill="{bg_hex_run}"/>"#
            );
            x = end;
        }
    }

    for y in 0..rows {
        let mut x = 0u16;
        let baseline = FONT_PX.mul_add(0.85, f64::from(y) * CELL_H);
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
            let mut tbuf = xml_escape(sym);
            let mut end = x + 1;
            while end < cols {
                let next = buf.cell((end, y)).unwrap();
                let nsym = next.symbol();
                if nsym == " " || nsym.is_empty() {
                    break;
                }
                if next.style().fg.and_then(color_to_hex) != fg
                    || next.style().add_modifier.contains(Modifier::BOLD) != bold
                {
                    break;
                }
                tbuf.push_str(&xml_escape(nsym));
                end += 1;
            }
            let tx = f64::from(start) * CELL_W;
            let fg_hex = fg.unwrap_or_else(|| "#eaebec".into());
            let weight = if bold { "700" } else { "400" };
            let span_w = f64::from(end - start) * CELL_W;
            let _ = write!(
                out,
                concat!(
                    r#"<text x="{tx:.2}" y="{baseline:.2}" fill="{fg_hex}" "#,
                    r#"font-weight="{weight}" textLength="{span_w:.2}" "#,
                    r#"lengthAdjust="spacingAndGlyphs">{text}</text>"#,
                ),
                tx = tx,
                baseline = baseline,
                fg_hex = fg_hex,
                weight = weight,
                span_w = span_w,
                text = tbuf,
            );
            x = end;
        }
    }

    out.push_str("</svg>");
    out
}

fn color_to_hex(c: Color) -> Option<String> {
    match c {
        Color::Rgb(r, g, b) => Some(format!("#{r:02x}{g:02x}{b:02x}")),
        Color::Reset | Color::Indexed(_) => None,
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

fn build_gallery(items: &[(String, &str)]) -> String {
    let mut out = String::new();
    out.push_str(
        "<!doctype html><meta charset=utf-8><title>rocm-cli UX mockups</title>\
         <style>body{background:#0d0e10;color:#eaebec;font:15px/1.5 ui-sans-serif,system-ui,sans-serif;\
         margin:0;padding:32px}h1{font-weight:700}h2{margin:2.5rem 0 .5rem;font-size:1rem;color:#00c2de}\
         img{width:100%;max-width:1200px;border:1px solid #2a2d31;border-radius:8px;display:block}\
         .hint{color:#b4b9bc;margin-bottom:2rem}</style>",
    );
    out.push_str("<h1>rocm-cli — proposed UX mockups</h1>");
    out.push_str(
        "<p class=hint>Rendered from <code>examples/gen_mockups.rs</code> via a real ratatui \
         framebuffer. See <code>docs/design/</code> for the annotated rationale.</p>",
    );
    for (file, title) in items {
        let _ = write!(
            out,
            "<h2>{}</h2><img src=\"{}\" alt=\"{}\">",
            title, file, title
        );
    }
    out
}
