// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The instance Detail modal: a summary strip, a "recent throughput" panel,
//! scrollable launch-args and env-vars panes, and a footer, drawn over a dimmed backdrop.
//!
//! Split out of `ui/tabs/instances.rs`. It reads `sorted_instances`,
//! `clamp_sel` and `status_meta` from there, which stay single-sourced beside
//! the table code that also uses them; nothing in `instances.rs` calls
//! back into this module. `ui/mod.rs` calls `draw_detail` for `Modal::Detail`
//! on the Observe tab and writes the returned max scroll offset to
//! `AppState::instance_detail_max_scroll`.

use chrono::{DateTime, Utc};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Row, Table, Wrap};

use rocm_dash_core::metrics::Instance;

use crate::app::{AppState, ScrollTarget};
use crate::ui::format;
use crate::ui::panel::{self, BoxRole, centered_rect, draw_popup_frame, grey_overlay};
use crate::ui::tabs::instances::{clamp_sel, sorted_instances, status_meta};
use crate::ui::theme::Theme;
use crate::ui::widgets::trunc;

/// Draws the instance Detail modal and returns the max scroll offset for its
/// body (see `render_body`), so the caller can write it back to
/// `AppState::instance_detail_max_scroll`.
pub(super) fn draw_detail(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) -> u16 {
    grey_overlay(f);
    let popup = centered_rect(85, 85, 120, 36, area);

    if state.instances.is_empty() {
        let inner = draw_popup_frame(f, popup, " Instance · (no selection) ", theme);
        let p = Paragraph::new(Line::from(Span::styled(
            "no instances to show",
            Style::default().fg(theme.muted),
        )));
        f.render_widget(p, inner);
        return 0;
    }

    let instances = sorted_instances(&state.instances);
    let sel = clamp_sel(state.instance_sel, instances.len());
    let Some(inst) = instances.get(sel) else {
        let inner = draw_popup_frame(f, popup, " Instance · (no selection) ", theme);
        let p = Paragraph::new(Line::from(Span::styled(
            "no selection",
            Style::default().fg(theme.muted),
        )));
        f.render_widget(p, inner);
        return 0;
    };

    let title = format!(" Instance · {} ", inst.container_name);
    let inner = draw_popup_frame(f, popup, &title, theme);
    if inner.height == 0 || inner.width == 0 {
        return 0;
    }

    // Vertical: summary (4 lines: status/id/port/tp · model/gpus/tpw/gen · partition/quant/vram · freshness)
    // | recent throughput (recent non-idle ticks, ROCMAI-196) | body (min) | footer (1)
    let snap_ts = state.latest.as_ref().map(|s| s.timestamp);
    let activity = recent_activity_rows(&state.history, &inst.container_id, RECENT_ACTIVITY_CAP);
    let has_observed_throughput = has_observed_throughput(&state.history, &inst.container_id);
    let (activity_h, visible_rows) = activity_layout(activity.len(), inner.height);
    let total_rows = activity.len();
    let activity = &activity[..visible_rows];
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(activity_h),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);

    render_summary(f, chunks[0], inst, snap_ts, theme);
    render_recent_activity(
        f,
        chunks[1],
        activity,
        total_rows,
        snap_ts,
        has_observed_throughput,
        theme,
    );
    let max_scroll = render_body(f, chunks[2], inst, state, theme);
    render_footer(f, chunks[3], inst, theme, max_scroll > 0);
    max_scroll
}

/// Max rows shown in the detail modal's "recent throughput" summary.
const RECENT_ACTIVITY_CAP: usize = 5;

/// Fewest rows the detail body keeps below the recent-activity panel: it holds
/// the launch args / env panes, the modal's main content, so the panel must not
/// squeeze it to a line or two (and its scrollbar needs a border pair plus a
/// content row to render at all).
const BODY_FLOOR_MIN: u16 = 6;

/// Rows reserved for the body in a modal `inner_h` rows tall: at least
/// `BODY_FLOOR_MIN`, and half the modal once it is tall enough.
fn body_floor(inner_h: u16) -> u16 {
    (inner_h / 2).max(BODY_FLOOR_MIN)
}

/// Sizes the "recent throughput" panel for `n_rows` entries in a modal body
/// `inner_h` rows tall; returns `(panel height, rows that fit)`.
///
/// The panel is capped so the body keeps its border pair plus a content row
/// (`body_floor`) — otherwise the launch args / env panes are starved and,
/// at 0 rows, the body's scrollbar stops rendering. The row count is derived from the same
/// budget, so a capped panel shows fewer (newest) rows instead of being
/// handed rows it can't draw. A panel too short for its header plus one row
/// (or the placeholder line) collapses to `(0, 0)` so the body gets the rows.
fn activity_layout(n_rows: usize, inner_h: u16) -> (u16, usize) {
    let avail = inner_h.saturating_sub(4 + 1 + body_floor(inner_h));
    // 2 border rows + (1 header row + 1 per entry), or 1 placeholder row when
    // there's none yet; `panel::bento` adds a top-padding row once the box is
    // tall enough (`padding_for`), which is reused here rather than re-derived.
    let height_for = |content: u16| {
        let raw = 2 + content;
        raw + panel::padding_for(Rect::new(0, 0, 1, raw)).top
    };
    if n_rows == 0 {
        let h = height_for(1);
        return if h <= avail { (h, 0) } else { (0, 0) };
    }
    (1..=n_rows)
        .rev()
        .map(|k| (height_for(1 + u16::try_from(k).unwrap_or(u16::MAX)), k))
        .find(|&(h, _)| h <= avail)
        .unwrap_or((0, 0))
}

/// One historical tick where an instance was actually generating tokens.
pub(crate) struct ActivityRow {
    pub(crate) ts: DateTime<Utc>,
    pub(crate) gen_tps: Option<f64>,
    pub(crate) gen_tps_observation: Option<rocm_dash_core::metrics::ObservationMetadata>,
    pub(crate) ttft_ms: Option<f64>,
    pub(crate) tpot_ms: Option<f64>,
}

/// Scan `history` (oldest-to-newest) for the most recent snapshot ticks where
/// `container_id` had non-zero generation throughput — i.e. was actually
/// serving, not idle. These are per-tick samples, not discrete serving runs.
/// The live "gen:" gauge in `render_summary` already covers the idle case, so
/// idle ticks are skipped. Repeated Held ticks of one scrape collapse to one row. Returns up to `cap` rows, newest first.
/// Pure — exposed for tests.
pub(crate) fn recent_activity_rows(
    history: &std::collections::VecDeque<rocm_dash_core::metrics::Snapshot>,
    container_id: &str,
    cap: usize,
) -> Vec<ActivityRow> {
    let mut rows: Vec<ActivityRow> = Vec::new();
    for snap in history.iter().rev() {
        let Some(inst) = snap
            .instances
            .iter()
            .find(|i| i.container_id == container_id)
        else {
            continue;
        };
        if !inst.gen_tps.is_some_and(|v| v.is_finite() && v > 0.0) {
            continue;
        }
        // A Held tick repeats an earlier scrape; age it from when the
        // value was observed, not from the tick that re-published it.
        let ts = inst
            .gen_tps_observation
            .as_ref()
            .map_or(snap.timestamp, |o| o.observed_at);
        let row = ActivityRow {
            ts,
            gen_tps: inst.gen_tps,
            gen_tps_observation: inst.gen_tps_observation.clone(),
            ttft_ms: inst.ttft_ms,
            tpot_ms: inst.tpot_ms,
        };
        // Ticks sharing one `observed_at` are one scrape. Walking newest
        // first, a later tick overwrites the group's row, so the group ends
        // up represented by its oldest tick: the scrape's own (Fresh) tick,
        // which carries that scrape's latencies, not a Held re-publication.
        if let Some(last) = rows.last_mut().filter(|l| l.ts == ts) {
            *last = row;
        } else if rows.len() >= cap {
            break;
        } else {
            rows.push(row);
        }
    }
    rows
}

/// Whether `history` has ever recorded a real throughput sample — even a
/// genuine `0.0` — for `container_id`. `recent_activity_rows` only returns
/// non-zero ticks, so an empty result alone can't tell "has been idle" (real
/// `Some(0.0)` samples seen) apart from "no throughput data" (too new, past
/// the observation's validity window, a managed non-vLLM engine that never
/// gets `gen_tps` wired up, or history freshly cleared by a replay seek).
/// Pure — exposed for tests.
pub(crate) fn has_observed_throughput(
    history: &std::collections::VecDeque<rocm_dash_core::metrics::Snapshot>,
    container_id: &str,
) -> bool {
    history.iter().any(|snap| {
        snap.instances
            .iter()
            .any(|i| i.container_id == container_id && i.gen_tps.is_some_and(f64::is_finite))
    })
}

/// Renders the "recent throughput" panel: the last few snapshot ticks where
/// the instance recorded non-zero throughput, each with tok/s + TTFT/TPOT at
/// that time (partial coverage of ROCMAI-196; no token counts). Named
/// distinctly from the Home tab's "Activity · node" panel (jobs/actions feed)
/// so the two aren't mistaken for the same concept.
fn render_recent_activity(
    f: &mut Frame,
    area: Rect,
    rows: &[ActivityRow],
    total_rows: usize,
    snap_ts: Option<DateTime<Utc>>,
    has_observed_throughput: bool,
    theme: &Theme,
) {
    // Say so when a short terminal hides older rows instead of clipping silently.
    // No "of N": `total_rows` is already capped, so it would understate the history.
    let mut title = if rows.len() < total_rows {
        format!("recent throughput (newest {})", rows.len())
    } else {
        "recent throughput".to_string()
    };
    // Explain the `*` the same way the table does, but only when a shown row has one.
    if rows.iter().any(|r| {
        r.gen_tps_observation
            .as_ref()
            .is_some_and(|m| m.freshness == rocm_dash_core::metrics::ObservationFreshness::Held)
    }) {
        title.push_str(" · ");
        title.push_str(format::HELD_LEGEND);
    }
    let inner = panel::bento(f, area, Some(&title), BoxRole::Secondary, false, theme);
    if inner.height == 0 {
        return;
    }
    if rows.is_empty() {
        let msg = if has_observed_throughput {
            "no recent throughput — instance has been idle".to_string()
        } else {
            "no throughput samples".to_string()
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                msg,
                Style::default().fg(theme.muted),
            ))),
            inner,
        );
        return;
    }

    let header = Row::new(["TIME", "TOK/S", "TTFT", "TPOT"]).style(
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    );
    let table_rows = rows.iter().map(|r| {
        // No snapshot (dashboard disconnected): the age is unknowable, so
        // show a placeholder rather than a fabricated "0s ago".
        let time = snap_ts.map_or_else(
            || "—".to_string(),
            |ts| format!("{}s ago", (ts - r.ts).num_seconds().max(0)),
        );
        let tps = format::gen_tps_cell(r.gen_tps, r.gen_tps_observation.as_ref());
        let ttft = format::ms_opt(r.ttft_ms);
        let tpot = format::ms_opt(r.tpot_ms);
        Row::new([time, tps, ttft, tpot]).style(Style::default().fg(theme.fg))
    });
    let widths = [
        Constraint::Length(10),
        Constraint::Length(14),
        Constraint::Length(7),
        Constraint::Length(7),
    ];
    let table = Table::new(table_rows, widths)
        .header(header)
        .column_spacing(1);
    f.render_widget(table, inner);
}

fn render_summary(
    f: &mut Frame,
    area: Rect,
    inst: &Instance,
    snap_ts: Option<DateTime<Utc>>,
    theme: &Theme,
) {
    let (status_color, status_text) = status_meta(inst.status, theme);
    let muted = Style::default().fg(theme.muted);
    let fg = Style::default().fg(theme.fg);

    let id_w = (area.width as usize).saturating_sub(16).max(8);
    let container_id = trunc(&inst.container_id, id_w);
    let partition = inst.partition_info.as_deref().unwrap_or("-");
    let quant = inst.quantization.as_deref().unwrap_or("-");
    let port = inst.port.map_or_else(|| "-".into(), |p| p.to_string());
    let gpus = if inst.gpu_ids.is_empty() {
        "-".to_string()
    } else {
        inst.gpu_ids.join(",")
    };

    // Freshness label: age = snapshot_ts − observed_at (never wall-clock).
    let freshness_str =
        format::gen_tps_detail_freshness(inst.gen_tps_observation.as_ref(), snap_ts);
    let observed_str = inst.gen_tps_observation.as_ref().map_or_else(
        || "-".to_string(),
        |m| m.observed_at.format("%H:%M:%S UTC").to_string(),
    );

    let lines = vec![
        Line::from(vec![
            Span::styled(
                format!(" {status_text} "),
                Style::default()
                    .fg(theme.bg)
                    .bg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("id: ", muted),
            Span::styled(container_id, fg),
            Span::raw("  "),
            Span::styled("port: ", muted),
            Span::styled(port, fg),
            Span::raw("  "),
            Span::styled("tp: ", muted),
            Span::styled(inst.tensor_parallel_size.to_string(), fg),
        ]),
        Line::from(vec![
            Span::styled("model: ", muted),
            Span::styled(inst.model_name.clone(), fg),
            Span::raw("  "),
            Span::styled("gpus: ", muted),
            Span::styled(gpus, Style::default().fg(theme.accent)),
            Span::raw("  "),
            Span::styled("tok/W: ", muted),
            Span::styled(
                format::tokens_per_watt_cell(
                    inst.tokens_per_watt,
                    inst.gen_tps_observation.as_ref(),
                ),
                Style::default().fg(theme.accent),
            ),
            Span::raw("  "),
            Span::styled("gen: ", muted),
            Span::styled(
                format::gen_tps_compact(inst.gen_tps, inst.gen_tps_observation.as_ref()),
                fg,
            ),
        ]),
        Line::from(vec![
            Span::styled("partition: ", muted),
            Span::styled(partition.to_string(), fg),
            Span::raw("  "),
            Span::styled("quantization: ", muted),
            Span::styled(quant.to_string(), fg),
            Span::raw("  "),
            Span::styled("vram: ", muted),
            Span::styled(format::mib_pair(inst.vram_used_mb, inst.vram_total_mb), fg),
        ]),
        // 4th line: gen_tps freshness and deterministic age (snapshot-time, never wall-clock).
        Line::from(vec![
            Span::styled("gen freshness: ", muted),
            Span::styled(freshness_str, fg),
            Span::raw("  "),
            Span::styled("observed: ", muted),
            Span::styled(observed_str, Style::default().fg(theme.muted)),
        ]),
    ];

    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

/// Renders `p` (already wrapped) into `inner`, reserving a vertical scrollbar
/// column when `reserve` is set, and returns the pane's max scroll offset
/// plus the final content rect (so the caller can register it for mouse
/// hit-testing — see `AppState::record_scrollbar`).
///
/// `reserve` is decided by the caller (see `render_body`) rather than by this
/// pane's own content, because the launch_args/env_vars panes share one
/// scroll position and must stay the same width — gating each pane's column
/// on its own overflow independently would let one reserve a column while its
/// sibling doesn't, purely because one has slightly less content.
///
/// `len` — the wrapped line count the scrollbar's thumb and `max_scroll` are
/// both derived from — is measured once, at the width the pane will actually
/// render at (`inner.width - 1` when reserving, `inner.width` otherwise),
/// *before* drawing the bar. Measuring the bar at the pre-reservation width
/// and `max_scroll` at the post-reservation width (two different measurements
/// of the same pane) let them disagree whenever reserving the column changes
/// how the content wraps: the bar can render "nothing to scroll" in the same
/// frame the footer says `max_scroll > 0`. Deriving both from one `len`
/// makes that impossible by construction.
fn render_scrollable_pane(
    f: &mut Frame,
    inner: Rect,
    p: Paragraph<'_>,
    full_len: usize,
    scroll: u16,
    reserve: bool,
    theme: &Theme,
) -> (u16, Rect) {
    // `vertical_scrollbar_forced` (`panel.rs`) bails out and returns `inner`
    // unmodified — no column actually reserved — whenever `inner.width < 2`,
    // regardless of `reserve`. `len` must be measured at whatever width the
    // pane will really render at, so mirror that exact guard here: measuring
    // at `inner.width - 1` (down to 0) when the bar can't fit would disagree
    // with content that's still drawn at the full, unreserved `inner.width`.
    let will_reserve = reserve && inner.width >= 2;
    let len = if will_reserve {
        p.line_count(inner.width - 1)
    } else {
        full_len
    };
    let content = if will_reserve {
        panel::vertical_scrollbar_forced(
            f,
            inner,
            len,
            inner.height as usize,
            scroll as usize,
            theme,
        )
    } else {
        inner
    };
    let max = u16::try_from(len)
        .unwrap_or(u16::MAX)
        .saturating_sub(content.height);
    f.render_widget(p.scroll((scroll.min(max), 0)), content);
    (max, content)
}

/// Renders the launch_args/env_vars panes, applying `state.instance_detail_scroll`
/// to both, and returns the larger of the two panes' max scroll offsets so the
/// caller can clamp future scroll input (see `AppState::scroll_instance_detail`).
fn render_body(f: &mut Frame, area: Rect, inst: &Instance, state: &AppState, theme: &Theme) -> u16 {
    let scroll = state.instance_detail_scroll;
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
        .split(area);

    // launch_args (left)
    let args_inner = panel::bento(
        f,
        chunks[0],
        Some("launch_args"),
        BoxRole::Secondary,
        false,
        theme,
    );

    let args_lines: Vec<Line> = if inst.launch_args.is_empty() {
        vec![Line::from(Span::styled(
            "(none)",
            Style::default().fg(theme.muted),
        ))]
    } else {
        inst.launch_args
            .iter()
            .map(|a| Line::from(Span::styled(a.clone(), Style::default().fg(theme.fg))))
            .collect()
    };
    let args_p = Paragraph::new(args_lines).wrap(Wrap { trim: false });

    // env_vars (right). BTreeMap iterates sorted by key.
    let env_inner = panel::bento(
        f,
        chunks[1],
        Some("env_vars"),
        BoxRole::Primary,
        false,
        theme,
    );

    let env_lines: Vec<Line> = if inst.env_vars.is_empty() {
        vec![Line::from(Span::styled(
            "(none)",
            Style::default().fg(theme.muted),
        ))]
    } else {
        inst.env_vars
            .iter()
            .map(|(k, v)| {
                Line::from(vec![
                    Span::styled(k.clone(), Style::default().fg(theme.accent)),
                    Span::styled("=", Style::default().fg(theme.muted)),
                    Span::styled(v.clone(), Style::default().fg(theme.fg)),
                ])
            })
            .collect()
    };
    let env_p = Paragraph::new(env_lines).wrap(Wrap { trim: false });

    // Decide reservation once, from both panes' pre-reservation overflow, so
    // a scrollbar in either pane reserves the column in *both* — the two
    // share one scroll position and would otherwise end up different widths
    // whenever only one pane's content happened to overflow.
    //
    // This measures each pane at its pre-reservation width; `render_scrollable_pane`
    // measures again at the post-reservation width when `reserve` ends up true
    // (up to 4 `line_count` calls total for the two panes). That looks like it
    // could be collapsed to one measurement per pane by reusing the narrower
    // (reserved) width's count either way — word-wrap only ever wraps to the
    // same or *more* lines as width shrinks, so a pane that already fits within
    // `height` at the narrower width provably also fits at the wider one. But
    // the reverse direction doesn't hold: a pane whose narrower-width count
    // exceeds `height` might still fit fine at the wider, unreserved width, and
    // deciding `reserve` from the narrower count there would show a scrollbar
    // for a pane that never actually overflows when rendered without one —
    // reintroducing, one width away, the exact class of bug that measuring
    // `reserve` and the final wrap at two different widths (4773c227) exists to
    // prevent. The two measurements are a deliberate cost of that fix, not an
    // oversight — don't collapse them.
    let args_full_len = args_p.line_count(args_inner.width);
    let env_full_len = env_p.line_count(env_inner.width);
    let reserve = args_full_len > usize::from(args_inner.height)
        || env_full_len > usize::from(env_inner.height);

    let (args_max, args_content) =
        render_scrollable_pane(f, args_inner, args_p, args_full_len, scroll, reserve, theme);
    let (env_max, env_content) =
        render_scrollable_pane(f, env_inner, env_p, env_full_len, scroll, reserve, theme);

    // Register both panes' bars for mouse drag using the *shared* max — the
    // authoritative clamp both keyboard scrolling (`instance_detail_max_scroll`)
    // and the other pane use — rather than each pane's own, possibly smaller,
    // local max. Using a pane's own max here would let dragging the shorter
    // pane's bar (e.g. env_vars, often just "(none)") clamp against its own
    // near-zero range instead of the real shared one, even though its thumb's
    // *visual* size/position (already handled above) correctly reflects its
    // own content.
    let shared_max = args_max.max(env_max);
    let content_len = usize::from(shared_max) + usize::from(args_inner.height);
    // Both panes register against the same `content_len` (the shared max) so
    // dragging either bar clamps consistently — looping over the two
    // (area, drawn) pairs instead of writing the call out twice keeps that
    // guarantee from silently drifting if only one call site is ever edited.
    for (area, drawn) in [(args_inner, args_content), (env_inner, env_content)] {
        state.record_scrollbar(
            area,
            drawn,
            false,
            content_len,
            usize::from(area.height),
            ScrollTarget::InstanceDetail,
        );
    }

    shared_max
}

fn render_footer(f: &mut Frame, area: Rect, inst: &Instance, theme: &Theme, scrollable: bool) {
    let log = inst.log_file.as_deref().unwrap_or("-");
    let mut spans = Vec::new();
    if scrollable {
        // Only shown once `render_body` reports overflow — the launch_args/
        // env_vars panes otherwise give no hint that ↑/↓ do anything here.
        // Rendered first (not appended after the log path) so the hint
        // stays visible even when a long log path gets clipped by the
        // footer's width — Paragraph here isn't wrapped, so anything past
        // `area.width` is silently dropped rather than truncated in place.
        spans.push(Span::styled(
            "↑/↓ scroll  ·  ",
            Style::default().fg(theme.muted),
        ));
    }
    spans.push(Span::styled("log: ", Style::default().fg(theme.muted)));
    spans.push(Span::styled(
        log.to_string(),
        Style::default().fg(theme.muted),
    ));
    let p = Paragraph::new(Line::from(spans));
    f.render_widget(p, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tabs::instances::tests::{
        SNAP_SECS, buffer_text, mk_inst, mk_inst_obs, mk_snap, mk_state, obs, state_with_snap,
    };
    use chrono::TimeZone;
    use rocm_dash_core::metrics::{ObservationFreshness, ObservationMetadata};
    use std::collections::HashMap;

    #[test]
    fn render_shows_quantization_and_vram_for_populated_instance() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Populate the previously-dead fields: quantization + per-instance VRAM.
        let mut inst = mk_inst("vllm");
        inst.quantization = Some("fp8".into());
        inst.vram_used_mb = 49152;
        inst.vram_total_mb = 196_608; // mib_pair → "48.0 / 192.0 GiB"
        let vram = format::mib_pair(inst.vram_used_mb, inst.vram_total_mb);
        assert_eq!(vram, "48.0 / 192.0 GiB"); // sanity on the expected string

        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst);
        let state = mk_state(m, 0);

        // Detail modal: shows the quantization value and the VRAM pair.
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let detail = buffer_text(&term);
        assert!(
            detail.contains("fp8"),
            "detail modal must render the quantization value; got:\n{detail}"
        );
        assert!(
            detail.contains(&vram),
            "detail modal must render the used / total MiB VRAM string; got:\n{detail}"
        );
    }

    #[test]
    fn draw_detail_dims_periphery_with_grey_overlay() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Empty instance map hits `draw_detail`'s early-return branch, right
        // after its `grey_overlay(f)` call — the shortest path that still
        // exercises it. Text-only assertions on the popup body would still
        // pass if that call were silently dropped, since the corner is blank
        // either way; assert on the corner cell's background directly so
        // this fails if `grey_overlay(f)` is ever removed.
        let state = mk_state(HashMap::new(), 0);
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let wash = ratatui::style::Color::Rgb(0x1c, 0x1e, 0x22);
        let corner = term.backend().buffer().cell((0, 0)).unwrap();
        assert_eq!(
            corner.style().bg,
            Some(wash),
            "corner cell must carry grey_overlay's wash bg, not plain theme bg"
        );
    }

    #[test]
    fn detail_modal_body_scrolls_launch_args_and_reports_nonzero_max_scroll() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Enough launch_args to overflow the body pane at a realistic
        // terminal height, so `render_body` has real content to scroll.
        let mut inst = mk_inst("overflow");
        inst.launch_args = (0..60).map(|i| format!("--flag-{i}=value{i}")).collect();
        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst);
        let mut state = mk_state(m, 0);

        // Render once at scroll=0 and capture the max_scroll draw_detail
        // reports back — it must be non-zero given how much content
        // overflows the pane.
        let mut max_scroll = 0u16;
        let mut term = Terminal::new(TestBackend::new(160, 30)).unwrap();
        term.draw(|f| {
            max_scroll = draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        assert!(
            max_scroll > 0,
            "60 launch_args must overflow the body pane, giving a nonzero max_scroll; got {max_scroll}"
        );

        let text_top = buffer_text(&term);
        assert!(
            text_top.contains("--flag-0=value0"),
            "unscrolled body must show the first launch_args line; got:\n{text_top}"
        );

        // Scroll to the end and confirm the visible text actually shifts:
        // the first line scrolls out of view while the last scrolls in.
        state.instance_detail_scroll = max_scroll;
        let mut term2 = Terminal::new(TestBackend::new(160, 30)).unwrap();
        term2
            .draw(|f| {
                draw_detail(f, f.area(), &state, &state.theme);
            })
            .unwrap();
        let text_scrolled = buffer_text(&term2);
        assert!(
            !text_scrolled.contains("--flag-0=value0"),
            "fully scrolled body must no longer show the first launch_args line; got:\n{text_scrolled}"
        );
        assert!(
            text_scrolled.contains("--flag-59=value59"),
            "fully scrolled body must show the last launch_args line; got:\n{text_scrolled}"
        );
    }

    #[test]
    fn detail_modal_footer_shows_scroll_hint_only_when_scrollable() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Overflow the body pane (same recipe as the scroll test above) so
        // `draw_detail` computes a nonzero max_scroll and passes
        // `scrollable = true` into `render_footer`.
        let mut inst = mk_inst("overflow");
        inst.launch_args = (0..60).map(|i| format!("--flag-{i}=value{i}")).collect();
        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst);
        let state = mk_state(m, 0);

        let mut term = Terminal::new(TestBackend::new(160, 30)).unwrap();
        let mut max_scroll = 0u16;
        term.draw(|f| {
            max_scroll = draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        assert!(
            max_scroll > 0,
            "60 launch_args must overflow the body pane, giving a nonzero max_scroll; got {max_scroll}"
        );
        let scrollable_text = buffer_text(&term);
        assert!(
            scrollable_text.contains("↑/↓ scroll"),
            "footer must show the scroll hint once the body overflows; got:\n{scrollable_text}"
        );

        // A non-overflowing instance (no launch_args/env_vars) yields
        // max_scroll == 0, so `scrollable` is false and the hint must be
        // absent from the footer.
        let inst_small = mk_inst("small");
        let mut m2 = HashMap::new();
        m2.insert(inst_small.container_id.clone(), inst_small);
        let state_small = mk_state(m2, 0);

        let mut term2 = Terminal::new(TestBackend::new(160, 30)).unwrap();
        term2
            .draw(|f| {
                draw_detail(f, f.area(), &state_small, &state_small.theme);
            })
            .unwrap();
        let non_scrollable_text = buffer_text(&term2);
        assert!(
            !non_scrollable_text.contains("↑/↓ scroll"),
            "footer must not show the scroll hint when the body does not overflow; got:\n{non_scrollable_text}"
        );
    }

    #[test]
    fn detail_modal_body_shows_scrollbar_only_when_overflowing() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // `render_scrollable_pane` draws a `║`/`█` scrollbar (see
        // `panel::vertical_scrollbar`) once the pane's wrapped content
        // overflows the viewport — the footer's `↑/↓ scroll` text hint is not
        // this app's only scrollable-content affordance, and every other
        // scrollable surface (job console, managers, chat, dock) gets one.
        let mut inst = mk_inst("overflow");
        inst.launch_args = (0..60).map(|i| format!("--flag-{i}=value{i}")).collect();
        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst);
        let state = mk_state(m, 0);

        let mut term = Terminal::new(TestBackend::new(160, 30)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let text = buffer_text(&term);
        assert!(
            text.contains('║') && text.contains('█'),
            "overflowing body must render a scrollbar track and thumb; got:\n{text}"
        );

        // A non-overflowing instance must not draw a scrollbar at all —
        // otherwise the reserved column would needlessly narrow content that
        // already fits.
        let inst_small = mk_inst("small");
        let mut m2 = HashMap::new();
        m2.insert(inst_small.container_id.clone(), inst_small);
        let state_small = mk_state(m2, 0);

        let mut term2 = Terminal::new(TestBackend::new(160, 30)).unwrap();
        term2
            .draw(|f| {
                draw_detail(f, f.area(), &state_small, &state_small.theme);
            })
            .unwrap();
        let non_scrollable_text = buffer_text(&term2);
        assert!(
            !non_scrollable_text.contains('║') && !non_scrollable_text.contains('█'),
            "non-overflowing body must not draw a scrollbar; got:\n{non_scrollable_text}"
        );
    }

    #[test]
    fn detail_modal_registers_scrollbar_for_mouse_drag() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // `render_body` draws the bar via `panel::vertical_scrollbar_forced`
        // but that alone doesn't make it mouse-draggable — proves
        // `state.record_scrollbar` actually fires from the real draw path
        // (not just a synthetic-handle unit test on the mouse-routing side),
        // and that it's reachable through `state.scrollbars`, which is what
        // `scrollbar_hit` reads at click time.
        let mut inst = mk_inst("overflow");
        inst.launch_args = (0..60).map(|i| format!("--flag-{i}=value{i}")).collect();
        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst);
        let state = mk_state(m, 0);

        let mut term = Terminal::new(TestBackend::new(160, 30)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();

        let bars = state.scrollbars.borrow();
        assert!(
            bars.iter()
                .any(|h| h.target == ScrollTarget::InstanceDetail),
            "draw_detail must register a scrollbar for ScrollTarget::InstanceDetail"
        );
    }

    #[test]
    fn detail_modal_reserves_scrollbar_symmetrically_when_only_one_pane_overflows() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // launch_args overflows; env_vars is default-empty ("(none)", one
        // line) and would never overflow on its own. The two panes share one
        // scroll position (see `render_body`), so both must reserve the
        // column — otherwise env_vars would render one column wider than
        // launch_args purely because it happens to have less content, an
        // alignment wobble with no functional meaning.
        let mut inst = mk_inst("overflow");
        inst.launch_args = (0..60).map(|i| format!("--flag-{i}=value{i}")).collect();
        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst);
        let state = mk_state(m, 0);

        let mut term = Terminal::new(TestBackend::new(160, 30)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();

        let buf = term.backend().buffer();
        let area = buf.area;
        let mut bar_columns = std::collections::BTreeSet::new();
        for y in 0..area.height {
            for x in 0..area.width {
                let sym = buf.cell((x, y)).unwrap().symbol();
                if sym == "║" || sym == "█" {
                    bar_columns.insert(x);
                }
            }
        }
        assert_eq!(
            bar_columns.len(),
            2,
            "launch_args and env_vars must each reserve exactly one scrollbar \
             column, even though only launch_args overflows on its own; got \
             columns {bar_columns:?}"
        );
    }

    #[test]
    fn detail_modal_footer_keeps_scroll_hint_visible_with_long_log_path() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // `render_footer` renders the hint before the log path specifically
        // so a long log path can't push it out of view — its Paragraph
        // isn't wrapped, so anything past the footer's width is silently
        // dropped rather than truncated in place. That ordering had no
        // test: reverting it left the whole suite green.
        let mut inst = mk_inst("overflow");
        inst.launch_args = (0..60).map(|i| format!("--flag-{i}=value{i}")).collect();
        inst.log_file = Some("x".repeat(300));
        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst);
        let state = mk_state(m, 0);

        let mut term = Terminal::new(TestBackend::new(160, 30)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let text = buffer_text(&term);
        assert!(
            text.contains("↑/↓ scroll"),
            "a 300-char log path must not push the scroll hint out of the \
             footer; got:\n{text}"
        );
    }

    #[test]
    fn render_scrollable_pane_remeasures_wrap_at_post_reservation_width() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // `reserve` can be forced true by the *other* pane overflowing (see
        // `render_body`) even when this pane's own content fits at the
        // pre-reservation width — exactly the case this test sets up. A
        // 20-char line fits one row at width 20 but wraps to two at width 19
        // (the width `vertical_scrollbar_forced` leaves after reserving its
        // column), so re-measuring after reservation is what makes the
        // second row reachable at all: measuring only at the pre-reservation
        // width would silently under-report `max_scroll` by exactly that row.
        let inner = Rect::new(0, 0, 20, 3);
        let lines: Vec<Line> = vec![Line::raw("a"), Line::raw("b"), Line::raw("x".repeat(20))];
        let p = Paragraph::new(lines).wrap(Wrap { trim: false });
        let full_len = p.line_count(inner.width);
        assert_eq!(
            full_len, 3,
            "the 20-char line must fit in one row at the pre-reservation width 20"
        );

        let theme = Theme::from_name("default-dark");
        let mut term = Terminal::new(TestBackend::new(20, 3)).unwrap();
        let mut max = 0u16;
        term.draw(|f| {
            (max, _) = render_scrollable_pane(f, inner, p, full_len, 0, true, &theme);
        })
        .unwrap();

        assert_eq!(
            max, 1,
            "the 20-char line wraps to 2 rows at the post-reservation width \
             19, so max_scroll must be 1 (4 wrapped rows - 3 visible), not 0 \
             as a stale pre-reservation measurement would report"
        );

        // The bar must agree with `max`: if the scrollbar were still measured
        // at the pre-reservation width (full_len=3, which reads as "fits"),
        // it would render a solid "nothing to scroll" thumb in the same
        // frame `max_scroll == 1` says otherwise — the two affordances must
        // be derived from the same measurement, not just each be correct in
        // isolation.
        let buf = term.backend().buffer();
        let bar_column: Vec<&str> = (0..3)
            .map(|y| buf.cell((inner.width - 1, y)).unwrap().symbol())
            .collect();
        assert_ne!(
            bar_column,
            ["█", "█", "█"],
            "the bar must not render as a full 'nothing to scroll' thumb \
             when max_scroll is nonzero; got {bar_column:?}"
        );
    }

    #[test]
    fn render_scrollable_pane_measures_at_full_width_when_too_narrow_to_reserve() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // `vertical_scrollbar_forced` (panel.rs) bails out and returns `area`
        // unmodified whenever `area.width < 2` — no column is actually
        // reserved at width 1, regardless of `reserve`. Before this fix, `len`
        // was still measured at `inner.width.saturating_sub(1)` == 0 in that
        // case, disagreeing with content that's really drawn at the full,
        // unreserved width 1 — this pins that the two stay consistent.
        let inner = Rect::new(0, 0, 1, 3);
        let lines: Vec<Line> = vec![Line::raw("a"), Line::raw("b")];
        let p = Paragraph::new(lines).wrap(Wrap { trim: false });
        let full_len = p.line_count(inner.width);

        let theme = Theme::from_name("default-dark");
        let mut term = Terminal::new(TestBackend::new(1, 3)).unwrap();
        let mut result = (0u16, Rect::default());
        term.draw(|f| {
            result = render_scrollable_pane(f, inner, p, full_len, 0, true, &theme);
        })
        .unwrap();
        let (max, content) = result;

        assert_eq!(
            content.width, inner.width,
            "no column can be reserved at width 1, so the content rect must \
             stay the full, unreserved width"
        );
        assert_eq!(
            max,
            u16::try_from(full_len)
                .unwrap()
                .saturating_sub(inner.height),
            "max_scroll must be derived from the same (full) width the \
             content is actually rendered at, not from a width-0 measurement \
             that never happens on screen"
        );
    }

    #[test]
    fn detail_held_shows_held_freshness_label() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs(
            "held-detail",
            Some(100.0),
            Some(obs(ObservationFreshness::Held, 30)),
        );
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("held"),
            "detail for held instance must show 'held' freshness; got:\n{out}"
        );
    }

    #[test]
    fn detail_held_shows_held_marker_on_tok_w() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut inst = mk_inst_obs(
            "held-detail-tpw",
            Some(100.0),
            Some(obs(ObservationFreshness::Held, 30)),
        );
        inst.tokens_per_watt = Some(1.5);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("1.50 tok/W*"),
            "held tok/W must carry the held marker in the detail pane; got:\n{out}"
        );
    }

    #[test]
    fn detail_fresh_shows_fresh_label() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs(
            "fresh-detail",
            Some(100.0),
            Some(obs(ObservationFreshness::Fresh, 0)),
        );
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("fresh"),
            "detail for fresh instance must show 'fresh'; got:\n{out}"
        );
    }

    #[test]
    fn detail_legacy_meta_none_shows_unknown() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs("legacy-detail", Some(100.0), None);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("unknown"),
            "detail for legacy (None metadata) must say 'unknown'; got:\n{out}"
        );
    }

    #[test]
    fn recent_activity_rows_skips_idle_ticks_and_orders_newest_first() {
        let mut busy_old = mk_inst("x");
        busy_old.gen_tps = Some(10.0);
        let mut idle = mk_inst("x");
        idle.gen_tps = Some(0.0);
        let mut no_sample = mk_inst("x");
        no_sample.gen_tps = None;
        let mut non_finite = mk_inst("x");
        non_finite.gen_tps = Some(f64::INFINITY);
        let mut busy_new = mk_inst("x");
        busy_new.gen_tps = Some(86.0);

        let mut history: std::collections::VecDeque<_> = std::collections::VecDeque::new();
        let mut snap = mk_snap(vec![busy_old]);
        snap.timestamp = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
        history.push_back(snap);
        let mut snap = mk_snap(vec![idle]);
        snap.timestamp = Utc.timestamp_opt(1_700_000_030, 0).unwrap();
        history.push_back(snap);
        let mut snap = mk_snap(vec![no_sample]);
        snap.timestamp = Utc.timestamp_opt(1_700_000_045, 0).unwrap();
        history.push_back(snap);
        let mut snap = mk_snap(vec![non_finite]);
        snap.timestamp = Utc.timestamp_opt(1_700_000_050, 0).unwrap();
        history.push_back(snap);
        let mut snap = mk_snap(vec![busy_new]);
        snap.timestamp = Utc.timestamp_opt(1_700_000_060, 0).unwrap();
        history.push_back(snap);

        let rows = recent_activity_rows(&history, "id-x", RECENT_ACTIVITY_CAP);
        assert_eq!(
            rows.len(),
            2,
            "idle, sample-less and non-finite ticks must be skipped"
        );
        assert_eq!(
            rows[0].gen_tps,
            Some(86.0),
            "newest activity must come first"
        );
        assert_eq!(rows[1].gen_tps, Some(10.0));
    }

    #[test]
    fn recent_activity_rows_caps_to_requested_len() {
        let mut history: std::collections::VecDeque<_> = std::collections::VecDeque::new();
        for s in 0..10i64 {
            let mut i = mk_inst("x");
            i.gen_tps = Some(5.0);
            let mut snap = mk_snap(vec![i]);
            snap.timestamp = Utc.timestamp_opt(1_700_000_000 + s, 0).unwrap();
            history.push_back(snap);
        }
        let rows = recent_activity_rows(&history, "id-x", 3);
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn recent_activity_rows_collapses_held_ticks_and_ages_from_observed_at() {
        let mut history: std::collections::VecDeque<_> = std::collections::VecDeque::new();
        // One scrape at t=100 (Fresh), re-published as Held on the next two
        // ticks, then a new scrape at t=103.
        for (tick, observed, fresh) in [
            (100, 100, ObservationFreshness::Fresh),
            (101, 100, ObservationFreshness::Held),
            (102, 100, ObservationFreshness::Held),
            (103, 103, ObservationFreshness::Fresh),
        ] {
            let mut i = mk_inst("x");
            i.gen_tps = Some(86.0);
            // A Held tick carries no latencies of its own (failed scrape).
            let is_fresh = fresh == ObservationFreshness::Fresh;
            i.ttft_ms =
                is_fresh.then_some(120.0 + f64::from(u8::try_from(observed - 100).unwrap()));
            i.tpot_ms = is_fresh.then_some(40.0);
            i.gen_tps_observation = Some(ObservationMetadata {
                observed_at: Utc.timestamp_opt(1_700_000_000 + observed, 0).unwrap(),
                freshness: fresh,
            });
            let mut snap = mk_snap(vec![i]);
            snap.timestamp = Utc.timestamp_opt(1_700_000_000 + tick, 0).unwrap();
            history.push_back(snap);
        }
        let rows = recent_activity_rows(&history, "id-x", RECENT_ACTIVITY_CAP);
        let ts: Vec<i64> = rows
            .iter()
            .map(|r| r.ts.timestamp() - 1_700_000_000)
            .collect();
        assert_eq!(ts, vec![103, 100]);
        // Each group is represented by its scrape's own Fresh tick, not the
        // later Held re-publication, so no row wrongly carries the held
        // marker and the latencies are the scrape's.
        for r in &rows {
            assert_eq!(
                r.gen_tps_observation.as_ref().map(|o| o.freshness),
                Some(ObservationFreshness::Fresh)
            );
        }
        assert_eq!(rows[0].ttft_ms, Some(123.0));
        assert_eq!(rows[1].ttft_ms, Some(120.0));
        assert_eq!(rows[1].tpot_ms, Some(40.0));
    }

    #[test]
    fn recent_activity_rows_ignores_sibling_instances() {
        let mut history: std::collections::VecDeque<_> = std::collections::VecDeque::new();
        let mut mine = mk_inst("x");
        mine.gen_tps = Some(11.0);
        let mut sibling = mk_inst("y");
        sibling.gen_tps = Some(99.0);
        history.push_back(mk_snap(vec![sibling.clone(), mine]));
        let mut quiet = mk_inst("x");
        quiet.gen_tps = Some(0.0);
        history.push_back(mk_snap(vec![sibling, quiet]));
        let rows = recent_activity_rows(&history, "id-x", RECENT_ACTIVITY_CAP);
        assert_eq!(rows.len(), 1, "only this instance's own samples");
        assert_eq!(rows[0].gen_tps, Some(11.0));
        assert!(has_observed_throughput(&history, "id-x"));
        let mut only_sibling: std::collections::VecDeque<_> = std::collections::VecDeque::new();
        let mut sib = mk_inst("y");
        sib.gen_tps = Some(0.0);
        only_sibling.push_back(mk_snap(vec![sib]));
        assert!(
            !has_observed_throughput(&only_sibling, "id-x"),
            "a sibling's samples are not this instance's"
        );
    }

    #[test]
    fn detail_modal_shows_no_data_placeholder_when_history_has_no_throughput_sample() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // gen_tps = None and an empty history: nothing has ever been
        // observed for this instance, so this must not claim it's "idle" —
        // that would be a false claim for a too-new instance, one past its
        // observation validity window, or a managed non-vLLM engine that
        // never gets gen_tps wired up.
        let inst = mk_inst_obs("idle-detail", None, None);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("no throughput samples"),
            "detail with no observed sample must not claim 'idle'; got:\n{out}"
        );
        assert!(
            !out.contains("has been idle"),
            "must not show the idle placeholder when nothing was ever observed; got:\n{out}"
        );
    }

    #[test]
    fn detail_modal_shows_idle_placeholder_when_history_has_only_zero_samples() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // A real Some(0.0) sample in history means the instance has actually
        // been observed and was genuinely idle — unlike the no-data case
        // above, "has been idle" is a true claim here.
        let inst = mk_inst_obs("idle-detail", None, None);
        let mut state = state_with_snap(inst.clone());
        let mut zero = inst;
        zero.gen_tps = Some(0.0);
        let mut snap = mk_snap(vec![zero]);
        snap.timestamp = Utc.timestamp_opt(SNAP_SECS - 30, 0).unwrap();
        state.history.push_back(snap);

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("has been idle"),
            "detail with an observed zero sample must show the idle placeholder; got:\n{out}"
        );
    }

    #[test]
    fn detail_modal_shows_recent_activity_row() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs("burst-detail", Some(0.0), None);
        let mut state = state_with_snap(inst.clone());
        let mut burst = inst;
        burst.gen_tps = Some(86.0);
        let mut snap = mk_snap(vec![burst]);
        snap.timestamp = Utc.timestamp_opt(SNAP_SECS - 30, 0).unwrap();
        state.history.push_back(snap);

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("86"),
            "must show the historical burst's tok/s; got:\n{out}"
        );
        assert!(
            out.contains("30s ago"),
            "must show the burst's age relative to the live snapshot; got:\n{out}"
        );
    }

    #[test]
    fn detail_modal_shows_held_marker_for_recent_activity_row() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs("held-burst-detail", Some(0.0), None);
        let mut state = state_with_snap(inst.clone());
        let mut burst = inst;
        burst.gen_tps = Some(86.0);
        burst.gen_tps_observation = Some(obs(ObservationFreshness::Held, 30));
        let mut snap = mk_snap(vec![burst]);
        snap.timestamp = Utc.timestamp_opt(SNAP_SECS - 30, 0).unwrap();
        state.history.push_back(snap);

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("86*"),
            "held gen_tps in the activity row must carry the held marker; got:\n{out}"
        );
    }

    #[test]
    fn detail_modal_shows_all_activity_rows_up_to_the_cap() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Regression: `activity_h` must account for `panel::bento`'s
        // top-padding row once the box reaches 5 rows tall, or
        // RECENT_ACTIVITY_CAP = 5 only shows 4 rows (the oldest clipped).
        let inst = mk_inst_obs("multi-burst-detail", Some(0.0), None);
        let mut state = state_with_snap(inst.clone());
        for (i, tps) in [101.0, 102.0, 103.0, 104.0, 105.0].into_iter().enumerate() {
            let mut row = inst.clone();
            row.gen_tps = Some(tps);
            let mut snap = mk_snap(vec![row]);
            snap.timestamp = Utc
                .timestamp_opt(SNAP_SECS - 14 + i64::try_from(i).unwrap(), 0)
                .unwrap();
            state.history.push_back(snap);
        }

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        for tps in ["101", "102", "103", "104", "105"] {
            assert!(
                out.contains(tps),
                "activity row for {tps} tok/s must not be clipped; got:\n{out}"
            );
        }
    }

    #[test]
    fn detail_modal_body_scrollbar_survives_a_full_activity_panel_on_a_short_terminal() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Regression for the body floor itself (not just the padding
        // fix above): at 90x18 the popup's inner height is 13, so the
        // panel collapses and the body keeps its rows. Without the floor, the
        // activity panel's `Min(0)` neighbor gets starved to 0 and the
        // scrollbar silently stops rendering even though the body still
        // overflows.
        let mut inst = mk_inst_obs("overflow-short-detail", Some(0.0), None);
        inst.launch_args = (0..60).map(|i| format!("--flag-{i}=value{i}")).collect();
        let mut state = state_with_snap(inst.clone());
        for (i, tps) in [101.0, 102.0, 103.0, 104.0, 105.0].into_iter().enumerate() {
            let mut row = inst.clone();
            row.gen_tps = Some(tps);
            let mut snap = mk_snap(vec![row]);
            snap.timestamp = Utc
                .timestamp_opt(SNAP_SECS - 14 + i64::try_from(i).unwrap(), 0)
                .unwrap();
            state.history.push_back(snap);
        }

        let mut term = Terminal::new(TestBackend::new(90, 18)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        // At this height the panel would leave the body under its floor, so
        // it collapses and the launch args get the rows.
        assert!(
            !out.contains("recent throughput"),
            "panel must collapse rather than starve the body; got:\n{out}"
        );
        // ...and the body still renders its scrollbar thumb beside it.
        assert!(
            out.contains('█'),
            "overflowing body must still render a scrollbar thumb next to \
             the activity panel; got:\n{out}"
        );
    }

    #[test]
    fn detail_modal_disconnected_activity_rows_show_no_fabricated_age() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // `Disconnected` clears `latest` but leaves history populated, so rows
        // render with no snapshot timestamp to age them against.
        let inst = mk_inst_obs("disc-detail", Some(0.0), None);
        let mut state = state_with_snap(inst.clone());
        let mut burst = inst;
        burst.gen_tps = Some(86.0);
        let mut snap = mk_snap(vec![burst]);
        snap.timestamp = Utc.timestamp_opt(SNAP_SECS - 30, 0).unwrap();
        state.history.push_back(snap);
        state.latest = None;

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("TOK/S"),
            "panel must still list the row; got:\n{out}"
        );
        assert!(
            !out.contains("0s ago"),
            "a missing snapshot must not render a made-up age; got:\n{out}"
        );
    }

    #[test]
    fn detail_modal_activity_row_shows_ttft_and_tpot_in_their_own_columns() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs("ttft-tpot-detail", Some(0.0), None);
        let mut state = state_with_snap(inst.clone());
        let mut burst = inst;
        burst.gen_tps = Some(86.0);
        burst.ttft_ms = Some(123.0);
        burst.tpot_ms = Some(45.0);
        let mut snap = mk_snap(vec![burst]);
        snap.timestamp = Utc.timestamp_opt(SNAP_SECS - 30, 0).unwrap();
        state.history.push_back(snap);

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        let row = out
            .lines()
            .find(|l| l.contains("30s ago"))
            .unwrap_or_else(|| panic!("no activity row; got:\n{out}"));
        let fields: Vec<&str> = row.split_whitespace().collect();
        let at = fields.iter().position(|f| *f == "ago").unwrap();
        assert_eq!(
            fields.get(at + 2).copied(),
            Some("123ms"),
            "TTFT column; row: {row}"
        );
        assert_eq!(
            fields.get(at + 3).copied(),
            Some("45ms"),
            "TPOT column; row: {row}"
        );
    }

    #[test]
    fn detail_modal_shows_no_data_placeholder_when_history_has_only_non_finite_samples() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs("inf-detail", None, None);
        let mut state = state_with_snap(inst.clone());
        let mut inf = inst;
        inf.gen_tps = Some(f64::INFINITY);
        let mut snap = mk_snap(vec![inf]);
        snap.timestamp = Utc.timestamp_opt(SNAP_SECS - 30, 0).unwrap();
        state.history.push_back(snap);

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("no throughput samples"),
            "non-finite samples are not observations; got:\n{out}"
        );
        assert!(
            !out.contains("has been idle"),
            "must not claim idle from non-finite samples; got:\n{out}"
        );
    }

    #[test]
    fn activity_layout_never_exceeds_budget_and_keeps_newest_rows() {
        for inner_h in 0..=60u16 {
            for n in 0..=RECENT_ACTIVITY_CAP {
                let (h, k) = activity_layout(n, inner_h);
                let avail = inner_h.saturating_sub(4 + 1 + body_floor(inner_h));
                assert!(h <= avail, "n={n} inner_h={inner_h}: h={h} > {avail}");
                assert!(k <= n);
                if n > 0 && h > 0 {
                    // header + k rows + borders (+ top pad at >= 5 rows)
                    assert!(h >= 2 + 1 + k as u16, "n={n} inner_h={inner_h}");
                }
                if n > 0 {
                    assert!(k >= 1 || h == 0, "a drawn panel shows at least one row");
                }
            }
        }
        // Spacious: all five rows. Mid band: partial, but sized to match.
        assert_eq!(activity_layout(5, 48).1, 5);
        let (h, k) = activity_layout(5, 24);
        assert!(
            (1..5).contains(&k),
            "mid-height band shows a partial panel: {k}"
        );
        assert!(h > 0);
        // Placeholder: shown when it fits, collapsed otherwise.
        assert_eq!(activity_layout(0, 48), (3, 0));
        assert_eq!(activity_layout(0, 8), (0, 0));
    }

    #[test]
    fn detail_modal_mid_height_terminal_shows_only_rows_that_fit_newest_first() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Regression: on a body height where the body floor caps the panel
        // without collapsing the panel, the panel used to be handed all five
        // rows and silently clip the oldest ones mid-table.
        let inst = mk_inst_obs("mid-height-detail", Some(0.0), None);
        let mut state = state_with_snap(inst.clone());
        for (i, tps) in [101.0, 102.0, 103.0, 104.0, 105.0].into_iter().enumerate() {
            let mut row = inst.clone();
            row.gen_tps = Some(tps);
            let mut snap = mk_snap(vec![row]);
            snap.timestamp = Utc
                .timestamp_opt(SNAP_SECS - 14 + i64::try_from(i).unwrap(), 0)
                .unwrap();
            state.history.push_back(snap);
        }
        let mut term = Terminal::new(TestBackend::new(100, 28)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        let shown: Vec<&str> = ["105", "104", "103", "102", "101"]
            .into_iter()
            .filter(|t| out.contains(t))
            .collect();
        assert!(
            !shown.is_empty(),
            "panel should show some rows; got:\n{out}"
        );
        assert!(
            shown.len() < 5 && out.contains(&format!("(newest {})", shown.len())),
            "a clipped panel must say how many rows it hides; got:\n{out}"
        );
        // Newest-first contiguous prefix: never a newer row missing while an older one shows.
        let prefix_len = shown.len();
        assert_eq!(
            shown,
            ["105", "104", "103", "102", "101"][..prefix_len].to_vec(),
            "rows must be the newest contiguous prefix; got:\n{out}"
        );
    }

    #[test]
    fn detail_modal_empty_history_on_short_terminal_collapses_or_shows_placeholder() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let state = state_with_snap(mk_inst_obs("empty-short-detail", None, None));
        let mut term = Terminal::new(TestBackend::new(90, 12)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        // At 90x12 the placeholder panel (3 rows) can't fit beside the body
        // floor, so it collapses entirely rather than drawing a border-only box.
        assert!(
            !out.contains("recent throughput") && !out.contains("no throughput data"),
            "placeholder panel must collapse on a short terminal; got:\n{out}"
        );
        // On a roomy terminal the same state shows the placeholder.
        let mut term = Terminal::new(TestBackend::new(90, 40)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        assert!(buffer_text(&term).contains("no throughput samples"));
    }

    #[test]
    fn detail_modal_at_80x24_keeps_several_launch_args_visible_beside_activity() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut inst = mk_inst_obs("default-size-detail", Some(0.0), None);
        inst.launch_args = (0..60).map(|i| format!("--flag-{i}=value{i}")).collect();
        let mut state = state_with_snap(inst.clone());
        for (i, tps) in [101.0, 102.0, 103.0, 104.0, 105.0].into_iter().enumerate() {
            let mut row = inst.clone();
            row.gen_tps = Some(tps);
            let mut snap = mk_snap(vec![row]);
            snap.timestamp = Utc
                .timestamp_opt(SNAP_SECS - 14 + i64::try_from(i).unwrap(), 0)
                .unwrap();
            state.history.push_back(snap);
        }
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        let visible = (0..60)
            .filter(|i| out.contains(&format!("--flag-{i}=value{i}")))
            .count();
        assert!(
            visible >= 5,
            "launch args must stay readable at 80x24 ({visible} visible); got:\n{out}"
        );
        assert!(out.contains("recent throughput"), "got:\n{out}");
    }

    #[test]
    fn detail_modal_explains_held_marker_only_when_a_shown_row_is_held() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let render = |freshness| {
            let inst = mk_inst_obs("held-legend-detail", Some(0.0), None);
            let mut state = state_with_snap(inst.clone());
            let mut row = inst;
            row.gen_tps = Some(86.0);
            row.gen_tps_observation = Some(obs(freshness, 30));
            let mut snap = mk_snap(vec![row]);
            snap.timestamp = Utc.timestamp_opt(SNAP_SECS - 5, 0).unwrap();
            state.history.push_back(snap);
            let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
            term.draw(|f| {
                draw_detail(f, f.area(), &state, &state.theme);
            })
            .unwrap();
            buffer_text(&term)
        };
        let held = render(ObservationFreshness::Held);
        assert!(held.contains("86*"), "held row marker; got:\n{held}");
        assert!(held.contains(format::HELD_LEGEND), "got:\n{held}");
        let fresh = render(ObservationFreshness::Fresh);
        assert!(!fresh.contains(format::HELD_LEGEND), "got:\n{fresh}");
    }

    #[test]
    fn detail_modal_clamps_a_future_sample_age_to_zero() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs("future-age-detail", Some(0.0), None);
        let mut state = state_with_snap(inst.clone());
        let mut row = inst;
        row.gen_tps = Some(77.0);
        let mut snap = mk_snap(vec![row]);
        snap.timestamp = Utc.timestamp_opt(SNAP_SECS + 30, 0).unwrap();
        state.history.push_back(snap);
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| {
            draw_detail(f, f.area(), &state, &state.theme);
        })
        .unwrap();
        let out = buffer_text(&term);
        assert!(out.contains("0s ago"), "got:\n{out}");
        assert!(!out.contains("-30s"), "got:\n{out}");
    }
}
