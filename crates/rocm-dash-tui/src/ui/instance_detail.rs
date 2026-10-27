// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The instance Detail modal: a summary strip, scrollable launch-args and
//! env-vars panes, and a footer, drawn over a dimmed backdrop.
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
use ratatui::widgets::{Paragraph, Wrap};

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
    // | body (min) | footer (1)
    let snap_ts = state.latest.as_ref().map(|s| s.timestamp);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);

    render_summary(f, chunks[0], inst, snap_ts, theme);
    let max_scroll = render_body(f, chunks[1], inst, state, theme);
    render_footer(f, chunks[2], inst, theme, max_scroll > 0);
    max_scroll
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
            Span::styled(
                if inst.gen_tps_last_request {
                    "last req tok/s: "
                } else {
                    "gen: "
                },
                muted,
            ),
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
        buffer_text, last_req_inst, mk_inst, mk_inst_obs, mk_state, obs, state_with_snap,
    };
    use rocm_dash_core::metrics::ObservationFreshness;
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
    fn detail_names_a_last_request_rate() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        for (last_request, label) in [(true, "last req tok/s:"), (false, "gen:")] {
            let state = state_with_snap(last_req_inst("rate-detail", last_request));
            let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
            term.draw(|f| {
                draw_detail(f, f.area(), &state, &state.theme);
            })
            .unwrap();
            let out = buffer_text(&term);
            assert!(out.contains(label), "{label:?} missing:\n{out}");
            assert_eq!(out.contains("last req tok/s:"), last_request, "{out}");
        }
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
}
