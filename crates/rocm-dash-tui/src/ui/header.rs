// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The dashboard header: the title line, the connection state and the warning
//! badge, plus the badge's click region.
//!
//! Split out of `ui/mod.rs`; `draw_header` returns the badge's rect so the
//! caller can record it for mouse hit-testing.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::{AppState, ConnState};
use crate::ui::panel;
use crate::ui::theme::Theme;

/// Draw the header and return where the `⚠ N` warning badge landed (clipped to
/// the header, `None` when no badge is shown) so a click on it can be resolved.
pub(super) fn draw_header(
    f: &mut Frame,
    area: Rect,
    state: &AppState,
    theme: &Theme,
) -> Option<Rect> {
    // In demo/replay the data is not live, so never present the session as
    // "connected" to a real daemon — the Connected case shows a simulated label
    // instead, and the SIMULATED DATA chip below makes the state unmistakable.
    // Other states (e.g. Disconnected on end-of-recording or a failed replay
    // file) still surface their reason so playback status is not masked.
    let (status_text, status_color) = match &state.conn {
        ConnState::Initial => ("starting".to_string(), theme.muted),
        ConnState::Connecting => ("connecting…".to_string(), theme.warn),
        ConnState::Connected { .. } if state.simulated => {
            ("simulated — not live".to_string(), theme.warn)
        }
        c @ ConnState::Connected { host, .. } if c.is_in_process() => {
            (format!("live · {host} · local collectors"), theme.ok)
        }
        ConnState::Connected { host, version } => (
            format!("connected · {host} · rocm daemon {version}"),
            theme.ok,
        ),
        ConnState::Disconnected { reason } => (format!("disconnected · {reason}"), theme.err),
    };

    let mut spans: Vec<Span> = vec![
        Span::styled(
            "rocm.ai",
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(theme.accent),
        ),
        Span::raw("  "),
    ];
    if state.simulated {
        spans.push(Span::styled(
            " SIMULATED DATA ",
            Style::default()
                .bg(theme.warn)
                .fg(theme.surface_2)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw("  "));
    }
    let warning_count = state.warning_count();
    // (column offset within the header, width) of the badge, once placed.
    let mut badge: Option<(u16, u16)> = None;
    // Placed right after the brand, ahead of the connection target and status:
    // the header is one unwrapped row, and on an 80-column terminal those two
    // already fill it, which would clip a trailing badge out of sight (and out of
    // reach of a click) when it is the part that asks the user to do something.
    if warning_count > 0 {
        let offset: usize = spans.iter().map(Span::width).sum();
        // Advertise the key only where it works; the click is gated the same way.
        let hint = if state.global_hotkeys_reachable() {
            " (!)"
        } else {
            ""
        };
        let badge_span = Span::styled(
            format!(" ⚠ {warning_count}{hint} "),
            Style::default()
                .bg(theme.warn)
                .fg(theme.surface_2)
                .add_modifier(Modifier::BOLD),
        );
        badge = Some((
            u16::try_from(offset).unwrap_or(u16::MAX),
            u16::try_from(badge_span.width()).unwrap_or(u16::MAX),
        ));
        spans.push(badge_span);
        spans.push(Span::raw("   "));
    }
    spans.push(Span::styled(
        format!("→ {}", state.connect),
        Style::default().fg(theme.muted),
    ));
    spans.push(Span::raw("   "));
    spans.push(Span::styled(status_text, Style::default().fg(status_color)));
    if let Some(r) = state.replay.as_ref() {
        spans.push(Span::raw("   "));
        let (icon, fg) = if r.paused {
            ("⏸", theme.warn)
        } else {
            ("▶", theme.ok)
        };
        spans.push(Span::styled(
            format!(" {icon} {:.2}× ", r.speed),
            Style::default()
                .bg(theme.surface_2)
                .fg(fg)
                .add_modifier(Modifier::BOLD),
        ));
        if r.total_s > 0 {
            spans.push(Span::raw(" "));
            spans.push(Span::styled(
                format!(
                    "{} / {}",
                    crate::app::format_mmss(r.elapsed_s),
                    crate::app::format_mmss(r.total_s)
                ),
                Style::default().fg(theme.muted),
            ));
        }
    }
    spans.push(Span::raw("   "));
    spans.push(Span::styled(
        format!("theme: {}", state.theme_name),
        Style::default().fg(theme.muted),
    ));
    // Headline chrome hint (per the mocks): Esc menu · t theme · ? help.
    spans.push(Span::raw("   "));
    spans.push(Span::styled(
        "Esc menu · t theme · ? help",
        Style::default().fg(theme.muted),
    ));
    let inner = panel::bento(f, area, None, panel::BoxRole::Neutral, false, theme);
    f.render_widget(Paragraph::new(vec![Line::from(spans)]), inner);
    // The paragraph is not wrapped, so whatever runs past `inner` is cut off; the
    // clickable region must be cut the same way.
    badge.and_then(|(offset, width)| {
        let width = width.min(inner.width.saturating_sub(offset));
        (inner.height > 0 && width > 0).then(|| Rect::new(inner.x + offset, inner.y, width, 1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::app::{ConnState, Modal};
    use crate::ui::tests::{draw_frame, state_with_warnings};
    use ratatui::layout::Rect;

    fn text_in(rows: &[String], r: Rect) -> String {
        rows[usize::from(r.y)]
            .chars()
            .skip(usize::from(r.x))
            .take(usize::from(r.width))
            .collect()
    }

    #[test]
    fn header_does_not_claim_a_daemon_for_the_in_process_collector() {
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.conn = ConnState::Connected {
            host: "localhost".into(),
            version: ConnState::in_process_version("1.2.3"),
        };
        let mut term = Terminal::new(TestBackend::new(120, 3)).unwrap();
        term.draw(|f| {
            draw_header(f, f.area(), &state, &theme);
        })
        .unwrap();
        let grid: String = (0..3)
            .flat_map(|y| (0..120).map(move |x| (x, y)))
            .map(|p| {
                term.backend()
                    .buffer()
                    .cell(p)
                    .unwrap()
                    .symbol()
                    .to_string()
            })
            .collect();
        assert!(grid.contains("local collectors"), "got: {grid}");
        assert!(!grid.contains("rocm daemon"), "got: {grid}");
    }

    #[test]
    fn header_badge_region_covers_the_drawn_badge_and_advertises_the_key() {
        let mut s = state_with_warnings(&["only one"]);
        let rows = draw_frame(&mut s, 140, 40);
        let badge = s.last_header_badge.expect("a warning is shown, so a badge");
        // Exactly the drawn badge, padding included: a region one column off
        // would still contain the glyph and the count.
        assert_eq!(text_in(&rows, badge), " ⚠ 1 (!) ");
    }

    #[test]
    fn header_has_no_badge_region_without_warnings() {
        let mut s = state_with_warnings(&[]);
        draw_frame(&mut s, 140, 40);
        assert_eq!(s.last_header_badge, None);
    }

    #[test]
    fn header_badge_drops_the_key_hint_while_another_modal_owns_the_keyboard() {
        let mut s = state_with_warnings(&["only one"]);
        s.modal = Modal::Help;
        let rows = draw_frame(&mut s, 140, 40);
        let badge = s.last_header_badge.expect("the badge is still counted");
        let under = text_in(&rows, badge);
        assert!(under.contains('⚠'), "{under:?}");
        assert!(!under.contains("(!)"), "`!` does nothing here: {under:?}");
    }

    #[test]
    fn header_badge_drops_the_key_hint_while_an_overlay_or_approval_owns_the_keyboard() {
        use crate::ui::services_manager::ServicesManagerState;
        let mut overlay = state_with_warnings(&["only one"]);
        overlay.services = Some(ServicesManagerState::default());
        let mut approval = state_with_warnings(&["only one"]);
        approval.open_approval(crate::tool_exec::ApprovalIntent {
            title: "run a command".into(),
            body: vec!["echo hi".into()],
            name: "shell".into(),
            arguments: serde_json::Value::Null,
        });
        for (what, mut s) in [
            ("a manager overlay", overlay),
            ("a pending approval", approval),
        ] {
            let rows = draw_frame(&mut s, 140, 40);
            let badge = s.last_header_badge.expect("the badge is still counted");
            let under = text_in(&rows, badge);
            assert!(under.contains('⚠'), "{what}: {under:?}");
            assert!(
                !under.contains("(!)"),
                "{what}: `!` does nothing here: {under:?}"
            );
        }
    }

    #[test]
    fn header_badge_is_visible_on_an_80_column_terminal_with_a_long_connection_line() {
        // The connection target and status already fill an 80-column header; a
        // badge drawn after them was cut off, so the count (and its click) could
        // not be seen on the smallest terminal the dashboard supports.
        let mut s = state_with_warnings(&["only one"]);
        s.connect = "replay:a-recording-with-a-fairly-long-name.ndjson".into();
        s.conn = ConnState::Connected {
            host: "a-host-with-a-long-name".into(),
            version: "0.1.0".into(),
        };
        let rows = draw_frame(&mut s, 80, 24);
        let badge = s.last_header_badge.expect("badge must fit at 80 columns");
        assert!(text_in(&rows, badge).contains('⚠'), "{:?}", rows[1]);
    }

    #[test]
    fn header_badge_region_is_clipped_to_a_header_narrower_than_the_badge() {
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let s = state_with_warnings(&["only one"]);
        let theme = Theme::from_name("default-dark");
        let badge_at = |w: u16| {
            let mut term = Terminal::new(TestBackend::new(w, 3)).unwrap();
            let mut badge = None;
            term.draw(|f| badge = draw_header(f, f.area(), &s, &theme))
                .unwrap();
            badge
        };
        let full = badge_at(120).expect("room for the whole badge").width;
        let mut clipped = false;
        for w in 1..=40u16 {
            if let Some(r) = badge_at(w) {
                assert!(r.x + r.width <= w, "width {w}: {r:?} runs off the screen");
                clipped |= r.width < full;
            }
        }
        assert!(
            clipped,
            "no width cut the badge short, so clipping is untested"
        );
    }
}
