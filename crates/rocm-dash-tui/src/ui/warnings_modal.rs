// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The Warnings modal (`Modal::Warnings`): the text behind the header `⚠ N`
//! badge, content-sized like the Help modals.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::ui::help_modal::popup_height_for;
use crate::ui::panel::{centered_rect, centered_width, draw_popup_frame, grey_overlay};
use crate::ui::theme::Theme;

/// Lines of text a multi-line warning is cut to once it is longer than this plus
/// one. A failed amd-smi call embeds its raw stderr, which can run to dozens of
/// lines; uncapped it would push every warning listed after it off a popup that
/// does not scroll. A warning only one line over the cap is shown whole, since
/// "1 more line" costs the row it saves.
///
/// The cap counts lines of the warning's text, not the rows they wrap to, so one
/// very long line can still take several rows. A popup that runs out of room
/// says so with its truncation marker (see [`draw_warnings`]).
const WARNING_MAX_LINES: usize = 4;

/// One warning as popup lines: a bullet on the first line, an indent on the
/// continuation lines of a multi-line one, and a count of what the cap cut.
fn warning_lines<'a>(warning: &'a str, theme: &Theme) -> Vec<Line<'a>> {
    let bullet = Style::default().fg(theme.warn);
    let text = Style::default().fg(theme.fg);
    let all: Vec<&str> = warning
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .collect();
    if all.is_empty() {
        return vec![Line::from(Span::styled("• ", bullet))];
    }
    let shown = if all.len() <= WARNING_MAX_LINES + 1 {
        all.len()
    } else {
        WARNING_MAX_LINES
    };
    let mut out: Vec<Line> = all[..shown]
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let lead = if i == 0 { "• " } else { "  " };
            Line::from(vec![Span::styled(lead, bullet), Span::styled(*l, text)])
        })
        .collect();
    if all.len() > shown {
        out.push(Line::from(Span::styled(
            format!("  … {} more line(s)", all.len() - shown),
            Style::default().fg(theme.muted),
        )));
    }
    out
}

/// Render the Warnings modal: the text behind the header `⚠ N` badge.
///
/// It lists the warnings the daemon currently reports. A multi-line one longer
/// than [`WARNING_MAX_LINES`] + 1 lines is cut to [`WARNING_MAX_LINES`] lines of
/// text with a count of the rest.
///
/// Content-sized like [`draw_help`](crate::ui::help_modal::draw_help) — the
/// warnings are few and short-lived, so the popup is sized to them rather than
/// growing a scroll position of its own.
/// On a terminal too short for the close hint, the hint goes first (the footer
/// still shows Esc); only when the warnings themselves do not fit does the last
/// row say so. That marker is in the body, not the border title as in
/// [`draw_help`](crate::ui::help_modal::draw_help): a title marker needs a popup
/// at least ~38 columns wide and is silently dropped on a narrower one, which
/// would leave cut-off warning text unmarked.
pub fn draw_warnings(f: &mut Frame, area: Rect, warnings: &[String], theme: &Theme) {
    grey_overlay(f);
    let body: Vec<Line> = if warnings.is_empty() {
        // The daemon can clear its warnings while the popup is open.
        vec![Line::from(Span::styled(
            "No active warnings.",
            Style::default().fg(theme.muted),
        ))]
    } else {
        warnings
            .iter()
            .flat_map(|w| warning_lines(w, theme))
            .collect()
    };
    let mut with_hint = body.clone();
    with_hint.push(Line::raw(""));
    with_hint.push(Line::from(Span::styled(
        "Esc / Enter / ! close",
        Style::default().fg(theme.muted),
    )));

    let width = centered_width(70, 80, area);
    let rows_for = |lines: &[Line]| {
        let p = Paragraph::new(lines.to_vec()).wrap(Wrap { trim: false });
        popup_height_for(p.line_count(width.saturating_sub(2)))
    };
    let needed = rows_for(&with_hint);
    let popup = centered_rect(70, 100, 80, needed, area);
    let (lines, popup, truncated) = if needed <= popup.height {
        (with_hint, popup, false)
    } else {
        let needed = rows_for(&body);
        let popup = centered_rect(70, 100, 80, needed, area);
        (body, popup, needed > popup.height)
    };

    let title = if warnings.is_empty() {
        "Warnings".to_string()
    } else {
        format!("Warnings ({})", warnings.len())
    };
    let inner = draw_popup_frame(f, popup, &title, theme);
    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    if truncated && inner.height > 0 {
        let body_area = Rect::new(inner.x, inner.y, inner.width, inner.height - 1);
        f.render_widget(p, body_area);
        let marker = Paragraph::new(Line::from(Span::styled(
            "… truncated — resize",
            Style::default().fg(theme.warn),
        )));
        let marker_row = Rect::new(inner.x, inner.y + body_area.height, inner.width, 1);
        f.render_widget(marker, marker_row);
    } else {
        f.render_widget(p, inner);
    }
}

#[cfg(test)]
mod tests {
    use crate::ui::tests::{bottom_border_row, rows};
    use crate::ui::theme::Theme;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    #[test]
    fn warnings_modal_lists_every_warning_with_a_count() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec![
            "amd-smi unavailable (alpha)".to_string(),
            "vram attribution: beta".to_string(),
            "vllm scrape: gamma".to_string(),
        ];
        let area = Rect::new(0, 0, 80, 30);
        let painted = rows(80, 30, |f| {
            super::draw_warnings(f, area, &warnings, &theme);
        })
        .join("\n");
        for w in &warnings {
            assert!(painted.contains(w.as_str()), "missing {w:?}:\n{painted}");
        }
        assert!(painted.contains("Warnings (3)"), "{painted}");
    }

    #[test]
    fn warnings_modal_does_not_clip_a_long_warning_on_a_narrow_terminal() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec![
            "amd-smi is missing, unresolvable, or failed to run, even though a GPU was \
             detected by other means - it may not be supported on this platform yet"
                .to_string(),
        ];
        let area = Rect::new(0, 0, 40, 30);
        let painted = rows(40, 30, |f| {
            super::draw_warnings(f, area, &warnings, &theme);
        })
        .join("\n");
        assert!(
            painted.contains("platform yet"),
            "the tail of the warning must survive wrapping:\n{painted}"
        );
    }

    #[test]
    fn warnings_modal_says_so_when_the_warnings_have_cleared() {
        let theme = Theme::from_name("default-dark");
        let area = Rect::new(0, 0, 80, 30);
        let painted = rows(80, 30, |f| super::draw_warnings(f, area, &[], &theme)).join("\n");
        assert!(painted.contains("No active warnings."), "{painted}");
    }

    #[test]
    fn warnings_modal_marks_truncation_when_the_terminal_is_too_short() {
        let theme = Theme::from_name("default-dark");
        let warnings: Vec<String> = (0..12).map(|i| format!("warning number {i}")).collect();
        // 80 columns is wide enough for a title marker; 40 is not. The cut must
        // be marked either way, since the warnings are what the popup is for.
        for (w, h) in [(80u16, 8u16), (40, 8)] {
            let area = Rect::new(0, 0, w, h);
            let painted = rows(w, h, |f| super::draw_warnings(f, area, &warnings, &theme));
            let bottom = bottom_border_row(&painted);
            assert!(
                painted[bottom - 1].contains("truncated"),
                "{w}x{h}: the marker takes the last inner row:\n{}",
                painted.join("\n")
            );
            assert!(
                painted[bottom - 2].contains("warning number"),
                "{w}x{h}: the marker must not overwrite the last body row:\n{}",
                painted.join("\n")
            );
        }
    }

    #[test]
    fn warnings_modal_does_not_mark_truncation_when_everything_fits() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec!["only one".to_string()];
        let area = Rect::new(0, 0, 80, 30);
        let painted = rows(80, 30, |f| super::draw_warnings(f, area, &warnings, &theme)).join("\n");
        assert!(!painted.contains("truncated"), "{painted}");
        assert!(painted.contains("Esc / Enter / ! close"), "{painted}");
    }

    #[test]
    fn warnings_modal_drops_the_close_hint_before_it_cuts_a_warning() {
        let theme = Theme::from_name("default-dark");
        let warnings: Vec<String> = (1..=3).map(|i| format!("warning {i}")).collect();
        // Five rows: three of warnings plus the borders. The blank row and the
        // close hint do not fit, and are the only things that should go.
        let area = Rect::new(0, 0, 80, 5);
        let painted = rows(80, 5, |f| super::draw_warnings(f, area, &warnings, &theme)).join("\n");
        for w in &warnings {
            assert!(painted.contains(w.as_str()), "{painted}");
        }
        assert!(!painted.contains("Esc / Enter"), "{painted}");
        assert!(!painted.contains("truncated"), "nothing was cut: {painted}");
    }

    /// Paint one warning of `n` lines (`row 1` .. `row n`, after a first line) at a
    /// size with room to spare, followed by a second warning.
    fn painted_multi_line_warning(n: usize) -> String {
        let theme = Theme::from_name("default-dark");
        let mut lines = vec!["first line".to_string()];
        lines.extend((1..n).map(|i| format!("row {i}")));
        let warnings = vec![lines.join("\n"), "the warning after it".to_string()];
        let area = Rect::new(0, 0, 80, 40);
        rows(80, 40, |f| super::draw_warnings(f, area, &warnings, &theme)).join("\n")
    }

    #[test]
    fn warnings_modal_cuts_a_long_multi_line_warning_to_four_lines_and_counts_the_rest() {
        // 31 lines: the first plus 30 of stderr.
        let painted = painted_multi_line_warning(31);
        for shown in ["• first line", "  row 1", "  row 2", "  row 3"] {
            assert!(painted.contains(shown), "{shown:?}:\n{painted}");
        }
        assert!(!painted.contains("row 4"), "{painted}");
        assert!(painted.contains("… 27 more line(s)"), "{painted}");
        assert!(painted.contains("the warning after it"), "{painted}");
    }

    #[test]
    fn warnings_modal_shows_a_warning_of_up_to_five_lines_whole() {
        // Four lines fit the cap; five would hide one line behind a row that
        // costs the same, so it is shown whole too.
        for n in [4usize, 5] {
            let painted = painted_multi_line_warning(n);
            assert!(
                painted.contains(&format!("row {}", n - 1)),
                "{n}:\n{painted}"
            );
            assert!(!painted.contains("more line(s)"), "{n}:\n{painted}");
        }
        let six = painted_multi_line_warning(6);
        assert!(!six.contains("row 4"), "{six}");
        assert!(six.contains("… 2 more line(s)"), "{six}");
    }

    #[test]
    fn warnings_modal_lists_a_blank_warning_as_a_bare_bullet() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec![String::new(), "second".to_string()];
        let area = Rect::new(0, 0, 80, 20);
        let painted = rows(80, 20, |f| super::draw_warnings(f, area, &warnings, &theme));
        let bullets = painted.iter().filter(|r| r.contains('•')).count();
        assert_eq!(
            bullets,
            2,
            "one bullet per warning:\n{}",
            painted.join("\n")
        );
    }

    // Warning text is often raw amd-smi stderr. Nothing in this crate strips it:
    // ratatui drops every grapheme holding a control character when it writes the
    // buffer, so an escape sequence cannot reach the terminal. This pins that.
    #[test]
    fn warnings_modal_paints_no_control_character_from_a_warning() {
        let theme = Theme::from_name("default-dark");
        let warnings = vec![
            "\x1b[31mred\x1b[0m\x07 bell\ttab\rreturn\x1b]0;title\x07\ndel\x7f after".to_string(),
        ];
        let area = Rect::new(0, 0, 80, 20);
        let backend = TestBackend::new(80, 20);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| super::draw_warnings(f, area, &warnings, &theme))
            .unwrap();
        let buf = term.backend().buffer();
        let painted: String = (0..20)
            .flat_map(|y| (0..80).map(move |x| (x, y)))
            .map(|pos| buf[pos].symbol().to_string())
            .collect();
        assert!(
            !painted.chars().any(char::is_control),
            "a control character reached the buffer: {painted:?}"
        );
        assert!(painted.contains("red"), "{painted:?}");
        assert!(
            painted.contains("after"),
            "the second line must survive: {painted:?}"
        );
    }
}
