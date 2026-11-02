// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The Help modals: the contextual per-tab `?` help (`draw_help`) and the
//! global two-column keyboard reference (`draw_global_help`). Both are sized
//! to their wrapped content rather than to a share of the screen, and
//! `popup_height_for` is the one place that sizing arithmetic lives, shared
//! with the warnings modal.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Wrap};

use crate::app::ActiveTab;
use crate::ui::panel::{self, centered_rect, centered_width, draw_popup_frame, grey_overlay};
use crate::ui::theme::Theme;

/// Render the Help modal for the active tab.
///
/// The popup is sized to the height its *wrapped* content actually needs,
/// clamped to `area`, rather than to a fixed share of the screen. A fixed share
/// silently truncated: at the 80x24 the e2e lane pins, the body is 20 rows, 70%
/// of that is 14, and two of the key hints wrap to a second line — so the modal
/// ended mid-list at `{ / }` and the per-tab guidance below it was never drawn,
/// with no scrollbar or indicator to say so. Content-sizing keeps every hint on
/// screen wherever the room exists, and adding a hint can no longer push an
/// unrelated one off the bottom.
///
/// Where the room does *not* exist — a terminal too short for the hint list even
/// at full height — the content is still cut, and the modal does not scroll.
/// [`help_title`] marks the title in that case rather than leaving the user to
/// guess, which is the whole of what is claimed here: an honest indicator, not a
/// guarantee that everything is visible.
pub(super) fn draw_help(f: &mut Frame, area: Rect, tab: ActiveTab, theme: &Theme) {
    grey_overlay(f);
    let mut lines: Vec<Line> = vec![
        key_line("q", "quit", theme),
        // Ctrl-C is a first-class quit gesture in both key loops (it restores the
        // terminal and exits 130), so it belongs on the help surface next to `q`.
        // The exception is worth stating: over a *running* job console it still
        // means "cancel this job".
        key_line("Ctrl-C", "quit (cancels a running job console)", theme),
        key_line("?", "toggle this help", theme),
        key_line("Tab / Shift-Tab", "next / previous tab", theme),
        key_line("1 .. 5", "jump to tab", theme),
        key_line("t", "open theme picker", theme),
        key_line("!", "show warnings (header ⚠ badge)", theme),
        key_line("Space", "pause / resume (replay only)", theme),
        key_line("+ / -", "speed up / slow down (replay only)", theme),
        key_line("[ / ]", "jump ±10s (replay only)", theme),
        key_line("{ / }", "jump ±60s (replay only)", theme),
        Line::raw(""),
    ];
    let tab_help: &[(&str, &str)] = match tab {
        ActiveTab::Home => &[("(no tab-specific keys — see the ROCm / Serving tabs)", "")],
        ActiveTab::Rocm | ActiveTab::Serving => &[
            ("j / k  ↑ / ↓", "select an action"),
            ("→ / Enter", "open it in Details (asks before mutating)"),
            ("←", "Details preview → Actions list"),
            ("Esc", "close an open manager (back to Actions)"),
        ],
        ActiveTab::Observe => &[
            ("j / Down", "select next instance"),
            ("k / Up", "select previous instance"),
            ("g / Home", "first instance"),
            ("G / End", "last instance"),
            ("Enter", "open instance detail"),
            ("s", "services manager"),
            (
                "w / e / d / u / i / l",
                "serve / engines / doctor / update / install / logs",
            ),
        ],
        ActiveTab::Chat => &[
            ("y / Enter", "accept the detected endpoint (consent prompt)"),
            ("n", "decline / disable chat"),
            ("d", "detect a local engine (gate)"),
            (
                "y / s / n",
                "detected engine: use now / use & save / dismiss",
            ),
            ("i / Enter", "focus the input (insert mode, once enabled)"),
            ("Esc", "leave insert mode"),
            ("Enter", "send the message (while focused)"),
            ("Backspace", "delete a character (while focused)"),
        ],
    };
    lines.push(Line::from(Span::styled(
        format!("— {tab:?} tab —"),
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    )));
    for (k, desc) in tab_help {
        lines.push(key_line(k, desc, theme));
    }

    let p = Paragraph::new(lines).wrap(Wrap { trim: false });

    // Width first — it does not depend on the height — then ask the paragraph
    // how many rows it wraps to at that width. `line_count` is the renderer's
    // own wrap, not an estimate of it, so this cannot drift from what lands on
    // screen. `pct_y = 100` with `max_h = needed` means "exactly the content,
    // or the whole area when the content is taller than it". `+ 2` is the
    // popup's top and bottom border rows; `- 2` on the width is the left and
    // right ones.
    //
    // `line_count` sits behind ratatui's `unstable-rendered-line-info` feature
    // (already enabled in this crate's Cargo.toml for the scroll bounds). It is
    // now load-bearing for *layout*, not just for how far a scrollbar may
    // travel: if a ratatui release changes or withdraws it, this modal silently
    // mis-sizes rather than merely mis-scrolling.
    // `help_modal_is_sized_to_exactly_its_content_on_every_tab` (in this
    // module's `tests`) is the tripwire for that.
    let width = centered_width(70, 80, area);
    let needed = popup_height_for(p.line_count(width.saturating_sub(2)));
    let popup = centered_rect(70, 100, 80, needed, area);
    let inner = draw_popup_frame(f, popup, &help_title("Help", needed, popup), theme);

    f.render_widget(p, inner);
}

/// Title for a content-sized help popup, marked when the content did not fit.
///
/// `needed` is the height the wrapped content asked for; `popup` is what the
/// area could actually give it. When the popup is shorter, the tail of the
/// content is off-screen — and these modals are *sized*, not scrolled, so there
/// is no scrollbar, no scroll position, and nothing the user can press to see
/// the rest. Saying so is the only honest option left; the alternative is the
/// silent truncation this module's doc comments call the original defect.
///
/// The marker goes in the border title rather than the body because a body
/// marker would cost a content row on precisely the terminal that has none to
/// spare — it would evict a hint to announce that hints were evicted.
///
/// Residual, stated rather than papered over: [`panel::title_fits`] is false on
/// a popup too narrow for the marked form, and a title that does not fit is
/// dropped rather than overflowing the border. Rather than lose the plain
/// "Help" as well, the bare title is used there and the truncation goes
/// unmarked. That is a popup under ~30 columns, i.e. a terminal well below
/// anything the dashboard lays out usefully.
fn help_title(base: &str, needed: u16, popup: Rect) -> String {
    let marked = format!("{base} (truncated — resize)");
    if needed > popup.height && panel::title_fits(popup.width, &marked) {
        marked
    } else {
        base.to_string()
    }
}

/// Popup height that shows `content_rows` rows of wrapped content: the content
/// plus the popup's top and bottom border rows. Shared by [`draw_help`],
/// [`draw_global_help`] and
/// [`draw_warnings`](crate::ui::warnings_modal::draw_warnings) so the
/// content-sized modals cannot drift apart.
pub(super) fn popup_height_for(content_rows: usize) -> u16 {
    u16::try_from(content_rows)
        .unwrap_or(u16::MAX)
        .saturating_add(2)
}

fn key_line<'a>(key: &'a str, desc: &'a str, theme: &Theme) -> Line<'a> {
    Line::from(vec![
        Span::styled(
            format!("  {key:<18} "),
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(desc, Style::default().fg(theme.fg)),
    ])
}

/// Global 2-column keyboard reference (NAVIGATE / OVERLAYS / ACTIONS / CHAT /
/// GLOBAL). Distinct from the contextual per-tab `?` help (`draw_help`).
///
/// Content-sized exactly like [`draw_help`], and for the same reason. The old
/// fixed `centered_rect(80, 80, 100, 26, area)` shape truncated silently, with
/// no scrollbar and no indicator: at the 80x24 the e2e lane pins it had *zero*
/// rows of margin (14 available, 14 needed), so the Ctrl-C hint this PR adds to
/// the right-hand column landed on the last usable row and one more hint — or
/// one more wrap, from a wording change — would have pushed it off. On a roomy
/// terminal the same fixed shape drew ten blank rows below the content.
///
/// Shares [`help_title`] with [`draw_help`], so the residual case — a terminal
/// too short even for the content-sized modal — is marked here the same way and
/// under the same caveat.
///
/// Holds no row for `!` (warnings): at the pinned 80x20 this card has no row to
/// spare and one more is silently cut. Like the replay keys it is listed in the
/// contextual [`draw_help`], and the header badge advertises it.
pub(super) fn draw_global_help(f: &mut Frame, area: Rect, theme: &Theme) {
    grey_overlay(f);
    let left: &[(&str, &[(&str, &str)])] = &[
        (
            "NAVIGATE",
            &[
                ("Tab / ⇧Tab", "next / prev tab"),
                ("1 .. 5", "jump to tab"),
                ("j / k", "select"),
            ],
        ),
        (
            "OVERLAYS",
            &[
                ("Esc", "main menu"),
                (":", "command palette"),
                ("? ", "this help"),
                ("t", "theme picker"),
            ],
        ),
    ];
    let right: &[(&str, &[(&str, &str)])] = &[
        (
            "ACTIONS",
            &[
                ("w / e / d", "serve / engines / doctor"),
                ("u / i / l", "update / install / logs"),
                ("Enter", "open / detail"),
            ],
        ),
        (
            "CHAT / GLOBAL",
            &[
                ("i / Enter", "focus chat input"),
                ("q", "quit"),
                ("Ctrl-C", "quit (cancels a running job)"),
            ],
        ),
    ];
    let left_p = Paragraph::new(help_group_lines(left, theme)).wrap(Wrap { trim: false });
    let right_p = Paragraph::new(help_group_lines(right, theme)).wrap(Wrap { trim: false });

    // Width first, then the height the *taller* column wraps to at the width
    // the 50/50 split will actually hand it. The split is run here on a stand-in
    // rect rather than the ratio being re-derived by hand, so the measurement
    // cannot drift from the render below.
    let width = centered_width(80, 100, area);
    let measure = help_columns(Rect::new(0, 0, width.saturating_sub(2), 1));
    let needed = popup_height_for(
        left_p
            .line_count(measure[0].width)
            .max(right_p.line_count(measure[1].width)),
    );

    let modal = centered_rect(80, 100, 100, needed, area);
    let inner = draw_popup_frame(f, modal, &help_title("Keyboard", needed, modal), theme);
    if inner.height == 0 {
        return;
    }
    f.render_widget(Clear, inner);
    let cols = help_columns(inner);
    f.render_widget(left_p, cols[0]);
    f.render_widget(right_p, cols[1]);
}

/// The two equal columns [`draw_global_help`] lays its groups out in. Returned
/// as an array so the same split serves both the measure and the render.
fn help_columns(area: Rect) -> [Rect; 2] {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    [cols[0], cols[1]]
}

/// Titled key/description groups as renderable lines.
///
/// The blank separator sits *between* groups, not after each one: a trailing
/// blank would make the content-sized popup one row taller than its content and
/// show as a gap above the bottom border.
fn help_group_lines<'a>(
    groups: &'a [(&'a str, &'a [(&'a str, &'a str)])],
    theme: &Theme,
) -> Vec<Line<'a>> {
    let mut lines: Vec<Line> = Vec::new();
    for (i, (title, rows)) in groups.iter().enumerate() {
        if i > 0 {
            lines.push(Line::raw(""));
        }
        lines.push(Line::from(Span::styled(
            *title,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        for (k, desc) in *rows {
            lines.push(key_line(k, desc, theme));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::draw_help;
    use crate::app::ActiveTab;
    use crate::ui::tests::{bottom_border_row, flat, rows};
    use crate::ui::theme::Theme;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    #[test]
    fn draw_help_dims_the_backdrop() {
        // `draw_help` sizes its popup to its content, so on a large area there
        // is backdrop left uncovered around it — the corner is always part of
        // that backdrop, not the popup. Before `draw_help` called
        // `grey_overlay`, that corner kept whatever the tab underneath had
        // painted there instead of the dimmed wash every other modal uses.
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(120, 30);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw_help(f, f.area(), ActiveTab::Observe, &theme))
            .unwrap();
        let wash = ratatui::style::Color::Rgb(0x1c, 0x1e, 0x22);
        let corner = term.backend().buffer().cell((0, 0)).unwrap();
        assert_eq!(
            corner.style().bg,
            Some(wash),
            "help modal must dim its backdrop like every other modal"
        );
    }

    /// The help modal must show ALL of its content at the smallest geometry the
    /// product supports, not as much of it as a fixed share of the screen
    /// happens to fit.
    ///
    /// Regression: adding the Ctrl-C hint pushed the per-tab guidance off the
    /// bottom at 80x24 and nothing said so — the modal just ended mid-list at
    /// `{ / }`. That reached CI as an e2e failure on an assertion about
    /// unrelated text ("Home tab"), because no unit test asserted the modal's
    /// last row was reachable.
    ///
    /// 80x20 is the *body* rect `ui::draw` hands `draw_help` on the 80x24 the
    /// e2e lane pins (3-row header, 1-row footer), so this is the real
    /// worst-case geometry rather than an invented one.
    #[test]
    fn help_modal_shows_every_hint_at_the_minimum_supported_geometry() {
        use crate::app::ActiveTab;
        let theme = Theme::from_name("default-dark");
        let area = Rect::new(0, 0, 80, 20);
        let backend = TestBackend::new(80, 20);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| super::draw_help(f, area, ActiveTab::Home, &theme))
            .unwrap();
        let out = flat(&term);

        // First hint, the hint that wraps, the last global hint, and the per-tab
        // section that used to fall off the bottom. The last two are the ones
        // that regress when the modal is sized to anything but its content.
        for needle in [
            "quit",
            "Ctrl-C",
            "toggle this help",
            "next / previous tab",
            "show warnings",
            "jump ±60s",
            "Home tab",
            "no tab-specific keys",
        ] {
            assert!(
                out.contains(needle),
                "help modal truncated before {needle:?} at 80x20:\n{out}"
            );
        }
    }

    /// The popup is sized to *exactly* its wrapped content whenever the area has
    /// room: the last hint lands on the row directly above the bottom border,
    /// with no filler row after it and nothing evicted before it.
    ///
    /// This is the property the crate actually owns — `needed` is computed here
    /// from `Paragraph::line_count` and handed to `centered_rect` as `max_h`.
    /// (That a popup cannot overrun its area is ratatui's own guarantee:
    /// `Layout::split` structurally cannot emit a segment larger than its input,
    /// so asserting it here would constrain no branch of this module.)
    ///
    /// Checked on every tab, because each has a different hint list and a
    /// different set of lines that wrap; the sibling test above pins a single
    /// geometry on a single tab.
    #[test]
    fn help_modal_is_sized_to_exactly_its_content_on_every_tab() {
        use crate::app::ActiveTab;
        let theme = Theme::from_name("default-dark");
        // (tab, the tail of that tab's LAST hint — the text that must land on
        // the row immediately above the bottom border). Several of these are
        // wrap continuations, which is the point: the sizing has to account for
        // the renderer's own wrapping, not the unwrapped line count.
        let cases = [
            (ActiveTab::Home, "Serving tabs)"),
            (ActiveTab::Rocm, "Actions)"),
            (ActiveTab::Serving, "Actions)"),
            (ActiveTab::Observe, "update / install / logs"),
            (ActiveTab::Chat, "focused)"),
        ];
        for (tab, tail) in cases {
            // Tall enough that every tab's help fits with room to spare, so any
            // mismatch is the sizing arithmetic and not the clamp.
            let area = Rect::new(0, 0, 80, 44);
            let painted = rows(80, 44, |f| super::draw_help(f, area, tab, &theme));
            let bottom = bottom_border_row(&painted);
            let last = &painted[bottom - 1];
            assert!(
                last.contains(tail),
                "{tab:?}: the last hint must sit directly above the bottom \
                 border, but row {} is {last:?} (expected it to contain \
                 {tail:?}) — the popup is over- or under-sized for its \
                 content:\n{}",
                bottom - 1,
                painted.join("\n")
            );
        }
    }

    /// When the content genuinely cannot fit, the modal takes **every** row of
    /// the area rather than a fixed share of it, so the most hints the terminal
    /// can hold are shown. This is what `pct_y = 100` buys over the old 70%.
    #[test]
    fn help_modal_fills_every_available_row_when_the_content_cannot_fit() {
        use crate::app::ActiveTab;
        let theme = Theme::from_name("default-dark");
        // Chat has the longest hint list; at 10 rows it cannot fit by a wide
        // margin, so the popup is clamp-bound rather than content-bound.
        let area = Rect::new(0, 0, 80, 10);
        let painted = rows(80, 10, |f| {
            super::draw_help(f, area, ActiveTab::Chat, &theme);
        });
        assert!(
            painted[0].contains('╭'),
            "the modal must start on the first row of the area:\n{}",
            painted.join("\n")
        );
        assert_eq!(
            bottom_border_row(&painted),
            9,
            "the modal must end on the last row of the area, using all 10 rows \
             rather than a fixed share of them:\n{}",
            painted.join("\n")
        );
        // Eight inner rows, so eight wrapped hint rows: `q`, the two the Ctrl-C
        // hint wraps to, `?`, Tab, `1 .. 5`, `t`, `!`. A 70% share would stop
        // five rows in, at `next / previous tab`.
        assert!(
            painted[8].contains("show warnings"),
            "a full-height modal must paint every row it has:\n{}",
            painted.join("\n")
        );
    }

    /// The global keyboard reference is content-sized the same way `draw_help`
    /// is: no filler rows on a roomy terminal, and — the regression that
    /// matters — nothing silently evicted at the smallest geometry the product
    /// supports, where the fixed 80%-of-area shape had zero rows of margin and
    /// this PR added a hint to it.
    #[test]
    fn global_help_is_sized_to_its_content_and_evicts_nothing() {
        let theme = Theme::from_name("default-dark");
        // 80x20 is the body rect on the 80x24 the e2e lane pins; 120x30 is a
        // roomy terminal, where a fixed 26-row modal left ten blank rows.
        for (w, h) in [(80u16, 20u16), (120, 30)] {
            let area = Rect::new(0, 0, w, h);
            let painted = rows(w, h, |f| super::draw_global_help(f, area, &theme));
            let bottom = bottom_border_row(&painted);
            let last = painted[bottom - 1].trim_end();
            // Two separate properties, in order. The first is *not* a
            // no-filler check: a blank row inside a bordered popup paints as
            // `│      │` and ends with '│' too, so `ends_with` can never tell
            // filler from content. What it does establish is that the row above
            // the bottom border is an interior row — a body row between the two
            // side borders, rather than the popup's own top border (`╭───╮`,
            // which ends with '╮') as it would be on a two-row popup. The
            // second assertion is the one that rules out filler.
            assert!(
                last.ends_with('│'),
                "{w}x{h}: the row above the bottom border must be a body row \
                 between the side borders, but row {} is {last:?}:\n{}",
                bottom - 1,
                painted.join("\n")
            );
            assert!(
                last.trim_matches(['│', ' ']).chars().count() > 0,
                "{w}x{h}: the row above the bottom border is blank — the popup \
                 is taller than its content:\n{}",
                painted.join("\n")
            );
            let flat = painted.join("");
            for needle in [
                "NAVIGATE",
                "OVERLAYS",
                "ACTIONS",
                "CHAT / GLOBAL",
                "main menu",
                "cancels a running",
            ] {
                assert!(
                    flat.contains(needle),
                    "{w}x{h}: the keyboard reference is truncated before \
                     {needle:?}:\n{}",
                    painted.join("\n")
                );
            }
        }
    }

    /// When the content still cannot fit at full height, the modal says so.
    ///
    /// It is sized, not scrolled: there is no scrollbar and nothing to press, so
    /// an unmarked short modal is indistinguishable from a complete one. The
    /// marker lives in the border title, which costs no content row.
    #[test]
    fn help_modals_mark_the_title_when_the_content_still_cannot_fit() {
        use crate::app::ActiveTab;
        let theme = Theme::from_name("default-dark");
        let painted = |w: u16, h: u16, global: bool| -> String {
            let area = Rect::new(0, 0, w, h);
            rows(w, h, |f| {
                if global {
                    super::draw_global_help(f, area, &theme);
                } else {
                    super::draw_help(f, area, ActiveTab::Chat, &theme);
                }
            })
            .join("\n")
        };

        // Ten rows holds neither hint list — the sibling test above pins that
        // `draw_help` is clamp-bound here — so both must be marked.
        for global in [false, true] {
            let out = painted(80, 10, global);
            assert!(
                out.contains("truncated"),
                "a modal that cut its content must say so (global={global}):\n{out}"
            );
        }
        // Roomy: every hint fits, so the marker must NOT appear — it would be a
        // false alarm on the geometry the product is normally used at.
        for (w, h, global) in [(80u16, 44u16, false), (80, 20, true), (120, 30, true)] {
            let out = painted(w, h, global);
            assert!(
                !out.contains("truncated"),
                "{w}x{h}: a modal that fits must not claim it was truncated \
                 (global={global}):\n{out}"
            );
        }
    }
}
