// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The dashboard footer: the key-hint chips, the Esc chip's context-dependent
//! label, and the chip regions the mouse hit-testing resolves against.
//!
//! Split out of `ui/mod.rs`; `draw_footer` returns the clickable chips so the
//! caller can record them in `last_footer_chips` for mouse hit-testing.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::Paragraph;

use crate::app::{ActiveTab, AppState, ChatConsent, FooterChip, KeyAction, Modal, PaneFocus};
use crate::ui::panel;
use crate::ui::theme::Theme;

/// One footer-legend segment: a key chip (optionally clickable) or plain text.
enum Seg {
    /// A keycap. `Some(action)` => left-clicking it dispatches that action and
    /// the cap is highlighted to signal it is interactive; `None` => a display
    /// keycap (e.g. the `1–4` range) that has no single click target.
    Key(&'static str, Option<KeyAction>),
    /// A plain text separator / label.
    Sep(&'static str),
}

/// Draw the footer legend and return the clickable chip geometry for hit-testing.
pub(super) fn draw_footer(
    f: &mut Frame,
    area: Rect,
    state: &AppState,
    theme: &Theme,
) -> Vec<FooterChip> {
    let is_action_tab = matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving);
    let enter_action = if is_action_tab {
        KeyAction::PaneActivate
    } else {
        KeyAction::OpenDetail
    };
    // "Does a global hotkey reach its normal `handle_key` arm right now, or
    // does something else own the keyboard first?" is answered by
    // `AppState::global_hotkeys_reachable` (the header warning badge gates on it
    // too); its terms are split out here because the footer needs subsets of
    // them separately (`q`'s label and clickability). It mirrors, in order,
    // `event_loop`'s own dispatch: a manager overlay or pending approval owns
    // every key before `handle_key` is ever called (`overlay_or_approval`); a
    // plain `Modal::*` fully returns from its own arm before reaching the
    // global match (`modal_open`); and
    // focused chat text entry absorbs every key except PageUp/PageDown
    // (`chat_typing`, see `handle_key`'s `chat.focused` branch). Any footer
    // chip whose click dispatches a *global* hotkey action must gate on this,
    // or the click can do something the key it displays never could.
    let overlay_or_approval = state.overlay_or_approval();
    let modal_open = state.modal != Modal::None;
    let chat_typing = state.active_tab == ActiveTab::Chat && state.chat_focused;
    let hotkeys_owned_elsewhere = !state.global_hotkeys_reachable();
    let mut segs: Vec<Seg> = vec![
        // A real Tab keypress reaches `SwitchTab` only in that same idle
        // state: every modal arm (including `Options`, which remaps Tab to
        // `OptionsTab`) returns before the global match, a manager overlay
        // owns keys entirely, and focused chat text entry consumes Tab as
        // `Nothing` rather than falling through.
        Seg::Key(
            "Tab",
            (!hotkeys_owned_elsewhere).then_some(KeyAction::SwitchTab(state.active_tab.next())),
        ),
        Seg::Sep(" next  "),
        Seg::Key("1–5", None),
        Seg::Sep(" jump  "),
    ];
    // Exactly one Esc chip is shown at all times, and it must match what Esc
    // actually does — the real routing priority (highest first) is: the
    // quit-confirm prompt (issue #145) or a pending chat approval owns every
    // key; then an open manager overlay backs itself out; then a `Modal::*`
    // overlay closes; then a focused/gating Chat tab absorbs Esc; only once
    // none of those apply does Esc fall through to the uniform "menu"
    // fallback (item #35). Mirror that order here so the chip never
    // advertises `menu` while the Esc *key* would actually do something else.
    // A click on the chip itself can no longer disagree with this label for
    // quit-confirm/approval: `resolve_mouse`'s `blocks_body_absolutely()`
    // guard swallows a footer-chip click there before `footer_chip_hit` ever
    // resolves one (issue #145) — this label logic still has to track the
    // manager-overlay case, where a chip click does dispatch.
    if state.blocks_body_absolutely() {
        segs.push(Seg::Key("Esc", None));
        segs.push(Seg::Sep(" cancel  "));
    } else if state.has_open_overlay() && state.active_overlay_at_root() {
        segs.push(Seg::Key("Esc", None));
        segs.push(Seg::Sep(" back out  "));
    } else if state.has_open_overlay()
        && let Some(id) = state.active_job_id()
    {
        let closes = crate::ui::job_console::console_esc_closes(state.jobs.job(id));
        // A manager's job console is showing a job — `closes` mirrors
        // `on_console_key`'s own terminality check (`console_esc_closes`) so
        // the label can't drift from what Esc actually does, matched once
        // here instead of as two separate (and negated) lookups.
        segs.push(Seg::Key("Esc", None));
        if closes {
            // Still running — Esc fully closes the overlay (the job keeps
            // running in the background), matching the console's own footer
            // hint ("Esc close (keeps running)"), not the generic sub-popup
            // "cancel" below.
            segs.push(Seg::Sep(" close  "));
        } else {
            // Finished (or vanished) — `on_console_key` only dismisses the
            // console back to the screen body (the overlay itself stays
            // open); nothing is left to cancel, matching the console's own
            // reworded footer hint ("Enter/Esc dismiss", see
            // `job_console::draw`). This must not fall through to the
            // generic sub-popup "cancel" arm below, which would otherwise
            // contradict that hint.
            segs.push(Seg::Sep(" dismiss  "));
        }
    } else if state.has_open_overlay() {
        // A manager is open but not at its root layer (sub-popup, picker, or
        // approval) — Esc is handled by that layer's own event-loop arm, not
        // by `should_pane_back_out`/`OpenMenu`. `None` keeps the chip
        // non-clickable so it can't dispatch the wrong action.
        segs.push(Seg::Key("Esc", None));
        segs.push(Seg::Sep(" cancel  "));
    } else if state.modal != Modal::None {
        segs.push(Seg::Key("Esc", Some(KeyAction::CloseModal)));
        segs.push(Seg::Sep(" close  "));
    } else if state.active_tab == ActiveTab::Chat
        && state.chat_detect_offer.is_some()
        && state.chat_consent != ChatConsent::Accepted
    {
        segs.push(Seg::Key("Esc", Some(KeyAction::ChatDetectDismiss)));
        segs.push(Seg::Sep(" dismiss  "));
    } else if state.active_tab == ActiveTab::Chat && state.chat_focused {
        segs.push(Seg::Key("Esc", Some(KeyAction::ChatBlur)));
        segs.push(Seg::Sep(" unfocus  "));
    } else {
        // Dispatch `PaneEscape`, not a hardcoded `OpenMenu` — `apply_action`
        // resolves `PaneEscape` against `pane_focus` exactly as a real
        // keypress does (Details → Actions on Rocm/Serving, else the menu),
        // so the chip can't promise "menu" when the key would actually just
        // step the pane back out.
        let steps_out_of_detail = is_action_tab && state.pane_focus == PaneFocus::Detail;
        segs.push(Seg::Key("Esc", Some(KeyAction::PaneEscape)));
        segs.push(Seg::Sep(if steps_out_of_detail {
            " back  "
        } else {
            " menu  "
        }));
        if matches!(
            state.active_tab,
            ActiveTab::Observe | ActiveTab::Rocm | ActiveTab::Serving
        ) {
            segs.push(Seg::Key("j/k", Some(KeyAction::Move(1))));
            segs.push(Seg::Sep(" select  "));
            segs.push(Seg::Key("Enter", Some(enter_action)));
            segs.push(Seg::Sep(if is_action_tab {
                " open  "
            } else {
                " detail  "
            }));
        }
    }
    // Guided-action letter hotkeys — Observe only (telemetry quick-jumps). On
    // ROCm/Serving the Actions list is the single path, so no letter chips.
    //
    // Gated on `overlay_or_approval || modal_open`, not just the former: a
    // real w/e/d/u/i/l/s/b keypress can't reach these `Open*` actions while a
    // manager overlay owns the event loop, a chat approval is pending, *or* a
    // plain `Modal::*` is open (every modal arm in `handle_key` fully returns
    // before this letter-hotkey match, none of them match these letters), but
    // a footer-chip click bypasses all three. Each `Open*` handler starts
    // with `close_overlays()`, which would silently discard whatever overlay,
    // approval, or modal was open — the same class of bug the t/?/q gating
    // below exists to prevent. (`chat_typing` doesn't apply: this whole block
    // is already tab-gated to Observe, and `chat_typing` can only be true on
    // the Chat tab.)
    if state.active_tab == ActiveTab::Observe && !overlay_or_approval && !modal_open {
        segs.push(Seg::Key("w", Some(KeyAction::OpenServeWizard)));
        segs.push(Seg::Sep(" serve  "));
        segs.push(Seg::Key("e", Some(KeyAction::OpenEngineManager)));
        segs.push(Seg::Sep(" engines  "));
        segs.push(Seg::Key("d", Some(KeyAction::OpenExamine)));
        segs.push(Seg::Sep(" examine  "));
        segs.push(Seg::Key("u", Some(KeyAction::OpenUpdate)));
        segs.push(Seg::Sep(" update  "));
        segs.push(Seg::Key("i", Some(KeyAction::OpenInstall)));
        segs.push(Seg::Sep(" install  "));
        segs.push(Seg::Key("l", Some(KeyAction::OpenLogs)));
        segs.push(Seg::Sep(" logs  "));
        segs.push(Seg::Key("s", Some(KeyAction::OpenServices)));
        segs.push(Seg::Sep(" services  "));
        segs.push(Seg::Key("b", Some(KeyAction::OpenBenchRun)));
        segs.push(Seg::Sep(" bench  "));
    }
    // Same reachability argument as the Tab chip above: a real Space/+/-
    // keypress only reaches `ReplayTogglePause`/`ReplaySpeedUp` when nothing
    // else — overlay, approval, modal, or focused chat text entry — owns the
    // key first.
    if state.replay.is_some() && !hotkeys_owned_elsewhere {
        segs.push(Seg::Key("Space", Some(KeyAction::ReplayTogglePause)));
        segs.push(Seg::Sep(" pause  "));
        segs.push(Seg::Key("+/-", Some(KeyAction::ReplaySpeedUp)));
        segs.push(Seg::Sep(" speed  "));
    }
    // While a manager overlay is open it owns every key (the event loop routes
    // each keypress to its `on_key`, never falling through to `apply_action`),
    // so a real `t`/`?`/`q` press can't reach
    // `OpenThemePicker`/`ToggleHelp`/`KeyAction::Quit` there — it cancels the
    // approval, closes the job console, or backs the manager out instead, but
    // it never opens the theme picker, toggles help, or tears down the app the
    // way those actions do. The same is true while an approval is pending:
    // `has_open_overlay()` deliberately excludes `approval` (it's a separate
    // gating layer), so `approval_pending()` has to be checked here too, or
    // these chips would stay clickable and let a mouse click silently discard
    // a pending approval that a real keypress never could.
    //
    // `t`/`?` additionally need `state.modal` itself checked: `OpenThemePicker`/
    // `ToggleHelp` both call `close_overlays()` then unconditionally overwrite
    // `state.modal`, silently replacing whatever plain `Modal::*` is already
    // open (Detail/Menu/Palette/Options/GlobalHelp) — but `handle_key` routes
    // `t`/`?` differently (or not at all) inside every one of those modals'
    // own arms, so a real keypress never reaches those actions there either.
    // And both need `chat_typing` checked: focused chat text entry consumes
    // any `Char` key (including `t`/`?`) as ordinary input, never as a global
    // hotkey — `hotkeys_owned_elsewhere` covers all three conditions.
    // `q`'s label ("close" vs "quit") only depends on `overlay_or_approval` —
    // that's the one condition under which a real `q` still does something
    // (the overlay's own `on_key` backs it out), so `chat_typing` alone must
    // not relabel it "close" with no overlay to close. But `q`'s
    // *clickability* also has to account for `chat_typing`: while typing, a
    // real `q` types the letter rather than reaching `Quit`, so the chip must
    // go non-clickable there too even though it keeps the "quit" label.
    // `modal_open` doesn't affect `q` either way: every modal arm already
    // maps a real `q` to `Quit` (see `handle_key`).
    if hotkeys_owned_elsewhere {
        segs.push(Seg::Key("t", None));
        segs.push(Seg::Sep(" theme  "));
        segs.push(Seg::Key("?", None));
        segs.push(Seg::Sep(" help  "));
    } else {
        segs.push(Seg::Key("t", Some(KeyAction::OpenThemePicker)));
        segs.push(Seg::Sep(" theme  "));
        segs.push(Seg::Key("?", Some(KeyAction::ToggleHelp)));
        segs.push(Seg::Sep(" help  "));
    }
    if overlay_or_approval {
        segs.push(Seg::Key("q", None));
        segs.push(Seg::Sep(" close"));
    } else {
        segs.push(Seg::Key("q", (!chat_typing).then_some(KeyAction::Quit)));
        segs.push(Seg::Sep(" quit"));
    }

    // Lay out left-to-right, rendering each segment in its own cell span so the
    // recorded chip geometry matches the painted columns exactly.
    let mut cx = area.x;
    let row = area.y;
    let max_x = area.x.saturating_add(area.width);
    let mut chips: Vec<FooterChip> = Vec::new();
    // Keycap surface tracks the theme's luminance (a small nudge off the bg)
    // rather than surface_2 (= br_black), which collides with text on light
    // themes. Clickable caps add UNDERLINE as a non-color affordance so
    // interactivity does not rely on hue alone.
    let cap_bg = panel::blend(theme.bg, theme.fg, 0.12);
    for seg in &segs {
        if cx >= max_x {
            break;
        }
        let (text, key, action): (&str, bool, Option<KeyAction>) = match seg {
            Seg::Key(cap, act) => (cap, true, *act),
            Seg::Sep(sep) => (sep, false, None),
        };
        let body = if key {
            format!(" {text} ")
        } else {
            text.to_string()
        };
        let width = body.chars().count() as u16;
        let draw_w = width.min(max_x - cx);
        let style = match (key, action.is_some()) {
            // Clickable keycap: accent + bold + underline (color-independent cue).
            (true, true) => Style::default()
                .bg(cap_bg)
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            // Display-only keycap (e.g. 1–4): same surface, muted, no underline.
            (true, false) => Style::default().bg(cap_bg).fg(theme.muted),
            // Separator / label.
            _ => Style::default().fg(theme.muted),
        };
        f.render_widget(
            Paragraph::new(Span::styled(body, style)),
            Rect::new(cx, row, draw_w, 1),
        );
        if let Some(act) = action {
            chips.push(FooterChip {
                x0: cx,
                x1: cx + width,
                y: row,
                action: act,
            });
        }
        cx = cx.saturating_add(width);
    }
    chips
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::app::{ActiveTab, ChatConsent, KeyAction, Modal, PaneFocus};

    #[test]
    fn footer_shows_esc_menu_chip_when_no_overlay_open() {
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let state = AppState::new("t".into(), "default-dark".into());
        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        // The chip dispatches `PaneEscape`, matching what a real Esc keypress
        // resolves to via `handle_key`'s catch-all — not a hardcoded
        // `OpenMenu` that would diverge from `apply_action`'s `pane_focus`
        // handling on Rocm/Serving.
        let _ = chips
            .iter()
            .find(|c| c.action == KeyAction::PaneEscape)
            .expect("a fallback Esc chip stepping the pane back out must always be present");
    }

    #[test]
    fn footer_esc_chip_dispatches_pane_escape_when_detail_focused() {
        // Regression: on Rocm/Serving with `pane_focus == Detail`, the real
        // Esc key steps Details → Actions first (`PaneEscape` in
        // `apply_action`); it does not open the menu. The chip must dispatch
        // the same `PaneEscape` action (not `OpenMenu`) so a click matches
        // the keypress, and its label must say so.
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.active_tab = ActiveTab::Rocm;
        state.pane_focus = PaneFocus::Detail;

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        assert!(
            chips.iter().any(|c| c.action == KeyAction::PaneEscape),
            "Esc chip must dispatch PaneEscape, matching the real key, while Detail is focused"
        );
        assert!(
            !chips.iter().any(|c| c.action == KeyAction::OpenMenu),
            "no chip may claim OpenMenu while Esc would actually step Detail back to Actions"
        );
        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("back"),
            "chip label should say the Esc key steps back out of Detail: {row:?}"
        );
    }

    #[test]
    fn footer_esc_chip_is_not_clickable_menu_when_overlay_has_a_sub_popup_open() {
        // Regression: with a manager open but not at its root layer (here, a
        // folder browser sub-popup), `has_open_overlay()` is true but
        // `active_overlay_at_root()` is false. The chip must not fall through
        // to the generic `OpenMenu` arm — that key is actually consumed by the
        // manager's own event-loop arm, which cancels the sub-layer, not the
        // menu. Any chip shown here must be non-clickable (`action == None`)
        // and labeled "cancel" (this sub-popup has no "close means job keeps
        // running" nuance, unlike a job console — see the "close" test below).
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState {
            browser: Some(crate::ui::folder_browser::FolderBrowser::new(
                "t",
                std::env::temp_dir(),
            )),
            ..Default::default()
        });
        assert!(state.has_open_overlay());
        assert!(!state.active_overlay_at_root());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        for chip in &chips {
            assert_ne!(
                chip.action,
                KeyAction::OpenMenu,
                "no chip may dispatch OpenMenu while a sub-popup owns Esc"
            );
        }
        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("Esc  cancel"),
            "sub-popup Esc chip should say cancel: {row:?}"
        );
        // Note: "close" legitimately appears elsewhere in this row (the `q`
        // chip always says "close" while any overlay is open, root or not —
        // see `footer_q_t_help_chips_are_not_clickable_when_a_manager_overlay_is_open`),
        // so the Esc chip's own label must be checked specifically rather
        // than scanning the whole row for the substring.
        assert!(
            !row.contains("Esc  close"),
            "sub-popup Esc chip should not say close: {row:?}"
        );
    }

    #[test]
    fn footer_esc_chip_is_noninteractive_while_quit_confirm_is_pending() {
        // Issue #145 regression: the Esc-chip label chain checked
        // `approval_pending()` but not `quit_confirm_pending()`, so with
        // nothing else open it fell to the generic `PaneEscape` arm — making
        // the chip genuinely clickable (and mislabeled "menu") while the
        // quit-confirm prompt was still open on top of it.
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.open_quit_confirm();
        assert!(!state.has_open_overlay());
        assert_eq!(state.modal, Modal::None);

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        for chip in &chips {
            assert_ne!(
                chip.action,
                KeyAction::PaneEscape,
                "no chip may dispatch PaneEscape while quit-confirm owns Esc"
            );
        }
        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("Esc  cancel"),
            "quit-confirm Esc chip should say cancel: {row:?}"
        );
        assert!(
            !row.contains("Esc  menu"),
            "quit-confirm Esc chip must not advertise menu: {row:?}"
        );
    }

    #[test]
    fn footer_esc_chip_labels_close_when_job_console_is_open() {
        // A manager's job console is showing a still-running job — Esc fully
        // closes the overlay there (matching the console's own "Esc close
        // (keeps running)" footer hint), so the dashboard footer chip must say
        // "close", not the generic sub-popup "cancel".
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use rocm_dash_core::state::StateEvent;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.jobs.apply(StateEvent::StartJob {
            id: "job".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        });
        state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState {
            active_job: Some("job".into()),
            ..Default::default()
        });
        assert!(state.has_open_overlay());
        assert!(!state.active_overlay_at_root());
        assert!(state.active_job_id().is_some());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            let _ = draw_footer(f, f.area(), &state, &theme);
        })
        .unwrap();

        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        // The `q` chip always says "close" while any overlay is open, root or
        // not, so an unscoped substring check here would pass even if the Esc
        // chip's own label were wrong — check the Esc chip specifically.
        assert!(
            row.contains("Esc  close"),
            "job console Esc chip should say close: {row:?}"
        );
        assert!(
            !row.contains("Esc  cancel"),
            "job console Esc chip should not say cancel: {row:?}"
        );
    }

    #[test]
    fn footer_esc_chip_labels_dismiss_when_job_console_shows_a_finished_job() {
        // Once the job console's job has finished, Esc only dismisses the
        // console back to the screen body (the overlay itself stays open) —
        // `on_console_key` never returns `Closed` for a terminal job. The
        // footer chip must not claim "close" here, and must not fall through
        // to the generic sub-popup "cancel" label either — that would
        // contradict the console's own reworded footer hint ("Enter/Esc
        // dismiss", see `job_console::draw`). It gets a dedicated "dismiss"
        // label instead.
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use rocm_dash_core::state::StateEvent;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.jobs.apply(StateEvent::StartJob {
            id: "job".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        });
        state.jobs.apply(StateEvent::JobDone {
            id: "job".into(),
            code: 0,
        });
        state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState {
            active_job: Some("job".into()),
            ..Default::default()
        });
        assert!(state.has_open_overlay());
        assert!(!state.active_overlay_at_root());
        assert!(state.active_job_id().is_some());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| {
            let _ = draw_footer(f, f.area(), &state, &theme);
        })
        .unwrap();

        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("Esc  dismiss"),
            "finished-job console Esc chip should say dismiss: {row:?}"
        );
        // Note: "close" legitimately appears elsewhere in this row (the `q`
        // chip always says "close" while any overlay is open — see
        // `footer_q_t_help_chips_are_not_clickable_when_a_manager_overlay_is_open`),
        // so the Esc chip's own label must be checked specifically rather
        // than scanning the whole row for the substring.
        assert!(
            !row.contains("Esc  close"),
            "finished-job console Esc chip should not say close: {row:?}"
        );
        assert!(
            !row.contains("Esc  cancel"),
            "finished-job console Esc chip should not say cancel: {row:?}"
        );
    }

    #[test]
    fn footer_q_t_help_chips_are_not_clickable_when_a_manager_overlay_is_open() {
        // Regression: while any manager overlay is open it owns every key
        // (the event loop routes each keypress to the manager's own `on_key`,
        // never falling through to `apply_action`), so a real `q`/`t`/`?`
        // press can never reach `KeyAction::Quit`/`OpenThemePicker`/
        // `ToggleHelp` there — it only cancels/closes the overlay. A click on
        // any of these footer chips must not diverge from that: it must not
        // tear down the app (killing a still-running job via `kill_on_drop`)
        // or swap the theme / pop the help overlay on top of the manager when
        // the key itself never would.
        use crate::ui::services_manager::ServicesManagerState;
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.services = Some(ServicesManagerState::default());
        assert!(state.has_open_overlay());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        for chip in &chips {
            assert_ne!(
                chip.action,
                KeyAction::Quit,
                "no chip may dispatch Quit while a manager overlay owns `q`"
            );
            assert_ne!(
                chip.action,
                KeyAction::OpenThemePicker,
                "no chip may dispatch OpenThemePicker while a manager overlay owns `t`"
            );
            assert_ne!(
                chip.action,
                KeyAction::ToggleHelp,
                "no chip may dispatch ToggleHelp while a manager overlay owns `?`"
            );
        }
        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("close"),
            "q chip should say close while an overlay is open: {row:?}"
        );
        assert!(
            !row.contains("quit"),
            "q chip should not say quit while an overlay is open: {row:?}"
        );
    }

    #[test]
    fn footer_q_t_help_chips_are_not_clickable_when_an_approval_is_pending() {
        // Regression: `has_open_overlay()` deliberately excludes `approval`
        // (it's a separate gating layer with its own routing — see its doc
        // comment), so a pending approval alone must still gate these chips.
        // Keyboard input has a dedicated `state.approval.is_some()` arm ahead
        // of general key dispatch that routes every key to `approval_key`;
        // a footer-chip click bypasses that dispatch entirely and would
        // otherwise silently discard the pending approval (or, for `q`, tear
        // the app down without ever recording a verdict).
        use crate::app::PendingApproval;
        use crate::ui::approval::{ApprovalChoice, ApprovalRequest};
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.approval = Some(PendingApproval {
            req: ApprovalRequest::new("run it", vec!["echo hi".into()]),
            choice: ApprovalChoice::default(),
            name: "tool".into(),
            arguments: serde_json::Value::Null,
        });
        assert!(!state.has_open_overlay());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        for chip in &chips {
            assert_ne!(
                chip.action,
                KeyAction::Quit,
                "no chip may dispatch Quit while an approval is pending"
            );
            assert_ne!(
                chip.action,
                KeyAction::OpenThemePicker,
                "no chip may dispatch OpenThemePicker while an approval is pending"
            );
            assert_ne!(
                chip.action,
                KeyAction::ToggleHelp,
                "no chip may dispatch ToggleHelp while an approval is pending"
            );
        }
    }

    #[test]
    fn observe_guided_action_chips_are_hidden_while_an_overlay_or_approval_owns_input() {
        // Regression: the w/e/d/u/i/l/s/b Observe-tab hotkey chips are the one
        // footer group that used to be pushed unconditionally, unlike every
        // other chip this file gates. A real w/e/d/u/i/l/s/b keypress can
        // never reach an `Open*` action while a manager overlay or a pending
        // approval owns the event loop (same reachability argument as the
        // q/t/? chips above) — but every `Open*` handler starts with
        // `close_overlays()`, which would silently discard whatever was open
        // (including an approval, with no verdict ever recorded) if a click
        // on one of these chips reached it anyway.
        use crate::app::PendingApproval;
        use crate::ui::approval::{ApprovalChoice, ApprovalRequest};
        use crate::ui::services_manager::ServicesManagerState;
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let open_actions = [
            KeyAction::OpenServeWizard,
            KeyAction::OpenEngineManager,
            KeyAction::OpenExamine,
            KeyAction::OpenUpdate,
            KeyAction::OpenInstall,
            KeyAction::OpenLogs,
            KeyAction::OpenServices,
            KeyAction::OpenBenchRun,
        ];
        let assert_no_open_chip = |state: &AppState, why: &str| {
            let backend = TestBackend::new(160, 1);
            let mut term = Terminal::new(backend).unwrap();
            let mut chips = Vec::new();
            term.draw(|f| chips = draw_footer(f, f.area(), state, &theme))
                .unwrap();
            for chip in &chips {
                assert!(
                    !open_actions.contains(&chip.action),
                    "no chip may dispatch {:?} {why}",
                    chip.action
                );
            }
        };

        let mut overlay_state = AppState::new("t".into(), "default-dark".into());
        overlay_state.active_tab = ActiveTab::Observe;
        overlay_state.services = Some(ServicesManagerState::default());
        assert!(overlay_state.has_open_overlay());
        assert_no_open_chip(&overlay_state, "while a manager overlay is open");

        let mut approval_state = AppState::new("t".into(), "default-dark".into());
        approval_state.active_tab = ActiveTab::Observe;
        approval_state.approval = Some(PendingApproval {
            req: ApprovalRequest::new("run it", vec!["echo hi".into()]),
            choice: ApprovalChoice::default(),
            name: "tool".into(),
            arguments: serde_json::Value::Null,
        });
        assert!(!approval_state.has_open_overlay());
        assert_no_open_chip(&approval_state, "while an approval is pending");

        let mut idle_state = AppState::new("t".into(), "default-dark".into());
        idle_state.active_tab = ActiveTab::Observe;
        let backend = TestBackend::new(160, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &idle_state, &theme))
            .unwrap();
        for action in open_actions {
            assert!(
                chips.iter().any(|c| c.action == action),
                "{action:?} chip must still be clickable on an idle Observe tab"
            );
        }
    }

    #[test]
    fn footer_t_help_chips_are_not_clickable_while_a_plain_modal_is_open() {
        // Regression: `OpenThemePicker`/`ToggleHelp` both call `close_overlays()`
        // then unconditionally overwrite `state.modal` (`open_theme_picker`,
        // and `ToggleHelp`'s handler) — but neither `has_open_overlay()` nor
        // `approval_pending()` is true while a plain `Modal::*` (e.g. Detail)
        // is open, so before this fix these chips stayed clickable and a click
        // would silently replace the open modal with ThemePicker/Help, which a
        // real keypress there can't do (`handle_key`'s `Modal::Detail` arm maps
        // `?` to `CloseModal` and doesn't match `t` at all).
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.modal = Modal::Detail;
        assert!(!state.has_open_overlay() && !state.approval_pending());

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        for chip in &chips {
            assert_ne!(
                chip.action,
                KeyAction::OpenThemePicker,
                "no chip may dispatch OpenThemePicker while a plain modal is open"
            );
            assert_ne!(
                chip.action,
                KeyAction::ToggleHelp,
                "no chip may dispatch ToggleHelp while a plain modal is open"
            );
            assert!(
                !matches!(chip.action, KeyAction::SwitchTab(_)),
                "no chip may dispatch SwitchTab while a plain modal is open \
                 (Modal::Detail doesn't remap Tab at all, so a real Tab \
                 keypress there is swallowed as Nothing)"
            );
        }
        // `q` is unaffected: every modal arm already maps a real `q` to `Quit`
        // (see `handle_key`), so the chip must stay clickable and correctly
        // labeled, not silently fall back to the overlay's "close" label.
        assert!(
            chips.iter().any(|c| c.action == KeyAction::Quit),
            "q chip must still dispatch Quit while only a plain modal is open"
        );
    }

    #[test]
    fn observe_guided_action_chips_are_hidden_while_a_plain_modal_is_open() {
        // Regression: `observe_guided_action_chips_are_hidden_while_an_overlay_or_approval_owns_input`
        // only covered the manager-overlay and pending-approval cases, missing
        // a plain `Modal::*` (e.g. `Detail`) — every modal arm in `handle_key`
        // fully returns before the w/e/d/u/i/l/s/b match, so a real keypress
        // never reaches an `Open*` action there either, but before this fix
        // the chips stayed clickable and their `Open*` handler's
        // `close_overlays()` would silently discard the open modal.
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let open_actions = [
            KeyAction::OpenServeWizard,
            KeyAction::OpenEngineManager,
            KeyAction::OpenExamine,
            KeyAction::OpenUpdate,
            KeyAction::OpenInstall,
            KeyAction::OpenLogs,
            KeyAction::OpenServices,
            KeyAction::OpenBenchRun,
        ];

        let mut state = AppState::new("t".into(), "default-dark".into());
        state.active_tab = ActiveTab::Observe;
        state.modal = Modal::Detail;
        assert!(!state.has_open_overlay() && !state.approval_pending());

        let backend = TestBackend::new(160, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();
        for chip in &chips {
            assert!(
                !open_actions.contains(&chip.action),
                "no chip may dispatch {:?} while a plain modal is open",
                chip.action
            );
        }
    }

    #[test]
    fn footer_global_hotkey_chips_are_not_clickable_while_chat_is_focused() {
        // Regression: focused chat text entry consumes every `Char` key
        // (including `t`/`?`/`q`, and — via the catch-all `_ => Nothing` arm —
        // `Tab`) as ordinary input, never as a global hotkey (see
        // `handle_key`'s `chat.focused` branch). Before this fix, `Tab`/`t`/`?`
        // stayed clickable while composing a chat message, and clicking them
        // would switch tabs or open the theme picker / help overlay out from
        // under the user mid-sentence — something no keystroke there could do.
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.active_tab = ActiveTab::Chat;
        state.chat_consent = ChatConsent::Accepted;
        state.chat_focused = true;

        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &state, &theme))
            .unwrap();

        for chip in &chips {
            assert!(
                !matches!(
                    chip.action,
                    KeyAction::SwitchTab(_)
                        | KeyAction::OpenThemePicker
                        | KeyAction::ToggleHelp
                        | KeyAction::Quit
                ),
                "no chip may dispatch {:?} while chat is focused",
                chip.action
            );
        }
        // `q`'s label stays "quit" here (unlike the overlay/approval case,
        // there is no overlay to "close" while merely typing) even though the
        // chip itself is non-clickable.
        let row: String = (0..90)
            .map(|x| term.backend().buffer().cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(
            row.contains("quit"),
            "q chip should still say quit while chat is focused: {row:?}"
        );
        assert!(
            !row.contains("close"),
            "q chip should not say close while chat is focused (no overlay to close): {row:?}"
        );
    }

    #[test]
    fn footer_replay_chips_are_not_clickable_while_an_overlay_or_modal_owns_input() {
        // Same reachability argument as the Tab/t/?/q chips: a real
        // Space/+/- keypress only reaches `ReplayTogglePause`/`ReplaySpeedUp`
        // when nothing else owns the key first.
        use crate::app::ReplayState;
        use crate::replay::ReplayController;
        use crate::ui::services_manager::ServicesManagerState;
        use crate::ui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::from_name("default-dark");
        let replay_actions = [KeyAction::ReplayTogglePause, KeyAction::ReplaySpeedUp];
        let assert_no_replay_chip = |state: &AppState, why: &str| {
            let backend = TestBackend::new(90, 1);
            let mut term = Terminal::new(backend).unwrap();
            let mut chips = Vec::new();
            term.draw(|f| chips = draw_footer(f, f.area(), state, &theme))
                .unwrap();
            for chip in &chips {
                assert!(
                    !replay_actions.contains(&chip.action),
                    "no chip may dispatch {:?} {why}",
                    chip.action
                );
            }
        };

        let mut overlay_state = AppState::new("t".into(), "default-dark".into());
        overlay_state.replay = Some(ReplayState::new(ReplayController::for_test()));
        overlay_state.services = Some(ServicesManagerState::default());
        assert!(overlay_state.has_open_overlay());
        assert_no_replay_chip(&overlay_state, "while a manager overlay is open");

        let mut modal_state = AppState::new("t".into(), "default-dark".into());
        modal_state.replay = Some(ReplayState::new(ReplayController::for_test()));
        modal_state.modal = Modal::Detail;
        assert_no_replay_chip(&modal_state, "while a plain modal is open");

        let mut idle_state = AppState::new("t".into(), "default-dark".into());
        idle_state.replay = Some(ReplayState::new(ReplayController::for_test()));
        let backend = TestBackend::new(90, 1);
        let mut term = Terminal::new(backend).unwrap();
        let mut chips = Vec::new();
        term.draw(|f| chips = draw_footer(f, f.area(), &idle_state, &theme))
            .unwrap();
        for action in replay_actions {
            assert!(
                chips.iter().any(|c| c.action == action),
                "{action:?} chip must still be clickable while idle and replaying"
            );
        }
    }
}
