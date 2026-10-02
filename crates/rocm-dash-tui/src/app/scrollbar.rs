// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Mouse/scroll hit-testing: resolving a raw `MouseEvent` against recorded
//! scrollbar tracks, the tab bar, and footer-legend chips into a `KeyAction`.
//! Split out of `app/mod.rs` to keep the core reducer + event loop focused.

#[cfg(test)]
use crossterm::event::KeyModifiers;
use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};

use crate::ui;

#[cfg(test)]
use super::actions::apply_action;
use super::actions::{KeyAction, handle_mouse, tab_bar_hit};
use super::{ActiveTab, AppState, Modal};

/// Convert a displayed position to the target's own offset units. Dock logs are
/// tail-anchored, so their displayed top-to-bottom position is inverted.
fn target_offset(h: &ScrollbarHandle, displayed: usize) -> usize {
    if h.target == ScrollTarget::DockLogs {
        h.max_position().saturating_sub(displayed)
    } else {
        displayed
    }
}

fn target_position(state: &AppState, h: &ScrollbarHandle) -> usize {
    let position = match h.target {
        ScrollTarget::Console => usize::from(state.console_scroll),
        ScrollTarget::ConsoleH => usize::from(state.console_hscroll),
        ScrollTarget::Chat => usize::from(state.chat_scroll),
        ScrollTarget::DockLogs => h
            .max_position()
            .saturating_sub(usize::from(state.dock_logs_scroll)),
        ScrollTarget::InstanceDetail => usize::from(state.instance_detail_scroll),
    };
    position.min(h.max_position())
}

/// If `(col, row)` lands on a recorded scrollbar, preserve a thumb grab or use
/// a proportional full-track jump for a track click.
fn scrollbar_hit(state: &AppState, col: u16, row: u16) -> Option<KeyAction> {
    let bars = state.scrollbars.borrow();
    let h = bars.iter().find(|h| point_in(h.track, col, row))?;
    let displayed = target_position(state, h);
    let grab_offset = h.grab_offset(col, row, displayed);
    let next = grab_offset.map_or_else(|| h.track_position_at(col, row), |_| displayed);
    Some(KeyAction::ScrollGrab(
        h.target,
        target_offset(h, next),
        grab_offset.unwrap_or(0),
    ))
}

pub(crate) fn resolve_mouse(me: MouseEvent, state: &AppState) -> KeyAction {
    // A held drag on a scrollbar keeps updating that offset until release, even
    // when the pointer slides off the narrow track.
    if me.kind == MouseEventKind::Drag(MouseButton::Left) {
        // A drag can start before an approval becomes pending (it's only
        // gated at the click that starts it, via `scrollbar_hit`'s own
        // `approval_pending()` check below) and then have an approval land
        // asynchronously mid-drag. Swallow it here too, or the drag would
        // keep mutating a scroll position hidden behind the approval modal —
        // "a pending approval owns the body with no exception" (see the
        // wheel-scroll swallow further down) applies to an in-flight drag
        // just as much as to input that starts fresh.
        if state.approval_pending() {
            return KeyAction::Nothing;
        }
        if let Some(drag) = state.scroll_drag
            && let Some(h) = state
                .scrollbars
                .borrow()
                .iter()
                .find(|h| h.target == drag.target)
        {
            let current = target_position(state, h);
            let displayed = h.position_at(me.column, me.row, drag.grab_offset, current);
            return KeyAction::ScrollGrab(
                drag.target,
                target_offset(h, displayed),
                drag.grab_offset,
            );
        }
        return KeyAction::Nothing;
    }
    // Any button release ends an active scrollbar drag.
    if matches!(me.kind, MouseEventKind::Up(_)) {
        return if state.scroll_drag.is_some() {
            KeyAction::ScrollRelease
        } else {
            KeyAction::Nothing
        };
    }

    if me.kind == MouseEventKind::Down(MouseButton::Left) {
        // Scrollbar tracks win over a plain open overlay (incl. its console
        // bar), so a click on the bar grabs it instead of falling through —
        // but NOT over a pending approval: nothing registers a scrollbar for
        // the approval modal itself, so any handle on screen while one is
        // pending belongs to content underneath it, which the swallow below
        // must still catch rather than let a scrollbar drag bypass it.
        if !state.approval_pending()
            && let Some(a) = scrollbar_hit(state, me.column, me.row)
        {
            return a;
        }
        if let Some(area) = state.last_tab_bar_area
            && let Some(tab) = tab_bar_hit(area, me.column, me.row)
        {
            return KeyAction::SwitchTab(tab);
        }
        // Footer legend: a click on a key chip acts exactly like the key press.
        if let Some(chip) = footer_chip_hit(&state.last_footer_chips, me.column, me.row) {
            return chip;
        }
        // While an operational manager is open — or a chat tool-call approval
        // is pending — it owns the body: swallow body clicks so they can't
        // fall THROUGH to the obscured Actions/Details list (which would
        // silently change the selection, re-open a verb, or switch tabs
        // underneath the approval modal). Tab-bar and footer-chip clicks
        // above still work, matching the manager-overlay swallow this
        // mirrors (see the analogous `overlay_or_approval()` check in
        // ui/mod.rs's footer-chip gating).
        if state.overlay_or_approval() {
            return KeyAction::Nothing;
        }
        if state.modal == Modal::None
            && let Some(area) = state.last_body_area
        {
            // ponytail: Observe folds the instances table into a stacked region;
            // body-click hit-testing best-efforts the instances rows. Keyboard
            // selection is the primary path.
            let action = match state.active_tab {
                // Observe's AI table is keyboard + scroll-wheel driven (the
                // scroll path maps to Move in `handle_mouse`); left-click select
                // is intentionally not wired (the table sits below the hero band,
                // so a body-relative row map would be wrong). No-op here.
                ActiveTab::Rocm => ui::tabs::rocm::hit_test(area, me.column, me.row),
                ActiveTab::Serving => ui::tabs::serving::hit_test(area, me.column, me.row),
                _ => None,
            };
            if let Some(a) = action {
                return a;
            }
        }
        return KeyAction::Nothing;
    }

    // Scroll wheel (incl. horizontal wheel where the device emits it). Per-notch
    // deltas: ±1 line / ±1 col here, scaled per target below.
    let (dv, dh): (i16, i16) = match me.kind {
        MouseEventKind::ScrollDown => (1, 0),
        MouseEventKind::ScrollUp => (-1, 0),
        MouseEventKind::ScrollRight => (0, 1),
        MouseEventKind::ScrollLeft => (0, -1),
        // Not a scroll (e.g. moves / other buttons): nothing to route.
        _ => return KeyAction::Nothing,
    };

    // A pending approval owns the body with no exception (mirrors the click
    // swallow a few lines above) — unlike a plain manager overlay, it never
    // has its own console to pan, so there is nothing to fall through to.
    if state.approval_pending() {
        return KeyAction::Nothing;
    }
    // An open manager owns the body. When it is showing its job console, the
    // wheel pans that log (bigger vertical step, wider horizontal step so long
    // command lines come into view). On a form screen there is nothing to pan —
    // swallow it so the wheel can't move the obscured Actions list underneath.
    if state.has_open_overlay() {
        return if state.has_active_console() {
            KeyAction::ScrollConsole(dv * 3, dh * 6)
        } else {
            KeyAction::Nothing
        };
    }

    // Wide-layout right LOGS dock: the wheel pans the log stream when the pointer
    // is over it (vertical only — it's a tail-anchored log).
    if state.modal == Modal::None
        && dv != 0
        && let Some(dock) = state.last_dock_area
        && point_in(dock, me.column, me.row)
    {
        return KeyAction::ScrollDock(dv * 3);
    }

    // No overlay: on a domain tab the wheel moves the Actions selection by ONE
    // row — but only while the pointer is actually over the Actions column, so
    // hovering the Details pane doesn't nudge the list. Anything else falls
    // through to the modal/tab scroll routing.
    if state.modal == Modal::None
        && matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving)
    {
        if dv != 0
            && let Some(body) = state.last_body_area
            && point_in(crate::ui::tabs::pane::actions_rect(body), me.column, me.row)
        {
            return KeyAction::Move(dv as isize);
        }
        return KeyAction::Nothing;
    }

    handle_mouse(me, &state.modal, state.active_tab)
}

/// Whether `(x, y)` lies inside `r` (end-exclusive on both axes).
const fn point_in(r: ratatui::layout::Rect, x: u16, y: u16) -> bool {
    x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height
}

/// Resolve a pointer `(col, row)` against the recorded footer-legend chips.
/// Returns the chip's action when the pointer lands inside a chip span.
fn footer_chip_hit(chips: &[FooterChip], col: u16, row: u16) -> Option<KeyAction> {
    chips
        .iter()
        .find(|c| row == c.y && col >= c.x0 && col < c.x1)
        .map(|c| c.action)
}

/// Where a domain tab's (ROCm/Serving) keyboard focus currently sits. Shared by
/// both tabs; each keeps its own selection cursor (`rocm_sel`/`serving_sel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaneFocus {
    /// Browsing the Actions list (left column).
    #[default]
    Actions,
    /// Inside the Details pane (right column), ready to start the operation.
    Detail,
}

/// A clickable footer-legend chip: an absolute screen span on the footer row
/// plus the action a left-click should dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FooterChip {
    pub x0: u16,
    /// End-exclusive.
    pub x1: u16,
    pub y: u16,
    pub action: KeyAction,
}

/// Active pointer drag for a scrollbar thumb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollDrag {
    pub target: ScrollTarget,
    pub grab_offset: u16,
}

/// Which scrollable surface a drawn scrollbar controls. Lets a mouse click on a
/// scrollbar track write the right offset field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollTarget {
    /// Job console vertical (`console_scroll`).
    Console,
    /// Job console horizontal (`console_hscroll`).
    ConsoleH,
    /// Wide-layout LOGS dock (`dock_logs_scroll`, tail-anchored / inverted).
    DockLogs,
    /// Chat transcript (`chat_scroll`).
    Chat,
    /// Instance detail modal's launch_args/env_vars panes (`instance_detail_scroll`).
    InstanceDetail,
}

/// A scrollbar drawn this frame, recorded so a mouse click/drag can hit-test it.
///
/// `track` is the screen rect of the bar; `content_len`/`viewport_len` size the
/// thumb; `target` says which offset to move. Vertical bars map the mouse row,
/// horizontal bars the column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollbarHandle {
    pub track: ratatui::layout::Rect,
    pub horizontal: bool,
    pub content_len: usize,
    pub viewport_len: usize,
    pub target: ScrollTarget,
}

impl ScrollbarHandle {
    /// Build a handle from the rect passed to a scrollbar helper (`area`) and its
    /// returned content rect (`drawn`). Returns `None` when they're equal — i.e.
    /// no bar was drawn because the content fit — so nothing gets hit-tested.
    pub(crate) fn new(
        area: ratatui::layout::Rect,
        drawn: ratatui::layout::Rect,
        horizontal: bool,
        content_len: usize,
        viewport_len: usize,
        target: ScrollTarget,
    ) -> Option<Self> {
        if drawn == area {
            return None;
        }
        let track = if horizontal {
            ratatui::layout::Rect::new(area.x, area.y + area.height - 1, area.width, 1)
        } else {
            ratatui::layout::Rect::new(area.x + area.width - 1, area.y, 1, area.height)
        };
        Some(Self {
            track,
            horizontal,
            content_len,
            viewport_len,
            target,
        })
    }

    const fn max_position(&self) -> usize {
        self.content_len.saturating_sub(self.viewport_len)
    }

    const fn axis(&self, col: u16, row: u16) -> (u16, u16, u16) {
        if self.horizontal {
            (col, self.track.x, self.track.width)
        } else {
            (row, self.track.y, self.track.height)
        }
    }

    /// Ratatui 0.30.2 `Scrollbar::part_lengths` geometry for the logical
    /// first-visible-unit position used by the dashboard.
    fn thumb_geometry(&self, logical_position: usize) -> (u16, u16) {
        let (_, _, span) = self.axis(0, 0);
        if span == 0 || self.content_len == 0 {
            return (0, 0);
        }
        let track_len = usize::from(span);
        let rendered_max = self.content_len.saturating_sub(1);
        let rendered_position =
            logical_position.min(self.max_position()) * rendered_max / self.max_position().max(1);
        let denominator = rendered_max.saturating_add(self.viewport_len);
        let rounded_divide =
            |numerator: usize| numerator.saturating_add(denominator / 2) / denominator.max(1);
        let thumb_len =
            rounded_divide(self.viewport_len.saturating_mul(track_len)).clamp(1, track_len);
        let thumb_start = rounded_divide(rendered_position.saturating_mul(track_len))
            .clamp(0, track_len.saturating_sub(thumb_len));
        (thumb_start as u16, thumb_len as u16)
    }

    fn grab_offset(&self, col: u16, row: u16, position: usize) -> Option<u16> {
        let (coord, track_start, _) = self.axis(col, row);
        let relative = coord.saturating_sub(track_start);
        let (thumb_start, thumb_len) = self.thumb_geometry(position);
        (relative >= thumb_start && relative < thumb_start.saturating_add(thumb_len))
            .then(|| relative - thumb_start)
    }

    /// Proportional track-click mapping. The endpoints map exactly to the
    /// logical endpoints and do not depend on thumb geometry.
    fn track_position_at(&self, col: u16, row: u16) -> usize {
        let max_position = self.max_position();
        let (coord, start, span) = self.axis(col, row);
        if max_position == 0 || span <= 1 {
            return 0;
        }
        usize::from(coord.saturating_sub(start).min(span - 1)) * max_position
            / usize::from(span - 1)
    }

    /// Invert Ratatui's rounded thumb-start mapping. When several logical
    /// positions render at the requested start, retain `current_position` if it
    /// lies on that plateau; otherwise choose the nearest plateau endpoint.
    fn position_at(&self, col: u16, row: u16, grab_offset: u16, current_position: usize) -> usize {
        let max_position = self.max_position();
        if max_position == 0 {
            return 0;
        }
        let (coord, start, span) = self.axis(col, row);
        let (_, thumb_len) = self.thumb_geometry(0);
        let desired_start = coord
            .saturating_sub(start)
            .saturating_sub(grab_offset)
            .min(span.saturating_sub(thumb_len));
        if desired_start == 0 {
            return 0;
        }
        if desired_start == span.saturating_sub(thumb_len) {
            return max_position;
        }

        let first_at_or_after = |wanted: u16| {
            let mut low = 0usize;
            let mut high = max_position;
            while low < high {
                let mid = low + (high - low) / 2;
                if self.thumb_geometry(mid).0 < wanted {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };
        let first = first_at_or_after(desired_start);
        if self.thumb_geometry(first).0 != desired_start {
            if first == 0 {
                return 0;
            }
            let before = first - 1;
            let before_start = self.thumb_geometry(before).0;
            let after_start = self.thumb_geometry(first).0;
            return if desired_start - before_start <= after_start - desired_start {
                before
            } else {
                first
            };
        }
        let after = first_at_or_after(desired_start.saturating_add(1));
        let last = if after == max_position && self.thumb_geometry(after).0 == desired_start {
            after
        } else {
            after.saturating_sub(1)
        };
        current_position.clamp(first, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::layout::Rect;

    #[test]
    fn body_clicks_are_swallowed_while_a_manager_is_open() {
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 90,
            row: 10,
            modifiers: KeyModifiers::NONE,
        };
        // No manager open → the click resolves against the tab's hit-test.
        assert_ne!(resolve_mouse(click, &s), KeyAction::Nothing);
        // Manager open → the body click is swallowed (no click-through).
        s.install_manager = Some(crate::ui::install_manager::InstallManagerState::default());
        assert_eq!(resolve_mouse(click, &s), KeyAction::Nothing);
    }

    #[test]
    fn body_clicks_are_swallowed_while_an_approval_is_pending() {
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 90,
            row: 10,
            modifiers: KeyModifiers::NONE,
        };
        // No approval pending → the click resolves against the tab's hit-test.
        assert_ne!(resolve_mouse(click, &s), KeyAction::Nothing);
        // `open_approval` clears every manager overlay (so `has_open_overlay()`
        // is false) but never touches `modal` — the body click must still be
        // swallowed instead of falling through to the obscured Actions/Details
        // list underneath the approval modal.
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "run a command".into(),
            body: vec!["echo hi".into()],
            name: "shell".into(),
            arguments: serde_json::Value::Null,
        });
        assert!(!s.has_open_overlay());
        assert_eq!(s.modal, Modal::None);
        assert_eq!(resolve_mouse(click, &s), KeyAction::Nothing);
    }

    /// Build a ScrollDown/Up/Left/Right event at a pointer position.
    fn wheel(kind: MouseEventKind, col: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column: col,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn scrollbar_position_maps_proportionally() {
        let h = ScrollbarHandle {
            track: Rect::new(50, 0, 1, 10),
            horizontal: false,
            content_len: 100,
            viewport_len: 10,
            target: ScrollTarget::Console,
        };
        assert_eq!(h.track_position_at(50, 0), 0);
        assert_eq!(h.track_position_at(50, 9), 90);
        assert_eq!(h.track_position_at(50, 5), 90 * 5 / 9);
        assert_eq!(h.track_position_at(50, 99), 90);
    }

    #[test]
    fn scrollbar_click_grabs_drag_scrolls_then_releases() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.scrollbars.borrow_mut().push(ScrollbarHandle {
            track: Rect::new(60, 0, 1, 10),
            horizontal: false,
            content_len: 100,
            viewport_len: 10,
            target: ScrollTarget::Console,
        });
        // Click near the bottom of the track → grab + jump near the end.
        let down = wheel(MouseEventKind::Down(MouseButton::Left), 60, 9);
        let a = resolve_mouse(down, &s);
        assert_eq!(a, KeyAction::ScrollGrab(ScrollTarget::Console, 90, 0));
        apply_action(&mut s, a);
        assert_eq!(s.console_scroll, 90);
        assert_eq!(
            s.scroll_drag,
            Some(ScrollDrag {
                target: ScrollTarget::Console,
                grab_offset: 0,
            })
        );
        // Drag to the top — the off-axis column is ignored, so it still tracks.
        let drag = wheel(MouseEventKind::Drag(MouseButton::Left), 40, 0);
        let a = resolve_mouse(drag, &s);
        assert_eq!(a, KeyAction::ScrollGrab(ScrollTarget::Console, 0, 0));
        apply_action(&mut s, a);
        assert_eq!(s.console_scroll, 0);
        // Release clears the drag.
        let up = wheel(MouseEventKind::Up(MouseButton::Left), 40, 0);
        let a = resolve_mouse(up, &s);
        assert_eq!(a, KeyAction::ScrollRelease);
        apply_action(&mut s, a);
        assert_eq!(s.scroll_drag, None);
    }

    #[test]
    fn scrollbar_hit_is_swallowed_while_an_approval_is_pending() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.scrollbars.borrow_mut().push(ScrollbarHandle {
            track: Rect::new(60, 0, 1, 10),
            horizontal: false,
            content_len: 100,
            viewport_len: 10,
            target: ScrollTarget::Console,
        });
        let click = wheel(MouseEventKind::Down(MouseButton::Left), 60, 9);
        // No approval pending → the scrollbar still wins, same as
        // `scrollbar_click_grabs_drag_scrolls_then_releases`.
        assert_eq!(
            resolve_mouse(click, &s),
            KeyAction::ScrollGrab(ScrollTarget::Console, 90, 0)
        );
        // Nothing registers a scrollbar for the approval modal itself, so a
        // handle on screen while one is pending belongs to content
        // underneath it — the click must be swallowed like every other body
        // click, not resolve to a drag on the obscured bar.
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "run a command".into(),
            body: vec!["echo hi".into()],
            name: "shell".into(),
            arguments: serde_json::Value::Null,
        });
        assert_eq!(resolve_mouse(click, &s), KeyAction::Nothing);
    }

    #[test]
    fn drag_is_swallowed_once_an_approval_becomes_pending_mid_drag() {
        // A drag can only start while no approval is pending (the click that
        // starts it goes through `scrollbar_hit`, which is itself gated), but
        // an approval can land asynchronously (a chat tool call) while a drag
        // started earlier is still in flight. The Drag branch must not keep
        // updating the scroll position once that happens — "a pending
        // approval owns the body with no exception" applies to an in-flight
        // drag, not just to input that starts fresh.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.scrollbars.borrow_mut().push(ScrollbarHandle {
            track: Rect::new(60, 0, 1, 10),
            horizontal: false,
            content_len: 100,
            viewport_len: 10,
            target: ScrollTarget::Console,
        });
        let down = wheel(MouseEventKind::Down(MouseButton::Left), 60, 9);
        let a = resolve_mouse(down, &s);
        apply_action(&mut s, a);
        assert!(s.scroll_drag.is_some(), "drag must have started");
        assert_eq!(s.console_scroll, 90);

        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "run a command".into(),
            body: vec!["echo hi".into()],
            name: "shell".into(),
            arguments: serde_json::Value::Null,
        });

        let drag = wheel(MouseEventKind::Drag(MouseButton::Left), 40, 0);
        assert_eq!(
            resolve_mouse(drag, &s),
            KeyAction::Nothing,
            "a drag in flight when an approval becomes pending must be swallowed"
        );
    }

    #[test]
    fn rendered_thumb_cells_are_grabbable() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        for horizontal in [false, true] {
            let mut s = AppState::new("t".into(), "default-dark".into());
            let area = if horizontal {
                Rect::new(0, 0, 4, 2)
            } else {
                Rect::new(0, 0, 2, 4)
            };
            let target = if horizontal {
                ScrollTarget::ConsoleH
            } else {
                ScrollTarget::Console
            };
            if horizontal {
                s.console_hscroll = 1;
            } else {
                s.console_scroll = 1;
            }
            let backend = TestBackend::new(area.width, area.height);
            let mut terminal = Terminal::new(backend).unwrap();
            let mut body = area;
            terminal
                .draw(|frame| {
                    body = if horizontal {
                        crate::ui::panel::horizontal_scrollbar(frame, area, 4, 2, 1, &s.theme)
                    } else {
                        crate::ui::panel::vertical_scrollbar(frame, area, 4, 2, 1, &s.theme)
                    };
                })
                .unwrap();
            s.record_scrollbar(area, body, horizontal, 4, 2, target);
            let handle = s.scrollbars.borrow()[0];
            let (thumb_start, thumb_len) = handle.thumb_geometry(1);
            assert_eq!((thumb_start, thumb_len), (1, 2));
            for cell in thumb_start..thumb_start + thumb_len {
                let (col, row) = if horizontal {
                    (cell, handle.track.y)
                } else {
                    (handle.track.x, cell)
                };
                assert_eq!(
                    resolve_mouse(wheel(MouseEventKind::Down(MouseButton::Left), col, row), &s),
                    KeyAction::ScrollGrab(target, 1, cell - thumb_start)
                );
            }
            let (before_col, before_row) = if horizontal {
                (0, handle.track.y)
            } else {
                (handle.track.x, 0)
            };
            assert_eq!(
                resolve_mouse(
                    wheel(
                        MouseEventKind::Down(MouseButton::Left),
                        before_col,
                        before_row
                    ),
                    &s,
                ),
                KeyAction::ScrollGrab(target, 0, 0)
            );
            let (after_col, after_row) = if horizontal {
                (3, handle.track.y)
            } else {
                (handle.track.x, 3)
            };
            assert_eq!(
                resolve_mouse(
                    wheel(
                        MouseEventKind::Down(MouseButton::Left),
                        after_col,
                        after_row
                    ),
                    &s,
                ),
                KeyAction::ScrollGrab(target, 2, 0)
            );
        }
    }

    #[test]
    fn stationary_thumb_drag_preserves_logical_position() {
        for target in [ScrollTarget::Console, ScrollTarget::DockLogs] {
            let mut s = AppState::new("t".into(), "default-dark".into());
            s.console_scroll = 5;
            s.dock_logs_scroll = 5;
            s.scrollbars.borrow_mut().push(ScrollbarHandle {
                track: Rect::new(60, 0, 1, 10),
                horizontal: false,
                content_len: 20,
                viewport_len: 10,
                target,
            });
            let down = wheel(MouseEventKind::Down(MouseButton::Left), 60, 4);
            let action = resolve_mouse(down, &s);
            apply_action(&mut s, action);
            let before = if target == ScrollTarget::DockLogs {
                s.dock_logs_scroll
            } else {
                s.console_scroll
            };
            let drag = wheel(MouseEventKind::Drag(MouseButton::Left), 60, 4);
            let action = resolve_mouse(drag, &s);
            apply_action(&mut s, action);
            let after = if target == ScrollTarget::DockLogs {
                s.dock_logs_scroll
            } else {
                s.console_scroll
            };
            assert_eq!(after, before, "stationary {target:?} drag moved");
        }
    }

    #[test]
    fn horizontal_thumb_drag_tracks_pointer_column() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.console_hscroll = 20;
        s.scrollbars.borrow_mut().push(ScrollbarHandle {
            track: Rect::new(10, 20, 20, 1),
            horizontal: true,
            content_len: 100,
            viewport_len: 20,
            target: ScrollTarget::ConsoleH,
        });
        let action = resolve_mouse(wheel(MouseEventKind::Down(MouseButton::Left), 14, 20), &s);
        apply_action(&mut s, action);
        assert_eq!(s.console_hscroll, 20);
        let action = resolve_mouse(wheel(MouseEventKind::Drag(MouseButton::Left), 19, 99), &s);
        apply_action(&mut s, action);
        assert!(s.console_hscroll > 20);
    }

    #[test]
    fn dock_scrollbar_grab_inverts_tail_anchored_offset() {
        let s = AppState::new("t".into(), "default-dark".into());
        s.scrollbars.borrow_mut().push(ScrollbarHandle {
            track: Rect::new(0, 0, 1, 10),
            horizontal: false,
            content_len: 100,
            viewport_len: 10,
            target: ScrollTarget::DockLogs,
        });
        // Top of the bar = oldest lines = fully scrolled up from the tail (max).
        let top = resolve_mouse(wheel(MouseEventKind::Down(MouseButton::Left), 0, 0), &s);
        assert_eq!(top, KeyAction::ScrollGrab(ScrollTarget::DockLogs, 90, 0));
        // Bottom of the bar = newest tail = offset 0.
        let bot = resolve_mouse(wheel(MouseEventKind::Down(MouseButton::Left), 0, 9), &s);
        assert_eq!(bot, KeyAction::ScrollGrab(ScrollTarget::DockLogs, 0, 0));
    }

    #[test]
    fn wheel_over_actions_list_moves_selection_by_one() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        // Left column (Actions) is the first ~46% — a low column is over the list.
        let over_list = wheel(MouseEventKind::ScrollDown, 10, 12);
        assert_eq!(resolve_mouse(over_list, &s), KeyAction::Move(1));
        let up = wheel(MouseEventKind::ScrollUp, 10, 12);
        assert_eq!(resolve_mouse(up, &s), KeyAction::Move(-1));
    }

    #[test]
    fn wheel_over_details_pane_does_not_move_the_list() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Serving;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        // A high column lands in the Details pane (right ~54%) → no list move.
        let over_detail = wheel(MouseEventKind::ScrollDown, 140, 12);
        assert_eq!(resolve_mouse(over_detail, &s), KeyAction::Nothing);
    }

    #[test]
    fn wheel_over_open_console_pans_the_log_not_the_list() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        s.logs_view = Some(crate::ui::logs_view::LogsViewState {
            active_job: Some("logs".into()),
            ..Default::default()
        });
        // Console showing → vertical wheel pans the log (×3 lines), not the list.
        assert_eq!(
            resolve_mouse(wheel(MouseEventKind::ScrollDown, 10, 12), &s),
            KeyAction::ScrollConsole(3, 0)
        );
        // Horizontal wheel pans columns (×6) for off-screen-wide log lines.
        assert_eq!(
            resolve_mouse(wheel(MouseEventKind::ScrollRight, 10, 12), &s),
            KeyAction::ScrollConsole(0, 6)
        );
    }

    #[test]
    fn wheel_over_logs_dock_scrolls_the_dock() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Serving;
        // A recorded dock rect off to the right of the body.
        s.last_dock_area = Some(Rect::new(160, 4, 52, 30));
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        let over_dock = wheel(MouseEventKind::ScrollDown, 180, 12);
        assert_eq!(resolve_mouse(over_dock, &s), KeyAction::ScrollDock(3));
        // A point outside the dock does not scroll it.
        let over_body = wheel(MouseEventKind::ScrollDown, 10, 12);
        assert_ne!(resolve_mouse(over_body, &s), KeyAction::ScrollDock(3));
    }

    #[test]
    fn instance_detail_scrollbar_click_grabs_drag_scrolls_then_releases() {
        // Mirrors `scrollbar_click_grabs_drag_scrolls_then_releases` for the
        // instance Detail modal's scrollbar (`instances.rs::render_body`
        // registers one for each of its two panes) — proves the
        // `ScrollTarget::InstanceDetail` wiring added for mouse-drag support
        // actually moves `instance_detail_scroll`, not just that the bar
        // renders.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.scrollbars.borrow_mut().push(ScrollbarHandle {
            track: Rect::new(60, 0, 1, 10),
            horizontal: false,
            content_len: 100,
            viewport_len: 10,
            target: ScrollTarget::InstanceDetail,
        });
        let down = wheel(MouseEventKind::Down(MouseButton::Left), 60, 9);
        let a = resolve_mouse(down, &s);
        assert_eq!(
            a,
            KeyAction::ScrollGrab(ScrollTarget::InstanceDetail, 90, 0)
        );
        apply_action(&mut s, a);
        assert_eq!(s.instance_detail_scroll, 90);
        let drag = wheel(MouseEventKind::Drag(MouseButton::Left), 40, 0);
        let a = resolve_mouse(drag, &s);
        assert_eq!(a, KeyAction::ScrollGrab(ScrollTarget::InstanceDetail, 0, 0));
        apply_action(&mut s, a);
        assert_eq!(s.instance_detail_scroll, 0);
        let up = wheel(MouseEventKind::Up(MouseButton::Left), 40, 0);
        let a = resolve_mouse(up, &s);
        assert_eq!(a, KeyAction::ScrollRelease);
        apply_action(&mut s, a);
        assert_eq!(s.scroll_drag, None);
    }

    #[test]
    fn wheel_over_form_screen_overlay_is_swallowed() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        // Overlay open but on its form (no active_job) → nothing to pan, and the
        // obscured Actions list must NOT move.
        s.install_manager = Some(crate::ui::install_manager::InstallManagerState::default());
        assert_eq!(
            resolve_mouse(wheel(MouseEventKind::ScrollDown, 10, 12), &s),
            KeyAction::Nothing
        );
    }

    #[test]
    fn wheel_is_swallowed_while_an_approval_is_pending() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.last_body_area = Some(Rect::new(2, 4, 150, 30));
        s.open_approval(crate::tool_exec::ApprovalIntent {
            title: "run a command".into(),
            body: vec!["echo hi".into()],
            name: "shell".into(),
            arguments: serde_json::Value::Null,
        });
        // `open_approval` clears every manager overlay (`has_open_overlay()` is
        // false) but never touches `modal` — the wheel must still be swallowed
        // instead of falling through to whatever's obscured underneath, the
        // same gap the click path was already fixed for (see
        // `body_clicks_are_swallowed_while_an_approval_is_pending`).
        assert!(!s.has_open_overlay());
        assert_eq!(
            resolve_mouse(wheel(MouseEventKind::ScrollDown, 10, 12), &s),
            KeyAction::Nothing
        );
    }

    #[test]
    fn footer_chip_hit_maps_click_to_action() {
        let chips = vec![
            FooterChip {
                x0: 0,
                x1: 5,
                y: 49,
                action: KeyAction::Quit,
            },
            FooterChip {
                x0: 6,
                x1: 9,
                y: 49,
                action: KeyAction::ToggleHelp,
            },
        ];
        // Inside the first chip.
        assert_eq!(footer_chip_hit(&chips, 2, 49), Some(KeyAction::Quit));
        // End-exclusive: column 5 is past the first chip, before the second.
        assert_eq!(footer_chip_hit(&chips, 5, 49), None);
        assert_eq!(footer_chip_hit(&chips, 7, 49), Some(KeyAction::ToggleHelp));
        // Wrong row never matches.
        assert_eq!(footer_chip_hit(&chips, 2, 48), None);
    }

    #[test]
    fn chat_scrollbar_grab_updates_follow_state() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_max_scroll = 20;
        s.chat_scroll = 20;

        s.apply_scroll_grab(ScrollTarget::Chat, 5, 0);
        assert_eq!(s.chat_scroll, 5);
        assert!(!s.chat_follow, "dragging above the bottom disables follow");

        s.apply_scroll_grab(ScrollTarget::Chat, 20, 0);
        assert_eq!(s.chat_scroll, 20);
        assert!(s.chat_follow, "dragging to the bottom restores follow");
    }
}
