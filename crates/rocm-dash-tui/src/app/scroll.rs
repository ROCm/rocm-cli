// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Scroll state: the job-console, log-dock, chat and instance-detail scroll
//! offsets (clamped through the shared `ScrollState`), and recording the
//! scrollbar tracks a mouse grab resolves against.
//!
//! A further `impl AppState` block, split out of `app/mod.rs` to keep the core
//! reducer focused. Deliberately not named `scrollbar`: `app/scrollbar.rs` is
//! the mouse hit-testing module, and `actions.rs` must never name that token.

use super::{AppState, ScrollDrag, ScrollTarget, ScrollbarHandle};

/// A scroll offset together with the largest offset it may take.
///
/// The one place the "add a signed delta, clamp into `[0, max]`, saturate to
/// `u16`" arithmetic lives, so the job console (both axes), the LOGS dock and
/// the instance Detail modal clamp identically. It is a short-lived value:
/// `AppState` keeps its plain `u16` offsets (the renderer and the scrollbar
/// hit-testing read those directly), and the scroll methods below build a
/// `ScrollState` from the stored offset and the bound they currently know,
/// call [`ScrollState::scroll_by`], then store `pos` back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScrollState {
    /// Current offset, in lines (or columns for a horizontal axis).
    pub(crate) pos: u16,
    /// Largest offset `scroll_by` will produce.
    pub(crate) max: u16,
}

impl ScrollState {
    pub(crate) const fn new(pos: u16, max: u16) -> Self {
        Self { pos, max }
    }

    /// Like [`ScrollState::new`] for a bound computed as a `usize` content
    /// length; a bound beyond `u16::MAX` saturates to it, the same ceiling a
    /// `u16` offset saturates to.
    pub(crate) fn capped_at(pos: u16, max: usize) -> Self {
        Self::new(pos, u16::try_from(max).unwrap_or(u16::MAX))
    }

    /// Move by `delta` (negative = toward 0), landing in `[0, max]`. An offset
    /// already past `max` (content shrank since it was set) is pulled back to
    /// `max`. A delta larger than the remaining distance stops at `0`/`max`
    /// instead of overflowing, so `i16::MIN`/`i16::MAX` ("jump to start/end")
    /// reach the ends for any range of up to 32767 lines.
    pub(crate) fn scroll_by(&mut self, delta: i16) {
        let next = (i32::from(self.pos) + i32::from(delta)).clamp(0, i32::from(self.max));
        self.pos = u16::try_from(next).unwrap_or(u16::MAX);
    }
}

impl AppState {
    /// Pan the active job console. `dv`/`dh` are line/column deltas (negative =
    /// up/left). Clamps at 0; the vertical offset is clamped against the active
    /// job's line count and the horizontal offset against its widest line, so it
    /// can't scroll past the content into blank space in either axis.
    pub(crate) fn scroll_console(&mut self, dv: i16, dh: i16) {
        let job = self.active_job_id().and_then(|id| self.jobs.job(id));
        let max_v = job.map_or(0, |j| j.output.len().saturating_sub(1));
        let max_h = job.map_or(0, |j| {
            j.output
                .iter()
                .map(|l| l.chars().count())
                .max()
                .unwrap_or(0)
                .saturating_sub(1)
        });
        let mut v = ScrollState::capped_at(self.console_scroll, max_v);
        let mut h = ScrollState::capped_at(self.console_hscroll, max_h);
        v.scroll_by(dv);
        h.scroll_by(dh);
        self.console_scroll = v.pos;
        self.console_hscroll = h.pos;
    }

    /// Total log lines the wide-layout LOGS dock aggregates across all jobs.
    /// Single source for both the renderer's window and the scroll clamp.
    pub(crate) fn dock_logs_total(&self) -> usize {
        self.jobs.jobs.values().map(|j| j.output.len()).sum()
    }

    /// Scroll the wide-layout LOGS dock by `dv` lines (negative = toward newer).
    /// Counted up from the tail and clamped against the buffer minus the dock's
    /// visible height (derived from the last drawn dock rect).
    pub(crate) fn scroll_dock(&mut self, dv: i16) {
        let cap = self
            .last_dock_area
            .map_or(0, |r| r.height.saturating_sub(3) as usize);
        let mut scroll = ScrollState::capped_at(
            self.dock_logs_scroll,
            self.dock_logs_total().saturating_sub(cap),
        );
        scroll.scroll_by(dv);
        self.dock_logs_scroll = scroll.pos;
    }

    /// Apply an absolute chat offset against the latest measured viewport and
    /// derive follow-tail state from whether the viewport is at its bottom.
    pub(super) fn set_chat_scroll(&mut self, position: usize) {
        self.chat_scroll = u16::try_from(position)
            .unwrap_or(u16::MAX)
            .min(self.chat_max_scroll);
        self.chat_follow = self.chat_scroll == self.chat_max_scroll;
    }

    /// Arm a scrollbar drag and apply `position` in the target's own offset
    /// units. The grab offset is retained for subsequent pointer moves.
    pub(crate) fn apply_scroll_grab(
        &mut self,
        target: ScrollTarget,
        position: usize,
        grab_offset: u16,
    ) {
        self.scroll_drag = Some(ScrollDrag {
            target,
            grab_offset,
        });
        let p = u16::try_from(position).unwrap_or(u16::MAX);
        match target {
            ScrollTarget::Console => self.console_scroll = p,
            ScrollTarget::ConsoleH => self.console_hscroll = p,
            ScrollTarget::Chat => self.set_chat_scroll(position),
            ScrollTarget::DockLogs => self.dock_logs_scroll = p,
            ScrollTarget::InstanceDetail => self.instance_detail_scroll = p,
        }
    }

    /// Record a scrollbar drawn this frame for later mouse hit-testing.
    ///
    /// `area` is the rect passed to the scrollbar helper and `drawn` its return
    /// value; when they're equal no bar was drawn (content fit) and nothing is
    /// recorded. The 1-cell track strip is derived from `area` and `horizontal`.
    pub(crate) fn record_scrollbar(
        &self,
        area: ratatui::layout::Rect,
        drawn: ratatui::layout::Rect,
        horizontal: bool,
        content_len: usize,
        viewport_len: usize,
        target: ScrollTarget,
    ) {
        if let Some(h) =
            ScrollbarHandle::new(area, drawn, horizontal, content_len, viewport_len, target)
        {
            self.scrollbars.borrow_mut().push(h);
        }
    }

    /// Reset the instance Detail modal's scroll offset (called when opening
    /// the modal, so a stale offset never carries over from a previous
    /// instance's selection).
    pub const fn reset_instance_detail_scroll(&mut self) {
        self.instance_detail_scroll = 0;
        self.instance_detail_max_scroll = 0;
    }

    /// Adjust the instance Detail modal's scroll. `delta` is in lines;
    /// clamped against `[0, instance_detail_max_scroll]` (the latter is last
    /// written back by the renderer, see `instance_detail_max_scroll`), so
    /// `i16::MIN`/`i16::MAX` ("jump to start/end") land exactly on
    /// `0`/`instance_detail_max_scroll` instead of overflowing into an offset
    /// far past the real content length.
    pub fn scroll_instance_detail(&mut self, delta: i16) {
        let mut scroll =
            ScrollState::new(self.instance_detail_scroll, self.instance_detail_max_scroll);
        scroll.scroll_by(delta);
        self.instance_detail_scroll = scroll.pos;
    }
}

#[cfg(test)]
mod tests {
    use super::ScrollState;
    use crate::app::test_support::st;
    use ratatui::layout::Rect;

    fn scrolled(pos: u16, max: u16, delta: i16) -> u16 {
        let mut s = ScrollState::new(pos, max);
        s.scroll_by(delta);
        s.pos
    }

    #[test]
    fn scroll_state_moves_within_bounds() {
        assert_eq!(scrolled(3, 10, 4), 7);
        assert_eq!(scrolled(7, 10, -4), 3);
        assert_eq!(scrolled(3, 10, 0), 3, "zero delta is a no-op");
    }

    #[test]
    fn scroll_state_clamps_at_zero() {
        assert_eq!(scrolled(3, 10, -4), 0, "overshoot up clamps to 0");
        assert_eq!(scrolled(0, 10, -1), 0, "already at 0 stays at 0");
    }

    #[test]
    fn scroll_state_clamps_at_max() {
        assert_eq!(scrolled(8, 10, 5), 10, "overshoot down clamps to max");
        assert_eq!(scrolled(10, 10, 1), 10, "already at max stays at max");
        assert_eq!(scrolled(0, 0, 5), 0, "zero max pins to 0");
        assert_eq!(scrolled(0, 0, -5), 0);
    }

    #[test]
    fn scroll_state_pulls_a_stale_offset_back_to_max() {
        // The content shrank after the offset was set: even a no-op or an
        // upward nudge that stays past `max` lands on `max`.
        assert_eq!(scrolled(50, 10, 0), 10);
        assert_eq!(scrolled(50, 10, -5), 10);
        assert_eq!(scrolled(50, 10, -45), 5, "a move back inside max is kept");
    }

    #[test]
    fn scroll_state_large_deltas_land_exactly_on_the_ends() {
        assert_eq!(scrolled(4, 9, i16::MAX), 9, "jump to end");
        assert_eq!(scrolled(4, 9, i16::MIN), 0, "jump to start");
        assert_eq!(scrolled(u16::MAX, u16::MAX, i16::MIN), 32_767);
    }

    #[test]
    fn scroll_state_saturates_at_the_u16_ceiling() {
        assert_eq!(scrolled(u16::MAX, u16::MAX, i16::MAX), u16::MAX);
        assert_eq!(scrolled(u16::MAX - 1, u16::MAX, 5), u16::MAX);
        assert_eq!(scrolled(u16::MAX, u16::MAX, 0), u16::MAX);
    }

    #[test]
    fn scroll_state_capped_at_saturates_an_oversized_bound() {
        let mut s = ScrollState::capped_at(0, usize::MAX);
        assert_eq!(s.max, u16::MAX, "bound saturates to the u16 ceiling");
        s.scroll_by(i16::MAX);
        s.scroll_by(i16::MAX);
        s.scroll_by(i16::MAX);
        assert_eq!(s.pos, u16::MAX, "offset saturates rather than wrapping");
        assert_eq!(ScrollState::capped_at(2, 7), ScrollState::new(2, 7));
    }

    #[test]
    fn scroll_dock_clamps_against_buffer() {
        let mut s = st();
        // Dock 10 rows tall → ~7 visible lines after border/padding.
        s.last_dock_area = Some(Rect::new(0, 0, 52, 10));
        s.jobs.apply(rocm_dash_core::state::StateEvent::StartJob {
            id: "logs".into(),
            cmd: "rocm".into(),
            args: vec!["logs".into()],
        });
        for i in 0..20 {
            s.jobs.apply(rocm_dash_core::state::StateEvent::JobLine {
                id: "logs".into(),
                line: format!("line {i}"),
            });
        }
        // 20 lines, ~7 visible → max scroll-up is bounded, never past the top.
        s.scroll_dock(100);
        assert!(s.dock_logs_scroll <= 13, "clamped: {}", s.dock_logs_scroll);
        assert!(s.dock_logs_scroll > 0, "scrolled up some");
        s.scroll_dock(-100);
        assert_eq!(s.dock_logs_scroll, 0, "back to the tail");
    }

    #[test]
    fn scroll_instance_detail_clamps_to_measured_max() {
        let mut s = st();
        s.instance_detail_max_scroll = 9;
        // i16::MAX is the jump-to-end gesture; it must land on the measured
        // max, not overflow past it.
        s.scroll_instance_detail(i16::MAX);
        assert_eq!(s.instance_detail_scroll, 9, "jump-to-end clamps to max");
        // i16::MIN is jump-to-start; it must land on 0, not underflow.
        s.scroll_instance_detail(i16::MIN);
        assert_eq!(s.instance_detail_scroll, 0, "jump-to-start clamps to 0");
    }

    #[test]
    fn scroll_console_clamps_and_tracks_output() {
        let mut s = st();
        // No console → both axes clamp to 0 (no content to pan over).
        s.scroll_console(5, 5);
        assert_eq!(s.console_scroll, 0);
        assert_eq!(s.console_hscroll, 0, "no console → horizontal clamps to 0");
        // With a console of N lines, vertical clamps to N-1.
        s.jobs.apply(rocm_dash_core::state::StateEvent::StartJob {
            id: "logs".into(),
            cmd: "rocm".into(),
            args: vec!["logs".into()],
        });
        for i in 0..4 {
            s.jobs.apply(rocm_dash_core::state::StateEvent::JobLine {
                id: "logs".into(),
                line: format!("line {i}"),
            });
        }
        s.logs_view = Some(crate::ui::logs_view::LogsViewState {
            active_job: Some("logs".into()),
            ..Default::default()
        });
        s.console_scroll = 0;
        s.scroll_console(100, 0);
        assert_eq!(s.console_scroll, 3, "clamped to output.len()-1 (4 lines)");
        s.scroll_console(-100, 0);
        assert_eq!(s.console_scroll, 0);
        // Horizontal clamps to the widest line minus one ("line 0" = 6 chars).
        s.scroll_console(0, 100);
        assert_eq!(s.console_hscroll, 5, "clamped to max line width - 1");
        s.scroll_console(0, -100);
        assert_eq!(s.console_hscroll, 0);
    }
}
