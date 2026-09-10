// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared TTY-gated status indicator for long-running CLI operations
//! (starting a server, downloading a large artifact). Written to stderr only,
//! so piped/redirected output — and stdout, which callers may still be
//! printing a final summary to — never sees control characters.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossterm::QueueableCommand;
use crossterm::cursor::MoveToColumn;
use crossterm::terminal::{Clear, ClearType};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Braille spinner frames (matching the dashboard's visual language).
const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// A byte-progress repaint fires at most this often. A 64 KiB read cadence on
/// a fast local link would otherwise flood the terminal with far more
/// repaints per second than a human can perceive.
const MIN_PROGRESS_REPAINT_INTERVAL: Duration = Duration::from_millis(100);

/// How often [`AnimatedSpinner`]'s background thread repaints while idle, so
/// a stalled transfer still visibly animates instead of looking hung.
const IDLE_TICK_INTERVAL: Duration = Duration::from_millis(200);

/// A carriage-return status indicator written to stderr. Disabled (a no-op) when
/// stderr is not a TTY, so piped/redirected output never receives control
/// characters. Keeps stdout clean for whatever the caller prints afterward.
pub(crate) struct Spinner {
    enabled: bool,
    idx: usize,
    label: String,
    active: bool,
    last_progress_paint: Option<Instant>,
    max_progress_bytes: u64,
}

impl Spinner {
    pub(crate) fn new(label: impl Into<String>) -> Self {
        Self {
            enabled: std::io::stderr().is_terminal(),
            idx: 0,
            label: label.into(),
            active: false,
            last_progress_paint: None,
            max_progress_bytes: 0,
        }
    }

    /// Change the message shown next to the spinner (e.g. "Running smoke test…").
    pub(crate) fn set_label(&mut self, label: impl Into<String>) {
        self.label = label.into();
        self.render_current();
    }

    /// Advance to the next animation frame and repaint.
    pub(crate) fn tick(&mut self) {
        self.idx = self.idx.wrapping_add(1);
        self.render_current();
    }

    /// Repaint with a byte-progress label. Throttled to at most one repaint
    /// per [`MIN_PROGRESS_REPAINT_INTERVAL`], except the very first call
    /// (`last_progress_paint` starts unset) always repaints, as does the
    /// final chunk when the total size is known (`bytes >= total`). With an
    /// unknown total there is no final-chunk signal to detect, so the true
    /// last frame is subject to the same throttle as any other and may be
    /// swallowed.
    ///
    /// `bytes` is clamped to a high-water mark: a retried transfer that
    /// restarts from zero (or resumes from an earlier offset than what was
    /// already shown) never visibly regresses the displayed count.
    pub(crate) fn set_progress(&mut self, prefix: &str, bytes: u64, total: Option<u64>) {
        let bytes = bytes.max(self.max_progress_bytes);
        self.max_progress_bytes = bytes;
        let is_final = total.is_some_and(|total| bytes >= total);
        let now = Instant::now();
        if !is_final
            && let Some(last) = self.last_progress_paint
            && now.duration_since(last) < MIN_PROGRESS_REPAINT_INTERVAL
        {
            return;
        }
        self.last_progress_paint = Some(now);
        self.idx = self.idx.wrapping_add(1);
        self.label = format_download_progress(prefix, bytes, total);
        self.render_current();
    }

    fn render_current(&mut self) {
        if !self.enabled {
            return;
        }
        let frame = SPINNER_FRAMES[self.idx % SPINNER_FRAMES.len()];
        let mut line = format!("{frame} {}", self.label);
        if let Ok((cols, _)) = crossterm::terminal::size() {
            // A line that fits exactly at `cols` still wraps on some terminals
            // once the cursor lands in the last column, and `Clear::CurrentLine`
            // on the next repaint can only erase the row the cursor ends up on
            // — not a wrapped-over first row. Leaving one column of slack keeps
            // every repaint confined to a single row.
            line = truncate_to_width(&line, cols.saturating_sub(1) as usize);
        }
        let mut err = std::io::stderr();
        let _ = err.queue(MoveToColumn(0));
        let _ = err.queue(Clear(ClearType::CurrentLine));
        let _ = write!(err, "{line}");
        let _ = err.flush();
        self.active = true;
    }

    /// Erase the spinner line so whatever prints next starts on a clean line.
    pub(crate) fn clear(&mut self) {
        if self.enabled && self.active {
            let mut err = std::io::stderr();
            let _ = err.queue(MoveToColumn(0));
            let _ = err.queue(Clear(ClearType::CurrentLine));
            let _ = err.flush();
            self.active = false;
        }
    }
}

/// Truncates `line` (by Unicode display width, not character count — a wide
/// CJK glyph occupies two terminal columns) to fit within `max_width`
/// columns, appending an ellipsis when it doesn't already fit, so a repaint
/// can never wrap to a second terminal row.
fn truncate_to_width(line: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if line.width() <= max_width {
        return line.to_owned();
    }
    let ellipsis_width = '…'.width().unwrap_or(1);
    let keep_width = max_width.saturating_sub(ellipsis_width);
    let mut truncated = String::new();
    let mut used_width = 0;
    for ch in line.chars() {
        let ch_width = ch.width().unwrap_or(0);
        if used_width + ch_width > keep_width {
            break;
        }
        truncated.push(ch);
        used_width += ch_width;
    }
    truncated.push('…');
    truncated
}

/// A [`Spinner`] kept animating by a background thread, for callers whose
/// progress signal can go quiet for long stretches — a stalled download's
/// `on_progress` callback only fires when bytes actually arrive, unlike
/// `serve`'s HTTP-polling wait loop, which already ticks on every iteration
/// regardless of readiness. Clears the line and stops the thread on drop.
pub(crate) struct AnimatedSpinner {
    inner: Arc<Mutex<Spinner>>,
    stop: Arc<AtomicBool>,
    ticker: Option<JoinHandle<()>>,
}

impl AnimatedSpinner {
    pub(crate) fn start(label: impl Into<String>) -> Self {
        Self::start_with_interval(label, IDLE_TICK_INTERVAL)
    }

    fn start_with_interval(label: impl Into<String>, interval: Duration) -> Self {
        Self::start_with_interval_impl(label, interval, None)
    }

    /// Like [`Self::start_with_interval`], but overrides whether the spinner
    /// is treated as enabled instead of probing stderr. Test-only: whether
    /// the ticker thread spawns depends on `Spinner::enabled`, which
    /// `Spinner::new` derives from the real `stderr().is_terminal()` — a
    /// property of however the test happens to be run, not of the behavior
    /// under test. Forcing it here keeps these tests deterministic whether
    /// `cargo test` is launched from an interactive terminal or not.
    #[cfg(test)]
    fn start_with_interval_enabled(
        label: impl Into<String>,
        interval: Duration,
        enabled: bool,
    ) -> Self {
        Self::start_with_interval_impl(label, interval, Some(enabled))
    }

    fn start_with_interval_impl(
        label: impl Into<String>,
        interval: Duration,
        enabled_override: Option<bool>,
    ) -> Self {
        let mut spinner = Spinner::new(label);
        if let Some(enabled) = enabled_override {
            spinner.enabled = enabled;
        }
        let inner = Arc::new(Mutex::new(spinner));
        inner.lock().unwrap().tick();
        let stop = Arc::new(AtomicBool::new(false));
        // A ticker thread only exists to keep repainting an already-visible
        // spinner; when stderr isn't a TTY every repaint it would trigger is
        // a no-op, so skip holding an OS thread open for the whole download.
        let ticker = if inner.lock().unwrap().enabled {
            let inner = Arc::clone(&inner);
            let stop = Arc::clone(&stop);
            Some(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    thread::sleep(interval);
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    inner.lock().unwrap().tick();
                }
            }))
        } else {
            None
        };
        Self {
            inner,
            stop,
            ticker,
        }
    }

    /// Repaint with a byte-progress label. See [`Spinner::set_progress`].
    pub(crate) fn set_progress(&self, prefix: &str, bytes: u64, total: Option<u64>) {
        self.inner
            .lock()
            .unwrap()
            .set_progress(prefix, bytes, total);
    }
}

impl Drop for AnimatedSpinner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(ticker) = self.ticker.take() {
            let _ = ticker.join();
        }
        self.inner.lock().unwrap().clear();
    }
}

/// e.g. `"Downloading SDK tarball… 842.1 MiB / 3.2 GiB (26%)"`, or
/// `"Downloading SDK tarball… 842.1 MiB"` when the total is unknown (the
/// server never reported a `Content-Length`).
pub(crate) fn format_download_progress(prefix: &str, bytes: u64, total: Option<u64>) -> String {
    match total {
        Some(total) if total > 0 => {
            // Floor rather than round: a multi-gigabyte transfer sitting at
            // 99.5% must not be shown as "complete" while bytes are still
            // outstanding. 100% is reserved for `bytes >= total`. Integer
            // arithmetic in u128 (rather than an f64 ratio) avoids adjacent
            // huge u64 values collapsing to the same float and reporting
            // 100% early.
            let pct = if bytes >= total {
                100
            } else {
                ((u128::from(bytes) * 100) / u128::from(total)) as u64
            };
            format!(
                "{prefix} {} / {} ({pct}%)",
                rocm_core::format_bytes(bytes),
                rocm_core::format_bytes(total)
            )
        }
        _ => format!("{prefix} {}", rocm_core::format_bytes(bytes)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_download_progress_shows_bytes_and_percent_when_total_is_known() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(
            format_download_progress("Downloading…", gib, Some(4 * gib)),
            "Downloading… 1.0 GiB / 4.0 GiB (25%)"
        );
    }

    #[test]
    fn format_download_progress_omits_total_when_unknown() {
        let rendered = format_download_progress("Downloading…", 883_147_264, None);
        assert!(
            !rendered.contains('/') && !rendered.contains('%'),
            "no total means no fraction or percentage: {rendered}"
        );
        assert!(rendered.starts_with("Downloading… "));
    }

    #[test]
    fn format_download_progress_clamps_percent_at_100_when_bytes_exceeds_total() {
        let rendered = format_download_progress("Downloading…", 105, Some(100));
        assert!(
            rendered.contains("(100%)"),
            "a server sending a few bytes past its declared length must not report over 100%: {rendered}"
        );
    }

    #[test]
    fn format_download_progress_does_not_round_up_to_100_before_completion() {
        let rendered = format_download_progress("Downloading…", 995, Some(1000));
        assert!(
            rendered.contains("(99%)"),
            "99.5% must floor to 99%, not round up to a premature 100%: {rendered}"
        );
    }

    #[test]
    fn format_download_progress_does_not_round_up_to_100_for_huge_totals() {
        // An f64 ratio can't distinguish adjacent values this close to
        // u64::MAX — it collapses to 1.0 and would misreport 100% while a
        // byte is still outstanding. Integer arithmetic must not.
        let rendered = format_download_progress("Downloading…", u64::MAX - 1, Some(u64::MAX));
        assert!(
            !rendered.contains("(100%)"),
            "a single outstanding byte out of u64::MAX must not show as complete: {rendered}"
        );
    }

    #[test]
    fn set_progress_never_displays_fewer_bytes_than_already_shown() {
        let mut spinner = Spinner::new("Downloading…");
        spinner.set_progress("Downloading…", 900, Some(1000));
        assert!(spinner.label.contains("900"));
        // A retried transfer restarts its own byte count from a lower offset.
        // Force this repaint past the throttle (via a small `total` that the
        // clamped byte count already exceeds) to prove the clamp itself, not
        // just that the repaint was skipped.
        spinner.set_progress("Downloading…", 100, Some(500));
        assert!(
            spinner.label.contains("900"),
            "progress must not regress after a retry: {}",
            spinner.label
        );
    }

    #[test]
    fn truncate_to_width_leaves_short_lines_untouched() {
        assert_eq!(truncate_to_width("⠋ short", 40), "⠋ short");
        assert_eq!(truncate_to_width("⠋ exact", 7), "⠋ exact");
    }

    #[test]
    fn truncate_to_width_ellipsizes_overlong_lines() {
        let truncated = truncate_to_width("⠋ a very long download progress line", 10);
        assert_eq!(truncated.width(), 10);
        assert!(
            truncated.ends_with('…'),
            "overlong line must end with an ellipsis marker: {truncated}"
        );
    }

    #[test]
    fn truncate_to_width_handles_zero_width() {
        assert_eq!(truncate_to_width("anything", 0), "");
    }

    #[test]
    fn truncate_to_width_accounts_for_wide_characters() {
        // Each 下 occupies two terminal columns, so a naive char-count
        // truncation would keep too many of them and still overflow the row.
        let truncated = truncate_to_width("下载中下载中下载中", 10);
        assert!(
            truncated.width() <= 10,
            "display width must respect max_width even with wide glyphs: {truncated} ({})",
            truncated.width()
        );
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn animated_spinner_keeps_ticking_without_progress_calls() {
        let spinner = AnimatedSpinner::start_with_interval_enabled(
            "Downloading…",
            Duration::from_millis(5),
            true,
        );
        thread::sleep(Duration::from_millis(60));
        let idx = spinner.inner.lock().unwrap().idx;
        assert!(
            idx >= 3,
            "the background ticker must keep advancing frames on its own: idx={idx}"
        );
    }

    #[test]
    fn animated_spinner_skips_the_ticker_thread_when_disabled() {
        // Force `enabled = false` explicitly rather than relying on stderr
        // not being a TTY in the test process, so this stays deterministic
        // whether `cargo test` runs piped (CI) or from an interactive shell.
        let spinner = AnimatedSpinner::start_with_interval_enabled(
            "Downloading…",
            Duration::from_millis(5),
            false,
        );
        assert!(
            spinner.ticker.is_none(),
            "no ticker thread should be spawned when stderr isn't a TTY"
        );
    }
}
