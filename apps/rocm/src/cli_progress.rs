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
    /// (`last_progress_paint` starts unset) and the final chunk (`bytes >=
    /// total`) always repaint, so the first and last frames shown are never
    /// stale.
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

/// Truncates `line` (by character count) to fit within `max_width` columns,
/// appending an ellipsis when it doesn't already fit, so a repaint can never
/// wrap to a second terminal row.
fn truncate_to_width(line: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if line.chars().count() <= max_width {
        return line.to_owned();
    }
    let keep = max_width.saturating_sub(1);
    let mut truncated: String = line.chars().take(keep).collect();
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
        let inner = Arc::new(Mutex::new(Spinner::new(label)));
        inner.lock().unwrap().tick();
        let stop = Arc::new(AtomicBool::new(false));
        let ticker = {
            let inner = Arc::clone(&inner);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    thread::sleep(interval);
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    inner.lock().unwrap().tick();
                }
            })
        };
        Self {
            inner,
            stop,
            ticker: Some(ticker),
        }
    }

    /// Repaint with a byte-progress label. See [`Spinner::set_progress`].
    pub(crate) fn set_progress(&self, prefix: &str, bytes: u64, total: Option<u64>) {
        self.inner.lock().unwrap().set_progress(prefix, bytes, total);
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
            // outstanding. 100% is reserved for `bytes >= total`.
            let pct = if bytes >= total {
                100
            } else {
                ((bytes as f64 / total as f64) * 100.0).floor() as u64
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
        assert_eq!(truncated.chars().count(), 10);
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
    fn animated_spinner_keeps_ticking_without_progress_calls() {
        let spinner = AnimatedSpinner::start_with_interval("Downloading…", Duration::from_millis(5));
        thread::sleep(Duration::from_millis(60));
        let idx = spinner.inner.lock().unwrap().idx;
        assert!(
            idx >= 3,
            "the background ticker must keep advancing frames on its own: idx={idx}"
        );
    }
}
