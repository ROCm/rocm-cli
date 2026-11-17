// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Validation of a requested terminal size, and the order in which the emulated
//! screen and the real PTY take it.

use std::num::NonZeroU16;
use std::sync::Mutex;

/// `rows` and `cols` as the `NonZeroU16`s the `vt100` emulator takes, or an
/// error naming the requested size if either is zero.
///
/// A pure function on purpose: the PTY driver lives in the cucumber binary,
/// where plain `#[test]`s never run, so the zero check is kept here where
/// `cargo test -p e2e-cucumber --lib` covers it.
pub fn non_zero_dimensions(rows: u16, cols: u16) -> Result<(NonZeroU16, NonZeroU16), String> {
    match (NonZeroU16::new(rows), NonZeroU16::new(cols)) {
        (Some(rows), Some(cols)) => Ok((rows, cols)),
        _ => Err(format!(
            "cannot resize to rows={rows} cols={cols}: zero dimension"
        )),
    }
}

/// Resize the emulated screen first, then the real PTY.
///
/// `resize_pty` is the call that resizes the real PTY and so signals the child.
/// It runs after the emulated screen has taken the new size and with the
/// emulator unlocked, so the reader thread keeps draining the child's output
/// while it runs. If it fails, the emulated screen goes back to its old size so
/// the two never disagree; that restores the size only, not any cells a shrink
/// truncated, which is acceptable because a failed resize fails the step.
///
/// The order is the point. The child redraws the moment it learns the new size,
/// and the reader thread feeds whatever it writes to the emulator. Resizing the
/// PTY first and the emulator second leaves a window between the two calls in
/// which a frame laid out for the new width is parsed by the old, narrower
/// screen: its lines wrap and scroll the top rows (the header among them) off,
/// and because a terminal UI only repaints the cells that changed, nothing ever
/// repairs the damage. Any scheduling stall of the harness thread between the
/// two calls (a loaded CI runner) opens that window. With the emulator resized
/// first, whatever the child writes in answer to the resize is parsed at the new
/// size.
///
/// The same argument is not made, and not tested, for shrinking: a frame the
/// child drew for the old, larger size can reach the already smaller emulator
/// and wrap there, but the child clears and redraws on a resize, so that residue
/// is repainted over. Shrinking relies on the child doing so; a test of it would
/// pass with either order and so would not guard anything.
///
/// The emulator is deliberately not held locked across `resize_pty`: the reader
/// drains the PTY under that lock, and a PTY resize that waits for the child's
/// output to drain (which a Windows pseudoconsole is not documented to rule out)
/// would then deadlock against it.
///
/// Pure of any PTY on purpose, like [`non_zero_dimensions`]: the driver lives in
/// the cucumber binary, where plain `#[test]`s never run, so the ordering is
/// covered here by `cargo test -p e2e-cucumber --lib`.
pub fn resize_screen_then_pty(
    parser: &Mutex<vt100::Parser>,
    rows: u16,
    cols: u16,
    resize_pty: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    // Validate before touching either side, so a zero dimension fails cleanly
    // instead of leaving the PTY and the emulated screen at different sizes.
    let (vt_rows, vt_cols) = non_zero_dimensions(rows, cols)?;
    // Recover rather than fail on a poisoned lock: resizing is still meaningful
    // even if some earlier operation panicked while holding it.
    let lock = || {
        parser
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    };
    let (old_rows, old_cols) = {
        let mut parser = lock();
        let old = parser.screen().size();
        parser.screen_mut().set_size(vt_rows, vt_cols);
        old
    };
    let resized = resize_pty();
    if resized.is_err() {
        lock().screen_mut().set_size(old_rows, old_cols);
    }
    resized
}

#[cfg(test)]
mod tests {
    use super::{non_zero_dimensions, resize_screen_then_pty};
    use std::fmt::Write as _;
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::Duration;

    #[test]
    fn accepts_non_zero_dimensions_and_keeps_their_order() {
        let (rows, cols) = non_zero_dimensions(24, 80).expect("24x80 is a valid size");
        assert_eq!((rows.get(), cols.get()), (24, 80));
    }

    #[test]
    fn rejects_zero_rows_and_reports_the_requested_size() {
        assert_eq!(
            non_zero_dimensions(0, 80),
            Err("cannot resize to rows=0 cols=80: zero dimension".to_owned())
        );
    }

    #[test]
    fn rejects_zero_cols_and_reports_the_requested_size() {
        assert_eq!(
            non_zero_dimensions(24, 0),
            Err("cannot resize to rows=24 cols=0: zero dimension".to_owned())
        );
    }

    #[test]
    fn rejects_zero_in_both_dimensions() {
        assert_eq!(
            non_zero_dimensions(0, 0),
            Err("cannot resize to rows=0 cols=0: zero dimension".to_owned())
        );
    }

    /// Cursor-addressed frame of `rows` full-width lines, as a terminal UI
    /// writes it after a resize: clear, then every row at an absolute position.
    fn frame(rows: u16, cols: u16) -> Vec<u8> {
        let mut out = String::from("\x1b[2J");
        for row in 0..rows {
            let label = format!("row{row:02}");
            let fill = "x".repeat(usize::from(cols) - label.len());
            let _ = write!(out, "\x1b[{};1H{label}{fill}", row + 1);
        }
        out.into_bytes()
    }

    fn parser(rows: u16, cols: u16) -> Arc<Mutex<vt100::Parser>> {
        let (rows, cols) = non_zero_dimensions(rows, cols).expect("a valid test size");
        Arc::new(Mutex::new(vt100::Parser::new(rows, cols, 0)))
    }

    fn size(parser: &Mutex<vt100::Parser>) -> (u16, u16) {
        let (rows, cols) = parser.lock().unwrap().screen().size();
        (rows.get(), cols.get())
    }

    #[test]
    fn a_frame_the_child_draws_in_answer_to_the_resize_lands_on_the_new_size() {
        let parser = parser(24, 80);
        // The reader thread parses the child's answer the moment it is released,
        // under the parser lock, exactly as the driver's reader does.
        let (go, released) = mpsc::channel::<()>();
        let (parsed, wait_parsed) = mpsc::channel::<()>();
        let reader = {
            let parser = Arc::clone(&parser);
            std::thread::spawn(move || {
                released
                    .recv()
                    .expect("the resize never signalled the child");
                parser.lock().unwrap().process(&frame(40, 120));
                parsed.send(()).unwrap();
            })
        };

        let resized = resize_screen_then_pty(&parser, 40, 120, || {
            // The child redraws as soon as it is signalled, and the harness
            // thread does not return from the PTY call until the reader has
            // parsed that redraw: the window a PTY-first order leaves open (a
            // frame parsed before the emulator has its new size), and, if the
            // emulator were kept locked across this call, one the reader could
            // never get through, so the wait below would time out.
            go.send(()).unwrap();
            wait_parsed
                .recv_timeout(Duration::from_secs(5))
                .map_err(|_| {
                    "the reader could not parse the child's redraw while the PTY resize ran"
                        .to_owned()
                })
        });
        assert_eq!(resized, Ok(()));
        reader.join().unwrap();

        assert_eq!(size(&parser), (40, 120));
        let guard = parser.lock().unwrap();
        let screen = guard.screen();
        for (row, text) in screen.rows(0, 120).enumerate() {
            let label = format!("row{row:02}");
            assert_eq!(
                text,
                format!("{label}{}", "x".repeat(120 - label.len())),
                "row {row} was not drawn intact: a frame laid out for 120 columns \
                 was parsed by a screen still sized for the old geometry"
            );
        }
    }

    #[test]
    fn a_failed_pty_resize_puts_the_emulated_screen_back_and_reports_the_error() {
        let parser = parser(24, 80);
        let resized =
            resize_screen_then_pty(
                &parser,
                40,
                120,
                || Err("failed to resize pty: boom".into()),
            );
        assert_eq!(resized, Err("failed to resize pty: boom".to_owned()));
        assert_eq!(size(&parser), (24, 80));
    }

    #[test]
    fn a_zero_dimension_resizes_neither_side() {
        let parser = parser(24, 80);
        let mut pty_resized = false;
        let resized = resize_screen_then_pty(&parser, 0, 120, || {
            pty_resized = true;
            Ok(())
        });
        assert_eq!(
            resized,
            Err("cannot resize to rows=0 cols=120: zero dimension".to_owned())
        );
        assert!(!pty_resized, "the PTY was resized for an invalid size");
        assert_eq!(size(&parser), (24, 80));
    }
}
