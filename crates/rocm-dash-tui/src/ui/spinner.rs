// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Animated braille spinner and progress-percentage parsing.
//!
//! Long jobs stream progress via carriage-return redraws (pip, tqdm,
//! huggingface). The console pairs an animated braille glyph with the
//! percentage parsed from the latest output line, so a running job reads as
//! live progress instead of a wall of text. Pure helpers — no I/O, no state.

/// Braille spinner frames, advanced one step per UI repaint tick (~250ms).
const BRAILLE_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// The spinner glyph for a given monotonic tick count.
pub const fn spinner_frame(tick: u64) -> char {
    BRAILLE_FRAMES[(tick % BRAILLE_FRAMES.len() as u64) as usize]
}

/// Parse a progress percentage (0–100) from a raw output line, if present.
///
/// Matches the last `NN%` token — the form pip, tqdm, and huggingface all
/// print. Returns `None` when no percent token is found or the value falls
/// outside 0–100 (e.g. a literal `%` with no leading number, or a bogus value).
pub fn parse_progress_pct(line: &str) -> Option<u8> {
    let pct_idx = line.rfind('%')?;
    // Walk back over the numeric run (digits and one-or-more dots) before '%'.
    let rev: String = line[..pct_idx]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    if rev.is_empty() {
        return None;
    }
    let num: String = rev.chars().rev().collect();
    let val: f32 = num.parse().ok()?;
    if (0.0..=100.0).contains(&val) {
        Some(val.round() as u8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinner_cycles_through_all_frames() {
        assert_eq!(spinner_frame(0), '⠋');
        assert_eq!(spinner_frame(1), '⠙');
        // Wraps after the last frame.
        assert_eq!(spinner_frame(10), spinner_frame(0));
        assert_eq!(spinner_frame(21), spinner_frame(1));
    }

    #[test]
    fn parses_percent_from_progress_lines() {
        assert_eq!(
            parse_progress_pct("Downloading model.safetensors:  45%|████▌     | 2.3G/5.2G"),
            Some(45)
        );
        assert_eq!(parse_progress_pct("100%|██████████| done"), Some(100));
        assert_eq!(parse_progress_pct("progress: 0%"), Some(0));
        // Decimal rounds to nearest whole percent.
        assert_eq!(parse_progress_pct("45.6% complete"), Some(46));
    }

    #[test]
    fn rejects_non_progress_lines() {
        assert_eq!(parse_progress_pct("no percent here"), None);
        assert_eq!(parse_progress_pct("just a % sign"), None);
        // Out of range is not a percentage.
        assert_eq!(parse_progress_pct("cpu at 250% load"), None);
    }
}
