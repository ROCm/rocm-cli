// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What of an assistant reply is read aloud.
//!
//! A reply can carry code, command lines, tables and long lists that are
//! useless spoken. Only a short plain-language summary is spoken; the full
//! reply always stays on screen.

/// Speak at most this many characters of a reply.
const MAX_SPOKEN_CHARS: usize = 280;
/// And at most this many sentences.
const MAX_SPOKEN_SENTENCES: usize = 2;

/// The part of `reply` worth speaking, or `None` when nothing in it is prose
/// (a reply that is all code, say).
#[must_use]
pub fn speakable_summary(reply: &str) -> Option<String> {
    let mut prose = String::new();
    let mut in_code_block = false;
    for line in reply.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block
            || trimmed.is_empty()
            || trimmed.starts_with('|')
            || trimmed.starts_with("$ ")
            || trimmed.starts_with("    ")
        {
            if !prose.is_empty() && !prose.ends_with(' ') {
                prose.push(' ');
            }
            continue;
        }
        let text: String = trimmed
            .trim_start_matches(['#', '-', '*', '>', ' '])
            .chars()
            .filter(|ch| !matches!(ch, '*' | '_' | '`' | '#'))
            .collect();
        if text.trim().is_empty() {
            continue;
        }
        if !prose.is_empty() && !prose.ends_with(' ') {
            prose.push(' ');
        }
        prose.push_str(text.trim());
    }
    let prose = prose.trim();
    if prose.is_empty() {
        return None;
    }
    let mut summary = String::new();
    let mut sentences = 0;
    for (index, ch) in prose.char_indices() {
        summary.push(ch);
        let ends_sentence = matches!(ch, '.' | '!' | '?')
            && prose[index + ch.len_utf8()..]
                .chars()
                .next()
                .is_none_or(char::is_whitespace);
        if ends_sentence {
            sentences += 1;
            if sentences == MAX_SPOKEN_SENTENCES {
                break;
            }
        }
        if summary.chars().count() >= MAX_SPOKEN_CHARS {
            // Cut at the last word boundary rather than mid-word.
            if let Some(space) = summary.rfind(' ') {
                summary.truncate(space);
            }
            summary.push_str(" ...");
            break;
        }
    }
    Some(summary.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_first_two_sentences_are_spoken() {
        assert_eq!(
            speakable_summary("Your GPU is ready. ROCm 7 is active. Nothing else to do.")
                .as_deref(),
            Some("Your GPU is ready. ROCm 7 is active.")
        );
    }

    #[test]
    fn code_and_command_lines_are_left_on_screen() {
        let reply = "Run this to start the server:\n```\nrocm serve qwen\n```\nIt takes a minute.";
        assert_eq!(
            speakable_summary(reply).as_deref(),
            Some("Run this to start the server: It takes a minute.")
        );
    }

    #[test]
    fn markdown_markup_is_not_read_out() {
        assert_eq!(
            speakable_summary("## Status\n- **GPU**: ready").as_deref(),
            Some("Status GPU: ready")
        );
    }

    #[test]
    fn a_reply_with_no_prose_is_not_spoken() {
        assert_eq!(speakable_summary("```\n$ rocm examine\n```"), None);
        assert_eq!(speakable_summary("   "), None);
    }

    #[test]
    fn a_long_sentence_is_cut_at_a_word() {
        let reply = "word ".repeat(200);
        let spoken = speakable_summary(&reply).expect("prose");
        assert!(spoken.chars().count() <= MAX_SPOKEN_CHARS + 4);
        assert!(spoken.ends_with(" ..."));
    }
}
