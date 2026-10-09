// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Split-invariance properties for the SSE line emitters (`OpenAiSseEmitter`
//! and `AnthropicSseEmitter`): however the bytes of a stream arrive, the
//! emitted events are the ones the whole body produces in one piece.
//!
//! The fixtures are the SSE bodies the unit tests in `providers.rs` already
//! use, plus the same shapes with CRLF line endings (what Python SSE servers
//! emit by default) and multi-byte UTF-8 content, so that a split can land
//! between `\r` and `\n` and inside a UTF-8 sequence. None of them generates
//! SSE text from scratch. The transport readers that feed these emitters
//! (chunked and close-delimited bodies) are not covered here.

use super::*;
use proptest::prelude::*;

const OPENAI_FIXTURES: &[&str] = &[
    // providers.rs `parses_openai_sse_chat_events`.
    "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n",
    // Same body, CRLF line endings.
    "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n",
    // Multi-byte content (2-, 3- and 4-byte sequences), CRLF, no space after
    // `data:`, a role-only first delta, and a usage chunk with empty choices.
    "data:{\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"caf\u{e9} \u{6f22}\u{5b57} \u{1f680}\"}}]}\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" \u{4e16}\u{754c}\"}}]}\r\n\r\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":3}}\r\n\r\ndata: [DONE]\r\n\r\n",
];

const ANTHROPIC_FIXTURE: &str = "event: content_block_delta\r\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"h\u{e9}l\u{1f680}\"}}\r\n\r\nevent: content_block_delta\r\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\r\n\r\nevent: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n";

/// Case count for the randomized property: `default`, unless `PROPTEST_CASES`
/// (proptest's own knob, which an explicit count would otherwise override)
/// asks for a longer run.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn openai_events_from_pieces(pieces: &[&[u8]]) -> Vec<ProviderStreamEvent> {
    let mut events = Vec::new();
    let mut on_event = |event: ProviderStreamEvent| {
        events.push(event);
        Ok(())
    };
    let mut emitter = OpenAiSseEmitter::new(&mut on_event);
    for piece in pieces {
        emitter.push_bytes(piece).expect("fixture lines parse");
    }
    emitter.finish().expect("fixture stream finishes");
    events
}

fn anthropic_events_from_pieces(pieces: &[&[u8]]) -> Vec<ProviderStreamEvent> {
    let mut events = Vec::new();
    let mut on_event = |event: ProviderStreamEvent| {
        events.push(event);
        Ok(())
    };
    let mut emitter = AnthropicSseEmitter::new(&mut on_event);
    for piece in pieces {
        emitter.push_bytes(piece).expect("fixture lines parse");
    }
    emitter.finish().expect("fixture stream finishes");
    events
}

/// Split `bytes` at the given (unsorted, possibly duplicated) offsets.
fn split_at_offsets<'a>(bytes: &'a [u8], offsets: &[usize]) -> Vec<&'a [u8]> {
    let mut cuts: Vec<usize> = offsets.iter().map(|o| o % (bytes.len() + 1)).collect();
    cuts.sort_unstable();
    cuts.dedup();
    let mut pieces = Vec::new();
    let mut start = 0;
    for cut in cuts {
        pieces.push(&bytes[start..cut]);
        start = cut;
    }
    pieces.push(&bytes[start..]);
    pieces
}

/// How many split points landed in each place a line emitter is most likely to
/// get wrong.
#[derive(Default, Debug)]
struct SplitReach {
    total: usize,
    mid_utf8: usize,
    mid_crlf: usize,
    mid_data_prefix: usize,
}

impl SplitReach {
    fn record(&mut self, bytes: &[u8], cut: usize) {
        self.total += 1;
        if cut > 0 && cut < bytes.len() {
            if bytes[cut] & 0b1100_0000 == 0b1000_0000 {
                self.mid_utf8 += 1;
            }
            if bytes[cut - 1] == b'\r' && bytes[cut] == b'\n' {
                self.mid_crlf += 1;
            }
            let line_start = bytes[..cut]
                .iter()
                .rposition(|b| *b == b'\n')
                .map_or(0, |i| i + 1);
            if cut - line_start < 5 && bytes[line_start..].starts_with(b"data:") && cut > line_start
            {
                self.mid_data_prefix += 1;
            }
        }
    }
}

/// Every single split point, and every pair of split points, of every fixture:
/// exhaustive, so mid-UTF-8 and mid-CRLF splits are guaranteed to be reached
/// rather than hoped for.
#[test]
fn sse_emitters_are_invariant_under_every_one_and_two_point_split() {
    let mut reach = SplitReach::default();
    for fixture in OPENAI_FIXTURES {
        let bytes = fixture.as_bytes();
        let whole = openai_events_from_pieces(&[bytes]);
        assert!(
            whole.iter().any(|e| !e.content.is_empty()),
            "fixture has content"
        );
        for a in 0..=bytes.len() {
            reach.record(bytes, a);
            assert_eq!(
                openai_events_from_pieces(&split_at_offsets(bytes, &[a])),
                whole,
                "split at {a}"
            );
            for b in a..=bytes.len() {
                assert_eq!(
                    openai_events_from_pieces(&split_at_offsets(bytes, &[a, b])),
                    whole,
                    "split at {a},{b}"
                );
            }
        }
    }
    let bytes = ANTHROPIC_FIXTURE.as_bytes();
    let whole = anthropic_events_from_pieces(&[bytes]);
    assert_eq!(
        whole.iter().map(|e| e.content.as_str()).collect::<String>(),
        "h\u{e9}l\u{1f680}lo"
    );
    for a in 0..=bytes.len() {
        reach.record(bytes, a);
        assert_eq!(
            anthropic_events_from_pieces(&split_at_offsets(bytes, &[a])),
            whole
        );
    }
    eprintln!("single-split reach: {reach:?}");
    assert!(reach.mid_utf8 > 0 && reach.mid_crlf > 0 && reach.mid_data_prefix > 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(256)))]

    /// Arbitrary line content never panics either line parser.
    #[test]
    fn sse_line_parsers_never_panic(line in ".{0,200}") {
        let _ = parse_openai_sse_line(&line);
        let _ = parse_anthropic_sse_line(&line);
        let data_line = format!("data: {line}");
        let _ = parse_openai_sse_line(&data_line);
        let _ = parse_anthropic_sse_line(&data_line);
    }
}

/// An OpenAI-compatible server that fails mid-generation reports it in-band:
/// vLLM's chat streaming handler yields `data: {"error": {...}}` and then
/// `data: [DONE]` (and older releases yield the bare error object). Both shapes
/// below are written from that handler, not captured from a live server.
///
/// Today the error line parses to "no content" and is skipped, so the caller
/// sees a normal, finished stream holding only the text produced before the
/// failure: a truncated answer presented as a complete one. The fix for #516
/// should un-ignore this.
#[test]
#[ignore = "known gap: in-band SSE error payloads are dropped and the stream reports a clean finish (#516)"]
fn in_band_stream_errors_are_not_reported_as_a_clean_finish() {
    for error_line in [
        "data: {\"error\":{\"object\":\"error\",\"message\":\"CUDA out of memory\",\"type\":\"InternalServerError\",\"code\":500}}\n\n",
        "data: {\"object\":\"error\",\"message\":\"CUDA out of memory\",\"type\":\"InternalServerError\",\"code\":500}\n\n",
    ] {
        let body = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"partial\"}}}}]}}\n\n{error_line}data: [DONE]\n\n"
        );
        let mut events = Vec::new();
        let mut on_event = |event: ProviderStreamEvent| {
            events.push(event);
            Ok(())
        };
        let mut emitter = OpenAiSseEmitter::new(&mut on_event);
        let pushed = emitter.push_bytes(body.as_bytes());
        let finished = emitter.finish();
        assert!(
            pushed.is_err() || finished.is_err(),
            "stream with an in-band error finished cleanly with {events:?}"
        );
    }
}
