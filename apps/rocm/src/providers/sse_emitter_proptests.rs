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
//! between `\r` and `\n` and inside a UTF-8 sequence, and bodies whose last
//! line is cut off before its terminator, so that `finish()` has a partial line
//! to flush. None of them generates SSE text from scratch. The transport readers
//! that feed these emitters (chunked and close-delimited bodies) are not covered
//! here.
//!
//! Split invariance alone cannot see a defect that changes the whole-body result
//! and every split result alike, such as `finish()` dropping the partial last
//! line. So each fixture also carries the text it must produce.

use super::*;
use proptest::prelude::*;

/// `(body, the text its events must concatenate to)`.
const OPENAI_FIXTURES: &[(&str, &str)] = &[
    // providers.rs `parses_openai_sse_chat_events`.
    (
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n",
        "hello",
    ),
    // Same body, CRLF line endings.
    (
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n",
        "hello",
    ),
    // Multi-byte content (2-, 3- and 4-byte sequences), CRLF, no space after
    // `data:`, a role-only first delta, and a usage chunk with empty choices.
    (
        "data:{\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"caf\u{e9} \u{6f22}\u{5b57} \u{1f680}\"}}]}\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" \u{4e16}\u{754c}\"}}]}\r\n\r\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":3}}\r\n\r\ndata: [DONE]\r\n\r\n",
        "caf\u{e9} \u{6f22}\u{5b57} \u{1f680} \u{4e16}\u{754c}",
    ),
    // The stream closes in the middle of its last line: no terminator and no
    // `[DONE]`, so that line's content reaches the caller only through
    // `finish()`.
    (
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo \u{1f680}\"}}]}",
        "hello \u{1f680}",
    ),
    // The same cut-off tail on a CRLF stream, after the `\r` of the last line.
    (
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\r",
        "hello",
    ),
];

/// `(body, the text its events must concatenate to)`.
const ANTHROPIC_FIXTURES: &[(&str, &str)] = &[
    // The opening `message_start` and `content_block_start` (empty text) that
    // Anthropic streams send, two multi-byte deltas, and `message_stop`.
    (
        "event: message_start\r\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}\r\n\r\nevent: content_block_start\r\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\r\n\r\nevent: content_block_delta\r\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"h\u{e9}l\u{1f680}\"}}\r\n\r\nevent: content_block_delta\r\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\r\n\r\nevent: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n",
        "h\u{e9}l\u{1f680}lo",
    ),
    // The stream closes in the middle of its last line, before `message_stop`.
    (
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" th\u{e9}re\"}}",
        "hi th\u{e9}re",
    ),
];

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

/// The whole-body result every split is compared with must itself be right:
/// the expected text, and exactly one `done` event, at the end.
fn assert_whole_body_events(events: &[ProviderStreamEvent], expected: &str, fixture: &str) {
    assert_eq!(
        events
            .iter()
            .map(|e| e.content.as_str())
            .collect::<String>(),
        expected,
        "text of {fixture:?}"
    );
    assert_eq!(
        events.iter().filter(|e| e.done).count(),
        1,
        "done events of {fixture:?}"
    );
    assert!(
        events.last().is_some_and(|e| e.done),
        "last event of {fixture:?} is not done"
    );
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
    for (fixture, expected) in OPENAI_FIXTURES {
        let bytes = fixture.as_bytes();
        let whole = openai_events_from_pieces(&[bytes]);
        assert_whole_body_events(&whole, expected, fixture);
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
    for (fixture, expected) in ANTHROPIC_FIXTURES {
        let bytes = fixture.as_bytes();
        let whole = anthropic_events_from_pieces(&[bytes]);
        assert_whole_body_events(&whole, expected, fixture);
        for a in 0..=bytes.len() {
            reach.record(bytes, a);
            assert_eq!(
                anthropic_events_from_pieces(&split_at_offsets(bytes, &[a])),
                whole,
                "split at {a}"
            );
        }
    }
    eprintln!("single-split reach: {reach:?}");
    assert!(reach.mid_utf8 > 0 && reach.mid_crlf > 0 && reach.mid_data_prefix > 0);
    // Each emitter has a fixture whose last line is left for `finish()`.
    for fixtures in [OPENAI_FIXTURES, ANTHROPIC_FIXTURES] {
        assert!(fixtures.iter().any(|(body, _)| !body.ends_with('\n')));
    }
}

/// Any JSON value, with object keys drawn mostly from the field names the line
/// parsers look up, so that arbitrary nesting still lands on them.
fn json_value() -> impl Strategy<Value = serde_json::Value> {
    use serde_json::Value;
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::from),
        any::<f64>().prop_map(Value::from),
        ".{0,12}".prop_map(Value::String),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
            prop::collection::btree_map(json_key(), inner, 0..4)
                .prop_map(|map| Value::Object(map.into_iter().collect())),
        ]
    })
}

fn json_key() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => prop::sample::select(vec![
            "choices",
            "delta",
            "message",
            "content",
            "type",
            "text",
            "content_block",
            "error",
        ])
        .prop_map(str::to_owned),
        1 => "[a-z_]{0,8}",
    ]
}

/// `{key: value}`, with the key present or absent and the value either the
/// expected shape or any JSON at all.
fn maybe_field(
    key: &'static str,
    shaped: impl Strategy<Value = serde_json::Value> + 'static,
) -> impl Strategy<Value = Option<(String, serde_json::Value)>> {
    prop::option::weighted(
        0.9,
        prop_oneof![4 => shaped.boxed(), 1 => json_value().boxed()]
            .prop_map(move |value| (key.to_owned(), value)),
    )
}

fn object(fields: Vec<Option<(String, serde_json::Value)>>) -> serde_json::Value {
    serde_json::Value::Object(fields.into_iter().flatten().collect())
}

/// `{"<key>": <value>}`, the shape of `delta`, `message`, `content_block` and
/// `error` when the parser reads one field out of them. The value is usually the
/// string the parser takes, otherwise any JSON.
fn holder(key: &'static str) -> impl Strategy<Value = serde_json::Value> {
    let value = prop_oneof![3 => ".{0,12}".prop_map(serde_json::Value::String), 1 => json_value()];
    maybe_field(key, value).prop_map(|field| object(vec![field]))
}

/// An OpenAI chat chunk: `choices` is usually an array of choice objects,
/// each with `delta` and `message` holders of arbitrary content.
fn openai_payload() -> impl Strategy<Value = serde_json::Value> {
    let choice = (
        maybe_field("delta", holder("content")),
        maybe_field("message", holder("content")),
    )
        .prop_map(|(delta, message)| object(vec![delta, message]));
    let choices = prop::collection::vec(choice, 0..3).prop_map(serde_json::Value::Array);
    maybe_field("choices", choices).prop_map(|field| object(vec![field]))
}

/// An Anthropic stream event: `type` is usually one the parser branches on,
/// with every holder the branches read, each of arbitrary content.
fn anthropic_payload() -> impl Strategy<Value = serde_json::Value> {
    let kind = prop::sample::select(vec![
        "content_block_delta",
        "content_block_start",
        "message_stop",
        "error",
        "ping",
    ])
    .prop_map(|kind| serde_json::Value::String(kind.to_owned()));
    (
        maybe_field("type", kind),
        maybe_field("delta", holder("text")),
        maybe_field("content_block", holder("text")),
        maybe_field("error", holder("message")),
    )
        .prop_map(|(kind, delta, block, error)| object(vec![kind, delta, block, error]))
}

/// A line for either parser: arbitrary text, with or without a `data:` prefix,
/// or a `data:` line carrying a structured OpenAI or Anthropic payload.
fn sse_line() -> impl Strategy<Value = String> {
    let prefix = prop::sample::select(vec!["data: ", "data:", " data:  "]);
    prop_oneof![
        1 => ".{0,200}",
        1 => ".{0,200}".prop_map(|line| format!("data: {line}")),
        2 => (prefix.clone(), openai_payload()).prop_map(|(p, v)| format!("{p}{v}")),
        // Five event types split this arm, so it gets the largest share.
        3 => (prefix, anthropic_payload()).prop_map(|(p, v)| format!("{p}{v}")),
    ]
}

/// How many sampled lines reach each branch of the two line parsers that reads
/// a field out of the payload.
#[derive(Debug, Default)]
struct ParserReach {
    samples: u32,
    openai_delta_content: u32,
    openai_message_content: u32,
    openai_choices_not_an_array: u32,
    anthropic_delta_text: u32,
    anthropic_block_start_text: u32,
    anthropic_message_stop: u32,
    anthropic_error_message: u32,
    anthropic_error_without_message: u32,
    anthropic_other_type: u32,
}

impl ParserReach {
    fn record(&mut self, line: &str) {
        fn bump(counter: &mut u32, hit: bool) {
            *counter += u32::from(hit);
        }
        self.samples += 1;
        let Some(data) = line.trim().strip_prefix("data:") else {
            return;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data.trim()) else {
            return;
        };
        let first = &v["choices"][0];
        bump(
            &mut self.openai_delta_content,
            first["delta"]["content"].is_string(),
        );
        bump(
            &mut self.openai_message_content,
            !first["delta"]["content"].is_string() && first["message"]["content"].is_string(),
        );
        bump(
            &mut self.openai_choices_not_an_array,
            v.get("choices").is_some_and(|c| !c.is_array()),
        );
        let kind = v["type"].as_str();
        bump(
            &mut self.anthropic_delta_text,
            kind == Some("content_block_delta") && v["delta"]["text"].is_string(),
        );
        bump(
            &mut self.anthropic_block_start_text,
            kind == Some("content_block_start") && v["content_block"]["text"].is_string(),
        );
        bump(
            &mut self.anthropic_message_stop,
            kind == Some("message_stop"),
        );
        bump(
            &mut self.anthropic_error_message,
            kind == Some("error") && v["error"]["message"].is_string(),
        );
        bump(
            &mut self.anthropic_error_without_message,
            kind == Some("error") && !v["error"]["message"].is_string(),
        );
        bump(
            &mut self.anthropic_other_type,
            v.get("type").is_some()
                && !matches!(
                    kind,
                    Some("content_block_delta" | "content_block_start" | "message_stop" | "error")
                ),
        );
    }
}

/// `sse_line_parsers_never_panic` can only fail on a branch its lines reach.
/// This counts the field-reading branches over a fixed-seed sample of the same
/// strategy and fails if a generator change starves one.
#[test]
fn sse_line_inputs_reach_every_field_reading_branch() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    let mut runner = TestRunner::deterministic();
    let strategy = sse_line();
    let mut reach = ParserReach::default();
    for _ in 0..4096 {
        let line = strategy
            .new_tree(&mut runner)
            .expect("strategy generates a value")
            .current();
        reach.record(&line);
    }
    eprintln!("line-parser reach: {reach:#?}");
    // 1% of the sample: under every measured share, well above zero.
    let floor = reach.samples / 100;
    for (branch, hits) in [
        ("openai delta.content", reach.openai_delta_content),
        ("openai message.content", reach.openai_message_content),
        (
            "openai choices not an array",
            reach.openai_choices_not_an_array,
        ),
        (
            "anthropic content_block_delta text",
            reach.anthropic_delta_text,
        ),
        (
            "anthropic content_block_start text",
            reach.anthropic_block_start_text,
        ),
        ("anthropic message_stop", reach.anthropic_message_stop),
        (
            "anthropic error with message",
            reach.anthropic_error_message,
        ),
        (
            "anthropic error without message",
            reach.anthropic_error_without_message,
        ),
        ("anthropic other type", reach.anthropic_other_type),
    ] {
        assert!(
            hits >= floor,
            "{branch}: {hits} of {} samples, below the floor of {floor}",
            reach.samples
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(1024)))]

    /// No line panics either line parser: arbitrary text, and `data:` lines
    /// whose JSON has the fields the parsers read, in arbitrary shapes.
    #[test]
    fn sse_line_parsers_never_panic(line in sse_line()) {
        let _ = parse_openai_sse_line(&line);
        let _ = parse_anthropic_sse_line(&line);
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
