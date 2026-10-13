// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Tests for the SSE line emitters (`OpenAiSseEmitter` and
//! `AnthropicSseEmitter`) and the line parsers under them.
//!
//! - An exhaustive sweep over hand-written fixtures: every one- and two-point
//!   split of each body produces the events (and the error, for a body that
//!   fails) of the whole body, and the whole body produces the exact events
//!   each fixture lists. OpenAI fixture 1 is the body of
//!   `parses_openai_sse_chat_events` in `providers.rs`. The others are written
//!   here, in the shapes those servers send, to reach what that one body does
//!   not: CRLF line endings (what Python SSE servers emit by default),
//!   multi-byte UTF-8 content so that a split can land inside a UTF-8 sequence,
//!   `data:` with no space before content, a role-only delta, a usage chunk with
//!   empty `choices`, a `message.content` chunk, malformed JSON, Anthropic's
//!   `message_start`, `content_block_start` and `error` events, a non-text
//!   delta, a `[DONE]` line in an Anthropic stream, lines after the end of the
//!   stream, and bodies whose last line is cut off before its terminator, so
//!   that `finish()` has a partial line to flush. Split invariance alone cannot
//!   see a defect that changes the whole-body result and every split result
//!   alike, such as `finish()` dropping the partial last line; the listed
//!   events can.
//! - Direct line-parser assertions on `\r`-terminated lines, and a test that an
//!   error from the caller's callback stops either emitter.
//! - A fuzz of both line parsers over generated `data:` lines (`sse_line()`),
//!   which is a panic guard only: it asserts nothing about what a line parses
//!   to. Its companion test checks that the generator keeps reaching every
//!   field-reading branch, so the guard does not go quiet on one.
//! - An ignored reproducer for #516 (in-band OpenAI error payloads).
//!
//! The transport readers the sibling `proptests` module covers (chunked,
//! `Content-Length` and close-delimited bodies) feed only the OpenAI emitter,
//! on the local-provider path. On the remote-provider path, the only one the
//! Anthropic emitter has, both emitters are fed by `stream_json_with_headers`
//! through `SseBytesEmitter`, which example tests in `providers.rs` exercise
//! but no property does.
//!
//! A trailing `\r` is trimmed in three places on the way to a parse: the
//! emitters' `trim_end_matches('\r')`, then each parser's `line.trim()` and
//! `data.trim()`. Each hides the others, so the CRLF fixtures prove only that
//! at least one of them exists, and the direct parser assertions only that one
//! of the parser's own two exists. A split between `\r` and `\n` is not a
//! separate hazard: the emitters break lines on `\n` alone.

use super::*;
use crate::test_support::{proptest_cases, split_at_offsets};
use proptest::prelude::*;

/// A body, the `(content, done)` events it must produce, and the error the
/// stream must then fail with, if it fails.
type Fixture = (
    &'static str,
    &'static [(&'static str, bool)],
    Option<&'static str>,
);

const OPENAI_FIXTURES: &[Fixture] = &[
    // providers.rs `parses_openai_sse_chat_events`.
    (
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n",
        &[("hel", false), ("lo", false), ("", true)],
        None,
    ),
    // Same body, CRLF line endings.
    (
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n",
        &[("hel", false), ("lo", false), ("", true)],
        None,
    ),
    // Multi-byte content (2-, 3- and 4-byte sequences), CRLF, a role-only first
    // delta (its empty content is still an event), content after `data:` with
    // no space, and a usage chunk with empty choices (no event).
    (
        "data:{\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\r\n\r\ndata:{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"caf\u{e9} \u{6f22}\u{5b57} \u{1f680}\"}}]}\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" \u{4e16}\u{754c}\"}}]}\r\n\r\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":3}}\r\n\r\ndata: [DONE]\r\n\r\n",
        &[
            ("", false),
            ("caf\u{e9} \u{6f22}\u{5b57} \u{1f680}", false),
            (" \u{4e16}\u{754c}", false),
            ("", true),
        ],
        None,
    ),
    // A `message.content` chunk (the non-delta shape some servers send), then
    // lines after `[DONE]`: neither a late chunk nor a second `[DONE]` may
    // reach the caller.
    (
        "data: {\"choices\":[{\"message\":{\"content\":\"wh\u{f6}le\"}}]}\n\ndata: [DONE]\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\ndata: [DONE]\n\n",
        &[("wh\u{f6}le", false), ("", true)],
        None,
    ),
    // The stream closes in the middle of its last line: no terminator and no
    // `[DONE]`, so that line's content reaches the caller only through
    // `finish()`.
    (
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo \u{1f680}\"}}]}",
        &[("hel", false), ("lo \u{1f680}", false), ("", true)],
        None,
    ),
    // The same cut-off tail on a CRLF stream, after the `\r` of the last line.
    (
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\r",
        &[("hel", false), ("lo", false), ("", true)],
        None,
    ),
    // A line that is not JSON fails the stream, after the text before it has
    // reached the caller; nothing after it is read.
    (
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: {\"choices\":[\n\ndata: [DONE]\n\n",
        &[("ok", false)],
        Some("failed to parse local provider SSE data"),
    ),
];

const ANTHROPIC_FIXTURES: &[Fixture] = &[
    // The opening `message_start` and `content_block_start` (empty text, which
    // is no event) that Anthropic streams send, two multi-byte deltas, and
    // `message_stop`.
    (
        "event: message_start\r\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}\r\n\r\nevent: content_block_start\r\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\r\n\r\nevent: content_block_delta\r\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"h\u{e9}l\u{1f680}\"}}\r\n\r\nevent: content_block_delta\r\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\r\n\r\nevent: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n",
        &[("h\u{e9}l\u{1f680}", false), ("lo", false), ("", true)],
        None,
    ),
    // A `content_block_start` that carries text, after `data:` with no space
    // (Anthropic sends it empty, but the parser reads it when it is not), then
    // lines after `message_stop`: neither a late delta nor a late `error` may
    // reach the caller or fail the finished stream.
    (
        "event: content_block_start\ndata:{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"H\u{e9}\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"y\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"late\"}}\n\nevent: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"late\"}}\n\n",
        &[("H\u{e9}", false), ("y", false), ("", true)],
        None,
    ),
    // The stream closes in the middle of its last line, before `message_stop`.
    (
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" th\u{e9}re\"}}",
        &[("hi", false), (" th\u{e9}re", false), ("", true)],
        None,
    ),
    // A non-text delta (tool input), which is no event, then a `[DONE]` line,
    // which Anthropic does not send but the parser takes as the end of the
    // stream: the late delta after it must not reach the caller.
    (
        "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"ok\"}}\n\ndata: [DONE]\n\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"late\"}}\n\n",
        &[("ok", false), ("", true)],
        None,
    ),
    // An `error` event fails the stream with the server's message, after the
    // text before it has reached the caller. `sse_line_parsers_never_panic`
    // reaches this branch but can only fail on a panic, not on a wrong result.
    (
        "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"par\"}}\n\nevent: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"Overloaded\"}}\n\n",
        &[("par", false)],
        Some("Anthropic provider stream failed: Overloaded"),
    ),
    // An `error` event with no message: a placeholder stands in for it.
    (
        "data: {\"type\":\"error\",\"error\":{}}\n\n",
        &[],
        Some("Anthropic provider stream failed: unknown Anthropic stream error"),
    ),
];

/// What a stream produced: the events the caller saw, and the error it failed
/// with, if it failed.
type Outcome = (Vec<ProviderStreamEvent>, Option<String>);

/// Run `stream` with a callback that records every event.
fn outcome(
    stream: impl FnOnce(&mut dyn FnMut(ProviderStreamEvent) -> Result<()>) -> Result<()>,
) -> Outcome {
    let mut events = Vec::new();
    let mut on_event = |event: ProviderStreamEvent| {
        events.push(event);
        Ok(())
    };
    let error = stream(&mut on_event).err().map(|error| error.to_string());
    (events, error)
}

fn openai_outcome(pieces: &[&[u8]]) -> Outcome {
    outcome(|on_event| {
        let mut emitter = OpenAiSseEmitter::new(on_event);
        for piece in pieces {
            emitter.push_bytes(piece)?;
        }
        emitter.finish()
    })
}

fn anthropic_outcome(pieces: &[&[u8]]) -> Outcome {
    outcome(|on_event| {
        let mut emitter = AnthropicSseEmitter::new(on_event);
        for piece in pieces {
            emitter.push_bytes(piece)?;
        }
        emitter.finish()
    })
}

/// How many single split points landed in each place a line emitter is most
/// likely to get wrong.
#[derive(Default, Debug)]
struct SplitReach {
    mid_utf8: usize,
    mid_data_prefix: usize,
}

impl SplitReach {
    fn record(&mut self, bytes: &[u8], cut: usize) {
        if cut > 0 && cut < bytes.len() {
            if bytes[cut] & 0b1100_0000 == 0b1000_0000 {
                self.mid_utf8 += 1;
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

/// Check the whole body of each fixture against its expected outcome, then
/// every one- and two-point split of it against the whole body.
fn sweep_splits(fixtures: &[Fixture], outcome_of: fn(&[&[u8]]) -> Outcome, reach: &mut SplitReach) {
    for (fixture, events, error) in fixtures {
        let bytes = fixture.as_bytes();
        let whole = outcome_of(&[bytes]);
        let expected: Outcome = (
            events
                .iter()
                .map(|&(content, done)| ProviderStreamEvent {
                    content: content.to_owned(),
                    done,
                })
                .collect(),
            error.map(str::to_owned),
        );
        assert_eq!(whole, expected, "outcome of {fixture:?}");
        for a in 0..=bytes.len() {
            reach.record(bytes, a);
            assert_eq!(
                outcome_of(&split_at_offsets(bytes, &[a])),
                whole,
                "split at {a} of {fixture:?}"
            );
            for b in a..=bytes.len() {
                assert_eq!(
                    outcome_of(&split_at_offsets(bytes, &[a, b])),
                    whole,
                    "split at {a},{b} of {fixture:?}"
                );
            }
        }
    }
}

/// Every single split point, and every pair of split points, of every fixture
/// of both emitters: exhaustive, so mid-UTF-8 and mid-`data:` splits are
/// guaranteed to be reached rather than hoped for.
#[test]
fn sse_emitters_are_invariant_under_every_one_and_two_point_split() {
    let mut reach = SplitReach::default();
    sweep_splits(OPENAI_FIXTURES, openai_outcome, &mut reach);
    sweep_splits(ANTHROPIC_FIXTURES, anthropic_outcome, &mut reach);
    eprintln!("single-split reach: {reach:?}");
    assert!(reach.mid_utf8 > 0 && reach.mid_data_prefix > 0);
    // Each emitter has a fixture whose last line is left for `finish()`.
    for fixtures in [OPENAI_FIXTURES, ANTHROPIC_FIXTURES] {
        assert!(fixtures.iter().any(|(body, ..)| !body.ends_with('\n')));
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
    openai_choices_empty: u32,
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
        bump(
            &mut self.openai_choices_empty,
            v["choices"].as_array().is_some_and(Vec::is_empty),
        );
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
            // Production drops an empty start text, so only non-empty counts.
            kind == Some("content_block_start")
                && v["content_block"]["text"]
                    .as_str()
                    .is_some_and(|text| !text.is_empty()),
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
        ("openai empty choices", reach.openai_choices_empty),
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
            "anthropic content_block_start non-empty text",
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
    #![proptest_config(ProptestConfig::with_cases(proptest_cases(1024)))]

    /// No line panics either line parser: arbitrary text, and `data:` lines
    /// whose JSON has the fields the parsers read, in arbitrary shapes.
    #[test]
    fn sse_line_parsers_never_panic(line in sse_line()) {
        let _ = parse_openai_sse_line(&line);
        let _ = parse_anthropic_sse_line(&line);
    }
}

/// Each parser accepts a line that still ends in `\r`, whatever the emitter in
/// front of it does. For a `[DONE]` line this reaches the parser's own trims
/// (`line.trim()` and `data.trim()`, either of which is enough); a JSON payload
/// would not, because `serde_json` reads a trailing `\r` as whitespace.
#[test]
fn sse_line_parsers_accept_a_cr_terminated_line() {
    let done = Some(ProviderStreamEvent {
        content: String::new(),
        done: true,
    });
    for line in ["data: [DONE]\r", "data:[DONE]\r"] {
        assert_eq!(
            parse_openai_sse_line(line).expect("parses"),
            done,
            "{line:?}"
        );
        assert_eq!(
            parse_anthropic_sse_line(line).expect("parses"),
            done,
            "{line:?}"
        );
    }
    assert_eq!(
        parse_anthropic_sse_line("data: {\"type\":\"message_stop\"}\r").expect("parses"),
        done
    );
}

/// An error the caller's callback returns stops the stream and is the error
/// the emitter returns: from a pushed line, and from the end-of-stream event
/// `finish()` sends when the body had none.
#[test]
fn a_callback_error_stops_either_emitter() {
    fn refuse(event: ProviderStreamEvent) -> Result<()> {
        bail!("caller refused {:?}", event.content)
    }
    let mut on_event = refuse;

    let mut emitter = OpenAiSseEmitter::new(&mut on_event);
    let error = emitter
        .push_bytes(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n")
        .expect_err("a callback error on a line propagates");
    assert_eq!(error.to_string(), "caller refused \"hi\"");
    let mut emitter = AnthropicSseEmitter::new(&mut on_event);
    let error = emitter
        .push_bytes(b"data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n")
        .expect_err("a callback error on a line propagates");
    assert_eq!(error.to_string(), "caller refused \"hi\"");

    let error = OpenAiSseEmitter::new(&mut on_event)
        .finish()
        .expect_err("a callback error on the end event propagates");
    assert_eq!(error.to_string(), "caller refused \"\"");
    let error = AnthropicSseEmitter::new(&mut on_event)
        .finish()
        .expect_err("a callback error on the end event propagates");
    assert_eq!(error.to_string(), "caller refused \"\"");
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
        let (events, error) = openai_outcome(&[body.as_bytes()]);
        assert!(
            error.is_some(),
            "stream with an in-band error finished cleanly with {events:?}"
        );
        // The text produced before the failure still reaches the caller.
        assert_eq!(
            events
                .iter()
                .map(|e| e.content.as_str())
                .collect::<String>(),
            "partial"
        );
    }
}
