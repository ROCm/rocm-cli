// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for the local provider's SSE transport readers.
//!
//! Each property re-frames one of the SSE bodies the unit tests in
//! `providers.rs` already use (plus the same shape with CRLF line endings and
//! multi-byte UTF-8 content): it re-chunks it, truncates it, or interrupts its
//! reads. None of them generates SSE text from scratch.

use super::*;
use proptest::prelude::*;
use std::io::{BufReader, Cursor, ErrorKind, Read};

const OPENAI_FIXTURES: &[&str] = &[
    // providers.rs `parses_openai_sse_chat_events`.
    "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n",
    // Same body, CRLF line endings.
    "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n",
    // Multi-byte content (2-, 3- and 4-byte sequences), CRLF, no space after
    // `data:`, a role-only first delta, and a usage chunk with empty choices.
    "data:{\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"caf\u{e9} \u{6f22}\u{5b57} \u{1f680}\"}}]}\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" \u{4e16}\u{754c}\"}}]}\r\n\r\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":3}}\r\n\r\ndata: [DONE]\r\n\r\n",
];

/// The events the emitter produces for `body` delivered in one piece.
fn events_for_whole_body(body: &[u8]) -> Vec<ProviderStreamEvent> {
    let mut events = Vec::new();
    let mut on_event = |event: ProviderStreamEvent| {
        events.push(event);
        Ok(())
    };
    let mut emitter = OpenAiSseEmitter::new(&mut on_event);
    emitter.push_bytes(body).expect("fixture lines parse");
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

/// Encode `body` with HTTP/1.1 chunked transfer coding, cutting chunks at
/// `offsets`, including the terminating zero chunk. Also returns the wire
/// offset of every chunk boundary (where a chunk-size line starts).
fn chunk_encode(body: &[u8], offsets: &[usize]) -> (Vec<u8>, Vec<usize>) {
    let mut out = Vec::new();
    let mut boundaries = Vec::new();
    for piece in split_at_offsets(body, offsets) {
        if piece.is_empty() {
            continue;
        }
        boundaries.push(out.len());
        out.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
        out.extend_from_slice(piece);
        out.extend_from_slice(b"\r\n");
    }
    boundaries.push(out.len());
    out.extend_from_slice(b"0\r\n\r\n");
    (out, boundaries)
}

/// A reader over fixed bytes that panics once it has been asked to read past
/// end-of-stream an absurd number of times: a caller in that state is spinning,
/// not reading. A real socket at EOF behaves the same way (each `read` returns
/// `Ok(0)` immediately), so this is how a busy loop shows up — as a fast test
/// failure instead of a hung test run.
struct EofSpinGuard {
    inner: Cursor<Vec<u8>>,
    eof_reads: usize,
}

impl EofSpinGuard {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            inner: Cursor::new(bytes),
            eof_reads: 0,
        }
    }
}

impl Read for EofSpinGuard {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buf)?;
        if read == 0 && !buf.is_empty() {
            self.eof_reads += 1;
            assert!(
                self.eof_reads < 10_000,
                "reader spun on end-of-stream {} times without returning",
                self.eof_reads
            );
        }
        Ok(read)
    }
}

/// A reader that fails every `every`-th read with `ErrorKind::Interrupted`
/// (what a socket read with a receive timeout returns when a signal handler
/// runs, since Linux does not restart it) and returns at most `max_read` bytes
/// from the others.
struct InterruptingReader {
    inner: Cursor<Vec<u8>>,
    every: usize,
    calls: usize,
    max_read: usize,
}

impl Read for InterruptingReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.calls += 1;
        if self.calls.is_multiple_of(self.every) {
            return Err(std::io::Error::from(ErrorKind::Interrupted));
        }
        let len = buf.len().min(self.max_read);
        self.inner.read(&mut buf[..len])
    }
}

/// A reader whose first `read` fails with `ErrorKind::Interrupted`.
struct FirstCallInterrupted(Cursor<Vec<u8>>, bool);

impl Read for FirstCallInterrupted {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if !self.1 {
            self.1 = true;
            return Err(std::io::Error::from(ErrorKind::Interrupted));
        }
        self.0.read(buf)
    }
}

fn chunked_events<R: BufRead>(reader: &mut R) -> Result<Vec<ProviderStreamEvent>> {
    let mut events = Vec::new();
    let mut on_event = |event: ProviderStreamEvent| {
        events.push(event);
        Ok(())
    };
    let mut emitter = OpenAiSseEmitter::new(&mut on_event);
    read_chunked_sse_body(reader, &mut emitter)?;
    emitter.finish()?;
    Ok(events)
}

fn close_delimited_events<R: Read>(reader: &mut R) -> Result<Vec<ProviderStreamEvent>> {
    let mut events = Vec::new();
    let mut on_event = |event: ProviderStreamEvent| {
        events.push(event);
        Ok(())
    };
    let mut emitter = OpenAiSseEmitter::new(&mut on_event);
    read_close_delimited_sse_body(reader, &mut emitter)?;
    emitter.finish()?;
    Ok(events)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Chunked transfer coding is transparent: any chunk layout yields the same
    /// events as the plain body.
    #[test]
    fn chunked_reader_is_chunk_layout_invariant(
        fixture in 0..OPENAI_FIXTURES.len(),
        offsets in prop::collection::vec(any::<usize>(), 0..32),
    ) {
        let body = OPENAI_FIXTURES[fixture].as_bytes();
        let (wire, _) = chunk_encode(body, &offsets);
        let mut reader = BufReader::new(EofSpinGuard::new(wire));
        prop_assert_eq!(chunked_events(&mut reader).unwrap(), events_for_whole_body(body));
    }

    /// The engine dying mid-response (connection closed before the zero chunk)
    /// ends the read with an error that says so — never a spin, and never a
    /// clean finish that would present a partial answer as a complete one —
    /// wherever the cut lands: between chunks, or inside a size line, a
    /// chunk's data, or its CRLF.
    #[test]
    fn chunked_reader_fails_on_a_truncated_stream(
        fixture in 0..OPENAI_FIXTURES.len(),
        offsets in prop::collection::vec(any::<usize>(), 0..32),
        cut in any::<usize>(),
    ) {
        let body = OPENAI_FIXTURES[fixture].as_bytes();
        let (wire, _) = chunk_encode(body, &offsets);
        // Cut anywhere before the `0` of the terminating chunk-size line.
        let cut = cut % (wire.len() - "0\r\n\r\n".len() + 1);
        let mut reader = BufReader::new(EofSpinGuard::new(wire[..cut].to_vec()));
        let result = chunked_events(&mut reader);
        prop_assert!(result.is_err(), "truncated at {} of {} finished cleanly", cut, wire.len());
        prop_assert_eq!(result.unwrap_err().to_string(), LOCAL_STREAM_TRUNCATED);
    }

    /// A truncation exactly on a chunk boundary is the shape a crashed engine
    /// produces (it wrote whole chunks, then the socket closed), and the error
    /// says so rather than blaming the framing.
    #[test]
    fn chunked_reader_names_a_close_on_a_chunk_boundary(
        fixture in 0..OPENAI_FIXTURES.len(),
        offsets in prop::collection::vec(any::<usize>(), 0..32),
        boundary in any::<usize>(),
    ) {
        let body = OPENAI_FIXTURES[fixture].as_bytes();
        let (wire, boundaries) = chunk_encode(body, &offsets);
        let cut = boundaries[boundary % boundaries.len()];
        let mut reader = BufReader::new(EofSpinGuard::new(wire[..cut].to_vec()));
        let error = chunked_events(&mut reader).unwrap_err();
        prop_assert_eq!(error.to_string(), LOCAL_STREAM_TRUNCATED);
    }

    /// A transient `Interrupted` read is retried, not surfaced as a failed
    /// stream, for both body framings.
    ///
    /// The close-delimited half covers this module's own retry. The chunked
    /// half relies on the retry inside std's `read_line`/`read_exact`, so it
    /// cannot fail if this module's retry is removed; it pins that the chunked
    /// reader keeps using those rather than a bare `read`.
    #[test]
    fn stream_readers_retry_interrupted_reads(
        fixture in 0..OPENAI_FIXTURES.len(),
        every in 2usize..8,
        max_read in 1usize..64,
        offsets in prop::collection::vec(any::<usize>(), 0..16),
    ) {
        let body = OPENAI_FIXTURES[fixture].as_bytes();
        let whole = events_for_whole_body(body);

        let mut chunked = BufReader::new(InterruptingReader {
            inner: Cursor::new(chunk_encode(body, &offsets).0),
            every,
            calls: 0,
            max_read,
        });
        prop_assert_eq!(&chunked_events(&mut chunked).unwrap(), &whole);

        let mut close_delimited = InterruptingReader {
            inner: Cursor::new(body.to_vec()),
            every,
            calls: 0,
            max_read,
        };
        let result = close_delimited_events(&mut close_delimited);
        prop_assert!(result.is_ok(), "close-delimited read failed: {:#}", result.unwrap_err());
        prop_assert_eq!(result.unwrap(), whole);
    }
}

/// Shrunk reproducer: a chunked body that ends at a chunk boundary. Zero bytes
/// is the minimal case (the engine closed the connection right after sending
/// the response headers); one complete chunk and then EOF is the realistic one.
#[test]
fn chunked_reader_returns_an_error_at_eof_on_a_chunk_boundary() {
    for wire in [
        Vec::new(),
        b"33\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"chunk\"}}]}\n\n\r\n".to_vec(),
    ] {
        let mut reader = BufReader::new(EofSpinGuard::new(wire));
        let error = chunked_events(&mut reader).expect_err("a truncated stream is not a success");
        assert_eq!(error.to_string(), LOCAL_STREAM_TRUNCATED);
    }
}

/// Shrunk reproducer: one `Interrupted` before any data must be retried, not
/// fail the whole close-delimited stream.
#[test]
fn close_delimited_reader_retries_one_interrupted_read() {
    let body = OPENAI_FIXTURES[0].as_bytes();
    let mut reader = FirstCallInterrupted(Cursor::new(body.to_vec()), false);
    let events = close_delimited_events(&mut reader).expect("Interrupted must be retried");
    assert_eq!(events, events_for_whole_body(body));
}
