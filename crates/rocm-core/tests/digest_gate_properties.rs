// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for the digest gate in `download_file_streaming`, the single
//! place a downloaded artifact's bytes are authenticated.
//!
//! The generator is built around *near misses*, because that is the only region
//! where this gate can plausibly fail open: a uniformly random 64-character
//! string differs from the real digest in ~every position and is rejected by
//! any implementation, correct or not. What distinguishes a correct comparison
//! from a prefix comparison, a truncating comparison, or a trimming one is a
//! candidate that shares most of the real digest.

#[path = "common/loopback.rs"]
mod loopback;
#[path = "common/proptest_cases.rs"]
mod proptest_cases;

use std::time::Duration;

use loopback::LoopbackServer;
use proptest::prelude::*;
use proptest_cases::proptest_cases;
use rocm_core::{DOWNLOAD_CHUNK_BYTES, DownloadRequest, download_file_streaming};
use sha2::{Digest, Sha256};

/// A fresh scratch directory, removed when the returned guard drops — including
/// when an assertion fails, so a failing run leaves nothing in the temp dir.
fn scratch_dir(tag: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("rocm-core-digest-gate-{tag}-"))
        .tempdir()
        .expect("scratch dir")
}

/// Run one download against a loopback server, returning whether the gate
/// accepted the body.
fn download_accepts(body: &[u8], expected_sha256: Option<&str>, tag: &str) -> bool {
    download_outcome(body, expected_sha256, tag).0.is_ok()
}

/// Run one download against a loopback server, returning the download's
/// result and every file it left in its otherwise empty scratch directory —
/// the destination and any partial file, whatever it is named.
///
/// Panics if the request never reached the server: a download that failed
/// before the body arrived proves nothing about the digest gate.
fn download_outcome(
    body: &[u8],
    expected_sha256: Option<&str>,
    tag: &str,
) -> (Result<(), String>, Vec<String>) {
    let scratch = scratch_dir(tag);
    let dir = scratch.path();
    let destination = dir.join("artifact.bin");
    let server = LoopbackServer::start(None, body.to_vec());
    let url = format!("{}/artifact.bin", server.url());
    let request = DownloadRequest {
        expected_sha256,
        ..DownloadRequest::new(&url, &destination, Duration::from_secs(10))
    };
    let result = download_file_streaming(&request)
        .map(drop)
        .map_err(|error| format!("{error:#}"));
    assert!(
        server.requests_served() > 0,
        "the download never reached the test server: {result:?}"
    );
    drop(server);
    let mut left_behind: Vec<String> = std::fs::read_dir(dir)
        .expect("read scratch dir")
        .map(|entry| {
            entry
                .expect("scratch entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    left_behind.sort();
    (result, left_behind)
}

/// Whether `result` is the digest gate's rejection, rather than success or
/// some other failure.
fn is_digest_rejection(result: &Result<(), String>) -> bool {
    result
        .as_ref()
        .is_err_and(|error| error.contains("SHA-256 mismatch"))
}

fn digest_of(body: &[u8]) -> String {
    format!("{:x}", Sha256::digest(body))
}

/// How to perturb the true digest. Each variant lands in the region where a
/// sloppy comparison would wrongly accept.
#[derive(Debug, Clone, Copy)]
enum NearMiss {
    /// One hex nibble changed — the smallest possible difference.
    OneNibble,
    /// Correct prefix, wrong suffix: accepted by a prefix comparison.
    WrongSuffix,
    /// Correct prefix only, truncated: accepted by a `starts_with` comparison.
    TruncatedPrefix,
    /// Correct digest with a trailing newline, as a `.sha256` sidecar carries it.
    TrailingNewline,
    /// Correct digest with surrounding spaces.
    Padded,
    /// Correct digest with a 65th character appended.
    Overlong,
    /// Correct digest with a leading zero prepended (shifts every position).
    LeadingZero,
    /// Empty — the "no expectation" string, which must not read as a match.
    Empty,
    /// All-zero digest of the right shape.
    AllZeroes,
}

const ALL_NEAR_MISSES: [NearMiss; 9] = [
    NearMiss::OneNibble,
    NearMiss::WrongSuffix,
    NearMiss::TruncatedPrefix,
    NearMiss::TrailingNewline,
    NearMiss::Padded,
    NearMiss::Overlong,
    NearMiss::LeadingZero,
    NearMiss::Empty,
    NearMiss::AllZeroes,
];

fn apply_near_miss(digest: &str, miss: NearMiss, position: usize, keep: usize) -> String {
    let position = position % digest.len();
    let keep = 1 + keep % (digest.len() - 1);
    match miss {
        NearMiss::OneNibble => {
            let mut out: Vec<char> = digest.chars().collect();
            out[position] = if out[position] == '0' { '1' } else { '0' };
            out.into_iter().collect()
        }
        NearMiss::WrongSuffix => {
            let mut out = digest[..keep].to_owned();
            out.push_str(&"f".repeat(digest.len() - keep));
            if out == digest {
                out.replace_range(keep..=keep, "0");
            }
            out
        }
        NearMiss::TruncatedPrefix => digest[..keep].to_owned(),
        NearMiss::TrailingNewline => format!("{digest}\n"),
        NearMiss::Padded => format!("  {digest}  "),
        NearMiss::Overlong => format!("{digest}a"),
        NearMiss::LeadingZero => format!("0{digest}"),
        NearMiss::Empty => String::new(),
        NearMiss::AllZeroes => "0".repeat(64),
    }
}

/// The positional near misses — one nibble changed, a correct prefix with a
/// wrong tail, a truncated prefix — at every position and every prefix length,
/// rather than at positions drawn at random: a regression that misses only one
/// position (a comparison that skips the last nibble, say) is caught on every
/// run, not on the runs that happen to draw it.
#[test]
fn positional_near_misses_are_rejected_at_every_position() {
    let body = b"artifact body for the positional near misses".to_vec();
    let truth = digest_of(&body);
    let mut not_rejected = Vec::new();
    for miss in [
        NearMiss::OneNibble,
        NearMiss::WrongSuffix,
        NearMiss::TruncatedPrefix,
    ] {
        for at in 0..truth.len() {
            // `apply_near_miss` reads `position` for a changed nibble and
            // `keep` (as `1 + keep % 63`) for a prefix length; `at` walks both.
            let candidate = apply_near_miss(&truth, miss, at, at);
            assert!(
                !candidate.eq_ignore_ascii_case(&truth),
                "near-miss kind {miss:?} at {at} reproduced the true digest"
            );
            let (result, _) = download_outcome(&body, Some(&candidate), "positional");
            if !is_digest_rejection(&result) {
                not_rejected.push(format!("{miss:?} at {at}: {candidate:?} gave {result:?}"));
            }
        }
    }
    assert!(
        not_rejected.is_empty(),
        "the digest gate did not reject these near misses of {truth:?}:\n{}",
        not_rejected.join("\n")
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: proptest_cases(24),
        ..ProptestConfig::default()
    })]

    /// A near-miss digest must never be accepted, whatever the body. The one
    /// deliberate exception is pure ASCII case folding, which the gate
    /// implements as equivalent and which is covered separately below.
    ///
    /// Every kind of near miss is tried for every generated case, rather than
    /// one kind drawn per case: a regression that admits just one kind would
    /// otherwise go unnoticed in any run that happened not to draw it. The
    /// positional kinds are also walked exhaustively by
    /// [`positional_near_misses_are_rejected_at_every_position`].
    #[test]
    fn near_miss_digests_are_rejected(
        body in prop::collection::vec(any::<u8>(), 1..512),
        position in any::<usize>(),
        keep in any::<usize>(),
    ) {
        let truth = digest_of(&body);
        for miss in ALL_NEAR_MISSES {
            let candidate = apply_near_miss(&truth, miss, position, keep);
            prop_assert!(
                !candidate.eq_ignore_ascii_case(&truth),
                "near-miss kind {miss:?} reproduced the true digest"
            );
            let (result, _) = download_outcome(&body, Some(&candidate), "near-miss");
            prop_assert!(
                is_digest_rejection(&result),
                "the digest gate did not reject {candidate:?} for an artifact whose digest is \
                 {truth:?} (near-miss kind {miss:?}): {result:?}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: proptest_cases(12),
        ..ProptestConfig::default()
    })]

    /// The true digest is accepted, in either ASCII case, and in mixed case —
    /// case-insensitivity must hold in both directions, not just for a fully
    /// lowercase expectation.
    #[test]
    fn the_true_digest_is_accepted_in_any_ascii_case(
        body in prop::collection::vec(any::<u8>(), 1..512),
        mask in prop::collection::vec(any::<bool>(), 64),
    ) {
        let truth = digest_of(&body);
        let mixed: String = truth
            .chars()
            .zip(mask)
            .map(|(ch, upper)| if upper { ch.to_ascii_uppercase() } else { ch })
            .collect();
        prop_assert!(download_accepts(&body, Some(&mixed), "mixed-case"));
    }

    /// Mutating the *body* while keeping the expected digest must be rejected,
    /// and nothing — no destination, no partial file — may be left behind for a
    /// later run to treat as cached. This checks the end state: the download
    /// cleans up on more than one path, and either one suffices here.
    #[test]
    fn a_tampered_body_is_rejected(
        body in prop::collection::vec(any::<u8>(), 2..512),
        index in any::<prop::sample::Index>(),
        bit in 0u8..8,
    ) {
        let truth = digest_of(&body);
        let mut tampered = body;
        let position = index.index(tampered.len());
        tampered[position] ^= 1 << bit;
        let (result, left_behind) = download_outcome(&tampered, Some(&truth), "tampered");
        prop_assert!(
            is_digest_rejection(&result),
            "a body differing by one bit was not rejected against the original digest: {result:?}"
        );
        prop_assert!(
            left_behind.is_empty(),
            "a rejected download left {left_behind:?} behind"
        );
    }
}

/// A body several streaming chunks long, tampered only in its last byte, is
/// rejected and leaves nothing behind, while the intact body is accepted: the
/// digest must cover every byte, not just the first buffer. (A hash over only a
/// prefix is caught by the intact half; the tampered half catches a gate that
/// is skipped or disabled for a body this large.)
#[test]
fn tampering_past_the_first_chunk_is_rejected() {
    // Three chunks plus a byte, so the body crosses a chunk boundary
    // regardless of how short the socket reads come back.
    let body: Vec<u8> = (0..=255_u8)
        .cycle()
        .take(3 * DOWNLOAD_CHUNK_BYTES + 1)
        .collect();
    let truth = digest_of(&body);
    assert!(
        download_accepts(&body, Some(&truth), "multi-chunk-intact"),
        "the untampered multi-chunk body must be accepted"
    );
    let mut tampered = body;
    *tampered.last_mut().expect("non-empty body") ^= 1;
    let (result, left_behind) = download_outcome(&tampered, Some(&truth), "multi-chunk");
    assert!(
        is_digest_rejection(&result),
        "a change in the last byte was not detected: {result:?}"
    );
    assert!(
        left_behind.is_empty(),
        "a rejected download left {left_behind:?} behind"
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: proptest_cases(8),
        ..ProptestConfig::default()
    })]

    /// Without an expected digest the download succeeds — this is the gate's
    /// documented fail-open mode, pinned here so it cannot be mistaken for
    /// verification. `DownloadOutcome::sha256` is reported but nothing compares
    /// it, so a caller that omits `expected_sha256` has authenticated nothing.
    #[test]
    fn an_absent_expectation_authenticates_nothing(
        body in prop::collection::vec(any::<u8>(), 1..256),
    ) {
        prop_assert!(
            download_accepts(&body, None, "no-expectation"),
            "callers rely on a digest-less download succeeding; it just proves nothing"
        );
    }
}
