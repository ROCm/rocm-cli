// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for the signature-verification path that decides whether a
//! downloaded artifact or metadata document is authentic.
//!
//! The generators here deliberately sample the *near-miss* region — a signature
//! with one byte changed, a signature re-encoded with leading zeros, a payload
//! with one bit flipped, a PEM mangled by Windows tooling — because a uniformly
//! random generator essentially never produces an input that reaches the
//! interesting branches of an RSA verifier.

use std::sync::OnceLock;

use proptest::prelude::*;
use rocm_core::{
    generate_rsa_signing_keypair, sign_rsa_pkcs1_sha256_signature,
    verify_rsa_pkcs1_sha256_signature,
};

/// One 2048-bit keypair, generated once: key generation takes seconds in a debug build, and every
/// property below only needs *a* valid key, not a fresh one per case.
fn keypair() -> &'static (String, String) {
    static KEYPAIR: OnceLock<(String, String)> = OnceLock::new();
    KEYPAIR.get_or_init(|| generate_rsa_signing_keypair().expect("keypair generation"))
}

/// A second, unrelated keypair, for "signed by the wrong key" cases.
fn other_keypair() -> &'static (String, String) {
    static KEYPAIR: OnceLock<(String, String)> = OnceLock::new();
    KEYPAIR.get_or_init(|| generate_rsa_signing_keypair().expect("keypair generation"))
}

fn sign(payload: &[u8]) -> Vec<u8> {
    sign_rsa_pkcs1_sha256_signature(&keypair().0, payload).expect("signing")
}

fn verify(payload: &[u8], signature: &[u8]) -> bool {
    verify_rsa_pkcs1_sha256_signature(&keypair().1, payload, signature, "test").is_ok()
}

/// Payloads shaped like the documents actually verified here — arbitrary bytes,
/// near-empty bodies, and printable text such as a metadata JSON blob — capped
/// at 512 bytes so shrinking stays fast.
fn payload_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        3 => prop::collection::vec(any::<u8>(), 0..512),
        1 => prop::collection::vec(any::<u8>(), 0..4),
        1 => "[ -~\\n]{0,400}".prop_map(String::into_bytes),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    /// Baseline: a genuine signature over the exact payload verifies.
    #[test]
    fn genuine_signature_verifies(payload in payload_strategy()) {
        let signature = sign(&payload);
        prop_assert!(verify(&payload, &signature));
    }

    /// Flipping a single bit of the payload must invalidate the signature.
    /// This is the "tampered artifact" case: the attacker edits one byte of a
    /// signed metadata document and keeps the original signature.
    #[test]
    fn single_bit_payload_mutation_is_rejected(
        payload in prop::collection::vec(any::<u8>(), 1..256),
        index in any::<prop::sample::Index>(),
        bit in 0u8..8,
    ) {
        let signature = sign(&payload);
        let mut tampered = payload.clone();
        let position = index.index(payload.len());
        tampered[position] ^= 1 << bit;
        prop_assert!(
            !verify(&tampered, &signature),
            "a payload differing by one bit must not verify"
        );
    }

    /// Truncating or extending the payload must invalidate the signature, even
    /// when the retained prefix is byte-identical.
    #[test]
    fn payload_prefix_or_extension_is_rejected(
        payload in prop::collection::vec(any::<u8>(), 2..256),
        cut in any::<prop::sample::Index>(),
        suffix in prop::collection::vec(any::<u8>(), 1..16),
    ) {
        let signature = sign(&payload);
        let keep = cut.index(payload.len());
        prop_assert!(!verify(&payload[..keep], &signature));

        let mut extended = payload;
        extended.extend_from_slice(&suffix);
        prop_assert!(!verify(&extended, &signature));
    }

    /// Mutating the signature — one byte changed, truncated, padded, or
    /// re-encoded with leading zeros — must be rejected. Leading-zero
    /// re-encoding is the classic PKCS#1 v1.5 pitfall: the integer value is
    /// unchanged, so a verifier that parses to a bignum without length-checking
    /// accepts it.
    #[test]
    fn signature_mutations_are_rejected(
        payload in payload_strategy(),
        index in any::<prop::sample::Index>(),
        delta in 1u8..=255,
        pad in 1usize..4,
    ) {
        let signature = sign(&payload);

        let mut flipped = signature.clone();
        let position = index.index(flipped.len());
        flipped[position] = flipped[position].wrapping_add(delta);
        prop_assert!(!verify(&payload, &flipped), "a mutated signature must not verify");

        prop_assert!(!verify(&payload, &signature[..signature.len() - 1]));
        prop_assert!(!verify(&payload, &signature[1..]));
        prop_assert!(!verify(&payload, &[]));

        let mut zero_padded = vec![0_u8; pad];
        zero_padded.extend_from_slice(&signature);
        prop_assert!(
            !verify(&payload, &zero_padded),
            "a leading-zero re-encoding carries the same integer value and must \
             still be rejected on length"
        );

        let mut trailing = signature.clone();
        trailing.extend(std::iter::repeat_n(0_u8, pad));
        prop_assert!(!verify(&payload, &trailing));
    }

    /// A signature produced by a different key must never verify.
    #[test]
    fn signature_from_another_key_is_rejected(payload in payload_strategy()) {
        let wrong = sign_rsa_pkcs1_sha256_signature(&other_keypair().0, &payload)
            .expect("signing with the second key");
        prop_assert!(!verify(&payload, &wrong));
    }

    /// Arbitrary bytes offered as a signature must be rejected, never accepted
    /// and never panic.
    #[test]
    fn arbitrary_bytes_are_never_a_valid_signature(
        payload in payload_strategy(),
        candidate in prop::collection::vec(any::<u8>(), 0..300),
    ) {
        prop_assert!(!verify(&payload, &candidate));
    }
}

/// PEM mangling that real tooling produces: CRLF, lone CR, a UTF-8 BOM, a stray
/// trailing space, a blank line, a tab. Verification must still succeed for
/// each. The BOM, space, blank-line and tab styles depend on `normalize_pem`;
/// CRLF and lone CR are also accepted by the PEM parser itself, so those two
/// styles pin the outcome rather than `normalize_pem`'s handling of them.
fn mangle_pem(pem: &str, style: u8) -> String {
    match style % 6 {
        0 => pem.replace('\n', "\r\n"),
        1 => pem.replace('\n', "\r"),
        2 => format!("\u{feff}{pem}"),
        3 => pem.replace('\n', " \n"),
        4 => pem.replace('\n', "\n\n"),
        _ => pem.replace('\n', "\t\r\n"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// Tolerated mangling must not change *which* key is trusted.
    #[test]
    fn pem_mangling_preserves_the_verification_verdict(
        payload in payload_strategy(),
        style in any::<u8>(),
    ) {
        let signature = sign(&payload);
        let mangled = mangle_pem(&keypair().1, style);
        prop_assert!(
            verify_rsa_pkcs1_sha256_signature(&mangled, &payload, &signature, "test").is_ok(),
            "whitespace/BOM mangling must not break a genuine signature"
        );

        // The same mangling applied to the *wrong* key must still reject.
        let wrong = mangle_pem(&other_keypair().1, style);
        prop_assert!(
            verify_rsa_pkcs1_sha256_signature(&wrong, &payload, &signature, "test").is_err(),
            "normalization must not turn the wrong key into an accepting one"
        );
    }

    /// Corrupting the base64 body of the public key must never yield a key that
    /// accepts a signature made by the real key.
    #[test]
    fn corrupted_public_key_never_accepts(
        payload in payload_strategy(),
        index in any::<prop::sample::Index>(),
        replacement in prop::sample::select(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
                .chars()
                .collect::<Vec<_>>(),
        ),
    ) {
        let signature = sign(&payload);
        let pem = &keypair().1;
        // Only the base64 body: everything after the header line and before
        // the footer.
        let body_start = pem.find('\n').expect("PEM header line") + 1;
        let body_end = pem.find("-----END").expect("PEM footer");
        let body_positions: Vec<usize> = pem[body_start..body_end]
            .char_indices()
            .filter(|(_, ch)| ch.is_ascii_alphanumeric() || *ch == '+' || *ch == '/')
            .map(|(position, _)| body_start + position)
            .collect();
        let position = body_positions[index.index(body_positions.len())];
        let mut corrupted = pem.clone();
        corrupted.replace_range(position..=position, &replacement.to_string());
        if corrupted == *pem {
            return Ok(());
        }
        prop_assert!(
            verify_rsa_pkcs1_sha256_signature(&corrupted, &payload, &signature, "test").is_err(),
            "a public key with a corrupted body must not accept the real signature"
        );
    }

    /// Garbage offered as a public key must produce an error, never a success
    /// and never a panic.
    #[test]
    fn arbitrary_public_key_text_is_rejected(
        payload in payload_strategy(),
        pem in prop_oneof![
            Just(String::new()),
            "[ -~\\n]{0,200}",
            "-----BEGIN PUBLIC KEY-----\n[A-Za-z0-9+/=\\n]{0,200}\n-----END PUBLIC KEY-----\n",
        ],
    ) {
        let signature = sign(&payload);
        prop_assert!(
            verify_rsa_pkcs1_sha256_signature(&pem, &payload, &signature, "test").is_err()
        );
    }
}
