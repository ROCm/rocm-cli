// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! RSASSA-PKCS#1 v1.5 over SHA-256 sign/verify, with no external `openssl` process.
//!
//! Generic — not model-catalog-specific; the model-recipe-index signed-catalog
//! verification built on top of these primitives lives elsewhere in the crate.

use anyhow::{Context, Result};

/// Normalize a PEM document the way the OpenSSL CLI tolerated input, so keys
/// produced or copied through other tooling still parse with the strict RFC 7468
/// reader. Strips a leading UTF-8 BOM, accepts any line-ending style (CRLF, lone
/// CR, or LF), and drops trailing whitespace from each line — Windows tooling
/// (e.g. PowerShell `Set-Content`) can introduce CRLF or a stray trailing space
/// on the `-----BEGIN ...-----` boundary that the parser would otherwise reject.
fn normalize_pem(pem: &str) -> String {
    let without_bom = pem.strip_prefix('\u{feff}').unwrap_or(pem);
    let unified = without_bom.replace("\r\n", "\n").replace('\r', "\n");
    let mut normalized: String = unified
        .split('\n')
        .map(|line| line.trim_end_matches([' ', '\t']))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    normalized.push('\n');
    normalized
}

/// Verify an RSASSA-PKCS#1 v1.5 signature over SHA-256 using a pure-Rust
/// implementation, with no external `openssl` process.
///
/// `public_key_pem` is a SubjectPublicKeyInfo PEM (`-----BEGIN PUBLIC KEY-----`),
/// exactly what `openssl rsa -pubout` emits and what `openssl dgst -sha256 -verify`
/// consumes, so verification is byte-compatible with that command. `label` names the
/// artifact being checked (e.g. `"metadata"`); on a bad signature the error reads
/// `"<label> signature verification failed"` to preserve existing diagnostics.
pub fn verify_rsa_pkcs1_sha256_signature(
    public_key_pem: &str,
    payload: &[u8],
    signature: &[u8],
    label: &str,
) -> Result<()> {
    use rsa::RsaPublicKey;
    use rsa::pkcs1v15::{Signature, VerifyingKey};
    use rsa::pkcs8::DecodePublicKey;
    use rsa::signature::Verifier;
    use sha2::Sha256;

    let public_key = RsaPublicKey::from_public_key_pem(&normalize_pem(public_key_pem))
        .with_context(|| format!("{label} public key is not a valid RSA public key"))?;
    let signature = Signature::try_from(signature)
        .with_context(|| format!("{label} signature is malformed"))?;
    VerifyingKey::<Sha256>::new(public_key)
        .verify(payload, &signature)
        .map_err(|error| anyhow::anyhow!("{label} signature verification failed: {error}"))
}

/// Produce an RSASSA-PKCS#1 v1.5 signature over SHA-256 with a pure-Rust
/// implementation, with no external `openssl` process.
///
/// `private_key_pem` is a PKCS#8 private-key PEM (`-----BEGIN PRIVATE KEY-----`),
/// as emitted by `openssl genpkey`. The signature is deterministic and
/// byte-identical to `openssl dgst -sha256 -sign`, so artifacts signed here verify
/// with either implementation.
pub fn sign_rsa_pkcs1_sha256_signature(private_key_pem: &str, payload: &[u8]) -> Result<Vec<u8>> {
    use rsa::RsaPrivateKey;
    use rsa::pkcs1v15::SigningKey;
    use rsa::pkcs8::DecodePrivateKey;
    use rsa::signature::{SignatureEncoding, Signer};
    use sha2::Sha256;

    let private_key = RsaPrivateKey::from_pkcs8_pem(&normalize_pem(private_key_pem))
        .context("signing private key is not a valid PKCS#8 RSA private key")?;
    let signature = SigningKey::<Sha256>::new(private_key)
        .try_sign(payload)
        .context("failed to produce RSA signature")?;
    Ok(signature.to_bytes().into_vec())
}

/// Generate a fresh 2048-bit RSA signing keypair, returned as
/// `(PKCS#8 private-key PEM, SubjectPublicKeyInfo public-key PEM)` — the same
/// formats `openssl genpkey` / `openssl rsa -pubout` produce.
pub fn generate_rsa_signing_keypair() -> Result<(String, String)> {
    use rsa::RsaPrivateKey;
    use rsa::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};

    // `rsa` 0.9 is built against `rand_core` 0.6, while the workspace `rand` is
    // 0.9 (`rand_core` 0.9) — the two trait sets are not interchangeable, so an
    // rng from `rand::rng()` does not satisfy `RsaPrivateKey::new`. Use the
    // `rand_core` that `rsa` itself re-exports, which keeps the versions matched
    // no matter which one `rand` moves to. `OsRng` draws straight from the
    // operating system CSPRNG.
    let mut rng = rsa::rand_core::OsRng;
    let private_key =
        RsaPrivateKey::new(&mut rng, 2048).context("failed to generate RSA signing key")?;
    let private_pem = private_key
        .to_pkcs8_pem(LineEnding::LF)
        .context("failed to encode private key")?
        .to_string();
    let public_pem = rsa::RsaPublicKey::from(&private_key)
        .to_public_key_pem(LineEnding::LF)
        .context("failed to encode public key")?;
    Ok((private_pem, public_pem))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::process::Command;

    /// Produce a `(private-key PEM, SPKI public-key PEM, payload, signature)`
    /// tuple with the OpenSSL CLI, cross-checking the pure-Rust sign/verify path
    /// against the same RSASSA-PKCS#1 v1.5 over SHA-256 scheme the installer
    /// (`install.sh`) and release packaging (`cargo xtask package`) use. Returns
    /// `None` when openssl is unavailable or its spawn fails, so this interop
    /// guard never reintroduces the build-failing flake we removed: the pure-Rust
    /// sign/verify path is fully covered by the round-trip test, and this only
    /// adds cross-checking against real openssl output where openssl exists.
    fn openssl_signed_vector(dir: &Path) -> Option<(String, String, Vec<u8>, Vec<u8>)> {
        let private_key = dir.join("interop-private.pem");
        let public_key = dir.join("interop-public.pem");
        let payload_path = dir.join("interop-payload.bin");
        let signature_path = dir.join("interop.sig");
        std::fs::write(&payload_path, b"version = 1\n").ok()?;

        let run = |args: &[&str]| -> Option<()> {
            let output = Command::new("openssl").args(args).output().ok()?;
            output.status.success().then_some(())
        };
        run(&[
            "genpkey",
            "-algorithm",
            "RSA",
            "-pkeyopt",
            "rsa_keygen_bits:2048",
            "-out",
            private_key.to_string_lossy().as_ref(),
        ])?;
        run(&[
            "rsa",
            "-in",
            private_key.to_string_lossy().as_ref(),
            "-pubout",
            "-out",
            public_key.to_string_lossy().as_ref(),
        ])?;
        run(&[
            "dgst",
            "-sha256",
            "-sign",
            private_key.to_string_lossy().as_ref(),
            "-out",
            signature_path.to_string_lossy().as_ref(),
            payload_path.to_string_lossy().as_ref(),
        ])?;

        Some((
            std::fs::read_to_string(&private_key).ok()?,
            std::fs::read_to_string(&public_key).ok()?,
            std::fs::read(&payload_path).ok()?,
            std::fs::read(&signature_path).ok()?,
        ))
    }

    #[test]
    fn signing_tolerates_crlf_and_trailing_whitespace_pems() -> Result<()> {
        // Windows tooling (PowerShell Set-Content, editors) can rewrite a PEM with
        // CRLF endings, a stray trailing space on the boundary line, or a UTF-8 BOM.
        // The OpenSSL CLI accepted these, so the Rust path must too.
        let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
        let payload = b"version = 1\n";

        let crlf_private = private_pem.replace('\n', "\r\n");
        let trailing_ws_private = private_pem
            .lines()
            .map(|line| format!("{line} "))
            .collect::<Vec<_>>()
            .join("\n");
        let bom_public = format!("\u{feff}{public_pem}");

        for variant in [crlf_private, trailing_ws_private] {
            let signature = sign_rsa_pkcs1_sha256_signature(&variant, payload)?;
            verify_rsa_pkcs1_sha256_signature(&public_pem, payload, &signature, "metadata")?;
        }
        let signature = sign_rsa_pkcs1_sha256_signature(&private_pem, payload)?;
        verify_rsa_pkcs1_sha256_signature(&bom_public, payload, &signature, "metadata")?;
        Ok(())
    }

    #[test]
    fn rsa_sign_verify_round_trips_in_pure_rust() -> Result<()> {
        let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
        let payload = b"version = 1\n";

        let signature = sign_rsa_pkcs1_sha256_signature(&private_pem, payload)?;
        verify_rsa_pkcs1_sha256_signature(&public_pem, payload, &signature, "metadata")?;

        let mut tampered = payload.to_vec();
        tampered[0] ^= 0x01;
        let error =
            verify_rsa_pkcs1_sha256_signature(&public_pem, &tampered, &signature, "metadata")
                .expect_err("a tampered payload must be rejected")
                .to_string();
        assert!(error.contains("metadata signature verification failed"));
        Ok(())
    }

    #[test]
    fn rsa_verifier_is_byte_compatible_with_openssl_output() -> Result<()> {
        let (root, _paths) = crate::test_support::temp_app_paths("openssl-interop-vector");
        std::fs::create_dir_all(&root)?;
        let Some((private_key_pem, public_key_pem, payload, openssl_signature)) =
            openssl_signed_vector(&root)
        else {
            eprintln!(
                "skipping openssl interop check: openssl CLI unavailable or failed to produce a signature"
            );
            std::fs::remove_dir_all(&root).ok();
            return Ok(());
        };

        // Our verifier accepts a signature produced by the openssl CLI.
        verify_rsa_pkcs1_sha256_signature(
            &public_key_pem,
            &payload,
            &openssl_signature,
            "metadata",
        )
        .expect("pure-Rust verifier must accept an openssl-produced signature");

        // Our signer is byte-identical to `openssl dgst -sha256 -sign`, so artifacts
        // signed by the Rust xtask verify with the openssl-based installers and vice-versa.
        let rust_signature = sign_rsa_pkcs1_sha256_signature(&private_key_pem, &payload)?;
        assert_eq!(
            rust_signature, openssl_signature,
            "Rust signature must match openssl byte-for-byte"
        );

        let mut tampered = payload;
        tampered[0] ^= 0x01;
        let error = verify_rsa_pkcs1_sha256_signature(
            &public_key_pem,
            &tampered,
            &openssl_signature,
            "metadata",
        )
        .expect_err("a tampered payload must be rejected")
        .to_string();
        assert!(error.contains("metadata signature verification failed"));
        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }
}
