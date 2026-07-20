// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Release-artifact signing in pure Rust (RSASSA-PKCS#1 v1.5 over SHA-256), so
//! the project's signing, CI verification, and test keygen do not depend on the
//! `openssl` CLI. The cryptography lives in `rocm-core`; this module is the thin
//! file-I/O layer the CLI dispatches to.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use rocm_core::{
    generate_rsa_signing_keypair, sign_rsa_pkcs1_sha256_signature,
    verify_rsa_pkcs1_sha256_signature,
};

/// Write a private-key PEM, restricting it to owner-only (`0o600`) on Unix so the
/// key is never group- or world-readable. On Unix the file is created with the
/// restricted mode up front, and any pre-existing file is tightened too.
fn write_private_key(path: &Path, pem: &str) -> Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .with_context(|| format!("failed to write {}", path.display()))?;
        // `.mode()` only applies when creating the file; tighten an existing one too.
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("failed to restrict permissions on {}", path.display()))?;
        file.write_all(pem.as_bytes())
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    #[cfg(not(unix))]
    {
        fs::write(path, pem.as_bytes())
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

/// Generate a 2048-bit RSA keypair and write the PKCS#8 private and SPKI public PEMs.
pub fn keygen(private_out: &Path, public_out: &Path) -> Result<()> {
    let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
    write_private_key(private_out, &private_pem)?;
    fs::write(public_out, public_pem.as_bytes())
        .with_context(|| format!("failed to write {}", public_out.display()))?;
    Ok(())
}

/// Sign `input` with the RSA private key and write the raw signature to `output`.
pub fn sign(private_key: &Path, input: &Path, output: &Path) -> Result<()> {
    let private_pem = fs::read_to_string(private_key)
        .with_context(|| format!("failed to read {}", private_key.display()))?;
    let payload = fs::read(input).with_context(|| format!("failed to read {}", input.display()))?;
    let signature = sign_rsa_pkcs1_sha256_signature(&private_pem, &payload)?;
    fs::write(output, signature)
        .with_context(|| format!("failed to write {}", output.display()))?;
    Ok(())
}

/// Verify `input`'s `signature` against the RSA public key.
///
/// When `public_key` is `None`, the key is taken from the
/// `ROCM_CLI_SIGNING_PUBLIC_KEY_PEM` environment variable (the inline PEM that
/// release/nightly CI wire from the signing-key secret) — so callers that already
/// have the key in the environment do not have to materialize a temp file. It is an
/// error if neither a path nor the env var is provided.
pub fn verify(public_key: Option<&Path>, input: &Path, signature: &Path) -> Result<()> {
    let public_pem = if let Some(path) = public_key {
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        let pem = std::env::var("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .context(
                "no signing public key: pass --public-key or set ROCM_CLI_SIGNING_PUBLIC_KEY_PEM",
            )?;
        if pem.ends_with('\n') {
            pem
        } else {
            format!("{pem}\n")
        }
    };
    let payload = fs::read(input).with_context(|| format!("failed to read {}", input.display()))?;
    let signature_bytes =
        fs::read(signature).with_context(|| format!("failed to read {}", signature.display()))?;
    let label = input.display().to_string();
    verify_rsa_pkcs1_sha256_signature(&public_pem, &payload, &signature_bytes, &label)?;
    Ok(())
}
