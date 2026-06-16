//! Repository task runner.
//!
//! Currently provides the release-artifact signing toolchain in pure Rust so the
//! project's signing, CI verification, and test keygen no longer depend on the
//! `openssl` CLI. Run via the workspace alias `cargo xtask <command>`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rocm_core::{
    generate_rsa_signing_keypair, sign_rsa_pkcs1_sha256_signature,
    verify_rsa_pkcs1_sha256_signature,
};

#[derive(Parser)]
#[command(name = "xtask", about = "rocm-cli repository tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a 2048-bit RSA signing keypair (PKCS#8 private + SPKI public PEM).
    Keygen {
        /// Path to write the PKCS#8 private-key PEM.
        #[arg(long)]
        private_out: PathBuf,
        /// Path to write the SubjectPublicKeyInfo public-key PEM.
        #[arg(long)]
        public_out: PathBuf,
    },
    /// Sign a file with an RSA private key (RSASSA-PKCS#1 v1.5 over SHA-256).
    Sign {
        /// Path to the PKCS#8 private-key PEM.
        #[arg(long)]
        private_key: PathBuf,
        /// File whose contents are signed.
        #[arg(long = "in")]
        input: PathBuf,
        /// Path to write the raw signature bytes.
        #[arg(long = "out")]
        output: PathBuf,
    },
    /// Verify a file's signature against an RSA public key.
    Verify {
        /// Path to the SubjectPublicKeyInfo public-key PEM.
        #[arg(long)]
        public_key: PathBuf,
        /// File whose signature is checked.
        #[arg(long = "in")]
        input: PathBuf,
        /// Path to the raw signature bytes.
        #[arg(long)]
        signature: PathBuf,
    },
}

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

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Keygen {
            private_out,
            public_out,
        } => {
            let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
            write_private_key(&private_out, &private_pem)?;
            fs::write(&public_out, public_pem.as_bytes())
                .with_context(|| format!("failed to write {}", public_out.display()))?;
        }
        Command::Sign {
            private_key,
            input,
            output,
        } => {
            let private_pem = fs::read_to_string(&private_key)
                .with_context(|| format!("failed to read {}", private_key.display()))?;
            let payload =
                fs::read(&input).with_context(|| format!("failed to read {}", input.display()))?;
            let signature = sign_rsa_pkcs1_sha256_signature(&private_pem, &payload)?;
            fs::write(&output, signature)
                .with_context(|| format!("failed to write {}", output.display()))?;
        }
        Command::Verify {
            public_key,
            input,
            signature,
        } => {
            let public_pem = fs::read_to_string(&public_key)
                .with_context(|| format!("failed to read {}", public_key.display()))?;
            let payload =
                fs::read(&input).with_context(|| format!("failed to read {}", input.display()))?;
            let signature_bytes = fs::read(&signature)
                .with_context(|| format!("failed to read {}", signature.display()))?;
            let label = input.display().to_string();
            verify_rsa_pkcs1_sha256_signature(&public_pem, &payload, &signature_bytes, &label)?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
