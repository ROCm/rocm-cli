// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Repository task runner.
//!
//! Provides release-artifact signing, the MANIFEST.md dependency-table
//! generator, commit-trust verification, and PowerShell linting in pure Rust.
//! Each command's implementation lives in its own module; this file holds only
//! the CLI definition and dispatch. Run via the workspace alias
//! `cargo xtask <command>`.

mod manifest;
mod powershell;
mod signing;
mod verify_commits;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};

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
    /// Regenerate the Cargo dependency table in MANIFEST.md from `cargo metadata`.
    Manifest {
        /// Verify the table is up to date without writing; exit non-zero if it would change.
        #[arg(long)]
        check: bool,
    },
    /// Verify that commits in a range are cryptographically signed and carry a
    /// DCO `Signed-off-by` trailer.
    VerifyCommits {
        /// Base ref for the `<base>..HEAD` range. Defaults to the PR base branch
        /// (`origin/$GITHUB_BASE_REF`) in CI, otherwise `origin/main`.
        #[arg(long)]
        base: Option<String>,
        /// Strict mode: require GitHub to report each signature as "Verified"
        /// (shells out to the `gh` CLI). Mutually exclusive with `--check-config`.
        #[arg(long, conflicts_with = "check_config")]
        require_verified: bool,
        /// Assert only that commit signing is configured locally, without
        /// inspecting any commits. Mutually exclusive with `--base`/`--require-verified`.
        #[arg(long, conflicts_with_all = ["base", "require_verified"])]
        check_config: bool,
    },
    /// Lint PowerShell scripts with PSScriptAnalyzer (parse errors are reported
    /// too, so this also covers syntax).
    PowershellLint {
        /// PowerShell executable to run (e.g. `pwsh` or `powershell`). When set,
        /// it is an error if that shell is missing — CI uses this to assert
        /// coverage under a specific edition. When omitted, `pwsh` then
        /// `powershell` is auto-detected and the check skips cleanly on a
        /// non-Windows host, or if neither the shell nor the PSScriptAnalyzer
        /// module is available.
        #[arg(long)]
        shell: Option<String>,
        /// `.ps1` files to lint. When none are given, all tracked `*.ps1`
        /// (excluding `third_party/`) are checked.
        files: Vec<PathBuf>,
    },
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Keygen {
            private_out,
            public_out,
        } => signing::keygen(&private_out, &public_out)?,
        Command::Sign {
            private_key,
            input,
            output,
        } => signing::sign(&private_key, &input, &output)?,
        Command::Verify {
            public_key,
            input,
            signature,
        } => signing::verify(&public_key, &input, &signature)?,
        Command::Manifest { check } => manifest::run(check)?,
        Command::VerifyCommits {
            base,
            require_verified,
            check_config,
        } => {
            // `--check-config` is declared `conflicts_with_all = ["base",
            // "require_verified"]`, so clap rejects those combinations before we
            // get here.
            if check_config {
                verify_commits::check_config()?;
            } else {
                verify_commits::run(base, require_verified)?;
            }
        }
        Command::PowershellLint { shell, files } => powershell::run(shell, files)?,
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
