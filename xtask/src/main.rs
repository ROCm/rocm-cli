// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Repository task runner.
//!
//! Provides release-artifact signing, the MANIFEST.md dependency-table
//! generator, third-party-notices generation, commit-trust verification, and
//! PowerShell linting. Each command's implementation lives in its own module;
//! this file holds only the CLI definition and dispatch. Run via the workspace
//! alias `cargo xtask <command>`.

mod affected;
mod demos;
mod e2e;
mod e2e_report;
mod manifest;
mod package;
mod paths;
mod powershell;
mod signing;
mod tpn;
mod verify_commits;
mod verify_pinned_keys;

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
        /// Path to the SubjectPublicKeyInfo public-key PEM. When omitted, the key is
        /// read from the `ROCM_CLI_SIGNING_PUBLIC_KEY_PEM` environment variable.
        #[arg(long)]
        public_key: Option<PathBuf>,
        /// File whose signature is checked.
        #[arg(long = "in")]
        input: PathBuf,
        /// Path to the raw signature bytes.
        #[arg(long)]
        signature: PathBuf,
    },
    /// Assert the release/metadata public keys pinned in the installers and
    /// `apps/rocm/src/therock.rs` match the canonical published keys under
    /// `docs/keys/`, and that any configured CI signing key matches the pinned
    /// current release key. A no-op while no canonical keys are published.
    VerifyPinnedKeys,
    /// Print the workspace crates affected by a git range (changed crates plus
    /// their transitive dependents) as `cargo` package-selection flags, so CI
    /// can build/test only what a change can reach instead of `--workspace`.
    ///
    /// Output is `--workspace` (fall back to the full workspace — used whenever a
    /// change can't be confined to specific crates, e.g. `Cargo.lock` or the
    /// toolchain file), or `-p <crate> ...`, or empty (nothing Rust-relevant
    /// changed). Note: selection follows the Cargo dependency graph, so a test
    /// that exercises another crate only indirectly (e.g. via a subprocess) is
    /// not captured — the conservative fallbacks cover the common such triggers.
    Affected {
        /// Base ref for the `<base>...HEAD` (merge-base) range. Defaults to
        /// `origin/main`.
        #[arg(long)]
        base: Option<String>,
    },
    /// Regenerate the Cargo dependency table in MANIFEST.md from `cargo metadata`.
    Manifest {
        /// Verify the table is up to date without writing; exit non-zero if it would change.
        #[arg(long)]
        check: bool,
    },
    /// Regenerate THIRD_PARTY_NOTICES.txt from the dependency tree via cargo-about
    /// (using the committed `about.toml` and `about.hbs`).
    Tpn {
        /// Verify the notices file is up to date without writing; exit non-zero if it would change.
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
    /// Build the demo binaries and render the deterministic CLI and Console GIFs.
    Demos {
        /// Demo names to render. When omitted, renders `cli` and `console`.
        #[arg(value_name = "DEMO")]
        names: Vec<String>,
        /// Use binaries already present in `ROCM_BIN_DIR` or the release target dir.
        #[arg(long)]
        skip_build: bool,
    },
    /// Package the built release binaries into a distribution archive
    /// (`.tar.gz` on Unix, `.zip` on Windows) with a SHA-256 sidecar and, when a
    /// signing key is configured, a detached signature.
    ///
    /// Replaces the former `scripts/package-{linux,windows}-release` scripts with
    /// one cross-platform command. Signing inputs are read from the environment
    /// for parity with those scripts: `ROCM_CLI_SIGNING_PRIVATE_KEY_PATH` (a PEM
    /// file) or `ROCM_CLI_SIGNING_PRIVATE_KEY_PEM` (an inline PEM); set
    /// `ROCM_CLI_REQUIRE_SIGNATURE=1` to fail unless a signature is produced.
    Package {
        /// Distribution name (the archive stem and top-level bundle directory),
        /// e.g. `rocm-cli-1.2.3-linux-amd64`.
        dist_name: String,
        /// Output directory for the bundle directory and archive. Relative paths
        /// are resolved against the workspace root. Defaults to `dist`.
        #[arg(default_value = "dist")]
        output_dir: PathBuf,
        /// Optional cross-compilation target triple selecting the
        /// `target/<triple>/release` binaries instead of `target/release`.
        #[arg(long)]
        target: Option<String>,
    },
    /// Build the release `rocm` binary and run the cucumber-rs E2E suite.
    ///
    /// The harness resolves every scenario's expectation (pass / xfail / skip)
    /// per host from its tags + a capability probe + `expectations.toml`, so no
    /// tier flag or tag filter is needed — one run covers a whole platform.
    /// Extra arguments after `--` are still forwarded to the cucumber CLI for
    /// ad-hoc local use, e.g. `cargo xtask e2e -- -n serve-inference`.
    E2e {
        /// Arguments forwarded verbatim to the cucumber test binary (name filter
        /// `-n`, `--fail-fast`, etc.). Not used by CI.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Consolidate per-platform E2E `report.json` files (one per CI job/runner)
    /// into a single cross-platform HTML report, and print a summary matrix to
    /// stdout for `$GITHUB_STEP_SUMMARY`.
    E2eReport {
        /// Directory holding the downloaded E2E artifacts. Each immediate subdir
        /// containing a `report.json` is treated as one platform; its name (an
        /// `e2e-*-report` artifact name) becomes the platform label.
        #[arg(long)]
        artifacts_dir: PathBuf,
        /// Path to write the consolidated HTML report.
        #[arg(long)]
        html_out: PathBuf,
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
        } => signing::verify(public_key.as_deref(), &input, &signature)?,
        Command::VerifyPinnedKeys => verify_pinned_keys::run()?,
        Command::Affected { base } => affected::run(base)?,
        Command::Manifest { check } => manifest::run(check)?,
        Command::Tpn { check } => tpn::run(check)?,
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
        Command::Demos { names, skip_build } => demos::run(&names, skip_build)?,
        Command::Package {
            dist_name,
            output_dir,
            target,
        } => package::run(&dist_name, &output_dir, target.as_deref())?,
        Command::E2e { args } => e2e::run(&args)?,
        Command::E2eReport {
            artifacts_dir,
            html_out,
        } => e2e_report::run(&artifacts_dir, &html_out)?,
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
