// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! PowerShell linting: run PSScriptAnalyzer over `.ps1` files via PowerShell 7+
//! (`pwsh`) or pre-installed Windows PowerShell (`powershell`).
//!
//! PSScriptAnalyzer parses each file, so syntax errors are reported as
//! `ParseError` findings too — this both lints and syntax-checks in one pass.
//! The file path is passed to PowerShell via the environment rather than as a
//! command argument because, with `-Command "<string>"`, trailing arguments are
//! not bound to `$args`.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};

/// PSScriptAnalyzer settings file at the repo root.
const SETTINGS: &str = "PSScriptAnalyzerSettings.psd1";

/// PowerShell snippet: analyze the file named by `PS_LINT_FILE` with the
/// settings named by `PS_LINT_SETTINGS`; print any findings and exit non-zero
/// when findings exist.
const ANALYZE_SCRIPT: &str = "$path = $env:PS_LINT_FILE; \
$settings = $env:PS_LINT_SETTINGS; \
$findings = Invoke-ScriptAnalyzer -Path $path -Settings $settings; \
if ($findings) { $findings | Format-Table -AutoSize | Out-String | Write-Output; exit 1 }";

/// Whether `shell` can be spawned (it exists on PATH and runs a trivial command).
fn shell_available(shell: &str) -> bool {
    Command::new(shell)
        .args(["-NoProfile", "-Command", "exit 0"])
        .status()
        .is_ok_and(|status| status.success())
}

/// First available of `pwsh` then `powershell`, if any.
fn detect_powershell() -> Option<&'static str> {
    ["pwsh", "powershell"]
        .into_iter()
        .find(|shell| shell_available(shell))
}

/// Whether the PSScriptAnalyzer module is importable from `shell`.
fn has_analyzer(shell: &str) -> bool {
    Command::new(shell)
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "if (Get-Module -ListAvailable -Name PSScriptAnalyzer) { exit 0 } else { exit 1 }",
        ])
        .status()
        .is_ok_and(|status| status.success())
}

/// All tracked `*.ps1` files, excluding `third_party/`.
fn discover_ps1_files() -> Result<Vec<PathBuf>> {
    let output = Command::new("git")
        .args(["ls-files", "*.ps1"])
        .output()
        .context("failed to run `git ls-files`")?;
    if !output.status.success() {
        bail!(
            "`git ls-files` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("third_party/"))
        .map(PathBuf::from)
        .collect())
}

/// Lint PowerShell scripts with PSScriptAnalyzer via `pwsh`/`powershell`.
pub fn run(shell: Option<String>, files: Vec<PathBuf>) -> Result<()> {
    // The local hook auto-detects a shell; CI requests one explicitly. CI is the
    // authority here — it lints on Windows under both PowerShell editions — and
    // PowerShell Core on a non-Windows host can report edition-specific
    // differences that won't match CI. So skip the auto-detected run off Windows
    // rather than lint with behavior CI won't reproduce. An explicit `--shell`
    // still runs everywhere, leaving a deliberate escape hatch.
    if shell.is_none() && !cfg!(windows) {
        eprintln!(
            "powershell-lint: skipping on non-Windows host; CI lints on Windows. \
             Run with `--shell pwsh` to force."
        );
        return Ok(());
    }

    let powershell = if let Some(requested) = &shell {
        if !shell_available(requested) {
            bail!("PowerShell executable `{requested}` not found");
        }
        requested.clone()
    } else if let Some(found) = detect_powershell() {
        found.to_string()
    } else {
        eprintln!(
            "powershell-lint: no PowerShell (pwsh/powershell) found; skipping. \
             CI enforces PSScriptAnalyzer."
        );
        return Ok(());
    };

    // In auto-detect mode (local hook), skip cleanly when the module is absent.
    // When a shell is requested explicitly (CI), let the analysis surface a
    // missing module as a failure instead of silently passing.
    if shell.is_none() && !has_analyzer(&powershell) {
        eprintln!(
            "powershell-lint: PSScriptAnalyzer module not found; skipping. \
             Install with `Install-Module PSScriptAnalyzer -Scope CurrentUser`."
        );
        return Ok(());
    }

    let files = if files.is_empty() {
        discover_ps1_files()?
    } else {
        files
    };
    if files.is_empty() {
        println!("powershell-lint: no .ps1 files to analyze.");
        return Ok(());
    }

    let settings =
        fs::canonicalize(SETTINGS).with_context(|| format!("failed to locate {SETTINGS}"))?;

    println!(
        "powershell-lint: analyzing {} file(s) with `{powershell}` (settings: {SETTINGS}).",
        files.len()
    );

    let mut failed = false;
    for file in &files {
        let absolute = fs::canonicalize(file)
            .with_context(|| format!("failed to locate {}", file.display()))?;
        // Inherit stdout/stderr so findings print in place.
        let status = Command::new(&powershell)
            .args(["-NoProfile", "-NonInteractive", "-Command", ANALYZE_SCRIPT])
            .env("PS_LINT_FILE", &absolute)
            .env("PS_LINT_SETTINGS", &settings)
            .status()
            .with_context(|| format!("failed to run {powershell} on {}", file.display()))?;
        if status.success() {
            println!("  ok   {}", file.display());
        } else {
            failed = true;
            eprintln!(
                "  FAIL {} (PSScriptAnalyzer reported findings)",
                file.display()
            );
        }
    }

    if failed {
        bail!("PSScriptAnalyzer reported findings");
    }
    println!("powershell-lint: {} file(s) clean.", files.len());
    Ok(())
}
