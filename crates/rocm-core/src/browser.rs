// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Handing a URL to the user's browser.
//!
//! Behind a trait because starting a browser is the one thing in this area
//! that cannot run in CI: the real implementation spawns a process against
//! whatever desktop the machine has, so any code path that opens a URL is
//! untestable unless the opening itself can be replaced. Callers take
//! `&dyn Opener`, tests pass a fake, and the decision to open stays separate
//! from the opening.
//!
//! One implementation, deliberately. A second opener somewhere else is how the
//! seam stops being a seam.

use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

use crate::{runtime_is_linux, runtime_is_windows};

/// Something that can show the user a URL.
pub trait Opener {
    /// Hand `url` to the user's browser.
    ///
    /// # Errors
    /// When no browser could be started, or the opener reported failure.
    fn open(&self, url: &str) -> Result<()>;
}

/// The real one: whatever this platform uses to open a link.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemOpener;

/// Environment variable that carries the URL to the Windows opener.
const WINDOWS_URL_ENV: &str = "ROCM_OPEN_URL";

/// The PowerShell that opens the URL in `WINDOWS_URL_ENV`.
///
/// `Stop` makes a failed open a non-zero exit, which `cmd /C start` and
/// `rundll32 url.dll,FileProtocolHandler` do not give: the latter exits 0
/// whether or not anything opened, so the caller would report a mail client
/// that never started.
const WINDOWS_OPEN_SCRIPT: &str =
    "$ErrorActionPreference = 'Stop'; Start-Process -FilePath $env:ROCM_OPEN_URL";

/// Run `script` in PowerShell with `url` in `WINDOWS_URL_ENV`.
///
/// The URL travels in the environment rather than the command line. A mail
/// report link contains `&body=`; through `cmd /C start "" <url>` that `&`
/// ends the command and `body=...` runs as a second one, so the link never
/// opens. No shell parses the environment, so nothing in the URL needs quoting.
fn windows_powershell(script: &str, url: &str) -> Command {
    let mut command = Command::new("powershell");
    command
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .env(WINDOWS_URL_ENV, url);
    command
}

fn windows_open_command(url: &str) -> Command {
    windows_powershell(WINDOWS_OPEN_SCRIPT, url)
}

impl Opener for SystemOpener {
    fn open(&self, url: &str) -> Result<()> {
        let status = if runtime_is_windows() {
            windows_open_command(url)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
        } else if cfg!(target_os = "macos") && !runtime_is_linux() {
            Command::new("open")
                .arg(url)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
        } else {
            Command::new("xdg-open")
                .arg(url)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
        }
        .context("failed to open browser")?;
        if status.success() {
            Ok(())
        } else {
            bail!("browser opener exited with status {status}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The URL must not appear on the command line, where a shell could split
    /// it at `&`. This pins how the command is built; the Windows-only tests
    /// below prove the behaviour.
    #[test]
    fn windows_open_command_keeps_the_url_off_the_command_line() {
        let url = "mailto:a@example.com?subject=s&body=b";
        let command = windows_open_command(url);
        let args: Vec<_> = command.get_args().collect();
        assert!(args.iter().all(|a| !a.to_string_lossy().contains(url)));
        let env: Vec<_> = command.get_envs().collect();
        assert!(env.iter().any(|(k, v)| {
            *k == WINDOWS_URL_ENV && v.is_some_and(|v| v.to_string_lossy() == url)
        }));
    }

    /// A URL with `&` reaches the opener whole. Prints it instead of opening
    /// it, because opening needs a desktop.
    #[cfg(windows)]
    #[test]
    fn windows_opener_passes_an_ampersand_url_through_intact() {
        let url = "mailto:a@example.com?subject=one&body=two%20words&x=3";
        let out = windows_powershell("[Console]::Out.Write($env:ROCM_OPEN_URL)", url)
            .output()
            .expect("powershell runs");
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout), url);
    }

    /// The failure this fixes: the old `cmd /C start "" <url>` shape cut the
    /// URL at `&` and exited 1. Kept as a negative control so the test above
    /// is known to be able to fail.
    #[cfg(windows)]
    #[test]
    fn the_old_cmd_shape_cut_the_url_at_the_ampersand() {
        let out = Command::new("cmd")
            .args(["/C", "echo", "mailto:x?subject=a&body=b"])
            .output()
            .expect("cmd runs");
        assert!(!out.status.success());
        assert!(!String::from_utf8_lossy(&out.stdout).contains("body=b"));
    }

    /// A failed open is an error, not a silent success.
    #[cfg(windows)]
    #[test]
    fn windows_opener_reports_failure_when_nothing_can_be_opened() {
        let missing = std::env::temp_dir().join("rocm-opener-no-such-file.rocmtest");
        assert!(!missing.exists());
        let result = SystemOpener.open(&missing.to_string_lossy());
        assert!(result.is_err(), "opening a missing file must fail");
    }
}
