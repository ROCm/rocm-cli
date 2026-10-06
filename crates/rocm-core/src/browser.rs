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

impl Opener for SystemOpener {
    fn open(&self, url: &str) -> Result<()> {
        let status = if runtime_is_windows() {
            Command::new("cmd")
                .args(["/C", "start", "", url])
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
