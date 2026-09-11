// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A stand-in for the Tailscale CLI, for scenarios that need a tailnet.
//!
//! `rocm remote` shells out to `tailscale status --json` to find candidate
//! machines and to check one is online before dialling it. A scenario that has
//! to be deterministic cannot depend on whether the developer's machine happens
//! to have Tailscale installed, connected, or peered with anything — so it puts
//! this on `PATH` instead and points it at a status document it wrote.
//!
//! A Rust binary rather than a shell script so the scenarios run on Windows too.
//!
//! `FAKE_TAILSCALE_STATUS` names the document to serve. Unset, it reports a
//! daemon that is installed but not logged in, which is its own scenario.

use std::io::Write as _;

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut stdout = std::io::stdout();

    match args.first().map(String::as_str) {
        Some("status") => {
            let document = std::env::var("FAKE_TAILSCALE_STATUS")
                .ok()
                .and_then(|path| std::fs::read_to_string(path).ok())
                .unwrap_or_else(|| r#"{"BackendState":"NeedsLogin"}"#.to_owned());
            let _ = writeln!(stdout, "{document}");
        }
        other => {
            // Anything else is a command a scenario did not set up. Failing
            // loudly beats a silent empty answer that the CLI would read as a
            // legitimate "nothing here".
            let _ = writeln!(
                std::io::stderr(),
                "fake tailscale: unsupported command: {}",
                other.unwrap_or("(none)")
            );
            std::process::exit(2);
        }
    }
}
