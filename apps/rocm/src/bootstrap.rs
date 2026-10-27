// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::Result;
use clap::Subcommand;
use rocm_core::interactive_terminal;

#[derive(Subcommand, Debug)]
pub(crate) enum BootstrapCommand {
    Setup,
}

pub(crate) fn run(command: Option<BootstrapCommand>) -> Result<()> {
    match command.unwrap_or(BootstrapCommand::Setup) {
        BootstrapCommand::Setup => run_setup(),
    }
}

/// Shown when `rocm bootstrap setup` runs without an interactive terminal —
/// must keep advertising the install-folder choice the onboarding wizard
/// offers (see the dash-tui onboarding Configure step).
const NON_INTERACTIVE_MESSAGE: &str = "ROCm setup needs an interactive terminal. Run `rocm bootstrap setup` from a terminal to choose an install folder and set up ROCm/TheRock.";

fn run_setup() -> Result<()> {
    if interactive_terminal() {
        crate::dash::run_bootstrap()
    } else {
        println!("{NON_INTERACTIVE_MESSAGE}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::NON_INTERACTIVE_MESSAGE;

    #[test]
    fn non_interactive_message_advertises_the_install_folder_choice() {
        assert!(NON_INTERACTIVE_MESSAGE.contains("choose an install folder"));
        assert!(NON_INTERACTIVE_MESSAGE.contains("rocm bootstrap setup"));
    }
}
