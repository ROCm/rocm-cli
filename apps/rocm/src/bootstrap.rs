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

fn run_setup() -> Result<()> {
    if interactive_terminal() {
        crate::dash::run_bootstrap()
    } else {
        println!(
            "ROCm setup needs an interactive terminal. Run `rocm bootstrap setup` from a terminal to set up ROCm/TheRock."
        );
        Ok(())
    }
}
