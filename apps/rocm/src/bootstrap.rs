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
        crate::tui::run_bootstrap_setup()
    } else {
        println!(
            "ROCm setup needs an interactive terminal. Run `rocm` from a terminal to choose an install folder and set up ROCm/TheRock."
        );
        Ok(())
    }
}
