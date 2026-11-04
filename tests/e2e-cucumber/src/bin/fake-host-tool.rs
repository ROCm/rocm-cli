// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Stand-in for the tools a simulated host is described by.
//!
//! The CLI learns about the machine partly from files (redirected by
//! `ROCM_CLI_TEST_HOST_ROOT`, see `rocm_core::host_path`) and partly by running
//! programs: `uname`, `lsmod`, `modinfo`, `ldconfig`, `lspci`, `rocminfo`,
//! `rocm_agent_enumerator`, `amd-smi`. A simulated-host scenario links this one
//! binary into a directory at the front of `PATH` under each of those names, so
//! the real host's answers cannot leak into the simulated one.
//!
//! It answers from `$ROCM_CLI_TEST_HOST_ROOT/.fake-tools/`: the file named after
//! the exact command line (`uname -r`, `lspci -nn -D`). Its contents are printed
//! and the exit code is 0. Only a tool in
//! [`ARGV_INDEPENDENT_TOOLS`](e2e_cucumber::simulated_host::ARGV_INDEPENDENT_TOOLS)
//! also answers from the file named after the tool alone, whatever its
//! arguments (see [`canned_output_names`]). Any other command line nobody
//! planted is reported the way a shell reports a missing command, on stderr
//! with exit code 127, naming the full command line: a stand-in that answered
//! a changed invocation with the old canned text would hide exactly the drift
//! between the CLI and the simulated machine this suite exists to catch.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use e2e_cucumber::simulated_host::{TEST_HOST_ROOT_ENV, TOOL_OUTPUT_DIR, canned_output_names};

fn main() -> ExitCode {
    let mut args = std::env::args();
    let tool = tool_name(&args.next().unwrap_or_default());
    let args: Vec<String> = args.collect();

    let Some(root) = std::env::var_os(TEST_HOST_ROOT_ENV).filter(|root| !root.is_empty()) else {
        eprintln!("{tool}: {TEST_HOST_ROOT_ENV} is not set; this is a simulated-host stand-in");
        return ExitCode::from(127);
    };
    let outputs = PathBuf::from(root).join(TOOL_OUTPUT_DIR);
    let names = canned_output_names(&tool, &args);
    for name in &names {
        if let Ok(text) = std::fs::read(outputs.join(name)) {
            // A closed stdout (the caller stopped reading) is not this tool's failure.
            let _ = std::io::stdout().write_all(&text);
            return ExitCode::SUCCESS;
        }
    }
    eprintln!(
        "{tool}: command not found (fake-host-tool: the simulated machine has no canned \
         answer for `{}`; plant one in simulated_host.rs if the CLI now runs it)",
        names[0]
    );
    ExitCode::from(127)
}

/// The tool this invocation stands in for: the full file name it was linked
/// as. `powershell.exe` keeps its extension — the CLI runs it by that exact
/// name inside WSL, and its canned answer is stored under it.
fn tool_name(argv0: &str) -> String {
    Path::new(argv0)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::tool_name;

    #[test]
    fn a_windows_interop_tool_keeps_its_extension() {
        assert_eq!(tool_name("/sim/bin/powershell.exe"), "powershell.exe");
        assert_eq!(tool_name("/sim/bin/uname"), "uname");
        assert_eq!(tool_name("lspci"), "lspci");
    }
}
