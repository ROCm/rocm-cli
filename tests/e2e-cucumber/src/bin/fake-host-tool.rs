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
//! the full command line (`uname -r`), else the one named after the tool alone
//! (`lsmod`). Its contents are printed and the exit code is 0. With neither file
//! the simulated host does not have that tool, which is reported the way a shell
//! reports a missing command: on stderr, with exit code 127.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Must match `simulated_host::TOOL_OUTPUT_DIR`.
const TOOL_OUTPUT_DIR: &str = ".fake-tools";
/// Must match `rocm_core::hardware_root::TEST_HOST_ROOT_ENV`.
const TEST_HOST_ROOT_ENV: &str = "ROCM_CLI_TEST_HOST_ROOT";

fn main() -> ExitCode {
    let mut args = std::env::args();
    let tool = tool_name(&args.next().unwrap_or_default());
    let args: Vec<String> = args.collect();

    let Some(root) = std::env::var_os(TEST_HOST_ROOT_ENV).filter(|root| !root.is_empty()) else {
        eprintln!("{tool}: {TEST_HOST_ROOT_ENV} is not set; this is a simulated-host stand-in");
        return ExitCode::from(127);
    };
    let outputs = PathBuf::from(root).join(TOOL_OUTPUT_DIR);
    let with_args = std::iter::once(tool.as_str())
        .chain(args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ");
    for name in [with_args.as_str(), tool.as_str()] {
        if let Ok(text) = std::fs::read(outputs.join(name)) {
            // A closed stdout (the caller stopped reading) is not this tool's failure.
            let _ = std::io::stdout().write_all(&text);
            return ExitCode::SUCCESS;
        }
    }
    eprintln!("{tool}: command not found");
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
