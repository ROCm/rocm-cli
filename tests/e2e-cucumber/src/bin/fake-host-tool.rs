// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Stand-in for the tools a simulated host is described by.
//!
//! The CLI learns about the machine partly from files (redirected by
//! `ROCM_CLI_TEST_HOST_ROOT`, see `rocm_core::host_path`) and partly by running
//! programs ([`HOST_TOOLS`](e2e_cucumber::simulated_host::HOST_TOOLS)). A
//! simulated-host scenario takes all of those off `PATH` and links this one
//! binary, at the front of it, under the name of each tool the simulated machine
//! has, so the real host's answers cannot leak into the simulated one and a tool
//! the machine lacks is simply absent.
//!
//! It answers from `$ROCM_CLI_TEST_HOST_ROOT/.fake-tools/`: the file named after
//! the exact command line (`uname -r`, `lspci -nn -D`). Its contents are printed
//! and the exit code is 0. Only a tool in
//! [`ARGV_INDEPENDENT_TOOLS`](e2e_cucumber::simulated_host::ARGV_INDEPENDENT_TOOLS)
//! also answers from the file named after the tool alone, whatever its
//! arguments (see [`canned_answer`]). Any other command line nobody
//! planted is reported the way a shell reports a missing command, on stderr
//! with exit code 127, naming the full command line: a stand-in that answered
//! a changed invocation with the old canned text would hide exactly the drift
//! between the CLI and the simulated machine this suite exists to catch.

use std::ffi::OsString;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use e2e_cucumber::simulated_host::{TEST_HOST_ROOT_ENV, TOOL_OUTPUT_DIR, canned_answer};

fn main() -> ExitCode {
    let (tool, args) = command_line(std::env::args_os());

    let Some(root) = std::env::var_os(TEST_HOST_ROOT_ENV).filter(|root| !root.is_empty()) else {
        eprintln!("{tool}: {TEST_HOST_ROOT_ENV} is not set; this is a simulated-host stand-in");
        return ExitCode::from(127);
    };
    match canned_answer(&PathBuf::from(root).join(TOOL_OUTPUT_DIR), &tool, &args) {
        Ok(text) => {
            // A closed stdout (the caller stopped reading) is not this tool's failure.
            let _ = std::io::stdout().write_all(&text);
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(127)
        }
    }
}

/// The tool name and arguments of this invocation. Lossy rather than
/// `std::env::args()`, which panics on an argument that is not UTF-8: such a
/// command line was planted for nobody, so it gets the missing-command answer
/// like any other instead of a crash.
fn command_line(mut argv: impl Iterator<Item = OsString>) -> (String, Vec<String>) {
    let tool = tool_name(&argv.next().unwrap_or_default().to_string_lossy());
    let args = argv.map(|arg| arg.to_string_lossy().into_owned()).collect();
    (tool, args)
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

    /// An argument that is not UTF-8 is carried lossily instead of panicking.
    #[cfg(unix)]
    #[test]
    fn an_argument_that_is_not_utf8_is_carried_lossily() {
        use super::command_line;
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt as _;
        let argv = [
            OsString::from("/sim/bin/lspci"),
            OsString::from_vec(vec![b'-', 0xff]),
        ];
        let (tool, args) = command_line(argv.into_iter());
        assert_eq!(tool, "lspci");
        assert_eq!(args, ["-\u{fffd}"]);
    }

    #[test]
    fn a_windows_interop_tool_keeps_its_extension() {
        assert_eq!(tool_name("/sim/bin/powershell.exe"), "powershell.exe");
        assert_eq!(tool_name("/sim/bin/uname"), "uname");
        assert_eq!(tool_name("lspci"), "lspci");
    }
}
