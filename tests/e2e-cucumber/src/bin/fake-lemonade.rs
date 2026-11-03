// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A stand-in for the two executables a Lemonade runtime installs: the `lemonade`
//! CLI and the `lemond` server.
//!
//! `@id:serve-lemonade-preparation-recovery` needs Lemonade's llama.cpp backend
//! install to fail deterministically, twice, so the scenario can pin the retry
//! count and the terminal recovery guidance. That failure used to be a
//! `#[cfg(feature = "e2e-test-hooks")]` branch compiled into the Lemonade engine
//! itself, plus a second one in `rocm serve` that waived its no-GPU pre-flight.
//! Driving it from here keeps both out of the binary under test: the CLI spawns
//! these as real subprocesses over the interface it uses for a real runtime, and
//! only the runtime is fake.
//!
//! The harness copies this one binary into a planted runtime directory under
//! both names, so the role is taken from `argv[0]`:
//!
//! - `lemond` -- the server `rocm serve` spawns. It only has to stay alive; the
//!   CLI's readiness check polls the `lemonade` CLI, not this process.
//! - `lemonade` -- the CLI the engine shells out to. Three calls matter, and the
//!   real binary's contract for each is what this reproduces:
//!   - `--host H --port P status` -> exit 0, which is how the engine decides the
//!     server came up.
//!   - `backends` -> [`BACKENDS_TABLE`], the recipe/backend/status table the
//!     engine scrapes. Reporting `llamacpp rocm` as `installable` (not
//!     `installed`) is what makes the engine attempt an install rather than
//!     reuse one.
//!   - `--host H --port P backends install llamacpp:rocm` -> non-zero, the
//!     failure the scenario is about.

use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use e2e_cucumber::fixture_binary::FAKE_LEMONADE_INSTALL_REFUSAL;

/// What `lemonade backends` prints here.
///
/// Read by the Lemonade engine's own unit tests (`include_str!` of this file),
/// which check that the engine parses it into a `llamacpp:rocm` install
/// attempt. Keep it a raw string literal without hashes: that test extracts it
/// textually.
const BACKENDS_TABLE: &str = r"Recipe    Backend   Status
--------  --------  -----------
llamacpp  rocm      installable
llamacpp  vulkan    unsupported
";

fn main() -> ExitCode {
    // Copied rather than symlinked (self-hosted Windows runners commonly lack
    // SeCreateSymbolicLinkPrivilege), so the file name is the only role signal.
    let argv0 = std::env::args_os().next().unwrap_or_default();
    let role = Path::new(&argv0)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let args: Vec<String> = std::env::args().skip(1).collect();

    match role.as_str() {
        "lemond" => run_lemond(),
        "lemonade" => run_lemonade_cli(&args),
        other => {
            eprintln!(
                "fake-lemonade: copied to an unexpected name {other:?}; expected `lemond` or \
                 `lemonade`"
            );
            ExitCode::FAILURE
        }
    }
}

/// Stay alive for as long as the process that spawned us does.
///
/// Exiting while the engine still runs would turn the scenario's expected
/// backend-install failure into a spurious "server exited" diagnosis. But the
/// engine does not stop `lemond` when the backend install fails (#729): it
/// exits and leaves the server running, and only the scenario's teardown
/// (`stop_managed_services`, best-effort) stops it. So this fake also exits on
/// its own once its parent is gone -- on Unix, when it has been re-parented --
/// and in any case after [`LEMOND_MAX_LIFETIME`], so a run whose teardown did
/// not get that far still leaves no process behind.
fn run_lemond() -> ExitCode {
    #[cfg(unix)]
    let spawned_by = std::os::unix::process::parent_id();
    let started = std::time::Instant::now();
    while started.elapsed() < LEMOND_MAX_LIFETIME {
        std::thread::sleep(Duration::from_millis(500));
        #[cfg(unix)]
        if std::os::unix::process::parent_id() != spawned_by {
            break;
        }
    }
    ExitCode::SUCCESS
}

/// Backstop on the fake server's lifetime, for a parent that never exits or a
/// platform where re-parenting is not observable. The scenario finishes in
/// seconds, so this is far longer than any run needs.
const LEMOND_MAX_LIFETIME: Duration = Duration::from_mins(10);

fn run_lemonade_cli(args: &[String]) -> ExitCode {
    if let Some(index) = args.iter().position(|arg| arg == "backends") {
        if args.get(index + 1).map(String::as_str) == Some("install") {
            // Fail loudly on stderr: the engine passes the child's stderr
            // through, and the scenario asserts this line to prove the install
            // it saw fail was this fake's rather than a real runtime's.
            eprintln!(
                "{FAKE_LEMONADE_INSTALL_REFUSAL} for @id:serve-lemonade-preparation-recovery"
            );
            return ExitCode::FAILURE;
        }
        print!("{BACKENDS_TABLE}");
        return ExitCode::SUCCESS;
    }

    if args.iter().any(|arg| arg == "status") {
        return ExitCode::SUCCESS;
    }

    // Any other subcommand: succeed quietly. The scenario never reaches model
    // load, and failing here would mask the failure it is actually asserting.
    ExitCode::SUCCESS
}
