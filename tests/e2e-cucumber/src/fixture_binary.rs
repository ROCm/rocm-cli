// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Locate a fixture stand-in built as a `[[bin]]` of this crate.
//!
//! A scenario that needs a fake subprocess — `fake-tailscale`, `fake-uv`,
//! `fake-python` — declares it a normal cargo `[[bin]]` instead of
//! hand-writing a shell or batch script, so it runs on every platform cargo
//! already builds for and gets normal incremental, cached builds. Cargo puts
//! it next to this test binary itself, so finding it is a matter of walking
//! up from `std::env::current_exe()`.
//!
//! Only the lookup lives here: planting the located binary at a scenario's
//! chosen path needs `rocm_core::fake_executable::write_fake_executable` to
//! avoid reopening the `ETXTBSY` race, and that helper is a dev-dependency —
//! unavailable to this crate's own library target. See `install_fixture_binary`
//! in `tests/e2e.rs`, the test binary that can reach it.

use std::path::PathBuf;

/// What the `fake-lemonade` stand-in prints on stderr each time it refuses a
/// `lemonade backends install`.
///
/// The Lemonade recovery scenario asserts it in `rocm serve`'s output, which
/// proves the install the CLI attempted was the planted runtime's. If the
/// planted manifest stopped matching what `Detect` expects, `serve` would run
/// the real runtime install instead, and this line would be missing.
pub const FAKE_LEMONADE_INSTALL_REFUSAL: &str =
    "fake-lemonade: refusing to install a backend; this runtime was planted by the E2E harness";

/// The path cargo built the `[[bin]]` named `name` at, alongside this test
/// binary.
pub fn fixture_binary(name: &str) -> PathBuf {
    let mut dir = std::env::current_exe().expect("test binary path");
    dir.pop();
    if dir.ends_with("deps") {
        dir.pop();
    }
    let candidate = dir.join(fixture_target_name(name));
    assert!(
        candidate.exists(),
        "the {name} stand-in was not built at {}; it is a [[bin]] of this crate",
        candidate.display()
    );
    candidate
}

/// The filename a planted fixture executable needs on this platform: `stem`
/// unchanged on unix, `stem.exe` on Windows.
pub fn fixture_target_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_owned()
    }
}
