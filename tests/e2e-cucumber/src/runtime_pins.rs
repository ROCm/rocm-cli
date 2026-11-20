// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The pinned third-party runtime versions, read from the workspace-root
//! `runtime-deps.toml` as plain data.
//!
//! That file is the single source of truth for the pins: the product turns it
//! into constants at build time, in the `rocm-deps` crate. The suite is
//! black-box and imports nothing from the rocm-cli codebase (see the README's
//! design principles), so it reads the same file instead. A scenario that must
//! plant something keyed by a pinned version, such as the Lemonade runtime's
//! `env_id`, then follows a version bump with no copy to keep in step.

use std::path::{Path, PathBuf};

/// The pin file, relative to this crate's manifest directory.
const PINS_FILE: &str = "../../runtime-deps.toml";

/// Path of the workspace-root pin file.
pub fn pins_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(PINS_FILE)
}

/// The `version` pinned for `[runtime.<runtime>]` in the workspace pin file.
///
/// Panics if the file cannot be read or has no such pin: a scenario that plants
/// a runtime under a missing or misspelled pin must fail loudly, not plant one
/// the CLI does not recognise.
pub fn pinned_version(runtime: &str) -> String {
    let path = pins_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    version_in(&text, runtime).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The `version` field of `[runtime.<runtime>]` in a pin file's text.
pub fn version_in(text: &str, runtime: &str) -> Result<String, String> {
    let doc: toml::Table = toml::from_str(text).map_err(|e| format!("not valid TOML: {e}"))?;
    doc.get("runtime")
        .and_then(toml::Value::as_table)
        .and_then(|runtimes| runtimes.get(runtime))
        .and_then(|entry| entry.get("version"))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("no string `version` under [runtime.{runtime}]"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_lemonade_pin_from_the_workspace_file() {
        let version = pinned_version("lemonade");
        // The pin is the value the file holds under `[runtime.lemonade]`, read
        // independently of `version_in`, so a parser that ignored the runtime
        // name or returned a constant would fail here.
        let text = std::fs::read_to_string(pins_path()).expect("pin file is readable");
        let in_section = text
            .split("[runtime.lemonade]")
            .nth(1)
            .expect("pin file has a [runtime.lemonade] table");
        let line = in_section
            .lines()
            .find(|line| line.trim_start().starts_with("version"))
            .expect("[runtime.lemonade] has a version line");
        assert!(
            line.contains(&format!("\"{version}\"")),
            "pinned_version returned {version:?}, the file says {line:?}"
        );
        // Same shape the product's own pin check requires: a bare version.
        assert!(
            !version.is_empty() && !version.starts_with('v'),
            "pin must be a bare version, got {version:?}"
        );
    }

    #[test]
    fn picks_the_named_runtime_and_rejects_a_missing_one() {
        let text =
            "[runtime.lemonade]\nversion = \"1.2.3\"\n\n[runtime.other]\nversion = \"9.9.9\"\n";
        assert_eq!(version_in(text, "lemonade").as_deref(), Ok("1.2.3"));
        assert_eq!(version_in(text, "other").as_deref(), Ok("9.9.9"));
        assert!(version_in(text, "absent").is_err());
        assert!(version_in("[runtime.lemonade]\n", "lemonade").is_err());
    }

    #[test]
    fn rejects_a_version_that_is_not_a_string() {
        // `version = 11.5` parses as a float: the pin must be text, as the
        // product's build script reads it, so this is an error, not "11.5".
        let error = version_in("[runtime.lemonade]\nversion = 11.5\n", "lemonade").unwrap_err();
        assert!(error.contains("no string `version`"), "{error}");
    }
}
