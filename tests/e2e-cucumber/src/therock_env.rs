// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

/// The environment variables through which a host can redirect where the CLI
/// reads TheRock artifacts from.
///
/// `ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE` is the opt-in the CLI requires before
/// it honours any `ROCM_CLI_THEROCK_*_BASE` variable, so unsetting it alone makes
/// the rest inert. The base variables are listed as well so that a scenario
/// asserting the canonical source does not depend on that gate staying where it
/// is.
///
/// A scenario that asserts what the CLI resolves by default removes these from
/// the child's environment. Self-hosted runners may set a mirror for every job,
/// and a mirror there reports itself as the source the scenario is checking.
pub const THEROCK_BASE_OVERRIDE_VARS: [&str; 7] = [
    "ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE",
    "ROCM_CLI_THEROCK_RELEASE_PIP_BASE",
    "ROCM_CLI_THEROCK_RELEASE_TARBALL_BASE",
    "ROCM_CLI_THEROCK_NIGHTLY_PIP_BASE",
    "ROCM_CLI_THEROCK_NIGHTLY_TARBALL_BASE",
    "ROCM_CLI_THEROCK_NEXT_PIP_BASE",
    "ROCM_CLI_THEROCK_NEXT_TARBALL_BASE",
];

#[cfg(test)]
mod tests {
    use super::THEROCK_BASE_OVERRIDE_VARS;

    /// Every `ROCM_CLI_THEROCK_*` name the CLI's source resolver reads.
    fn names_read_by_the_cli() -> Vec<String> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../apps/rocm/src/therock/source.rs"
        );
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("could not read the CLI's source resolver at {path}: {e}"));
        let mut names: Vec<String> = source
            .split('"')
            .skip(1)
            .step_by(2)
            .filter(|literal| {
                literal.starts_with("ROCM_CLI_THEROCK_")
                    && literal
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            })
            .map(str::to_owned)
            .collect();
        names.sort();
        names.dedup();
        names
    }

    #[test]
    fn every_override_the_cli_reads_is_scrubbed() {
        let read = names_read_by_the_cli();
        assert!(
            read.iter()
                .any(|name| name == "ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE"),
            "the scan of the CLI's resolver found no opt-in flag, so it is reading the wrong \
             file or the wrong shape: {read:?}"
        );
        // `ROCM_CLI_THEROCK_FAMILY` and the like select a target, not a source, and
        // are not in this file's string literals, so this names only the bases
        // and the opt-in.
        for name in &read {
            assert!(
                THEROCK_BASE_OVERRIDE_VARS.contains(&name.as_str()),
                "the CLI reads {name} but THEROCK_BASE_OVERRIDE_VARS does not scrub it, so a \
                 runner that sets it would again change what a canonical-source scenario sees"
            );
        }
    }

    #[test]
    fn the_opt_in_flag_is_scrubbed() {
        assert!(
            THEROCK_BASE_OVERRIDE_VARS.contains(&"ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE"),
            "without the opt-in flag removed, a runner-set base override is still honoured"
        );
    }
}
