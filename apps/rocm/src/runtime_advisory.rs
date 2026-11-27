// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Advisories about an installed ROCm runtime that hold only on some hosts.
//!
//! A runtime can be perfectly installed and still behave badly on one kind of
//! machine. The CLI cannot fix that, but it can say so where the user would
//! otherwise be left staring at a stalled command: when a runtime is activated
//! (`rocm runtimes activate`, `rocm runtimes rollback` and
//! `rocm update --apply --activate`), when an SDK install makes it the active one,
//! and when a Lemonade `rocm serve` is about to use it.
//!
//! One advisory exists today: ROCm SDK 10.2.0 on WSL2, where pageable host-to-GPU
//! copies were measured at about 20 MiB/s (issue #1143). The decision is a pure
//! function of the host and the recorded SDK version ([`advisory_for`]), and every
//! surface prints the text from [`advisory_for`], so the surfaces cannot disagree
//! about when it applies or what it says.

use rocm_core::{AppPaths, is_wsl_host};

use crate::engine_runtime_select::EngineSelection;
use crate::runtime_registry::load_runtime_manifests;

/// The one SDK version the slow-copy measurement was made on. Matched exactly: a
/// newer 10.2.x or a 10.3 has not been measured, and a warning that claims more
/// than was measured would send users away from a build that may be fine.
const SLOW_PAGEABLE_COPY_SDK_VERSION: &str = "10.2.0";

/// The SDK version the advisory points users to, which is the version
/// `rocm install sdk --version` is told to install.
const KNOWN_GOOD_SDK_VERSION: &str = "10.1.0";

/// The public issue that records the measurement and its status.
const ISSUE_URL: &str = "https://github.com/ROCm/rocm-cli/issues/1143";

/// The advisory for a runtime whose recorded SDK version is `sdk_version`, on a
/// host that is (`is_wsl`) or is not WSL2. `None` when there is nothing to say.
pub(crate) fn advisory_for(is_wsl: bool, sdk_version: &str) -> Option<String> {
    (is_wsl && sdk_version.trim() == SLOW_PAGEABLE_COPY_SDK_VERSION).then(|| {
        let silent_minutes = rocm_core::SERVE_SILENT_LOAD_LIMIT.as_secs() / 60;
        format!(
            "warning: ROCm SDK {SLOW_PAGEABLE_COPY_SDK_VERSION} on WSL2 copied pageable host memory \
             to the GPU at about 20 MiB/s on a gfx1151 host, against at least several hundred \
             MiB/s on {KNOWN_GOOD_SDK_VERSION}. Loading a model can then take far longer than usual, so a \
             serve can stay not-ready for a long time while its server is still loading, and a \
             Lemonade serve is stopped once its log has been silent for {silent_minutes} minutes. To switch \
             to {KNOWN_GOOD_SDK_VERSION}, run `rocm install sdk --version {KNOWN_GOOD_SDK_VERSION}`, \
             or `rocm runtimes activate <runtime_key>` for a {KNOWN_GOOD_SDK_VERSION} runtime that \
             `rocm runtimes list` already shows. Details: {ISSUE_URL}"
        )
    })
}

/// [`advisory_for`] on this host, for the registered runtime `runtime_key`.
///
/// `None` when the host is not WSL2, when the runtime is not registered, or when
/// its manifests cannot be read: an advisory is a courtesy, and failing to find
/// one must never fail the command it rides on.
pub(crate) fn advisory_for_runtime_key(paths: &AppPaths, runtime_key: &str) -> Option<String> {
    advisory_for_runtime_key_on(is_wsl_host(), paths, runtime_key)
}

/// [`advisory_for_runtime_key`] for a host that is (`is_wsl`) or is not WSL2.
fn advisory_for_runtime_key_on(
    is_wsl: bool,
    paths: &AppPaths,
    runtime_key: &str,
) -> Option<String> {
    // The host is checked first: it is cheap, and on every other host the
    // manifests need not be read at all.
    if !is_wsl {
        return None;
    }
    let manifests = load_runtime_manifests(paths).ok()?;
    let manifest = manifests
        .iter()
        .find(|manifest| manifest.runtime_key == runtime_key)?;
    advisory_for(true, &manifest.version)
}

/// Print the advisory for the runtime a `rocm serve` is about to launch on, if any.
///
/// On stderr, ahead of the launch and its wait: a runtime with a known slow path
/// on this host is what the user will be waiting on, and stdout stays the plan or
/// the machine-readable launch report.
pub(crate) fn advise_serve(
    paths: &AppPaths,
    engine: &str,
    cpu_only: bool,
    selection: &EngineSelection,
) {
    let runtime_key = selection.runtime_id.as_deref();
    if let Some(advisory) = serve_advisory_on(is_wsl_host(), paths, engine, cpu_only, runtime_key) {
        eprintln!("{advisory}");
    }
}

/// The advisory a serve of `engine` on `runtime_key` owes, on a host that is
/// (`is_wsl`) or is not WSL2. The measurement was a Lemonade `llama-server` on a
/// GPU, so a CPU-only serve and another engine are not warned.
fn serve_advisory_on(
    is_wsl: bool,
    paths: &AppPaths,
    engine: &str,
    cpu_only: bool,
    runtime_key: Option<&str>,
) -> Option<String> {
    if cpu_only || !engine.eq_ignore_ascii_case("lemonade") {
        return None;
    }
    advisory_for_runtime_key_on(is_wsl, paths, runtime_key?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_advisory_applies_only_to_the_measured_version_on_wsl() {
        assert!(advisory_for(true, "10.2.0").is_some());
        assert!(advisory_for(true, " 10.2.0 ").is_some());
        // Not WSL2: the measurement says nothing about native Linux or Windows.
        assert_eq!(advisory_for(false, "10.2.0"), None);
        // Versions that were not measured, or that measured fast.
        for version in [
            "10.1.0",
            "10.2.1",
            "10.2.0a20260822",
            "10.3.0",
            "7.14.1",
            "",
        ] {
            assert_eq!(advisory_for(true, version), None, "{version}");
        }
    }

    /// The text names a command, so the command has to be one the CLI accepts, and
    /// the version it installs has to be the one the text calls known-good.
    #[test]
    fn the_advisory_names_the_workaround_version_and_the_issue() {
        let text = advisory_for(true, "10.2.0").unwrap();
        assert!(text.starts_with("warning: "), "{text}");
        assert!(
            text.contains("`rocm install sdk --version 10.1.0`"),
            "{text}"
        );
        assert!(
            text.contains("`rocm runtimes activate <runtime_key>`"),
            "{text}"
        );
        assert!(text.contains(ISSUE_URL), "{text}");
        // One line: it is printed beside spinner and progress rows.
        assert!(!text.contains('\n'), "{text}");
    }

    /// The `rocm install sdk --version <V>` the advisory prints must parse as an
    /// install of exactly `<V>`. That the install then ends the warning is not
    /// covered: it needs a download, so only the `runtimes activate` route is shown
    /// to clear it (`runtime-lifecycle-33`).
    #[test]
    fn the_named_install_command_parses_to_the_known_good_version() {
        use clap::Parser as _;
        let cli = crate::Cli::try_parse_from([
            "rocm",
            "install",
            "sdk",
            "--version",
            KNOWN_GOOD_SDK_VERSION,
        ])
        .expect("the advisory's install command must parse");
        let Some(crate::Command::Install {
            target: crate::InstallTarget::Sdk { version, .. },
        }) = cli.command
        else {
            panic!("expected `install sdk`");
        };
        assert_eq!(version.as_deref(), Some(KNOWN_GOOD_SDK_VERSION));
    }

    fn plant_runtime(paths: &AppPaths, key: &str, version: &str, stamp: u128) {
        crate::test_support::write_test_pip_runtime(
            paths,
            key,
            "therock-release:gfx1151",
            version,
            stamp,
        )
        .unwrap();
    }

    /// The lookup that connects a runtime key to its recorded version: a planted
    /// 10.2.0 manifest warns on WSL2 only, and an unregistered key or another
    /// version does not.
    #[test]
    fn the_runtime_key_lookup_reads_the_registered_version() {
        let (_root, paths) = crate::tests::test_paths("advisory-runtime-key");
        plant_runtime(&paths, "slow-10-2-0", "10.2.0", 2);
        plant_runtime(&paths, "fast-10-1-0", "10.1.0", 1);
        assert!(advisory_for_runtime_key_on(true, &paths, "slow-10-2-0").is_some());
        assert_eq!(
            advisory_for_runtime_key_on(false, &paths, "slow-10-2-0"),
            None
        );
        assert_eq!(
            advisory_for_runtime_key_on(true, &paths, "fast-10-1-0"),
            None
        );
        assert_eq!(advisory_for_runtime_key_on(true, &paths, "missing"), None);
    }

    /// `rocm serve` warns for a Lemonade GPU serve on the slow runtime, and for
    /// nothing else: not another engine, not `--device cpu`, not an unresolved runtime.
    #[test]
    fn a_serve_is_warned_only_for_a_lemonade_gpu_serve_on_the_slow_runtime() {
        let (_root, paths) = crate::tests::test_paths("advisory-serve");
        plant_runtime(&paths, "slow-10-2-0", "10.2.0", 2);
        let warned = serve_advisory_on(true, &paths, "lemonade", false, Some("slow-10-2-0"));
        assert!(warned.is_some());
        for (engine, cpu_only, key) in [
            ("vllm", false, Some("slow-10-2-0")),
            ("lemonade", true, Some("slow-10-2-0")),
            ("lemonade", false, None),
        ] {
            assert_eq!(
                serve_advisory_on(true, &paths, engine, cpu_only, key),
                None,
                "{engine} cpu_only={cpu_only} {key:?}"
            );
        }
        assert_eq!(
            serve_advisory_on(false, &paths, "lemonade", false, Some("slow-10-2-0")),
            None
        );
    }
}
