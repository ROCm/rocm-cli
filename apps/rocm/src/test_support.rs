// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared test-only fixtures for `rocm`'s module test suites.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use rocm_core::AppPaths;

use crate::runtimes_cmd::{runtime_manifest_path, runtime_registry_dir};
use crate::therock;

/// Serializes the `therock` tests (those in `therock.rs` and its child modules)
/// that replace process-global environment variables while they run: `PATH`, the
/// `ROCM_CLI_THEROCK_*` and managed-Python variables, and the Windows temp-dir
/// variables. Because env is shared across all test threads, any `therock` test
/// that spawns a bare-name binary (e.g. `tar`) via `PATH` lookup must also hold
/// this lock, or it can fail with ENOENT while another test has temporarily
/// narrowed `PATH`.
///
/// It stays one static for those tests: two locks guarding the same variables
/// there would let the tests that hold them race each other. It is not the only
/// env lock in this crate: `ScopedTestEnv` in `main.rs` serializes the tests
/// that use it, and `engine_ipc.rs` has a private production static,
/// `BUILTIN_ENGINE_ENV_LOCK`, that `with_scoped_builtin_engine_env` takes in
/// production code. Neither is ordered against this lock, so a test that
/// depends on one of those groups' variables is not protected from the
/// `therock` tests by it.
pub(crate) static PROCESS_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A name suffix no other call in this test process returns.
///
/// Test scratch roots are named `<label>-<suffix>`. The pid separates
/// processes and the timestamp keeps names readable across runs, but neither
/// separates two tests in one process: `cargo test` runs them on parallel
/// threads, and two that start in the same millisecond with the same label got
/// the same root, so one test's cleanup deleted the other's files. The counter
/// is per-process and monotonic, so no two calls here can agree.
pub(crate) fn unique_suffix() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}-{}",
        std::process::id(),
        rocm_core::unix_time_millis(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// Case count for a property test: `PROPTEST_CASES` (proptest's own knob,
/// which an explicit `with_cases` would otherwise override) when it is set and
/// above `default`, otherwise `default`.
///
/// Never below `default`: the counts are chosen so a property catches the
/// defects it targets with near certainty, and some of those are hit by a
/// small share of cases, so a lower count would pass with them present. `0`
/// would pass vacuously.
pub(crate) fn proptest_cases(default: u32) -> u32 {
    case_count_from_env("PROPTEST_CASES", default)
}

/// Case count read from `var`, raising `default` but never lowering it, for
/// the same reason as [`proptest_cases`]. The uninstall properties use it with
/// their own knob, `ROCM_DELETION_PROP_CASES`. A value that is ignored,
/// because it is lower or does not parse, is reported on stderr.
pub(crate) fn case_count_from_env(var: &str, default: u32) -> u32 {
    let requested = std::env::var(var).ok();
    let cases = raised_case_count(requested.as_deref(), default);
    if let Some(requested) = requested
        && requested.parse::<u32>().ok() != Some(cases)
    {
        eprintln!("{var}={requested} ignored: this property runs at least {default} cases");
    }
    cases
}

/// The pure half of [`case_count_from_env`].
fn raised_case_count(requested: Option<&str>, default: u32) -> u32 {
    requested
        .and_then(|value| value.parse().ok())
        .map_or(default, |requested: u32| requested.max(default))
}

/// Split `bytes` at the given (unsorted, possibly duplicated) offsets, each
/// taken modulo `bytes.len() + 1`.
pub(crate) fn split_at_offsets<'a>(bytes: &'a [u8], offsets: &[usize]) -> Vec<&'a [u8]> {
    let mut cuts: Vec<usize> = offsets.iter().map(|o| o % (bytes.len() + 1)).collect();
    cuts.sort_unstable();
    cuts.dedup();
    let mut pieces = Vec::new();
    let mut start = 0;
    for cut in cuts {
        pieces.push(&bytes[start..cut]);
        start = cut;
    }
    pieces.push(&bytes[start..]);
    pieces
}

/// Call `make_root` from many threads at once, remove what it made, and
/// assert no two calls returned the same root.
///
/// Concurrent rather than sequential, because that is how tests call these
/// helpers: a run of calls made one after another can let the clock tick
/// between them and pass by luck, while threads released together all read the
/// same millisecond, so a root keyed on pid and time alone repeats.
pub(crate) fn assert_each_call_gets_its_own_root(make_root: impl Fn() -> PathBuf + Sync) {
    const CALLERS: usize = 16;
    let start = std::sync::Barrier::new(CALLERS);
    let roots: Vec<PathBuf> = std::thread::scope(|scope| {
        let callers: Vec<_> = (0..CALLERS)
            .map(|_| {
                scope.spawn(|| {
                    start.wait();
                    make_root()
                })
            })
            .collect();
        callers
            .into_iter()
            .map(|caller| caller.join().expect("a caller panicked"))
            .collect()
    });
    for root in &roots {
        let _ = std::fs::remove_dir_all(root);
    }
    let distinct: std::collections::HashSet<_> = roots.iter().collect();
    assert_eq!(
        distinct.len(),
        CALLERS,
        "two calls with one label must not share a root"
    );
}

/// Write a fake `release-pip` runtime install (manifest, registry entry, and
/// the on-disk layout `InstalledRuntimeManifest` expects) under `paths`, for
/// tests across the runtime/SDK-install test suites that need a usable
/// runtime fixture without a real ROCm SDK install.
pub(crate) fn write_test_pip_runtime(
    paths: &AppPaths,
    runtime_key: &str,
    runtime_id: &str,
    version: &str,
    installed_at_unix_ms: u128,
) -> Result<therock::InstalledRuntimeManifest> {
    let install_root = paths
        .data_dir
        .join("runtimes")
        .join("wheel")
        .join(runtime_key);
    let scripts_dir = install_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
    let python_executable = scripts_dir.join(if cfg!(windows) {
        "python.exe"
    } else {
        "python"
    });
    let sdk_root = install_root.join("_rocm_sdk_devel");
    let sdk_bin = sdk_root.join("bin");
    fs::create_dir_all(&scripts_dir)?;
    fs::create_dir_all(&sdk_bin)?;
    let amdhip = sdk_bin.join(if cfg!(windows) {
        "amdhip64_7.dll"
    } else {
        "libamdhip64.so"
    });
    let hipblas = sdk_bin.join(if cfg!(windows) {
        "hipblas.dll"
    } else {
        "libhipblas.so"
    });
    fs::write(&python_executable, "python")?;
    fs::write(&amdhip, "amdhip")?;
    fs::write(&hipblas, "hipblas")?;

    let manifest = therock::InstalledRuntimeManifest {
        runtime_key: runtime_key.to_owned(),
        runtime_id: runtime_id.to_owned(),
        channel: "release".to_owned(),
        format: "wheel".to_owned(),
        family: "gfx120X-all".to_owned(),
        family_source: "test".to_owned(),
        version: version.to_owned(),
        install_root: install_root.clone(),
        selected_artifact_url: "https://example.invalid/therock".to_owned(),
        source_layout_generation: None,
        index_url: Some("https://example.invalid/therock".to_owned()),
        tarball_file_name: None,
        python_launcher: Some("python".to_owned()),
        python_executable: Some(python_executable.display().to_string()),
        pip_cache_dir: Some(paths.cache_dir.join("uv").join("therock")),
        rocm_sdk: Some(therock::RocmSdkPythonProbe {
            import_ok: true,
            root_path: Some(sdk_root.clone()),
            bin_path: Some(sdk_bin.clone()),
            runtime_roots: vec![sdk_root],
            bin_paths: vec![sdk_bin.clone()],
            library_paths: vec![sdk_bin],
            resolved_libraries: vec![
                therock::RocmSdkLibraryProbe {
                    shortname: "amdhip64".to_owned(),
                    paths: vec![amdhip],
                },
                therock::RocmSdkLibraryProbe {
                    shortname: "hipblas".to_owned(),
                    paths: vec![hipblas],
                },
            ],
            ..therock::RocmSdkPythonProbe::default()
        }),
        sdk_torch: None,
        wheel_composition: None,
        read_only: false,
        imported_from: None,
        devel: true,
        installed_at_unix_ms,
    };
    fs::create_dir_all(runtime_registry_dir(paths))?;
    fs::write(
        runtime_manifest_path(paths, runtime_key),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::write(
        install_root.join(".rocm-cli-runtime.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

/// A `rocm serve` argv the serve safety policy refuses, with the refusal each
/// surface reports for it.
///
/// Two surfaces enforce the policy on a model-proposed `serve`: the chat tool
/// policy (`tool_policy::validate_chat_rocm_command_safety`) and the
/// provider-assisted planner (`freeform_planner::validate_provider_planner_tool_call`).
/// Both suites run this one table, so a rule missing from either surface fails
/// that surface's tests, and the same argv cannot be refused by one for a
/// different reason than the other without the table saying so.
pub(crate) struct ServeSafetyCase {
    /// The rule the argv breaks.
    pub(crate) rule: &'static str,
    pub(crate) args: &'static [&'static str],
    /// The full message the chat tool policy refuses it with.
    pub(crate) chat_error: &'static str,
    /// The full message the planner refuses it with.
    pub(crate) planner_error: &'static str,
}

/// Every refusal the serve safety policy has, each rule reached by more than one
/// spelling of its trigger. A case breaks exactly one rule, so the message names
/// the rule that fired and nothing else could have.
pub(crate) const SERVE_SAFETY_REFUSALS: &[ServeSafetyCase] = &[
    ServeSafetyCase {
        rule: "public bind",
        args: &[
            "serve",
            "qwen",
            "--engine",
            "vllm",
            "--allow-public-bind",
            "--managed",
        ],
        chat_error: "local assistant cannot request public network binding",
        planner_error: "provider planner cannot request public network binding",
    },
    ServeSafetyCase {
        rule: "public bind",
        args: &["serve", "qwen", "--managed", "--allow-public-bind"],
        chat_error: "local assistant cannot request public network binding",
        planner_error: "provider planner cannot request public network binding",
    },
    ServeSafetyCase {
        rule: "cpu device",
        args: &[
            "serve",
            "qwen",
            "--engine",
            "vllm",
            "--device",
            "cpu",
            "--managed",
        ],
        chat_error: "local assistant cannot request CPU execution; ROCm GPU execution is required",
        planner_error: "provider planner cannot request CPU execution; ROCm GPU execution is required",
    },
    ServeSafetyCase {
        rule: "cpu device",
        args: &["serve", "qwen", "--device=cpu", "--managed"],
        chat_error: "local assistant cannot request CPU execution; ROCm GPU execution is required",
        planner_error: "provider planner cannot request CPU execution; ROCm GPU execution is required",
    },
    ServeSafetyCase {
        rule: "cpu device",
        args: &["serve", "qwen", "--device", "cpu_only", "--managed"],
        chat_error: "local assistant cannot request CPU execution; ROCm GPU execution is required",
        planner_error: "provider planner cannot request CPU execution; ROCm GPU execution is required",
    },
    ServeSafetyCase {
        rule: "non-loopback host",
        args: &["serve", "qwen", "--host", "0.0.0.0", "--managed"],
        chat_error: "local assistant cannot request non-local host `0.0.0.0`",
        planner_error: "provider planner cannot request non-local host `0.0.0.0`",
    },
    ServeSafetyCase {
        rule: "non-loopback host",
        args: &["serve", "qwen", "--host=example.test", "--managed"],
        chat_error: "local assistant cannot request non-local host `example.test`",
        planner_error: "provider planner cannot request non-local host `example.test`",
    },
    ServeSafetyCase {
        // Classified public on purpose: only the three exact loopback spellings
        // pass (see `is_public_bind_host`).
        rule: "non-loopback host",
        args: &["serve", "qwen", "--host", "LOCALHOST", "--managed"],
        chat_error: "local assistant cannot request non-local host `LOCALHOST`",
        planner_error: "provider planner cannot request non-local host `LOCALHOST`",
    },
    ServeSafetyCase {
        rule: "managed required",
        args: &["serve", "qwen", "--engine", "vllm"],
        chat_error: "local assistant must request managed serving with --managed",
        planner_error: "provider planner must request managed serving with --managed",
    },
    ServeSafetyCase {
        rule: "managed required",
        args: &["serve", "qwen", "--engine", "vllm", "--foreground"],
        chat_error: "local assistant must request managed serving with --managed",
        planner_error: "provider planner must request managed serving with --managed",
    },
];

/// `--foreground` next to `--managed`: the safety check refuses the pair, but
/// clap does too (the flags conflict), so the planner never reaches its safety
/// check with it and only the chat check is what refuses it first.
pub(crate) const SERVE_FOREGROUND_AND_MANAGED: &[&str] =
    &["serve", "qwen", "--managed", "--foreground"];

/// `rocm serve` argvs the serve safety policy lets through, and that also parse
/// as a `rocm` command, so the planner reaches its safety check with them.
pub(crate) const SERVE_SAFETY_ACCEPTED: &[&[&str]] = &[
    &["serve", "qwen", "--managed"],
    &[
        "serve",
        "qwen",
        "--engine",
        "vllm",
        "--device",
        "gpu_required",
        "--managed",
    ],
    &["serve", "qwen", "--device=gpu_required", "--managed"],
    &["serve", "qwen", "--host", "127.0.0.1", "--managed"],
    &["serve", "qwen", "--host=localhost", "--managed"],
    &["serve", "qwen", "--host", "::1", "--managed"],
];

/// `args` as the owned argv both surfaces take.
pub(crate) fn owned_argv(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

mod tests {
    use super::*;

    #[test]
    fn unique_suffix_never_repeats_within_a_process() {
        let suffixes: std::collections::HashSet<String> =
            (0..64).map(|_| unique_suffix()).collect();
        assert_eq!(suffixes.len(), 64, "two calls must never share a suffix");
    }

    #[test]
    fn case_counts_are_raised_but_never_lowered() {
        assert_eq!(raised_case_count(None, 256), 256);
        assert_eq!(raised_case_count(Some("4096"), 256), 4096);
        assert_eq!(raised_case_count(Some("16"), 256), 256);
        assert_eq!(raised_case_count(Some("0"), 256), 256);
        assert_eq!(raised_case_count(Some("lots"), 256), 256);
    }
}
