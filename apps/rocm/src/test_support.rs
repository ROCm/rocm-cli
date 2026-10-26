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
    std::env::var("PROPTEST_CASES")
        .ok()
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

mod tests {
    use super::*;

    #[test]
    fn unique_suffix_never_repeats_within_a_process() {
        let suffixes: std::collections::HashSet<String> =
            (0..64).map(|_| unique_suffix()).collect();
        assert_eq!(suffixes.len(), 64, "two calls must never share a suffix");
    }
}
