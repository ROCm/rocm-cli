// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What a Lemonade runtime tree says about its llama.cpp backend, for a scenario
//! that found the backend in a state it did not expect (ROCMAI-1318).
//!
//! The runtime-33 install scenario expects the next install to reuse the pinned
//! backend. When it reinstalls instead, the install output alone cannot say
//! whether Lemonade did not list the backend as installed, the pin file moved, or
//! the build on disk is not the one the pin names. Each of those is one read of
//! the tree, so they are all taken at once and attached to the failure.
//!
//! Read-only apart from the `lemonade backends` listing, which is the same
//! command the engine runs to choose a backend. It runs in the harness's own
//! environment, not the engine's managed library environment, so a listing that
//! fails here is evidence about that environment and not about the engine.
//!
//! Never panics: it only runs while a scenario failure is being reported, and a
//! second panic would replace that report.

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use crate::serve_log::slugify;

/// Most bytes of one file or one output stream kept in the evidence.
const MAX_BYTES: u64 = 64 * 1024;
/// Time allowed for the `lemonade backends` listing. It queries the local
/// server's catalog and can reach for the network to check for updates.
const BACKENDS_TIMEOUT: Duration = Duration::from_secs(30);
/// Where the evidence is archived, under the suite's results directory.
const ARCHIVE_SUBDIR: &str = "lemonade-runtime-evidence";

/// The evidence for the Lemonade runtime tree at `runtime_dir`: the
/// `lemonade backends` listing, then the files that decide what that listing
/// says.
pub async fn collect(runtime_dir: &Path) -> String {
    let mut report = format!("Lemonade runtime tree: {}\n\n", runtime_dir.display());
    report.push_str(&lemonade_backends(runtime_dir, BACKENDS_TIMEOUT).await);
    report.push('\n');
    report.push_str(&files(runtime_dir));
    report
}

/// Write `evidence` under `results_dir`, which CI uploads, so it outlives the
/// scenario's temporary root. Returns where it went, or why it did not.
#[must_use]
pub fn archive(evidence: &str, results_dir: &Path, scenario: &str) -> String {
    let dir = results_dir.join(ARCHIVE_SUBDIR);
    if let Err(error) = std::fs::create_dir_all(&dir) {
        return format!(
            "<not archived: failed to create {}: {error}>",
            dir.display()
        );
    }
    let name = format!("{}.txt", slugify(scenario));
    match std::fs::write(dir.join(&name), evidence) {
        Ok(()) => format!("archived to {ARCHIVE_SUBDIR}/{name}"),
        Err(error) => format!(
            "<not archived: failed to write {}: {error}>",
            dir.join(&name).display()
        ),
    }
}

fn lemonade_binary(runtime_dir: &Path) -> std::path::PathBuf {
    runtime_dir.join(if cfg!(windows) {
        "lemonade.exe"
    } else {
        "lemonade"
    })
}

async fn lemonade_backends(runtime_dir: &Path, timeout: Duration) -> String {
    let binary = lemonade_binary(runtime_dir);
    let mut report = format!(
        "== `{} backends` (run in the harness environment, not the engine's managed one) ==\n",
        binary.display()
    );
    let mut command = tokio::process::Command::new(&binary);
    command
        .arg("backends")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    match tokio::time::timeout(timeout, command.output()).await {
        Err(_) => {
            let _ = writeln!(report, "<did not finish within {timeout:?}; killed>");
        }
        Ok(Err(error)) => {
            let _ = writeln!(report, "<failed to run: {error}>");
        }
        Ok(Ok(output)) => {
            let _ = writeln!(report, "result: {}", output.status);
            for (stream, bytes) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
                let _ = writeln!(report, "-- {stream} --");
                report.push_str(&clamp(bytes));
                if !report.ends_with('\n') {
                    report.push('\n');
                }
            }
        }
    }
    report
}

/// `resources/backend_versions.json`, every `bin/llamacpp/rocm*/version.txt`
/// and `install-lemond.log`, each named by its path whether or not it exists.
fn files(runtime_dir: &Path) -> String {
    let mut report = String::new();
    section(
        &mut report,
        &runtime_dir.join("resources/backend_versions.json"),
    );
    let llamacpp = runtime_dir.join("bin").join("llamacpp");
    match std::fs::read_dir(&llamacpp) {
        Err(error) => {
            let _ = writeln!(
                report,
                "== {} ==\n<cannot list: {error}>\n",
                llamacpp.display()
            );
        }
        Ok(entries) => {
            let mut dirs: Vec<_> = entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("rocm"))
                .map(|entry| entry.path())
                .collect();
            dirs.sort();
            if dirs.is_empty() {
                let _ = writeln!(
                    report,
                    "== {}/rocm* ==\n<no such directory>\n",
                    llamacpp.display()
                );
            }
            for dir in dirs {
                section(&mut report, &dir.join("version.txt"));
            }
        }
    }
    section(&mut report, &runtime_dir.join("install-lemond.log"));
    report
}

fn section(report: &mut String, path: &Path) {
    let _ = writeln!(report, "== {} ==", path.display());
    match read_clamped(path) {
        Ok(text) => report.push_str(&text),
        Err(error) => {
            let _ = write!(report, "<cannot read: {error}>");
        }
    }
    if !report.ends_with('\n') {
        report.push('\n');
    }
    report.push('\n');
}

fn read_clamped(path: &Path) -> std::io::Result<String> {
    let mut body = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut body)?;
    Ok(clamp(&body))
}

fn clamp(bytes: &[u8]) -> String {
    let max = usize::try_from(MAX_BYTES).unwrap_or(usize::MAX);
    if bytes.len() <= max {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    format!(
        "{}\n<truncated at {MAX_BYTES} bytes>",
        String::from_utf8_lossy(&bytes[..max])
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, relative: &str, body: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn every_rocm_build_version_file_is_included_and_other_backends_are_not() {
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            "resources/backend_versions.json",
            "{\"pin\":1}",
        );
        write(root.path(), "bin/llamacpp/rocm-stable/version.txt", "b1111");
        write(root.path(), "bin/llamacpp/rocm/version.txt", "b2222");
        write(root.path(), "bin/llamacpp/vulkan/version.txt", "b3333");
        write(root.path(), "install-lemond.log", "installed ok");
        let report = files(root.path());
        for expected in ["{\"pin\":1}", "b1111", "b2222", "installed ok"] {
            assert!(report.contains(expected), "missing `{expected}`:\n{report}");
        }
        assert!(!report.contains("b3333"), "vulkan is not ROCm:\n{report}");
    }

    #[test]
    fn a_missing_file_is_named_not_skipped() {
        let root = tempfile::tempdir().unwrap();
        let report = files(root.path());
        for name in ["backend_versions.json", "install-lemond.log"] {
            assert!(report.contains(name), "`{name}` not named:\n{report}");
        }
        assert!(report.contains("<cannot read:"), "{report}");
        assert!(report.contains("<cannot list:"), "{report}");
    }

    #[test]
    fn a_llamacpp_directory_with_no_rocm_build_says_so() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("bin/llamacpp/vulkan")).unwrap();
        let report = files(root.path());
        assert!(
            report.contains("rocm*/ ==") || report.contains("/rocm* =="),
            "{report}"
        );
        assert!(report.contains("<no such directory>"), "{report}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_listing_that_outlives_its_timeout_is_killed_and_reported() {
        use std::os::unix::fs::PermissionsExt as _;
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "lemonade", "#!/bin/sh\nexec sleep 30\n");
        let binary = root.path().join("lemonade");
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let started = std::time::Instant::now();
        let report = lemonade_backends(root.path(), Duration::from_millis(200)).await;
        assert!(report.contains("did not finish within"), "{report}");
        assert!(started.elapsed() < Duration::from_secs(10), "not bounded");
    }

    #[test]
    fn a_rocm_build_directory_without_a_version_file_is_still_named() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("bin/llamacpp/rocm-nightly")).unwrap();
        let report = files(root.path());
        assert!(report.contains("rocm-nightly/version.txt"), "{report}");
    }

    #[test]
    fn a_large_file_is_truncated_and_says_so() {
        let root = tempfile::tempdir().unwrap();
        let big = "x".repeat(usize::try_from(MAX_BYTES).unwrap() + 10);
        write(root.path(), "install-lemond.log", &big);
        let report = files(root.path());
        assert!(report.contains("<truncated at"), "{report}");
        assert!(report.len() < big.len() + 1024);
    }

    #[tokio::test]
    async fn a_missing_lemonade_binary_is_reported_and_the_files_still_follow() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "install-lemond.log", "log body");
        let report = collect(root.path()).await;
        assert!(report.contains("backends` ("), "{report}");
        assert!(report.contains("<failed to run:"), "{report}");
        assert!(report.contains("log body"), "{report}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_backends_listing_is_captured_with_its_exit_status() {
        use std::os::unix::fs::PermissionsExt as _;
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            "lemonade",
            "#!/bin/sh\necho \"args: $*\"\necho warn >&2\nexit 3\n",
        );
        let binary = root.path().join("lemonade");
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let report = lemonade_backends(root.path(), BACKENDS_TIMEOUT).await;
        assert!(report.contains("args: backends"), "{report}");
        assert!(report.contains("warn"), "{report}");
        assert!(report.contains("result: exit status: 3"), "{report}");
    }

    #[test]
    fn the_archive_lands_under_the_results_directory() {
        let results = tempfile::tempdir().unwrap();
        let outcome = archive("evidence body", results.path(), "runtime-33 - A scenario");
        assert_eq!(
            outcome,
            "archived to lemonade-runtime-evidence/runtime-33-a-scenario.txt"
        );
        let body = std::fs::read_to_string(
            results
                .path()
                .join("lemonade-runtime-evidence/runtime-33-a-scenario.txt"),
        )
        .unwrap();
        assert_eq!(body, "evidence body");
    }
}
