// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Read-only backend evidence for a failed Lemonade serve, before teardown.

use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt as _};

use crate::serve_log::read_clamped;

const MAX_EVIDENCE_BYTES: u64 = 64 * 1024;
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CANDIDATES: usize = 8;
const TOTAL_PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// Record the installed version and pins separately from executable build output:
/// a pin describes what should be installed, not what a runner actually cached.
///
/// Every candidate is labelled as such; finding a binary does not prove that the
/// router launched it. No install, download, model load, or cache repair is done.
pub async fn backend_evidence(manifest_path: &Path) -> String {
    let mut report = String::new();
    let _ = writeln!(report, "manifest: {}", manifest_path.display());
    let manifest = match read_clamped(manifest_path, MAX_EVIDENCE_BYTES).and_then(|(bytes, _)| {
        serde_json::from_slice::<serde_json::Value>(&bytes).map_err(std::io::Error::other)
    }) {
        Ok(manifest) => manifest,
        Err(error) => return format!("{report}unavailable: {error}"),
    };
    for key in ["version", "backend_recipe", "backend_name", "runtime_dir"] {
        let _ = writeln!(
            report,
            "{key}: {}",
            manifest.get(key).unwrap_or(&serde_json::Value::Null)
        );
    }
    let Some(runtime) = manifest
        .get("runtime_dir")
        .and_then(serde_json::Value::as_str)
    else {
        return format!("{report}backend inventory unavailable: no runtime_dir");
    };
    let runtime = Path::new(runtime);
    let pins = runtime.join("resources/backend_versions.json");
    let _ = writeln!(report, "configured pins (not executable identity):");
    match read_clamped(&pins, MAX_EVIDENCE_BYTES) {
        Ok((bytes, note)) => {
            let _ = writeln!(report, "{}{note}", String::from_utf8_lossy(&bytes));
        }
        Err(error) => {
            let _ = writeln!(report, "{}: {error}", pins.display());
        }
    }
    let candidates = backend_candidates(runtime);
    let _ = writeln!(
        report,
        "probes inherit the harness environment, not the engine's managed library environment; \
         a probe loader failure does not establish a serving failure"
    );
    if candidates.is_empty() {
        let _ = writeln!(
            report,
            "no llama-server candidates under bin/llamacpp (backend download may not have completed)"
        );
    }
    let deadline = tokio::time::Instant::now() + TOTAL_PROBE_TIMEOUT;
    for server in candidates.iter().take(MAX_CANDIDATES) {
        let _ = writeln!(
            report,
            "candidate (not proof of selection): {}",
            server.display()
        );
        match fingerprint(server) {
            Ok(hash) => {
                let _ = writeln!(report, "sha256: {hash}");
            }
            Err(error) => {
                let _ = writeln!(report, "sha256 unavailable: {error}");
            }
        }
        if let Some(parent) = server.parent() {
            let version = parent.join("version.txt");
            if let Ok((bytes, note)) = read_clamped(&version, MAX_EVIDENCE_BYTES) {
                let _ = writeln!(
                    report,
                    "{}: {}{note}",
                    version.display(),
                    String::from_utf8_lossy(&bytes)
                );
            }
        }
        for flag in ["--version", "--help"] {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                let _ = writeln!(
                    report,
                    "remaining probes skipped: 20s evidence budget exhausted"
                );
                return report;
            }
            let output = probe(server, flag, remaining.min(PROBE_TIMEOUT)).await;
            let _ = writeln!(report, "{flag}:\n{output}");
        }
    }
    if candidates.len() > MAX_CANDIDATES {
        let _ = writeln!(
            report,
            "backend inventory truncated after {MAX_CANDIDATES} candidates"
        );
    }
    report
}

fn backend_candidates(runtime: &Path) -> Vec<PathBuf> {
    let binary = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let mut candidates = Vec::new();
    for backend in ["rocm-stable", "rocm-nightly", "rocm", "vulkan"] {
        let dir = runtime.join("bin/llamacpp").join(backend);
        let direct = dir.join(binary);
        if direct.is_file() {
            candidates.push(direct);
        }
        // Lemonade archives also unpack one level below the backend directory.
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut nested: Vec<_> = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path().join(binary))
                .filter(|path| path.is_file())
                .collect();
            nested.sort();
            candidates.extend(nested);
        }
    }
    candidates
}

fn fingerprint(server: &Path) -> std::io::Result<String> {
    let file = std::fs::File::open(server)?;
    // A real server is much smaller; do not hash an arbitrarily large corrupt
    // cache entry while reporting a failure. Never label a partial hash as SHA256.
    let limit = 512 * 1024 * 1024;
    let mut hasher = Sha256::new();
    let count = std::io::copy(&mut file.take(limit + 1), &mut hasher)?;
    if count > limit {
        return Err(std::io::Error::other(
            "executable exceeds 512 MiB hash limit",
        ));
    }
    Ok(format!("{:x}", hasher.finalize()))
}

async fn probe(server: &Path, flag: &str, timeout: Duration) -> String {
    let mut command = tokio::process::Command::new(server);
    command
        .arg(flag)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => return format!("probe could not start: {error}"),
    };
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let mut out = Vec::new();
    let mut err = Vec::new();
    let status = if let Ok((status, stdout_result, stderr_result)) =
        tokio::time::timeout(timeout, async {
            tokio::join!(
                child.wait(),
                drain_output(stdout, &mut out),
                drain_output(stderr, &mut err),
            )
        })
        .await
    {
        format!(
            "{}\nstdout capture: {stdout_result:?}; stderr capture: {stderr_result:?}",
            status.map_or_else(|error| error.to_string(), |status| status.to_string())
        )
    } else {
        // kill_on_drop also covers cancellation of the whole reporter.
        // Explicitly wait here so a normal timeout leaves no zombie child.
        let result = child.kill().await;
        format!(
            "probe timed out after {}s; kill/reap: {result:?}",
            timeout.as_secs_f64()
        )
    };
    format!(
        "{status}\nstdout:\n{}\nstderr:\n{}{}",
        String::from_utf8_lossy(&out),
        String::from_utf8_lossy(&err),
        if out.len() == MAX_EVIDENCE_BYTES as usize / 2
            || err.len() == MAX_EVIDENCE_BYTES as usize / 2
        {
            "\n<probe output capped at 64 KiB>"
        } else {
            ""
        }
    )
}

async fn drain_output(
    mut reader: impl AsyncRead + Unpin,
    retained: &mut Vec<u8>,
) -> std::io::Result<()> {
    let limit = MAX_EVIDENCE_BYTES as usize / 2;
    let mut buffer = [0; 4096];
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 {
            return Ok(());
        }
        // Continue draining both pipes after the cap, without retaining more
        // bytes or writing them to disk. A noisy child must not block on output.
        let keep = count.min(limit.saturating_sub(retained.len()));
        retained.extend_from_slice(&buffer[..keep]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_identity_does_not_depend_on_the_dynamic_loader() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("not-executable");
        std::fs::write(&server, b"abc").unwrap();
        assert_eq!(
            fingerprint(&server).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn inventory_covers_flat_and_nested_gpu_backends_not_cpu() {
        let root = tempfile::tempdir().unwrap();
        let binary = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let mut wanted = Vec::new();
        for relative in ["rocm-stable", "rocm-nightly/llama-b9752", "vulkan/build"] {
            let server = root.path().join("bin/llamacpp").join(relative).join(binary);
            std::fs::create_dir_all(server.parent().unwrap()).unwrap();
            std::fs::write(&server, "").unwrap();
            wanted.push(server);
        }
        let cpu = root.path().join("bin/llamacpp/cpu").join(binary);
        std::fs::create_dir_all(cpu.parent().unwrap()).unwrap();
        std::fs::write(cpu, "").unwrap();
        assert_eq!(backend_candidates(root.path()), wanted);
    }

    #[tokio::test]
    async fn missing_manifest_is_reported_without_panicking() {
        let root = tempfile::tempdir().unwrap();
        let report = backend_evidence(&root.path().join("missing.json")).await;
        assert!(report.contains("unavailable:"), "{report}");
    }

    #[tokio::test]
    async fn failed_download_keeps_pins_but_does_not_invent_an_installed_build() {
        let root = tempfile::tempdir().unwrap();
        let runtime = root.path().join("runtime");
        std::fs::create_dir_all(runtime.join("resources")).unwrap();
        std::fs::write(
            runtime.join("resources/backend_versions.json"),
            r#"{"llamacpp":{"rocm-stable":"b9752"}}"#,
        )
        .unwrap();
        let manifest = root.path().join("runtime.json");
        std::fs::write(
            &manifest,
            serde_json::json!({
                "version": "fixture",
                "runtime_dir": runtime,
                "backend_name": "rocm"
            })
            .to_string(),
        )
        .unwrap();
        let report = backend_evidence(&manifest).await;
        assert!(
            report.contains("configured pins (not executable identity):"),
            "{report}"
        );
        assert!(report.contains("\"rocm-stable\":\"b9752\""), "{report}");
        assert!(report.contains("no llama-server candidates"), "{report}");
    }

    #[tokio::test]
    async fn probe_start_error_is_evidence_not_a_second_failure() {
        let root = tempfile::tempdir().unwrap();
        let report = probe(&root.path().join("missing"), "--version", PROBE_TIMEOUT).await;
        assert!(report.contains("probe could not start:"), "{report}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn inventory_probes_the_cached_executable_not_a_path_lookup() {
        let root = tempfile::tempdir().unwrap();
        let runtime = root.path().join("runtime");
        let dir = runtime.join("bin/llamacpp/rocm-stable");
        std::fs::create_dir_all(&dir).unwrap();
        let server = dir.join("llama-server");
        rocm_core::fake_executable::write_fake_executable(
            &server,
            "\
#!/bin/sh
case \"$1\" in
  --version) echo 'version: 9752 (fixture-build)' ;;
  --help) echo '--mmap, --no-mmap'; echo 'fixture help stderr' >&2; exit 2 ;;
  *) exit 99 ;;
esac
",
        )
        .unwrap();
        std::fs::write(dir.join("version.txt"), "cached-build").unwrap();
        let manifest = root.path().join("runtime.json");
        std::fs::write(
            &manifest,
            serde_json::json!({
                "version": "fixture",
                "runtime_dir": runtime,
                "backend_name": "rocm"
            })
            .to_string(),
        )
        .unwrap();
        let report = backend_evidence(&manifest).await;
        assert!(report.contains(&server.display().to_string()), "{report}");
        assert!(report.contains("version: 9752 (fixture-build)"), "{report}");
        assert!(report.contains("version.txt: cached-build"), "{report}");
        assert!(report.contains("--help:\nexit status: 2"), "{report}");
        assert!(report.contains("--mmap, --no-mmap"), "{report}");
        assert!(report.contains("fixture help stderr"), "{report}");
        assert!(
            server.is_file(),
            "diagnostics must not repair or remove the cache"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn probe_timeout_is_bounded() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("llama-server");
        // exec keeps the sleeper in the child's PID, so cancellation kills it
        // rather than leaving a grandchild behind.
        rocm_core::fake_executable::write_fake_executable(
            &server,
            "#!/bin/sh\necho $$\nexec /bin/sleep 30\n",
        )
        .unwrap();
        let start = std::time::Instant::now();
        let report = probe(&server, "--help", Duration::from_millis(50)).await;
        assert!(report.contains("probe timed out after 0.05s"), "{report}");
        assert!(start.elapsed() < Duration::from_secs(5));
        let pid = report
            .split("stdout:\n")
            .nth(1)
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .parse::<u32>()
            .unwrap();
        assert!(!rocm_core::process_is_running(pid), "{report}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn noisy_probe_output_is_capped() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("llama-server");
        rocm_core::fake_executable::write_fake_executable(
            &server,
            "#!/bin/sh\nhead -c 100000 /dev/zero\n",
        )
        .unwrap();
        let report = probe(&server, "--help", PROBE_TIMEOUT).await;
        assert!(
            report.ends_with("<probe output capped at 64 KiB>"),
            "{report}"
        );
        assert!(report.len() < MAX_EVIDENCE_BYTES as usize + 200);
    }
}
