// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Read-only backend evidence for a failed Lemonade serve, before teardown.

use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt as _};

use crate::serve_log::read_clamped;

const MAX_EVIDENCE_BYTES: u64 = 64 * 1024;
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CANDIDATES: usize = 8;
const TOTAL_EVIDENCE_BUDGET: Duration = Duration::from_secs(20);
const MAX_HASH_BYTES: u64 = 32 * 1024 * 1024;

/// Record the installed version and pins separately from executable build output:
/// a pin describes what should be installed, not what a runner actually cached.
///
/// Every candidate is labelled as such; finding a binary does not prove that the
/// router launched it. No install, download, model load, or cache repair is done.
pub async fn backend_evidence(manifest_path: &Path) -> String {
    backend_evidence_with_budget(manifest_path, TOTAL_EVIDENCE_BUDGET).await
}

async fn backend_evidence_with_budget(manifest_path: &Path, budget: Duration) -> String {
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
    let deadline = Instant::now() + budget;
    for server in candidates.iter().take(MAX_CANDIDATES) {
        if Instant::now() >= deadline {
            let _ = writeln!(
                report,
                "remaining evidence skipped: evidence budget exhausted"
            );
            return report;
        }
        let _ = writeln!(
            report,
            "candidate (not proof of selection): {}",
            server.display()
        );
        let path = server.clone();
        match tokio::task::spawn_blocking(move || fingerprint(&path, deadline))
            .await
            .map_err(std::io::Error::other)
            .and_then(|result| result)
        {
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
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let _ = writeln!(
                    report,
                    "remaining evidence skipped: evidence budget exhausted"
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

fn fingerprint(server: &Path, deadline: Instant) -> std::io::Result<String> {
    let mut file = std::fs::File::open(server)?;
    // A real server is much smaller; do not hash an arbitrarily large corrupt
    // cache entry while reporting a failure. Never label a partial hash as SHA256.
    if file.metadata()?.len() > MAX_HASH_BYTES {
        return Err(std::io::Error::other(
            "executable exceeds 32 MiB hash limit",
        ));
    }
    let mut hasher = Sha256::new();
    let mut count = 0;
    let mut buffer = vec![0; 64 * 1024];
    loop {
        // Check inside the worker, not just around its JoinHandle: timing out a
        // spawn_blocking await alone would leave the hashing thread running.
        if Instant::now() >= deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "hash evidence budget exhausted",
            ));
        }
        let read = file.read(&mut buffer)?;
        count += read as u64;
        if count > MAX_HASH_BYTES {
            return Err(std::io::Error::other(
                "executable exceeds 32 MiB hash limit",
            ));
        }
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Default)]
struct CapturedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

impl CapturedOutput {
    fn truncation_note(&self, stream: &str) -> String {
        if self.truncated {
            format!("\n<{stream} truncated at 32 KiB>")
        } else {
            String::new()
        }
    }
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
    let mut out = CapturedOutput::default();
    let mut err = CapturedOutput::default();
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
        let mut status = status.map_or_else(|error| error.to_string(), |status| status.to_string());
        for (stream, result) in [("stdout", stdout_result), ("stderr", stderr_result)] {
            if let Err(error) = result {
                let _ = write!(status, "\n{stream} capture failed: {error}");
            }
        }
        status
    } else {
        // kill_on_drop also covers cancellation of the whole reporter.
        // Explicitly wait here so a normal timeout leaves no zombie child.
        let mut status = format!("probe timed out after {}s", timeout.as_secs_f64());
        if let Err(error) = child.kill().await {
            let _ = write!(status, "; kill/reap failed: {error}");
        }
        status
    };
    format!(
        "{status}\nstdout:\n{}{}\nstderr:\n{}{}",
        String::from_utf8_lossy(&out.bytes),
        out.truncation_note("stdout"),
        String::from_utf8_lossy(&err.bytes),
        err.truncation_note("stderr"),
    )
}

async fn drain_output(
    mut reader: impl AsyncRead + Unpin,
    retained: &mut CapturedOutput,
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
        let keep = count.min(limit.saturating_sub(retained.bytes.len()));
        retained.bytes.extend_from_slice(&buffer[..keep]);
        retained.truncated |= keep < count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_manifest(runtime: &Path) -> PathBuf {
        let manifest = runtime.join("runtime.json");
        std::fs::write(
            &manifest,
            serde_json::json!({"runtime_dir": runtime}).to_string(),
        )
        .unwrap();
        manifest
    }

    #[test]
    fn executable_identity_does_not_depend_on_the_dynamic_loader() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("not-executable");
        std::fs::write(&server, b"abc").unwrap();
        assert_eq!(
            fingerprint(&server, Instant::now() + TOTAL_EVIDENCE_BUDGET).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hash_limit_accepts_the_boundary_but_never_returns_a_partial_hash() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("llama-server");
        let file = std::fs::File::create(&server).unwrap();
        file.set_len(MAX_HASH_BYTES).unwrap();
        assert_eq!(
            fingerprint(&server, Instant::now() + TOTAL_EVIDENCE_BUDGET)
                .unwrap()
                .len(),
            64
        );
        file.set_len(MAX_HASH_BYTES + 1).unwrap();
        assert_eq!(
            fingerprint(&server, Instant::now() + TOTAL_EVIDENCE_BUDGET)
                .unwrap_err()
                .to_string(),
            "executable exceeds 32 MiB hash limit"
        );
    }

    #[test]
    fn hash_worker_obeys_its_own_deadline() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("llama-server");
        std::fs::write(&server, b"abc").unwrap();
        assert_eq!(
            fingerprint(&server, Instant::now()).unwrap_err().kind(),
            std::io::ErrorKind::TimedOut
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
    async fn evidence_truncates_candidates_without_hashing_or_probing_the_rest() {
        let root = tempfile::tempdir().unwrap();
        let binary = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        for i in 0..=MAX_CANDIDATES {
            let server = root
                .path()
                .join(format!("bin/llamacpp/rocm-stable/{i:02}/{binary}"));
            std::fs::create_dir_all(server.parent().unwrap()).unwrap();
            std::fs::write(server, "").unwrap();
        }
        let report = backend_evidence(&write_manifest(root.path())).await;
        assert_eq!(
            report
                .matches("candidate (not proof of selection):")
                .count(),
            MAX_CANDIDATES,
            "{report}"
        );
        assert_eq!(report.matches("sha256:").count(), MAX_CANDIDATES);
        assert_eq!(report.matches("--help:").count(), MAX_CANDIDATES);
        assert!(report.contains("backend inventory truncated after 8 candidates"));
        let skipped = root
            .path()
            .join(format!("bin/llamacpp/rocm-stable/08/{binary}"));
        assert!(!report.contains(&skipped.display().to_string()), "{report}");
    }

    #[tokio::test]
    async fn exhausted_budget_skips_hashing_and_probes() {
        let root = tempfile::tempdir().unwrap();
        let binary = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let server = root.path().join("bin/llamacpp/rocm-stable").join(binary);
        std::fs::create_dir_all(server.parent().unwrap()).unwrap();
        std::fs::write(server, "").unwrap();
        let report =
            backend_evidence_with_budget(&write_manifest(root.path()), Duration::ZERO).await;
        assert!(report.contains("remaining evidence skipped: evidence budget exhausted"));
        for skipped in ["sha256", "--version:", "--help:", "candidate ("] {
            assert!(!report.contains(skipped), "{report}");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timed_out_probe_leaves_no_budget_for_the_next_probe() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("bin/llamacpp/rocm-stable/llama-server");
        std::fs::create_dir_all(server.parent().unwrap()).unwrap();
        rocm_core::fake_executable::write_fake_executable(
            &server,
            "#!/bin/sh\nexec /bin/sleep 30\n",
        )
        .unwrap();
        let report =
            backend_evidence_with_budget(&write_manifest(root.path()), Duration::from_secs(1))
                .await;
        assert!(report.contains("--version:\nprobe timed out"), "{report}");
        assert!(report.contains("remaining evidence skipped: evidence budget exhausted"));
        assert!(!report.contains("--help:"), "{report}");
    }

    #[tokio::test]
    async fn stream_capture_marks_only_bytes_actually_discarded() {
        let limit = MAX_EVIDENCE_BYTES as usize / 2;
        for length in [0, limit - 1, limit, limit + 1] {
            let bytes = vec![b'x'; length];
            let mut captured = CapturedOutput::default();
            drain_output(bytes.as_slice(), &mut captured).await.unwrap();
            assert_eq!(captured.bytes, vec![b'x'; length.min(limit)]);
            assert_eq!(captured.truncated, length > limit);
            assert_eq!(
                captured.truncation_note("stdout"),
                if length > limit {
                    "\n<stdout truncated at 32 KiB>"
                } else {
                    ""
                }
            );
        }
    }

    #[tokio::test]
    async fn stream_truncation_survives_timeout_before_eof() {
        use tokio::io::AsyncWriteExt as _;
        let length = MAX_EVIDENCE_BYTES as usize / 2 + 1;
        let (reader, mut writer) = tokio::io::duplex(length);
        writer.write_all(&vec![b'x'; length]).await.unwrap();
        let mut captured = CapturedOutput::default();
        assert!(
            tokio::time::timeout(
                Duration::from_millis(50),
                drain_output(reader, &mut captured)
            )
            .await
            .is_err()
        );
        assert_eq!(captured.bytes.len(), length - 1);
        assert_eq!(
            captured.truncation_note("stderr"),
            "\n<stderr truncated at 32 KiB>"
        );
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
        assert!(
            report.contains("--help:\nexit status: 2\nstdout:"),
            "{report}"
        );
        assert!(!report.contains("capture"), "{report}");
        assert!(!report.contains("Ok("), "{report}");
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
        let report = probe(&server, "--help", Duration::from_secs(1)).await;
        assert!(
            report.contains("probe timed out after 1s\nstdout:"),
            "{report}"
        );
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
    async fn exact_stream_limit_is_not_reported_as_truncated() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("llama-server");
        rocm_core::fake_executable::write_fake_executable(
            &server,
            "#!/bin/sh\nhead -c 32768 /dev/zero\n",
        )
        .unwrap();
        let report = probe(&server, "--help", PROBE_TIMEOUT).await;
        assert!(report.contains("exit status: 0"), "{report}");
        assert_eq!(report.bytes().filter(|byte| *byte == 0).count(), 32768);
        assert!(!report.contains("capped"), "{report}");
        assert!(!report.contains("truncated"), "{report}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn noisy_probe_output_is_capped() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("llama-server");
        rocm_core::fake_executable::write_fake_executable(
            &server,
            "#!/bin/sh\nhead -c 100000 /dev/zero\nhead -c 100000 /dev/zero >&2\n",
        )
        .unwrap();
        let report = probe(&server, "--help", PROBE_TIMEOUT).await;
        assert!(report.contains("<stdout truncated at 32 KiB>"), "{report}");
        assert!(report.ends_with("<stderr truncated at 32 KiB>"), "{report}");
        assert_eq!(
            report.bytes().filter(|byte| *byte == 0).count(),
            MAX_EVIDENCE_BYTES as usize
        );
        assert!(report.len() < MAX_EVIDENCE_BYTES as usize + 300);
    }
}
