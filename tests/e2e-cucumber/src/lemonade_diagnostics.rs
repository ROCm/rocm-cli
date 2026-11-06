// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Read-only backend evidence for a failed Lemonade serve.
//!
//! The evidence is taken in two steps with the service stop between them (see
//! [`backend_evidence_around_stop`]): what is only read from disk (manifest,
//! pins, candidate paths, fingerprints) is collected while the failed service is
//! still in the state being diagnosed, and the `llama-server` probes, which
//! execute a backend binary, run after it has been stopped.

use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt as _};

use crate::serve_log::{ENGINE_UNKNOWN_EVIDENCE, read_clamped};

const MAX_EVIDENCE_BYTES: u64 = 64 * 1024;
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CANDIDATES: usize = 8;
/// Time for the read-only inventory (pins, candidate paths, fingerprints),
/// taken before the service is stopped.
const INVENTORY_BUDGET: Duration = Duration::from_secs(20);
/// Time for the probes, started after the stop returns so a slow stop does not
/// use it up.
const PROBE_BUDGET: Duration = Duration::from_secs(20);
const MAX_HASH_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Budgets {
    inventory: Duration,
    probes: Duration,
}

const DEFAULT_BUDGETS: Budgets = Budgets {
    inventory: INVENTORY_BUDGET,
    probes: PROBE_BUDGET,
};

/// Runs one `llama-server` flag and renders what came back. A trait so the order
/// of probes against the service stop is testable without executing a binary.
trait Probe {
    async fn run(&self, server: &Path, flag: &str, timeout: Duration) -> String;
}

struct RealProbe;

impl Probe for RealProbe {
    async fn run(&self, server: &Path, flag: &str, timeout: Duration) -> String {
        probe(server, flag, timeout).await
    }
}

/// Collect the backend evidence for a failed serve of `engine` around the call
/// that stops the service, and return the stop's own status with it.
///
/// `stop` always runs exactly once, whatever the engine, so a caller cannot
/// forget it or run it twice. For Lemonade the order is fixed: the read-only
/// inventory first, then `stop`, then the probes. A probe executes a backend
/// binary, and llama.cpp builds commonly initialise a GPU backend even for
/// `--version`, so running one next to a still-live router on a shared-memory
/// device could change the very state being diagnosed. Reading files cannot.
/// `manifest_path` is only called for Lemonade.
///
/// The report says `engine unknown` when no engine was determined, so a skipped
/// inventory stays visible, and returns no evidence for another engine.
pub async fn backend_evidence_around_stop(
    manifest_path: impl FnOnce() -> PathBuf,
    engine: Option<&str>,
    stop: impl FnOnce() -> String,
) -> (String, Option<String>) {
    match engine {
        Some("lemonade") => {
            let (stop_status, evidence) =
                evidence_around_stop(&manifest_path(), DEFAULT_BUDGETS, &RealProbe, stop).await;
            (stop_status, Some(evidence))
        }
        Some(_) => (stop(), None),
        None => (stop(), Some(ENGINE_UNKNOWN_EVIDENCE.to_owned())),
    }
}

async fn evidence_around_stop(
    manifest_path: &Path,
    budgets: Budgets,
    probe: &impl Probe,
    stop: impl FnOnce() -> String,
) -> (String, String) {
    let inventory = take_inventory(manifest_path, budgets.inventory).await;
    let stop_status = stop();
    let probes = inventory.probe_all(probe, budgets.probes).await;
    (stop_status, format!("{}{probes}", inventory.report))
}

/// What was read from disk before the stop: the report so far, and the
/// candidates the probes will run against once the service is stopped.
struct Inventory {
    report: String,
    servers: Vec<PathBuf>,
}

impl Inventory {
    fn without_probes(report: String) -> Self {
        Self {
            report,
            servers: Vec::new(),
        }
    }

    async fn probe_all(&self, probe: &impl Probe, budget: Duration) -> String {
        if self.servers.is_empty() {
            return String::new();
        }
        let mut report = String::new();
        let _ = writeln!(
            report,
            "probes, run after the service stop was attempted; they inherit the harness \
             environment, not the engine's managed library environment, so a probe loader \
             failure does not establish a serving failure:"
        );
        let deadline = Instant::now() + budget;
        for server in &self.servers {
            let _ = writeln!(report, "probe target: {}", server.display());
            for flag in ["--version", "--help"] {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    let _ = writeln!(
                        report,
                        "remaining evidence skipped: evidence budget exhausted"
                    );
                    return report;
                }
                let output = probe.run(server, flag, remaining.min(PROBE_TIMEOUT)).await;
                let _ = writeln!(report, "{flag}:\n{output}");
            }
        }
        report
    }
}

/// Record the installed version and pins separately from executable build output:
/// a pin describes what should be installed, not what a runner actually cached.
///
/// Every candidate is labelled as such; finding a binary does not prove that the
/// router launched it. Nothing here executes anything. No install, download,
/// model load, or cache repair is done.
async fn take_inventory(manifest_path: &Path, budget: Duration) -> Inventory {
    let mut report = String::new();
    let _ = writeln!(report, "manifest: {}", manifest_path.display());
    let manifest = match read_clamped(manifest_path, MAX_EVIDENCE_BYTES).and_then(|(bytes, _)| {
        serde_json::from_slice::<serde_json::Value>(&bytes).map_err(std::io::Error::other)
    }) {
        Ok(manifest) => manifest,
        Err(error) => {
            return Inventory::without_probes(format!("{report}unavailable: {error}"));
        }
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
        return Inventory::without_probes(format!(
            "{report}backend inventory unavailable: no runtime_dir"
        ));
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
    if candidates.is_empty() {
        let _ = writeln!(report, "{}", empty_search_note(runtime));
    }
    let deadline = Instant::now() + budget;
    let mut servers = Vec::new();
    for server in candidates.iter().take(MAX_CANDIDATES) {
        if Instant::now() >= deadline {
            let _ = writeln!(
                report,
                "remaining evidence skipped: evidence budget exhausted"
            );
            return Inventory { report, servers };
        }
        let _ = writeln!(
            report,
            "candidate (not proof of selection): {}",
            server.display()
        );
        servers.push(server.clone());
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
    }
    if candidates.len() > MAX_CANDIDATES {
        let _ = writeln!(
            report,
            "backend inventory truncated after {MAX_CANDIDATES} candidates"
        );
    }
    Inventory { report, servers }
}

/// The GPU backend directories under `bin/llamacpp` that are searched. The cpu
/// backend is left out on purpose: a CPU build is never what a GPU serve should
/// have launched, and listing it would read as one.
const BACKEND_DIRS: [&str; 4] = ["rocm-stable", "rocm-nightly", "rocm", "vulkan"];
/// How many directory levels below a backend directory are searched. Lemonade
/// unpacks some llama.cpp builds into a build-numbered directory, and the layout
/// is the router's to choose, so the search is not pinned to one level.
const MAX_SEARCH_DEPTH: usize = 3;
/// Directory entries read per directory, so a corrupt tree cannot make the
/// search unbounded.
const MAX_DIR_ENTRIES: usize = 256;
/// Names listed when explaining an empty search.
const MAX_LISTED_NAMES: usize = 16;

fn server_binary() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

fn backend_candidates(runtime: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for backend in BACKEND_DIRS {
        collect_candidates(
            &runtime.join("bin/llamacpp").join(backend),
            0,
            &mut candidates,
        );
    }
    candidates
}

fn collect_candidates(dir: &Path, depth: usize, candidates: &mut Vec<PathBuf>) {
    let direct = dir.join(server_binary());
    if direct.is_file() {
        candidates.push(direct);
    }
    if depth >= MAX_SEARCH_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subdirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .take(MAX_DIR_ENTRIES)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    subdirs.sort();
    for subdir in subdirs {
        collect_candidates(&subdir, depth + 1, candidates);
    }
}

/// Why a search found nothing, in terms of where it looked. An absent binary can
/// mean a backend download that did not finish, a layout the search does not
/// cover, or a build that was never installed, and the report cannot tell which,
/// so it says what was searched and what `bin/llamacpp` actually holds.
fn empty_search_note(runtime: &Path) -> String {
    let llamacpp = runtime.join("bin/llamacpp");
    let searched = BACKEND_DIRS
        .iter()
        .map(|backend| llamacpp.join(backend).display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let present = match std::fs::read_dir(&llamacpp) {
        Ok(entries) => {
            let mut names: Vec<String> = entries
                .filter_map(Result::ok)
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            if names.is_empty() {
                format!("{} is empty", llamacpp.display())
            } else {
                let more = names.len().saturating_sub(MAX_LISTED_NAMES);
                names.truncate(MAX_LISTED_NAMES);
                let more = if more > 0 {
                    format!(" and {more} more")
                } else {
                    String::new()
                };
                format!(
                    "{} contains: {}{more}",
                    llamacpp.display(),
                    names.join(", ")
                )
            }
        }
        Err(error) => format!("{} cannot be read: {error}", llamacpp.display()),
    };
    format!(
        "no llama-server candidates found; searched {searched}, down to {MAX_SEARCH_DEPTH} \
         directory levels below each (the cpu backend is not searched); {present}"
    )
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

    async fn evidence(manifest: &Path) -> String {
        evidence_with(manifest, DEFAULT_BUDGETS).await
    }

    async fn evidence_with(manifest: &Path, budgets: Budgets) -> String {
        evidence_around_stop(manifest, budgets, &RealProbe, String::new)
            .await
            .1
    }

    fn write_manifest(runtime: &Path) -> PathBuf {
        let manifest = runtime.join("runtime.json");
        std::fs::write(
            &manifest,
            serde_json::json!({"runtime_dir": runtime}).to_string(),
        )
        .unwrap();
        manifest
    }

    /// The candidate paths a report lists, parsed back into paths.
    ///
    /// Compared as paths, never as printed text: the inventory joins
    /// `bin/llamacpp` as one string, so on Windows a report line mixes `/` and
    /// `\` while a test that builds the expected path another way prints it
    /// differently. `Path` equality compares components, so both spellings agree,
    /// and a negative check ("this candidate was skipped") cannot pass vacuously
    /// just because the two spellings differ.
    fn candidate_paths(report: &str) -> Vec<PathBuf> {
        report
            .lines()
            .filter_map(|line| line.strip_prefix("candidate (not proof of selection): "))
            .map(PathBuf::from)
            .collect()
    }

    #[test]
    fn executable_identity_does_not_depend_on_the_dynamic_loader() {
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("not-executable");
        std::fs::write(&server, b"abc").unwrap();
        assert_eq!(
            fingerprint(&server, Instant::now() + INVENTORY_BUDGET).unwrap(),
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
            fingerprint(&server, Instant::now() + INVENTORY_BUDGET)
                .unwrap()
                .len(),
            64
        );
        file.set_len(MAX_HASH_BYTES + 1).unwrap();
        assert_eq!(
            fingerprint(&server, Instant::now() + INVENTORY_BUDGET)
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

    #[test]
    fn candidate_search_reaches_three_levels_below_a_backend_and_no_further() {
        let root = tempfile::tempdir().unwrap();
        let backend = root.path().join("bin/llamacpp/rocm-nightly");
        // Literal layouts, not derived from MAX_SEARCH_DEPTH: the depth is the
        // behaviour under test, so changing the constant must turn this red.
        let found_at_three = backend.join("a/b/llama-b9752").join(server_binary());
        let found_at_two = backend.join("a/build").join(server_binary());
        let beyond = backend.join("a/b/c/llama-b9752").join(server_binary());
        for server in [&found_at_three, &found_at_two, &beyond] {
            std::fs::create_dir_all(server.parent().unwrap()).unwrap();
            std::fs::write(server, "").unwrap();
        }
        assert_eq!(
            backend_candidates(root.path()),
            // Sorted by path, so `a/b/...` precedes `a/build/...`.
            vec![found_at_three, found_at_two]
        );
    }

    /// An empty search says where it looked and what `bin/llamacpp` holds, and does
    /// not guess that a download was interrupted: the same report follows from a
    /// layout the search does not cover or a build that was never installed.
    #[tokio::test]
    async fn empty_search_lists_the_searched_locations_and_blames_nothing() {
        let root = tempfile::tempdir().unwrap();
        let cpu = root.path().join("bin/llamacpp/cpu");
        std::fs::create_dir_all(&cpu).unwrap();
        std::fs::write(cpu.join(server_binary()), "").unwrap();
        let report = evidence(&write_manifest(root.path())).await;
        let llamacpp = root.path().join("bin/llamacpp");
        for backend in BACKEND_DIRS {
            assert!(
                report.contains(&llamacpp.join(backend).display().to_string()),
                "{report}"
            );
        }
        assert!(
            report.contains("no llama-server candidates found"),
            "{report}"
        );
        assert!(
            report.contains(&format!("{} contains: cpu", llamacpp.display())),
            "{report}"
        );
        assert!(
            !report.contains("download") && !report.contains("complete"),
            "{report}"
        );
        assert!(!report.contains("candidate (not proof"), "{report}");

        let bare = tempfile::tempdir().unwrap();
        let report = evidence(&write_manifest(bare.path())).await;
        assert!(report.contains("cannot be read"), "{report}");
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
        let report = evidence(&write_manifest(root.path())).await;
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
        let listed = candidate_paths(&report);
        assert_eq!(listed.len(), MAX_CANDIDATES, "{report}");
        assert!(!listed.contains(&skipped), "{report}");
        let first = root
            .path()
            .join(format!("bin/llamacpp/rocm-stable/00/{binary}"));
        assert!(
            listed.contains(&first),
            "the comparison must be able to match a listed candidate: {report}"
        );
    }

    /// An executable past the hash cap is not fingerprinted, but the report says
    /// so instead of omitting the line, and still lists the candidate and probes it.
    #[tokio::test]
    async fn oversized_executable_reports_why_it_has_no_fingerprint() {
        let root = tempfile::tempdir().unwrap();
        let binary = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let server = root.path().join("bin/llamacpp/rocm-stable").join(binary);
        std::fs::create_dir_all(server.parent().unwrap()).unwrap();
        std::fs::File::create(&server)
            .unwrap()
            .set_len(MAX_HASH_BYTES + 1)
            .unwrap();
        let report = evidence(&write_manifest(root.path())).await;
        assert!(
            report.contains("sha256 unavailable: executable exceeds 32 MiB hash limit"),
            "{report}"
        );
        assert!(!report.contains("sha256: "), "{report}");
        assert_eq!(candidate_paths(&report), vec![server], "{report}");
        assert!(report.contains("--version:"), "{report}");
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
        let report = evidence_with(
            &write_manifest(root.path()),
            Budgets {
                inventory: Duration::ZERO,
                probes: PROBE_BUDGET,
            },
        )
        .await;
        assert!(report.contains("remaining evidence skipped: evidence budget exhausted"));
        for skipped in ["sha256", "--version:", "--help:", "candidate ("] {
            assert!(!report.contains(skipped), "{report}");
        }
    }

    struct RecordingProbe<'a> {
        events: &'a std::cell::RefCell<Vec<String>>,
    }

    impl Probe for RecordingProbe<'_> {
        async fn run(&self, _server: &Path, flag: &str, _timeout: Duration) -> String {
            self.events.borrow_mut().push(format!("probe {flag}"));
            "recorded, not executed".to_owned()
        }
    }

    /// A lemonade data dir whose manifest sits where the engine writes it, with one
    /// small candidate whose SHA-256 is a known value.
    fn lemonade_data_dir_with_candidate() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let paths = rocm_core::AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let runtime = root.path().join("runtime");
        let binary = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let server = runtime.join("bin/llamacpp/rocm-stable").join(binary);
        std::fs::create_dir_all(server.parent().unwrap()).unwrap();
        std::fs::write(&server, b"abc").unwrap();
        let manifest = paths.engine_runtime_manifest_path("lemonade");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &manifest,
            serde_json::json!({"runtime_dir": runtime}).to_string(),
        )
        .unwrap();
        (root, manifest, server)
    }

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    /// The inventory is read while the failed service is still in place and the
    /// probes, which execute a backend binary, run only after the stop.
    ///
    /// The stop deletes the candidate. A fingerprint in the report therefore proves
    /// the file was read before the stop, and the recorded events prove the probes
    /// came after it; running either step on the wrong side turns this red.
    #[tokio::test]
    async fn inventory_precedes_the_stop_and_probes_follow_it() {
        let (_root, manifest, server) = lemonade_data_dir_with_candidate();
        let events = std::cell::RefCell::new(Vec::new());
        let (stop_status, report) = evidence_around_stop(
            &manifest,
            DEFAULT_BUDGETS,
            &RecordingProbe { events: &events },
            || {
                events.borrow_mut().push("stop".to_owned());
                std::fs::remove_file(&server).unwrap();
                "stopped".to_owned()
            },
        )
        .await;
        assert_eq!(stop_status, "stopped");
        assert_eq!(
            *events.borrow(),
            ["stop", "probe --version", "probe --help"],
            "{report}"
        );
        assert!(
            report.contains(&format!("sha256: {ABC_SHA256}")),
            "{report}"
        );
        assert!(!report.contains("sha256 unavailable"), "{report}");
        let inventory = report.find("candidate (not proof of selection):").unwrap();
        let probes = report.find("probe target:").unwrap();
        assert!(inventory < probes, "{report}");
        assert!(report.contains("recorded, not executed"), "{report}");
    }

    #[tokio::test]
    async fn stop_runs_once_for_every_engine_and_an_unknown_engine_stays_visible() {
        let stops = std::cell::Cell::new(0);
        let stop = || {
            stops.set(stops.get() + 1);
            "stopped".to_owned()
        };
        let no_manifest = || -> PathBuf { panic!("the manifest is only read for lemonade") };

        assert_eq!(
            backend_evidence_around_stop(no_manifest, Some("vllm"), stop).await,
            ("stopped".to_owned(), None)
        );
        assert_eq!(
            backend_evidence_around_stop(no_manifest, None, stop).await,
            (
                "stopped".to_owned(),
                Some(ENGINE_UNKNOWN_EVIDENCE.to_owned())
            )
        );
        assert_eq!(stops.get(), 2);

        let (_root, manifest, _server) = lemonade_data_dir_with_candidate();
        let (status, evidence) =
            backend_evidence_around_stop(|| manifest, Some("lemonade"), stop).await;
        assert_eq!(status, "stopped");
        assert!(evidence.unwrap().contains("manifest: "));
        assert_eq!(stops.get(), 3);
    }

    #[tokio::test]
    async fn an_unreadable_manifest_is_reported_and_the_service_is_still_stopped() {
        let root = tempfile::tempdir().unwrap();
        let events = std::cell::RefCell::new(Vec::new());
        let (stop_status, report) = evidence_around_stop(
            &root.path().join("missing.json"),
            DEFAULT_BUDGETS,
            &RecordingProbe { events: &events },
            || "stopped".to_owned(),
        )
        .await;
        assert_eq!(stop_status, "stopped");
        assert!(report.contains("unavailable:"), "{report}");
        assert!(events.borrow().is_empty(), "no candidate, no probe");
        assert!(!report.contains("probe target:"), "{report}");
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
        let report = evidence_with(
            &write_manifest(root.path()),
            Budgets {
                inventory: INVENTORY_BUDGET,
                probes: Duration::from_secs(1),
            },
        )
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
        let report = evidence(&root.path().join("missing.json")).await;
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
        let report = evidence(&manifest).await;
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
        let report = evidence(&manifest).await;
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
