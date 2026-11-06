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
/// Largest executable that is fingerprinted.
///
/// A fingerprint that is absent on the lane the report is for is a defect, so the
/// cap is set well above a dynamically linked `llama-server` (a few MiB; the HIP
/// kernels live in separate shared libraries). It is still finite: a build that
/// statically linked kernels for several GPU targets can reach hundreds of MiB, and
/// a corrupt cache entry should not be hashed at all. Hashing streams 64 KiB at a
/// time, so memory stays bounded whatever this is; the time is bounded by the
/// inventory budget, which the hashing loop checks between reads.
const MAX_HASH_BYTES: u64 = 512 * 1024 * 1024;

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
trait Probe: Sync {
    fn run(
        &self,
        server: &Path,
        flag: &str,
        timeout: Duration,
    ) -> impl std::future::Future<Output = String> + Send;
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
    manifest_path: impl FnOnce() -> PathBuf + Send,
    engine: Option<&str>,
    stop: impl FnOnce() -> String + Send,
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
    stop: impl FnOnce() -> String + Send,
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
    const fn without_probes(report: String) -> Self {
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

/// Answers "has the inventory's time run out?". Injected so a test can drive the
/// traversal's deadline without sleeping; the real one compares against an
/// [`Instant`]. `Arc` because the scan runs on a blocking thread.
type Expired = std::sync::Arc<dyn Fn() -> bool + Send + Sync>;

/// Record the installed version and pins separately from executable build output:
/// a pin describes what should be installed, not what a runner actually cached.
///
/// Every candidate is labelled as such; finding a binary does not prove that the
/// router launched it. Nothing here executes anything. No install, download,
/// model load, or cache repair is done.
///
/// The budget starts here, at entry, and covers the manifest and pin reads, the
/// directory search and the hashing alike.
async fn take_inventory(manifest_path: &Path, budget: Duration) -> Inventory {
    let deadline = Instant::now() + budget;
    take_inventory_with(
        manifest_path,
        deadline,
        SEARCH_LIMITS,
        std::sync::Arc::new(move || Instant::now() >= deadline),
    )
    .await
}

/// What [`scan_runtime`] read from disk.
enum Scan {
    /// The manifest could not be used; the report says why.
    Unavailable(String),
    Found {
        /// Manifest fields and pins.
        report: String,
        discovery: Discovery,
        /// Where an empty search looked, when it found nothing and was not cut short
        /// by the clock.
        empty_note: Option<String>,
    },
}

/// Everything the inventory reads before it hashes anything: the manifest, the
/// pins and the directory search. Blocking filesystem work, so it runs on a
/// blocking thread; it carries its own clock because a `timeout` around
/// `spawn_blocking` would only drop the join handle and leave this running.
fn scan_runtime(manifest_path: &Path, limits: SearchLimits, expired: &dyn Fn() -> bool) -> Scan {
    let mut report = String::new();
    let _ = writeln!(report, "manifest: {}", manifest_path.display());
    let manifest = match read_clamped(manifest_path, MAX_EVIDENCE_BYTES).and_then(|(bytes, _)| {
        serde_json::from_slice::<serde_json::Value>(&bytes).map_err(std::io::Error::other)
    }) {
        Ok(manifest) => manifest,
        Err(error) => return Scan::Unavailable(format!("{report}unavailable: {error}")),
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
        return Scan::Unavailable(format!(
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
    let discovery = discover_candidates(runtime, limits, expired);
    let empty_note = (discovery.candidates.is_empty()
        && !matches!(discovery.stop, Some(Stop::Deadline)))
    .then(|| empty_search_note(runtime));
    Scan::Found {
        report,
        discovery,
        empty_note,
    }
}

/// One candidate's fingerprint and `version.txt`, rendered. Blocking reads, run on
/// a blocking thread with the inventory's deadline checked between hash reads.
fn inspect_candidate(server: &Path, deadline: Instant) -> String {
    let mut report = String::new();
    match fingerprint(server, deadline) {
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
    report
}

async fn take_inventory_with(
    manifest_path: &Path,
    deadline: Instant,
    limits: SearchLimits,
    expired: Expired,
) -> Inventory {
    let scan_path = manifest_path.to_owned();
    let scan_clock = expired.clone();
    let scan =
        tokio::task::spawn_blocking(move || scan_runtime(&scan_path, limits, scan_clock.as_ref()))
            .await;
    let (mut report, discovery, empty_note) = match scan {
        Ok(Scan::Found {
            report,
            discovery,
            empty_note,
        }) => (report, discovery, empty_note),
        Ok(Scan::Unavailable(report)) => return Inventory::without_probes(report),
        Err(error) => {
            return Inventory::without_probes(format!(
                "manifest: {}\nunavailable: inventory scan failed: {error}",
                manifest_path.display()
            ));
        }
    };
    if let Some(note) = empty_note {
        let _ = writeln!(report, "{note}");
    }
    match discovery.stop {
        Some(Stop::Deadline) => {
            let _ = writeln!(
                report,
                "remaining evidence skipped: evidence budget exhausted"
            );
        }
        Some(Stop::Directories) => {
            let _ = writeln!(
                report,
                "candidate search stopped: {} directories visited (limit)",
                discovery.dirs_visited
            );
        }
        Some(Stop::Entries) => {
            let _ = writeln!(
                report,
                "candidate search stopped: {} directory entries read (limit)",
                discovery.entries_read
            );
        }
        Some(Stop::Candidates) | None => {}
    }
    if discovery.symlinks_skipped > 0 {
        let _ = writeln!(
            report,
            "{} symlinked entries under the backend directories were not followed",
            discovery.symlinks_skipped
        );
    }
    let mut servers = Vec::new();
    for server in &discovery.candidates {
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
        match tokio::task::spawn_blocking(move || inspect_candidate(&path, deadline)).await {
            Ok(lines) => report.push_str(&lines),
            Err(error) => {
                let _ = writeln!(report, "sha256 unavailable: {error}");
            }
        }
    }
    if matches!(discovery.stop, Some(Stop::Candidates)) {
        let _ = writeln!(
            report,
            "backend inventory truncated after {MAX_CANDIDATES} candidates \
             (the search stops there; more may exist)"
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
/// Names listed when explaining an empty search.
const MAX_LISTED_NAMES: usize = 16;
/// Directory entries read to build that listing. Bounded so a huge `bin/llamacpp`
/// cannot make the explanation of an empty search expensive; the report says how
/// many names it left out.
const MAX_LISTING_SCAN: usize = 64;

/// Global bounds on one candidate search. They apply to the whole search, across
/// every backend directory, not per directory: a tree of many shallow directories
/// is as expensive as a deep one.
#[derive(Clone, Copy)]
struct SearchLimits {
    /// Directories whose entries are read.
    dirs: usize,
    /// Directory entries read, in total.
    entries: usize,
    /// Candidates collected; the search stops as soon as it has this many.
    candidates: usize,
}

const SEARCH_LIMITS: SearchLimits = SearchLimits {
    dirs: 128,
    entries: 2048,
    candidates: MAX_CANDIDATES,
};

/// Why a candidate search ended before it had looked everywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    Deadline,
    Directories,
    Entries,
    Candidates,
}

#[derive(Debug, Default)]
struct Discovery {
    /// In a fixed order: backend directories as listed, then sibling directories
    /// sorted by path.
    candidates: Vec<PathBuf>,
    stop: Option<Stop>,
    /// Symlinks met and not followed. A cache that links out of its backend
    /// directory, or back into it, is neither searched nor allowed to loop the
    /// search.
    symlinks_skipped: usize,
    dirs_visited: usize,
    entries_read: usize,
}

const fn server_binary() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

/// Look for `llama-server` under each backend directory, within `limits` and until
/// `expired` says time is up. Symlinks are never followed: only real directories
/// are entered and only a regular file counts as a candidate, so a link cycle
/// ends and a link out of the tree is not read.
fn discover_candidates(
    runtime: &Path,
    limits: SearchLimits,
    expired: &dyn Fn() -> bool,
) -> Discovery {
    let mut search = Search {
        limits,
        expired,
        out: Discovery::default(),
    };
    for backend in BACKEND_DIRS {
        let dir = runtime.join("bin/llamacpp").join(backend);
        match std::fs::symlink_metadata(&dir) {
            Ok(meta) if meta.is_dir() => search.visit(&dir, 0),
            Ok(meta) if meta.file_type().is_symlink() => search.out.symlinks_skipped += 1,
            _ => {}
        }
        if search.out.stop.is_some() {
            break;
        }
    }
    search.out
}

struct Search<'a> {
    limits: SearchLimits,
    expired: &'a dyn Fn() -> bool,
    out: Discovery,
}

impl Search<'_> {
    fn stop(&mut self, why: Stop) {
        self.out.stop.get_or_insert(why);
    }

    fn visit(&mut self, dir: &Path, depth: usize) {
        if self.out.stop.is_some() {
            return;
        }
        if (self.expired)() {
            self.stop(Stop::Deadline);
            return;
        }
        if self.out.dirs_visited >= self.limits.dirs {
            self.stop(Stop::Directories);
            return;
        }
        self.out.dirs_visited += 1;
        let direct = dir.join(server_binary());
        // A symlink named llama-server is not a candidate (and is counted below,
        // as a directory entry, not here).
        if std::fs::symlink_metadata(&direct).is_ok_and(|meta| meta.is_file()) {
            self.out.candidates.push(direct);
        }
        if self.out.candidates.len() >= self.limits.candidates {
            self.stop(Stop::Candidates);
            return;
        }
        if depth >= MAX_SEARCH_DEPTH {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut subdirs = Vec::new();
        for entry in entries {
            if (self.expired)() {
                self.stop(Stop::Deadline);
                break;
            }
            if self.out.entries_read >= self.limits.entries {
                self.stop(Stop::Entries);
                break;
            }
            self.out.entries_read += 1;
            let Ok(entry) = entry else {
                continue;
            };
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => subdirs.push(entry.path()),
                Ok(kind) if kind.is_symlink() => self.out.symlinks_skipped += 1,
                _ => {}
            }
        }
        subdirs.sort();
        for subdir in subdirs {
            self.visit(&subdir, depth + 1);
        }
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
                .take(MAX_LISTING_SCAN)
                .filter_map(Result::ok)
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect();
            let hit_scan_cap = names.len() >= MAX_LISTING_SCAN;
            names.sort();
            if names.is_empty() {
                format!("{} is empty", llamacpp.display())
            } else {
                let omitted = names.len().saturating_sub(MAX_LISTED_NAMES);
                names.truncate(MAX_LISTED_NAMES);
                let more = match (omitted, hit_scan_cap) {
                    (0, false) => String::new(),
                    (_, false) => format!(" and {omitted} more"),
                    (_, true) => format!(
                        " and at least {omitted} more (only the first {MAX_LISTING_SCAN} \
                         entries were read)"
                    ),
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

/// `limit` as the report words it: whole MiB when it is one, else bytes.
fn describe_hash_limit(limit: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    if limit.is_multiple_of(MIB) {
        format!("{} MiB", limit / MIB)
    } else {
        format!("{limit} bytes")
    }
}

fn fingerprint(server: &Path, deadline: Instant) -> std::io::Result<String> {
    fingerprint_with_limit(server, deadline, MAX_HASH_BYTES)
}

fn fingerprint_with_limit(server: &Path, deadline: Instant, limit: u64) -> std::io::Result<String> {
    let too_large = || {
        std::io::Error::other(format!(
            "executable exceeds {} hash limit",
            describe_hash_limit(limit)
        ))
    };
    let mut file = std::fs::File::open(server)?;
    // Do not hash an arbitrarily large corrupt cache entry while reporting a
    // failure. Never label a partial hash as SHA256.
    if file.metadata()?.len() > limit {
        return Err(too_large());
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
        if count > limit {
            return Err(too_large());
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
    command.arg(flag);
    run_probe(command, timeout).await
}

/// Run `command` with bounded time and output, and render what came back.
///
/// Split from [`probe`] so the timeout, kill and reap path can be driven with any
/// child, not only a `llama-server` stand-in: shell-script stand-ins exist on unix
/// only, and this path is what must also hold on Windows.
async fn run_probe(mut command: tokio::process::Command, timeout: Duration) -> String {
    command
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

    fn candidates(runtime: &Path) -> Vec<PathBuf> {
        discover_candidates(runtime, SEARCH_LIMITS, &|| false).candidates
    }

    fn touch_server(dir: &Path) -> PathBuf {
        let server = dir.join(server_binary());
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(&server, "").unwrap();
        server
    }

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
        let limit = 3 * 64 * 1024 + 5;
        let file = std::fs::File::create(&server).unwrap();
        file.set_len(limit).unwrap();
        assert_eq!(
            fingerprint_with_limit(&server, Instant::now() + INVENTORY_BUDGET, limit)
                .unwrap()
                .len(),
            64
        );
        file.set_len(limit + 1).unwrap();
        assert_eq!(
            fingerprint_with_limit(&server, Instant::now() + INVENTORY_BUDGET, limit)
                .unwrap_err()
                .to_string(),
            format!("executable exceeds {limit} bytes hash limit")
        );
    }

    /// The cap is not the old 32 MiB: a server above it, such as a static HIP build,
    /// must still get a fingerprint on the lane this report is for. The file is
    /// sparse, so the test hashes zeros rather than holding real data.
    #[test]
    fn a_server_above_the_old_32_mib_cap_is_still_fingerprinted() {
        const OLD_CAP: u64 = 32 * 1024 * 1024;
        const { assert!(MAX_HASH_BYTES >= 256 * 1024 * 1024) };
        let root = tempfile::tempdir().unwrap();
        let server = root.path().join("llama-server");
        std::fs::File::create(&server)
            .unwrap()
            .set_len(OLD_CAP + 1)
            .unwrap();
        assert_eq!(
            fingerprint(&server, Instant::now() + INVENTORY_BUDGET)
                .unwrap()
                .len(),
            64
        );
    }

    #[test]
    fn hash_limit_is_worded_in_the_unit_it_was_set_in() {
        assert_eq!(describe_hash_limit(512 * 1024 * 1024), "512 MiB");
        assert_eq!(describe_hash_limit(1000), "1000 bytes");
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
        assert_eq!(candidates(root.path()), wanted);
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
            candidates(root.path()),
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

    fn limits(dirs: usize, entries: usize) -> SearchLimits {
        SearchLimits {
            dirs,
            entries,
            candidates: MAX_CANDIDATES,
        }
    }

    /// The clock is read while the tree is walked, not once before it: it reports
    /// time up from its third reading on, so the walk must end partway through,
    /// with what it had already found and the reason it stopped.
    #[test]
    fn the_clock_is_checked_during_the_walk() {
        let root = tempfile::tempdir().unwrap();
        let wanted: Vec<PathBuf> = BACKEND_DIRS
            .iter()
            .map(|backend| touch_server(&root.path().join("bin/llamacpp").join(backend)))
            .collect();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let found = discover_candidates(root.path(), SEARCH_LIMITS, &|| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= 2
        });
        assert_eq!(found.stop, Some(Stop::Deadline));
        assert_eq!(found.candidates, wanted[..1], "{found:?}");
        let all = discover_candidates(root.path(), SEARCH_LIMITS, &|| false);
        assert_eq!(all.candidates, wanted);
        assert_eq!(all.stop, None);
    }

    /// The same, through the inventory: a clock that is already up stops the search
    /// even though the hashing deadline is far off, so the walk consults the
    /// inventory's clock and not only the hash loop's.
    #[tokio::test]
    async fn an_inventory_whose_time_is_up_skips_the_search() {
        let root = tempfile::tempdir().unwrap();
        touch_server(&root.path().join("bin/llamacpp/rocm-stable"));
        let inventory = take_inventory_with(
            &write_manifest(root.path()),
            Instant::now() + INVENTORY_BUDGET,
            SEARCH_LIMITS,
            std::sync::Arc::new(|| true),
        )
        .await;
        let report = &inventory.report;
        assert!(
            report.contains("remaining evidence skipped: evidence budget exhausted"),
            "{report}"
        );
        assert!(!report.contains("candidate (not proof"), "{report}");
        assert!(
            !report.contains("no llama-server candidates found"),
            "an unfinished search must not claim to have found nothing: {report}"
        );
        assert!(inventory.servers.is_empty());
    }

    /// The scan is blocking filesystem work and must not run on the async executor.
    /// On a current-thread runtime the test's own thread is the executor, so a clock
    /// read from any other thread shows the walk ran on a blocking one.
    #[tokio::test]
    async fn the_directory_scan_runs_off_the_async_executor() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("bin/llamacpp/rocm-stable/build")).unwrap();
        let threads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = threads.clone();
        take_inventory_with(
            &write_manifest(root.path()),
            Instant::now() + INVENTORY_BUDGET,
            SEARCH_LIMITS,
            std::sync::Arc::new(move || {
                recorded.lock().unwrap().push(std::thread::current().id());
                false
            }),
        )
        .await;
        let threads = threads.lock().unwrap();
        assert!(!threads.is_empty(), "the walk never read the clock");
        let executor = std::thread::current().id();
        assert!(
            threads.iter().all(|thread| *thread != executor),
            "the scan ran on the async executor"
        );
    }

    /// The limits cover the whole search: two backend directories with three
    /// subdirectories each read six entries between them, so a limit of four stops
    /// the second directory, which a per-directory limit of four would not.
    #[test]
    fn entry_and_directory_limits_apply_to_the_whole_search() {
        let root = tempfile::tempdir().unwrap();
        for backend in ["rocm-stable", "rocm-nightly"] {
            for build in ["a", "b", "c"] {
                std::fs::create_dir_all(root.path().join("bin/llamacpp").join(backend).join(build))
                    .unwrap();
            }
        }
        let found = discover_candidates(root.path(), limits(100, 4), &|| false);
        assert_eq!(found.stop, Some(Stop::Entries), "{found:?}");
        assert_eq!(found.entries_read, 4, "{found:?}");

        let found = discover_candidates(root.path(), limits(2, 100), &|| false);
        assert_eq!(found.stop, Some(Stop::Directories), "{found:?}");
        assert_eq!(found.dirs_visited, 2, "{found:?}");

        let found = discover_candidates(root.path(), SEARCH_LIMITS, &|| false);
        assert_eq!(
            found.stop, None,
            "the default limits are not the small ones"
        );
    }

    #[tokio::test]
    async fn a_search_cut_short_by_a_limit_says_so_and_does_not_claim_an_empty_result() {
        let root = tempfile::tempdir().unwrap();
        for build in ["a", "b", "c"] {
            std::fs::create_dir_all(root.path().join("bin/llamacpp/rocm-stable").join(build))
                .unwrap();
        }
        let inventory = take_inventory_with(
            &write_manifest(root.path()),
            Instant::now() + INVENTORY_BUDGET,
            limits(2, 100),
            std::sync::Arc::new(|| false),
        )
        .await;
        assert!(
            inventory
                .report
                .contains("candidate search stopped: 2 directories visited (limit)"),
            "{}",
            inventory.report
        );
        let inventory = take_inventory_with(
            &write_manifest(root.path()),
            Instant::now() + INVENTORY_BUDGET,
            limits(100, 2),
            std::sync::Arc::new(|| false),
        )
        .await;
        assert!(
            inventory
                .report
                .contains("candidate search stopped: 2 directory entries read (limit)"),
            "{}",
            inventory.report
        );
    }

    /// The candidate cap ends the walk as soon as it is reached, instead of the walk
    /// finishing and the list being cut afterwards: with twenty candidate
    /// directories only the root and the first eight are entered, in sorted order.
    #[test]
    fn the_candidate_cap_stops_the_walk_when_it_is_reached() {
        let root = tempfile::tempdir().unwrap();
        let backend = root.path().join("bin/llamacpp/rocm-stable");
        let wanted: Vec<PathBuf> = (0..20)
            .map(|i| touch_server(&backend.join(format!("{i:02}"))))
            .collect();
        let found = discover_candidates(root.path(), SEARCH_LIMITS, &|| false);
        assert_eq!(found.stop, Some(Stop::Candidates));
        assert_eq!(found.candidates, wanted[..MAX_CANDIDATES]);
        assert_eq!(found.dirs_visited, 1 + MAX_CANDIDATES, "{found:?}");
    }

    /// A symlink is neither entered nor read as a candidate. The tree holds a link
    /// back to its own backend directory (a cycle), a link out to a directory with a
    /// real `llama-server`, a `llama-server` that is a link to a file outside, and a
    /// whole backend directory that is a link. The walk must end, find only the real
    /// file, and count what it declined to follow.
    #[cfg(unix)]
    #[test]
    fn symlinks_are_not_followed_and_a_cycle_ends() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_server = touch_server(outside.path());
        let backends = root.path().join("bin/llamacpp");
        let real = touch_server(&backends.join("rocm-stable/real"));
        symlink(
            backends.join("rocm-stable"),
            backends.join("rocm-stable/loop"),
        )
        .unwrap();
        symlink(outside.path(), backends.join("rocm-stable/out")).unwrap();
        symlink(
            &outside_server,
            backends.join("rocm-stable").join(server_binary()),
        )
        .unwrap();
        symlink(outside.path(), backends.join("rocm-nightly")).unwrap();

        let found = discover_candidates(root.path(), SEARCH_LIMITS, &|| false);
        assert_eq!(found.candidates, vec![real], "{found:?}");
        assert_eq!(found.stop, None, "{found:?}");
        assert_eq!(found.symlinks_skipped, 4, "{found:?}");
        assert_eq!(found.dirs_visited, 2, "{found:?}");
    }

    #[tokio::test]
    async fn the_empty_search_listing_is_bounded_and_counts_what_it_omits() {
        let root = tempfile::tempdir().unwrap();
        let llamacpp = root.path().join("bin/llamacpp");
        for i in 0..(MAX_LISTED_NAMES + 4) {
            std::fs::create_dir_all(llamacpp.join(format!("other-{i:03}"))).unwrap();
        }
        let report = evidence(&write_manifest(root.path())).await;
        assert!(report.contains(" and 4 more"), "{report}");
        assert!(!report.contains("at least"), "{report}");

        for i in 0..(2 * MAX_LISTING_SCAN) {
            std::fs::create_dir_all(llamacpp.join(format!("other-{i:03}"))).unwrap();
        }
        let report = evidence(&write_manifest(root.path())).await;
        let listing = report
            .split("contains: ")
            .nth(1)
            .and_then(|rest| rest.lines().next())
            .unwrap();
        let (names, omitted) = listing.split_once(" and at least ").unwrap();
        assert_eq!(names.split(", ").count(), MAX_LISTED_NAMES, "{report}");
        assert!(
            omitted.starts_with(&format!("{} more", MAX_LISTING_SCAN - MAX_LISTED_NAMES)),
            "{report}"
        );
        assert!(
            omitted.contains(&format!("first {MAX_LISTING_SCAN} entries")),
            "{report}"
        );
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
            report.contains(&format!(
                "sha256 unavailable: executable exceeds {} hash limit",
                describe_hash_limit(MAX_HASH_BYTES)
            )),
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

    type Events = std::sync::Mutex<Vec<String>>;

    struct RecordingProbe<'a> {
        events: &'a Events,
    }

    impl Probe for RecordingProbe<'_> {
        async fn run(&self, _server: &Path, flag: &str, _timeout: Duration) -> String {
            self.events.lock().unwrap().push(format!("probe {flag}"));
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
        let events = Events::default();
        let (stop_status, report) = evidence_around_stop(
            &manifest,
            DEFAULT_BUDGETS,
            &RecordingProbe { events: &events },
            || {
                events.lock().unwrap().push("stop".to_owned());
                std::fs::remove_file(&server).unwrap();
                "stopped".to_owned()
            },
        )
        .await;
        assert_eq!(stop_status, "stopped");
        assert_eq!(
            *events.lock().unwrap(),
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
        let stops = std::sync::atomic::AtomicUsize::new(0);
        let stop = || {
            stops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
        assert_eq!(stops.load(std::sync::atomic::Ordering::SeqCst), 2);

        let (_root, manifest, _server) = lemonade_data_dir_with_candidate();
        let (status, evidence) =
            backend_evidence_around_stop(|| manifest, Some("lemonade"), stop).await;
        assert_eq!(status, "stopped");
        assert!(evidence.unwrap().contains("manifest: "));
        assert_eq!(stops.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn an_unreadable_manifest_is_reported_and_the_service_is_still_stopped() {
        let root = tempfile::tempdir().unwrap();
        let events = Events::default();
        let (stop_status, report) = evidence_around_stop(
            &root.path().join("missing.json"),
            DEFAULT_BUDGETS,
            &RecordingProbe { events: &events },
            || "stopped".to_owned(),
        )
        .await;
        assert_eq!(stop_status, "stopped");
        assert!(report.contains("unavailable:"), "{report}");
        assert!(events.lock().unwrap().is_empty(), "no candidate, no probe");
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

    /// Selects what [`probe_child`] does when a probe test re-runs this test binary
    /// as its child process.
    const PROBE_CHILD_ENV: &str = "ROCM_E2E_PROBE_CHILD";

    /// Not a test: the child process the probe tests below start.
    ///
    /// The probe's timeout, kill and reap path has to hold on Windows too, and a
    /// shell-script stand-in does not exist there. This test binary does, on every
    /// platform, so the tests re-run it with `--ignored --exact` and this function
    /// plays the stand-in. It is `#[ignore]`d so a normal run skips it, and does
    /// nothing unless [`PROBE_CHILD_ENV`] selects a behaviour.
    #[test]
    #[ignore = "child process of the probe tests, not a test of its own"]
    fn probe_child() {
        match std::env::var(PROBE_CHILD_ENV).as_deref() {
            Ok("sleep") => {
                println!("pid:{}", std::process::id());
                std::thread::sleep(Duration::from_secs(30));
            }
            Ok("exit") => println!("child-ran"),
            _ => {}
        }
    }

    fn probe_child_command(behaviour: &str) -> tokio::process::Command {
        let module = module_path!().split_once("::").unwrap().1;
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                &format!("{module}::probe_child"),
                "--ignored",
                "--nocapture",
            ])
            .env(PROBE_CHILD_ENV, behaviour);
        command
    }

    /// The timeout path on every platform: the child is still running when the
    /// timeout fires, so the report must say so, return promptly, and leave no
    /// process behind. `process_is_running` is false only once the child has been
    /// killed and reaped, so a probe that merely abandoned it turns this red.
    #[tokio::test]
    async fn a_probe_that_outlives_its_timeout_is_killed_and_reaped() {
        let start = Instant::now();
        let report = run_probe(probe_child_command("sleep"), Duration::from_secs(5)).await;
        assert!(report.starts_with("probe timed out after 5s"), "{report}");
        assert!(!report.contains("kill/reap failed"), "{report}");
        assert!(start.elapsed() < Duration::from_secs(20), "{report}");
        let pid = report
            .lines()
            .find_map(|line| line.strip_prefix("pid:"))
            .unwrap_or_else(|| {
                panic!("the child did not report its pid before the timeout: {report}")
            })
            .parse::<u32>()
            .unwrap();
        assert!(!rocm_core::process_is_running(pid), "{report}");
    }

    /// The normal path on every platform: status and output come back, and the
    /// timeout branch is not taken.
    #[tokio::test]
    async fn a_probe_that_exits_reports_its_status_and_output() {
        let report = run_probe(probe_child_command("exit"), Duration::from_mins(1)).await;
        let status = if cfg!(windows) {
            "exit code: 0\n"
        } else {
            "exit status: 0\n"
        };
        assert!(report.starts_with(status), "{report}");
        assert!(report.contains("child-ran"), "{report}");
        assert!(!report.contains("timed out"), "{report}");
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
