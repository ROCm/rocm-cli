// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The pre-warm's check that the active runtime can `import torchvision.ops`.
//!
//! Split out of [`crate::e2e_prewarm`], which calls [`repair_torchvision_mismatch`]
//! before it decides what to install and [`verify_runtime_imports`] after the
//! engine check. A persisted runtime can pass every freshness check and still hold
//! a torch and a torchvision that do not match; this finds that once, before the
//! suite, instead of in every vLLM serve scenario after its readiness timeout.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::e2e_prewarm::{StatusOk, default_engine, probe, rocm_command, runtime_ahead_of_index};

/// What importing torchvision's operators did in the active runtime's Python.
#[derive(Debug, PartialEq, Eq)]
enum VisionImport {
    /// The runtime ships no torchvision, so there is nothing to check.
    Absent,
    /// `import torchvision.ops` succeeded.
    Working,
    /// It failed, or did not finish; the tail of stderr (or the deadline) says how.
    Broken(String),
}

/// What [`TORCHVISION_PROBE`] prints, and [`classify_vision_import`] reads, when
/// the runtime has no torchvision. A test ties the two together.
const ABSENT_MARKER: &str = "absent";

/// How long the probe may run before it is killed and reported as broken. An
/// import that does not finish is a runtime this pre-warm must not wait on.
const PROBE_TIMEOUT: Duration = Duration::from_mins(2);

/// Run by the active runtime's Python. Prints [`ABSENT_MARKER`] and exits 0 when
/// the runtime has no torchvision, so such a runtime (a wheel runtime built
/// without it) is not mistaken for a broken one. `torchvision.ops` is the import
/// that fails when torchvision's compiled operators were built for a different
/// torch than the one installed next to it.
const TORCHVISION_PROBE: &str = "import importlib.util\n\
if importlib.util.find_spec('torchvision') is None:\n    print('absent')\n\
else:\n    import torchvision.ops\n";

/// Whether `text` carries torchvision's operator-mismatch error, such as
/// `operator torchvision::nms does not exist`. Both parts are required: this
/// decides whether a runtime is deleted, so an unrelated error that merely
/// mentions a `torchvision::` name must not match.
fn is_vision_mismatch(text: &str) -> bool {
    text.contains("torchvision::") && text.contains("does not exist")
}

/// Read what the probe did from its exit status and output.
fn classify_vision_import(success: bool, stdout: &str, stderr: &str) -> VisionImport {
    if !success {
        let lines: Vec<&str> = stderr
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect();
        let mut tail: Vec<&str> = lines.iter().rev().take(3).rev().copied().collect();
        // The mismatch error is not always in the last lines (a chained
        // traceback ends on the outer exception), and the callers read the
        // signature from the tail, so keep the first line that carries it.
        if !tail.iter().any(|line| is_vision_mismatch(line))
            && let Some(line) = lines.iter().find(|line| is_vision_mismatch(line))
        {
            tail.insert(0, line);
        }
        let tail = if tail.is_empty() {
            "the interpreter printed nothing to stderr".to_owned()
        } else {
            tail.join(" | ")
        };
        return VisionImport::Broken(tail);
    }
    if stdout.trim() == ABSENT_MARKER {
        VisionImport::Absent
    } else {
        VisionImport::Working
    }
}

/// How long to wait for a pipe's reader once the probe has ended. A grandchild
/// that inherited the pipe can hold it open after the probe is gone, and the
/// pre-warm must not wait on it.
const PIPE_GRACE: Duration = Duration::from_secs(2);

/// Read a child's pipe to its end on its own thread, so a full pipe cannot stall
/// the child while the caller waits on its deadline. The bytes come back over a
/// channel so the caller can bound its wait; a reader still blocked on a pipe
/// something else holds is left running (detached) and ends when that closes.
fn drain<R: std::io::Read + Send + 'static>(pipe: Option<R>) -> std::sync::mpsc::Receiver<Vec<u8>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_end(&mut bytes);
        }
        let _ = sender.send(bytes);
    });
    receiver
}

/// Run [`TORCHVISION_PROBE`] with `python`, with `library_paths` (the runtime's
/// recorded ROCm library directories) on the loader path the way the CLI's own
/// probes put them, and kill it once `timeout` has passed.
fn probe_vision_import(
    python: &Path,
    library_paths: &[PathBuf],
    timeout: Duration,
) -> std::io::Result<VisionImport> {
    let mut command = Command::new(python);
    command
        .args(["-c", TORCHVISION_PROBE])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if !library_paths.is_empty() {
        // The recorded directories plus the amd-smi shim, ahead of whatever the
        // job already exports: `runtime_loader_path_entries` is what a served
        // process gets, so a probe that skips it judges a healthy runtime
        // unloadable.
        let mut entries = rocm_core::runtime_loader_path_entries(python, library_paths);
        if let Some(existing) = std::env::var_os(rocm_core::RUNTIME_LIBRARY_PATH_ENV) {
            entries.extend(std::env::split_paths(&existing));
        }
        let joined = std::env::join_paths(entries).map_err(std::io::Error::other)?;
        command.env(rocm_core::RUNTIME_LIBRARY_PATH_ENV, joined);
    }
    // A script written an instant ago can be refused with `ETXTBSY` while another
    // thread's fork still holds the write handle; that clears within
    // milliseconds, so a few tries are made before giving up.
    let mut attempts = 0;
    let mut child = loop {
        match command.spawn() {
            Err(error)
                if error.kind() == std::io::ErrorKind::ExecutableFileBusy && attempts < 20 =>
            {
                attempts += 1;
                std::thread::sleep(Duration::from_millis(50));
            }
            spawned => break spawned?,
        }
    };
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if Instant::now() >= deadline {
            // Killing the probe closes its own ends of the pipes; a process it
            // spawned may still hold them, which `PIPE_GRACE` below bounds.
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let read = |receiver: std::sync::mpsc::Receiver<Vec<u8>>| {
        String::from_utf8_lossy(&receiver.recv_timeout(PIPE_GRACE).unwrap_or_default()).into_owned()
    };
    let stdout = read(stdout);
    let stderr = read(stderr);
    let Some(status) = status else {
        return Ok(VisionImport::Broken(format!(
            "`import torchvision.ops` did not finish within {}s and was killed",
            timeout.as_secs_f32()
        )));
    };
    let verdict = classify_vision_import(status.success(), &stdout, &stderr);
    Ok(match verdict {
        // A crash during an ABI mismatch leaves no stderr, so say how it ended.
        VisionImport::Broken(tail) => VisionImport::Broken(format!("{tail}; {status}")),
        other => other,
    })
}

/// The active runtime as the tree records it.
#[derive(Debug, PartialEq, Eq)]
struct ActiveRuntime {
    key: String,
    python: PathBuf,
    /// `rocm_sdk.library_paths` from the runtime's manifest: the ROCm library
    /// directories a served process gets on its loader path. Empty when the
    /// manifest records none.
    library_paths: Vec<PathBuf>,
}

/// The active runtime's key, Python and library directories, from the tree's
/// `active.json` and that runtime's registry manifest.
///
/// `Ok(None)` only when there is nothing to check: the tree records no active
/// runtime, or the manifest records no interpreter (absent or null; the CLI
/// writes it as optional, so a renamed field cannot be told from that). A marker
/// or manifest that exists but cannot be read or parsed, or a marker without
/// `runtime_key` (which the CLI always writes), is an error.
fn active_runtime(prewarm_dir: &Path) -> Result<Option<ActiveRuntime>> {
    let runtimes = prewarm_dir.join("data").join("runtimes");
    let marker = runtimes.join("active.json");
    let active = match std::fs::read_to_string(&marker) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", marker.display()));
        }
    };
    let active: serde_json::Value = serde_json::from_str(&active)
        .with_context(|| format!("{} is not valid JSON", marker.display()))?;
    let key = active
        .get("runtime_key")
        .and_then(serde_json::Value::as_str)
        .with_context(|| format!("{} records no `runtime_key`", marker.display()))?;
    let manifest_path = runtimes.join("registry").join(format!("{key}.json"));
    let manifest = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest)
        .with_context(|| format!("{} is not valid JSON", manifest_path.display()))?;
    let Some(python) = manifest
        .get("python_executable")
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    let library_paths = manifest
        .get("rocm_sdk")
        .and_then(|sdk| sdk.get("library_paths"))
        .and_then(serde_json::Value::as_array)
        .map(|paths| {
            paths
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default();
    Ok(Some(ActiveRuntime {
        key: key.to_owned(),
        python: PathBuf::from(python),
        library_paths,
    }))
}

/// What the torchvision check found in the active runtime.
#[derive(Debug, PartialEq, Eq)]
enum ImportCheck {
    /// Nothing to check: no active runtime, or none with a recorded interpreter.
    NoRuntime,
    /// The runtime ships no torchvision.
    Absent { key: String },
    /// `import torchvision.ops` worked.
    Working { key: String },
    /// A shared library would not load and the manifest records no library
    /// directories to have given it: the probe was not configured the way the
    /// CLI's are, which says nothing about the runtime.
    Unconfigured { key: String, tail: String },
    /// The import failed or hung.
    Broken { key: String, tail: String },
}

/// Probe the active runtime's Python for `import torchvision.ops`.
fn check_runtime_imports(prewarm_dir: &Path, timeout: Duration) -> Result<ImportCheck> {
    let Some(runtime) = active_runtime(prewarm_dir)? else {
        return Ok(ImportCheck::NoRuntime);
    };
    let key = runtime.key.clone();
    let verdict = probe_vision_import(&runtime.python, &runtime.library_paths, timeout)
        .with_context(|| format!("failed to run the Python of runtime {key}"))?;
    Ok(match verdict {
        VisionImport::Absent => ImportCheck::Absent { key },
        VisionImport::Working => ImportCheck::Working { key },
        VisionImport::Broken(tail)
            if runtime.library_paths.is_empty()
                && tail.contains("cannot open shared object file") =>
        {
            ImportCheck::Unconfigured { key, tail }
        }
        VisionImport::Broken(tail) => ImportCheck::Broken { key, tail },
    })
}

/// Whether serves on `engine` import torch, and so whether a runtime whose torch
/// and torchvision do not match is one they cannot use.
///
/// The Lemonade engine runs its own llama.cpp backends and imports no torch
/// (`engines/lemonade/src/lib.rs` records `torch_version: None`), so a mismatch
/// in the runtime beside it does not fail its serves. Every other engine is
/// treated as needing torch: the cost of being wrong that way is a failed
/// pre-warm that names the cause, where the other way is a lane that waits out
/// every readiness timeout.
fn engine_imports_torch(engine: &str) -> bool {
    engine != "lemonade"
}

/// Remove the active runtime when its torch and torchvision do not match and the
/// engine about to serve needs torch, so the decision that follows sees a tree
/// without it and installs afresh.
///
/// The same repair as `repair_poisoned_runtimes`: `rocm runtimes uninstall` drops
/// the registry entry, the active marker and the config pointers, and
/// `decide` on a registry with no runtime for the channel answers
/// `Decision::Install` (see the `no_managed_runtime_installs` test). Narrower on
/// purpose: only the operator-mismatch signature, because a hang, a library that
/// will not load or any other import error is not known to be cured by a
/// reinstall, and deleting a healthy runtime on a flaky probe would cost a
/// reinstall every run. Those reach [`verify_runtime_imports`] and fail there.
///
/// Best effort until the uninstall itself: a tree whose engine or runtime state
/// cannot be read here is left for the fatal check after the engine install.
///
/// Not done for a runtime the pre-warm must not replace: when the run is pinned
/// to an exact version, or the report says the runtime is `ahead_of_index` (see
/// `decide`: a reinstall could only roll it back). Those, and a report that
/// cannot be read, are declined: the runtime stays, this returns `true`, and
/// [`verify_runtime_imports`] fails after the engine check with the manual advice.
///
/// Returns whether a mismatch was found and left in place.
pub(crate) fn repair_torchvision_mismatch(
    rocm: &Path,
    prewarm_dir: &Path,
    pinned: bool,
) -> Result<bool> {
    repair_torchvision_mismatch_within(rocm, prewarm_dir, PROBE_TIMEOUT, pinned)
}

fn repair_torchvision_mismatch_within(
    rocm: &Path,
    prewarm_dir: &Path,
    timeout: Duration,
    pinned: bool,
) -> Result<bool> {
    let Ok(engine) = default_engine(rocm, prewarm_dir) else {
        return Ok(false);
    };
    if !engine_imports_torch(&engine) {
        return Ok(false);
    }
    let Ok(ImportCheck::Broken { key, tail }) = check_runtime_imports(prewarm_dir, timeout) else {
        return Ok(false);
    };
    if !is_vision_mismatch(&tail) {
        return Ok(false);
    }
    let declined = if pinned {
        Some("this pre-warm is pinned to an exact version".to_owned())
    } else {
        match probe(rocm, prewarm_dir) {
            Ok(report) if runtime_ahead_of_index(&report, &key) => Some(
                "it is ahead of the channel index, so a reinstall would roll it back".to_owned(),
            ),
            Ok(_) => None,
            Err(error) => Some(format!(
                "`rocm update` could not say whether it is ahead of the index ({error:#})"
            )),
        }
    };
    if let Some(why) = declined {
        println!(
            "::warning::pre-warm: runtime {key} cannot `import torchvision.ops` ({tail}), so its \
             torch and torchvision do not match; it is not removed because {why}."
        );
        return Ok(true);
    }
    println!(
        "::warning::pre-warm: removing the shared runtime {key} — it cannot `import \
         torchvision.ops` ({tail}), so its torch and torchvision do not match. A reinstall follows."
    );
    rocm_command(rocm, prewarm_dir)
        .args(["runtimes", "uninstall", &key, "--yes"])
        .status_ok("rocm runtimes uninstall")?;
    Ok(false)
}

/// Fail the pre-warm when the active runtime cannot import torchvision's operators
/// and the engine serving on it needs them.
///
/// A persisted runtime can pass every freshness check and still be broken: when
/// its torch and torchvision come from different installs, torchvision's compiled
/// operators no longer match torch and `import torchvision` dies with `operator
/// torchvision::nms does not exist`. Nothing before the suite imports it, so each
/// vLLM serve scenario would find out for itself after its readiness timeout.
/// This finds it once, before the suite starts.
///
/// Fatal where `engine` imports torch, like `ensure_default_engine`: serving
/// against a runtime already known to be broken fails later and names none of
/// this. For an engine that does not, the same finding is a warning. The probe
/// targets the runtime's own Python because that is where the engine runs: the
/// vLLM install lays its packages, torchvision included, into the managed
/// runtime's interpreter (`install_vllm_with_uv(&managed.python_executable, ..)` in
/// `engines/vllm/src/install.rs`) and creates no venv of its own.
pub(crate) fn verify_runtime_imports(
    rocm: &Path,
    prewarm_dir: &Path,
    installed_this_run: bool,
    engine: &str,
    repair_declined: bool,
) -> Result<()> {
    match check_runtime_imports(prewarm_dir, PROBE_TIMEOUT)? {
        ImportCheck::NoRuntime => println!(
            "pre-warm: no active runtime interpreter recorded; skipping the torchvision check"
        ),
        ImportCheck::Absent { key } => {
            println!("pre-warm: runtime {key} has no torchvision; nothing to check");
        }
        ImportCheck::Working { key } => {
            println!("pre-warm: runtime {key} imports torchvision.ops");
        }
        // Last resort: reached only when the manifest records no library
        // directories, so nothing could be put on the loader path.
        ImportCheck::Unconfigured { key, tail } => println!(
            "::warning::pre-warm: runtime {key} could not load a shared library for the \
             torchvision check and records no library directories to supply ({tail}); skipping it"
        ),
        ImportCheck::Broken { key, tail } => {
            let mismatch = if is_vision_mismatch(&tail) {
                " Its torch and torchvision do not match."
            } else {
                ""
            };
            if !engine_imports_torch(engine) {
                println!(
                    "::warning::pre-warm: runtime {key} cannot `import torchvision.ops` \
                     ({tail}).{mismatch} The {engine} engine does not import torch, so this \
                     does not fail the pre-warm."
                );
                return Ok(());
            }
            let advice = if installed_this_run {
                "This run installed the runtime, so the installer produced the broken pair and \
                 a reinstall would reproduce it; fix the install."
                    .to_owned()
            } else if mismatch.is_empty() || repair_declined {
                format!(
                    "To reinstall it, set ROCM_CLI_CONFIG_DIR={dir}/config, \
                     ROCM_CLI_DATA_DIR={dir}/data and ROCM_CLI_CACHE_DIR={dir}/cache, run \
                     `{rocm} runtimes uninstall {key} --yes`, and run the pre-warm again.",
                    dir = prewarm_dir.display(),
                    rocm = rocm.display()
                )
            } else {
                // `repair_torchvision_mismatch` removes this runtime at the start
                // of the next pre-warm, once it is the active one.
                "Run the pre-warm again: it removes a runtime whose torch and torchvision do \
                 not match and installs afresh."
                    .to_owned()
            };
            bail!(
                "runtime {key} cannot `import torchvision.ops` ({tail}).{mismatch} Serves that \
                 import torch would fail. {advice}"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that write a script and then run it hold this lock for their whole
    /// body. Writing an executable and exec'ing it fails with `ETXTBSY` when
    /// another thread forks while the write handle is still open, and these
    /// tests fork constantly.
    #[cfg(unix)]
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn vision_probe_is_read_from_exit_status_and_output() {
        assert_eq!(
            classify_vision_import(true, "", ""),
            VisionImport::Working,
            "a clean import prints nothing"
        );
        assert_eq!(
            classify_vision_import(true, "absent\n", ""),
            VisionImport::Absent
        );
        // The failure that cost the MI300X lane its green runs: only the tail of
        // the traceback is kept, and the cause is its last line.
        let stderr = "Traceback (most recent call last):\n  File \"x.py\", line 1\n    a\n    b\n\
                      RuntimeError: operator torchvision::nms does not exist\n";
        match classify_vision_import(false, "", stderr) {
            VisionImport::Broken(tail) => {
                assert!(tail.ends_with("RuntimeError: operator torchvision::nms does not exist"));
                assert_eq!(tail.matches(" | ").count(), 2, "three lines kept: {tail}");
            }
            other => panic!("expected Broken, got {other:?}"),
        }
        // A failure that prints "absent" first is still a failure.
        assert!(matches!(
            classify_vision_import(false, "absent\n", "boom"),
            VisionImport::Broken(_)
        ));
    }

    /// A tree whose active runtime records `python`, laid out as the CLI does.
    fn tree_with_python(python: &Path) -> tempfile::TempDir {
        tree_with_runtime(python, &[])
    }

    /// [`tree_with_python`] whose manifest also records `library_paths`.
    fn tree_with_runtime(python: &Path, library_paths: &[&Path]) -> tempfile::TempDir {
        let tree = tempfile::tempdir().expect("tempdir");
        let runtimes = tree.path().join("data").join("runtimes");
        std::fs::create_dir_all(runtimes.join("registry")).expect("registry dir");
        std::fs::write(
            runtimes.join("active.json"),
            r#"{"runtime_key": "release-wheel-test"}"#,
        )
        .expect("active.json");
        let manifest = if library_paths.is_empty() {
            serde_json::json!({ "python_executable": python })
        } else {
            serde_json::json!({
                "python_executable": python,
                "rocm_sdk": { "library_paths": library_paths },
            })
        };
        std::fs::write(
            runtimes.join("registry").join("release-wheel-test.json"),
            manifest.to_string(),
        )
        .expect("manifest");
        tree
    }

    #[test]
    fn active_runtime_follows_the_active_marker_to_its_manifest() {
        let tree = tree_with_python(Path::new("/runtime/bin/python"));
        assert_eq!(
            active_runtime(tree.path()).expect("reads"),
            Some(ActiveRuntime {
                key: "release-wheel-test".to_owned(),
                python: PathBuf::from("/runtime/bin/python"),
                library_paths: Vec::new(),
            })
        );
        // The library directories the CLI records under `rocm_sdk` are read too.
        let with_libs = tree_with_runtime(
            Path::new("/runtime/bin/python"),
            &[Path::new("/sdk/lib"), Path::new("/sdk/bin")],
        );
        assert_eq!(
            active_runtime(with_libs.path())
                .expect("reads")
                .expect("an active runtime")
                .library_paths,
            vec![PathBuf::from("/sdk/lib"), PathBuf::from("/sdk/bin")]
        );
        // No marker, nothing to check.
        let empty = tempfile::tempdir().expect("tempdir");
        assert_eq!(active_runtime(empty.path()).expect("reads"), None);
    }

    #[test]
    fn unreadable_runtime_state_is_an_error_not_a_skip() {
        let tree = tree_with_python(Path::new("/runtime/bin/python"));
        let runtimes = tree.path().join("data").join("runtimes");
        // A marker that exists but is not JSON.
        std::fs::write(runtimes.join("active.json"), "not json").expect("corrupt marker");
        assert!(active_runtime(tree.path()).is_err());
        assert!(
            verify_runtime_imports(Path::new("rocm"), tree.path(), false, "vllm", false).is_err()
        );
        // A marker naming a runtime whose manifest is gone.
        std::fs::write(
            runtimes.join("active.json"),
            r#"{"runtime_key": "no-such-runtime"}"#,
        )
        .expect("marker");
        assert!(active_runtime(tree.path()).is_err());
        // A marker that cannot be read at all (here, a directory) is not "absent".
        std::fs::remove_file(runtimes.join("active.json")).expect("remove marker");
        std::fs::create_dir(runtimes.join("active.json")).expect("directory marker");
        assert!(active_runtime(tree.path()).is_err());
        std::fs::remove_dir(runtimes.join("active.json")).expect("remove directory");
        // A marker without a runtime key is a shape change, not a skip.
        std::fs::write(runtimes.join("active.json"), "{}").expect("marker");
        assert!(active_runtime(tree.path()).is_err());
        // A manifest that exists but is not JSON.
        std::fs::write(
            runtimes.join("active.json"),
            r#"{"runtime_key": "release-wheel-test"}"#,
        )
        .expect("marker");
        std::fs::write(
            runtimes.join("registry").join("release-wheel-test.json"),
            "not json",
        )
        .expect("corrupt manifest");
        assert!(active_runtime(tree.path()).is_err());
        // A manifest that records no interpreter leaves nothing to check.
        std::fs::write(
            runtimes.join("registry").join("release-wheel-test.json"),
            r#"{"python_executable": null}"#,
        )
        .expect("manifest");
        std::fs::write(
            runtimes.join("active.json"),
            r#"{"runtime_key": "release-wheel-test"}"#,
        )
        .expect("marker");
        assert_eq!(active_runtime(tree.path()).expect("reads"), None);
    }

    #[test]
    fn the_probe_prints_the_marker_the_classifier_reads() {
        assert!(TORCHVISION_PROBE.contains(&format!("print('{ABSENT_MARKER}')")));
        assert_eq!(
            classify_vision_import(true, &format!("{ABSENT_MARKER}\n"), ""),
            VisionImport::Absent
        );
        assert!(
            matches!(
                classify_vision_import(false, "", "\n\n"),
                VisionImport::Broken(ref tail) if tail.contains("nothing to stderr")
            ),
            "an empty stderr still says something"
        );
    }

    /// A stand-in interpreter: a script that ignores `-c` and behaves as told.
    #[cfg(unix)]
    fn fake_python(dir: &Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("python");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("script");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        path
    }

    #[cfg(unix)]
    #[test]
    fn the_probe_is_passed_to_the_interpreter_with_dash_c() {
        let _serial = serial();
        // Succeeds only when its first argument is `-c`, as the probe passes it.
        let scripts = tempfile::tempdir().expect("tempdir");
        let python = fake_python(
            scripts.path(),
            "[ \"$1\" = -c ] || { echo \"unexpected arguments: $*\" >&2; exit 7; }",
        );
        assert_eq!(
            probe_vision_import(&python, &[], PROBE_TIMEOUT).expect("runs"),
            VisionImport::Working
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_advice_depends_on_whether_this_run_installed_the_runtime() {
        let _serial = serial();
        let scripts = tempfile::tempdir().expect("tempdir");
        let python = fake_python(scripts.path(), "echo 'ImportError: boom' >&2; exit 1");
        let tree = tree_with_python(&python);
        let reused = format!(
            "{:#}",
            verify_runtime_imports(Path::new("rocm"), tree.path(), false, "vllm", false)
                .expect_err("fails")
        );
        assert!(
            reused.contains("runtimes uninstall release-wheel-test --yes"),
            "{reused}"
        );
        assert!(
            !reused.contains("do not match"),
            "no torchvision:: in the tail: {reused}"
        );
        let fresh = format!(
            "{:#}",
            verify_runtime_imports(Path::new("rocm"), tree.path(), true, "vllm", false)
                .expect_err("fails")
        );
        assert!(fresh.contains("This run installed the runtime"), "{fresh}");
        assert!(!fresh.contains("runtimes uninstall"), "{fresh}");
    }

    #[cfg(unix)]
    #[test]
    fn the_runtimes_library_directories_reach_the_probe() {
        let _serial = serial();
        // Exits 0 only when its loader path starts with the recorded directory.
        let scripts = tempfile::tempdir().expect("tempdir");
        let python = fake_python(
            scripts.path(),
            "case \"$LD_LIBRARY_PATH\" in /sdk/lib:*|/sdk/lib) exit 0 ;; \
             *) echo \"loader path was: $LD_LIBRARY_PATH\" >&2; exit 3 ;; esac",
        );
        let with_libs = tree_with_runtime(&python, &[Path::new("/sdk/lib")]);
        assert!(
            verify_runtime_imports(Path::new("rocm"), with_libs.path(), false, "vllm", false)
                .is_ok()
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_library_that_will_not_load_fails_when_directories_were_supplied() {
        let _serial = serial();
        // The runtime's own directories were on the path and the load still
        // failed: that is a broken runtime, not an unconfigured probe.
        let scripts = tempfile::tempdir().expect("tempdir");
        let python = fake_python(
            scripts.path(),
            "echo 'libroctx64.so.4: cannot open shared object file' >&2; exit 1",
        );
        let tree = tree_with_runtime(&python, &[Path::new("/sdk/lib")]);
        let error = verify_runtime_imports(Path::new("rocm"), tree.path(), false, "vllm", false)
            .expect_err("must fail");
        assert!(
            format!("{error:#}").contains("libroctx64.so.4"),
            "{error:#}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_library_that_will_not_load_is_skipped_only_when_nothing_could_be_supplied() {
        let _serial = serial();
        let scripts = tempfile::tempdir().expect("tempdir");
        let python = fake_python(
            scripts.path(),
            "echo 'libroctx64.so.4: cannot open shared object file' >&2; exit 1",
        );
        let tree = tree_with_python(&python);
        assert!(matches!(
            check_runtime_imports(tree.path(), PROBE_TIMEOUT).expect("runs"),
            ImportCheck::Unconfigured { .. }
        ));
        assert!(
            verify_runtime_imports(Path::new("rocm"), tree.path(), false, "vllm", false).is_ok()
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_probe_that_never_finishes_is_killed_and_reported() {
        let _serial = serial();
        // `exec` so the killed process is the sleeper itself and holds the pipes.
        let scripts = tempfile::tempdir().expect("tempdir");
        let python = fake_python(scripts.path(), "exec sleep 60");
        let started = Instant::now();
        let verdict = probe_vision_import(&python, &[], Duration::from_millis(300)).expect("runs");
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the deadline held"
        );
        match verdict {
            VisionImport::Broken(tail) => assert!(tail.contains("did not finish"), "{tail}"),
            other => panic!("expected Broken, got {other:?}"),
        }
    }

    #[cfg(unix)]
    const MISMATCH: &str =
        "echo 'RuntimeError: operator torchvision::nms does not exist' >&2; exit 1";

    #[cfg(unix)]
    #[test]
    fn a_runtime_the_engine_does_not_need_torch_for_only_warns() {
        let _serial = serial();
        let scripts = tempfile::tempdir().expect("tempdir");
        let tree = tree_with_python(&fake_python(scripts.path(), MISMATCH));
        assert!(
            verify_runtime_imports(Path::new("rocm"), tree.path(), false, "lemonade", false)
                .is_ok()
        );
        assert!(
            verify_runtime_imports(Path::new("rocm"), tree.path(), false, "vllm", false).is_err()
        );
    }

    /// A stand-in `rocm`: `engines list` marks `engine` as the default, every call
    /// is appended to `calls`, and `runtimes uninstall <key>` removes that
    /// runtime's registry entry and the active marker from the tree the call is
    /// scoped to, as the CLI does. `update` prints the `update_report` file in `dir`
    /// and fails when `update_fails` exists.
    #[cfg(unix)]
    fn fake_rocm(dir: &Path, engine: &str) -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let calls = dir.join("calls");
        let path = dir.join("rocm");
        std::fs::write(
            &path,
            format!(
                "#!/bin/sh\necho \"$*\" >> {calls}\n\
                 [ \"$1 $2\" = \"engines list\" ] && echo '* {engine}  default'\n\
                 if [ \"$1\" = update ]; then\n\
                 [ -e {dir}/update_fails ] && exit 1\n\
                 cat {dir}/update_report 2>/dev/null\nexit 0\nfi\n\
                 if [ \"$1 $2\" = \"runtimes uninstall\" ]; then\n\
                 rm -f \"$ROCM_CLI_DATA_DIR/runtimes/registry/$3.json\" \
                 \"$ROCM_CLI_DATA_DIR/runtimes/active.json\"\nfi\nexit 0\n",
                calls = calls.display(),
                dir = dir.display()
            ),
        )
        .expect("script");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        (path, calls)
    }

    #[cfg(unix)]
    fn uninstalls(calls: &Path) -> String {
        std::fs::read_to_string(calls)
            .unwrap_or_default()
            .lines()
            .filter(|line| line.starts_with("runtimes uninstall"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[cfg(unix)]
    #[test]
    fn a_torch_and_torchvision_mismatch_is_uninstalled_so_the_next_decision_installs() {
        let _serial = serial();
        let scripts = tempfile::tempdir().expect("tempdir");
        let (rocm, calls) = fake_rocm(scripts.path(), "vllm");
        let tree = tree_with_python(&fake_python(scripts.path(), MISMATCH));
        let manifest = tree
            .path()
            .join("data/runtimes/registry/release-wheel-test.json");
        assert!(manifest.exists());
        repair_torchvision_mismatch(&rocm, tree.path(), false).expect("repairs");
        assert_eq!(
            uninstalls(&calls),
            "runtimes uninstall release-wheel-test --yes"
        );
        // Red if the uninstall is not requested for this runtime: the stand-in
        // removes the registry entry only for the key it is given. That the
        // emptied registry then installs afresh is `decide`'s own behavior,
        // pinned by `no_managed_runtime_installs` in `e2e_prewarm.rs`.
        assert!(!manifest.exists());
        assert_eq!(active_runtime(tree.path()).expect("reads"), None);
    }

    #[cfg(unix)]
    #[test]
    fn a_runtime_the_pre_warm_must_not_replace_is_not_uninstalled() {
        let _serial = serial();
        let line =
            |status: &str| format!("runtime release-wheel-test channel=release status={status}\n");
        // (pinned, `rocm update` report, `rocm update` fails, why)
        let cases: [(bool, String, bool, &str); 3] = [
            (true, line("update_available"), false, "a pinned run"),
            (
                false,
                line("ahead_of_index"),
                false,
                "a runtime ahead of the index",
            ),
            (false, String::new(), true, "a report that cannot be read"),
        ];
        for (pinned, report, fails, why) in cases {
            let scripts = tempfile::tempdir().expect("tempdir");
            let (rocm, calls) = fake_rocm(scripts.path(), "vllm");
            std::fs::write(scripts.path().join("update_report"), report).expect("report");
            if fails {
                std::fs::write(scripts.path().join("update_fails"), "").expect("marker");
            }
            let tree = tree_with_python(&fake_python(scripts.path(), MISMATCH));
            let declined = repair_torchvision_mismatch(&rocm, tree.path(), pinned).expect("runs");
            assert!(declined, "{why}: the mismatch is reported as left in place");
            assert_eq!(uninstalls(&calls), "", "{why}");
        }
        // The same mismatch, with the index ahead of the runtime, is removed.
        let scripts = tempfile::tempdir().expect("tempdir");
        let (rocm, calls) = fake_rocm(scripts.path(), "vllm");
        std::fs::write(
            scripts.path().join("update_report"),
            line("update_available"),
        )
        .expect("report");
        let tree = tree_with_python(&fake_python(scripts.path(), MISMATCH));
        assert!(!repair_torchvision_mismatch(&rocm, tree.path(), false).expect("runs"));
        assert_eq!(
            uninstalls(&calls),
            "runtimes uninstall release-wheel-test --yes"
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_advice_for_a_mismatch_depends_on_whether_the_repair_was_declined() {
        let _serial = serial();
        let scripts = tempfile::tempdir().expect("tempdir");
        let tree = tree_with_python(&fake_python(scripts.path(), MISMATCH));
        let message = |declined| {
            format!(
                "{:#}",
                verify_runtime_imports(Path::new("rocm"), tree.path(), false, "vllm", declined)
                    .expect_err("fails")
            )
        };
        assert!(message(false).contains("Run the pre-warm again"));
        let declined = message(true);
        assert!(
            declined.contains("runtimes uninstall release-wheel-test --yes")
                && !declined.contains("Run the pre-warm again"),
            "{declined}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn only_a_real_mismatch_on_a_torch_engine_is_uninstalled() {
        let _serial = serial();
        // Another import error, a hang, a working runtime, a runtime without
        // torchvision and a Lemonade lane are all left alone.
        let cases: [(&str, &str, &str); 6] = [
            (
                "vllm",
                "echo 'ImportError: boom' >&2; exit 1",
                "another import error",
            ),
            (
                "vllm",
                "echo 'ImportError: torchvision::nms failed to load' >&2; exit 1",
                "an error that names a torchvision operator without saying it is missing",
            ),
            ("vllm", "exit 0", "a working runtime"),
            ("vllm", "echo absent", "no torchvision"),
            ("lemonade", MISMATCH, "an engine that does not import torch"),
            // `exec` so the killed process is the sleeper itself.
            ("vllm", "exec sleep 60", "a probe that never finishes"),
        ];
        for (engine, body, why) in cases {
            let scripts = tempfile::tempdir().expect("tempdir");
            let (rocm, calls) = fake_rocm(scripts.path(), engine);
            let tree = tree_with_python(&fake_python(scripts.path(), body));
            repair_torchvision_mismatch_within(
                &rocm,
                tree.path(),
                Duration::from_millis(300),
                false,
            )
            .expect("runs");
            assert_eq!(uninstalls(&calls), "", "{why}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_grandchild_holding_the_pipes_cannot_stall_the_probe() {
        let _serial = serial();
        // The probe starts a background process that inherits its pipes, then
        // becomes a sleeper. Killing the probe leaves the grandchild holding
        // both pipes open, so the readers never see the end of the stream.
        let scripts = tempfile::tempdir().expect("tempdir");
        let python = fake_python(scripts.path(), "sleep 60 &\nexec sleep 60");
        let started = Instant::now();
        let verdict = probe_vision_import(&python, &[], Duration::from_millis(300)).expect("runs");
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the wait on the readers is bounded: {:?}",
            started.elapsed()
        );
        assert!(matches!(verdict, VisionImport::Broken(ref t) if t.contains("did not finish")));
    }

    #[test]
    fn a_mismatch_error_above_the_last_lines_is_still_recognised() {
        let stderr = "RuntimeError: operator torchvision::nms does not exist\n\
                      During handling of the above exception, another exception occurred:\n\
                      Traceback (most recent call last):\n  File \"x.py\", line 1\n\
                      ImportError: could not import torchvision.ops\n";
        match classify_vision_import(false, "", stderr) {
            VisionImport::Broken(tail) => {
                assert!(is_vision_mismatch(&tail), "{tail}");
                assert!(tail.ends_with("could not import torchvision.ops"), "{tail}");
            }
            other => panic!("expected Broken, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_runtime_that_cannot_import_torchvision_ops_fails_the_pre_warm() {
        let _serial = serial();
        let scripts = tempfile::tempdir().expect("tempdir");
        let python = fake_python(
            scripts.path(),
            "echo 'RuntimeError: operator torchvision::nms does not exist' >&2; exit 1",
        );
        let tree = tree_with_python(&python);
        let error = verify_runtime_imports(Path::new("rocm"), tree.path(), false, "vllm", false)
            .expect_err("must fail");
        let message = format!("{error:#}");
        assert!(
            message.starts_with("runtime release-wheel-test cannot"),
            "{message}"
        );
        assert!(
            message.contains(" Its torch and torchvision do not match."),
            "{message}"
        );
        // A mismatch is what `repair_torchvision_mismatch` removes at the start of
        // the next pre-warm (`a_torch_and_torchvision_mismatch_is_uninstalled_…`),
        // so the advice is to run it again.
        assert!(
            message.contains("Run the pre-warm again: it removes a runtime whose torch"),
            "{message}"
        );
        assert!(message.contains("exit status: 1"), "{message}");
        assert!(
            message.contains("torchvision::nms does not exist"),
            "{message}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn working_and_torchvision_free_runtimes_pass_the_pre_warm() {
        let _serial = serial();
        let scripts = tempfile::tempdir().expect("tempdir");
        let working = tree_with_python(&fake_python(scripts.path(), "exit 0"));
        assert!(
            verify_runtime_imports(Path::new("rocm"), working.path(), false, "vllm", false).is_ok()
        );

        let scripts = tempfile::tempdir().expect("tempdir");
        let absent = tree_with_python(&fake_python(scripts.path(), "echo absent"));
        assert!(
            verify_runtime_imports(Path::new("rocm"), absent.path(), false, "vllm", false).is_ok()
        );

        // No active runtime recorded: skipped, not failed.
        let empty = tempfile::tempdir().expect("tempdir");
        assert!(
            verify_runtime_imports(Path::new("rocm"), empty.path(), false, "vllm", false).is_ok()
        );
    }

    #[test]
    fn a_recorded_interpreter_that_cannot_start_fails_the_pre_warm() {
        // The manifest names a Python that is not there: the runtime is not
        // usable, which is the same verdict as one that cannot import.
        let tree = tree_with_python(Path::new("/nonexistent/runtime/bin/python"));
        let error = verify_runtime_imports(Path::new("rocm"), tree.path(), false, "vllm", false)
            .expect_err("must fail");
        assert!(
            format!("{error:#}").contains("failed to run the Python of runtime release-wheel-test")
        );
    }

    /// Runs the real probe with the host's `python3` against a throwaway
    /// `torchvision` package, so the probe text itself is exercised and not only
    /// the classifier. Skipped where no `python3` is on the PATH.
    #[cfg(unix)]
    #[test]
    fn the_probe_imports_torchvision_ops_in_a_real_interpreter() {
        let _serial = serial();
        let python = Path::new("python3");
        let usable = Command::new(python)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success());
        // Skipped, and said so, on an image without `python3`: the probe is run
        // by the runtime's own interpreter in the pre-warm, not by this one.
        if !usable {
            eprintln!("skipping: no working python3 on PATH");
            return;
        }
        let write_package = |ops: &str| {
            let dir = tempfile::tempdir().expect("tempdir");
            let package = dir.path().join("torchvision");
            std::fs::create_dir_all(&package).expect("package dir");
            std::fs::write(package.join("__init__.py"), "").expect("__init__");
            std::fs::write(package.join("ops.py"), ops).expect("ops");
            dir
        };
        let classify = |output: std::process::Output| {
            classify_vision_import(
                output.status.success(),
                &String::from_utf8_lossy(&output.stdout),
                &String::from_utf8_lossy(&output.stderr),
            )
        };
        let run = |path: &Path| {
            // The package goes on sys.path by hand so the probe text runs as is.
            let script = format!(
                "import sys; sys.path.insert(0, {:?})\n{TORCHVISION_PROBE}",
                path.to_string_lossy()
            );
            classify(
                Command::new(python)
                    .args(["-c", &script])
                    .output()
                    .expect("python3 runs"),
            )
        };

        let working = write_package("");
        assert_eq!(run(working.path()), VisionImport::Working);

        let broken =
            write_package("raise RuntimeError('operator torchvision::nms does not exist')");
        match run(broken.path()) {
            VisionImport::Broken(tail) => {
                assert!(tail.contains("torchvision::nms does not exist"), "{tail}");
            }
            other => panic!("expected Broken, got {other:?}"),
        }

        // `-I -S`: no site-packages and no current directory, so no torchvision
        // can be found and the probe takes its `find_spec` branch.
        let absent = classify(
            Command::new(python)
                .args(["-I", "-S", "-c", TORCHVISION_PROBE])
                .output()
                .expect("python3 runs"),
        );
        assert_eq!(absent, VisionImport::Absent);
    }
}
