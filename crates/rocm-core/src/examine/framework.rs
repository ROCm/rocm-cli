// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The framework probes: PyTorch (through the active runtime or `PATH`) and
//! llama.cpp.

use super::{Examination, FrameworkProbe};
use crate::probe_exec::{run, run_with_env, which};
use crate::{FrameworkInterpreter, RUNTIME_LIBRARY_PATH_ENV};
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

const PYTORCH_PROBE: &str = concat!(
    "import json,sys\n",
    "out={'ok':False}\n",
    "try:\n",
    "  import torch\n",
    "  out['ok']=True\n",
    "  out['version']=torch.__version__\n",
    "  out['hip']=getattr(torch.version,'hip',None)\n",
    "  out['cuda']=getattr(torch.version,'cuda',None)\n",
    "  out['is_available']=bool(torch.cuda.is_available())\n",
    "  try: out['device_count']=int(torch.cuda.device_count())\n",
    "  except Exception: out['device_count']=0\n",
    "  try: out['arch_list']=list(torch.cuda.get_arch_list())\n",
    "  except Exception: out['arch_list']=[]\n",
    "except Exception as ex:\n",
    "  out['error']=type(ex).__name__+': '+str(ex)\n",
    "sys.stdout.write(json.dumps(out))\n",
);

pub(super) fn probe_framework(
    e: &mut Examination,
    framework: FrameworkProbe,
    interpreter: Option<&FrameworkInterpreter>,
) {
    match framework {
        FrameworkProbe::Skip => e.framework = "skipped".to_owned(),
        FrameworkProbe::PyTorch => probe_pytorch(e, interpreter),
        FrameworkProbe::LlamaCpp => probe_llama_cpp(e),
        FrameworkProbe::Auto => {
            // A managed runtime brings its own interpreter, so the ambient PATH
            // no longer decides whether torch is worth asking about.
            if interpreter.is_some() || which("python") || which("python3") {
                probe_pytorch(e, interpreter);
                if e.framework == "pytorch" {
                    return;
                }
            }
            probe_llama_cpp(e);
        }
    }
}

fn probe_pytorch(e: &mut Examination, interpreter: Option<&FrameworkInterpreter>) {
    match interpreter {
        Some(interpreter) => probe_pytorch_in_runtime(e, interpreter),
        None => probe_pytorch_on_path(e),
    }
}

/// Probe the torch the engines will actually load.
///
/// Deliberately does not fall back to the `PATH` interpreter when the runtime's
/// torch fails to import. Falling back would double the timeout budget and
/// produce an examination whose fields describe one interpreter while its notes
/// describe another — and a broken active runtime *is* the host's real answer,
/// because that is the torch `rocm serve` will run.
fn probe_pytorch_in_runtime(e: &mut Examination, interpreter: &FrameworkInterpreter) {
    e.framework_source = "managed-runtime".to_owned();
    e.framework_notes.push(format!(
        "Probed torch with the active managed runtime's interpreter: {}",
        interpreter.python.display()
    ));
    let env = runtime_library_path_env(e, &interpreter.library_paths);
    let (rc, out, err) = run_with_env(
        &interpreter.python.display().to_string(),
        &["-c", PYTORCH_PROBE],
        &env,
        Duration::from_secs(20),
    );
    record_pytorch_probe(e, rc, &out, &err);
}

/// Compose the loader path a runtime's torch needs, runtime entries ahead of
/// whatever the host already has, so the runtime's own ROCm wins.
///
/// A failure to compose it is recorded rather than swallowed: the import that
/// follows would die on a missing ROCm library and read as a broken runtime,
/// which is the misdiagnosis this whole path exists to avoid.
fn runtime_library_path_env(
    e: &mut Examination,
    library_paths: &[PathBuf],
) -> Vec<(String, OsString)> {
    if library_paths.is_empty() {
        return Vec::new();
    }
    let mut entries = library_paths.to_vec();
    if let Some(existing) = std::env::var_os(RUNTIME_LIBRARY_PATH_ENV) {
        // `runtime_path_list_split`, not `std::env::split_paths`: on Windows it
        // trims, drops empty entries and host-normalises each one, on top of the
        // same quoting rules `split_paths` applies. The sibling composition in
        // `probe_runtime_devices` (`apps/rocm/src/therock/probe.rs`, a different crate)
        // already uses it, and the two building the same variable differently is
        // how they drift.
        //
        // The quoting is what keeps the `join_paths` below on its `Ok` arm: this
        // variable is `PATH` on Windows, a quoted entry anywhere in the inherited
        // one is legal, and a splitter that left the `"` in place would fail the
        // join and cost the interpreter every library path rather than one.
        entries.extend(crate::runtime_path_list_split(&existing));
    }
    match std::env::join_paths(entries) {
        Ok(joined) => vec![(RUNTIME_LIBRARY_PATH_ENV.to_owned(), joined)],
        Err(err) => {
            e.framework_notes.push(format!(
                "Could not compose {RUNTIME_LIBRARY_PATH_ENV} for the runtime's interpreter ({err}); \
                 its torch may fail to load the runtime's ROCm libraries."
            ));
            Vec::new()
        }
    }
}

fn probe_pytorch_on_path(e: &mut Examination) {
    let py = if which("python") {
        "python"
    } else if which("python3") {
        "python3"
    } else {
        e.framework_notes
            .push("No python interpreter found to probe torch.".to_owned());
        return;
    };
    e.framework_source = "path".to_owned();
    let (rc, out, err) = run(py, &["-c", PYTORCH_PROBE], Duration::from_secs(20));
    let (rc, out, err) = if (rc != 0 || out.trim().is_empty()) && py == "python" && which("python3")
    {
        let (rc2, out2, err2) = run("python3", &["-c", PYTORCH_PROBE], Duration::from_secs(20));
        if out2.trim().is_empty() {
            (rc, out, err)
        } else {
            (rc2, out2, err2)
        }
    } else {
        (rc, out, err)
    };
    record_pytorch_probe(e, rc, &out, &err);
}

fn record_pytorch_probe(e: &mut Examination, rc: i32, out: &str, err: &str) {
    if out.trim().is_empty() {
        // `run` reports 127 when the program could not be spawned and 124 on
        // timeout. Those are facts about the interpreter, not about torch, and
        // the venv advice below is actively wrong for a managed runtime — its
        // interpreter is the one that was asked.
        e.framework_notes.push(match rc {
            127 => "The torch probe interpreter could not be started.".to_owned(),
            124 => "The torch probe timed out.".to_owned(),
            _ if e.framework_source == "managed-runtime" => {
                "The active managed runtime's interpreter returned nothing for the torch probe."
                    .to_owned()
            }
            _ => "Could not import torch; if PyTorch is in a venv, activate it and re-run inside that venv."
                .to_owned(),
        });
        if let Some(last) = err.trim().lines().last() {
            let snippet: String = last.chars().take(200).collect();
            e.framework_notes.push(format!("python stderr: {snippet}"));
        }
        return;
    }
    let Ok(data) = serde_json::from_str::<serde_json::Value>(out.trim()) else {
        let snippet: String = out.chars().take(200).collect();
        e.framework_notes
            .push(format!("torch probe returned non-JSON: {snippet}"));
        return;
    };
    if data.get("ok").and_then(serde_json::Value::as_bool) != Some(true) {
        let err = data
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        e.framework_notes
            .push(format!("torch import failed: {err}"));
        return;
    }
    e.framework = "pytorch".to_owned();
    e.framework_version = data
        .get("version")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let hip = data.get("hip").and_then(serde_json::Value::as_str);
    let cuda = data.get("cuda").and_then(serde_json::Value::as_str);
    if let Some(hip) = hip.filter(|h| !h.is_empty()) {
        e.framework_rocm_version = format!("hip={hip}");
    } else if let Some(cuda) = cuda.filter(|c| !c.is_empty()) {
        e.framework_rocm_version = format!("cuda={cuda}");
        e.framework_notes.push(
            "This torch wheel is a CUDA build, not a ROCm build. Reinstall from the ROCm wheel index."
                .to_owned(),
        );
    }
    if let Some(arch) = data.get("arch_list").and_then(serde_json::Value::as_array) {
        e.framework_arch_list = arch
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
    }
    if data
        .get("is_available")
        .and_then(serde_json::Value::as_bool)
        == Some(false)
    {
        e.framework_notes.push(
            "torch.cuda.is_available() returned False -- runtime can't see a GPU.".to_owned(),
        );
    }
}

fn probe_llama_cpp(e: &mut Examination) {
    let binary = ["llama-cli", "llama-server", "main"]
        .into_iter()
        .find(|name| which(name));
    let Some(binary) = binary else {
        e.framework_notes
            .push("No llama.cpp binary (llama-cli/llama-server/main) on PATH.".to_owned());
        return;
    };
    let (rc, out, err) = run(binary, &["--version"], Duration::from_secs(10));
    let body = format!("{out}{err}");
    if rc != 0 && body.is_empty() {
        e.framework_notes
            .push(format!("{binary} --version exited rc={rc}"));
        return;
    }
    // Set only now that this probe is the one answering. Setting it on finding
    // the binary would, on the `Auto` fall-through from a failed runtime probe,
    // relabel the runtime's answer as the ambient one.
    e.framework_source = "path".to_owned();
    e.framework = "llama-cpp".to_owned();
    e.framework_version = body.trim().lines().next().map_or_else(
        || "unknown".to_owned(),
        |line| line.chars().take(200).collect(),
    );
    if body.contains("HIP") || body.contains("ROCm") || body.contains("hipBLAS") {
        e.framework_rocm_version = "GGML_HIP=ON".to_owned();
    } else {
        e.framework_notes.push(
            "llama.cpp binary doesn't advertise HIP/ROCm support; was it built with `cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx>`?"
                .to_owned(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialises the tests that read or replace the process-global
    /// `RUNTIME_LIBRARY_PATH_ENV` while they run. Env is shared by every test
    /// thread, so a test that sets it and one that composes a child env from it
    /// can otherwise see each other's value mid-test.
    #[cfg(unix)]
    static PROCESS_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Replaces a process-global environment variable for as long as it lives,
    /// putting the previous value back on drop.
    ///
    /// A scope guard rather than straight-line save/restore because the restore
    /// has to survive a panic: an assertion firing between the two halves skips
    /// the restore, and the mutated variable then leaks into every later test in
    /// this binary. The lock above only serialises those tests -- it does not
    /// undo the write, and recovering from its poison hands the next test the
    /// leaked value.
    #[cfg(unix)]
    struct EnvVarGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    #[cfg(unix)]
    impl EnvVarGuard {
        #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let previous = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }
    }

    #[cfg(unix)]
    impl Drop for EnvVarGuard {
        #[allow(unsafe_code)] // std::env::set_var/remove_var are unsafe in edition 2024
        fn drop(&mut self) {
            unsafe {
                match self.previous.take() {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn asking_to_skip_the_frameworks_is_recorded_as_skipped() {
        // Host-independent on purpose: every other variant depends on what is
        // installed, so this is the one the CLI flag can be pinned against
        // anywhere. "skipped" is a distinct answer from "unknown" -- the latter
        // means the probe ran and found nothing.
        let mut e = Examination::default();
        probe_framework(&mut e, FrameworkProbe::Skip, None);
        assert_eq!(e.framework, "skipped");
    }

    #[test]
    fn a_skipped_framework_probe_runs_no_interpreter() {
        // The reason to offer `skip` at all: the other variants start Python to
        // read the framework's ROCm build. Nothing else on the Examination may
        // move, or "skip" would be quietly doing work.
        let mut e = Examination::default();
        probe_framework(&mut e, FrameworkProbe::Skip, None);
        assert!(e.framework_version.is_empty());
        assert!(e.framework_rocm_version.is_empty());
        assert!(e.framework_arch_list.is_empty());
        assert!(e.framework_notes.is_empty());
    }

    /// What a healthy ROCm torch answers the probe, shared by the fakes below so
    /// a change to the probe's contract lands in one place.
    #[cfg(unix)]
    const RUNTIME_TORCH_OK_JSON: &str = r#"{"ok":true,"version":"2.11.0+rocm7.14.1","hip":"7.14.60850","cuda":null,"is_available":true,"device_count":1,"arch_list":["gfx942"]}"#;

    /// A stand-in for a managed runtime's interpreter.
    ///
    /// It answers like a ROCm torch **only** when the runtime's library
    /// directory reached it on the loader path, which is what the real thing
    /// does: a TheRock runtime's torch resolves HIP from a sibling
    /// `_rocm_sdk_core` package, so without those directories the import dies on
    /// `libroctx64.so.4`. Keying the fake on that means a probe that forgets the
    /// library paths fails the test instead of quietly reporting a broken
    /// runtime.
    #[cfg(unix)]
    fn plant_fake_runtime_interpreter(label: &str) -> (std::path::PathBuf, FrameworkInterpreter) {
        use crate::fake_executable::write_fake_executable;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-examine-{label}-{}",
            crate::test_support::unique_suffix()
        ));
        let libs = root.join("lib");
        std::fs::create_dir_all(&libs).expect("plant the runtime library dir");
        let python = root.join("python");
        write_fake_executable(&python,
            format!(
                "#!/bin/sh\n\
                 case \"${env}\" in\n\
                 *{marker}*) printf '%s' '{ok}' ;;\n\
                 *) printf '%s' '{broken}' ;;\n\
                 esac\n",
                env = RUNTIME_LIBRARY_PATH_ENV,
                marker = libs.display(),
                ok = RUNTIME_TORCH_OK_JSON,
                broken = r#"{"ok":false,"error":"ImportError: libroctx64.so.4: cannot open shared object file"}"#,
            ),
        )
        .expect("plant the fake interpreter");

        let interpreter = FrameworkInterpreter {
            python,
            library_paths: vec![libs],
        };
        (root, interpreter)
    }

    #[cfg(unix)]
    #[test]
    fn plant_fake_runtime_interpreter_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| {
            plant_fake_runtime_interpreter("same-label").0
        });
    }

    #[test]
    #[cfg(unix)]
    fn the_framework_probe_reads_the_active_runtimes_torch() {
        // The bug this pins: torch lives only inside the managed runtime, so a
        // probe that resolves its interpreter from PATH reports `unknown` for a
        // host that has a working one.
        let _guard = PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (root, interpreter) = plant_fake_runtime_interpreter("runtime-torch");
        let mut e = Examination::default();
        probe_framework(&mut e, FrameworkProbe::PyTorch, Some(&interpreter));
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(e.framework, "pytorch");
        assert_eq!(e.framework_version, "2.11.0+rocm7.14.1");
        assert_eq!(e.framework_rocm_version, "hip=7.14.60850");
        assert_eq!(e.framework_arch_list, vec!["gfx942".to_owned()]);
        assert_eq!(e.framework_source, "managed-runtime");
        assert!(
            e.framework_notes
                .iter()
                .any(|note| note.contains("active managed runtime's interpreter")),
            "the shift away from PATH must be self-describing: {:?}",
            e.framework_notes
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_runtime_without_its_library_path_reports_the_import_failure() {
        // Withholding the library paths turns "no torch" into "torch import
        // failed", which reads as a broken runtime -- a worse answer than the
        // silence it replaced. This pins how that case is REPORTED; the
        // composition itself is pinned by
        // `the_framework_probe_reads_the_active_runtimes_torch`, whose fake
        // interpreter only answers when the loader path actually reached it.
        let _guard = PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (root, mut interpreter) = plant_fake_runtime_interpreter("runtime-no-libs");
        interpreter.library_paths.clear();
        let mut e = Examination::default();
        probe_framework(&mut e, FrameworkProbe::PyTorch, Some(&interpreter));
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(e.framework, "unknown");
        assert_eq!(e.framework_source, "managed-runtime");
        assert!(
            e.framework_notes
                .iter()
                .any(|note| note.contains("libroctx64.so.4")),
            "expected the loader failure to be reported: {:?}",
            e.framework_notes
        );
    }

    /// A stand-in interpreter that records the loader path it was handed.
    ///
    /// `plant_fake_runtime_interpreter` only judges whether the runtime's own
    /// directory arrived, which cannot see what became of the entries the host
    /// already had. This one writes the whole variable out, so a test can assert
    /// on both halves of the merge and on their ORDER -- which is what decides
    /// whose ROCm the runtime's torch loads.
    #[cfg(unix)]
    fn plant_loader_path_recording_interpreter(
        label: &str,
    ) -> (std::path::PathBuf, std::path::PathBuf, FrameworkInterpreter) {
        use crate::fake_executable::write_fake_executable;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-examine-{label}-{}",
            crate::test_support::unique_suffix()
        ));
        let libs = root.join("lib");
        std::fs::create_dir_all(&libs).expect("plant the runtime library dir");
        let recorded = root.join("loader-path");
        let python = root.join("python");
        write_fake_executable(
            &python,
            format!(
                "#!/bin/sh\n\
                 printf '%s' \"${env}\" > '{recorded}'\n\
                 printf '%s' '{ok}'\n",
                env = RUNTIME_LIBRARY_PATH_ENV,
                recorded = recorded.display(),
                ok = RUNTIME_TORCH_OK_JSON,
            ),
        )
        .expect("plant the recording interpreter");

        let interpreter = FrameworkInterpreter {
            python,
            library_paths: vec![libs],
        };
        (root, recorded, interpreter)
    }

    #[cfg(unix)]
    #[test]
    fn plant_loader_path_recording_interpreter_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| {
            plant_loader_path_recording_interpreter("same-label").0
        });
    }

    #[test]
    #[cfg(unix)]
    fn the_runtime_loader_path_keeps_the_entries_the_host_already_had() {
        // The merge branch neither sibling reaches: one clears `library_paths`
        // and returns before the merge, the other never sets the variable, so on
        // a host that leaves it unset `if let Some(existing)` is skipped and the
        // extend below it never runs.
        //
        // What it guards is that PREPENDING the runtime's directories does not
        // DISCARD the inherited ones. A runtime's torch still resolves its C++
        // runtime and other system libraries from the host, so dropping them
        // would fail the import and report a healthy runtime broken -- the same
        // misdiagnosis `runtime_library_path_env` exists to avoid, reached by a
        // different route.
        let _guard = PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (root, recorded, interpreter) =
            plant_loader_path_recording_interpreter("runtime-loader-merge");
        // Under `root` so teardown takes it too, and named so it cannot appear
        // as a substring of the runtime's own `lib` entry.
        let inherited = root.join("host-lib");

        // Dropped before `_guard`, so the variable is restored while this test
        // still holds the lock, and restored at all if an assertion below panics.
        let _env = EnvVarGuard::set(RUNTIME_LIBRARY_PATH_ENV, &inherited);
        let mut e = Examination::default();
        probe_framework(&mut e, FrameworkProbe::PyTorch, Some(&interpreter));

        let seen = std::fs::read_to_string(&recorded)
            .expect("the recording interpreter must have written its loader path");
        let runtime_lib = interpreter.library_paths[0].display().to_string();
        let inherited = inherited.display().to_string();
        std::fs::remove_dir_all(&root).ok();

        let runtime_at = seen.find(&runtime_lib);
        let inherited_at = seen.find(&inherited);
        assert!(
            runtime_at.is_some(),
            "the runtime's own library directory must reach the child: {seen:?}"
        );
        assert!(
            inherited_at.is_some(),
            "the inherited entry must survive the merge, or the runtime's torch \
             loses the system libraries it still loads: {seen:?}"
        );
        assert!(
            runtime_at < inherited_at,
            "the runtime's ROCm must win over the host's, so its entries lead: {seen:?}"
        );
    }
}
