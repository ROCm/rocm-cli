// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `comfyui.feature`.
//!
//! `comfyui-01` and `comfyui-02` drive a dependency install against an entirely
//! planted runtime: a `wheel` manifest with a satisfying `rocm_sdk` probe, a
//! pre-existing source checkout (so the CLI never attempts the real network
//! download of ComfyUI's source archive) and a fake Python interpreter that
//! answers the torch-stack version probe with valid, empty JSON. `comfyui-01`
//! covers what happens when the `uv` dependency install that follows all of
//! that fails; `comfyui-02` covers a `uv` install that succeeds, which runs on
//! into the post-install GPU check and needs the fake Python to also answer
//! that second probe.
//!
//! `comfyui-03` covers runtime *selection* rather than the install that follows
//! it: `rocm comfyui install` refuses when more than one managed ROCm runtime is
//! ready and none is activated, rather than guessing which one to install into.
//! Those steps plant two ready wheel runtimes on disk (readiness is filesystem
//! and manifest state, so no GPU is needed) and assert the refusal is actionable
//! in `rocm comfyui install`'s command output (`--runtime-id`,
//! `rocm runtimes activate`, and the `/runtimes` pointer). The text is CLI-only
//! today — not for want of a TUI error path: `/comfyui install` is
//! approval-gated, and a non-zero `rocm` exit is *captured* into an
//! `isError: true` envelope rather than raised, so the seam yields
//! `RocmToolOutcome::Result` (never the `Error` arm that prints a message
//! verbatim) and `summarize_json_value` collapses the envelope to
//! `content: [1 items]`. Both links in that chain are pinned:
//! `seam_execute_approved_captures_a_failing_command_as_a_result`
//! (`apps/rocm/src/dash_seam.rs`) replays a real refusing `rocm` subprocess
//! through the seam and asserts the `Result`/`isError: true` envelope, and
//! `approved_command_failure_stays_a_collapsed_envelope`
//! (`crates/rocm-dash-tui/src/app/mod.rs`) asserts that envelope is collapsed
//! out of the chat.
//!
//! `comfyui-04` covers the source-archive download spinner under a real PTY,
//! mirroring `download_progress_pty.feature`'s tarball scenario but for
//! `download_and_extract_source`'s in-process `GzDecoder`/`tar` unpack, which
//! has no separate extraction phase (unlike TheRock's subprocess `tar -xf`, it
//! never renders its own "Extracting…" frame — only the download spinner
//! line matters here). It plants a ready wheel runtime, points
//! `ROCM_CLI_COMFYUI_SOURCE_ARCHIVE_URL_OVERRIDE` at a paced loopback server,
//! and gives the fixture's `requirements.txt` only torch-stack entries so
//! `install()`'s dependency filter empties out and skips the `uv` block
//! entirely — this scenario is about the download spinner, not the
//! dependency install already covered above.
//!
//! `comfyui-05` is the GPU-only EAI-8051 guard: it installs ComfyUI into a real,
//! isolated managed runtime and asserts the runtime's torch is unchanged, no
//! `nvidia-*` distributions appeared, and the install really added packages. It
//! uses its own When phrase because the planted-runtime step above hard-codes
//! `--runtime-id`.
//!
//! Black-box throughout: the planted registry manifests are plain JSON matching
//! the CLI's on-disk schema, not typed imports from the product crates.

use std::path::{Path, PathBuf};
use std::time::Duration;

use cucumber::{given, then, when};
use e2e_cucumber::paced_download::{
    PacedDownloadServer, build_gzip_tarball, deterministic_payload,
};

use crate::E2eWorld;
use crate::e2e::tui_driver::TuiSession;

const RUNTIME_KEY: &str = "e2e-comfyui-runtime";

/// The two runtime keys planted for the ambiguity scenario. Distinct so the
/// assertion that the refusal lists both is meaningful.
const RUNTIME_KEYS: [&str; 2] = [
    "release-wheel-gfx94x-dcgpu-7-13-0",
    "nightly-wheel-gfx94x-dcgpu-7-14-0",
];

fn root(world: &E2eWorld) -> &Path {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
}

fn install_root(world: &E2eWorld) -> PathBuf {
    root(world).join("comfyui-fixture").join("install-root")
}

/// The scenario's isolated `data` dir — where the CLI reads its runtime registry
/// (`ROCM_CLI_DATA_DIR`, set by `isolate_env`).
fn data_dir(world: &E2eWorld) -> PathBuf {
    root(world).join("data")
}

fn write_fixture(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create fixture directory");
    }
    std::fs::write(path, contents).expect("failed to write fixture file");
}

/// Writes an executable POSIX shell script, standing in for a real binary the
/// CLI shells out to (`uv`, the runtime's Python).
fn write_shim(path: &Path, body: &str) {
    write_fixture(path, body);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .unwrap_or_else(|e| panic!("failed to chmod fake {}: {e}", path.display()));
    }
}

/// Writes a fake runtime Python that answers both forms `probe_comfyui`'s
/// post-install GPU check shells out to: `-c <script>` (the pre-install
/// torch-stack probe) and `<script-path> <result-path>` (the post-install
/// check, which writes its JSON result to the given path instead of stdout).
fn write_gpu_probe_shim(path: &Path) {
    write_shim(
        path,
        "#!/bin/sh\n\
         if [ \"$1\" = \"-c\" ]; then\n\
         \tprintf '{}'\n\
         \texit 0\n\
         fi\n\
         cat > \"$2\" <<'JSON'\n\
         {\"torch_version\": \"2.4.0\", \"torch_cuda_available\": true, \"device_count\": 1, \"devices\": [\"Fake GPU\"]}\n\
         JSON\n",
    );
}

/// Plant one ready wheel runtime: the on-disk stubs the CLI's readiness check
/// requires (an install root holding its local manifest, a Python executable,
/// and a rocm_sdk bin exposing amdhip64 + hipblas) plus the registry manifest
/// that points at them. The readiness gate validates recorded manifest state and
/// that these paths exist — it never executes anything — so a GPU-less host can
/// present a runtime the CLI accepts as "ready".
///
/// This JSON must stay schema-exact. `therock::load_runtime_manifests` skips
/// registry entries that fail to deserialize *silently* (`if let Ok(manifest)`),
/// so a typo or dropped required field here does not fail the run — it quietly
/// turns "two ready runtimes" into one or zero, and the scenario then fails on a
/// confusing downstream assertion instead of on the fixture. If this step starts
/// failing after an edit here, suspect the manifest shape first.
fn plant_ready_runtime(data: &Path, key: &str) {
    let install_root = data.join("runtimes").join("roots").join(key);
    let sdk_root = install_root.join("sdk");
    let sdk_bin = sdk_root.join("bin");
    let python = install_root.join("bin").join("python3");
    let amdhip = sdk_bin.join("libamdhip64.so");
    let hipblas = sdk_bin.join("libhipblas.so");

    write_fixture(&install_root.join(".rocm-cli-runtime.json"), "{}");
    write_fixture(&python, "#!/bin/sh\nexit 0\n");
    write_fixture(&amdhip, "stub");
    write_fixture(&hipblas, "stub");

    let manifest = serde_json::json!({
        "runtime_key": key,
        "runtime_id": key,
        "channel": "release",
        "format": "wheel",
        "family": "gfx94X-dcgpu",
        "family_source": "e2e",
        "version": "7.13.0",
        "install_root": install_root.display().to_string(),
        "selected_artifact_url": "https://example.invalid/e2e.whl",
        "python_executable": python.display().to_string(),
        "rocm_sdk": {
            "import_ok": true,
            "root_path": sdk_root.display().to_string(),
            "bin_path": sdk_bin.display().to_string(),
            "resolved_libraries": [
                {"shortname": "amdhip64", "paths": [amdhip.display().to_string()]},
                {"shortname": "hipblas", "paths": [hipblas.display().to_string()]},
            ],
        },
        "installed_at_unix_ms": 1,
    });

    let registry = data.join("runtimes").join("registry");
    std::fs::create_dir_all(&registry)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", registry.display()));
    std::fs::write(
        registry.join(format!("{key}.json")),
        serde_json::to_vec_pretty(&manifest).expect("manifest serialises"),
    )
    .unwrap_or_else(|e| panic!("failed to write the planted runtime manifest: {e}"));
}

/// Combined stdout+stderr of the recorded `rocm` invocation. The refusal is an
/// `anyhow` error printed to stderr, so both streams are searched.
fn refusal_text(world: &E2eWorld) -> String {
    let stdout = world.cli_output.clone().unwrap_or_default();
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    format!("{stdout}\n{stderr}")
}

#[given("a ready ROCm runtime with a ComfyUI checkout pending dependencies")]
async fn ready_install_pending_dependencies(world: &mut E2eWorld) {
    let install_root = install_root(world);

    // `validate_runtime_manifest_for_activation` requires this marker to exist
    // directly inside `install_root` for a non-read-only runtime.
    write_fixture(&install_root.join(".rocm-cli-runtime.json"), "{}");

    // A pre-existing source checkout with a non-torch-stack requirement makes
    // `install()` take the "use existing checkout" branch (no network) and
    // reach the `uv` install step (a requirements file with only
    // torch/torchvision/torchaudio would be filtered down to nothing and skip
    // it entirely).
    let source_dir = install_root.join("apps").join("comfyui").join("source");
    write_fixture(&source_dir.join("requirements.txt"), "numpy==1.26.0\n");

    // Fake Python interpreter: succeeds and prints valid (empty) JSON, so the
    // torch-stack constraint probe that runs before `uv` passes cleanly.
    let python = root(world)
        .join("comfyui-fixture")
        .join("python")
        .join("rocm-python");
    write_shim(&python, "#!/bin/sh\nprintf '{}'\n");

    // A `rocm_sdk` probe that satisfies `validate_rocm_sdk_runtime_probe`:
    // importable, with an existing root/bin dir and resolved amdhip64/hipblas
    // libraries.
    let sdk_root = install_root.join("rocm_sdk").join("root");
    let sdk_bin = install_root.join("rocm_sdk").join("bin");
    std::fs::create_dir_all(&sdk_root).expect("failed to create fake rocm_sdk root dir");
    std::fs::create_dir_all(&sdk_bin).expect("failed to create fake rocm_sdk bin dir");
    let amdhip64 = sdk_root.join("libamdhip64.so");
    let hipblas = sdk_root.join("libhipblas.so");
    write_fixture(&amdhip64, "");
    write_fixture(&hipblas, "");

    let registry = root(world).join("data").join("runtimes").join("registry");
    let manifest = serde_json::to_string_pretty(&serde_json::json!({
        "runtime_key": RUNTIME_KEY,
        "runtime_id": RUNTIME_KEY,
        "channel": "release",
        "format": "wheel",
        "family": "gfx94X-dcgpu",
        "family_source": "e2e",
        "version": "7.14.0",
        "install_root": install_root,
        "selected_artifact_url": "https://example.invalid/e2e.whl",
        "installed_at_unix_ms": 1u64,
        "python_executable": python,
        "rocm_sdk": {
            "import_ok": true,
            "root_path": sdk_root,
            "bin_path": sdk_bin,
            "resolved_libraries": [
                {"shortname": "amdhip64", "paths": [amdhip64]},
                {"shortname": "hipblas", "paths": [hipblas]},
            ],
        },
    }))
    .expect("failed to serialize runtime manifest");
    write_fixture(&registry.join(format!("{RUNTIME_KEY}.json")), &manifest);
}

#[given("the ComfyUI dependency install with uv fails")]
async fn uv_install_fails(world: &mut E2eWorld) {
    let uv = root(world)
        .join("comfyui-fixture")
        .join("uv-bin")
        .join("uv");
    write_shim(&uv, "#!/bin/sh\nexit 7\n");
    world
        .command_env
        .push(("ROCM_CLI_UV_BINARY", uv.into_os_string()));
}

#[given("the ComfyUI dependency install with uv prints progress and succeeds")]
async fn uv_install_prints_progress_and_succeeds(world: &mut E2eWorld) {
    let uv = root(world)
        .join("comfyui-fixture")
        .join("uv-bin")
        .join("uv");
    write_shim(&uv, "#!/bin/sh\necho 'Resolved 3 packages'\nexit 0\n");
    world
        .command_env
        .push(("ROCM_CLI_UV_BINARY", uv.into_os_string()));

    // The success path runs past `uv` into `probe_comfyui`'s post-install GPU
    // check, which shells out to the runtime's Python a second time with two
    // path arguments (a generated probe script, then where to write its JSON
    // result) rather than `-c <script>` like the pre-install torch-stack probe.
    // The fixture Python from the `Given` above only answers the `-c` form, so
    // it must be replaced here with one that answers both: unlike
    // `comfyui-01`, this scenario runs `install()` far enough to reach that
    // second call.
    let python = root(world)
        .join("comfyui-fixture")
        .join("python")
        .join("rocm-python");
    write_gpu_probe_shim(&python);
}

#[when("the user installs ComfyUI")]
async fn install_comfyui(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(
        world,
        &["comfyui", "install", "--runtime-id", RUNTIME_KEY],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the CLI fails and names the install log it wrote")]
async fn cli_names_install_log(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    let rc = world.cli_rc.unwrap_or(0);
    assert!(
        rc != 0,
        "expected `rocm comfyui install` to fail, got rc={rc}\nstderr:\n{stderr}"
    );

    let logs_dir = install_root(world)
        .join("apps")
        .join("comfyui")
        .join("logs");
    let entries: Vec<_> = std::fs::read_dir(&logs_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", logs_dir.display()))
        .map(|entry| entry.expect("failed to read log dir entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("install-"))
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("log"))
        })
        .collect();
    assert!(
        entries.len() == 1,
        "expected exactly one install log under {}, found {entries:?}",
        logs_dir.display()
    );
    let log_path = entries[0].display().to_string();

    assert!(
        stderr.contains("install ComfyUI dependencies: uv exited with"),
        "expected the uv-failure message in stderr, got:\n{stderr}"
    );
    assert!(
        stderr.contains(&log_path),
        "expected stderr to name the install log {log_path}, got:\n{stderr}"
    );
}

#[then("the CLI succeeds and shows the install progress")]
async fn cli_succeeds_and_shows_progress(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    let rc = world.cli_rc.unwrap_or(-1);
    assert!(
        rc == 0,
        "expected `rocm comfyui install` to succeed, got rc={rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // This is the non-TTY fallback under test: the `AnimatedSpinner` is a
    // no-op off a terminal, so `uv`'s own stdout must be the thing that
    // proves the install wasn't silent for its whole run.
    assert!(
        stdout.contains("Resolved 3 packages"),
        "expected uv's progress output to be streamed through to stdout, got:\n{stdout}"
    );
}

#[given("two ready ROCm runtimes and no active default")]
async fn plant_two_ready_runtimes(world: &mut E2eWorld) {
    // No `active.json` and no `activate` step: with two ready runtimes and no
    // configured default, the CLI must refuse to guess rather than auto-select.
    let data = data_dir(world);
    for key in RUNTIME_KEYS {
        plant_ready_runtime(&data, key);
    }
}

#[when("the user installs ComfyUI without choosing a runtime")]
async fn install_comfyui_without_runtime(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["comfyui", "install"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("ComfyUI install is refused as ambiguous")]
async fn comfyui_install_refused(world: &mut E2eWorld) {
    let text = refusal_text(world);
    assert_ne!(
        world.cli_rc,
        Some(0),
        "expected a non-zero refusal, got rc={:?}\n{text}",
        world.cli_rc
    );
    assert!(
        text.contains("Multiple ROCm runtimes are ready"),
        "expected the ambiguity refusal, got:\n{text}"
    );
}

#[then("the refusal offers the /runtimes picker")]
async fn refusal_offers_runtimes_picker(world: &mut E2eWorld) {
    let text = refusal_text(world);
    assert!(
        text.contains("/runtimes"),
        "refusal should point to the TUI `/runtimes` picker, got:\n{text}"
    );
}

#[then("the refusal names the --runtime-id flag")]
async fn refusal_names_runtime_id(world: &mut E2eWorld) {
    let text = refusal_text(world);
    assert!(
        text.contains("--runtime-id"),
        "refusal should name the `--runtime-id` flag, got:\n{text}"
    );
}

#[then("the refusal names rocm runtimes activate")]
async fn refusal_names_activate(world: &mut E2eWorld) {
    let text = refusal_text(world);
    assert!(
        text.contains("rocm runtimes activate"),
        "refusal should name the durable `rocm runtimes activate` remedy, got:\n{text}"
    );
}

#[then("the refusal lists both runtime keys")]
async fn refusal_lists_both_keys(world: &mut E2eWorld) {
    let text = refusal_text(world);
    for key in RUNTIME_KEYS {
        assert!(
            text.contains(key),
            "refusal should list runtime key `{key}`, got:\n{text}"
        );
    }
}

/// Pacing knobs for `comfyui-04`'s download server, mirroring
/// `therock_steps.rs`'s `PACED_TARBALL_*` constants: large enough that several
/// chunk boundaries land before the transfer completes, slow enough per chunk
/// that the PTY's poll cadence reliably samples an intermediate frame. Smaller
/// than TheRock's tarball fixture since there is no extraction phase here to
/// also keep observable — only the download needs to take a moment.
const PACED_ARCHIVE_PAYLOAD_BYTES: usize = 8_000_000;
const PACED_ARCHIVE_CHUNK_BYTES: usize = 650_000;
const PACED_ARCHIVE_CHUNK_DELAY: Duration = Duration::from_millis(150);
/// Wait budget for this scenario's PTY assertions, matching
/// `therock_steps.rs`'s file-local `PTY_SCREEN_TIMEOUT` convention.
const PTY_SCREEN_TIMEOUT: Duration = Duration::from_secs(30);

#[given("a paced ComfyUI source archive fixture")]
async fn paced_comfyui_source_archive_fixture(world: &mut E2eWorld) {
    let data = data_dir(world);
    plant_ready_runtime(&data, RUNTIME_KEY);

    // `probe_comfyui`'s post-install GPU check runs unconditionally in
    // `install()`, regardless of whether the `uv` block ran, so the runtime's
    // stub Python must answer it — `plant_ready_runtime`'s default (`exit 0`,
    // no output) is not enough. This scenario's requirements.txt (torch-stack
    // only, below) empties the dependency list and skips the `-c` probe, but
    // the shim answers both forms anyway for parity with the other fixtures.
    let python = data
        .join("runtimes")
        .join("roots")
        .join(RUNTIME_KEY)
        .join("bin")
        .join("python3");
    write_gpu_probe_shim(&python);

    // Build a real `.tar.gz`: one top-level directory holding a
    // `requirements.txt` naming only the torch stack, so `install()`'s
    // dependency filter empties the spec list and skips `uv` entirely — this
    // scenario is about the download spinner, not the dependency install.
    // Padded with high-entropy filler (`deterministic_payload`) so the paced
    // server has enough incompressible bytes to stream in more than one
    // chunk.
    let build_dir = root(world).join("comfyui-fixture").join("archive-build");
    let source_dir = build_dir.join("ComfyUI-master");
    std::fs::create_dir_all(&source_dir).expect("failed to create ComfyUI source directory");
    write_fixture(
        &source_dir.join("requirements.txt"),
        "torch==2.4.0\ntorchvision==0.19.0\ntorchaudio==2.4.0\n",
    );
    std::fs::write(
        source_dir.join("payload.bin"),
        deterministic_payload(PACED_ARCHIVE_PAYLOAD_BYTES),
    )
    .expect("failed to write archive filler payload");
    let contents = build_gzip_tarball(&build_dir, "comfyui-source.tar.gz", "ComfyUI-master").await;

    let served = root(world).join("comfyui-fixture").join("archive-serve");
    std::fs::create_dir_all(&served).expect("failed to create the archive fixture serve root");
    world.paced_download_server = Some(PacedDownloadServer::start(
        &served,
        "archive/comfyui-source.tar.gz",
        contents,
        PACED_ARCHIVE_CHUNK_BYTES,
        PACED_ARCHIVE_CHUNK_DELAY,
    ));
    let base = world
        .paced_download_server
        .as_ref()
        .expect("paced download server was just started")
        .base_url();
    world.command_env.push((
        "ROCM_CLI_COMFYUI_SOURCE_ARCHIVE_URL_OVERRIDE",
        format!("{base}/archive/comfyui-source.tar.gz").into(),
    ));
}

#[when("the user installs ComfyUI under a real terminal")]
async fn install_comfyui_under_pty(world: &mut E2eWorld) {
    let mut session =
        TuiSession::spawn(world, &["comfyui", "install", "--runtime-id", RUNTIME_KEY])
            .unwrap_or_else(|e| panic!("failed to spawn `rocm comfyui install` under a pty: {e}"));
    // The completion report is only ~10 lines today, safely inside the default
    // 24-row screen (0 lines of scrollback) — but nothing guards against that
    // growing and silently turning `assert_comfyui_spinner_line_cleared`'s
    // negative check into a tautology, the same trap `therock_steps.rs`'s
    // identical check hit once its own (much longer) summary was added. Grow
    // rows now, matching that fix, rather than waiting for it to recur here.
    //
    // As in `therock_steps.rs`: issued right after spawn with no
    // synchronization point, on the assumption that the child's fork/exec
    // and first repaint take longer than this resize call.
    session
        .grow_rows(60)
        .unwrap_or_else(|e| panic!("failed to grow the pty's row count: {e}"));
    world.tui = Some(session);
}

#[then("the terminal shows an intermediate ComfyUI download progress frame")]
async fn assert_intermediate_comfyui_download_progress_frame(world: &mut E2eWorld) {
    let session = world
        .tui
        .as_mut()
        .expect("no pty session for the ComfyUI install");
    session
        .assert_intermediate_download_progress_frame("the ComfyUI install", PTY_SCREEN_TIMEOUT)
        .await;
}

#[then("the ComfyUI install exits cleanly")]
async fn assert_comfyui_install_exits_cleanly(world: &mut E2eWorld) {
    let session = world
        .tui
        .as_mut()
        .expect("no pty session for the ComfyUI install");
    session
        .assert_exits_cleanly("the ComfyUI install", PTY_SCREEN_TIMEOUT)
        .await;
}

#[then("the final terminal screen shows no ComfyUI download spinner line")]
async fn assert_comfyui_spinner_line_cleared(world: &mut E2eWorld) {
    let session = world
        .tui
        .as_ref()
        .expect("no pty session for the ComfyUI install");
    let screen = session.screen_text();
    assert!(
        !screen.contains("Fetching ComfyUI source archive"),
        "download spinner line was not cleared on completion:\n{screen}"
    );
}

// --- comfyui-05: installing ComfyUI must not damage the managed ROCm runtime ---

/// Report the installed torch distribution's version string via `importlib.metadata`
/// — WITHOUT importing torch. Emits JSON `{version}` (e.g. `2.7.0+rocm6.4` for a
/// ROCm build, `2.7.0+cu128` for a CUDA build) or `{error}` when no torch
/// distribution is installed.
///
/// Deliberately does not `import torch`: importing it loads the ROCm/CUDA shared
/// libraries, which need the runtime's `LD_LIBRARY_PATH`/`ROCM_PATH` set up (the
/// product runs its own torch probe *with* that env, ours runs the interpreter
/// bare). The `+rocm` / `+cu` local-version label in the dist metadata is the
/// definitive ROCm-vs-CUDA discriminator and is readable with no native load — so
/// a bare interpreter suffices, and a runtime whose torch is present but whose
/// native libs aren't on our env no longer reads as "not a ROCm build".
const TORCH_DIST_PROBE: &str = "import json,sys\n\
     from importlib import metadata\n\
     out={}\n\
     try:\n\
     \x20 out['version']=metadata.version('torch')\n\
     except Exception as ex:\n\
     \x20 out['error']=type(ex).__name__+': '+str(ex)\n\
     sys.stdout.write(json.dumps(out))\n";

/// Locate the managed runtime's venv interpreter. `rocm runtimes list` prints an
/// `install_root: <path>` line for each installed runtime; the interpreter lives
/// under a `bin/python` (Unix) / `Scripts/python.exe` (Windows) inside that tree.
/// The exact env sub-layout is an internal detail, so search for the interpreter
/// rather than reconstruct the path — black-box, and tolerant of layout changes.
///
/// Reads `runtimes list` rather than `examine`: examine only prints a `Folder:`
/// line for the *active* runtime and takes a different branch when none is marked
/// active, so it is not a reliable source for the install root (this cost a GPU
/// dispatch — the scenario panicked on a missing `Folder:` there). `runtimes list`
/// prints `install_root:` for every installed runtime unconditionally.
fn active_runtime_python(world: &E2eWorld) -> PathBuf {
    let (listing, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    let roots: Vec<&str> = listing
        .lines()
        .filter_map(|l| l.trim().strip_prefix("install_root:"))
        .map(str::trim)
        .collect();
    assert_eq!(
        roots.len(),
        1,
        "expected exactly one installed runtime in the isolated world, found {}:\n{listing}",
        roots.len()
    );
    let root = roots[0];
    find_venv_python(Path::new(root)).unwrap_or_else(|| {
        panic!("could not locate a venv python under the runtime install_root {root}")
    })
}

/// Locate a `bin/python` (Unix) or `Scripts/python.exe` (Windows) under `root`.
///
/// The documented layout is probed FIRST: `root` itself is the initial frontier
/// entry, so a managed runtime — whose interpreter is exactly `<install_root>/bin/
/// python` — is found on the first iteration with no directory traversal at all.
/// The depth-first walk below is only a fallback for a tree that does not match, and
/// is depth-capped so a pathological one cannot hang the scenario rather than
/// being a cost the normal path pays.
fn find_venv_python(root: &Path) -> Option<PathBuf> {
    #[cfg(windows)]
    let (bin, exe) = ("Scripts", "python.exe");
    #[cfg(not(windows))]
    let (bin, exe) = ("bin", "python");
    let mut frontier = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = frontier.pop() {
        let candidate = dir.join(bin).join(exe);
        if candidate.is_file() {
            return Some(candidate);
        }
        if depth >= 6 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                frontier.push((entry.path(), depth + 1));
            }
        }
    }
    None
}

/// The installed torch distribution's version string, or `None` if no torch
/// distribution is installed. Reads dist metadata without importing torch (see
/// [`TORCH_DIST_PROBE`]), so it works against a bare interpreter.
///
/// The local-version label identifies the build. A CUDA wheel is unmistakable:
/// `+cu` (e.g. `2.7.0+cu128`). A ROCm build is NOT reliably `+rocm`, though —
/// TheRock's managed torch labels the local version with a git hash
/// (`2.11.0+gitd0c8b1f`, measured on the MI300X lane), so callers judge "ROCm" as
/// "torch is present and is NOT a CUDA build", which is exactly the flip the
/// EAI-8051 corruption would cause.
fn torch_version(python: &Path) -> Result<String, String> {
    let output = std::process::Command::new(python)
        .args(["-c", TORCH_DIST_PROBE])
        .output()
        .unwrap_or_else(|e| panic!("failed to run runtime python {}: {e}", python.display()));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let data: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|_| {
        panic!("torch version probe returned non-JSON:\nstdout: {stdout}\nstderr: {stderr}")
    });
    match data.get("version").and_then(serde_json::Value::as_str) {
        Some(v) => Ok(v.to_owned()),
        None => Err(data
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("probe reported no version and no error")
            .to_owned()),
    }
}

/// Whether `version` is a CUDA torch build (carries a `+cuNNN` local label). Used
/// only for the baseline PREMISE check — that the runtime didn't start out on a
/// CUDA torch. The post-install invariant is stronger: the torch version must be
/// byte-for-byte unchanged (a ComfyUI install must not replace the runtime's torch
/// at all), which also catches a swap to a plain non-`+cu` wheel (e.g. the observed
/// `2.11.0+gitd0c8b1f` → `2.13.0`) that this label check alone would miss.
fn is_cuda_torch(version: &str) -> bool {
    version.to_ascii_lowercase().contains("+cu")
}

/// Enumerate installed distributions via `importlib.metadata` and emit their names
/// as a JSON array. Used instead of `pip list` because uv-created managed runtimes
/// have no `pip` module — `python -m pip` there exits non-zero with empty stdout,
/// which a naive reader would misread as "no packages installed" and pass the
/// nvidia check while the runtime is actually corrupted. `importlib.metadata` is in
/// the stdlib, so it is always present; the probe emits `{names}` on success or
/// `{error}` on failure so the caller can fail loudly rather than treat a broken
/// probe as a clean result.
const DISTRIBUTIONS_PROBE: &str = "import json,sys\n\
     out={}\n\
     try:\n\
     \x20 from importlib import metadata\n\
     \x20 out['names']=sorted({(d.metadata['Name'] or '') for d in metadata.distributions()})\n\
     except Exception as ex:\n\
     \x20 out['error']=type(ex).__name__+': '+str(ex)\n\
     sys.stdout.write(json.dumps(out))\n";

/// Every distribution name installed in the interpreter's environment, sorted.
/// Panics if the interpreter cannot be run or the probe reports an error — a probe
/// that cannot enumerate packages must NOT read as an empty environment, which
/// would pass the nvidia check on a runtime it never actually inspected.
fn installed_distributions(python: &Path) -> Vec<String> {
    let output = std::process::Command::new(python)
        .args(["-c", DISTRIBUTIONS_PROBE])
        .output()
        .unwrap_or_else(|e| panic!("failed to run runtime python {}: {e}", python.display()));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let data: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|_| {
        panic!("distributions probe returned non-JSON:\nstdout: {stdout}\nstderr: {stderr}")
    });
    if let Some(error) = data.get("error").and_then(serde_json::Value::as_str) {
        panic!(
            "could not enumerate installed distributions on {}: {error}",
            python.display()
        );
    }
    let names = data
        .get("names")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("distributions probe returned no 'names' array:\n{stdout}"));
    names
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_owned)
        .collect()
}

/// The `nvidia-*` CUDA distributions installed in the interpreter's environment.
/// A ROCm runtime should have none; ComfyUI's install dragging any in is the
/// EAI-8051 defect.
fn nvidia_distributions(python: &Path) -> Vec<String> {
    installed_distributions(python)
        .into_iter()
        .filter(|name| name.to_ascii_lowercase().starts_with("nvidia-"))
        .collect()
}

#[given("an isolated machine with a managed ROCm runtime")]
async fn setup_isolated_runtime(world: &mut E2eWorld) {
    // DELIBERATELY do NOT call `world.use_shared_runtimes()`: this scenario may
    // corrupt the runtime (that is the bug it pins), so it must own a private,
    // throwaway runtime prefix. Each World already has isolated ROCM_CLI_* dirs,
    // so a plain `install sdk` here lands in this scenario's own tree.
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    if stdout.contains("installed: none") {
        crate::run_rocm_ok(world, &["install", "sdk"]);
    }
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    assert!(
        !stdout.contains("installed: none"),
        "no managed runtime is active after install:\n{stdout}"
    );
}

#[given("the runtime's torch is a ROCm build")]
async fn assert_baseline_rocm_torch(world: &mut E2eWorld) {
    let python = active_runtime_python(world);
    let version = torch_version(&python);
    assert!(
        version.as_ref().is_ok_and(|v| !is_cuda_torch(v)),
        "baseline runtime torch is absent or already a CUDA build; scenario premise absent \
         (torch version: {version:?}, python: {})",
        python.display()
    );
    let distributions = installed_distributions(&python);
    assert!(
        !distributions
            .iter()
            .any(|name| name.to_ascii_lowercase().starts_with("nvidia-")),
        "runtime already has nvidia-* distributions before ComfyUI install; premise absent"
    );
    // Record the exact baseline version so the post-install step can require it to
    // be unchanged (see `assert_torch_still_rocm`), and the baseline package set so
    // it can require the install to have actually added something (see
    // `assert_dependencies_installed`).
    world.comfyui_baseline_torch = version.ok();
    world.comfyui_baseline_distributions = Some(distributions);
}

#[when("the user installs ComfyUI into the isolated runtime")]
async fn user_installs_comfyui(world: &mut E2eWorld) {
    // Capture the outcome rather than asserting here: the exit code is checked by
    // its own Then step, so a failure is reported as that step failing rather than
    // as a mid-scenario panic in the action.
    let (stdout, stderr, rc) = crate::run_rocm(world, &["comfyui", "install"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the install succeeds")]
async fn assert_install_succeeded(world: &mut E2eWorld) {
    // The premise for every invariant below. `comfyui::install` bails early on
    // several paths (no managed runtime, a runtime that isn't `ready`, a non-wheel
    // format, a failed source download or `uv` acquisition); on any of those the
    // runtime is TRIVIALLY unchanged and the torch/nvidia assertions would pass
    // having exercised nothing. It is also the check that catches a revert of
    // #298: without the torch constraint, the post-install GPU probe in
    // `comfyui::install` bails, so the install exits non-zero and the torch and
    // nvidia steps never run. Do not relax it as "just a premise".
    let rc = world.cli_rc.expect("no ComfyUI install was run");
    assert_eq!(
        rc,
        0,
        "{}",
        e2e_cucumber::cli_failure_report(
            &["comfyui", "install"],
            rc,
            world.cli_output.as_deref().unwrap_or(""),
            world.cli_stderr.as_deref().unwrap_or(""),
        )
    );
}

#[then("ComfyUI's dependencies were installed into the runtime")]
async fn assert_dependencies_installed(world: &mut E2eWorld) {
    // Closes the last vacuity path a zero exit code leaves open. `comfyui::install`
    // guards the whole `uv` install with `if !packages.is_empty()`
    // (`apps/rocm/src/comfyui.rs`), so an empty filtered requirement list skips it
    // and still exits 0 — leaving the runtime untouched and every invariant below
    // passing having installed nothing.
    //
    // Deliberately asserts the package set GREW rather than naming an expected
    // dependency: ComfyUI's requirements drift upstream independently of this
    // contract, so a named package would rot, while "the install put something in
    // the runtime" is exactly the premise the invariants need and cannot go stale.
    let python = active_runtime_python(world);
    let baseline = world
        .comfyui_baseline_distributions
        .as_ref()
        .expect("no baseline distribution set was captured");
    let after = installed_distributions(&python);
    assert!(
        after.iter().any(|name| !baseline.contains(name)),
        "ComfyUI install added no distributions to the runtime, so it installed \
         nothing and the runtime-preservation checks would pass vacuously \
         ({} distributions before and after, python: {})",
        after.len(),
        python.display()
    );
}

#[then("the runtime's torch is unchanged")]
async fn assert_torch_still_rocm(world: &mut E2eWorld) {
    let python = active_runtime_python(world);
    let version = torch_version(&python);
    let baseline = world
        .comfyui_baseline_torch
        .as_deref()
        .expect("no baseline torch version was captured");
    assert_eq!(
        version.as_deref(),
        Ok(baseline),
        "ComfyUI install replaced the managed runtime's torch \
         (before: {baseline}, after: {version:?}, python: {})",
        python.display()
    );
}

#[then("no CUDA nvidia packages were added to the runtime")]
async fn assert_no_nvidia_packages(world: &mut E2eWorld) {
    let python = active_runtime_python(world);
    let nvidia = nvidia_distributions(&python);
    assert!(
        nvidia.is_empty(),
        "ComfyUI install added CUDA nvidia-* distributions to the ROCm runtime: {}",
        nvidia.join(", ")
    );
}
