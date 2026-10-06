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
//! `comfyui-05` to `comfyui-08` cover `--reinstall` over a used install: one
//! file of the user's in every entry of `source/` a reinstall keeps, plus
//! release code. The new release comes from the same loopback server
//! (unpaced), at a path that 404s for the failed-download case. Each `Then`
//! that checks a printed claim about what was kept also checks the files.
//! `comfyui-08` plants a running stand-in in the saved state and runs the
//! `rocm comfyui stop` the refusal names before reinstalling again.
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

/// The user's own content in a used ComfyUI install, relative to `source/`:
/// one file in each entry a reinstall keeps.
const USER_FILES: [(&str, &str); 9] = [
    ("models/checkpoints/my-model.safetensors", "user model"),
    ("user/default/workflows/my-workflow.json", "user workflow"),
    ("output/ComfyUI_00001_.png", "user image"),
    ("input/my-upload.png", "user upload"),
    ("custom_nodes/my-node/__init__.py", "user node"),
    ("datasets/my-set/0001.png", "user dataset"),
    ("extra_model_paths.yaml", "user model paths"),
    // No ComfyUI release ships these; a reinstall leaves them where they are.
    (".git/HEAD", "ref: refs/heads/master"),
    ("styles.csv", "user styles"),
];

/// The kept-entries list the CLI prints, in its order.
const KEPT_LIST: &str =
    "models, user, output, input, custom_nodes, datasets, extra_model_paths.yaml";

/// The entries no release shipped, as the CLI lists them.
const LEFT_IN_PLACE_LIST: &str = ".git, styles.csv";

/// `source/` of the ComfyUI install that belongs to the planted runtime.
fn comfyui_source_dir(world: &E2eWorld) -> PathBuf {
    data_dir(world)
        .join("runtimes")
        .join("roots")
        .join(RUNTIME_KEY)
        .join("apps")
        .join("comfyui")
        .join("source")
}

fn assert_user_files_intact(world: &E2eWorld) {
    let source = comfyui_source_dir(world);
    for (relative, contents) in USER_FILES {
        let path = source.join(relative);
        assert_eq!(
            std::fs::read_to_string(&path).ok().as_deref(),
            Some(contents),
            "the user's {} must survive the reinstall",
            path.display()
        );
    }
}

/// `run_rocm_with_scenario_env` consumes `command_env`; a scenario that runs
/// `rocm` more than once (`comfyui-08`) needs the archive-URL override on
/// every run, or a later run would fetch the real ComfyUI from the network.
fn run_keeping_scenario_env(world: &mut E2eWorld, args: &[&str]) -> (String, String, i32) {
    let env = world.command_env.clone();
    let result = crate::run_rocm_with_scenario_env(world, args);
    world.command_env = env;
    result
}

fn run_comfyui_reinstall(world: &mut E2eWorld, extra: &[&str]) {
    let mut args = vec![
        "comfyui",
        "install",
        "--runtime-id",
        RUNTIME_KEY,
        "--reinstall",
    ];
    args.extend_from_slice(extra);
    let (stdout, stderr, rc) = run_keeping_scenario_env(world, &args);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[given("a ComfyUI install holding the user's models, workflows, images and custom nodes")]
async fn used_comfyui_install(world: &mut E2eWorld) {
    let data = data_dir(world);
    plant_ready_runtime(&data, RUNTIME_KEY);
    // Answers the post-install GPU check `install()` runs after the swap.
    write_gpu_probe_shim(
        &data
            .join("runtimes")
            .join("roots")
            .join(RUNTIME_KEY)
            .join("bin")
            .join("python3"),
    );
    let source = comfyui_source_dir(world);
    for (relative, contents) in [
        ("main.py", "old release"),
        ("requirements.txt", "torch\n"),
        ("comfy/dropped_upstream.py", "old release"),
    ]
    .into_iter()
    .chain(USER_FILES)
    {
        write_fixture(&source.join(relative), contents);
    }
}

/// Serves a small release archive. Its `requirements.txt` names only the
/// torch stack, so the dependency install is filtered down to nothing and
/// skipped — this scenario is about the swap, not the `uv` step.
async fn serve_comfyui_release(world: &mut E2eWorld, served_path: &str, extra: &[(&str, &str)]) {
    let build_dir = root(world).join("comfyui-fixture").join("release-build");
    let release = build_dir.join("ComfyUI-master");
    for (relative, contents) in [
        ("main.py", "new release"),
        ("requirements.txt", "torch==2.4.0\n"),
        ("models/checkpoints/put_checkpoints_here", ""),
        ("output/_output_images_will_be_put_here", ""),
        ("custom_nodes/websocket_image_save.py", "new release"),
        ("comfy/added_upstream.py", "new release"),
    ]
    .iter()
    .chain(extra)
    {
        write_fixture(&release.join(relative), contents);
    }
    let contents = build_gzip_tarball(&build_dir, "comfyui-release.tar.gz", "ComfyUI-master").await;
    let served = root(world).join("comfyui-fixture").join("release-serve");
    std::fs::create_dir_all(&served).expect("failed to create the archive fixture serve root");
    world.paced_download_server = Some(PacedDownloadServer::start(
        &served,
        "archive/comfyui-release.tar.gz",
        contents,
        1 << 20,
        Duration::ZERO,
    ));
    let base = world
        .paced_download_server
        .as_ref()
        .expect("archive server was just started")
        .base_url();
    world.command_env.push((
        "ROCM_CLI_COMFYUI_SOURCE_ARCHIVE_URL_OVERRIDE",
        format!("{base}/{served_path}").into(),
    ));
}

#[given("a newer ComfyUI release is available to download")]
async fn newer_comfyui_release(world: &mut E2eWorld) {
    serve_comfyui_release(world, "archive/comfyui-release.tar.gz", &[]).await;
}

#[given("the ComfyUI release download fails")]
async fn comfyui_release_download_fails(world: &mut E2eWorld) {
    // The server is up but the archive is not at this path: a 404, which the
    // downloader does not retry.
    serve_comfyui_release(world, "archive/missing.tar.gz", &[]).await;
}

#[when("the user reinstalls ComfyUI")]
async fn reinstall_comfyui(world: &mut E2eWorld) {
    run_comfyui_reinstall(world, &[]);
}

#[when("the user previews reinstalling ComfyUI")]
async fn preview_comfyui_reinstall(world: &mut E2eWorld) {
    run_comfyui_reinstall(world, &["--dry-run"]);
}

#[then("the reinstall reports the user's folders as kept and they still hold the user's files")]
async fn reinstall_reports_and_keeps(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    assert_eq!(
        world.cli_rc,
        Some(0),
        "expected the reinstall to succeed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // The claim and the state it describes, asserted together.
    assert_user_files_intact(world);
    assert!(
        stdout.contains(&format!("  kept: {KEPT_LIST}\n")),
        "the reinstall must name what it kept, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&format!("  left in place: {LEFT_IN_PLACE_LIST}\n")),
        "the reinstall must name what it left alone, got:\n{stdout}"
    );
}

#[then("ComfyUI's code is the newer release")]
async fn comfyui_code_is_newer_release(world: &mut E2eWorld) {
    let source = comfyui_source_dir(world);
    assert_eq!(
        std::fs::read_to_string(source.join("main.py"))
            .ok()
            .as_deref(),
        Some("new release")
    );
    assert!(
        !source.join("comfy/dropped_upstream.py").exists(),
        "code the new release does not ship must be gone"
    );
    let app_root = source.parent().expect("source has a parent");
    let leftovers: Vec<_> = std::fs::read_dir(app_root)
        .expect("failed to read the ComfyUI folder")
        .map(|entry| entry.expect("dir entry").file_name())
        .filter(|name| name != "source" && name.to_string_lossy().starts_with("source"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "the previous code must be removed once the swap is done: {leftovers:?}"
    );
}

/// What the release the planted install came from shipped, as rocm-cli
/// records it when it installs one.
const INSTALLED_RELEASE_RECORD: &str = "comfy\nmain.py\nrequirements.txt";

#[given("the user keeps an app folder there that the installed ComfyUI release did not ship")]
async fn user_app_folder(world: &mut E2eWorld) {
    let source = comfyui_source_dir(world);
    write_fixture(&source.join("app/settings.json"), "user app");
    write_fixture(
        &source.join(".rocm-cli-release-entries"),
        INSTALLED_RELEASE_RECORD,
    );
}

#[given("a newer ComfyUI release that ships an app folder is available to download")]
async fn newer_comfyui_release_with_app(world: &mut E2eWorld) {
    serve_comfyui_release(
        world,
        "archive/comfyui-release.tar.gz",
        &[("app/__init__.py", "new release")],
    )
    .await;
}

#[then(
    "the reinstall reports the user's app folder as set aside and it still holds the user's files"
)]
async fn reinstall_reports_set_aside(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    let renamed = stdout
        .lines()
        .find_map(|line| line.strip_prefix("  set aside: app (now "))
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or_else(|| panic!("the reinstall must say where the user's app/ went:\n{stdout}"));
    assert!(
        renamed.starts_with("app.rocm-cli-kept-") && !renamed.contains('/'),
        "unexpected set-aside name {renamed:?}"
    );
    // The claim and the state it describes, asserted together.
    let source = comfyui_source_dir(world);
    assert_eq!(
        std::fs::read_to_string(source.join(renamed).join("settings.json"))
            .ok()
            .as_deref(),
        Some("user app"),
        "the folder the reinstall names must hold the user's files"
    );
    assert_eq!(
        std::fs::read_to_string(source.join("app/__init__.py"))
            .ok()
            .as_deref(),
        Some("new release"),
        "the new release's app/ is installed under its own name"
    );
    assert!(
        !source.join("app/settings.json").exists(),
        "the release's app/ must not be mixed with the user's"
    );
}

#[then("the reinstall lists the code it replaced")]
async fn reinstall_lists_replaced(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    // The planted install predates the release record, so the entries named
    // like the new release's code are replaced, and listed.
    assert!(
        stdout.contains("  replaced: comfy, main.py, requirements.txt\n"),
        "the reinstall must list what it replaced, got:\n{stdout}"
    );
    let source = comfyui_source_dir(world);
    assert_eq!(
        std::fs::read_to_string(source.join("requirements.txt"))
            .ok()
            .as_deref(),
        Some("torch==2.4.0\n"),
        "requirements.txt is listed as replaced, so it is the new release's"
    );
    assert!(!source.join("comfy/dropped_upstream.py").exists());
}

#[then("the reinstall fails")]
async fn comfyui_reinstall_fails(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    assert_ne!(
        world.cli_rc,
        Some(0),
        "expected the reinstall to fail\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("404"),
        "expected the failed download in stderr, got:\n{stderr}"
    );
}

#[then("the existing ComfyUI code and the user's files are untouched")]
async fn existing_comfyui_untouched(world: &mut E2eWorld) {
    let source = comfyui_source_dir(world);
    assert_eq!(
        std::fs::read_to_string(source.join("main.py"))
            .ok()
            .as_deref(),
        Some("old release"),
        "the existing ComfyUI code must be left in place"
    );
    assert!(source.join("comfy/dropped_upstream.py").is_file());
    assert_user_files_intact(world);
}

#[then("the preview says the ComfyUI code is replaced and names the kept folders")]
async fn preview_names_replaced_and_kept(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    assert_eq!(world.cli_rc, Some(0), "dry run failed:\n{stdout}");
    let source = comfyui_source_dir(world);
    assert!(
        stdout.contains(&format!(
            "  reinstall: replaces the ComfyUI code in {}\n",
            source.display()
        )),
        "the preview must say what is replaced, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&format!("  keeps: {KEPT_LIST}\n")),
        "the preview must name what is kept, got:\n{stdout}"
    );
    // Every folder the preview promises to keep exists in this install, so
    // the promise is about real content (the dry run itself changes nothing;
    // `comfyui-05` proves a real reinstall keeps them).
    for name in KEPT_LIST.split(", ") {
        assert!(
            source.join(name).exists(),
            "{name} is named as kept but missing"
        );
    }
}

#[then("the preview's install command includes --reinstall")]
async fn preview_command_includes_reinstall(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    let command = stdout
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("install command: "))
        .unwrap_or_else(|| panic!("no install command in the preview:\n{stdout}"));
    assert!(
        command.split_whitespace().any(|arg| arg == "--reinstall"),
        "the previewed command must be the reinstall that was asked for, got: {command}"
    );
}

/// Starts a detached long-lived process to stand in for a running ComfyUI and
/// records it the way `rocm comfyui start` does (`<data>/apps/comfyui/state/
/// running.json`), pointing at the planted install's `source/`. Detached via
/// the shell so that once `rocm comfyui stop` kills it, init reaps it and it
/// does not linger as a zombie the CLI's liveness check would still see. It
/// exits on its own if the scenario fails before stopping it.
#[given("the ComfyUI that rocm-cli started is running from that install")]
async fn comfyui_running_from_install(world: &mut E2eWorld) {
    let output = std::process::Command::new("sh")
        .args(["-c", "sleep 120 >/dev/null 2>&1 & echo $!"])
        .output()
        .expect("failed to start the stand-in ComfyUI process");
    let pid: u32 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("the shell printed the stand-in's pid");
    // A port nothing listens on, so the CLI's view rests on the process alone.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("failed to pick a free port")
        .port();
    let source = comfyui_source_dir(world);
    let state = serde_json::json!({
        "app_id": "comfyui",
        "url": format!("http://127.0.0.1:{port}"),
        "host": "127.0.0.1",
        "port": port,
        "pid": pid,
        "source_path": source,
        "python_executable": "python3",
        "log_path": source.parent().expect("source has a parent").join("logs").join("start.log"),
        "started_at_unix_ms": 1u64,
    });
    write_fixture(
        &data_dir(world)
            .join("apps")
            .join("comfyui")
            .join("state")
            .join("running.json"),
        &serde_json::to_string_pretty(&state).expect("state serialises"),
    );
}

#[then("the reinstall is refused and names rocm comfyui stop")]
async fn reinstall_refused_while_running(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    assert_ne!(
        world.cli_rc,
        Some(0),
        "expected the reinstall to be refused\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("ComfyUI is running from") && stderr.contains("`rocm comfyui stop`"),
        "the refusal must say ComfyUI is running and name `rocm comfyui stop`, got:\n{stderr}"
    );
}

#[when("the user stops ComfyUI")]
async fn stop_comfyui(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = run_keeping_scenario_env(world, &["comfyui", "stop"]);
    assert_eq!(
        rc, 0,
        "`rocm comfyui stop` failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

/// Leaves the installed folder as a reinstall killed part-way through
/// replacing the code would: the swap's marker present (naming what the user
/// had) and `main.py`, the first thing removed, gone. The swap's own unit test
/// (`interrupted_swap_converges_when_run_again`) interrupts it for real at
/// every step; this plants the end state so the CLI's handling of it can be
/// driven through the binary.
#[given("the reinstall was cut short while replacing ComfyUI's code")]
async fn reinstall_cut_short(world: &mut E2eWorld) {
    let source = comfyui_source_dir(world);
    write_fixture(
        &source.join(".rocm-cli-reinstall-in-progress"),
        &KEPT_LIST.replace(", ", "\n"),
    );
    std::fs::remove_file(source.join("main.py")).expect("main.py was installed");
}

#[when("the user starts ComfyUI")]
async fn start_comfyui(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = run_keeping_scenario_env(
        world,
        &["comfyui", "start", "--no-open-browser", "--port", "1"],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

const FINISH_ADVICE: &str = "a reinstall was interrupted; run `rocm comfyui install --runtime-id e2e-comfyui-runtime` to finish it";

#[then("start refuses and names the command that finishes the reinstall")]
async fn start_refuses_mid_swap(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    assert_ne!(world.cli_rc, Some(0), "start must refuse:\n{stderr}");
    assert!(
        stderr.contains(FINISH_ADVICE),
        "start must name the command that finishes the reinstall, got:\n{stderr}"
    );
}

/// Runs the command `start` named, taken from its own output rather than
/// restated, so the advice is what gets proven.
#[when("the user runs the command start named")]
async fn run_named_command(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    let command = stderr
        .split('`')
        .nth(1)
        .unwrap_or_else(|| panic!("no backticked command in:\n{stderr}"))
        .to_owned();
    let args: Vec<&str> = command
        .strip_prefix("rocm ")
        .unwrap_or_else(|| panic!("not a rocm command: {command}"))
        .split_whitespace()
        .collect();
    let (stdout, stderr, rc) = run_keeping_scenario_env(world, &args);
    assert_eq!(
        rc, 0,
        "`{command}` failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the user's own files are untouched")]
async fn user_files_untouched(world: &mut E2eWorld) {
    assert_user_files_intact(world);
}

#[then("ComfyUI status reports the interrupted reinstall")]
async fn status_reports_interruption(world: &mut E2eWorld) {
    let (stdout, _, rc) = run_keeping_scenario_env(world, &["comfyui", "status"]);
    assert_eq!(rc, 0, "{stdout}");
    assert!(
        stdout.contains(&format!("  note: {FINISH_ADVICE}\n")),
        "status must report the interrupted reinstall, got:\n{stdout}"
    );
}

#[then("ComfyUI status no longer reports an interrupted reinstall")]
async fn status_clear_of_interruption(world: &mut E2eWorld) {
    let (stdout, _, rc) = run_keeping_scenario_env(world, &["comfyui", "status"]);
    assert_eq!(rc, 0, "{stdout}");
    assert!(
        !stdout.contains("interrupted"),
        "status still reports the interruption:\n{stdout}"
    );
    assert!(
        !comfyui_source_dir(world)
            .join(".rocm-cli-reinstall-in-progress")
            .exists()
    );
}
