// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `comfyui.feature`.
//!
//! `comfyui-01` and `comfyui-02` drive a dependency install against an entirely
//! planted runtime: a `wheel` manifest with a satisfying `rocm_sdk` probe, a
//! pre-existing source checkout (so the CLI never attempts the real network
//! download of ComfyUI's source archive) and a `fake-python` stand-in
//! (`src/bin/fake-python.rs`) that answers both the pre-install torch-stack
//! version probe and the post-install GPU check. `comfyui-01` covers what
//! happens when the `uv` dependency install that follows all of that fails;
//! `comfyui-02` covers a `uv` install that succeeds, which runs on into the
//! post-install GPU check that `fake-python` also answers.
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
//! `comfyui-05` to `comfyui-13` cover `--reinstall` over a used install: one
//! file of the user's in every entry of `source/` a reinstall keeps, plus
//! release code. The new release comes from the same loopback server
//! (unpaced), at a path that 404s for the failed-download case. Each `Then`
//! that checks a printed claim about what was kept also checks the files.
//! `comfyui-08` plants a running stand-in in the saved state, with a port
//! held open so it reads as running, and runs the `rocm comfyui stop` the
//! refusal names before reinstalling again. `comfyui-12` plants it from the
//! loopback server once the download has started, so only the check just
//! before the swap can see it.
//! `comfyui-11` makes the runtime's GPU check fail after the swap, then pass,
//! and runs the command the failed reinstall names.
//! `comfyui-08` and `comfyui-09` also ask `rocm` for ComfyUI's status in plain
//! words, which is answered without a model.
//!
//! The planted registry manifests are plain JSON matching the CLI's on-disk
//! schema, not typed imports from the product crates. Two scenarios also write
//! files the CLI keeps inside `source/`, by name and format: `comfyui-09` the
//! marker an interrupted reinstall leaves (`.rocm-cli-reinstall-in-progress`),
//! because a kill cannot be timed from outside, and `comfyui-10` the record of
//! what the installed release shipped (`.rocm-cli-release-entries`), which an
//! install made by this CLI would have. `source_swap`'s unit tests interrupt
//! the swap for real at every step.

use std::path::{Path, PathBuf};
use std::time::Duration;

use cucumber::{given, then, when};
use e2e_cucumber::fixture_binary::fixture_target_name;
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
/// CLI shells out to.
///
/// Written via [`rocm_core::fake_executable::write_fake_executable`], not
/// `std::fs::write`: scenarios run concurrently, and a write fd another
/// scenario's spawn inherits makes the CLI's first run of the shim fail with
/// `Text file busy`.
///
/// comfyui-11 is the only remaining caller (its GPU-check failing transition,
/// via `gpu_check_fails`): every other scenario's runtime Python is the
/// Windows-portable `fake-python` `[[bin]]` planted by `install_fake_python`,
/// including comfyui-11's own passing-again half (`gpu_check_passes_again`).
fn write_shim(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create fixture directory");
    }
    rocm_core::fake_executable::write_fake_executable(path, body)
        .unwrap_or_else(|e| panic!("failed to write fake {}: {e}", path.display()));
}

/// Installs the `fake-uv` stand-in at `path`, configured via env vars it
/// reads (see `src/bin/fake-uv.rs`). Returns the env additions the invoking
/// `rocm` process needs so its own child `uv` process — which inherits
/// `rocm`'s environment — picks them up.
fn install_fake_uv(
    path: &Path,
    exit_code: i32,
    stdout_line: Option<&str>,
) -> Vec<(&'static str, std::ffi::OsString)> {
    crate::install_fixture_binary("fake-uv", path);
    let mut env = vec![("FAKE_UV_EXIT_CODE", exit_code.to_string().into())];
    if let Some(line) = stdout_line {
        env.push(("FAKE_UV_STDOUT", line.into()));
    }
    env
}

/// Installs the `fake-python` stand-in at `path` (see `src/bin/fake-python.rs`).
/// It always answers both of the runtime Python's invocation forms
/// (`probe_torch_stack_versions`'s `-c`, `probe_comfyui`'s script form), so
/// one install serves every scenario regardless of how far `install()` runs.
fn install_fake_python(path: &Path) {
    crate::install_fixture_binary("fake-python", path);
}

// A simplified stand-in for `rocm_core::runtime_rocm_library_filename` rather
// than imported: only the filename's look matters here, since readiness only
// checks `path.is_file()` on whatever the manifest lists (see
// `probe_has_resolved_library` in `apps/rocm/src/therock.rs`) — never the
// filename itself. Importing the product crate for that would buy nothing,
// so it stays a local stand-in.
fn rocm_library_filename(shortname: &str) -> String {
    if cfg!(windows) {
        format!("{shortname}.dll")
    } else {
        format!("lib{shortname}.so")
    }
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
    let python = install_root
        .join("bin")
        .join(fixture_target_name("python3"));
    let amdhip = sdk_bin.join(rocm_library_filename("amdhip64"));
    let hipblas = sdk_bin.join(rocm_library_filename("hipblas"));

    write_fixture(&install_root.join(".rocm-cli-runtime.json"), "{}");
    install_fake_python(&python);
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

    // `install_fake_python` answers both of the runtime Python's invocation
    // forms, but this scenario only reaches the torch-stack constraint probe
    // (`-c`) before `uv`; the empty JSON it prints is what makes that probe
    // pass cleanly.
    let python = root(world)
        .join("comfyui-fixture")
        .join("python")
        .join(fixture_target_name("rocm-python"));
    install_fake_python(&python);

    // A `rocm_sdk` probe that satisfies `validate_rocm_sdk_runtime_probe`:
    // importable, with an existing root/bin dir and resolved amdhip64/hipblas
    // libraries.
    let sdk_root = install_root.join("rocm_sdk").join("root");
    let sdk_bin = install_root.join("rocm_sdk").join("bin");
    std::fs::create_dir_all(&sdk_root).expect("failed to create fake rocm_sdk root dir");
    std::fs::create_dir_all(&sdk_bin).expect("failed to create fake rocm_sdk bin dir");
    let amdhip64 = sdk_root.join(rocm_library_filename("amdhip64"));
    let hipblas = sdk_root.join(rocm_library_filename("hipblas"));
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
        .join(fixture_target_name("uv"));
    world.command_env.extend(install_fake_uv(&uv, 7, None));
    world
        .command_env
        .push(("ROCM_CLI_UV_BINARY", uv.into_os_string()));
}

#[given("the ComfyUI dependency install with uv prints progress and succeeds")]
async fn uv_install_prints_progress_and_succeeds(world: &mut E2eWorld) {
    let uv = root(world)
        .join("comfyui-fixture")
        .join("uv-bin")
        .join(fixture_target_name("uv"));
    world
        .command_env
        .extend(install_fake_uv(&uv, 0, Some("Resolved 3 packages")));
    world
        .command_env
        .push(("ROCM_CLI_UV_BINARY", uv.into_os_string()));

    // The success path runs past `uv` into `probe_comfyui`'s post-install GPU
    // check, which shells out to the runtime's Python a second time with two
    // path arguments (a generated probe script, then where to write its JSON
    // result) rather than `-c <script>` like the pre-install torch-stack probe.
    // The fixture Python the `Given` above installed already answers both
    // forms, so nothing more is needed here for `install()` to reach that
    // second call, unlike `comfyui-01`, which fails before reaching it.
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

/// The `folder:` line `install` and `status` both print names this
/// directory, not the runtime root or the bare app root; the fixture
/// plants only `requirements.txt` here (production also has `main.py`).
fn expected_comfyui_folder_line(world: &E2eWorld) -> String {
    let source_dir = install_root(world)
        .join("apps")
        .join("comfyui")
        .join("source");
    format!("  folder: {}\n", source_dir.display())
}

#[then("the install output's folder: line names the ComfyUI source checkout")]
async fn install_folder_line_names_source_checkout(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    let expected = expected_comfyui_folder_line(world);
    assert!(
        stdout.contains(&expected),
        "expected install output to contain {expected:?}, got:\n{stdout}"
    );
}

#[when("the user checks ComfyUI status")]
async fn user_checks_comfyui_status(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["comfyui", "status"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("status names the same ComfyUI source folder as the install")]
async fn status_names_same_comfyui_source_folder(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    let rc = world.cli_rc.unwrap_or(-1);
    let expected = expected_comfyui_folder_line(world);
    assert!(
        rc == 0,
        "expected `rocm comfyui status` to succeed, got rc={rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains(&expected),
        "expected status output to contain {expected:?}, got:\n{stdout}"
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
    // `probe_comfyui`'s post-install GPU check runs unconditionally in
    // `install()`, regardless of whether the `uv` block ran, so the runtime's
    // stub Python must answer it — `plant_ready_runtime`'s `fake-python`
    // stand-in already does, answering both invocation forms. This scenario's
    // requirements.txt (torch-stack only, below) empties the dependency list
    // and skips the `-c` probe, but the stand-in answers it anyway for parity
    // with the other fixtures.
    plant_ready_runtime(&data, RUNTIME_KEY);

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
    // Every ComfyUI release has it; a reinstall refuses an archive without one.
    write_fixture(&source_dir.join("main.py"), "");
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
    // With a runtime installed, a scenario's first `rocm` call otherwise runs
    // the CLI's startup update check for this planted runtime: a real `uv`
    // download, a managed-Python setup and a package-index query, 10-15s on a
    // hosted runner. Nothing here is about updates, and while that call runs
    // it blocks the thread cucumber polls every other scenario from (#661).
    // Every command these scenarios run goes through
    // `run_keeping_scenario_env`, so this reaches all of them.
    world
        .command_env
        .push(("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK", "1".into()));
}

/// Serves a small release archive. Its `requirements.txt` names only the
/// torch stack, so the dependency install is filtered down to nothing and
/// skipped — this scenario is about the swap, not the `uv` step.
async fn serve_comfyui_release(world: &mut E2eWorld, served_path: &str, extra: &[(&str, &str)]) {
    serve_comfyui_release_with(world, served_path, extra, true, || {}).await;
}

/// [`serve_comfyui_release`], without `main.py` unless `ships_main_py`, and
/// running `on_request` when the release is requested, before any of it is
/// sent.
async fn serve_comfyui_release_with(
    world: &mut E2eWorld,
    served_path: &str,
    extra: &[(&str, &str)],
    ships_main_py: bool,
    on_request: impl Fn() + Send + Sync + 'static,
) {
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
    if !ships_main_py {
        std::fs::remove_file(release.join("main.py")).expect("main.py was just written");
    }
    let contents = build_gzip_tarball(&build_dir, "comfyui-release.tar.gz", "ComfyUI-master").await;
    let served = root(world).join("comfyui-fixture").join("release-serve");
    std::fs::create_dir_all(&served).expect("failed to create the archive fixture serve root");
    world.paced_download_server = Some(PacedDownloadServer::start_with_on_request(
        &served,
        "archive/comfyui-release.tar.gz",
        contents,
        1 << 20,
        Duration::ZERO,
        on_request,
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

/// A well-formed archive laid out like a release, but with no `main.py`: not
/// ComfyUI, or a layout this CLI does not know.
#[given("the newer ComfyUI release offered for download has no main.py")]
async fn comfyui_release_without_main_py(world: &mut E2eWorld) {
    serve_comfyui_release_with(world, "archive/comfyui-release.tar.gz", &[], false, || {}).await;
}

#[then("the reinstall fails saying the download is not a ComfyUI release, and does not keep it")]
async fn reinstall_rejects_non_comfyui_download(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    assert_ne!(
        world.cli_rc,
        Some(0),
        "expected the reinstall to fail\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("is not a ComfyUI release") && stderr.contains("main.py"),
        "the failure must say the download is not a ComfyUI release, got:\n{stderr}"
    );
    // Nothing was changed, so no command is named to finish anything.
    assert!(!stderr.contains("rocm comfyui install"), "{stderr}");
    let app_root = comfyui_source_dir(world)
        .parent()
        .expect("source has a parent")
        .to_path_buf();
    assert!(
        !app_root
            .join("downloads")
            .join("ComfyUI-master.tar.gz")
            .exists(),
        "the rejected download must not be kept for the next attempt"
    );
    assert!(
        !comfyui_source_dir(world)
            .join(".rocm-cli-reinstall-in-progress")
            .exists(),
        "no reinstall was started, so none is left unfinished"
    );
    assert_staging_folder_empty(world);
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
    assert_staging_folder_empty(world);
}

/// Nothing is left in the staging folder (`<app>/extract`) the release is
/// unpacked into before it is installed.
fn assert_staging_folder_empty(world: &E2eWorld) {
    let extract = comfyui_source_dir(world)
        .parent()
        .expect("source has a parent")
        .join("extract");
    let left: Vec<_> = match std::fs::read_dir(&extract) {
        Ok(entries) => entries
            .map(|entry| entry.expect("dir entry").file_name())
            .collect(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => panic!("failed to read {}: {error}", extract.display()),
    };
    assert!(
        left.is_empty(),
        "the staging folder must be emptied once the release is installed: {left:?}"
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
    // Nothing was changed, so no command is named to finish anything.
    assert!(!stderr.contains("rocm comfyui install"), "{stderr}");
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
    // No reinstall was interrupted here, so the preview claims none was.
    assert!(
        !stdout.contains("interrupted"),
        "the preview must not claim an interrupted reinstall, got:\n{stdout}"
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

/// A ComfyUI started with `rocm comfyui start` that has loaded and answers on
/// its port, which rocm-cli reads as running. The port is held open on the
/// World for the rest of the scenario.
#[given("the ComfyUI that rocm-cli started is running from that install and answering on its port")]
async fn comfyui_running_from_install(world: &mut E2eWorld) {
    let port = hold_answering_port(world);
    plant_running_state(world, spawn_stand_in(), port);
    // What the rest of the scenario rests on: rocm-cli reads it as running,
    // not as starting.
    let (stdout, stderr, rc) = run_keeping_scenario_env(world, &["comfyui", "status"]);
    assert_eq!(
        rc, 0,
        "`rocm comfyui status` failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("  status: running\n"),
        "the planted ComfyUI must read as running, got:\n{stdout}"
    );
}

/// A ComfyUI started with `rocm comfyui start` that is still loading: its
/// process is alive but nothing answers on its port yet, which rocm-cli reads
/// as starting.
#[given("a ComfyUI started by rocm-cli is still loading from that install")]
async fn comfyui_loading_from_install(world: &mut E2eWorld) {
    // A port nothing listens on, so the CLI's view rests on the process alone.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("failed to pick a free port")
        .port();
    plant_running_state(world, spawn_stand_in(), port);
}

/// `rocm comfyui start` is run while a reinstall downloads the newer release:
/// the server records the started ComfyUI, answering on its port, when the
/// download begins, which is after the reinstall's first check that nothing
/// runs from the folder. Only on the first request: the reinstall after
/// `stop` reuses the downloaded archive, and should it download again it
/// still must not find ComfyUI started a second time.
#[given(
    "a newer ComfyUI release is available, and ComfyUI is started from that install while it downloads"
)]
async fn comfyui_started_during_download(world: &mut E2eWorld) {
    let port = hold_answering_port(world);
    let path = running_state_path(world);
    let state = running_state_json(world, spawn_stand_in(), port);
    let fired = std::sync::atomic::AtomicBool::new(false);
    serve_comfyui_release_with(
        world,
        "archive/comfyui-release.tar.gz",
        &[],
        true,
        move || {
            if !fired.swap(true, std::sync::atomic::Ordering::SeqCst) {
                write_fixture(&path, &state);
            }
        },
    )
    .await;
}

/// Binds a loopback port that answers connections for the rest of the
/// scenario.
fn hold_answering_port(world: &mut E2eWorld) -> u16 {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind a loopback port");
    let port = listener.local_addr().expect("bound port").port();
    world.comfyui_port_listener = Some(listener);
    port
}

/// Starts a detached long-lived process to stand in for a running ComfyUI.
/// Detached via the shell so that once `rocm comfyui stop` kills it, init
/// reaps it and it does not linger as a zombie the CLI's liveness check would
/// still see. It exits on its own if the scenario fails before stopping it.
fn spawn_stand_in() -> u32 {
    let output = std::process::Command::new("sh")
        .args(["-c", "sleep 120 >/dev/null 2>&1 & echo $!"])
        .output()
        .expect("failed to start the stand-in ComfyUI process");
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("the shell printed the stand-in's pid")
}

/// Where `rocm comfyui start` records the ComfyUI it started.
fn running_state_path(world: &E2eWorld) -> PathBuf {
    data_dir(world)
        .join("apps")
        .join("comfyui")
        .join("state")
        .join("running.json")
}

/// The record `rocm comfyui start` saves, for process `pid` serving on
/// `port` from the planted install's `source/`.
fn running_state_json(world: &E2eWorld, pid: u32, port: u16) -> String {
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
    serde_json::to_string_pretty(&state).expect("state serialises")
}

fn plant_running_state(world: &E2eWorld, pid: u32, port: u16) {
    write_fixture(
        &running_state_path(world),
        &running_state_json(world, pid, port),
    );
}

#[then("the reinstall was refused before it downloaded anything")]
async fn refused_before_download(world: &mut E2eWorld) {
    let downloads = comfyui_source_dir(world)
        .parent()
        .expect("source has a parent")
        .join("downloads");
    let downloaded: Vec<_> = match std::fs::read_dir(&downloads) {
        Ok(entries) => entries
            .map(|entry| entry.expect("dir entry").file_name())
            .collect(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => panic!("failed to read {}: {error}", downloads.display()),
    };
    assert!(
        downloaded.is_empty(),
        "a reinstall refused up front must not download first: {downloaded:?}"
    );
}

#[then("the newer release had downloaded before the reinstall was refused")]
async fn release_downloaded_before_refusal(world: &mut E2eWorld) {
    let app_root = comfyui_source_dir(world)
        .parent()
        .expect("source has a parent")
        .to_path_buf();
    // The CLI keeps the downloaded archive; that it is there shows the
    // refusal came after the download, from the check just before the swap.
    assert!(
        app_root
            .join("downloads")
            .join("ComfyUI-master.tar.gz")
            .is_file(),
        "the reinstall must have downloaded the release before it was refused"
    );
    assert!(
        running_state_path(world).is_file(),
        "ComfyUI was recorded as started while the release downloaded"
    );
    assert_staging_folder_empty(world);
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
    // What the marker holds: the user's entries that were there before, then
    // the names of the release being installed.
    let marker = KEPT_LIST
        .split(", ")
        .map(str::to_owned)
        .chain(
            INSTALLED_RELEASE_RECORD
                .lines()
                .map(|name| format!("release\t{name}")),
        )
        .collect::<Vec<_>>()
        .join("\n");
    write_fixture(&source.join(".rocm-cli-reinstall-in-progress"), &marker);
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
async fn run_command_start_named(world: &mut E2eWorld) {
    run_named_command(world);
}

#[when("the user runs the command the reinstall named")]
async fn run_command_reinstall_named(world: &mut E2eWorld) {
    run_named_command(world);
}

/// Runs the first backticked `rocm` command in the last command's stderr.
fn run_named_command(world: &mut E2eWorld) {
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

/// Asked as one argument, as a user types `rocm "<question>"`; answered
/// without a model, and without changing anything.
const PLAIN_WORDS_STATUS_QUESTION: &str = "what is the comfyui status";

fn plain_words_status_answer(world: &mut E2eWorld) -> String {
    let (stdout, stderr, rc) = run_keeping_scenario_env(world, &[PLAIN_WORDS_STATUS_QUESTION]);
    assert_eq!(
        rc, 0,
        "the status question failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.starts_with("ComfyUI status\n") && stdout.contains("\nNothing was changed.\n"),
        "expected the ComfyUI status answer, got:\n{stdout}"
    );
    stdout
}

#[when("the user asks in plain words for ComfyUI's status")]
async fn ask_status_in_plain_words(world: &mut E2eWorld) {
    let answer = plain_words_status_answer(world);
    world.cli_output = Some(answer);
}

#[then("the answer says ComfyUI is installed and starting and does not advise starting it")]
async fn answer_says_starting(world: &mut E2eWorld) {
    let answer = world.cli_output.clone().unwrap_or_default();
    assert!(answer.contains("ComfyUI: installed\n"), "{answer}");
    assert!(answer.contains("Running: starting\n"), "{answer}");
    assert!(
        !answer.contains("comfyui start"),
        "a second ComfyUI must not be advised while one starts:\n{answer}"
    );
    // What "starting" claims: rocm-cli's own view is the same.
    let (status, _, rc) = run_keeping_scenario_env(world, &["comfyui", "status"]);
    assert_eq!(rc, 0, "{status}");
    assert!(status.contains("  status: starting\n"), "{status}");
}

#[then("the preview says the reinstall is refused until ComfyUI is stopped")]
async fn preview_says_refused_while_running(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    assert_eq!(world.cli_rc, Some(0), "dry run failed:\n{stdout}");
    assert!(
        stdout.lines().any(|line| {
            line.starts_with("  note: ComfyUI is running from ")
                && line.ends_with(
                    "Stop it with `rocm comfyui stop`; a reinstall is refused until then",
                )
        }),
        "the preview must say the reinstall waits for `rocm comfyui stop`, got:\n{stdout}"
    );
    // The scenario's next steps run that reinstall and prove the refusal.
}

#[then("the plain-words status answer names that command instead of start")]
async fn answer_names_finishing_command(world: &mut E2eWorld) {
    let answer = plain_words_status_answer(world);
    assert!(
        answer.contains(&format!("Note: {FINISH_ADVICE}.\n")),
        "the answer must name the command that finishes the reinstall, got:\n{answer}"
    );
    assert!(
        !answer.contains("comfyui start"),
        "`start` refuses here, so it must not be advised:\n{answer}"
    );
}

#[then("the plain-words status answer no longer reports an interrupted reinstall")]
async fn answer_clear_of_interruption(world: &mut E2eWorld) {
    let answer = plain_words_status_answer(world);
    assert!(!answer.contains("interrupted"), "{answer}");
    // Nothing rocm-cli started is running here, but the host may run a
    // ComfyUI of its own on ComfyUI's port; `start` is advised only if not.
    if answer.contains("Running: yes\n") {
        assert!(!answer.contains("comfyui start"), "{answer}");
    } else {
        assert!(answer.contains("Running: no\n"), "{answer}");
        assert!(answer.contains("run `rocm comfyui start`"), "{answer}");
    }
}

#[given("rocm-cli's saved record of the ComfyUI it started cannot be read")]
async fn unreadable_saved_state(world: &mut E2eWorld) {
    write_fixture(
        &data_dir(world)
            .join("apps")
            .join("comfyui")
            .join("state")
            .join("running.json"),
        "{\"app_id\": \"comfy",
    );
}

#[then("the reinstall says it could not tell whether ComfyUI was running")]
async fn reinstall_notes_unreadable_state(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    let state = data_dir(world)
        .join("apps")
        .join("comfyui")
        .join("state")
        .join("running.json");
    let prefix = format!("Note: failed to parse {}", state.display());
    let suffix = format!(
        "; whether a ComfyUI started by rocm-cli is running from {} is unknown, so the \
         reinstall is not refused for it",
        comfyui_source_dir(world).display()
    );
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with(&prefix) && line.ends_with(&suffix)),
        "the reinstall must say why it was not refused, got:\n{stdout}"
    );
    // What the note claims: the reinstall was not refused. The scenario's
    // other steps check what it did.
    assert_eq!(world.cli_rc, Some(0), "{stdout}");
}

#[given("the runtime's AMD GPU check fails")]
async fn gpu_check_fails(world: &mut E2eWorld) {
    write_shim(
        &runtime_python(world),
        "#!/bin/sh\n\
         if [ \"$1\" = \"-c\" ]; then\n\
         \tprintf '{}'\n\
         \texit 0\n\
         fi\n\
         cat > \"$2\" <<'JSON'\n\
         {\"torch_version\": \"2.4.0\", \"torch_cuda_available\": false, \"device_count\": 0, \"devices\": []}\n\
         JSON\n",
    );
}

#[given("the runtime's AMD GPU check passes again")]
async fn gpu_check_passes_again(world: &mut E2eWorld) {
    install_fake_python(&runtime_python(world));
}

fn runtime_python(world: &E2eWorld) -> PathBuf {
    data_dir(world)
        .join("runtimes")
        .join("roots")
        .join(RUNTIME_KEY)
        .join("bin")
        .join(fixture_target_name("python3"))
}

#[then("the reinstall fails and names what it replaced and the command that finishes the install")]
async fn reinstall_fails_after_swap(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    assert_ne!(world.cli_rc, Some(0), "the reinstall must fail:\n{stderr}");
    assert!(
        stderr.contains(&format!(
            "the new ComfyUI code is in place (replaced: comfy, main.py, requirements.txt), but \
             the install did not finish; once the cause below is fixed, run `rocm comfyui install \
             --runtime-id {RUNTIME_KEY}` to finish it"
        )),
        "the failure must name what was replaced and how to finish, got:\n{stderr}"
    );
    assert!(stderr.contains("AMD GPU check failed"), "{stderr}");
    // What "replaced" claims: those are the new release's now.
    let source = comfyui_source_dir(world);
    assert_eq!(
        std::fs::read_to_string(source.join("requirements.txt"))
            .ok()
            .as_deref(),
        Some("torch==2.4.0\n")
    );
}

#[then("ComfyUI is installed and its AMD GPU check is ready")]
async fn comfyui_installed_and_ready(world: &mut E2eWorld) {
    let (stdout, _, rc) = run_keeping_scenario_env(world, &["comfyui", "status"]);
    assert_eq!(rc, 0, "{stdout}");
    assert!(stdout.contains("  installed: yes\n"), "{stdout}");
    assert!(stdout.contains("  AMD GPU check: ready\n"), "{stdout}");
}
