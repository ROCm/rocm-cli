// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `therock_next_generation.feature`.
//!
//! Scenarios 01-05 serve both the canonical release layout and the ROCm 10
//! ("next") layout from one loopback server, under `current/` and `next/`
//! prefixes, and point the gated `ROCM_CLI_THEROCK_*_BASE` overrides at them.
//! Serving both — rather than only the one a scenario expects to be used — is
//! what makes "the canonical stream is still canonical" and "the next stream is
//! only reached when explicitly pinned" assertable: a dispatch regression
//! resolves the *other* fixture instead of failing to resolve anything.
//!
//! Scenario 06 sets an override without the trust opt-in and resolves the live
//! default index, proving the child CLI ignores the untrusted value end to end.
//!
//! Scenario 07 installs for real against `stable.repo.amd.com` on a self-hosted
//! GPU runner, with no `--family` override. The fixture scenarios prove dispatch
//! and refusal logic for a supplied exact arch; this one proves the exact arch
//! can come from `resolve_family`'s host probe (`detect_host_gfx_target`) and that
//! the resulting managed SDK/Torch stack executes a real GPU kernel.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use cucumber::{given, then, when};
use e2e_cucumber::cli_failure_report;
use e2e_cucumber::loopback_http::LoopbackServer;
use e2e_cucumber::paced_download::{
    PacedDownloadServer, build_gzip_tarball, deterministic_payload,
};

use crate::E2eWorld;
use crate::e2e::tui_driver::TuiSession;

/// The exact GFX arch the next layout needs. Not a group label: the aggregate
/// source publishes one `rocm-sdk-device-<arch>` payload per arch, and
/// `device-gfx120X-all` is not an extra it declares at all.
const RAW_ARCH: &str = "gfx1200";
/// The family label `RAW_ARCH` normalizes to, used for the tarball file names
/// and the canonical (non-next) resolution.
const GROUP_FAMILY: &str = "gfx120X-all";

const NEXT_ROCM_VERSION: &str = "10.0.0";
const NEXT_TORCH_VERSION: &str = "2.10.0+rocm10.0.0";
const NEXT_TORCHVISION_VERSION: &str = "0.25.0+rocm10.0.0";
const NEXT_TORCHAUDIO_VERSION: &str = "2.9.1+rocm10.0.0";

const CURRENT_ROCM_VERSION: &str = "7.10.0";
const CURRENT_TORCH_VERSION: &str = "2.9.0+rocm7.10.0";
const CURRENT_TORCHVISION_VERSION: &str = "0.24.0+rocm7.10.0";
const CURRENT_TORCHAUDIO_VERSION: &str = "2.9.0+rocm7.10.0";

const NEXT_REAL_TARBALL: &str = "therock-dist-linux-gfx120X-all-10.0.0.tar.gz";
/// The non-release sibling the live catalog publishes beside the real archive.
const NEXT_TESTS_TARBALL: &str = "therock-dist-linux-gfx120X-all-tests-10.0.0.tar.gz";
const CURRENT_TARBALL: &str = "therock-dist-linux-gfx120X-all-7.10.0.tar.gz";

/// How many leading bytes of `"Downloading {CURRENT_TARBALL}"` to check for in
/// [`assert_spinner_lines_cleared`] below. Comfortably under the ~48-column
/// label budget `assemble_status_line` leaves on an 80-column terminal even at
/// the widest realistic progress suffix (see that assertion's comment) — the
/// full string would never fully render while the line is live, so a prefix
/// this short is what actually needs to disappear on clear.
const DOWNLOADING_PREFIX_LEN: usize = 30;

/// The `current/` fixture's served base, i.e. what the canonical release
/// overrides are pointed at.
fn current_pip_base(world: &E2eWorld) -> String {
    format!("{}/current", server_base(world))
}

/// The `next/` fixture's served base, i.e. what the ROCm 10 overrides are
/// pointed at.
fn next_pip_base(world: &E2eWorld) -> String {
    format!("{}/next", server_base(world))
}

fn next_tarball_base(world: &E2eWorld) -> String {
    format!("{}/tarball/next/", server_base(world))
}

fn server_base(world: &E2eWorld) -> String {
    world
        .artifact_server
        .as_ref()
        .expect("scenario started no fixture server")
        .base_url()
}

fn root(world: &E2eWorld) -> &Path {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
}

fn write_fixture(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create fixture directory");
    }
    std::fs::write(path, contents).expect("failed to write fixture file");
}

/// A PEP 503 style root listing. `validate_aggregate_index_layout` requires the
/// four stack packages, and `parse_aggregate_device_targets` reads the exact
/// published arches out of the `rocm-sdk-device-*` links — so the fixture has to
/// publish `rocm-sdk-device-gfx1200` for `device-gfx1200` to be requestable.
fn aggregate_root_html() -> String {
    [
        "rocm",
        "torch",
        "torchvision",
        "torchaudio",
        "rocm-sdk-device-gfx1200",
    ]
    .iter()
    .fold(String::new(), |mut html, name| {
        writeln!(html, "<a href=\"{name}/\">{name}</a>").expect("write fixture HTML");
        html
    })
}

/// One PEP 503 package page with PEP 658 metadata. The metadata lets uv resolve
/// the published dependency contract without downloading a wheel.
fn write_wheel_index(served: &Path, package: &str, version: &str, requires: &[String]) {
    let file_name = format!("{package}-{version}-py3-none-any.whl");
    let html =
        format!("<a href=\"{file_name}\" data-dist-info-metadata=\"true\">{file_name}</a>\n");
    write_fixture(&served.join("index.html"), &html);
    let mut metadata = format!("Metadata-Version: 2.3\nName: {package}\nVersion: {version}\n");
    if matches!(package, "rocm" | "torch" | "torchvision") {
        metadata.push_str("Provides-Extra: device-gfx1200\n");
    }
    if package == "rocm" {
        metadata.push_str("Provides-Extra: libraries\nProvides-Extra: devel\n");
    }
    for requirement in requires {
        writeln!(metadata, "Requires-Dist: {requirement}").expect("write fixture metadata");
    }
    write_fixture(&served.join(format!("{file_name}.metadata")), &metadata);
}

fn write_pip_index(served: &Path, versions: [(&str, &str); 4]) {
    write_fixture(&served.join("index.html"), &aggregate_root_html());
    let torch_version = versions
        .iter()
        .find_map(|(package, version)| (*package == "torch").then_some(*version))
        .expect("fixture must include torch");
    for (package, version) in versions {
        let requires = if matches!(package, "torchvision" | "torchaudio") {
            vec![format!("torch=={torch_version}")]
        } else {
            Vec::new()
        };
        write_wheel_index(&served.join(package), package, version, &requires);
    }
}

/// The scrapeable listing the tarball catalog publishes: a JS array of
/// `{name, mtime}` records.
fn tarball_index_html(files: &[(&str, f64)]) -> String {
    let entries = files
        .iter()
        .map(|(name, mtime)| format!("{{\"name\": \"{name}\", \"mtime\": {mtime:?}}}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("<html><body><script>const files = [{entries}];</script></body></html>")
}

fn allow_base_overrides(world: &mut E2eWorld) {
    world
        .command_env
        .push(("ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE", "1".into()));
}

#[given("a canonical release pip index fixture and a ROCm 10 pip index fixture")]
async fn pip_index_fixtures(world: &mut E2eWorld) {
    let served = root(world).join("therock-pip-fixtures");
    write_pip_index(
        &served.join("current"),
        [
            ("rocm", CURRENT_ROCM_VERSION),
            ("torch", CURRENT_TORCH_VERSION),
            ("torchvision", CURRENT_TORCHVISION_VERSION),
            ("torchaudio", CURRENT_TORCHAUDIO_VERSION),
        ],
    );
    write_pip_index(
        &served.join("next"),
        [
            ("rocm", NEXT_ROCM_VERSION),
            ("torch", NEXT_TORCH_VERSION),
            ("torchvision", NEXT_TORCHVISION_VERSION),
            ("torchaudio", NEXT_TORCHAUDIO_VERSION),
        ],
    );
    world.artifact_server = Some(LoopbackServer::start(&served));
    allow_base_overrides(world);
    let current = current_pip_base(world);
    let next = next_pip_base(world);
    world
        .command_env
        .push(("ROCM_CLI_THEROCK_RELEASE_PIP_BASE", current.into()));
    world
        .command_env
        .push(("ROCM_CLI_THEROCK_NEXT_PIP_BASE", next.into()));
}

/// Registry key for the planted manifest below. Filename-safe (no `:`), same
/// reasoning as `runtime_lifecycle_steps.rs`'s planted keys.
const NEXT_GROUPED_FAMILY_RUNTIME_KEY: &str = "release-wheel-next-v1-gfx120x-all-9-0-0";

/// Plants a `wheel`/`next-v1` runtime manifest whose family is a *group* label
/// (`gfx120X-all` — every family this PR adds is one), with the exact arch it
/// was installed for recorded in `wheel_composition.rocm_sdk_target`, exactly
/// the shape a real ROCm 10 install produces. `apply_runtime_update` ->
/// `install_sdk_for_update` -> `install_wheel_runtime` must resolve past this
/// without the user ever typing `--family gfx1200` again.
///
/// Version `9.0.0` (older than the fixture's published `NEXT_ROCM_VERSION`) so
/// the update plan finds one available; nothing about this scenario depends on
/// a real ROCm 9 having existed, since the layout is read from
/// `source_layout_generation`, not derived from the version string.
#[given("a registered ROCm 10 wheel runtime with a grouped family")]
async fn registered_next_wheel_runtime_with_grouped_family(world: &mut E2eWorld) {
    let install_root = root(world).join("runtime-next-grouped-family");
    std::fs::create_dir_all(&install_root).expect("failed to create install root");

    let registry = root(world).join("data").join("runtimes").join("registry");
    std::fs::create_dir_all(&registry).expect("failed to create registry dir");
    let manifest = serde_json::to_string_pretty(&serde_json::json!({
        "runtime_key": NEXT_GROUPED_FAMILY_RUNTIME_KEY,
        "runtime_id": format!("therock-release:{GROUP_FAMILY}"),
        "channel": "release",
        "format": "wheel",
        "family": GROUP_FAMILY,
        "family_source": "manual",
        "version": "9.0.0",
        "install_root": install_root,
        "selected_artifact_url": "https://example.invalid/rocm",
        "source_layout_generation": "next-v1",
        "read_only": false,
        "wheel_composition": {
            "source_layout_generation": "next-v1",
            "package_specs": [
                format!("rocm[libraries,devel,device-{RAW_ARCH}]==9.0.0"),
                "torch==2.9.0+rocm9.0.0",
                "torchvision==0.24.0+rocm9.0.0",
                "torchaudio==2.9.0+rocm9.0.0",
            ],
            "rocm_sdk_target": RAW_ARCH,
        },
        "installed_at_unix_ms": 1_700_000_000_000u64,
    }))
    .expect("failed to serialize runtime manifest");
    std::fs::write(
        registry.join(format!("{NEXT_GROUPED_FAMILY_RUNTIME_KEY}.json")),
        manifest,
    )
    .expect("failed to write runtime manifest");
}

#[when("the user previews applying the pending update to that runtime")]
async fn preview_apply_pending_update(world: &mut E2eWorld) {
    preview_ok(
        world,
        &[
            "update",
            "--apply",
            "--dry-run",
            "--runtime",
            NEXT_GROUPED_FAMILY_RUNTIME_KEY,
        ],
    );
}

#[given("an untrusted ROCm 10 pip base override")]
async fn untrusted_pip_index_fixture(world: &mut E2eWorld) {
    let served = root(world).join("untrusted-therock-pip-fixture");
    write_pip_index(
        &served,
        [
            ("rocm", NEXT_ROCM_VERSION),
            ("torch", NEXT_TORCH_VERSION),
            ("torchvision", NEXT_TORCHVISION_VERSION),
            ("torchaudio", NEXT_TORCHAUDIO_VERSION),
        ],
    );
    world.artifact_server = Some(LoopbackServer::start(&served));
    let untrusted = server_base(world);
    world
        .command_env
        .push(("ROCM_CLI_THEROCK_NEXT_PIP_BASE", untrusted.into()));
}

#[given("a canonical release tarball fixture and a ROCm 10 tarball fixture with a tests sibling")]
async fn tarball_index_fixtures(world: &mut E2eWorld) {
    let served = root(world).join("therock-tarball-fixtures");
    write_fixture(
        &served.join("tarball").join("current").join("index.html"),
        &tarball_index_html(&[(CURRENT_TARBALL, 1_787_000_000.0)]),
    );
    // The real archive is OLDER than its `-tests-` sibling, exactly as the live
    // catalog publishes them, so mtime alone selects the wrong file.
    write_fixture(
        &served.join("tarball").join("next").join("index.html"),
        &tarball_index_html(&[
            (NEXT_REAL_TARBALL, 1_787_612_008.0),
            (NEXT_TESTS_TARBALL, 1_787_612_032.0),
        ]),
    );
    world.artifact_server = Some(LoopbackServer::start(&served));
    allow_base_overrides(world);
    let base = server_base(world);
    let next = next_tarball_base(world);
    world.command_env.push((
        "ROCM_CLI_THEROCK_RELEASE_TARBALL_BASE",
        format!("{base}/tarball/current/").into(),
    ));
    world
        .command_env
        .push(("ROCM_CLI_THEROCK_NEXT_TARBALL_BASE", next.into()));
}

/// Size and pacing for the paced tarball fixture below: large enough (versus
/// the chunk size) that several chunk boundaries — and therefore several
/// observable progress frames — land before the transfer completes, and slow
/// enough per chunk that the PTY's poll cadence reliably samples an
/// intermediate, sub-100% frame rather than racing straight to completion.
///
/// The payload is tens of MB, not a few hundred KB, so that `tar -xf`
/// (spawned synchronously once the download completes — see
/// `extract_tarball`) takes long enough, via its own subprocess-spawn and
/// real disk I/O over a ~20MB archive, for the "Extracting …" spinner frame
/// to still be on screen the next time the PTY's poll checks it — the poll
/// cadence only governs how often the already-rendered screen is sampled, not
/// how fast extraction itself runs. The chunk size scales with the payload,
/// so the number of paced chunks — and therefore the download's observed
/// wall time — stays the same as before.
const PACED_TARBALL_PAYLOAD_BYTES: usize = 20_000_000;
const PACED_TARBALL_CHUNK_BYTES: usize = 1_600_000;
const PACED_TARBALL_CHUNK_DELAY: Duration = Duration::from_millis(150);
/// Wait budget for the PTY-driven download scenario below, mirroring
/// `engines_steps.rs`'s file-local `SCREEN_TIMEOUT` convention.
const PTY_SCREEN_TIMEOUT: Duration = Duration::from_secs(30);

#[given("a paced canonical release tarball fixture")]
async fn paced_tarball_fixture(world: &mut E2eWorld) {
    let served = root(world).join("therock-paced-tarball-fixture");
    write_fixture(
        &served.join("tarball").join("current").join("index.html"),
        &tarball_index_html(&[(CURRENT_TARBALL, 1_787_000_000.0)]),
    );

    // Build a real gzip tarball so `extract_tarball` (auto-detecting `-xf`) has
    // a genuine archive to unpack once the paced download completes. The
    // payload bytes come from a seeded CSPRNG (`StdRng`) rather than a simple
    // multiplicative-hash sequence: the latter looked scrambled but gzip still
    // crushed it down to under 2 KB (well under one paced chunk), collapsing
    // the whole "transfer" into a single unpaced chunk and defeating the
    // pacing entirely. `StdRng` output is high-entropy enough that gzip
    // cannot shrink it, keeping the wire transfer close to
    // `PACED_TARBALL_PAYLOAD_BYTES`.
    let build_dir = root(world).join("therock-paced-tarball-build");
    let payload_dir = build_dir.join("payload");
    std::fs::create_dir_all(&payload_dir).expect("failed to create tarball payload directory");
    let payload = deterministic_payload(PACED_TARBALL_PAYLOAD_BYTES);
    std::fs::write(payload_dir.join("payload.bin"), &payload)
        .expect("failed to write tarball payload");
    let contents = build_gzip_tarball(&build_dir, CURRENT_TARBALL, "payload").await;

    world.paced_download_server = Some(PacedDownloadServer::start(
        &served,
        &format!("tarball/current/{CURRENT_TARBALL}"),
        contents,
        PACED_TARBALL_CHUNK_BYTES,
        PACED_TARBALL_CHUNK_DELAY,
    ));
    allow_base_overrides(world);
    let base = world
        .paced_download_server
        .as_ref()
        .expect("paced download server was just started")
        .base_url();
    world.command_env.push((
        "ROCM_CLI_THEROCK_RELEASE_TARBALL_BASE",
        format!("{base}/tarball/current/").into(),
    ));
}

#[when("the user installs the tarball SDK for family gfx120X-all under a real terminal")]
async fn install_tarball_sdk_under_pty(world: &mut E2eWorld) {
    // No `--version`: tarball installs only accept an explicit version pin for
    // a stable ROCm 10+ selector (see `resolve_tarball_artifact_with_timeout`'s
    // rejection message), so an unpinned request is what reaches the canonical
    // release catalog here — which the given-step populated with exactly one
    // candidate, `CURRENT_TARBALL`.
    //
    // This scenario is testing the download spinner, not the post-install
    // torch-runtime-dependency setup — but a real SDK install completing
    // successfully triggers `ensure_libatomic_for_torch`/`ensure_libnuma_for_torch`,
    // which install a missing PyTorch runtime library through the real system
    // package manager whenever the test host happens to lack it. That is slow,
    // network-dependent, and mutates host state, none of which this scenario
    // should depend on, so disable it here.
    world
        .command_env
        .push(("ROCM_CLI_DISABLE_TORCH_RUNTIME_DEP_CHECKS", "1".into()));
    let session = TuiSession::spawn(
        world,
        &[
            "install",
            "sdk",
            "--channel",
            "release",
            "--format",
            "tarball",
            "--family",
            GROUP_FAMILY,
            "--yes",
        ],
    )
    .unwrap_or_else(|e| panic!("failed to spawn `rocm install sdk` under a pty: {e}"));
    world.tui = Some(session);
    // A successful install's summary is ~20 lines; on the default 24-row
    // screen (no scrollback) it would scroll the download/extraction spinner
    // rows off the top before `assert_spinner_lines_cleared` ever reads them,
    // turning that assertion into a tautology regardless of whether
    // `Spinner::clear` actually ran. Grow rows only — not `use_detail_size`,
    // which also widens the terminal and would stop the label from
    // truncating, defeating the whole point of this scenario.
    //
    // Issued immediately after spawn with no synchronization point: this
    // assumes the child process's fork/exec and first spinner repaint take
    // longer than this resize call, so the grow always lands before more
    // than 24 lines could have been written. True at real process-startup
    // timings; there's no signal to wait on that would prove it instead.
    world
        .tui
        .as_mut()
        .expect("tui session was just set")
        .grow_rows(60)
        .unwrap_or_else(|e| panic!("failed to grow the pty's row count: {e}"));
}

#[then("the terminal shows an intermediate download progress frame")]
async fn assert_intermediate_download_progress_frame(world: &mut E2eWorld) {
    let session = world
        .tui
        .as_mut()
        .expect("no pty session for the tarball install");
    session
        .assert_intermediate_download_progress_frame("the tarball install", PTY_SCREEN_TIMEOUT)
        .await;
}

#[then("the terminal shows the archive being extracted")]
async fn assert_extraction_frame_is_shown(world: &mut E2eWorld) {
    let session = world
        .tui
        .as_mut()
        .expect("no pty session for the tarball install");
    session
        .wait_for_screen(&format!("Extracting {CURRENT_TARBALL}"), PTY_SCREEN_TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("extraction spinner frame never appeared: {e}"));
}

#[then("the tarball install exits cleanly")]
async fn assert_tarball_install_exits_cleanly(world: &mut E2eWorld) {
    let session = world
        .tui
        .as_mut()
        .expect("no pty session for the tarball install");
    session
        .assert_exits_cleanly("the tarball install", PTY_SCREEN_TIMEOUT)
        .await;
}

#[then("the final terminal screen shows neither spinner line")]
async fn assert_spinner_lines_cleared(world: &mut E2eWorld) {
    let session = world
        .tui
        .as_ref()
        .expect("no pty session for the tarball install");
    let screen = session.screen_text();
    // A live progress suffix (kept intact by `assemble_status_line`) can
    // truncate this label to a fraction of its length on an 80-column
    // terminal, so the download line never actually contains the full
    // "Downloading {CURRENT_TARBALL}" string while it's showing — checking
    // for that full string here would pass trivially whether or not the line
    // was cleared. Truncation always keeps the label's head intact and cuts
    // its tail, so a short prefix is present whenever the line is live and
    // gone once `Spinner::clear` erases it.
    //
    // Note this download-line check by itself can't distinguish a real
    // `Drop`-time clear from the extraction spinner's first repaint simply
    // overwriting the same row with `Clear(CurrentLine)` — it would pass
    // either way, since extraction always starts once the download spinner
    // stops. The extraction assertion just below, and the ComfyUI scenario's
    // equivalent check (a spinner with nothing after it to overwrite its
    // row), are what actually prove `Spinner::clear` runs on `Drop`.
    let downloading = format!("Downloading {CURRENT_TARBALL}");
    let downloading_prefix = downloading
        .get(..DOWNLOADING_PREFIX_LEN)
        .unwrap_or_else(|| {
            panic!(
                "DOWNLOADING_PREFIX_LEN ({DOWNLOADING_PREFIX_LEN}) is not a char boundary in \
             {downloading:?} — pick a length that lands on one, otherwise this assertion \
             would silently degrade to checking the untruncated label, which a live \
             truncated line never shows"
            )
        });
    assert!(
        !screen.contains(downloading_prefix),
        "download spinner line was not cleared on completion:\n{screen}"
    );
    assert!(
        !screen.contains(&format!("Extracting {CURRENT_TARBALL}")),
        "extract spinner line was not cleared on completion:\n{screen}"
    );
}

fn preview(world: &mut E2eWorld, args: &[&str]) -> i32 {
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(world, args);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
    rc
}

fn preview_ok(world: &mut E2eWorld, args: &[&str]) {
    let rc = preview(world, args);
    assert!(
        rc == 0,
        "{}",
        cli_failure_report(
            args,
            rc,
            world.cli_output.as_deref().unwrap_or_default(),
            world.cli_stderr.as_deref().unwrap_or_default(),
        )
    );
}

#[when("the user previews a wheel SDK install for arch gfx1200 with no version pin")]
async fn preview_unpinned_wheel_install(world: &mut E2eWorld) {
    preview_ok(
        world,
        &[
            "install",
            "sdk",
            "--channel",
            "release",
            "--format",
            "wheel",
            "--family",
            RAW_ARCH,
            "--dry-run",
        ],
    );
}

#[when("the user previews a wheel SDK install for arch gfx1200 pinned to ROCm 10.0.0")]
async fn preview_pinned_wheel_install(world: &mut E2eWorld) {
    preview_ok(
        world,
        &[
            "install",
            "sdk",
            "--channel",
            "release",
            "--format",
            "wheel",
            "--family",
            RAW_ARCH,
            "--version",
            NEXT_ROCM_VERSION,
            "--dry-run",
        ],
    );
}

#[when(
    "the user previews a wheel SDK install for arch gfx1200 pinned to ROCm 10.0.0 with the toolchain"
)]
async fn preview_pinned_wheel_install_with_devel(world: &mut E2eWorld) {
    preview_ok(
        world,
        &[
            "install",
            "sdk",
            "--channel",
            "release",
            "--format",
            "wheel",
            "--family",
            RAW_ARCH,
            "--version",
            NEXT_ROCM_VERSION,
            "--devel",
            "--dry-run",
        ],
    );
}

#[then("the preview requests the gfx1200 device extras with the toolchain")]
async fn preview_requests_device_extras_with_devel(world: &mut E2eWorld) {
    // The opt-in half of the same assertion. Whole line for the same reason:
    // `devel` has to be added to the rocm extras without disturbing the device
    // payload on any of the four requirements.
    let expected = format!(
        "package_specs: rocm[libraries,devel,device-{RAW_ARCH}]=={NEXT_ROCM_VERSION} \
         torch[device-{RAW_ARCH}]=={NEXT_TORCH_VERSION} \
         torchvision[device-{RAW_ARCH}]=={NEXT_TORCHVISION_VERSION} \
         torchaudio=={NEXT_TORCHAUDIO_VERSION}"
    );
    assert_contains(world, &expected, "device extras with devel");
    assert_version_resolution_extras(world, true);
}

/// Distinct from [`preview_pinned_wheel_install`]: this scenario's whole point
/// is that the override is ignored and resolution falls through to the real
/// `DEFAULT_NEXT_RELEASE_PIP_BASE`, so the pin has to be a version that source
/// actually publishes right now — a hardcoded `10.0.0` would go stale the
/// moment it is superseded, same reasoning as `discover_latest_next_rocm_version`'s
/// other caller.
#[when(
    "the user previews a wheel SDK install for arch gfx1200 pinned to the latest published ROCm 10 version"
)]
async fn preview_pinned_wheel_install_latest_next_version(world: &mut E2eWorld) {
    let version = discover_latest_next_rocm_version().await;
    preview_ok(
        world,
        &[
            "install",
            "sdk",
            "--channel",
            "release",
            "--format",
            "wheel",
            "--family",
            RAW_ARCH,
            "--version",
            &version,
            "--dry-run",
        ],
    );
}

#[when("the user previews a wheel SDK install for family gfx120X-all pinned to ROCm 10.0.0")]
async fn preview_pinned_wheel_install_with_group_family(world: &mut E2eWorld) {
    preview(
        world,
        &[
            "install",
            "sdk",
            "--channel",
            "release",
            "--format",
            "wheel",
            "--family",
            GROUP_FAMILY,
            "--version",
            NEXT_ROCM_VERSION,
            "--dry-run",
        ],
    );
}

#[when("the user previews a tarball SDK install for arch gfx1200 pinned to ROCm 10.0.0")]
async fn preview_pinned_tarball_install(world: &mut E2eWorld) {
    preview_ok(
        world,
        &[
            "install",
            "sdk",
            "--channel",
            "release",
            "--format",
            "tarball",
            "--family",
            RAW_ARCH,
            "--version",
            NEXT_ROCM_VERSION,
            "--dry-run",
        ],
    );
}

#[when("the user previews a tarball SDK install for family gfx120X-all pinned to ROCm 10.0.0")]
async fn preview_pinned_tarball_install_with_group_family(world: &mut E2eWorld) {
    preview(
        world,
        &[
            "install",
            "sdk",
            "--channel",
            "release",
            "--format",
            "tarball",
            "--family",
            GROUP_FAMILY,
            "--version",
            NEXT_ROCM_VERSION,
            "--dry-run",
        ],
    );
}

fn stdout(world: &E2eWorld) -> &str {
    world.cli_output.as_deref().unwrap_or_default()
}

fn assert_contains(world: &E2eWorld, needle: &str, what: &str) {
    assert!(
        stdout(world).contains(needle),
        "{what}: expected {needle:?} in the preview:\n{}",
        stdout(world)
    );
}

#[then("the preview resolves the canonical release pip index")]
async fn preview_resolves_canonical_pip_index(world: &mut E2eWorld) {
    let base = current_pip_base(world);
    assert_contains(
        world,
        &format!("canonical_source: {base}"),
        "canonical source",
    );
    assert_contains(world, &format!("index_url: {base}"), "resolved index");
    assert_contains(
        world,
        &format!("latest_compatible_version: {CURRENT_ROCM_VERSION}"),
        "resolved canonical version",
    );
}

#[then("the preview reports the canonical multi-arch source layout generation")]
async fn preview_reports_canonical_generation(world: &mut E2eWorld) {
    assert_contains(
        world,
        "source_layout_generation: multi-arch-v2",
        "canonical layout generation",
    );
}

#[then("the preview never mentions the ROCm 10 pip index")]
async fn preview_never_mentions_next_index(world: &mut E2eWorld) {
    let next = next_pip_base(world);
    assert!(
        !stdout(world).contains(&next),
        "an unpinned release install reached the ROCm 10 index {next}:\n{}",
        stdout(world)
    );
    assert!(
        !stdout(world).contains("next-v1"),
        "an unpinned release install selected the next source layout:\n{}",
        stdout(world)
    );
}

#[then("the preview resolves the ROCm 10 pip index")]
async fn preview_resolves_next_pip_index(world: &mut E2eWorld) {
    let base = next_pip_base(world);
    assert_contains(world, &format!("canonical_source: {base}"), "next source");
    assert_contains(world, &format!("index_url: {base}"), "resolved index");
    assert_contains(
        world,
        &format!("latest_compatible_version: {NEXT_ROCM_VERSION}"),
        "resolved next version",
    );
}

#[then("the preview resolves the default ROCm 10 pip index")]
async fn preview_resolves_default_next_pip_index(world: &mut E2eWorld) {
    assert_contains(
        world,
        &format!("canonical_source: {DEFAULT_NEXT_RELEASE_PIP_BASE}"),
        "default next source",
    );
    assert_contains(
        world,
        &format!("index_url: {DEFAULT_NEXT_RELEASE_PIP_BASE}"),
        "default resolved index",
    );
}

#[then("the preview never mentions the untrusted ROCm 10 pip index")]
async fn preview_never_mentions_untrusted_next_index(world: &mut E2eWorld) {
    let untrusted = server_base(world);
    assert!(
        !stdout(world).contains(&untrusted),
        "an override without explicit trust redirected the install to {untrusted}:\n{}",
        stdout(world)
    );
}

#[then("the preview reports the next source layout generation")]
async fn preview_reports_next_generation(world: &mut E2eWorld) {
    assert_contains(
        world,
        "source_layout_generation: next-v1",
        "next layout generation",
    );
}

#[then("the preview requests the gfx1200 device extras")]
async fn preview_requests_device_extras(world: &mut E2eWorld) {
    // The exact line, not four substring checks: the value of this assertion is
    // that the install would request one exact device payload for rocm, torch
    // and torchvision and none for torchaudio, which only the whole spec list
    // shows. A group-bucket or `device-all` regression still matches any subset.
    let expected = format!(
        "package_specs: rocm[libraries,device-{RAW_ARCH}]=={NEXT_ROCM_VERSION} \
         torch[device-{RAW_ARCH}]=={NEXT_TORCH_VERSION} \
         torchvision[device-{RAW_ARCH}]=={NEXT_TORCHVISION_VERSION} \
         torchaudio=={NEXT_TORCHAUDIO_VERSION}"
    );
    assert_contains(world, &expected, "device extras");
    assert_version_resolution_extras(world, false);
    assert_contains(
        world,
        &format!("device_target: {RAW_ARCH}"),
        "device target",
    );
}

/// What version resolution asked `uv pip compile` for, as opposed to what the
/// install plan says it will install.
///
/// These are produced by two different code paths from one `include_devel`, and
/// only this assertion covers the first. `package_specs` alone passed while the
/// requirements handed to `uv` were hardcoded to `rocm[libraries,devel,...]`,
/// which constrained the chosen versions by a toolchain a default install never
/// asked for.
fn assert_version_resolution_extras(world: &mut E2eWorld, include_devel: bool) {
    let extras = if include_devel {
        "libraries,devel"
    } else {
        "libraries"
    };
    let expected = format!(
        "version_resolution_specs: rocm[{extras},device-{RAW_ARCH}]=={NEXT_ROCM_VERSION} \
         torch[device-{RAW_ARCH}] torchvision[device-{RAW_ARCH}] torchaudio"
    );
    assert_contains(world, &expected, "version resolution extras");
}

#[then("the preview resolves the ROCm 10 tarball catalog")]
async fn preview_resolves_next_tarball_catalog(world: &mut E2eWorld) {
    let base = next_tarball_base(world);
    assert_contains(
        world,
        &format!("canonical_source: {base}"),
        "next tarball catalog",
    );
    assert_contains(
        world,
        &format!("tarball_url: {base}{NEXT_REAL_TARBALL}"),
        "selected artifact url",
    );
}

#[then("the preview selects the real tarball artifact")]
async fn preview_selects_real_tarball(world: &mut E2eWorld) {
    assert_contains(
        world,
        &format!("tarball: {NEXT_REAL_TARBALL}"),
        "selected artifact",
    );
    assert_contains(
        world,
        &format!("latest_version: {NEXT_ROCM_VERSION}"),
        "selected artifact version",
    );
}

#[then("the preview does not select the tests artifact")]
async fn preview_does_not_select_tests_tarball(world: &mut E2eWorld) {
    assert!(
        !stdout(world).contains(NEXT_TESTS_TARBALL),
        "the preview selected the non-release tests sibling:\n{}",
        stdout(world)
    );
}

#[then("the install fails")]
async fn install_fails(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no recorded exit code");
    assert!(
        rc != 0,
        "expected a refusal, but the install succeeded:\n{}",
        stdout(world)
    );
}

#[then("the failure asks for an exact GPU arch and names --family gfx1200")]
async fn failure_names_the_exact_arch_flag(world: &mut E2eWorld) {
    // Both streams: what matters is that the user is told how to fix it, not
    // which file descriptor carried the sentence.
    let reported = format!(
        "{}\n{}",
        stdout(world),
        world.cli_stderr.as_deref().unwrap_or_default()
    );
    assert!(
        reported.contains("requires an exact GPU arch"),
        "the refusal did not say an exact arch is required:\n{reported}"
    );
    assert!(
        reported.contains(&format!("--family {RAW_ARCH}")),
        "the refusal did not name the flag that fixes it:\n{reported}"
    );
}

/// The default (un-overridden) ROCm 10 preview pip index base. Matches
/// `THEROCK_NEXT_RELEASE_PIP_INDEX_BASE` in `apps/rocm/src/therock.rs`.
const DEFAULT_NEXT_RELEASE_PIP_BASE: &str = "https://stable.repo.amd.com/rocm/whl-next";

/// The newest `rocm` version the live ROCm 10 preview source actually
/// publishes right now, discovered at run time rather than hardcoded: the
/// preview source is a moving target, and a version this scenario doesn't
/// control would go stale the moment a newer one is published.
async fn discover_latest_next_rocm_version() -> String {
    let url = format!("{DEFAULT_NEXT_RELEASE_PIP_BASE}/rocm/");
    let html = reqwest::get(&url)
        .await
        .unwrap_or_else(|error| {
            panic!("failed to fetch the ROCm 10 preview pip index at {url}: {error}")
        })
        .text()
        .await
        .unwrap_or_else(|error| {
            panic!("failed to read the ROCm 10 preview pip index body from {url}: {error}")
        });
    let mut versions = Vec::new();
    let marker = "rocm-";
    let mut rest = html.as_str();
    while let Some(start) = rest.find(marker) {
        let after = &rest[start + marker.len()..];
        // The `rocm` aggregate meta-package publishes as an sdist (`.tar.gz`),
        // not a wheel (confirmed live) — check both suffixes, whichever comes
        // first, same as the CLI's own `parse_simple_index_version_candidate`.
        let end = match (after.find(".tar.gz"), after.find(".whl")) {
            (Some(tar), Some(whl)) => Some(tar.min(whl)),
            (Some(tar), None) => Some(tar),
            (None, Some(whl)) => Some(whl),
            (None, None) => None,
        };
        let Some(end) = end else {
            // No recognized suffix after this "rocm-": advance past just the
            // marker itself (not the whole rest of the page) so the next
            // search still finds a later real "rocm-" occurrence instead of
            // skipping the page outright.
            rest = &after[1.min(after.len())..];
            continue;
        };
        // A wheel's stem is `rocm-<version>-<python tag>-...`; only the sdist
        // form's `end` names the version directly, so drop anything past the
        // first remaining `-` for the wheel case.
        let candidate = &after[..end];
        let version = candidate.split('-').next().unwrap_or(candidate);
        versions.push(version.to_owned());
        rest = &after[end..];
    }
    versions.sort_by(|left, right| compare_dotted_versions(left, right));
    versions
        .into_iter()
        .next_back()
        .unwrap_or_else(|| panic!("no `rocm-*` package versions were found at {url}:\n{html}"))
}

/// Loose dotted-version comparison good enough to pick "the newest" out of a
/// live index: numeric components first, falling back to plain string order
/// for anything a simple `x.y.z` split doesn't cover (e.g. a `+local` suffix).
fn compare_dotted_versions(left: &str, right: &str) -> std::cmp::Ordering {
    fn numeric_prefix(value: &str) -> Vec<u64> {
        value
            .split(['.', '+'])
            .map_while(|part| part.parse().ok())
            .collect()
    }
    numeric_prefix(left)
        .cmp(&numeric_prefix(right))
        .then_with(|| left.cmp(right))
}

#[when("the user installs the SDK from the ROCm 10 preview source with no family override")]
async fn user_installs_sdk_from_next_source_auto_detected(world: &mut E2eWorld) {
    let version = discover_latest_next_rocm_version().await;
    // No `--family`: this is the whole point of the scenario. `resolve_family`
    // falls back to `detect_host_gfx_target()` when nothing overrides it, so
    // the runner's real GPU is what picks the exact arch the next layout needs.
    let args = [
        "install",
        "sdk",
        "--channel",
        "release",
        "--format",
        "wheel",
        "--version",
        &version,
    ];
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(world, &args);
    assert!(
        rc == 0,
        "{}",
        cli_failure_report(&args, rc, &stdout, &stderr)
    );
    world.cli_output = Some(stdout);
}

#[then("the install used the ROCm 10 preview source")]
async fn assert_install_used_next_source(world: &mut E2eWorld) {
    assert_contains(
        world,
        "source_layout_generation: next-v1",
        "next layout generation",
    );
    assert_contains(
        world,
        &format!("canonical_source: {DEFAULT_NEXT_RELEASE_PIP_BASE}"),
        "live next source",
    );
}

#[then("the install requested the device extras for this host's detected GPU")]
async fn assert_install_used_detected_device_target(world: &mut E2eWorld) {
    let (examine_stdout, _, _) = crate::run_rocm(world, &["examine"]);
    let detected = examine_stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("detected_gfx_target: "))
        .unwrap_or_else(|| {
            panic!("host reported no detected_gfx_target, so auto-detection had nothing to sniff:\n{examine_stdout}")
        });
    assert_contains(
        world,
        &format!("device_target: {detected}"),
        "install did not target this host's auto-detected GPU",
    );
}

#[then("the ROCm 10 runtime passes SDK and Torch GPU probes")]
async fn assert_next_runtime_passes_sdk_and_gpu_probes(world: &mut E2eWorld) {
    let install_output = stdout(world);
    for field in ["rocm_sdk_version:", "rocm_sdk_root:", "rocm_sdk_bin:"] {
        assert!(
            install_output
                .lines()
                .any(|line| line.trim().starts_with(field)),
            "live install did not report {field}\n{install_output}"
        );
    }
    let python = install_output
        .lines()
        .find_map(|line| line.trim().strip_prefix("python_executable: "))
        .expect("live install did not report its managed Python executable");
    let script = r#"
import json
import rocm_sdk
rocm_sdk.initialize_process()
import torch
assert torch.cuda.is_available(), "torch reports no usable AMD GPU"
assert torch.cuda.device_count() > 0, "torch reports zero AMD GPUs"
probe = torch.ones(1, device="cuda")
probe.add_(1)
torch.cuda.synchronize()
print(json.dumps({"rocm_sdk": rocm_sdk.__version__, "torch": torch.__version__, "hip": torch.version.hip, "devices": torch.cuda.device_count()}))
"#;
    let result = std::process::Command::new(python)
        .args(["-c", script])
        .output()
        .unwrap_or_else(|error| panic!("failed to launch managed ROCm X Python {python}: {error}"));
    assert!(
        result.status.success(),
        "managed ROCm X SDK/Torch GPU probe failed (status {}):\nstdout:\n{}\nstderr:\n{}",
        result.status,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[then("the ROCm 10 runtime provisioned a cp314 Python interpreter")]
async fn assert_next_runtime_provisioned_cp314(world: &mut E2eWorld) {
    let install_output = stdout(world);
    let python = install_output
        .lines()
        .find_map(|line| line.trim().strip_prefix("python_executable: "))
        .expect("live install did not report its managed Python executable");
    let result = std::process::Command::new(python)
        .args([
            "-c",
            "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')",
        ])
        .output()
        .unwrap_or_else(|error| panic!("failed to launch managed Python {python}: {error}"));
    assert!(result.status.success(), "failed to read {python}'s tag");
    let tag = String::from_utf8_lossy(&result.stdout).trim().to_owned();
    assert_eq!(
        tag, "cp314",
        "ROCm 10.x's vLLM/flash-attn/amd-aiter wheels are cp314-only, but the runtime \
         provisioned {python} as {tag}"
    );
}

#[when("the user reinstalls vllm")]
async fn user_reinstalls_vllm(world: &mut E2eWorld) {
    // `--reinstall` so the adapter's ROCm 10.x discovery route (dry-run
    // resolve, then reinstall pinned to what that reported) runs even though
    // the SDK install above may already have installed vLLM once.
    let args = ["engines", "install", "vllm", "--reinstall", "--yes"];
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(world, &args);
    assert!(
        rc == 0,
        "{}",
        cli_failure_report(&args, rc, &stdout, &stderr)
    );
    world.cli_output = Some(stdout);
}

#[then("the install reports the vLLM ROCm 10.x discovery pins")]
async fn assert_vllm_rocm10_discovery_pins(world: &mut E2eWorld) {
    let install_output = stdout(world);
    assert!(
        install_output.contains("vLLM ROCm 10.x discovery pinned:"),
        "vLLM install against a live ROCm 10.x runtime did not report discovery pins:\n{install_output}"
    );
}

// ── Rebuilding a broken environment in an install folder (#533) ─────────
//
// `install sdk --format wheel --prefix <folder>` reuses the folder it is given.
// When the Python already there no longer runs, the install used to delete the
// whole folder and start again, whoever made it. These steps drive that path
// for real, up to and including the rebuild decision, against the loopback
// canonical index (so no network) and an exact `--family` (so no GPU).
//
// A fake `uv` stands in for the real one: it creates a venv on `uv venv` and
// fails everything else, so the install stops at the package step right after
// the decision under test. Like uv 0.10, it refuses a folder with entries
// unless `--allow-existing` is passed, and wipes it on `--clear` or
// `UV_VENV_CLEAR=1`, so an install that leaned on uv to cope with an existing
// folder fails here as it would for real. The fake is a shell script, hence
// these scenarios are Linux-only.

/// The user's home folder in the reported case: their files beside a
/// `bin/python` that fails `--version`.
const USER_HOME_FOLDER: &str = "user-home";
/// The folder the refusal tells the user to choose instead.
const EMPTY_INSTALL_FOLDER: &str = "empty-install-folder";
/// A runtime folder an earlier `install sdk --prefix` created, now holding
/// ComfyUI's models as well as the venv.
const ROCM_CLI_MADE_FOLDER: &str = "rocm-cli-made-prefix";

fn install_folder(world: &E2eWorld, name: &str) -> std::path::PathBuf {
    let folder = root(world).join(name);
    std::fs::create_dir_all(&folder).expect("failed to create install folder");
    // Canonical, because that is the spelling the install resolves `--prefix`
    // to and records in its runtime manifest.
    folder
        .canonicalize()
        .expect("failed to canonicalize install folder")
}

fn write_executable(path: &Path, body: &str) {
    write_fixture(path, body);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .unwrap_or_else(|e| panic!("failed to chmod {}: {e}", path.display()));
    }
}

fn plant_python_that_no_longer_runs(folder: &Path) {
    write_executable(&folder.join("bin").join("python"), "#!/bin/sh\nexit 1\n");
}

/// Run a real (not `--dry-run`) wheel install into `folder`.
///
/// Pinned to the ROCm 10 layout with an exact arch because that layout takes
/// its device payload from the arch it is given rather than probing the host
/// GPU, so it reaches the install step on a machine with no GPU. The fake `uv`
/// answers the three calls this path makes: `pip compile` (pins the fixture's
/// versions), `venv` (creates a working interpreter at `$4`), and the package
/// install, which it fails. The fake launcher reports the `cp314` tag that
/// layout requires, so the host's Python version does not matter and no managed
/// Python is bootstrapped over the network.
///
/// `HOME` is the user's home folder from runtime-19 throughout, so the CLI sees
/// the same home the user would.
fn install_sdk_into(world: &mut E2eWorld, folder: &Path) {
    run_sdk_install_into(world, folder, false);
}

fn run_sdk_install_into(world: &mut E2eWorld, folder: &Path, dry_run: bool) {
    let uv = root(world).join("fake-uv").join("uv");
    write_executable(
        &uv,
        &format!(
            r#"#!/bin/sh
if [ "$1" = pip ] && [ "$2" = compile ]; then
  cat > /dev/null
  echo rocm=={NEXT_ROCM_VERSION}
  echo torch=={NEXT_TORCH_VERSION}
  echo torchvision=={NEXT_TORCHVISION_VERSION}
  echo torchaudio=={NEXT_TORCHAUDIO_VERSION}
  exit 0
fi
if [ "$1" = venv ] && [ "$2" = --python ]; then
  target="$4"; allow=0; clear=0
  for arg in "$@"; do
    [ "$arg" = --allow-existing ] && allow=1
    [ "$arg" = --clear ] && clear=1
  done
  [ "$allow" = 0 ] && [ "${{UV_VENV_CLEAR-}}" = 1 ] && clear=1
  if [ -d "$target" ] && [ -n "$(ls -A "$target")" ]; then
    if [ "$clear" = 1 ]; then
      rm -rf "$target"
    elif [ "$allow" = 0 ]; then
      echo "error: A directory already exists at: $target" >&2
      exit 2
    fi
  fi
  mkdir -p "$target/bin"
  printf '#!/bin/sh\necho Python 3.14.0\n' > "$target/bin/python"
  chmod +x "$target/bin/python"
  echo 'home = /usr/bin' > "$target/pyvenv.cfg"
  exit 0
fi
echo "fake uv: $*" >&2
exit 1
"#
        ),
    );
    let launcher = root(world).join("fake-python").join("python3.14");
    write_executable(
        &launcher,
        "#!/bin/sh\nif [ \"$1\" = -c ]; then echo cp314; fi\nexit 0\n",
    );
    let home = install_folder(world, USER_HOME_FOLDER);
    world
        .command_env
        .push(("ROCM_CLI_UV_BINARY", uv.into_os_string()));
    world
        .command_env
        .push(("ROCM_CLI_PYTHON", launcher.into_os_string()));
    world.command_env.push(("HOME", home.into_os_string()));
    // `command_env` is consumed by each command, so the fixture index the
    // `Given` pointed the CLI at has to be named again for every run here, or
    // the second command of a scenario resolves the live index instead.
    allow_base_overrides(world);
    let current = current_pip_base(world);
    let next = next_pip_base(world);
    world
        .command_env
        .push(("ROCM_CLI_THEROCK_RELEASE_PIP_BASE", current.into()));
    world
        .command_env
        .push(("ROCM_CLI_THEROCK_NEXT_PIP_BASE", next.into()));
    let prefix = folder.display().to_string();
    let mut args = vec![
        "install",
        "sdk",
        "--channel",
        "release",
        "--format",
        "wheel",
        "--family",
        RAW_ARCH,
        "--version",
        NEXT_ROCM_VERSION,
        "--prefix",
        &prefix,
    ];
    if dry_run {
        args.push("--dry-run");
    }
    preview(world, &args);
}

fn reported(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        stdout(world),
        world.cli_stderr.as_deref().unwrap_or_default()
    )
}

#[given("the user's home folder holds their files and a Python that no longer runs")]
async fn home_folder_with_failing_python(world: &mut E2eWorld) {
    let home = install_folder(world, USER_HOME_FOLDER);
    plant_python_that_no_longer_runs(&home);
    write_fixture(&home.join("Documents").join("thesis.txt"), "years of work");
}

#[given("a folder ROCm CLI installed into holds ComfyUI models and a Python that no longer runs")]
async fn rocm_cli_folder_with_comfyui_models(world: &mut E2eWorld) {
    let folder = install_folder(world, ROCM_CLI_MADE_FOLDER);
    plant_python_that_no_longer_runs(&folder);
    write_fixture(&folder.join("pyvenv.cfg"), "home = /usr/bin\n");
    write_fixture(
        &folder
            .join("lib")
            .join("python3.12")
            .join("site-packages")
            .join("stale.py"),
        "old",
    );
    // What `save_runtime_manifest` writes beside every install it completes.
    let marker = serde_json::to_string_pretty(&serde_json::json!({
        "runtime_key": "release-wheel-gfx1200-7-10-0",
        "runtime_id": format!("therock-release:{GROUP_FAMILY}"),
        "format": "wheel",
        "install_root": folder,
    }))
    .expect("failed to serialize runtime marker");
    write_fixture(&folder.join(".rocm-cli-runtime.json"), &marker);
    write_fixture(
        &folder
            .join("apps")
            .join("comfyui")
            .join("source")
            .join("models")
            .join("checkpoint.safetensors"),
        "the user's model",
    );
}

#[when("the user installs the SDK into their home folder")]
async fn install_sdk_into_home(world: &mut E2eWorld) {
    let home = install_folder(world, USER_HOME_FOLDER);
    install_sdk_into(world, &home);
}

#[when("the user installs the SDK into an empty folder instead")]
async fn install_sdk_into_empty_folder(world: &mut E2eWorld) {
    let empty = install_folder(world, EMPTY_INSTALL_FOLDER);
    install_sdk_into(world, &empty);
}

#[when("the user previews installing the SDK into their home folder")]
async fn preview_sdk_install_into_home(world: &mut E2eWorld) {
    let home = install_folder(world, USER_HOME_FOLDER);
    run_sdk_install_into(world, &home, true);
}

#[when("the user previews installing the SDK into the folder ROCm CLI installed into")]
async fn preview_sdk_install_into_rocm_cli_folder(world: &mut E2eWorld) {
    let folder = install_folder(world, ROCM_CLI_MADE_FOLDER);
    run_sdk_install_into(world, &folder, true);
}

#[then("the preview predicts the refusal of the home folder and changes nothing")]
async fn preview_predicts_home_refusal(world: &mut E2eWorld) {
    let home = install_folder(world, USER_HOME_FOLDER);
    let reported = reported(world);
    assert_eq!(world.cli_rc, Some(0), "the preview failed:\n{reported}");
    assert!(
        reported.contains(&format!(
            "python_env: the install will refuse this folder: refusing to create the Python environment at {}",
            home.display()
        )),
        "the preview did not predict the refusal:\n{reported}"
    );
    // No command for an install that will not run one.
    assert!(
        reported.contains("command: none; the install will refuse this folder"),
        "the preview did not say no command will run:\n{reported}"
    );
    assert!(
        !reported.contains("command: uv venv"),
        "the preview shows a uv venv command the install will never run:\n{reported}"
    );
    assert_eq!(
        std::fs::read_to_string(home.join("Documents").join("thesis.txt")).ok(),
        Some("years of work".to_owned()),
        "a preview changed the home folder"
    );
    assert!(home.join("bin").join("python").is_file());
}

#[then("the preview shows a rebuild in place that keeps the folder")]
async fn preview_shows_rebuild_in_place(world: &mut E2eWorld) {
    let folder = install_folder(world, ROCM_CLI_MADE_FOLDER);
    let reported = reported(world);
    assert_eq!(world.cli_rc, Some(0), "the preview failed:\n{reported}");
    assert!(
        reported.contains("python_env: rebuild only the venv files ROCm CLI created here"),
        "the preview did not describe the in-place rebuild:\n{reported}"
    );
    assert!(
        reported.contains(&format!("{} --allow-existing &&", folder.display())),
        "the previewed uv venv command is not the one a rebuild runs:\n{reported}"
    );
    assert!(
        folder.join("lib").is_dir() && folder.join("pyvenv.cfg").is_file(),
        "a preview changed the folder"
    );
}

#[when("the user installs the SDK into the folder ROCm CLI installed into")]
async fn install_sdk_into_rocm_cli_folder(world: &mut E2eWorld) {
    let folder = install_folder(world, ROCM_CLI_MADE_FOLDER);
    install_sdk_into(world, &folder);
}

#[then("the install refuses to rebuild the home folder and changes nothing in it")]
async fn install_refuses_home_folder(world: &mut E2eWorld) {
    let home = install_folder(world, USER_HOME_FOLDER);
    let reported = reported(world);
    assert_ne!(world.cli_rc, Some(0), "the install succeeded:\n{reported}");
    assert!(
        reported.contains(&format!(
            "refusing to create the Python environment at {}",
            home.display()
        )),
        "the refusal did not name the home folder:\n{reported}"
    );
    assert!(
        reported.contains("Nothing was changed"),
        "the refusal did not say nothing was changed:\n{reported}"
    );
    // The claim and the disk, together.
    assert_eq!(
        std::fs::read_to_string(home.join("Documents").join("thesis.txt")).ok(),
        Some("years of work".to_owned()),
        "the user's file is gone or changed although the install said nothing was changed"
    );
    assert_eq!(
        std::fs::read_to_string(home.join("bin").join("python")).ok(),
        Some("#!/bin/sh\nexit 1\n".to_owned()),
        "the user's bin/python is gone or changed"
    );
    assert!(
        reported.contains("rocm install sdk --prefix <empty-folder>"),
        "the refusal did not say what to do instead:\n{reported}"
    );
}

#[then("the install gets past the refusal and creates a Python environment in the empty folder")]
async fn install_proceeds_in_empty_folder(world: &mut E2eWorld) {
    let empty = install_folder(world, EMPTY_INSTALL_FOLDER);
    let reported = reported(world);
    assert!(
        !reported.contains("refusing to create"),
        "following the refusal's advice was refused again:\n{reported}"
    );
    assert!(
        empty.join("bin").join("python").is_file(),
        "no Python environment was created in the empty folder:\n{reported}"
    );
    let home = install_folder(world, USER_HOME_FOLDER);
    assert!(
        home.join("Documents").join("thesis.txt").is_file(),
        "installing elsewhere touched the home folder"
    );
}

#[then(
    "the install rebuilds only the environment's own files and reports keeping the ComfyUI data"
)]
async fn install_rebuilds_and_keeps_comfyui(world: &mut E2eWorld) {
    let folder = install_folder(world, ROCM_CLI_MADE_FOLDER);
    let reported = reported(world);
    let line = format!(
        "Existing Python environment at {} no longer runs or does not match the required interpreter; rebuilding only its venv files (kept: .rocm-cli-runtime.json, apps).",
        folder.display()
    );
    assert!(
        reported.contains(&line),
        "the install did not report a rebuild that kept the ComfyUI data:\n{reported}"
    );
    // The line says `apps` was kept; the disk has to agree.
    assert_eq!(
        std::fs::read_to_string(
            folder
                .join("apps")
                .join("comfyui")
                .join("source")
                .join("models")
                .join("checkpoint.safetensors")
        )
        .ok(),
        Some("the user's model".to_owned()),
        "the ComfyUI model is gone although the install reported keeping it:\n{reported}"
    );
    assert!(
        folder.join(".rocm-cli-runtime.json").is_file(),
        "the runtime manifest is gone although the install reported keeping it"
    );
    assert!(
        !folder
            .join("lib")
            .join("python3.12")
            .join("site-packages")
            .join("stale.py")
            .exists(),
        "the old environment's files were not cleared"
    );
    assert!(
        folder.join("bin").join("python").is_file(),
        "no new Python environment was created:\n{reported}"
    );
}
