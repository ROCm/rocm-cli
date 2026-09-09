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

use cucumber::{given, then, when};
use e2e_cucumber::cli_failure_report;
use e2e_cucumber::loopback_http::LoopbackServer;

use crate::E2eWorld;

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
        "package_specs: rocm[libraries,devel,device-{RAW_ARCH}]=={NEXT_ROCM_VERSION} \
         torch[device-{RAW_ARCH}]=={NEXT_TORCH_VERSION} \
         torchvision[device-{RAW_ARCH}]=={NEXT_TORCHVISION_VERSION} \
         torchaudio=={NEXT_TORCHAUDIO_VERSION}"
    );
    assert_contains(world, &expected, "device extras");
    assert_contains(
        world,
        &format!("device_target: {RAW_ARCH}"),
        "device target",
    );
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
