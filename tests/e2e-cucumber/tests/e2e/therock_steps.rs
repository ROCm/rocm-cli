// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::path::Path;

use cucumber::{given, then, when};
use e2e_cucumber::cli_failure_report;
use e2e_cucumber::loopback_http::LoopbackServer;

use crate::E2eWorld;

const RAW_ARCH: &str = "gfx1200";
const ROCM_VERSION: &str = "10.0.0";
const TORCH_VERSION: &str = "2.10.0+rocm10.0.0";
const TORCHVISION_VERSION: &str = "0.25.0+rocm10.0.0";
const TORCHAUDIO_VERSION: &str = "2.10.0+rocm10.0.0";
const REAL_TARBALL: &str = "therock-dist-linux-gfx120X-all-10.0.0.tar.gz";
const TESTS_TARBALL: &str = "therock-dist-linux-gfx120X-all-tests-10.0.0.tar.gz";

fn root(world: &E2eWorld) -> &Path {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
}

fn wheel_index_html(package: &str, version: &str) -> String {
    format!("<a href=\"{package}-{version}-py3-none-any.whl\">{package}-{version}-py3-none-any.whl</a>")
}

#[given("a ROCm 10 pip index fixture for family gfx1200")]
async fn rocm10_pip_index_fixture(world: &mut E2eWorld) {
    let served = root(world).join("therock-next-pip");
    for (package, version) in [
        ("rocm", ROCM_VERSION),
        ("torch", TORCH_VERSION),
        ("torchvision", TORCHVISION_VERSION),
        ("torchaudio", TORCHAUDIO_VERSION),
    ] {
        let package_dir = served.join(package);
        std::fs::create_dir_all(&package_dir).expect("failed to create fixture pip package dir");
        std::fs::write(
            package_dir.join("index.html"),
            wheel_index_html(package, version),
        )
        .expect("failed to write fixture pip index");
    }
    let server = LoopbackServer::start(&served);
    world.command_env.push((
        "ROCM_CLI_THEROCK_NEXT_RELEASE_PIP_BASE",
        server.base_url().into(),
    ));
    world.artifact_server = Some(server);
}

#[given("a ROCm 10 tarball index fixture for family gfx1200 with a tests sibling")]
async fn rocm10_tarball_index_fixture(world: &mut E2eWorld) {
    let served = root(world).join("therock-next-tarball");
    std::fs::create_dir_all(&served).expect("failed to create fixture tarball dir");
    // The real dist archive has an earlier mtime than its `-tests-` sibling
    // (confirmed on the live AMD index), so naive highest-mtime selection picks
    // the wrong file. `select_tarball_artifact` instead filters out any
    // candidate whose extracted version fails `parse_version` first.
    let index_html = format!(
        "<html><body><script>const files = [{{\"name\": \"{REAL_TARBALL}\", \"mtime\": 1787612008.0}}, {{\"name\": \"{TESTS_TARBALL}\", \"mtime\": 1787612032.0}}];</script></body></html>"
    );
    std::fs::write(served.join("index.html"), index_html)
        .expect("failed to write fixture tarball index");
    let server = LoopbackServer::start(&served);
    world.command_env.push((
        "ROCM_CLI_THEROCK_NEXT_RELEASE_TARBALL_BASE",
        format!("{}/", server.base_url()).into(),
    ));
    world.artifact_server = Some(server);
}

#[when("the user previews a wheel SDK install for family gfx1200")]
async fn preview_wheel_install(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(
        world,
        &[
            "install",
            "sdk",
            "--format",
            "wheel",
            "--family",
            RAW_ARCH,
            "--dry-run",
        ],
    );
    assert!(
        rc == 0,
        "{}",
        cli_failure_report(&["install", "sdk"], rc, &stdout, &stderr)
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user previews a tarball SDK install for family gfx1200")]
async fn preview_tarball_install(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(
        world,
        &[
            "install",
            "sdk",
            "--format",
            "tarball",
            "--family",
            RAW_ARCH,
            "--dry-run",
        ],
    );
    assert!(
        rc == 0,
        "{}",
        cli_failure_report(&["install", "sdk"], rc, &stdout, &stderr)
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the dry-run output selects the ROCm 10 pip index")]
async fn output_selects_next_pip_index(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    let server = world
        .artifact_server
        .as_ref()
        .expect("no fixture server recorded");
    assert!(
        stdout.contains(&format!("index_url: {}", server.base_url())),
        "dry-run output did not select the fixture pip index: {stdout}"
    );
    assert!(
        stdout.contains(&format!("latest_compatible_version: {ROCM_VERSION}")),
        "dry-run output did not resolve the fixture rocm version: {stdout}"
    );
}

#[then("the dry-run output requests the gfx1200 device extras")]
async fn output_requests_device_extras(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    let expected = format!(
        "package_specs: rocm[libraries,devel,device-{RAW_ARCH}]=={ROCM_VERSION} torch[device-{RAW_ARCH}]=={TORCH_VERSION} torchvision[device-{RAW_ARCH}]=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"
    );
    assert!(
        stdout.contains(&expected),
        "dry-run output did not request the expected device extras: {stdout}"
    );
}

#[then("the dry-run output selects the real tarball artifact")]
async fn output_selects_real_tarball(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    assert!(
        stdout.contains(&format!("tarball: {REAL_TARBALL}")),
        "dry-run output did not select the real tarball artifact: {stdout}"
    );
    assert!(
        stdout.contains(&format!("latest_version: {ROCM_VERSION}")),
        "dry-run output did not resolve the expected tarball version: {stdout}"
    );
}

#[then("the dry-run output does not select the tests artifact")]
async fn output_does_not_select_tests_tarball(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    assert!(
        !stdout.contains(TESTS_TARBALL),
        "dry-run output unexpectedly selected the tests artifact: {stdout}"
    );
}
